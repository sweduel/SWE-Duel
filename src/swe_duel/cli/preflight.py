"""Preflight environment checks enforced before every run-critical CLI.

``swe_duel.cli._common.setup`` calls :func:`run_preflight` once configs are
loaded: it verifies the Docker daemon is reachable, every selected target
repo's image exists, and the pinned clones are present — failing fast with
the exact ``swe-duel setup`` remediation command instead of dying mid-run.

Escape hatches: ``--skip-preflight`` on any run CLI, or the
``SWE_DUEL_SKIP_PREFLIGHT=1`` environment variable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import docker
from docker.errors import ImageNotFound

from swe_duel.config import RepoConfig


class PreflightError(RuntimeError):
    """Raised when a prerequisite for running is missing."""


def preflight_error(message: str) -> None:
    """Raise a PreflightError (indirection keeps the CLI import surface tiny)."""
    raise PreflightError(message)


def _missing_images(repo_configs: dict[str, RepoConfig], client: docker.DockerClient) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for name, rc in sorted(repo_configs.items()):
        try:
            client.images.get(rc.docker_image)
        except ImageNotFound:
            missing.append((name, rc.docker_image))
    return missing


def _missing_clones(repo_configs: dict[str, RepoConfig], repos_dir: Path) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for name, rc in sorted(repo_configs.items()):
        if not (repos_dir / name).exists():
            missing.append((name, rc.commit))
    return missing


def _clone_head_matches(repo_dir: Path, commit: str) -> bool | None:
    """Whether the clone's HEAD resolves to the pinned commit.

    Returns True/False when determinable, None when git or the pinned ref is
    unavailable (shallow clones of tags/branches may not resolve a sha).
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pinned = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--verify", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if head.returncode != 0 or pinned.returncode != 0:
        return None
    return head.stdout.strip() == pinned.stdout.strip()


def run_preflight(
    *,
    repo_configs: dict[str, RepoConfig],
    repos_dir: Path,
    config_dir: Path,
) -> None:
    """Fail fast when the environment is not ready to run.

    Checks (selected repos only, so partial setups stay usable):

    1. Docker daemon reachable.
    2. Every selected repo's ``swe-duel-<repo>`` image exists
       → remediation: ``swe-duel setup docker --only <names>``.
    3. Every selected repo's clone exists
       → remediation: ``swe-duel setup repos --only <names>``.
    4. Clone HEAD matches the pinned commit → warning only (agents never
       mutate the clone; scoring runs against fresh workspace copies).
    5. ``SWE_DUEL_OPENROUTER_API_KEY`` present → warning only (pure
       offline commands keep working without it).
    """
    problems: list[str] = []
    warnings: list[str] = []

    # 1. Docker daemon
    try:
        client = docker.from_env(timeout=10)
        client.ping()
    except Exception as e:
        problems.append(
            f"Docker daemon not reachable ({type(e).__name__}: {e}).\n"
            "  Start Docker, then re-run. All agents and test execution are containerized."
        )
        print(_format(problems, warnings), file=sys.stderr)
        raise PreflightError("Docker daemon not reachable") from e

    # 2. Images
    missing_images = _missing_images(repo_configs, client)
    if missing_images:
        names = " ".join(n for n, _ in missing_images)
        problems.append(
            "Docker images missing for selected repos: "
            + ", ".join(f"{n} ({img})" for n, img in missing_images)
            + f".\n  Build them with: swe-duel setup docker --only {names}"
        )

    # 3. Clones
    missing_clones = _missing_clones(repo_configs, repos_dir)
    if missing_clones:
        names = " ".join(n for n, _ in missing_clones)
        problems.append(
            "Target repo clones missing: "
            + ", ".join(f"{n} (pin {c})" for n, c in missing_clones)
            + f" under {repos_dir}.\n  Clone them with: swe-duel setup repos --only {names}"
        )

    # 4. Pinned-commit drift (warning only)
    for name, rc in sorted(repo_configs.items()):
        repo_dir = repos_dir / name
        if repo_dir.exists():
            matches = _clone_head_matches(repo_dir, rc.commit)
            if matches is False:
                warnings.append(
                    f"repos/{name}: HEAD does not match pinned commit {rc.commit} — "
                    "workspace copies come from the clone as-is; "
                    f"re-pin with: swe-duel setup repos --only {name}"
                )

    # 5. API key (warning only)
    if not os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"):
        warnings.append(
            "SWE_DUEL_OPENROUTER_API_KEY is not set — LLM-backed phases "
            "(generation, defenses) will fail until you export it."
        )

    if problems:
        print(_format(problems, warnings), file=sys.stderr)
        raise PreflightError(
            "preflight failed (skip with --skip-preflight or SWE_DUEL_SKIP_PREFLIGHT=1)"
        )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)


def _format(problems: list[str], warnings: list[str]) -> str:
    lines = ["preflight failed:"]
    for p in problems:
        lines.append(f"  - {p}")
    for w in warnings:
        lines.append(f"  warning: {w}")
    lines.append("(remediate above, or bypass with --skip-preflight / SWE_DUEL_SKIP_PREFLIGHT=1)")
    return "\n".join(lines)
