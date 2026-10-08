"""``swe-duel setup`` — prepare the environment: clone target repos, build images.

Subcommands
-----------
``swe-duel setup repos [--only flask ...]``
    Clone every configured target repo at its pinned commit into ``./repos``
    (Python port of the former ``scripts/setup_repos.sh``). Existing clones
    are re-pinned with fetch + checkout.

``swe-duel setup docker [--only flask ...] [--force]``
    Build ``swe-duel-base``, ``swe-duel-mock`` and the per-repo ``swe-duel-<name>`` images
    from the Dockerfiles bundled inside the installed package (replaces
    ``scripts/build_docker.sh``). The build context is staged into a temporary
    directory that mirrors the canonical layout the Dockerfiles expect, so a
    clean ``pip install swe-duel`` can build everything without a source checkout.
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from swe_duel.cli._common import resolve_config_dir
from swe_duel.config import RepoConfig, load_all_repo_configs
from swe_duel.sandbox.image_build import (
    BASE_IMAGE,
    MOCK_IMAGE,
    build_base_image,
    build_mock_image,
    build_repo_image,
    image_exists,
    repo_image_tag,
    staged_build_context,
)


def _select_repos(
    repo_configs: dict[str, RepoConfig], only: list[str] | None
) -> dict[str, RepoConfig]:
    if not only:
        return dict(repo_configs)
    unknown = [n for n in only if n not in repo_configs]
    if unknown:
        known = ", ".join(sorted(repo_configs))
        print(
            f"error: unknown repo(s) {unknown}; configured: {known}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return {n: repo_configs[n] for n in only}


# ── repos ────────────────────────────────────────────────────


def _git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def clone_or_repin(name: str, rc: RepoConfig, repos_dir: Path) -> bool:
    """Clone ``rc`` at its pinned ref, or re-pin an existing clone.

    Mirrors the historical shell script exactly: fresh clones use
    ``--depth=1 --branch <ref>``; existing clones fetch the ref and check it
    out. Returns True when anything was cloned/fetched, False when the clone
    already sat at the pinned ref.
    """
    target = repos_dir / name
    if target.exists():
        head = _git(["rev-parse", "HEAD"], cwd=target)
        pinned = _git(["rev-parse", "--verify", f"{rc.commit}^{{commit}}"], cwd=target)
        if head.returncode == 0 and pinned.returncode == 0 and head.stdout.strip() == pinned.stdout.strip():
            print(f"[{name}] already at pinned {rc.commit}")
            return False
        print(f"[{name}] fetching pinned {rc.commit} ...")
        fetch = _git(["fetch", "--depth=1", "origin", rc.commit], cwd=target)
        if fetch.returncode != 0:
            print(f"error: git fetch failed for {name}:\n{fetch.stderr}", file=sys.stderr)
            raise SystemExit(1)
        checkout = _git(["checkout", rc.commit], cwd=target)
        if checkout.returncode != 0:
            print(f"error: git checkout failed for {name}:\n{checkout.stderr}", file=sys.stderr)
            raise SystemExit(1)
        print(f"[{name}] checked out {rc.commit}")
        return True

    print(f"[{name}] cloning {rc.url} at {rc.commit} ...")
    clone = _git(["clone", "--depth=1", "--branch", rc.commit, rc.url, str(target)])
    if clone.returncode != 0:
        print(f"error: git clone failed for {name}:\n{clone.stderr}", file=sys.stderr)
        raise SystemExit(1)
    return True


def cmd_repos(args: argparse.Namespace) -> int:
    config_dir = resolve_config_dir(args)
    repo_configs = _select_repos(load_all_repo_configs(config_dir), args.only)
    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    for name, rc in sorted(repo_configs.items()):
        clone_or_repin(name, rc, repos_dir)
    print(f"All repos set up in {repos_dir.resolve()}")
    return 0


# ── docker ───────────────────────────────────────────────────


def _build_or_skip(
    label: str,
    tag: str,
    build_fn: Callable[[], subprocess.CompletedProcess[str]],
    *,
    force: bool,
) -> bool:
    """Build ``tag`` unless it already exists (and not ``force``).

    Returns True when a build was attempted, False when skipped.
    """
    if image_exists(tag) and not force:
        print(f"[{label}] image {tag} already built — skipping (use --force to rebuild)")
        return False
    print(f"[{label}] building {tag} ...")
    result = build_fn()
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"error: docker build failed for {label} ({tag})", file=sys.stderr)
        raise SystemExit(1)
    return True


def cmd_docker(args: argparse.Namespace) -> int:
    config_dir = resolve_config_dir(args)
    repo_configs = _select_repos(load_all_repo_configs(config_dir), args.only)
    repos_dir = Path(args.repos_dir)

    # Nothing to do? Skip the (copy-heavy) context staging entirely.
    wanted_tags = [BASE_IMAGE, MOCK_IMAGE] + [
        repo_image_tag(n) for n in repo_configs
    ]
    if not args.force and all(image_exists(t) for t in wanted_tags):
        print("All requested Docker images already built (use --force to rebuild).")
        return 0

    # base + mock build first (per-repo images FROM swe-duel-base; doctor and the
    # unit suite use swe-duel-mock).
    with staged_build_context({}) as base_ctx:
        # ``{}`` clones: base/mock need no repo clone in the context.
        _build_or_skip("base", BASE_IMAGE, lambda: build_base_image(base_ctx), force=args.force)
        _build_or_skip("mock", MOCK_IMAGE, lambda: build_mock_image(base_ctx), force=args.force)

    if not repo_configs:
        return 0

    missing = [n for n in repo_configs if not (repos_dir / n).is_dir()]
    if missing:
        print(
            "error: clone(s) missing for: "
            + ", ".join(missing)
            + f"\n  run first: swe-duel setup repos --only {' '.join(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    clones = {name: repos_dir / name for name in repo_configs}
    with staged_build_context(clones) as ctx:
        for name in sorted(repo_configs):
            tag = repo_image_tag(name)
            _build_or_skip(
                name, tag, functools.partial(build_repo_image, name, ctx), force=args.force
            )
    print("All requested Docker images built.")
    return 0


# ── CLI ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swe-duel-setup",
        description="Prepare the environment: clone target repos and build Docker images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config-dir", default=None)
        p.add_argument("--repos-dir", default="repos")

    p_repos = sub.add_parser("repos", help="Clone target repos at pinned commits")
    _common_flags(p_repos)
    p_repos.add_argument("--only", nargs="+", default=None, metavar="REPO")
    p_repos.set_defaults(func=cmd_repos)

    p_docker = sub.add_parser("docker", help="Build swe-duel-base + swe-duel-mock + per-repo images")
    _common_flags(p_docker)
    p_docker.add_argument("--only", nargs="+", default=None, metavar="REPO")
    p_docker.add_argument("--force", action="store_true", help="Rebuild existing images")
    p_docker.set_defaults(func=cmd_docker)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
