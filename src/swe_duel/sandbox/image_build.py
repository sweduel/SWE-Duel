"""Build the SWE-Duel per-repo Docker images from the bundled docker/ context.

The Dockerfiles (shipped as package data under ``swe_duel/docker/``) expect a
build context whose root contains::

    docker/                 Dockerfiles + shared helper scripts
    tests/fixtures/mock_repo   mock target repo (for swe-duel-mock)
    repos/<name>/           cloned target repo (for swe-duel-<name>)

Source checkouts have this layout naturally (historically the project root was
the build context); pip installs stage the same layout into a temporary
directory via :func:`stage_docker_context` and build from there, so the
Dockerfiles work unchanged in both worlds.

Image tags are stable (``swe-duel-base``, ``swe-duel-mock``, ``swe-duel-<repo>``, all
``:latest``) — everything downstream (gates, harness containers, the test
suite) references them by name.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import swe_duel
import swe_duel.validation

# Build order: base image (all per-repo images FROM it), then the mock repo
# image (used by the unit suite and `swe-duel doctor`), then every target repo.
BASE_IMAGE = "swe-duel-base:latest"
MOCK_IMAGE = "swe-duel-mock:latest"


def docker_data_dir() -> Path:
    """On-disk location of the bundled Dockerfiles + helper scripts."""
    return Path(swe_duel.__file__).resolve().parent / "docker"


def bundled_fixtures_dir() -> Path:
    """On-disk location of the bundled fixtures (contains mock_repo)."""
    return Path(swe_duel.validation.__file__).resolve().parent / "fixtures"


def repo_image_tag(repo_name: str) -> str:
    return f"swe-duel-{repo_name}:latest"


def image_exists(tag: str) -> bool:
    """Whether ``tag`` is present in the local Docker daemon."""
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def stage_docker_context(
    dest: Path,
    repo_clones: dict[str, Path],
    *,
    include_mock: bool = True,
) -> Path:
    """Materialize the canonical build context at ``dest``.

    Copies the bundled ``docker/`` tree, the bundled mock repo (into
    ``tests/fixtures/mock_repo`` — the path ``docker/mock/Dockerfile`` COPYs),
    and each target repo clone into ``repos/<name>``. Copies preserve current
    image semantics (clones go in as-is, including ``.git`` — that is what
    building from the project root always produced).
    """
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copytree(docker_data_dir(), dest / "docker")
    if include_mock:
        shutil.copytree(
            bundled_fixtures_dir() / "mock_repo",
            dest / "tests" / "fixtures" / "mock_repo",
        )
    for name, clone_dir in repo_clones.items():
        if not clone_dir.is_dir():
            raise FileNotFoundError(
                f"clone for {name!r} not found at {clone_dir} — run: "
                f"swe-duel setup repos --only {name}"
            )
        shutil.copytree(clone_dir, dest / "repos" / name)
    return dest


def docker_build(
    tag: str,
    dockerfile_rel: str,
    context_dir: Path,
    *,
    capture: bool = False,
    timeout: int = 2700,
) -> subprocess.CompletedProcess[str]:
    """``docker build -t <tag> -f <dockerfile_rel> <context_dir>``."""
    cmd = [
        "docker",
        "build",
        "-t",
        tag,
        "-f",
        str(context_dir / dockerfile_rel),
        str(context_dir),
    ]
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def build_base_image(context_dir: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return docker_build(BASE_IMAGE, "docker/base/Dockerfile", context_dir, capture=capture)


def build_mock_image(context_dir: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return docker_build(MOCK_IMAGE, "docker/mock/Dockerfile", context_dir, capture=capture)


def build_repo_image(
    repo_name: str, context_dir: Path, *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return docker_build(
        repo_image_tag(repo_name), f"docker/{repo_name}/Dockerfile", context_dir, capture=capture
    )


def staged_build_context(
    repo_clones: dict[str, Path], *, include_mock: bool = True
) -> "_StagedContext":
    """Context manager yielding a temp directory with the canonical layout."""
    return _StagedContext(repo_clones, include_mock=include_mock)


class _StagedContext:
    def __init__(self, repo_clones: dict[str, Path], *, include_mock: bool) -> None:
        self.repo_clones = repo_clones
        self.include_mock = include_mock
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory(prefix="swe-duel-docker-ctx-")
        self.path = stage_docker_context(
            Path(self._tmp.name), self.repo_clones, include_mock=self.include_mock
        )
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = None
        self.path = None
