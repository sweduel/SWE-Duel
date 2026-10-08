"""Docker container lifecycle and command execution."""

from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar

import docker
import requests
from docker.errors import ImageNotFound

from swe_duel.models import ExecutionResult


SESSION_PID_LABEL = "swe_duel.session_pid"
"""Label key applied to every container so interrupted runs can kill their own."""

_T = TypeVar("_T")

_DOCKER_HTTP_TIMEOUT_S = 300
"""HTTP timeout for the docker-py client (docker-py's default is 60s).

docker-py applies the client-level timeout to every request that does not
override one — including the two calls that CANNOT be configured per request:
the ``get_archive`` tar stream (seeding a gate workspace from the image) and
the follow-logs stream. Under parallel generation/scoring load a loaded daemon
can easily stall a read past 60s, raising ``requests.exceptions.ReadTimeout``
(note: NOT a ``docker.errors.APIError``), which used to escape and kill an
entire generation pool. ``Container.wait`` is unaffected: it passes its own
explicit request timeout (``timeout_s``).
"""

_DOCKER_HTTP_RETRY_DELAYS_S: tuple[float, ...] = (2.0, 8.0)
"""Backoff sleeps between attempts of a retried docker-py HTTP call."""

_TRANSIENT_DOCKER_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


def _retry_docker(fn: Callable[[], _T]) -> _T:
    """Run a docker-py call, retrying transient HTTP transport failures.

    ReadTimeout / ConnectTimeout / ConnectionError on the local daemon socket
    are almost always load-induced (a big tar stream, a daemon busy starting
    sibling containers) and clear within seconds, so retry with backoff
    instead of letting one slow read abort the surrounding pool / gate.
    Deterministic API errors (``docker.errors.APIError``) do not match the
    transient tuple and still raise immediately.
    """
    for delay in _DOCKER_HTTP_RETRY_DELAYS_S:
        try:
            return fn()
        except _TRANSIENT_DOCKER_HTTP_ERRORS:
            time.sleep(delay)
    return fn()


SWE_DUEL_CONTAINER_PREFIX = "swe-duel-"
"""Every container this framework launches is named with this prefix so it is
unambiguously ours — never colliding with a co-tenant's ``minisweagent-*`` /
``agent-server-*`` containers, and reapable by name as a cleanup fallback."""


def _session_labels() -> dict[str, str]:
    """Return labels that tie a container to the current process."""
    return {SESSION_PID_LABEL: str(os.getpid()), "swe_duel.managed": "true"}


def _sanitize_name_segment(value: str) -> str:
    """Reduce an arbitrary string to a Docker-name-safe segment.

    Docker container names must match ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``; we
    lowercase, replace every other run of characters with a single ``-`` and
    strip leading/trailing separators.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-._")
    return cleaned


def repo_from_image(image: str | None) -> str:
    """Derive a short repo label from a per-repo Docker image name.

    e.g. ``swe-duel-flask`` → ``flask``, ``swe-duel-java-html-sanitizer`` →
    ``java-html-sanitizer``, ``swe-duel-flask:latest`` → ``flask``,
    ``registry/swe-duel-jwt:tag`` → ``jwt``. Falls back to ``"repo"`` if nothing
    usable remains.
    """
    if not image:
        return "repo"
    name = str(image).rsplit("/", 1)[-1]  # drop any registry/namespace
    name = name.split(":", 1)[0]          # drop :tag
    name = name.removeprefix("swe-duel-")
    sanitized = _sanitize_name_segment(name)
    return sanitized or "repo"


def swe_duel_container_name(repo: str | None, harness: str | None = None) -> str:
    """Build a unique ``swe-duel-``-prefixed container name.

    Format: ``swe-duel-<repo>[-<harness>]-<pid>-<uuid8>``. The pid scopes the name
    to this session and the short uuid guarantees uniqueness across concurrent
    runs in the same process. ``harness`` is omitted at the sandbox layer
    (``docker_executor``), which only knows the image.
    """
    parts = [SWE_DUEL_CONTAINER_PREFIX.rstrip("-"), _sanitize_name_segment(repo or "") or "repo"]
    if harness:
        parts.append(_sanitize_name_segment(harness) or "harness")
    parts.append(str(os.getpid()))
    parts.append(uuid.uuid4().hex[:8])
    return "-".join(parts)


class DockerExecutor:
    """Run commands inside isolated Docker containers."""

    def __init__(
        self,
        docker_image: str,
        timeout_s: int = 120,
        memory_mb: int = 512,
        cpu_limit: float = 1.0,
    ) -> None:
        self.docker_image = docker_image
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.cpu_limit = cpu_limit
        self._client = docker.from_env(timeout=_DOCKER_HTTP_TIMEOUT_S)

        try:
            _retry_docker(lambda: self._client.images.get(docker_image))
        except ImageNotFound:
            raise ValueError(f"Docker image not found: {docker_image}")

    def execute(
        self,
        file_overrides: dict[str, str],
        command: str,
        extra_files: dict[str, str] | None = None,
        stream_file: Path | None = None,
    ) -> ExecutionResult:
        """Execute a command in a fresh container with file overrides applied.

        (1) Create temp dir. (2) Copy repo into temp dir. (3) Write file_overrides.
        (4) Write extra_files. (5) docker run. (6) Capture output. (7) Cleanup.
        """
        tmpdir = tempfile.mkdtemp(prefix="swe_duel_exec_")
        try:
            workspace = Path(tmpdir) / "workspace"
            self._seed_workspace_from_image(workspace)
            for rel_path, content in file_overrides.items():
                target = workspace / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            if extra_files:
                for rel_path, content in extra_files.items():
                    target = workspace / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
            return self._run_container(workspace, command, stream_file=stream_file)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def execute_in_workspace(
        self,
        workspace_path: Path,
        command: str,
        extra_files: dict[str, str] | None = None,
        stream_file: Path | None = None,
    ) -> ExecutionResult:
        """Execute a command in an existing workspace directory.

        Like execute() but mounts the workspace directly instead of copying.
        """
        if extra_files:
            for rel_path, content in extra_files.items():
                target = workspace_path / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        return self._run_container(workspace_path, command, stream_file=stream_file)

    def health_check(self) -> bool:
        """Run a trivial Python command in the container to verify it works."""
        try:
            result = self.execute(file_overrides={}, command="python -c 'print(\"ok\")'")
            return result.return_code == 0 and "ok" in result.stdout
        except Exception:
            return False

    def _seed_workspace_from_image(self, workspace: Path) -> None:
        """Extract /workspace from the image into the local workspace path."""

        def _fetch_tar() -> bytes:
            # Fresh container name per attempt: if the previous attempt's
            # HTTP response timed out after the daemon had already created
            # the container, reusing the name would 409 on the retry. Any
            # such orphan carries the swe-duel- prefix + session labels and is
            # reaped by the session/leak cleanup.
            seed = self._client.containers.create(
                self.docker_image,
                name=swe_duel_container_name(repo_from_image(self.docker_image)),
                labels=_session_labels(),
            )
            try:
                # Buffer the full tar before extracting: the ReadTimeout
                # surfaces while ITERATING the stream (docker-py cannot be
                # given a per-request timeout for get_archive), so the
                # retry has to own the iteration. A failed attempt then
                # leaves no partially-extracted tree behind.
                bits, _ = seed.get_archive("/workspace")
                return b"".join(bits)
            finally:
                try:
                    seed.remove(force=True)
                except Exception:
                    pass

        tar_bytes = _retry_docker(_fetch_tar)
        bio = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=bio) as tar:
            tar.extractall(workspace.parent)

    def _run_container(
        self, workspace: Path, command: str, stream_file: Path | None = None
    ) -> ExecutionResult:
        """Run a Docker container with the workspace mounted and command executed.

        If ``stream_file`` is given, stdout and stderr are appended to that file
        in near real-time so long-running tests can be monitored.
        """
        start = time.monotonic()
        timed_out = False
        return_code = -1
        stdout = ""
        stderr = ""

        if stream_file is not None:
            stream_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            container = _retry_docker(
                lambda: self._client.containers.run(
                    image=self.docker_image,
                    name=swe_duel_container_name(repo_from_image(self.docker_image)),
                    command=["bash", "-c", command],
                    volumes={str(workspace.resolve()): {"bind": "/workspace", "mode": "rw"}},
                    mem_limit=f"{self.memory_mb}m",
                    nano_cpus=int(self.cpu_limit * 1e9),
                    detach=True,
                    stdout=True,
                    stderr=True,
                    labels=_session_labels(),
                    network_mode="none",
                )
            )
            try:
                chunks: list[str] = []
                stream_exc: list[Exception | None] = [None]

                def _stream_logs() -> None:
                    try:
                        for chunk in container.logs(
                            stdout=True, stderr=True, stream=True, follow=True
                        ):
                            text = chunk.decode("utf-8", errors="replace")
                            chunks.append(text)
                            if stream_file is not None:
                                with open(stream_file, "a", encoding="utf-8") as fh:
                                    fh.write(text)
                                    fh.flush()
                    except Exception as e:
                        stream_exc[0] = e

                streamer = threading.Thread(target=_stream_logs, daemon=True)
                streamer.start()

                try:
                    result = container.wait(timeout=self.timeout_s)
                    return_code = result.get("StatusCode", -1)
                except Exception:
                    timed_out = True
                    try:
                        container.kill()
                    except Exception:
                        pass
                    return_code = -1

                streamer.join(timeout=5)
                stdout = "".join(chunks)
                # The follow stream applies the client HTTP timeout to every
                # socket read (docker-py cannot be configured per request
                # here either): a container silent for that window raises
                # ReadTimeout inside the streamer and the stream dies with
                # truncated output. Once the container has exited, one
                # non-stream fetch is authoritative — REPLACE the partial
                # capture (never append: the fetch replays logs from the
                # start, so appending would duplicate).
                if stream_exc[0] is not None or not stdout:
                    try:
                        stdout = _retry_docker(
                            lambda: container.logs(stdout=True, stderr=True)
                        ).decode("utf-8", errors="replace")
                    except Exception:
                        pass

            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        except docker.errors.APIError as e:
            return ExecutionResult(
                return_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        return ExecutionResult(
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )


def kill_session_containers(session_pid: int | None = None) -> None:
    """Kill all running containers tagged with ``session_pid``.

    When ``session_pid`` is omitted, the current process PID is used. This is
    used by CLI signal handlers to ensure Ctrl+C / Ctrl+Z / terminal close
    does not leave SWE-Duel sandbox or agent containers running in the background.
    """
    pid = session_pid if session_pid is not None else os.getpid()
    try:
        client = docker.from_env()
    except Exception:
        return
    label_filter: dict[str, str | list[str] | bool] = {"label": f"{SESSION_PID_LABEL}={pid}"}
    try:
        victims = client.containers.list(filters=label_filter, all=False)
    except Exception:
        victims = []
    for container in victims:
        try:
            container.kill()
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass


def _kill_leaked_swe_duel_containers() -> None:
    """Best-effort kill of any running ``swe-duel-*`` containers.

    Every container this framework launches — sandbox/gate/scoring containers
    (``docker_executor``) and the agent-harness containers (mini-swe-agent's
    ``DockerEnvironment`` and OpenHands' ``DockerWorkspace``, both renamed to
    the ``swe-duel-`` prefix) — is named ``swe-duel-…``. We reap them by prefix as a
    fallback when the session-PID label cleanup misses one.

    This intentionally targets the ``swe-duel-`` prefix, NOT ``minisweagent-``: a
    co-tenant on the same host may run their own ``minisweagent-*`` containers,
    and killing those by prefix would interfere with their work.
    """
    try:
        client = docker.from_env()
    except Exception:
        return
    try:
        running = client.containers.list()
    except Exception:
        return
    for container in running:
        if not (container.name or "").startswith(SWE_DUEL_CONTAINER_PREFIX):
            continue
        try:
            container.kill()
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass


def kill_all_swe_duel_containers() -> None:
    """Kill every SWE-Duel-managed container for this session plus leaked agents."""
    kill_session_containers(os.getpid())
    _kill_leaked_swe_duel_containers()
