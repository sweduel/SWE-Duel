"""Shared Docker lifecycle helpers for CLI-in-container harnesses.

Codex and Claude Code run as autonomous CLI processes *inside* the same
per-repo Docker image the validation gates use. The host-side harness:

1. starts a long-lived ``sleep infinity`` container with the host workspace
   bind-mounted at :data:`CONTAINER_WORKSPACE`;
2. ``docker exec``s the CLI against that container;
3. cleans the container up when the run finishes.

This mirrors mini-swe-agent's container lifetime without depending on the
mini-swe-agent Python library.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from swe_duel.agents.harness.base import CONTAINER_WORKSPACE
from swe_duel.sandbox.docker_executor import swe_duel_container_name, repo_from_image

_RECOVERY_BUDGET_PER_TURN = 6


@dataclass
class CliRunResult:
    """Outcome of one ``docker exec`` / local CLI invocation."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    killed_for_budget: bool = False


@dataclass
class StreamState:
    """Mutable progress state while a CLI streams JSONL events."""

    step: int = 0
    limit: int = 0
    wall_hit: bool = False
    budget_hit: bool = False
    started: float = field(default_factory=time.monotonic)
    last_step_at: float = field(default_factory=time.monotonic)
    lines: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0


class CliContainer:
    """Long-lived container (or host cwd) for running CLI coding agents."""

    def __init__(
        self,
        *,
        docker_image: str | None,
        host_workspace: Path,
        harness_id: str,
        preserve_paths: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        login_shell: bool = True,
    ) -> None:
        self.docker_image = docker_image
        self.host_workspace = Path(host_workspace).resolve()
        self.harness_id = harness_id
        self.preserve_paths = list(preserve_paths or [])
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self.login_shell = login_shell
        self.container_name: str | None = None
        self._started = False
        # Set True when ensure_running() rebuilt a dead container so callers can
        # drop in-container session state (Claude --resume, etc.).
        self.restarted: bool = False

    def __enter__(self) -> CliContainer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def is_running(self) -> bool:
        """True if the long-lived container exists and is running (host mode: True)."""
        if not self.docker_image:
            return True
        if not self.container_name:
            return False
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def ensure_running(self) -> bool:
        """Recreate the agent container if it died (OOM, ``--rm`` after exit, …).

        Returns True iff a restart happened. Ephemeral dirs under
        ``/var/tmp/swe-duel-*`` are container-local, so any resumed CLI session ids
        become invalid after a restart and must be cleared by the caller.
        """
        self.restarted = False
        if not self.docker_image:
            return False
        if self.is_running():
            return False
        old = self.container_name
        if old:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", old],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
            except Exception:
                pass
        self.container_name = None
        self._started = False
        self.start()
        self.restarted = True
        return True

    def start(self) -> None:
        if self._started:
            return
        if not self.docker_image:
            self._started = True
            return

        name = swe_duel_container_name(repo_from_image(self.docker_image), self.harness_id)
        uid, gid = os.getuid(), os.getgid()
        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--user",
            f"{uid}:{gid}",
            "-v",
            f"{self.host_workspace}:{CONTAINER_WORKSPACE}",
            "--label",
            f"swe_duel.session_pid={os.getpid()}",
            "--label",
            "swe_duel.managed=true",
            "-w",
            CONTAINER_WORKSPACE,
        ]
        for sub in self.preserve_paths:
            sub = sub.strip().strip("/")
            if sub:
                cmd += ["-v", f"{CONTAINER_WORKSPACE}/{sub}"]
        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]
        # Writable homes for the host uid (image HOME may be root-owned).
        # Prefer /var/tmp over /tmp: Codex refuses CODEX_HOME under temporary
        # dirs named "/tmp" when creating helper binaries.
        for key, value in (
            ("HOME", "/var/tmp/swe-duel-home"),
            ("GOCACHE", "/var/tmp/swe-duel-home/.gocache"),
            ("GOPATH", "/var/tmp/swe-duel-home/go"),
            ("npm_config_cache", "/var/tmp/swe-duel-home/.npm"),
            ("CODEX_HOME", "/var/tmp/swe-duel-codex"),
            ("CLAUDE_CONFIG_DIR", "/var/tmp/swe-duel-claude"),
        ):
            if key not in self.env:
                cmd += ["-e", f"{key}={value}"]
        cmd += [self.docker_image, "sleep", "infinity"]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.container_name = name
        self._started = True
        # Ensure the writable home dirs exist for the non-root container user.
        try:
            self.run_command(
                [
                    "bash",
                    "-c",
                    "mkdir -p /var/tmp/swe-duel-home /var/tmp/swe-duel-codex /var/tmp/swe-duel-claude "
                    "/var/tmp/swe-duel-home/.gocache /var/tmp/swe-duel-home/go /var/tmp/swe-duel-home/.npm",
                ],
                timeout=30,
            )
        except Exception:
            pass

    def cleanup(self) -> None:
        if self.container_name is None:
            self._started = False
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except Exception:
            pass
        self.container_name = None
        self._started = False

    def run_command(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        stdin_text: str | None = None,
        on_stdout_line: Callable[[str], None] | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> CliRunResult:
        """Run ``argv`` in the container (or on the host) and stream output."""
        # Revive before every exec — early-finish/recovery reuses this handle and
        # the sleep-infinity main process can be SIGKILL'd (OOM 137) while the
        # harness still holds the dead name.
        if self.docker_image:
            self.ensure_running()
        cmd = self._build_command(argv)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=None if self.docker_image else str(self.host_workspace),
            env=None if self.docker_image else {**os.environ, **self.env},
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        timed_out = False
        killed_for_budget = False
        deadline = (time.monotonic() + timeout) if timeout and timeout > 0 else None

        def _reader(stream: IO[str] | None, sink: list[str], cb: Callable[[str], None] | None) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                sink.append(line)
                if cb is not None:
                    try:
                        cb(line.rstrip("\n"))
                    except Exception:
                        pass

        import threading

        t_out = threading.Thread(
            target=_reader, args=(proc.stdout, stdout_chunks, on_stdout_line), daemon=True
        )
        t_err = threading.Thread(
            target=_reader, args=(proc.stderr, stderr_chunks, on_stderr_line), daemon=True
        )
        t_out.start()
        t_err.start()

        if stdin_text is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            except Exception:
                pass

        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if deadline is not None and time.monotonic() > deadline:
                timed_out = True
                _kill_process_tree(proc)
                break
            if should_stop is not None:
                try:
                    if should_stop():
                        killed_for_budget = True
                        _kill_process_tree(proc)
                        break
                except Exception:
                    pass
            time.sleep(0.05)

        t_out.join(timeout=5)
        t_err.join(timeout=5)
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        try:
            rc_final = proc.wait(timeout=5)
        except Exception:
            rc_final = proc.returncode if proc.returncode is not None else -1

        return CliRunResult(
            return_code=int(rc_final if rc_final is not None else -1),
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            timed_out=timed_out,
            killed_for_budget=killed_for_budget,
        )

    def write_text(self, container_path: str, content: str) -> None:
        """Write a small text file into the container (or host workspace)."""
        if not self.docker_image:
            # Host mode: container_path is absolute; only allow under workspace.
            target = Path(container_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return
        assert self.container_name is not None
        # Prefer a here-doc via bash so we do not need an extra copy step.
        quoted = shlex.quote(container_path)
        script = f"mkdir -p $(dirname {quoted}) && cat > {quoted}"
        result = self.run_command(["bash", "-c", script], stdin_text=content, timeout=30)
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to write {container_path} in container: {result.stderr[:400]}"
            )

    def _build_command(self, argv: Sequence[str]) -> list[str]:
        if not self.docker_image:
            return list(argv)
        assert self.container_name is not None
        # Nested shell so PATH / login profile (nvm, etc.) is available when the
        # image needs it. login_shell mirrors mini-swe-agent behaviour.
        shell_flag = "-lc" if self.login_shell else "-c"
        inner = " ".join(shlex.quote(a) for a in argv)
        cmd = [
            "docker",
            "exec",
            "-i",
            "-w",
            CONTAINER_WORKSPACE,
        ]
        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += [self.container_name, "bash", shell_flag, inner]
        return cmd


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def is_docker_infra_failure(result: CliRunResult | None) -> bool:
    """True when docker exec failed because the container/daemon is gone."""
    if result is None:
        return False
    text = f"{result.stderr or ''}\n{result.stdout or ''}"
    needles = (
        "Error response from daemon",
        "is not running",
        "No such container",
        "Cannot connect to the Docker daemon",
        "docker: Error",
    )
    return any(n in text for n in needles)


def truncate(s: str | None, n: int) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ⏎ ")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + f"… ⟨+{len(s) - n} chars⟩"


def openrouter_api_key() -> str:
    """Prefer SWE_DUEL_OPENROUTER_API_KEY (SWE-Duel convention); fall back to OPENROUTER_API_KEY."""
    key = os.environ.get("SWE_DUEL_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "SWE_DUEL_OPENROUTER_API_KEY (or OPENROUTER_API_KEY) is required for CLI harnesses"
        )
    return key


def iter_json_lines(text: str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL stream, ignoring non-JSON noise lines."""
    import json

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] not in "{[":
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj
