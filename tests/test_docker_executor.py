"""Tests for DockerExecutor — Phase 2 verification."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

import swe_duel.sandbox.docker_executor as docker_executor
from swe_duel.sandbox.docker_executor import DockerExecutor

MOCK_IMAGE = "swe-duel-mock:latest"
BASE_IMAGE = "swe-duel-base:latest"


def _image_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", name],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.fixture()
def mock_executor() -> DockerExecutor:
    if not _image_exists(MOCK_IMAGE):
        pytest.skip(f"Docker image {MOCK_IMAGE} not built — run: swe-duel setup docker")
    return DockerExecutor(docker_image=MOCK_IMAGE, timeout_s=30, memory_mb=256)


@pytest.fixture()
def base_executor() -> DockerExecutor:
    if not _image_exists(BASE_IMAGE):
        pytest.skip(f"Docker image {BASE_IMAGE} not built — run: swe-duel setup docker")
    return DockerExecutor(docker_image=BASE_IMAGE, timeout_s=30, memory_mb=256)


def _er(result) -> dict:
    return {
        "return_code": result.return_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "duration_ms": result.duration_ms,
    }


def test_health_check(base_executor: DockerExecutor, record) -> None:
    ok = base_executor.health_check()
    record("health_check", ok)
    assert ok is True


def test_execute_simple_command(base_executor: DockerExecutor, record) -> None:
    result = base_executor.execute(file_overrides={}, command="echo hello")
    record("command", "echo hello")
    record("result", _er(result))
    assert result.return_code == 0
    assert result.stdout.strip() == "hello"
    assert not result.timed_out


def test_execute_timeout(base_executor: DockerExecutor, record) -> None:
    executor = DockerExecutor(docker_image=BASE_IMAGE, timeout_s=5, memory_mb=256)
    result = executor.execute(file_overrides={}, command="sleep 300")
    record("command", "sleep 300")
    record("timeout_s", 5)
    record("result", _er(result))
    assert result.timed_out is True


def test_execute_network_disabled(base_executor: DockerExecutor, record) -> None:
    cmd = "curl -s --max-time 5 http://example.com 2>&1 || echo 'network_failed'"
    result = base_executor.execute(file_overrides={}, command=cmd)
    record("command", cmd)
    record("result", _er(result))
    assert "network_failed" in result.stdout or result.return_code != 0


def test_execute_file_overrides(mock_executor: DockerExecutor, record) -> None:
    content = 'def greet(name): return f"hello {name}"\n'
    overrides = {"src/calculator/basic.py": content}
    result = mock_executor.execute(
        file_overrides=overrides,
        command="cat src/calculator/basic.py",
    )
    record("file_overrides", overrides)
    record("result", _er(result))
    assert result.return_code == 0
    assert "greet" in result.stdout


def test_execute_extra_files(mock_executor: DockerExecutor, record) -> None:
    test_code = "def test_extra(): assert 1 + 1 == 2\n"
    extra = {"test_swe_duel_extra.py": test_code}
    result = mock_executor.execute(
        file_overrides={},
        command="cat test_swe_duel_extra.py",
        extra_files=extra,
    )
    record("extra_files", extra)
    record("result", _er(result))
    assert result.return_code == 0
    assert "test_extra" in result.stdout


def test_execute_in_workspace(mock_executor: DockerExecutor, record) -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="swe_duel_workspace_") as tmpdir:
        ws = Path(tmpdir)
        (ws / "hello.txt").write_text("world")
        result = mock_executor.execute_in_workspace(
            workspace_path=ws,
            command="cat hello.txt",
        )
        record("workspace_contents", {"hello.txt": "world"})
        record("result", _er(result))
        assert result.return_code == 0
        assert "world" in result.stdout


# ── daemon-free unit tests: HTTP read-timeout resilience ──────────────────
#
# docker-py applies its client HTTP timeout to every request that does not
# override one — including get_archive's tar stream and the follow-logs
# stream, which cannot be configured per call. A loaded daemon can stall
# those past the timeout and raise requests.exceptions.ReadTimeout (NOT a
# docker.errors.APIError), which before the retry wrapper escaped and killed
# a whole generation pool. These tests fake the docker client; no images or
# daemon are needed.


def _read_timeout() -> requests.exceptions.ReadTimeout:
    return requests.exceptions.ReadTimeout(
        "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out. (read timeout=60)"
    )


def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_executor, "_DOCKER_HTTP_RETRY_DELAYS_S", (0.0, 0.0))


def _bare_executor(fake_client: Any) -> DockerExecutor:
    ex = DockerExecutor.__new__(DockerExecutor)
    ex.docker_image = MOCK_IMAGE
    ex.timeout_s = 30
    ex.memory_mb = 256
    ex.cpu_limit = 1.0
    ex._client = fake_client
    return ex


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return bio.getvalue()


class _FakeSeedContainer:
    def __init__(self, archive_results: list[Any]) -> None:
        self._archive_results = list(archive_results)
        self.removed = 0

    def get_archive(self, path: str) -> Any:
        result = self._archive_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def remove(self, force: bool = False) -> None:
        self.removed += 1


class _FakeContainers:
    def __init__(
        self, create_results: list[Any] | None = None, run_results: list[Any] | None = None
    ) -> None:
        self._create_results = list(create_results or [])
        self._run_results = list(run_results or [])
        self.create_calls = 0
        self.run_calls = 0

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.create_calls += 1
        return self._create_results.pop(0)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.run_calls += 1
        result = self._run_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeRunContainer:
    def __init__(
        self,
        *,
        wait_result: dict[str, int] | None = None,
        stream_chunks: list[bytes] | None = None,
        stream_exc: Exception | None = None,
        full_logs: bytes = b"",
        logs_calls: list[dict[str, bool]] | None = None,
    ) -> None:
        self._wait_result = wait_result or {"StatusCode": 0}
        self._stream_chunks = list(stream_chunks or [])
        self._stream_exc = stream_exc
        self._full_logs = full_logs
        self._logs_calls = logs_calls
        self.removed = 0

    def logs(
        self, stdout: bool = True, stderr: bool = True, stream: bool = False, follow: bool = False
    ) -> Any:
        if self._logs_calls is not None:
            self._logs_calls.append({"stream": stream, "follow": follow})
        if stream:

            def _gen():
                for chunk in self._stream_chunks:
                    yield chunk
                if self._stream_exc is not None:
                    raise self._stream_exc

            return _gen()
        return self._full_logs

    def wait(self, timeout: float | None = None) -> dict[str, int]:
        return dict(self._wait_result)

    def kill(self) -> None:
        pass

    def remove(self, force: bool = False) -> None:
        self.removed += 1


def test_client_uses_extended_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    fake_client = SimpleNamespace(images=SimpleNamespace(get=lambda name: name))
    monkeypatch.setattr(
        docker_executor.docker,
        "from_env",
        lambda **kwargs: (seen.update(kwargs), fake_client)[1],
    )
    DockerExecutor(docker_image=MOCK_IMAGE)
    assert seen["timeout"] == docker_executor._DOCKER_HTTP_TIMEOUT_S


def test_retry_docker_recovers_from_read_timeout(monkeypatch: pytest.MonkeyPatch, record) -> None:
    _no_backoff(monkeypatch)
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _read_timeout()
        return "ok"

    assert docker_executor._retry_docker(flaky) == "ok"
    record("attempts", len(calls))
    assert len(calls) == 3


def test_retry_docker_reraises_after_exhaustion(monkeypatch: pytest.MonkeyPatch, record) -> None:
    monkeypatch.setattr(docker_executor, "_DOCKER_HTTP_RETRY_DELAYS_S", (0.0,))
    calls: list[int] = []

    def always_down() -> None:
        calls.append(1)
        raise requests.exceptions.ConnectionError("daemon socket reset")

    with pytest.raises(requests.exceptions.ConnectionError):
        docker_executor._retry_docker(always_down)
    record("attempts", len(calls))
    assert len(calls) == 2


def test_seed_workspace_retries_read_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record
) -> None:
    _no_backoff(monkeypatch)

    def _raising_stream() -> Any:
        raise _read_timeout()
        yield  # pragma: no cover

    archive = _tar_bytes({"workspace/hello.txt": b"seeded"})
    # First attempt: create succeeds, tar stream dies on first read.
    seed_containers = [
        _FakeSeedContainer([(_raising_stream(), {})]),
        _FakeSeedContainer([(iter([archive]), {})]),
    ]
    fake_client = SimpleNamespace(containers=_FakeContainers(create_results=seed_containers))
    executor = _bare_executor(fake_client)

    workspace = tmp_path / "workspace"
    executor._seed_workspace_from_image(workspace)

    record("create_calls", fake_client.containers.create_calls)
    record("removed", [c.removed for c in seed_containers])
    assert fake_client.containers.create_calls == 2
    assert all(c.removed == 1 for c in seed_containers)
    assert (workspace / "hello.txt").read_bytes() == b"seeded"


def test_run_container_replaces_truncated_logs_after_stream_dies(tmp_path: Path, record) -> None:
    container = _FakeRunContainer(
        wait_result={"StatusCode": 7},
        stream_chunks=[b"partial line\n"],
        stream_exc=_read_timeout(),
        full_logs=b"full authoritative log\n",
    )
    executor = _bare_executor(SimpleNamespace(containers=_FakeContainers(run_results=[container])))

    result = executor._run_container(tmp_path, "some test command")

    record("result", _er(result))
    assert result.return_code == 7
    assert result.stdout == "full authoritative log\n"
    assert container.removed == 1


def test_run_container_keeps_streamed_logs_on_happy_path(tmp_path: Path, record) -> None:
    logs_calls: list[dict[str, bool]] = []
    container = _FakeRunContainer(
        wait_result={"StatusCode": 0},
        stream_chunks=[b"all streamed output\n"],
        logs_calls=logs_calls,
        full_logs=b"should not be fetched\n",
    )
    executor = _bare_executor(SimpleNamespace(containers=_FakeContainers(run_results=[container])))

    result = executor._run_container(tmp_path, "some test command")

    record("result", _er(result))
    record("logs_calls", logs_calls)
    assert result.return_code == 0
    assert result.stdout == "all streamed output\n"
    # Exactly one logs call: the follow stream. No post-exit refetch needed.
    assert logs_calls == [{"stream": True, "follow": True}]


def test_run_container_retries_start_read_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record
) -> None:
    _no_backoff(monkeypatch)
    container = _FakeRunContainer(
        wait_result={"StatusCode": 0},
        stream_chunks=[b"recovered output\n"],
    )
    fake_containers = _FakeContainers(run_results=[_read_timeout(), container])
    executor = _bare_executor(SimpleNamespace(containers=fake_containers))

    result = executor._run_container(tmp_path, "some test command")

    record("run_calls", fake_containers.run_calls)
    record("result", _er(result))
    assert fake_containers.run_calls == 2
    assert result.return_code == 0
    assert result.stdout == "recovered output\n"
