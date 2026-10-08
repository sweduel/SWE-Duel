"""Tests for TestRunner — Phase 2 verification."""

from __future__ import annotations

import subprocess

import pytest

from swe_duel.config import RepoConfig
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner

MOCK_IMAGE = "swe-duel-mock:latest"


def _image_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", name],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.fixture()
def mock_repo_config() -> RepoConfig:
    return RepoConfig(
        name="mock",
        url="",
        commit="",
        test_command="python -m pytest tests/ -x -q --tb=short --json-report --json-report-file=report.json",
        docker_image=MOCK_IMAGE,
    )


@pytest.fixture()
def executor() -> DockerExecutor:
    if not _image_exists(MOCK_IMAGE):
        pytest.skip(f"Docker image {MOCK_IMAGE} not built — run: swe-duel setup docker")
    return DockerExecutor(docker_image=MOCK_IMAGE, timeout_s=60, memory_mb=256)


@pytest.fixture()
def runner(executor: DockerExecutor, mock_repo_config: RepoConfig) -> TestRunner:
    return TestRunner(executor=executor, repo_config=mock_repo_config, retries=1)


def _ter(result) -> dict:
    return {
        "passed": result.passed,
        "total": result.total,
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "error_count": result.error_count,
        "failure_messages": result.failure_messages,
        "duration_ms": result.duration_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def test_run_existing_tests_all_pass(runner: TestRunner, record) -> None:
    result = runner.run_existing_tests(file_overrides={})
    record("result", _ter(result))
    assert result.passed is True
    assert result.total > 0
    assert result.failed_count == 0


def test_run_existing_tests_with_breakage(runner: TestRunner, record) -> None:
    broken = (
        "def add(a, b): return a - b\n"
        "def subtract(a, b): return a - b\n"
        "def multiply(a, b): return a * b\n"
        "def divide(a, b): return a / b\n"
    )
    overrides = {"src/calculator/basic.py": broken}
    result = runner.run_existing_tests(file_overrides=overrides)
    record("file_overrides", overrides)
    record("result", _ter(result))
    assert result.passed is False
    assert result.failed_count > 0


def test_run_injected_tests_pass(runner: TestRunner, record) -> None:
    test_code = "def test_trivial(): assert 1 + 1 == 2\n"
    result = runner.run_injected_tests(
        file_overrides={},
        test_code=test_code,
        test_filename="test_swe_duel_trivial.py",
    )
    record("test_code", test_code)
    record("result", _ter(result))
    assert result.passed is True
    assert result.passed_count >= 1


def test_run_injected_tests_fail(runner: TestRunner, record) -> None:
    test_code = "def test_will_fail(): assert False\n"
    result = runner.run_injected_tests(
        file_overrides={},
        test_code=test_code,
        test_filename="test_swe_duel_fail.py",
    )
    record("test_code", test_code)
    record("result", _ter(result))
    assert result.passed is False
    assert result.failed_count == 1


def test_parse_pytest_json_valid(runner: TestRunner, record) -> None:
    from swe_duel.models import ExecutionResult

    sample_json = """{
        "summary": {"total": 5, "passed": 3, "failed": 2, "error": 0},
        "tests": [
            {"nodeid": "test_a.py::test_1", "outcome": "passed"},
            {"nodeid": "test_a.py::test_2", "outcome": "failed",
             "call": {"crash": {"message": "AssertionError"}}},
            {"nodeid": "test_b.py::test_3", "outcome": "failed",
             "call": {"crash": {"message": "ValueError"}}}
        ]
    }"""
    exec_result = ExecutionResult(
        return_code=1, stdout=sample_json, stderr="", timed_out=False, duration_ms=100,
    )
    parsed = runner._parse_pytest_json(sample_json, exec_result)
    record("input_json", sample_json)
    record("parsed", _ter(parsed))

    assert parsed.total == 5
    assert parsed.passed_count == 3
    assert parsed.failed_count == 2
    assert parsed.error_count == 0
    assert parsed.passed is False
    assert len(parsed.failure_messages) == 2


def test_parse_pytest_json_fallback(runner: TestRunner, record) -> None:
    from swe_duel.models import ExecutionResult

    plain_output = """\
test_a.py ...F.
FAILED test_a.py::test_2 - AssertionError
1 failed, 3 passed in 0.5s
"""
    exec_result = ExecutionResult(
        return_code=1, stdout=plain_output, stderr="", timed_out=False, duration_ms=500,
    )
    parsed = runner._parse_pytest_json(plain_output, exec_result)
    record("plain_output", plain_output)
    record("parsed", _ter(parsed))

    assert parsed.passed is False
    assert parsed.passed_count == 3
    assert parsed.failed_count == 1
