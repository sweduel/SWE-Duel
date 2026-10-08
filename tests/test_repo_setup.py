"""Phase 1 tests: repo setup and mock repo."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from swe_duel.cli.setup import _select_repos, clone_or_repin
from swe_duel.config import RepoConfig


def _toy_repo_config(url: str) -> RepoConfig:
    return RepoConfig(
        name="toy",
        url=url,
        commit="v1",
        test_command="true",
        docker_image="swe-duel-toy:latest",
    )


class TestSetupReposCommand:
    def test_select_repos_passthrough_and_filter(self, record):
        rc = _toy_repo_config("https://example.invalid/toy")
        cfgs = {"toy": rc, "other": None}
        assert _select_repos({"toy": rc}, None) == {"toy": rc}
        assert _select_repos(cfgs, ["toy"]) == {"toy": rc}
        record("selected", sorted(_select_repos(cfgs, ["toy"])))
        with pytest.raises(SystemExit):
            _select_repos(cfgs, ["missing"])

    def test_clone_or_repin(self, tmp_path: Path, record):
        # Local "remote": a git repo with one commit tagged v1.
        remote = tmp_path / "remote"
        remote.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["git", "init", "-q"], ["git", "checkout", "-q", "-b", "main"],
                    ["git", "commit", "--allow-empty", "-m", "init", "-q"],
                    ["git", "tag", "v1"]):
            subprocess.run(cmd, cwd=remote, env=env, check=True, capture_output=True)
        rc = _toy_repo_config(str(remote))

        repos_dir = tmp_path / "repos"
        cloned = clone_or_repin("toy", rc, repos_dir)
        record("first_call_cloned", cloned)
        assert (repos_dir / "toy").is_dir()

        # Idempotent re-run: already at the pinned ref, no fetch.
        repinned = clone_or_repin("toy", rc, repos_dir)
        record("second_call_cloned", repinned)
        assert repinned is False

    def test_setup_cli_help_smoke(self, record):
        from swe_duel.cli import setup as setup_cli

        record("has_repos_cmd", callable(setup_cli.cmd_repos))
        record("has_docker_cmd", callable(setup_cli.cmd_docker))
        assert callable(setup_cli.cmd_repos)
        assert callable(setup_cli.cmd_docker)


class TestMockRepoStructure:
    def test_mock_repo_structure(self, mock_repo_dir: Path, record):
        expected = [
            "pyproject.toml",
            "src/calculator/__init__.py",
            "src/calculator/basic.py",
            "src/calculator/advanced.py",
            "tests/test_basic.py",
            "tests/test_advanced.py",
        ]
        presence = {p: (mock_repo_dir / p).exists() for p in expected}
        record("file_presence", presence)
        for p, exists in presence.items():
            assert exists, p


class TestMockRepoTestsPass:
    def test_mock_repo_tests_pass_locally(self, mock_repo_dir: Path, record):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            cwd=mock_repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        record("return_code", result.returncode)
        record("stdout", result.stdout)
        record("stderr", result.stderr)
        assert result.returncode == 0, f"Tests failed:\n{result.stdout}\n{result.stderr}"
        output = result.stdout
        import re

        match = re.search(r"(\d+) passed", output)
        record("passed_count", int(match.group(1)) if match else None)
        assert match, f"Could not find pass count in:\n{output}"
        assert int(match.group(1)) == 20
