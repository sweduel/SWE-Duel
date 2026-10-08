"""Phase 5: Red agent (feature-only) tests.

Unit tests exercise the extraction and validation logic using the
`sample_red_workspace` fixture. The integration test drives a real
mini-swe-agent session against mock_repo through OpenRouter.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from swe_duel.agents.agent_wrapper import AgentWrapper
from swe_duel.agents.red import RedAgent, RedOutputError
from swe_duel.config import ModelConfig, RepoConfig
from swe_duel.models import AgentTrajectory, Workspace
from swe_duel.sandbox.workspace import WorkspaceManager

from conftest import FIXTURES_DIR, load_integration_models, integration_model_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_REPO_DIR = FIXTURES_DIR / "mock_repo"
SAMPLE_RED_WS = FIXTURES_DIR / "sample_red_workspace"
PROMPT_DIR = PROJECT_ROOT / "src" / "swe_duel" / "agents" / "prompts"

HAS_API_KEY = bool(os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"))
_skip_no_key = pytest.mark.skipif(not HAS_API_KEY, reason="SWE_DUEL_OPENROUTER_API_KEY not set")

_INTEGRATION_MODELS = load_integration_models() if HAS_API_KEY else []


# ── Helpers ────────────────────────────────────────────────


def _fake_trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[],
        total_steps=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        model_id="test/mock",
        duration_seconds=0.0,
    )


def _build_workspace_from_fixture(tmp_path: Path) -> Workspace:
    """Stage a Workspace: reference = mock_repo, working = mock_repo + sample overrides."""
    base = tmp_path / "ws"
    base.mkdir()
    reference = base / "reference"
    working = base / "working"
    shutil.copytree(MOCK_REPO_DIR, reference)
    shutil.copytree(MOCK_REPO_DIR, working)

    # Overlay the sample_red_workspace contents onto the working copy.
    for src in SAMPLE_RED_WS.rglob("*"):
        if src.is_file():
            rel = src.relative_to(SAMPLE_RED_WS)
            dst = working / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return Workspace(
        workspace_id=str(uuid.uuid4()),
        repo_name="mock_repo",
        path=working,
        reference_path=reference,
    )


def _make_red_agent(monkeypatch, repos_dir: Path, tmp_root: Path) -> RedAgent:
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy-key")
    mc = ModelConfig(
        model_id="deepseek/deepseek-v4-flash-0731",
        temperature=0.0,
        max_tokens=2048,
    )
    wrapper = AgentWrapper(mc)
    wm = WorkspaceManager(repos_dir=repos_dir, tmp_root=tmp_root)
    return RedAgent(agent_wrapper=wrapper, workspace_manager=wm, prompt_dir=PROMPT_DIR)


# ── Unit tests (no API call) ───────────────────────────────


class TestExtractChallengeFromFixture:
    def test_extract_challenge_from_fixture(self, tmp_path: Path, monkeypatch, record):
        workspace = _build_workspace_from_fixture(tmp_path)
        red = _make_red_agent(monkeypatch, repos_dir=tmp_path, tmp_root=tmp_path / "tmp")

        metadata = red._load_full_metadata(workspace)
        feature_test_code = red._load_feature_tests(workspace)
        bug_test_code = red._load_bug_tests(workspace)
        challenge = red._extract_challenge(
            workspace=workspace,
            metadata=metadata,
            feature_test_code=feature_test_code,
            bug_test_code=bug_test_code,
            feature_only_file_contents={},
            feature_trajectory=_fake_trajectory(),
            bug_trajectory=_fake_trajectory(),
        )

        record("target_files", challenge.target_files)
        record("exploration_summary", challenge.exploration_summary)
        record("feature_spec", challenge.feature_spec)
        record("feature_rationale", challenge.feature_rationale)
        record("pr_diff", challenge.pr_diff)
        record("modified_file_contents", challenge.modified_file_contents)
        record("original_file_contents", challenge.original_file_contents)
        record("feature_test_code", challenge.feature_test_code)
        record("bug_type", challenge.bug_type)
        record("bug_description", challenge.bug_description)
        record("bug_location", challenge.bug_location)
        record("bug_test_code", challenge.bug_test_code)

        assert challenge.target_files == ["src/calculator/basic.py"]
        assert challenge.exploration_summary
        assert challenge.feature_spec
        assert challenge.feature_rationale
        assert "def modulo" in "".join(challenge.modified_file_contents.values())
        assert challenge.pr_diff.strip(), "pr_diff must be non-empty"
        assert "src/calculator/basic.py" in challenge.modified_file_contents
        assert "src/calculator/basic.py" in challenge.original_file_contents
        # Bug fields populated since Phase 6
        assert challenge.bug_type is not None
        assert challenge.bug_description
        assert challenge.bug_location
        assert challenge.bug_test_code
        # Feature tests captured
        assert "def test_modulo_positive" in challenge.feature_test_code
        # _swe-duel/ files must not leak into file maps
        assert not any(p.startswith("_swe-duel") for p in challenge.modified_file_contents)


    def test_build_feature_only_challenge_has_pr_diff(self, tmp_path: Path, monkeypatch, record):
        workspace = _build_workspace_from_fixture(tmp_path)
        red = _make_red_agent(monkeypatch, repos_dir=tmp_path, tmp_root=tmp_path / "tmp")

        metadata = red._load_feature_metadata(workspace)
        feature_test_code = red._load_feature_tests(workspace)
        feature_only_file_contents = red._snapshot_non_swe_duel_files(workspace)
        challenge = red._build_feature_only_challenge(
            workspace=workspace,
            metadata=metadata,
            feature_test_code=feature_test_code,
            feature_only_file_contents=feature_only_file_contents,
            feature_trajectory=_fake_trajectory(),
        )

        record("pr_diff", challenge.pr_diff)
        record("modified_file_contents", challenge.modified_file_contents)
        assert challenge.pr_diff.strip(), "feature-only pr_diff must be non-empty"
        assert not challenge.bug_test_code
        assert challenge.bug_type is None
        assert "src/calculator/basic.py" in challenge.modified_file_contents
        assert "def modulo" in challenge.modified_file_contents["src/calculator/basic.py"]


class TestValidateMetadata:
    def _valid(self) -> dict:
        return {
            "target_files": ["src/foo.py"],
            "exploration_summary": "summary",
            "feature_spec": "spec",
            "feature_rationale": "rationale",
            "bug_type": "logic_error",
            "bug_description": "desc",
            "bug_location": "src/foo.py:fn",
        }

    def test_validate_metadata_valid(self, record):
        data = self._valid()
        record("input", data)
        RedAgent._validate_metadata(data)
        record("result", "accepted")

    def test_validate_metadata_missing_key(self, record):
        bad = self._valid()
        del bad["feature_spec"]
        record("input", bad)
        with pytest.raises(RedOutputError, match="feature_spec") as exc_info:
            RedAgent._validate_metadata(bad)
        record("error_message", str(exc_info.value))

    def test_validate_metadata_empty_target_files(self, record):
        bad = self._valid()
        bad["target_files"] = []
        record("input", bad)
        with pytest.raises(RedOutputError) as exc_info:
            RedAgent._validate_metadata(bad)
        record("error_message", str(exc_info.value))


class TestValidateTests:
    def test_validate_tests_sufficient(self, record):
        code = (
            "def test_a():\n    assert 1\n\n"
            "def test_b():\n    assert 2\n\n"
            "def test_c():\n    assert 3\n"
        )
        record("test_code", code)
        record("test_function_count", code.count("def test_"))
        RedAgent._validate_tests(code, label="feature", min_test_functions=3)
        record("result", "accepted")

    def test_validate_tests_insufficient(self, record):
        code = "def test_only():\n    assert True\n"
        record("test_code", code)
        record("test_function_count", code.count("def test_"))
        with pytest.raises(RedOutputError) as exc_info:
            RedAgent._validate_tests(code, label="feature", min_test_functions=3)
        record("error_message", str(exc_info.value))


# ── Integration test (API call) ────────────────────────────


@pytest.mark.integration
@_skip_no_key
@pytest.mark.parametrize("model_cfg", _INTEGRATION_MODELS, ids=[integration_model_id(m) for m in _INTEGRATION_MODELS])
class TestGenerateFeature:
    def test_generate_feature_mock_repo(self, tmp_path: Path, record, model_cfg):
        # Arrange: use fixtures/ as repos_dir so "mock_repo" resolves correctly.
        repo_config = RepoConfig(
            name="mock_repo",
            url="local://mock_repo",
            commit="HEAD",
            language="python",
            test_command="python -m pytest -x -q",
            docker_image="swe-duel-mock",
        )
        mc = ModelConfig(
            model_id=model_cfg.model_id,
            temperature=0.0,
            max_tokens=4096,
        )
        record("model_id", mc.model_id)
        wrapper = AgentWrapper(mc)
        wm = WorkspaceManager(repos_dir=FIXTURES_DIR, tmp_root=tmp_path)
        red = RedAgent(
            agent_wrapper=wrapper,
            workspace_manager=wm,
            prompt_dir=PROMPT_DIR,
            feature_steps=30,
            bug_steps=30,
        )

        challenge, workspace = red.generate_challenge(repo_config)

        try:
            record("target_files", challenge.target_files)
            record("exploration_summary", challenge.exploration_summary)
            record("feature_spec", challenge.feature_spec)
            record("feature_rationale", challenge.feature_rationale)
            record("pr_diff", challenge.pr_diff)
            record("modified_file_contents", challenge.modified_file_contents)
            record("feature_test_code", challenge.feature_test_code)
            record("agent_trajectory", {
                "model_id": challenge.agent_trajectory.model_id,
                "total_steps": challenge.agent_trajectory.total_steps,
                "total_input_tokens": challenge.agent_trajectory.total_input_tokens,
                "total_output_tokens": challenge.agent_trajectory.total_output_tokens,
                "total_cost_usd": challenge.agent_trajectory.total_cost_usd,
                "duration_seconds": challenge.agent_trajectory.duration_seconds,
                "steps": challenge.agent_trajectory.steps,
            })
            # Required artifacts exist in workspace
            assert (workspace.path / "_swe-duel" / "metadata.json").exists()
            assert (workspace.path / "_swe-duel" / "feature_tests.py").exists()
            # Challenge assembled correctly
            assert challenge.pr_diff.strip(), "pr_diff must be non-empty"
            assert len(challenge.modified_file_contents) >= 1
            fn_count = challenge.feature_test_code.count("def test_")
            assert fn_count >= 3, f"expected ≥3 test_ functions, got {fn_count}"
            assert challenge.target_files
            assert challenge.feature_spec
        finally:
            wm.cleanup(workspace)


