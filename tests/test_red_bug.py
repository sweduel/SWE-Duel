"""Phase 6: Red agent (bug embedding + bug test generation) tests.

Unit tests reuse the `sample_red_workspace` fixture, now updated with a
subtle off-by-sign bug in modulo() and an accompanying `_swe-duel/bug_tests.py`.
The integration test drives a real mini-swe-agent session against mock_repo
through OpenRouter and verifies the bug portion of the output.
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
from swe_duel.models import AgentTrajectory, BugType, Workspace
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
    base = tmp_path / "ws"
    base.mkdir()
    reference = base / "reference"
    working = base / "working"
    shutil.copytree(MOCK_REPO_DIR, reference)
    shutil.copytree(MOCK_REPO_DIR, working)

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


def _valid_metadata() -> dict:
    return {
        "target_files": ["src/foo.py"],
        "exploration_summary": "summary",
        "feature_spec": "spec",
        "feature_rationale": "rationale",
        "bug_type": "logic_error",
        "bug_description": "desc",
        "bug_location": "src/foo.py:fn",
    }


# ── Unit tests ─────────────────────────────────────────────


class TestExtractChallengeWithBug:
    def test_extract_challenge_with_bug(self, tmp_path: Path, monkeypatch, record):
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

        record("bug_type", challenge.bug_type)
        record("bug_description", challenge.bug_description)
        record("bug_location", challenge.bug_location)
        record("bug_test_code", challenge.bug_test_code)

        assert challenge.bug_type == BugType.LOGIC_ERROR
        assert challenge.bug_description and "modulo" in challenge.bug_description
        assert challenge.bug_location == "src/calculator/basic.py:modulo"
        assert challenge.bug_test_code and "def test_" in challenge.bug_test_code


class TestValidateMetadataWithBug:
    def test_validate_metadata_with_bug_valid(self, record):
        data = _valid_metadata()
        record("input", data)
        RedAgent._validate_metadata(data)
        record("result", "accepted")

    def test_validate_metadata_bug_type_invalid(self, record):
        bad = _valid_metadata()
        bad["bug_type"] = "not_a_type"
        record("input", bad)
        with pytest.raises(RedOutputError, match="bug_type") as exc_info:
            RedAgent._validate_metadata(bad)
        record("error_message", str(exc_info.value))

    def test_validate_metadata_bug_description_empty(self, record):
        bad = _valid_metadata()
        bad["bug_description"] = "   "
        record("input", bad)
        with pytest.raises(RedOutputError, match="bug_description") as exc_info:
            RedAgent._validate_metadata(bad)
        record("error_message", str(exc_info.value))

    def test_validate_metadata_bug_location_missing(self, record):
        bad = _valid_metadata()
        del bad["bug_location"]
        record("input", bad)
        with pytest.raises(RedOutputError, match="bug_location") as exc_info:
            RedAgent._validate_metadata(bad)
        record("error_message", str(exc_info.value))


class TestBugTestCodeHasTestFunction:
    def test_bug_test_code_has_test_function(self, record):
        code = "def test_only_bug():\n    assert 1 == 2\n"
        record("test_code", code)
        record("test_function_count", code.count("def test_"))
        RedAgent._validate_tests(code, label="bug", min_test_functions=1)
        record("result", "accepted")

    def test_bug_test_code_empty_rejected(self, record):
        with pytest.raises(RedOutputError):
            RedAgent._validate_tests("", label="bug", min_test_functions=1)
        record("result", "rejected")


# ── Integration test (API call) ────────────────────────────


@pytest.mark.integration
@_skip_no_key
@pytest.mark.parametrize("model_cfg", _INTEGRATION_MODELS, ids=[integration_model_id(m) for m in _INTEGRATION_MODELS])
class TestGenerateChallengeFull:
    def test_generate_challenge_full(self, tmp_path: Path, record, model_cfg):
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
            record("feature_spec", challenge.feature_spec)
            record("feature_test_code", challenge.feature_test_code)
            record("bug_type", challenge.bug_type)
            record("bug_description", challenge.bug_description)
            record("bug_location", challenge.bug_location)
            record("bug_test_code", challenge.bug_test_code)
            record("pr_diff", challenge.pr_diff)
            record("modified_file_contents", challenge.modified_file_contents)
            record("agent_trajectory", {
                "model_id": challenge.agent_trajectory.model_id,
                "total_steps": challenge.agent_trajectory.total_steps,
                "total_input_tokens": challenge.agent_trajectory.total_input_tokens,
                "total_output_tokens": challenge.agent_trajectory.total_output_tokens,
                "total_cost_usd": challenge.agent_trajectory.total_cost_usd,
                "duration_seconds": challenge.agent_trajectory.duration_seconds,
                "steps": challenge.agent_trajectory.steps,
            })

            assert (workspace.path / "_swe-duel" / "bug_tests.py").exists()
            assert isinstance(challenge.bug_type, BugType)
            assert challenge.bug_description and challenge.bug_description.strip()
            assert challenge.bug_location and challenge.bug_location.strip()
            bug_fn_count = challenge.bug_test_code.count("def test_")
            assert bug_fn_count >= 1, f"expected ≥1 bug test_ function, got {bug_fn_count}"
        finally:
            wm.cleanup(workspace)
