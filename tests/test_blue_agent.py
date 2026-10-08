"""Phase 9: Blue agent (review + fix) tests.

Unit tests exercise `BlueAgent._extract_fix` using the `sample_blue_workspace`
fixture layered on top of `mock_repo`. The integration test drives a real
mini-swe-agent session against mock_repo through OpenRouter.
"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from swe_duel.agents.agent_wrapper import AgentWrapper
from swe_duel.agents.blue import BlueAgent
from swe_duel.agents.red import RedAgent
from swe_duel.config import ModelConfig, RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    ChallengeRecord,
    GateResult,
    GateStatus,
    RedValidationResult,
    Workspace,
)
from swe_duel.sandbox.workspace import WorkspaceManager

from conftest import FIXTURES_DIR, load_integration_models, integration_model_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_REPO_DIR = FIXTURES_DIR / "mock_repo"
SAMPLE_RED_WS = FIXTURES_DIR / "sample_red_workspace"
SAMPLE_BLUE_WS = FIXTURES_DIR / "sample_blue_workspace"
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


def _overlay(src_dir: Path, dst_dir: Path) -> None:
    for src in src_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _build_blue_workspace_from_fixture(tmp_path: Path) -> Workspace:
    """Reference = pristine mock_repo; working = mock_repo + sample_blue_workspace overlay."""
    base = tmp_path / "ws"
    base.mkdir()
    reference = base / "reference"
    working = base / "working"
    shutil.copytree(MOCK_REPO_DIR, reference)
    shutil.copytree(MOCK_REPO_DIR, working)
    _overlay(SAMPLE_BLUE_WS, working)

    return Workspace(
        workspace_id=str(uuid.uuid4()),
        repo_name="mock_repo",
        path=working,
        reference_path=reference,
    )


def _build_blue_workspace_no_changes(tmp_path: Path) -> Workspace:
    base = tmp_path / "ws"
    base.mkdir()
    reference = base / "reference"
    working = base / "working"
    shutil.copytree(MOCK_REPO_DIR, reference)
    shutil.copytree(MOCK_REPO_DIR, working)
    return Workspace(
        workspace_id=str(uuid.uuid4()),
        repo_name="mock_repo",
        path=working,
        reference_path=reference,
    )


def _make_blue_agent(monkeypatch, repos_dir: Path, tmp_root: Path) -> BlueAgent:
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy-key")
    mc = ModelConfig(
        model_id="z-ai/glm-5.3-flash",
        temperature=0.0,
        max_tokens=2048,
    )
    wrapper = AgentWrapper(mc)
    wm = WorkspaceManager(repos_dir=repos_dir, tmp_root=tmp_root)
    return BlueAgent(agent_wrapper=wrapper, workspace_manager=wm, prompt_dir=PROMPT_DIR)


def _build_challenge_record_from_red_fixture(tmp_path: Path) -> ChallengeRecord:
    """Run RedAgent._extract_challenge on sample_red_workspace to get a full RedChallenge,
    then wrap it in a ChallengeRecord for the integration test."""
    base = tmp_path / "red_ws"
    base.mkdir()
    reference = base / "reference"
    working = base / "working"
    shutil.copytree(MOCK_REPO_DIR, reference)
    shutil.copytree(MOCK_REPO_DIR, working)
    _overlay(SAMPLE_RED_WS, working)

    workspace = Workspace(
        workspace_id=str(uuid.uuid4()),
        repo_name="mock_repo",
        path=working,
        reference_path=reference,
    )

    wm = WorkspaceManager(repos_dir=tmp_path, tmp_root=tmp_path / "red_tmp")
    os.environ.setdefault("SWE_DUEL_OPENROUTER_API_KEY", "dummy-key")
    mc = ModelConfig(
        model_id="z-ai/glm-5.3-flash",
        temperature=0.0,
        max_tokens=2048,
    )
    wrapper = AgentWrapper(mc)
    red = RedAgent(agent_wrapper=wrapper, workspace_manager=wm, prompt_dir=PROMPT_DIR)
    challenge = red._extract_challenge(workspace, _fake_trajectory())

    validation = RedValidationResult(
        passed=True,
        gate_results=[
            GateResult(
                gate_name="fixture",
                status=GateStatus.PASSED,
                message="fixture-derived; no gates actually executed",
            )
        ],
        attempt_number=1,
    )
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id="z-ai/glm-5.3-flash",
        repo_name="mock_repo",
        repo_commit_sha="HEAD",
        target_files=challenge.target_files,
        challenge=challenge,
        validation=validation,
        generated_at=datetime.utcnow(),
        generation_cost_usd=0.0,
        generation_retries=0,
    )


# ── Unit tests ─────────────────────────────────────────────


class TestExtractFixFromFixture:
    def test_extract_fix_from_fixture(self, tmp_path: Path, monkeypatch, record):
        workspace = _build_blue_workspace_from_fixture(tmp_path)
        blue = _make_blue_agent(monkeypatch, repos_dir=tmp_path, tmp_root=tmp_path / "tmp")

        fix = blue._extract_fix(workspace, _fake_trajectory())

        record("fix_diff", fix.fix_diff)
        record("modified_file_contents", fix.modified_file_contents)
        record("fix_explanation", fix.fix_explanation)
        record(
            "review_findings",
            [
                {"location": f.location, "severity": f.severity, "description": f.description}
                for f in fix.review_findings
            ],
        )

        assert fix.fix_diff.strip(), "fix_diff must be non-empty"
        assert "src/calculator/basic.py" in fix.modified_file_contents
        assert "a % b" in fix.modified_file_contents["src/calculator/basic.py"]
        assert len(fix.review_findings) >= 1
        assert fix.review_findings[0].severity == "high"
        assert "modulo" in fix.fix_explanation.lower() or "%" in fix.fix_explanation
        # _swe-duel/ files must not leak into modified_file_contents
        assert not any(p.startswith("_swe-duel") for p in fix.modified_file_contents)


class TestExtractFixNoReviewJson:
    def test_extract_fix_no_review_json(self, tmp_path: Path, monkeypatch, record):
        workspace = _build_blue_workspace_from_fixture(tmp_path)
        # Delete the review.json to simulate the missing-file case
        review_path = workspace.path / "_swe-duel" / "review.json"
        if review_path.exists():
            review_path.unlink()

        blue = _make_blue_agent(monkeypatch, repos_dir=tmp_path, tmp_root=tmp_path / "tmp")
        fix = blue._extract_fix(workspace, _fake_trajectory())

        record("fix_diff", fix.fix_diff)
        record("fix_explanation", fix.fix_explanation)
        record("review_findings_count", len(fix.review_findings))

        assert fix.review_findings == []
        assert fix.fix_explanation == "no issues found"
        # fix_diff still reflects the code change that is present in the working copy
        assert fix.fix_diff.strip()


class TestExtractFixNoChanges:
    def test_extract_fix_no_changes(self, tmp_path: Path, monkeypatch, record):
        workspace = _build_blue_workspace_no_changes(tmp_path)
        blue = _make_blue_agent(monkeypatch, repos_dir=tmp_path, tmp_root=tmp_path / "tmp")

        fix = blue._extract_fix(workspace, _fake_trajectory())

        record("fix_diff", fix.fix_diff)
        record("modified_file_contents", fix.modified_file_contents)
        record("fix_explanation", fix.fix_explanation)

        assert fix.fix_diff == ""
        assert fix.modified_file_contents == {}
        assert fix.review_findings == []


# ── Integration test (API call) ────────────────────────────


@pytest.mark.integration
@_skip_no_key
@pytest.mark.parametrize("model_cfg", _INTEGRATION_MODELS, ids=[integration_model_id(m) for m in _INTEGRATION_MODELS])
class TestReviewAndFixMockRepo:
    def test_review_and_fix_mock_repo(self, tmp_path: Path, record, model_cfg):
        repo_config = RepoConfig(
            name="mock_repo",
            url="local://mock_repo",
            commit="HEAD",
            language="python",
            test_command="python -m pytest -x -q",
            docker_image="swe-duel-mock",
        )
        challenge_record = _build_challenge_record_from_red_fixture(tmp_path)

        mc = ModelConfig(
            model_id=model_cfg.model_id,
            temperature=0.0,
            max_tokens=4096,
        )
        record("model_id", mc.model_id)
        wrapper = AgentWrapper(mc)
        wm = WorkspaceManager(repos_dir=FIXTURES_DIR, tmp_root=tmp_path / "blue_tmp")
        blue = BlueAgent(
            agent_wrapper=wrapper,
            workspace_manager=wm,
            prompt_dir=PROMPT_DIR,
            steps=25,
        )

        fix, workspace = blue.review_and_fix(repo_config, challenge_record)

        try:
            record("fix_diff", fix.fix_diff)
            record("fix_explanation", fix.fix_explanation)
            record(
                "review_findings",
                [
                    {"location": f.location, "severity": f.severity, "description": f.description}
                    for f in fix.review_findings
                ],
            )
            record("modified_file_contents", fix.modified_file_contents)
            record("review_json_present", (workspace.path / "_swe-duel" / "review.json").exists())
            record(
                "agent_trajectory",
                {
                    "model_id": fix.agent_trajectory.model_id,
                    "total_steps": fix.agent_trajectory.total_steps,
                    "total_input_tokens": fix.agent_trajectory.total_input_tokens,
                    "total_output_tokens": fix.agent_trajectory.total_output_tokens,
                    "total_cost_usd": fix.agent_trajectory.total_cost_usd,
                    "duration_seconds": fix.agent_trajectory.duration_seconds,
                    "steps": fix.agent_trajectory.steps,
                },
            )

            assert fix.fix_diff.strip(), "fix_diff must be non-empty"
            assert (workspace.path / "_swe-duel" / "review.json").exists()
            assert len(fix.modified_file_contents) >= 1
        finally:
            wm.cleanup(workspace)
