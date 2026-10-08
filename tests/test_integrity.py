"""Phase 8: Challenge freshness tests."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path


from swe_duel.challenge_bank.integrity import (
    prune_stale_challenges,
    verify_challenge_freshness,
)
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    BugType,
    ChallengeRecord,
    GateResult,
    GateStatus,
    RedChallenge,
    RedValidationResult,
)


def _record(commit: str, repo: str = "flask") -> ChallengeRecord:
    traj = AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, model_id="m", duration_seconds=0.0,
    )
    ch = RedChallenge(
        target_files=["f.py"],
        exploration_summary="e",
        feature_spec="s",
        feature_rationale="r",
        pr_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n x\n+y\n",
        modified_file_contents={"f.py": "c"},
        original_file_contents={"f.py": "o"},
        feature_test_code="def test_a():\n    assert 1",
        bug_type=BugType.LOGIC_ERROR,
        bug_description="d",
        bug_location="l",
        bug_test_code="def test_b():\n    assert 1",
        agent_trajectory=traj,
    )
    val = RedValidationResult(
        passed=True,
        gate_results=[GateResult("g", GateStatus.PASSED, "ok")],
        attempt_number=1,
    )
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id="m1",
        repo_name=repo,
        repo_commit_sha=commit,
        target_files=["f.py"],
        challenge=ch,
        validation=val,
        generated_at=datetime(2026, 4, 19),
        generation_cost_usd=0.0,
        generation_retries=0,
    )


def _repo(commit: str, name: str = "flask") -> RepoConfig:
    return RepoConfig(
        name=name,
        url="local://x",
        commit=commit,
        test_command="pytest",
        docker_image="swe-duel-mock",
    )


def test_verify_freshness_match(record):
    rec = _record(commit="abc")
    cfg = _repo(commit="abc")
    result = verify_challenge_freshness(rec, cfg)
    record("fresh", result)
    assert result is True


def test_verify_freshness_mismatch(record):
    rec = _record(commit="abc")
    cfg = _repo(commit="def")
    result = verify_challenge_freshness(rec, cfg)
    record("fresh", result)
    assert result is False


def test_prune_stale(tmp_path: Path, record):
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    fresh_ids = [store.store(_record(commit="abc")) for _ in range(2)]
    stale_id = store.store(_record(commit="old"))
    repo_configs = {"flask": _repo(commit="abc")}
    pruned = prune_stale_challenges(store, repo_configs)
    record("pruned_count", pruned)
    record("remaining_pools", store.list_pools())
    assert pruned == 1
    remaining = [r.challenge_id for r in store.query()]
    assert set(remaining) == set(fresh_ids)
    assert stale_id not in remaining
