"""Phase 8: ChallengeStore tests — no LLM, no Docker."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from swe_duel.challenge_bank.store import (
    ChallengeNotFoundError,
    ChallengeStore,
    InsufficientChallengesError,
)
from swe_duel.models import (
    AgentTrajectory,
    BugType,
    ChallengeRecord,
    GateResult,
    GateStatus,
    RedChallenge,
    RedValidationResult,
)


# ── builders ───────────────────────────────────────────────


def _trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[{"thought": "t", "action": "a", "observation": "o"}],
        total_steps=1,
        total_input_tokens=10,
        total_output_tokens=20,
        total_cost_usd=0.01,
        model_id="test/mock",
        duration_seconds=0.5,
    )


def _challenge(target_files: list[str]) -> RedChallenge:
    return RedChallenge(
        target_files=list(target_files),
        exploration_summary="explored",
        feature_spec="spec",
        feature_rationale="rationale",
        pr_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n x\n+y\n",
        modified_file_contents={target_files[0]: "content"},
        original_file_contents={target_files[0]: "orig"},
        feature_test_code="def test_a():\n    assert 1",
        bug_type=BugType.LOGIC_ERROR,
        bug_description="d",
        bug_location="loc",
        bug_test_code="def test_b():\n    assert 1",
        agent_trajectory=_trajectory(),
    )


def _validation() -> RedValidationResult:
    return RedValidationResult(
        passed=True,
        gate_results=[
            GateResult(
                gate_name="gate_diff_valid",
                status=GateStatus.PASSED,
                message="ok",
                duration_ms=1,
            )
        ],
        attempt_number=1,
    )


def _record(
    red_model: str = "anthropic/claude-x",
    repo: str = "flask",
    commit: str = "abc123",
    target_files: list[str] | None = None,
    retries: int = 0,
    cost: float = 0.01,
    harness: str = "mini-swe-agent",
    effort: str = "",
    provider: str = "",
) -> ChallengeRecord:
    return ChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id=red_model,
        repo_name=repo,
        repo_commit_sha=commit,
        target_files=target_files or ["src/foo.py"],
        challenge=_challenge(target_files or ["src/foo.py"]),
        validation=_validation(),
        generated_at=datetime(2026, 4, 19, 12, 0, 0),
        generation_cost_usd=cost,
        generation_retries=retries,
        red_harness_id=harness,
        red_reasoning_effort=effort,
        red_provider=provider,
    )


@pytest.fixture
def bank_dir(tmp_path: Path) -> Path:
    return tmp_path / "bank"


@pytest.fixture
def store(bank_dir: Path) -> ChallengeStore:
    return ChallengeStore(bank_dir=bank_dir)


# ── tests ──────────────────────────────────────────────────


def test_store_and_get(store: ChallengeStore, record):
    rec = _record()
    cid = store.store(rec)
    loaded = store.get(cid)
    record("stored_id", cid)
    record("loaded", loaded)
    assert loaded.challenge_id == rec.challenge_id
    assert loaded.red_model_id == rec.red_model_id
    assert loaded.challenge.bug_type == BugType.LOGIC_ERROR
    assert loaded.validation.gate_results[0].status == GateStatus.PASSED
    assert loaded.generated_at == rec.generated_at


def test_store_updates_index(store: ChallengeStore, bank_dir: Path, record):
    for _ in range(3):
        store.store(_record(red_model="m1", repo="flask"))
    idx = json.loads((bank_dir / "index.json").read_text())
    record("index", idx)
    assert "m1#mini-swe-agent::flask" in idx["pools"]
    assert len(idx["pools"]["m1#mini-swe-agent::flask"]) == 3


def test_list_ids_in_index_order(store: ChallengeStore, record):
    ids = [
        store.store(_record(red_model="m1", repo="flask"))
        for _ in range(3)
    ]
    listed = store.list_ids_in_index_order("m1", "flask")
    record("listed_ids", listed)
    assert listed == ids
    assert store.list_ids_in_index_order("m1", "missing") == []


def test_query_by_model(store: ChallengeStore, record):
    store.store(_record(red_model="m1", repo="flask"))
    store.store(_record(red_model="m1", repo="jinja"))
    store.store(_record(red_model="m2", repo="flask"))
    results = store.query(red_model_id="m1")
    record("count", len(results))
    assert len(results) == 2
    assert all(r.red_model_id == "m1" for r in results)


def test_query_by_repo(store: ChallengeStore, record):
    store.store(_record(red_model="m1", repo="flask"))
    store.store(_record(red_model="m2", repo="flask"))
    store.store(_record(red_model="m1", repo="jinja"))
    results = store.query(repo_name="flask")
    record("count", len(results))
    assert len(results) == 2
    assert all(r.repo_name == "flask" for r in results)


def test_sample_basic(store: ChallengeStore, record):
    ids = [store.store(_record(target_files=[f"f{i}.py"])) for i in range(5)]
    sampled = store.sample("anthropic/claude-x", "flask", n=3, seed=1)
    record("sampled_ids", [s.challenge_id for s in sampled])
    assert len(sampled) == 3
    assert len({s.challenge_id for s in sampled}) == 3
    assert all(s.challenge_id in ids for s in sampled)


def test_sample_with_seed(store: ChallengeStore, record):
    for i in range(5):
        store.store(_record(target_files=[f"f{i}.py"]))
    a = store.sample("anthropic/claude-x", "flask", n=3, seed=42)
    b = store.sample("anthropic/claude-x", "flask", n=3, seed=42)
    record("seed_run_a", [s.challenge_id for s in a])
    record("seed_run_b", [s.challenge_id for s in b])
    assert [s.challenge_id for s in a] == [s.challenge_id for s in b]


def test_sample_with_exclusions(store: ChallengeStore, record):
    ids = [store.store(_record(target_files=[f"f{i}.py"])) for i in range(5)]
    excluded = {ids[0], ids[1]}
    sampled = store.sample(
        "anthropic/claude-x", "flask", n=3, seed=1, exclude_ids=excluded
    )
    record("excluded", list(excluded))
    record("sampled_ids", [s.challenge_id for s in sampled])
    assert len(sampled) == 3
    assert not any(s.challenge_id in excluded for s in sampled)


def test_sample_insufficient(store: ChallengeStore, record):
    store.store(_record())
    store.store(_record())
    with pytest.raises(InsufficientChallengesError) as exc:
        store.sample("anthropic/claude-x", "flask", n=5)
    record("error", str(exc.value))


def test_sample_file_diversity(store: ChallengeStore, record):
    # 6 challenges across 3 files (2 each).
    for f in ["a.py", "b.py", "c.py"]:
        for _ in range(2):
            store.store(_record(target_files=[f]))
    sampled = store.sample("anthropic/claude-x", "flask", n=3, seed=0)
    files = {tuple(s.target_files) for s in sampled}
    record("sampled_files", [s.target_files for s in sampled])
    assert len(files) == 3


def test_pool_stats(store: ChallengeStore, record):
    for f, cost, retries in [
        ("a.py", 0.1, 0),
        ("a.py", 0.2, 1),
        ("b.py", 0.3, 2),
        ("b.py", 0.4, 1),
        ("c.py", 0.5, 0),
    ]:
        store.store(_record(target_files=[f], cost=cost, retries=retries))
    stats = store.pool_stats("anthropic/claude-x", "flask")
    record("stats", stats)
    assert stats.total_challenges == 5
    assert stats.target_file_distribution == {"a.py": 2, "b.py": 2, "c.py": 1}
    assert stats.bug_type_distribution == {"logic_error": 5}
    assert stats.total_generation_cost_usd == pytest.approx(1.5)
    assert stats.avg_retries == pytest.approx(0.8)


def test_has_sufficient_pool(store: ChallengeStore, record):
    for _ in range(5):
        store.store(_record())
    record("has_3", store.has_sufficient_pool("anthropic/claude-x", "flask", 3))
    record("has_8", store.has_sufficient_pool("anthropic/claude-x", "flask", 8))
    assert store.has_sufficient_pool("anthropic/claude-x", "flask", 3) is True
    assert store.has_sufficient_pool("anthropic/claude-x", "flask", 8) is False


def test_delete(store: ChallengeStore, record):
    cid = store.store(_record())
    store.store(_record())
    store.delete(cid)
    with pytest.raises(ChallengeNotFoundError):
        store.get(cid)
    pools = store.index["pools"]
    record("pools_after_delete", pools)
    for ids in pools.values():
        assert cid not in ids


def test_list_pools(store: ChallengeStore, record):
    store.store(_record(red_model="m1", repo="r1"))
    store.store(_record(red_model="m1", repo="r1"))
    store.store(_record(red_model="m2", repo="r2"))
    pools = sorted(store.list_pools())
    record("pools", pools)
    assert pools == [
        ("m1", "mini-swe-agent", "", "", "r1", 2),
        ("m2", "mini-swe-agent", "", "", "r2", 1),
    ]


def test_slot_defaults_to_one_and_persists(store: ChallengeStore, record):
    """A record stored without an explicit slot defaults to 1 and round-trips,
    and the slot is mirrored into index.entries."""
    rec = _record(red_model="m1", repo="flask")
    cid = store.store(rec)
    loaded = store.get(cid)
    record("loaded_slot", loaded.slot)
    assert loaded.slot == 1
    idx = store.index
    assert idx["entries"][cid]["slot"] == 1


def test_next_slot_and_distinct_slot_count(store: ChallengeStore, record):
    """next_slot advances past recorded slots; distinct_slot_count counts them."""
    # Empty pool: next slot is 1, no distinct slots yet.
    assert store.next_slot("m1", "flask") == 1
    assert store.distinct_slot_count("m1", "flask") == 0

    # Store slot 1.
    from dataclasses import replace
    store.store(replace(_record(red_model="m1", repo="flask"), slot=1))
    assert store.distinct_slot_count("m1", "flask") == 1
    assert store.next_slot("m1", "flask") == 2

    # A second effort in slot 2 — distinct count rises, next advances.
    store.store(replace(_record(red_model="m1", repo="flask"), slot=2))
    record("distinct", store.distinct_slot_count("m1", "flask"))
    assert store.distinct_slot_count("m1", "flask") == 2
    assert store.next_slot("m1", "flask") == 3

    # Other (model, repo) pools are unaffected.
    assert store.next_slot("m1", "jwt") == 1
    assert store.next_slot("m2", "flask") == 1


def test_failed_record_carries_slot(store: ChallengeStore, record):
    """A failed record's slot persists and lands in failed_pools + entries.

    A failed effort in its own slot is counted as a distinct slot, so a repo
    with only failed slots still reports slot activity (eligibility)."""
    from swe_duel.models import FailedChallengeRecord

    frec = FailedChallengeRecord(
        challenge_id=str(uuid.uuid4()),
        red_model_id="m1",
        repo_name="helmet",
        repo_commit_sha="abc",
        kind="validation",
        error_message="gate_bug_tests=failed",
        attempt_number=1,
        target_files=["src/foo.py"],
        challenge=None,
        validation=None,
        feature_trajectory=None,
        bug_trajectory=None,
        self_review_trajectory=None,
        generated_at=datetime(2026, 4, 19, 12, 0, 0),
        generation_cost_usd=0.0,
        elapsed_seconds=1.0,
        slot=1,
    )
    cid = store.store_failed(frec)
    loaded = store.get_failed(cid)
    record("failed_slot", loaded.slot)
    assert loaded.slot == 1
    idx = store.index
    assert idx["entries"][cid]["slot"] == 1
    fp = idx["failed_pools"]["m1#mini-swe-agent::helmet"]
    assert fp[0]["slot"] == 1
    # A pool with only failed slots still reports a distinct slot.
    assert store.distinct_slot_count("m1", "helmet") == 1
    assert store.next_slot("m1", "helmet") == 2


def test_list_attempts_by_slot_groups_success_and_failed(store: ChallengeStore, record):
    """list_attempts_by_slot surfaces both pools and failed_pools by slot."""
    from dataclasses import replace
    from swe_duel.models import FailedChallengeRecord

    store.store(replace(_record(red_model="m1", repo="flask"), slot=1))
    for attempt in (1, 2):
        store.store_failed(
            FailedChallengeRecord(
                challenge_id=str(uuid.uuid4()),
                red_model_id="m1",
                repo_name="flask",
                repo_commit_sha="abc",
                kind="incomplete-feature",
                error_message="missing metadata",
                attempt_number=attempt,
                target_files=[],
                challenge=None,
                validation=None,
                feature_trajectory=None,
                bug_trajectory=None,
                self_review_trajectory=None,
                generated_at=datetime(2026, 4, 19, 12, 0, 0),
                generation_cost_usd=0.0,
                elapsed_seconds=1.0,
                slot=2,
            )
        )

    by_slot = store.list_attempts_by_slot("m1", "flask")
    record("slots", sorted(by_slot))
    assert set(by_slot) == {1, 2}
    assert any(a["status"] == "success" for a in by_slot[1])
    assert len(by_slot[2]) == 2
    assert all(a["status"] == "failed" for a in by_slot[2])
    assert [a["attempt"] for a in by_slot[2]] == [1, 2]
    # Unrelated pool is empty.
    assert store.list_attempts_by_slot("m2", "flask") == {}


def test_atomic_index_write(store: ChallengeStore, record):
    """_save_index must use a temp file + os.replace for atomicity."""
    rec = _record()
    with patch("swe_duel.challenge_bank.store.os.replace") as mock_replace, patch(
        "swe_duel.challenge_bank.store.tempfile.mkstemp",
        wraps=__import__("tempfile").mkstemp,
    ) as mock_mkstemp:
        try:
            store.store(rec)
        except Exception:
            pass
    record("mkstemp_called", mock_mkstemp.called)
    record("replace_called", mock_replace.called)
    assert mock_mkstemp.called
    assert mock_replace.called


# ── participant-identity pools (model, harness, effort, provider) ─────────


def test_pools_separate_by_effort_and_provider(store: ChallengeStore, record):
    """The same (model, harness) at different effort/provider selections
    populates distinct pools — each is a distinct competitor."""
    store.store(_record(effort="", provider=""))
    store.store(_record(effort="high", provider=""))
    store.store(_record(effort="high", provider="fireworks"))

    assert store.count_in_pool("anthropic/claude-x", "flask", "mini-swe-agent") == 1
    assert store.count_in_pool(
        "anthropic/claude-x", "flask", "mini-swe-agent", "high"
    ) == 1
    assert store.count_in_pool(
        "anthropic/claude-x", "flask", "mini-swe-agent", "high", "fireworks"
    ) == 1
    # Wrong-identity lookups miss.
    assert store.count_in_pool(
        "anthropic/claude-x", "flask", "mini-swe-agent", "low"
    ) == 0
    assert store.count_in_pool(
        "anthropic/claude-x", "flask", "mini-swe-agent", "high", "cloudflare"
    ) == 0

    pools = sorted(store.list_pools())
    record("pools", pools)
    assert pools == [
        ("anthropic/claude-x", "mini-swe-agent", "", "", "flask", 1),
        ("anthropic/claude-x", "mini-swe-agent", "high", "", "flask", 1),
        ("anthropic/claude-x", "mini-swe-agent", "high", "fireworks", "flask", 1),
    ]


def test_identity_roundtrips_through_json(store: ChallengeStore, bank_dir: Path):
    """Effort/provider persist into the record JSON and deserialize back."""
    store.store(_record(effort="low", provider="cloudflare"))
    cid = store.list_ids_in_index_order(
        "anthropic/claude-x", "flask", "mini-swe-agent", "low", "cloudflare"
    )[0]
    data = json.loads((bank_dir / "challenges" / f"{cid}.json").read_text())
    assert data["red_reasoning_effort"] == "low"
    assert data["red_provider"] == "cloudflare"
    rec = store.get(data["challenge_id"])
    assert rec.red_reasoning_effort == "low"
    assert rec.red_provider == "cloudflare"


def test_legacy_default_identity_pool_key_unchanged(store: ChallengeStore, bank_dir: Path):
    """Records with empty effort/provider keep the exact legacy pool key
    (`model#harness::repo`) so pre-existing banks stay addressable."""
    store.store(_record(effort="", provider=""))
    index = json.loads((bank_dir / "index.json").read_text())
    assert list(index["poles"] if "poles" in index else index["pools"]) == [
        "anthropic/claude-x#mini-swe-agent::flask"
    ]


def test_slots_and_attempts_split_by_identity(store: ChallengeStore, record):
    """Slot bookkeeping (next_slot / list_attempts_by_slot) is per identity."""
    store.store(_record(effort="high", provider=""))
    assert store.next_slot("anthropic/claude-x", "flask", "mini-swe-agent", "high") == 2
    # The default identity is untouched by the effort-selected pool.
    assert store.next_slot("anthropic/claude-x", "flask", "mini-swe-agent") == 1
    attempts = store.list_attempts_by_slot(
        "anthropic/claude-x", "flask", "mini-swe-agent", "high"
    )
    record("attempts", {str(k): v for k, v in attempts.items()})
    assert set(attempts) == {1}
    assert attempts[1][0]["status"] == "success"


def test_query_filters_by_identity(store: ChallengeStore):
    store.store(_record(effort=""))
    store.store(_record(effort="high"))
    assert len(store.query(red_model_id="anthropic/claude-x")) == 2
    only_high = store.query(
        red_model_id="anthropic/claude-x", red_reasoning_effort="high"
    )
    assert len(only_high) == 1
    assert only_high[0].red_reasoning_effort == "high"
