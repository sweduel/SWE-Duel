"""Phase 8: ChallengeGenerator orchestration tests (mocked agent + validator)."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

from jinja2 import Template

from swe_duel.challenge_bank.generator import ChallengeGenerator
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import ArenaConfig, ChallengeBankConfig, ModelConfig, RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    BugType,
    GateResult,
    GateStatus,
    RedChallenge,
    RedValidationResult,
    Workspace,
)


# ── builders ───────────────────────────────────────────────


def _trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        steps=[],
        total_steps=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.02,
        model_id="test/mock",
        duration_seconds=0.1,
    )


def _challenge() -> RedChallenge:
    return RedChallenge(
        target_files=["src/foo.py"],
        exploration_summary="e",
        feature_spec="s",
        feature_rationale="r",
        pr_diff="--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n x\n+y\n",
        modified_file_contents={"src/foo.py": "content"},
        original_file_contents={"src/foo.py": "orig"},
        feature_test_code="def test_a():\n    assert 1",
        bug_type=BugType.LOGIC_ERROR,
        bug_description="d",
        bug_location="l",
        bug_test_code="def test_b():\n    assert 1",
        agent_trajectory=_trajectory(),
    )


def _passing_validation() -> RedValidationResult:
    return RedValidationResult(
        passed=True,
        gate_results=[GateResult("gate_diff_valid", GateStatus.PASSED, "ok")],
        attempt_number=1,
    )


def _failing_validation() -> RedValidationResult:
    return RedValidationResult(
        passed=False,
        gate_results=[GateResult("gate_bug_tests", GateStatus.FAILED, "no bug")],
        attempt_number=1,
    )


def _workspace(tmp_root: Path) -> Workspace:
    wid = str(uuid.uuid4())
    return Workspace(
        workspace_id=wid,
        repo_name="flask",
        path=tmp_root / wid / "working",
        reference_path=tmp_root / wid / "reference",
    )


def _repo_cfg() -> RepoConfig:
    return RepoConfig(
        name="flask",
        url="local://flask",
        commit="abc123",
        test_command="pytest",
        docker_image="swe-duel-flask",
    )


def _arena_cfg(max_retries: int = 3) -> ArenaConfig:
    return ArenaConfig(
        challenge_bank=ChallengeBankConfig(
            target_challenges_per_model_repo=5,
            max_generation_attempts=max_retries,
        )
    )


def _make_generator(tmp_path: Path, max_retries: int = 3):
    store = ChallengeStore(bank_dir=tmp_path / "bank")
    validator = MagicMock()
    wm = MagicMock()
    wm.cleanup = MagicMock()
    gen = ChallengeGenerator(
        store=store,
        red_gate_validator=validator,
        workspace_manager=wm,
        config=_arena_cfg(max_retries=max_retries),
    )
    return gen, store, validator, wm


# ── tests ──────────────────────────────────────────────────


def test_generate_pool_mock_agent(tmp_path, record):
    gen, store, validator, wm = _make_generator(tmp_path)
    validator.validate.return_value = _passing_validation()

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    stats = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=3,
        red_model_id="m1",
    )
    record("stats", stats)
    record("agent_calls", red_agent.generate_challenge.call_count)
    assert stats.total_challenges == 3
    assert red_agent.generate_challenge.call_count == 3
    assert validator.validate.call_count == 3
    assert wm.cleanup.call_count == 3


def test_generate_pool_with_gate_failures(tmp_path, record):
    gen, store, validator, wm = _make_generator(tmp_path, max_retries=3)
    # Fail twice then pass.
    validator.validate.side_effect = [
        _failing_validation(),
        _failing_validation(),
        _passing_validation(),
    ]

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    stats = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="m1",
    )
    record("stats", stats)
    record("validator_calls", validator.validate.call_count)
    assert stats.total_challenges == 1
    assert validator.validate.call_count == 3
    pools = store.list_pools()
    record("pools", pools)
    assert pools == [("m1", "mini-swe-agent", "", "", "flask", 1)]
    # Stored record should reflect 2 retries before success.
    rec = store.query(red_model_id="m1")[0]
    assert rec.generation_retries == 2


def test_generate_pool_skips_existing(tmp_path, record):
    gen, store, validator, wm = _make_generator(tmp_path)
    validator.validate.return_value = _passing_validation()

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    # Seed pool with 3.
    gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=3,
        red_model_id="m1",
    )
    calls_before = red_agent.generate_challenge.call_count

    # Ask for 5 total — only 2 more should be generated.
    stats = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=5,
        red_model_id="m1",
    )
    record("stats", stats)
    record("new_agent_calls", red_agent.generate_challenge.call_count - calls_before)
    assert stats.total_challenges == 5
    assert red_agent.generate_challenge.call_count - calls_before == 2


def test_collect_gists_includes_prior_feature_and_bug_locations(tmp_path, record):
    gen, _, validator, _ = _make_generator(tmp_path)
    validator.validate.return_value = _passing_validation()

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )
    gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="m1",
    )

    gists = gen._collect_gists("m1", "flask", "mini-swe-agent")
    record("gists", gists)
    assert gists == [
        {
            "target_files": ["src/foo.py"],
            "bug_location": "l",
            "bug_type": "logic_error",
            "feature_spec": "s",
        }
    ]
    prompt_path = (
        Path(__file__).parents[1] / "src" / "swe_duel" / "agents" / "prompts" / "red_feature_task.md"
    )
    rendered_prompt = Template(prompt_path.read_text()).render(previous_gists=gists)
    assert "feature location(s): ['src/foo.py']" in rendered_prompt
    assert "bug location(s): l" in rendered_prompt


def test_generate_pool_skips_failed_only_slot(tmp_path, record):
    """A fully-exhausted failed slot must not be re-run (tournament auto-win).

    Mirrors libexpat/glm: slot 1 has max_attempts failures and no success.
    Re-invoking generate_pool(target_count=1) must make zero agent calls.
    """
    gen, store, validator, wm = _make_generator(tmp_path, max_retries=3)
    validator.validate.return_value = _failing_validation()

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    stats1 = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="m1",
    )
    record("first_stats", stats1)
    assert stats1.total_challenges == 0
    assert store.distinct_slot_count("m1", "flask") == 1
    assert store.count_failed_in_pool("m1", "flask") == 3
    calls_after_exhaust = red_agent.generate_challenge.call_count
    assert calls_after_exhaust == 3

    # Re-run: slot 1 already attempted → skip, no new calls.
    stats2 = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="m1",
    )
    record("second_stats", stats2)
    record(
        "new_calls_after_skip",
        red_agent.generate_challenge.call_count - calls_after_exhaust,
    )
    assert stats2.total_challenges == 0
    assert red_agent.generate_challenge.call_count == calls_after_exhaust
    assert store.count_failed_in_pool("m1", "flask") == 3


def test_generate_pool_fills_unattempted_slot_after_failed(tmp_path, record):
    """target_count=2 with slot 1 failed-only must only generate slot 2."""
    gen, store, validator, wm = _make_generator(tmp_path, max_retries=2)
    # First pool run: exhaust slot 1 (all fails). Second: pass on slot 2.
    validator.validate.side_effect = [
        _failing_validation(),
        _failing_validation(),
        _passing_validation(),
    ]
    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="m1",
    )
    calls_mid = red_agent.generate_challenge.call_count
    assert calls_mid == 2
    assert store.count_in_pool("m1", "flask") == 0

    stats = gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=2,
        red_model_id="m1",
    )
    record("stats", stats)
    record("slot2_calls", red_agent.generate_challenge.call_count - calls_mid)
    assert stats.total_challenges == 1
    assert red_agent.generate_challenge.call_count - calls_mid == 1
    recs = store.query(red_model_id="m1", repo_name="flask")
    assert len(recs) == 1
    assert recs[0].slot == 2


def test_generate_all_multi_model(tmp_path, record):
    gen, store, validator, wm = _make_generator(tmp_path)
    validator.validate.return_value = _passing_validation()

    red_agent_a = MagicMock()
    red_agent_a.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent_a.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )
    red_agent_b = MagicMock()
    red_agent_b.agent_wrapper.harness_id = "mini-swe-agent"
    red_agent_b.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    model_cfgs = {
        "a": ModelConfig(model_id="vendor/a"),
        "b": ModelConfig(model_id="vendor/b"),
    }
    repo_cfgs = {"flask": _repo_cfg()}

    stats = gen.generate_all(
        red_agents={"a": red_agent_a, "b": red_agent_b},
        model_configs=model_cfgs,
        repo_configs=repo_cfgs,
        target_count_per_repo=2,
    )
    record("stats_keys", list(stats.keys()))
    record("pools", store.list_pools())
    assert set(stats.keys()) == {"vendor/a#mini-swe-agent::flask", "vendor/b#mini-swe-agent::flask"}
    assert stats["vendor/a#mini-swe-agent::flask"].total_challenges == 2
    assert stats["vendor/b#mini-swe-agent::flask"].total_challenges == 2


def test_generate_pool_identity_follows_bound_harness_selection(tmp_path, record):
    """A rankings-pinned participant (swe-duel-tournament-as) binds its
    effort/provider selection into the Red harness's ModelConfig before
    generation. generate_pool must derive the pool identity from that bound
    config so challenges land in the identity-scoped pool the match
    orchestrator later selects from — never in the default-identity pool."""
    gen, store, validator, wm = _make_generator(tmp_path)
    validator.validate.return_value = _passing_validation()

    red_agent = MagicMock()
    red_agent.agent_wrapper.harness_id = "mini-swe-agent"
    # Exactly what get_harness(harness_id, model_configs[cid]) binds for a
    # rankings entry whose recorded columns were pinned: the config carries
    # the selected effort/provider.
    red_agent.agent_wrapper.model_config = ModelConfig(
        model_id="z-ai/glm-5.2", max_tokens=128
    ).with_selection("high", "cloudflare")
    red_agent.generate_challenge.side_effect = lambda *args, **kwargs: (
        _challenge(),
        _workspace(tmp_path),
    )

    gen.generate_pool(
        red_agent=red_agent,
        repo_config=_repo_cfg(),
        target_count=1,
        red_model_id="z-ai/glm-5.2",
    )
    record("pools", store.list_pools())
    # The pool key carries the pinned effort/provider, not the default identity.
    assert store.list_pools() == [
        ("z-ai/glm-5.2", "mini-swe-agent", "high", "cloudflare", "flask", 1)
    ]
    rec = store.query(
        red_model_id="z-ai/glm-5.2",
        red_harness_id="mini-swe-agent",
        red_reasoning_effort="high",
        red_provider="cloudflare",
    )[0]
    assert rec.red_reasoning_effort == "high"
    assert rec.red_provider == "cloudflare"
    # The default-identity pool stays untouched (identity separation): the
    # match orchestrator only finds the challenge under the pinned identity.
    assert store.count_in_pool("z-ai/glm-5.2", "flask") == 0
    assert (
        store.count_in_pool(
            "z-ai/glm-5.2", "flask", "mini-swe-agent", "high", "cloudflare"
        )
        == 1
    )
