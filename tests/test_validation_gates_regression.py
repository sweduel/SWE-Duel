"""Regression tests for language adapters against real Red/Blue fixtures.

The suite replays the deterministic validation-gate / scoring paths inside the
per-repo Docker image against **committed** fixtures under
``tests/fixtures/gate_regression/<repo>/`` — gate-validated challenges and
winning Blue defenses copied from a real generation run (see that directory's
``provenance.json`` for source ids; the only transformation is emptying
``agent_trajectory.steps``, which the gates never replay):

  - **Red feature** (``validate_feature_only`` path): existing + feature tests
    pass on ``feature_only_file_contents``.
  - **Red bug** (full bug-admittance path): existing + feature pass on bugged
    code; bug tests fail on bugged code and pass on feature-only code.
  - **Red self-review** (stored outcome): admitted challenges must record a
    passed ``gate_self_review`` with ``detected=True`` (emulated Blue *did*
    remove the bug — confirming solvability/fairness). The LLM half is not
    re-run; only the sealed record is checked so a refactor cannot silently
    drop or invert the gate.
  - **Blue scoring** (``TurnScorer``): re-score a successful cached Blue fix and
    assert ``blue_composite == 1.0`` (regression + feature + bugfix all pass).

Repos covered:

  - original: flask, jwt, java-html-sanitizer, helmet, cjson
  - extension: jinja, expressjs, java-jwt, libexpat, chi

These are integration tests: they need Docker + the per-repo images built
(``make build-docker``) and skip gracefully when images are missing. No
``data/`` runtime output is required — the fixtures are part of the repo, so
the suite runs on a fresh clone.

**Parallelism.** Running this file alone auto-enables pytest-xdist
(``-n <capped auto> --dist=worksteal``) via ``tests/conftest.py``. Containers
are named with pid+uuid so concurrent Docker gate runs do not collide. Tune
with ``SWE_DUEL_GATE_TEST_WORKERS`` or pass ``-n`` explicitly::

    ./venv/bin/pytest tests/test_validation_gates_regression.py -v
    SWE_DUEL_GATE_TEST_WORKERS=8 ./venv/bin/pytest tests/test_validation_gates_regression.py -v
    ./venv/bin/pytest tests/test_validation_gates_regression.py -n 4 -v   # explicit
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_duel.config import ArenaConfig, RepoConfig, load_repo_config
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    ReviewFinding,
)
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.languages import get_adapter
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.turn_scorer import TurnScorer

from conftest import FIXTURES_DIR

pytestmark = pytest.mark.integration

# Original five repos that first exercised the multi-language adapters.
_ORIGINAL_REPOS = ["flask", "jwt", "java-html-sanitizer", "helmet", "cjson"]
# Second wave exercised only after multi-repo-per-language support landed.
_EXTENSION_REPOS = ["jinja", "expressjs", "java-jwt", "libexpat", "chi"]
_ALL_REPOS = _ORIGINAL_REPOS + _EXTENSION_REPOS

_FIXTURES_DIR = FIXTURES_DIR / "gate_regression"


def _load_challenge(repo_name: str) -> dict:
    """Return the committed gate-validated challenge fixture for ``repo_name``."""
    path = _FIXTURES_DIR / repo_name / "challenge.json"
    if not path.exists():
        pytest.skip(f"{repo_name}: committed gate fixture missing: {path}")
    return json.loads(path.read_text())


def _load_winning_defense(repo_name: str) -> tuple[dict, dict]:
    """Return (defense_json, its challenge_json) for a Blue that scored 1.0."""
    base = _FIXTURES_DIR / repo_name
    dpath = base / "defense.json"
    cpath = base / "defense_challenge.json"
    if not (dpath.exists() and cpath.exists()):
        pytest.skip(f"{repo_name}: committed defense fixture missing under {base}")
    return json.loads(dpath.read_text()), json.loads(cpath.read_text())


def _make_runner(repo_config: RepoConfig) -> TestRunner:
    image_tag = f"{repo_config.docker_image}:latest"
    import subprocess

    inspect = subprocess.run(
        ["docker", "image", "inspect", image_tag], capture_output=True
    )
    if inspect.returncode != 0:
        pytest.skip(f"Docker image {image_tag} not built — run: make build-docker")
    executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
    return TestRunner(executor=executor, repo_config=repo_config, retries=1)


def _challenge_record_for_scorer(challenge_data: dict) -> ChallengeRecord:
    """ChallengeRecord from on-disk challenge JSON for TurnScorer."""
    from swe_duel.challenge_bank.store import _deserialise_record

    return _deserialise_record(challenge_data)


def _blue_fix_from_defense(defense: dict) -> BlueFix:
    bf = defense["blue_fix"]
    findings = [
        ReviewFinding(
            location=str(f.get("location", "")),
            severity=str(f.get("severity", "info")),
            description=str(f.get("description", "")),
        )
        for f in (bf.get("review_findings") or [])
        if isinstance(f, dict)
    ]
    traj_data = bf.get("agent_trajectory") or {}
    if traj_data:
        trajectory = AgentTrajectory(**traj_data)
    else:
        trajectory = AgentTrajectory(
            steps=[],
            total_steps=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            model_id="",
            duration_seconds=0.0,
        )
    return BlueFix(
        review_findings=findings,
        fix_explanation=str(bf.get("fix_explanation", "")),
        fix_diff=str(bf.get("fix_diff", "")),
        modified_file_contents=dict(bf.get("modified_file_contents") or {}),
        agent_trajectory=trajectory,
    )


# ── Red feature gates ──────────────────────────────────────


@pytest.mark.parametrize("repo_name", _ALL_REPOS)
class TestRedFeatureGates:
    """Re-run the feature-only admittance path (existing + feature tests)."""

    def test_existing_and_feature_pass_on_feature_only(
        self, repo_name: str, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        challenge = _load_challenge(repo_name)
        ch = challenge["challenge"]
        feature_only = ch.get("feature_only_file_contents") or {}
        if not feature_only:
            pytest.skip(f"{repo_name}: no feature_only_file_contents in fixture")
        record("challenge_id", challenge["challenge_id"])

        runner = _make_runner(rc)
        existing = runner.run_existing_tests(
            file_overrides=feature_only,
            extra_deselect=list(ch.get("pre_existing_failures") or []),
        )
        feature = runner.run_injected_tests(
            file_overrides=feature_only,
            test_code=ch["feature_test_code"],
            test_filename="test_swe_duel_feature.py",
            target_files=ch["target_files"],
        )
        record(
            "feature_only",
            {
                "existing_passed": existing.passed,
                "existing_total": existing.total,
                "feature_passed": feature.passed,
                "feature_total": feature.total,
            },
        )
        assert existing.passed, (
            f"{repo_name}: feature-only code regresses existing suite — "
            f"{existing.failure_messages[:2]}"
        )
        assert feature.passed, (
            f"{repo_name}: feature tests fail on feature-only code — "
            f"{feature.failure_messages[:2]}"
        )


# ── Red bug gates ──────────────────────────────────────────


@pytest.mark.parametrize("repo_name", _ALL_REPOS)
class TestRedBugGates:
    """Re-run the full bug-admittance path on a sealed challenge."""

    def test_existing_tests_pass_on_bugged_code(
        self, repo_name: str, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        challenge = _load_challenge(repo_name)
        ch = challenge["challenge"]
        record("challenge_id", challenge["challenge_id"])

        runner = _make_runner(rc)
        result = runner.run_existing_tests(
            file_overrides=ch["modified_file_contents"],
            extra_deselect=list(ch.get("pre_existing_failures") or []),
        )
        record(
            "existing_result",
            {
                "passed": result.passed,
                "total": result.total,
                "failed_count": result.failed_count,
                "error_count": result.error_count,
                "command": result.command,
                "stdout_tail": result.stdout[-2000:],
            },
        )
        assert result.passed, (
            f"{repo_name}: existing tests regressed under the bugged challenge — "
            f"{result.failure_messages[:2]}"
        )

    def test_feature_tests_pass_on_bugged_code(
        self, repo_name: str, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        runner = _make_runner(rc)
        challenge = _load_challenge(repo_name)
        ch = challenge["challenge"]

        result = runner.run_injected_tests(
            file_overrides=ch["modified_file_contents"],
            test_code=ch["feature_test_code"],
            test_filename="test_swe_duel_feature.py",
            target_files=ch["target_files"],
        )
        record(
            "feature_result",
            {
                "passed": result.passed,
                "total": result.total,
                "failed_count": result.failed_count,
                "command": result.command,
            },
        )
        assert result.passed, (
            f"{repo_name}: feature tests failed on bugged code — "
            f"{result.failure_messages[:2]}"
        )

    def test_bug_tests_fail_on_bugged_and_pass_on_feature_only(
        self, repo_name: str, config_dir: Path, record
    ):
        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        runner = _make_runner(rc)
        challenge = _load_challenge(repo_name)
        ch = challenge["challenge"]
        feature_only = ch.get("feature_only_file_contents") or {}

        bugged = runner.run_injected_tests(
            file_overrides=ch["modified_file_contents"],
            test_code=ch["bug_test_code"],
            test_filename="test_swe_duel_bug.py",
            target_files=ch["target_files"],
        )
        record(
            "bug_on_bugged",
            {
                "passed": bugged.passed,
                "total": bugged.total,
                "failed_count": bugged.failed_count,
                "error_count": bugged.error_count,
                "command": bugged.command,
            },
        )
        assert (bugged.failed_count + bugged.error_count) >= 1, (
            f"{repo_name}: bug tests did not detect the bug on bugged code"
        )

        if feature_only:
            fair = runner.run_injected_tests(
                file_overrides=feature_only,
                test_code=ch["bug_test_code"],
                test_filename="test_swe_duel_bug.py",
                target_files=ch["target_files"],
            )
            record(
                "bug_on_feature_only",
                {
                    "passed": fair.passed,
                    "total": fair.total,
                    "failed_count": fair.failed_count,
                    "error_count": fair.error_count,
                },
            )
            assert fair.passed, (
                f"{repo_name}: bug tests failed on pre-bug feature code — "
                f"{fair.failure_messages[:2]}"
            )


# ── Red self-review (stored gate outcome) ──────────────────


@pytest.mark.parametrize("repo_name", _ALL_REPOS)
class TestRedSelfReviewStored:
    """Admitted challenges must record a passed self-review (bug *was* fixed)."""

    def test_self_review_gate_passed_and_bug_was_detected(
        self, repo_name: str, record
    ):
        challenge = _load_challenge(repo_name)
        validation = challenge.get("validation") or {}
        record("challenge_id", challenge["challenge_id"])
        record("validation_passed", validation.get("passed"))

        gate_results = validation.get("gate_results") or []
        self_review_gates = [
            g for g in gate_results if g.get("gate_name") == "gate_self_review"
        ]
        record(
            "self_review_gates",
            [{"status": g.get("status"), "message": g.get("message")} for g in self_review_gates],
        )
        # Older fixtures / disabled self-review still admit challenges; skip
        # rather than fail when the gate was never run.
        if not self_review_gates:
            pytest.skip(f"{repo_name}: no gate_self_review in fixture")
        assert all(g.get("status") == "passed" for g in self_review_gates), (
            f"{repo_name}: **admitted** challenge has non-passed gate_self_review"
        )

        self_review = validation.get("self_review")
        if self_review is None:
            pytest.skip(f"{repo_name}: gate_self_review recorded without payload")
        record(
            "self_review",
            {
                "detected": self_review.get("detected"),
                "detection_reason": (self_review.get("detection_reason") or "")[:200],
            },
        )
        # Passing the self-review gate means the emulated Blue *did* remove the
        # bug (detected=True), confirming the challenge is solvable/fair.
        # detected=False fails the gate and rejects the challenge as unfair.
        assert self_review.get("detected") is True, (
            f"{repo_name}: self-review did not confirm bug was fixable, "
            "but the challenge was admitted"
        )


# ── Blue scoring ───────────────────────────────────────────


@pytest.mark.parametrize("repo_name", _ALL_REPOS)
class TestBlueScoring:
    """Re-score a successful Blue defense through TurnScorer."""

    def test_winning_blue_fix_scores_composite_one(
        self, repo_name: str, config_dir: Path, record
    ):
        defense, challenge_data = _load_winning_defense(repo_name)
        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        record("challenge_id", challenge_data["challenge_id"])
        record("defense_id", defense.get("defense_id"))

        record_obj = _challenge_record_for_scorer(challenge_data)
        blue_fix = _blue_fix_from_defense(defense)
        runner = _make_runner(rc)
        scorer = TurnScorer(test_runner=runner, config=ArenaConfig())
        score = scorer.score(rc, record_obj, blue_fix)
        record(
            "score",
            {
                "s_regression": score.s_regression,
                "s_feature": score.s_feature,
                "s_bugfix": score.s_bugfix,
                "blue_composite": score.blue_composite,
            },
        )
        assert score.s_regression == 1.0, (
            f"{repo_name}: Blue fix regressed existing tests"
        )
        assert score.s_feature == 1.0, f"{repo_name}: Blue fix broke feature tests"
        assert score.s_bugfix == 1.0, f"{repo_name}: Blue fix did not pass bug tests"
        assert score.blue_composite == 1.0


# ── Adapter selection ──────────────────────────────────────


@pytest.mark.parametrize("repo_name", _ALL_REPOS)
class TestAdapterSelection:
    """The TestRunner must select the per-repo adapter (not a generic default)."""

    def test_adapter_selected_correctly_for_repo(
        self, repo_name: str, config_dir: Path, record
    ):
        from swe_duel.sandbox.languages import CAdapter, GoAdapter, JavaAdapter, NodeAdapter

        rc = load_repo_config(config_dir / "repos" / f"{repo_name}.yaml")
        adapter = get_adapter(rc.language, rc.name)
        record("adapter", {"name": adapter.name, "repo": rc.name})
        if rc.language == "java":
            assert isinstance(adapter, JavaAdapter)
            if rc.name == "java-html-sanitizer":
                assert not adapter.is_gradle
            if rc.name == "java-jwt":
                assert adapter.is_gradle
        elif rc.language in ("node", "javascript", "typescript"):
            assert isinstance(adapter, NodeAdapter)
            if rc.name == "expressjs":
                assert adapter.is_mocha
            elif rc.name == "helmet":
                assert not adapter.is_mocha
        elif rc.language == "c":
            assert isinstance(adapter, CAdapter)
            assert adapter.repo_name == rc.name
        elif rc.language == "go":
            assert isinstance(adapter, GoAdapter)
            assert adapter.prompt_hints().get("import_path")
