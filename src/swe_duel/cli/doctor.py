"""``swe-duel doctor`` — staged environment validation, no pytest required.

Mirrors the staged validation suite from the README ("Validating the
environment") so a **pip-installed** user — who has no test suite — can verify
their environment is ready for challenge generation and tournaments. Every
stage reuses the exact library primitives the real runs use (``DockerExecutor``
offline execution, ``TestRunner``, ``TurnScorer``), so a green doctor means
the same code paths the arena will exercise are working.

Stages (skipped stages report as ``skip`` and never fail the run):

===========================  ==========================================  ==============
Stage                        What it proves                              Needs
===========================  ==========================================  ==============
install                      package data (docker/, fixtures, prompts,   nothing
                             config templates) shipped with the wheel
config                       arena.yaml / models.yaml / repos/*.yaml     config dir
                             parse and are internally consistent
docker-daemon                a Docker daemon is reachable                Docker
images                       swe-duel-base / swe-duel-mock / per-repo images     `swe-duel setup
                             exist                                       docker`
clones                       target repos cloned at pinned commits       `swe-duel setup
                                                                         repos`
offline-repos                every selected repo's own native test       images
                             suite passes fully offline in its image
scoring-replay               deterministic s_regression/s_feature/       swe-duel-mock
                             s_bugfix decomposition on a sealed fixture
gate-regression              sealed per-repo challenge/defense fixtures  images
                             replay through the validation-gate path
openrouter (``--live``)      every configured model answers one tiny     API key
                             call with recoverable token/cost telemetry
agent-smoke (``--smoke-      one real mini-swe-agent session solves a    API key +
agent NICK``)                 task inside the containerized environment   swe-duel-mock
===========================  ==========================================  ==============

Exit code is 0 only when every non-skipped stage passed.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import swe_duel
import swe_duel.config_defaults
from swe_duel.cli._common import resolve_config_dir
from swe_duel.config import (
    ArenaConfig,
    ModelConfig,
    RepoConfig,
    load_arena_config,
    load_all_repo_configs,
    load_models_config,
)
from swe_duel.sandbox.image_build import (
    BASE_IMAGE,
    MOCK_IMAGE,
    bundled_fixtures_dir,
    image_exists,
)
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.turn_scorer import TurnScorer

# Sealed gate-regression fixtures shipped with the package (one per repo).
_GATE_FIXTURE_REPOS = [
    "flask", "jwt", "java-html-sanitizer", "helmet", "cjson",
    "jinja", "expressjs", "java-jwt", "libexpat", "chi",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    skipped: bool = False


class Report:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail))

    def skip(self, name: str, detail: str) -> None:
        print(f"  [SKIP] {name} — {detail}")
        self.checks.append(Check(name=name, ok=True, detail=detail, skipped=True))

    def run_stage(self, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        """Run one stage, converting exceptions into a failed check."""
        started = time.monotonic()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 — doctor reports, never crashes
            ok, detail = False, f"{type(e).__name__}: {e}"
        secs = time.monotonic() - started
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({secs:.1f}s) — {detail}")
        self.add(name, ok, detail)

    def passed(self, name: str) -> bool:
        return any(c.name == name and c.ok and not c.skipped for c in self.checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.skipped]

    def print_summary(self) -> None:
        passed = sum(1 for c in self.checks if c.ok and not c.skipped)
        skipped = sum(1 for c in self.checks if c.skipped)
        failed = self.failed
        print()
        print(f"doctor: {passed} passed, {len(failed)} failed, {skipped} skipped")
        for c in failed:
            print(f"  FAIL  {c.name}: {c.detail}")
        if not failed:
            print("environment is ready for challenge generation and tournaments.")

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(),
            "checks": [asdict(c) for c in self.checks],
            "ok": not self.failed,
        }


# ── package-integrity stage ─────────────────────────────────


def check_install() -> tuple[bool, str]:
    pkg_root = Path(swe_duel.__file__).resolve().parent
    missing: list[str] = []
    for rel in [
        "docker/base/Dockerfile",
        "docker/mock/Dockerfile",
        "docker/openhands-constraints.txt",
        "agents/prompts/blue_task.md",
        "agents/prompts/red_feature_task.md",
        "config_defaults/arena.yaml",
        "config_defaults/models.yaml",
    ]:
        if not (pkg_root / rel).is_file():
            missing.append(rel)
    fixtures = bundled_fixtures_dir()
    for rel in [
        "mock_repo/pyproject.toml",
        "gate_regression/provenance.json",
        "sample_red_workspace",
        "sample_blue_workspace",
    ]:
        if not (fixtures / rel).exists():
            missing.append(f"validation/fixtures/{rel}")
    defaults = Path(swe_duel.config_defaults.__file__).resolve().parent
    n_repo_templates = len(list((defaults / "repos").glob("*.yaml")))
    if n_repo_templates == 0:
        missing.append("config_defaults/repos/*.yaml")
    if missing:
        return False, "package data missing: " + ", ".join(missing)
    return (
        True,
        f"package data intact ({n_repo_templates} repo config templates, "
        "prompts + fixtures + docker context bundled)",
    )


# ── config stage ─────────────────────────────────────────────


def check_config(
    config_dir: Path, arena: ArenaConfig
) -> tuple[dict[str, ModelConfig], dict[str, RepoConfig], str]:
    """Parse every config file; raise on any inconsistency."""
    models = load_models_config(config_dir, max_tokens=arena.agent_model.max_tokens)
    repos = load_all_repo_configs(config_dir)
    bad_models = [nick for nick, m in models.items() if "/" not in m.model_id]
    if bad_models:
        raise ValueError(f"bad model_id entries in models.yaml: {bad_models}")
    if not models:
        raise ValueError("models.yaml declares no models")
    if not repos:
        raise ValueError("config/repos/ declares no target repos")
    detail = (
        f"arena.yaml ok (max_tokens={arena.agent_model.max_tokens}, "
        f"output_dir={arena.paths.output_dir}); "
        f"{len(models)} models, {len(repos)} repos in {config_dir}"
    )
    return models, repos, detail


# ── docker stages ─────────────────────────────────────────────


def check_docker_daemon() -> tuple[bool, str]:
    import docker

    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH — install Docker"
    client = docker.from_env(timeout=10)
    client.ping()
    info = client.version()
    return True, f"Docker daemon reachable (server {info.get('Version', '?')})"


def check_images(repo_configs: dict[str, RepoConfig]) -> tuple[bool, str]:
    missing: list[str] = []
    for tag in [BASE_IMAGE, MOCK_IMAGE] + [
        rc.docker_image for rc in repo_configs.values()
    ]:
        if not image_exists(tag):
            missing.append(tag)
    if missing:
        return (
            False,
            "missing images: "
            + ", ".join(missing)
            + " — run: swe-duel setup docker"
            + (f" --only {' '.join(sorted(repo_configs))}" if repo_configs else ""),
        )
    return True, f"base + mock + {len(repo_configs)} repo images present"


def check_clones(
    repo_configs: dict[str, RepoConfig], repos_dir: Path
) -> tuple[bool, str]:
    missing = [name for name in repo_configs if not (repos_dir / name).is_dir()]
    if missing:
        return (
            False,
            f"clones missing under {repos_dir}: {', '.join(missing)} — run: "
            f"swe-duel setup repos --only {' '.join(missing)}",
        )
    return True, f"{len(repo_configs)} clones present under {repos_dir}"


def check_offline_repo(
    rc: RepoConfig, known_failures: list[str] | None = None
) -> tuple[bool, str]:
    """Run the repo's native suite in its image, fully offline (network=none).

    ``known_failures`` are pre-existing failures recorded in the repo's sealed
    gate-regression fixture (the same ones every real challenge deselects via
    ``pre_existing_failures``) — they are passed as ``--deselect`` so this
    stage asserts the same thing the arena does: nothing NEW is broken.
    """
    command = rc.test_command
    if known_failures:
        command += " " + " ".join(f"--deselect {f}" for f in known_failures)
    executor = DockerExecutor(
        docker_image=rc.docker_image, timeout_s=900, memory_mb=2048
    )
    result = executor.execute(file_overrides={}, command=command)
    if result.timed_out:
        return False, f"{rc.name}: native test suite timed out"
    if result.return_code != 0:
        tail = (result.stdout or "")[-400:]
        return (
            False,
            f"{rc.name}: native test suite failed (rc={result.return_code}): {tail}",
        )
    extra = (
        f", {len(known_failures)} known pre-existing failure(s) deselected"
        if known_failures
        else ""
    )
    return True, f"{rc.name}: native suite passes offline in {rc.docker_image}{extra}"


def _sealed_pre_existing_failures(repo_name: str) -> list[str]:
    """Known pre-existing failures from the repo's sealed gate fixture (if any)."""
    path = bundled_fixtures_dir() / "gate_regression" / repo_name / "challenge.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return list(data["challenge"].get("pre_existing_failures") or [])
    except (json.JSONDecodeError, KeyError):
        return []


# ── scoring-replay stage (swe-duel-mock) ────────────────────────


def _mock_repo_config() -> RepoConfig:
    return RepoConfig(
        name="mock",
        url="",
        commit="",
        test_command=(
            "python -m pytest tests/ -x -q --tb=short --json-report "
            "--json-report-file=report.json"
        ),
        docker_image=MOCK_IMAGE,
    )


def _empty_trajectory() -> Any:
    from swe_duel.models import AgentTrajectory

    return AgentTrajectory(
        steps=[],
        total_steps=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        model_id="",
        duration_seconds=0.0,
    )


def check_scoring_replay(arena: ArenaConfig) -> tuple[bool, str]:
    """Replay the sealed sample Red/Blue fixtures through the real TestRunner.

    Asserts the deterministic s_regression × s_feature × s_bugfix
    decomposition: the correct Blue fix scores 1.0 everywhere; Red's buggy
    file leaves s_bugfix at 0 (the hidden-bug gate the whole arena rests on).
    """
    from swe_duel.models import (
        BlueFix,
        ChallengeRecord,
        GateResult,
        GateStatus,
        RedChallenge,
        RedValidationResult,
    )

    fixtures = bundled_fixtures_dir()
    red_dir = fixtures / "sample_red_workspace"
    feature_tests = (red_dir / "_swe-duel" / "feature_tests.py").read_text()
    bug_tests = (red_dir / "_swe-duel" / "bug_tests.py").read_text()
    buggy_basic = (red_dir / "src" / "calculator" / "basic.py").read_text()
    blue_basic = (
        fixtures / "sample_blue_workspace" / "src" / "calculator" / "basic.py"
    ).read_text()

    challenge = RedChallenge(
        target_files=["src/calculator/basic.py"],
        exploration_summary="",
        feature_spec="modulo()",
        feature_rationale="",
        pr_diff=(
            "--- a/src/calculator/basic.py\n+++ b/src/calculator/basic.py\n"
            "@@\n+def modulo\n"
        ),
        modified_file_contents={"src/calculator/basic.py": buggy_basic},
        original_file_contents={},
        feature_test_code=feature_tests,
        bug_type="modulo truncates toward zero for negative operands",
        bug_description="hidden",
        bug_location="src/calculator/basic.py:modulo",
        bug_test_code=bug_tests,
        agent_trajectory=_empty_trajectory(),
    )
    record = ChallengeRecord(
        challenge_id="doctor-scoring-replay",
        red_model_id="doctor/red",
        repo_name="mock",
        repo_commit_sha="",
        target_files=challenge.target_files,
        challenge=challenge,
        validation=RedValidationResult(
            passed=True,
            gate_results=[
                GateResult(
                    gate_name="gate_diff_valid", status=GateStatus.PASSED, message="ok"
                )
            ],
            attempt_number=1,
        ),
        generated_at=datetime(2026, 1, 1),
        generation_cost_usd=0.0,
        generation_retries=0,
    )

    executor = DockerExecutor(docker_image=MOCK_IMAGE, timeout_s=120, memory_mb=512)
    runner = TestRunner(executor=executor, repo_config=_mock_repo_config(), retries=1)
    scorer = TurnScorer(test_runner=runner, config=arena)
    rc = _mock_repo_config()

    good = BlueFix(
        review_findings=[],
        fix_explanation="use Python %",
        fix_diff="--- a\n+++ b\n@@\n+fix\n",
        modified_file_contents={"src/calculator/basic.py": blue_basic},
        agent_trajectory=_empty_trajectory(),
    )
    s_good = scorer.score(rc, record, good)
    if (s_good.s_regression, s_good.s_feature, s_good.s_bugfix) != (1.0, 1.0, 1.0):
        return (
            False,
            f"correct fix did not sweep: s_regression={s_good.s_regression} "
            f"s_feature={s_good.s_feature} s_bugfix={s_good.s_bugfix} "
            f"(details: {s_good.test_details})",
        )

    buggy = BlueFix(
        review_findings=[],
        fix_explanation="no-op",
        fix_diff="--- a\n+++ b\n@@\nunchanged\n",
        modified_file_contents={"src/calculator/basic.py": buggy_basic},
        agent_trajectory=_empty_trajectory(),
    )
    s_bad = scorer.score(rc, record, buggy)
    if s_bad.s_bugfix != 0.0:
        return (
            False,
            f"hidden bug not caught by bug tests: s_bugfix={s_bad.s_bugfix} "
            f"(details: {s_bad.test_details})",
        )
    return (
        True,
        "deterministic scoring decomposition verified (1.0 sweep, hidden bug caught)",
    )


# ── gate-regression stage (sealed per-repo fixtures) ─────────


def check_gate_regression(
    repo_name: str, repo_rc: RepoConfig, arena: ArenaConfig
) -> tuple[bool, str]:
    """Replay a sealed gate-validated challenge + winning defense per repo.

    Mirrors ``tests/test_validation_gates_regression.py`` distilled to the
    core invariants: existing+feature gates pass on feature-only code, the
    hidden bug tests fail on bugged code and pass on clean code, and the
    sealed Blue defense re-scores to ``blue_composite == 1.0``.
    """
    from swe_duel.challenge_bank.store import _deserialise_record
    from swe_duel.models import AgentTrajectory, BlueFix, ReviewFinding

    base = bundled_fixtures_dir() / "gate_regression" / repo_name
    challenge_path = base / "challenge.json"
    defense_path = base / "defense.json"
    defense_challenge_path = base / "defense_challenge.json"
    for p in (challenge_path, defense_path, defense_challenge_path):
        if not p.is_file():
            return False, f"{repo_name}: sealed fixture missing: {p.name}"

    ch = json.loads(challenge_path.read_text())["challenge"]
    image_tag = f"{repo_rc.docker_image}:latest"
    executor = DockerExecutor(docker_image=image_tag, timeout_s=600, memory_mb=2048)
    runner = TestRunner(executor=executor, repo_config=repo_rc, retries=1)

    deselect = list(ch.get("pre_existing_failures") or [])
    feature_only = ch.get("feature_only_file_contents") or {}
    existing = runner.run_existing_tests(
        file_overrides=feature_only, extra_deselect=deselect
    )
    if not existing.passed:
        return False, f"{repo_name}: existing suite fails on feature-only code"
    feature = runner.run_injected_tests(
        file_overrides=feature_only,
        test_code=ch["feature_test_code"],
        test_filename="test_swe_duel_feature.py",
        target_files=ch["target_files"],
    )
    if not feature.passed:
        return False, f"{repo_name}: feature tests fail on feature-only code"

    bugged = ch["modified_file_contents"]
    bug_on_bugged = runner.run_injected_tests(
        file_overrides=bugged,
        test_code=ch["bug_test_code"],
        test_filename="test_swe_duel_bug.py",
        target_files=ch["target_files"],
    )
    if bug_on_bugged.passed:
        return False, f"{repo_name}: bug tests PASSED on bugged code (fixture corrupt?)"
    if feature_only:
        bug_on_clean = runner.run_injected_tests(
            file_overrides=feature_only,
            test_code=ch["bug_test_code"],
            test_filename="test_swe_duel_bug.py",
            target_files=ch["target_files"],
        )
        if not bug_on_clean.passed:
            return (
                False,
                f"{repo_name}: bug tests fail on feature-only code (should pass)",
            )

    defense = json.loads(defense_path.read_text())
    defense_challenge = json.loads(defense_challenge_path.read_text())
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
    traj = bf.get("agent_trajectory") or {}
    blue_fix = BlueFix(
        review_findings=findings,
        fix_explanation=str(bf.get("fix_explanation", "")),
        fix_diff=str(bf.get("fix_diff", "")),
        modified_file_contents=dict(bf.get("modified_file_contents") or {}),
        agent_trajectory=AgentTrajectory(**traj) if traj else _empty_trajectory(),
    )
    scorer = TurnScorer(test_runner=runner, config=arena)
    s = scorer.score(repo_rc, _deserialise_record(defense_challenge), blue_fix)
    if s.blue_composite != 1.0:
        return (
            False,
            f"{repo_name}: sealed winning defense re-scores to "
            f"blue_composite={s.blue_composite} (expected 1.0)",
        )
    return (
        True,
        f"{repo_name}: feature gates pass, hidden bug caught, sealed defense "
        f"re-scores 1.0 in {image_tag}",
    )


# ── live stages (need SWE_DUEL_OPENROUTER_API_KEY) ──────────


def check_openrouter_model(mc: ModelConfig) -> tuple[bool, str]:
    """One tiny live call: model reachable + token/cost telemetry recoverable."""
    import litellm

    from swe_duel.agents.harness.base import openrouter_body_extras

    litellm.suppress_debug_info = True
    body: dict[str, Any] = {
        "model": mc.openrouter_model_id,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 16,
        "temperature": mc.temperature,
    }
    extras = openrouter_body_extras(mc)
    if extras:
        body["extra_body"] = extras
    started = time.monotonic()
    response = litellm.completion(**body, timeout=60)
    usage = getattr(response, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = float(
        getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
    )
    ok = (in_tok + out_tok) > 0
    return ok, (
        f"{mc.model_id} answered in {time.monotonic() - started:.1f}s "
        f"(tokens {in_tok} in / {out_tok} out, cost ${cost:.5f})"
    )


def check_agent_smoke(nick: str, models: dict[str, ModelConfig]) -> tuple[bool, str]:
    """One real mini-swe-agent session solving a task inside swe-duel-mock."""
    from swe_duel.agents.agent_wrapper import AgentWrapper

    mc = models[nick]
    wrapper = AgentWrapper(mc)
    with tempfile.TemporaryDirectory(prefix="swe-duel-doctor-agent-") as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(bundled_fixtures_dir() / "mock_repo", ws)
        task = (
            "Add a function `square(n)` to src/calculator/basic.py that returns n*n, "
            "and add a test for it in tests/test_basic.py. Run the tests to verify."
        )
        trajectory = wrapper.run(ws, task, max_steps=15)
        steps = len(trajectory.steps)
        if steps < 1:
            return False, f"{nick}: agent session produced {steps} steps"
        return (
            True,
            f"{nick}: mini-swe session ran {steps} steps, "
            f"{trajectory.total_output_tokens} output tokens",
        )


# ── orchestration ───────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swe-duel-doctor",
        description=(
            "Validate the environment for running the arena (staged; mirrors "
            "the README 'Validating the environment' suite without pytest)."
        ),
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        metavar="REPO",
        help="restrict repo-targeted stages to these repos",
    )
    parser.add_argument(
        "--skip-offline",
        action="store_true",
        help="skip running each repo's native suite in its image",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true", help="skip the swe-duel-mock scoring replay"
    )
    parser.add_argument(
        "--skip-gates", action="store_true", help="skip the sealed gate-regression replay"
    )
    parser.add_argument(
        "--all-repos",
        action="store_true",
        help="replay gate fixtures for every bundled repo (slow)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="also live-probe every configured model through OpenRouter",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NICK",
        help="restrict --live probes to these models.yaml nicks",
    )
    parser.add_argument(
        "--smoke-agent",
        default=None,
        metavar="NICK",
        help="live stage: run one real mini-swe session as NICK",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the full report to this JSON file"
    )
    args = parser.parse_args(argv)

    import os

    report = Report()
    print("swe-duel doctor — staged environment validation")
    print(f"python {sys.version.split()[0]}, swe-duel at {Path(swe_duel.__file__).resolve().parent}")

    config_dir = resolve_config_dir(args)
    print(f"config: {config_dir.resolve()}")

    # 1–2. package integrity + config parse
    report.run_stage("install", check_install)
    arena = load_arena_config(config_dir)
    models: dict[str, ModelConfig] = {}
    repo_configs: dict[str, RepoConfig] = {}

    def _config_stage() -> tuple[bool, str]:
        nonlocal models, repo_configs
        models, repo_configs, detail = check_config(config_dir, arena)
        return True, detail

    report.run_stage("config", _config_stage)

    if args.repos:
        repo_configs = {n: repo_configs[n] for n in args.repos if n in repo_configs}
        unknown = [n for n in args.repos if n not in repo_configs]
        if unknown:
            print(f"warning: --repos names not in config: {unknown}", file=sys.stderr)

    # 3. docker daemon
    report.run_stage("docker-daemon", check_docker_daemon)

    # 4–5. images + clones
    if report.passed("docker-daemon"):
        report.run_stage("images", lambda: check_images(repo_configs))
    else:
        report.skip("images", "skipped (no Docker daemon)")
    report.run_stage("clones", lambda: check_clones(repo_configs, Path(args.repos_dir)))

    # 6. offline native suites
    if args.skip_offline:
        report.skip("offline-repos", "skipped by --skip-offline")
    elif not (report.passed("images") and report.passed("clones")):
        report.skip("offline-repos", "skipped (missing images or clones)")
    else:
        for rc in sorted(repo_configs.values(), key=lambda r: r.name):
            report.run_stage(
                f"offline-repos[{rc.name}]",
                functools.partial(
                    check_offline_repo, rc, _sealed_pre_existing_failures(rc.name)
                ),
            )

    # 7. scoring replay (needs swe-duel-mock)
    if args.skip_scoring:
        report.skip("scoring-replay", "skipped by --skip-scoring")
    elif not image_exists(MOCK_IMAGE):
        report.skip(
            "scoring-replay", f"skipped ({MOCK_IMAGE} not built — run: swe-duel setup docker)"
        )
    else:
        report.run_stage("scoring-replay", lambda: check_scoring_replay(arena))

    # 8. gate-regression replay
    if args.skip_gates:
        report.skip("gate-regression", "skipped by --skip-gates")
    elif not report.passed("images"):
        report.skip("gate-regression", "skipped (missing images)")
    else:
        gate_targets = _GATE_FIXTURE_REPOS if args.all_repos else ["flask"]
        if args.repos:
            gate_targets = [r for r in gate_targets if r in repo_configs]
        for name in gate_targets:
            gate_rc = repo_configs.get(name)
            if gate_rc is None:
                report.skip(f"gate-regression[{name}]", "repo not selected")
                continue
            report.run_stage(
                f"gate-regression[{name}]",
                functools.partial(check_gate_regression, name, gate_rc, arena),
            )

    # 9. live stages
    if args.live or args.smoke_agent:
        if not os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"):
            report.skip("openrouter", "skipped (SWE_DUEL_OPENROUTER_API_KEY not set)")
        else:
            probe_models = models
            if args.models:
                probe_models = {n: models[n] for n in args.models if n in models}
                unknown = [n for n in args.models if n not in models]
                if unknown:
                    print(
                        f"warning: --models nicks not in models.yaml: {unknown}",
                        file=sys.stderr,
                    )
            for nick, mc in sorted(probe_models.items()):
                report.run_stage(
                    f"openrouter[{nick}]", functools.partial(check_openrouter_model, mc)
                )
        if args.smoke_agent:
            if args.smoke_agent not in models:
                report.add(
                    f"agent-smoke[{args.smoke_agent}]",
                    False,
                    f"unknown models.yaml nick: {args.smoke_agent}",
                )
            elif not image_exists(MOCK_IMAGE):
                report.skip(
                    f"agent-smoke[{args.smoke_agent}]",
                    f"skipped ({MOCK_IMAGE} not built — run: swe-duel setup docker)",
                )
            else:
                report.run_stage(
                    f"agent-smoke[{args.smoke_agent}]",
                    functools.partial(check_agent_smoke, args.smoke_agent, models),
                )

    report.print_summary()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=2))
        print(f"report written to {args.json}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
