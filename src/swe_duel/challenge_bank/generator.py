"""Batch orchestration: drive Red agent → validate → store in Challenge Bank."""

from __future__ import annotations

import html as html_mod
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from swe_duel.agents.red import AgentTimeoutError, FeatureGateFailure, RedPhaseIncomplete
from swe_duel.challenge_bank.progress import Status as _Status
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import ArenaConfig, ModelConfig, RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    ChallengePoolStats,
    ChallengeRecord,
    FailedChallengeRecord,
    GateStatus,
    RedChallenge,
    RedValidationResult,
    Workspace,
)
from swe_duel.sandbox.diff_utils import _CHALLENGE_CSS, _render_trajectory_steps
from swe_duel.sandbox.workspace import WorkspaceManager
from swe_duel.validation.red_gates import RedGateValidator

# Per-thread flag: when a worker drives generation under the live TUI it sets
# this False so the module's diagnostic `print()` calls stay silent (the live
# tree owns the console). Full agent reasoning is still written to _swe-duel/*.log.
_echo = threading.local()


def _echo_enabled() -> bool:
    return getattr(_echo, "enabled", True)


def _log(*args, **kwargs) -> None:
    """`_log()` that is suppressed on threads running under the live UI."""
    if _echo_enabled():
        print(*args, **kwargs)


@dataclass
class _FailedAttempt:
    attempt: int
    kind: str  # "feature-gate", "validation", "timeout-*", or "incomplete-*"
    validation: RedValidationResult | None
    feature_trajectory: AgentTrajectory | None
    bug_trajectory: AgentTrajectory | None
    elapsed_s: float
    workspace_path: Path | None = None
    failed_gates: list[dict] = field(default_factory=list)
    error_message: str = ""


class ChallengeGenerator:
    """Populate the Challenge Bank by repeatedly invoking a Red agent."""

    def __init__(
        self,
        store: ChallengeStore,
        red_gate_validator: RedGateValidator,
        workspace_manager: WorkspaceManager,
        config: ArenaConfig,
    ) -> None:
        self.store = store
        self.red_gate_validator = red_gate_validator
        self.workspace_manager = workspace_manager
        self.config = config

    def _gate_log_dir(
        self,
        red_model_id: str,
        repo_name: str,
        target_index: int,
        attempt: int,
    ) -> Path:
        """Directory where live gate logs for this attempt are streamed."""
        return (
            self.store.bank_dir
            / "_gate_logs"
            / red_model_id
            / repo_name
            / f"t{target_index:03d}"
            / f"a{attempt:02d}"
        )

    # ── single pool ───────────────────────────────────────

    def generate_pool(
        self,
        red_agent,
        repo_config: RepoConfig,
        target_count: int,
        red_model_id: str | None = None,
        pool_reporter=None,
    ) -> ChallengePoolStats:
        """Generate challenges for slots ``1..target_count`` of (red_model, repo).

        Each slot is one generation effort (up to ``max_generation_attempts``
        attempts). Slots that already have any record in ``pools`` or
        ``failed_pools`` are skipped and surfaced on the live TUI as already
        cached (success) or already attempted fails — they are not re-run, so a
        fully-exhausted failed slot stays empty for tournament auto-wins.

        `red_agent.generate_challenge(repo_config)` must return `(RedChallenge, Workspace)`.

        ``pool_reporter`` is an optional ``PoolReporter`` (see
        ``swe_duel.challenge_bank.progress``) the live TUI uses to render
        target/attempt/phase progress. When supplied, this thread's diagnostic
        prints are suppressed (the live tree owns the console).
        """
        model_cfg_obj = getattr(
            getattr(red_agent, "agent_wrapper", None),
            "model_config",
            None,
        )
        model_id = red_model_id or model_cfg_obj
        if model_id is not None and not isinstance(model_id, str):
            model_id = model_id.model_id

        if model_id is None:
            raise ValueError("red_model_id could not be inferred; pass it explicitly")

        # Competitor identity = (model, harness, reasoning_effort, provider).
        # The harness and the selected effort/provider come from the red
        # agent's bound harness; no defaulting beyond legacy fallback. The
        # isinstance guard keeps mocks (tests) and foreign config objects from
        # leaking non-string values into the pool key.
        harness_id = getattr(
            getattr(red_agent, "agent_wrapper", None), "harness_id", "mini-swe-agent"
        )
        if isinstance(model_cfg_obj, ModelConfig):
            reasoning_effort = model_cfg_obj.reasoning_effort or ""
            provider = model_cfg_obj.provider or ""
        else:
            reasoning_effort = ""
            provider = ""

        if pool_reporter is not None:
            _echo.enabled = False

        from swe_duel.challenge_bank.store import _pool_key

        pool_key = _pool_key(
            model_id, repo_config.name, harness_id, reasoning_effort, provider
        )
        # Slot index is identity: target_count = n means generation slots 1..n.
        # A slot is "already attempted" if any success (pools) or failure
        # (failed_pools) is recorded for it — both live in index.entries via
        # list_attempts_by_slot. Skipping attempted slots is essential for
        # tournaments: a failed-only slot stays empty so the match orchestrator
        # awards the defender an auto-win rather than spawning slot n+1.
        attempts_by_slot = self.store.list_attempts_by_slot(
            model_id, repo_config.name, harness_id, reasoning_effort, provider
        )
        existing_successes = len(
            self.store.index.get("pools", {}).get(pool_key, [])
        )
        max_attempts = self.config.challenge_bank.max_generation_attempts
        slots_to_generate = [
            s for s in range(1, target_count + 1) if s not in attempts_by_slot
        ]
        _log(
            f"Found {existing_successes} successful / "
            f"{len(attempts_by_slot)} attempted slots for {pool_key}; "
            f"will generate slots {slots_to_generate or '(none)'}"
        )

        if pool_reporter is not None:
            pool_reporter.set_target_total(target_count)
            pool_reporter.start_pool()

        for slot_i in range(1, target_count + 1):
            prior = attempts_by_slot.get(slot_i)
            if prior is not None:
                self._report_prior_slot(
                    pool_reporter=pool_reporter,
                    slot_i=slot_i,
                    prior=prior,
                )
                continue

            self._generate_one(
                red_agent=red_agent,
                repo_config=repo_config,
                red_model_id=model_id,
                red_harness_id=harness_id,
                max_attempts=max_attempts,
                previous_gists=self._collect_gists(
                    model_id, repo_config.name, harness_id,
                    reasoning_effort, provider,
                ),
                pool_reporter=pool_reporter,
                target_index=slot_i,
                slot=slot_i,
                red_reasoning_effort=reasoning_effort,
                red_provider=provider,
            )

        stats = self.store.pool_stats(
            model_id, repo_config.name, harness_id, reasoning_effort, provider
        )
        if pool_reporter is not None:
            final = stats.total_challenges
            status = _Status.SUCCESS if final >= target_count else _Status.FAIL
            pool_reporter.finish_pool(
                status, detail=f"{final}/{target_count} cached"
            )
        return stats

    @staticmethod
    def _report_prior_slot(
        pool_reporter: object | None,
        slot_i: int,
        prior: list[dict[str, object]],
    ) -> None:
        """Replay a previously-attempted slot onto the live TUI without regenerating.

        Successful slots collapse to ``Target N SUCCESS — already cached``.
        Failed-only slots replay each prior attempt as
        ``Attempt k FAIL — <kind>`` and finish the target as FAIL so the operator
        can see the slot was already exhausted (and will be an auto-win in
        tournament play).
        """
        if pool_reporter is None:
            return
        # PoolReporter is duck-typed to avoid a circular import with progress.py.
        reporter = pool_reporter
        has_success = any(str(a.get("status")) == "success" for a in prior)
        start_target = getattr(reporter, "start_target")
        finish_target = getattr(reporter, "finish_target")
        start_attempt = getattr(reporter, "start_attempt")
        finish_attempt = getattr(reporter, "finish_attempt")
        start_target(slot_i)
        if has_success:
            finish_target(slot_i, _Status.SUCCESS, detail="already cached")
            return
        last_kind = "failed"
        for a in prior:
            raw_att = a.get("attempt", 1)
            if isinstance(raw_att, int) and not isinstance(raw_att, bool):
                attempt_no = raw_att
            elif isinstance(raw_att, str) and raw_att.isdigit():
                attempt_no = int(raw_att)
            else:
                attempt_no = 1
            kind = str(a.get("kind") or "failed")
            last_kind = kind
            start_attempt(slot_i, attempt_no)
            finish_attempt(slot_i, attempt_no, _Status.FAIL, detail=kind)
        finish_target(
            slot_i,
            _Status.FAIL,
            detail=f"already attempted — {last_kind}",
        )

    def _collect_gists(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> list[dict]:
        """Return compact summaries of already-generated challenges for this pool."""
        records = self.store.query(
            red_model_id=red_model_id,
            repo_name=repo_name,
            red_harness_id=red_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )
        gists: list[dict] = []
        for r in records:
            spec = r.challenge.feature_spec or ""
            if len(spec) > 200:
                spec = spec[:200].rstrip() + "…"
            gists.append(
                {
                    "target_files": list(r.target_files),
                    "bug_location": r.challenge.bug_location or "",
                    "bug_type": r.challenge.bug_type.value if r.challenge.bug_type else "unknown",
                    "feature_spec": spec,
                }
            )
        return gists

    # ── persistence helpers ──────────────────────────────────

    def _persist_failed_to_bank(
        self,
        *,
        red_model_id: str,
        repo_config: RepoConfig,
        kind: str,
        attempt: int,
        error_message: str,
        elapsed_seconds: float,
        partial_challenge: RedChallenge | None,
        validation: RedValidationResult | None,
        feature_trajectory: AgentTrajectory | None,
        bug_trajectory: AgentTrajectory | None,
        previous_attempts: list[dict] | None = None,
        red_harness_id: str = "mini-swe-agent",
        slot: int = 1,
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> str | None:
        """Build and persist a FailedChallengeRecord; return its challenge_id."""
        cost = 0.0
        for traj in (feature_trajectory, bug_trajectory):
            if traj is not None:
                cost += float(traj.total_cost_usd or 0.0)
        self_review_trajectory: AgentTrajectory | None = None
        if validation is not None and validation.self_review is not None:
            self_review_trajectory = validation.self_review.agent_trajectory
            cost += float(self_review_trajectory.total_cost_usd or 0.0)
        target_files: list[str] = []
        if partial_challenge is not None:
            target_files = list(partial_challenge.target_files)
        record = FailedChallengeRecord(
            challenge_id=str(uuid.uuid4()),
            red_model_id=red_model_id,
            repo_name=repo_config.name,
            repo_commit_sha=repo_config.commit,
            kind=kind,
            error_message=error_message,
            attempt_number=attempt,
            target_files=target_files,
            challenge=partial_challenge,
            validation=validation,
            feature_trajectory=feature_trajectory,
            bug_trajectory=bug_trajectory,
            self_review_trajectory=self_review_trajectory,
            generated_at=datetime.now(timezone.utc),
            generation_cost_usd=cost,
            elapsed_seconds=elapsed_seconds,
            red_harness_id=red_harness_id,
            slot=slot,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )
        try:
            return self.store.store_failed(
                record, previous_attempts=previous_attempts
            )
        except Exception as e:
            _log(f"    [warn] could not persist failed attempt to bank: {e}", flush=True)
            return None

    def _generate_one(
        self,
        red_agent,
        repo_config: RepoConfig,
        red_model_id: str,
        max_attempts: int,
        previous_gists: list[dict] | None = None,
        pool_reporter=None,
        target_index: int | None = None,
        red_harness_id: str = "mini-swe-agent",
        slot: int = 1,
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> ChallengeRecord | None:
        total_cost = 0.0
        workspace = None
        gists = list(previous_gists or [])
        failed_attempts: list[_FailedAttempt] = []
        # One-glance chain of previous attempts inside THIS slot.
        slot_chain: list[dict] = []

        if pool_reporter is not None and target_index is not None:
            pool_reporter.start_target(target_index)

        def _progress(phase: str, step: int, max_steps: int) -> None:
            if pool_reporter is not None and target_index is not None:
                pool_reporter.update_phase(
                    target_index, attempt, phase, step, max_steps
                )

        def _chain_snapshot() -> list[dict]:
            return [dict(p) for p in slot_chain]

        def _record_chain(cid: str | None, kind: str, attempt_no: int) -> None:
            slot_chain.append(
                {
                    "id": cid or "",
                    "kind": kind,
                    "attempt": attempt_no,
                    "status": "success" if kind == "success" else "failed",
                }
            )

        for attempt in range(1, max_attempts + 1):
            _log(
                f"  [attempt {attempt}/{max_attempts}] stage=agent-run "
                f"(red_model={red_model_id}, repo={repo_config.name})",
                flush=True,
            )
            if pool_reporter is not None and target_index is not None:
                pool_reporter.start_attempt(target_index, attempt)
            attempt_started = time.time()

            gate_log_dir = self._gate_log_dir(
                red_model_id, repo_config.name, target_index or 0, attempt
            )
            gate_log_dir.mkdir(parents=True, exist_ok=True)
            self.red_gate_validator.log_dir = gate_log_dir

            def _feature_validator(
                partial: RedChallenge, attempt_no: int
            ) -> RedValidationResult:
                return self.red_gate_validator.validate_feature_only(
                    repo_config=repo_config,
                    challenge=partial,
                    attempt_number=attempt_no,
                    phase="feature-only",
                    progress=_progress if pool_reporter is not None else None,
                )

            try:
                challenge, workspace = red_agent.generate_challenge(
                    repo_config,
                    previous_gists=gists,
                    feature_validator=_feature_validator,
                    attempt_number=attempt,
                    progress=_progress if pool_reporter is not None else None,
                    console_echo=pool_reporter is None,
                )
            except FeatureGateFailure as fgf:
                # Phase A produced a feature that fails existing or feature
                # tests — skip Phase B entirely to save tokens.
                total_cost += fgf.feature_trajectory.total_cost_usd
                elapsed = time.time() - attempt_started
                gate_summary = ", ".join(
                    f"{g.gate_name}={g.status.value}"
                    for g in fgf.validation.gate_results
                )
                _log(
                    f"    stage=feature-gate FAILED (skipping bug phase): {gate_summary}",
                    flush=True,
                )
                for g in fgf.validation.gate_results:
                    if g.status != GateStatus.PASSED:
                        _log(f"      {g.gate_name}: {g.message}", flush=True)
                _dump_failed_attempt(
                    workspace=fgf.workspace,
                    validation=fgf.validation,
                    attempt=attempt,
                    feature_trajectory=fgf.feature_trajectory,
                    bug_trajectory=None,
                    elapsed_s=elapsed,
                    kind="feature-gate",
                )
                cid = self._persist_failed_to_bank(
                    red_model_id=red_model_id,
                    repo_config=repo_config,
                    kind="feature-gate",
                    attempt=attempt,
                    error_message=str(fgf),
                    elapsed_seconds=elapsed,
                    partial_challenge=fgf.partial_challenge,
                    validation=fgf.validation,
                    feature_trajectory=fgf.feature_trajectory,
                    bug_trajectory=None,
                    previous_attempts=_chain_snapshot(),
                    red_harness_id=red_harness_id,
                    slot=slot,
                    red_reasoning_effort=red_reasoning_effort,
                    red_provider=red_provider,
                )
                _record_chain(cid, "feature-gate", attempt)
                failed_attempts.append(
                    _FailedAttempt(
                        attempt=attempt,
                        kind="feature-gate",
                        validation=fgf.validation,
                        feature_trajectory=fgf.feature_trajectory,
                        bug_trajectory=None,
                        elapsed_s=elapsed,
                        workspace_path=fgf.workspace.path if fgf.workspace else None,
                    )
                )
                if pool_reporter is not None and target_index is not None:
                    pool_reporter.finish_attempt(
                        target_index, attempt, _Status.FAIL, detail="feature-gate"
                    )
                workspace = None
                continue
            except AgentTimeoutError as te:
                elapsed = time.time() - attempt_started
                err_msg = str(te)
                _log(
                    f"    stage=agent-run TIMEOUT ({te.phase} phase, "
                    f"wall_limit={te.wall_seconds}s): persisting failure artefacts",
                    flush=True,
                )
                if te.feature_trajectory is not None:
                    total_cost += float(te.feature_trajectory.total_cost_usd or 0.0)
                if te.bug_trajectory is not None:
                    total_cost += float(te.bug_trajectory.total_cost_usd or 0.0)
                _dump_failed_attempt(
                    workspace=te.workspace,
                    validation=None,
                    attempt=attempt,
                    feature_trajectory=te.feature_trajectory,
                    bug_trajectory=te.bug_trajectory,
                    elapsed_s=elapsed,
                    kind=f"timeout-{te.phase}",
                    error_message=err_msg,
                )
                cid = self._persist_failed_to_bank(
                    red_model_id=red_model_id,
                    repo_config=repo_config,
                    kind=f"timeout-{te.phase}",
                    attempt=attempt,
                    error_message=err_msg,
                    elapsed_seconds=elapsed,
                    partial_challenge=None,
                    validation=None,
                    feature_trajectory=te.feature_trajectory,
                    bug_trajectory=te.bug_trajectory,
                    previous_attempts=_chain_snapshot(),
                    red_harness_id=red_harness_id,
                    slot=slot,
                    red_reasoning_effort=red_reasoning_effort,
                    red_provider=red_provider,
                )
                _record_chain(cid, f"timeout-{te.phase}", attempt)
                failed_attempts.append(
                    _FailedAttempt(
                        attempt=attempt,
                        kind=f"timeout-{te.phase}",
                        validation=None,
                        feature_trajectory=te.feature_trajectory,
                        bug_trajectory=te.bug_trajectory,
                        elapsed_s=elapsed,
                        workspace_path=te.workspace.path,
                        error_message=err_msg,
                    )
                )
                if pool_reporter is not None and target_index is not None:
                    pool_reporter.finish_attempt(
                        target_index, attempt, _Status.FAIL,
                        detail=f"timeout-{te.phase}",
                    )
                workspace = None
                continue
            except RedPhaseIncomplete as inc:
                elapsed = time.time() - attempt_started
                err_msg = str(inc)
                _log(
                    f"    stage=agent-run INCOMPLETE ({inc.phase} phase): "
                    f"{err_msg}; persisting failure artefacts",
                    flush=True,
                )
                if inc.feature_trajectory is not None:
                    total_cost += float(inc.feature_trajectory.total_cost_usd or 0.0)
                if inc.bug_trajectory is not None:
                    total_cost += float(inc.bug_trajectory.total_cost_usd or 0.0)
                _dump_failed_attempt(
                    workspace=inc.workspace,
                    validation=None,
                    attempt=attempt,
                    feature_trajectory=inc.feature_trajectory,
                    bug_trajectory=inc.bug_trajectory,
                    elapsed_s=elapsed,
                    kind=f"incomplete-{inc.phase}",
                    error_message=err_msg,
                )
                cid = self._persist_failed_to_bank(
                    red_model_id=red_model_id,
                    repo_config=repo_config,
                    kind=f"incomplete-{inc.phase}",
                    attempt=attempt,
                    error_message=err_msg,
                    elapsed_seconds=elapsed,
                    partial_challenge=None,
                    validation=None,
                    feature_trajectory=inc.feature_trajectory,
                    bug_trajectory=inc.bug_trajectory,
                    previous_attempts=_chain_snapshot(),
                    red_harness_id=red_harness_id,
                    slot=slot,
                    red_reasoning_effort=red_reasoning_effort,
                    red_provider=red_provider,
                )
                _record_chain(cid, f"incomplete-{inc.phase}", attempt)
                failed_attempts.append(
                    _FailedAttempt(
                        attempt=attempt,
                        kind=f"incomplete-{inc.phase}",
                        validation=None,
                        feature_trajectory=inc.feature_trajectory,
                        bug_trajectory=inc.bug_trajectory,
                        elapsed_s=elapsed,
                        workspace_path=inc.workspace.path,
                        error_message=err_msg,
                    )
                )
                if pool_reporter is not None and target_index is not None:
                    pool_reporter.finish_attempt(
                        target_index, attempt, _Status.FAIL,
                        detail=f"incomplete-{inc.phase}",
                    )
                workspace = None
                continue
            except Exception as e:
                _log(f"    stage=agent-run FAILED: {type(e).__name__}: {e}", flush=True)
                if workspace is not None:
                    self.workspace_manager.cleanup(workspace)
                    workspace = None
                if pool_reporter is not None and target_index is not None:
                    pool_reporter.finish_attempt(
                        target_index, attempt, _Status.FAIL,
                        detail=f"error: {type(e).__name__}",
                    )
                continue

            _log(
                f"    stage=agent-run OK: target_files={challenge.target_files} "
                f"bug_type={challenge.bug_type.value if challenge.bug_type else None} "
                f"cost=${challenge.agent_trajectory.total_cost_usd:.4f} "
                f"steps={challenge.agent_trajectory.total_steps}",
                flush=True,
            )
            _log("    stage=validation running gates...", flush=True)

            total_cost += challenge.agent_trajectory.total_cost_usd

            validation = self.red_gate_validator.validate(
                repo_config=repo_config,
                challenge=challenge,
                attempt_number=attempt,
                progress=_progress if pool_reporter is not None else None,
                console_echo=pool_reporter is None,
                phase="full",
            )
            elapsed = time.time() - attempt_started
            gate_summary = ", ".join(
                f"{g.gate_name}={g.status.value}" for g in validation.gate_results
            )
            _log(f"    stage=validation done: {gate_summary}", flush=True)
            if not validation.passed:
                for g in validation.gate_results:
                    if g.status != GateStatus.PASSED:
                        _log(f"      {g.gate_name}: {g.message}", flush=True)

            if validation.passed:
                _log("    stage=store persisting challenge to bank", flush=True)
                record = ChallengeRecord(
                    challenge_id=str(uuid.uuid4()),
                    red_model_id=red_model_id,
                    repo_name=repo_config.name,
                    target_files=list(challenge.target_files),
                    repo_commit_sha=repo_config.commit,
                    challenge=challenge,
                    validation=validation,
                    generated_at=datetime.now(timezone.utc),
                    generation_cost_usd=total_cost,
                    generation_retries=attempt - 1,
                    red_harness_id=red_harness_id,
                    slot=slot,
                    red_reasoning_effort=red_reasoning_effort,
                    red_provider=red_provider,
                )
                self.store.store(record, previous_attempts=_chain_snapshot())
                _record_chain(record.challenge_id, "success", attempt)
                _augment_success_artifacts(
                    store=self.store,
                    record=record,
                    failed_attempts=failed_attempts,
                    successful_elapsed_s=elapsed,
                )
                if workspace is not None:
                    self.workspace_manager.cleanup(workspace)
                if pool_reporter is not None and target_index is not None:
                    pool_reporter.finish_attempt(
                        target_index, attempt, _Status.SUCCESS
                    )
                    pool_reporter.finish_target(
                        target_index, _Status.SUCCESS,
                        detail=f"attempt {attempt}/{max_attempts}",
                    )
                return record

            # Validation failed: persist gate output (json + html) alongside
            # metadata.json so the user can inspect which gates failed and why.
            # Skip cleanup of the workspace so _swe-duel/ stays on disk.
            if workspace is not None:
                _dump_failed_attempt(
                    workspace=workspace,
                    validation=validation,
                    attempt=attempt,
                    feature_trajectory=challenge.feature_trajectory,
                    bug_trajectory=challenge.bug_trajectory,
                    elapsed_s=elapsed,
                    kind="validation",
                )
                cid = self._persist_failed_to_bank(
                    red_model_id=red_model_id,
                    repo_config=repo_config,
                    kind="validation",
                    attempt=attempt,
                    error_message=", ".join(
                        f"{g.gate_name}={g.status.value}"
                        for g in validation.gate_results
                        if g.status != GateStatus.PASSED
                    ),
                    elapsed_seconds=elapsed,
                    partial_challenge=challenge,
                    validation=validation,
                    feature_trajectory=challenge.feature_trajectory,
                    bug_trajectory=challenge.bug_trajectory,
                    previous_attempts=_chain_snapshot(),
                    red_harness_id=red_harness_id,
                    slot=slot,
                    red_reasoning_effort=red_reasoning_effort,
                    red_provider=red_provider,
                )
                _record_chain(cid, "validation", attempt)
                failed_attempts.append(
                    _FailedAttempt(
                        attempt=attempt,
                        kind="validation",
                        validation=validation,
                        feature_trajectory=challenge.feature_trajectory,
                        bug_trajectory=challenge.bug_trajectory,
                        elapsed_s=elapsed,
                        workspace_path=workspace.path,
                    )
                )
                workspace = None
            if pool_reporter is not None and target_index is not None:
                gate_detail = ", ".join(
                    g.gate_name
                    for g in validation.gate_results
                    if g.status != GateStatus.PASSED
                ) or "validation"
                pool_reporter.finish_attempt(
                    target_index, attempt, _Status.FAIL, detail=gate_detail
                )

        # All attempts exhausted for this slot.
        if pool_reporter is not None and target_index is not None:
            pool_reporter.finish_target(
                target_index, _Status.FAIL,
                detail=f"{max_attempts} attempts exhausted",
            )
        return None

    # ── multi-pool ────────────────────────────────────────

    def generate_all(
        self,
        red_agents: dict[str, object],
        model_configs: dict[str, ModelConfig],
        repo_configs: dict[str, RepoConfig],
        target_count_per_repo: int,
    ) -> dict[str, ChallengePoolStats]:
        """Drive generation across every (model, repo) pair.

        `red_agents` maps model nickname → a ready RedAgent for that model.
        """
        stats: dict[str, ChallengePoolStats] = {}
        for model_nick, model_cfg in model_configs.items():
            red_agent = red_agents[model_nick]
            wrapper_cfg = getattr(getattr(red_agent, "agent_wrapper", None), "model_config", None)
            harness_id = getattr(
                getattr(red_agent, "agent_wrapper", None),
                "harness_id",
                "mini-swe-agent",
            )
            if isinstance(wrapper_cfg, ModelConfig):
                reasoning_effort = wrapper_cfg.reasoning_effort or ""
                provider = wrapper_cfg.provider or ""
            else:
                reasoning_effort = ""
                provider = ""
            for repo_name, repo_cfg in repo_configs.items():
                pool_stats = self.generate_pool(
                    red_agent=red_agent,
                    repo_config=repo_cfg,
                    target_count=target_count_per_repo,
                    red_model_id=model_cfg.model_id,
                )
                from swe_duel.challenge_bank.store import _pool_key

                stats[_pool_key(
                    model_cfg.model_id, repo_name, harness_id,
                    reasoning_effort, provider,
                )] = pool_stats
        return stats


# ── per-attempt artifact dumping ─────────────────────────────


def _traj_stats(traj: AgentTrajectory | None) -> dict:
    if traj is None:
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "duration_s": 0.0, "steps": 0}
    return {
        "input_tokens": int(getattr(traj, "total_input_tokens", 0) or 0),
        "output_tokens": int(getattr(traj, "total_output_tokens", 0) or 0),
        "cost_usd": float(getattr(traj, "total_cost_usd", 0.0) or 0.0),
        "duration_s": float(getattr(traj, "duration_seconds", 0.0) or 0.0),
        "steps": int(getattr(traj, "total_steps", 0) or 0),
    }


def _attempt_combined(
    feat: AgentTrajectory | None, bug: AgentTrajectory | None
) -> dict:
    f = _traj_stats(feat)
    b = _traj_stats(bug)
    return {
        "input_tokens": f["input_tokens"] + b["input_tokens"],
        "output_tokens": f["output_tokens"] + b["output_tokens"],
        "cost_usd": f["cost_usd"] + b["cost_usd"],
        "duration_s": f["duration_s"] + b["duration_s"],
        "steps": f["steps"] + b["steps"],
    }


def _dump_failed_attempt(
    workspace: Workspace,
    validation: RedValidationResult | None,
    attempt: int,
    feature_trajectory: AgentTrajectory | None,
    bug_trajectory: AgentTrajectory | None,
    elapsed_s: float,
    kind: str,
    error_message: str = "",
) -> None:
    """Write JSON + HTML for a failed attempt into the workspace's _swe-duel/.

    The JSON keeps the legacy `gate_failures.json` shape (extended with
    per-phase stats). The HTML embeds the agent reasoning steps for both
    phases using the same step-rendering format as successful challenges.
    """
    swe_duel_dir = workspace.path / "_swe-duel"
    swe_duel_dir.mkdir(parents=True, exist_ok=True)

    failed = [
        {
            "gate_name": g.gate_name,
            "status": g.status.value,
            "message": g.message,
            "stdout": g.stdout,
            "stderr": g.stderr,
            "duration_ms": g.duration_ms,
        }
        for g in (validation.gate_results if validation is not None else [])
        if g.status != GateStatus.PASSED
    ]

    feat_stats = _traj_stats(feature_trajectory)
    bug_stats = _traj_stats(bug_trajectory)
    combined = _attempt_combined(feature_trajectory, bug_trajectory)

    payload = {
        "attempt_number": attempt,
        "kind": kind,
        "workspace_id": workspace.workspace_id,
        "passed": validation.passed if validation is not None else False,
        "elapsed_seconds": elapsed_s,
        "error_message": error_message,
        "feature_exit_status": (
            feature_trajectory.exit_status if feature_trajectory is not None else ""
        ),
        "bug_exit_status": (
            bug_trajectory.exit_status if bug_trajectory is not None else ""
        ),
        "phase_stats": {
            "feature": feat_stats,
            "bug": bug_stats,
            "combined": combined,
        },
        "failed_gates": failed,
        "all_gates": [
            {"gate_name": g.gate_name, "status": g.status.value, "message": g.message}
            for g in (validation.gate_results if validation is not None else [])
        ],
    }

    (swe_duel_dir / "gate_failures.json").write_text(json.dumps(payload, indent=2))
    (swe_duel_dir / "gate_failures.html").write_text(
        _render_failed_attempt_html(
            workspace_id=workspace.workspace_id,
            attempt=attempt,
            kind=kind,
            validation=validation,
            feature_trajectory=feature_trajectory,
            bug_trajectory=bug_trajectory,
            elapsed_s=elapsed_s,
            error_message=error_message,
        )
    )


# ── HTML rendering ──────────────────────────────────────────


def _gate_rows_html(validation: RedValidationResult | None) -> str:
    if validation is None:
        return "<p><em>No gate results — agent did not produce a valid challenge.</em></p>"
    rows = []
    details_blocks = []
    for g in validation.gate_results:
        status = g.status.value
        cls = "pass" if g.status == GateStatus.PASSED else "fail"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><code>{html_mod.escape(g.gate_name)}</code></td>"
            f"<td>{html_mod.escape(status)}</td>"
            f"<td>{html_mod.escape(g.message or '')}</td>"
            f"</tr>"
        )
        details_blocks.append(_gate_detail_html(g))
    return (
        "<table class='gates'><thead><tr>"
        "<th>Gate</th><th>Status</th><th>Message</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
        f"{''.join(details_blocks)}"
    )


def _gate_detail_html(g) -> str:
    """Collapsible per-gate block showing the exact command run + its output.

    Lets the operator see, for every gate that ran, what command executed in the
    sandbox and the captured stdout/stderr — for both passed and failed gates.
    Gates that never ran simply don't appear (so a missing self-review section
    means the self-review gate was skipped because a cheaper gate failed first).
    """
    cmd = getattr(g, "command", "") or ""
    stdout = g.stdout or ""
    stderr = g.stderr or ""
    if not (cmd or stdout or stderr):
        return ""
    cls = "pass" if g.status == GateStatus.PASSED else "fail"
    parts = [
        f"<details class='gate-detail {cls}'>"
        f"<summary><code>{html_mod.escape(g.gate_name)}</code> "
        f"— {html_mod.escape(g.status.value)} "
        f"<span class='gate-msg'>{html_mod.escape((g.message or '')[:160])}</span>"
        f"</summary>"
    ]
    if cmd:
        parts.append(
            "<div class='gate-cmd'><b>command</b>"
            f"<pre>{html_mod.escape(cmd)}</pre></div>"
        )
    if stdout:
        parts.append(
            "<div class='gate-out'><b>stdout</b>"
            f"<pre>{html_mod.escape(stdout[-20000:])}</pre></div>"
        )
    if stderr:
        parts.append(
            "<div class='gate-out'><b>stderr</b>"
            f"<pre>{html_mod.escape(stderr[-20000:])}</pre></div>"
        )
    parts.append("</details>")
    return "".join(parts)


def _stats_row(label: str, stats: dict) -> str:
    return (
        f"<tr><td>{html_mod.escape(label)}</td>"
        f"<td>{stats['steps']:,}</td>"
        f"<td>{stats['input_tokens']:,}</td>"
        f"<td>{stats['output_tokens']:,}</td>"
        f"<td>${stats['cost_usd']:.4f}</td>"
        f"<td>{stats['duration_s']:.2f}s</td></tr>"
    )


def _stats_table(
    feature_trajectory: AgentTrajectory | None,
    bug_trajectory: AgentTrajectory | None,
    elapsed_s: float | None,
) -> str:
    feat = _traj_stats(feature_trajectory)
    bug = _traj_stats(bug_trajectory)
    combined = _attempt_combined(feature_trajectory, bug_trajectory)
    rows = [
        _stats_row("Feature generation", feat),
        _stats_row("Bug embedding", bug),
        f"<tr class='combined'>"
        f"<td><b>Combined</b></td>"
        f"<td>{combined['steps']:,}</td>"
        f"<td>{combined['input_tokens']:,}</td>"
        f"<td>{combined['output_tokens']:,}</td>"
        f"<td>${combined['cost_usd']:.4f}</td>"
        f"<td>{combined['duration_s']:.2f}s</td></tr>",
    ]
    elapsed_html = (
        f"<p class='wallclock'><em>Wall-clock attempt time: "
        f"<b>{elapsed_s:.2f}s</b></em></p>"
        if elapsed_s is not None
        else ""
    )
    return (
        "<table class='stats'><thead><tr>"
        "<th>Phase</th><th>Steps</th><th>Input tokens</th>"
        "<th>Output tokens</th><th>Cost (USD)</th><th>Time</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>{elapsed_html}"
    )


_EXTRA_CSS = (
    "table.gates,table.stats{border-collapse:collapse;margin:8px 0;"
    "font-size:11px}"
    "table.gates td,table.gates th,table.stats td,table.stats th{"
    "border:1px solid #d0d7de;padding:4px 8px;text-align:left}"
    "table.gates tr.fail td{background:#fde2e4}"
    "table.gates tr.pass td{background:#d1f0df}"
    "table.stats tr.combined td{background:#eef2f6;font-weight:600}"
    "table.stats tr.timeout td{background:#fff1d0}"
    "table.stats tr.fail td{background:#fde2e4}"
    "table.stats tr.incomplete td{background:#ffe8c2}"
    "table.stats tr.pass td{background:#d1f0df}"
    "table.stats .err-msg{color:#820014;font-style:italic}"
    "table.stats th{background:#f6f8fa}"
    "table.stats tr.sub-phase td{background:#f9f9f9;color:#444;font-style:italic;"
    "font-size:11px}"
    "table.stats tr.sub-combined td{background:#eef2f6;font-style:italic;font-size:11px}"
    "details.attempt{border:1px solid #d0d7de;border-radius:6px;"
    "margin:8px 0;background:#fafbfc}"
    "details.attempt > summary{padding:8px 12px;cursor:pointer;"
    "font-weight:600;background:#f6f8fa;border-radius:6px}"
    "details.attempt[open] > summary{border-bottom:1px solid #d0d7de;"
    "border-radius:6px 6px 0 0}"
    "details.attempt > .body{padding:10px 14px}"
    ".gen-stats{margin:1em 0;padding:0.75em 1em;border:1px solid #ccc;"
    "background:#fafafa;border-radius:6px}"
    ".wallclock{margin:4px 0 0 0;color:#555}"
    ".error-banner{margin:8px 0;padding:8px 12px;background:#fff1f0;"
    "border:1px solid #ffa39e;border-radius:6px;color:#820014;font-size:13px}"
    "ul.exit-status{margin:6px 0;padding-left:20px;font-size:12px;color:#555}"
    "ul.findings{margin:6px 0;padding-left:20px;font-size:12px}"
    # Per-gate command/output disclosure blocks.
    "details.gate-detail{border:1px solid #d0d7de;border-radius:5px;margin:6px 0;"
    "background:#fff}"
    "details.gate-detail.fail{border-color:#ffa39e}"
    "details.gate-detail.pass{border-color:#9bd9b4}"
    "details.gate-detail > summary{padding:5px 10px;cursor:pointer;font-size:12px}"
    "details.gate-detail .gate-msg{color:#666;font-weight:400}"
    "details.gate-detail .gate-cmd,details.gate-detail .gate-out{padding:4px 10px}"
    "details.gate-detail .gate-cmd b,details.gate-detail .gate-out b{"
    "display:block;font-size:11px;color:#555;margin-top:4px}"
    "details.gate-detail pre{background:#0d1117;color:#c9d1d9;padding:8px;"
    "border-radius:4px;overflow:auto;max-height:340px;font-size:11px;"
    "white-space:pre-wrap;word-break:break-word}"
    "details.gate-detail .gate-cmd pre{background:#1f2937;color:#e5e7eb}"
)


def _render_failed_attempt_html(
    workspace_id: str,
    attempt: int,
    kind: str,
    validation: RedValidationResult | None,
    feature_trajectory: AgentTrajectory | None,
    bug_trajectory: AgentTrajectory | None,
    elapsed_s: float,
    error_message: str = "",
) -> str:
    title = f"Failed attempt {attempt} — {kind}"
    error_banner = ""
    if error_message:
        error_banner = (
            f"<div class='error-banner'><b>Error:</b> "
            f"{html_mod.escape(error_message)}</div>"
        )
    feat_exit = feature_trajectory.exit_status if feature_trajectory else ""
    bug_exit = bug_trajectory.exit_status if bug_trajectory else ""
    exit_lines = []
    if feat_exit:
        exit_lines.append(
            f"<li>Feature phase exit: <code>{html_mod.escape(feat_exit)}</code></li>"
        )
    if bug_exit:
        exit_lines.append(
            f"<li>Bug phase exit: <code>{html_mod.escape(bug_exit)}</code></li>"
        )
    exit_html = (
        f"<ul class='exit-status'>{''.join(exit_lines)}</ul>" if exit_lines else ""
    )
    feature_steps_html = _render_trajectory_steps(
        list(feature_trajectory.steps) if feature_trajectory else None,
        "Red agent reasoning — feature generation phase",
        f"traj-attempt-{attempt}-feature",
    )
    bug_steps_html = _render_trajectory_steps(
        list(bug_trajectory.steps) if bug_trajectory else None,
        "Red agent reasoning — bug embedding phase",
        f"traj-attempt-{attempt}-bug",
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html_mod.escape(title)}</title>"
        f"<style>{_CHALLENGE_CSS}{_EXTRA_CSS}</style></head><body>"
        f"<h1>{html_mod.escape(title)}</h1>"
        f"<p class='meta'>workspace_id: <code>{html_mod.escape(workspace_id)}</code></p>"
        f"{error_banner}{exit_html}"
        f"<h2>Gate results</h2>{_gate_rows_html(validation)}"
        f"<h2>Phase stats</h2>"
        f"{_stats_table(feature_trajectory, bug_trajectory, elapsed_s)}"
        f"<h2>Red agent reasoning</h2>"
        f"{feature_steps_html}{bug_steps_html}"
        f"</body></html>"
    )


def _attempt_combined_with_self_review(
    feat: AgentTrajectory | None,
    bug: AgentTrajectory | None,
    self_review_traj: AgentTrajectory | None,
) -> dict:
    f = _traj_stats(feat)
    b = _traj_stats(bug)
    s = _traj_stats(self_review_traj)
    return {
        "input_tokens": f["input_tokens"] + b["input_tokens"] + s["input_tokens"],
        "output_tokens": f["output_tokens"] + b["output_tokens"] + s["output_tokens"],
        "cost_usd": f["cost_usd"] + b["cost_usd"] + s["cost_usd"],
        "duration_s": f["duration_s"] + b["duration_s"] + s["duration_s"],
        "steps": f["steps"] + b["steps"] + s["steps"],
    }


def _phase_breakdown_rows(
    attempt_label: str,
    attempt_class: str,
    badge: str,
    err_suffix: str,
    feat: AgentTrajectory | None,
    bug: AgentTrajectory | None,
    self_review_traj: AgentTrajectory | None,
    elapsed_s: float,
) -> list[str]:
    """Return one header row + sub-rows (phase 1 / phase 2 / self-review) + combined row."""
    feat_s = _traj_stats(feat)
    bug_s = _traj_stats(bug)
    sr_s = _traj_stats(self_review_traj)
    combined = _attempt_combined_with_self_review(feat, bug, self_review_traj)

    rows = [
        f"<tr class='{attempt_class}'>"
        f"<td><b>{html_mod.escape(attempt_label)}</b> <b>{badge}</b>{err_suffix}</td>"
        f"<td colspan='5'></td></tr>",
        f"<tr class='sub-phase'>"
        f"<td>&nbsp;&nbsp;&nbsp;Phase 1 — feature generation</td>"
        f"<td>{feat_s['steps']:,}</td>"
        f"<td>{feat_s['input_tokens']:,}</td>"
        f"<td>{feat_s['output_tokens']:,}</td>"
        f"<td>${feat_s['cost_usd']:.4f}</td>"
        f"<td>{feat_s['duration_s']:.2f}s</td></tr>",
        f"<tr class='sub-phase'>"
        f"<td>&nbsp;&nbsp;&nbsp;Phase 2 — bug embedding</td>"
        f"<td>{bug_s['steps']:,}</td>"
        f"<td>{bug_s['input_tokens']:,}</td>"
        f"<td>{bug_s['output_tokens']:,}</td>"
        f"<td>${bug_s['cost_usd']:.4f}</td>"
        f"<td>{bug_s['duration_s']:.2f}s</td></tr>",
    ]
    if self_review_traj is not None:
        rows.append(
            f"<tr class='sub-phase'>"
            f"<td>&nbsp;&nbsp;&nbsp;Self-review agent</td>"
            f"<td>{sr_s['steps']:,}</td>"
            f"<td>{sr_s['input_tokens']:,}</td>"
            f"<td>{sr_s['output_tokens']:,}</td>"
            f"<td>${sr_s['cost_usd']:.4f}</td>"
            f"<td>{sr_s['duration_s']:.2f}s</td></tr>"
        )
    rows.append(
        f"<tr class='sub-combined'>"
        f"<td>&nbsp;&nbsp;&nbsp;<i>Attempt combined</i></td>"
        f"<td>{combined['steps']:,}</td>"
        f"<td>{combined['input_tokens']:,}</td>"
        f"<td>{combined['output_tokens']:,}</td>"
        f"<td>${combined['cost_usd']:.4f}</td>"
        f"<td>{elapsed_s:.2f}s wall</td></tr>"
    )
    return rows


def _augment_success_artifacts(
    store: ChallengeStore,
    record: ChallengeRecord,
    failed_attempts: list[_FailedAttempt],
    successful_elapsed_s: float,
) -> None:
    """Inject collapsible failed-attempt sections + combined stats into the
    persisted challenge HTML.

    Only attempts collected during *this* generation call are included.
    """
    html_path = store.challenges_dir / f"{record.challenge_id}.html"
    if not html_path.exists():
        return

    content = html_path.read_text()

    # Per-attempt rows with per-phase breakdown + grand total.
    success_feat = record.challenge.feature_trajectory
    success_bug = record.challenge.bug_trajectory
    success_sr_traj = (
        record.validation.self_review.agent_trajectory
        if record.validation and record.validation.self_review
        else None
    )

    rows: list[str] = []
    for fa in failed_attempts:
        is_timeout = fa.kind.startswith("timeout")
        is_incomplete = fa.kind.startswith("incomplete")
        if is_timeout:
            row_class, badge = "timeout", "⏱ TIMEOUT"
        elif is_incomplete:
            row_class, badge = "incomplete", "⚠ INCOMPLETE"
        else:
            row_class, badge = "fail", "✗ FAILED"
        err_suffix = (
            f" — <span class='err-msg'>{html_mod.escape(fa.error_message)}</span>"
            if fa.error_message
            else ""
        )
        rows.extend(
            _phase_breakdown_rows(
                attempt_label=f"Attempt {fa.attempt} ({fa.kind})",
                attempt_class=row_class,
                badge=badge,
                err_suffix=err_suffix,
                feat=fa.feature_trajectory,
                bug=fa.bug_trajectory,
                self_review_traj=None,
                elapsed_s=fa.elapsed_s,
            )
        )

    success_attempt_no = len(failed_attempts) + 1
    rows.extend(
        _phase_breakdown_rows(
            attempt_label=f"Attempt {success_attempt_no} (full)",
            attempt_class="pass",
            badge="✓ PASSED",
            err_suffix="",
            feat=success_feat,
            bug=success_bug,
            self_review_traj=success_sr_traj,
            elapsed_s=successful_elapsed_s,
        )
    )

    # Grand total across all attempts including self-review of the successful one.
    grand = _attempt_combined_with_self_review(success_feat, success_bug, success_sr_traj)
    for fa in failed_attempts:
        fa_combined = _attempt_combined(fa.feature_trajectory, fa.bug_trajectory)
        grand["steps"] += fa_combined["steps"]
        grand["input_tokens"] += fa_combined["input_tokens"]
        grand["output_tokens"] += fa_combined["output_tokens"]
        grand["cost_usd"] += fa_combined["cost_usd"]
        grand["duration_s"] += fa_combined["duration_s"]

    grand_wall = successful_elapsed_s + sum(fa.elapsed_s for fa in failed_attempts)
    rows.append(
        f"<tr class='combined'><td><b>All attempts combined</b></td>"
        f"<td>{grand['steps']:,}</td>"
        f"<td>{grand['input_tokens']:,}</td>"
        f"<td>{grand['output_tokens']:,}</td>"
        f"<td>${grand['cost_usd']:.4f}</td>"
        f"<td>{grand_wall:.2f}s wall</td></tr>"
    )

    stats_table = (
        "<table class='stats'><thead><tr>"
        "<th>Attempt / Phase</th><th>Steps</th><th>Input tokens</th>"
        "<th>Output tokens</th><th>Cost (USD)</th><th>Time</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
    )

    # Collapsible per-failed-attempt reasoning sections.
    attempt_blocks: list[str] = []
    for fa in failed_attempts:
        feature_steps_html = _render_trajectory_steps(
            list(fa.feature_trajectory.steps) if fa.feature_trajectory else None,
            "Red agent reasoning — feature generation phase",
            f"failed-attempt-{fa.attempt}-feature",
        )
        bug_steps_html = _render_trajectory_steps(
            list(fa.bug_trajectory.steps) if fa.bug_trajectory else None,
            "Red agent reasoning — bug embedding phase",
            f"failed-attempt-{fa.attempt}-bug",
        )
        if fa.validation is not None:
            failed_gates_summary = ", ".join(
                f"{g.gate_name}={g.status.value}"
                for g in fa.validation.gate_results
                if g.status != GateStatus.PASSED
            ) or "(none)"
        else:
            failed_gates_summary = fa.error_message or "(no validation — agent did not finish)"
        error_html = (
            f"<div class='error-banner'><b>Error:</b> "
            f"{html_mod.escape(fa.error_message)}</div>"
            if fa.error_message
            else ""
        )
        body = (
            f"<div class='body'>"
            f"{error_html}"
            f"<p><b>Kind:</b> {html_mod.escape(fa.kind)} "
            f"&nbsp;|&nbsp; <b>Elapsed:</b> {fa.elapsed_s:.2f}s "
            f"&nbsp;|&nbsp; <b>Failed gates:</b> {html_mod.escape(failed_gates_summary)}</p>"
            f"<h4>Gate results</h4>{_gate_rows_html(fa.validation)}"
            f"<h4>Phase stats</h4>"
            f"{_stats_table(fa.feature_trajectory, fa.bug_trajectory, fa.elapsed_s)}"
            f"<h4>Reasoning</h4>{feature_steps_html}{bug_steps_html}"
            f"</div>"
        )
        attempt_blocks.append(
            f"<details class='attempt'>"
            f"<summary>Failed attempt {fa.attempt} "
            f"({html_mod.escape(fa.kind)}) — {html_mod.escape(failed_gates_summary)}</summary>"
            f"{body}</details>"
        )

    failed_section_inner: str
    if failed_attempts:
        failed_section_inner = (
            "<p>Reasoning steps from each failed attempt in this generation run. "
            "Each is collapsed by default.</p>" + "".join(attempt_blocks)
        )
    else:
        failed_section_inner = (
            "<p><em>No failed attempts — generated on first try.</em></p>"
        )

    # Gate results for the SUCCESSFUL attempt — every gate that ran, with its
    # exact command and captured stdout/stderr (feature-only gates pre-bug, full
    # gates post-bug, and the self-review gate). Absent gates (e.g. self-review
    # when a cheaper gate failed — not applicable for a success, where all gates
    # passed) simply don't appear.
    success_gates_block = (
        "<details class='attempt' open>"
        "<summary>Gate results — successful attempt "
        "(commands + outputs)</summary>"
        f"<div class='body'>{_gate_rows_html(record.validation)}</div>"
        "</details>"
    )

    block = (
        "<div class='gen-stats'>"
        "<h2 style='margin-top:0'>Generation stats</h2>"
        f"<p><b>Total attempts:</b> {success_attempt_no} "
        f"&nbsp;|&nbsp; <b>Failed attempts:</b> {len(failed_attempts)}</p>"
        f"{stats_table}"
        f"{success_gates_block}"
        "<details class='attempt'>"
        f"<summary>Previous failed attempts ({len(failed_attempts)})</summary>"
        f"<div class='body'>{failed_section_inner}</div>"
        "</details>"
        "</div>"
    )

    # Inject extra CSS once and place the block above the first section.
    extra_style = f"<style>{_EXTRA_CSS}</style>"
    if extra_style not in content:
        content = content.replace("</head>", f"{extra_style}</head>", 1)

    marker = "<h2 id='feature'>"
    if marker in content:
        content = content.replace(marker, block + marker, 1)
    else:
        content = content.replace("</body>", block + "</body>", 1)
    html_path.write_text(content)

    # Mirror the failed-attempt summary into the persisted challenge JSON.
    json_path = store.challenges_dir / f"{record.challenge_id}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            data["previous_failed_attempts"] = [
                {
                    "attempt": fa.attempt,
                    "kind": fa.kind,
                    "error_message": fa.error_message,
                    "elapsed_seconds": fa.elapsed_s,
                    "feature_exit_status": (
                        fa.feature_trajectory.exit_status
                        if fa.feature_trajectory is not None
                        else ""
                    ),
                    "bug_exit_status": (
                        fa.bug_trajectory.exit_status
                        if fa.bug_trajectory is not None
                        else ""
                    ),
                    "phase_stats": _attempt_combined(
                        fa.feature_trajectory, fa.bug_trajectory
                    ),
                    "failed_gates": [
                        {
                            "gate_name": g.gate_name,
                            "status": g.status.value,
                            "message": g.message,
                        }
                        for g in (
                            fa.validation.gate_results
                            if fa.validation is not None
                            else []
                        )
                        if g.status != GateStatus.PASSED
                    ],
                    "workspace_path": (
                        str(fa.workspace_path) if fa.workspace_path else ""
                    ),
                }
                for fa in failed_attempts
            ]
            json_path.write_text(json.dumps(data, indent=2, default=str))
