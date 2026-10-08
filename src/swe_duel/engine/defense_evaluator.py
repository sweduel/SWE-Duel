"""One Blue agent attempting one cached challenge."""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from swe_duel.agents.blue import BlueAgent
from swe_duel.challenge_bank.integrity import verify_challenge_freshness
from swe_duel.config import ArenaConfig, RepoConfig
from swe_duel.models import ChallengeRecord, DefenseResult
from swe_duel.scoring.turn_scorer import TurnScorer


class SubturnProgress(Protocol):
    """Minimal progress sink for one defense sub-turn (e.g. a TUI handle).

    Implemented by ``swe_duel.engine.round_progress.SubturnHandle``.
    """

    def step(self, step: int, max_steps: int) -> None: ...
    def phase(self, phase: str, status: str) -> None: ...


class StaleChallengeError(RuntimeError):
    """Raised when a ChallengeRecord's commit no longer matches the current RepoConfig."""


class DefenseEvaluator:
    def __init__(
        self,
        blue_agent: BlueAgent,
        scorer: TurnScorer,
        config: ArenaConfig,
    ) -> None:
        self.blue_agent = blue_agent
        self.scorer = scorer
        self.config = config

    # ── public ─────────────────────────────────────────────

    def evaluate(
        self,
        repo_config: RepoConfig,
        challenge_record: ChallengeRecord,
        *,
        progress: SubturnProgress | None = None,
        console_echo: bool = True,
        log_dir: Path | None = None,
    ) -> DefenseResult:
        """Run one Blue defense and score it.

        ``progress`` (optional) receives per-step ticks (``step``) during the
        Blue agent run and per-phase signals (``phase``) during scoring, so a
        live TUI can render this sub-turn. ``console_echo`` is set False when a
        TUI owns the console. ``log_dir`` (optional) is where the workspace's
        ``_swe-duel/blue.log`` is copied after the run, so the agent's reasoning
        survives workspace cleanup and sits beside the defense JSON/HTML.
        """
        if not verify_challenge_freshness(challenge_record, repo_config):
            raise StaleChallengeError(
                f"Challenge {challenge_record.challenge_id} has commit "
                f"{challenge_record.repo_commit_sha!r} but repo_config is at "
                f"{repo_config.commit!r}"
            )

        step_cb = None
        if progress is not None:
            step_cb = lambda c, m: progress.step(c, m)  # noqa: E731
        scorer_progress = None
        if progress is not None:
            scorer_progress = lambda ph, st: progress.phase(ph, st)  # noqa: E731

        started = time.perf_counter()
        blue_fix, workspace = self.blue_agent.review_and_fix(
            repo_config,
            challenge_record,
            step_callback=step_cb,
            console_echo=console_echo,
        )
        defense_id = str(uuid.uuid4())
        try:
            score = self.scorer.score(
                repo_config, challenge_record, blue_fix, progress=scorer_progress
            )
        finally:
            # Preserve the Blue agent's reasoning log beside the defense
            # artifacts before the workspace (and its _swe-duel/) is removed.
            if log_dir is not None:
                src = workspace.path / "_swe-duel" / "blue.log"
                try:
                    if src.exists():
                        Path(log_dir).mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, Path(log_dir) / f"{defense_id}.blue.log")
                    elif not console_echo:
                        # In TUI mode (console echo absorbed) this log is the
                        # ONLY surviving record of the Blue agent's reasoning —
                        # losing it silently is how a defense can end up with
                        # no trace at all. Console runs stream the reasoning to
                        # the operator instead, so they stay quiet.
                        print(
                            f"[defense] WARNING: no _swe-duel/blue.log in "
                            f"the workspace to preserve for defense "
                            f"{defense_id}",
                            file=sys.stderr,
                        )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[defense] WARNING: preserving the blue.log for "
                        f"defense {defense_id} failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
            try:
                self.blue_agent.workspace_manager.cleanup(workspace)
            except Exception as e:  # noqa: BLE001
                # Residue is a disk problem, not a correctness one — but a
                # multi-MB workspace leaked per defense adds up fast, and the
                # operator needs to know before the drive fills.
                print(
                    f"[defense] WARNING: cleaning up workspace "
                    f"{workspace.workspace_id} failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
        duration = time.perf_counter() - started

        blue_model_id = self.blue_agent.agent_wrapper.model_config.model_id
        blue_harness_id = getattr(
            self.blue_agent.agent_wrapper, "harness_id", "mini-swe-agent"
        )
        # The participant's selected reasoning effort / provider ride on the
        # harness's bound ModelConfig (see ModelConfig.with_selection).
        blue_model_config = getattr(self.blue_agent.agent_wrapper, "model_config", None)
        blue_reasoning_effort = str(
            getattr(blue_model_config, "reasoning_effort", "") or ""
        )
        blue_provider = str(getattr(blue_model_config, "provider", "") or "")
        return DefenseResult(
            defense_id=defense_id,
            challenge_id=challenge_record.challenge_id,
            blue_model_id=blue_model_id,
            blue_fix=blue_fix,
            score=score,
            duration_seconds=duration,
            cost_usd=blue_fix.agent_trajectory.total_cost_usd,
            timestamp=datetime.now(timezone.utc),
            blue_harness_id=blue_harness_id,
            blue_reasoning_effort=blue_reasoning_effort,
            blue_provider=blue_provider,
        )

    def evaluate_batch(
        self,
        repo_config: RepoConfig,
        challenges: list[ChallengeRecord],
    ) -> list[DefenseResult]:
        return [self.evaluate(repo_config, ch) for ch in challenges]
