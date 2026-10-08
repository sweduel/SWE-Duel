"""Multi-round match between two models drawing from the Challenge Bank.

A match is decomposed into three phases so a tournament round can pool every
defense sub-turn across all of its matches into one ``ThreadPoolExecutor``:

* :meth:`MatchOrchestrator.plan_match` — deterministic challenge selection plus
  defense-cache resolution. Returns a :class:`MatchPlan` carrying the cache-miss
  :class:`DefenseTask` list (the work to run) and the already-resolved defenses
  (cache hits).
* :meth:`MatchOrchestrator.run_defense_task` — runs ONE Blue defense (a fresh,
  per-task evaluator → fresh harness + Blue agent in the repo's Docker image,
  then containerized scoring) and returns its :class:`DefenseResult`. This is
  the unit of parallelism; it carries no shared mutable state.
* :meth:`MatchOrchestrator.assemble_match` — stitches the cached + freshly-run
  defenses back into a :class:`MatchResult`.

:meth:`MatchOrchestrator.execute` chains the three sequentially and remains the
single-match entry point (used by single-match CLIs and tests). The tournament
REPL instead calls ``plan_match`` for every pairing, flattens the pending tasks
across the whole round, runs them through one pool, then ``assemble_match``.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from swe_duel.agents.blue import BlueAgent
from swe_duel.agents.harness import get_harness
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import ArenaConfig, ModelConfig, RepoConfig
from swe_duel.engine.defense_evaluator import DefenseEvaluator
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    DefenseResult,
    MatchOutcome,
    MatchResult,
    RedChallenge,
    RedValidationResult,
    TurnResult,
    TurnScore,
    split_composite_id,
)
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.sandbox.workspace import WorkspaceManager
from swe_duel.scoring.turn_scorer import TurnScorer


DEFAULT_DRAW_MARGIN = 0.25


@dataclass
class DefenseTask:
    """One Blue agent defending one challenge — the unit of round-level work.

    A task is created only for a cache MISS; cache hits are resolved during
    planning and never enter the pool. ``run_defense_task`` consumes a task and
    produces a :class:`DefenseResult`. Tasks are independent and fully
    containerized, so they parallelize across a round's ThreadPoolExecutor.
    """

    repo_config: RepoConfig
    challenge: ChallengeRecord
    blue_model_id: str          # bare model id (Blue defender)
    blue_harness_id: str
    # Identifies which match/side/repo this task belongs to so the round-level
    # pool can route the finished DefenseResult back to the right MatchPlan.
    match_key: str
    side: str                   # "b_defends_a" | "a_defends_b"
    repo_idx: int = 0           # index into MatchPlan.repo_configs
    slot: int = 0               # index within that repo's defense slot list
    # Blue's selected reasoning effort / OpenRouter provider (empty = default
    # identity) — parts of Blue's competitor 4-tuple and of the defense cache key.
    blue_reasoning_effort: str = ""
    blue_provider: str = ""


@dataclass
class MatchPlan:
    """The fully-selected, partially-cached state of one match before its
    cache-miss defenses are run.

    Created by :meth:`MatchOrchestrator.plan_match`. The defense slots are
    pre-sized lists where cache hits are populated immediately and cache misses
    are left ``None`` until the corresponding :class:`DefenseTask` completes and
    is placed back via :meth:`MatchOrchestrator.place_result`.
    """

    model_a_id: str
    model_b_id: str
    a_model: str
    b_model: str
    a_harness: str
    b_harness: str
    repo_configs: list[RepoConfig]
    match_key: str
    a_reasoning_effort: str = ""
    b_reasoning_effort: str = ""
    a_provider: str = ""
    b_provider: str = ""
    # Per-repo selected challenges and the auto-win (missing-Red) counts.
    a_challenges: list[list[ChallengeRecord]] = field(default_factory=list)
    b_challenges: list[list[ChallengeRecord]] = field(default_factory=list)
    a_missing: list[int] = field(default_factory=list)
    b_missing: list[int] = field(default_factory=list)
    # Defense slots, parallel to the per-repo challenge lists. None = pending.
    # b_defenses[repo_idx][i] = B defending A's i-th challenge for that repo.
    b_defenses: list[list[DefenseResult | None]] = field(default_factory=list)
    a_defenses: list[list[DefenseResult | None]] = field(default_factory=list)
    # The cache-miss tasks this plan still needs run (empty ⇒ ready to assemble).
    pending_tasks: list[DefenseTask] = field(default_factory=list)
    started: float = 0.0

    def cache_hits(self) -> int:
        hits = 0
        for repo_slots in (self.b_defenses, self.a_defenses):
            for slots in repo_slots:
                hits += sum(1 for s in slots if s is not None)
        return hits

    def pending_count(self) -> int:
        return len(self.pending_tasks)


class MatchOrchestrator:
    def __init__(
        self,
        model_configs: dict[str, ModelConfig],
        challenge_store: ChallengeStore,
        workspace_manager: WorkspaceManager,
        config: ArenaConfig,
        artifact_logger: ArtifactLogger,
        *,
        prompt_dir=None,
        test_runner: TestRunner | None = None,
        test_runner_factory=None,
        blue_agent_factory=None,
        defense_evaluator_factory=None,
        draw_margin: float = DEFAULT_DRAW_MARGIN,
    ) -> None:
        self.model_configs = model_configs
        self.challenge_store = challenge_store
        self.workspace_manager = workspace_manager
        self.config = config
        self.artifact_logger = artifact_logger
        self.prompt_dir = prompt_dir
        # `test_runner` is a single-repo runner (legacy / single-repo matches).
        # `test_runner_factory(repo_config) -> TestRunner` lets a multi-repo
        # match build the correct runner (image + test command + language
        # adapter) per repo.
        self.test_runner = test_runner
        self.test_runner_factory = test_runner_factory
        self.blue_agent_factory = blue_agent_factory
        self.defense_evaluator_factory = defense_evaluator_factory
        self.draw_margin = draw_margin

    # ── public ─────────────────────────────────────────────

    def execute(
        self,
        model_a_id: str,
        model_b_id: str,
        repo_config: "RepoConfig | list[RepoConfig]",
        seed: int | None = None,
    ) -> MatchResult:
        """Run a match between two competitors across one or more repos.

        ``model_a_id`` / ``model_b_id`` are **composite identities**
        (``"<model_id>#<harness_id>"``) — a competitor is a (model, harness)
        pair. They are split for challenge selection and Blue-agent
        construction; the composite forms are kept on the MatchResult so the
        Swiss/rating layers key on competitor identity.

        ``repo_config`` may be a single ``RepoConfig`` (legacy, single-repo
        match) or a list of them. A multi-repo match runs ``turns_per_player``
        turns **per repo per side** and aggregates every turn into one
        ``MatchResult`` — so the Swiss/rating layer sees one combined outcome
        per pairing.

        **Auto-win on a missing Red challenge.** A competitor may have fewer
        than ``turns_per_player`` challenges for a given repo (challenge
        generation can fail every attempt for a repo). The missing turns are
        not skipped: each is recorded as an *auto-win for the defending
        opponent* (``blue_composite=1.0``, ``red_composite=0.0``), so a repo a
        player could not generate for is never silently dropped — the opponent
        is simply handed those points.
        """
        # `seed` is unused for challenge selection (deterministic index order);
        # kept for API stability.
        plan = self.plan_match(model_a_id, model_b_id, repo_config, seed=seed)
        # Run every cache-miss defense sequentially, then assemble. The
        # tournament REPL bypasses this and runs pending tasks across the whole
        # round through a shared ThreadPoolExecutor instead.
        for task in plan.pending_tasks:
            result = self.run_defense_task(task)
            self.place_result(plan, task, result)
        return self.assemble_match(plan)

    # ── phased API (used by the round-level parallel pool) ─────────────────

    def plan_match(
        self,
        model_a_id: str,
        model_b_id: str,
        repo_config: "RepoConfig | list[RepoConfig]",
        seed: int | None = None,
    ) -> MatchPlan:
        """Select challenges + resolve the defense cache, WITHOUT running any
        Blue agent. Cache hits are filled in immediately; cache misses become
        :class:`DefenseTask` entries in ``plan.pending_tasks``.
        """
        del seed  # deterministic selection; see execute() docstring
        turns_per_player = self.config.match.turns_per_player
        repo_configs = (
            list(repo_config) if isinstance(repo_config, (list, tuple))
            else [repo_config]
        )

        a_model, a_harness, a_effort, a_provider = split_composite_id(model_a_id)
        b_model, b_harness, b_effort, b_provider = split_composite_id(model_b_id)
        a_harness = a_harness or "mini-swe-agent"
        b_harness = b_harness or "mini-swe-agent"

        match_key = str(uuid.uuid4())
        plan = MatchPlan(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            a_model=a_model,
            b_model=b_model,
            a_harness=a_harness,
            b_harness=b_harness,
            a_reasoning_effort=a_effort,
            b_reasoning_effort=b_effort,
            a_provider=a_provider,
            b_provider=b_provider,
            repo_configs=repo_configs,
            match_key=match_key,
            started=time.perf_counter(),
        )

        for repo_idx, rc in enumerate(repo_configs):
            a_challenges, a_missing = self._select_first_n(
                red_model_id=a_model, red_harness_id=a_harness,
                repo_name=rc.name, n=turns_per_player,
                red_reasoning_effort=a_effort, red_provider=a_provider,
            )
            b_challenges, b_missing = self._select_first_n(
                red_model_id=b_model, red_harness_id=b_harness,
                repo_name=rc.name, n=turns_per_player,
                red_reasoning_effort=b_effort, red_provider=b_provider,
            )
            plan.a_challenges.append(a_challenges)
            plan.b_challenges.append(b_challenges)
            plan.a_missing.append(a_missing)
            plan.b_missing.append(b_missing)

            # B defends A's challenges; A defends B's. Resolve cache per slot.
            b_slots = self._resolve_or_task(
                plan, rc, repo_idx, a_challenges,
                blue_model_id=b_model, blue_harness_id=b_harness,
                blue_reasoning_effort=b_effort, blue_provider=b_provider,
                side="b_defends_a",
            )
            a_slots = self._resolve_or_task(
                plan, rc, repo_idx, b_challenges,
                blue_model_id=a_model, blue_harness_id=a_harness,
                blue_reasoning_effort=a_effort, blue_provider=a_provider,
                side="a_defends_b",
            )
            plan.b_defenses.append(b_slots)
            plan.a_defenses.append(a_slots)

        return plan

    def _resolve_or_task(
        self,
        plan: MatchPlan,
        repo_config: RepoConfig,
        repo_idx: int,
        challenges: list[ChallengeRecord],
        *,
        blue_model_id: str,
        blue_harness_id: str,
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
        side: str,
    ) -> list[DefenseResult | None]:
        """For each challenge: fill a cached DefenseResult, or append a pending
        DefenseTask and leave the slot ``None``."""
        slots: list[DefenseResult | None] = []
        for slot_i, ch in enumerate(challenges):
            cached = self.artifact_logger.find_defense(
                challenge_id=ch.challenge_id,
                blue_model_id=blue_model_id,
                blue_harness_id=blue_harness_id,
                blue_reasoning_effort=blue_reasoning_effort,
                blue_provider=blue_provider,
            )
            if cached is not None:
                print(
                    f"[match] cache hit: blue={blue_model_id} "
                    f"harness={blue_harness_id} challenge={ch.challenge_id} "
                    f"→ defense={cached.defense_id}"
                )
                slots.append(cached)
                continue
            slots.append(None)
            plan.pending_tasks.append(
                DefenseTask(
                    repo_config=repo_config,
                    challenge=ch,
                    blue_model_id=blue_model_id,
                    blue_harness_id=blue_harness_id,
                    blue_reasoning_effort=blue_reasoning_effort,
                    blue_provider=blue_provider,
                    match_key=plan.match_key,
                    side=side,
                    repo_idx=repo_idx,
                    slot=slot_i,
                )
            )
        return slots

    def run_defense_task(
        self,
        task: DefenseTask,
        *,
        progress=None,
        console_echo: bool = True,
    ) -> DefenseResult:
        """Run one Blue defense (fresh evaluator → fresh harness + Blue agent in
        the repo image, then containerized scoring) and log its artifacts.

        This is the unit of parallelism: it touches no MatchPlan state and may
        run concurrently with other tasks. A fresh evaluator per call guarantees
        each task gets its own harness/agent (harnesses carry per-run mutable
        state and must not be shared across threads); workspaces and container
        names are UUID-unique, and artifact writes are one-file-per-id.

        ``progress`` (optional) is a per-sub-turn handle (e.g. a TUI
        ``SubturnHandle``) receiving per-step + per-phase updates; ``console_echo``
        is set False when a live TUI owns the console. The Blue agent's reasoning
        log is preserved as ``<defenses_dir>/<defense_id>.blue.log``.
        """
        evaluator = self._make_defense_evaluator(
            task.blue_model_id,
            task.blue_harness_id,
            task.repo_config,
            blue_reasoning_effort=getattr(task, "blue_reasoning_effort", "") or "",
            blue_provider=getattr(task, "blue_provider", "") or "",
        )
        if progress is not None:
            try:
                progress.start()
            except Exception:
                pass
        result = evaluator.evaluate(
            task.repo_config,
            task.challenge,
            progress=progress,
            console_echo=console_echo,
            log_dir=self.artifact_logger.defenses_dir,
        )
        try:
            self.artifact_logger.log_defense(result)
            self.artifact_logger.log_defense_html(result, task.challenge)
        except Exception as e:  # noqa: BLE001
            # The defense itself succeeded and its turn will carry the score,
            # but a lost record silently holes the defense cache: future runs
            # re-run this (challenge, blue) pair at real cost, and a
            # contribution zip can never ship it. Never swallow silently.
            print(
                f"[match] WARNING: persisting defense {result.defense_id} "
                f"(challenge {result.challenge_id}, blue={task.blue_model_id}) "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        return result

    @staticmethod
    def place_result(
        plan: MatchPlan, task: DefenseTask, result: DefenseResult
    ) -> None:
        """Place a finished DefenseResult into its plan slot."""
        slots = (
            plan.b_defenses if task.side == "b_defends_a" else plan.a_defenses
        )
        slots[task.repo_idx][task.slot] = result

    def assemble_match(self, plan: MatchPlan) -> MatchResult:
        """Stitch a fully-resolved plan into a MatchResult and log it.

        Any defense slot still ``None`` (its task failed and was dropped)
        truncates that repo's turn build the same way a short challenge list
        would, so a failed sub-turn never crashes the whole round.
        """
        all_turns: list[TurnResult] = []
        turn_index = 0
        for repo_idx, rc in enumerate(plan.repo_configs):
            # Drop slots whose defense task failed (still None), keeping each
            # challenge aligned with its defense. A dropped slot is simply not
            # scored — it neither credits nor penalizes either side.
            a_challenges, b_defenses = _pair_drop_none(
                plan.a_challenges[repo_idx], plan.b_defenses[repo_idx]
            )
            b_challenges, a_defenses = _pair_drop_none(
                plan.b_challenges[repo_idx], plan.a_defenses[repo_idx]
            )
            repo_turns = self._build_turns(
                a_challenges=a_challenges,
                b_defenses=b_defenses,
                b_challenges=b_challenges,
                a_defenses=a_defenses,
                model_a_id=plan.model_a_id,
                model_b_id=plan.model_b_id,
                a_harness_id=plan.a_harness,
                b_harness_id=plan.b_harness,
                repo_name=rc.name,
                a_missing=plan.a_missing[repo_idx],
                b_missing=plan.b_missing[repo_idx],
                start_index=turn_index,
                a_model=plan.a_model,
                b_model=plan.b_model,
                a_reasoning_effort=plan.a_reasoning_effort,
                b_reasoning_effort=plan.b_reasoning_effort,
                a_provider=plan.a_provider,
                b_provider=plan.b_provider,
            )
            all_turns.extend(repo_turns)
            turn_index += len(repo_turns)

        turns = all_turns
        score_a, score_b = self._aggregate_scores(
            turns, plan.model_a_id, plan.model_b_id
        )
        outcome = self._determine_outcome(score_a, score_b)
        duration = time.perf_counter() - plan.started
        total_cost = sum(r.defense_result.cost_usd for r in turns)

        result = MatchResult(
            match_id=str(uuid.uuid4()),
            model_a_id=plan.model_a_id,
            model_b_id=plan.model_b_id,
            repo_name=",".join(rc.name for rc in plan.repo_configs),
            turns=turns,
            model_a_total=score_a,
            model_b_total=score_b,
            outcome=outcome,
            duration_seconds=duration,
            total_cost_usd=total_cost,
            timestamp=datetime.now(timezone.utc),
        )
        try:
            self.artifact_logger.log_match(result)
        except Exception:
            pass
        return result

    # ── internals ─────────────────────────────────────────

    def _select_first_n(
        self,
        red_model_id: str,
        repo_name: str,
        n: int,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> tuple[list[ChallengeRecord], int]:
        """Deterministically pick the first `n` challenges for (red_model,
        harness, effort, provider, repo) in index.json order. Every Blue agent
        in the tournament therefore faces the same challenge set per Red
        competitor.

        Returns ``(picked, missing)`` where ``missing = max(0, n - len(picked))``
        is the number of challenge slots this Red competitor could NOT fill for
        this repo (challenge generation failed every attempt). The caller turns
        each missing slot into an auto-win for the defending opponent instead of
        aborting the whole match — a repo a player can't generate for must not
        exclude that repo from the tournament.
        """
        records = self.challenge_store.list_in_index_order(
            red_model_id=red_model_id,
            repo_name=repo_name,
            red_harness_id=red_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )
        picked = records[:n]
        missing = max(0, n - len(picked))
        ident = (
            f"red={red_model_id} harness={red_harness_id}"
            + (f" effort={red_reasoning_effort}" if red_reasoning_effort else "")
            + (f" provider={red_provider}" if red_provider else "")
        )
        if missing:
            print(
                f"[match] {ident} repo={repo_name}: "
                f"only {len(picked)}/{n} challenges — "
                f"{missing} auto-win(s) to the defender"
            )
        else:
            print(
                f"[match] using first {len(picked)} challenges (deterministic) "
                f"for {ident} repo={repo_name}"
            )
        return picked, missing

    def _make_defense_evaluator(
        self,
        blue_model_id: str,
        blue_harness_id: str = "mini-swe-agent",
        repo_config: RepoConfig | None = None,
        *,
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
    ) -> DefenseEvaluator:
        if self.defense_evaluator_factory is not None:
            return self.defense_evaluator_factory(
                blue_model_id, blue_harness_id,
                blue_reasoning_effort, blue_provider,
            )

        if self.blue_agent_factory is not None:
            blue_agent = self.blue_agent_factory(
                blue_model_id, blue_harness_id,
                blue_reasoning_effort, blue_provider,
            )
        else:
            model_cfg = self._resolve_model_config(
                blue_model_id, blue_harness_id,
                blue_reasoning_effort, blue_provider,
            )
            harness = get_harness(blue_harness_id, model_cfg)
            if self.prompt_dir is None:
                raise ValueError(
                    "MatchOrchestrator needs `prompt_dir` to instantiate BlueAgent "
                    "without an explicit blue_agent_factory"
                )
            blue_agent = BlueAgent(
                agent_wrapper=harness,
                workspace_manager=self.workspace_manager,
                prompt_dir=self.prompt_dir,
                wall_seconds=self.config.agent_timeouts.blue_seconds,
                steps=self.config.agent_steps.blue_steps,
            )

        # Pick the repo-appropriate TestRunner: prefer the per-repo factory
        # (multi-repo matches), else fall back to the single bound runner.
        test_runner = None
        if self.test_runner_factory is not None and repo_config is not None:
            test_runner = self.test_runner_factory(repo_config)
        if test_runner is None:
            test_runner = self.test_runner
        if test_runner is None:
            raise ValueError(
                "MatchOrchestrator needs `test_runner` or `test_runner_factory` "
                "to build a TurnScorer"
            )
        scorer = TurnScorer(test_runner=test_runner, config=self.config)
        return DefenseEvaluator(
            blue_agent=blue_agent, scorer=scorer, config=self.config
        )

    def _resolve_model_config(
        self,
        model_id: str,
        harness_id: str = "mini-swe-agent",
        reasoning_effort: str = "",
        provider: str = "",
    ) -> ModelConfig:
        """ModelConfig for a Blue participant, selection applied.

        Lookup order:
        1. Exact participant composite id (``model#harness[#effort#provider]``)
           — what the tournament scripts pass, so each entrant gets its own
           pre-selected config.
        2. Any config whose ``model_id`` matches (legacy callers that key the
           dict by nickname or bare model id).
        3. The dict key itself (model nickname).

        In every case the participant's selected effort/provider are applied on
        top of the found base config via :meth:`ModelConfig.with_selection`.
        """
        from swe_duel.models import composite_id

        cid = composite_id(model_id, harness_id, reasoning_effort, provider)
        if cid in self.model_configs:
            return self.model_configs[cid]

        base: ModelConfig | None = None
        for cfg in self.model_configs.values():
            if cfg.model_id == model_id:
                base = cfg
                break
        if base is None and model_id in self.model_configs:
            base = self.model_configs[model_id]
        if base is None:
            raise KeyError(f"No ModelConfig for {model_id!r}")
        if not reasoning_effort and not provider:
            return base
        return base.with_selection(reasoning_effort, provider)

    @staticmethod
    def _build_turns(
        a_challenges: list[ChallengeRecord],
        b_defenses: list[DefenseResult],
        b_challenges: list[ChallengeRecord],
        a_defenses: list[DefenseResult],
        model_a_id: str,
        model_b_id: str,
        a_harness_id: str = "mini-swe-agent",
        b_harness_id: str = "mini-swe-agent",
        repo_name: str = "",
        a_missing: int = 0,
        b_missing: int = 0,
        start_index: int = 0,
        a_model: str = "",
        b_model: str = "",
        a_reasoning_effort: str = "",
        b_reasoning_effort: str = "",
        a_provider: str = "",
        b_provider: str = "",
    ) -> list[TurnResult]:
        """Interleave A-Red/B-Blue and B-Red/A-Blue turns for one repo.

        ``a_missing`` / ``b_missing`` are the number of challenge slots A / B
        (as Red) could not fill for this repo; each is emitted as an auto-win
        turn crediting the defending opponent (see ``_auto_win_turn``).
        """
        turns: list[TurnResult] = []
        idx = start_index
        n = max(len(a_challenges), len(b_challenges))
        for i in range(n):
            # A-Red / B-Blue
            if i < len(a_challenges) and i < len(b_defenses):
                turns.append(
                    TurnResult(
                        turn_id=str(uuid.uuid4()),
                        turn_index=idx,
                        challenge_record=a_challenges[i],
                        defense_result=b_defenses[i],
                        red_model_id=model_a_id,
                        blue_model_id=model_b_id,
                        red_harness_id=a_harness_id,
                        blue_harness_id=b_harness_id,
                        red_reasoning_effort=a_reasoning_effort,
                        red_provider=a_provider,
                        blue_reasoning_effort=b_reasoning_effort,
                        blue_provider=b_provider,
                    )
                )
                idx += 1
            # B-Red / A-Blue
            if i < len(b_challenges) and i < len(a_defenses):
                turns.append(
                    TurnResult(
                        turn_id=str(uuid.uuid4()),
                        turn_index=idx,
                        challenge_record=b_challenges[i],
                        defense_result=a_defenses[i],
                        red_model_id=model_b_id,
                        blue_model_id=model_a_id,
                        red_harness_id=b_harness_id,
                        blue_harness_id=a_harness_id,
                        red_reasoning_effort=b_reasoning_effort,
                        red_provider=b_provider,
                        blue_reasoning_effort=a_reasoning_effort,
                        blue_provider=a_provider,
                    )
                )
                idx += 1

        # ── Auto-win turns for challenge slots Red could not fill ──
        # A missing as Red → B (defender) wins by default, and vice-versa.
        for _ in range(a_missing):
            turns.append(
                MatchOrchestrator._auto_win_turn(
                    turn_index=idx,
                    red_model_id=model_a_id, blue_model_id=model_b_id,
                    red_harness_id=a_harness_id, blue_harness_id=b_harness_id,
                    red_model=a_model, blue_model=b_model,
                    repo_name=repo_name,
                    red_reasoning_effort=a_reasoning_effort,
                    red_provider=a_provider,
                    blue_reasoning_effort=b_reasoning_effort,
                    blue_provider=b_provider,
                )
            )
            idx += 1
        for _ in range(b_missing):
            turns.append(
                MatchOrchestrator._auto_win_turn(
                    turn_index=idx,
                    red_model_id=model_b_id, blue_model_id=model_a_id,
                    red_harness_id=b_harness_id, blue_harness_id=a_harness_id,
                    red_model=b_model, blue_model=a_model,
                    repo_name=repo_name,
                    red_reasoning_effort=b_reasoning_effort,
                    red_provider=b_provider,
                    blue_reasoning_effort=a_reasoning_effort,
                    blue_provider=a_provider,
                )
            )
            idx += 1
        return turns

    @staticmethod
    def _auto_win_turn(
        *,
        turn_index: int,
        red_model_id: str,
        blue_model_id: str,
        red_harness_id: str,
        blue_harness_id: str,
        red_model: str,
        blue_model: str,
        repo_name: str,
        red_reasoning_effort: str = "",
        red_provider: str = "",
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
    ) -> TurnResult:
        """A turn for a repo where Red has no challenge → Blue wins by default.

        The defending opponent (Blue) is credited a full win
        (``blue_composite=1.0``, ``red_composite=0.0``). A synthetic, clearly
        marked placeholder challenge/defense carries no diff or trajectory, so
        scoring/reporting can recognise it as an auto-win rather than a real
        defended turn.
        """
        empty_traj = AgentTrajectory(
            steps=[], total_steps=0, total_input_tokens=0,
            total_output_tokens=0, total_cost_usd=0.0,
            model_id=blue_model, duration_seconds=0.0,
            exit_status="auto_win_red_missing_challenge",
        )
        score = TurnScore(
            s_regression=1.0, s_feature=1.0, s_bugfix=1.0,
            blue_composite=1.0, red_composite=0.0,
            test_details={"auto_win": True, "reason": "red_missing_challenge"},
        )
        defense = DefenseResult(
            defense_id=str(uuid.uuid4()),
            challenge_id="",
            blue_model_id=blue_model,
            blue_fix=BlueFix(
                review_findings=[], fix_explanation="auto-win: Red had no "
                "challenge for this repo", fix_diff="",
                modified_file_contents={}, agent_trajectory=empty_traj,
            ),
            score=score,
            duration_seconds=0.0,
            cost_usd=0.0,
            timestamp=datetime.now(timezone.utc),
            blue_harness_id=blue_harness_id,
            blue_reasoning_effort=blue_reasoning_effort,
            blue_provider=blue_provider,
        )
        placeholder = ChallengeRecord(
            challenge_id="",
            red_model_id=red_model,
            repo_name=repo_name,
            repo_commit_sha="",
            target_files=[],
            challenge=RedChallenge(
                target_files=[], exploration_summary="",
                feature_spec="", feature_rationale="", pr_diff="",
                modified_file_contents={}, original_file_contents={},
                feature_test_code="", bug_type=None, bug_description=None,
                bug_location=None, bug_test_code=None,
                agent_trajectory=empty_traj,
            ),
            validation=RedValidationResult(
                passed=False, gate_results=[], attempt_number=0
            ),
            generated_at=datetime.now(timezone.utc),
            generation_cost_usd=0.0,
            generation_retries=0,
            red_harness_id=red_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )
        return TurnResult(
            turn_id=str(uuid.uuid4()),
            turn_index=turn_index,
            challenge_record=placeholder,
            defense_result=defense,
            red_model_id=red_model_id,
            blue_model_id=blue_model_id,
            red_harness_id=red_harness_id,
            blue_harness_id=blue_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
            blue_reasoning_effort=blue_reasoning_effort,
            blue_provider=blue_provider,
        )

    @staticmethod
    def _aggregate_scores(
        turns: list[TurnResult], model_a_id: str, model_b_id: str
    ) -> tuple[float, float]:
        score_a = 0.0
        score_b = 0.0
        for r in turns:
            sc = r.defense_result.score
            if r.red_model_id == model_a_id:
                score_a += sc.red_composite
            if r.red_model_id == model_b_id:
                score_b += sc.red_composite
            if r.blue_model_id == model_a_id:
                score_a += sc.blue_composite
            if r.blue_model_id == model_b_id:
                score_b += sc.blue_composite
        return score_a, score_b

    def _determine_outcome(self, score_a: float, score_b: float) -> MatchOutcome:
        if score_a > score_b + self.draw_margin:
            return MatchOutcome.MODEL_A_WINS
        if score_b > score_a + self.draw_margin:
            return MatchOutcome.MODEL_B_WINS
        return MatchOutcome.DRAW


def _pair_drop_none(
    challenges: list[ChallengeRecord],
    defenses: list[DefenseResult | None],
) -> tuple[list[ChallengeRecord], list[DefenseResult]]:
    """Zip challenges with their (positional) defense slots, dropping any pair
    whose defense is ``None`` (its task failed and was discarded).

    Keeps each surviving challenge aligned with its defense so ``_build_turns``
    pairs them correctly. ``defenses`` may be shorter than ``challenges`` (no
    slot was ever created); those trailing challenges are dropped too.
    """
    kept_ch: list[ChallengeRecord] = []
    kept_def: list[DefenseResult] = []
    for i, ch in enumerate(challenges):
        d = defenses[i] if i < len(defenses) else None
        if d is None:
            continue
        kept_ch.append(ch)
        kept_def.append(d)
    return kept_ch, kept_def
