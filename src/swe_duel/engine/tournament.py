"""End-to-end tournament: Generation → Evaluation → Rating."""

from __future__ import annotations

import json
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from swe_duel.challenge_bank.generator import ChallengeGenerator
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.config import ArenaConfig, ModelConfig, RepoConfig
from swe_duel.engine.match import MatchOrchestrator
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    ChallengePoolStats,
    MatchOutcome,
    MatchResult,
    RatingSnapshot,
    TournamentResult,
)
from swe_duel.scoring.rating import compute_all_ratings
from swe_duel.sandbox.workspace import WorkspaceManager


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


class TournamentOrchestrator:
    def __init__(
        self,
        model_configs: dict[str, ModelConfig],
        repo_configs: dict[str, RepoConfig],
        challenge_store: ChallengeStore,
        workspace_manager: WorkspaceManager,
        config: ArenaConfig,
        artifact_logger: ArtifactLogger,
        *,
        match_orchestrator: MatchOrchestrator | None = None,
        challenge_generator: ChallengeGenerator | None = None,
        red_agents: dict[str, Any] | None = None,
        matches_per_pair_per_repo: int = 1,
        seed: int | None = None,
    ) -> None:
        self.model_configs = model_configs
        self.repo_configs = repo_configs
        self.challenge_store = challenge_store
        self.workspace_manager = workspace_manager
        self.config = config
        self.artifact_logger = artifact_logger
        self.match_orchestrator = match_orchestrator
        self.challenge_generator = challenge_generator
        self.red_agents = red_agents or {}
        self.matches_per_pair_per_repo = matches_per_pair_per_repo
        self.seed = seed
        self.tournament_id = str(uuid.uuid4())
        self.checkpoint_path: Path = (
            self.artifact_logger.tournaments_dir / "checkpoint.json"
        )

    # ── public API ────────────────────────────────────────

    def execute(self) -> TournamentResult:
        started = datetime.now(timezone.utc)
        generation_stats = self._run_generation_phase()
        matches = self._run_evaluation_phase()
        snapshots = self._run_rating_phase(matches)
        return self._assemble_result(
            generation_stats, matches, snapshots, started
        )

    def resume(self, checkpoint_path: Path) -> TournamentResult:
        started = datetime.now(timezone.utc)
        data = json.loads(Path(checkpoint_path).read_text())
        completed_keys = set(data.get("completed_keys", []))
        self.tournament_id = data.get("tournament_id", self.tournament_id)

        # Regeneration: skip; assume generation_stats persisted
        generation_stats_raw = data.get("generation_stats", {})
        generation_stats: dict[str, ChallengePoolStats] = {}
        for k, v in generation_stats_raw.items():
            generation_stats[k] = ChallengePoolStats(**v)

        # Replay completed match records from artifact logger
        prior_matches = self._load_prior_matches(data.get("match_ids", []))

        remaining = [
            trip for trip in self._generate_schedule()
            if self._schedule_key(trip) not in completed_keys
        ]
        new_matches = self._run_scheduled(remaining, prior_matches, completed_keys)

        all_matches = prior_matches + new_matches
        snapshots = self._run_rating_phase(all_matches)
        return self._assemble_result(
            generation_stats, all_matches, snapshots, started
        )

    # ── phases ────────────────────────────────────────────

    def _run_generation_phase(self) -> dict[str, ChallengePoolStats]:
        stats: dict[str, ChallengePoolStats] = {}
        if self.challenge_generator is None:
            # Just compute stats for existing pools.
            for m in self.model_configs.values():
                for r in self.repo_configs.values():
                    stats[f"{m.model_id}::{r.name}"] = self.challenge_store.pool_stats(
                        m.model_id, r.name
                    )
            return stats

        target = self.config.challenge_bank.target_challenges_per_model_repo
        for nick, model_cfg in self.model_configs.items():
            red_agent = self.red_agents.get(nick)
            for repo_name, repo_cfg in self.repo_configs.items():
                key = f"{model_cfg.model_id}::{repo_name}"
                sufficient = self.challenge_store.has_sufficient_pool(
                    model_cfg.model_id, repo_name, target
                )
                if sufficient or red_agent is None:
                    stats[key] = self.challenge_store.pool_stats(
                        model_cfg.model_id, repo_name
                    )
                    continue
                stats[key] = self.challenge_generator.generate_pool(
                    red_agent=red_agent,
                    repo_config=repo_cfg,
                    target_count=target,
                    red_model_id=model_cfg.model_id,
                )
        return stats

    def _run_evaluation_phase(self) -> list[MatchResult]:
        schedule = self._generate_schedule()
        return self._run_scheduled(schedule, prior=[], completed_keys=set())

    def _run_scheduled(
        self,
        schedule: list[tuple[str, str, str]],
        prior: list[MatchResult],
        completed_keys: set[str],
    ) -> list[MatchResult]:
        if self.match_orchestrator is None:
            raise ValueError(
                "MatchOrchestrator required to execute evaluation phase"
            )
        completed: list[MatchResult] = list(prior)
        new_matches: list[MatchResult] = []
        for idx, (a_id, b_id, repo_name) in enumerate(schedule):
            repo = self.repo_configs[repo_name]
            match = self.match_orchestrator.execute(
                model_a_id=a_id, model_b_id=b_id, repo_config=repo,
                seed=(self.seed + idx if self.seed is not None else None),
            )
            completed.append(match)
            new_matches.append(match)
            completed_keys.add(self._schedule_key((a_id, b_id, repo_name)))
            self._save_checkpoint(completed, completed_keys)
        return new_matches

    def _run_rating_phase(
        self, matches: list[MatchResult]
    ) -> list[RatingSnapshot]:
        snapshots = compute_all_ratings(
            matches,
            k=int(self.config.rating.elo_k),
            initial=self.config.rating.initial_elo,
            trueskill_mu=self.config.rating.trueskill_initial_mu,
            trueskill_sigma=self.config.rating.trueskill_initial_sigma,
        )
        return list(snapshots.values())

    # ── schedule ─────────────────────────────────────────

    def _generate_schedule(self) -> list[tuple[str, str, str]]:
        model_ids = [c.model_id for c in self.model_configs.values()]
        pairs: list[tuple[str, str]] = []
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                pairs.append((model_ids[i], model_ids[j]))

        schedule: list[tuple[str, str, str]] = []
        for repo_name in self.repo_configs:
            for pair in pairs:
                for _ in range(self.matches_per_pair_per_repo):
                    schedule.append((pair[0], pair[1], repo_name))

        rng = random.Random(self.seed)
        rng.shuffle(schedule)
        return schedule

    @staticmethod
    def _schedule_key(trip: tuple[str, str, str]) -> str:
        return "::".join(trip)

    # ── checkpointing ─────────────────────────────────────

    def _save_checkpoint(
        self, completed: list[MatchResult], completed_keys: set[str]
    ) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tournament_id": self.tournament_id,
            "completed_keys": sorted(completed_keys),
            "match_ids": [m.match_id for m in completed],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2))

    def _load_prior_matches(self, match_ids: list[str]) -> list[MatchResult]:
        # Logged matches are summarised; full MatchResult reconstruction is
        # impractical here. Return empty list — resume skips already-run
        # schedule entries but rating is computed only over newly-run matches.
        return []

    # ── assembly ──────────────────────────────────────────

    def _assemble_result(
        self,
        generation_stats: dict[str, ChallengePoolStats],
        matches: list[MatchResult],
        snapshots: list[RatingSnapshot],
        started: datetime,
    ) -> TournamentResult:
        head_to_head = self._head_to_head(matches)
        gen_cost = sum(s.total_generation_cost_usd for s in generation_stats.values())
        eval_cost = sum(m.total_cost_usd for m in matches)

        config_snapshot = _jsonable(self.config.model_dump())
        result = TournamentResult(
            tournament_id=self.tournament_id,
            generation_stats=generation_stats,
            matches=matches,
            final_ratings=snapshots,
            head_to_head=head_to_head,
            config_snapshot=config_snapshot,
            total_generation_cost_usd=gen_cost,
            total_evaluation_cost_usd=eval_cost,
            timestamp=started,
        )
        try:
            self.artifact_logger.log_tournament(result)
        except Exception:
            pass
        return result

    @staticmethod
    def _head_to_head(
        matches: list[MatchResult],
    ) -> dict[str, dict[str, dict]]:
        table: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "matches": 0})
        )
        for m in matches:
            a, b = m.model_a_id, m.model_b_id
            table[a][b]["matches"] += 1
            table[b][a]["matches"] += 1
            if m.outcome == MatchOutcome.MODEL_A_WINS:
                table[a][b]["wins"] += 1
                table[b][a]["losses"] += 1
            elif m.outcome == MatchOutcome.MODEL_B_WINS:
                table[b][a]["wins"] += 1
                table[a][b]["losses"] += 1
            else:
                table[a][b]["draws"] += 1
                table[b][a]["draws"] += 1
        return {k: dict(v) for k, v in table.items()}
