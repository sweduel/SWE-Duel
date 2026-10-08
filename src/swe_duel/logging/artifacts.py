"""Persist round/match/rating/tournament artefacts as JSON on disk."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    DefenseResult,
    MatchResult,
    RatingSnapshot,
    ReviewFinding,
    TurnScore,
    TournamentResult,
)
from swe_duel.sandbox.diff_utils import render_defense_html

# (challenge_id, blue_model_id, blue_harness_id, blue_reasoning_effort,
#  blue_provider) → (timestamp, path)
_DefenseIndexKey = tuple[str, str, str, str, str]
_DefenseIndexEntry = tuple[str, Path]
_DefenseIndex = dict[_DefenseIndexKey, _DefenseIndexEntry]


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


def _defense_lookup_key(
    challenge_id: str,
    blue_model_id: str,
    blue_harness_id: str = "mini-swe-agent",
    blue_reasoning_effort: str = "",
    blue_provider: str = "",
) -> _DefenseIndexKey:
    return (
        challenge_id, blue_model_id, blue_harness_id,
        blue_reasoning_effort, blue_provider,
    )


def _timestamp_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return ""


class ArtifactLogger:
    def __init__(
        self, data_dir: Path, defenses_dir: Path | None = None
    ) -> None:
        self.data_dir = Path(data_dir)
        self.defenses_dir = (
            Path(defenses_dir)
            if defenses_dir is not None
            else self.data_dir / "defenses"
        )
        self.matches_dir = self.data_dir / "matches"
        self.ratings_dir = self.data_dir / "ratings"
        self.tournaments_dir = self.data_dir / "tournaments"
        for d in (
            self.defenses_dir,
            self.matches_dir,
            self.ratings_dir,
            self.tournaments_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        # Lazy in-memory index: most-recent defense path per
        # (challenge_id, blue_model_id, blue_harness_id, blue_reasoning_effort,
        # blue_provider). Built once on first lookup / existence check; kept
        # current by log_defense.
        self._defense_index: _DefenseIndex | None = None
        self._defense_index_lock = threading.Lock()

    # ── writes ────────────────────────────────────────────

    def log_defense(self, result: DefenseResult) -> Path:
        path = self.defenses_dir / f"{result.defense_id}.json"
        path.write_text(json.dumps(_jsonable(result), indent=2, default=str))
        key = _defense_lookup_key(
            result.challenge_id,
            result.blue_model_id,
            result.blue_harness_id or "mini-swe-agent",
            getattr(result, "blue_reasoning_effort", "") or "",
            getattr(result, "blue_provider", "") or "",
        )
        ts = _timestamp_str(result.timestamp)
        with self._defense_index_lock:
            if self._defense_index is None:
                self._defense_index = self._scan_defense_index()
            prev = self._defense_index.get(key)
            if prev is None or ts >= prev[0]:
                self._defense_index[key] = (ts, path)
        return path

    def log_defense_html(
        self, result: DefenseResult, challenge_record: ChallengeRecord
    ) -> Path:
        ch = challenge_record.challenge
        blue_files = {
            rel: result.blue_fix.modified_file_contents.get(
                rel, ch.modified_file_contents[rel]
            )
            for rel in ch.modified_file_contents
        }
        for rel, content in result.blue_fix.modified_file_contents.items():
            blue_files.setdefault(rel, content)

        html_str = render_defense_html(
            defense_id=result.defense_id,
            challenge_id=result.challenge_id,
            red_model_id=challenge_record.red_model_id,
            blue_model_id=result.blue_model_id,
            repo_name=challenge_record.repo_name,
            target_files=list(challenge_record.target_files),
            feature_spec=ch.feature_spec,
            bug_type=ch.bug_type,
            bug_description=ch.bug_description,
            bug_location=ch.bug_location,
            review_findings=[
                (f.location, f.severity, f.description)
                for f in result.blue_fix.review_findings
            ],
            fix_explanation=result.blue_fix.fix_explanation,
            original_file_contents=dict(ch.original_file_contents),
            red_file_contents=dict(ch.modified_file_contents),
            blue_file_contents=blue_files,
            s_regression=result.score.s_regression,
            s_feature=result.score.s_feature,
            s_bugfix=result.score.s_bugfix,
            blue_composite=result.score.blue_composite,
            blue_trajectory_steps=list(result.blue_fix.agent_trajectory.steps),
            test_details=dict(result.score.test_details or {}),
        )
        path = self.defenses_dir / f"{result.defense_id}.html"
        path.write_text(html_str)
        return path

    # ── defense index ─────────────────────────────────────

    def _scan_defense_index(self) -> _DefenseIndex:
        """Full-directory bootstrap: keep only the most recent path per key.

        Legacy defenses written before harness selection default to
        mini-swe-agent, and defenses written before effort/provider selection
        default to empty selections, so they keep matching the default identity.
        """
        index: _DefenseIndex = {}
        for p in self.defenses_dir.glob("*.json"):
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            challenge_id = payload.get("challenge_id")
            blue_model_id = payload.get("blue_model_id")
            if not isinstance(challenge_id, str) or not isinstance(blue_model_id, str):
                continue
            harness_raw = payload.get("blue_harness_id", "mini-swe-agent")
            harness = (
                harness_raw
                if isinstance(harness_raw, str) and harness_raw
                else "mini-swe-agent"
            )
            effort_raw = payload.get("blue_reasoning_effort", "")
            effort = effort_raw if isinstance(effort_raw, str) else ""
            provider_raw = payload.get("blue_provider", "")
            provider = provider_raw if isinstance(provider_raw, str) else ""
            ts = _timestamp_str(payload.get("timestamp", ""))
            key = _defense_lookup_key(
                challenge_id, blue_model_id, harness, effort, provider
            )
            prev = index.get(key)
            if prev is None or ts >= prev[0]:
                index[key] = (ts, p)
        return index

    def _ensure_defense_index(self) -> _DefenseIndex:
        with self._defense_index_lock:
            if self._defense_index is None:
                self._defense_index = self._scan_defense_index()
            return self._defense_index

    def invalidate_defense_index(self) -> None:
        """Drop the cached index (next lookup rescans defenses_dir)."""
        with self._defense_index_lock:
            self._defense_index = None

    # ── reads ─────────────────────────────────────────────

    def has_defense(
        self, challenge_id: str, blue_model_id: str,
        blue_harness_id: str = "mini-swe-agent",
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
    ) -> bool:
        """True if a prior defense exists for this Blue competitor identity.

        Index-only — does not load or deserialize the defense body.
        """
        key = _defense_lookup_key(
            challenge_id, blue_model_id, blue_harness_id,
            blue_reasoning_effort, blue_provider,
        )
        return key in self._ensure_defense_index()

    def find_defense(
        self, challenge_id: str, blue_model_id: str,
        blue_harness_id: str = "mini-swe-agent",
        blue_reasoning_effort: str = "",
        blue_provider: str = "",
    ) -> DefenseResult | None:
        """Return a prior DefenseResult for (challenge_id, blue_model_id,
        blue_harness_id, blue_reasoning_effort, blue_provider), else None.
        Blue's harness, effort and provider are part of its competitor identity,
        so a defense by the same model under a different harness / effort /
        provider is a cache miss. Uses an in-memory path index; loads at most
        one JSON file.
        """
        key = _defense_lookup_key(
            challenge_id, blue_model_id, blue_harness_id,
            blue_reasoning_effort, blue_provider,
        )
        index = self._ensure_defense_index()
        entry = index.get(key)
        if entry is None:
            return None
        path = entry[1]
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return _defense_from_dict(payload)

    def log_match(self, result: MatchResult) -> Path:
        payload = {
            "match_id": result.match_id,
            "model_a_id": result.model_a_id,
            "model_b_id": result.model_b_id,
            "repo_name": result.repo_name,
            "model_a_total": result.model_a_total,
            "model_b_total": result.model_b_total,
            "outcome": result.outcome.value,
            "duration_seconds": result.duration_seconds,
            "total_cost_usd": result.total_cost_usd,
            "timestamp": result.timestamp.isoformat(),
            "turns": [
                {
                    "turn_id": r.turn_id,
                    "turn_index": r.turn_index,
                    "red_model_id": r.red_model_id,
                    "blue_model_id": r.blue_model_id,
                    "red_harness_id": r.red_harness_id,
                    "blue_harness_id": r.blue_harness_id,
                    "red_reasoning_effort": getattr(r, "red_reasoning_effort", "") or "",
                    "red_provider": getattr(r, "red_provider", "") or "",
                    "blue_reasoning_effort": getattr(r, "blue_reasoning_effort", "") or "",
                    "blue_provider": getattr(r, "blue_provider", "") or "",
                    "challenge_id": r.challenge_record.challenge_id,
                    "defense_id": r.defense_result.defense_id,
                    # Per-turn repo: a match spans multiple repos, and an
                    # auto-win turn has an empty challenge_id, so the repo can't
                    # be recovered from the challenge index. Persisting it here
                    # lets the report group every turn (auto-win included) under
                    # its real repo instead of one shared "(auto-win)" bucket.
                    "repo_name": r.challenge_record.repo_name,
                    "score": _jsonable(r.defense_result.score),
                }
                for r in result.turns
            ],
        }
        path = self.matches_dir / f"{result.match_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_rating_snapshot(self, ratings: list[RatingSnapshot]) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.ratings_dir / f"ratings_{timestamp}.json"
        payload = {"ratings": [_jsonable(r) for r in ratings]}
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_tournament(self, result: TournamentResult) -> Path:
        path = self.tournaments_dir / f"{result.tournament_id}.json"
        path.write_text(json.dumps(_jsonable(result), indent=2, default=str))
        return path


def _defense_from_dict(d: dict) -> DefenseResult:
    traj_d = d["blue_fix"]["agent_trajectory"]
    trajectory = AgentTrajectory(
        steps=traj_d.get("steps", []),
        total_steps=traj_d.get("total_steps", 0),
        total_input_tokens=traj_d.get("total_input_tokens", 0),
        total_output_tokens=traj_d.get("total_output_tokens", 0),
        total_cost_usd=traj_d.get("total_cost_usd", 0.0),
        model_id=traj_d.get("model_id", ""),
        duration_seconds=traj_d.get("duration_seconds", 0.0),
    )
    bf = d["blue_fix"]
    blue_fix = BlueFix(
        review_findings=[ReviewFinding(**f) for f in bf.get("review_findings", [])],
        fix_explanation=bf.get("fix_explanation", ""),
        fix_diff=bf.get("fix_diff", ""),
        modified_file_contents=bf.get("modified_file_contents", {}),
        agent_trajectory=trajectory,
    )
    sc = d["score"]
    score = TurnScore(
        s_regression=sc["s_regression"],
        s_feature=sc["s_feature"],
        s_bugfix=sc["s_bugfix"],
        blue_composite=sc["blue_composite"],
        red_composite=sc["red_composite"],
        test_details=sc.get("test_details", {}),
    )
    ts = d["timestamp"]
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return DefenseResult(
        defense_id=d["defense_id"],
        challenge_id=d["challenge_id"],
        blue_model_id=d["blue_model_id"],
        blue_fix=blue_fix,
        score=score,
        duration_seconds=d["duration_seconds"],
        cost_usd=d["cost_usd"],
        timestamp=ts,
        blue_harness_id=d.get("blue_harness_id", "mini-swe-agent"),
        blue_reasoning_effort=str(d.get("blue_reasoning_effort", "") or ""),
        blue_provider=str(d.get("blue_provider", "") or ""),
    )
