"""Seeded bootstrap diagnostics over a tournament's merged match history.

One pass produces, per participant: the match-level Bradley--Terry point
estimate + percentile CI, the contested-turn BT point + CI, the Elo point
+ CI, and the full bootstrap rank histogram (median rank + central 95%
interval). Consumers: ``swe-duel-rankings`` (writes the CIs / rank
statistics into ``rankings_<id>.json`` so every export carries fresh
uncertainty for the whole field) and ``swe-duel-tournament-update``
(renders the leaderboard's ``Strength diagnostics`` card and ``Bootstrap
rank distribution`` histogram, for appended late entrants AND refreshed
incumbent rows).

Determinism: the fixed seed makes re-exports reproduce identical numbers.
A resample in which a participant's matches were all left out (possible
for a late entrant — its 5-of-55 match share means ~0.5% of resamples) is
counted as *unranked* for that participant and skipped in its CI draws,
never fabricated as rank 0 / strength 0.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from swe_duel.models import MatchOutcome, MatchResult
from swe_duel.scoring.rating import BradleyTerryRating, EloRating

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261006


@dataclass
class BTMatch:
    """Minimal Bradley--Terry observation: a match outcome, or one contested
    turn recast as a Red-vs-Blue duel (auto-wins carry no observation)."""

    model_a_id: str
    model_b_id: str
    outcome: MatchOutcome


@dataclass
class StrengthDiagnostics:
    """Bootstrap strength diagnostics for one participant: match-level
    Bradley--Terry point + CI, contested-turn BT point + CI, Elo point +
    CI, and the full bootstrap rank histogram (median rank + central 95%
    interval) over the merged tournament history."""

    draws: int
    ranked: int
    n_participants: int
    rank_hist: list[float]
    median_rank: int
    rank_lo: int
    rank_hi: int
    bt: float
    bt_lo: float
    bt_hi: float
    contested: float
    contested_lo: float
    contested_hi: float
    elo: float
    elo_lo: float
    elo_hi: float


def match_level_observations(combined: list[dict[str, Any]]) -> list[MatchResult]:
    """One BT observation per merged match (win/draw/loss as 1/0.5/0)."""
    out: list[BTMatch] = []
    for raw in combined:
        a = str(raw.get("model_a_id") or "")
        b = str(raw.get("model_b_id") or "")
        if not a or not b or a == b:
            continue
        out.append(BTMatch(model_a_id=a, model_b_id=b, outcome=MatchOutcome(str(raw.get("outcome") or "draw"))))
    return cast("list[MatchResult]", out)


def contested_turn_matches(combined: list[dict[str, Any]]) -> list[MatchResult]:
    """One BT observation per CONTESTED turn: Red vs Blue, Red winning when
    its bug shipped (``red_composite`` > 0.5). A missing-Red auto-win turn
    carries an empty ``challenge_id`` and a synthetic defense — no defense
    ran, so it contributes no strength observation."""
    out: list[BTMatch] = []
    for raw in combined:
        for turn in raw.get("turns") or []:
            if not isinstance(turn, dict) or not turn.get("challenge_id"):
                continue
            red = str(turn.get("red_model_id") or "")
            blue = str(turn.get("blue_model_id") or "")
            if not red or not blue or red == blue:
                continue
            red_composite = float((turn.get("score") or {}).get("red_composite") or 0.0)
            if red_composite > 0.5:
                outcome = MatchOutcome.MODEL_A_WINS
            elif red_composite < 0.5:
                outcome = MatchOutcome.MODEL_B_WINS
            else:
                outcome = MatchOutcome.DRAW
            out.append(BTMatch(model_a_id=red, model_b_id=blue, outcome=outcome))
    return cast("list[MatchResult]", out)


def _pct(sorted_values: list[Any], q: float) -> Any:
    """The q-th percentile by index of an (unsorted) values list."""
    if not sorted_values:
        return 0
    ordered = sorted(sorted_values)
    idx = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[idx]


def bootstrap_field_diagnostics(
    combined: list[dict[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, StrengthDiagnostics]:
    """One seeded match-level bootstrap over the merged tournament history
    (original schedule + imported active-sampled matches), producing every
    participant's rank histogram, match-level BT CI, contested-turn BT CI,
    and Elo CI."""
    matches = match_level_observations(combined)
    if not matches:
        return {}
    contested = contested_turn_matches(combined)
    rng = random.Random(seed)

    bt_point = BradleyTerryRating().fit(matches)
    contested_point = BradleyTerryRating().fit(contested) if contested else {}

    def _elo_of(sample: list[MatchResult]) -> dict[str, float]:
        elo = EloRating()
        for m in sample:
            elo.update(m.model_a_id, m.model_b_id, m.outcome)
        return dict(elo.ratings)

    elo_point = _elo_of(matches)

    players = sorted(bt_point)
    n = len(matches)
    nc = len(contested)
    rank_counts: dict[str, Counter[int]] = {p: Counter() for p in players}
    bt_boot: dict[str, list[float]] = {p: [] for p in players}
    elo_boot: dict[str, list[float]] = {p: [] for p in players}
    contested_boot: dict[str, list[float]] = {p: [] for p in players}

    for _ in range(draws):
        sample = [matches[rng.randrange(n)] for _ in range(n)]
        strengths = BradleyTerryRating().fit(sample)
        order = sorted(strengths, key=lambda m: (-strengths[m], m))
        for pos, pid in enumerate(order, 1):
            rank_counts[pid][pos] += 1
        for pid, value in strengths.items():
            bt_boot[pid].append(value)
        for pid, value in _elo_of(sample).items():
            elo_boot[pid].append(value)
        if nc:
            c_sample = [contested[rng.randrange(nc)] for _ in range(nc)]
            c_strengths = BradleyTerryRating().fit(c_sample)
            for pid, value in c_strengths.items():
                contested_boot[pid].append(value)

    out: dict[str, StrengthDiagnostics] = {}
    for pid in players:
        ranked = sum(rank_counts[pid].values())
        if not ranked:
            continue
        ranks_sorted = sorted(rank_counts[pid].elements())
        out[pid] = StrengthDiagnostics(
            draws=draws,
            ranked=ranked,
            n_participants=len(players),
            rank_hist=[
                rank_counts[pid].get(r, 0) / ranked for r in range(1, len(players) + 1)
            ],
            median_rank=int(statistics.median_low(ranks_sorted)),
            rank_lo=int(ranks_sorted[int(0.025 * len(ranks_sorted))]),
            rank_hi=int(
                ranks_sorted[min(len(ranks_sorted) - 1, int(0.975 * len(ranks_sorted)))]
            ),
            bt=bt_point.get(pid, 0.0),
            bt_lo=float(_pct(bt_boot[pid], 0.025)),
            bt_hi=float(_pct(bt_boot[pid], 0.975)),
            contested=contested_point.get(pid, 0.0),
            contested_lo=float(_pct(contested_boot[pid], 0.025)),
            contested_hi=float(_pct(contested_boot[pid], 0.975)),
            elo=float(elo_point.get(pid, 1500.0)),
            elo_lo=float(_pct(elo_boot[pid], 0.025) or 1500.0),
            elo_hi=float(_pct(elo_boot[pid], 0.975) or 1500.0),
        )
    return out