"""Swiss-system tournament pairing.

Tracks per-player Swiss points across rounds and pairs by closest current
score, avoiding rematches when possible. Byes (odd N) award 1 point to the
lowest-scoring player who has not yet had a bye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx


@dataclass
class SwissPlayer:
    model_id: str
    points: float = 0.0
    opponents: list[str] = field(default_factory=list)
    had_bye: bool = False


@dataclass
class SwissPairing:
    model_a: str
    model_b: str | None  # None == bye for model_a


class SwissTournament:
    """In-memory Swiss bookkeeping.

    Round 0 = the initial pairing (typically random or rating-based). Subsequent
    rounds pair by descending score, scanning for non-repeat partners.

    **Late entrants.** :meth:`add_players` seats newcomers at 0 Swiss points
    (no prior opponents). The next :meth:`pair_next_round` naturally folds them
    into score-group matching; no special intake budget is needed because Swiss
    never promises every-vs-every pairing.
    """

    def __init__(self, model_ids: Iterable[str]) -> None:
        self.players: dict[str, SwissPlayer] = {
            m: SwissPlayer(model_id=m) for m in model_ids
        }
        self.round_history: list[list[SwissPairing]] = []
        # Late entrants registered after the original field was seated. Kept for
        # state persistence / TUI annotations; pairing treats them as normal
        # zero-point players.
        self.intake_players: list[str] = []

    # ── public ───────────────────────────────────────────────

    @property
    def n_models(self) -> int:
        return len(self.players)

    @property
    def recommended_rounds(self) -> int:
        n = self.n_models
        return max(1, math.ceil(math.log2(n))) if n > 1 else 0

    @property
    def current_round_index(self) -> int:
        return len(self.round_history)

    def standings(self) -> list[SwissPlayer]:
        return sorted(
            self.players.values(),
            key=lambda p: (-p.points, p.model_id),
        )

    def add_players(self, model_ids: Iterable[str]) -> list[str]:
        """Seat late entrants at 0 points. Returns ids actually added."""
        added: list[str] = []
        for mid in model_ids:
            if mid in self.players:
                continue
            self.players[mid] = SwissPlayer(model_id=mid)
            self.intake_players.append(mid)
            added.append(mid)
        return added

    def pair_next_round(self, *, initial_seed_by: list[str] | None = None) -> list[SwissPairing]:
        """Return the pairings for the next round. Does NOT record them yet —
        call ``record_result`` after each pairing's match completes."""
        if not self.round_history and initial_seed_by:
            order = [m for m in initial_seed_by if m in self.players]
            order += [m for m in self.players if m not in order]
        else:
            standings = self.standings()
            order = [p.model_id for p in standings]
        return self._greedy_pair(order)

    def commit_round(self, pairings: list[SwissPairing]) -> None:
        self.round_history.append(pairings)
        for p in pairings:
            if p.model_b is None:
                # Bye
                self.players[p.model_a].had_bye = True
                self.players[p.model_a].points += 1.0
            else:
                self.players[p.model_a].opponents.append(p.model_b)
                self.players[p.model_b].opponents.append(p.model_a)

    def record_result(self, model_a: str, model_b: str, score_a: float) -> None:
        """Apply Swiss points after a (non-bye) match. score_a∈{0,0.5,1}."""
        self.players[model_a].points += score_a
        self.players[model_b].points += 1.0 - score_a

    # ── internals ────────────────────────────────────────────

    def _greedy_pair(self, ordered_ids: list[str]) -> list[SwissPairing]:
        """Pair the round via minimum-weight perfect matching.

        Builds a complete graph over the unpaired models with edge weights
        equal to the squared score gap plus a large penalty for rematches,
        then runs NetworkX's blossom-algorithm matching. This finds a
        rematch-free pairing whenever one exists (unlike the previous greedy
        approach, which could paint itself into a corner where the last two
        unpaired players had already played each other).

        Handles odd counts by giving a bye to the lowest-scoring player who
        has not yet had one. The output is sorted with the higher-ranked
        player of each pair listed first and pairs ordered by that player's
        position in ``ordered_ids``, matching the previous convention."""
        remaining = list(ordered_ids)
        bye_target: str | None = None
        if len(remaining) % 2 == 1:
            for cand in reversed(remaining):
                if not self.players[cand].had_bye:
                    bye_target = cand
                    break
            if bye_target is None:
                bye_target = remaining[-1]
            remaining.remove(bye_target)

        rank = {m: i for i, m in enumerate(remaining)}
        REMATCH_PENALTY = 1e9

        g: nx.Graph = nx.Graph()
        g.add_nodes_from(remaining)
        for i, a in enumerate(remaining):
            a_opp = set(self.players[a].opponents)
            for b in remaining[i + 1 :]:
                gap = self.players[a].points - self.players[b].points
                w = gap * gap
                if b in a_opp:
                    w += REMATCH_PENALTY
                g.add_edge(a, b, weight=w)

        matching = nx.min_weight_matching(g)

        if len(matching) * 2 != len(remaining):
            raise RuntimeError(
                "Min-weight matching failed to produce a perfect pairing for "
                f"{remaining!r}. Swiss tournament is complete or impossible "
                "at this size."
            )

        ordered_pairs: list[tuple[str, str]] = []
        for a, b in matching:
            if rank[a] > rank[b]:
                a, b = b, a
            ordered_pairs.append((a, b))
        ordered_pairs.sort(key=lambda ab: rank[ab[0]])

        used_rematch = [
            (a, b) for a, b in ordered_pairs if b in self.players[a].opponents
        ]
        if used_rematch:
            # Matching is still valid — emit pairings — but warn so the user
            # knows the bracket has exhausted fresh opponents.
            import warnings

            warnings.warn(
                "Swiss pairing required rematches; no fully fresh pairing "
                f"exists this round. Repeats: {used_rematch!r}",
                stacklevel=2,
            )

        pairings = [SwissPairing(model_a=a, model_b=b) for a, b in ordered_pairs]
        if bye_target is not None:
            pairings.append(SwissPairing(model_a=bye_target, model_b=None))
        return pairings


def swiss_points_table(
    matches: list[dict],
    *,
    draw_value: float = 0.5,
) -> dict[str, float]:
    """Aggregate Swiss points from raw match dicts (as stored by ArtifactLogger).

    Win = 1.0, Draw = ``draw_value``, Loss = 0.0. Byes are not represented in
    match logs (they are tournament-level events), so callers that need them
    must add them externally.
    """
    table: dict[str, float] = {}
    for m in matches:
        a, b = m["model_a_id"], m["model_b_id"]
        table.setdefault(a, 0.0)
        table.setdefault(b, 0.0)
        outcome = m.get("outcome")
        if outcome == "model_a_wins":
            table[a] += 1.0
        elif outcome == "model_b_wins":
            table[b] += 1.0
        else:
            table[a] += draw_value
            table[b] += draw_value
    return table
