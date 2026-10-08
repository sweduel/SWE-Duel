"""Round-robin tournament pairing.

Every player meets every other player exactly once. The full schedule is fixed
up-front (a circle-method pairing table), so round R is always the same set of
pairings regardless of intermediate outcomes — there is no re-pairing step.

Odd-N handling: a dummy "BYE" seat is appended so the circle method produces a
clean N+1 (even) bracket; the player drawn against BYE in a given round sits out
that round (no match is run, no points awarded). This keeps every player's
games-played count within one of every other player's, which is the round-robin
fairness property.

**Late entrants.** After the base schedule is exhausted (or mid-tournament via
:meth:`add_players`), newcomers are onboarded with Chatbot-Arena *active
sampling* (Chiang et al., ICML 2024): each intake round pairs residual-budget
newcomers against the yet-unplayed opponents that most reduce win-matrix CI
width, so a new entrant need not face every incumbent.

This mirrors :class:`swe_duel.engine.swiss.SwissTournament` in surface API
(``players``, ``round_history``, ``current_round_index``, ``standings``,
``pair_next_round``, ``commit_round``, ``record_result``,
``recommended_rounds``) so the tournament REPL can be a near-clone of the
Swiss one.
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from swe_duel.engine.swiss import SwissPairing, SwissPlayer
from swe_duel.models import MatchResult
from swe_duel.scoring.active_sampling import (
    default_intake_budget,
    pair_key,
    select_intake_pairings,
)

# Re-export SwissPairing / SwissPlayer under the round-robin module so callers
# can `from swe_duel.engine.round_robin import RoundRobinPairing` if they prefer;
# the shapes are identical to the Swiss ones and the REPL uses them
# interchangeably.
RoundRobinPairing = SwissPairing
RoundRobinPlayer = SwissPlayer

_BYE = "__bye__"


class RoundRobinTournament:
    """In-memory round-robin bookkeeping.

    The full base schedule is computed eagerly in ``__init__`` (circle method),
    then exposed one round at a time via :meth:`pair_next_round`. Each base
    round's pairings are fixed — :meth:`record_result` only updates the
    cumulative points table, never the next base round's pairings.

    After :meth:`add_players`, further rounds are *intake* rounds produced by
    active sampling (see module docstring) until every newcomer's residual
    budget is spent.
    """

    def __init__(
        self,
        model_ids: Iterable[str],
        *,
        initial_seed_by: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        ids = list(model_ids)
        # Round 1 ordering: if the caller gave an explicit seed order use it;
        # otherwise if a numeric seed was provided, shuffle deterministically
        # by it; otherwise use the input order.
        if initial_seed_by is not None:
            order = [m for m in initial_seed_by if m in ids]
            order += [m for m in ids if m not in order]
        elif seed is not None:
            rng = random.Random(seed)
            order = list(ids)
            rng.shuffle(order)
        else:
            order = list(ids)

        self.base_model_ids: list[str] = list(ids)
        self.players: dict[str, SwissPlayer] = {
            m: SwissPlayer(model_id=m) for m in ids
        }
        self._schedule: list[list[SwissPairing]] = _circle_schedule(order)
        self.round_history: list[list[SwissPairing]] = []
        # Late-entrant intake state (empty until add_players).
        self.intake_players: list[str] = []
        # Residual matches each newcomer still owes under active sampling.
        self.intake_budget: dict[str, int] = {}
        # Optional tip from the REPL: MatchResults used to re-rank intake pairs
        # after each completed intake round. Not persisted; rebuilt on resume.
        self._intake_match_results: list[MatchResult] = []
        self._intake_secondary: dict[str, float] = {}

    # ── public ───────────────────────────────────────────────

    @property
    def n_models(self) -> int:
        return len(self.players)

    @property
    def recommended_rounds(self) -> int:
        """Base circle rounds + 0/1 pending active-sampling intake batch.

        For N even base size this is ``N - 1``; for N odd (with a bye slot) it
        is ``N``. A lone intake round (if any residual newcomer budget remains)
        dumps every residual newcomer×opponent pairing in one batch — not one
        match per round.
        """
        base = len(self._schedule)
        pending_intake = 0 if self.intake_complete else 1
        already_intake = max(0, self.current_round_index - base)
        return base + already_intake + pending_intake

    @property
    def current_round_index(self) -> int:
        return len(self.round_history)

    @property
    def base_schedule_complete(self) -> bool:
        return self.current_round_index >= len(self._schedule)

    @property
    def intake_complete(self) -> bool:
        if not self.intake_players:
            return True
        return all(b <= 0 for b in self.intake_budget.values())

    def standings(self) -> list[SwissPlayer]:
        return sorted(
            self.players.values(),
            key=lambda p: (-p.points, p.model_id),
        )

    def add_players(
        self,
        model_ids: Iterable[str],
        *,
        matches_per_newcomer: int | None = None,
        remaining_budget: Mapping[str, int] | None = None,
    ) -> list[str]:
        """Register late entrants for active-sampling intake rounds.

        Returns the list of ids actually added (already-present ids are
        skipped). ``matches_per_newcomer`` defaults to
        :func:`default_intake_budget` over the pre-addition field size.
        ``remaining_budget`` (id → residual count) is used on resume to restore
        a partially-spent intake without resetting.
        """
        incumbents = len(self.players)
        default_k = (
            matches_per_newcomer
            if matches_per_newcomer is not None
            else default_intake_budget(incumbents)
        )
        added: list[str] = []
        for mid in model_ids:
            if mid in self.players:
                continue
            self.players[mid] = SwissPlayer(model_id=mid)
            self.intake_players.append(mid)
            if remaining_budget is not None and mid in remaining_budget:
                self.intake_budget[mid] = max(0, int(remaining_budget[mid]))
            else:
                # Cap at the number of other players alive after this addition.
                # (Filled after the loop so count is final.)
                self.intake_budget[mid] = default_k
            added.append(mid)
        # Cap budgets against the final field (can't face oneself).
        n_others = max(0, len(self.players) - 1)
        for mid in added:
            self.intake_budget[mid] = min(self.intake_budget[mid], n_others)
        return added

    def set_intake_context(
        self,
        matches: Sequence[MatchResult] | None = None,
        secondary_score: Mapping[str, float] | None = None,
    ) -> None:
        """Feed match history + optional tie-break scores for intake ranking.

        Called by the REPL before each intake ``pair_next_round`` so gains are
        recomputed from the latest results (the paper's sequential rule).
        """
        if matches is not None:
            self._intake_match_results = list(matches)
        if secondary_score is not None:
            self._intake_secondary = dict(secondary_score)

    def pair_next_round(
        self, *, initial_seed_by: list[str] | None = None  # noqa: ARG002
    ) -> list[SwissPairing]:
        """Return the pairings for the next round.

        Base rounds are fixed at construction time, so ``initial_seed_by`` is
        accepted for API parity with
        :class:`SwissTournament.pair_next_round` but ignored. Once the circle
        schedule is exhausted, switches to active-sampling intake. Raises
        ``RuntimeError`` when both the base schedule and intake are done.
        """
        idx = self.current_round_index
        if idx < len(self._schedule):
            return self._pair_base_round(idx)
        return self._pair_intake_round()

    def remaining_base_rounds(self) -> list[list[SwissPairing]]:
        """Uncommitted base circle rounds, in schedule order (each as pair_next_round).

        Empty when the circle table is exhausted. Callers that want intake as
        well should append :meth:`_pair_intake_round` after the base schedule
        would complete (see the round-robin REPL ``all`` command).
        """
        start = self.current_round_index
        if start >= len(self._schedule):
            return []
        return [self._pair_base_round(i) for i in range(start, len(self._schedule))]

    def commit_round(self, pairings: list[SwissPairing]) -> None:
        """Record the round's pairings as committed.

        Bye pairings (``model_b is None``) mark the sitting-out player as
        having had a bye but award no points (round-robin byes are schedule
        artifacts, not free wins). Real pairings append each side to the
        other's opponents list. Intake rounds also decrement each newcomer's
        residual budget.
        """
        self.round_history.append(pairings)
        for p in pairings:
            if p.model_b is None:
                self.players[p.model_a].had_bye = True
            else:
                self.players[p.model_a].opponents.append(p.model_b)
                self.players[p.model_b].opponents.append(p.model_a)
                for mid in (p.model_a, p.model_b):
                    if mid in self.intake_budget and self.intake_budget[mid] > 0:
                        self.intake_budget[mid] -= 1

    def record_result(self, model_a: str, model_b: str, score_a: float) -> None:
        """Apply cumulative points after a (non-bye) match. score_a∈{0,0.5,1}.

        Unlike Swiss, points here are pure bookkeeping for the live standings
        display — they never influence future base pairings (those are fixed).
        Intake pairings *do* re-rank from match history via
        :meth:`set_intake_context`, which is independent of this points table.
        """
        self.players[model_a].points += score_a
        self.players[model_b].points += 1.0 - score_a

    def already_played_pairs(self) -> set[frozenset[str]]:
        """Unordered pairs that have already been committed (any round)."""
        seen: set[frozenset[str]] = set()
        for rnd in self.round_history:
            for p in rnd:
                if p.model_b is None:
                    continue
                seen.add(pair_key(p.model_a, p.model_b))
        return seen

    # ── internals ────────────────────────────────────────────

    def _pair_base_round(self, idx: int) -> list[SwissPairing]:
        # Strip the bye slot from the returned pairings — a bye is represented
        # as model_b=None, matching the Swiss convention.
        out: list[SwissPairing] = []
        for p in self._schedule[idx]:
            if p.model_b == _BYE:
                out.append(SwissPairing(model_a=p.model_a, model_b=None))
            elif p.model_a == _BYE:
                # Bye was listed first; normalise so model_a is the real player.
                # p.model_b is guaranteed to be a real id here (not _BYE),
                # because the circle method never pairs two byes.
                out.append(SwissPairing(model_a=p.model_b or _BYE, model_b=None))
            else:
                out.append(p)
        return out

    def _pair_intake_round(self) -> list[SwissPairing]:
        if self.intake_complete:
            raise RuntimeError(
                "Round-robin schedule complete: every base pairing has been "
                "played and active-sampling intake budgets are exhausted."
            )
        pairings = select_intake_pairings(
            newcomers=list(self.intake_players),
            all_players=list(self.players.keys()),
            already_played=self.already_played_pairs(),
            remaining_budget=self.intake_budget,
            matches=self._intake_match_results,
            secondary_score=self._intake_secondary,
        )
        if not pairings:
            # Budgets remain but no legal unplayed pair exists (e.g. budget
            # exceeded the possible opponent pool) — treat as complete.
            raise RuntimeError(
                "Active-sampling intake has no legal remaining pairs "
                f"(budgets={dict(self.intake_budget)})."
            )
        return pairings


# ── internals ────────────────────────────────────────────


def _circle_schedule(order: list[str]) -> list[list[SwissPairing]]:
    """Build the full round-robin schedule via the circle method.

    Appends a ``_BYE`` sentinel when N is odd so every round pairs evenly.
    Returns ``R`` lists of pairings (``R = N-1`` for even N, ``N`` for odd N),
    each pairing carrying two real player ids or one real id + ``_BYE``.
    """
    ids = list(order)
    if len(ids) < 2:
        return []
    if len(ids) % 2 == 1:
        ids.append(_BYE)

    n = len(ids)
    n_rounds = n - 1
    # Fix the first player, rotate the rest.
    fixed = ids[0]
    rest = ids[1:]

    rounds: list[list[SwissPairing]] = []
    for _ in range(n_rounds):
        row = [fixed] + rest
        pairs: list[SwissPairing] = []
        for i in range(n // 2):
            a = row[i]
            b = row[n - 1 - i]
            # Order pair deterministically (higher in `order` first) so the
            # Swiss/round-robin REPL renders consistently.
            if _rank_of(a, order) > _rank_of(b, order):
                a, b = b, a
            pairs.append(SwissPairing(model_a=a, model_b=b))
        rounds.append(pairs)
        # Rotate rest: [r0, r1, ..., r_{k-1}] -> [r_{k-1}, r0, r1, ...]
        rest = [rest[-1]] + rest[:-1]
    return rounds


def _rank_of(player: str, order: list[str]) -> int:
    """Index of `player` in the original `order` (BYE sorts last)."""
    if player == _BYE:
        return len(order)
    try:
        return order.index(player)
    except ValueError:
        return len(order)


# Total pairings across the schedule = C(N, 2). Used by the REPL for sizing.
def total_pairings(n_models: int) -> int:
    """C(N, 2) — total distinct pairings in a full round-robin."""
    return n_models * (n_models - 1) // 2


def residual_intake_rounds(budget: Mapping[str, int]) -> int:
    """Worst-case remaining intake rounds given residual per-newcomer budgets."""
    if not budget:
        return 0
    return max(int(v) for v in budget.values()) if budget else 0


def suggested_intake_rounds(n_incumbents: int) -> int:
    """Public wrapper around default_intake_budget for CLIs."""
    return default_intake_budget(n_incumbents)
