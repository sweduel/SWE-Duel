"""Active pairwise sampling (Chatbot Arena / Chiang et al., ICML 2024).

Chooses which model pair to play next so that win-matrix confidence intervals
shrink fastest. The sampling weight for pair ``a`` is the reduction in the
standard error of ``θ̂_a`` from one more observation (eq. 9 of the paper):

    P_t(a) ∝ √(Σ̂_{t,a,a} / n_a) − √(Σ̂_{t,a,a} / (n_a + 1))

where ``n_a`` is the number of prior matchups of that pair and ``Σ̂_{t,a,a}`` is
the current variance estimate of the (Bernoulli / soft) win rate.

Used by the tournament REPLs when onboarding a newcomer into an already-run
round-robin (or Swiss) so the new entrant need not face every incumbent.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from swe_duel.engine.swiss import SwissPairing
from swe_duel.models import MatchOutcome, MatchResult


# ── primitives ────────────────────────────────────────────────


def se_reduction(n_obs: int, variance: float) -> float:
    """Reduction in SE from one extra observation of a pair (paper eq. 9).

    ``n_obs == 0`` is treated as never-sampled and returns a large finite
    priority so unplayed pairs always outrank played ones (the continuous
    formula diverges at n=0).
    """
    var = max(float(variance), 1e-12)
    if n_obs <= 0:
        # Never sampled: maximum priority. Magnitude is large but finite so
        # still sortable / summable.
        return math.sqrt(var) * 1e6
    return math.sqrt(var / n_obs) - math.sqrt(var / (n_obs + 1))


def bernoulli_variance(p_hat: float) -> float:
    """Bernoulli variance ``p(1-p)`` clamped away from the {0,1} poles."""
    p = min(max(float(p_hat), 1e-6), 1.0 - 1e-6)
    return p * (1.0 - p)


def laplace_win_rate(wins_a: float, wins_b: float, *, prior: float = 1.0) -> float:
    """P(A beats B) with Laplace/Beta(prior, prior) smoothing."""
    pr = max(float(prior), 0.0)
    return (wins_a + pr) / (wins_a + wins_b + 2.0 * pr)


@dataclass(frozen=True)
class PairStats:
    """Aggregate outcome counts for an unordered pair {a, b}, oriented a→b."""

    model_a: str
    model_b: str
    wins_a: float = 0.0  # incl. 0.5 per draw
    wins_b: float = 0.0
    n_obs: int = 0

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.model_a, self.model_b))

    def p_a_wins(self) -> float:
        return laplace_win_rate(self.wins_a, self.wins_b)

    def variance(self) -> float:
        return bernoulli_variance(self.p_a_wins())

    def gain(self) -> float:
        return se_reduction(self.n_obs, self.variance())


def pair_key(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def default_intake_budget(n_incumbents: int) -> int:
    """How many opponents a newcomer should face under active sampling.

    Swiss-like information-theoretic lower bound floor(log2 N)+1, floored at 3
    and capped at the full incumbent pool so a tiny field still finishes.
    """
    if n_incumbents <= 0:
        return 0
    if n_incumbents <= 2:
        return n_incumbents
    # ceil(log2 N) + 1 — enough to place the newcomer on a binary-search tree
    # of the existing field, plus one margin sample against a similar peer.
    k = math.ceil(math.log2(n_incumbents)) + 1
    return min(n_incumbents, max(3, k))


# ── history aggregation ───────────────────────────────────────


def aggregate_pair_stats(
    matches: Sequence[MatchResult],
) -> dict[frozenset[str], PairStats]:
    """Fold match outcomes into per-unordered-pair win counts.

    Orientation inside each ``PairStats`` is lexicographic on model id so the
    structure is stable across rebuilds (``wins_a`` is the lex-smaller id).
    """
    wins: dict[frozenset[str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    # list = [wins_lex0, wins_lex1, n_obs]
    for m in matches:
        a, b = m.model_a_id, m.model_b_id
        if a == b:
            continue
        key = pair_key(a, b)
        lex = sorted((a, b))
        sa, sb = 0.5, 0.5
        if m.outcome == MatchOutcome.MODEL_A_WINS:
            sa, sb = 1.0, 0.0
        elif m.outcome == MatchOutcome.MODEL_B_WINS:
            sa, sb = 0.0, 1.0
        # Map match-oriented scores onto lex orientation.
        if a == lex[0]:
            wins[key][0] += sa
            wins[key][1] += sb
        else:
            wins[key][0] += sb
            wins[key][1] += sa
        wins[key][2] += 1.0

    out: dict[frozenset[str], PairStats] = {}
    for key, (wa, wb, n) in wins.items():
        lex = sorted(key)
        out[key] = PairStats(
            model_a=lex[0],
            model_b=lex[1],
            wins_a=wa,
            wins_b=wb,
            n_obs=int(n),
        )
    return out


def pair_gain(
    a: str,
    b: str,
    stats: Mapping[frozenset[str], PairStats],
) -> float:
    """Active-sampling gain for the unordered pair {a, b}."""
    st = stats.get(pair_key(a, b))
    if st is None:
        # Never observed — maximum Bernoulli variance under a flat prior.
        return se_reduction(0, bernoulli_variance(0.5))
    return st.gain()


# ── ranking / matching ────────────────────────────────────────


def rank_candidate_pairs(
    candidates: Iterable[tuple[str, str]],
    stats: Mapping[frozenset[str], PairStats],
    *,
    secondary_score: Mapping[str, float] | None = None,
) -> list[tuple[str, str, float]]:
    """Score candidate pairs by eq. 9 gain (desc).

    ``secondary_score`` (e.g. TrueSkill σ of the opponent, or 0) is used only as
    a tie-break so that among equally-unobserved pairs we prefer the opponent
    whose rating is currently most uncertain — more ranking information per
    Waterloo sample.
    """
    sec = secondary_score or {}
    scored: list[tuple[str, str, float, float]] = []
    for a, b in candidates:
        if a == b:
            continue
        g = pair_gain(a, b, stats)
        # Prefer higher secondary on either side as a soft tie-break.
        tie = max(sec.get(a, 0.0), sec.get(b, 0.0))
        scored.append((a, b, g, tie))
    scored.sort(key=lambda t: (-t[2], -t[3], t[0], t[1]))
    return [(a, b, g) for a, b, g, _ in scored]


def greedy_active_matching(
    ranked: Sequence[tuple[str, str, float]],
    *,
    blocked: set[str] | None = None,
) -> list[SwissPairing]:
    """Greedy one-round matching: take highest-gain pairs without reusing a seat.

    Deterministic given ``ranked``. Pair orientation keeps the first element of
    each scored tuple as ``model_a``. Used when a classical single-seat-per-
    round constraint is desired (not the default intake batcher).
    """
    used: set[str] = set(blocked or ())
    out: list[SwissPairing] = []
    for a, b, _gain in ranked:
        if a in used or b in used:
            continue
        out.append(SwissPairing(model_a=a, model_b=b))
        used.add(a)
        used.add(b)
    return out


def select_intake_pairings(
    *,
    newcomers: Sequence[str],
    all_players: Sequence[str],
    already_played: set[frozenset[str]],
    remaining_budget: Mapping[str, int],
    matches: Sequence[MatchResult],
    secondary_score: Mapping[str, float] | None = None,
    include_newcomer_pairs: bool = True,
) -> list[SwissPairing]:
    """Pick one intake batch of pairings via active sampling.

    Emits **the full residual budget** in a single round: a newcomer with
    budget K appears in up to K pairings at once (defense sub-turns for those
    matches already run concurrently in the REPL pool). Pair choice is still
    the Chatbot-Arena SE-reduction ranking over unplayed newcomer pairs —
    top-K by gain, not a full re-round-robin.

    Newcomer–newcomer pairs cost one budget unit on each side; pass
    ``include_newcomer_pairs=False`` for **cross-only** intake (every
    pairing is newcomer × incumbent — used when entrants must anchor to
    the existing field instead of each other, e.g. the one-shot
    active-sampling tournament console). Incumbent–incumbent pairs are
    never emitted (those belong to the original schedule).
    """
    budget_left: dict[str, int] = {
        n: int(remaining_budget.get(n, 0))
        for n in newcomers
        if remaining_budget.get(n, 0) > 0
    }
    if not budget_left:
        return []

    player_set = set(all_players)
    stats = aggregate_pair_stats(matches)
    new_list = sorted(budget_left)
    others = sorted(player_set)

    candidates: list[tuple[str, str]] = []
    for i, a in enumerate(new_list):
        if include_newcomer_pairs:
            for b in new_list[i + 1 :]:
                if pair_key(a, b) not in already_played:
                    candidates.append((a, b))
        for b in others:
            if b == a or b in budget_left:
                continue
            if pair_key(a, b) not in already_played:
                candidates.append((a, b))

    ranked = rank_candidate_pairs(
        candidates, stats, secondary_score=secondary_score
    )

    # Fill each newcomer's residual seats, highest-gain first. A seat may have
    # a newcomer on both sides; incumbents may appear in many pairings —
    # SWE-Duel matches are independent workspaces so this batches fine under
    # the existing parallel defense pool.
    selected: list[SwissPairing] = []
    selected_keys: set[frozenset[str]] = set()
    for a, b, _gain in ranked:
        if not any(budget_left.get(x, 0) > 0 for x in (a, b)):
            continue
        # Both endpoints that are newcomers need residual budget.
        need = [x for x in (a, b) if x in budget_left]
        if any(budget_left[x] <= 0 for x in need):
            continue
        key = pair_key(a, b)
        if key in selected_keys or key in already_played:
            continue
        selected.append(SwissPairing(model_a=a, model_b=b))
        selected_keys.add(key)
        for x in need:
            budget_left[x] -= 1
        if all(v <= 0 for v in budget_left.values()):
            break
    return selected
