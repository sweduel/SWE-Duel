"""Rating systems: ELO, TrueSkill, Bradley-Terry; unified snapshot output."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import trueskill

from swe_duel.models import MatchOutcome, MatchResult, RatingSnapshot


# ── ELO ─────────────────────────────────────────────────────


class EloRating:
    def __init__(self, k: int = 32, initial: float = 1500.0) -> None:
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def _ensure(self, model: str) -> None:
        if model not in self.ratings:
            self.ratings[model] = self.initial

    @staticmethod
    def expected_score(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(
        self, model_a: str, model_b: str, outcome: MatchOutcome
    ) -> None:
        self._ensure(model_a)
        self._ensure(model_b)
        ra = self.ratings[model_a]
        rb = self.ratings[model_b]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        if outcome == MatchOutcome.MODEL_A_WINS:
            sa, sb = 1.0, 0.0
        elif outcome == MatchOutcome.MODEL_B_WINS:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        self.ratings[model_a] = ra + self.k * (sa - ea)
        self.ratings[model_b] = rb + self.k * (sb - eb)

    def update_score(
        self, model_a: str, model_b: str, score_a: float
    ) -> None:
        """ELO update with continuous score ∈ [0,1]."""
        self._ensure(model_a)
        self._ensure(model_b)
        ra = self.ratings[model_a]
        rb = self.ratings[model_b]
        ea = self.expected_score(ra, rb)
        sa = max(0.0, min(1.0, score_a))
        sb = 1.0 - sa
        eb = 1.0 - ea
        self.ratings[model_a] = ra + self.k * (sa - ea)
        self.ratings[model_b] = rb + self.k * (sb - eb)

    def get_ratings(self) -> dict[str, float]:
        return dict(self.ratings)


# ── Bradley-Terry ───────────────────────────────────────────


class BradleyTerryRating:
    """MLE fit via Minorization-Maximization."""

    def __init__(self, max_iter: int = 1000, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.strengths: dict[str, float] = {}

    def fit(self, matches: list[MatchResult]) -> dict[str, float]:
        wins: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        models: set[str] = set()
        for m in matches:
            a, b = m.model_a_id, m.model_b_id
            models.update([a, b])
            if m.outcome == MatchOutcome.MODEL_A_WINS:
                wins[a][b] += 1.0
            elif m.outcome == MatchOutcome.MODEL_B_WINS:
                wins[b][a] += 1.0
            else:
                wins[a][b] += 0.5
                wins[b][a] += 0.5

        models_list = sorted(models)
        if not models_list:
            return {}
        p = {m: 1.0 for m in models_list}

        for _ in range(self.max_iter):
            new_p: dict[str, float] = {}
            max_delta = 0.0
            for i in models_list:
                num = sum(wins[i][j] for j in models_list if j != i)
                den = 0.0
                for j in models_list:
                    if j == i:
                        continue
                    games = wins[i][j] + wins[j][i]
                    if games > 0:
                        den += games / (p[i] + p[j])
                if den == 0 or num == 0:
                    new_p[i] = p[i]
                else:
                    new_p[i] = num / den
                max_delta = max(max_delta, abs(new_p[i] - p[i]))
            # normalise to geometric mean = 1
            prod = 1.0
            for v in new_p.values():
                prod *= max(v, 1e-12)
            norm = prod ** (1.0 / len(new_p))
            if norm > 0:
                new_p = {k: v / norm for k, v in new_p.items()}
            p = new_p
            if max_delta < self.tol:
                break

        import math

        self.strengths = {m: math.log(max(p[m], 1e-12)) for m in models_list}
        return dict(self.strengths)


# ── TrueSkill ───────────────────────────────────────────────


class TrueSkillRating:
    def __init__(self, mu: float = 25.0, sigma: float = 8.333) -> None:
        self.env = trueskill.TrueSkill(mu=mu, sigma=sigma, draw_probability=0.1)
        self.ratings: dict[str, trueskill.Rating] = {}
        self._mu = mu
        self._sigma = sigma

    def _ensure(self, model: str) -> None:
        if model not in self.ratings:
            self.ratings[model] = self.env.create_rating()

    def update(
        self, model_a: str, model_b: str, outcome: MatchOutcome
    ) -> None:
        self._ensure(model_a)
        self._ensure(model_b)
        ra = self.ratings[model_a]
        rb = self.ratings[model_b]
        if outcome == MatchOutcome.DRAW:
            new_a, new_b = self.env.rate_1vs1(ra, rb, drawn=True)
        elif outcome == MatchOutcome.MODEL_A_WINS:
            new_a, new_b = self.env.rate_1vs1(ra, rb)
        else:
            new_b, new_a = self.env.rate_1vs1(rb, ra)
        self.ratings[model_a] = new_a
        self.ratings[model_b] = new_b

    def get_ratings(self) -> dict[str, tuple[float, float]]:
        return {m: (r.mu, r.sigma) for m, r in self.ratings.items()}


# ── Unified ─────────────────────────────────────────────────


def _decompose_per_role(match: MatchResult) -> list[tuple[str, str, float]]:
    """Return per-round (red_model, blue_model, red_score∈{0,1}) triples."""
    out: list[tuple[str, str, float]] = []
    for r in match.turns:
        sc = r.defense_result.score
        red_score = sc.red_composite
        out.append((r.red_model_id, r.blue_model_id, red_score))
    return out


def compute_all_ratings(
    matches: list[MatchResult],
    *,
    k: int = 32,
    initial: float = 1500.0,
    trueskill_mu: float = 25.0,
    trueskill_sigma: float = 8.333,
    model_ids: list[str] | None = None,
) -> dict[str, RatingSnapshot]:
    """Run ELO, TrueSkill, Bradley-Terry + per-role ELO. Return snapshots keyed by model_id.

    Optional ``model_ids`` seeds competitors that never appear in ``matches`` so
    tournament rosters stay complete.
    """
    elo = EloRating(k=k, initial=initial)
    ts = TrueSkillRating(mu=trueskill_mu, sigma=trueskill_sigma)
    red_elo = EloRating(k=k, initial=initial)
    blue_elo = EloRating(k=k, initial=initial)

    matches_played: dict[str, int] = defaultdict(int)
    for m in matches:
        elo.update(m.model_a_id, m.model_b_id, m.outcome)
        ts.update(m.model_a_id, m.model_b_id, m.outcome)
        matches_played[m.model_a_id] += 1
        matches_played[m.model_b_id] += 1

        for red_id, blue_id, red_score in _decompose_per_role(m):
            # Role-specific ELO: only the role-appropriate player's rating
            # updates. Opponent's rating in the *opposite* role is used as
            # reference.
            red_elo._ensure(red_id)
            red_elo._ensure(blue_id)
            blue_elo._ensure(red_id)
            blue_elo._ensure(blue_id)

            ref_blue = blue_elo.ratings[blue_id]
            ra = red_elo.ratings[red_id]
            expected_red = 1.0 / (1.0 + 10 ** ((ref_blue - ra) / 400.0))
            red_elo.ratings[red_id] = ra + red_elo.k * (red_score - expected_red)

            ref_red = red_elo.ratings[red_id]  # newly updated
            bb = blue_elo.ratings[blue_id]
            blue_score = 1.0 - red_score
            expected_blue = 1.0 / (1.0 + 10 ** ((ref_red - bb) / 400.0))
            blue_elo.ratings[blue_id] = bb + blue_elo.k * (blue_score - expected_blue)

    bt_strengths = BradleyTerryRating().fit(matches)

    all_models = (
        set(elo.ratings)
        | set(ts.ratings)
        | set(red_elo.ratings)
        | set(blue_elo.ratings)
        | set(model_ids or [])
    )
    for mid in model_ids or []:
        elo._ensure(mid)
        ts._ensure(mid)
        red_elo._ensure(mid)
        blue_elo._ensure(mid)
    ts_ratings = ts.get_ratings()
    now = datetime.now(timezone.utc)

    snapshots: dict[str, RatingSnapshot] = {}
    for model in sorted(all_models):
        mu, sigma = ts_ratings.get(model, (trueskill_mu, trueskill_sigma))
        snapshots[model] = RatingSnapshot(
            model_id=model,
            elo=elo.ratings.get(model, initial),
            trueskill_mu=mu,
            trueskill_sigma=sigma,
            red_elo=red_elo.ratings.get(model, initial),
            blue_elo=blue_elo.ratings.get(model, initial),
            matches_played=matches_played.get(model, 0),
            timestamp=now,
            bradley_terry=bt_strengths.get(model, 0.0),
        )
    return snapshots
