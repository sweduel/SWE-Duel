"""Active sampling + late-entrant intake tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from swe_duel.engine.round_robin import RoundRobinTournament
from swe_duel.engine.swiss import SwissPairing, SwissTournament
from swe_duel.models import MatchOutcome, MatchResult
from swe_duel.scoring.active_sampling import (
    aggregate_pair_stats,
    default_intake_budget,
    pair_gain,
    pair_key,
    se_reduction,
    select_intake_pairings,
)


def _match(a: str, b: str, outcome: MatchOutcome) -> MatchResult:
    return MatchResult(
        match_id=str(uuid.uuid4()),
        model_a_id=a,
        model_b_id=b,
        repo_name="r",
        turns=[],
        model_a_total=1.0 if outcome == MatchOutcome.MODEL_A_WINS else 0.0,
        model_b_total=1.0 if outcome == MatchOutcome.MODEL_B_WINS else 0.0,
        outcome=outcome,
        duration_seconds=0.0,
        total_cost_usd=0.0,
        timestamp=datetime.now(timezone.utc),
    )


def test_se_reduction_prefers_unplayed() -> None:
    unplayed = se_reduction(0, 0.25)
    once = se_reduction(1, 0.25)
    twice = se_reduction(2, 0.25)
    assert unplayed > once > twice > 0


def test_default_intake_budget_bounds() -> None:
    assert default_intake_budget(0) == 0
    assert default_intake_budget(1) == 1
    assert default_intake_budget(2) == 2
    # N=8 → ceil(log2 8)+1 = 4
    assert default_intake_budget(8) == 4
    # Never exceeds the incumbent pool.
    assert default_intake_budget(3) <= 3


def test_aggregate_pair_stats_counts_draws() -> None:
    matches = [
        _match("a", "b", MatchOutcome.MODEL_A_WINS),
        _match("b", "a", MatchOutcome.DRAW),
    ]
    stats = aggregate_pair_stats(matches)
    st = stats[pair_key("a", "b")]
    assert st.n_obs == 2
    # lex orientation: a < b so wins_a accumulates A's score from both rows.
    assert st.wins_a == 1.5
    assert st.wins_b == 0.5


def test_select_intake_pairings_single_newcomer() -> None:
    # Incumbents a..d fully played; newcomer e faces none yet. Budget 2 → two
    # pairings in one batch (e appears in both).
    incumbents = ["a", "b", "c", "d"]
    matches = [
        _match(x, y, MatchOutcome.MODEL_A_WINS)
        for i, x in enumerate(incumbents)
        for y in incumbents[i + 1 :]
    ]
    already = {pair_key(m.model_a_id, m.model_b_id) for m in matches}
    pairings = select_intake_pairings(
        newcomers=["e"],
        all_players=incumbents + ["e"],
        already_played=already,
        remaining_budget={"e": 2},
        matches=matches,
    )
    assert len(pairings) == 2
    opps: set[str] = set()
    for pair in pairings:
        assert pair.model_b is not None
        assert "e" in (pair.model_a, pair.model_b)
        opps.add(pair.model_b if pair.model_a == "e" else pair.model_a)
    assert len(opps) == 2


def test_select_intake_pairings_can_pair_newcomers() -> None:
    # Default intake allows newcomer–newcomer pairs (one budget unit on
    # each side): with every incumbent already played, the two newcomers
    # face each other.
    already = {
        pair_key("e", "a"),
        pair_key("e", "b"),
        pair_key("f", "a"),
        pair_key("f", "b"),
    }
    pairings = select_intake_pairings(
        newcomers=["e", "f"],
        all_players=["a", "b", "e", "f"],
        already_played=already,
        remaining_budget={"e": 1, "f": 1},
        matches=[],
    )
    assert len(pairings) == 1
    assert pairings[0].model_b is not None
    assert {pairings[0].model_a, pairings[0].model_b} == {"e", "f"}
    # Cross-only intake (the one-shot active-sampling console) refuses the
    # entrant-vs-entrant fallback: no legal matchups.
    assert (
        select_intake_pairings(
            newcomers=["e", "f"],
            all_players=["a", "b", "e", "f"],
            already_played=already,
            remaining_budget={"e": 1, "f": 1},
            matches=[],
            include_newcomer_pairs=False,
        )
        == []
    )


def test_select_intake_pairings_cross_only_never_pairs_newcomers() -> None:
    # include_newcomer_pairs=False = cross-only intake (the one-shot
    # active-sampling console): every matchup is newcomer × incumbent —
    # entrants never play each other (their budget anchors them to the
    # existing field instead), and incumbents never play each other.
    incumbents = ["a", "b"]
    newcomers = ["e", "f"]
    pairings = select_intake_pairings(
        newcomers=newcomers,
        all_players=incumbents + newcomers,
        already_played=set(),
        remaining_budget={"e": 2, "f": 2},
        matches=[],
        include_newcomer_pairs=False,
    )
    assert len(pairings) == 4  # 2 entrants × 2 incumbents
    for p in pairings:
        assert p.model_b is not None
        pair = {p.model_a, p.model_b}
        # Exactly one entrant and one incumbent per matchup.
        assert len(pair & set(newcomers)) == 1
        assert len(pair & set(incumbents)) == 1


def test_remaining_base_rounds_drains_schedule() -> None:
    rr = RoundRobinTournament(["a", "b", "c", "d"], seed=0)
    # Round 0 committed → remaining includes every later circle round.
    first = rr.pair_next_round()
    rr.commit_round(first)
    rem = rr.remaining_base_rounds()
    assert len(rem) == len(rr._schedule) - 1
    # Flat remaining playable pairs plus committed ones = C(N,2).
    played = {
        frozenset((p.model_a, p.model_b))
        for p in first
        if p.model_b is not None
    }
    remaining_pairs = {
        frozenset((p.model_a, p.model_b))
        for rnd in rem
        for p in rnd
        if p.model_b is not None
    }
    assert played.isdisjoint(remaining_pairs)
    assert len(played) + len(remaining_pairs) == 6  # C(4,2)
    # Drain via remaining_base_rounds + commit leaves schedule complete.
    for pairings in rem:
        rr.commit_round(pairings)
    assert rr.remaining_base_rounds() == []
    assert rr.base_schedule_complete


def test_round_robin_add_players_intake_batch() -> None:
    rr = RoundRobinTournament(["m1", "m2", "m3", "m4"], seed=0)
    # Drain the base schedule.
    while rr.current_round_index < len(rr._schedule):
        pairings = rr.pair_next_round()
        rr.commit_round(pairings)
        for p in pairings:
            if p.model_b is not None:
                rr.record_result(p.model_a, p.model_b, 1.0)

    assert rr.base_schedule_complete
    added = rr.add_players(["new#h"], matches_per_newcomer=2)
    assert added == ["new#h"]
    assert rr.intake_budget["new#h"] == 2
    # One intake round carries the full residual budget.
    assert rr.recommended_rounds == rr.current_round_index + 1

    pairings = rr.pair_next_round()
    assert len(pairings) == 2
    seen_opponents: set[str] = set()
    for p in pairings:
        assert p.model_b is not None
        assert "new#h" in (p.model_a, p.model_b)
        opp = p.model_b if p.model_a == "new#h" else p.model_a
        seen_opponents.add(opp)
    rr.commit_round(pairings)
    for p in pairings:
        assert p.model_b is not None
        rr.record_result(p.model_a, p.model_b, 0.5)

    assert rr.intake_budget["new#h"] == 0
    assert len(seen_opponents) == 2
    try:
        rr.pair_next_round()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_round_robin_intake_skips_already_played() -> None:
    rr = RoundRobinTournament(["a", "b", "c", "d"], seed=1)
    while rr.current_round_index < len(rr._schedule):
        pairings = rr.pair_next_round()
        rr.commit_round(pairings)
        for p in pairings:
            if p.model_b is not None:
                rr.record_result(p.model_a, p.model_b, 1.0)

    rr.add_players(["z"], matches_per_newcomer=3)
    # Pretend z already faced "a".
    rr.players["z"].opponents.append("a")
    rr.players["a"].opponents.append("z")
    rr.round_history.append([SwissPairing(model_a="z", model_b="a")])
    # Budget still 3 but only 3 unplayed incumbents remain (b,c,d).
    pairings = rr.pair_next_round()
    assert len(pairings) == 3
    opps = {
        (p.model_b if p.model_a == "z" else p.model_a)
        for p in pairings
        if p.model_b is not None
    }
    assert "a" not in opps
    assert opps == {"b", "c", "d"}


def test_swiss_add_players() -> None:
    s = SwissTournament(["a", "b", "c", "d"])
    s.commit_round(s.pair_next_round(initial_seed_by=["a", "b", "c", "d"]))
    added = s.add_players(["e", "f"])
    assert added == ["e", "f"]
    assert set(s.intake_players) == {"e", "f"}
    assert s.players["e"].points == 0.0
    # Next round can pair with odd/even field (6 players).
    pairings = s.pair_next_round()
    seated = {p.model_a for p in pairings} | {
        p.model_b for p in pairings if p.model_b
    }
    assert "e" in seated or "f" in seated


def test_pair_gain_unplayed_beats_played() -> None:
    matches = [_match("a", "b", MatchOutcome.MODEL_A_WINS)]
    stats = aggregate_pair_stats(matches)
    g_unplayed = pair_gain("a", "c", stats)
    g_played = pair_gain("a", "b", stats)
    assert g_unplayed > g_played
