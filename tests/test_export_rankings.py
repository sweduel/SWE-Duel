"""swe-duel-rankings (export_rankings) unit tests.

Builds a miniature data/ tree (state file + matches), exports its rankings,
and checks the export carries everything `swe-duel-tournament-as` needs:
standings, BT/Elo, head-to-head, raw matches, and the enforced arena
(repos / targets_per_repo). No Docker or LLM is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_duel.cli.export_rankings import (
    _find_state_file,
    _points_from_matches,
    _targets_per_repo,
    export_rankings,
)

_TID = "tid-1111"


def _turn(
    idx: int, red: str, blue: str, repo: str, challenge_id: str = "c-1"
) -> dict:
    return {
        "turn_id": f"t-{idx}",
        "turn_index": idx,
        "red_model_id": red,
        "blue_model_id": blue,
        "red_harness_id": "mini-swe-agent",
        "blue_harness_id": "mini-swe-agent",
        "red_reasoning_effort": "high",
        "red_provider": "",
        "blue_reasoning_effort": "high",
        "blue_provider": "",
        "challenge_id": challenge_id,
        "defense_id": f"d-{idx}",
        "repo_name": repo,
        "score": {
            "s_regression": 1.0, "s_feature": 1.0, "s_bugfix": 1.0,
            "blue_composite": 1.0, "red_composite": 0.0, "test_details": {},
        },
    }


def _match(
    mid: str,
    a: str,
    b: str,
    outcome: str,
    repos: list[str],
    timestamp: str = "2026-10-06T00:00:00+00:00",
    targets_per_repo: int = 1,
) -> dict:
    turns = []
    idx = 0
    for repo in repos:
        for red, blue in ((a, b), (b, a)):
            for _ in range(targets_per_repo):
                turns.append(_turn(idx, red, blue, repo))
                idx += 1
    return {
        "match_id": mid,
        "model_a_id": a,
        "model_b_id": b,
        "repo_name": ",".join(repos),
        "model_a_total": 1.0,
        "model_b_total": 0.0,
        "outcome": outcome,
        "duration_seconds": 1.0,
        "total_cost_usd": 0.0,
        "timestamp": timestamp,
        "turns": turns,
    }


_A = "alpha/model-a#mini-swe-agent#high#"
_B = "alpha/model-b#mini-swe-agent#high#"
_C = "alpha/model-c#mini-swe-agent"


@pytest.fixture
def mini_data(tmp_path: Path) -> Path:
    """3 participants, 2 matches across 2 repos, one unplayed participant."""
    data = tmp_path / "data"
    (data / "matches").mkdir(parents=True)
    (data / "tournaments").mkdir()
    (data / "matches" / "m-1.json").write_text(
        json.dumps(_match("m-1", _A, _B, "model_a_wins", ["flask", "jwt"]))
    )
    (data / "matches" / "m-2.json").write_text(
        json.dumps(_match("m-2", _A, _C, "draw", ["flask", "jwt"]))
    )
    (data / "tournaments" / f"round_robin_state_{_TID}.json").write_text(
        json.dumps(
            {
                "tournament_id": _TID,
                "format": "round_robin",
                "timestamp": "2026-10-06T00:00:00+00:00",
                "repo_names": ["flask", "jwt"],
                "players": [
                    {"model_id": _A, "points": 2.0, "opponents": []},
                    {"model_id": _B, "points": 0.0, "opponents": []},
                    {"model_id": _C, "points": 1.0, "opponents": []},
                ],
                "rounds": [
                    [
                        {"model_a": _A, "model_b": _B},
                        {"model_a": _A, "model_b": _C},
                    ]
                ],
            }
        )
    )
    return data


def test_export_rankings_full_payload(mini_data: Path, tmp_path: Path) -> None:
    out = export_rankings(_TID, mini_data, tmp_path / "rankings")
    assert out == tmp_path / "rankings" / f"rankings_{_TID}.json"
    payload = json.loads(out.read_text())

    assert payload["tournament_id"] == _TID
    assert payload["format"] == "round_robin"
    assert payload["repos"] == ["flask", "jwt"]
    assert payload["targets_per_repo"] == 1

    # Standings: state points (authoritative), BT/Elo computed from the
    # claimed matches, identity columns split from the pinned cids.
    rankings = {r["model"]: r for r in payload["rankings"]}
    assert [r["model"] for r in payload["rankings"]] == [
        "alpha/model-a", "alpha/model-c", "alpha/model-b",
    ]
    a = rankings["alpha/model-a"]
    assert a["rank"] == 1 and a["points"] == 2.0
    assert a["matches_played"] == 2
    assert a["harness"] == "mini-swe-agent"
    assert a["reasoning_effort"] == "high"
    assert a["provider"] == "(OpenRouter auto-route)"
    assert a["bradley_terry"] > rankings["alpha/model-b"]["bradley_terry"]
    assert a["elo_rank"] == 1
    # Every entry carries the seeded bootstrap uncertainty the leaderboard
    # renders: 95% Elo CI, match-level BT CI, contested-turn BT, and the
    # full rank distribution (median + interval + histogram + draw counts).
    assert len(a["elo_ci"]) == 2 and a["elo_ci"][0] <= a["elo"] <= a["elo_ci"][1]
    assert len(a["bradley_terry_ci"]) == 2
    assert a["bradley_terry_ci"][0] <= a["bradley_terry"] <= a["bradley_terry_ci"][1]
    assert len(a["contested_bt_ci"]) == 2
    assert a["contested_bt_ci"][0] <= a["contested_bt"] <= a["contested_bt_ci"][1]
    assert a["bootstrap_draws"] > 0 and a["bootstrap_ranked"] <= a["bootstrap_draws"]
    lo_rank, hi_rank = a["bootstrap_rank_interval"]
    assert lo_rank <= a["bootstrap_median_rank"] <= hi_rank
    assert abs(sum(a["bootstrap_rank_histogram"]) - 1.0) < 1e-6
    assert len(a["bootstrap_rank_histogram"]) == len(rankings)
    # The dominant player's bootstrap median rank is #1.
    assert a["bootstrap_median_rank"] == 1
    # Determinism: re-exporting reproduces the identical bootstrap numbers.
    old_hist = a["bootstrap_rank_histogram"]
    export_rankings(_TID, mini_data, tmp_path / "rankings")
    again = json.loads((tmp_path / "rankings" / f"rankings_{_TID}.json").read_text())
    a2 = {r["model"]: r for r in again["rankings"]}["alpha/model-a"]
    assert a2["bootstrap_rank_histogram"] == old_hist
    assert a2["elo_ci"] == a["elo_ci"]
    # The state points still order the field even where ratings tie.
    c = rankings["alpha/model-c"]
    assert c["matches_played"] == 1
    assert c["points"] == 1.0

    # Head-to-head records: one entry per played pair, W/L/D attributed to
    # the sorted-first endpoint (model-a sorts before model-b).
    h2h = {(r["a"], r["b"]): r for r in payload["head_to_head"]}
    ab = h2h[(_A, _B)]
    assert ab["a_wins"] == 1 and ab["b_wins"] == 0 and ab["draws"] == 0
    assert ab["match_ids"] == ["m-1"]
    ac = h2h[(_A, _C)]
    assert ac["draws"] == 1 and ac["match_ids"] == ["m-2"]

    # Raw match list: enough to reconstruct the history offline.
    assert [m["match_id"] for m in payload["matches"]] == ["m-1", "m-2"]
    assert payload["matches"][0]["model_a"] == _A


def test_export_rankings_targets_per_repo_reconstructed(
    mini_data: Path, tmp_path: Path
) -> None:
    # Re-run the tournament with 2 slots per repo: the export must report
    # targets_per_repo=2, reconstructed from the turn structure.
    (mini_data / "matches" / "m-1.json").write_text(
        json.dumps(_match("m-1", _A, _B, "model_a_wins", ["flask"], targets_per_repo=2))
    )
    (mini_data / "matches" / "m-2.json").unlink()
    out = export_rankings(_TID, mini_data, tmp_path / "rankings")
    payload = json.loads(out.read_text())
    assert payload["targets_per_repo"] == 2


def test_export_rankings_rejects_unknown_tournament(mini_data: Path) -> None:
    with pytest.raises(SystemExit, match="no tournament state"):
        export_rankings("nope-tid", mini_data, mini_data / "rankings")


def test_export_rankings_rejects_matchless_tournament(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "matches").mkdir(parents=True)
    (data / "tournaments").mkdir()
    (data / "tournaments" / f"round_robin_state_{_TID}.json").write_text(
        json.dumps({"tournament_id": _TID, "players": [], "rounds": []})
    )
    with pytest.raises(SystemExit, match="claimed no matches"):
        export_rankings(_TID, data, tmp_path / "rankings")


def test_find_state_file_lists_available(mini_data: Path) -> None:
    assert _find_state_file(mini_data / "tournaments", _TID).name == (
        f"round_robin_state_{_TID}.json"
    )
    with pytest.raises(SystemExit, match="tid-1111"):
        _find_state_file(mini_data / "tournaments", "other-tid")


def test_points_from_matches_fallback() -> None:
    pts = _points_from_matches(
        [
            _match("m-1", "a#h", "b#h", "model_a_wins", ["flask"]),
            _match("m-2", "a#h", "b#h", "draw", ["flask"]),
        ]
    )
    assert pts == {"a#h": 1.5, "b#h": 0.5}


def test_targets_per_repo_needs_turns() -> None:
    with pytest.raises(SystemExit, match="cannot determine targets_per_repo"):
        _targets_per_repo([{"match_id": "m", "turns": []}])


# ── extra_matches: imported active-sampling contributions ──────

_NEWCOMER = "alpha/model-n#mini-swe-agent"
_AS_MATCH = _match(
    "m-as-1", _NEWCOMER, _A, "model_a_wins", ["flask", "jwt"],
    timestamp="2026-10-07T00:00:00+00:00",
)


def test_extra_matches_add_newcomer_and_points(mini_data: Path, tmp_path: Path) -> None:
    # The state points stay authoritative for the incumbents; the folded-in
    # contribution match adds its win/draw points on top and seats the
    # newcomer in the field with full standings.
    out = export_rankings(
        _TID, mini_data, tmp_path / "rankings", extra_matches=[_AS_MATCH]
    )
    payload = json.loads(out.read_text())
    by_model = {r["model"]: r for r in payload["rankings"]}
    assert "alpha/model-n" in by_model
    n = by_model["alpha/model-n"]
    assert n["points"] == 1.0 and n["matches_played"] == 1
    assert n["harness"] == "mini-swe-agent"
    # Incumbent points = state points only (no double count of claimed
    # matches), plus the loss from the imported match.
    assert by_model["alpha/model-a"]["points"] == 2.0
    assert by_model["alpha/model-a"]["matches_played"] == 3
    # Head-to-head + raw match list cover the merged history (keys are the
    # sorted pair, so the incumbent sorts first).
    assert "m-as-1" in [m["match_id"] for m in payload["matches"]]
    h2h = {(r["a"], r["b"]): r for r in payload["head_to_head"]}
    assert h2h[(_A, _NEWCOMER)]["b_wins"] == 1
    # targets_per_repo is unchanged (the AS match ran the same arena).
    assert payload["targets_per_repo"] == 1


def test_extra_matches_dedupe_by_match_id(mini_data: Path, tmp_path: Path) -> None:
    out = export_rankings(
        _TID, mini_data, tmp_path / "rankings", extra_matches=[_AS_MATCH, _AS_MATCH]
    )
    payload = json.loads(out.read_text())
    ids = [m["match_id"] for m in payload["matches"]]
    assert ids.count("m-as-1") == 1
    # Same match twice must not double the newcomer's points.
    by_model = {r["model"]: r for r in payload["rankings"]}
    assert by_model["alpha/model-n"]["points"] == 1.0


def test_extra_matches_claims_win_over_extras(mini_data: Path, tmp_path: Path) -> None:
    # An extra whose match_id collides with an already-claimed local match is
    # dropped (local files win) — the export stays at one copy of it.
    claimed = _match("m-1", _A, _B, "draw", ["flask", "jwt"])
    out = export_rankings(
        _TID, mini_data, tmp_path / "rankings", extra_matches=[claimed]
    )
    payload = json.loads(out.read_text())
    ids = [m["match_id"] for m in payload["matches"]]
    assert ids.count("m-1") == 1


def test_as_extra_matches_auto_fold(mini_data: Path, tmp_path: Path) -> None:
    # AS states on disk (an imported contribution, or a local
    # swe-duel-tournament-as run) fold in automatically on every re-export,
    # so the late entrants never silently drop out of a fresh export.
    from swe_duel.cli.export_rankings import (
        _as_entered_tournament,
        _load_as_extra_matches,
    )

    matches_dir = mini_data / "matches"
    (matches_dir / "m-as-1.json").write_text(json.dumps(_AS_MATCH))
    tournaments_dir = mini_data / "tournaments"
    # Explicit entered_tournament form.
    (tournaments_dir / "active_sampling_state_as-1.json").write_text(
        json.dumps(
            {
                "tournament_id": "as-1",
                "format": "active_sampling",
                "entered_tournament": _TID,
                "repo_names": ["flask", "jwt"],
                "newcomers": [_NEWCOMER],
                "pairings": [{"model_a": _NEWCOMER, "model_b": _A}],
                "results": [{"match_id": "m-as-1", "outcome": "model_a_wins"}],
                "status": "complete",
                "timestamp": "2026-10-07T00:00:00+00:00",
            }
        )
    )
    # Legacy form: the tournament is only recoverable from rankings_source.
    (tournaments_dir / "active_sampling_state_as-2.json").write_text(
        json.dumps(
            {
                "tournament_id": "as-2",
                "format": "active_sampling",
                "rankings_source": "rankings/rankings_tid-1111.json",
                "repo_names": ["flask"],
                "newcomers": [_NEWCOMER],
                "pairings": [{"model_a": _NEWCOMER, "model_b": _B}],
                "results": [],
                "status": "sampled",
                "timestamp": "2026-10-07T00:00:00+00:00",
            }
        )
    )
    # A state entering a DIFFERENT tournament must not leak in.
    (tournaments_dir / "active_sampling_state_as-3.json").write_text(
        json.dumps(
            {
                "tournament_id": "as-3",
                "format": "active_sampling",
                "entered_tournament": "other-tid",
                "pairings": [{"model_a": _NEWCOMER, "model_b": _C}],
                "results": [],
                "status": "sampled",
                "timestamp": "2026-10-07T00:00:00+00:00",
            }
        )
    )
    assert _as_entered_tournament(
        {"entered_tournament": _TID}
    ) == _TID
    assert _as_entered_tournament(
        {"rankings_source": "rankings/rankings_tid-1111.json"}
    ) == _TID
    assert _as_entered_tournament({}) == ""

    extras = _load_as_extra_matches(tournaments_dir, matches_dir, _TID)
    assert [m["match_id"] for m in extras] == ["m-as-1"]
    # as-2 scheduled a pairing but no match file exists → contributes nothing
    # beyond what as-1 already folded; as-3 is out of scope entirely.

    # Re-export now absorbs the AS match without an explicit parameter…
    out = export_rankings(_TID, mini_data, tmp_path / "rankings")
    payload = json.loads(out.read_text())
    assert "m-as-1" in [m["match_id"] for m in payload["matches"]]
    assert "alpha/model-n" in {r["model"] for r in payload["rankings"]}
    # …and is idempotent (a second export does not double the points).
    out2 = export_rankings(_TID, mini_data, tmp_path / "rankings")
    payload2 = json.loads(out2.read_text())
    assert payload2["rankings"] == payload["rankings"]