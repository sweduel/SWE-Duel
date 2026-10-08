"""swe-duel-tournament-update unit tests.

Offline end-to-end coverage for the external-contribution importer: safe zip
extraction, tournament resolution, the additive merge (matches / defenses /
challenges scoped to the AS run's field identities + challenge-bank index
merge), the rankings re-export (via export_rankings' AS auto-fold), the
root index.html leaderboard refresh, the contribution archive/history, and
the duplicate-import guard. No Docker or LLM is touched.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pytest

from swe_duel.cli.tournament_update import (
    _Identity,
    _as_entered_tournament,
    _as_field,
    _as_newcomers,
    _bar_width,
    _bootstrap_field_diagnostics,
    _locate_data_prefix,
    _merge_index,
    _merge_submission,
    _newcomer_row_html,
    _newcomer_stats,
    _page_axis,
    _resolve_tournament,
    _StrengthDiag,
    _update_leaderboard,
    _zip_sha256,
    main as update_main,
)

# ── shared synthetic field ──────────────────────────────────────

_TID = "tid-2222"
_A = "openai/gpt-5.5#codex#medium#"
_B = "anthropic/claude-opus-4.8#mini-swe-agent#high#"
_N = "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai"
_REPOS = ["flask", "jwt"]


def _turn(idx: int, red: str, blue: str, repo: str, cid: str, did: str,
          red_comp: float = 0.0) -> dict[str, Any]:
    return {
        "turn_id": f"t-{idx}", "turn_index": idx,
        "red_model_id": red, "blue_model_id": blue,
        "red_harness_id": red.split("#")[1],
        "blue_harness_id": blue.split("#")[1],
        "red_reasoning_effort": "", "red_provider": "",
        "blue_reasoning_effort": "", "blue_provider": "",
        "challenge_id": cid, "defense_id": did, "repo_name": repo,
        "score": {"s_regression": 1.0, "s_feature": 1.0, "s_bugfix": 1.0,
                  "blue_composite": 1.0 - red_comp, "red_composite": red_comp,
                  "test_details": {}},
    }


def _match(mid: str, a: str, b: str, outcome: str, repos: list[str],
           ts: str = "2026-10-05T00:00:00+00:00") -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    idx = 0
    for repo in repos:
        for red, blue in ((a, b), (b, a)):
            turns.append(_turn(idx, red, blue, repo, f"c-{idx}", f"d-{mid}-{idx}"))
            idx += 1
    return {
        "match_id": mid, "model_a_id": a, "model_b_id": b,
        "repo_name": ",".join(repos), "model_a_total": 2.0,
        "model_b_total": 0.0, "outcome": outcome, "duration_seconds": 1.0,
        "total_cost_usd": 0.5, "timestamp": ts, "turns": turns,
    }


def _as_state(tid: str = _TID, *, newcomer: str = _N, results: list | None = None,
              path_name: str = "as-run-1") -> dict[str, Any]:
    return {
        "tournament_id": path_name, "format": "active_sampling",
        "entered_tournament": tid, "repo_names": _REPOS,
        "targets_per_repo": 1, "field": [_A, _B, newcomer],
        "targets": [newcomer], "newcomers": [newcomer], "budget": 1,
        "pairings": [{"model_a": newcomer, "model_b": _A}],
        "results": results if results is not None else [
            {"match_id": "m-as-1", "model_a": newcomer, "model_b": _A,
             "outcome": "model_a_wins"}
        ],
        "status": "complete", "timestamp": "2026-10-05T00:00:00+00:00",
    }


def _challenge(cid: str, red: str, repo: str, cost: float = 1.5,
               out_tok: int = 1000) -> dict[str, Any]:
    parts = red.split("#")
    return {
        "challenge_id": cid, "red_model_id": parts[0],
        "red_harness_id": parts[1],
        "red_reasoning_effort": parts[2] if len(parts) > 2 else "",
        "red_provider": parts[3] if len(parts) > 3 else "",
        "repo_name": repo, "repo_commit_sha": "abc",
        "target_files": ["src/a.py"], "slot": 1,
        "generation_cost_usd": cost, "generation_retries": 0,
        "generated_at": "2026-10-05T00:00:00+00:00",
        "challenge": {
            "feature_trajectory": {"steps": [], "total_input_tokens": 10,
                                  "total_output_tokens": out_tok,
                                  "total_cost_usd": cost, "model_id": parts[0],
                                  "duration_seconds": 1.0},
            "bug_trajectory": {"steps": [], "total_input_tokens": 5,
                               "total_output_tokens": out_tok // 2,
                               "total_cost_usd": cost / 2,
                               "model_id": parts[0], "duration_seconds": 1.0},
        },
        "validation": {
            "passed": True, "gate_results": [], "attempt_number": 1,
            "self_review": {"detected": True, "findings": [], "detection_reason": "",
                            "fix_explanation": "",
                            "agent_trajectory": {"steps": [], "total_input_tokens": 3,
                                                 "total_output_tokens": out_tok // 4,
                                                 "total_cost_usd": 0.25,
                                                 "model_id": "rev",
                                                 "duration_seconds": 1.0}},
        },
    }


def _defense(did: str, blue: str, cid: str, cost: float = 0.75,
             out_tok: int = 800) -> dict[str, Any]:
    parts = blue.split("#")
    return {
        "defense_id": did, "challenge_id": cid, "blue_model_id": parts[0],
        "blue_harness_id": parts[1],
        "blue_reasoning_effort": parts[2] if len(parts) > 2 else "",
        "blue_provider": parts[3] if len(parts) > 3 else "",
        "duration_seconds": 1.0, "cost_usd": cost,
        "timestamp": "2026-10-05T00:00:00+00:00",
        "blue_fix": {
            "review_findings": [], "fix_explanation": "", "fix_diff": "",
            "modified_file_contents": {},
            "agent_trajectory": {"steps": [], "total_input_tokens": 7,
                                "total_output_tokens": out_tok,
                                "total_cost_usd": cost, "model_id": parts[0],
                                "duration_seconds": 1.0},
        },
        "score": {"s_regression": 1.0, "s_feature": 1.0, "s_bugfix": 1.0,
                  "blue_composite": 1.0, "red_composite": 0.0,
                  "test_details": {}},
    }


# ── zip prefix location + extraction ────────────────────────────


def test_locate_data_prefix_shapes() -> None:
    assert _locate_data_prefix(["data/matches/m.json"]) == "data/"
    assert _locate_data_prefix(["matches/m.json", "defenses/d.json"]) == ""
    assert _locate_data_prefix(["wrap/data/matches/m.json"]) == "wrap/data/"
    # The prefix with the most marker hits wins; ties prefer the shortest.
    assert _locate_data_prefix(
        ["data/matches/a.json", "data/defenses/b.json", "other/matches/c.json"]
    ) == "data/"
    with pytest.raises(SystemExit, match="none of the expected data"):
        _locate_data_prefix(["workspaces/junk.txt"])


def test_zip_sha256_stable(tmp_path: Path) -> None:
    p = tmp_path / "s.zip"
    p.write_bytes(b"hello")
    assert _zip_sha256(p) == _zip_sha256(p)
    assert len(_zip_sha256(p)) == 64


def test_extract_rejects_zip_slip(tmp_path: Path) -> None:
    from swe_duel.cli.tournament_update import _extract_submission

    zip_path = tmp_path / "slip.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/../../evil.txt", "boom")
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SystemExit, match="zip-slip"):
        _extract_submission(zip_path, dest)
    assert not list(dest.rglob("evil.txt"))


def test_extract_skips_unrelated_subtrees(tmp_path: Path) -> None:
    from swe_duel.cli.tournament_update import _extract_submission

    zip_path = tmp_path / "s.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/matches/m.json", "{}")
        zf.writestr("data/workspaces/huge.txt", "x" * 5000)
        zf.writestr("unrelated/other.txt", "y")
    dest = tmp_path / "dest"
    dest.mkdir()
    data_root, total = _extract_submission(zip_path, dest)
    assert data_root == dest / "data"
    assert total == len("{}")  # workspaces/ is never extracted → not counted
    assert (data_root / "matches" / "m.json").is_file()
    # Only the data subtree is extracted.
    assert not (dest / "unrelated").exists()
    assert not (data_root / "workspaces").exists()


# ── AS-state readers + tournament resolution ────────────────────


def test_as_state_readers() -> None:
    state = _as_state()
    assert _as_entered_tournament(state) == _TID
    assert _as_field(state) == [_A, _B, _N]
    assert _as_newcomers(state) == [_N]
    # Fallbacks when the recorded keys are missing: field derives from the
    # pairing/result endpoints, newcomers from the pairing host (side A).
    bare = {
        "format": "active_sampling", "entered_tournament": _TID,
        "pairings": [{"model_a": _N, "model_b": _A}],
        "results": [{"model_a": _N, "model_b": _A}],
    }
    assert _as_field(bare) == [_N, _A]
    assert _as_newcomers(bare) == [_N]
    # Legacy states name the tournament only via the rankings export path.
    assert _as_entered_tournament(
        {"rankings_source": "rankings/rankings_tid-9.json"}
    ) == "tid-9"
    assert _as_entered_tournament({}) == ""


def test_resolve_tournament_single_and_multi() -> None:
    states = [(Path("as-1.json"), _as_state())]
    tid, selected = _resolve_tournament(states, None)
    assert tid == _TID and selected == states
    # --tournament must match an entered tournament.
    with pytest.raises(SystemExit, match="not entered by any"):
        _resolve_tournament(states, "other-tid")
    assert _resolve_tournament(states, _TID) == (_TID, states)
    # Several tournaments → fail fast listing the ids.
    states.append((Path("as-2.json"), _as_state("tid-9999", path_name="as-2")))
    with pytest.raises(SystemExit, match="pass --tournament"):
        _resolve_tournament(states, None)
    # …and --tournament picks one, skipping the other.
    tid, selected = _resolve_tournament(states, _TID)
    assert tid == _TID and len(selected) == 1
    # Unlabeled states cannot be scoped without the flag.
    unlabeled = [(Path("as-x.json"), {"format": "active_sampling"})]
    with pytest.raises(SystemExit, match="pass --tournament"):
        _resolve_tournament(unlabeled, None)


# ── merge ───────────────────────────────────────────────────────


@pytest.fixture
def arena(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal organizer arena rooted at tmp_path (CWD for the command)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "arena.yaml").write_text("{}\n")
    data = tmp_path / "data"
    (data / "matches").mkdir(parents=True)
    (data / "tournaments").mkdir()
    (data / "defenses").mkdir()
    (data / "challenge_bank" / "challenges").mkdir(parents=True)
    (data / "challenge_bank" / "index.json").write_text(
        json.dumps({"pools": {}, "failed_pools": {}, "entries": {}})
    )
    (data / "matches" / "m-1.json").write_text(
        json.dumps(_match("m-1", _A, _B, "model_a_wins", _REPOS,
                          ts="2026-10-01T00:00:00+00:00"))
    )
    (data / "tournaments" / f"round_robin_state_{_TID}.json").write_text(
        json.dumps({
            "tournament_id": _TID, "format": "round_robin",
            "timestamp": "2026-10-01T00:00:00+00:00", "repo_names": _REPOS,
            "players": [{"model_id": _A, "points": 1.0},
                        {"model_id": _B, "points": 0.0}],
            "rounds": [[{"model_a": _A, "model_b": _B}]],
        })
    )
    return tmp_path


@pytest.fixture
def submission(arena: Path) -> Path:
    """The contributor's ./data zip (newcomer N actively sampled vs A)."""
    sub = arena / "sub"
    as_state = _as_state()
    (sub / "data" / "tournaments").mkdir(parents=True)
    (sub / "data" / "tournaments" / "active_sampling_state_as-run-1.json").write_text(
        json.dumps(as_state)
    )
    (sub / "data" / "matches").mkdir(parents=True)
    (sub / "data" / "matches" / "m-as-1.json").write_text(
        json.dumps(_match("m-as-1", _N, _A, "model_a_wins", _REPOS))
    )
    # An unrelated local match of the contributor's — must NOT be imported.
    (sub / "data" / "matches" / "m-unrelated.json").write_text(
        json.dumps(_match("m-unrelated", "x/y#mini-swe-agent",
                          "z/w#mini-swe-agent", "draw", ["flask"]))
    )
    ch_dir = sub / "data" / "challenge_bank" / "challenges"
    ch_dir.mkdir(parents=True)
    for cid, red, repo in (
        ("c-n-flask", _N, "flask"), ("c-n-jwt", _N, "jwt"),
        ("c-a-flask", _A, "flask"),
        ("c-foreign", "x/y#mini-swe-agent", "flask"),
    ):
        (ch_dir / f"{cid}.json").write_text(json.dumps(_challenge(cid, red, repo)))
    failed_dir = sub / "data" / "challenge_bank" / "failed_challenges"
    failed_dir.mkdir(parents=True)
    (failed_dir / "f-n-flask.json").write_text(json.dumps({
        "challenge_id": "f-n-flask", "red_model_id": "z-ai/glm-5.3-flash",
        "red_harness_id": "mini-swe-agent", "red_reasoning_effort": "high",
        "red_provider": "z-ai", "repo_name": "flask", "kind": "validation",
        "error_message": "gate_bug_tests=failed", "attempt_number": 3,
        "slot": 1, "generation_cost_usd": 0.4, "elapsed_seconds": 2.0,
        "generated_at": "2026-10-05T00:00:00+00:00",
        "challenge": None, "validation": None,
    }))
    def_dir = sub / "data" / "defenses"
    def_dir.mkdir(parents=True)
    for did, blue, cid in (
        ("d-as-1", _N, "c-a-flask"), ("d-as-2", _A, "c-n-flask"),
        ("d-as-3", _B, "c-a-flask"), ("d-foreign", "x/y#mini-swe-agent", "c-foreign"),
    ):
        (def_dir / f"{did}.json").write_text(json.dumps(_defense(did, blue, cid)))
    (sub / "data" / "workspaces").mkdir(parents=True)
    (sub / "data" / "workspaces" / "huge.txt").write_text("x" * 5000)

    zip_path = arena / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in sorted(sub.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(sub))
    return zip_path


def test_merge_scopes_to_field_and_merges_index(
    arena: Path, submission: Path
) -> None:
    import tempfile

    from swe_duel.cli.tournament_update import _extract_submission, _load_as_states

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "x"
        dest.mkdir()
        sub_data, _total = _extract_submission(submission, dest)
        states = _load_as_states(sub_data / "tournaments")
        selected = _resolve_tournament(states, _TID)[1]
        counts, _copied, as_matches, challenges, defenses = _merge_submission(
            sub_data, arena / "data", selected, _TID
        )
    assert [m["match_id"] for m in as_matches] == ["m-as-1"]
    assert counts.matches == 1 and counts.defenses == 3 and counts.challenges == 3
    assert counts.failed_challenges == 1 and counts.as_states == 1
    # Foreign records are excluded; the unrelated match is not claimed.
    data = arena / "data"
    assert sorted(p.name for p in (data / "matches").glob("*.json")) == [
        "m-1.json", "m-as-1.json"
    ]
    assert not (data / "defenses" / "d-foreign.json").exists()
    assert not (data / "challenge_bank" / "challenges" / "c-foreign.json").exists()
    assert (data / "defenses" / "d-as-1.json").is_file()
    assert (data / "challenge_bank" / "failed_challenges" / "f-n-flask.json").is_file()
    assert (
        data / "tournaments" / "active_sampling_state_as-run-1.json"
    ).is_file()
    # index.json: pools + failed_pools + entries merged under pinned keys.
    idx = json.loads((data / "challenge_bank" / "index.json").read_text())
    pool_key_n = f"{_N}::flask"
    assert idx["pools"][pool_key_n] == ["c-n-flask"]
    assert idx["pools"][f"{_A}::flask"] == ["c-a-flask"]
    assert idx["failed_pools"][pool_key_n][0]["id"] == "f-n-flask"
    assert "c-foreign" not in idx["entries"]
    assert idx["entries"]["c-n-flask"]["red_provider"] == "z-ai"

    # Idempotent: re-merging the same submission skips everything existing.
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "x"
        dest.mkdir()
        sub_data, _total = _extract_submission(submission, dest)
        selected = _resolve_tournament(
            _load_as_states(sub_data / "tournaments"), _TID
        )[1]
        counts2, _c2, as_matches2, _ch2, _d2 = _merge_submission(
            sub_data, arena / "data", selected, _TID
        )
    assert counts2.matches == 0 and counts2.defenses == 0
    assert counts2.challenges == 0 and counts2.as_states == 0
    assert counts2.skipped_existing > 0
    assert [m["match_id"] for m in as_matches2] == ["m-as-1"]
    idx2 = json.loads((data / "challenge_bank" / "index.json").read_text())
    assert idx2["pools"][pool_key_n] == ["c-n-flask"]  # no duplicates


def test_merge_index_rejects_bad_shape(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"pools": "not-a-dict"}))
    with pytest.raises(SystemExit, match="unexpected shape"):
        _merge_index(
            path,
            [{
                "key": "k", "failed": False,
                "entry": {"id": "c-1"},
            }],
            None,  # type: ignore[arg-type]
        )


# ── newcomer stats ─────────────────────────────────────────────


def test_newcomer_stats_from_contribution() -> None:
    as_matches = [_match("m-as-1", _N, _A, "model_a_wins", _REPOS)]
    # Newcomer's challenges (incl. one failed attempt) + its defense.
    challenges = [
        _challenge("c-n-flask", _N, "flask", 1.5, 1000),
        _challenge("c-n-jwt", _N, "jwt", 2.5, 2000),
        {"challenge_id": "f-n", "red_model_id": "z-ai/glm-5.3-flash",
         "red_harness_id": "mini-swe-agent", "red_reasoning_effort": "high",
         "red_provider": "z-ai", "repo_name": "flask",
         "generation_cost_usd": 0.4, "challenge": None, "validation": None},
        _challenge("c-a-flask", _A, "flask"),  # incumbent's — not N's
    ]
    defenses = [
        _defense("d-1", _N, "c-a-flask", 0.75, 800),
        _defense("d-2", _A, "c-n-flask"),  # incumbent's — not N's
    ]
    stats = _newcomer_stats(_N, as_matches, challenges, defenses)
    # gen: 1.5 + 2.5 + 0.4 (failed attempt burned real generation cost)
    assert stats.gen_cost == pytest.approx(4.4)
    assert stats.val_cost == pytest.approx(0.5)  # 0.25 self-review × 2
    assert stats.eval_cost == pytest.approx(0.75)
    # tokens: challenges (1000+500+250) + (2000+1000+500) + defense 800
    assert stats.out_tokens == 6050
    assert stats.match_out_tokens == 800
    assert stats.record == "1W · 0D · 0L"
    # Turn-level duel counters over contested turns only.
    assert stats.turns_defended == 2
    assert stats.defenses_broken == 0
    assert stats.defense_success == 1.0
    assert stats.attacks_landed == 0


# ── per-match cost/token math (the page's cell semantics) ───────


def test_pm_cost_tokens_and_cells() -> None:
    """Per-match cell semantics: spend / output tokens divided by the
    entry's matches played, zero-guarded when no matches were played."""
    from swe_duel.cli.tournament_update import _cost_tok_cells, _pm_cost_tokens

    stats = _newcomer_stats(
        _N,
        [_match("m-as-1", _N, _A, "model_a_wins", _REPOS)],
        [_challenge("c-n-flask", _N, "flask", cost=3.0, out_tok=2000)],
        [_defense("d-1", _N, "c-a-flask", cost=1.0, out_tok=1000)],
    )
    # 3.0 gen + 0.25 val + 1.0 eval = $4.25 over 3,500 + 1,000 tokens.
    entry4 = _entry("z-ai/glm-5.3-flash", "mini-swe-agent", 1500.0, 2.0, 4,
                    effort="high", provider="z-ai")
    assert _pm_cost_tokens(entry4, stats) == (pytest.approx(1.0625), 1125)
    entry0 = _entry("z-ai/glm-5.3-flash", "mini-swe-agent", 1500.0, 0.0, 0,
                    effort="high", provider="z-ai")
    assert _pm_cost_tokens(entry0, stats) == (0.0, 0)

    cost_cell, tok_cell = _cost_tok_cells(entry4, stats, 2.125, 2250)
    assert '<span class="cv">$1.06</span>' in cost_cell
    assert '<span class="cbar"><i style="width:50.0%"></i></span>' in cost_cell
    assert "$1.06 per match over 4 matches" in cost_cell
    assert "1,125 output tokens per match (4,500 total over 4 matches)" in tok_cell
    assert '<span class="cv">1K</span>' in tok_cell
    assert '<span class="cbar"><i style="width:50.0%"></i></span>' in tok_cell

    # No matches played: zeroed averages, no division, disclosed titles.
    cost0, tok0 = _cost_tok_cells(entry0, stats, 2.125, 2250)
    assert '<span class="cv">$0.00</span>' in cost0
    assert '<span class="cv">0</span>' in tok0
    assert "no matches played" in cost0 and "no matches played" in tok0
    assert 'width:0.0%' in cost0 and 'width:0.0%' in tok0


def test_parse_row_converts_legacy_cumulative_attrs() -> None:
    """A pre-per-match row (cumulative data-total_usd / data-out_tokens)
    is converted via its own 'Matches played' count; the current shape
    parses its per-match attrs directly."""
    from swe_duel.cli.tournament_update import _parse_row

    block = (
        '<details class="rwrap" data-rank="1" data-elo="1500.0" '
        'data-total_usd="20.00" data-out_tokens="9000">\n'
        '<summary><div class="row"><div class="cell cell-rank">1</div>'
        '<div class="cell entrant" title="openai/gpt-5.5 · OpenAI">'
        '<span class="enames"><span class="nm">GPT-5.5</span>'
        '<span class="pills"><span class="hpill">Codex CLI</span></span></span></div>'
        "</div></summary>\n"
        '<div class="detail"><div class="kv"><dt>Matches played</dt><dd>4</dd></div></div>'
        "\n</details>"
    )
    parsed = _parse_row(block)
    assert parsed is not None
    assert parsed["model_harness"] == ("openai/gpt-5.5", "codex")
    assert parsed["cost"] == pytest.approx(5.0)
    assert parsed["tokens"] == 2250
    migrated = block.replace(
        'data-total_usd="20.00" data-out_tokens="9000"',
        'data-usd_per_match="5.00" data-tokens_per_match="2250"',
    )
    parsed2 = _parse_row(migrated)
    assert parsed2 is not None
    assert parsed2["cost"] == pytest.approx(5.0)
    assert parsed2["tokens"] == 2250


# ── leaderboard page math (regression vs the hand-built page) ───


def test_page_axis_and_bars_match_original_page() -> None:
    # The shipped index.html was generated with axis [min elo, ceil50(max)+50]
    # and bars clamped at a 3% minimum; the real 72510128 elos must reproduce
    # the shipped bars within the 1-decimal Elo rounding tolerance.
    real_elos = [1589.9, 1582.1, 1556.1, 1537.5, 1527.7, 1513.6,
                 1481.1, 1451.5, 1436.5, 1424.2, 1399.8]
    shipped_bars = [75.961, 72.850, 62.484, 55.021, 51.119, 45.469,
                    32.487, 20.676, 14.660, 9.748, 3.000]
    lo, hi = _page_axis(real_elos)
    assert lo == pytest.approx(1399.8)
    assert hi == pytest.approx(1650.0)
    for elo, want in zip(real_elos, shipped_bars):
        assert _bar_width(elo, lo, hi) == pytest.approx(want, abs=0.1)
    # The weakest entrant's bar clamps at 3%.
    assert _bar_width(1399.8, lo, hi) == pytest.approx(3.0)


def test_identity_and_display_helpers() -> None:
    from swe_duel.cli.tournament_update import (
        _display_name,
        _org_of,
    )

    ident = _Identity.from_cid(_N)
    assert ident == _Identity("z-ai/glm-5.3-flash", "mini-swe-agent", "high", "z-ai")
    assert _Identity.from_cid("m/x#mini-swe-agent").harness == "mini-swe-agent"
    assert _display_name("z-ai/glm-5.3-flash") == "GLM-5.3-Flash"
    assert _display_name("openai/gpt-5.5") == "GPT-5.5"
    assert _display_name("qwen/qwen3.7-max") == "Qwen3.7-Max"
    assert _org_of("z-ai/glm-5.3-flash") == "Z.ai"
    assert _org_of("unknown-lab/model-x") == "Unknown-lab"


# ── leaderboard page update ─────────────────────────────────────


_MINI_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="description" content="Elo leaderboard: 2 entrants.">
<style>.hrow{display:grid} .row{display:grid} .gridlines{position:relative}</style>
</head><body>
<div class="wrap">
  <div class="board-card"><div class="board" id="board">
      <div class="gridlines"><div class="gl-line" style="left:60.0%"></div><div class="gl-label" style="left:60.0%">1500</div></div>
      <div class="hrow" id="hrow"><div class="hcell sortable" data-sort="rank">Rank</div></div>
      <div class="rows" id="rows">
<details class="rwrap" data-rank="1" data-elo="1516.0" data-total_usd="10.00" data-out_tokens="100000">
<summary><div class="row"><div class="cell cell-rank">1</div><div class="cell entrant" title="openai/gpt-5.5 · OpenAI"><span class="fdot" style="background:#59c886"></span><span class="enames"><span class="nm">GPT-5.5</span><span class="pills"><span class="hpill">Codex CLI</span><span class="org">OpenAI</span></span></span></div><div class="cell chart"><div class="track"><div class="bar pos" style="left:0.000%;width:88.888%"></div></div><div class="val"><span class="btv">1516.0</span><span class="btci">[1490.0, 1540.0]</span></div></div><div class="cell cnum cell-cost"><span class="cv">$10.00</span></div><div class="cell cnum cell-tok"><span class="cv">100K</span></div><div class="cell cell-chev">▸</div></div></summary>
<div class="detail"><div class="dgrid"><section class="dcard"><h4>Match record</h4><div class="dhead">1W · 0D · 0L<span class="dsub">win · draw · loss</span></div></section></div></div>
</details>
<details class="rwrap" data-rank="2" data-elo="1484.0" data-total_usd="5.00" data-out_tokens="50000">
<summary><div class="row"><div class="cell cell-rank">2</div><div class="cell entrant" title="anthropic/claude-opus-4.8 · Anthropic"><span class="fdot" style="background:#fe9b61"></span><span class="enames"><span class="nm">Claude Opus 4.8</span><span class="pills"><span class="hpill">mini-swe-agent</span><span class="org">Anthropic</span></span></span></div><div class="cell chart"><div class="track"><div class="bar pos" style="left:0.000%;width:44.444%"></div></div><div class="val"><span class="btv">1484.0</span><span class="btci">[1460.0, 1510.0]</span></div></div><div class="cell cnum cell-cost"><span class="cv">$5.00</span></div><div class="cell cnum cell-tok"><span class="cv">50K</span></div><div class="cell cell-chev">▸</div></div></summary>
<div class="detail"><div class="dgrid"><section class="dcard"><h4>Match record</h4><div class="dhead">0W · 0D · 1L<span class="dsub">win · draw · loss</span></div></section></div></div>
</details>
            </div>
    </div>
  </div>
  <div class="legend">
    <span class="chip">bracketed values = 95% percentile CI</span>
    <span class="chip"><a href="methodology.html">full methodology ↗</a></span>
    <span class="chip"><a href="contributions/index.html">1 late entrant(s) joined via active sampling · contribution history ↗</a></span>
  </div>
</div>
<footer></footer>
</body></html>
"""


def _mini_rankings(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rankings": entries}


def _entry(model: str, harness: str, elo: float, points: float,
           matches: int, effort: str | None = None,
           provider: str | None = None) -> dict[str, Any]:
    return {
        "model": model, "harness": harness, "elo": elo, "points": points,
        "matches_played": matches, "reasoning_effort": effort,
        "provider": provider or "(OpenRouter auto-route)",
    }


_OLD = _mini_rankings([
    _entry("openai/gpt-5.5", "codex", 1516.0, 1.0, 1, effort="medium"),
    _entry("anthropic/claude-opus-4.8", "mini-swe-agent", 1484.0, 0.0, 1,
           effort="high"),
])
_NEW = _mini_rankings([
    _entry("openai/gpt-5.5", "codex", 1499.3, 1.0, 2, effort="medium"),
    _entry("anthropic/claude-opus-4.8", "mini-swe-agent", 1484.0, 0.0, 1,
           effort="high"),
    _entry("z-ai/glm-5.3-flash", "mini-swe-agent", 1516.7, 1.0, 2,
           effort="high", provider="z-ai"),
])


class _Snap:
    red_elo = 1472.0
    blue_elo = 1526.6
    trueskill_mu = 31.0
    trueskill_sigma = 6.87


def test_update_leaderboard_refreshes_core_columns(tmp_path: Path) -> None:
    import re

    page = tmp_path / "index.html"
    page.write_text(_MINI_PAGE)
    stats = {
        "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai": _newcomer_stats(
            "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai",
            [
                _match("m-as-1", _N, _A, "model_a_wins", _REPOS),
                _match("m-as-2", _N, _B, "model_b_wins", _REPOS),
            ],
            [_challenge("c-n-flask", _N, "flask")],
            [_defense("d-1", _N, "c-a-flask")],
        )
    }
    appended = _update_leaderboard(
        page, _OLD, _NEW, stats, {"z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai": _Snap()},
        {
            "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai": _StrengthDiag(
                draws=10,
                ranked=10,
                n_participants=3,
                rank_hist=[0.7, 0.3, 0.0],
                median_rank=1,
                rank_lo=1,
                rank_hi=2,
                bt=1.2,
                bt_lo=0.3,
                bt_hi=2.4,
                contested=0.1,
                contested_lo=-0.2,
                contested_hi=0.4,
                elo=1516.7,
                elo_lo=1490.0,
                elo_hi=1540.0,
            )
        },
    )
    assert appended == ["z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai"]
    text = page.read_text()

    rows = re.findall(r'<details class="rwrap".*?</details>', text, re.S)
    assert len(rows) == 3
    # Incumbent A: rank 1 → 2 (newcomer's Elo is higher), Elo + bar refreshed.
    a = rows[0]
    assert 'data-rank="2"' in a and 'data-elo="1499.3"' in a
    assert re.search(r'cell-rank">2<', a)
    assert re.search(r'btv">1499\.3<', a)
    assert 'width:13.190%"' in a  # (1499.3-1484.0)/116.0
    # Its untouched parts: CI badge, cost/tok columns, detail card.
    assert '<span class="btci">[1490.0, 1540.0]</span>' in a
    assert '<span class="cv">$10.00</span>' in a
    assert "<h4>Match record</h4>" in a and "1W · 0D · 0L" in a
    # Incumbent rows never grow the new components — only appended rows do.
    assert "Strength diagnostics" not in a
    # Incumbent B keeps its Elo but is re-ranked below the newcomer.
    b = rows[1]
    assert 'data-rank="3"' in b and 'data-elo="1484.0"' in b
    assert "width:3.000%" in b  # weakest entrant → 3% clamp
    # Newcomer row appended last, in the page's late-entrant shape.
    n = rows[2]
    assert 'data-rank="1"' in n and 'data-elo="1516.7"' in n
    assert "GLM-5.3-Flash" in n and 'data-late-entrant="1"' in n
    assert "Z.ai" in n and "z-ai/glm-5.3-flash · Z.ai" in n
    assert "1W · 0D · 1L" in n
    assert "Turns defended" in n
    # Per-match cells: $2.50 spend (1.5 gen + 0.25 val + 0.75 eval) and
    # 2,550 output tokens averaged over the entry's 2 matches.
    assert 'data-usd_per_match="1.25"' in n and 'data-tokens_per_match="1275"' in n
    assert "$1.25" in n
    assert (
        '<div class="cell cnum cell-cost" '
        'title="$1.25 per match over 2 matches · '
        'generation $1.50 · validation $0.25 · match play $0.75">' in n
    )
    assert "Cost per match</dt><dd>$1.25</dd>" in n
    assert "Tokens per match</dt><dd>1,275</dd>" in n
    # The appended row carries the same strength components as the
    # original field's rows: diagnostics card + bootstrap rank histogram.
    assert "<h4>Strength diagnostics</h4>" in n
    assert "<h4>Bootstrap rank distribution</h4>" in n
    assert "median bootstrap rank" in n
    assert 'class="dhead">#1<span class="dsub">median bootstrap rank' in n
    assert "95% rank interval</dt><dd>1–2" in n
    assert "Match-level BT</dt><dd><b>+1.20</b></dd>" in n
    assert "Contested BT</dt><dd>+0.10</dd>" in n
    assert 'class="hb med in" title="rank 1: 70.0% of resamples"' in n
    assert 'class="hb in" title="rank 2: 30.0% of resamples"' in n
    assert 'class="hb out" title="rank 3: 0.0% of resamples"' in n
    assert "<h4>Bootstrap rank distribution</h4>" in n

    # Axis gridlines regenerated for [1484.0, 1600.0].
    assert 'left:13.793%"' in text and ">1500<" in text
    assert 'left:56.897%"' in text and ">1550<" in text
    # Meta description follows the field.
    assert "3 entrants" in text
    # The legacy late-entrant / contribution-history chip is removed.
    assert "contributions/index.html" not in text
    assert "late entrant(s)" not in text
    # No model-color dots anywhere (legacy incumbents' dots stripped too).
    assert "fdot" not in text
    # Reasoning effort + provider live only in the expanded detail block.
    for row in rows:
        summary, detail = row.split("</summary>", 1)
        assert "Reasoning effort" not in summary and "Provider" not in summary
        assert detail.count('<div class="dident">') == 1
    assert (
        '<span class="ik">Reasoning effort</span><span class="iv">high</span>' in n
        and '<span class="ik">Provider</span><span class="iv">z-ai</span>' in n
        and '<span class="ik">Model</span><span class="iv">z-ai/glm-5.3-flash</span>' in n
    )
    assert '<span class="ik">Provider</span><span class="iv">(auto-route)</span>' in a
    assert '<span class="ik">Harness</span><span class="iv">Codex CLI</span>' in a
    assert ".dident{" in text  # stylesheet gains the strip's CSS

    # A later import with another fresh entrant appends exactly one more row
    # (identity strips are rewritten, never duplicated).
    new2 = {
        "rankings": _NEW["rankings"]
        + [_entry("moonshotai/kimi-k2.7-code", "mini-swe-agent", 1490.0, 0.5, 1)]
    }
    stats2 = dict(stats)
    stats2["moonshotai/kimi-k2.7-code#mini-swe-agent"] = stats[
        "z-ai/glm-5.3-flash#mini-swe-agent#high#z-ai"
    ]
    old2 = {"rankings": _NEW["rankings"]}
    appended2 = _update_leaderboard(
        page, old2, new2, stats2,
        {"moonshotai/kimi-k2.7-code#mini-swe-agent": _Snap()},
    )
    assert appended2 == ["moonshotai/kimi-k2.7-code#mini-swe-agent"]
    text2 = page.read_text()
    assert text2.count('class="rwrap"') == 4
    assert "contributions/index.html" not in text2
    assert text2.count('<div class="dident">') == 4
    assert text2.count(".dident{") == 1
    assert (
        '<span class="ik">Reasoning effort</span><span class="iv">(default)</span>' in text2
    )


def test_update_leaderboard_skips_missing_page(tmp_path: Path) -> None:
    assert _update_leaderboard(
        tmp_path / "missing.html", _OLD, _NEW, {}, {}
    ) == []


def test_update_leaderboard_needs_entries(tmp_path: Path) -> None:
    page = tmp_path / "index.html"
    page.write_text(_MINI_PAGE)
    with pytest.raises(SystemExit, match="no entries"):
        _update_leaderboard(page, _OLD, _mini_rankings([]), {}, {})


# ── end to end: main() ─────────────────────────────────────────


def _update_args(zip_path: Path, arena: Path, *extra: str) -> list[str]:
    return [
        str(zip_path), *extra,
        "--output-dir", str(arena / "data"),
        "--rankings-dir", str(arena / "rankings"),
        "--contributions-dir", str(arena / "contributions"),
        "--index-html", str(arena / "index.html"),
    ]


def test_main_end_to_end(arena: Path, submission: Path) -> None:
    (arena / "index.html").write_text(_MINI_PAGE)
    from swe_duel.cli.export_rankings import export_rankings

    # The organizer's pre-contribution export + leaderboard baseline.
    export_rankings(_TID, arena / "data", arena / "rankings")
    assert update_main(_update_args(submission, arena)) == 0

    # ── rankings: newcomer seated, incumbents re-rated ──────────
    payload = json.loads(
        (arena / "rankings" / f"rankings_{_TID}.json").read_text()
    )
    by_model = {r["model"]: r for r in payload["rankings"]}
    assert "z-ai/glm-5.3-flash" in by_model
    assert by_model["z-ai/glm-5.3-flash"]["points"] == 1.0
    assert by_model["z-ai/glm-5.3-flash"]["provider"] == "z-ai"
    assert by_model["openai/gpt-5.5"]["points"] == 1.0  # state points only
    assert [m["match_id"] for m in payload["matches"]] == ["m-1", "m-as-1"]

    # ── leaderboard: refreshed + one appended row ───────────────
    page = (arena / "index.html").read_text()
    assert page.count('class="rwrap"') == 3
    assert "GLM-5.3-Flash" in page and 'data-late-entrant="1"' in page
    # The refresh never unbalances the page's markup (orphan closing tags
    # mangle the rendered layout).
    assert len(re.findall(r"<div\b", page)) == len(re.findall(r"</div>", page))
    assert len(re.findall(r"<details\b", page)) == len(re.findall(r"</details>", page))
    assert len(re.findall(r"<span\b", page)) == len(re.findall(r"</span>", page))

    # ── archive + history ───────────────────────────────────────
    sha = _zip_sha256(submission)
    contribution_dir = arena / "contributions" / sha
    assert (contribution_dir / "submission.zip").is_file()
    manifest = json.loads((contribution_dir / "manifest.json").read_text())
    assert manifest["hash"] == sha and manifest["tournament_id"] == _TID
    assert manifest["added"]["matches"] == 1
    assert (contribution_dir / "data" / "matches" / "m-as-1.json").is_file()
    registry = json.loads(
        (arena / "contributions" / "contributions.json").read_text()
    )
    assert [e["hash"] for e in registry["contributions"]] == [sha]
    history = (arena / "contributions" / "index.html").read_text()
    assert "GLM-5.3-Flash" in history and "tid-2222" in history

    # ── duplicate guard: the same zip is rejected ───────────────
    with pytest.raises(SystemExit, match="already"):
        update_main(_update_args(submission, arena))


def test_main_second_contribution_grows_history(arena: Path, submission: Path) -> None:
    (arena / "index.html").write_text(_MINI_PAGE)
    from swe_duel.cli.export_rankings import export_rankings

    export_rankings(_TID, arena / "data", arena / "rankings")
    assert update_main(_update_args(submission, arena)) == 0

    # A second, different zip: the SAME entrant re-enters and plays B now.
    as2 = _as_state(results=[{"match_id": "m-as-2", "model_a": _N,
                              "model_b": _B, "outcome": "draw"}],
                    path_name="as-run-2")
    as2["pairings"] = [{"model_a": _N, "model_b": _B}]
    sub2 = arena / "sub2"
    (sub2 / "data" / "tournaments").mkdir(parents=True)
    (sub2 / "data" / "tournaments" / "active_sampling_state_as-run-2.json").write_text(
        json.dumps(as2)
    )
    (sub2 / "data" / "matches").mkdir(parents=True)
    (sub2 / "data" / "matches" / "m-as-2.json").write_text(
        json.dumps(_match("m-as-2", _N, _B, "draw", _REPOS,
                          ts="2026-10-08T00:00:00+00:00"))
    )
    zip2 = arena / "submission-2.zip"
    with zipfile.ZipFile(zip2, "w") as zf:
        for p in sorted(sub2.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(sub2))

    assert update_main(_update_args(zip2, arena)) == 0

    payload = json.loads(
        (arena / "rankings" / f"rankings_{_TID}.json").read_text()
    )
    by_model = {r["model"]: r for r in payload["rankings"]}
    # Returning entrant: accumulated points + matches across BOTH imports.
    assert by_model["z-ai/glm-5.3-flash"]["points"] == 1.5
    assert by_model["z-ai/glm-5.3-flash"]["matches_played"] == 2
    assert [m["match_id"] for m in payload["matches"]] == [
        "m-1", "m-as-1", "m-as-2"
    ]
    # The returning entrant already has a leaderboard row: its rank/Elo are
    # refreshed in place, and NO duplicate row is appended (its invisible
    # marker stays unique).
    page = (arena / "index.html").read_text()
    assert page.count('class="rwrap"') == 3
    assert page.count('data-late-entrant="1"') == 1

    registry = json.loads(
        (arena / "contributions" / "contributions.json").read_text()
    )
    assert len(registry["contributions"]) == 2
    assert {e["tournament_id"] for e in registry["contributions"]} == {_TID}
    history = (arena / "contributions" / "index.html").read_text()
    assert history.count("<section class='card'") == 2


def test_main_drops_history_chip_on_nested_page(
    arena: Path, submission: Path
) -> None:
    """The default leaderboard lives at ./docs/index.html (GitHub Pages
    root) while the history stays at ./contributions/ — the page no longer
    links it: a legacy late-entrant / contribution-history chip is removed,
    while the history itself is still written."""
    from swe_duel.cli.export_rankings import export_rankings

    (arena / "docs").mkdir()
    (arena / "docs" / "index.html").write_text(_MINI_PAGE)
    export_rankings(_TID, arena / "data", arena / "rankings")
    args = _update_args(submission, arena)
    args[args.index("--index-html") + 1] = str(arena / "docs" / "index.html")
    assert update_main(args) == 0

    page = (arena / "docs" / "index.html").read_text()
    assert "contributions/index.html" not in page
    assert "late entrant(s)" not in page
    assert (arena / "contributions" / "index.html").is_file()


def test_main_requires_existing_data_dir(arena: Path, submission: Path) -> None:
    import shutil

    shutil.rmtree(arena / "data" / "defenses")
    with pytest.raises(SystemExit, match="not found"):
        update_main(_update_args(submission, arena))


def test_main_rejects_missing_zip(arena: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        update_main(_update_args(arena / "nope.zip", arena))

# ── bootstrap strength diagnostics (late-entrant cards) ────────


def test_contested_turn_matches_excludes_auto_wins() -> None:
    """The contested-turn BT fit sees one observation per real duel: a
    missing-Red auto-win turn (empty challenge_id, synthetic defense_id)
    never contributes — no defense ran."""
    from swe_duel.scoring.bootstrap import contested_turn_matches

    raw = [
        {
            "match_id": "m-1",
            "model_a_id": _A,
            "model_b_id": _B,
            "repo_name": "flask",
            "outcome": "model_a_wins",
            "turns": [
                _turn(0, _A, _B, "flask", "c-1", "d-1", red_comp=1.0),
                _turn(1, _A, _B, "flask", "", "d-auto", red_comp=0.0),
            ],
        }
    ]
    got = contested_turn_matches(raw)
    assert len(got) == 1
    assert got[0].model_a_id == _A
    assert got[0].model_b_id == _B
    assert got[0].outcome.value == "model_a_wins"


def test_bootstrap_field_diagnostics_ranking_and_determinism() -> None:
    """One seeded bootstrap pass yields per-participant rank histograms and
    BT CIs that respect the match ordering; a weakly-connected player whose
    matches a resample dropped is counted unranked, never fabricated."""
    _C = "moonshotai/kimi-k2.7-code#mini-swe-agent"
    combined = [
        _match("m-1", _A, _B, "model_a_wins", ["flask"]),
        _match("m-2", _A, _C, "model_a_wins", ["flask"]),
        _match("m-3", _B, _C, "model_a_wins", ["flask"]),
        _match("m-4", _A, _B, "model_a_wins", ["flask"]),
    ]
    diags = _bootstrap_field_diagnostics(combined, draws=200, seed=7)
    assert set(diags) == {_A, _B, _C}
    a, b, c = diags[_A], diags[_B], diags[_C]
    # Dominance ordering: match-level BT strictly follows A > B > C, and
    # the bootstrap medians agree.
    assert a.bt > b.bt > c.bt
    assert a.median_rank == 1 and c.median_rank == 3
    # A resample CAN drop even a well-connected player's matches (it
    # samples matches with replacement) — those draws count as unranked.
    assert 0 < c.ranked < a.ranked <= 200
    assert abs(sum(a.rank_hist) - 1.0) < 1e-9
    assert len(a.rank_hist) == 3
    assert a.rank_lo <= a.median_rank <= a.rank_hi
    assert a.bt_lo <= a.bt <= a.bt_hi
    assert a.contested_lo <= a.contested <= a.contested_hi
    # _C is the least-connected player (two of the four matches): it
    # missed some resamples entirely, and those draws are counted
    # unranked (its .ranked sits below every better-connected player),
    # never fabricated as rank 0.
    assert c.ranked < 200
    # Determinism: the same seed reproduces identical diagnostics.
    diags2 = _bootstrap_field_diagnostics(combined, draws=200, seed=7)
    assert diags2[_A] == a and diags2[_C] == c


def test_newcomer_row_html_strength_components() -> None:
    """The appended row's markup matches the incumbent rows' shape: the
    diagnostics dcard (signed BT, CIs, interval) and the rank histogram
    (in/out dimming, median accent, hp labels, resample caption)."""
    stats = _newcomer_stats(
        _N,
        [_match("m-as-1", _N, _A, "model_a_wins", _REPOS)],
        [_challenge("c-n-flask", _N, "flask")],
        [_defense("d-1", _N, "c-a-flask")],
    )
    diag = _StrengthDiag(
        draws=1000,
        ranked=995,
        n_participants=12,
        rank_hist=[0.01, 0.05, 0.4, 0.3, 0.14, 0.05, 0.02, 0.01, 0.01, 0.005, 0.0, 0.0],
        median_rank=3,
        rank_lo=2,
        rank_hi=7,
        bt=1.9,
        bt_lo=0.4,
        bt_hi=5.2,
        contested=0.2,
        contested_lo=-0.3,
        contested_hi=0.7,
        elo=1573.5,
        elo_lo=1520.0,
        elo_hi=1640.0,
    )
    entry = _entry(
        "z-ai/glm-5.3-flash", "mini-swe-agent", 1516.7, 1.0, 1,
        effort="high", provider="z-ai",
    )
    out = _newcomer_row_html(entry, 1, stats, _Snap(), 1484.0, 1600.0, 10.0, 100000, diag)
    # The 95% Elo CI badge sits directly under the Elo value, exactly as
    # the original field's rows carry it.
    assert '<span class="btv">1516.7</span><span class="btci">[1520.0, 1640.0]</span>' in out
    # Per-match cost/token attrs and cells (1 match → averages = totals).
    assert 'data-usd_per_match="2.50"' in out
    assert 'data-tokens_per_match="2550"' in out
    assert '<span class="cv">$2.50</span>' in out
    assert "Cost per match</dt><dd>$2.50</dd>" in out
    assert "Tokens per match</dt><dd>2,550</dd>" in out
    assert "<h4>Strength diagnostics</h4>" in out
    assert 'class="dhead">#3<span class="dsub">median bootstrap rank' in out
    assert "95% rank interval</dt><dd>2–7" in out
    assert "Match-level BT</dt><dd><b>+1.90</b></dd>" in out
    assert '<span class="ci">[+0.40, +5.20]</span>' in out
    assert "Contested BT</dt><dd>+0.20</dd>" in out
    assert "[−0.30, +0.70]" in out
    assert "<h4>Bootstrap rank distribution</h4>" in out
    # Unranked resamples are disclosed, not silently dropped.
    assert "995 of 1,000 match-level resamples" in out
    assert "5 resample(s) contained none of its matches" in out
    # Median bar: med + in classes, the palette accent, full height.
    assert 'class="hb med in" title="rank 3: 40.0% of resamples"' in out
    assert 'style="height:100.00%;--hc:#59c886"' in out
    # Interval membership drives the dimming; rank 1 sits outside 2–7.
    assert 'class="hb out" title="rank 1: 1.0% of resamples"' in out
    # hp labels only for shares rounding to ≥1%; zero shares keep the
    # hairline height.
    assert '<span class="hp">40%</span>' in out
    assert 'title="rank 11: 0.0% of resamples"><span class="hp"></span>' in out
    assert 'title="rank 11: 0.0% of resamples"><span class="hp"></span><i style="height:1.20%"' in out
    # The rank axis spans the full merged field.
    assert "<div class=\"hlabs\"><span>1</span><span>2</span><span>3</span>" in out
    assert "<span>12</span></div>" in out


def test_newcomer_row_html_without_diag_keeps_basic_cards() -> None:
    """Without a bootstrap (e.g. an entrant whose matches never connect),
    the row still renders the basic cards and no empty components."""
    stats = _newcomer_stats(
        _N,
        [_match("m-as-1", _N, _A, "model_a_wins", _REPOS)],
        [_challenge("c-n-flask", _N, "flask")],
        [_defense("d-1", _N, "c-a-flask")],
    )
    entry = _entry(
        "z-ai/glm-5.3-flash", "mini-swe-agent", 1516.7, 1.0, 1,
        effort="high", provider="z-ai",
    )
    out = _newcomer_row_html(entry, 1, stats, _Snap(), 1484.0, 1600.0, 10.0, 100000)
    assert "<h4>Match record</h4>" in out
    assert "<h4>Duel outcomes</h4>" in out
    assert "<h4>Cost &amp; tokens</h4>" in out
    assert "Strength diagnostics" not in out
    assert "Bootstrap rank distribution" not in out


def test_refresh_row_block_rewrites_incumbent_in_place(tmp_path: Path) -> None:
    """A matched incumbent row is fully refreshed from merged-history stats:
    the summary keeps its identity visuals (pills, old btci badge; the
    legacy model-color dot is dropped
    shape) while rank/Elo attrs, the 95% Elo CI badge, the cost/token
    cells, and the entire detail block (all five cards + rank histogram)
    are regenerated — even for a row whose cells predate the cbar/title
    markup."""
    import re
    from swe_duel.cli.tournament_update import _refresh_row_block

    block = (
        '<details class="rwrap" data-rank="7" data-elo="1481.1" '
        'data-total_usd="5.00" data-out_tokens="50000">\n'
        '<summary><div class="row"><div class="cell cell-rank">7</div>'
        '<div class="cell entrant" title="moonshotai/kimi-k2.7-code · Moonshot AI">'
        '<span class="fdot" style="background:#fe9b61"></span>'
        '<span class="enames"><span class="nm">Kimi-K2.7-Code</span>'
        '<span class="pills"><span class="hpill">mini-swe-agent</span>'
        '<span class="org">Moonshot AI</span></span></span></div>'
        '<div class="cell chart"><div class="track">'
        '<div class="bar pos" style="left:0.000%;width:12.000%"></div></div>'
        '<div class="val"><span class="btv">1481.1</span>'
        '<span class="btci">[1400.0, 1560.0]</span></div></div>'
        '<div class="cell cnum cell-cost"><span class="cv">$5.00</span></div>'
        '<div class="cell cnum cell-tok"><span class="cv">50K</span></div>'
        '<div class="cell cell-chev">▸</div></div></summary>\n'
        '<div class="detail"><div class="dgrid">'
        '<section class="dcard"><h4>Match record</h4>'
        '<div class="dhead">STALE<span class="dsub">win · draw · loss</span></div>'
        "</section></div></div>\n</details>"
    )
    stats = _newcomer_stats(
        _B,
        [_match("m-1", _B, _A, "model_b_wins", _REPOS)],
        [_challenge("c-b", _B, "flask", cost=2.0)],
        [_defense("d-b", _B, "c-a", cost=0.75)],
    )
    diag = _StrengthDiag(
        draws=100,
        ranked=100,
        n_participants=3,
        rank_hist=[0.0, 0.85, 0.15],
        median_rank=2,
        rank_lo=2,
        rank_hi=3,
        bt=0.8,
        bt_lo=0.1,
        bt_hi=2.2,
        contested=0.05,
        contested_lo=-0.2,
        contested_hi=0.3,
        elo=1510.0,
        elo_lo=1495.0,
        elo_hi=1535.0,
    )
    entry = {
        "model": "anthropic/claude-opus-4.8",
        "harness": "mini-swe-agent",
        "reasoning_effort": "high",
        "provider": "(OpenRouter auto-route)",
        "elo": 1510.0,
        "points": 3.0,
        "matches_played": 9,
    }
    out = _refresh_row_block(
        block, 3, entry, stats, _Snap(), diag, 1484.0, 1600.0, 10.0, 100000
    )
    # Tag-balanced (an orphan closing </div> per cell was the original bug
    # that mangled the page layout).
    assert len(re.findall(r"<div\b", out)) == len(re.findall(r"</div>", out))
    assert len(re.findall(r"<span\b", out)) == len(re.findall(r"</span>", out))
    # Idempotent: a second pass over its own output is byte-identical, so
    # a re-application can never stack orphan tags.
    out2 = _refresh_row_block(
        out, 3, entry, stats, _Snap(), diag, 1484.0, 1600.0, 10.0, 100000
    )
    assert out2 == out
    # Identity visuals preserved minus the legacy dot (no late-entrant
    # marker creeps in); effort/provider appear only in the detail block.
    assert "fdot" not in out and "data-late-entrant" not in out
    assert 'class="hpill">mini-swe-agent<' in out
    summary_part, detail_part = out.split("</summary>", 1)
    assert "dident" not in summary_part
    assert detail_part.count('<div class="dident">') == 1
    # Attrs + core cells rewritten.
    assert 'data-rank="3"' in out and 'data-elo="1510.0"' in out
    # Per-match attrs: $3.00 spend + 2,550 tokens over the entry's 9 matches.
    assert 'data-usd_per_match="0.33"' in out
    assert 'data-tokens_per_match="283"' in out
    assert re.search(r'cell-rank">3<', out)
    assert re.search(r'btv">1510\.0<', out)
    # The 95% Elo CI badge refreshed in place.
    assert '<span class="btci">[1495.0, 1535.0]</span>' in out
    assert "[1400.0, 1560.0]" not in out
    # Cost/token cells normalized to the full shape (title + cbar added) and
    # carrying the PER-MATCH averages, not the cumulative totals.
    assert (
        '<div class="cell cnum cell-cost" '
        'title="$0.33 per match over 9 matches · '
        'generation $2.00 · validation $0.25 · match play $0.75">' in out
    )
    assert re.search(
        r'cell-cost"[^>]*><span class="cv">\$0\.33</span>'
        r'<span class="cbar"><i style="width:3\.3%"></i></span>',
        out,
    )
    assert re.search(
        r'cell-tok"[^>]*><span class="cv">283</span>'
        r'<span class="cbar"><i style="width:0\.3%"></i></span>',
        out,
    )
    # The per-match averages also surface in the regenerated cost card.
    assert "Cost per match</dt><dd>$0.33</dd>" in out
    assert "Tokens per match</dt><dd>283</dd>" in out
    # The detail block fully regenerated: no stale value survives.
    assert "STALE" not in out
    assert "League points</dt><dd>3.0</dd>" in out
    assert "<h4>Strength diagnostics</h4>" in out
    assert 'class="dhead">#2<span class="dsub">median bootstrap rank' in out
    assert "<h4>Bootstrap rank distribution</h4>" in out
    assert 'class="hb med in" title="rank 2: 85.0% of resamples"' in out
