#!/usr/bin/env python
"""Tournament rankings exporter (`swe-duel-rankings`).

Exports one tournament's ranking information from ``./data/`` into
``./rankings/rankings_<tournament_id>.json`` so `swe-duel-tournament-as`
can let new entrants join that tournament's field.

The export is self-contained — everything active sampling needs to place a
newcomer's matchups is in the file:

* ``rankings``: per-participant standings — tournament points, matches
  played, match-level Bradley-Terry + Elo (with ranks), TrueSkill μ/σ —
  keyed by the participants' pinned identities (``model``, ``harness``,
  ``reasoning_effort`` = the pinned effort or null, ``provider`` = the
  pinned slug or ``(OpenRouter auto-route)``);
* ``head_to_head``: W/L/D records per participant pair (with match ids);
* ``matches``: the raw claimed match list (id, endpoints, outcome,
  timestamp, repos) — enough to reconstruct the tournament's match history
  for pair-gain/already-played bookkeeping even without local
  ``data/matches`` files;
* ``repos`` + ``targets_per_repo``: the exact arena the tournament ran —
  `swe-duel-tournament-as` enforces both for later entrants so nobody can
  bias the field by running a different repo set or challenge count.

Data sources: the tournament's state file (``round_robin_state_<id>.json``
or ``swiss_state_<id>.json`` under ``data/tournaments/`` — standings, repo
set, participant identities) and the matches it claimed in
``data/matches`` (the same claim pipeline `swe-duel-report` uses). The
tournament id positions/challenges/repo set are *not* reconstructed from
elsewhere — export exactly what that tournament ran.

``export_rankings(..., extra_matches=...)`` folds externally supplied match
payloads (imported ``swe-duel-tournament-update`` contributions) into the
export: their endpoints join the field, BT/Elo/TrueSkill and head-to-head are
computed over the merged match list, and their win/draw points are added on
top of the state scoreboard. This is how a tournament's rankings absorb
newcomer matches run by external contributors via
``swe-duel-tournament-as``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from swe_duel.cli._common import resolve_config_dir, resolve_output_dir
from swe_duel.cli.build_report import (
    _claim_matches_per_tournament,
    _dedupe,
    _load_matches,
    _load_round_robin_states,
    _load_swiss_states,
    _to_match_result,
)
from swe_duel.config import load_arena_config
from swe_duel.models import MatchResult, split_composite_id
from swe_duel.scoring.bootstrap import bootstrap_field_diagnostics
from swe_duel.scoring.rating import compute_all_ratings

_STATE_PREFIXES = ("round_robin_state_", "swiss_state_")


def _find_state_file(tournaments_dir: Path, tournament_id: str) -> Path:
    """Locate the state file for ``tournament_id`` (fail fast, list options)."""
    for prefix in _STATE_PREFIXES:
        cand = tournaments_dir / f"{prefix}{tournament_id}.json"
        if cand.is_file():
            return cand
    available = sorted(
        {
            p.name[len(prefix): -len(".json")]
            for p in tournaments_dir.glob("*.json")
            for prefix in _STATE_PREFIXES
            if p.name.startswith(prefix)
        }
    )
    raise SystemExit(
        f"no tournament state for id {tournament_id!r} under "
        f"{tournaments_dir} (looked for "
        + " / ".join(f"{p}{tournament_id}.json" for p in _STATE_PREFIXES)
        + ").\n"
        + (
            f"tournaments with state on disk: {available}"
            if available
            else "no round_robin_state_* / swiss_state_* files exist yet."
        )
    )


def _points_from_matches(
    claimed: list[dict[str, Any]],
) -> dict[str, float]:
    """Fallback standings (win=1, draw=0.5 each, loss=0) when the state
    carries none."""
    pts: dict[str, float] = defaultdict(float)
    from swe_duel.models import MatchOutcome

    for m in claimed:
        a, b = m["model_a_id"], m["model_b_id"]
        out = MatchOutcome(str(m.get("outcome") or "draw"))
        if out == MatchOutcome.MODEL_A_WINS:
            pts[a] += 1.0
        elif out == MatchOutcome.MODEL_B_WINS:
            pts[b] += 1.0
        else:
            pts[a] += 0.5
            pts[b] += 0.5
    return dict(pts)


def _targets_per_repo(claimed: list[dict[str, Any]]) -> int:
    """Challenge slots per (participant, repo) the tournament ran.

    Reconstructed from the claimed matches' turn records: per match, the
    maximum number of turns any (red participant, repo) pair contributed —
    exactly the match's ``turns_per_player`` even with dropped sub-turns or
    missing-Red auto-wins. All matches must agree; if they do not, the
    median is used with a warning.
    """
    per_match: list[int] = []
    for m in claimed:
        counts: Counter[tuple[str, str]] = Counter()
        for t in m.get("turns") or []:
            red = str(t.get("red_model_id") or "")
            repo = str(t.get("repo_name") or "")
            if red and repo:
                counts[(red, repo)] += 1
        per_match.append(max(counts.values()) if counts else 0)
    if not per_match or all(v == 0 for v in per_match):
        raise SystemExit("claimed matches carry no turns — cannot determine targets_per_repo")
    vals = sorted(per_match)
    mid = vals[len(vals) // 2]
    if len(set(per_match)) > 1:
        print(
            f"[warn] matches disagree on turns-per-player ({dict(Counter(per_match))}); "
            f"using the median {mid}",
            file=sys.stderr,
        )
    return mid


def _rank_by(value_map: dict[str, float]) -> dict[str, int]:
    """Dense descending rank per key (ties share a rank)."""
    order = sorted(set(value_map.values()), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of[v] for k, v in value_map.items()}


def _identity_columns(cid: str) -> dict[str, Any]:
    """Split a pinned composite id into the export's identity columns."""
    model, harness, effort, provider = split_composite_id(cid)
    return {
        "model": model,
        "harness": harness,
        # null = no effort pin (model default); provider column keeps the
        # human-readable auto-route marker, empty = OpenRouter default.
        "reasoning_effort": effort or None,
        "provider": provider or "(OpenRouter auto-route)",
    }


def _as_entered_tournament(state: dict[str, Any]) -> str:
    """Which tournament an active-sampling state entered.

    ``swe-duel-tournament-as`` records ``entered_tournament`` when invoked
    with ``--tournament <id>``; legacy states instead carry the rankings
    export path they loaded (``rankings_source``) — the ``rankings_<id>.json``
    stem names the same tournament.
    """
    entered = state.get("entered_tournament")
    if isinstance(entered, str) and entered:
        return entered
    source = state.get("rankings_source")
    if isinstance(source, str) and source:
        stem = PurePosixPath(source.replace("\\", "/")).name
        if stem.startswith("rankings_") and stem.endswith(".json"):
            return stem[len("rankings_") : -len(".json")]
    return ""


def _load_as_extra_matches(
    tournaments_dir: Path, matches_dir: Path, tournament_id: str
) -> list[dict[str, Any]]:
    """Match payloads run by active-sampling states that entered this tournament.

    Scans ``active_sampling_state_*.json`` files whose entered tournament is
    ``tournament_id`` and returns the match payloads those runs produced:
    every match file referenced by a state's ``results`` plus every match
    claimed by walking its ``pairings`` (the same claim pipeline the round
    robin / Swiss exports use). Deduplicated by ``match_id``. Missing match
    files (a sampled-but-declined or interrupted run) are skipped silently —
    a match that never ran contributes nothing.
    """
    if not tournaments_dir.is_dir():
        return []
    selected: list[dict[str, Any]] = []
    for path in sorted(tournaments_dir.glob("active_sampling_state_*.json")):
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(state, dict) or state.get("format") != "active_sampling":
            continue
        if _as_entered_tournament(state) != tournament_id:
            continue
        state["_as_state_path"] = str(path)
        selected.append(state)
    if not selected:
        return []

    raw = _load_matches(matches_dir) if matches_dir.is_dir() else []
    if not raw:
        return []
    kept, _dropped = _dedupe(raw)
    by_id = {str(m.get("match_id") or ""): m for m in kept}
    # Walk each state's pairings through the standard claim pipeline so a
    # match is attributed to the state that actually scheduled it.
    pseudo_states = [
        {
            "tournament_id": str(st.get("tournament_id") or st["_as_state_path"]),
            "repo_names": list(st.get("repo_names") or []),
            "rounds": [list(st.get("pairings") or [])],
            "timestamp": st.get("timestamp"),
        }
        for st in selected
    ]
    claimed = _claim_matches_per_tournament(pseudo_states, kept)
    ids: set[str] = set()
    for matches in claimed.values():
        for m in matches:
            mid = str(m.get("match_id") or "")
            if mid:
                ids.add(mid)
    for st in selected:
        for r in st.get("results") or []:
            if not isinstance(r, dict):
                continue
            mid = str(r.get("match_id") or "")
            if mid in by_id:
                ids.add(mid)
    return [by_id[mid] for mid in sorted(ids) if mid in by_id]


def export_rankings(
    tournament_id: str,
    data_dir: Path,
    rankings_dir: Path,
    *,
    stdout: Any = sys.stdout,
    extra_matches: list[dict[str, Any]] | None = None,
) -> Path:
    """Build and write ``rankings_<tournament_id>.json``; return its path.

    ``extra_matches`` folds external match payloads (e.g. the active-sampled
    newcomer-vs-incumbent matches an imported contribution ran via
    ``swe-duel-tournament-as``) into the export. They are de-duplicated
    against the tournament's claimed matches by ``match_id`` (claimed local
    matches win), their win/draw points are ADDED on top of the state's
    authoritative scoreboard, and their endpoints (the newcomers) join the
    field with full BT/Elo/TrueSkill standings — so a later
    ``swe-duel-tournament-as`` run sees the newcomers and the merged
    head-to-head history.
    """
    tournaments_dir = data_dir / "tournaments"
    matches_dir = data_dir / "matches"
    state_path = _find_state_file(tournaments_dir, tournament_id)
    state = json.loads(state_path.read_text())

    raw = _load_matches(matches_dir)
    kept, _dropped = _dedupe(raw)
    all_states = _load_round_robin_states(tournaments_dir) + _load_swiss_states(tournaments_dir)
    claimed = _claim_matches_per_tournament(all_states, kept).get(tournament_id, [])
    if not claimed:
        raise SystemExit(
            f"tournament {tournament_id} claimed no matches in {matches_dir} — "
            "export rankings only for tournaments that have run matches."
        )
    claimed.sort(key=lambda m: str(m.get("timestamp") or ""))

    # Standings: the state file's points are the authoritative scoreboard;
    # fall back to recomputing from the claimed match outcomes.
    players: list[dict[str, Any]] = [
        p for p in (state.get("players") or []) if isinstance(p, dict)
    ]
    points: dict[str, float] = {
        str(p.get("model_id")): float(p.get("points", 0.0)) for p in players
    }
    participants: list[str] = list(points)
    if not participants:
        points = _points_from_matches(claimed)
        participants = list(points)

    # Fold external matches (imported contributions) into the claimed list:
    # dedupe by match_id (claimed local matches win) and add their win/draw
    # points ON TOP of the scoreboard above — the state points already cover
    # the originally claimed matches, so only the extras contribute here.
    # The active-sampling matches already on disk (from imported
    # contributions or a local swe-duel-tournament-as run) fold in
    # automatically, so re-exporting a tournament never silently drops the
    # late entrants again.
    extras = list(extra_matches or [])
    extras.extend(
        _load_as_extra_matches(tournaments_dir, matches_dir, tournament_id)
    )
    n_extra = 0
    if extras:
        claimed_ids = {str(m.get("match_id") or "") for m in claimed}
        accepted: list[dict[str, Any]] = []
        for m in extras:
            mid = str(m.get("match_id") or "")
            if mid and mid in claimed_ids:
                continue
            claimed_ids.add(mid)
            accepted.append(m)
        n_extra = len(accepted)
        for cid, pts in _points_from_matches(accepted).items():
            points[cid] = points.get(cid, 0.0) + pts
        claimed.extend(accepted)
        claimed.sort(key=lambda m: str(m.get("timestamp") or ""))

    for m in claimed:
        for side in ("model_a_id", "model_b_id"):
            participants.append(str(m[side]))
    field: list[str] = []
    seen: set[str] = set()
    for p in participants:
        if p not in seen:
            seen.add(p)
            field.append(p)

    # _ShimMatch structurally mirrors the MatchResult fields the rating
    # pipeline reads (same shim build_report uses).
    shims = [_to_match_result(m) for m in claimed]
    snaps = compute_all_ratings(
        cast("list[MatchResult]", shims), model_ids=field
    )

    pts_rank = _rank_by({cid: points.get(cid, 0.0) for cid in field})
    bt_rank = _rank_by({cid: s.bradley_terry for cid, s in snaps.items()})
    elo_rank = _rank_by({cid: s.elo for cid, s in snaps.items()})

    # Seeded bootstrap over the merged history (fixed seed → re-exports are
    # byte-identical): every entry carries its 95% Elo CI, match-level BT
    # CI, contested-turn BT point + CI, and the full bootstrap rank
    # distribution — the uncertainty numbers the leaderboard renders and
    # tournament-update reuses so the page and the JSON always agree.
    diags = bootstrap_field_diagnostics(claimed)

    rankings = []
    for cid in sorted(field, key=lambda c: (-points.get(c, 0.0), field.index(c))):
        s = snaps[cid]
        rankings.append(
            {
                "rank": pts_rank[cid],
                **_identity_columns(cid),
                "points": points.get(cid, 0.0),
                "matches_played": int(s.matches_played),
                "bradley_terry": round(float(s.bradley_terry), 3),
                "bradley_terry_rank": bt_rank[cid],
                "elo": round(float(s.elo), 1),
                "elo_rank": elo_rank[cid],
                "trueskill_mu": round(float(s.trueskill_mu), 2),
                "trueskill_sigma": round(float(s.trueskill_sigma), 2),
                **(
                    {
                        "elo_ci": [round(d.elo_lo, 1), round(d.elo_hi, 1)],
                        "bradley_terry_ci": [round(d.bt_lo, 3), round(d.bt_hi, 3)],
                        "contested_bt": round(d.contested, 3),
                        "contested_bt_ci": [
                            round(d.contested_lo, 3),
                            round(d.contested_hi, 3),
                        ],
                        "bootstrap_draws": d.draws,
                        "bootstrap_ranked": d.ranked,
                        "bootstrap_median_rank": d.median_rank,
                        "bootstrap_rank_interval": [d.rank_lo, d.rank_hi],
                        "bootstrap_rank_histogram": [round(s_, 4) for s_ in d.rank_hist],
                    }
                    if (d := diags.get(cid)) is not None
                    else {}
                ),
            }
        )

    matches_out = [
        {
            "match_id": str(m["match_id"]),
            "model_a": str(m["model_a_id"]),
            "model_b": str(m["model_b_id"]),
            "outcome": str(m.get("outcome") or "draw"),
            "timestamp": str(m.get("timestamp") or ""),
            "repo_name": str(m.get("repo_name") or ""),
        }
        for m in claimed
    ]

    head_to_head: dict[tuple[str, str], dict[str, Any]] = {}
    for m in claimed:
        a, b = sorted((str(m["model_a_id"]), str(m["model_b_id"])))
        outcome = str(m.get("outcome") or "draw")
        rec = head_to_head.setdefault(
            (a, b),
            {"a": a, "b": b, "a_wins": 0, "b_wins": 0, "draws": 0, "match_ids": []},
        )
        if outcome == "model_a_wins" and m["model_a_id"] == a:
            rec["a_wins"] += 1
        elif outcome == "model_b_wins" and m["model_b_id"] == a:
            rec["a_wins"] += 1
        elif outcome == "model_a_wins":
            rec["b_wins"] += 1
        elif outcome == "model_b_wins":
            rec["b_wins"] += 1
        else:
            rec["draws"] += 1
        rec["match_ids"].append(str(m["match_id"]))

    repos = [str(r) for r in (state.get("repo_names") or [])]
    if not repos:
        for m in claimed:
            for t in m.get("turns") or []:
                r = str(t.get("repo_name") or "")
                if r and r not in repos:
                    repos.append(r)

    source = f"{state_path} + {len(claimed)} claimed match(es) from {matches_dir}"
    if n_extra:
        source += f" (+{n_extra} imported contribution match(es))"
    payload = {
        "tournament_id": tournament_id,
        "format": str(state.get("format") or ""),
        "source": source,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "repos": repos,
        "targets_per_repo": _targets_per_repo(claimed),
        "rankings": rankings,
        "head_to_head": sorted(
            head_to_head.values(), key=lambda r: (r["a"], r["b"])
        ),
        "matches": matches_out,
    }

    rankings_dir.mkdir(parents=True, exist_ok=True)
    out_path = rankings_dir / f"rankings_{tournament_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"exported {len(rankings)} participant(s) from tournament "
        f"{tournament_id} ({len(claimed)} matches, repos={len(repos)}, "
        f"targets_per_repo={payload['targets_per_repo']}) -> {out_path}",
        file=stdout,
    )
    for e in rankings:
        print(
            f"  {e['rank']:>2}. {e['model']} [{e['harness']}]"
            f"{' effort=' + e['reasoning_effort'] if e['reasoning_effort'] else ''}"
            f"{' provider=' + e['provider'] if e['provider'] != '(OpenRouter auto-route)' else ''}"
            f"  pts={e['points']:.1f}  matches={e['matches_played']}"
            f"  BT={e['bradley_terry']}  Elo={e['elo']}",
            file=stdout,
        )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swe-duel-rankings",
        description=(
            "Export one tournament's rankings from ./data/ to "
            "./rankings/rankings_<tournament_id>.json (consumed by "
            "swe-duel-tournament-as --tournament <id>)."
        ),
    )
    parser.add_argument("tournament_id", help="tournament id to export")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="config directory (arena.yaml paths.output_dir resolves the data dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="data directory (default: arena.yaml paths.output_dir, i.e. ./data)",
    )
    parser.add_argument(
        "--rankings-dir",
        default="./rankings",
        help="where rankings_<tournament_id>.json is written (default ./rankings)",
    )
    args = parser.parse_args(argv)

    arena_config = load_arena_config(resolve_config_dir(args))
    data_dir = resolve_output_dir(args, arena_config)
    export_rankings(
        args.tournament_id, data_dir, Path(args.rankings_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())