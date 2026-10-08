#!/usr/bin/env python
"""Interactive round-robin tournament console.

Mirrors ``swe_duel/cli/run_tournament.py`` (the Swiss-system REPL) but uses a
**round-robin** schedule: every selected competitor plays every other
competitor exactly once. The full pairing table is fixed at construction time
(circle method), so round R is always the same set of pairings regardless of
intermediate outcomes — there is no re-pairing step and no rematch-avoidance
logic (each pair meets exactly once by construction).

A tournament runs over one or more repos (``--repos``); a single match between
two competitors spans ALL selected repos (``turns_per_player`` turns per repo
per side, aggregated into one ``MatchResult``). A competitor that lacks a
challenge for some repo is still played — the opponent auto-wins those turns.

Features carried over from ``run_tournament.py``:
  * interactive paged selection of (model, harness, reasoning_effort,
    provider) competitors — one page per dimension, then a landing page;
  * per-identity slot/success table printed before selection;
  * prompt for the round-1 ordering seed (used only to shuffle the initial
    circle ordering — it does NOT affect which pairs meet, only the order in
    which rounds surface them);
  * resume detection against ``round_robin_state_*.json`` for the same repo
    set, reconstructing prior ``MatchResult`` objects from ``data/matches/``;
  * post-resume "add participant" flow: seat eligible newcomers and onboard
    them with Chatbot-Arena active sampling (SE-reduction) so each newcomer
    faces only K≈ceil(log2 N)+1 incumbents rather than a full re-round-robin;
  * per-sub-turn defense-cache reuse: each (challenge, blue_model,
    blue_harness) defense JSON from a prior tournament (Swiss OR round-robin)
    is reused at zero cost;
  * parallel Blue agents via one ``ThreadPoolExecutor`` per round with a live
    ``rich`` progress tree (``DefenseRoundReporter``);
  * REPL: ``status``, ``standings``, ``history [N]``, ``pairings``, ``next``,
    ``all``, ``report``, ``quit`` (final report generated on exit).

REPL commands (history N takes an argument):
    status     (s)   tournament overview
    standings  (ls)  cumulative points + ELO leaderboard (with round counter)
    history    (h)   list past rounds with pairings + outcomes
    history N        ELO table as it stood after round N
    pairings   (p)   preview next round's pairings with per-pairing reuse counts
    next       (n)   run the next round
    all        (a)   run all remaining matches in one parallel batch
                     (circle rounds do not gate each other; intake included)
    report     (r)   regenerate figures via build_report.py
    quit/q           exit (final report generated on the way out)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import questionary

from swe_duel.cli._common import install_container_cleanup_handlers
from swe_duel.cli._common import default_prompt_dir
from swe_duel.cli._common import setup
from swe_duel.agents.harness.cost_tracking import warn_missing_pricing
from swe_duel.agents.harness.rate_limit import (
    summarize_rate_limits,
    warn_unenforceable_rate_limits,
)
from swe_duel.engine.match import (
    DefenseTask,
    MatchOrchestrator,
    MatchPlan,
)
from swe_duel.engine.round_progress import (
    DefenseRoundReporter,
    Status as SubturnStatus,
    live as round_live,
)
from swe_duel.engine.round_robin import RoundRobinTournament
from swe_duel.engine.swiss import SwissPairing
from swe_duel.models import MatchResult
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.rating import compute_all_ratings

# Reuse the interactive-selection, slot/success-count, validation, and
# resume/reconstruction helpers verbatim from the Swiss tournament script —
# they are repo/competitor-shaped, not pairing-algorithm-shaped.
from swe_duel.cli.run_tournament import (
    _load_match_results_for_pairs,
    _merge_model_configs_for_ids,
    _outcome_score_a,
    _print_slot_table,
    _prompt_intake_budget,
    _prompt_seed,
    _select_addable_competitors,
    _select_models_interactive,
    _short,
    _slot_counts,
    _success_counts,
    _validate_pools,
)


# ── helpers ───────────────────────────────────────────────────


def _persist_round_robin_state(
    path: Path,
    tournament_id: str,
    rr: RoundRobinTournament,
    repo_names: list[str],
    selected_model_ids: list[str],
    seed: int,
) -> None:
    payload = {
        "tournament_id": tournament_id,
        "format": "round_robin",
        "repo_name": ",".join(repo_names),
        "repo_names": list(repo_names),
        "selected_model_ids": selected_model_ids,
        # Base field the circle schedule was built over. Late entrants stay in
        # ``intake_players`` so resume rebuilds the same base schedule.
        "base_model_ids": list(rr.base_model_ids),
        "intake_players": list(rr.intake_players),
        "intake_budget": {k: int(v) for k, v in rr.intake_budget.items()},
        "seed": seed,
        "current_round_index": rr.current_round_index,
        "players": [
            {
                "model_id": p.model_id,
                "points": p.points,
                "opponents": list(p.opponents),
                "had_bye": p.had_bye,
            }
            for p in rr.players.values()
        ],
        "rounds": [
            [
                {"model_a": pp.model_a, "model_b": pp.model_b}
                for pp in rnd
            ]
            for rnd in rr.round_history
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _find_resume_states_rr(
    tournaments_dir: Path, repo_names: list[str]
) -> list[tuple[Path, dict]]:
    """Round-robin resume states. Filters on ``format == "round_robin"``.

    Falls back to scanning ``round_robin_state_*.json`` files (separate
    namespace from the Swiss ``swiss_state_*.json`` files).
    """
    want = set(repo_names)
    candidates = sorted(
        tournaments_dir.glob("round_robin_state_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    results: list[tuple[Path, dict]] = []
    for p in candidates:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        repos = payload.get("repo_names")
        if not repos:
            rn = payload.get("repo_name", "")
            repos = [r for r in rn.split(",") if r] if rn else []
        if set(repos) == want:
            results.append((p, payload))
    return results


def _reconstruct_round_robin(
    state_path: Path, model_ids: list[str], seed: int
) -> tuple[RoundRobinTournament, str, list[str]]:
    from swe_duel.engine.swiss import SwissPlayer

    payload = json.loads(state_path.read_text())
    # Rebuild the circle schedule from the *base* field only. Late entrants
    # (``intake_players``) must not disturb the original pairing table.
    base_ids = list(payload.get("base_model_ids") or [])
    intake_ids = list(payload.get("intake_players") or [])
    if not base_ids:
        # Legacy state: everything in selected_model_ids was part of the base
        # field; strip known intake ids if present.
        base_ids = [m for m in model_ids if m not in set(intake_ids)]
        if not base_ids:
            base_ids = list(model_ids)

    rr = RoundRobinTournament(base_ids, seed=seed)
    # Restore late entrants + residual budgets *before* replaying player
    # bookkeeping so player dicts exist for every id that has points/opponents.
    residual = payload.get("intake_budget") or {}
    if intake_ids:
        rr.add_players(
            intake_ids,
            remaining_budget={
                mid: int(residual.get(mid, 0)) for mid in intake_ids
            },
        )
        # Preserve the order from the state file.
        rr.intake_players = [m for m in intake_ids if m in rr.players]

    for p in payload.get("players", []):
        mid = p["model_id"]
        if mid not in rr.players:
            # Defensive: seat any stray player recorded in the points table.
            rr.players[mid] = SwissPlayer(model_id=mid)
        pl = rr.players[mid]
        pl.points = p.get("points", 0.0)
        pl.opponents = list(p.get("opponents", []))
        pl.had_bye = p.get("had_bye", False)

    rr.round_history = [
        [SwissPairing(model_a=pp["model_a"], model_b=pp["model_b"]) for pp in rnd]
        for rnd in payload.get("rounds", [])
    ]
    tournament_id = payload.get("tournament_id", str(uuid.uuid4()))
    selected = payload.get("selected_model_ids", model_ids)
    return rr, tournament_id, selected


def _maybe_add_participants_rr(
    *,
    rr: RoundRobinTournament,
    selected_ids: list[str],
    all_model_configs: dict,
    slot_counts: dict[str, int],
    success_counts: dict[str, int],
    repo_names: list[str],
    turns_per_player: int,
    challenge_store,
    tournament_id: str,
    state_path: Path,
    seed: int,
    prior_matches: list[MatchResult],
) -> list[str]:
    """Post-resume prompt: onboard newcomers via active-sampling intake.

    Newcomers do **not** rebuild the circle schedule. Each is given a residual
    match-budget (default ``ceil(log2 N)+1`` against the incumbent field) and
    subsequent ``next`` rounds pair them against the yet-unfaced opponents that
    most reduce win-matrix CI width (Chatbot Arena eq. 9).
    """
    from swe_duel.models import display_composite_id
    from swe_duel.scoring.active_sampling import default_intake_budget

    add = questionary.confirm(
        "Add a new participant to this tournament?",
        default=False,
    ).ask()
    if not add:
        return selected_ids

    _print_slot_table(
        challenge_store, all_model_configs, repo_names, turns_per_player
    )
    newcomers = _select_addable_competitors(
        all_model_configs,
        slot_counts,
        success_counts,
        len(repo_names),
        already_selected=set(selected_ids),
    )
    if not newcomers:
        print("  no newcomers selected.")
        return selected_ids

    _validate_pools(newcomers, slot_counts, require_even=False, min_count=1)
    if not rr.base_schedule_complete:
        print(
            "\n  Note: base circle rounds still remaining — newcomers sit out "
            "until the original schedule finishes, then enter active-sampling "
            "intake via `next`.\n"
        )
    n_incumbents = len(rr.players)
    default_k = default_intake_budget(n_incumbents)
    print(
        f"\n  Active sampling will pick up to K opponents per newcomer from "
        f"the {n_incumbents} incumbents (Chatbot Arena SE-reduction rule) — "
        f"not a full re-round-robin.\n"
        f"  Suggested K = {default_k} (= ceil(log2 N)+1, capped at N).\n"
    )
    k = _prompt_intake_budget(n_incumbents)
    added = rr.add_players(newcomers, matches_per_newcomer=k)
    if not added:
        print("  (all selected competitors were already seated)")
        return selected_ids

    # Seed intake ranking context so the first intake pairings see the full
    # incumbent win matrix (variance tie-breaks use TrueSkill σ).
    secondary: dict[str, float] = {}
    if prior_matches:
        snaps = compute_all_ratings(prior_matches)
        secondary = {m: s.trueskill_sigma for m, s in snaps.items()}
    rr.set_intake_context(matches=prior_matches, secondary_score=secondary)

    extended = list(selected_ids) + [c for c in added if c not in selected_ids]
    print("\n  Added newcomers (active-sampling intake):")
    for cid in added:
        budget = rr.intake_budget.get(cid, 0)
        print(f"    + {display_composite_id(cid)}  (budget={budget} matches)")
    n_intake_matches = sum(rr.intake_budget.values())
    print(
        f"  Field size {len(selected_ids)} → {len(extended)}. "
        f"One `next` intake round will run {n_intake_matches} match(es) "
        f"(full residual budget, active-sampled opponents).\n"
    )
    _persist_round_robin_state(
        state_path, tournament_id, rr, repo_names, extended, seed
    )
    return extended


# ── REPL ──────────────────────────────────────────────────────



class RoundRobinConsole:
    def __init__(
        self,
        ctx: dict,
        match_orchestrator: MatchOrchestrator,
        repo_names: list[str],
        seed: int,
        tournament_id: str,
        selected_model_ids: list[str],
        rr: RoundRobinTournament,
        resumed: bool,
        prior_matches: list[MatchResult],
        match_round: dict[str, int],
    ) -> None:
        self.ctx = ctx
        self.orchestrator = match_orchestrator
        self.repo_names = list(repo_names)
        self.repo_cfgs = [ctx["repo_configs"][rn] for rn in repo_names]
        self.repo_label = ",".join(repo_names)
        self.seed = seed
        self.tournament_id = tournament_id
        self.selected_model_ids = selected_model_ids
        self.matches: list[MatchResult] = list(prior_matches)
        self.match_round: dict[str, int] = dict(match_round)
        self.resumed = resumed
        self.rr = rr
        self.pending_pairings: list[SwissPairing] | None = None
        self.state_path = (
            ctx["artifact_logger"].tournaments_dir
            / f"round_robin_state_{self.tournament_id}.json"
        )

    # ── command handlers ──────────────────────────────────────

    def cmd_status(self) -> None:
        n = len(self.selected_model_ids)
        rec = self.rr.recommended_rounds
        base_rounds = len(self.rr._schedule)
        total_pairs = n * (n - 1) // 2 if n > 1 else 0
        print()
        print(f"  Tournament ID : {self.tournament_id}")
        print("  Format        : round-robin")
        print(f"  Repos         : {self.repo_label}")
        print(f"  Models        : {n}")
        if self.rr.intake_players:
            budgets = ", ".join(
                f"{_short(m, 28)}={self.rr.intake_budget.get(m, 0)}"
                for m in self.rr.intake_players
            )
            print(f"  Intake        : {len(self.rr.intake_players)} newcomer(s)"
                  f"  residual budgets [{budgets}]")
            print(
                f"  Base pairs    : C({len(self.rr.base_model_ids)},2) pathway done"
                if self.rr.base_schedule_complete
                else f"  Base rounds   : {self.rr.current_round_index}/{base_rounds}"
            )
        else:
            print(f"  Total pairs   : {total_pairs} (C(N,2))")
        print(f"  Recommended   : {rec} rounds")
        print(f"  Current round : {self.rr.current_round_index} of {rec}")
        print(f"  Matches run   : {len(self.matches)}")
        print(f"  Seed          : {self.seed}")
        print(
            f"  Total cost    : ${sum(m.total_cost_usd for m in self.matches):.4f}"
        )
        print()

    def cmd_standings(self) -> None:
        completed = self.rr.current_round_index
        rec = self.rr.recommended_rounds
        std = self.rr.standings()
        elo_map = self._elo_snapshot(self.matches)
        print()
        print(
            f"  ── Round {completed} of {rec} complete — "
            f"Round {completed + 1} next ──"
        )
        print(
            f"  {'#':>2}  {'Model':<42}  {'Pts':>6}  {'ELO':>7}  {'Played':>6}"
        )
        print(f"  {'-' * 2}  {'-' * 42}  {'-' * 6}  {'-' * 7}  {'-' * 6}")
        for i, p in enumerate(std, 1):
            elo = elo_map.get(p.model_id, 1500.0)
            played = len(p.opponents) + (1 if p.had_bye else 0)
            print(
                f"  {i:>2}  {_short(p.model_id, 42):<42}  "
                f"{p.points:>6.1f}  {elo:>7.1f}  {played:>6}"
            )
        print()

    def cmd_history(self, arg: str = "") -> None:
        if arg.strip():
            try:
                n = int(arg)
            except ValueError:
                print("  usage: history <round_number>")
                return
            prefix = [
                m for m in self.matches
                if self.match_round.get(m.match_id, 10**9) <= n
            ]
            if not prefix:
                print(f"  no matches recorded for rounds 1..{n}")
                return
            elo_map = self._elo_snapshot(prefix)
            print(f"\n  ── ELO table after Round {n} ──")
            print(f"  {'#':>2}  {'Model':<42}  {'ELO':>7}  {'Played':>6}")
            print(f"  {'-' * 2}  {'-' * 42}  {'-' * 7}  {'-' * 6}")
            played_counts: dict[str, int] = {}
            for m in prefix:
                played_counts[m.model_a_id] = played_counts.get(m.model_a_id, 0) + 1
                played_counts[m.model_b_id] = played_counts.get(m.model_b_id, 0) + 1
            for i, (model_id, elo) in enumerate(
                sorted(elo_map.items(), key=lambda kv: -kv[1]), 1
            ):
                print(
                    f"  {i:>2}  {_short(model_id, 42):<42}  "
                    f"{elo:>7.1f}  {played_counts.get(model_id, 0):>6}"
                )
            print()
            return

        if not self.matches:
            print("  no rounds played yet.")
            return
        by_round: dict[int, list[MatchResult]] = {}
        for m in self.matches:
            r = self.match_round.get(m.match_id)
            if r is None:
                continue
            by_round.setdefault(r, []).append(m)
        for r in sorted(by_round):
            print(f"\n  Round {r}:")
            for m in by_round[r]:
                print(
                    f"    {_short(m.model_a_id, 38):<38}  vs  "
                    f"{_short(m.model_b_id, 38):<38}  → {m.outcome.value:<14}  "
                    f"A={m.model_a_total:.2f} B={m.model_b_total:.2f}  "
                    f"${m.total_cost_usd:.4f}"
                )
        print()

    def cmd_pairings(self) -> None:
        try:
            pairings = self._ensure_pending_pairings()
        except RuntimeError as e:
            print(f"\n  [tournament complete] {e}\n")
            return
        rnd = self.rr.current_round_index + 1
        intake = (
            self.rr.base_schedule_complete and bool(self.rr.intake_players)
        )
        label = "active-sampling intake" if intake else "circle"
        print()
        print(f"  Round {rnd} pairings ({len(pairings)} · {label}):")
        playable = [p for p in pairings if p.model_b is not None]
        counts = {id(p): self._turn_reuse_counts(p) for p in playable}
        for i, p in enumerate(pairings, 1):
            if p.model_b is None:
                # Bye round for model_a (odd-N round-robin only).
                print(
                    f"   {i:>2}. {_short(p.model_a):<42}  — BYE (sits out)"
                )
                continue
            a_pts = self.rr.players[p.model_a].points
            b_pts = self.rr.players[p.model_b].points
            reuse, new = counts[id(p)]
            print(
                f"   {i:>2}. {_short(p.model_a):<42}  ({a_pts:.1f})  "
                f"vs  {_short(p.model_b):<42}  ({b_pts:.1f})  "
                f"— {reuse} reusable / {new} new"
            )
        total_reuse = sum(r for r, _ in counts.values())
        total_new = sum(n for _, n in counts.values())
        print(
            f"\n  Totals: {total_reuse} reusable / {total_new} new "
            f"({total_reuse + total_new} turns this round)"
        )
        print()

    def cmd_next(self) -> None:
        try:
            pairings = self._ensure_pending_pairings()
        except RuntimeError as e:
            print(f"\n  [tournament complete] {e}\n")
            return
        rnd = self.rr.current_round_index + 1
        self.cmd_pairings()
        playable = [p for p in pairings if p.model_b is not None]
        if not playable:
            # Round is all byes (only possible with 1 player — shouldn't happen
            # because _validate_pools requires ≥2). Commit and move on.
            self.rr.commit_round(pairings)
            self.pending_pairings = None
            _persist_round_robin_state(
                self.state_path, self.tournament_id, self.rr,
                self.repo_names, self.selected_model_ids, self.seed,
            )
            print(f"\n[round {rnd}] no playable pairings; round committed.\n")
            return
        confirm = questionary.confirm(
            f"Run Round {rnd} ({len(playable)} matches)?", default=True
        ).ask()
        if not confirm:
            print("  cancelled.\n")
            return

        print(f"\n[round {rnd}] running {len(playable)} pairings...\n")
        seed_base = self.seed + 1000 * rnd
        executed = self._run_round_parallel(playable, rnd, seed_base)
        for p, m in executed:
            if m is None:
                continue
            self.matches.append(m)
            self.match_round[m.match_id] = rnd

        self.rr.commit_round(pairings)
        for pairing, result in executed:
            if result is None:
                continue
            sa = _outcome_score_a(result.outcome)
            self.rr.record_result(pairing.model_a, pairing.model_b, sa)

        self.pending_pairings = None
        _persist_round_robin_state(
            self.state_path,
            self.tournament_id,
            self.rr,
            self.repo_names,
            self.selected_model_ids,
            self.seed,
        )
        print(f"\n[round {rnd}] done. State → {self.state_path}\n")

    def cmd_all(self) -> None:
        """Run every remaining circle (and pending intake) match in one pool.

        Round-robin base pairings are independent across rounds, so unlike Swiss
        there is no reason to wait for round R standings before starting R+1.
        Batches are still committed in schedule order so history/resume keep
        their per-round shape.
        """
        try:
            batches = self._collect_remaining_batches()
        except RuntimeError as e:
            print(f"\n  [tournament complete] {e}\n")
            return

        # Flatten playable pairings while remembering schedule round numbers.
        scheduled: list[tuple[int, SwissPairing]] = []
        intake_ids = set(self.rr.intake_players)
        print()
        for rnd, pairings in batches:
            is_intake = bool(intake_ids) and any(
                p.model_a in intake_ids
                or (p.model_b is not None and p.model_b in intake_ids)
                for p in pairings
            )
            label = "active-sampling intake" if is_intake else "circle"
            playable = [p for p in pairings if p.model_b is not None]
            print(
                f"  Round {rnd} ({len(pairings)} · {label}"
                f"{'' if playable else '; byes only'}):"
            )
            counts = {id(p): self._turn_reuse_counts(p) for p in playable}
            for i, p in enumerate(pairings, 1):
                if p.model_b is None:
                    print(
                        f"   {i:>2}. {_short(p.model_a):<42}  — BYE (sits out)"
                    )
                    continue
                a_pts = self.rr.players[p.model_a].points
                b_pts = self.rr.players[p.model_b].points
                reuse, new = counts[id(p)]
                print(
                    f"   {i:>2}. {_short(p.model_a):<42}  ({a_pts:.1f})  "
                    f"vs  {_short(p.model_b):<42}  ({b_pts:.1f})  "
                    f"— {reuse} reusable / {new} new"
                )
                scheduled.append((rnd, p))
            total_reuse = sum(r for r, _ in counts.values())
            total_new = sum(n for _, n in counts.values())
            if playable:
                print(
                    f"     subtotal: {total_reuse} reusable / {total_new} new "
                    f"({total_reuse + total_new} turns)"
                )
            print()

        if not scheduled:
            # Only byes left — commit every batch and advance state.
            for _rnd, pairings in batches:
                self.rr.commit_round(pairings)
            self.pending_pairings = None
            _persist_round_robin_state(
                self.state_path, self.tournament_id, self.rr,
                self.repo_names, self.selected_model_ids, self.seed,
            )
            print("[all] no playable pairings; remaining rounds committed.\n")
            return

        n_matches = len(scheduled)
        n_rounds = len(batches)
        confirm = questionary.confirm(
            f"Run all remaining ({n_matches} matches across {n_rounds} "
            f"scheduled round(s)) in one parallel batch?",
            default=True,
        ).ask()
        if not confirm:
            print("  cancelled.\n")
            return

        print(f"\n[all] running {n_matches} pairings...\n")
        playable = [p for _, p in scheduled]
        # Distinct seeds per pairing; offset past ordinary per-round bases.
        seed_base = self.seed + 1000 * (self.rr.current_round_index + 1)
        # Use the first remaining round index for the live TUI header only.
        tui_rnd = batches[0][0]
        executed = self._run_round_parallel(playable, tui_rnd, seed_base)
        result_by_pair: dict[tuple[str, str | None], MatchResult] = {}
        for p, m in executed:
            if m is None:
                continue
            result_by_pair[(p.model_a, p.model_b)] = m

        for rnd, pairings in batches:
            for p in pairings:
                if p.model_b is None:
                    continue
                m = result_by_pair.get((p.model_a, p.model_b))
                if m is None:
                    continue
                self.matches.append(m)
                self.match_round[m.match_id] = rnd
            self.rr.commit_round(pairings)
            for p in pairings:
                if p.model_b is None:
                    continue
                m = result_by_pair.get((p.model_a, p.model_b))
                if m is None:
                    continue
                sa = _outcome_score_a(m.outcome)
                self.rr.record_result(p.model_a, p.model_b, sa)

        self.pending_pairings = None
        _persist_round_robin_state(
            self.state_path,
            self.tournament_id,
            self.rr,
            self.repo_names,
            self.selected_model_ids,
            self.seed,
        )
        print(f"\n[all] done ({n_matches} matches). State → {self.state_path}\n")

    def _collect_remaining_batches(self) -> list[tuple[int, list[SwissPairing]]]:
        """All uncommitted base rounds + one intake batch if residual budget.

        Drop and re-peek past any single-round ``pending_pairings`` so ``all``
        always sees the full remainder even after a prior ``pairings`` preview.
        """
        self.pending_pairings = None
        batches: list[tuple[int, list[SwissPairing]]] = []
        base_remaining = self.rr.remaining_base_rounds()
        start_idx = self.rr.current_round_index
        for offset, pairings in enumerate(base_remaining):
            batches.append((start_idx + offset + 1, pairings))

        # Intake is legal once the base circle is exhausted; if base rounds
        # remain in this batch they will be committed first, so intake’s
        # residual-already_played set is still correct (newcomers are not on
        # the circle table). Prefer illustrating intake in the same confirm.
        base_done_after = self.rr.current_round_index + len(base_remaining) >= len(
            self.rr._schedule
        )
        if (
            base_done_after
            and self.rr.intake_players
            and not self.rr.intake_complete
        ):
            secondary: dict[str, float] = {}
            if self.matches:
                snaps = compute_all_ratings(self.matches)
                secondary = {m: s.trueskill_sigma for m, s in snaps.items()}
            self.rr.set_intake_context(
                matches=self.matches, secondary_score=secondary
            )
            try:
                intake = self.rr._pair_intake_round()
            except RuntimeError:
                intake = []
            if intake:
                intake_rnd = start_idx + len(base_remaining) + 1
                batches.append((intake_rnd, intake))

        if not batches:
            raise RuntimeError(
                "Round-robin schedule complete: every base pairing has been "
                "played and active-sampling intake budgets are exhausted."
            )
        return batches

    def _run_round_parallel(
        self,
        pairings: list[SwissPairing],
        rnd: int,
        seed_base: int,
    ) -> list[tuple[SwissPairing, MatchResult | None]]:
        """Run every pairing in a round by pooling ALL of their defense
        sub-turns into one ThreadPoolExecutor.

        Mirrors ``TournamentConsole._run_round_parallel`` in
        ``run_tournament.py``: plan every pairing (no agent runs), flatten
        cache-miss tasks across the round, run them concurrently under a live
        ``rich`` progress tree, then assemble each match.
        """
        # 1) Plan every pairing (no agent runs yet).
        plans: dict[str, MatchPlan] = {}
        plan_by_pairing: list[tuple[SwissPairing, MatchPlan | None]] = []
        for i, p in enumerate(pairings):
            try:
                plan = self.orchestrator.plan_match(
                    model_a_id=p.model_a,
                    model_b_id=p.model_b,
                    repo_config=self.repo_cfgs,
                    seed=seed_base + i,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  ! planning failed for "
                    f"{_short(p.model_a)} vs {_short(p.model_b)}: {e}"
                )
                plan_by_pairing.append((p, None))
                continue
            plans[plan.match_key] = plan
            plan_by_pairing.append((p, plan))

        # 2) Flatten cache-miss tasks across the whole round and register
        #    every sub-turn in the live reporter.
        reporter = DefenseRoundReporter(round_index=rnd)
        all_tasks: list[DefenseTask] = []
        handles: dict[int, object] = {}
        for p, plan in plan_by_pairing:
            if plan is None:
                continue
            label = f"{_short(p.model_a)} vs {_short(p.model_b)}"
            reporter.register_match(plan.match_key, label)
            hits = plan.cache_hits()
            if hits:
                reporter.finish_match(
                    plan.match_key, SubturnStatus.RUNNING,
                    detail=f"{hits} cached",
                )
            for task in plan.pending_tasks:
                if task.side == "b_defends_a":
                    blue_cid, red_cid = p.model_b, p.model_a
                else:
                    blue_cid, red_cid = p.model_a, p.model_b
                sid = (
                    f"{task.repo_config.name}:{task.side}:{task.slot}:"
                    f"{task.challenge.challenge_id[:8]}"
                )
                handle = reporter.register_subturn(
                    plan.match_key, sid,
                    label=(
                        f"blue={_short(blue_cid)} defends "
                        f"{_short(red_cid)}'s {task.repo_config.name} challenge"
                    ),
                    blue_model=task.blue_model_id,
                    repo=task.repo_config.name,
                )
                handles[id(task)] = handle
                all_tasks.append(task)

        total_hits = sum(
            plan.cache_hits() for _, plan in plan_by_pairing if plan is not None
        )
        max_workers = self.orchestrator.config.match.max_workers
        effective = min(max_workers, len(all_tasks)) or 1
        print(
            f"  pool: {len(all_tasks)} defense sub-turns to run "
            f"({total_hits} reused from cache) across {len(plans)} matches "
            f"— max_workers={max_workers}, using {effective}\n"
        )

        # 3) Run the pool under the live TUI.
        t0 = time.perf_counter()
        done = 0
        if all_tasks:
            with round_live(reporter):
                with ThreadPoolExecutor(max_workers=effective) as pool:
                    fut_to_task = {
                        pool.submit(
                            self.orchestrator.run_defense_task,
                            t,
                            progress=handles.get(id(t)),
                            console_echo=False,
                        ): t
                        for t in all_tasks
                    }
                    for fut in as_completed(fut_to_task):
                        task = fut_to_task[fut]
                        done += 1
                        handle = handles.get(id(task))
                        try:
                            result = fut.result()
                        except Exception as e:  # noqa: BLE001
                            if handle is not None:
                                handle.finish(
                                    SubturnStatus.FAIL,
                                    detail=f"error: {str(e)[:60]}",
                                )
                            continue
                        self.orchestrator.place_result(
                            plans[task.match_key], task, result
                        )
                        sc = result.score
                        if handle is not None:
                            verdict = (
                                SubturnStatus.SUCCESS
                                if sc.blue_composite > 0
                                else SubturnStatus.FAIL
                            )
                            handle.finish(
                                verdict,
                                detail=(
                                    f"blue={sc.blue_composite:.2f} "
                                    f"${result.cost_usd:.4f}"
                                ),
                            )
        dt = time.perf_counter() - t0
        print(
            f"\n  pool drained in {dt:.1f}s ({done}/{len(all_tasks)} sub-turns)\n"
        )

        # 4) Assemble each planned match.
        executed: list[tuple[SwissPairing, MatchResult | None]] = []
        for p, plan in plan_by_pairing:
            if plan is None:
                executed.append((p, None))
                continue
            try:
                m = self.orchestrator.assemble_match(plan)
            except Exception as e:  # noqa: BLE001
                print(
                    f"    ! assembling match failed for "
                    f"{_short(p.model_a)} vs {_short(p.model_b)}: {e}"
                )
                executed.append((p, None))
                continue
            print(
                f"  → {_short(p.model_a)} vs {_short(p.model_b)}: "
                f"{m.outcome.value}  A={m.model_a_total:.2f} "
                f"B={m.model_b_total:.2f}  ${m.total_cost_usd:.4f}"
            )
            executed.append((p, m))
        return executed

    def cmd_report(self) -> None:
        print("\n[report] invoking swe-duel-report ...\n")
        env = {**os.environ, "SWE_DUEL_REPORT_FORMAT": "round-robin"}
        subprocess.run(
            [sys.executable, "-m", "swe_duel.cli.build_report"],
            check=False,
            env=env,
        )

    # ── helpers ───────────────────────────────────────────────

    def _ensure_pending_pairings(self) -> list[SwissPairing]:
        if self.pending_pairings is None:
            # Refresh intake ranking from the latest match history before each
            # active-sampling round (paper's sequential rule).
            if self.rr.base_schedule_complete and self.rr.intake_players:
                secondary: dict[str, float] = {}
                if self.matches:
                    snaps = compute_all_ratings(self.matches)
                    secondary = {
                        m: s.trueskill_sigma for m, s in snaps.items()
                    }
                self.rr.set_intake_context(
                    matches=self.matches, secondary_score=secondary
                )
            self.pending_pairings = self.rr.pair_next_round()
        return self.pending_pairings

    def _elo_snapshot(self, matches: list[MatchResult]) -> dict[str, float]:
        if not matches:
            return {}
        snaps = compute_all_ratings(matches)
        return {m: s.elo for m, s in snaps.items()}

    def _turn_reuse_counts(self, pairing: SwissPairing) -> tuple[int, int]:
        """Count reusable (cached) vs new real-defense turns for this pairing.

        Identical to ``TournamentConsole._turn_reuse_counts``: summed over
        every repo in the match; auto-win turns (Red with no challenge for a
        repo) are counted as neither reusable nor new.
        """
        turns_per = self.orchestrator.config.match.turns_per_player
        store = self.orchestrator.challenge_store
        al = self.orchestrator.artifact_logger
        reuse = 0
        new = 0
        from swe_duel.models import split_composite_id

        for red_id, blue_id in [
            (pairing.model_a, pairing.model_b),
            (pairing.model_b, pairing.model_a),
        ]:
            red_model, red_harness, red_effort, red_provider = split_composite_id(red_id)
            blue_model, blue_harness, blue_effort, blue_provider = split_composite_id(blue_id)
            for repo_name in self.repo_names:
                challenge_ids = store.list_ids_in_index_order(
                    red_model, repo_name,
                    red_harness_id=red_harness or "mini-swe-agent",
                    red_reasoning_effort=red_effort,
                    red_provider=red_provider,
                )[:turns_per]
                for cid in challenge_ids:
                    if al.has_defense(
                        challenge_id=cid,
                        blue_model_id=blue_model,
                        blue_harness_id=blue_harness or "mini-swe-agent",
                        blue_reasoning_effort=blue_effort,
                        blue_provider=blue_provider,
                    ):
                        reuse += 1
                    else:
                        new += 1
        return reuse, new

    # ── REPL loop ─────────────────────────────────────────────

    def run(self) -> None:
        if self.resumed:
            print(
                f"\n[resume] tournament {self.tournament_id} — "
                f"round {self.rr.current_round_index} complete"
            )
            self.cmd_standings()
            self._help()
        else:
            self._help()
            self.cmd_status()

        commands_noarg = {
            "status": self.cmd_status,
            "s": self.cmd_status,
            "standings": self.cmd_standings,
            "ls": self.cmd_standings,
            "pairings": self.cmd_pairings,
            "p": self.cmd_pairings,
            "next": self.cmd_next,
            "n": self.cmd_next,
            "all": self.cmd_all,
            "a": self.cmd_all,
            "report": self.cmd_report,
            "r": self.cmd_report,
            "help": self._help,
            "?": self._help,
        }
        while True:
            try:
                line = input("swe-duel> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            parts = line.split(maxsplit=1)
            head = parts[0].lower()
            rest = parts[1] if len(parts) > 1 else ""
            if head in ("quit", "q", "exit"):
                break
            if head in ("history", "h"):
                try:
                    self.cmd_history(rest)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! error: {e}")
                continue
            fn = commands_noarg.get(head)
            if fn is None:
                print(f"  unknown command: {line!r}  (type ? for help)")
                continue
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"  ! error: {e}")

        print("\n[exit] generating final report...")
        self.cmd_report()
        print(f"[exit] tournament {self.tournament_id} complete.")

    def _help(self) -> None:
        print(
            "\n  Commands:\n"
            "    status     (s)   tournament overview\n"
            "    standings  (ls)  cumulative points + ELO leaderboard\n"
            "    history    (h)   list past rounds with pairings + outcomes\n"
            "    history N        ELO table as it stood after round N\n"
            "    pairings   (p)   preview next round's pairings (reuse counts)\n"
            "    next       (n)   confirm and run next round's matches\n"
            "                     (after add-participant: active-sampling intake)\n"
            "    all        (a)   run all remaining matches in one parallel batch\n"
            "                     (full remainder of circle + pending intake)\n"
            "    report     (r)   regenerate figures via build_report.py\n"
            "    quit/q           exit (generates report on the way out)\n"
        )


# ── entry point ───────────────────────────────────────────────


def main() -> int:
    install_container_cleanup_handlers()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Skip the interactive picker and use these model nicknames/IDs.",
    )
    parser.add_argument(
        "--harnesses",
        nargs="+",
        default=None,
        help=(
            "Agent harness(es) to pair with each --models entry (e.g. "
            "mini-swe-agent openhands codex claude-code). Each "
            "(model, harness, effort, provider) combination is a distinct "
            "competitor. Required when --models is given; no default."
        ),
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=None,
        help=(
            "Reasoning-effort selection(s), cross-producted with "
            "--models/--harnesses/--providers. Must be in each model's "
            "reasoning_efforts menu in config/models.yaml. Default: model "
            "default."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help=(
            "OpenRouter provider slug(s), cross-producted with "
            "--models/--harnesses/--efforts. Must be in each model's providers "
            "menu in config/models.yaml; only mini-swe-agent/openhands can pin "
            "a provider. Default: OpenRouter auto-routing."
        ),
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help=(
            "One or more repos for the tournament. A single match spans ALL of "
            "them: turns_per_player turns per repo per side. Unknown names "
            "(e.g. a typo'd repo) are skipped with a warning."
        ),
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--prompt-dir", default=default_prompt_dir())
    parser.add_argument(
        "--turns-per-player",
        "--turns-per-agent",  # back-compat alias
        dest="turns_per_player",
        type=int,
        default=None,
        help="Override arena.yaml match.turns_per_player.",
    )
    parser.add_argument(
        "--match-workers",
        dest="match_workers",
        type=int,
        default=None,
        help="Override arena.yaml match.max_workers (defense sub-turns run "
        "concurrently per round).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    args_for_setup = argparse.Namespace(**vars(args))
    args_for_setup.models = None  # don't pre-filter; we filter after selection
    ctx = setup(args_for_setup)
    arena_cfg = ctx["arena_config"]

    if args.turns_per_player is not None:
        arena_cfg.match.turns_per_player = args.turns_per_player
    if args.match_workers is not None:
        arena_cfg.match.max_workers = args.match_workers
    turns_per_player = arena_cfg.match.turns_per_player

    # Resolve --repos against config/repos/, skipping unknown names.
    repo_names: list[str] = []
    for rn in args.repos:
        if rn in ctx["repo_configs"]:
            if rn not in repo_names:
                repo_names.append(rn)
        else:
            print(
                f"[warn] repo {rn!r} not found in config/repos/ — skipping. "
                f"Available: {sorted(ctx['repo_configs'])}",
                file=sys.stderr,
            )
    if not repo_names:
        raise SystemExit(
            f"none of --repos {args.repos!r} matched config/repos/. "
            f"Available: {sorted(ctx['repo_configs'])}"
        )

    all_model_configs = ctx["model_configs"]
    slot_counts = _slot_counts(ctx["challenge_store"], all_model_configs, repo_names)
    success_counts = _success_counts(
        ctx["challenge_store"], all_model_configs, repo_names
    )

    # ── Resume detection ─────────────────────────────────────
    state_candidates = _find_resume_states_rr(
        ctx["artifact_logger"].tournaments_dir, repo_names
    )
    resumed = False
    rr: RoundRobinTournament | None = None
    tournament_id: str | None = None
    prior_matches: list[MatchResult] = []
    match_round: dict[str, int] = {}
    seed: int = args.seed if args.seed is not None else 0
    selected_ids: list[str] = []
    state_path: Path | None = None

    if state_candidates:
        new_value = "__new__"
        choices = []
        for p, payload in state_candidates:
            tid = payload.get("tournament_id", "?")
            rounds_played = len(payload.get("rounds", []))
            n_models = len(
                payload.get("selected_model_ids", payload.get("players", []))
            )
            ts = payload.get("timestamp", "")
            title = (
                f"{tid[:8]}…  rounds={rounds_played}  models={n_models}  "
                f"{ts[:19]}  ({p.name})"
            )
            choices.append(questionary.Choice(title=title, value=str(p)))
        choices.append(questionary.Choice(title="Start a new tournament", value=new_value))

        picked = questionary.select(
            f"Found {len(state_candidates)} existing round-robin state(s) for "
            f"repos={','.join(repo_names)}. "
            "Pick one to resume, or start new:",
            choices=choices,
        ).ask()
        if picked is None:
            raise SystemExit("aborted.")
        if picked != new_value:
            # Participant composite ids (model#harness[#effort#provider]);
            # legacy state files may hold bare/2-part ids — split tolerates both.
            from swe_duel.models import split_composite_id

            state_path = Path(picked)
            tmp_payload = json.loads(state_path.read_text())
            persisted_ids = tmp_payload.get(
                "selected_model_ids",
                [pl["model_id"] for pl in tmp_payload.get("players", [])],
            )
            persisted_seed = tmp_payload.get("seed", seed)
            seed = persisted_seed
            rr, tournament_id, _ = _reconstruct_round_robin(
                state_path, persisted_ids, seed
            )
            persisted_models = {split_composite_id(cid)[0] for cid in persisted_ids}
            selected_nicks = [
                nick
                for nick, cfg in all_model_configs.items()
                if cfg.model_id in persisted_models
            ]
            ctx["model_configs"] = {n: all_model_configs[n] for n in selected_nicks}
            selected_ids = list(persisted_ids)
            prior_matches, match_round = _load_match_results_for_pairs(
                ctx["artifact_logger"].matches_dir, rr, repo_names
            )
            # Feed match history so any residual intake pairings re-rank correctly.
            if rr.intake_players:
                secondary = {}
                if prior_matches:
                    snaps = compute_all_ratings(prior_matches)
                    secondary = {
                        m: s.trueskill_sigma for m, s in snaps.items()
                    }
                rr.set_intake_context(
                    matches=prior_matches, secondary_score=secondary
                )
            resumed = True

            assert state_path is not None and tournament_id is not None
            selected_ids = _maybe_add_participants_rr(
                rr=rr,
                selected_ids=selected_ids,
                all_model_configs=all_model_configs,
                slot_counts=slot_counts,
                success_counts=success_counts,
                repo_names=repo_names,
                turns_per_player=turns_per_player,
                challenge_store=ctx["challenge_store"],
                tournament_id=tournament_id,
                state_path=state_path,
                seed=seed,
                prior_matches=prior_matches,
            )
            ctx["model_configs"] = _merge_model_configs_for_ids(
                all_model_configs, selected_ids
            )

    if not resumed:
        # ── Fresh start: pick competitors (model, harness, effort, provider) ──
        if args.models:
            from swe_duel.cli.participant_select import resolve_cli_participants

            if not args.harnesses:
                from swe_duel.agents.harness import HARNESS_IDS

                raise SystemExit(
                    "--harnesses is required when --models is given "
                    f"(choose from {list(HARNESS_IDS)})."
                )
            participants = resolve_cli_participants(
                all_model_configs,
                models=list(args.models),
                harnesses=list(args.harnesses),
                efforts=args.efforts,
                providers=args.providers,
            )
            selected_ids = [p.cid for p in participants]
        else:
            _print_slot_table(
                ctx["challenge_store"], all_model_configs, repo_names,
                turns_per_player,
            )
            selected_ids = _select_models_interactive(
                all_model_configs, slot_counts, success_counts, len(repo_names)
            )

        _validate_pools(selected_ids, slot_counts, require_even=False)
        # Participant-keyed configs with each entrant's selection pre-bound.
        ctx["model_configs"] = _merge_model_configs_for_ids(
            all_model_configs, selected_ids
        )
        # Prompt for seed (used both for the initial circle ordering AND as
        # the persisted resume key).
        default_seed = args.seed if args.seed is not None else 0
        seed = _prompt_seed(default_seed)
        rr = RoundRobinTournament(selected_ids, seed=seed)
        tournament_id = str(uuid.uuid4())

    # Warn up front when OpenRouter has no pricing for a selected entrant's
    # (model, provider) — cost tracking for those competitors reports $0.00.
    # Both the fresh and resume paths leave ctx["model_configs"] keyed by
    # participant composite id with each entrant's selection pre-bound.
    warn_missing_pricing(ctx["model_configs"].values())
    # Surface request-rate caps before the live TUI starts: the throttle is
    # process-wide (workers divide one budget), and CLI harnesses cannot
    # enforce a configured cap at all.
    from swe_duel.models import split_composite_id

    summarize_rate_limits(ctx["model_configs"].values())
    warn_unenforceable_rate_limits(
        (cfg, split_composite_id(cid)[1]) for cid, cfg in ctx["model_configs"].items()
    )

    # If we resumed, we still need to prompt for the seed? No — seed was
    # restored from the state file. But if the user passed --seed explicitly
    # on a fresh start, _prompt_seed already ran above. For a resume we skip
    # the prompt (the initial ordering is fixed by the persisted seed).
    if resumed and args.seed is not None and args.seed != seed:
        print(
            f"[warn] --seed={args.seed} ignored on resume; using persisted "
            f"seed={seed} (the schedule is fixed at construction time)."
        )

    # ── Build orchestrator ───────────────────────────────────
    _test_runner_cache: dict[str, TestRunner] = {}

    def _test_runner_factory(rc) -> TestRunner:
        tr = _test_runner_cache.get(rc.name)
        if tr is None:
            ex = DockerExecutor(
                docker_image=rc.docker_image,
                timeout_s=arena_cfg.sandbox.timeout_seconds,
                memory_mb=arena_cfg.sandbox.memory_mb,
            )
            tr = TestRunner(executor=ex, repo_config=rc)
            _test_runner_cache[rc.name] = tr
        return tr

    # ctx["model_configs"] is keyed by participant composite id with the
    # effort/provider selection applied (see _merge_model_configs_for_ids).
    model_configs_by_id = dict(ctx["model_configs"])
    selected_model_ids = selected_ids

    match_orchestrator = MatchOrchestrator(
        model_configs=model_configs_by_id,
        challenge_store=ctx["challenge_store"],
        workspace_manager=ctx["workspace_manager"],
        config=arena_cfg,
        artifact_logger=ctx["artifact_logger"],
        prompt_dir=Path(args.prompt_dir),
        test_runner_factory=_test_runner_factory,
    )

    console = RoundRobinConsole(
        ctx=ctx,
        match_orchestrator=match_orchestrator,
        repo_names=repo_names,
        seed=seed,
        tournament_id=tournament_id,
        selected_model_ids=selected_model_ids,
        rr=rr,
        resumed=resumed,
        prior_matches=prior_matches,
        match_round=match_round,
    )
    console.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
