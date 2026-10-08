#!/usr/bin/env python
"""Interactive Swiss-system tournament console.

A tournament runs over one or more repos (`--repos`); a single match between
two competitors spans ALL selected repos (turns_per_player turns per repo per
side, aggregated into one MatchResult). A competitor that lacks a challenge for
some repo is still played — the opponent auto-wins those turns.

Launch flow:
    1. Interactive paged wizard to pick competitors — a competitor is the
       4-tuple (model, harness, reasoning_effort, provider): a model page,
       then a harness page, then an effort page (menu from models.yaml), then
       a provider page (menu from models.yaml), then back to a landing page
       listing the selected participants (add another / remove / start).
       Options without challenge slots across the selected repos are dimmed.
       Even count required.
    2. Dim a competitor only if it has NO slots at all (no success, no failed
       attempt) across the selected repos — nothing to play. Missing-but-some
       repos are fine (opponent auto-wins those turns).
    3. Prompt for the round-1 pairing seed (default from --seed or 0).
    4. If a Swiss state file exists for this exact repo set, offer to resume —
       landing page becomes the ELO league table with the current round header.
       After choosing a tournament to resume, optionally add late entrants
       (newcomers seat at 0 Swiss points and fold into subsequent pairings).

REPL commands (history N takes an argument):
    status     (s)   tournament overview
    standings  (ls)  current Swiss + ELO leaderboard (with round counter)
    history    (h)   list past rounds with pairings + outcomes
    history N        ELO table as it stood after round N
    pairings   (p)   preview next round's pairings with per-pairing reuse counts
    next       (n)   run the next round
    report     (r)   regenerate figures via build_report.py
    quit/q           exit (final report generated on the way out)
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
from swe_duel.engine.swiss import SwissPairing, SwissTournament
from swe_duel.models import (
    BlueFix,
    DefenseResult,
    MatchOutcome,
    MatchResult,
    TurnResult,
    TurnScore,
)
from swe_duel.models import AgentTrajectory
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.rating import compute_all_ratings


# ── helpers ───────────────────────────────────────────────────


def _outcome_score_a(outcome: MatchOutcome) -> float:
    if outcome == MatchOutcome.MODEL_A_WINS:
        return 1.0
    if outcome == MatchOutcome.MODEL_B_WINS:
        return 0.0
    return 0.5


def _short(model_id: str, width: int = 40) -> str:
    # `model_id` here is a competitor identity (composite "model#harness");
    # render it human-friendly as "model [harness]" before truncating.
    from swe_duel.models import display_composite_id

    disp = display_composite_id(model_id)
    return disp if len(disp) <= width else disp[: width - 1] + "…"


def _persist_swiss_state(
    path: Path,
    tournament_id: str,
    swiss: SwissTournament,
    repo_names: list[str],
    selected_model_ids: list[str],
) -> None:
    payload = {
        "tournament_id": tournament_id,
        # A match spans all of these repos. `repo_name` (joined) is kept for
        # backwards-compatible readers; `repo_names` is the authoritative list.
        "repo_name": ",".join(repo_names),
        "repo_names": list(repo_names),
        "selected_model_ids": selected_model_ids,
        "intake_players": list(swiss.intake_players),
        "current_round_index": swiss.current_round_index,
        "players": [
            {
                "model_id": p.model_id,
                "points": p.points,
                "opponents": list(p.opponents),
                "had_bye": p.had_bye,
            }
            for p in swiss.players.values()
        ],
        "rounds": [
            [{"model_a": pp.model_a, "model_b": pp.model_b} for pp in rnd]
            for rnd in swiss.round_history
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ── interactive selection ─────────────────────────────────────


def _slot_counts(
    store, model_configs, repo_names: list[str]
) -> dict[str, int]:
    """Per-competitor *slot* counts keyed by participant composite id.

    A "slot" is one generation effort for a (competitor, repo): it ends either
    in a successful challenge or in a fully-failed attempt chain. A competitor
    is eligible for a repo as long as it has at least one slot there — a
    *successful* challenge lets it pose a real challenge, while a repo with only
    failed slots is still played (the opponent simply auto-wins those turns).

    The count returned here is the number of repos (among ``repo_names``) for
    which the competitor has **any** slot (success OR failure). It is used only
    to decide eligibility/dimming, not to size the match.

    Derived from the bank index (not the yaml cartesian product): every
    persisted entry carries its participant identity
    (model/harness/effort/provider) + repo, so enumerating the entries gives
    exactly the playable competitor set without exploding over ungenerated
    yaml combinations.
    """
    from swe_duel.models import composite_id

    repos = set(repo_names)
    counts: dict[str, set[str]] = {}
    for entry in store.index.get("entries", {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("repo_name") not in repos:
            continue
        cid = composite_id(
            str(entry.get("red_model_id", "")),
            str(entry.get("red_harness_id", "mini-swe-agent") or "mini-swe-agent"),
            str(entry.get("red_reasoning_effort", "") or ""),
            str(entry.get("red_provider", "") or ""),
        )
        counts.setdefault(cid, set()).add(str(entry.get("repo_name")))
    for model_id, harness_id, effort, provider, repo, _n in store.list_pools():
        if repo in repos:
            cid = composite_id(model_id, harness_id, effort, provider)
            counts.setdefault(cid, set()).add(repo)
    return {cid: len(rs) for cid, rs in counts.items()}


def _success_counts(
    store, model_configs, repo_names: list[str]
) -> dict[str, int]:
    """Per-competitor count of repos with at least one *successful* challenge.

    Surfaced in the selector so the user can see how many repos a competitor
    can actually pose challenges for (the rest become opponent auto-wins).
    """
    from swe_duel.models import composite_id

    repos = set(repo_names)
    counts: dict[str, set[str]] = {}
    for model_id, harness_id, effort, provider, repo, _n in store.list_pools():
        if repo in repos:
            cid = composite_id(model_id, harness_id, effort, provider)
            counts.setdefault(cid, set()).add(repo)
    return {cid: len(rs) for cid, rs in counts.items()}


def _print_slot_table(
    store, all_models: dict, repo_names: list[str], turns_per_player: int
) -> None:
    """Print the per-identity bank pool table (mirrors generate_challenges.py).

    One row per *existing* participant identity (model, harness, effort,
    provider) with a successful_slots/total_attempted_slots cell per repo, so
    the user can see — before selecting — how many real challenges each
    competitor can pose per repo (the rest become opponent auto-wins)."""
    from swe_duel.cli.participant_select import print_pool_table

    print(f"turns_per_player = {turns_per_player}")
    print_pool_table(store, repo_names)


def _select_models_interactive(
    all_models: dict,
    slot_counts: dict[str, int],
    success_counts: dict[str, int],
    n_repos: int,
) -> list[str]:
    """Paged competitor selection: model → harness → effort → provider pages,
    then a landing page (add another / remove / start). Returns participant
    composite ids ("model#harness[#effort#provider]").

    A competitor is dimmed only when it has **zero slots** across all selected
    repos (no successful challenge and no failed attempt anywhere) — there is
    nothing to play. Competitors missing only *some* repos are selectable: the
    opponent auto-wins the turns for repos they couldn't generate for.
    """
    from swe_duel.cli.participant_select import select_participants
    from swe_duel.models import split_composite_id

    def _status(slots: int, succ: int) -> str:
        return f"{succ}/{n_repos} repos w/ challenges, {slots} w/ slots"

    bank_ids = {cid for cid, n in slot_counts.items() if n > 0}
    slot_status = {
        cid: _status(slot_counts.get(cid, 0), success_counts.get(cid, 0))
        for cid in bank_ids
    }
    # Model-page annotation: aggregate every identity of the model.
    model_status: dict[str, str] = {}
    for nick, cfg in all_models.items():
        slots = {
            cid for cid in bank_ids
            if split_composite_id(cid)[0] == cfg.model_id
        }
        if slots:
            succ = sum(1 for cid in slots if success_counts.get(cid, 0) > 0)
            model_status[nick] = (
                f"{succ}/{n_repos} repos w/ challenges, {len(slots)} w/ slots"
            )
    participants = select_participants(
        all_models,
        start_verb="Start the tournament",
        require_slots=True,
        bank_ids=bank_ids,
        model_status=model_status,
        slot_status=slot_status,
    )
    return [p.cid for p in participants]


def _validate_pools(
    selected_ids: list[str],
    slot_counts: dict[str, int],
    *,
    require_even: bool = True,
    min_count: int = 2,
) -> None:
    """`selected_ids` are participant composite ids
    (model#harness[#effort#provider]).

    Dimming is by *slots*, not successful challenges: a competitor is only
    rejected when it has no slots at all (nothing to play). Repos a competitor
    lacks challenges for are handled at match time as opponent auto-wins.

    ``require_even`` is True for a fresh Swiss start (byes disabled at kickoff)
    but False when late-adding entrants — Swiss pairing already handles odd N
    via a single bye.
    """
    from swe_duel.models import display_composite_id

    if len(selected_ids) < min_count:
        raise SystemExit(f"Need at least {min_count} competitor(s).")
    if require_even and len(selected_ids) % 2 != 0:
        raise SystemExit(
            f"Tournament requires an even number of competitors (selected "
            f"{len(selected_ids)}). Byes are disabled — please add or "
            "remove one."
        )
    empty = [cid for cid in selected_ids if slot_counts.get(cid, 0) <= 0]
    if empty:
        raise SystemExit(
            "No challenges or slots (cannot play) for:\n"
            + "\n".join(f"  {display_composite_id(cid)}" for cid in empty)
        )


def _select_addable_competitors(
    all_models: dict,
    slot_counts: dict[str, int],
    success_counts: dict[str, int],
    n_repos: int,
    already_selected: set[str],
) -> list[str]:
    """Paged selection of newcomer entrants (same wizard as the fresh start).

    Same dimming rules as :func:`_select_models_interactive` — zero-slot
    competitors are disabled. Returns the composite ids that were NOT already
    seated (possibly empty if the user adds nothing new).
    """
    from swe_duel.cli.participant_select import select_participants

    bank_ids = {cid for cid, n in slot_counts.items() if n > 0}
    preselected = _participants_from_ids(all_models, already_selected)
    picked = select_participants(
        all_models,
        start_verb="Done adding newcomers",
        require_slots=True,
        bank_ids=bank_ids,
        preselected=preselected,
    )
    return [p.cid for p in picked if p.cid not in already_selected]


def _participants_from_ids(
    all_models: dict, cids: set[str]
) -> list:  # list[participant_select.Participant]
    """Rebuild Participant objects from composite ids (for wizard seeding)."""
    from swe_duel.cli.participant_select import Participant
    from swe_duel.models import split_composite_id

    by_model = {cfg.model_id: (nick, cfg) for nick, cfg in all_models.items()}
    out: list[Participant] = []
    for cid in sorted(cids):
        model_id, harness_id, effort, provider = split_composite_id(cid)
        base = by_model.get(model_id)
        if base is None:
            continue
        nick, cfg = base
        out.append(
            Participant(
                nick=nick,
                model_config=cfg.with_selection(effort, provider),
                harness_id=harness_id or "mini-swe-agent",
                reasoning_effort=effort,
                provider=provider,
            )
        )
    return out


def _prompt_seed(default_seed: int) -> int:
    raw = questionary.text(
        f"Seed for round-1 pairing (default {default_seed}):",
        default=str(default_seed),
    ).ask()
    if raw is None:
        raise SystemExit("aborted.")
    raw = raw.strip()
    if not raw:
        return default_seed
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"invalid seed: {raw!r}")


def _prompt_intake_budget(n_incumbents: int) -> int:
    """Ask how many active-sampling opponents each newcomer should face."""
    from swe_duel.scoring.active_sampling import default_intake_budget

    default_k = default_intake_budget(n_incumbents)
    raw = questionary.text(
        f"Active-sampling matches per newcomer "
        f"(default {default_k} = ceil(log2 N)+1, max {n_incumbents}):",
        default=str(default_k),
    ).ask()
    if raw is None:
        raise SystemExit("aborted.")
    raw = raw.strip()
    if not raw:
        return default_k
    try:
        k = int(raw)
    except ValueError:
        raise SystemExit(f"invalid budget: {raw!r}")
    if k < 1:
        raise SystemExit("budget must be ≥ 1")
    return min(k, n_incumbents)


def _merge_model_configs_for_ids(
    all_model_configs: dict, selected_ids: list[str]
) -> dict:
    """Model-config map keyed by participant composite id, selection applied.

    The orchestrator resolves a Blue participant's harness by looking its
    composite id up here, so each entrant — including its reasoning-effort /
    provider selection — gets its own pre-bound ModelConfig. Legacy 2-part ids
    map to the default (empty) selection.
    """
    from swe_duel.models import split_composite_id

    by_model = {cfg.model_id: cfg for cfg in all_model_configs.values()}
    out: dict = {}
    for cid in selected_ids:
        model_id, _harness_id, effort, provider = split_composite_id(cid)
        base = by_model.get(model_id)
        if base is None:
            continue
        out[cid] = base.with_selection(effort, provider)
    return out


def _maybe_add_participants_swiss(
    *,
    swiss: SwissTournament,
    selected_ids: list[str],
    all_model_configs: dict,
    slot_counts: dict[str, int],
    success_counts: dict[str, int],
    repo_names: list[str],
    turns_per_player: int,
    challenge_store,
    tournament_id: str,
    state_path: Path,
) -> list[str]:
    """Post-resume prompt: seat newcomers into an existing Swiss field.

    Newcomers enter at 0 Swiss points; subsequent ``pair_next_round`` calls fold
    them into score-group matching. Returns the (possibly extended) selected-id
    list.
    """
    from swe_duel.models import display_composite_id

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
    added = swiss.add_players(newcomers)
    if not added:
        print("  (all selected competitors were already seated)")
        return selected_ids

    extended = list(selected_ids) + [c for c in added if c not in selected_ids]
    print("\n  Added newcomers (Swiss seat at 0 pts):")
    for cid in added:
        print(f"    + {display_composite_id(cid)}")
    print(
        f"  Field size {len(selected_ids)} → {len(extended)}. "
        f"Recommended rounds now {swiss.recommended_rounds}.\n"
    )
    _persist_swiss_state(
        state_path, tournament_id, swiss, repo_names, extended
    )
    return extended


# ── resume detection / reconstruction ─────────────────────────


def _state_repo_names(payload: dict) -> list[str]:
    """Repo set of a swiss-state payload (tolerates legacy single-repo files)."""
    if payload.get("repo_names"):
        return list(payload["repo_names"])
    rn = payload.get("repo_name", "")
    return [r for r in rn.split(",") if r] if rn else []


def _find_resume_states(
    tournaments_dir: Path, repo_names: list[str]
) -> list[tuple[Path, dict]]:
    """Return all swiss state files matching the repo *set*, newest-first."""
    want = set(repo_names)
    candidates = sorted(
        tournaments_dir.glob("swiss_state_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    results: list[tuple[Path, dict]] = []
    for p in candidates:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if set(_state_repo_names(payload)) == want:
            results.append((p, payload))
    return results


def _reconstruct_swiss(
    state_path: Path, model_ids: list[str]
) -> tuple[SwissTournament, str, list[str]]:
    payload = json.loads(state_path.read_text())
    swiss = SwissTournament(model_ids)
    for p in payload.get("players", []):
        if p["model_id"] in swiss.players:
            pl = swiss.players[p["model_id"]]
            pl.points = p.get("points", 0.0)
            pl.opponents = list(p.get("opponents", []))
            pl.had_bye = p.get("had_bye", False)
    swiss.round_history = [
        [SwissPairing(model_a=pp["model_a"], model_b=pp["model_b"]) for pp in rnd]
        for rnd in payload.get("rounds", [])
    ]
    # Restore late entrants recorded after the original field was seated.
    intake = [
        mid
        for mid in payload.get("intake_players", [])
        if mid in swiss.players
    ]
    swiss.intake_players = list(intake)
    tournament_id = payload.get("tournament_id", str(uuid.uuid4()))
    selected = payload.get("selected_model_ids", model_ids)
    return swiss, tournament_id, selected


def _load_match_results_for_pairs(
    matches_dir: Path, swiss: SwissTournament, repo_names: list[str]
) -> tuple[list[MatchResult], dict[str, int]]:
    """Reconstruct minimal MatchResult objects for matches recorded in this
    tournament's Swiss state. Returns (matches, match_id → round_number).

    Matching is done by (pair, repo-set, timestamp): for each round, claim the
    earliest unclaimed match JSON whose participants equal the pairing and whose
    repo set matches this tournament's repos."""
    want_repos = set(repo_names)
    results: list[MatchResult] = []
    match_round: dict[str, int] = {}
    used_match_ids: set[str] = set()
    for round_index, pairings in enumerate(swiss.round_history, start=1):
        for sp in pairings:
            if sp.model_b is None:
                continue
            candidates: list[dict] = []
            for p in matches_dir.glob("*.json"):
                try:
                    payload = json.loads(p.read_text())
                except Exception:
                    continue
                payload_repos = {
                    r for r in str(payload.get("repo_name", "")).split(",") if r
                }
                if payload_repos != want_repos:
                    continue
                if payload.get("match_id") in used_match_ids:
                    continue
                pair = {payload.get("model_a_id"), payload.get("model_b_id")}
                if pair == {sp.model_a, sp.model_b}:
                    candidates.append(payload)
            if not candidates:
                continue
            candidates.sort(key=lambda x: x.get("timestamp", ""))
            payload = candidates[0]
            used_match_ids.add(payload["match_id"])
            try:
                m = _match_result_from_payload(payload)
            except Exception:
                continue
            results.append(m)
            match_round[m.match_id] = round_index
    return results, match_round


def _match_result_from_payload(p: dict) -> MatchResult:
    turns: list[TurnResult] = []
    for t in p.get("turns", []):
        sc = t.get("score", {})
        score = TurnScore(
            s_regression=sc.get("s_regression", 0.0),
            s_feature=sc.get("s_feature", 0.0),
            s_bugfix=sc.get("s_bugfix", 0.0),
            blue_composite=sc.get("blue_composite", 0.0),
            red_composite=sc.get("red_composite", 0.0),
            test_details=sc.get("test_details", {}),
        )
        # We only need score + red/blue model ids on the defense_result for
        # ELO + Swiss; the rest is left empty/placeholder.
        empty_traj = AgentTrajectory(
            steps=[], total_steps=0, total_input_tokens=0,
            total_output_tokens=0, total_cost_usd=0.0,
            model_id=t.get("blue_model_id", ""), duration_seconds=0.0,
        )
        empty_fix = BlueFix(
            review_findings=[], fix_explanation="", fix_diff="",
            modified_file_contents={}, agent_trajectory=empty_traj,
        )
        defense = DefenseResult(
            defense_id=t.get("defense_id", ""),
            challenge_id=t.get("challenge_id", ""),
            blue_model_id=t.get("blue_model_id", ""),
            blue_fix=empty_fix,
            score=score,
            duration_seconds=0.0,
            cost_usd=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        # ChallengeRecord is fully required by TurnResult dataclass; pass a
        # placeholder built from the challenge_id alone.
        turns.append(
            TurnResult(
                turn_id=t.get("turn_id", ""),
                turn_index=t.get("turn_index", 0),
                challenge_record=_placeholder_challenge_record(
                    t.get("challenge_id", ""), t.get("red_model_id", ""), p.get("repo_name", ""),
                ),
                defense_result=defense,
                red_model_id=t.get("red_model_id", ""),
                blue_model_id=t.get("blue_model_id", ""),
            )
        )
    ts = p.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return MatchResult(
        match_id=p["match_id"],
        model_a_id=p["model_a_id"],
        model_b_id=p["model_b_id"],
        repo_name=p.get("repo_name", ""),
        turns=turns,
        model_a_total=p.get("model_a_total", 0.0),
        model_b_total=p.get("model_b_total", 0.0),
        outcome=MatchOutcome(p.get("outcome", "draw")),
        duration_seconds=p.get("duration_seconds", 0.0),
        total_cost_usd=p.get("total_cost_usd", 0.0),
        timestamp=ts or datetime.now(timezone.utc),
    )


def _placeholder_challenge_record(challenge_id: str, red_model_id: str, repo_name: str):
    from swe_duel.models import (
        ChallengeRecord,
        RedChallenge,
        RedValidationResult,
        AgentTrajectory,
    )
    empty_traj = AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0,
        total_output_tokens=0, total_cost_usd=0.0,
        model_id=red_model_id, duration_seconds=0.0,
    )
    challenge = RedChallenge(
        target_files=[], exploration_summary="",
        feature_spec="", feature_rationale="", pr_diff="",
        modified_file_contents={}, original_file_contents={},
        feature_test_code="",
        bug_type=None, bug_description=None,
        bug_location=None, bug_test_code=None,
        agent_trajectory=empty_traj,
    )
    validation = RedValidationResult(passed=True, gate_results=[], attempt_number=1)
    return ChallengeRecord(
        challenge_id=challenge_id,
        red_model_id=red_model_id,
        repo_name=repo_name,
        repo_commit_sha="",
        target_files=[],
        challenge=challenge,
        validation=validation,
        generated_at=datetime.now(timezone.utc),
        generation_cost_usd=0.0,
        generation_retries=0,
    )


# ── REPL ──────────────────────────────────────────────────────


class TournamentConsole:
    def __init__(
        self,
        ctx: dict,
        match_orchestrator: MatchOrchestrator,
        repo_names: list[str],
        seed: int,
        tournament_id: str,
        selected_model_ids: list[str],
        swiss: SwissTournament,
        resumed: bool,
        prior_matches: list[MatchResult],
        match_round: dict[str, int],
    ) -> None:
        self.ctx = ctx
        self.orchestrator = match_orchestrator
        self.repo_names = list(repo_names)
        # A match spans all selected repos.
        self.repo_cfgs = [ctx["repo_configs"][rn] for rn in repo_names]
        self.repo_label = ",".join(repo_names)
        self.seed = seed
        self.tournament_id = tournament_id
        self.selected_model_ids = selected_model_ids
        self.matches: list[MatchResult] = list(prior_matches)
        self.match_round: dict[str, int] = dict(match_round)
        self.resumed = resumed
        self.swiss = swiss
        self.pending_pairings: list[SwissPairing] | None = None
        self.state_path = (
            ctx["artifact_logger"].tournaments_dir
            / f"swiss_state_{self.tournament_id}.json"
        )

    # ── command handlers ──────────────────────────────────────

    def cmd_status(self) -> None:
        n = len(self.selected_model_ids)
        rec = self.swiss.recommended_rounds
        log2n = math.log2(n) if n > 1 else 0.0
        print()
        print(f"  Tournament ID : {self.tournament_id}")
        print(f"  Repos         : {self.repo_label}")
        print(f"  Models        : {n}")
        print(f"  log2(N)       : {log2n:.2f}")
        print(f"  Recommended   : {rec} rounds (ceil log2 N)")
        print(f"  Current round : {self.swiss.current_round_index} of {rec}+")
        print(f"  Matches run   : {len(self.matches)}")
        print(f"  Seed          : {self.seed}")
        print(f"  Total cost    : ${sum(m.total_cost_usd for m in self.matches):.4f}")
        print()

    def cmd_standings(self) -> None:
        completed = self.swiss.current_round_index
        rec = self.swiss.recommended_rounds
        std = self.swiss.standings()
        elo_map = self._elo_snapshot(self.matches)
        print()
        print(f"  ── Round {completed} of {rec} complete — Round {completed + 1} next ──")
        print(f"  {'#':>2}  {'Model':<42}  {'Swiss':>6}  {'ELO':>7}  {'Played':>6}")
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
                m for m in self.matches if self.match_round.get(m.match_id, 10**9) <= n
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

        # No arg → list all rounds with pairings + outcomes.
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
        rnd = self.swiss.current_round_index + 1
        print()
        print(f"  Round {rnd} pairings ({len(pairings)} pairings):")
        counts = {id(p): self._turn_reuse_counts(p) for p in pairings}
        for i, p in enumerate(pairings, 1):
            a_pts = self.swiss.players[p.model_a].points
            b_pts = self.swiss.players[p.model_b].points
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
        rnd = self.swiss.current_round_index + 1
        self.cmd_pairings()
        confirm = questionary.confirm(
            f"Run Round {rnd} ({len(pairings)} matches)?", default=True
        ).ask()
        if not confirm:
            print("  cancelled.\n")
            return

        print(f"\n[round {rnd}] running {len(pairings)} pairings...\n")
        seed_base = self.seed + 1000 * rnd
        executed = self._run_round_parallel(pairings, rnd, seed_base)
        for p, m in executed:
            if m is None:
                continue
            self.matches.append(m)
            self.match_round[m.match_id] = rnd

        self.swiss.commit_round([p for p, _ in executed])
        for pairing, result in executed:
            if result is None:
                continue
            sa = _outcome_score_a(result.outcome)
            self.swiss.record_result(pairing.model_a, pairing.model_b, sa)

        self.pending_pairings = None
        _persist_swiss_state(
            self.state_path,
            self.tournament_id,
            self.swiss,
            self.repo_names,
            self.selected_model_ids,
        )
        print(f"\n[round {rnd}] done. State → {self.state_path}\n")

    def _run_round_parallel(
        self,
        pairings: list[SwissPairing],
        rnd: int,
        seed_base: int,
    ) -> list[tuple[SwissPairing, "MatchResult | None"]]:
        """Run every pairing in a round by pooling ALL of their defense
        sub-turns into one ThreadPoolExecutor.

        Each pairing is first *planned* (deterministic challenge selection +
        defense-cache resolution) without running any Blue agent. The cache-miss
        defense tasks from every pairing are then flattened into a single pool
        and dispatched concurrently (each task = one Blue agent + its
        containerized test runs, all inside the repo's Docker image). When the
        pool drains, each plan is assembled into its MatchResult.

        Parallelism is bounded by ``match.max_workers``. With P pairings, R
        repos, and ``turns_per_player`` k, a fully-cold round contributes up to
        ``P × R × 2 × k`` tasks to the pool.
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
                print(f"  ! planning failed for "
                      f"{_short(p.model_a)} vs {_short(p.model_b)}: {e}")
                plan_by_pairing.append((p, None))
                continue
            plans[plan.match_key] = plan
            plan_by_pairing.append((p, plan))

        # 2) Flatten cache-miss tasks across the whole round into one pool and
        #    register every sub-turn in the live reporter (one row per task).
        reporter = DefenseRoundReporter(round_index=rnd)
        all_tasks: list[DefenseTask] = []
        handles: dict[int, object] = {}  # id(task) → SubturnHandle
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
                # Which competitor is defending (Blue) in this sub-turn.
                if task.side == "b_defends_a":
                    blue_cid, red_cid = p.model_b, p.model_a
                else:
                    blue_cid, red_cid = p.model_a, p.model_b
                sid = f"{task.repo_config.name}:{task.side}:{task.slot}:{task.challenge.challenge_id[:8]}"
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

        # 3) Run the pool under the live TUI. Each task is independent and fully
        #    containerized; worker threads only mutate the reporter via their
        #    SubturnHandle, the Live view re-renders on a timer. console_echo is
        #    disabled so per-step reasoning goes to _swe-duel/blue.log (preserved as
        #    <defenses_dir>/<defense_id>.blue.log), not over the progress bars.
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
                            # Drop this sub-turn; assemble skips its None slot.
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
        print(f"\n  pool drained in {dt:.1f}s ({done}/{len(all_tasks)} sub-turns)\n")

        # 4) Assemble each planned match.
        executed: list[tuple[SwissPairing, MatchResult | None]] = []
        for p, plan in plan_by_pairing:
            if plan is None:
                executed.append((p, None))
                continue
            try:
                m = self.orchestrator.assemble_match(plan)
            except Exception as e:  # noqa: BLE001
                print(f"    ! assembling match failed for "
                      f"{_short(p.model_a)} vs {_short(p.model_b)}: {e}")
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
        subprocess.run(
            [sys.executable, "-m", "swe_duel.cli.build_report"],
            check=False,
        )

    # ── helpers ───────────────────────────────────────────────

    def _ensure_pending_pairings(self) -> list[SwissPairing]:
        if self.pending_pairings is None:
            self.pending_pairings = self.swiss.pair_next_round(
                initial_seed_by=self._initial_seed()
            )
        for sp in self.pending_pairings:
            assert sp.model_b is not None, "byes are disabled"
        return self.pending_pairings

    def _initial_seed(self) -> list[str]:
        if self.swiss.round_history:
            return [p.model_id for p in self.swiss.standings()]
        # Round 1: deterministic shuffle of selected model ids by the seed.
        rng = random.Random(self.seed)
        order = list(self.selected_model_ids)
        rng.shuffle(order)
        return order

    def _elo_snapshot(self, matches: list[MatchResult]) -> dict[str, float]:
        if not matches:
            return {}
        snaps = compute_all_ratings(matches)
        return {m: s.elo for m, s in snaps.items()}

    def _turn_reuse_counts(self, pairing: SwissPairing) -> tuple[int, int]:
        """Count reusable (cached) vs new real-defense turns for this pairing.

        Summed over every repo in the match. Auto-win turns (a Red competitor
        with no challenge for a repo) need no Blue defense and are counted as
        neither reusable nor new."""
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
                f"round {self.swiss.current_round_index} complete"
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
            "    standings  (ls)  Swiss + ELO leaderboard (with round counter)\n"
            "    history    (h)   list past rounds with pairings + outcomes\n"
            "    history N        ELO table as it stood after round N\n"
            "    pairings   (p)   preview next round's pairings (reuse counts)\n"
            "    next       (n)   confirm and run next round's matches\n"
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

    # Resolve --repos against config/repos/, skipping unknown names (e.g. a
    # typo like "python" that has no config). At least one must resolve.
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
    # Per-competitor slot counts aggregated across the selected repos.
    slot_counts = _slot_counts(
        ctx["challenge_store"], all_model_configs, repo_names
    )
    success_counts = _success_counts(
        ctx["challenge_store"], all_model_configs, repo_names
    )

    # ── Resume detection ─────────────────────────────────────
    state_candidates = _find_resume_states(
        ctx["artifact_logger"].tournaments_dir, repo_names
    )
    resumed = False
    swiss: SwissTournament | None = None
    tournament_id: str | None = None
    selected_nicks: list[str]
    prior_matches: list[MatchResult] = []
    match_round: dict[str, int] = {}

    state_path: Path | None = None
    if state_candidates:
        new_value = "__new__"
        choices = []
        for p, payload in state_candidates:
            tid = payload.get("tournament_id", "?")
            rounds_played = len(payload.get("rounds", []))
            n_models = len(payload.get("selected_model_ids", payload.get("players", [])))
            ts = payload.get("timestamp", "")
            title = (
                f"{tid[:8]}…  rounds={rounds_played}  models={n_models}  "
                f"{ts[:19]}  ({p.name})"
            )
            choices.append(questionary.Choice(title=title, value=str(p)))
        choices.append(questionary.Choice(title="Start a new tournament", value=new_value))

        picked = questionary.select(
            f"Found {len(state_candidates)} existing Swiss state(s) for "
            f"repos={','.join(repo_names)}. "
            "Pick one to resume, or start new:",
            choices=choices,
        ).ask()
        if picked is None:
            raise SystemExit("aborted.")
        if picked != new_value:
            from swe_duel.models import split_composite_id

            state_path = Path(picked)
            tmp_payload = json.loads(state_path.read_text())
            # selected_model_ids are participant composite ids
            # (model#harness[#effort#provider]); legacy state files may hold
            # bare model ids or 2-part ids — split tolerates both.
            persisted_ids = tmp_payload.get(
                "selected_model_ids",
                [pl["model_id"] for pl in tmp_payload.get("players", [])],
            )
            swiss, tournament_id, _ = _reconstruct_swiss(state_path, persisted_ids)
            persisted_models = {split_composite_id(cid)[0] for cid in persisted_ids}
            selected_nicks = [
                nick
                for nick, cfg in all_model_configs.items()
                if cfg.model_id in persisted_models
            ]
            ctx["model_configs"] = {n: all_model_configs[n] for n in selected_nicks}
            selected_ids = list(persisted_ids)
            prior_matches, match_round = _load_match_results_for_pairs(
                ctx["artifact_logger"].matches_dir, swiss, repo_names
            )
            resumed = True

            selected_ids = _maybe_add_participants_swiss(
                swiss=swiss,
                selected_ids=selected_ids,
                all_model_configs=all_model_configs,
                slot_counts=slot_counts,
                success_counts=success_counts,
                repo_names=repo_names,
                turns_per_player=turns_per_player,
                challenge_store=ctx["challenge_store"],
                tournament_id=tournament_id,
                state_path=state_path,
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

        _validate_pools(selected_ids, slot_counts)
        # Orchestrator configs are keyed by participant composite id, each with
        # the entrant's effort/provider selection pre-bound.
        ctx["model_configs"] = _merge_model_configs_for_ids(
            all_model_configs, selected_ids
        )
        swiss = SwissTournament(selected_ids)
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

    # ── Seed prompt ──────────────────────────────────────────
    default_seed = args.seed if args.seed is not None else 0
    seed = _prompt_seed(default_seed)

    # ── Build orchestrator ───────────────────────────────────
    # A match spans multiple repos, each with its own image + test command +
    # language adapter, so the orchestrator builds a TestRunner per repo on
    # demand (cached) instead of holding a single bound runner.
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

    # ctx["model_configs"] is keyed by participant composite id
    # (model#harness[#effort#provider]) with each entrant's selection applied,
    # so the orchestrator can resolve a Blue participant's exact ModelConfig
    # from its identity. The Swiss tournament / console stay keyed on the ids.
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

    console = TournamentConsole(
        ctx=ctx,
        match_orchestrator=match_orchestrator,
        repo_names=repo_names,
        seed=seed,
        tournament_id=tournament_id,
        selected_model_ids=selected_model_ids,
        swiss=swiss,
        resumed=resumed,
        prior_matches=prior_matches,
        match_round=match_round,
    )
    console.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
