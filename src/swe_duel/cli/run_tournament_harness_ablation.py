#!/usr/bin/env python
"""Interactive **harness-ablation** round-robin tournament console.

Variant of ``swe_duel/cli/run_tournament_round_robin.py`` that holds the **model**
fixed and treats **agent harnesses** as the tournament participants.

Field
-----
* participants = harness ids (e.g. ``codex``, ``claude-code``, ``mini-swe-agent``)
* after harnesses are chosen, the user picks a **subset** of models to hold
  fixed (not the full bank — fewer models ⇒ cheaper). A model is eligible only
  if it has ``≥ turns_per_player`` attempted slots (``distinct_slot_count``:
  success or failed-only) on **every** selected repo under **every** selected
  harness — so each same-model sub-match has a full turn budget. Validation
  of that rectangular field runs after selection.

Pairings
--------
A pairing between harness A and harness B expands into one real match **per
fixed model**, always same-model:

    A vs B
      model_1 [A]  vs  model_1 [B]
      model_2 [A]  vs  model_2 [B]
      ...

Each underlying match is the usual multi-repo MatchOrchestrator game
(``turns_per_player`` per repo per side) and is logged to ``data/matches/``
under composite competitor ids. The harness-level outcome used for RR
standings is the sum of sub-match totals (same draw margin as arena.yaml).

State files live in a separate namespace:
``harness_ablation_rr_state_*.json`` (``format == "harness_ablation_rr"``).

REPL commands match the round-robin console (status / standings / history /
pairings / next / all / report / quit). Late-entry intake is disabled —
ablations keep a fixed rectangular field.
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
from swe_duel.models import (
    MatchOutcome,
    MatchResult,
    composite_id,
    split_composite_id,
)
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.scoring.rating import compute_all_ratings

from swe_duel.cli.run_tournament import (
    _match_result_from_payload,
    _outcome_score_a,
    _print_slot_table,
    _prompt_seed,
    _slot_counts,
    _success_counts,
)


STATE_FORMAT = "harness_ablation_rr"
STATE_GLOB = "harness_ablation_rr_state_*.json"


# ── selection / validation ────────────────────────────────────


def _nick_for_model_id(all_model_configs: dict, model_id: str) -> str:
    for nick, cfg in all_model_configs.items():
        if cfg.model_id == model_id:
            return nick
    return model_id.rsplit("/", 1)[-1]


def _cell_attempted_slots(
    store,
    model_id: str,
    harness_id: str,
    repo_names: list[str],
    effort: str = "",
    provider: str = "",
) -> dict[str, int]:
    """Per-repo ``distinct_slot_count`` for the pinned participant identity."""
    return {
        rn: store.distinct_slot_count(model_id, rn, harness_id, effort, provider)
        for rn in repo_names
    }


def _model_field_gaps(
    store,
    model_id: str,
    harness_ids: list[str],
    repo_names: list[str],
    turns_per_player: int,
    effort: str = "",
    provider: str = "",
) -> list[str]:
    """Cells where attempted slots < turns_per_player.

    Mirrors the challenge-bank notion used by tournament scripts: a slot is one
    generation effort (success or failed-only). For a fair harness ablation a
    fixed model needs ``turns_per_player`` attempted slots on **every**
    (repo × harness) cell so each same-model sub-match has a full turn budget.
    The whole ablation runs under ONE pinned (effort, provider) identity.
    """
    gaps: list[str] = []
    for h in harness_ids:
        for rn in repo_names:
            n = store.distinct_slot_count(model_id, rn, h, effort, provider)
            if n < turns_per_player:
                gaps.append(f"{h}/{rn}={n}/{turns_per_player}")
    return gaps


def _model_covers_field(
    store,
    model_id: str,
    harness_ids: list[str],
    repo_names: list[str],
    turns_per_player: int,
    effort: str = "",
    provider: str = "",
) -> bool:
    return not _model_field_gaps(
        store, model_id, harness_ids, repo_names, turns_per_player,
        effort, provider,
    )


def _validate_rectangular_field(
    store,
    harness_ids: list[str],
    model_ids: list[str],
    repo_names: list[str],
    turns_per_player: int,
    *,
    all_model_configs: dict | None = None,
    effort: str = "",
    provider: str = "",
) -> None:
    """Require every (model × harness × repo) cell ≥ turns_per_player attempts
    under the ablation's pinned (effort, provider) identity."""
    if len(harness_ids) < 2:
        raise SystemExit("Need at least 2 harnesses for a harness ablation.")
    if len(model_ids) < 1:
        raise SystemExit("Need at least 1 fixed model.")
    missing: list[str] = []
    for mid in model_ids:
        gaps = _model_field_gaps(
            store, mid, harness_ids, repo_names, turns_per_player,
            effort, provider,
        )
        if gaps:
            nick = (
                _nick_for_model_id(all_model_configs, mid)
                if all_model_configs is not None
                else mid
            )
            missing.append(f"{nick} [{mid}]: " + ", ".join(gaps))
    if missing:
        raise SystemExit(
            "Harness ablation requires each fixed model to have "
            f"≥{turns_per_player} attempted slot(s) "
            f"(turns_per_player) on every selected repo under every "
            "selected harness. Shortfalls:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


def _select_harnesses_interactive(
    *,
    slot_counts: dict[str, int],
    success_counts: dict[str, int],
    all_model_configs: dict,
    n_repos: int,
    effort: str = "",
    provider: str = "",
) -> list[str]:
    from swe_duel.agents.harness import (
        HARNESS_IDS,
        harness_display_name,
        harness_supported_efforts,
        harness_supports_provider,
    )

    # A harness is selectable if any model has slots under it (at the pinned
    # identity) AND the harness can express that identity.
    choices = []
    for hid in HARNESS_IDS:
        if effort and harness_supported_efforts(hid) is not None \
                and effort not in harness_supported_efforts(hid):
            choices.append(
                questionary.Choice(
                    title=f"{harness_display_name(hid):<16}  (cannot express effort {effort!r})",
                    value=hid,
                    disabled=f"cannot express reasoning effort {effort!r}",
                )
            )
            continue
        if provider and not harness_supports_provider(hid):
            choices.append(
                questionary.Choice(
                    title=f"{harness_display_name(hid):<16}  (cannot pin provider {provider!r})",
                    value=hid,
                    disabled=f"cannot pin provider {provider!r}",
                )
            )
            continue
        models_with = sum(
            1
            for cfg in all_model_configs.values()
            if slot_counts.get(
                composite_id(cfg.model_id, hid, effort, provider), 0
            ) > 0
        )
        succ_models = sum(
            1
            for cfg in all_model_configs.values()
            if success_counts.get(
                composite_id(cfg.model_id, hid, effort, provider), 0
            ) > 0
        )
        title = (
            f"{harness_display_name(hid):<16}  "
            f"({succ_models} models w/ challenges, {models_with} w/ slots "
            f"across {n_repos} repo(s))"
        )
        disabled = None if models_with > 0 else "no models with slots"
        choices.append(
            questionary.Choice(title=title, value=hid, disabled=disabled)
        )

    selected = questionary.checkbox(
        "Select harnesses (participants) — space=toggle, enter=confirm:",
        choices=choices,
    ).ask()
    if selected is None:
        raise SystemExit("aborted.")
    if len(selected) < 2:
        raise SystemExit("Need at least 2 harnesses.")
    return list(selected)


def _select_models_interactive(
    *,
    store,
    harness_ids: list[str],
    all_model_configs: dict,
    repo_names: list[str],
    turns_per_player: int,
    effort: str = "",
    provider: str = "",
) -> list[str]:
    """Pick a **subset** of models that fully cover the harness × repo field
    at the pinned (effort, provider) identity.

    A model is selectable only when every (harness × repo) cell has
    ``distinct_slot_count >= turns_per_player`` attempted slots — same bank
    accounting as ``run_tournament.py`` (success + failed-only efforts), but
    requiring a full turn budget on every cell so same-model sub-matches are
    comparable. Checkbox starts empty so cost stays under user control.
    """
    from swe_duel.agents.harness import harness_display_name

    n_repos = len(repo_names)
    print(
        f"\n  Pick a subset of models to hold fixed "
        f"(turns_per_player={turns_per_player}).\n"
        f"  Eligible = ≥{turns_per_player} attempted slot(s) on every "
        f"selected repo under every selected harness.\n"
        f"  Only models you toggle on will be played (fewer ⇒ cheaper).\n"
    )

    choices = []
    n_eligible = 0
    for nick, cfg in all_model_configs.items():
        gaps = _model_field_gaps(
            store, cfg.model_id, harness_ids, repo_names, turns_per_player,
            effort, provider,
        )
        # Summary: min attempted slots across the whole field + per-harness
        # (min over repos) so shortfalls are obvious.
        harness_summaries: list[str] = []
        min_attempted = turns_per_player
        for h in harness_ids:
            per_repo = _cell_attempted_slots(
                store, cfg.model_id, h, repo_names, effort, provider
            )
            h_min = min(per_repo.values()) if per_repo else 0
            min_attempted = min(min_attempted, h_min)
            succ_repos = sum(
                1
                for rn in repo_names
                if store.count_in_pool(cfg.model_id, rn, h, effort, provider) > 0
            )
            harness_summaries.append(
                f"{harness_display_name(h)}:min={h_min},succ_repos={succ_repos}/{n_repos}"
            )
        eligible = not gaps
        if eligible:
            n_eligible += 1
            disabled = None
            flag = "ok"
        else:
            short = "; ".join(gaps[:4])
            if len(gaps) > 4:
                short += f"; +{len(gaps) - 4} more"
            disabled = f"need ≥{turns_per_player} attempted slots/cell ({short})"
            flag = "short"
        title = (
            f"{nick:<28}  [{flag}]  field_min={min_attempted}/{turns_per_player}  "
            f"[{', '.join(harness_summaries)}]  [{cfg.model_id}]"
        )
        choices.append(
            questionary.Choice(
                title=title, value=cfg.model_id, disabled=disabled
            )
        )

    if n_eligible == 0:
        raise SystemExit(
            f"No model has ≥{turns_per_player} attempted slot(s) on every "
            f"repo under every selected harness "
            f"({', '.join(harness_ids)} × {', '.join(repo_names)}). "
            "Generate more challenges first."
        )

    selected = questionary.checkbox(
        f"Select a subset of eligible models ({n_eligible} eligible) "
        "(space=toggle, enter=confirm — nothing is pre-selected):",
        choices=choices,
    ).ask()
    if selected is None:
        raise SystemExit("aborted.")
    if not selected:
        raise SystemExit(
            "Need at least 1 fixed model (pick a subset; do not leave empty)."
        )
    return list(selected)


def _resolve_cli_models(
    specs: list[str], all_model_configs: dict
) -> list[str]:
    """Map nicknames / bare model ids to canonical model_id strings."""
    by_nick = {nick: cfg.model_id for nick, cfg in all_model_configs.items()}
    by_id = {cfg.model_id: cfg.model_id for cfg in all_model_configs.values()}
    out: list[str] = []
    bad: list[str] = []
    for s in specs:
        if s in by_nick:
            mid = by_nick[s]
        elif s in by_id:
            mid = by_id[s]
        else:
            bad.append(s)
            continue
        if mid not in out:
            out.append(mid)
    if bad:
        raise SystemExit(
            f"unknown --models {bad!r}; known nicks: {sorted(by_nick)}"
        )
    return out


# ── state / resume ────────────────────────────────────────────


def _persist_state(
    path: Path,
    tournament_id: str,
    rr: RoundRobinTournament,
    repo_names: list[str],
    harness_ids: list[str],
    fixed_model_ids: list[str],
    seed: int,
    effort: str = "",
    provider: str = "",
) -> None:
    payload = {
        "tournament_id": tournament_id,
        "format": STATE_FORMAT,
        "repo_name": ",".join(repo_names),
        "repo_names": list(repo_names),
        "selected_harness_ids": list(harness_ids),
        # Pinned participant identity shared by every fixed model.
        "fixed_reasoning_effort": effort or "",
        "fixed_provider": provider or "",
        # Players of the RR schedule are harness ids (alias selected for
        # compatibility with generic RR helpers that read selected_model_ids).
        "selected_model_ids": list(harness_ids),
        "base_model_ids": list(rr.base_model_ids),
        "fixed_model_ids": list(fixed_model_ids),
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
            [{"model_a": pp.model_a, "model_b": pp.model_b} for pp in rnd]
            for rnd in rr.round_history
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _find_resume_states(
    tournaments_dir: Path, repo_names: list[str]
) -> list[tuple[Path, dict]]:
    want = set(repo_names)
    candidates = sorted(
        tournaments_dir.glob(STATE_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    results: list[tuple[Path, dict]] = []
    for p in candidates:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if payload.get("format") not in (None, STATE_FORMAT):
            continue
        repos = payload.get("repo_names")
        if not repos:
            rn = payload.get("repo_name", "")
            repos = [r for r in rn.split(",") if r] if rn else []
        if set(repos) == want:
            results.append((p, payload))
    return results


def _reconstruct_rr(
    state_path: Path, harness_ids: list[str], seed: int
) -> tuple[RoundRobinTournament, str, list[str], list[str]]:
    from swe_duel.engine.swiss import SwissPlayer

    payload = json.loads(state_path.read_text())
    base_ids = list(payload.get("base_model_ids") or harness_ids)
    rr = RoundRobinTournament(base_ids, seed=seed)

    for p in payload.get("players", []):
        mid = p["model_id"]
        if mid not in rr.players:
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
    selected = list(payload.get("selected_harness_ids") or harness_ids)
    fixed = list(payload.get("fixed_model_ids") or [])
    return rr, tournament_id, selected, fixed


def _load_submatch_results(
    matches_dir: Path,
    rr: RoundRobinTournament,
    repo_names: list[str],
    fixed_model_ids: list[str],
    effort: str = "",
    provider: str = "",
) -> tuple[list[MatchResult], dict[str, int]]:
    """Load underlying composite-id matches for each harness pairing × model."""
    want_repos = set(repo_names)
    results: list[MatchResult] = []
    match_round: dict[str, int] = {}
    used: set[str] = set()

    # Pre-scan match files once.
    payloads: list[dict] = []
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
        payloads.append(payload)

    for round_index, pairings in enumerate(rr.round_history, start=1):
        for sp in pairings:
            if sp.model_b is None:
                continue
            h_a, h_b = sp.model_a, sp.model_b
            for mid in fixed_model_ids:
                want_pair = {
                    composite_id(mid, h_a, effort, provider),
                    composite_id(mid, h_b, effort, provider),
                }
                candidates = [
                    pl
                    for pl in payloads
                    if pl.get("match_id") not in used
                    and {
                        pl.get("model_a_id"),
                        pl.get("model_b_id"),
                    }
                    == want_pair
                ]
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x.get("timestamp", ""))
                payload = candidates[0]
                used.add(payload["match_id"])
                try:
                    m = _match_result_from_payload(payload)
                except Exception:
                    continue
                # Normalize side so model_a is always h_a for this mid.
                if split_composite_id(m.model_a_id)[1] != h_a:
                    m = MatchResult(
                        match_id=m.match_id,
                        model_a_id=m.model_b_id,
                        model_b_id=m.model_a_id,
                        repo_name=m.repo_name,
                        turns=m.turns,
                        model_a_total=m.model_b_total,
                        model_b_total=m.model_a_total,
                        outcome=(
                            MatchOutcome.MODEL_B_WINS
                            if m.outcome == MatchOutcome.MODEL_A_WINS
                            else MatchOutcome.MODEL_A_WINS
                            if m.outcome == MatchOutcome.MODEL_B_WINS
                            else MatchOutcome.DRAW
                        ),
                        duration_seconds=m.duration_seconds,
                        total_cost_usd=m.total_cost_usd,
                        timestamp=m.timestamp,
                    )
                results.append(m)
                match_round[m.match_id] = round_index
    return results, match_round


def _submatches_for_pairing(
    matches: list[MatchResult],
    h_a: str,
    h_b: str,
    fixed_model_ids: list[str],
) -> list[MatchResult]:
    """Return sub-matches for harness pairing, ordered by fixed_model_ids."""
    by_model: dict[str, MatchResult] = {}
    for m in matches:
        a_mid, a_h, _a_effort, _a_provider = split_composite_id(m.model_a_id)
        b_mid, b_h, _b_effort, _b_provider = split_composite_id(m.model_b_id)
        if a_mid != b_mid:
            continue
        if {a_h, b_h} != {h_a, h_b}:
            continue
        # orient to h_a on the left
        if a_h == h_a and b_h == h_b:
            by_model[a_mid] = m
        elif a_h == h_b and b_h == h_a:
            by_model[a_mid] = MatchResult(
                match_id=m.match_id,
                model_a_id=m.model_b_id,
                model_b_id=m.model_a_id,
                repo_name=m.repo_name,
                turns=m.turns,
                model_a_total=m.model_b_total,
                model_b_total=m.model_a_total,
                outcome=(
                    MatchOutcome.MODEL_B_WINS
                    if m.outcome == MatchOutcome.MODEL_A_WINS
                    else MatchOutcome.MODEL_A_WINS
                    if m.outcome == MatchOutcome.MODEL_B_WINS
                    else MatchOutcome.DRAW
                ),
                duration_seconds=m.duration_seconds,
                total_cost_usd=m.total_cost_usd,
                timestamp=m.timestamp,
            )
    return [by_model[mid] for mid in fixed_model_ids if mid in by_model]


def _aggregate_harness_outcome(
    sub_matches: list[MatchResult],
    draw_margin: float,
) -> tuple[float, float, MatchOutcome, float]:
    """Sum sub-match totals → (score_a, score_b, outcome, cost)."""
    score_a = sum(m.model_a_total for m in sub_matches)
    score_b = sum(m.model_b_total for m in sub_matches)
    cost = sum(m.total_cost_usd for m in sub_matches)
    if abs(score_a - score_b) <= draw_margin:
        outcome = MatchOutcome.DRAW
    elif score_a > score_b:
        outcome = MatchOutcome.MODEL_A_WINS
    else:
        outcome = MatchOutcome.MODEL_B_WINS
    return score_a, score_b, outcome, cost


def _synthetic_harness_match(
    h_a: str,
    h_b: str,
    sub_matches: list[MatchResult],
    draw_margin: float,
    repo_label: str,
) -> MatchResult:
    """In-memory MatchResult keyed by harness ids (for ELO / history summary)."""
    score_a, score_b, outcome, cost = _aggregate_harness_outcome(
        sub_matches, draw_margin
    )
    duration = sum(m.duration_seconds for m in sub_matches)
    ts = (
        max(m.timestamp for m in sub_matches)
        if sub_matches
        else datetime.now(timezone.utc)
    )
    return MatchResult(
        match_id=f"harness:{h_a}|{h_b}:{sub_matches[0].match_id if sub_matches else uuid.uuid4()}",
        model_a_id=h_a,
        model_b_id=h_b,
        repo_name=repo_label,
        turns=[],
        model_a_total=score_a,
        model_b_total=score_b,
        outcome=outcome,
        duration_seconds=duration,
        total_cost_usd=cost,
        timestamp=ts,
    )


# ── REPL ──────────────────────────────────────────────────────


class HarnessAblationConsole:
    def __init__(
        self,
        ctx: dict,
        match_orchestrator: MatchOrchestrator,
        repo_names: list[str],
        seed: int,
        tournament_id: str,
        harness_ids: list[str],
        fixed_model_ids: list[str],
        rr: RoundRobinTournament,
        resumed: bool,
        prior_matches: list[MatchResult],
        match_round: dict[str, int],
        model_nick_by_id: dict[str, str],
        effort: str = "",
        provider: str = "",
    ) -> None:
        self.ctx = ctx
        self.orchestrator = match_orchestrator
        self.repo_names = list(repo_names)
        self.repo_cfgs = [ctx["repo_configs"][rn] for rn in repo_names]
        self.repo_label = ",".join(repo_names)
        self.seed = seed
        self.tournament_id = tournament_id
        self.harness_ids = list(harness_ids)
        self.fixed_model_ids = list(fixed_model_ids)
        # The ablation's pinned participant identity (same effort/provider for
        # every fixed model; empty = default identity).
        self.effort = effort or ""
        self.provider = provider or ""
        self.model_nick_by_id = dict(model_nick_by_id)
        self.matches: list[MatchResult] = list(prior_matches)
        self.match_round: dict[str, int] = dict(match_round)
        self.resumed = resumed
        self.rr = rr
        self.pending_pairings: list[SwissPairing] | None = None
        self.state_path = (
            ctx["artifact_logger"].tournaments_dir
            / f"harness_ablation_rr_state_{self.tournament_id}.json"
        )
        self.draw_margin = float(getattr(match_orchestrator, "draw_margin", 0.0))

    def _model_label(self, mid: str, width: int = 28) -> str:
        nick = self.model_nick_by_id.get(mid, mid.rsplit("/", 1)[-1])
        return nick if len(nick) <= width else nick[: width - 1] + "…"

    def _hlabel(self, hid: str, width: int = 16) -> str:
        from swe_duel.agents.harness import harness_display_name

        d = harness_display_name(hid)
        return d if len(d) <= width else d[: width - 1] + "…"

    def _persist(self) -> None:
        _persist_state(
            self.state_path,
            self.tournament_id,
            self.rr,
            self.repo_names,
            self.harness_ids,
            self.fixed_model_ids,
            self.seed,
            effort=self.effort,
            provider=self.provider,
        )

    def _harness_matches(self) -> list[MatchResult]:
        """One synthetic MatchResult per completed harness pairing."""
        out: list[MatchResult] = []
        for pairings in self.rr.round_history:
            for sp in pairings:
                if sp.model_b is None:
                    continue
                subs = _submatches_for_pairing(
                    self.matches,
                    sp.model_a,
                    sp.model_b,
                    self.fixed_model_ids,
                )
                if not subs:
                    continue
                out.append(
                    _synthetic_harness_match(
                        sp.model_a,
                        sp.model_b,
                        subs,
                        self.draw_margin,
                        self.repo_label,
                    )
                )
        return out

    # ── commands ──────────────────────────────────────────────

    def cmd_status(self) -> None:
        n = len(self.harness_ids)
        rec = self.rr.recommended_rounds
        total_pairs = n * (n - 1) // 2 if n > 1 else 0
        print()
        print(f"  Tournament ID : {self.tournament_id}")
        print("  Format        : harness-ablation round-robin")
        print(f"  Repos         : {self.repo_label}")
        print(
            f"  Harnesses     : {n}  "
            f"[{', '.join(self._hlabel(h) for h in self.harness_ids)}]"
        )
        print(
            f"  Fixed models  : {len(self.fixed_model_ids)}  "
            f"[{', '.join(self._model_label(m) for m in self.fixed_model_ids)}]"
        )
        print(
            f"  Total pairs   : {total_pairs} harness pairings × "
            f"{len(self.fixed_model_ids)} models = "
            f"{total_pairs * len(self.fixed_model_ids)} sub-matches"
        )
        print(f"  Recommended   : {rec} rounds")
        print(f"  Current round : {self.rr.current_round_index} of {rec}")
        print(
            f"  Sub-matches   : {len(self.matches)}  "
            f"(harness aggregates played via record_result)"
        )
        print(f"  Seed          : {self.seed}")
        print(
            f"  Total cost    : "
            f"${sum(m.total_cost_usd for m in self.matches):.4f}"
        )
        print()

    def cmd_standings(self) -> None:
        completed = self.rr.current_round_index
        rec = self.rr.recommended_rounds
        std = self.rr.standings()
        elo_map = self._elo_snapshot(self._harness_matches())
        print()
        print(
            f"  ── Round {completed} of {rec} complete — "
            f"Round {completed + 1} next ──"
        )
        print(
            f"  {'#':>2}  {'Harness':<22}  {'Pts':>6}  {'ELO':>7}  {'Played':>6}"
        )
        print(f"  {'-' * 2}  {'-' * 22}  {'-' * 6}  {'-' * 7}  {'-' * 6}")
        for i, p in enumerate(std, 1):
            elo = elo_map.get(p.model_id, 1500.0)
            played = len(p.opponents) + (1 if p.had_bye else 0)
            print(
                f"  {i:>2}  {self._hlabel(p.model_id, 22):<22}  "
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
            hmatches = self._harness_matches()
            # Filter by rounds using underlying match_round of first sub-match.
            prefix: list[MatchResult] = []
            for hm in hmatches:
                # synthetic id embeds a real sub-match id after last ':'
                real_id = hm.match_id.rsplit(":", 1)[-1]
                rnd = self.match_round.get(real_id)
                if rnd is not None and rnd <= n:
                    prefix.append(hm)
            if not prefix:
                print(f"  no harness matches recorded for rounds 1..{n}")
                return
            elo_map = self._elo_snapshot(prefix)
            print(f"\n  ── ELO table after Round {n} ──")
            print(f"  {'#':>2}  {'Harness':<22}  {'ELO':>7}")
            print(f"  {'-' * 2}  {'-' * 22}  {'-' * 7}")
            for i, (hid, elo) in enumerate(
                sorted(elo_map.items(), key=lambda kv: -kv[1]), 1
            ):
                print(f"  {i:>2}  {self._hlabel(hid, 22):<22}  {elo:>7.1f}")
            print()
            return

        if not self.matches:
            print("  no rounds played yet.")
            return
        by_round: dict[int, list[SwissPairing]] = {}
        for rnd_i, pairings in enumerate(self.rr.round_history, start=1):
            by_round[rnd_i] = pairings
        for r in sorted(by_round):
            print(f"\n  Round {r}:")
            for sp in by_round[r]:
                if sp.model_b is None:
                    print(f"    {self._hlabel(sp.model_a):<18}  — BYE")
                    continue
                subs = _submatches_for_pairing(
                    self.matches, sp.model_a, sp.model_b, self.fixed_model_ids
                )
                if not subs:
                    print(
                        f"    {self._hlabel(sp.model_a):<18} vs "
                        f"{self._hlabel(sp.model_b):<18}  (no sub-matches found)"
                    )
                    continue
                sa, sb, outcome, cost = _aggregate_harness_outcome(
                    subs, self.draw_margin
                )
                print(
                    f"    {self._hlabel(sp.model_a):<18} vs "
                    f"{self._hlabel(sp.model_b):<18}  → {outcome.value:<14}  "
                    f"A={sa:.2f} B={sb:.2f}  ${cost:.4f}"
                )
                for m in subs:
                    mid, _h, _e, _pv = split_composite_id(m.model_a_id)
                    print(
                        f"       · {self._model_label(mid):<22}  "
                        f"{m.outcome.value:<14}  "
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
        print()
        print(
            f"  Round {rnd} pairings ({len(pairings)} harness matchups × "
            f"{len(self.fixed_model_ids)} models):"
        )
        playable = [p for p in pairings if p.model_b is not None]
        counts = {id(p): self._turn_reuse_counts(p) for p in playable}
        for i, p in enumerate(pairings, 1):
            if p.model_b is None:
                print(
                    f"   {i:>2}. {self._hlabel(p.model_a):<22}  — BYE (sits out)"
                )
                continue
            a_pts = self.rr.players[p.model_a].points
            b_pts = self.rr.players[p.model_b].points
            reuse, new = counts[id(p)]
            print(
                f"   {i:>2}. {self._hlabel(p.model_a):<18}  ({a_pts:.1f})  "
                f"vs  {self._hlabel(p.model_b):<18}  ({b_pts:.1f})  "
                f"— {reuse} reusable / {new} new"
            )
            for mid in self.fixed_model_ids:
                print(
                    f"       · {self._model_label(mid):<22}  "
                    f"{self._model_label(mid)} [{self._hlabel(p.model_a)}] vs "
                    f"{self._model_label(mid)} [{self._hlabel(p.model_b)}]"
                )
        total_reuse = sum(r for r, _ in counts.values())
        total_new = sum(n for _, n in counts.values())
        print(
            f"\n  Totals: {total_reuse} reusable / {total_new} new "
            f"({total_reuse + total_new} defense turns this round)"
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
            self.rr.commit_round(pairings)
            self.pending_pairings = None
            self._persist()
            print(f"\n[round {rnd}] no playable pairings; round committed.\n")
            return
        n_sub = len(playable) * len(self.fixed_model_ids)
        confirm = questionary.confirm(
            f"Run Round {rnd} ({len(playable)} harness pairings / "
            f"{n_sub} sub-matches)?",
            default=True,
        ).ask()
        if not confirm:
            print("  cancelled.\n")
            return

        print(
            f"\n[round {rnd}] running {len(playable)} harness pairings "
            f"({n_sub} sub-matches)...\n"
        )
        seed_base = self.seed + 1000 * rnd
        executed = self._run_round_parallel(playable, rnd, seed_base)
        self.rr.commit_round(pairings)
        for pairing, sub_matches in executed:
            for m in sub_matches:
                self.matches.append(m)
                self.match_round[m.match_id] = rnd
            if not sub_matches or pairing.model_b is None:
                continue
            _, _, outcome, _ = _aggregate_harness_outcome(
                sub_matches, self.draw_margin
            )
            self.rr.record_result(
                pairing.model_a, pairing.model_b, _outcome_score_a(outcome)
            )

        self.pending_pairings = None
        self._persist()
        print(f"\n[round {rnd}] done. State → {self.state_path}\n")

    def cmd_all(self) -> None:
        try:
            batches = self._collect_remaining_batches()
        except RuntimeError as e:
            print(f"\n  [tournament complete] {e}\n")
            return

        scheduled: list[tuple[int, SwissPairing]] = []
        print()
        for rnd, pairings in batches:
            playable = [p for p in pairings if p.model_b is not None]
            print(
                f"  Round {rnd} ({len(pairings)} harness pairings"
                f"{'' if playable else '; byes only'}):"
            )
            counts = {id(p): self._turn_reuse_counts(p) for p in playable}
            for i, p in enumerate(pairings, 1):
                if p.model_b is None:
                    print(
                        f"   {i:>2}. {self._hlabel(p.model_a):<22}  — BYE"
                    )
                    continue
                a_pts = self.rr.players[p.model_a].points
                b_pts = self.rr.players[p.model_b].points
                reuse, new = counts[id(p)]
                print(
                    f"   {i:>2}. {self._hlabel(p.model_a):<18}  ({a_pts:.1f})  "
                    f"vs  {self._hlabel(p.model_b):<18}  ({b_pts:.1f})  "
                    f"— {reuse} reusable / {new} new"
                )
                scheduled.append((rnd, p))
            if playable:
                total_reuse = sum(r for r, _ in counts.values())
                total_new = sum(n for _, n in counts.values())
                print(
                    f"     subtotal: {total_reuse} reusable / {total_new} new"
                )
            print()

        if not scheduled:
            for _rnd, pairings in batches:
                self.rr.commit_round(pairings)
            self.pending_pairings = None
            self._persist()
            print("[all] no playable pairings; remaining rounds committed.\n")
            return

        n_matches = len(scheduled)
        n_sub = n_matches * len(self.fixed_model_ids)
        n_rounds = len(batches)
        confirm = questionary.confirm(
            f"Run all remaining ({n_matches} harness pairings / {n_sub} "
            f"sub-matches across {n_rounds} round(s)) in one parallel batch?",
            default=True,
        ).ask()
        if not confirm:
            print("  cancelled.\n")
            return

        print(f"\n[all] running {n_matches} harness pairings...\n")
        playable = [p for _, p in scheduled]
        seed_base = self.seed + 1000 * (self.rr.current_round_index + 1)
        tui_rnd = batches[0][0]
        executed = self._run_round_parallel(playable, tui_rnd, seed_base)
        result_by_pair: dict[tuple[str, str], list[MatchResult]] = {}
        for p, subs in executed:
            result_by_pair[(p.model_a, p.model_b or "")] = subs

        for rnd, pairings in batches:
            for p in pairings:
                if p.model_b is None:
                    continue
                subs = result_by_pair.get((p.model_a, p.model_b), [])
                for m in subs:
                    self.matches.append(m)
                    self.match_round[m.match_id] = rnd
            self.rr.commit_round(pairings)
            for p in pairings:
                if p.model_b is None:
                    continue
                subs = result_by_pair.get((p.model_a, p.model_b), [])
                if not subs:
                    continue
                _, _, outcome, _ = _aggregate_harness_outcome(
                    subs, self.draw_margin
                )
                self.rr.record_result(
                    p.model_a, p.model_b, _outcome_score_a(outcome)
                )

        self.pending_pairings = None
        self._persist()
        print(
            f"\n[all] done ({n_matches} harness pairings / {n_sub} sub-matches). "
            f"State → {self.state_path}\n"
        )

    def _collect_remaining_batches(
        self,
    ) -> list[tuple[int, list[SwissPairing]]]:
        self.pending_pairings = None
        batches: list[tuple[int, list[SwissPairing]]] = []
        base_remaining = self.rr.remaining_base_rounds()
        start_idx = self.rr.current_round_index
        for offset, pairings in enumerate(base_remaining):
            batches.append((start_idx + offset + 1, pairings))
        if not batches:
            raise RuntimeError(
                "Harness-ablation schedule complete: every harness pairing "
                "has been played."
            )
        return batches

    def _expand_pairing(
        self, pairing: SwissPairing
    ) -> list[tuple[str, str, str]]:
        """(model_id, composite_a, composite_b) for every fixed model."""
        assert pairing.model_b is not None
        return [
            (
                mid,
                composite_id(mid, pairing.model_a, self.effort, self.provider),
                composite_id(mid, pairing.model_b, self.effort, self.provider),
            )
            for mid in self.fixed_model_ids
        ]

    def _run_round_parallel(
        self,
        pairings: list[SwissPairing],
        rnd: int,
        seed_base: int,
    ) -> list[tuple[SwissPairing, list[MatchResult]]]:
        """Expand each harness pairing into model sub-matches; pool all defenses."""
        # 1) Plan every (pairing, model) sub-match.
        plan_entries: list[
            tuple[SwissPairing, str, MatchPlan | None]
        ] = []  # pairing, model_id, plan
        seed_i = 0
        for p in pairings:
            for mid, ca, cb in self._expand_pairing(p):
                try:
                    plan = self.orchestrator.plan_match(
                        model_a_id=ca,
                        model_b_id=cb,
                        repo_config=self.repo_cfgs,
                        seed=seed_base + seed_i,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  ! planning failed for "
                        f"{self._model_label(mid)} "
                        f"[{self._hlabel(p.model_a)}] vs "
                        f"[{self._hlabel(p.model_b or '')}]: {e}"
                    )
                    plan_entries.append((p, mid, None))
                    seed_i += 1
                    continue
                plan_entries.append((p, mid, plan))
                seed_i += 1

        # 2) Flatten cache-miss tasks + register TUI rows.
        reporter = DefenseRoundReporter(round_index=rnd)
        all_tasks: list[DefenseTask] = []
        handles: dict[int, object] = {}
        plans_by_key: dict[str, MatchPlan] = {}
        for p, mid, plan in plan_entries:
            if plan is None:
                continue
            plans_by_key[plan.match_key] = plan
            label = (
                f"{self._model_label(mid)} · "
                f"{self._hlabel(p.model_a)} vs {self._hlabel(p.model_b or '')}"
            )
            reporter.register_match(plan.match_key, label)
            hits = plan.cache_hits()
            if hits:
                reporter.finish_match(
                    plan.match_key,
                    SubturnStatus.RUNNING,
                    detail=f"{hits} cached",
                )
            for task in plan.pending_tasks:
                if task.side == "b_defends_a":
                    blue_h, red_h = p.model_b, p.model_a
                else:
                    blue_h, red_h = p.model_a, p.model_b
                sid = (
                    f"{task.repo_config.name}:{task.side}:{task.slot}:"
                    f"{task.challenge.challenge_id[:8]}"
                )
                handle = reporter.register_subturn(
                    plan.match_key,
                    sid,
                    label=(
                        f"{self._model_label(mid)} blue=[{self._hlabel(blue_h or '')}] "
                        f"defends [{self._hlabel(red_h or '')}] "
                        f"{task.repo_config.name}"
                    ),
                    blue_model=task.blue_model_id,
                    repo=task.repo_config.name,
                )
                handles[id(task)] = handle
                all_tasks.append(task)

        total_hits = sum(
            plan.cache_hits()
            for _, _, plan in plan_entries
            if plan is not None
        )
        max_workers = self.orchestrator.config.match.max_workers
        effective = min(max_workers, len(all_tasks)) or 1
        n_plans = sum(1 for _, _, plan in plan_entries if plan is not None)
        print(
            f"  pool: {len(all_tasks)} defense sub-turns to run "
            f"({total_hits} reused from cache) across {n_plans} sub-matches "
            f"— max_workers={max_workers}, using {effective}\n"
        )

        # 3) Drain pool.
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
                            plans_by_key[task.match_key], task, result
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
            f"\n  pool drained in {dt:.1f}s "
            f"({done}/{len(all_tasks)} sub-turns)\n"
        )

        # 4) Assemble sub-matches, group by harness pairing.
        subs_by_pairing: dict[tuple[str, str | None], list[MatchResult]] = {
            (p.model_a, p.model_b): [] for p in pairings
        }
        for p, mid, plan in plan_entries:
            if plan is None:
                continue
            try:
                m = self.orchestrator.assemble_match(plan)
            except Exception as e:  # noqa: BLE001
                print(
                    f"    ! assemble failed for {self._model_label(mid)} "
                    f"[{self._hlabel(p.model_a)}] vs "
                    f"[{self._hlabel(p.model_b or '')}]: {e}"
                )
                continue
            print(
                f"  → {self._model_label(mid):<22} "
                f"[{self._hlabel(p.model_a)}] vs [{self._hlabel(p.model_b or '')}]: "
                f"{m.outcome.value}  A={m.model_a_total:.2f} "
                f"B={m.model_b_total:.2f}  ${m.total_cost_usd:.4f}"
            )
            subs_by_pairing[(p.model_a, p.model_b)].append(m)

        executed: list[tuple[SwissPairing, list[MatchResult]]] = []
        for p in pairings:
            subs = subs_by_pairing.get((p.model_a, p.model_b), [])
            if p.model_b is not None and subs:
                sa, sb, outcome, cost = _aggregate_harness_outcome(
                    subs, self.draw_margin
                )
                print(
                    f"  ★ harness {self._hlabel(p.model_a)} vs "
                    f"{self._hlabel(p.model_b)}: {outcome.value}  "
                    f"A={sa:.2f} B={sb:.2f}  ${cost:.4f}  "
                    f"({len(subs)} models)"
                )
            executed.append((p, subs))
        return executed

    def cmd_report(self) -> None:
        print("\n[report] invoking swe-duel-report ...\n")
        env = {**os.environ, "SWE_DUEL_REPORT_FORMAT": "harness-ablation-rr"}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "swe_duel.cli.build_report",
            ],
            check=False,
            env=env,
        )

    def _ensure_pending_pairings(self) -> list[SwissPairing]:
        if self.pending_pairings is None:
            self.pending_pairings = self.rr.pair_next_round()
        return self.pending_pairings

    def _elo_snapshot(self, matches: list[MatchResult]) -> dict[str, float]:
        if not matches:
            return {}
        snaps = compute_all_ratings(matches)
        return {m: s.elo for m, s in snaps.items()}

    def _turn_reuse_counts(self, pairing: SwissPairing) -> tuple[int, int]:
        """Cached vs new defense turns across all fixed-model sub-matches."""
        turns_per = self.orchestrator.config.match.turns_per_player
        store = self.orchestrator.challenge_store
        al = self.orchestrator.artifact_logger
        reuse = 0
        new = 0
        assert pairing.model_b is not None
        for mid in self.fixed_model_ids:
            for red_h, blue_h in (
                (pairing.model_a, pairing.model_b),
                (pairing.model_b, pairing.model_a),
            ):
                for repo_name in self.repo_names:
                    challenge_ids = store.list_ids_in_index_order(
                        mid,
                        repo_name,
                        red_harness_id=red_h,
                    )[:turns_per]
                    for cid in challenge_ids:
                        if al.has_defense(
                            challenge_id=cid,
                            blue_model_id=mid,
                            blue_harness_id=blue_h,
                        ):
                            reuse += 1
                        else:
                            new += 1
        return reuse, new

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
                line = input("swe-duel-ablation> ").strip()
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
            "\n  Commands (harness-ablation RR):\n"
            "    status     (s)   tournament overview\n"
            "    standings  (ls)  harness points + ELO leaderboard\n"
            "    history    (h)   past rounds (harness + per-model sub-matches)\n"
            "    history N        ELO table as it stood after round N\n"
            "    pairings   (p)   next harness pairings + model expansions\n"
            "    next       (n)   run next round\n"
            "    all        (a)   run all remaining rounds in one parallel batch\n"
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
        help=(
            "Subset of model nicknames/IDs to hold fixed (cost control — "
            "omit to pick interactively after harnesses). Each chosen model "
            "must have challenge-bank slots under every selected harness."
        ),
    )
    parser.add_argument(
        "--harnesses",
        nargs="+",
        default=None,
        help=(
            "Harnesses to rank against each other (tournament participants). "
            "Omit to pick interactively. E.g. mini-swe-agent codex claude-code."
        ),
    )
    parser.add_argument(
        "--effort",
        default="",
        help=(
            "Pin ONE reasoning effort for every fixed model in the ablation "
            "(must be in each model's reasoning_efforts menu in models.yaml "
            "and expressible by every selected harness). Default: the model "
            "default (legacy identity)."
        ),
    )
    parser.add_argument(
        "--provider",
        default="",
        help=(
            "Pin ONE OpenRouter provider for every fixed model (must be in "
            "each model's providers menu; only mini-swe-agent/openhands can "
            "pin a provider). Default: OpenRouter auto-routing."
        ),
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        required=True,
        help="One or more repos; each sub-match spans ALL of them.",
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--prompt-dir", default=default_prompt_dir())
    parser.add_argument(
        "--turns-per-player",
        "--turns-per-agent",
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
        help="Override arena.yaml match.max_workers.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    args_for_setup = argparse.Namespace(**vars(args))
    args_for_setup.models = None
    ctx = setup(args_for_setup)
    arena_cfg = ctx["arena_config"]

    if args.turns_per_player is not None:
        arena_cfg.match.turns_per_player = args.turns_per_player
    if args.match_workers is not None:
        arena_cfg.match.max_workers = args.match_workers
    turns_per_player = arena_cfg.match.turns_per_player

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
    slot_counts = _slot_counts(
        ctx["challenge_store"], all_model_configs, repo_names
    )
    success_counts = _success_counts(
        ctx["challenge_store"], all_model_configs, repo_names
    )

    # ── Resume ────────────────────────────────────────────────
    state_candidates = _find_resume_states(
        ctx["artifact_logger"].tournaments_dir, repo_names
    )
    resumed = False
    rr: RoundRobinTournament | None = None
    tournament_id: str | None = None
    prior_matches: list[MatchResult] = []
    match_round: dict[str, int] = {}
    seed: int = args.seed if args.seed is not None else 0
    harness_ids: list[str] = []
    fixed_model_ids: list[str] = []
    fixed_effort = args.effort or ""
    fixed_provider = args.provider or ""

    if state_candidates:
        new_value = "__new__"
        choices = []
        for p, payload in state_candidates:
            tid = payload.get("tournament_id", "?")
            rounds_played = len(payload.get("rounds", []))
            n_h = len(payload.get("selected_harness_ids", []))
            n_m = len(payload.get("fixed_model_ids", []))
            ts = payload.get("timestamp", "")
            title = (
                f"{tid[:8]}…  rounds={rounds_played}  harnesses={n_h}  "
                f"models={n_m}  {ts[:19]}  ({p.name})"
            )
            choices.append(questionary.Choice(title=title, value=str(p)))
        choices.append(
            questionary.Choice(title="Start a new tournament", value=new_value)
        )

        picked = questionary.select(
            f"Found {len(state_candidates)} harness-ablation state(s) for "
            f"repos={','.join(repo_names)}. Pick one to resume, or start new:",
            choices=choices,
        ).ask()
        if picked is None:
            raise SystemExit("aborted.")
        if picked != new_value:
            state_path = Path(picked)
            tmp_payload = json.loads(state_path.read_text())
            harness_ids = list(
                tmp_payload.get("selected_harness_ids")
                or tmp_payload.get("selected_model_ids")
                or []
            )
            fixed_model_ids = list(tmp_payload.get("fixed_model_ids") or [])
            seed = int(tmp_payload.get("seed", seed))
            # Restore the pinned identity the schedule was generated under.
            fixed_effort = str(tmp_payload.get("fixed_reasoning_effort", "") or "")
            fixed_provider = str(tmp_payload.get("fixed_provider", "") or "")
            rr, tournament_id, harness_ids, fixed_model_ids = _reconstruct_rr(
                state_path, harness_ids, seed
            )
            _validate_rectangular_field(
                ctx["challenge_store"],
                harness_ids,
                fixed_model_ids,
                repo_names,
                turns_per_player,
                all_model_configs=all_model_configs,
                effort=fixed_effort,
                provider=fixed_provider,
            )
            prior_matches, match_round = _load_submatch_results(
                ctx["artifact_logger"].matches_dir,
                rr,
                repo_names,
                fixed_model_ids,
                fixed_effort,
                fixed_provider,
            )
            resumed = True

    if not resumed:
        from swe_duel.agents.harness import HARNESS_IDS

        def _pick_harnesses_from_cli(specs: list[str]) -> list[str]:
            bad = [h for h in specs if h not in HARNESS_IDS]
            if bad:
                raise SystemExit(
                    f"Unknown harness(es) {bad}; known: {list(HARNESS_IDS)}"
                )
            return list(dict.fromkeys(specs))  # preserve order, de-dupe

        # Validate the pinned identity against menus + harness capabilities.
        from swe_duel.agents.harness import (
            harness_supported_efforts,
            harness_supports_provider,
        )

        for nick, cfg in all_model_configs.items():
            if fixed_effort and fixed_effort not in cfg.reasoning_efforts:
                raise SystemExit(
                    f"--effort {fixed_effort!r} is not in the models.yaml menu "
                    f"for {nick} ({cfg.model_id}); menu: {cfg.reasoning_efforts}"
                )
            if fixed_provider and fixed_provider not in cfg.providers:
                raise SystemExit(
                    f"--provider {fixed_provider!r} is not in the models.yaml "
                    f"menu for {nick} ({cfg.model_id}); menu: {cfg.providers}"
                )
        if fixed_effort and any(
            harness_supported_efforts(h) is not None
            and fixed_effort not in harness_supported_efforts(h)
            for h in harness_ids
        ):
            raise SystemExit(
                f"--effort {fixed_effort!r} is not expressible by every "
                f"selected harness {harness_ids}"
            )
        if fixed_provider and any(
            not harness_supports_provider(h) for h in harness_ids
        ):
            raise SystemExit(
                f"--provider {fixed_provider!r} cannot be pinned by every "
                f"selected harness {harness_ids} (CLI harnesses have no "
                "request-body routing control)"
            )

        # Always: harnesses first, then a user-chosen *subset* of models, then
        # validate rectangular coverage. Nothing auto-selects every model.
        if args.harnesses:
            harness_ids = _pick_harnesses_from_cli(args.harnesses)
        else:
            _print_slot_table(
                ctx["challenge_store"],
                all_model_configs,
                repo_names,
                turns_per_player,
            )
            harness_ids = _select_harnesses_interactive(
                slot_counts=slot_counts,
                success_counts=success_counts,
                all_model_configs=all_model_configs,
                n_repos=len(repo_names),
                effort=fixed_effort,
                provider=fixed_provider,
            )

        if args.models:
            fixed_model_ids = _resolve_cli_models(args.models, all_model_configs)
        else:
            if args.harnesses:
                # Harnesses came from CLI — still show the bank table before
                # the interactive subset picker.
                _print_slot_table(
                    ctx["challenge_store"],
                    all_model_configs,
                    repo_names,
                    turns_per_player,
                )
            fixed_model_ids = _select_models_interactive(
                store=ctx["challenge_store"],
                harness_ids=harness_ids,
                all_model_configs=all_model_configs,
                repo_names=repo_names,
                turns_per_player=turns_per_player,
                effort=fixed_effort,
                provider=fixed_provider,
            )

        _validate_rectangular_field(
            ctx["challenge_store"],
            harness_ids,
            fixed_model_ids,
            repo_names,
            turns_per_player,
            all_model_configs=all_model_configs,
            effort=fixed_effort,
            provider=fixed_provider,
        )
        default_seed = args.seed if args.seed is not None else 0
        seed = _prompt_seed(default_seed)
        rr = RoundRobinTournament(harness_ids, seed=seed)
        tournament_id = str(uuid.uuid4())

        print("\n  Harness-ablation field (rectangular):")
        print(f"    harnesses : {', '.join(harness_ids)}")
        print(
            "    identity  : "
            + (f"effort={fixed_effort} " if fixed_effort else "effort=default ")
            + (f"provider={fixed_provider}" if fixed_provider else "provider=auto")
        )
        print(
            "    models    : "
            + ", ".join(
                _nick_for_model_id(all_model_configs, m) for m in fixed_model_ids
            )
        )
        print(
            f"    pairings  : C({len(harness_ids)},2) × "
            f"{len(fixed_model_ids)} models = "
            f"{len(harness_ids) * (len(harness_ids) - 1) // 2 * len(fixed_model_ids)} "
            "sub-matches\n"
        )

    if resumed and args.seed is not None and args.seed != seed:
        print(
            f"[warn] --seed={args.seed} ignored on resume; using persisted "
            f"seed={seed}."
        )

    assert rr is not None and tournament_id is not None

    # Restrict model_configs to the fixed models so factories resolve cleanly.
    ctx["model_configs"] = {
        nick: cfg
        for nick, cfg in all_model_configs.items()
        if cfg.model_id in set(fixed_model_ids)
    }
    model_nick_by_id = {
        cfg.model_id: nick for nick, cfg in all_model_configs.items()
    }

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

    # Participant-cid-keyed configs: every fixed model carries the ablation's
    # pinned effort/provider selection so the orchestrator resolves each
    # sub-match's Blue harness to the exact identity it competes under.
    model_configs_by_id = {
        composite_id(cfg.model_id, h, fixed_effort, fixed_provider): (
            cfg.with_selection(fixed_effort, fixed_provider)
        )
        for cfg in ctx["model_configs"].values()
        for h in harness_ids
    }

    # Warn up front when OpenRouter has no pricing for the pinned (model,
    # provider) combos — cost tracking for those competitors reports $0.00.
    # The (model, provider) pairs repeat across harnesses; the helper warns
    # once per unique pair.
    warn_missing_pricing(cfg for cfg in model_configs_by_id.values())
    # Surface request-rate caps before the live TUI starts. The ablation
    # requires every selected harness — if the pinned provider has a rate
    # limit, the CLI harnesses (codex / claude-code) cannot enforce it and
    # are warned about here.
    summarize_rate_limits(cfg for cfg in model_configs_by_id.values())
    warn_unenforceable_rate_limits(
        (cfg, split_composite_id(cid)[1])
        for cid, cfg in model_configs_by_id.items()
    )

    match_orchestrator = MatchOrchestrator(
        model_configs=model_configs_by_id,
        challenge_store=ctx["challenge_store"],
        workspace_manager=ctx["workspace_manager"],
        config=arena_cfg,
        artifact_logger=ctx["artifact_logger"],
        prompt_dir=Path(args.prompt_dir),
        test_runner_factory=_test_runner_factory,
    )

    console = HarnessAblationConsole(
        ctx=ctx,
        match_orchestrator=match_orchestrator,
        repo_names=repo_names,
        seed=seed,
        tournament_id=tournament_id,
        harness_ids=harness_ids,
        fixed_model_ids=fixed_model_ids,
        rr=rr,
        resumed=resumed,
        prior_matches=prior_matches,
        match_round=match_round,
        model_nick_by_id=model_nick_by_id,
        effort=fixed_effort,
        provider=fixed_provider,
    )
    console.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
