#!/usr/bin/env python
"""Active-sampling tournament console (`swe-duel-tournament-as`).

Enters a tournament whose rankings were exported by `swe-duel-rankings`:
``--tournament <id>`` loads ``./rankings/rankings_<id>.json`` (the
participant field with standings, BT/Elo, head-to-head history, and the
exact arena the tournament ran). **New entrants** — participants not in
the export — are added via the same paged model → harness → effort →
provider wizard as the other consoles, and the Chatbot-Arena
SE-reduction rule — the same ``swe_duel.scoring.active_sampling`` module
that powers newcomer intake in ``swe-duel-tournament`` /
``swe-duel-tournament-rr`` — picks each entrant's most
ranking-informative opponents from the WHOLE existing field (never a
hand-picked subset — that would bias the standings), so only the most
informative entrant-vs-field pairings run instead of a full C(N, 2)
round-robin.

Flow (one-shot, two confirm prompts):

  1. load the field from ``rankings_<id>.json`` (rank order). Each
     participant's composite identity carries the entry's
     ``reasoning_effort`` / ``provider`` columns pinned — the OpenRouter
     defaults the exported tournament actually ran at (auto-route ⇒ empty
     provider) — so the migrated bank pools and defense-cache entries
     marked with the same identity are visible to this command.
     claude-code entries keep the empty selection (its CLI has no
     effort/provider control). **The export's repo set and per-repo
     challenge target** (``repos`` / ``targets_per_repo``) are enforced:
     ``--repos`` / ``--turns-per-player`` may only repeat them (a different
     arena would bias the standings against the existing field). The
     export's raw ``matches`` list supplies the head-to-head history for
     pair gains and the already-played exclusion alongside ``data/matches``.
  2. add the **new entrants** — participants not in the export that join
     the field unrated — via the paged model → harness → effort →
     provider wizard (``swe_duel.cli.participant_select`` — shown
     without bank-slot annotations, since a participant is uniquely the
     (model, harness, reasoning_effort, provider) 4-tuple and the
     selection is only complete on the last page), or
     ``--participants model#harness[#effort#provider]`` composites
     non-interactively. At least one entrant is required — entering a
     tournament means bringing a new participant;
  3. active-sample the matchups: **every matchup pairs a new entrant with
     an EXISTING participant** (an entrant never plays another entrant
     here, and existing–existing pairs belong to the original schedule
     and are never re-sampled). Per entrant, up to ``--budget`` incumbent
     opponents (default ``ceil(log2 N)+1`` over the field) chosen by
     SE-reduction gain over the prior matches (never-played pairs first;
     already-played pairs are not re-sampled);
  4. show the sampled matches to run plus the challenge slots each
     (participant, repo) pool still needs — slots that already have any
     attempt record in ``./data/`` (admitted challenge OR an exhausted
     5-attempt failure chain) are reused from the cache, only never-attempted
     slots are generated;
  5. prompt: continue with challenge generation (live TUI), then
  6. prompt: continue with running the sampled matches (pooled defenses,
     live TUI; per-(challenge, Blue) defenses cached in ``data/defenses/``
     are reused at zero cost), then
7. prompt: prepare the **contribution zip** — the matchup-relevant
      slice of ``./data`` the organizer's ``swe-duel-tournament-update``
      import consumes for this tournament (every local active-sampling
      state entering it, the matches those states claim, the
      challenge/defense records those matches' **turns reference**, the
      new entrants' failed attempts, plus ``challenge_bank/index.json``)
      **and the whole ``./config`` directory** (the deployment's editable
      arena/models/repos configuration that shaped every packaged
      record, stored under a ``config/`` archive root the import leaves
      in the zip), written to ``--submission-dir`` (default
      ``./submissions``). NOT
      the whole field's bank: the incumbents' challenge/defense records
      already live on the organizer's side, so packaging by field
      identity only ballooned the zip (the entire original tournament's
      records) without adding anything the import needs. The member set
      is verified before writing and the written archive re-verified
      after: only those matchup-relevant records ever package —
      ``workspaces/`` and ``logs/`` are excluded (all needed facts live
      in the ``.json`` records). Declining the match runs (step 6) offers
      the same prompt, so a run that only refreshed the caches can still
      be submitted.

A run interrupted before its contribution zip (a closed terminal, or a
declined phase worth re-running) is resumable: ``--resume-state <id>``
replays a persisted ``active_sampling_state_<id>.json`` — its entrants,
sampled pairings, and recorded arena — instead of sampling fresh
matchups (no new LLM/Docker spend just to reach the packaging prompt).
Phases the state already completed are skipped (a ``complete`` state
goes straight to the contribution-zip prompt) and the state file is
continued in place, never duplicated.

Sampled pairings + outcomes are persisted to
``data/tournaments/active_sampling_state_<id>.json``; match records land in
``data/matches/`` like every other command, so re-invoking the command
naturally samples the next most-informative (yet-unplayed) matchups — to
revisit a specific earlier run instead (re-run a declined phase, or reach
the packaging prompt a closed terminal ate), pass ``--resume-state <id>``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import questionary

from swe_duel.agents.harness import (
    HARNESS_IDS,
    get_harness,
    harness_supported_efforts,
    harness_supports_provider,
)
from swe_duel.agents.harness.cost_tracking import warn_missing_pricing
from swe_duel.agents.harness.rate_limit import (
    summarize_rate_limits,
    warn_unenforceable_rate_limits,
)
from swe_duel.agents.red import RedAgent
from swe_duel.challenge_bank.generator import ChallengeGenerator
from swe_duel.challenge_bank.progress import (
    GenerationReporter,
    Status as _Status,
    live as reporter_live,
)
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.cli._common import (
    default_prompt_dir,
    install_container_cleanup_handlers,
    resolve_config_dir,
    resolve_output_dir,
    setup,
)
from swe_duel.cli.export_rankings import _as_entered_tournament, _load_as_extra_matches
from swe_duel.cli.participant_select import select_participants
from swe_duel.cli.run_tournament import _short
from swe_duel.cli.tournament_update import (
    _Identity,
    _as_field,
    _as_newcomers,
    _load_as_states,
    _record_identity,
    _zip_sha256,
)
from swe_duel.config import ArenaConfig, ModelConfig, RepoConfig, load_arena_config
from swe_duel.engine.match import DefenseTask, MatchOrchestrator, MatchPlan
from swe_duel.engine.round_progress import (
    DefenseRoundReporter,
    Status as SubturnStatus,
    SubturnHandle,
    live as round_live,
)
from swe_duel.engine.swiss import SwissPairing
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.models import (
    MatchOutcome,
    MatchResult,
    composite_id,
    display_composite_id,
    split_composite_id,
)
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.diff_utils import iter_files_safe
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.sandbox.workspace import WorkspaceManager
from swe_duel.scoring.active_sampling import (
    PairStats,
    aggregate_pair_stats,
    default_intake_budget,
    pair_gain,
    pair_key,
    select_intake_pairings,
)
from swe_duel.scoring.rating import compute_all_ratings
from swe_duel.validation.red_gates import RedGateValidator

# ── rankings.json ──────────────────────────────────────────────


@dataclass(frozen=True)
class RankingsField:
    """Parsed ``rankings_<id>.json`` export: the ranked participants plus the
    tournament's arena (repo set + per-repo challenge target) and match
    history.

    ``repos`` / ``targets_per_repo`` come from the export file (the arenas
    actually run): every match spanned every repo with ``targets_per_repo``
    challenges per repo per side, so the active-sampling command regenerates
    exactly that shape — and *enforces* it for new entrants (a different
    arena would bias the comparison against the existing field). Both may
    be absent in a legacy file — ``repos`` then defaults to empty (the
    caller requires ``--repos``) and ``targets_per_repo`` falls back to the
    arena's ``match.turns_per_player``.

    ``matches`` is the export's raw match list (id, endpoints, outcome,
    timestamp, repos) — the head-to-head history active sampling needs for
    pair gains and the already-played exclusion, even when those match
    files are not present under ``data/matches/``.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    targets_per_repo: int | None = None
    matches: list[dict[str, Any]] = field(default_factory=list)


def _pair_ids(p: SwissPairing) -> tuple[str, str]:
    """Both endpoints of a sampled pairing as bare strings.

    ``select_intake_pairings`` never emits byes, so ``model_b`` is always a
    real participant here; the assert narrows the ``str | None`` field for
    the type checker.
    """
    assert p.model_b is not None
    return p.model_a, p.model_b


def _full_identity(cid: str) -> str:
    """A participant's entire 4-tuple identity for the plan TUI lines.

    Renders ``model [harness] effort=<e> provider=<p>`` — every component
    shown, with explicit ``(default)`` / ``(auto-route)`` placeholders when a
    segment is unset — unlike ``display_composite_id`` (which omits unset
    segments) or ``_short`` (which truncates long ids and hides the
    effort/provider segments). The plan sections (matchups, defense-turn
    plan, challenge slots) print this so the operator can verify each
    matchup/slot runs under the exact effort/provider pinned from the
    rankings export (or chosen for the new entrant) — the identity that
    challenge generation and defense runs actually use.
    """
    model, harness, effort, provider = split_composite_id(cid)
    return (
        f"{model} [{harness or 'mini-swe-agent'}] "
        f"effort={effort or '(default)'} "
        f"provider={provider or '(auto-route)'}"
    )


def resolve_rankings_selection(
    harness: str, effort_raw: object, provider_raw: object, label: str
) -> tuple[str, str]:
    """Translate a rankings entry's effort/provider columns into the pinned
    participant selection, respecting harness capability.

    The columns record the OpenRouter defaults the original experiments ran
    at (``reasoning_effort`` = the model's default effort or null;
    ``provider`` = ``(OpenRouter auto-route)`` or a slug). Pinning exactly
    those values reproduces the original runs' behaviour while giving the
    records the explicit effort/provider identity fields — so bank pools,
    the defense cache and match records keyed under them line up.

    Harness capability rules (same fail-fast semantics as ``swe-duel-match``):

    * claude-code has *no* reasoning-effort / provider control, so its
      entries always keep the empty selection (model default, auto-route) —
      the informational column is never a pin there.
    * codex expresses a closed effort enum; a value outside it is a real
      conflict and fails fast instead of silently running unpinned.
    * codex / claude-code cannot pin providers; a non-auto-route slug for
      them fails fast.
    """
    effort = effort_raw if isinstance(effort_raw, str) and effort_raw else ""
    provider = (
        provider_raw
        if isinstance(provider_raw, str) and provider_raw and provider_raw != "(OpenRouter auto-route)"
        else ""
    )
    supported = harness_supported_efforts(harness)
    if supported == frozenset():
        if effort:
            print(
                f"[as] {label}: harness {harness!r} has no reasoning-effort "
                "control — keeping the empty selection (the model runs at "
                "its default effort either way)."
            )
        effort = ""
    elif supported is not None and effort and effort not in supported:
        raise SystemExit(
            f"rankings entry {label!r}: harness {harness!r} cannot express "
            f"reasoning effort {effort!r} (supported: {sorted(supported)})"
        )
    if provider and not harness_supports_provider(harness):
        raise SystemExit(
            f"rankings entry {label!r}: harness {harness!r} cannot pin "
            f"provider {provider!r}"
        )
    return effort, provider


def _load_rankings(path: Path) -> RankingsField:
    """Parse ``rankings.json`` into a validated :class:`RankingsField`.

    Each participant entry must carry ``model`` + ``harness``; the composite
    identity is rebuilt with the entry's ``reasoning_effort`` / ``provider``
    columns pinned (see :func:`resolve_rankings_selection`) — those record
    the OpenRouter defaults the original runs used, and pinning them makes
    the migrated bank pools / defense-cache entries under the same identity
    visible to this command. Entries with null effort / auto-route (and
    claude-code entries, which cannot pin) keep the legacy 2-part
    ``model#harness`` form.

    Top-level ``repos`` (the tournament's repo list) and ``targets_per_repo``
    (challenge slots per participant per repo, i.e. the match's
    ``turns_per_player``) are optional but validated when present; they are
    what the active-sampling command uses to decide *which* repositories the
    sampled matchups' participants generate challenges for, and *how many*
    slots each (participant, repo) pool needs — and, when present, the only
    values ``--repos`` / ``--turns-per-player`` may repeat (enforced, see
    :func:`_enforce_exported_arena`). An optional top-level ``matches`` list
    carries the export's raw head-to-head history for active sampling.
    """
    if not path.is_file():
        raise SystemExit(
            f"rankings file not found: {path}\n"
            "  (expected a rankings_<tournament_id>.json export produced by "
            "swe-duel-rankings; pass --tournament <id> or --rankings <path>)"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"rankings file {path} is not valid JSON: {e}") from e
    entries = data.get("rankings")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"rankings file {path} has no non-empty 'rankings' array")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict) or not raw.get("model") or not raw.get("harness"):
            raise SystemExit(
                f"rankings entry #{i + 1} in {path} must be an object with "
                "'model' and 'harness' fields"
            )
        harness = str(raw["harness"])
        if harness not in HARNESS_IDS:
            raise SystemExit(
                f"rankings entry #{i + 1}: unknown harness {harness!r} "
                f"(choose from {sorted(HARNESS_IDS)})"
            )
        label = f"{raw['model']}#{harness}"
        effort, provider = resolve_rankings_selection(
            harness, raw.get("reasoning_effort"), raw.get("provider"), label
        )
        cid = composite_id(str(raw["model"]), harness, effort, provider)
        if cid in seen:
            raise SystemExit(f"duplicate participant {cid!r} in {path}")
        seen.add(cid)
        entry: dict[str, Any] = {
            "rank": int(raw.get("rank", i + 1)),
            "model": str(raw["model"]),
            "harness": harness,
            "cid": cid,
            "points": float(raw.get("points", 0.0)),
            "elo": raw.get("elo"),
            "bradley_terry": raw.get("bradley_terry"),
        }
        out.append(entry)

    repos: list[str] = []
    raw_repos = data.get("repos")
    if raw_repos is not None:
        if not isinstance(raw_repos, list) or not all(
            isinstance(r, str) and r for r in raw_repos
        ):
            raise SystemExit(
                f"rankings file {path}: 'repos' must be a list of repo names"
            )
        for r in raw_repos:
            if r not in repos:
                repos.append(r)

    targets_per_repo: int | None = None
    raw_targets = data.get("targets_per_repo")
    if raw_targets is not None:
        if isinstance(raw_targets, bool) or not isinstance(raw_targets, int):
            raise SystemExit(
                f"rankings file {path}: 'targets_per_repo' must be a positive "
                "integer"
            )
        if raw_targets < 1:
            raise SystemExit(
                f"rankings file {path}: 'targets_per_repo' must be ≥ 1 "
                f"(got {raw_targets})"
            )
        targets_per_repo = raw_targets

    matches: list[dict[str, Any]] = []
    raw_matches = data.get("matches")
    if raw_matches is not None:
        # The export's raw match history (head-to-head reconstruction). Only
        # the fields active sampling reads are validated here.
        if not isinstance(raw_matches, list):
            raise SystemExit(f"rankings file {path}: 'matches' must be a list")
        for i, m in enumerate(raw_matches):
            if not isinstance(m, dict) or not m.get("model_a") or not m.get("model_b"):
                raise SystemExit(
                    f"rankings file {path}: matches[{i}] must carry 'model_a' "
                    "and 'model_b'"
                )
            matches.append(dict(m))

    return RankingsField(
        entries=out,
        repos=repos,
        targets_per_repo=targets_per_repo,
        matches=matches,
    )


def _build_model_configs(
    all_model_configs: dict[str, ModelConfig],
    cids: list[str],
    max_tokens: int,
) -> dict[str, ModelConfig]:
    """Per-participant ModelConfigs keyed by composite id.

    A model present in ``config/models.yaml`` uses that (curated) entry; a
    rankings-only model gets an auto-constructed config. Either way the
    config is bound to the participant's pinned effort/provider selection
    (from the rankings entry's recorded defaults) — the selection is part
    of the competitor identity, so it must reach the harness *and* the
    persisted records, not just the pool key lookups.
    """
    by_model = {cfg.model_id: cfg for cfg in all_model_configs.values()}
    out: dict[str, ModelConfig] = {}
    for cid in cids:
        model_id, _harness, effort, provider = split_composite_id(cid)
        base = by_model.get(model_id)
        if base is not None:
            out[cid] = base.with_selection(effort, provider)
        else:
            print(
                f"[as] {model_id} is not in config/models.yaml — using an "
                "auto-constructed config (pinned to the rankings entry's "
                "default effort/provider selection, no curated rate "
                "limits)."
            )
            out[cid] = ModelConfig(
                model_id=model_id, max_tokens=max_tokens
            ).with_selection(effort, provider)
    return out


def _parse_newcomer_id(raw: str) -> str:
    """Validate a new-entrant id and return its pinned composite cid.

    A newcomer is any ``model#harness[#effort#provider]`` composite that is
    not one of the rankings file's participants — a new model joining the
    tournament's field. The harness must be known and the effort/provider
    selection must respect harness capability (same rules as rankings
    entries, :func:`resolve_rankings_selection`); anything unparseable
    returns ``""`` so the caller can report it as unknown.
    """
    model, harness, effort, provider = split_composite_id(raw)
    if not model or not harness or harness not in HARNESS_IDS:
        return ""
    effort, provider = resolve_rankings_selection(
        harness, effort, provider, raw
    )
    return composite_id(model, harness, effort, provider)


def _resolve_newcomers(
    requested: list[str], field: list[str], by_cid: dict[str, dict[str, Any]]
) -> list[str]:
    """Resolve ``--participants`` values to new-entrant composite cids.

    Active sampling always covers the whole rankings field, so the flag
    only names **new entrants** — participants that join the field unrated.
    A value matching a rankings participant (its exact pinned composite
    id, the bare ``model#harness`` form, or a composite that parses down
    to one because the harness cannot express the pins) fails fast with
    the new semantics instead of silently (de)selecting a field member.
    Anything not a parseable ``model#harness[#effort#provider]`` composite
    fails fast listing the valid ids.
    """
    by_two_part = {
        composite_id(e["model"], e["harness"]): cid for cid, e in by_cid.items()
    }
    newcomers: list[str] = []
    unknown: list[str] = []
    for p in requested:
        if p in by_cid or p in by_two_part:
            raise SystemExit(
                f"--participants names {p!r}, which is already in the "
                "rankings field — active sampling always covers every "
                "field participant, so the flag only names NEW entrants; "
                "drop it from the list."
            )
        newcomer = _parse_newcomer_id(p)
        if not newcomer:
            unknown.append(p)
        elif newcomer in by_cid:
            raise SystemExit(
                f"--participants names {p!r}, which resolves to {newcomer!r} "
                "— already in the rankings field, so it is not a new "
                "entrant (active sampling always covers every field "
                "participant); drop it from the list."
            )
        else:
            newcomers.append(newcomer)
    if unknown:
        raise SystemExit(
            f"--participants must be new-entrant ids: {unknown}\n"
            f"(expected model#harness[#effort#provider]; the rankings field "
            f"is {field})"
        )
    return list(dict.fromkeys(newcomers))


def _resolve_rankings_path(args: argparse.Namespace) -> Path:
    """Which rankings export backs this run: ``--tournament <id>`` resolves
    ``./rankings/rankings_<id>.json``; ``--rankings <path>`` is an explicit
    file. Exactly one is required — the tournament id is how a new entrant
    chooses which tournament to enter, so no silent default exists.
    """
    rankings_dir = Path("./rankings")
    if args.tournament and args.rankings:
        raise SystemExit(
            "--tournament and --rankings are mutually exclusive — pick one."
        )
    if args.tournament:
        path = rankings_dir / f"rankings_{args.tournament}.json"
        if not path.is_file():
            available = sorted(
                p.name[len("rankings_"):-len(".json")]
                for p in rankings_dir.glob("rankings_*.json")
            )
            raise SystemExit(
                f"no rankings export for tournament {args.tournament!r}: "
                f"{path} not found.\n"
                + (
                    f"tournaments exported under {rankings_dir}: {available}\n"
                    "(generate one with: swe-duel-rankings <tournament_id>)"
                    if available
                    else f"no rankings_*.json under {rankings_dir} — generate "
                    "one with: swe-duel-rankings <tournament_id>"
                )
            )
        return path
    if args.rankings:
        return Path(args.rankings)
    available = sorted(
        p.name[len("rankings_"):-len(".json")]
        for p in rankings_dir.glob("rankings_*.json")
    )
    raise SystemExit(
        "choose which tournament to enter: pass --tournament <id> "
        "(loads ./rankings/rankings_<id>.json) or --rankings <path>."
        + (f"\ntournaments exported under {rankings_dir}: {available}"
           if available else
           f"\nno rankings_*.json under {rankings_dir} — generate one with: "
           "swe-duel-rankings <tournament_id>")
    )


def _enforce_exported_arena(
    args: argparse.Namespace, field_data: RankingsField, path: Path
) -> None:
    """``--repos`` / ``--turns-per-player`` must repeat the export's arena.

    The rankings export records the exact repos and per-repo challenge
    target the tournament ran; its BT/Elo/head-to-head are measured on that
    arena. Comparing a new entrant against the field on a *different* repo
    set or target count would bias the standings, so any differing value
    fails fast. Legacy exports without the fields keep the old
    CLI-overrides-arena behaviour.
    """
    if args.repos and field_data.repos and set(args.repos) != set(field_data.repos):
        raise SystemExit(
            f"--repos {sorted(args.repos)} differs from the arena the "
            f"tournament ran —\n  {path} declares repos="
            f"{field_data.repos}.\n"
            "  New entrants must be compared on the original arena: pass "
            "the same repos or drop the flag."
        )
    if (
        args.turns_per_player is not None
        and field_data.targets_per_repo is not None
        and args.turns_per_player != field_data.targets_per_repo
    ):
        raise SystemExit(
            f"--turns-per-player {args.turns_per_player} differs from the "
            f"arena the tournament ran —\n  {path} declares "
            f"targets_per_repo={field_data.targets_per_repo}.\n"
            "  New entrants must face the original challenge count: pass "
            "the same value or drop the flag."
        )


# ── resume (--resume-state) ─────────────────────────────────────


def _resolve_resume_state_path(raw: str, tournaments_dir: Path) -> Path:
    """Locate the ``active_sampling_state_<id>.json`` file to resume.

    Accepts an existing file path, the bare state id (the uuid the console
    printed when it sampled the matchups), or the full file stem. An
    unknown id fails fast listing the states available under
    ``tournaments_dir`` — resuming the wrong run would re-drive the wrong
    matchups against the wrong entrants.
    """
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    stem = candidate.name or raw
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    if not stem.startswith("active_sampling_state_"):
        stem = f"active_sampling_state_{stem}"
    hit = tournaments_dir / f"{stem}.json"
    if hit.is_file():
        return hit
    available = (
        sorted(
            p.stem[len("active_sampling_state_"):]
            for p in tournaments_dir.glob("active_sampling_state_*.json")
        )
        if tournaments_dir.is_dir()
        else []
    )
    raise SystemExit(
        f"--resume-state: no active-sampling state {raw!r} under "
        f"{tournaments_dir}"
        + (
            f" — available: {available}"
            if available
            else " (no active_sampling_state_*.json files there yet)"
        )
    )


@dataclass(frozen=True)
class _ResumePlan:
    """Everything ``--resume-state`` re-derives from a persisted AS state:
    the state file/payload to continue in place (never duplicated), the
    entrants it brought, its sampled pairings, and the arena it ran."""

    state_path: Path
    state_payload: dict[str, Any]
    newcomers: list[str]
    pairings: list[SwissPairing]
    repo_names: list[str]
    targets_per_repo: int | None


def _load_resume_plan(raw: str, tournaments_dir: Path) -> _ResumePlan:
    """Load + validate the state to resume (a pure disk read, pre-setup).

    Fail-fast validation: the file must be an active-sampling state that
    recorded at least one entrant and one sampled pairing and the arena it
    ran — a state that never sampled, or a foreign format's file, would
    re-drive the wrong matchups. Pairing endpoints are validated against
    the field later, once the rankings export is loaded (the export, not
    the state, defines the incumbent field).
    """
    state_path = _resolve_resume_state_path(raw, tournaments_dir)
    payload = _zip_payload(state_path)
    if payload is None:
        raise SystemExit(
            f"--resume-state: {state_path} is not a parseable JSON object"
        )
    if payload.get("format") != "active_sampling":
        raise SystemExit(
            f"--resume-state: {state_path} is not an active-sampling state "
            f"(format={payload.get('format')!r})"
        )
    newcomers = list(dict.fromkeys(_as_newcomers(payload)))
    if not newcomers:
        raise SystemExit(
            f"--resume-state: {state_path} records no new entrants — "
            "nothing to resume."
        )
    pairings: list[SwissPairing] = []
    for i, pr in enumerate(payload.get("pairings") or []):
        if not isinstance(pr, dict):
            raise SystemExit(
                f"--resume-state: pairing #{i + 1} in {state_path} is malformed"
            )
        a = str(pr.get("model_a") or "")
        b = str(pr.get("model_b") or "")
        if not a or not b:
            raise SystemExit(
                f"--resume-state: pairing #{i + 1} in {state_path} lacks an "
                "endpoint — the state file is corrupted"
            )
        pairings.append(SwissPairing(model_a=a, model_b=b))
    if not pairings:
        raise SystemExit(
            f"--resume-state: {state_path} records no sampled pairings — "
            "the run never sampled its matchups; re-run without "
            "--resume-state instead."
        )
    repo_names = [str(r) for r in payload.get("repo_names") or [] if r]
    if not repo_names:
        raise SystemExit(
            f"--resume-state: {state_path} records no repo_names — the "
            "arena it ran cannot be reconstructed; re-run without "
            "--resume-state."
        )
    raw_targets = payload.get("targets_per_repo")
    targets_per_repo: int | None = None
    if (
        isinstance(raw_targets, int)
        and not isinstance(raw_targets, bool)
        and raw_targets >= 1
    ):
        targets_per_repo = raw_targets
    return _ResumePlan(
        state_path=state_path,
        state_payload=payload,
        newcomers=newcomers,
        pairings=pairings,
        repo_names=repo_names,
        targets_per_repo=targets_per_repo,
    )


def _resolve_resume_rankings(
    args: argparse.Namespace, plan: _ResumePlan, rankings_dir: Path
) -> Path:
    """The rankings export a resumed state entered.

    Explicit ``--tournament`` / ``--rankings`` must agree with what the
    state recorded (its ``entered_tournament`` / ``rankings_source``) —
    resuming a state against a different export would silently change the
    incumbent field its pairings were sampled against. Without flags, the
    state's own record decides: the entered tournament's
    ``rankings_<id>.json``, else its recorded export path; a state that
    records neither cannot be placed and fails fast.
    """
    entered = _as_entered_tournament(plan.state_payload)
    source = plan.state_payload.get("rankings_source")
    if args.tournament and args.rankings:
        raise SystemExit(
            "--tournament and --rankings are mutually exclusive — pick one."
        )
    if args.tournament:
        if entered and entered != args.tournament:
            raise SystemExit(
                f"--resume-state: {plan.state_path.name} entered tournament "
                f"{entered!r}, not {args.tournament!r}."
            )
        path = rankings_dir / f"rankings_{args.tournament}.json"
        if not path.is_file():
            raise SystemExit(
                f"no rankings export for tournament {args.tournament!r}: "
                f"{path} not found."
            )
        return path
    if args.rankings:
        if (
            isinstance(source, str)
            and source
            and Path(source).name != Path(args.rankings).name
        ):
            raise SystemExit(
                f"--resume-state: {plan.state_path.name} was sampled from "
                f"{source!r}, not {args.rankings!r}."
            )
        return Path(args.rankings)
    if entered:
        path = rankings_dir / f"rankings_{entered}.json"
        if path.is_file():
            return path
        raise SystemExit(
            f"--resume-state: the state entered tournament {entered!r} but "
            f"{path} is missing — pass --rankings <path> explicitly."
        )
    if isinstance(source, str) and source and Path(source).is_file():
        return Path(source)
    raise SystemExit(
        "--resume-state: cannot tell which rankings export this state "
        f"entered ({plan.state_path.name} records no usable "
        "entered_tournament/rankings_source) — pass --tournament <id> or "
        "--rankings <path>."
    )


# ── prior-match history ────────────────────────────────────────


def _load_prior_matches(
    matches_dir: Path,
    field: set[str],
    extra_payloads: list[dict[str, Any]] | None = None,
) -> tuple[list[MatchResult], set[frozenset[str]], dict[str, float]]:
    """Prior matches among ``field``: ``data/matches/`` files plus the
    rankings export's raw match list.

    Returns ``(matches, already_played_pairs, trueskill_sigma)``. Only
    outcomes matter here (SE-reduction gains + the TrueSkill-σ tie-break),
    so each payload becomes a turns-free MatchResult. ``extra_payloads``
    (the export's ``matches`` list — model_a/model_b/outcome/timestamp)
    fills the history when the match files are not on disk; matches are
    de-duplicated by ``match_id`` with the local files winning, so an
    export of a tournament whose matches also live under ``data/matches``
    is never double-counted. Unreadable files are skipped.
    """
    payloads: dict[str, dict[str, Any]] = {}
    if matches_dir.is_dir():
        for p in sorted(matches_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            mid = str(payload.get("match_id") or p.stem)
            payloads.setdefault(mid, payload)
    for extra in extra_payloads or []:
        mid = str(extra.get("match_id") or "")
        if mid and mid not in payloads:
            payloads[mid] = {
                "match_id": mid,
                "model_a_id": extra.get("model_a"),
                "model_b_id": extra.get("model_b"),
                "repo_name": extra.get("repo_name", ""),
                "outcome": extra.get("outcome"),
                "timestamp": extra.get("timestamp"),
            }
    matches: list[MatchResult] = []
    played: set[frozenset[str]] = set()
    for mid in sorted(payloads):
        payload = payloads[mid]
        a = str(payload.get("model_a_id") or "")
        b = str(payload.get("model_b_id") or "")
        if not a or not b or a not in field or b not in field or a == b:
            continue
        ts = payload.get("timestamp")
        timestamp = (
            datetime.fromisoformat(ts)
            if isinstance(ts, str)
            else datetime.now(timezone.utc)
        )
        matches.append(
            MatchResult(
                match_id=mid,
                model_a_id=a,
                model_b_id=b,
                repo_name=str(payload.get("repo_name") or ""),
                turns=[],
                model_a_total=0.0,
                model_b_total=0.0,
                outcome=MatchOutcome(str(payload.get("outcome") or "draw")),
                duration_seconds=0.0,
                total_cost_usd=0.0,
                timestamp=timestamp,
            )
        )
        played.add(pair_key(a, b))
    secondary: dict[str, float] = {}
    if matches:
        snaps = compute_all_ratings(matches)
        secondary = {m: s.trueskill_sigma for m, s in snaps.items()}
    return matches, played, secondary


# ── challenge-slot plan ────────────────────────────────────────


def _slot_plan(
    store: ChallengeStore,
    cid: str,
    repo_names: list[str],
    turns_per_player: int,
) -> dict[str, dict[str, list[int]]]:
    """Per-repo slot status for a participant's (Red) challenge pool.

    A slot is **reused from the cache** when it has any attempt record in the
    bank (an admitted challenge, or an exhausted failure chain); only slots
    with no record at all are generated. Mirrors ``ChallengeGenerator.
    generate_pool``'s skip rule exactly.
    """
    model, harness, effort, provider = split_composite_id(cid)
    harness = harness or "mini-swe-agent"
    plan: dict[str, dict[str, list[int]]] = {}
    for repo in repo_names:
        attempts = store.list_attempts_by_slot(
            model, repo, harness, effort, provider
        )
        admitted: list[int] = []
        exhausted: list[int] = []
        to_generate: list[int] = []
        for slot in range(1, turns_per_player + 1):
            prior = attempts.get(slot)
            if prior is None:
                to_generate.append(slot)
            elif any(str(a.get("status")) == "success" for a in prior):
                admitted.append(slot)
            else:
                exhausted.append(slot)
        plan[repo] = {
            "admitted": admitted,
            "exhausted": exhausted,
            "to_generate": to_generate,
        }
    return plan


# ── generation phase ────────────────────────────────────────────


def _run_generation_pools(
    pool_tasks: list[tuple[str, ModelConfig, str, RepoConfig, int]],
    *,
    store: ChallengeStore,
    workspace_manager: WorkspaceManager,
    arena_cfg: ArenaConfig,
    prompt_dir: Path,
    max_workers: int,
) -> None:
    """Generate challenges for the given (cid, config, harness, repo, target)
    pools in parallel under the live generation TUI.

    Identical worker construction to ``swe-duel-generate``; ``generate_pool``
    itself skips every slot that already has an attempt record, so this only
    ever generates never-attempted slots.
    """
    reporter = GenerationReporter(title="Active-sampling challenge generation")
    handles = {
        (cid, rc.name): reporter.register_pool(
            display_composite_id(cid), rc.name, target, harness=harness
        )
        for cid, _cfg, harness, rc, target in pool_tasks
    }

    def _run_pool(task: tuple[str, ModelConfig, str, RepoConfig, int]) -> None:
        cid, model_cfg, harness_id, repo_cfg, target = task
        handle = handles[(cid, repo_cfg.name)]
        wrapper = get_harness(harness_id, model_cfg)
        red_agent = RedAgent(
            agent_wrapper=wrapper,
            workspace_manager=workspace_manager,
            prompt_dir=prompt_dir,
            feature_wall_seconds=arena_cfg.agent_timeouts.red_feature_seconds,
            bug_wall_seconds=arena_cfg.agent_timeouts.red_bug_seconds,
            feature_steps=arena_cfg.agent_steps.red_feature_steps,
            bug_steps=arena_cfg.agent_steps.red_bug_steps,
            min_diff_lines=arena_cfg.red_gates.min_diff_lines,
            min_test_assertions=arena_cfg.red_gates.min_test_assertions,
            min_test_functions=arena_cfg.red_gates.min_test_functions,
        )
        executor = DockerExecutor(
            docker_image=repo_cfg.docker_image,
            timeout_s=arena_cfg.sandbox.timeout_seconds,
            memory_mb=arena_cfg.sandbox.memory_mb,
        )
        test_runner = TestRunner(executor=executor, repo_config=repo_cfg)
        validator = RedGateValidator(
            test_runner=test_runner,
            config=arena_cfg,
            agent_wrapper=wrapper,
            workspace_manager=workspace_manager,
            prompt_dir=prompt_dir,
            self_review_wall_seconds=arena_cfg.agent_timeouts.red_self_review_seconds,
            self_review_steps=arena_cfg.agent_steps.red_self_review_steps,
        )
        generator = ChallengeGenerator(
            store=store,
            red_gate_validator=validator,
            workspace_manager=workspace_manager,
            config=arena_cfg,
        )
        generator.generate_pool(
            red_agent=red_agent,
            repo_config=repo_cfg,
            target_count=target,
            red_model_id=model_cfg.model_id,
            pool_reporter=handle,
        )

    workers = max(1, min(max_workers, len(pool_tasks)))
    print(
        f"[as] generating challenges for {len(pool_tasks)} pool(s) "
        f"(max_workers={workers})"
    )
    with reporter_live(reporter):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_pool, t): t for t in pool_tasks}
            try:
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001
                        handles[(task[0], task[3].name)].finish_pool(
                            _Status.FAIL,
                            detail=f"error: {type(e).__name__}: {str(e)[:80]}",
                        )
            except KeyboardInterrupt:
                print(
                    "\n[as] interrupted — cancelling pending pools...",
                    file=sys.stderr,
                )
                for fut in futures:
                    fut.cancel()
                raise


# ── match phase ─────────────────────────────────────────────────


def _turn_reuse_counts(
    orchestrator: MatchOrchestrator,
    pairing: SwissPairing,
    repo_names: list[str],
    turns_per_player: int,
) -> tuple[int, int]:
    """(reusable, new) defense sub-turn counts for one pairing.

    Auto-win turns (Red with no challenge for a repo) count as neither —
    identical to the tournament consoles' preview helper.
    """
    store = orchestrator.challenge_store
    al = orchestrator.artifact_logger
    a_id, b_id = _pair_ids(pairing)
    reuse = 0
    new = 0
    for red_id, blue_id in ((a_id, b_id), (b_id, a_id)):
        red_model, red_harness, red_effort, red_provider = split_composite_id(red_id)
        blue_model, blue_harness, blue_effort, blue_provider = split_composite_id(blue_id)
        for repo_name in repo_names:
            challenge_ids = store.list_ids_in_index_order(
                red_model,
                repo_name,
                red_harness_id=red_harness or "mini-swe-agent",
                red_reasoning_effort=red_effort,
                red_provider=red_provider,
            )[:turns_per_player]
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


def _run_matchups_parallel(
    orchestrator: MatchOrchestrator,
    pairings: list[SwissPairing],
    repo_cfgs: list[RepoConfig],
    *,
    max_workers: int,
    round_index: int = 1,
) -> list[tuple[SwissPairing, MatchResult | None]]:
    """Run every sampled pairing by pooling all of their cache-miss defense
    sub-turns into one ThreadPoolExecutor under the live round TUI.

    Same phased plan → pool → assemble walk as the tournament consoles'
    ``_run_round_parallel``: cached defenses are reused at zero cost, failed
    sub-turns truncate that turn instead of crashing the batch.
    """
    plans: dict[str, MatchPlan] = {}
    plan_by_pairing: list[tuple[SwissPairing, MatchPlan | None]] = []
    for i, p in enumerate(pairings):
        a_id, b_id = _pair_ids(p)
        try:
            plan = orchestrator.plan_match(
                model_a_id=a_id,
                model_b_id=b_id,
                repo_config=repo_cfgs,
                seed=1000 + i,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  ! planning failed for {_short(a_id)} vs "
                f"{_short(b_id)}: {e}"
            )
            plan_by_pairing.append((p, None))
            continue
        plans[plan.match_key] = plan
        plan_by_pairing.append((p, plan))

    reporter = DefenseRoundReporter(round_index=round_index)
    all_tasks: list[DefenseTask] = []
    handles: dict[int, SubturnHandle] = {}
    for p, mplan in plan_by_pairing:
        if mplan is None:
            continue
        a_id, b_id = _pair_ids(p)
        label = f"{_short(a_id)} vs {_short(b_id)}"
        reporter.register_match(mplan.match_key, label)
        hits = mplan.cache_hits()
        if hits:
            reporter.finish_match(
                mplan.match_key, SubturnStatus.RUNNING, detail=f"{hits} cached"
            )
        for task in mplan.pending_tasks:
            if task.side == "b_defends_a":
                blue_cid, red_cid = b_id, a_id
            else:
                blue_cid, red_cid = a_id, b_id
            sid = (
                f"{task.repo_config.name}:{task.side}:{task.slot}:"
                f"{task.challenge.challenge_id[:8]}"
            )
            reg_handle = reporter.register_subturn(
                mplan.match_key,
                sid,
                label=(
                    f"blue={_short(blue_cid)} defends "
                    f"{_short(red_cid)}'s {task.repo_config.name} challenge"
                ),
                blue_model=task.blue_model_id,
                repo=task.repo_config.name,
            )
            handles[id(task)] = reg_handle
            all_tasks.append(task)

    total_hits = sum(
        mplan.cache_hits() for _, mplan in plan_by_pairing if mplan is not None
    )
    effective = min(max_workers, len(all_tasks)) or 1
    print(
        f"  pool: {len(all_tasks)} defense sub-turns to run "
        f"({total_hits} reused from cache) across {len(plans)} matches "
        f"— max_workers={max_workers}, using {effective}\n"
    )

    t0 = time.perf_counter()
    done = 0
    if all_tasks:
        with round_live(reporter):
            with ThreadPoolExecutor(max_workers=effective) as pool:
                fut_to_task = {
                    pool.submit(
                        orchestrator.run_defense_task,
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
                    orchestrator.place_result(plans[task.match_key], task, result)
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

    executed: list[tuple[SwissPairing, MatchResult | None]] = []
    for p, fplan in plan_by_pairing:
        if fplan is None:
            executed.append((p, None))
            continue
        a_id, b_id = _pair_ids(p)
        try:
            m = orchestrator.assemble_match(fplan)
        except Exception as e:  # noqa: BLE001
            print(
                f"    ! assembling match failed for {_short(a_id)} vs "
                f"{_short(b_id)}: {e}"
            )
            executed.append((p, None))
            continue
        print(
            f"  → {_short(a_id)} vs {_short(b_id)}: "
            f"{m.outcome.value}  A={m.model_a_total:.2f} "
            f"B={m.model_b_total:.2f}  ${m.total_cost_usd:.4f}"
        )
        executed.append((p, m))
    return executed


# ── state persistence ───────────────────────────────────────────


def _persist_as_state(path: Path, payload: dict[str, Any]) -> None:
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ── contribution zip ─────────────────────────────────────────────

# Only these ./data subtrees may ever enter a contribution zip — the record
# trees the organizer's import reads. Everything else is excluded, above all
# workspaces/ (gigabytes of bind-mount residue whose every needed fact is
# already extracted into the .json records) and logs/.
_ZIP_ALLOWED_TOP_DIRS = frozenset({"tournaments", "matches", "challenge_bank", "defenses"})


def _zip_payload(src: Path) -> dict[str, Any] | None:
    """Parsed JSON object of a packaged record, or ``None`` when the file is
    missing/unparseable/not a JSON object."""
    try:
        payload = json.loads(src.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _verify_contribution_zip_members(
    data_dir: Path,
    members: list[Path],
    *,
    state_paths: list[Path],
    match_ids: set[str],
    identities: set[_Identity],
    turn_challenge_ids: set[str],
    turn_defense_ids: set[str],
    newcomer_identities: set[_Identity],
) -> None:
    """Fail fast unless ``members`` is exactly the matchup-relevant records.

    Re-derives every member's relevance from disk, independently of the
    collection walk, so a packaging bug can never silently ship junk (or
    miss the scope): only the record subtrees of ``./data`` are ever
    allowed — ``workspaces/`` and ``logs/`` never are; a state file must be
    one of the selected same-tournament active-sampling states; a match
    file must be one the selected states claim, with both endpoints in the
    field; a challenge must be one the claimed matches' turns reference
    (by ``challenge_id``), a defense one they reference by ``defense_id``
    — the incumbents' other bank records are the organizer's data
    already and never package; a failed attempt must belong to one of the
    selected states' new entrants (the new bank state the matchups' slot
    plan drew from); an ``.html`` member must sit next to its verified
    ``.json`` record. Any violation raises ``SystemExit`` before the zip
    is written.
    """
    state_set = set(state_paths)
    member_set = set(members)

    def _fail(src: Path, why: str) -> SystemExit:
        try:
            rel = src.relative_to(data_dir).as_posix()
        except ValueError:
            rel = str(src)
        return SystemExit(f"[as] refusing to package {rel}: {why}")

    for src in members:
        try:
            rel = src.relative_to(data_dir)
        except ValueError:
            raise _fail(src, "not inside the packaged ./data directory") from None
        top = rel.parts[0] if rel.parts else ""
        if top not in _ZIP_ALLOWED_TOP_DIRS:
            raise _fail(
                src,
                "only the record subtrees ("
                + ", ".join(sorted(_ZIP_ALLOWED_TOP_DIRS))
                + ") of ./data belong in a contribution zip — workspaces/ "
                "and logs/ never do (every fact the import needs is "
                "already extracted into the .json records)",
            )
        if src.suffix == ".html":
            if src.with_suffix(".json") not in member_set:
                raise _fail(src, "html twin without its .json record")
            continue
        if top == "tournaments":
            if src not in state_set:
                raise _fail(src, "not one of the selected active-sampling state files")
        elif top == "matches":
            if src.stem not in match_ids:
                raise _fail(
                    src,
                    "not one of the matches the selected states claim "
                    "(pairings walk + results references)",
                )
            payload = _zip_payload(src)
            if payload is None:
                raise _fail(src, "unparseable match record")
            endpoints = {
                _Identity.from_cid(str(payload.get(key) or ""))
                for key in ("model_a_id", "model_b_id")
            }
            if not endpoints <= identities:
                raise _fail(
                    src,
                    "a match endpoint is not one of the field participants "
                    "the selected states cover",
                )
        elif top == "defenses":
            payload = _zip_payload(src)
            defense_id = payload.get("defense_id") if payload is not None else None
            if not isinstance(defense_id, str) or defense_id not in turn_defense_ids:
                raise _fail(
                    src,
                    "not one of the defenses the claimed matches' turns "
                    "reference — the incumbents' pre-existing defenses are "
                    "the organizer's data already",
                )
        elif top == "challenge_bank":
            if len(rel.parts) == 2 and src.name == "index.json":
                continue
            if len(rel.parts) == 3 and rel.parts[1] in (
                "challenges",
                "failed_challenges",
            ):
                payload = _zip_payload(src)
                if payload is None:
                    raise _fail(src, "unparseable bank record")
                if rel.parts[1] == "challenges":
                    challenge_id = payload.get("challenge_id")
                    if (
                        not isinstance(challenge_id, str)
                        or challenge_id not in turn_challenge_ids
                    ):
                        raise _fail(
                            src,
                            "not one of the challenges the claimed matches' "
                            "turns reference — the incumbents' pre-existing "
                            "bank is the organizer's data already",
                        )
                else:
                    identity = _record_identity(payload, "red")
                    if identity is None or identity not in newcomer_identities:
                        raise _fail(
                            src,
                            "Red identity is not one of the new entrants "
                            "the selected states brought — only an entrant's "
                            "own failure chains package",
                        )
                continue
            raise _fail(src, "unexpected challenge_bank file")


def _config_zip_members(config_dir: Path) -> list[Path]:
    """Every file under ``config_dir``, sorted by its archive member name.

    The whole ``./config`` directory is matchup-relevant — it is the
    deployment's editable arena/models/repos configuration that shaped
    every packaged record — so unlike the ``./data`` slice there is no
    per-record scoping: every file found by the unreadable-subtree-safe
    walk (:func:`swe_duel.sandbox.diff_utils.iter_files_safe`) packages.
    """
    if not config_dir.is_dir():
        raise SystemExit(
            f"[as] refusing to package config: {config_dir} is not a "
            "directory — expected the resolved ./config (swe-duel init)."
        )
    return sorted(
        iter_files_safe(config_dir),
        key=lambda p: "config/" + p.relative_to(config_dir).as_posix(),
    )


def _verify_config_zip_members(config_dir: Path, members: list[Path]) -> None:
    """Fail fast unless ``members`` is exactly every file under
    ``config_dir``.

    Re-derives the expected set from disk, independently of the collection
    walk — the same guarantee the data-slice verifier gives the record
    subtrees: a file added (or deleted) under ``./config`` mid-run, or a
    path outside the directory, fails packaging before the zip is written.
    """
    member_set = set(members)
    expected = set(iter_files_safe(config_dir))
    unexpected = sorted(member_set - expected)
    missing = sorted(expected - member_set)
    if unexpected or missing:
        detail = (
            f"unexpected config member(s): {unexpected}. " if unexpected else ""
        ) + (f"missing config member(s): {missing}." if missing else "")
        raise SystemExit(
            f"[as] refusing to package config: {detail} The config/ archive "
            f"root must mirror {config_dir} exactly — re-run the command to "
            "prepare the zip again."
        )


def _verify_written_zip(target: Path, expected_arcs: set[str]) -> None:
    """Reopen the written archive and fail fast — deleting it — unless it
    contains exactly ``expected_arcs``: the verified member set and nothing
    else (a ``workspaces/`` leak or a mid-write mutation lands here)."""
    with zipfile.ZipFile(target) as zf:
        actual = set(zf.namelist())
    if actual == expected_arcs:
        return
    unexpected = sorted(actual - expected_arcs)
    missing = sorted(expected_arcs - actual)
    target.unlink()
    raise SystemExit(
        f"[as] contribution zip verification failed for {target}: "
        + (f"unexpected member(s): {unexpected}. " if unexpected else "")
        + (f"missing member(s): {missing}." if missing else "")
        + " The zip was deleted; nothing was packaged — re-run the "
        "command to prepare it again."
    )


def _prepare_contribution_zip(
    data_dir: Path,
    state_path: Path,
    state_payload: dict[str, Any],
    match_ids: list[str],
    out_dir: Path,
    *,
    zip_path: Path | None = None,
    config_dir: Path,
) -> Path:
    """Write the end-of-flow contribution zip for the organizer's
    ``swe-duel-tournament-update`` import.

    The zip carries the **matchup-relevant slice** of this machine's
    ``./data`` — workspaces, logs, and every record the sampled matchups
    never touched stay out of it:

    * every local ``active_sampling_state_*.json`` entering the same
      tournament (this run's own state included; earlier sessions that
      entered the same tournament join so their matches and entrants are
      claimed by the import too — without a resolvable entered tournament
      the scope falls back to this run's state plus the ``match_ids``
      argument);
    * the match files those states claim — via
      :func:`swe_duel.cli.export_rankings._load_as_extra_matches`, the
      import's own claiming pipeline (pairings walk + ``results``
      references, deduped by ``match_id``);
    * the challenge records **those matches' turns reference** (by
      ``challenge_id``) — not every bank record under a field
      participant's identity: the incumbents' pools already live on the
      organizer's side, and identity scoping ballooned earlier
      submissions to the whole original tournament's bank;
    * the defense records the same turns reference (by ``defense_id``);
    * the failed-attempt chains of the **new entrants** the selected
      states brought — the new bank state the matchups' slot plan drew
      from, so an exhausted slot stays exhausted on the organizer's side
      too (future runs reuse it instead of regenerating);
    * each packaged record with its ``.html`` twin when present, plus
      ``challenge_bank/index.json`` — the import rebuilds its index
      entries from the payload files, but the index keeps the zip a
      complete bank snapshot of the contributed pools.

    On top of the data slice, the **whole ``config_dir``** (the
    deployment's editable ``./config`` — arena.yaml / models.yaml /
    repos/*.yaml, the configuration that shaped every packaged record)
    packages under a ``config/`` archive root: every file found by the
    unreadable-subtree-safe walk, verified
    (``_verify_config_zip_members``) to mirror the directory exactly.
    The import's extraction only ever reads the ``data/`` root, so the
    config subtree rides along as provenance (the organizer's archive
    keeps the whole zip).

    The data member set is **verified**
    (``_verify_contribution_zip_members``) before the zip is written, and
    the written archive re-verified (``_verify_written_zip``): every
    member must be one of the matchup-relevant records above —
    ``workspaces/`` / ``logs/`` (and anything else under ``./data``)
    never package, and any violation fails fast with the offending path.
    Data members are stored under a ``data/`` archive root, one of the
    layouts the import's ``_locate_data_prefix`` accepts. Returns the zip
    path.
    """
    tournaments_dir = data_dir / "tournaments"
    entered = _as_entered_tournament(state_payload)
    if entered:
        states = [
            (p, s)
            for p, s in _load_as_states(tournaments_dir)
            if _as_entered_tournament(s) == entered
        ]
        claimed = _load_as_extra_matches(tournaments_dir, data_dir / "matches", entered)
        match_ids = [str(m.get("match_id") or "") for m in claimed]
    else:
        states = [(state_path, state_payload)]
    state_paths = list(dict.fromkeys(p for p, _s in states))
    identities = {_Identity.from_cid(cid) for _p, s in states for cid in _as_field(s)}
    newcomer_identities = {
        _Identity.from_cid(cid) for _p, s in states for cid in _as_newcomers(s)
    }

    members: list[Path] = list(state_paths)
    counts: dict[str, int] = {
        "states": len(state_paths),
        "matches": 0,
        "challenges": 0,
        "failed_challenges": 0,
        "defenses": 0,
    }

    # Turn-referenced record ids, re-derived from the claimed match
    # payloads on disk: only a real defense's challenge/defense records
    # are matchup-relevant. A missing-Red auto-win turn (empty
    # challenge_id) carries a *synthetic* defense_id and a perfect score —
    # no defense ever ran, so no record/log/workspace ever existed for it
    # by design; it must be neither scoped in nor warned about.
    turn_challenge_ids: set[str] = set()
    turn_defense_ids: set[str] = set()

    matches_dir = data_dir / "matches"
    for mid in dict.fromkeys(m for m in match_ids if m):
        src = matches_dir / f"{mid}.json"
        if not src.is_file():
            continue
        members.append(src)
        counts["matches"] += 1
        payload = _zip_payload(src)
        if payload is None:
            print(
                f"[as] warning: claimed match {mid} is unparseable — its "
                "turns' records are not scoped in",
                file=sys.stderr,
            )
            continue
        for turn in payload.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            challenge_id = turn.get("challenge_id")
            defense_id = turn.get("defense_id")
            if not (isinstance(challenge_id, str) and challenge_id):
                continue
            turn_challenge_ids.add(challenge_id)
            if isinstance(defense_id, str) and defense_id:
                turn_defense_ids.add(defense_id)

    bank_dir = data_dir / "challenge_bank"
    for cid in sorted(turn_challenge_ids):
        src = bank_dir / "challenges" / f"{cid}.json"
        if src.is_file():
            members.append(src)
            counts["challenges"] += 1
        else:
            print(
                f"[as] warning: turn-referenced challenge {cid} has no "
                "record file under the bank",
                file=sys.stderr,
            )

    def _is_newcomer_record(src: Path) -> bool:
        payload = _zip_payload(src)
        if payload is None:
            print(f"[as] warning: skipping unparseable {src}", file=sys.stderr)
            return False
        identity = _record_identity(payload, "red")
        return identity is not None and identity in newcomer_identities

    for src in sorted((bank_dir / "failed_challenges").glob("*.json")):
        if _is_newcomer_record(src):
            members.append(src)
            counts["failed_challenges"] += 1

    for did in sorted(turn_defense_ids):
        src = data_dir / "defenses" / f"{did}.json"
        if src.is_file():
            members.append(src)
            counts["defenses"] += 1
        else:
            print(
                f"[as] warning: turn-referenced defense {did} has no "
                "record file under data/defenses",
                file=sys.stderr,
            )

    twins = [t for t in (m.with_suffix(".html") for m in members) if t.is_file()]
    index_path = bank_dir / "index.json"
    members = sorted(
        dict.fromkeys(
            [
                *members,
                *twins,
                *([index_path] if index_path.is_file() else []),
            ]
        ),
        key=lambda p: "data/" + p.relative_to(data_dir).as_posix(),
    )
    config_members = _config_zip_members(config_dir)

    # Packaging verification: the member set must be exactly the matchup-
    # relevant records (selected states + their claimed matches + the
    # records those matches' turns reference + the entrants' failed
    # chains + the bank index) — workspaces/, logs/ and anything else
    # never packages — plus the ./config directory mirrored exactly.
    # Runs before the zip is written and again on the written archive.
    _verify_contribution_zip_members(
        data_dir,
        members,
        state_paths=state_paths,
        match_ids={m for m in match_ids if m},
        identities=identities,
        turn_challenge_ids=turn_challenge_ids,
        turn_defense_ids=turn_defense_ids,
        newcomer_identities=newcomer_identities,
    )
    _verify_config_zip_members(config_dir, config_members)

    out_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", entered) or "tournament"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = (
        zip_path if zip_path is not None else out_dir / f"as_submission_{label}_{ts}.zip"
    )
    arcs = [
        *((src, "data/" + src.relative_to(data_dir).as_posix()) for src in members),
        *(
            (src, "config/" + src.relative_to(config_dir).as_posix())
            for src in config_members
        ),
    ]
    total = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in arcs:
            total += src.stat().st_size
            zf.write(src, arc)

    _verify_written_zip(target, {arc for _src, arc in arcs})
    print(
        f"[as] contribution zip → {target}\n"
        f"     verified {len(arcs)} member(s): {counts['states']} "
        f"active-sampling state(s), {counts['matches']} match(es), "
        f"{counts['challenges']} challenge(s), "
        f"{counts['failed_challenges']} failed attempt(s), "
        f"{counts['defenses']} defense(s), {len(twins)} html twin(s)"
        + (", bank index" if index_path.is_file() else "")
        + f", {len(config_members)} config file(s)"
        f" — {total / 1024 / 1024:.1f} MiB raw, "
        f"{target.stat().st_size / 1024 / 1024:.1f} MiB zipped\n"
        "     only the records of this tournament's active-sampled matchups "
        "plus the ./config directory; workspaces/ and logs/ are excluded "
        "(every fact the import needs lives in the .json records)"
    )
    print(f"     SHA-256 {_zip_sha256(target)}")
    print(
        "     submit it to the organizer: swe-duel-tournament-update "
        f"{target}{f' --tournament {entered}' if entered else ''}"
    )
    return target


def _offer_contribution_zip(
    data_dir: Path,
    state_path: Path,
    state_payload: dict[str, Any],
    match_ids: list[str],
    out_dir: Path,
    *,
    config_dir: Path,
) -> None:
    """End-of-flow prompt: prepare the contribution zip (declining, or
    Ctrl+C at the prompt, only skips it — every record stays persisted
    under ``./data`` either way)."""
    confirm = questionary.confirm(
        "Prepare a contribution .zip of the matchup-relevant ./data/ (this "
        "tournament's active-sampling states + their claimed matches + the "
        "challenges/defenses those matches' turns reference + the entrants' "
        "failed attempts) plus your ./config/ directory for the "
        "organizer's swe-duel-tournament-update import?",
        default=True,
    ).ask()
    if not confirm:
        print("[as] contribution zip skipped.")
        return
    _prepare_contribution_zip(
        data_dir,
        state_path,
        state_payload,
        match_ids,
        out_dir,
        config_dir=config_dir,
    )


# ── entry point ─────────────────────────────────────────────────


def main() -> int:
    install_container_cleanup_handlers()

    parser = argparse.ArgumentParser(
        description=(
            "Active-sampled tournament entry: pick a tournament via "
            "--tournament <id> (its ./rankings/rankings_<id>.json export, "
            "produced by swe-duel-rankings), add new entrants via the paged "
            "participant wizard, and the Chatbot-Arena SE-reduction rule "
            "picks each entrant's opponents from the whole rankings field "
            "(every matchup is new entrant vs. an existing participant). "
            "Interrupted runs are resumable: --resume-state <id> replays a "
            "persisted active_sampling_state_<id>.json (its entrants, "
            "pairings, and arena) without re-sampling and ends at the "
            "contribution-zip prompt."
        )
    )
    parser.add_argument(
        "--tournament",
        default=None,
        help=(
            "Tournament id to enter: loads ./rankings/rankings_<id>.json "
            "(the export swe-duel-rankings wrote for it). Required unless "
            "--rankings is given — this is how new participants choose "
            "which tournament to enter."
        ),
    )
    parser.add_argument(
        "--rankings",
        default=None,
        help="Explicit rankings export path (alternative to --tournament).",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=None,
        help=(
            "Skip the interactive wizard and add these NEW entrants "
            "(participants not in the rankings export; they join the field "
            "unrated): any 'model#harness[#effort#provider]' composite. "
            "Every rankings participant is always actively sampled — the "
            "flag cannot (de)select field members and naming one fails fast."
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Incumbent matches per new entrant under active sampling "
            "(default: ceil(log2 N)+1 over the existing field, the "
            "Chatbot-Arena intake rule)."
        ),
    )
    parser.add_argument(
        "--resume-state",
        dest="resume_state",
        default=None,
        help=(
            "Resume a previous active-sampling run instead of sampling "
            "fresh matchups: the active-sampling state id (the uuid the "
            "console printed when it sampled, e.g. 39e956d4-…), its file "
            "stem, or a direct path to the "
            "active_sampling_state_<id>.json under the output dir. The "
            "state's own entrants, sampled pairings, and recorded arena "
            "are replayed (no re-sampling, no new spend), phases it "
            "already completed are skipped, and the flow ends at the "
            "contribution-zip prompt — this is how a run interrupted "
            "before packaging (e.g. a closed terminal) gets its zip "
            "without re-running anything. The rankings export defaults "
            "to the one the state entered."
        ),
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help=(
            "One or more repos for the sampled matches (a single match spans "
            "ALL of them). Default: the 'repos' list declared in the "
            "rankings export — the exact set the tournament ran. When the "
            "export carries repos, any --repos value MUST be the same set "
            "(differing values are rejected: a different arena would bias "
            "the standings)."
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
        "--turns-per-agent",
        dest="turns_per_player",
        type=int,
        default=None,
        help=(
            "Challenge slots per repo per participant (the match's "
            "turns_per_player). Default: the rankings export's "
            "'targets_per_repo', else arena.yaml match.turns_per_player. "
            "When the export carries targets_per_repo, any --turns-per-player "
            "value MUST equal it (differing values are rejected: a different "
            "challenge count would bias the standings)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel (participant × repo) generation pools (override "
        "arena.yaml challenge_bank.max_workers).",
    )
    parser.add_argument(
        "--match-workers",
        dest="match_workers",
        type=int,
        default=None,
        help="Parallel defense sub-turns (override arena.yaml "
        "match.max_workers).",
    )
    parser.add_argument(
        "--submission-dir",
        default="./submissions",
        help=(
            "Where the end-of-run contribution .zip is written (default "
            "./submissions). The zip carries the matchup-relevant ./data "
            "records a swe-duel-tournament-update import consumes (this "
            "tournament's active-sampling states, the matches they claim, "
            "the challenges/defenses those matches' turns reference, and "
            "the entrants' failed attempts) plus the whole ./config "
            "directory."
        ),
    )
    args = parser.parse_args()

    # ── Resume (--resume-state): replay a persisted run ─────────
    # The state file is located BEFORE setup: its recorded repo_names drive
    # the setup filter (exactly like the export's repos do for a fresh run),
    # so preflight only checks the repos the resumed matchups actually span.
    plan: _ResumePlan | None = None
    if args.resume_state:
        if args.participants:
            raise SystemExit(
                "--participants cannot be combined with --resume-state — "
                "the resumed state already carries its new entrants."
            )
        if args.budget is not None:
            raise SystemExit(
                "--budget cannot be combined with --resume-state — the "
                "resumed state's matchups were already sampled."
            )
        pre_arena = load_arena_config(resolve_config_dir(args))
        pre_out = resolve_output_dir(args, pre_arena)
        plan = _load_resume_plan(args.resume_state, pre_out / "tournaments")
        print(
            f"[as] resuming {plan.state_path.name} — status "
            f"{plan.state_payload.get('status')!r}, {len(plan.pairings)} "
            "sampled matchup(s), entrants: "
            + ", ".join(display_composite_id(c) for c in plan.newcomers)
        )

    # Parse the rankings export FIRST (a pure file read): its repo list and
    # per-repo target count feed the setup filter, so preflight only checks
    # the repos this run actually touches. --tournament resolves the export
    # for the tournament the new participants are entering; a resumed state
    # defaults to the export it entered (its record must agree with any
    # explicit flag — a different export means a different field).
    rankings_path = (
        _resolve_resume_rankings(args, plan, Path("./rankings"))
        if plan is not None
        else _resolve_rankings_path(args)
    )
    field_data = _load_rankings(rankings_path)
    # The export's arena (repos / targets_per_repo) is enforced, not
    # overridable — a different arena would bias the standings.
    _enforce_exported_arena(args, field_data, rankings_path)
    entries = field_data.entries
    field: list[str] = [e["cid"] for e in entries]
    by_cid = {e["cid"]: e for e in entries}
    if plan is not None:
        # The state's recorded arena is authoritative for what actually ran
        # (its repo list is the export's set minus any name missing from
        # config/repos/ at the time); a mismatch means the export changed
        # since the run — resuming against it would bias the standings.
        if field_data.repos and not set(plan.repo_names) <= set(field_data.repos):
            raise SystemExit(
                f"--resume-state: the state ran repos {plan.repo_names}, "
                f"not all of which are in the export's arena "
                f"{field_data.repos} ({rankings_path}) — the export changed "
                "since the run."
            )
        if (
            plan.targets_per_repo is not None
            and field_data.targets_per_repo is not None
            and plan.targets_per_repo != field_data.targets_per_repo
        ):
            raise SystemExit(
                f"--resume-state: the state ran targets_per_repo="
                f"{plan.targets_per_repo}, the export declares "
                f"{field_data.targets_per_repo} ({rankings_path}) — the "
                "export changed since the run."
            )

    # Repos: default to the tournament's own repo set from the rankings
    # export; --repos must repeat exactly that set (validated above).
    # A resumed state replays the repos it actually ran. Unknown names are
    # skipped with a warning below.
    if plan is not None:
        if args.repos and set(args.repos) != set(plan.repo_names):
            print(
                f"[as] note: --repos {sorted(args.repos)} ignored — the "
                f"resumed state ran {plan.repo_names}"
            )
        requested_repos = list(plan.repo_names)
        repo_source = "the resumed state"
    else:
        requested_repos = list(args.repos) if args.repos else list(field_data.repos)
        repo_source = "--repos" if args.repos else "the rankings export"
    if not requested_repos:
        raise SystemExit(
            "no repos to run over: pass --repos REPO... or export a 'repos' "
            f"list in {rankings_path}"
        )

    args_for_setup = argparse.Namespace(**vars(args))
    args_for_setup.models = None
    args_for_setup.repos = requested_repos
    ctx = setup(args_for_setup)
    arena_cfg: ArenaConfig = ctx["arena_config"]

    # Challenge slots per (participant, repo): the rankings export records the
    # tournament's own target count; a CLI flag still wins; a legacy export
    # without it falls back to arena.yaml match.turns_per_player. The value
    # doubles as the match's turns_per_player (one defense turn per slot).
    if args.turns_per_player is not None:
        arena_cfg.match.turns_per_player = args.turns_per_player
    elif field_data.targets_per_repo is not None:
        arena_cfg.match.turns_per_player = field_data.targets_per_repo
    if plan is not None and plan.targets_per_repo is not None:
        if args.turns_per_player is not None and args.turns_per_player != plan.targets_per_repo:
            print(
                f"[as] note: --turns-per-player {args.turns_per_player} "
                "ignored — the resumed state ran "
                f"{plan.targets_per_repo}"
            )
        arena_cfg.match.turns_per_player = plan.targets_per_repo
    if args.match_workers is not None:
        arena_cfg.match.max_workers = args.match_workers
    turns_per_player = arena_cfg.match.turns_per_player

    repo_names: list[str] = []
    for rn in requested_repos:
        if rn not in ctx["repo_configs"]:
            print(
                f"[warn] repo {rn!r} not found in config/repos/ — skipping. "
                f"Available: {sorted(ctx['repo_configs'])}",
                file=sys.stderr,
            )
        elif rn not in repo_names:
            repo_names.append(rn)
    if not repo_names:
        raise SystemExit(
            f"none of the requested repos {requested_repos!r} matched "
            f"config/repos/. Available: {sorted(ctx['repo_configs'])}"
        )
    repo_cfgs: list[RepoConfig] = [ctx["repo_configs"][rn] for rn in repo_names]
    if plan is not None and set(repo_names) != set(plan.repo_names):
        raise SystemExit(
            f"--resume-state: the resumed state ran {plan.repo_names} but "
            f"only {repo_names} are present in config/repos/ — resuming on "
            "a different arena would bias the standings. Restore the repo "
            "configs and re-run."
        )

    print(
        f"\n[as] field = {len(field)} participant(s) from {rankings_path} — "
        f"repos={len(repo_names)} from {repo_source}, "
        + (
            f"targets_per_repo={turns_per_player} (rankings)"
            if field_data.targets_per_repo is not None
            else f"turns_per_player={turns_per_player} (arena default)"
        )
        + ":"
    )
    for e in entries:
        extra = ""
        if e.get("bradley_terry") is not None:
            extra = (
                f"  pts={e['points']:.1f}  BT={e['bradley_terry']}"
                f"  Elo={e['elo']}"
            )
        print(f"    {e['rank']:>2}. {display_composite_id(e['cid'])}{extra}")

    # ── New entrants ────────────────────────────────────────────
    # Every sampled matchup pairs a NEW ENTRANT with an EXISTING
    # participant: the entrant joins the field unrated and the
    # Chatbot-Arena SE-reduction rule picks its most ranking-informative
    # opponents from the WHOLE rankings field — never a hand-picked subset
    # (that would bias the standings) and never existing–existing pairs
    # (those belong to the original tournament's schedule). New entrants
    # are picked with the same paged model → harness → effort → provider
    # wizard as the other consoles; at least one is required — entering a
    # tournament means bringing a new participant.
    newcomers: list[str] = []
    if plan is not None:
        # The entrants come from the state itself (validated at load); an
        # entrant that has since appeared in the export means the organizer
        # already folded this run's contribution in — nothing to resume.
        for c in plan.newcomers:
            if c in by_cid:
                raise SystemExit(
                    f"--resume-state: entrant {display_composite_id(c)} is "
                    "now a rankings participant in the export — the "
                    "tournament already folded this run's contribution in "
                    "(re-export with swe-duel-rankings); nothing to resume."
                )
        newcomers = list(plan.newcomers)
    elif args.participants:
        newcomers = _resolve_newcomers(args.participants, field, by_cid)
    else:
        print(
            "\n[as] add the new entrants entering the tournament below — "
            "each joins the field unrated and is actively sampled against "
            "the existing participants (matchups are always new entrant "
            "vs. an existing participant)."
        )
        picked = select_participants(
            ctx["model_configs"],
            start_verb="Done adding new entrants",
        )
        for part in picked:
            if part.cid in by_cid:
                print(
                    f"[as] {part.label} is already a rankings participant — "
                    "not a new entrant; skipped."
                )
                continue
            newcomers.append(part.cid)
    newcomers = list(dict.fromkeys(newcomers))
    if not newcomers:
        raise SystemExit(
            "no new entrants were added — active sampling only runs "
            "matchups between new entrants and the rankings field, so "
            "there is nothing to sample. Add at least one new entrant "
            "(interactive wizard or --participants) to enter the "
            "tournament."
        )
    n_incumbents = len(field)

    # New entrants join the field unrated (no points/BT/Elo yet) and are
    # always sampled — entering a tournament means playing it.
    for c in newcomers:
        model, harness, _eff, _prov = split_composite_id(c)
        print(
            f"[as] new entrant: {display_composite_id(c)} — not in the "
            "rankings export; joins the field unrated."
        )
        entries.append(
            {
                "rank": None,
                "model": model,
                "harness": harness,
                "cid": c,
                "points": None,
                "elo": None,
                "bradley_terry": None,
                "newcomer": True,
            }
        )
        field.append(c)
        by_cid[c] = entries[-1]
    targets = list(newcomers)
    print(
        f"\n[as] active-sampling targets = the {len(targets)} new "
        f"entrant(s); opponents = all {n_incumbents} existing "
        "participant(s), picked by SE-reduction gain (no subset "
        "selection, no existing-vs-existing matchups)."
    )

    # ── Participant model configs (rankings models may not be in models.yaml)
    model_configs = _build_model_configs(
        ctx["model_configs"], field, arena_cfg.agent_model.max_tokens
    )
    warn_missing_pricing(model_configs.values())
    summarize_rate_limits(model_configs.values())
    warn_unenforceable_rate_limits(
        (cfg, split_composite_id(cid)[1]) for cid, cfg in model_configs.items()
    )

    # ── Matchup selection: fresh active sampling vs resume ──────
    stats_map: dict[frozenset[str], PairStats] = {}
    result_by_pair: dict[frozenset[str], dict[str, Any]] = {}
    pairings: list[SwissPairing]
    if plan is not None:
        # Resume: NO re-sampling — replay the state's recorded matchups and
        # continue its state file in place (never duplicated). Every pairing
        # endpoint must be a member of the field (export participants + the
        # state's entrants); a missing endpoint means the export changed
        # since the run.
        pairings = list(plan.pairings)
        for p in pairings:
            for endpoint in _pair_ids(p):
                if endpoint not in by_cid:
                    raise SystemExit(
                        f"--resume-state: pairing endpoint "
                        f"{display_composite_id(endpoint)} is neither a "
                        "rankings participant nor a recorded entrant — the "
                        f"export {rankings_path} changed since the run."
                    )
        for r in plan.state_payload.get("results") or []:
            if not isinstance(r, dict):
                continue
            a = str(r.get("model_a") or "")
            b = str(r.get("model_b") or "")
            if a and b:
                result_by_pair[pair_key(a, b)] = r
        state_path = plan.state_path
        state_payload = plan.state_payload
        print(
            f"[as] replaying the state's {len(pairings)} sampled matchup(s) "
            "— no re-sampling; the state file is continued in place."
        )
    else:
        # ── Prior matches among the field (data/matches + export head-to-head)
        prior_matches, already_played, secondary = _load_prior_matches(
            ctx["artifact_logger"].matches_dir,
            set(field),
            extra_payloads=field_data.matches,
        )
        print(
            f"[as] prior matches among the field: {len(prior_matches)} "
            f"({len(already_played)} pair(s) already played — not "
            "re-sampled; data/matches + the export's head-to-head)"
        )

        # ── Active sampling ────────────────────────────────────────
        # Each entrant faces up to `budget` EXISTING participants: cross-only
        # intake (an entrant never plays another entrant here, and
        # existing–existing pairs are never re-sampled — those belong to the
        # original tournament's schedule). n_incumbents ≥ 1 is guaranteed
        # (the rankings array is validated non-empty at load time), so the
        # budget is always ≥ 1.
        if args.budget is not None and args.budget < 1:
            raise SystemExit("--budget must be ≥ 1")
        budget = (
            args.budget
            if args.budget is not None
            else default_intake_budget(max(n_incumbents, 1))
        )
        budget = min(budget, n_incumbents)
        print(
            f"[as] budget = {budget} incumbent match(es) per new entrant "
            "(Chatbot-Arena SE-reduction picks the opponents from the whole "
            "field)"
        )

        pairings = select_intake_pairings(
            newcomers=newcomers,
            all_players=field,
            already_played=already_played,
            remaining_budget={c: budget for c in newcomers},
            matches=prior_matches,
            secondary_score=secondary,
            include_newcomer_pairs=False,
        )
        if not pairings:
            print(
                "\n[as] active sampling found no legal matchups — every new "
                "entrant has already played every existing participant. "
                "Re-run with a larger field or clear data/matches to resample."
            )
            return 0

        stats_map = aggregate_pair_stats(prior_matches)
        tournament_id = str(uuid.uuid4())
        state_path = (
            ctx["artifact_logger"].tournaments_dir
            / f"active_sampling_state_{tournament_id}.json"
        )
        state_payload = {
            "tournament_id": tournament_id,
            "format": "active_sampling",
            "rankings_source": str(rankings_path),
            "entered_tournament": args.tournament,
            "repo_names": list(repo_names),
            "targets_per_repo": turns_per_player,
            "field": list(field),
            "targets": list(targets),
            "newcomers": list(newcomers),
            "budget": budget,
            "prior_matches": len(prior_matches),
            "pairings": [
                {"model_a": a, "model_b": b}
                for a, b in (_pair_ids(p) for p in pairings)
            ],
            "status": "sampled",
        }
        _persist_as_state(state_path, state_payload)

    # ── Show the sampled matchups + challenge plan ─────────────
    # One participant-column width for every plan section below, wide enough
    # for the longest full 4-tuple identity so no segment is ever truncated.
    ident_w = max(
        len(_full_identity(c)) for p in pairings for c in _pair_ids(p)
    )
    print(f"\n  ── Active-sampled matchups ({len(pairings)}) ──")
    for i, p in enumerate(pairings, 1):
        a_id, b_id = _pair_ids(p)
        if plan is not None:
            # Resume: show what the state actually recorded per pairing —
            # the operator verifies the replayed run at a glance.
            r = result_by_pair.get(pair_key(a_id, b_id))
            if r is None:
                gain_note = "not run yet"
            elif str(r.get("status") or "") == "failed":
                gain_note = "FAILED (infra) — re-run offered below"
            else:
                gain_note = (
                    f"→ {r.get('outcome')}  "
                    f"A={float(r.get('score_a') or 0.0):.2f} "
                    f"B={float(r.get('score_b') or 0.0):.2f}"
                )
        elif pair_key(a_id, b_id) in stats_map:
            gain_note = f"SE-gain {pair_gain(a_id, b_id, stats_map):.3g}"
        else:
            gain_note = "unplayed (max SE-gain)"
        print(
            f"   {i:>2}. {_full_identity(a_id):<{ident_w}} vs  "
            f"{_full_identity(b_id):<{ident_w}}  {gain_note}"
        )
    print()

    players_needed: list[str] = []
    for p in pairings:
        for cid in _pair_ids(p):
            if cid not in players_needed:
                players_needed.append(cid)

    store: ChallengeStore = ctx["challenge_store"]
    artifact_logger: ArtifactLogger = ctx["artifact_logger"]

    # ── Orchestrator (for reuse counts + the match phase) ──────
    def _test_runner_factory(rc: RepoConfig) -> TestRunner:
        return TestRunner(
            executor=DockerExecutor(
                docker_image=rc.docker_image,
                timeout_s=arena_cfg.sandbox.timeout_seconds,
                memory_mb=arena_cfg.sandbox.memory_mb,
            ),
            repo_config=rc,
        )

    orchestrator = MatchOrchestrator(
        model_configs=model_configs,
        challenge_store=store,
        workspace_manager=ctx["workspace_manager"],
        config=arena_cfg,
        artifact_logger=artifact_logger,
        prompt_dir=Path(args.prompt_dir),
        test_runner_factory=_test_runner_factory,
    )

    print(
        f"  ── Defense-turn plan (turns_per_player={turns_per_player} "
        f"× {len(repo_names)} repo(s) × both sides) ──"
    )
    for i, p in enumerate(pairings, 1):
        reuse, new = _turn_reuse_counts(
            orchestrator, p, repo_names, turns_per_player
        )
        a_id, b_id = _pair_ids(p)
        print(
            f"   {i:>2}. {_full_identity(a_id):<{ident_w}} vs  "
            f"{_full_identity(b_id):<{ident_w}}  — {reuse} reusable / {new} new"
        )
    print()

    slot_plans = {
        cid: _slot_plan(store, cid, repo_names, turns_per_player)
        for cid in players_needed
    }
    gen_pools: list[tuple[str, ModelConfig, str, RepoConfig, int]] = []
    total_new_slots = 0
    print("  ── Challenge slots (reused from ./data/ cache vs to generate) ──")
    for cid in players_needed:
        for repo in repo_names:
            sp = slot_plans[cid][repo]
            n_new = len(sp["to_generate"])
            total_new_slots += n_new
            label = f"{_full_identity(cid):<{ident_w}} × {repo:<22}"
            if n_new:
                print(
                    f"   {label}: {len(sp['admitted'])} admitted / "
                    f"{len(sp['exhausted'])} exhausted (cached) — "
                    f"{n_new} to generate {sp['to_generate']}"
                )
                model_cfg = model_configs[cid]
                harness = split_composite_id(cid)[1] or "mini-swe-agent"
                gen_pools.append(
                    (cid, model_cfg, harness, ctx["repo_configs"][repo],
                     turns_per_player)
                )
            else:
                print(
                    f"   {label}: {len(sp['admitted'])} admitted / "
                    f"{len(sp['exhausted'])} exhausted (cached) — up to date"
                )
    print(
        f"\n  Totals: {len(pairings)} match(es) to run, "
        f"{total_new_slots} challenge slot(s) to generate "
        f"across {len(gen_pools)} pool(s)"
    )
    print()

    # ── Prompt 1: challenge generation ────────────────────────
    if gen_pools:
        if plan is not None and str(state_payload.get("status") or "") in (
            "generated",
            "complete",
        ):
            print(
                "[as] skipping challenge generation — the resumed state "
                "already generated its pools."
            )
        else:
            confirm = questionary.confirm(
                f"Continue with challenge generation "
                f"({total_new_slots} unattempted slot(s) over {len(gen_pools)} "
                f"pool(s); attempted slots are reused from the cache)?",
                default=True,
            ).ask()
            if not confirm:
                print(
                    "\n[as] declined challenge generation — nothing was run. "
                    f"Sampled plan persisted to {state_path}"
                )
                if plan is not None:
                    _offer_contribution_zip(
                        ctx["output_dir"],
                        state_path,
                        state_payload,
                        [],
                        Path(args.submission_dir),
                        config_dir=ctx["config_dir"],
                    )
                return 0
            _run_generation_pools(
                gen_pools,
                store=store,
                workspace_manager=ctx["workspace_manager"],
                arena_cfg=arena_cfg,
                prompt_dir=Path(args.prompt_dir),
                max_workers=args.max_workers or arena_cfg.challenge_bank.max_workers,
            )
            state_payload["status"] = "generated"
            _persist_as_state(state_path, state_payload)

        print("\n  ── Challenge-bank recap ──")
        for cid in players_needed:
            model, harness, effort, provider = split_composite_id(cid)
            for repo in repo_names:
                admitted = store.count_in_pool(
                    model, repo, harness or "mini-swe-agent", effort, provider
                )
                short = max(0, turns_per_player - admitted)
                note = (
                    f" — {short} auto-win(s) for the defender"
                    if short
                    else ""
                )
                print(
                    f"   {_full_identity(cid):<{ident_w}} × {repo:<22}"
                    f" {admitted}/{turns_per_player} admitted{note}"
                )
        print()
    else:
        print(
            "[as] every needed challenge slot is already attempted in the "
            "cache — skipping generation entirely."
        )

    # ── Prompt 2: match runs ──────────────────────────────────
    if plan is not None and str(state_payload.get("status") or "") == "complete":
        # The resumed state already ran its matches — nothing to re-run
        # (the defenses are cached anyway); go straight to the end-of-flow
        # contribution-zip prompt, the packaging the earlier session missed.
        n_run = 0
        total_cost = 0.0
        for r in state_payload.get("results") or []:
            if isinstance(r, dict) and r.get("match_id"):
                n_run += 1
                total_cost += float(r.get("cost_usd") or 0.0)
        print(
            f"\n[as] the resumed state already ran its matches "
            f"({n_run}/{len(pairings)} recorded, total cost "
            f"${total_cost:.4f}) — skipping the match runs."
        )
    else:
        confirm = questionary.confirm(
            f"Continue with running the {len(pairings)} sampled match(es) "
            f"across {len(repo_names)} repo(s)?",
            default=True,
        ).ask()
        if not confirm:
            print(
                "\n[as] declined match runs — challenges are cached. "
                f"Sampled plan persisted to {state_path}"
            )
            _offer_contribution_zip(
                ctx["output_dir"],
                state_path,
                state_payload,
                [],
                Path(args.submission_dir),
                config_dir=ctx["config_dir"],
            )
            return 0

        print(f"\n[as] running {len(pairings)} sampled match(es)...\n")
        executed = _run_matchups_parallel(
            orchestrator,
            pairings,
            repo_cfgs,
            max_workers=arena_cfg.match.max_workers,
        )

        results: list[dict[str, Any]] = []
        total_cost = 0.0
        n_run = 0
        for p, m in executed:
            a_id, b_id = _pair_ids(p)
            if m is None:
                results.append(
                    {"model_a": a_id, "model_b": b_id, "status": "failed"}
                )
                continue
            n_run += 1
            total_cost += m.total_cost_usd
            results.append(
                {
                    "model_a": a_id,
                    "model_b": b_id,
                    "match_id": m.match_id,
                    "outcome": m.outcome.value,
                    "score_a": m.model_a_total,
                    "score_b": m.model_b_total,
                    "cost_usd": m.total_cost_usd,
                }
            )
        state_payload["results"] = results
        state_payload["status"] = "complete"
        _persist_as_state(state_path, state_payload)

        print(
            f"\n[as] done: {n_run}/{len(pairings)} match(es) ran, "
            f"total cost ${total_cost:.4f}. "
            f"State → {state_path}\n"
        )
    _offer_contribution_zip(
        ctx["output_dir"],
        state_path,
        state_payload,
        [
            str(r["match_id"])
            for r in state_payload.get("results") or []
            if isinstance(r, dict) and r.get("match_id")
        ],
        Path(args.submission_dir),
        config_dir=ctx["config_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())