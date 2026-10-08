#!/usr/bin/env python
"""External-contribution import console (`swe-duel-tournament-update`).

An external contributor enters a tournament with
``swe-duel-tournament-as --tournament <id>`` on their own machine (their
newcomer is actively sampled against the field, generating challenges,
defenses and matches under their ``./data/``), then submits that ``./data/``
directory as a ``.zip``. This command imports such a submission into the
organizer's arena:

  1. **duplicate guard** — the zip's SHA-256 names the contribution; a hash
     already present under ``./contributions/`` is rejected, so the same zip
     can never be imported twice;
  2. **merge** (additive, idempotent: existing local records always win) —
     the selected active-sampling state files, the matches those runs
     scheduled (claimed by walking each state's pairings, plus every match id
     its ``results`` reference), and the challenges / failed attempts /
     defenses whose Red or Blue identity is one of the runs' field
     participants. ``data/challenge_bank/index.json`` is merged (pool lists
     appended, entries added) so the bank, the defense cache and future
     ``swe-duel-tournament-as`` runs see the contributed slots. Everything
     else in the zip (workspaces, logs, unrelated runs) is not even
     extracted;
  3. **rankings update** — ``./rankings/rankings_<id>.json`` is re-exported
     through :func:`swe_duel.cli.export_rankings.export_rankings`, which now
     folds the active-sampling matches in automatically: the newcomers join
     the field with standings, BT/Elo/TrueSkill are recomputed over the
     merged match history, and the state scoreboard gains the new win/draw
     points;
4. **leaderboard update** — the ``./docs/index.html`` page (the GitHub
      Pages site root) is refreshed
      for every participant from the merged history: matched rows keep
      their identity visuals (name / harness + org pills; no model-color
      dot) while their rank/Elo attrs + cells and the whole detail block
      (led by an identity strip carrying the reasoning-effort + provider
      pins, shown only when the row is expanded) are
      regenerated; newcomers are appended as ``late entrant`` rows
      (invisible ``data-late-entrant`` marker, no visible pill or legend
      chip) with
      match record / duel outcome / per-match cost & token cards
      computed from the contribution; the Elo axis gridlines and the
      meta-description entrant count follow the new field;
  5. **archive + history** — the imported files, the original zip and a
     manifest are stored under ``./contributions/<sha256>/`` for future
     analysis, and ``./contributions/contributions.json`` +
     ``./contributions/index.html`` visualize the overall contribution
     history of newly added models against the original rankings.

Offline command: no Docker, no LLM calls, no preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from swe_duel.challenge_bank.store import _pool_key
from swe_duel.cli._common import resolve_config_dir, resolve_output_dir
from swe_duel.cli.build_report import (
    _claim_matches_per_tournament,
    _dedupe,
    _load_matches,
    _load_round_robin_states,
    _load_swiss_states,
    _to_match_result,
)
from swe_duel.cli.export_rankings import (
    _as_entered_tournament,
    _load_as_extra_matches,
    export_rankings,
)
from swe_duel.config import load_arena_config
from swe_duel.models import MatchResult, composite_id, split_composite_id
from swe_duel.scoring.bootstrap import (
    StrengthDiagnostics as _StrengthDiag,
    bootstrap_field_diagnostics as _bootstrap_field_diagnostics,
)
from swe_duel.scoring.rating import compute_all_ratings

# Only these subtrees of a submission's data/ directory are ever read; the
# zip is not even extracted beyond them (a contributor's workspaces/ can be
# gigabytes and is worthless to the organizer).
_DATA_MARKERS = ("matches", "defenses", "challenge_bank", "tournaments")
_MAX_EXTRACT_BYTES = 8 * 1024**3  # safety valve against zip bombs

# Harness id → the label the root leaderboard page prints in its hpill
# (codex renders as "Codex CLI" there, not the CLI's "Codex").
_PAGE_HARNESS_LABELS = {
    "mini-swe-agent": "mini-swe-agent",
    "openhands": "OpenHands",
    "codex": "Codex CLI",
    "claude-code": "Claude Code",
}

# model-id prefix → organization label shown on the leaderboard page.
_ORG_BY_PREFIX = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "moonshotai": "Moonshot AI",
    "deepseek": "DeepSeek",
    "z-ai": "Z.ai",
    "qwen": "Alibaba",
    "nvidia": "NVIDIA",
    "minimax": "MiniMax",
    "meta": "Meta",
    "mistralai": "Mistral AI",
    "x-ai": "xAI",
    "cohere": "Cohere",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "ai21": "AI21 Labs",
    "perplexity": "Perplexity",
    "01-ai": "01.AI",
    "stepfun": "StepFun",
    "turing": "Turing",
}

# Name tokens always rendered uppercase on the leaderboard page ("glm" →
# "GLM"); everything else keeps its first letter capitalized.
_UPPER_NAME_TOKENS = {"glm", "gpt"}


# ── submission zip handling ─────────────────────────────────────


def _zip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate_data_prefix(names: list[str]) -> str:
    """ZIP-member prefix of the submission's ``data/`` directory.

    Contributors zip ``./data`` in different shapes — ``data/...`` at the
    archive root, a single wrapper directory around it, or the marker
    subdirectories directly at the root. The prefix with the most marker
    hits (ties: shortest) wins, so only that subtree is ever extracted.
    """
    votes: dict[str, int] = {}
    for name in names:
        parts = PurePosixPath(name).parts
        for i, part in enumerate(parts[:-1]):
            prefix: str | None = None
            if part == "data":
                prefix = "/".join(parts[: i + 1]) + "/"
            elif part in _DATA_MARKERS:
                prefix = "/".join(parts[:i]) + "/" if i else ""
            if prefix is not None:
                votes[prefix] = votes.get(prefix, 0) + 1
    if not votes:
        raise SystemExit(
            "submission zip carries none of the expected data subdirectories "
            f"({', '.join(_DATA_MARKERS)}) — expected the ./data output of a "
            "swe-duel-tournament-as run."
        )
    return sorted(votes, key=lambda p: (-votes[p], len(p)))[0]


def _extract_submission(zip_path: Path, dest: Path) -> tuple[Path, int]:
    """Extract ONLY the data subtree of the submission, safely.

    Returns ``(data_root, total_bytes)``. Every extracted member is checked
    against zip-slip (absolute paths / ``..`` traversal) and the aggregate
    uncompressed size against :data:`_MAX_EXTRACT_BYTES` before extraction.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        prefix = _locate_data_prefix(names)
        root = dest.resolve()
        # Zip-slip defense first: EVERY member under the data prefix must
        # resolve inside the extraction root, regardless of whether it is
        # one of the subtrees we would import.
        for info in zf.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            if root not in (dest / info.filename).resolve().parents:
                raise SystemExit(
                    f"submission zip member {info.filename!r} escapes the "
                    "extraction directory (zip-slip) — refusing to import."
                )
        members: list[str] = []
        total = 0
        for info in zf.infolist():
            if info.is_dir() or info.file_size > _MAX_EXTRACT_BYTES:
                continue
            name = info.filename
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if not rel or rel.startswith("/"):
                continue
            # Only the marker subtrees are ever extracted — a contributor's
            # workspaces/ or logs/ inside data/ stay in the zip.
            if PurePosixPath(rel).parts[0] not in _DATA_MARKERS:
                continue
            total += info.file_size
            members.append(name)
        if total > _MAX_EXTRACT_BYTES:
            raise SystemExit(
                f"submission zip's data subtree expands to {total:,} bytes — "
                "above the import cap; submit only the ./data of the "
                "swe-duel-tournament-as run."
            )
        root = dest.resolve()
        for name in sorted(members):
            target = (dest / name).resolve()
            if root not in target.parents:
                raise SystemExit(
                    f"submission zip member {name!r} escapes the extraction "
                    "directory (zip-slip) — refusing to import."
                )
            zf.extract(name, dest)
        return dest / prefix, total


# ── active-sampling states inside a submission ──────────────────


def _load_as_states(tournaments_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Parse every active-sampling state under a tournaments directory."""
    states: list[tuple[Path, dict[str, Any]]] = []
    if not tournaments_dir.is_dir():
        return states
    for path in sorted(tournaments_dir.glob("active_sampling_state_*.json")):
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"[update] warning: skipping unparseable {path}", file=sys.stderr)
            continue
        if isinstance(state, dict) and state.get("format") == "active_sampling":
            states.append((path, state))
    return states


def _as_field(state: dict[str, Any]) -> list[str]:
    """The field identities an AS run covered (its own record, or derived)."""
    raw = state.get("field")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw if x]
    out: list[str] = []
    for pairing in state.get("pairings") or []:
        if isinstance(pairing, dict):
            for side in ("model_a", "model_b"):
                if pairing.get(side):
                    out.append(str(pairing[side]))
    for result in state.get("results") or []:
        if isinstance(result, dict):
            for side in ("model_a", "model_b"):
                if result.get(side):
                    out.append(str(result[side]))
    return list(dict.fromkeys(out))


def _as_newcomers(state: dict[str, Any]) -> list[str]:
    """The new entrants an AS run brought (recorded, or the pairing hosts)."""
    raw = state.get("newcomers") or state.get("targets")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw if x]
    # Cross-only intake always seats the newcomer on side A; that is the only
    # shape swe-duel-tournament-as emits.
    out = [
        str(p["model_a"])
        for p in state.get("pairings") or []
        if isinstance(p, dict) and p.get("model_a")
    ]
    return list(dict.fromkeys(out))


def _resolve_tournament(
    as_states: list[tuple[Path, dict[str, Any]]], requested: str | None
) -> tuple[str, list[tuple[Path, dict[str, Any]]]]:
    """Pick the tournament this import updates + the AS states in scope.

    Without ``--tournament`` the submission must reference exactly one
    tournament; several distinct ids fail fast listing them (the operator
    re-runs with ``--tournament <id>`` to pick one). With ``--tournament``,
    the states entering that tournament are imported and states for other
    tournaments are skipped with a notice (one import per tournament).
    """
    by_tid: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, state in as_states:
        by_tid.setdefault(_as_entered_tournament(state), []).append((path, state))
    if not by_tid:
        raise SystemExit(
            "submission carries no active_sampling_state_*.json — only the "
            "./data output of a swe-duel-tournament-as run can be imported."
        )
    if requested:
        if requested not in by_tid:
            raise SystemExit(
                f"--tournament {requested!r} is not entered by any "
                "active-sampling state in the submission; detected: "
                + ", ".join(sorted(tid or "(unlabeled)" for tid in by_tid))
            )
        for other in sorted(tid for tid in by_tid if tid and tid != requested):
            print(
                f"[update] skipping {len(by_tid[other])} AS state(s) entering "
                f"tournament {other!r} — imports are one tournament per zip."
            )
        return requested, by_tid[requested]
    if len(by_tid) == 1:
        tid, states = next(iter(by_tid.items()))
        if not tid:
            raise SystemExit(
                "the submission's active-sampling states carry no "
                "entered_tournament / rankings_source — pass --tournament <id>."
            )
        return tid, states
    raise SystemExit(
        "submission mixes active-sampling runs for several tournaments; "
        "pass --tournament <id> to pick one. Detected: "
        + ", ".join(sorted(by_tid))
    )


# ── identity scoping ────────────────────────────────────────────


@dataclass(frozen=True)
class _Identity:
    """A participant 4-tuple identity."""

    model: str
    harness: str
    effort: str
    provider: str

    @classmethod
    def from_cid(cls, cid: str) -> _Identity:
        model, harness, effort, provider = split_composite_id(cid)
        return cls(model, harness or "mini-swe-agent", effort, provider)


def _record_identity(payload: dict[str, Any], prefix: str) -> _Identity | None:
    """Identity columns of a challenge/defense record (``red_``/``blue_``)."""
    model = payload.get(f"{prefix}_model_id")
    if not isinstance(model, str) or not model:
        return None
    harness = payload.get(f"{prefix}_harness_id", "mini-swe-agent")
    effort = payload.get(f"{prefix}_reasoning_effort", "")
    provider = payload.get(f"{prefix}_provider", "")
    return _Identity(
        model,
        harness if isinstance(harness, str) and harness else "mini-swe-agent",
        effort if isinstance(effort, str) else "",
        provider if isinstance(provider, str) else "",
    )


# ── merge ───────────────────────────────────────────────────────


@dataclass
class _MergeCounts:
    matches: int = 0
    defenses: int = 0
    challenges: int = 0
    failed_challenges: int = 0
    as_states: int = 0
    skipped_existing: int = 0
    index_entries_added: int = 0


def _copy_new(src: Path, dst: Path, counts: _MergeCounts, copied: list[Path]) -> bool:
    """Copy ``src`` → ``dst`` when missing; existing local files always win."""
    if dst.exists():
        counts.skipped_existing += 1
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copied.append(src)
    return True


def _load_scope_payloads(
    sub_data: Path, field: set[_Identity]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The submission's field-scoped challenge and defense payloads."""
    challenges: list[dict[str, Any]] = []
    for rel_dir in ("challenges", "failed_challenges"):
        src_dir = sub_data / "challenge_bank" / rel_dir
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.glob("*.json")):
            try:
                payload = json.loads(src.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            identity = _record_identity(payload, "red")
            if identity is not None and identity in field:
                challenges.append(payload)
    defenses: list[dict[str, Any]] = []
    def_dir = sub_data / "defenses"
    if def_dir.is_dir():
        for src in sorted(def_dir.glob("*.json")):
            try:
                payload = json.loads(src.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            identity = _record_identity(payload, "blue")
            if identity is not None and identity in field:
                defenses.append(payload)
    return challenges, defenses


def _merge_index(
    local_index_path: Path,
    additions: list[dict[str, Any]],
    counts: _MergeCounts,
) -> None:
    """Fold imported challenge records into ``challenge_bank/index.json``.

    ``additions`` carries per-record merge instructions (pool key, entry
    dict, failed flag). Pool lists are appended (local order first, so
    ``plan_match``'s first-N selection over local pools is unchanged) and
    ``entries`` are only added when absent. A corrupt local index is treated
    exactly like :class:`ChallengeStore` treats it (empty) and rewritten
    with what the import contributes. The index is rewritten via the store's
    temp-then-rename pattern.
    """
    if not additions:
        return
    if local_index_path.exists():
        try:
            index: dict[str, Any] = json.loads(local_index_path.read_text())
        except json.JSONDecodeError:
            print(
                f"[update] warning: {local_index_path} is corrupt — rewriting "
                "it with only the imported entries (the store treats a "
                "corrupt index as empty too)."
            )
            index = {}
        if not isinstance(index, dict):
            index = {}
    else:
        index = {}
    pools = index.get("pools")
    failed_pools = index.get("failed_pools")
    entries = index.get("entries")
    if not isinstance(pools, dict) or not isinstance(failed_pools, dict) or not isinstance(entries, dict):
        raise SystemExit(f"{local_index_path} has an unexpected shape — aborting.")
    for add in additions:
        cid = str(add["entry"]["id"])
        if add["failed"]:
            bucket = failed_pools.setdefault(add["key"], [])
            if not any(isinstance(e, dict) and e.get("id") == cid for e in bucket):
                bucket.append(add["entry"])
                counts.index_entries_added += 1
        else:
            bucket = pools.setdefault(add["key"], [])
            if isinstance(bucket, list) and cid not in bucket:
                bucket.append(cid)
        if cid not in entries:
            entries[cid] = add["entry"]
            counts.index_entries_added += 1
    tmp = local_index_path.with_name(f".{local_index_path.name}.tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True, default=str))
    os.replace(tmp, local_index_path)


def _challenge_index_addition(
    payload: dict[str, Any], *, failed: bool
) -> dict[str, Any] | None:
    """Merge instruction for one challenge / failed-challenge record."""
    model = payload.get("red_model_id")
    repo = payload.get("repo_name")
    if not isinstance(model, str) or not model or not isinstance(repo, str) or not repo:
        return None
    identity = _record_identity(payload, "red") or _Identity(
        model, "mini-swe-agent", "", ""
    )
    retries = int(payload.get("generation_retries", 0) or 0)
    cid = str(payload.get("challenge_id") or "")
    if not cid:
        return None
    key = _pool_key(
        identity.model, repo, identity.harness, identity.effort, identity.provider
    )
    return {
        "key": key,
        "failed": failed,
        "entry": {
            "id": cid,
            "status": "failed" if failed else "success",
            "kind": str(payload.get("kind") or ("success" if not failed else "failed")),
            "attempt": int(payload.get("attempt_number", retries + 1) or retries + 1),
            "slot": int(payload.get("slot", 1) or 1),
            "red_model_id": identity.model,
            "red_harness_id": identity.harness,
            "repo_name": repo,
            "red_reasoning_effort": identity.effort,
            "red_provider": identity.provider,
            "previous_attempts": list(payload.get("previous_attempts") or []),
        },
    }


def _merge_submission(
    sub_data: Path,
    local_data: Path,
    selected_states: list[tuple[Path, dict[str, Any]]],
    tournament_id: str,
) -> tuple[_MergeCounts, list[Path], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge the in-scope submission records into the local ``./data/``.

    Returns ``(counts, copied_sources, as_match_payloads, challenge_payloads,
    defense_payloads)`` — copied sources are for archiving; the payload
    lists feed the newcomer leaderboard stats (challenges/defenses are the
    submission's field-scoped records, whether newly copied or already
    present locally).
    """
    counts = _MergeCounts()
    copied: list[Path] = []
    field = {
        _Identity.from_cid(cid)
        for _path, state in selected_states
        for cid in _as_field(state)
    }
    if not field:
        raise SystemExit(
            "the selected active-sampling states record no field identities "
            "— nothing to scope the import to."
        )

    # ── AS state files (only the ones entering this tournament) ──
    for src, _state in selected_states:
        if _copy_new(src, local_data / "tournaments" / src.name, counts, copied):
            counts.as_states += 1

    # ── matches scheduled by the AS runs ─────────────────────────
    as_matches = _load_as_extra_matches(
        sub_data / "tournaments", sub_data / "matches", tournament_id
    )
    for match in sorted(as_matches, key=lambda m: str(m.get("match_id") or "")):
        mid = str(match.get("match_id") or "")
        src = sub_data / "matches" / f"{mid}.json"
        if src.is_file() and _copy_new(
            src, local_data / "matches" / f"{mid}.json", counts, copied
        ):
            counts.matches += 1

    # ── challenges / failed attempts by field identities ─────────
    index_additions: list[dict[str, Any]] = []
    for rel_dir, failed, counter in (
        ("challenges", False, "challenges"),
        ("failed_challenges", True, "failed_challenges"),
    ):
        src_dir = sub_data / "challenge_bank" / rel_dir
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.glob("*.json")):
            try:
                payload = json.loads(src.read_text())
            except (json.JSONDecodeError, OSError):
                print(
                    f"[update] warning: skipping unparseable {src}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(payload, dict):
                continue
            identity = _record_identity(payload, "red")
            if identity is None or identity not in field:
                continue  # not one of this run's participants
            if _copy_new(
                src,
                local_data / "challenge_bank" / rel_dir / src.name,
                counts,
                copied,
            ):
                setattr(counts, counter, getattr(counts, counter) + 1)
            html_src = src.with_suffix(".html")
            if html_src.is_file():
                _copy_new(
                    html_src,
                    local_data / "challenge_bank" / rel_dir / html_src.name,
                    counts,
                    copied,
                )
            addition = _challenge_index_addition(payload, failed=failed)
            if addition is not None:
                index_additions.append(addition)

    # ── defenses by field identities ──────────────────────────────
    def_dir = sub_data / "defenses"
    if def_dir.is_dir():
        for src in sorted(def_dir.glob("*.json")):
            try:
                payload = json.loads(src.read_text())
            except (json.JSONDecodeError, OSError):
                print(
                    f"[update] warning: skipping unparseable {src}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(payload, dict):
                continue
            identity = _record_identity(payload, "blue")
            if identity is None or identity not in field:
                continue
            if _copy_new(src, local_data / "defenses" / src.name, counts, copied):
                counts.defenses += 1
            html_src = src.with_suffix(".html")
            if html_src.is_file():
                _copy_new(
                    html_src, local_data / "defenses" / html_src.name, counts, copied
                )

    _merge_index(
        local_data / "challenge_bank" / "index.json", index_additions, counts
    )
    challenges, defenses = _load_scope_payloads(sub_data, field)
    return counts, copied, as_matches, challenges, defenses


# ── newcomer stats (leaderboard rows + cards) ───────────────────


@dataclass
class _NewcomerStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gen_cost: float = 0.0
    val_cost: float = 0.0
    eval_cost: float = 0.0
    out_tokens: int = 0
    match_out_tokens: int = 0
    defense_costs: list[float] = field(default_factory=list)
    defense_out_tokens: list[int] = field(default_factory=list)
    attacks_landed: int = 0
    defenses_broken: int = 0
    turns_defended: int = 0

    @property
    def defense_success(self) -> float | None:
        if not self.turns_defended:
            return None
        return 1.0 - self.defenses_broken / self.turns_defended

    @property
    def record(self) -> str:
        return f"{self.wins}W · {self.draws}D · {self.losses}L"


def _trajectory_tokens(traj: Any) -> tuple[int, int, float]:
    if not isinstance(traj, dict):
        return 0, 0, 0.0
    return (
        int(traj.get("total_input_tokens") or 0),
        int(traj.get("total_output_tokens") or 0),
        float(traj.get("total_cost_usd") or 0.0),
    )


def _newcomer_stats(
    cid: str,
    as_matches: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
    defenses: list[dict[str, Any]],
) -> _NewcomerStats:
    """Compute a newcomer's contribution-scoped stats.

    Challenges/defenses are the imported payloads whose Red/Blue identity is
    the newcomer; matches are the imported AS matches it played. Turn-level
    outcome counters consider only contested turns (an empty challenge_id is
    a missing-Red auto-win — not a real duel).
    """
    identity = _Identity.from_cid(cid)
    stats = _NewcomerStats()
    for challenge in challenges:
        if _record_identity(challenge, "red") != identity:
            continue
        stats.gen_cost += float(challenge.get("generation_cost_usd") or 0.0)
        ch = challenge.get("challenge") or {}
        ft = ch.get("feature_trajectory")
        bt = ch.get("bug_trajectory")
        if isinstance(ft, dict) or isinstance(bt, dict):
            for traj in (ft, bt):
                stats.out_tokens += _trajectory_tokens(traj)[1]
        else:
            stats.out_tokens += _trajectory_tokens(ch.get("agent_trajectory"))[1]
        validation = challenge.get("validation") or {}
        sr = (validation.get("self_review") or {}).get("agent_trajectory")
        _in, out, cost = _trajectory_tokens(sr)
        stats.val_cost += cost
        stats.out_tokens += out
    for defense in defenses:
        if _record_identity(defense, "blue") != identity:
            continue
        cost = float(defense.get("cost_usd") or 0.0)
        stats.eval_cost += cost
        out = _trajectory_tokens((defense.get("blue_fix") or {}).get("agent_trajectory"))[1]
        stats.out_tokens += out
        stats.match_out_tokens += out
        stats.defense_costs.append(cost)
        stats.defense_out_tokens.append(out)
    for match in as_matches:
        a = str(match.get("model_a_id") or "")
        b = str(match.get("model_b_id") or "")
        if cid not in (a, b):
            continue
        outcome = str(match.get("outcome") or "draw")
        if outcome == "draw":
            stats.draws += 1
        elif (outcome == "model_a_wins" and a == cid) or (
            outcome == "model_b_wins" and b == cid
        ):
            stats.wins += 1
        else:
            stats.losses += 1
        for turn in match.get("turns") or []:
            if not turn.get("challenge_id"):
                continue  # auto-win turn — not a real duel
            red = str(turn.get("red_model_id") or "")
            blue = str(turn.get("blue_model_id") or "")
            red_won = float((turn.get("score") or {}).get("red_composite") or 0.0) > 0.5
            if red == cid and red_won:
                stats.attacks_landed += 1
            if blue == cid:
                stats.turns_defended += 1
                if red_won:
                    stats.defenses_broken += 1
    return stats


# ── strength diagnostics rendering (late-entrant cards) ─────────


# Accent colors for the median bar of the rank histogram (the page's palette).
_MED_PALETTE = (
    "#59c886",
    "#fe9b61",
    "#ff787d",
    "#ae8bff",
    "#3bcfcf",
    "#00c5ee",
    "#5fadff",
    "#a9aebb",
)


def _fmt_bt(value: float) -> str:
    """Signed Bradley--Terry strength in the page's style (true minus)."""
    return f"{'\u2212' if value < 0 else '+'}{abs(value):.2f}"


def _strength_card_html(diag: _StrengthDiag, accent: str) -> tuple[str, str]:
    """The ``Strength diagnostics`` dcard and the ``Bootstrap rank
    distribution`` dhist, in exactly the incumbent rows' markup shape so
    the appended late-entrant row is indistinguishable from the original
    field's cards."""
    card = (
        '<section class="dcard"><h4>Strength diagnostics</h4>'
        f'<div class="dhead">#{diag.median_rank}'
        '<span class="dsub">median bootstrap rank</span></div>'
        '<dl><div class="kv"><dt>Median rank</dt>'
        f"<dd><b>{diag.median_rank}</b></dd></div>"
        f'<div class="kv"><dt>95% rank interval</dt>'
        f"<dd>{diag.rank_lo}\u2013{diag.rank_hi}</dd></div>"
        f'<div class="kv"><dt>Match-level BT</dt>'
        f"<dd><b>{_fmt_bt(diag.bt)}</b></dd>"
        f'<span class="ci">[{_fmt_bt(diag.bt_lo)}, {_fmt_bt(diag.bt_hi)}]</span></div>'
        f'<div class="kv"><dt>Contested BT</dt><dd>{_fmt_bt(diag.contested)}</dd>'
        f'<span class="ci">[{_fmt_bt(diag.contested_lo)}, '
        f"{_fmt_bt(diag.contested_hi)}]</span></div></dl></section>"
    )
    max_share = max(diag.rank_hist) if diag.rank_hist else 0.0
    bars = []
    for rank_i, share in enumerate(diag.rank_hist, 1):
        inside = diag.rank_lo <= rank_i <= diag.rank_hi
        classes = "hb" + (" med" if rank_i == diag.median_rank else "")
        classes += " in" if inside else " out"
        pct = share * 100.0
        if share > 0 and max_share > 0:
            height = max(share / max_share * 100.0, 3.0)
        else:
            height = 1.2
        hp = f"{round(pct)}%" if round(pct) >= 1 else ""
        hc = f";--hc:{accent}" if rank_i == diag.median_rank else ""
        bars.append(
            f'<div class="{classes}" title="rank {rank_i}: {pct:.1f}% of resamples">'
            f'<span class="hp">{hp}</span>'
            f'<i style="height:{height:.2f}%{hc}"></i></div>'
        )
    labs = "".join(f"<span>{i}</span>" for i in range(1, diag.n_participants + 1))
    caption = (
        f"Share of {diag.ranked:,} of {diag.draws:,} match-level resamples "
        "in which this entrant finished at each rank"
        + (
            f" ({diag.draws - diag.ranked} resample(s) contained none of "
            "its matches)"
            if diag.ranked < diag.draws
            else ""
        )
        + f". Median <b>{diag.median_rank}</b>; 95% interval "
        f"<b>{diag.rank_lo}\u2013{diag.rank_hi}</b> (ranks outside it are dimmed)."
    )
    hist = (
        '<div class="dhist"><div class="dhist-cap">'
        "<h4>Bootstrap rank distribution</h4>"
        f"<p>{caption}</p></div>"
        f'<div class="dhist-plot"><div class="hbars">{"".join(bars)}</div>'
        f'<div class="hlabs">{labs}</div><div class="hax">rank</div></div></div>'
    )
    return card, hist


def _diag_from_entry(entry: dict[str, Any]) -> _StrengthDiag | None:
    """Rebuild a :class:`StrengthDiagnostics` from a rankings export entry.

    ``swe-duel-rankings`` writes the seeded bootstrap results into every
    entry (``elo_ci`` / ``bradley_terry_ci`` / ``contested_bt`` /
    ``contested_bt_ci`` / ``bootstrap_*``), so the page and the JSON show
    byte-identical numbers for the same tournament state. Entries without
    the fields (a legacy export) return ``None`` so the caller can fall
    back to computing the bootstrap locally.
    """
    elo_ci = entry.get("elo_ci")
    bt_ci = entry.get("bradley_terry_ci")
    hist = entry.get("bootstrap_rank_histogram")
    if (
        not isinstance(elo_ci, list)
        or len(elo_ci) != 2
        or not isinstance(bt_ci, list)
        or len(bt_ci) != 2
        or not isinstance(hist, list)
        or not hist
    ):
        return None
    interval = entry.get("bootstrap_rank_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        return None
    contested_ci = entry.get("contested_bt_ci")
    if not isinstance(contested_ci, list) or len(contested_ci) != 2:
        return None
    return _StrengthDiag(
        draws=int(entry.get("bootstrap_draws") or 0),
        ranked=int(entry.get("bootstrap_ranked") or 0),
        n_participants=len(hist),
        rank_hist=[float(s) for s in hist],
        median_rank=int(entry.get("bootstrap_median_rank") or 0),
        rank_lo=int(interval[0]),
        rank_hi=int(interval[1]),
        bt=float(entry.get("bradley_terry") or 0.0),
        bt_lo=float(bt_ci[0]),
        bt_hi=float(bt_ci[1]),
        contested=float(entry.get("contested_bt") or 0.0),
        contested_lo=float(contested_ci[0]),
        contested_hi=float(contested_ci[1]),
        elo=float(entry.get("elo") or 1500.0),
        elo_lo=float(elo_ci[0]),
        elo_hi=float(elo_ci[1]),
    )


# ── leaderboard page update ─────────────────────────────────────


_ROW_BLOCK_RE = re.compile(r'<details class="rwrap".*?</details>', re.DOTALL)
_ROW_ATTRS_RE = re.compile(r'<details class="rwrap"([^>]*)>', re.DOTALL)
_TITLE_RE = re.compile(r'<div class="cell entrant" title="([^"]*)"')
_HPILL_RE = re.compile(r'<span class="hpill">(.*?)</span>', re.DOTALL)
_RANK_CELL_RE = re.compile(r'(<div class="cell cell-rank">)([^<]*)(</div>)')
_BAR_RE = re.compile(r'(<div class="bar pos" style="left:[\d.]+%;width:)[\d.]+(%"></div>)')
_BTV_RE = re.compile(r'(<span class="btv">)([\d.]+)(</span>)')
_GRIDLINES_RE = re.compile(
    r'(<div class="gridlines">).*?(</div>\s*<div class="hrow")', re.DOTALL
)


def _ceil50(value: float) -> int:
    return int(math.ceil(value / 50.0)) * 50


def _page_axis(elos: list[float]) -> tuple[float, float]:
    """Leaderboard Elo axis: ``[min elo, ceil50(max)+50]`` (one gridline step
    of headroom above the strongest entrant), reproducing the page's scale."""
    lo = min(elos)
    hi = float(_ceil50(max(elos)) + 50)
    return lo, (hi if hi > lo else lo + 50.0)


def _bar_width(elo: float, lo: float, hi: float) -> float:
    """Elo bar width in % — the page's mapping, clamped at a 3% minimum."""
    if hi <= lo:
        return 3.0
    return max(3.0, (elo - lo) / (hi - lo) * 100.0)


def _gridlines_html(lo: float, hi: float) -> str:
    """The axis gridline divs (multiples of 50 inside the axis, edge-safe)."""
    cells = []
    tick = (int(lo) // 50 + 1) * 50
    while tick < hi:
        pos = (tick - lo) / (hi - lo) * 100.0 if hi > lo else 0.0
        if 1.0 <= pos <= 99.0:
            cells.append(
                f'<div class="gl-line" style="left:{pos:.3f}%"></div>'
                f'<div class="gl-label" style="left:{pos:.3f}%">{tick}</div>'
            )
        tick += 50
    return "".join(cells)


def _display_name(model_id: str) -> str:
    """Human label for a model id, mirroring the page's naming style."""
    segment = model_id.rsplit("/", 1)[-1]
    tokens = []
    for token in segment.split("-"):
        if token in _UPPER_NAME_TOKENS:
            tokens.append(token.upper())
        elif token and token[0].isalpha():
            tokens.append(token[0].upper() + token[1:])
        else:
            tokens.append(token)
    return "-".join(tokens)


def _org_of(model_id: str) -> str:
    prefix = model_id.split("/", 1)[0]
    if prefix in _ORG_BY_PREFIX:
        return _ORG_BY_PREFIX[prefix]
    if not prefix:
        return model_id
    return prefix[:1].upper() + prefix[1:]


_AUTO_ROUTE = "(OpenRouter auto-route)"
_FDOT_RE = re.compile(r'<span class="fdot"[^>]*></span>')
_DIDENT_RE = re.compile(r'<div class="dident">.*?</div>', re.DOTALL)
_HISTORY_CHIP_RE = re.compile(
    r'\s*<span class="chip"><a href="[^"]*contributions/index\.html">.*?</a></span>',
    re.DOTALL,
)
# Styles for the expanded row's identity strip; injected into pages whose
# stylesheet predates it so a refreshed page always renders the strip.
_DIDENT_CSS = (
    "  .dident{display:flex;flex-wrap:wrap;gap:8px 28px;margin-bottom:12px;"
    "background:var(--card);border:1px solid var(--border3);padding:10px 16px;"
    "font-family:var(--mono);font-size:12px;line-height:1.4}\n"
    "  .dident .ik{color:var(--text-faint);font-size:10px;letter-spacing:.12em;"
    "text-transform:uppercase;margin-right:8px}\n"
    "  .dident .iv{color:var(--text-bright)}\n"
)


def _identity_html(entry: dict[str, Any]) -> str:
    """The expanded row's identity strip: model id, harness, reasoning
    effort and OpenRouter provider (the full competitor 4-tuple). Shown
    only inside the detail block — the collapsed summary stays compact."""
    harness = str(entry.get("harness") or "")
    effort = str(entry.get("reasoning_effort") or "") or "(default)"
    provider = str(entry.get("provider") or "")
    if not provider or provider == _AUTO_ROUTE:
        provider = "(auto-route)"
    items = [
        ("Model", str(entry.get("model") or "")),
        ("Harness", _PAGE_HARNESS_LABELS.get(harness, harness)),
        ("Reasoning effort", effort),
        ("Provider", provider),
    ]
    esc = html.escape
    return (
        '<div class="dident">'
        + "".join(
            f'<span><span class="ik">{esc(k)}</span><span class="iv">{esc(v)}</span></span>'
            for k, v in items
        )
        + "</div>"
    )


def _ensure_dident_css(text: str) -> str:
    if ".dident{" in text or "</style>" not in text:
        return text
    return text.replace("</style>", _DIDENT_CSS + "</style>", 1)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _pm_cost_tokens(entry: dict[str, Any], stats: _NewcomerStats) -> tuple[float, int]:
    """Per-match averages: total spend / output tokens divided by the
    entry's ``matches_played`` (0 when the participant has no matches —
    the page's cost/token cell semantics)."""
    n = int(entry.get("matches_played") or 0)
    total_cost = stats.gen_cost + stats.val_cost + stats.eval_cost
    pm_cost = total_cost / n if n > 0 else 0.0
    pm_tokens = int(round(stats.out_tokens / n)) if n > 0 else 0
    return pm_cost, pm_tokens


def _cost_tok_cells(
    entry: dict[str, Any],
    stats: _NewcomerStats,
    max_cost: float,
    max_tokens: int,
) -> tuple[str, str]:
    """The row's cost/token cells — per-match averages with bars scaled to
    the field's per-match maxima; the tooltips carry the cumulative
    breakdown the averages derive from."""
    n = int(entry.get("matches_played") or 0)
    pm_cost, pm_tokens = _pm_cost_tokens(entry, stats)
    if n > 0:
        cost_title = (
            f"{_fmt_money(pm_cost)} per match over {n} matches \u00b7 "
            f"generation ${stats.gen_cost:.2f} \u00b7 validation ${stats.val_cost:.2f} \u00b7 "
            f"match play ${stats.eval_cost:.2f}"
        )
        tok_title = (
            f"{pm_tokens:,} output tokens per match "
            f"({stats.out_tokens:,} total over {n} matches)"
        )
    else:
        cost_title = (
            f"generation ${stats.gen_cost:.2f} \u00b7 validation ${stats.val_cost:.2f} \u00b7 "
            f"match play ${stats.eval_cost:.2f} \u2014 no matches played"
        )
        tok_title = f"{stats.out_tokens:,} output tokens \u2014 no matches played"
    cost_bar = f"{pm_cost / max_cost * 100:.1f}" if max_cost > 0 else "0.0"
    tok_bar = f"{pm_tokens / max_tokens * 100:.1f}" if max_tokens > 0 else "0.0"
    cost_cell = (
        f'<div class="cell cnum cell-cost" title="{cost_title}">'
        f'<span class="cv">{_fmt_money(pm_cost)}</span>'
        f'<span class="cbar"><i style="width:{cost_bar}%"></i></span></div>'
    )
    tok_cell = (
        f'<div class="cell cnum cell-tok" title="{tok_title}">'
        f'<span class="cv">{_fmt_tokens(pm_tokens)}</span>'
        f'<span class="cbar"><i style="width:{tok_bar}%"></i></span></div>'
    )
    return cost_cell, tok_cell


def _row_model_harness(title: str, hpill: str) -> tuple[str, str] | None:
    """(model, harness) for a leaderboard row, via its page labels."""
    model = html.unescape(title).split(" · ")[0].strip()
    for harness, label in _PAGE_HARNESS_LABELS.items():
        if label == hpill.strip():
            return (model, harness)
    return None


def _parse_row(block: str) -> dict[str, Any] | None:
    """Extract a leaderboard row's identity anchors and current columns.

    The cost/tokens parsed here are the row's PER-MATCH averages (the
    page's cell semantics); a legacy row still carrying the cumulative
    ``data-total_usd``/``data-out_tokens`` attrs is converted via its own
    "Matches played" count so every figure feeds one bar scale."""
    attrs_match = _ROW_ATTRS_RE.match(block)
    title_match = _TITLE_RE.search(block)
    hpill_match = _HPILL_RE.search(block)
    if attrs_match is None or title_match is None or hpill_match is None:
        return None
    attrs = attrs_match.group(1)
    data_rank = re.search(r'data-rank="([^"]*)"', attrs)
    data_elo = re.search(r'data-elo="([^"]*)"', attrs)
    data_cost = re.search(r'data-usd_per_match="([^"]*)"', attrs)
    data_tok = re.search(r'data-tokens_per_match="([^"]*)"', attrs)
    cost = float(data_cost.group(1)) if data_cost else 0.0
    tokens = int(data_tok.group(1)) if data_tok else 0
    if data_cost is None or data_tok is None:
        # Pre-per-match row: divide its cumulative attrs by the matches
        # it actually played (parsed from its own detail card).
        legacy_cost = re.search(r'data-total_usd="([^"]*)"', attrs)
        legacy_tok = re.search(r'data-out_tokens="([^"]*)"', attrs)
        n_match = re.search(r"Matches played</dt><dd>(\d+)</dd>", block)
        n = int(n_match.group(1)) if n_match else 0
        if n > 0:
            cost = float(legacy_cost.group(1)) / n if legacy_cost else 0.0
            tokens = int(round(int(legacy_tok.group(1)) / n)) if legacy_tok else 0
    return {
        "model_harness": _row_model_harness(
            title_match.group(1), hpill_match.group(1)
        ),
        "old_rank": data_rank.group(1) if data_rank else None,
        "old_elo": float(data_elo.group(1)) if data_elo else None,
        "cost": cost,
        "tokens": tokens,
        "has_bar": _BAR_RE.search(block) is not None,
        "has_btv": _BTV_RE.search(block) is not None,
        "has_rank_cell": _RANK_CELL_RE.search(block) is not None,
    }


def _update_row_block(
    block: str,
    rank: int,
    elo: float,
    lo: float,
    hi: float,
    entry: dict[str, Any] | None = None,
) -> str:
    """Rewrite a row's core ranking columns (rank, Elo bar + value).

    Also drops any legacy model-color dot from the summary and, given the
    rankings ``entry``, (re)writes the detail block's identity strip."""
    block = _FDOT_RE.sub("", block)
    if entry is not None:
        ident = _identity_html(entry)
        if _DIDENT_RE.search(block):
            block = _DIDENT_RE.sub(lambda _m: ident, block, count=1)
        else:
            block = block.replace('<div class="detail">', '<div class="detail">' + ident, 1)
    block = re.sub(
        r'(<details class="rwrap"[^>]*?)data-rank="[^"]*"',
        rf'\g<1>data-rank="{rank}"',
        block,
        count=1,
    )
    block = re.sub(
        r'(<details class="rwrap"[^>]*?)data-elo="[^"]*"',
        rf'\g<1>data-elo="{elo:.1f}"',
        block,
        count=1,
    )
    block = _RANK_CELL_RE.sub(rf"\g<1>{rank}\g<3>", block, count=1)
    block = _BAR_RE.sub(rf"\g<1>{_bar_width(elo, lo, hi):.3f}\g<2>", block, count=1)
    block = _BTV_RE.sub(rf"\g<1>{elo:.1f}\g<3>", block, count=1)
    return block


def _refresh_row_block(
    block: str,
    rank: int,
    entry: dict[str, Any],
    stats: _NewcomerStats,
    snapshot: Any,
    diag: _StrengthDiag | None,
    lo: float,
    hi: float,
    max_cost: float,
    max_tokens: int,
) -> str:
    """Rewrite an existing participant row in place with merged-history stats.

    The summary keeps its identity visuals (name, harness/org pills; any
    legacy model-color dot is dropped) and has every statistic refreshed — the rank/Elo attrs,
    rank cell, Elo bar + value, the 95% Elo CI badge (inserted when the row
    predates it), and the per-match cost/token cells — while the whole
    detail block is regenerated from the same renderer the appended entrant
    rows use, so incumbent and entrant cards stay indistinguishable and
    consistent with the re-exported rankings JSON."""
    elo = float(entry["elo"])
    pm_cost, pm_tokens = _pm_cost_tokens(entry, stats)
    cost_cell, tok_cell = _cost_tok_cells(entry, stats, max_cost, max_tokens)

    split = block.index("</summary>") + len("</summary>")
    summary = _FDOT_RE.sub("", block[:split])
    summary = re.sub(
        r'(<details class="rwrap"[^>]*?)data-rank="[^"]*"',
        rf'\g<1>data-rank="{rank}"',
        summary,
        count=1,
    )
    summary = re.sub(
        r'(<details class="rwrap"[^>]*?)data-elo="[^"]*"',
        rf'\g<1>data-elo="{elo:.1f}"',
        summary,
        count=1,
    )
    # Per-match attrs: the current names are rewritten in place; a legacy
    # row (cumulative data-total_usd/data-out_tokens page) is migrated by
    # the rename — either way the row ends up carrying the per-match pair.
    summary = re.sub(
        r'(<details class="rwrap"[^>]*?)data-(?:total_usd|usd_per_match)="[^"]*"',
        rf'\g<1>data-usd_per_match="{pm_cost:.2f}"',
        summary,
        count=1,
    )
    summary = re.sub(
        r'(<details class="rwrap"[^>]*?)data-(?:out_tokens|tokens_per_match)="[^"]*"',
        rf'\g<1>data-tokens_per_match="{pm_tokens}"',
        summary,
        count=1,
    )
    summary = _RANK_CELL_RE.sub(rf"\g<1>{rank}\g<3>", summary, count=1)
    summary = _BAR_RE.sub(rf"\g<1>{_bar_width(elo, lo, hi):.3f}\g<2>", summary, count=1)
    summary = _BTV_RE.sub(rf"\g<1>{elo:.1f}\g<3>", summary, count=1)
    if diag is not None:
        btci = f"[{diag.elo_lo:.1f}, {diag.elo_hi:.1f}]"
        if '<span class="btci">' in summary:
            summary = re.sub(
                r'(<span class="btci">)[^<]*(</span>)',
                rf"\g<1>{btci}\g<2>",
                summary,
                count=1,
            )
        else:
            summary = re.sub(
                r'(<span class="btv">[\d.]+</span>)',
                rf'\g<1><span class="btci">{btci}</span>',
                summary,
                count=1,
            )
    # The old cell may carry a title and/or a cbar (or predate both) — the
    # rewrite normalizes to the full shape either way. The pattern MUST
    # consume the cell's own closing </div>: without it every rewrite
    # leaves the original close behind and the rewrite is not idempotent
    # (a re-application would stack orphans and mangle the layout).
    summary, n_cost = re.subn(
        r'<div class="cell cnum cell-cost"(?: title="[^"]*")?>'
        r'<span class="cv">[^<]*</span>'
        r'(?:<span class="cbar"><i style="width:[^"]*"></i></span>)?'
        r"</div>",
        cost_cell,
        summary,
        count=1,
    )
    summary, n_tok = re.subn(
        r'<div class="cell cnum cell-tok"(?: title="[^"]*")?>'
        r'<span class="cv">[^<]*</span>'
        r'(?:<span class="cbar"><i style="width:[^"]*"></i></span>)?'
        r"</div>",
        tok_cell,
        summary,
        count=1,
    )
    if n_cost != 1 or n_tok != 1:
        raise SystemExit(
            "[update] refusing to rewrite a leaderboard row whose "
            "cost/token cells do not match the page's markup shape"
        )
    return (
        summary
        + "\n"
        + _detail_html(stats, snapshot, diag, entry)
        + "\n</details>"
    )


def _detail_html(
    stats: _NewcomerStats,
    snapshot: Any,
    diag: _StrengthDiag | None,
    entry: dict[str, Any],
) -> str:
    """A participant's expanded detail: the four dcards + the bootstrap
    rank histogram — the markup shape every original row uses, so
    appended entrants and refreshed incumbents are indistinguishable. The
    identity strip (model / harness / reasoning effort / provider) leads
    the block: those pins are visible only once the row is expanded."""
    total_cost = stats.gen_cost + stats.val_cost + stats.eval_cost
    success = stats.defense_success
    success_txt = f"{success * 100:.1f}%" if success is not None else "\u2014"
    cost_title = (
        f"generation ${stats.gen_cost:.2f} \u00b7 validation ${stats.val_cost:.2f} \u00b7 "
        f"match play ${stats.eval_cost:.2f}"
    )
    med_cost = statistics.median(stats.defense_costs) if stats.defense_costs else 0.0
    med_out = statistics.median(stats.defense_out_tokens) if stats.defense_out_tokens else 0
    pm_cost, pm_tokens = _pm_cost_tokens(entry, stats)
    elo = float(entry["elo"])
    detail = (
        '<div class="detail">'
        f"{_identity_html(entry)}"
        '<div class="dgrid">'
        '<section class="dcard"><h4>Match record</h4>'
        f'<div class="dhead">{stats.record}<span class="dsub">win \u00b7 draw \u00b7 loss</span></div>'
        '<dl><div class="kv"><dt>Matches played</dt>'
        f"<dd>{entry['matches_played']}</dd></div>"
        f'<div class="kv"><dt>League points</dt><dd>{entry["points"]:.1f}</dd></div>'
        f'<div class="kv"><dt>Elo (overall)</dt><dd><b>{elo:.1f}</b></dd></div>'
        f'<div class="kv"><dt>Elo as Red</dt><dd>{snapshot.red_elo:.1f}</dd></div>'
        f'<div class="kv"><dt>Elo as Blue</dt><dd>{snapshot.blue_elo:.1f}</dd></div>'
        f'<div class="kv"><dt>TrueSkill</dt>'
        f"<dd>{snapshot.trueskill_mu:.2f} \u00b1 {snapshot.trueskill_sigma:.2f}</dd></div></dl></section>"
        '<section class="dcard"><h4>Duel outcomes</h4>'
        f'<div class="dhead">{success_txt}<span class="dsub">defense success</span></div>'
        '<dl><div class="kv"><dt>Attacks landed (Red)</dt>'
        f"<dd><b>{stats.attacks_landed}</b></dd></div>"
        f'<div class="kv"><dt>Defenses broken (Blue)</dt>'
        f"<dd>{stats.defenses_broken}</dd></div>"
        f'<div class="kv"><dt>Defense success</dt>'
        f"<dd><b>{success_txt}</b></dd></div>"
        f'<div class="kv"><dt>Turns defended</dt>'
        f"<dd>{stats.turns_defended}</dd></div></dl></section>"
        '<section class="dcard"><h4>Cost &amp; tokens</h4>'
        f'<div class="dhead">{_fmt_money(total_cost)}<span class="dsub">total spend</span></div>'
        f'<div class="costbar" title="{cost_title}">'
        f'<i class="sw-gen" style="width:{stats.gen_cost / total_cost * 100 if total_cost else 0:.2f}%"></i>'
        f'<i class="sw-val" style="width:{stats.val_cost / total_cost * 100 if total_cost else 0:.2f}%"></i>'
        f'<i class="sw-eval" style="width:{stats.eval_cost / total_cost * 100 if total_cost else 0:.2f}%"></i></div>'
        '<dl><div class="kv"><dt><i class="sw sw-gen"></i>Generation</dt>'
        f"<dd>{_fmt_money(stats.gen_cost)}</dd></div>"
        f'<div class="kv"><dt><i class="sw sw-val"></i>Validation</dt>'
        f"<dd>{_fmt_money(stats.val_cost)}</dd></div>"
        f'<div class="kv"><dt><i class="sw sw-eval"></i>Match play</dt>'
        f"<dd>{_fmt_money(stats.eval_cost)}</dd></div>"
        '<div class="kv"><dt>Cost per match</dt>'
        f"<dd>{_fmt_money(pm_cost)}</dd></div>"
        '<div class="kv"><dt>Median cost / turn</dt>'
        f"<dd>{_fmt_money(med_cost)}</dd></div>"
        '<div class="kv"><dt>Output tokens</dt>'
        f"<dd><b>{_fmt_tokens(stats.out_tokens)}</b></dd>"
        f'<span class="ci">{_fmt_tokens(stats.match_out_tokens)} in matches</span></div>'
        '<div class="kv"><dt>Tokens per match</dt>'
        f"<dd>{pm_tokens:,}</dd></div>"
        '<div class="kv"><dt>Median output / turn</dt>'
        f"<dd>{med_out:,}</dd></div></dl></section>"
    )
    if diag is None:
        return detail + "</div></div>"
    strength_card, rank_hist_html = _strength_card_html(diag, "#59c886")
    return detail + strength_card + "</div>" + rank_hist_html + "</div>"


def _newcomer_row_html(
    entry: dict[str, Any],
    rank: int,
    stats: _NewcomerStats,
    snapshot: Any,
    lo: float,
    hi: float,
    max_cost: float,
    max_tokens: int,
    diag: _StrengthDiag | None = None,
) -> str:
    """A full leaderboard row for a new entrant (the appended 'late entrant' shape).

    ``diag`` (read from the re-exported rankings' bootstrap fields) adds
    the 95% Elo CI badge, the ``Strength diagnostics`` card, and the
    ``Bootstrap rank distribution`` histogram — the same components the
    original field's rows carry, so the appended row is complete. The row
    carries no visible late-entrant pill — only the invisible
    ``data-late-entrant`` marker (provenance only; nothing on the page
    renders it).
    """
    model = str(entry["model"])
    harness = str(entry["harness"])
    elo = float(entry["elo"])
    name = _display_name(model)
    org = _org_of(model)
    harness_label = _PAGE_HARNESS_LABELS.get(harness, harness)
    pm_cost, pm_tokens = _pm_cost_tokens(entry, stats)
    cost_cell, tok_cell = _cost_tok_cells(entry, stats, max_cost, max_tokens)
    esc = html.escape
    btci = (
        f'<span class="btci">[{diag.elo_lo:.1f}, {diag.elo_hi:.1f}]</span>'
        if diag is not None
        else ""
    )
    return (
        f'<details class="rwrap" data-late-entrant="1" data-rank="{rank}" '
        f'data-elo="{elo:.1f}" '
        f'data-usd_per_match="{pm_cost:.2f}" data-tokens_per_match="{pm_tokens}">\n'
        f'<summary><div class="row"><div class="cell cell-rank">{rank}</div>'
        f'<div class="cell entrant" title="{esc(model)} \u00b7 {esc(org)}">'
        f'<span class="enames"><span class="nm">{esc(name)}</span>'
        f'<span class="pills"><span class="hpill">{esc(harness_label)}</span>'
        f'<span class="org">{esc(org)}</span></span></span></div>'
        f'<div class="cell chart"><div class="track">'
        f'<div class="bar pos" style="left:0.000%;width:{_bar_width(elo, lo, hi):.3f}%"></div>'
        f'</div><div class="val"><span class="btv">{elo:.1f}</span>{btci}</div></div>'
        f"{cost_cell}"
        f"{tok_cell}"
        f'<div class="cell cell-chev">\u25b8</div></div></summary>\n'
        f"{_detail_html(stats, snapshot, diag, entry)}\n</details>"
    )


def _entry_cid(entry: dict[str, Any]) -> str:
    """Composite cid for a rankings entry (auto-route providers collapse)."""
    provider = str(entry.get("provider") or "")
    if provider == "(OpenRouter auto-route)":
        provider = ""
    return composite_id(
        str(entry["model"]),
        str(entry["harness"]),
        str(entry.get("reasoning_effort") or ""),
        provider,
    )


def _update_leaderboard(
    index_path: Path,
    old_rankings: dict[str, Any] | None,
    new_rankings: dict[str, Any],
    stats_by_cid: dict[str, _NewcomerStats],
    snapshots: dict[str, Any],
    diags: dict[str, _StrengthDiag] | None = None,
) -> list[str]:
    """Rewrite the leaderboard page; returns the appended entrant cids.

    Every participant matched to a rankings entry gets its statistics
    refreshed from the merged history: the summary keeps its identity
    visuals (name / harness + org pills — no model-color dot; legacy dots
    are stripped) while its
    attrs, rank cell, Elo bar + value, 95% Elo CI badge, per-match
    cost/token cells, and the whole detail block (identity strip with the
    reasoning effort + provider pins / match record / duel
    outcomes / cost & tokens / Strength diagnostics / Bootstrap rank
    distribution) are
    regenerated — so the page stays consistent with the re-exported
    rankings JSON after every import. Participants in ``stats_by_cid``
    without an existing row are appended as full ``late entrant`` rows.
    The Elo axis gridlines and the meta-description entrant count follow
    the new field. No late-entrant / contribution-history legend chip is
    rendered (a legacy one is removed).
    """
    if not index_path.is_file():
        print(
            f"[update] {index_path} not found — skipping the leaderboard "
            "page update (rankings JSON is still updated)."
        )
        return []
    text = index_path.read_text()

    entries = [dict(e) for e in (new_rankings.get("rankings") or [])]
    if not entries:
        raise SystemExit(
            "refreshed rankings carry no entries — refusing to touch the page."
        )
    for entry in entries:
        entry["cid"] = _entry_cid(entry)
    by_cid = {entry["cid"]: entry for entry in entries}

    # Dense Elo-descending rank (the page's rank semantics).
    elo_order = sorted(entries, key=lambda e: (-float(e["elo"]), e["cid"]))
    elo_rank: dict[str, int] = {}
    for entry in elo_order:
        elo_rank.setdefault(entry["cid"], len(elo_rank) + 1)
    lo, hi = _page_axis([float(e["elo"]) for e in entries])

    old_entries = [dict(e) for e in (old_rankings or {}).get("rankings") or []]
    for entry in old_entries:
        entry["cid"] = _entry_cid(entry)

    # Parse every existing row first (identity anchors), decide its target
    # entry, and only then fix the per-match cost/token bar scale: the
    # REFRESHED stats set the scale for matched rows; unmatched rows keep
    # their parsed maxima (their cells are left untouched).
    parsed_rows: list[tuple[str, dict[str, Any] | None, str | None]] = []
    unmatched = 0
    for match in _ROW_BLOCK_RE.finditer(text):
        parsed = _parse_row(match.group(0))
        cid: str | None = None
        if parsed is None:
            unmatched += 1
        elif parsed["model_harness"] is not None:
            model, harness = parsed["model_harness"]
            candidates = [
                e
                for e in old_entries
                if e["model"] == model and e["harness"] == harness
            ] or [e for e in entries if e["model"] == model and e["harness"] == harness]
            if parsed["old_elo"] is not None and len(candidates) > 1:
                exact = [
                    e
                    for e in candidates
                    if abs(float(e["elo"]) - parsed["old_elo"]) < 0.05
                ]
                candidates = exact or candidates
            if len(candidates) == 1 and candidates[0]["cid"] in by_cid:
                cid = candidates[0]["cid"]
        if parsed is not None and cid is None:
            unmatched += 1
        parsed_rows.append((match.group(0), parsed, cid))

    # Bar maxima are per-match averages (the page's cell semantics), so
    # entrants with different match counts compare fairly.
    max_cost = max(
        [
            _pm_cost_tokens(by_cid[cid], s)[0]
            for cid, s in stats_by_cid.items()
            if cid in by_cid
        ]
        + [
            parsed["cost"]
            for _b, parsed, cid in parsed_rows
            if parsed is not None and cid is None
        ],
        default=0.0,
    )
    max_tokens = max(
        [
            _pm_cost_tokens(by_cid[cid], s)[1]
            for cid, s in stats_by_cid.items()
            if cid in by_cid
        ]
        + [
            parsed["tokens"]
            for _b, parsed, cid in parsed_rows
            if parsed is not None and cid is None
        ],
        default=0,
    )

    # Rewrite matched rows; participants that already have a row never get
    # an appended duplicate.
    matched_cids: set[str] = set()
    out_blocks: list[str] = []
    for block, parsed, cid in parsed_rows:
        if parsed is None or cid is None:
            out_blocks.append(block)
            continue
        matched_cids.add(cid)
        stats = stats_by_cid.get(cid)
        snapshot = snapshots.get(cid)
        if stats is None or snapshot is None:
            # No merged-history stats for this row: refresh the core
            # ranking columns only (the pre-refresh behaviour).
            out_blocks.append(
                _update_row_block(
                    block, elo_rank[cid], float(by_cid[cid]["elo"]), lo, hi, by_cid[cid]
                )
            )
            continue
        out_blocks.append(
            _refresh_row_block(
                block,
                elo_rank[cid],
                by_cid[cid],
                stats,
                snapshot,
                (diags or {}).get(cid),
                lo,
                hi,
                max_cost,
                max_tokens,
            )
        )
    text = _ROW_BLOCK_RE.sub(lambda _m: out_blocks.pop(0), text)

    # Append the fresh entrants (in rank order) after the last existing row.
    # Maxima already include every participant's refreshed stats above.
    appended: list[str] = []
    rendered: list[str] = []
    for entry in elo_order:
        cid = entry["cid"]
        if cid in matched_cids:
            continue  # already refreshed in place — never duplicated
        stats = stats_by_cid.get(cid)
        if stats is None:
            continue
        snapshot = snapshots.get(cid)
        if snapshot is None:
            continue
        rendered.append(
            _newcomer_row_html(
                entry,
                elo_rank[cid],
                stats,
                snapshot,
                lo,
                hi,
                max_cost,
                max_tokens,
                diag=(diags or {}).get(cid),
            )
        )
        appended.append(cid)
    if rendered:
        last_row_end = text.rindex("</details>")
        insert_at = last_row_end + len("</details>")
        text = text[:insert_at] + "\n" + "\n".join(rendered) + text[insert_at:]

    # Elo axis gridlines + meta-description entrant count.
    text = _GRIDLINES_RE.sub(rf"\g<1>{_gridlines_html(lo, hi)}\g<2>", text, count=1)
    text = re.sub(r"(\d+) entrants", f"{len(entries)} entrants", text, count=1)
    # The page no longer advertises late entrants / the contribution
    # history: drop the legacy legend chip if an older page still has it.
    text = _HISTORY_CHIP_RE.sub("", text)
    text = _ensure_dident_css(text)

    index_path.write_text(text)
    if unmatched:
        print(
            f"[update] note: {unmatched} leaderboard row(s) could not be "
            "matched to a rankings participant and were left untouched."
        )
    return appended


# ── contributions archive + history ───────────────────────────────


def _load_registry(contributions_dir: Path) -> dict[str, Any]:
    path = contributions_dir / "contributions.json"
    if path.is_file():
        try:
            registry = json.loads(path.read_text())
        except json.JSONDecodeError:
            registry = {}
        if isinstance(registry, dict) and isinstance(registry.get("contributions"), list):
            return registry
    return {"contributions": []}


def _entry_columns(cid: str) -> dict[str, Any]:
    model, harness, effort, provider = split_composite_id(cid)
    return {
        "cid": cid,
        "model": model,
        "harness": harness or "mini-swe-agent",
        "reasoning_effort": effort or None,
        "provider": provider or "(OpenRouter auto-route)",
    }


def _render_contributions_html(registry: dict[str, Any]) -> str:
    """A small self-contained page visualizing the contribution history."""
    cards = []
    for entry in registry.get("contributions") or []:
        newcomers = " ".join(
            (
                f'<li><span class="nm">{html.escape(_display_name(str(n.get("model") or "")))}</span> '
                f'<span class="mono">{html.escape(str(n.get("cid") or ""))}</span> '
                f'<span class="pill">rank #{n.get("rank", "?")}</span> '
                f'<span class="pill">{float(n.get("points") or 0):.1f} pts</span> '
                f'<span class="pill">Elo {float(n.get("elo") or 0):.1f}</span> '
                f'<span class="pill">{html.escape(str(n.get("record") or ""))}</span></li>'
            )
            for n in entry.get("newcomers") or []
        )
        matches = " ".join(
            (
                f'<li><span class="mono">{html.escape(str(m.get("match_id") or "")[:13])}…</span> '
                f'{html.escape(str(m.get("model_a") or ""))} vs '
                f'{html.escape(str(m.get("model_b") or ""))} → '
                f'<b>{html.escape(str(m.get("outcome") or ""))}</b></li>'
            )
            for m in entry.get("matches") or []
        )
        added = entry.get("added") or {}
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(str(entry.get('imported_at') or '')[:19].replace('T', ' · '))}</h2>"
            "<p class='meta'>tournament <span class='mono'>"
            f"{html.escape(str(entry.get('tournament_id') or ''))}</span> · "
            f"zip <span class='mono'>{html.escape(str(entry.get('zip_name') or ''))}</span> · "
            f"<span class='mono'>{html.escape(str(entry.get('hash') or '')[:16])}…</span></p>"
            f"<p class='meta'>added {added.get('matches', 0)} match(es), "
            f"{added.get('defenses', 0)} defense(s), "
            f"{added.get('challenges', 0)} challenge(s), "
            f"{added.get('failed_challenges', 0)} failed attempt(s) · "
            f"{added.get('skipped_existing', 0)} already present (skipped)</p>"
            f"<h3>Entrants added</h3><ul class='newcomers'>{newcomers}</ul>"
            f"<h3>Matches</h3><ul class='matches'>"
            f"{matches or '<li>(none recorded)</li>'}</ul>"
            "</section>"
        )
    if not cards:
        cards = [
            "<section class='card'><p class='meta'>"
            "No contributions imported yet.</p></section>"
        ]
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SWE-Duel — Contribution History</title>
<style>
body{{background:#151515;color:#fafafa;font-family:ui-monospace,"SF Mono",Menlo,monospace;
     margin:0;padding:32px;line-height:1.55}}
h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:15px;margin:0 0 6px;color:#e8e8ea}}
h3{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#8a8a90;margin:14px 0 6px}}
a{{color:#9fd0ff;text-decoration:none}}
.mono{{color:#b0b0b6}}
.meta{{color:#b0b0b6;font-size:12.5px;margin:2px 0}}
.pill{{display:inline-block;border:1px solid #2a2a2a;border-radius:9999px;padding:1px 9px;
      font-size:11px;color:#b0b0b6;margin-left:4px}}
.card{{background:#1d1d1d;border:1px solid #2a2a2a;border-radius:10px;padding:18px 20px;
      margin-bottom:18px;max-width:900px}}
ul{{list-style:none;padding:0;margin:0}}
li{{padding:5px 0;border-bottom:1px solid #232326;font-size:13px}}
li:last-child{{border-bottom:none}}
.nm{{color:#fafafa;font-weight:600;margin-right:8px}}
footer{{color:#8a8a90;font-size:12px;margin-top:28px}}
</style></head><body>
<h1>SWE-Duel contribution history</h1>
<p class="meta">External entrants imported via <span class="mono">swe-duel-tournament-update</span>
 — new participants actively sampled against the original tournament field
(<span class="mono">swe-duel-tournament-as</span>), merged into the organizer's
<span class="mono">./data/</span>, and folded into
<span class="mono">./rankings/rankings_&lt;id&gt;.json</span>.</p>
{body}
<footer>one zip per tournament · duplicate submissions are rejected by SHA-256</footer>
</body></html>
"""


def _write_history(contributions_dir: Path, registry: dict[str, Any]) -> tuple[Path, Path]:
    """Persist the registry JSON + regenerate the history page."""
    contributions_dir.mkdir(parents=True, exist_ok=True)
    json_path = contributions_dir / "contributions.json"
    json_path.write_text(json.dumps(registry, indent=2))
    html_path = contributions_dir / "index.html"
    html_path.write_text(_render_contributions_html(registry))
    return json_path, html_path


# ── duplicate-import guard ──────────────────────────────────────


def _ensure_not_imported(contributions_dir: Path, sha: str) -> None:
    registry = _load_registry(contributions_dir)
    for entry in registry.get("contributions") or []:
        if entry.get("hash") == sha:
            raise SystemExit(
                f"this exact submission (sha256 {sha[:16]}…) was already "
                f"imported on {entry.get('imported_at', 'an unknown date')} — "
                "duplicate imports are rejected; submit a NEW zip."
            )
    if (contributions_dir / sha).is_dir():
        raise SystemExit(
            f"./contributions/{sha[:16]}… already exists — this submission "
            "was already imported (or a previous import of it partially "
            "completed). Remove that directory manually to force a re-import."
        )


# ── entry point ─────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swe-duel-tournament-update",
        description=(
            "Import an external contributor's ./data submission (a .zip of "
            "the output of swe-duel-tournament-as --tournament <id>) into "
            "the local arena: merge the contributed matches / defenses / "
            "challenges, re-export ./rankings/rankings_<id>.json with the "
            "new entrants, refresh the ./docs/index.html leaderboard, and archive "
            "the contribution under ./contributions/<sha256>/."
        ),
    )
    parser.add_argument(
        "submission_zip",
        help="path to the contributor's zipped ./data directory",
    )
    parser.add_argument(
        "--tournament",
        default=None,
        help=(
            "tournament id the submission entered (loads/updates "
            "./rankings/rankings_<id>.json). Required when the zip carries "
            "active-sampling runs for more than one tournament."
        ),
    )
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
        help="where rankings_<tournament_id>.json is updated (default ./rankings)",
    )
    parser.add_argument(
        "--contributions-dir",
        default="./contributions",
        help="where submissions are archived + the history is rendered "
        "(default ./contributions)",
    )
    parser.add_argument(
        "--index-html",
        default="./docs/index.html",
        help="leaderboard page to refresh (default ./docs/index.html, the "
        "GitHub Pages site root)",
    )
    args = parser.parse_args(argv)

    zip_path = Path(args.submission_zip)
    if not zip_path.is_file():
        raise SystemExit(f"submission zip not found: {zip_path}")
    contributions_dir = Path(args.contributions_dir)
    rankings_dir = Path(args.rankings_dir)
    index_path = Path(args.index_html)

    arena_config = load_arena_config(resolve_config_dir(args))
    data_dir = resolve_output_dir(args, arena_config)
    for needed in ("matches", "defenses", "challenge_bank", "tournaments"):
        if not (data_dir / needed).is_dir():
            raise SystemExit(
                f"{data_dir / needed} not found — this command updates an "
                "existing arena's ./data (run swe-duel init / a tournament first)."
            )

    # ── 1. duplicate guard ────────────────────────────────────
    sha = _zip_sha256(zip_path)
    _ensure_not_imported(contributions_dir, sha)
    print(f"[update] submission {zip_path.name} sha256={sha[:16]}… — new contribution")

    with tempfile.TemporaryDirectory(prefix="swe-duel-update-") as tmp:
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir()
        sub_data, _extracted_bytes = _extract_submission(zip_path, extract_dir)

        # ── 2. scope: the AS runs + the tournament they entered ──
        as_states = _load_as_states(sub_data / "tournaments")
        tournament_id, selected_states = _resolve_tournament(as_states, args.tournament)
        newcomers = list(
            dict.fromkeys(
                cid for _path, state in selected_states for cid in _as_newcomers(state)
            )
        )
        print(
            f"[update] tournament {tournament_id} — "
            f"{len(selected_states)} active-sampling state(s), "
            f"{len(newcomers)} new entrant(s)"
        )
        for cid in newcomers:
            print(f"[update]   new entrant: {cid}")

        # Snapshot the pre-update rankings (row matching needs the old Elo).
        old_rankings_path = rankings_dir / f"rankings_{tournament_id}.json"
        old_rankings: dict[str, Any] | None = None
        if old_rankings_path.is_file():
            try:
                old_rankings = json.loads(old_rankings_path.read_text())
            except json.JSONDecodeError:
                old_rankings = None

        # ── 3. merge into ./data/ ───────────────────────────────
        counts, copied, as_matches, challenges, defenses = _merge_submission(
            sub_data, data_dir, selected_states, tournament_id
        )
        print(
            f"[update] merged: {counts.matches} match(es), "
            f"{counts.defenses} defense(s), {counts.challenges} challenge(s), "
            f"{counts.failed_challenges} failed attempt(s), "
            f"{counts.as_states} AS state(s); "
            f"{counts.skipped_existing} file(s) already present (skipped); "
            f"{counts.index_entries_added} index.json entr(y/ies) added"
        )

        # ── 4. re-export the tournament's rankings ───────────────
        export_rankings(tournament_id, data_dir, rankings_dir)
        new_rankings = json.loads(
            (rankings_dir / f"rankings_{tournament_id}.json").read_text()
        )
        entry_by_cid = {
            _entry_cid(e): e for e in new_rankings.get("rankings") or []
        }
        old_cids = {
            _entry_cid(e) for e in (old_rankings or {}).get("rankings") or []
        }
        fresh_newcomers = [cid for cid in newcomers if cid not in old_cids]
        print(
            f"[update] rankings re-exported: {len(entry_by_cid)} participants, "
            f"{len(new_rankings.get('matches') or [])} matches "
            f"({len(fresh_newcomers)} new entrant(s))"
        )

        # ── 5. refresh the leaderboard page ──────────────────────
        # Rating snapshots + per-participant stats over the MERGED history
        # (the original schedule + the imported active-sampled matches):
        # every participant's row — incumbent or entrant — is refreshed
        # from the same numbers the re-exported rankings JSON carries.
        kept, _dropped = _dedupe(_load_matches(data_dir / "matches"))
        states = _load_round_robin_states(data_dir / "tournaments") + _load_swiss_states(
            data_dir / "tournaments"
        )
        claimed = _claim_matches_per_tournament(states, kept).get(tournament_id, [])
        claimed_ids = {str(m.get("match_id") or "") for m in claimed}
        combined = claimed + [
            m for m in as_matches if str(m.get("match_id") or "") not in claimed_ids
        ]
        snapshots = compute_all_ratings(
            cast("list[MatchResult]", [_to_match_result(m) for m in combined]),
            model_ids=list(entry_by_cid),
        )
        # Full-arena, field-scoped bank/defense payloads: incumbents' stats
        # span their entire history (the original tournament + the import).
        field_identities = {_Identity.from_cid(cid) for cid in entry_by_cid}
        arena_challenges, arena_defenses = _load_scope_payloads(
            data_dir, field_identities
        )
        stats_by_cid = {
            cid: _newcomer_stats(cid, combined, arena_challenges, arena_defenses)
            for cid in entry_by_cid
        }
        # Bootstrap strength diagnostics: read them from the re-exported
        # rankings (the export computes the SAME seeded bootstrap, so the
        # page and the JSON show identical numbers); fall back to one local
        # pass when the export lacks the fields (legacy export shape).
        diags = {
            cid: diag
            for cid, entry in (
                (_entry_cid(e), e) for e in new_rankings.get("rankings") or []
            )
            if (diag := _diag_from_entry(entry)) is not None
        }
        if len(diags) != len(entry_by_cid) and index_path.is_file():
            local = _bootstrap_field_diagnostics(combined)
            diags = {cid: local[cid] for cid in entry_by_cid if cid in local}
        appended = _update_leaderboard(
            index_path,
            old_rankings,
            new_rankings,
            stats_by_cid,
            snapshots,
            diags,
        )
        if index_path.is_file():
            print(
                f"[update] leaderboard: {len(appended)} row(s) appended, "
                "existing rows fully refreshed from the merged history "
                "(rank/Elo + 95% Elo CI, per-match cost/tokens, duel "
                "outcomes, strength diagnostics + bootstrap rank distribution)"
            )

        # ── 6. archive the contribution + update the history ─────
        contribution_dir = contributions_dir / sha
        (contribution_dir / "data").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(zip_path, contribution_dir / "submission.zip")
        standings = []
        for cid in newcomers:
            entry = entry_by_cid.get(cid)
            if entry is None:
                continue
            stats = stats_by_cid.get(cid)
            standings.append(
                {
                    **_entry_columns(cid),
                    "points": float(entry.get("points") or 0.0),
                    "matches_played": int(entry.get("matches_played") or 0),
                    "elo": float(entry.get("elo") or 0.0),
                    "rank": int(entry.get("rank") or 0),
                    "record": stats.record if stats is not None else "",
                }
            )
        manifest = {
            "hash": sha,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "zip_name": zip_path.name,
            "zip_bytes": zip_path.stat().st_size,
            "tournament_id": tournament_id,
            "as_states": [
                str(state.get("tournament_id") or "") for _path, state in selected_states
            ],
            "newcomers": standings,
            "added": {
                "matches": counts.matches,
                "defenses": counts.defenses,
                "challenges": counts.challenges,
                "failed_challenges": counts.failed_challenges,
                "as_states": counts.as_states,
                "skipped_existing": counts.skipped_existing,
            },
            "matches": [
                {
                    "match_id": str(m.get("match_id") or ""),
                    "model_a": str(m.get("model_a_id") or ""),
                    "model_b": str(m.get("model_b_id") or ""),
                    "outcome": str(m.get("outcome") or "draw"),
                    "timestamp": str(m.get("timestamp") or ""),
                }
                for m in as_matches
            ],
        }
        for src in copied:
            rel = src.relative_to(sub_data)
            archive_dst = contribution_dir / "data" / rel
            archive_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, archive_dst)
        (contribution_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        registry = _load_registry(contributions_dir)
        registry.setdefault("contributions", []).append(manifest)
        registry["contributions"].sort(key=lambda e: str(e.get("imported_at") or ""))
        registry_json, registry_html = _write_history(contributions_dir, registry)

    print(
        f"\n[update] done. contribution archived → {contribution_dir} "
        "(submission.zip + manifest.json + imported files)\n"
        f"  rankings → {rankings_dir / f'rankings_{tournament_id}.json'}\n"
        f"  history → {registry_json} + {registry_html}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())