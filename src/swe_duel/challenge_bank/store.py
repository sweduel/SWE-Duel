"""Persistent, indexed store of validated Red challenges."""

from __future__ import annotations

import html as html_mod
import json
import os
import random
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from swe_duel.models import (
    AgentTrajectory,
    ChallengePoolStats,
    ChallengeRecord,
    FailedChallengeRecord,
    GateResult,
    GateStatus,
    RedChallenge,
    RedSelfReview,
    RedValidationResult,
    ReviewFinding,
)
from swe_duel.sandbox.diff_utils import _CHALLENGE_CSS, _render_trajectory_steps, render_challenge_html


class ChallengeNotFoundError(KeyError):
    """Raised when a challenge_id cannot be located in the bank."""


class InsufficientChallengesError(ValueError):
    """Raised when the requested sample size exceeds available pool."""


def _pool_key(
    red_model_id: str,
    repo_name: str,
    red_harness_id: str = "mini-swe-agent",
    red_reasoning_effort: str = "",
    red_provider: str = "",
) -> str:
    """Pool key = competitor identity (model, harness, effort, provider) + repo.

    The harness, reasoning effort and OpenRouter provider are part of the key
    so the same model run under different harnesses / efforts / providers
    populates distinct pools and competes as distinct entrants. The
    `model_id#harness_id[#effort#provider]` composite uses `#` (model ids
    contain `/`, harness ids only `[a-z-]`, effort/provider slugs
    `[a-z0-9-]`), and `::` separates the composite id from the repo. An empty
    effort+provider collapses to the legacy 2-part composite so pools written
    before this dimension existed keep their exact keys.
    """
    from swe_duel.models import composite_id

    cid = composite_id(red_model_id, red_harness_id, red_reasoning_effort, red_provider)
    return f"{cid}::{repo_name}"


def _split_pool_key(key: str) -> tuple[str, str, str, str, str]:
    """Inverse of :func:`_pool_key` → (red_model_id, red_harness_id,
    red_reasoning_effort, red_provider, repo_name)."""
    from swe_duel.models import split_composite_id

    cid, repo = key.split("::", 1)
    model_id, harness_id, effort, provider = split_composite_id(cid)
    return (
        model_id,
        harness_id or "mini-swe-agent",
        effort,
        provider,
        repo,
    )


# ── (de)serialisation ───────────────────────────────────────


def _serialise_record(record: ChallengeRecord) -> dict:
    d = asdict(record)
    # asdict turns enums into their values for str-enum subclasses, but datetime
    # must be converted manually.
    d["generated_at"] = record.generated_at.isoformat()
    # normalise gate statuses to their string values
    for gr in d["validation"]["gate_results"]:
        if not isinstance(gr["status"], str):
            gr["status"] = gr["status"].value
    # self_review dict is already produced by asdict; nothing else to convert.
    return d


def _deserialise_record(data: dict) -> ChallengeRecord:
    ch = data["challenge"]
    # Free-text bug label; legacy records carry the old single-word category
    # tags ("logic_error", ...) which load unchanged as plain strings.
    bug_type = str(ch["bug_type"]) if ch.get("bug_type") else None
    trajectory = AgentTrajectory(**ch["agent_trajectory"])
    feature_trajectory = (
        AgentTrajectory(**ch["feature_trajectory"])
        if ch.get("feature_trajectory")
        else None
    )
    bug_trajectory = (
        AgentTrajectory(**ch["bug_trajectory"])
        if ch.get("bug_trajectory")
        else None
    )
    challenge = RedChallenge(
        target_files=list(ch["target_files"]),
        exploration_summary=ch["exploration_summary"],
        feature_spec=ch["feature_spec"],
        feature_rationale=ch["feature_rationale"],
        pr_diff=ch["pr_diff"],
        modified_file_contents=dict(ch["modified_file_contents"]),
        original_file_contents=dict(ch["original_file_contents"]),
        feature_test_code=ch["feature_test_code"],
        bug_type=bug_type,
        bug_description=ch.get("bug_description"),
        bug_location=ch.get("bug_location"),
        bug_test_code=ch.get("bug_test_code"),
        agent_trajectory=trajectory,
        feature_only_file_contents=dict(ch.get("feature_only_file_contents") or {}),
        feature_trajectory=feature_trajectory,
        bug_trajectory=bug_trajectory,
    )
    val_data = data["validation"]
    gate_results = [
        GateResult(
            gate_name=g["gate_name"],
            status=GateStatus(g["status"]),
            message=g["message"],
            stdout=g.get("stdout", ""),
            stderr=g.get("stderr", ""),
            duration_ms=g.get("duration_ms", 0),
            command=g.get("command", ""),
        )
        for g in val_data["gate_results"]
    ]
    self_review = _deserialise_self_review(val_data.get("self_review"))
    validation = RedValidationResult(
        passed=val_data["passed"],
        gate_results=gate_results,
        attempt_number=val_data["attempt_number"],
        self_review=self_review,
    )
    return ChallengeRecord(
        challenge_id=data["challenge_id"],
        red_model_id=data["red_model_id"],
        repo_name=data["repo_name"],
        repo_commit_sha=data["repo_commit_sha"],
        target_files=list(data["target_files"]),
        challenge=challenge,
        validation=validation,
        generated_at=datetime.fromisoformat(data["generated_at"]),
        generation_cost_usd=float(data["generation_cost_usd"]),
        generation_retries=int(data["generation_retries"]),
        red_harness_id=str(data.get("red_harness_id", "mini-swe-agent")),
        slot=int(data.get("slot", 1)),
        red_reasoning_effort=str(data.get("red_reasoning_effort", "")),
        red_provider=str(data.get("red_provider", "")),
    )


def _deserialise_self_review(data: dict | None) -> RedSelfReview | None:
    if not data:
        return None
    findings = [
        ReviewFinding(
            location=str(f.get("location", "")),
            severity=str(f.get("severity", "info")),
            description=str(f.get("description", "")),
        )
        for f in (data.get("findings") or [])
        if isinstance(f, dict)
    ]
    traj_data = data.get("agent_trajectory") or {}
    trajectory = AgentTrajectory(**traj_data) if traj_data else AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, model_id="", duration_seconds=0.0,
    )
    return RedSelfReview(
        detected=bool(data.get("detected", False)),
        findings=findings,
        fix_explanation=str(data.get("fix_explanation", "")),
        agent_trajectory=trajectory,
        detection_reason=str(data.get("detection_reason", "")),
    )


def _deserialise_challenge_or_none(ch: dict | None) -> RedChallenge | None:
    if not ch:
        return None
    bug_type = str(ch["bug_type"]) if ch.get("bug_type") else None
    trajectory = AgentTrajectory(**ch["agent_trajectory"]) if ch.get("agent_trajectory") else AgentTrajectory(
        steps=[], total_steps=0, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, model_id="", duration_seconds=0.0,
    )
    feature_trajectory = (
        AgentTrajectory(**ch["feature_trajectory"]) if ch.get("feature_trajectory") else None
    )
    bug_trajectory = (
        AgentTrajectory(**ch["bug_trajectory"]) if ch.get("bug_trajectory") else None
    )
    return RedChallenge(
        target_files=list(ch.get("target_files") or []),
        exploration_summary=str(ch.get("exploration_summary", "")),
        feature_spec=str(ch.get("feature_spec", "")),
        feature_rationale=str(ch.get("feature_rationale", "")),
        pr_diff=str(ch.get("pr_diff", "")),
        modified_file_contents=dict(ch.get("modified_file_contents") or {}),
        original_file_contents=dict(ch.get("original_file_contents") or {}),
        feature_test_code=str(ch.get("feature_test_code", "")),
        bug_type=bug_type,
        bug_description=ch.get("bug_description"),
        bug_location=ch.get("bug_location"),
        bug_test_code=ch.get("bug_test_code"),
        agent_trajectory=trajectory,
        feature_only_file_contents=dict(ch.get("feature_only_file_contents") or {}),
        feature_trajectory=feature_trajectory,
        bug_trajectory=bug_trajectory,
        pre_existing_failures=list(ch.get("pre_existing_failures") or []),
    )


_FAILED_CSS = (
    "table.stats{border-collapse:collapse;margin:8px 0;font-size:12px}"
    "table.stats td,table.stats th{border:1px solid #d0d7de;padding:4px 8px;"
    "text-align:left}"
    "table.stats tr.combined td{background:#eef2f6;font-weight:600}"
    "table.stats th{background:#f6f8fa}"
    "table.gates{border-collapse:collapse;margin:8px 0;font-size:12px}"
    "table.gates td,table.gates th{border:1px solid #d0d7de;padding:4px 8px;"
    "text-align:left}"
    "table.gates tr.fail td{background:#fde2e4}"
    "table.gates tr.pass td{background:#d1f0df}"
    "table.gates th{background:#f6f8fa}"
    ".error-banner{margin:8px 0;padding:8px 12px;background:#fff1f0;"
    "border:1px solid #ffa39e;border-radius:6px;color:#820014;font-size:13px}"
    ".kind-banner{margin:8px 0;padding:6px 12px;background:#fff7e6;"
    "border:1px solid #ffd591;border-radius:6px;font-size:12px}"
    ".meta{color:#555;font-size:12px;margin:2px 0}"
    ".attempt-chain{margin:8px 0;padding:6px 12px;background:#f0f5ff;"
    "border:1px solid #adc6ff;border-radius:6px;font-size:12px}"
    "ul.findings{margin:6px 0;padding-left:20px;font-size:12px}"
    "details.gate-detail{border:1px solid #d0d7de;border-radius:5px;margin:6px 0;"
    "background:#fff}"
    "details.gate-detail.fail{border-color:#ffa39e}"
    "details.gate-detail.pass{border-color:#9bd9b4}"
    "details.gate-detail > summary{padding:5px 10px;cursor:pointer;font-size:12px}"
    "details.gate-detail .gate-msg{color:#666;font-weight:400}"
    "details.gate-detail .gate-cmd,details.gate-detail .gate-out{padding:4px 10px}"
    "details.gate-detail .gate-cmd b,details.gate-detail .gate-out b{"
    "display:block;font-size:11px;color:#555;margin-top:4px}"
    "details.gate-detail pre{background:#0d1117;color:#c9d1d9;padding:8px;"
    "border-radius:4px;overflow:auto;max-height:340px;font-size:11px;"
    "white-space:pre-wrap;word-break:break-word}"
    "details.gate-detail .gate-cmd pre{background:#1f2937;color:#e5e7eb}"
)


def _phases_to_show(kind: str) -> tuple[bool, bool, bool, bool]:
    """Return (show_feature, show_bug, show_self_review, show_gates) per kind."""
    k = kind or ""
    if k.startswith("timeout-feature") or k.startswith("incomplete-feature"):
        return True, False, False, False
    if k == "feature-gate":
        # Phase A finished but its gates failed; bug phase never ran.
        return True, False, False, True
    if k.startswith("timeout-bug") or k.startswith("incomplete-bug"):
        return True, True, False, False
    if k == "validation":
        # Full attempt: all phases relevant + gate results.
        return True, True, True, True
    # Unknown kind — show everything we have.
    return True, True, True, True


def _traj_stats_row(label: str, traj: AgentTrajectory | None) -> str:
    if traj is None:
        return ""
    return (
        f"<tr><td>{html_mod.escape(label)}</td>"
        f"<td>{int(traj.total_steps or 0):,}</td>"
        f"<td>{int(traj.total_input_tokens or 0):,}</td>"
        f"<td>{int(traj.total_output_tokens or 0):,}</td>"
        f"<td>${float(traj.total_cost_usd or 0.0):.4f}</td>"
        f"<td>{float(traj.duration_seconds or 0.0):.2f}s</td></tr>"
    )


def _combined_stats_row(
    label: str, trajs: list[AgentTrajectory | None], elapsed_seconds: float | None
) -> str:
    steps = sum(int(t.total_steps or 0) for t in trajs if t is not None)
    in_tok = sum(int(t.total_input_tokens or 0) for t in trajs if t is not None)
    out_tok = sum(int(t.total_output_tokens or 0) for t in trajs if t is not None)
    cost = sum(float(t.total_cost_usd or 0.0) for t in trajs if t is not None)
    dur = sum(float(t.duration_seconds or 0.0) for t in trajs if t is not None)
    time_str = (
        f"{dur:.2f}s (wall {elapsed_seconds:.2f}s)"
        if elapsed_seconds is not None
        else f"{dur:.2f}s"
    )
    return (
        f"<tr class='combined'><td><b>{html_mod.escape(label)}</b></td>"
        f"<td>{steps:,}</td>"
        f"<td>{in_tok:,}</td>"
        f"<td>{out_tok:,}</td>"
        f"<td>${cost:.4f}</td>"
        f"<td>{time_str}</td></tr>"
    )


def _gates_table_html(validation: RedValidationResult | None) -> str:
    if validation is None or not validation.gate_results:
        return "<p><em>No gate results recorded.</em></p>"
    rows = []
    details = []
    for g in validation.gate_results:
        status = g.status.value if hasattr(g.status, "value") else str(g.status)
        cls = "pass" if status == "passed" else "fail"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><code>{html_mod.escape(g.gate_name)}</code></td>"
            f"<td>{html_mod.escape(status)}</td>"
            f"<td>{html_mod.escape(g.message or '')}</td>"
            "</tr>"
        )
        details.append(_gate_detail_html(g, status, cls))
    return (
        "<table class='gates'><thead><tr>"
        "<th>Gate</th><th>Status</th><th>Message</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
        f"{''.join(details)}"
    )


def _gate_detail_html(g, status: str, cls: str) -> str:
    """Collapsible per-gate command + stdout/stderr (for both pass and fail)."""
    cmd = getattr(g, "command", "") or ""
    stdout = g.stdout or ""
    stderr = g.stderr or ""
    if not (cmd or stdout or stderr):
        return ""
    parts = [
        f"<details class='gate-detail {cls}'>"
        f"<summary><code>{html_mod.escape(g.gate_name)}</code> — "
        f"{html_mod.escape(status)} "
        f"<span class='gate-msg'>{html_mod.escape((g.message or '')[:160])}</span>"
        "</summary>"
    ]
    if cmd:
        parts.append(
            f"<div class='gate-cmd'><b>command</b><pre>{html_mod.escape(cmd)}</pre></div>"
        )
    if stdout:
        parts.append(
            f"<div class='gate-out'><b>stdout</b><pre>{html_mod.escape(stdout[-20000:])}</pre></div>"
        )
    if stderr:
        parts.append(
            f"<div class='gate-out'><b>stderr</b><pre>{html_mod.escape(stderr[-20000:])}</pre></div>"
        )
    parts.append("</details>")
    return "".join(parts)


def _self_review_block_html(
    record: FailedChallengeRecord,
) -> str:
    """Render findings + reasoning for the self-review gate, when it ran."""
    self_review = None
    if record.validation is not None and record.validation.self_review is not None:
        self_review = record.validation.self_review
    if self_review is None and record.self_review_trajectory is None:
        return ""
    parts: list[str] = ["<h2 id='self-review'>Self-review (emulated Blue)</h2>"]
    if self_review is not None:
        verdict = (
            "<b style='color:#389e0d'>FIXED — bug tests pass on reviewer's code (challenge solvable)</b>"
            if self_review.detected
            else "<b style='color:#c41d7f'>NOT fixed — reviewer could not remove the bug (rejected)</b>"
        )
        parts.append(f"<p>{verdict}</p>")
        if self_review.detection_reason:
            parts.append(
                f"<p class='meta'>{html_mod.escape(self_review.detection_reason)}</p>"
            )
        if self_review.findings:
            items = "".join(
                f"<li><b>{html_mod.escape(f.severity)}</b> @ "
                f"{html_mod.escape(f.location)}: "
                f"{html_mod.escape(f.description)}</li>"
                for f in self_review.findings
            )
            parts.append(f"<ul class='findings'>{items}</ul>")
        else:
            parts.append("<p><em>No findings reported.</em></p>")
        if self_review.fix_explanation:
            parts.append(
                f"<p><b>fix_explanation:</b> "
                f"{html_mod.escape(self_review.fix_explanation)}</p>"
            )
    traj = record.self_review_trajectory or (
        self_review.agent_trajectory if self_review is not None else None
    )
    parts.append(
        _render_trajectory_steps(
            list(traj.steps) if traj else None,
            "Red agent reasoning — self-review phase",
            "traj-self-review",
        )
    )
    return "".join(parts)


def _render_failed_record_html(
    record: FailedChallengeRecord,
    previous_attempts: list[dict] | None = None,
) -> str:
    """Render a failed-challenge HTML page scoped by record.kind.

    Mirrors the successful challenge HTML in look and feel: a summary banner,
    a combined-statistics table, gate results (if any), and the agent
    reasoning steps for every phase that actually ran.
    """
    show_feature, show_bug, show_self_review, show_gates = _phases_to_show(
        record.kind
    )

    title = (
        f"Failed challenge — attempt {record.attempt_number} "
        f"({record.kind}) — {record.red_model_id} × {record.repo_name}"
    )
    kind_banner = (
        f"<div class='kind-banner'><b>Failure kind:</b> "
        f"<code>{html_mod.escape(record.kind)}</code> "
        f"&nbsp;|&nbsp; <b>Attempt:</b> {record.attempt_number} "
        f"&nbsp;|&nbsp; <b>Elapsed:</b> {record.elapsed_seconds:.2f}s "
        f"&nbsp;|&nbsp; <b>Total cost:</b> ${record.generation_cost_usd:.4f}"
        "</div>"
    )
    error_banner = (
        f"<div class='error-banner'><b>Error:</b> "
        f"{html_mod.escape(record.error_message)}</div>"
        if record.error_message
        else ""
    )

    # Combined / overall stats table.
    trajs_for_combined: list[AgentTrajectory | None] = []
    rows: list[str] = []
    if show_feature:
        rows.append(_traj_stats_row("Feature generation", record.feature_trajectory))
        trajs_for_combined.append(record.feature_trajectory)
    if show_bug:
        rows.append(_traj_stats_row("Bug embedding", record.bug_trajectory))
        trajs_for_combined.append(record.bug_trajectory)
    if show_self_review and record.self_review_trajectory is not None:
        rows.append(
            _traj_stats_row("Self-review", record.self_review_trajectory)
        )
        trajs_for_combined.append(record.self_review_trajectory)
    rows.append(
        _combined_stats_row("Combined", trajs_for_combined, record.elapsed_seconds)
    )
    stats_table = (
        "<table class='stats'><thead><tr>"
        "<th>Phase</th><th>Steps</th><th>Input tokens</th>"
        "<th>Output tokens</th><th>Cost (USD)</th><th>Time</th>"
        "</tr></thead><tbody>"
        f"{''.join(r for r in rows if r)}</tbody></table>"
    )

    # Attempt chain.
    chain_html = ""
    if previous_attempts:
        items = "".join(
            f"<li>Attempt {p.get('attempt', '?')} — "
            f"<code>{html_mod.escape(str(p.get('kind', '')))}</code> "
            f"(id <code>{html_mod.escape(str(p.get('id', '')))}</code>)</li>"
            for p in previous_attempts
        )
        chain_html = (
            "<div class='attempt-chain'><b>Previous attempts in this slot:</b>"
            f"<ul>{items}</ul></div>"
        )

    # Gates.
    gates_html = ""
    if show_gates and record.validation is not None:
        gates_html = (
            "<h2 id='gates'>Gate results</h2>"
            f"{_gates_table_html(record.validation)}"
        )

    # Trajectories.
    feature_html = (
        _render_trajectory_steps(
            list(record.feature_trajectory.steps) if record.feature_trajectory else None,
            "Red agent reasoning — feature generation phase",
            "traj-feature",
        )
        if show_feature
        else ""
    )
    bug_html = (
        _render_trajectory_steps(
            list(record.bug_trajectory.steps) if record.bug_trajectory else None,
            "Red agent reasoning — bug embedding phase",
            "traj-bug",
        )
        if show_bug
        else ""
    )
    self_review_html = _self_review_block_html(record) if show_self_review else ""

    body = (
        f"<h1>{html_mod.escape(title)}</h1>"
        f"<p class='meta'>challenge_id: <code>"
        f"{html_mod.escape(record.challenge_id)}</code> &nbsp;|&nbsp; "
        f"target_files: <code>"
        f"{html_mod.escape(', '.join(record.target_files) or '—')}</code></p>"
        f"{kind_banner}{error_banner}{chain_html}"
        "<h2 id='stats'>Overall statistics</h2>"
        f"{stats_table}{gates_html}"
        + ("<h2 id='feature'>Feature phase</h2>" + feature_html if show_feature else "")
        + ("<h2 id='bug'>Bug embedding phase</h2>" + bug_html if show_bug else "")
        + self_review_html
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html_mod.escape(title)}</title>"
        f"<style>{_CHALLENGE_CSS}{_FAILED_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def _serialise_failed(record: FailedChallengeRecord) -> dict:
    d = asdict(record)
    d["generated_at"] = record.generated_at.isoformat()
    if d.get("validation"):
        for gr in d["validation"]["gate_results"]:
            if not isinstance(gr["status"], str):
                gr["status"] = gr["status"].value
    return d


def _deserialise_failed(data: dict) -> FailedChallengeRecord:
    challenge = _deserialise_challenge_or_none(data.get("challenge"))
    validation = None
    val_data = data.get("validation")
    if val_data:
        gate_results = [
            GateResult(
                gate_name=g["gate_name"],
                status=GateStatus(g["status"]),
                message=g["message"],
                stdout=g.get("stdout", ""),
                stderr=g.get("stderr", ""),
                duration_ms=g.get("duration_ms", 0),
                command=g.get("command", ""),
            )
            for g in val_data.get("gate_results", [])
        ]
        validation = RedValidationResult(
            passed=bool(val_data.get("passed", False)),
            gate_results=gate_results,
            attempt_number=int(val_data.get("attempt_number", 0)),
            self_review=_deserialise_self_review(val_data.get("self_review")),
        )

    def _t(d: dict | None) -> AgentTrajectory | None:
        return AgentTrajectory(**d) if d else None

    return FailedChallengeRecord(
        challenge_id=data["challenge_id"],
        red_model_id=data["red_model_id"],
        repo_name=data["repo_name"],
        repo_commit_sha=data["repo_commit_sha"],
        kind=str(data.get("kind", "")),
        error_message=str(data.get("error_message", "")),
        attempt_number=int(data.get("attempt_number", 0)),
        target_files=list(data.get("target_files") or []),
        challenge=challenge,
        validation=validation,
        feature_trajectory=_t(data.get("feature_trajectory")),
        bug_trajectory=_t(data.get("bug_trajectory")),
        self_review_trajectory=_t(data.get("self_review_trajectory")),
        generated_at=datetime.fromisoformat(data["generated_at"]),
        generation_cost_usd=float(data.get("generation_cost_usd", 0.0)),
        elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
        red_harness_id=str(data.get("red_harness_id", "mini-swe-agent")),
        slot=int(data.get("slot", 1)),
        red_reasoning_effort=str(data.get("red_reasoning_effort", "")),
        red_provider=str(data.get("red_provider", "")),
    )


# ── Store ───────────────────────────────────────────────────


class ChallengeStore:
    """On-disk store: `challenges/{id}.json` files + `index.json` master index.

    SLOT IDENTITY (formerly technical debt, now recorded). A "slot" is one
    generation effort for a (competitor, repo) — up to `max_generation_attempts`
    attempts that end in one success or a chain of failures. Every record
    carries a 1-based ``slot`` number (``ChallengeRecord.slot`` /
    ``FailedChallengeRecord.slot``), persisted in the per-record JSON and
    mirrored into the index's ``entries[id]`` map and the ``failed_pools[key]``
    entry dicts. All attempts of a single effort (the success plus the failed
    attempts it chains via ``previous_attempts``) share one ``slot``; the next
    independent effort for the same (competitor, repo) gets ``slot + 1``. So the
    number of *distinct* slots per (competitor, repo) is now reconstructible
    from the recorded slot numbers — see :meth:`next_slot` and
    :meth:`distinct_slot_count`. Consumers that count slots (e.g.
    `run_tournament._slot_counts` for dimming) read these directly.

    Records written before slot tracking existed default to ``slot=1``.
    """

    def __init__(self, bank_dir: Path) -> None:
        self.bank_dir = Path(bank_dir)
        self.challenges_dir = self.bank_dir / "challenges"
        self.failed_dir = self.bank_dir / "failed_challenges"
        self.index_path = self.bank_dir / "index.json"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.challenges_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        # Reentrant so a lock-holding method may call another locked method.
        # Guards every read-modify-write of the in-memory `self.index` so
        # concurrent (model, repo) generation pools cannot corrupt it.
        self._lock = threading.RLock()
        self.index = self._load_index()

    def _load_index(self) -> dict:
        empty = {"pools": {}, "failed_pools": {}, "entries": {}}
        if not self.index_path.exists():
            return empty
        try:
            data = json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return empty
        for key, default in empty.items():
            if key not in data:
                data[key] = default
        return data

    def _save_index(self) -> None:
        """Atomically write `index.json` via temp-then-rename."""
        payload = json.dumps(self.index, indent=2, sort_keys=True, default=str)
        dir_ = str(self.bank_dir)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".index-", suffix=".json.tmp", dir=dir_
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp_path, self.index_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ── CRUD ────────────────────────────────────────────

    def store(
        self,
        record: ChallengeRecord,
        previous_attempts: list[dict] | None = None,
    ) -> str:
        challenge_path = self.challenges_dir / f"{record.challenge_id}.json"
        payload = _serialise_record(record)
        # Embed the slot chain directly in the JSON for one-glance inspection.
        prior = [dict(p) for p in (previous_attempts or [])]
        payload["previous_attempts"] = prior
        challenge_path.write_text(json.dumps(payload, indent=2, default=str))

        html_path = self.challenges_dir / f"{record.challenge_id}.html"
        html_path.write_text(_render_record_html(record))

        key = _pool_key(
            record.red_model_id,
            record.repo_name,
            record.red_harness_id,
            getattr(record, "red_reasoning_effort", "") or "",
            getattr(record, "red_provider", "") or "",
        )
        with self._lock:
            pool = self.index["pools"].setdefault(key, [])
            if record.challenge_id not in pool:
                pool.append(record.challenge_id)
            self.index.setdefault("entries", {})[record.challenge_id] = {
                "id": record.challenge_id,
                "status": "success",
                "kind": "success",
                "attempt": record.generation_retries + 1,
                "slot": record.slot,
                "red_model_id": record.red_model_id,
                "red_harness_id": record.red_harness_id,
                "repo_name": record.repo_name,
                "red_reasoning_effort": getattr(record, "red_reasoning_effort", "") or "",
                "red_provider": getattr(record, "red_provider", "") or "",
                "previous_attempts": prior,
            }
            self._save_index()
        return record.challenge_id

    def store_failed(
        self,
        record: FailedChallengeRecord,
        previous_attempts: list[dict] | None = None,
    ) -> str:
        """Persist a failed/partial generation attempt to the bank.

        Failure records live in `failed_challenges/{id}.{json,html}` and are
        indexed under `index.json::failed_pools[key]` plus the global
        `entries[challenge_id]` map. Tournament/match code consults only
        `pools` so these records do not affect pairing.
        """
        path = self.failed_dir / f"{record.challenge_id}.json"
        payload = _serialise_failed(record)
        prior = [dict(p) for p in (previous_attempts or [])]
        payload["previous_attempts"] = prior
        path.write_text(json.dumps(payload, indent=2, default=str))

        # Emit a stand-alone HTML view of the failed attempt, scoped to the
        # phases that actually ran.
        html_path = self.failed_dir / f"{record.challenge_id}.html"
        html_path.write_text(
            _render_failed_record_html(record, previous_attempts=prior)
        )

        key = _pool_key(
            record.red_model_id,
            record.repo_name,
            record.red_harness_id,
            getattr(record, "red_reasoning_effort", "") or "",
            getattr(record, "red_provider", "") or "",
        )
        with self._lock:
            entries = self.index.setdefault("failed_pools", {}).setdefault(key, [])
            if not any(e.get("id") == record.challenge_id for e in entries):
                entries.append(
                    {
                        "id": record.challenge_id,
                        "kind": record.kind,
                        "attempt": record.attempt_number,
                        "slot": record.slot,
                        "status": "failed",
                    }
                )
            self.index.setdefault("entries", {})[record.challenge_id] = {
                "id": record.challenge_id,
                "status": "failed",
                "kind": record.kind,
                "attempt": record.attempt_number,
                "slot": record.slot,
                "red_model_id": record.red_model_id,
                "red_harness_id": record.red_harness_id,
                "repo_name": record.repo_name,
                "red_reasoning_effort": getattr(record, "red_reasoning_effort", "") or "",
                "red_provider": getattr(record, "red_provider", "") or "",
                "previous_attempts": prior,
            }
            self._save_index()
        return record.challenge_id

    def get_failed(self, challenge_id: str) -> FailedChallengeRecord:
        path = self.failed_dir / f"{challenge_id}.json"
        if not path.exists():
            raise ChallengeNotFoundError(challenge_id)
        return _deserialise_failed(json.loads(path.read_text()))

    def count_failed_in_pool(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> int:
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        with self._lock:
            return len(self.index.get("failed_pools", {}).get(key, []))

    def _slots_in_pool(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> set[int]:
        """Distinct generation-slot numbers recorded for this (competitor, repo).

        Reads the global ``entries[id]`` map (which carries ``slot`` and the
        owning ``red_model_id``/``red_harness_id``/``repo_name`` (plus
        ``red_reasoning_effort``/``red_provider``) for every persisted attempt —
        successful or failed). Records written before slot tracking default to
        slot 1; records written before effort/provider selection default to
        empty selections.
        """
        slots: set[int] = set()
        with self._lock:
            entries = list(self.index.get("entries", {}).values())
        for e in entries:
            if not isinstance(e, dict):
                continue
            if e.get("red_model_id") != red_model_id:
                continue
            if e.get("red_harness_id", "mini-swe-agent") != red_harness_id:
                continue
            if e.get("repo_name") != repo_name:
                continue
            if str(e.get("red_reasoning_effort", "") or "") != red_reasoning_effort:
                continue
            if str(e.get("red_provider", "") or "") != red_provider:
                continue
            slots.add(int(e.get("slot", 1)))
        return slots

    def distinct_slot_count(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> int:
        """Number of distinct generation slots (efforts) for (competitor, repo)."""
        return len(self._slots_in_pool(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        ))

    def next_slot(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> int:
        """1-based slot number for the *next* generation effort in this pool.

        ``max(recorded slots) + 1`` (1 when the pool is empty). Used by the
        generator so each new effort lands in its own slot rather than reusing
        slot 1 — letting follow-up generations be tracked as distinct slots.
        """
        slots = self._slots_in_pool(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        return (max(slots) + 1) if slots else 1

    def list_attempts_by_slot(
        self,
        red_model_id: str,
        repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> dict[int, list[dict[str, object]]]:
        """Group persisted bank entries by generation slot for this pool.

        Returns ``{slot: [{id, status, kind, attempt}, ...]}`` with each list
        sorted by ``attempt`` ascending. Both successes (``pools``/entries with
        ``status=success``) and failed attempts (``failed_pools``) are included,
        so a fully-exhausted failed-only slot still registers as attempted.
        Callers use this to skip re-running slots ``1..n`` that already have
        any record.
        """
        by_slot: dict[int, list[dict[str, object]]] = defaultdict(list)
        with self._lock:
            entries = list(self.index.get("entries", {}).values())
        for e in entries:
            if not isinstance(e, dict):
                continue
            if e.get("red_model_id") != red_model_id:
                continue
            if e.get("red_harness_id", "mini-swe-agent") != red_harness_id:
                continue
            if e.get("repo_name") != repo_name:
                continue
            if str(e.get("red_reasoning_effort", "") or "") != red_reasoning_effort:
                continue
            if str(e.get("red_provider", "") or "") != red_provider:
                continue
            slot = int(e.get("slot", 1))
            by_slot[slot].append(
                {
                    "id": e.get("id", ""),
                    "status": e.get("status", "failed"),
                    "kind": e.get("kind", "failed"),
                    "attempt": int(e.get("attempt", 1)),
                }
            )
        for slot, items in by_slot.items():
            # De-dupe by id (index can theoretically list the same id twice)
            # then sort chronologically within the slot.
            seen: set[str] = set()
            unique: list[dict[str, object]] = []
            for item in items:
                cid = str(item.get("id") or "")
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                unique.append(item)
            def _attempt_key(x: dict[str, object]) -> int:
                raw = x.get("attempt", 1)
                if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                    return 1
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return 1

            unique.sort(key=_attempt_key)
            by_slot[slot] = unique
        return dict(by_slot)

    def list_failed_in_pool(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> list[FailedChallengeRecord]:
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        out: list[FailedChallengeRecord] = []
        for entry in self.index.get("failed_pools", {}).get(key, []):
            cid = entry.get("id") if isinstance(entry, dict) else entry
            if not cid:
                continue
            try:
                out.append(self.get_failed(cid))
            except (ChallengeNotFoundError, json.JSONDecodeError):
                continue
        return out

    def get(self, challenge_id: str) -> ChallengeRecord:
        path = self.challenges_dir / f"{challenge_id}.json"
        if not path.exists():
            raise ChallengeNotFoundError(challenge_id)
        data = json.loads(path.read_text())
        return _deserialise_record(data)

    def delete(self, challenge_id: str) -> None:
        path = self.challenges_dir / f"{challenge_id}.json"
        if path.exists():
            path.unlink()
        html_path = self.challenges_dir / f"{challenge_id}.html"
        if html_path.exists():
            html_path.unlink()
        with self._lock:
            for key, ids in list(self.index["pools"].items()):
                if challenge_id in ids:
                    ids.remove(challenge_id)
                    if not ids:
                        del self.index["pools"][key]
            self._save_index()

    # ── Queries ─────────────────────────────────────────

    def _all_pool_keys(self) -> list[str]:
        return list(self.index.get("pools", {}).keys())

    def query(
        self,
        red_model_id: str | None = None,
        repo_name: str | None = None,
        red_harness_id: str | None = None,
        red_reasoning_effort: str | None = None,
        red_provider: str | None = None,
    ) -> list[ChallengeRecord]:
        records: list[ChallengeRecord] = []
        # Snapshot the (key, ids) pairs under the lock so a concurrent writer
        # cannot mutate the dict mid-iteration; file reads happen lock-free.
        with self._lock:
            pool_items = [
                (key, list(ids))
                for key, ids in self.index.get("pools", {}).items()
            ]
        for key, ids in pool_items:
            k_model, k_harness, k_effort, k_provider, k_repo = _split_pool_key(key)
            if red_model_id is not None and k_model != red_model_id:
                continue
            if red_harness_id is not None and k_harness != red_harness_id:
                continue
            if red_reasoning_effort is not None and k_effort != red_reasoning_effort:
                continue
            if red_provider is not None and k_provider != red_provider:
                continue
            if repo_name is not None and k_repo != repo_name:
                continue
            for cid in ids:
                try:
                    records.append(self.get(cid))
                except ChallengeNotFoundError:
                    continue
        return records

    def list_ids_in_index_order(
        self,
        red_model_id: str,
        repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> list[str]:
        """Return challenge IDs for (red_model, harness, effort, provider, repo)
        in index.json order.

        Does not load challenge bodies — cheap for reuse / cache bookkeeping.
        """
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        with self._lock:
            return list(self.index.get("pools", {}).get(key, []))

    def list_in_index_order(
        self,
        red_model_id: str,
        repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> list[ChallengeRecord]:
        """Return all challenges for (red_model, harness, effort, provider,
        repo) in the order they appear under the pool key in index.json.
        Missing/unparseable challenge files are skipped silently."""
        out: list[ChallengeRecord] = []
        for cid in self.list_ids_in_index_order(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        ):
            try:
                out.append(self.get(cid))
            except ChallengeNotFoundError:
                continue
        return out

    def count_in_pool(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> int:
        """Cheap count of challenge IDs in the pool (does not load files)."""
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        with self._lock:
            return len(self.index.get("pools", {}).get(key, []))

    def list_pools(self) -> list[tuple[str, str, str, str, str, int]]:
        """Return (red_model_id, red_harness_id, red_reasoning_effort,
        red_provider, repo_name, count) per pool."""
        out: list[tuple[str, str, str, str, str, int]] = []
        for key, ids in self.index.get("pools", {}).items():
            model, harness, effort, provider, repo = _split_pool_key(key)
            out.append((model, harness, effort, provider, repo, len(ids)))
        return out

    def has_sufficient_pool(
        self, red_model_id: str, repo_name: str, required: int,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> bool:
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        return len(self.index.get("pools", {}).get(key, [])) >= required

    def sample(
        self,
        red_model_id: str,
        repo_name: str,
        n: int,
        seed: int | None = None,
        exclude_ids: set[str] | None = None,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> list[ChallengeRecord]:
        key = _pool_key(
            red_model_id, repo_name, red_harness_id,
            red_reasoning_effort, red_provider,
        )
        candidate_ids = [
            cid
            for cid in self.index.get("pools", {}).get(key, [])
            if not exclude_ids or cid not in exclude_ids
        ]
        if len(candidate_ids) < n:
            raise InsufficientChallengesError(
                f"Pool {key!r} has {len(candidate_ids)} challenges available "
                f"(exclusions applied), requested {n}"
            )

        records = [self.get(cid) for cid in candidate_ids]
        rng = random.Random(seed)

        # Bucket by the tuple of target_files so we can round-robin for diversity.
        buckets: dict[tuple[str, ...], list[ChallengeRecord]] = defaultdict(list)
        for r in records:
            buckets[tuple(sorted(r.target_files))].append(r)

        bucket_keys = list(buckets.keys())
        rng.shuffle(bucket_keys)
        for k in bucket_keys:
            rng.shuffle(buckets[k])

        picked: list[ChallengeRecord] = []
        while len(picked) < n:
            progressed = False
            for k in bucket_keys:
                if not buckets[k]:
                    continue
                picked.append(buckets[k].pop())
                progressed = True
                if len(picked) >= n:
                    break
            if not progressed:
                break
        return picked

    def pool_stats(
        self, red_model_id: str, repo_name: str,
        red_harness_id: str = "mini-swe-agent",
        red_reasoning_effort: str = "",
        red_provider: str = "",
    ) -> ChallengePoolStats:
        records = self.query(
            red_model_id=red_model_id,
            repo_name=repo_name,
            red_harness_id=red_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )
        target_dist: Counter[str] = Counter()
        bug_dist: Counter[str] = Counter()
        total_cost = 0.0
        total_retries = 0
        for r in records:
            for tf in r.target_files:
                target_dist[tf] += 1
            bt = r.challenge.bug_type
            if bt is not None:
                bug_dist[bt] += 1
            total_cost += r.generation_cost_usd
            total_retries += r.generation_retries
        avg_retries = total_retries / len(records) if records else 0.0
        return ChallengePoolStats(
            red_model_id=red_model_id,
            repo_name=repo_name,
            total_challenges=len(records),
            target_file_distribution=dict(target_dist),
            bug_type_distribution=dict(bug_dist),
            total_generation_cost_usd=total_cost,
            avg_retries=avg_retries,
            failed_challenges=self.count_failed_in_pool(
                red_model_id, repo_name, red_harness_id,
                red_reasoning_effort, red_provider,
            ),
            red_harness_id=red_harness_id,
            red_reasoning_effort=red_reasoning_effort,
            red_provider=red_provider,
        )


def _render_record_html(record: ChallengeRecord) -> str:
    ch = record.challenge
    bug_type = ch.bug_type
    feature_steps = (
        list(ch.feature_trajectory.steps) if ch.feature_trajectory else None
    )
    bug_steps = list(ch.bug_trajectory.steps) if ch.bug_trajectory else None
    self_review = record.validation.self_review if record.validation else None
    return render_challenge_html(
        challenge_id=record.challenge_id,
        red_model_id=record.red_model_id,
        repo_name=record.repo_name,
        target_files=list(record.target_files),
        feature_spec=ch.feature_spec,
        feature_rationale=ch.feature_rationale,
        bug_type=bug_type,
        bug_description=ch.bug_description,
        bug_location=ch.bug_location,
        original_file_contents=dict(ch.original_file_contents),
        modified_file_contents=dict(ch.modified_file_contents),
        feature_test_code=ch.feature_test_code,
        bug_test_code=ch.bug_test_code,
        feature_only_file_contents=dict(ch.feature_only_file_contents or {}),
        feature_trajectory_steps=feature_steps,
        bug_trajectory_steps=bug_steps,
        self_review=self_review,
    )
