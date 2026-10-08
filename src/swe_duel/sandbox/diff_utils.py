"""Diff parsing, application, and generation utilities."""

from __future__ import annotations

import difflib
import html
import re
from collections.abc import Iterator
from pathlib import Path

from unidiff import PatchSet, UnidiffParseError

from swe_duel.models import DiffResult, DiffStats


def apply_diff(original: str, diff: str) -> DiffResult:
    """Apply unified diff to original content string."""
    if not diff.strip():
        return DiffResult(success=False, patched="", error="Empty diff")
    try:
        patch = PatchSet(diff)
    except UnidiffParseError as e:
        return DiffResult(success=False, patched="", error=f"Parse error: {e}")

    if not patch:
        return DiffResult(success=False, patched="", error="No patches found")

    lines = original.splitlines(keepends=True)
    patched_lines = list(lines)

    for patched_file in patch:
        offset = 0
        for hunk in patched_file:
            source_start = hunk.source_start - 1 + offset
            removes = [line.value for line in hunk if line.is_removed]
            adds = [line.value for line in hunk if line.is_added]

            # Verify the lines we're about to remove match
            chunk = patched_lines[source_start : source_start + len(removes)]
            if list(chunk) != removes:
                # Try to find the correct position
                found = False
                for i in range(max(0, source_start - 5), min(len(patched_lines), source_start + 10)):
                    if patched_lines[i : i + len(removes)] == removes:
                        source_start = i
                        found = True
                        break
                if not found:
                    return DiffResult(success=False, patched="", error="Hunk context mismatch")

            patched_lines[source_start : source_start + len(removes)] = adds
            offset += len(adds) - len(removes)

    return DiffResult(success=True, patched="".join(patched_lines), error="")


def validate_diff_format(diff: str) -> bool:
    """Return True if diff is a parseable unified diff."""
    if not diff.strip():
        return False
    try:
        patch = PatchSet(diff)
        return len(patch) > 0
    except UnidiffParseError:
        return False


def compute_diff_stats(diff: str) -> DiffStats:
    """Count lines added, removed, hunks, files changed from a unified diff."""
    lines_added = 0
    lines_removed = 0
    hunks = 0
    files_changed = 0

    for line in diff.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+"):
            lines_added += 1
        elif line.startswith("-"):
            lines_removed += 1

    # Count file pairs (each +++ line that isn't /dev/null)
    files_changed = sum(
        1 for line in diff.splitlines()
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null")
    )

    return DiffStats(
        lines_added=lines_added,
        lines_removed=lines_removed,
        hunks=hunks,
        files_changed=files_changed,
    )


def normalize_diff(diff: str) -> str:
    """Normalise ---/+++ headers, strip timestamps."""
    normalized = []
    for line in diff.splitlines(keepends=True):
        # Strip timestamps from --- and +++ headers (e.g. "--- a/file.py\t2024-01-01 ...")
        if line.startswith("--- ") or line.startswith("+++ "):
            line = re.sub(r"\t\d{4}-\d{2}-\d{2}.*", "", line)
            # Normalise a/ b/ prefixes: strip them if present
            line = re.sub(r"^(---|\+\+\+) [ab]/", r"\1 ", line)
        normalized.append(line)
    return "".join(normalized)


def generate_diff(original: str, modified: str, filename: str) -> str:
    """Generate unified diff from two file content strings."""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )
    return "".join(diff_lines)


def render_html_diff(
    original_files: dict[str, str],
    modified_files: dict[str, str],
    title: str,
    annotation_html: str = "",
    trailing_html: str = "",
) -> str:
    """Render a multi-file side-by-side HTML diff.

    Files that appear in either dict are compared; paths missing from one side
    are treated as empty. `annotation_html` is inserted as a banner above the
    diff tables (already-escaped HTML allowed). `trailing_html` is appended
    after the diff tables (also already-escaped HTML). Uses the same `sbs`
    table format and CSS as the challenge-bank HTML.
    """
    body = _render_diff_tables(original_files, modified_files)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CHALLENGE_CSS}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>"
        f"{annotation_html}{body}{trailing_html}</body></html>"
    )


_CHALLENGE_CSS = (
    "body{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;"
    "margin:20px;background:#fff;color:#111}"
    "h1{font-size:20px}h2{font-size:16px;margin-top:28px;padding-bottom:4px;"
    "border-bottom:2px solid #ccc}"
    "h3{margin-top:18px;font-size:13px;color:#333}"
    "table.sbs{border-collapse:collapse;width:100%;table-layout:fixed;"
    "border:1px solid #d0d7de;margin-top:6px}"
    "table.sbs thead th{background:#f6f8fa;border-bottom:1px solid #d0d7de;"
    "text-align:left;padding:6px 8px;font-weight:600;font-size:11px;color:#555}"
    "table.sbs td{vertical-align:top;padding:1px 8px;white-space:pre-wrap;"
    "word-break:break-word;width:50%;border-right:1px solid #eaecef;"
    "font-family:inherit;line-height:1.45}"
    "table.sbs td:last-child{border-right:none}"
    "table.sbs tr.eq td{background:#fff}"
    "table.sbs tr.del td.before{background:#ffd6d6}"
    "table.sbs tr.ins td.after{background:#d4fcdc}"
    "table.sbs tr.chg td.before{background:#ffeef0}"
    "table.sbs tr.chg td.after{background:#e6ffed}"
    "table.sbs tr.skip td{background:#f6f8fa;color:#888;text-align:center;"
    "font-style:italic;padding:4px 8px}"
    ".banner{background:#fff3cd;border:1px solid #ffe69c;padding:10px 14px;"
    "border-radius:6px;margin:12px 0;white-space:pre-wrap}"
    ".banner.bug{background:#fde2e4;border-color:#f5a7ad}"
    ".banner.feature{background:#d1f0df;border-color:#8fcfae}"
    ".meta{color:#555;font-size:11px;margin-bottom:12px}"
    "pre.code{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;"
    "padding:12px;overflow-x:auto;white-space:pre;font-size:12px;line-height:1.4}"
    "nav{position:sticky;top:0;background:#fff;padding:8px 0;"
    "border-bottom:1px solid #eee;margin-bottom:16px}"
    "nav a{margin-right:14px;color:#0366d6;text-decoration:none}"
    "nav a:hover{text-decoration:underline}"
    ".step{border:1px solid #d0d7de;border-radius:6px;margin:10px 0;"
    "padding:10px 12px;background:#fafbfc}"
    ".step-hdr{font-weight:600;color:#333;font-size:12px;margin-bottom:6px;"
    "padding-bottom:4px;border-bottom:1px solid #eaecef}"
    ".step-stats{font-size:11px;color:#444;background:#eef2f6;border:1px solid #d0d7de;"
    "border-radius:4px;padding:4px 8px;margin:0 0 6px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}"
    ".step-part{margin-top:6px}"
    ".step-part .label{font-size:11px;color:#666;text-transform:uppercase;"
    "letter-spacing:0.5px;margin-right:6px}"
    ".step-part pre{margin:4px 0 0 0;background:#fff;border:1px solid #e1e4e8;"
    "border-radius:4px;padding:8px;white-space:pre-wrap;word-break:break-word;"
    "font-size:11px;max-height:240px;overflow:auto}"
    ".traj-empty{color:#888;font-style:italic}"
    "ul.findings{margin:6px 0;padding-left:20px;font-size:12px}"
    # Post-Blue test-run table (mirrors the challenge HTML's gate table).
    "table.gates{border-collapse:collapse;width:100%;border:1px solid #d0d7de;"
    "margin-top:6px}"
    "table.gates th,table.gates td{border:1px solid #e1e4e8;padding:6px 8px;"
    "text-align:left;vertical-align:top}"
    "table.gates thead th{background:#f6f8fa;font-size:11px;color:#555}"
    "table.gates tr.pass td.gate-status{color:#1a7f37;font-weight:600}"
    "table.gates tr.fail td.gate-status{color:#cf222e;font-weight:600}"
    "table.gates td.mono{font-family:ui-monospace,Menlo,monospace;color:#444}"
    "details.gate-detail{margin-top:6px}"
    "details.gate-detail summary{cursor:pointer;color:#0366d6}"
    "pre.gate-out{background:#0d1117;color:#e6edf3;border-radius:4px;padding:8px;"
    "white-space:pre-wrap;word-break:break-word;font-size:11px;max-height:320px;"
    "overflow:auto;margin:4px 0}"
)


def _render_trajectory_steps(
    steps: list[dict] | None, heading: str, anchor: str
) -> str:
    """Render an agent trajectory's steps (thought/action/observation) as HTML."""
    steps = steps or []
    if not steps:
        return (
            f"<h3 id='{anchor}'>{html.escape(heading)}</h3>"
            "<p class='traj-empty'>No steps recorded.</p>"
        )
    blocks: list[str] = []
    cum_in = 0
    cum_out = 0
    cum_cached = 0
    cum_cost = 0.0
    cum_time = 0.0
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        thought = str(step.get("thought") or "")
        action = str(step.get("action") or "")
        observation = str(step.get("observation") or "")
        in_tok = int(step.get("input_tokens", 0) or 0)
        out_tok = int(step.get("output_tokens", 0) or 0)
        cached_tok = int(step.get("cached_tokens", 0) or 0)
        step_cost = float(step.get("cost_usd", 0.0) or 0.0)
        step_time = float(step.get("step_seconds", 0.0) or 0.0)
        # Prefer recorded cum_seconds when available (more accurate than summing
        # step_seconds, which can drift if some steps weren't timed).
        recorded_cum = float(step.get("cum_seconds", 0.0) or 0.0)
        cum_in += in_tok
        cum_out += out_tok
        cum_cached += cached_tok
        cum_cost += step_cost
        cum_time = recorded_cum if recorded_cum > 0.0 else cum_time + step_time
        stats_html = (
            "<div class='step-stats'>"
            f"<span>step tokens: in={in_tok:,} out={out_tok:,} cached={cached_tok:,}</span>"
            f" &nbsp;|&nbsp; <span>step cost: ${step_cost:.4f}</span>"
            f" &nbsp;|&nbsp; <span>step time: {step_time:.2f}s</span>"
            f" &nbsp;|&nbsp; <span>cum tokens: in={cum_in:,} out={cum_out:,} cached={cum_cached:,}</span>"
            f" &nbsp;|&nbsp; <span>cum cost: ${cum_cost:.4f}</span>"
            f" &nbsp;|&nbsp; <span>cum time: {cum_time:.2f}s</span>"
            "</div>"
        )
        parts: list[str] = [
            f"<div class='step-hdr'>Step {i}</div>",
            stats_html,
            # Always emit Observation / Thought / Action so the three fields
            # stay aligned with mini-swe-agent rendering even when empty
            # (empty means the harness failed to capture that channel, not that
            # the section should be omitted — that was hiding missing thoughts).
            "<div class='step-part'><span class='label'>Observation</span>"
            f"<pre>{html.escape(observation) if observation else '(empty)'}</pre></div>",
            "<div class='step-part'><span class='label'>Thought</span>"
            f"<pre>{html.escape(thought) if thought else '(empty)'}</pre></div>",
            "<div class='step-part'><span class='label'>Action</span>"
            f"<pre>{html.escape(action) if action else '(empty)'}</pre></div>",
        ]
        blocks.append(f"<div class='step'>{''.join(parts)}</div>")
    return f"<h3 id='{anchor}'>{html.escape(heading)}</h3>{''.join(blocks)}"


def _render_two_col_table(
    a_lines: list[str], b_lines: list[str], rel: str, context: int = 3
) -> str:
    """Render a two-column (before | after) diff table with context folding."""
    matcher = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    rows: list[tuple[str, str, str]] = []  # (cls, left, right)

    opcodes = matcher.get_opcodes()
    # Build a flat list of aligned rows with a per-row class.
    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            n = i2 - i1
            # Collapse long equal runs, keeping `context` lines near changes.
            keep_head = context if idx != 0 else 0
            keep_tail = context if idx != len(opcodes) - 1 else 0
            if n <= keep_head + keep_tail:
                for k in range(n):
                    rows.append(("eq", a_lines[i1 + k], b_lines[j1 + k]))
            else:
                for k in range(keep_head):
                    rows.append(("eq", a_lines[i1 + k], b_lines[j1 + k]))
                skipped = n - keep_head - keep_tail
                rows.append(("skip", f"… {skipped} unchanged lines …", ""))
                for k in range(keep_tail):
                    off = n - keep_tail + k
                    rows.append(("eq", a_lines[i1 + off], b_lines[j1 + off]))
        elif tag == "delete":
            for k in range(i2 - i1):
                rows.append(("del", a_lines[i1 + k], ""))
        elif tag == "insert":
            for k in range(j2 - j1):
                rows.append(("ins", "", b_lines[j1 + k]))
        elif tag == "replace":
            left = a_lines[i1:i2]
            right = b_lines[j1:j2]
            n = max(len(left), len(right))
            for k in range(n):
                l_line = left[k] if k < len(left) else ""
                r = right[k] if k < len(right) else ""
                rows.append(("chg", l_line, r))

    trs: list[str] = []
    for cls, left, right in rows:
        if cls == "skip":
            trs.append(
                f"<tr class='skip'><td colspan='2'>{html.escape(left)}</td></tr>"
            )
            continue
        trs.append(
            f"<tr class='{cls}'>"
            f"<td class='before'>{html.escape(left)}</td>"
            f"<td class='after'>{html.escape(right)}</td>"
            f"</tr>"
        )

    return (
        f"<h3>{html.escape(rel)}</h3>"
        "<table class='sbs'>"
        f"<thead><tr><th>before — a/{html.escape(rel)}</th>"
        f"<th>after — b/{html.escape(rel)}</th></tr></thead>"
        f"<tbody>{''.join(trs)}</tbody>"
        "</table>"
    )


def _render_diff_tables(
    original_files: dict[str, str], modified_files: dict[str, str]
) -> str:
    all_paths = sorted(set(original_files) | set(modified_files))
    tables: list[str] = []
    for rel in all_paths:
        a = (original_files.get(rel) or "").splitlines()
        b = (modified_files.get(rel) or "").splitlines()
        if a == b:
            continue
        tables.append(_render_two_col_table(a, b, rel))
    return "\n<hr/>\n".join(tables) if tables else "<p><em>No changes.</em></p>"


def render_challenge_html(
    *,
    challenge_id: str,
    red_model_id: str,
    repo_name: str,
    target_files: list[str],
    feature_spec: str,
    feature_rationale: str,
    bug_type: str | None,
    bug_description: str | None,
    bug_location: str | None,
    original_file_contents: dict[str, str],
    modified_file_contents: dict[str, str],
    feature_test_code: str,
    bug_test_code: str | None,
    feature_only_file_contents: dict[str, str] | None = None,
    feature_trajectory_steps: list[dict] | None = None,
    bug_trajectory_steps: list[dict] | None = None,
    self_review: object | None = None,
) -> str:
    """Render a consolidated challenge visualization with four sections:
    new feature, embedded bug, feature tests (fixed-feature contract), and
    bug-detection tests.
    """
    feature_only = feature_only_file_contents or {}
    # If we have a pre-bug snapshot, show feature diff against the original
    # (pre-feature) and the bug diff against the pre-bug feature code.
    if feature_only:
        feature_tables = _render_diff_tables(original_file_contents, feature_only)
        bug_diff_tables = _render_diff_tables(feature_only, modified_file_contents)
    else:
        feature_tables = _render_diff_tables(
            original_file_contents, modified_file_contents
        )
        bug_diff_tables = feature_tables

    feature_banner = (
        "<div class='banner feature'>"
        f"<b>Feature:</b> {html.escape(feature_spec or '')}<br/><br/>"
        f"<b>Rationale:</b> {html.escape(feature_rationale or '')}"
        "</div>"
    )
    bug_banner = (
        "<div class='banner bug'>"
        f"<b>Bug type:</b> {html.escape(bug_type or 'unknown')}<br/><br/>"
        f"<b>Description:</b> {html.escape(bug_description or '')}<br/><br/>"
        f"<b>Location:</b> {html.escape(bug_location or '')}"
        "</div>"
    )

    feature_tests = (
        f"<pre class='code'>{html.escape(feature_test_code or '')}</pre>"
        if feature_test_code
        else "<p><em>No feature tests provided.</em></p>"
    )
    bug_tests = (
        f"<pre class='code'>{html.escape(bug_test_code or '')}</pre>"
        if bug_test_code
        else "<p><em>No bug-detection tests provided.</em></p>"
    )

    target_files_html = ", ".join(html.escape(tf) for tf in target_files) or "—"
    meta = (
        f"<div class='meta'>"
        f"<b>challenge_id:</b> {html.escape(challenge_id)} &nbsp;|&nbsp; "
        f"<b>red_model:</b> {html.escape(red_model_id)} &nbsp;|&nbsp; "
        f"<b>repo:</b> {html.escape(repo_name)} &nbsp;|&nbsp; "
        f"<b>target_files:</b> {target_files_html}"
        f"</div>"
    )

    nav = (
        "<nav>"
        "<a href='#feature'>New feature</a>"
        "<a href='#bug'>Embedded bug</a>"
        "<a href='#feature-tests'>Fixed-feature tests</a>"
        "<a href='#bug-tests'>Bug-detection tests</a>"
        "<a href='#traj-feature'>Red reasoning — feature</a>"
        "<a href='#traj-bug'>Red reasoning — bug</a>"
        "<a href='#self-review'>Self-review audit</a>"
        "</nav>"
    )

    feature_traj_html = _render_trajectory_steps(
        feature_trajectory_steps,
        "Red agent reasoning — feature generation phase",
        "traj-feature",
    )
    bug_traj_html = _render_trajectory_steps(
        bug_trajectory_steps,
        "Red agent reasoning — bug embedding phase",
        "traj-bug",
    )

    self_review_html = _render_self_review_section(self_review)

    bug_section_note = (
        "<p><em>Diff: pre-bug feature code → bugged code. Shows exactly the "
        "bug edits, isolated from the feature.</em></p>"
        if feature_only
        else "<p><em>Same diff as above — bug is intentionally woven into the "
        "feature. Banner above names the location.</em></p>"
    )

    title = f"Challenge {challenge_id} — {repo_name}"
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CHALLENGE_CSS}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{meta}{nav}"
        f"<h2 id='feature'>1. New feature</h2>{feature_banner}{feature_tables}"
        f"<h2 id='bug'>2. Embedded bug</h2>{bug_banner}"
        f"{bug_section_note}{bug_diff_tables}"
        f"<h2 id='feature-tests'>3. Fixed-feature tests</h2>"
        "<p><em>These must pass on Blue's fix to prove the feature survived.</em></p>"
        f"{feature_tests}"
        f"<h2 id='bug-tests'>4. Bug-detection tests</h2>"
        "<p><em>These fail on Red's code (bug present) and must pass on Blue's fix "
        "(bug removed).</em></p>"
        f"{bug_tests}"
        f"<h2 id='reasoning'>5. Red agent reasoning</h2>"
        f"{feature_traj_html}{bug_traj_html}"
        f"{self_review_html}"
        f"</body></html>"
    )


def _render_self_review_section(self_review: object | None) -> str:
    """Render the self-review audit section for a successful challenge HTML."""
    if self_review is None:
        return (
            "<h2 id='self-review'>6. Self-review audit</h2>"
            "<p><em>No self-review data available.</em></p>"
        )

    detected = bool(getattr(self_review, "detected", False))
    detection_reason = str(getattr(self_review, "detection_reason", "") or "")
    findings = getattr(self_review, "findings", None) or []
    fix_explanation = str(getattr(self_review, "fix_explanation", "") or "")
    agent_trajectory = getattr(self_review, "agent_trajectory", None)

    verdict_color = "#389e0d" if detected else "#c41d7f"
    verdict_text = (
        "FIXED — bug tests pass on reviewer's code (challenge solvable)"
        if detected
        else "NOT fixed — reviewer could not remove the bug (rejected)"
    )
    verdict_html = (
        f"<p><b style='color:{verdict_color}'>{html.escape(verdict_text)}</b></p>"
    )

    reason_html = (
        f"<p class='meta'>{html.escape(detection_reason)}</p>"
        if detection_reason
        else ""
    )

    if findings:
        items = "".join(
            f"<li><b>{html.escape(getattr(f, 'severity', ''))}</b> @ "
            f"{html.escape(getattr(f, 'location', ''))}: "
            f"{html.escape(getattr(f, 'description', ''))}</li>"
            for f in findings
        )
        findings_html = f"<ul class='findings'>{items}</ul>"
    else:
        findings_html = "<p><em>No findings reported.</em></p>"

    fix_html = (
        f"<p><b>fix_explanation:</b> {html.escape(fix_explanation)}</p>"
        if fix_explanation
        else ""
    )

    traj_steps = (
        list(agent_trajectory.steps)
        if agent_trajectory is not None and hasattr(agent_trajectory, "steps")
        else None
    )
    traj_html = _render_trajectory_steps(
        traj_steps,
        "Self-review agent reasoning — emulated Blue (no bug knowledge)",
        "traj-self-review",
    )

    return (
        "<h2 id='self-review'>6. Self-review audit</h2>"
        "<p><em>The Red model reviewed its own PR without bug knowledge. "
        "A genuine bug must be identifiable by a same-skill reviewer.</em></p>"
        f"{verdict_html}{reason_html}"
        "<h3>Review findings</h3>"
        f"{findings_html}{fix_html}"
        f"{traj_html}"
    )


def render_defense_html(
    *,
    defense_id: str,
    challenge_id: str,
    red_model_id: str,
    blue_model_id: str,
    repo_name: str,
    target_files: list[str],
    feature_spec: str,
    bug_type: str | None,
    bug_description: str | None,
    bug_location: str | None,
    review_findings: list[tuple[str, str, str]],
    fix_explanation: str,
    original_file_contents: dict[str, str],
    red_file_contents: dict[str, str],
    blue_file_contents: dict[str, str],
    s_regression: float,
    s_feature: float,
    s_bugfix: float,
    blue_composite: float,
    blue_trajectory_steps: list[dict] | None = None,
    test_details: dict | None = None,
) -> str:
    """Render a defense result visualization with four sections: detected bug
    (original → Red's buggy code, annotated with Blue's findings), bug removal
    (Red → Blue), feature retention (original → Blue, showing the feature without
    the bug), and test scores.

    ``test_details`` (optional) is ``TurnScore.test_details`` — a dict keyed by
    "regression"/"feature"/"bugfix", each a serialized ``TestExecutionResult``
    (command + stdout/stderr + counts). When present, the post-Blue scoring
    commands and console output are rendered in collapsible blocks, mirroring
    how Red's validation-gate output is shown in the challenge HTML — so a
    defense can be debugged from its command + captured output.
    """
    bug_tables = _render_diff_tables(original_file_contents, red_file_contents)
    fix_tables = _render_diff_tables(red_file_contents, blue_file_contents)
    retain_tables = _render_diff_tables(original_file_contents, blue_file_contents)

    if review_findings:
        findings_html = "<ul>" + "".join(
            f"<li><b>{html.escape(sev)}</b> @ {html.escape(loc)}: "
            f"{html.escape(desc)}</li>"
            for loc, sev, desc in review_findings
        ) + "</ul>"
    else:
        findings_html = "<p><em>No findings reported.</em></p>"

    detected_banner = (
        "<div class='banner bug'>"
        f"<b>Bug type:</b> {html.escape(bug_type or 'unknown')}<br/>"
        f"<b>Description:</b> {html.escape(bug_description or '')}<br/>"
        f"<b>Location:</b> {html.escape(bug_location or '')}<br/>"
        f"<b>Blue review findings:</b>{findings_html}"
        "</div>"
    )

    fix_banner = (
        "<div class='banner'>"
        f"<b>Fix explanation:</b> {html.escape(fix_explanation or '')}"
        "</div>"
    )

    feature_banner = (
        "<div class='banner feature'>"
        f"<b>Feature (should survive):</b> {html.escape(feature_spec or '')}"
        "</div>"
    )

    def _pill(label: str, val: float) -> str:
        color = "#d4fcdc" if val >= 1.0 else "#ffd6d6"
        return (
            f"<span style='display:inline-block;padding:4px 10px;margin-right:8px;"
            f"border-radius:4px;background:{color};border:1px solid #bbb'>"
            f"<b>{html.escape(label)}:</b> {val:.2f}</span>"
        )

    score_banner = (
        "<div class='banner'>"
        f"{_pill('s_regression', s_regression)}"
        f"{_pill('s_feature', s_feature)}"
        f"{_pill('s_bugfix', s_bugfix)}"
        f"{_pill('blue_composite', blue_composite)}"
        "</div>"
    )

    target_files_html = ", ".join(html.escape(tf) for tf in target_files) or "—"
    meta = (
        f"<div class='meta'>"
        f"<b>defense_id:</b> {html.escape(defense_id)} &nbsp;|&nbsp; "
        f"<b>challenge_id:</b> {html.escape(challenge_id)} &nbsp;|&nbsp; "
        f"<b>red_model:</b> {html.escape(red_model_id)} &nbsp;|&nbsp; "
        f"<b>blue_model:</b> {html.escape(blue_model_id)} &nbsp;|&nbsp; "
        f"<b>repo:</b> {html.escape(repo_name)} &nbsp;|&nbsp; "
        f"<b>target_files:</b> {target_files_html}"
        f"</div>"
    )

    nav = (
        "<nav>"
        "<a href='#detected'>Detected bug</a>"
        "<a href='#fix'>Bug fix</a>"
        "<a href='#retain'>Feature retained</a>"
        "<a href='#score'>Score</a>"
        "<a href='#tests'>Test runs</a>"
        "<a href='#traj-blue'>Blue reasoning</a>"
        "</nav>"
    )

    tests_html = _render_defense_test_details(test_details)

    blue_traj_html = _render_trajectory_steps(
        blue_trajectory_steps,
        "Blue agent reasoning — bug detection and fixing",
        "traj-blue",
    )

    title = f"Defense {defense_id} — {repo_name}"
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CHALLENGE_CSS}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{meta}{nav}"
        f"<h2 id='detected'>1. Detected bug</h2>"
        "<p><em>Diff: original → Red's buggy code (what Blue had to review).</em></p>"
        f"{detected_banner}{bug_tables}"
        f"<h2 id='fix'>2. Bug fix</h2>"
        "<p><em>Diff: Red's buggy code → Blue's fixed code.</em></p>"
        f"{fix_banner}{fix_tables}"
        f"<h2 id='retain'>3. Feature retained</h2>"
        "<p><em>Diff: original → Blue's fixed code. Should show the feature "
        "without the bug.</em></p>"
        f"{feature_banner}{retain_tables}"
        f"<h2 id='score'>4. Score</h2>{score_banner}"
        f"<h2 id='tests'>5. Post-Blue test runs</h2>"
        "<p><em>Commands and console output of the regression / feature / bug "
        "test suites executed against Blue's fixed code (inside the repo Docker "
        "image). These determine the scores above.</em></p>"
        f"{tests_html}"
        f"<h2 id='reasoning'>6. Blue agent reasoning</h2>{blue_traj_html}"
        f"</body></html>"
    )


# Suites in display order, with the score field each one decides.
_DEFENSE_TEST_SUITES = (
    ("regression", "regression tests (existing suite — s_regression)"),
    ("feature", "feature tests (s_feature)"),
    ("bugfix", "bug tests (s_bugfix — pass ⇒ bug removed)"),
)


def _render_defense_test_details(test_details: dict | None) -> str:
    """Render each post-Blue test suite's command + stdout/stderr in a
    collapsible block, mirroring the per-gate detail blocks in the challenge
    HTML. A suite absent from ``test_details`` (e.g. feature/bug skipped because
    regression already failed, or an empty-fix defense) is shown as not run.
    """
    if not test_details:
        return (
            "<p><em>No test details recorded (e.g. Blue made no changes, so "
            "scoring short-circuited to a total loss).</em></p>"
        )
    rows = []
    for key, label in _DEFENSE_TEST_SUITES:
        d = test_details.get(key)
        if not isinstance(d, dict):
            rows.append(
                f"<tr class='fail'><td><code>{html.escape(key)}</code></td>"
                f"<td class='gate-status'>not run</td><td>{html.escape(label)}</td>"
                f"<td class='mono'>—</td></tr>"
            )
            continue
        passed = bool(d.get("passed"))
        cls = "pass" if passed else "fail"
        status = "passed" if passed else "failed"
        total = int(d.get("total", 0) or 0)
        pc = int(d.get("passed_count", 0) or 0)
        fc = int(d.get("failed_count", 0) or 0)
        ec = int(d.get("error_count", 0) or 0)
        msg = f"{html.escape(label)} — {pc}/{total} passed, {fc} failed, {ec} errors"
        command = str(d.get("command", "") or "")
        stdout = str(d.get("stdout", "") or "")
        stderr = str(d.get("stderr", "") or "")
        inner_parts = []
        if command.strip():
            inner_parts.append(
                f"<b>command:</b><pre class='gate-out'>{html.escape(command)}</pre>"
            )
        if stdout.strip():
            inner_parts.append(
                f"<b>stdout:</b><pre class='gate-out'>{html.escape(stdout[:8000])}"
                + ("…" if len(stdout) > 8000 else "")
                + "</pre>"
            )
        if stderr.strip():
            inner_parts.append(
                f"<b>stderr:</b><pre class='gate-out'>{html.escape(stderr[:4000])}"
                + ("…" if len(stderr) > 4000 else "")
                + "</pre>"
            )
        detail = ""
        if inner_parts:
            detail = (
                "<details class='gate-detail'><summary>Show command &amp; output"
                "</summary>" + "".join(inner_parts) + "</details>"
            )
        dur_ms = int(d.get("duration_ms", 0) or 0)
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><code>{html.escape(key)}</code></td>"
            f"<td class='gate-status'>{html.escape(status)}</td>"
            f"<td>{msg}{detail}</td>"
            f"<td class='mono'>{dur_ms:,}ms</td>"
            "</tr>"
        )
    return (
        "<table class='gates'>"
        "<thead><tr><th>Suite</th><th>Status</th><th>Detail</th><th>Time</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _is_probably_binary(path: Path) -> bool:
    """Heuristic: a file is binary if its first 8 KiB contain a NUL byte."""
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return False


def iter_files_safe(root: Path) -> "Iterator[Path]":
    """Yield every file under ``root``, skipping unreadable subtrees.

    A plain ``Path.rglob('*')`` raises ``PermissionError`` mid-iteration if any
    directory is unreadable (e.g. a root-owned dir left by a containerised agent
    that ran as root). We walk with ``os.walk(onerror=...)`` so such directories
    are skipped instead of aborting the whole traversal — diffing/snapshotting
    must never crash because one stray dir is unreadable.
    """
    import os as _os

    for dirpath, dirnames, filenames in _os.walk(root, onerror=lambda e: None):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.is_file():
                    yield p
            except OSError:
                continue


def generate_tree_diff(
    dir_a: Path,
    dir_b: Path,
    exclude: list[str] | None = None,
) -> str:
    """Recursively diff two directories, returning combined unified diff."""
    import fnmatch

    exclude = exclude or []
    diffs: list[str] = []

    def is_excluded(rel: str) -> bool:
        for pattern in exclude:
            if any(ch in pattern for ch in "*?["):
                if fnmatch.fnmatch(rel, pattern):
                    return True
            elif rel.startswith(pattern) or ("/" + pattern) in rel:
                return True
        return False

    all_rels: set[str] = set()
    for p in iter_files_safe(dir_a):
        all_rels.add(str(p.relative_to(dir_a)))
    for p in iter_files_safe(dir_b):
        all_rels.add(str(p.relative_to(dir_b)))

    for rel in sorted(all_rels):
        if is_excluded(rel):
            continue

        path_a = dir_a / rel
        path_b = dir_b / rel

        # Skip binaries: reading them as text corrupts the content and can
        # produce a meaningless, enormous diff.
        if (path_a.exists() and _is_probably_binary(path_a)) or (
            path_b.exists() and _is_probably_binary(path_b)
        ):
            continue

        content_a = path_a.read_text(errors="replace") if path_a.exists() else ""
        content_b = path_b.read_text(errors="replace") if path_b.exists() else ""

        if content_a == content_b:
            continue

        diff = generate_diff(content_a, content_b, rel)
        if diff:
            diffs.append(diff)

    return "".join(diffs)
