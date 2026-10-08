#!/usr/bin/env python
"""Build ELO LaTeX table + pairwise matchup PDF matrix from data/matches/*.json.

Deduplicates matches that reuse the exact same defense set (a symptom of the
tournament being re-run while the defense cache was warm): for each unique
(sorted model pair, repo, sorted defense_ids) we keep only the latest match.
Legitimate rematches (same pair but different defenses) are preserved.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swe_duel.engine.swiss import swiss_points_table
from swe_duel.models import (
    MatchOutcome,
    TurnScore,
    composite_id,
    display_composite_id,
    split_composite_id,
)
from swe_duel.scoring.rating import compute_all_ratings

# Set in main() from SWE_DUEL_REPORT_FORMAT. Controls ranking-table labels
# (composite "model [harness]" vs bare harness display name).
_REPORT_FORMAT: str = "swiss"  # swiss | round-robin | harness-ablation-rr


def _is_harness_ablation() -> bool:
    return _REPORT_FORMAT in {
        "harness-ablation-rr",
        "harness_ablation_rr",
        "harness-ablation",
    }


def _participant_label(cid: str) -> str:
    """Human label for a ranking-table participant id.

    Harness-ablation rankings key on bare harness ids (``codex``, …); other
    formats key on composite ``model#harness`` competitor identities.
    """
    if _is_harness_ablation():
        from swe_duel.agents.harness import HARNESS_IDS, harness_display_name

        # Bare harness id, or a composite whose harness half we still show.
        if cid in HARNESS_IDS:
            return harness_display_name(cid)
        _mid, hid, _effort, _provider = split_composite_id(cid)
        if hid in HARNESS_IDS:
            return harness_display_name(hid)
        return cid
    return display_composite_id(cid)


def _participant_col() -> str:
    return "Harness" if _is_harness_ablation() else "Model"


@dataclass
class _ShimDefense:
    score: TurnScore


@dataclass
class _ShimRound:
    red_model_id: str
    blue_model_id: str
    defense_result: _ShimDefense


@dataclass
class _ShimMatch:
    match_id: str
    model_a_id: str
    model_b_id: str
    outcome: MatchOutcome
    turns: list
    model_a_total: float = 0.0
    model_b_total: float = 0.0


def _load_matches(matches_dir: Path) -> list[dict]:
    out = []
    for f in sorted(matches_dir.glob("*.json")):
        with f.open() as fh:
            out.append(json.load(fh))
    return out


def _dedupe(matches: list[dict]) -> tuple[list[dict], list[dict]]:
    """Group by (pair, repo, defense_id set); keep the latest timestamp per group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in matches:
        pair = tuple(sorted([m["model_a_id"], m["model_b_id"]]))
        defense_ids = tuple(sorted(r["defense_id"] for r in m["turns"]))
        groups[(pair, m["repo_name"], defense_ids)].append(m)

    kept, dropped = [], []
    for _, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda x: x["timestamp"])
        kept.append(grp_sorted[-1])
        dropped.extend(grp_sorted[:-1])
    return kept, dropped


def _to_match_result(raw: dict) -> _ShimMatch:
    """Minimal shim for compute_all_ratings — only needs the fields it reads."""
    turns = []
    for r in raw["turns"]:
        s = r["score"]
        score = TurnScore(
            s_regression=s["s_regression"],
            s_feature=s["s_feature"],
            s_bugfix=s["s_bugfix"],
            blue_composite=s["blue_composite"],
            red_composite=s["red_composite"],
            test_details={},
        )
        turns.append(
            _ShimRound(r["red_model_id"], r["blue_model_id"], _ShimDefense(score))
        )
    return _ShimMatch(
        match_id=raw["match_id"],
        model_a_id=raw["model_a_id"],
        model_b_id=raw["model_b_id"],
        outcome=MatchOutcome(raw["outcome"]),
        turns=turns,
        model_a_total=float(raw.get("model_a_total") or 0.0),
        model_b_total=float(raw.get("model_b_total") or 0.0),
    )


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_elo_table(snapshots: list, out_path: Path) -> None:
    rows = sorted(snapshots, key=lambda s: -s.elo)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        rf"{_participant_col()} & ELO & Red ELO & Blue ELO & TrueSkill ($\mu \pm \sigma$) & Bradley-Terry & Matches \\",
        r"\midrule",
    ]
    for s in rows:
        lines.append(
            f"{_latex_escape(_participant_label(s.model_id))} & {s.elo:.1f} & {s.red_elo:.1f} & "
            f"{s.blue_elo:.1f} & ${s.trueskill_mu:.2f} \\pm {s.trueskill_sigma:.2f}$ & "
            f"{getattr(s, 'bradley_terry', 0.0):.3f} & "
            f"{s.matches_played} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_elo_markdown(snapshots: list, out_path: Path) -> None:
    rows = sorted(snapshots, key=lambda s: -s.elo)
    lines = [
        f"| {_participant_col()} | ELO | Red ELO | Blue ELO | TrueSkill (μ ± σ) | Bradley-Terry | Matches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in rows:
        lines.append(
            f"| {_participant_label(s.model_id)} | {s.elo:.1f} | {s.red_elo:.1f} | {s.blue_elo:.1f} | "
            f"{s.trueskill_mu:.2f} ± {s.trueskill_sigma:.2f} | {getattr(s, 'bradley_terry', 0.0):.3f} | "
            f"{s.matches_played} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_elo_pdf(snapshots: list, out_path: Path) -> None:
    rows = sorted(snapshots, key=lambda s: -s.elo)
    headers = [
        _participant_col(),
        "ELO",
        "Red ELO",
        "Blue ELO",
        "TrueSkill (μ ± σ)",
        "Bradley-Terry",
        "Matches",
    ]
    cells = [
        [
            _participant_label(s.model_id),
            f"{s.elo:.1f}",
            f"{s.red_elo:.1f}",
            f"{s.blue_elo:.1f}",
            f"{s.trueskill_mu:.2f} ± {s.trueskill_sigma:.2f}",
            f"{getattr(s, 'bradley_terry', 0.0):.3f}",
            str(s.matches_played),
        ]
        for s in rows
    ]

    n_rows = len(cells) + 1
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * n_rows))
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for j in range(len(headers)):
        table[(0, j)].set_text_props(weight="bold")
    ax.set_title("ELO Ratings", fontsize=11, pad=10)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_swiss_tables(
    matches: list[dict],
    elo_by_model: dict[str, float],
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Emit (swiss.md, combined.md, combined.pdf).

    - swiss.md: Swiss points only, sorted descending.
    - combined.md: Swiss points with ELO as tiebreak.
    - combined.pdf: same combined table as a printable figure.
    """
    swiss = swiss_points_table(matches)
    # Ensure every ELO-known model appears even if it sat out / had a bye.
    for m in elo_by_model:
        swiss.setdefault(m, 0.0)

    swiss_rows = sorted(swiss.items(), key=lambda kv: (-kv[1], kv[0]))
    pcol = _participant_col()
    swiss_md = [f"| # | {pcol} | Swiss points |", "|---:|---|---:|"]
    for i, (m, pts) in enumerate(swiss_rows, 1):
        swiss_md.append(f"| {i} | {_participant_label(m)} | {pts:.1f} |")
    swiss_path = out_dir / "swiss_table.md"
    swiss_path.write_text("\n".join(swiss_md) + "\n")

    combined_rows = sorted(
        swiss.items(),
        key=lambda kv: (-kv[1], -elo_by_model.get(kv[0], 1500.0), kv[0]),
    )
    combined_md = [
        f"| # | {pcol} | Swiss points | ELO (tiebreak) |",
        "|---:|---|---:|---:|",
    ]
    for i, (m, pts) in enumerate(combined_rows, 1):
        elo = elo_by_model.get(m, 1500.0)
        combined_md.append(
            f"| {i} | {_participant_label(m)} | {pts:.1f} | {elo:.1f} |"
        )
    combined_path = out_dir / "combined_table.md"
    combined_path.write_text("\n".join(combined_md) + "\n")

    # PDF version
    headers = ["#", pcol, "Swiss", "ELO (tiebreak)"]
    cells = [
        [
            str(i),
            _participant_label(m),
            f"{pts:.1f}",
            f"{elo_by_model.get(m, 1500.0):.1f}",
        ]
        for i, (m, pts) in enumerate(combined_rows, 1)
    ]
    n_rows = len(cells) + 1
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * n_rows))
    ax.axis("off")
    table = ax.table(
        cellText=cells, colLabels=headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for j in range(len(headers)):
        table[(0, j)].set_text_props(weight="bold")
    ax.set_title("Combined ranking (Swiss points, ELO tiebreak)", fontsize=11, pad=10)
    combined_pdf = out_dir / "combined_table.pdf"
    fig.savefig(combined_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return swiss_path, combined_path, combined_pdf


def _load_swiss_states(tournaments_dir: Path) -> list[dict]:
    """Load all swiss_state_*.json files written by run_tournament.py."""
    out: list[dict] = []
    for f in sorted(tournaments_dir.glob("swiss_state_*.json")):
        try:
            out.append(json.load(f.open()))
        except Exception:
            continue
    return out


def _load_round_robin_states(tournaments_dir: Path) -> list[dict]:
    """Load all round_robin_state_*.json files written by run_tournament_round_robin.py."""
    out: list[dict] = []
    for f in sorted(tournaments_dir.glob("round_robin_state_*.json")):
        try:
            out.append(json.load(f.open()))
        except Exception:
            continue
    return out


def write_bracket_figure(swiss_states: list[dict], out_paths: list[Path]) -> None:
    """Round-by-round bracket visualization for each Swiss tournament.

    Each tournament gets its own figure rendered into a multi-panel layout (one
    panel per round). Within a panel, every pairing is drawn as two stacked
    model boxes linked by a connecting line. Byes are flagged.
    """
    if not swiss_states:
        # Placeholder
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No Swiss tournaments logged yet",
                ha="center", va="center", fontsize=12)
        for p in out_paths:
            fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        return

    # Build one figure stacking all tournaments vertically.
    panels = []
    for st in swiss_states:
        rounds = st.get("rounds") or []
        if rounds:
            panels.append((st, rounds))

    if not panels:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "Swiss tournaments have no recorded rounds yet",
                ha="center", va="center", fontsize=12)
        for p in out_paths:
            fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        return

    # Figure size: each tournament panel ~ (cols=max_rounds_in_tourney, rows=max_pairings)
    max_rounds = max(len(r) for _, r in panels)
    max_pairings = max(
        max((len(rnd) for rnd in r), default=1) for _, r in panels
    )
    panel_h = 0.55 * max_pairings + 1.2
    fig_h = panel_h * len(panels)
    fig_w = 3.0 + 3.2 * max_rounds
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(fig_w, fig_h), squeeze=False,
    )

    for panel_idx, (st, rounds) in enumerate(panels):
        ax = axes[panel_idx, 0]
        tid = st.get("tournament_id", "?")
        repo = st.get("repo_name", "?")
        ax.set_title(
            f"Swiss bracket — tournament {tid[:8]}… on {repo} "
            f"({len(rounds)} round{'s' if len(rounds) != 1 else ''})",
            fontsize=10,
            loc="left",
        )
        ax.set_xlim(0, max_rounds + 0.5)
        ax.set_ylim(0, max_pairings + 0.5)
        ax.invert_yaxis()
        ax.axis("off")

        for r_idx, pairings in enumerate(rounds):
            ax.text(
                r_idx + 0.5, 0.2, f"Round {r_idx + 1}",
                ha="center", va="top", fontsize=9, fontweight="bold",
            )
            for p_idx, pp in enumerate(pairings):
                y = p_idx + 1
                a = pp.get("model_a", "?")
                b = pp.get("model_b")
                short_a = _participant_label(a).split("/")[-1][:22]
                box_a = ax.text(
                    r_idx + 0.5, y - 0.18, short_a,
                    ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#ddf4ff", edgecolor="#0969da"),
                )
                if b is None:
                    ax.text(
                        r_idx + 0.5, y + 0.18, "BYE",
                        ha="center", va="center", fontsize=7, style="italic",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="#fff8c5", edgecolor="#9a6700"),
                    )
                else:
                    short_b = _participant_label(b).split("/")[-1][:22]
                    ax.text(
                        r_idx + 0.5, y + 0.18, short_b,
                        ha="center", va="center", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="#fbefff", edgecolor="#8250df"),
                    )
                del box_a  # silence linter

    fig.tight_layout()
    for p in out_paths:
        fmt = p.suffix.lstrip(".").lower() or "pdf"
        kw: dict = {"format": fmt, "bbox_inches": "tight"}
        if fmt == "png":
            kw["dpi"] = 160
        fig.savefig(p, **kw)
    plt.close(fig)


def write_matchup_figure(
    matches: list[dict],
    out_paths: list[Path],
    *,
    elo_by_model: dict[str, float] | None = None,
) -> None:
    """Per-role matrix: cell (red=row, blue=col) = mean red_composite across turns.

    Axes are sorted by ELO descending when ``elo_by_model`` is supplied, so the
    strongest model is in the top-left corner. Cells for pairs that have never
    faced each other render as a gray "—" and never raise.
    """
    discovered = {m["model_a_id"] for m in matches} | {m["model_b_id"] for m in matches}
    if elo_by_model:
        discovered |= set(elo_by_model)
    if not discovered:
        # Emit an empty placeholder figure rather than crashing.
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No matches yet", ha="center", va="center", fontsize=12)
        for out_path in out_paths:
            fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    if elo_by_model:
        models = sorted(discovered, key=lambda m: -elo_by_model.get(m, 1500.0))
    else:
        models = sorted(discovered)
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    sums = np.zeros((n, n))
    counts = np.zeros((n, n))
    for m in matches:
        for r in m.get("turns", []) or []:
            ri, bj = idx.get(r["red_model_id"]), idx.get(r["blue_model_id"])
            if ri is None or bj is None:
                continue
            sums[ri, bj] += r["score"]["red_composite"]
            counts[ri, bj] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e6e6e6")

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * n, 1.5 + 1.1 * n))
    im = ax.imshow(
        np.ma.masked_invalid(mean),
        cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal",
    )
    label_for = [_participant_label(m) for m in models]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_for, rotation=30, ha="right")
    ax.set_yticklabels(label_for)
    ax.set_xlabel("Blue (defender)")
    ax.set_ylabel("Red (attacker)")
    title = "Mean red_composite per (Red, Blue)"
    if elo_by_model:
        title += " — axes sorted by ELO ↓"
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                v = mean[i, j]
                text_color = "white" if (v <= 0.25 or v >= 0.75) else "black"
                ax.text(
                    j, i, f"{v:.2f}\n(n={int(counts[i, j])})",
                    ha="center", va="center", fontsize=8,
                    color=text_color,
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#888")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    caption = textwrap.fill(
        "Rows: Red (attacker) model that generated the challenge. "
        "Columns: Blue (defender) model attempting the fix. "
        "Cell = mean red_composite over n turns (1.0 = Red wins, 0.0 = Blue wins); "
        "n is the turn count. Empty cells (gray '—') = the two models have not yet "
        "faced each other in that role assignment. Axes are sorted by ELO (descending) "
        "when ELO data is available, so the strongest model sits in the top-left.",
        width=max(80, 18 * n),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8)
    for out_path in out_paths:
        fmt = out_path.suffix.lstrip(".").lower() or "pdf"
        save_kwargs: dict = {"format": fmt, "bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = 160
        fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# HTML tournament report
# ───────────────────────────────────────────────────────────────────────

def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def _render_diff(diff: str) -> str:
    if not diff:
        return '<div class="empty">(no diff)</div>'
    out = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            cls = "diff-file"
        elif line.startswith("@@"):
            cls = "diff-hunk"
        elif line.startswith("+"):
            cls = "diff-add"
        elif line.startswith("-"):
            cls = "diff-del"
        else:
            cls = ""
        out.append(f'<span class="{cls}">{_esc(line)}</span>')
    return '<pre class="diff">' + "\n".join(out) + "</pre>"


def _render_trajectory(traj: dict | None, label: str) -> str:
    if not traj:
        return f'<div class="empty">(no {_esc(label)} trajectory)</div>'
    steps = traj.get("steps") or []
    meta = (
        f'<div class="traj-meta">'
        f'{len(steps)} steps · '
        f'{traj.get("total_input_tokens", 0):,} in / '
        f'{traj.get("total_output_tokens", 0):,} out tokens · '
        f'${traj.get("total_cost_usd", 0):.4f} · '
        f'{traj.get("duration_seconds", 0):.1f}s · '
        f'<span class="mono">{_esc(traj.get("model_id"))}</span>'
        f"</div>"
    )
    if not steps:
        return meta + '<div class="empty">(no steps)</div>'
    items = []
    for i, step in enumerate(steps):
        in_tok = int(step.get("input_tokens") or 0)
        out_tok = int(step.get("output_tokens") or 0)
        step_cost = float(step.get("cost_usd") or 0.0)
        cost_meta = (
            f'<div class="traj-meta">'
            f'{in_tok:,} in / {out_tok:,} out tokens · '
            f'${step_cost:.6f}'
            f"</div>"
        )
        items.append(
            f'<details class="step"><summary>Step {i + 1} '
            f'<span class="step-cost">({in_tok:,} in / {out_tok:,} out · ${step_cost:.6f})</span>'
            f'</summary>'
            f"{cost_meta}"
            f'<div class="step-block"><div class="step-label">thought</div>'
            f"<pre>{_esc(step.get('thought') or '')}</pre></div>"
            f'<div class="step-block"><div class="step-label">action</div>'
            f"<pre>{_esc(step.get('action') or '')}</pre></div>"
            f'<div class="step-block"><div class="step-label">observation</div>'
            f"<pre>{_esc(step.get('observation') or '')}</pre></div>"
            f"</details>"
        )
    return meta + '<div class="steps">' + "".join(items) + "</div>"


def _league_table(matches: list[dict]) -> dict[str, dict]:
    """Soccer-league-style stats per model, computed from raw match dicts.

    For every leaf turn (a single repo defense) the attacking Red model "scores
    a goal" iff ``red_composite > 0.5`` (the defender failed to stop the
    attack). Aggregated across a match, whoever scores more goals wins the
    match (1 point); a tie splits 0.5 each. Columns:

    - ``gf``: goals scored (successful attacks as Red)
    - ``ga``: goals conceded (failed defenses as Blue — opponent scored)
    - ``gd``: attack differential (``gf - ga``)
    - ``points``: cumulative league points (1 win / 0.5 draw / 0 loss)
    - ``played``: matches played
    - ``wins``: matches won
    - ``losses``: matches lost
    """
    table: dict[str, dict] = {}
    for m in matches:
        a, b = m.get("model_a_id"), m.get("model_b_id")
        if not a or not b:
            continue
        for mid in (a, b):
            table.setdefault(mid, {
                "gf": 0, "ga": 0, "points": 0.0, "played": 0,
                "wins": 0, "losses": 0, "draws": 0,
            })
        gf_a = gf_b = 0
        for t in m.get("turns", []) or []:
            red = t.get("red_model_id")
            sc = t.get("score") or {}
            red_comp = float(sc.get("red_composite") or 0.0)
            if red_comp > 0.5:
                if red == a:
                    gf_a += 1
                elif red == b:
                    gf_b += 1
        ga_a, ga_b = gf_b, gf_a
        if gf_a > gf_b:
            pa, pb = 1.0, 0.0
            table[a]["wins"] += 1
            table[b]["losses"] += 1
        elif gf_a < gf_b:
            pa, pb = 0.0, 1.0
            table[a]["losses"] += 1
            table[b]["wins"] += 1
        else:
            pa, pb = 0.5, 0.5
            table[a]["draws"] += 1
            table[b]["draws"] += 1
        table[a]["gf"] += gf_a
        table[a]["ga"] += ga_a
        table[a]["points"] += pa
        table[a]["played"] += 1
        table[b]["gf"] += gf_b
        table[b]["ga"] += ga_b
        table[b]["points"] += pb
        table[b]["played"] += 1
    for st in table.values():
        st["gd"] = st["gf"] - st["ga"]
    return table


def _render_elo_html(
    ratings: list[dict],
    per_model_usage: dict[str, dict] | None = None,
    *,
    league_table: dict[str, dict] | None = None,
) -> str:
    usage = per_model_usage or {}
    if league_table is not None:
        # Default sort: Bradley-Terry descending (league points as tiebreak).
        rows = sorted(
            ratings,
            key=lambda r: (
                # -float(r.get("bradley_terry", 0.0)),
                -float(r.get("elo", 0.0)),
                -league_table.get(r.get("model_id"), {}).get("points", 0.0),
                r.get("model_id", ""),
            ),
        )
        body = []
        for i, r in enumerate(rows, 1):
            mid = r.get("model_id")
            u = usage.get(mid, {})
            gen_cost = u.get("gen_cost", 0.0)
            val_cost = u.get("val_cost", 0.0)
            eval_cost = u.get("eval_cost", 0.0)
            total_cost = gen_cost + val_cost + eval_cost
            in_tok = u.get("in_tokens", 0)
            out_tok = u.get("out_tokens", 0)
            lt = league_table.get(mid, {
                "gf": 0, "ga": 0, "gd": 0, "points": 0.0,
                "played": 0, "wins": 0, "losses": 0, "draws": 0,
            })
            body.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td class='mono'>{_esc(_participant_label(mid))}</td>"
                f"<td>{lt['played']}</td>"
                f"<td>{lt['wins']}</td>"
                f"<td>{lt['draws']}</td>"
                f"<td>{lt['losses']}</td>"
                f"<td>{lt['gf']}</td>"
                f"<td>{lt['ga']}</td>"
                f"<td>{lt['gd']:+d}</td>"
                # f"<td>{lt['points']:.1f}</td>"
                f"<td>{r.get('elo', 0):.1f}</td>"
                f"<td>{r.get('red_elo', 0):.1f}</td>"
                f"<td>{r.get('blue_elo', 0):.1f}</td>"
                f"<td>{r.get('bradley_terry', 0):.3f}</td>"
                f"<td>${gen_cost:.4f}</td>"
                f"<td>${val_cost:.4f}</td>"
                f"<td>${eval_cost:.4f}</td>"
                f"<td>${total_cost:.4f}</td>"
                f"<td>{in_tok:,}</td>"
                f"<td>{out_tok:,}</td>"
                "</tr>"
            )
        headers = [
            ("#", "num"),
            (_participant_col(), "str"),
            ("Played", "num"),
            ("Won", "num"),
            ("Drawn", "num"),
            ("Lost", "num"),
            ("Successful<br>attacks", "num"),
            ("Unsuccessful<br>defenses", "num"),
            ("Attack<br>differential", "num"),
            # ("League points", "num"),
            ("ELO", "num"),
            ("Red ELO", "num"),
            ("Blue ELO", "num"),
            ("Bradley-Terry", "num"),
            ("Gen. $<br>(Red)", "num"),
            ("Val. $<br>(Emulated Blue)", "num"),
            ("Def. $<br>(Blue)", "num"),
            ("Total $", "num"),
            ("Input tokens", "num"),
            ("Output tokens", "num"),
        ]
        # Bradley-Terry is the default sort column — mark it so the visual
        # arrow matches the server-side emission order.
        bt_col_idx = next(
            # i for i, (label, _) in enumerate(headers) if "Bradley-Terry" in label
            i for i, (label, _) in enumerate(headers) if "ELO" in label
        )
        head_html = "".join(
            f'<th data-sort="{kind}" data-col="{i}"'
            f'{" class=\"sorted-desc\"" if i == bt_col_idx else ""}>'
            f"{label}<span class=\"sort-arrow\"></span></th>"
            for i, (label, kind) in enumerate(headers)
        )
        return (
            '<div class="elo-scroll"><table class="elo sortable"><thead><tr>'
            + head_html
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
        )

    rows = sorted(ratings, key=lambda r: r.get("elo", 0), reverse=True)
    body = []
    for i, r in enumerate(rows, 1):
        mid = r.get("model_id")
        u = usage.get(mid, {})
        gen_cost = u.get("gen_cost", 0.0)
        val_cost = u.get("val_cost", 0.0)
        eval_cost = u.get("eval_cost", 0.0)
        total_cost = gen_cost + val_cost + eval_cost
        in_tok = u.get("in_tokens", 0)
        out_tok = u.get("out_tokens", 0)
        body.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td class='mono'>{_esc(_participant_label(mid))}</td>"
            f"<td>{r.get('elo', 0):.1f}</td>"
            f"<td>{r.get('red_elo', 0):.1f}</td>"
            f"<td>{r.get('blue_elo', 0):.1f}</td>"
            f"<td>{r.get('trueskill_mu', 0):.2f} ± {r.get('trueskill_sigma', 0):.2f}</td>"
            f"<td>{r.get('bradley_terry', 0):.3f}</td>"
            f"<td>{r.get('matches_played', 0)}</td>"
            f"<td>${gen_cost:.4f}</td>"
            f"<td>${val_cost:.4f}</td>"
            f"<td>${eval_cost:.4f}</td>"
            f"<td>${total_cost:.4f}</td>"
            f"<td>{in_tok:,}</td>"
            f"<td>{out_tok:,}</td>"
            "</tr>"
        )
    headers = [
        ("#", "num"),
        (_participant_col(), "str"),
        ("ELO", "num"),
        ("Red ELO", "num"),
        ("Blue ELO", "num"),
        ("TrueSkill", "num"),
        ("Bradley-Terry", "num"),
        ("Matches", "num"),
        ("Gen $ (Red)", "num"),
        ("Val $ (Emulated Blue)", "num"),
        ("Eval $ (Blue)", "num"),
        ("Total $", "num"),
        ("Input tokens", "num"),
        ("Output tokens", "num"),
    ]
    head_html = "".join(
        f'<th data-sort="{kind}" data-col="{i}">{label}<span class="sort-arrow"></span></th>'
        for i, (label, kind) in enumerate(headers)
    )
    return (
        '<div class="elo-scroll"><table class="elo sortable"><thead><tr>'
        + head_html
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _render_findings(findings: list[dict]) -> str:
    if not findings:
        return '<div class="empty">(no findings)</div>'
    rows = "".join(
        "<tr>"
        f"<td>{_esc(f.get('severity'))}</td>"
        f"<td class='mono'>{_esc(f.get('location'))}</td>"
        f"<td>{_esc(f.get('description'))}</td>"
        "</tr>"
        for f in findings
    )
    return (
        '<table class="findings"><thead><tr>'
        "<th>Severity</th><th>Location</th><th>Description</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def _render_score(score: dict) -> str:
    keys = [
        ("s_regression", "Regression"),
        ("s_feature", "Feature"),
        ("s_bugfix", "Bug fix"),
        ("blue_composite", "Blue"),
        ("red_composite", "Red"),
    ]
    badges = []
    for k, label in keys:
        if k not in score:
            continue
        v = float(score.get(k) or 0)
        cls = "ok" if v >= 1.0 else ("mid" if v > 0 else "bad")
        badges.append(f'<span class="badge {cls}">{_esc(label)}: {v:.2f}</span>')
    return '<div class="score-row">' + "".join(badges) + "</div>"


def _render_round(rnd: dict, label: str = "") -> str:
    """Render one leaf turn (a single repo defense).

    ``label`` is the repo name shown in the leaf summary; the slot/side context
    is supplied by the enclosing turn/sub-turn ``<details>`` wrappers built in
    :func:`_render_match`.
    """
    lead = f"{_esc(label)} — " if label else ""
    cr = rnd.get("challenge_record")
    if cr is None:
        # Slim match JSON: only score + ids embedded; full challenge/defense bodies
        # live elsewhere. Render a compact summary instead of crashing.
        score = rnd.get("score", {})
        return f"""
        <details class="round">
          <summary>{lead}<span class="mono">{_esc(display_composite_id(rnd.get('red_model_id') or ''))}</span>
          → <span class="mono">{_esc(display_composite_id(rnd.get('blue_model_id') or ''))}</span>
          {_render_score(score)}</summary>
          <div class="round-body">
            <div class="meta">
              <div><b>Challenge:</b> <span class="mono">{_esc(rnd.get('challenge_id'))}</span></div>
              <div><b>Defense:</b> <span class="mono">{_esc(rnd.get('defense_id'))}</span></div>
            </div>
            <div class="empty">(full challenge / defense bodies are not embedded in this match JSON)</div>
          </div>
        </details>
        """
    ch = cr["challenge"]
    dr = rnd["defense_result"]
    bf = dr["blue_fix"]
    score = dr.get("score", {})

    feat_traj = ch.get("feature_trajectory")
    bug_traj = ch.get("bug_trajectory")
    if feat_traj or bug_traj:
        red_trajs = (
            '<details class="sub"><summary>Red reasoning — feature phase</summary>'
            + _render_trajectory(feat_traj, "feature") + "</details>"
            '<details class="sub"><summary>Red reasoning — bug-embedding phase</summary>'
            + _render_trajectory(bug_traj, "bug") + "</details>"
        )
    else:
        red_trajs = (
            '<details class="sub"><summary>Red reasoning</summary>'
            + _render_trajectory(ch.get("agent_trajectory"), "red") + "</details>"
        )

    blue_trajs = (
        '<details class="sub"><summary>Blue reasoning — detection &amp; fix</summary>'
        + _render_trajectory(bf.get("agent_trajectory"), "blue") + "</details>"
    )

    target_files = ", ".join(ch.get("target_files") or []) or "(unknown)"

    return f"""
    <details class="round">
      <summary>{lead}target: <span class="mono">{_esc(target_files)}</span> {_render_score(score)}</summary>
      <div class="round-body">
        <div class="meta">
          <div><b>Red:</b> <span class="mono">{_esc(display_composite_id(rnd.get('red_model_id') or ''))}</span></div>
          <div><b>Blue:</b> <span class="mono">{_esc(display_composite_id(rnd.get('blue_model_id') or ''))}</span></div>
          <div><b>Bug type:</b> {_esc(ch.get('bug_type'))}</div>
          <div><b>Bug location:</b> <span class="mono">{_esc(ch.get('bug_location'))}</span></div>
          <div><b>Duration:</b> {dr.get('duration_seconds', 0):.1f}s · <b>Cost:</b> ${dr.get('cost_usd', 0):.4f}</div>
        </div>

        <h4>Feature spec</h4>
        <pre class="text">{_esc(ch.get('feature_spec'))}</pre>

        <details class="sub"><summary>Feature rationale</summary>
          <pre class="text">{_esc(ch.get('feature_rationale'))}</pre>
        </details>

        <details class="sub"><summary>Bug description (hidden from Blue at match time)</summary>
          <pre class="text">{_esc(ch.get('bug_description'))}</pre>
        </details>

        <h4>Red PR diff (feature + embedded bug)</h4>
        {_render_diff(ch.get('pr_diff', ''))}

        {red_trajs}

        <h4>Blue review &amp; fix</h4>
        <div><b>Explanation:</b></div>
        <pre class="text">{_esc(bf.get('fix_explanation'))}</pre>

        <h5>Review findings</h5>
        {_render_findings(bf.get('review_findings') or [])}

        <h5>Blue fix diff</h5>
        {_render_diff(bf.get('fix_diff', ''))}

        {blue_trajs}
      </div>
    </details>
    """


def _trajectory_totals(traj: dict | None) -> tuple[int, int, float]:
    if not traj:
        return 0, 0, 0.0
    return (
        int(traj.get("total_input_tokens", 0) or 0),
        int(traj.get("total_output_tokens", 0) or 0),
        float(traj.get("total_cost_usd", 0.0) or 0.0),
    )


def _match_token_totals(m: dict) -> tuple[int, int]:
    in_tok = out_tok = 0
    for rnd in m.get("turns", []) or []:
        cr = rnd.get("challenge_record") or {}
        ch = cr.get("challenge") or {}
        for key in ("agent_trajectory", "feature_trajectory", "bug_trajectory"):
            i, o, _ = _trajectory_totals(ch.get(key))
            in_tok += i
            out_tok += o
        bf = (rnd.get("defense_result") or {}).get("blue_fix") or {}
        i, o, _ = _trajectory_totals(bf.get("agent_trajectory"))
        in_tok += i
        out_tok += o
    return in_tok, out_tok


def _classify_turns(
    m: dict, challenge_idx: dict[str, dict] | None
) -> list[tuple[int, str, str, dict]]:
    """Tag each flat turn with ``(slot, side, repo_label, turn)``.

    A match spans ``turns_per_player`` *slots* (targets), two *sides* (A attacks
    vs B attacks), and one or more *repos*. The slim match JSON stores none of
    these directly, so we recover them:

    - **side**: ``A`` if the turn's Red is ``model_a_id`` (A is attacking),
      else ``B``.
    - **repo**: the turn's own ``repo_name`` when present (written by
      ``log_match`` / back-filled by ``migrate_match_repo_names.py``). This is
      authoritative even for an **auto-win** turn (empty ``challenge_id``), which
      otherwise has no recoverable repo. Falls back to the challenge index by
      ``challenge_id``, then to ``(auto-win)`` / ``(unknown)`` for the oldest
      un-migrated records. Grouping auto-win turns under one shared
      ``(auto-win)`` bucket was the cause of phantom extra "Turn N" rows when a
      side had ≥2 auto-wins.
    - **slot**: the running index within each ``(repo, side)`` group — turns
      were emitted slot-by-slot per repo by ``MatchOrchestrator._build_turns``,
      so the Nth turn for a given repo+side is slot N.
    """
    chal_idx = challenge_idx or {}
    a_id = m.get("model_a_id")
    counters: dict[tuple[str, str], int] = defaultdict(int)
    out: list[tuple[int, str, str, dict]] = []
    for t in m.get("turns", []) or []:
        cid = t.get("challenge_id")
        repo = t.get("repo_name") or (chal_idx.get(cid or "", {}) or {}).get("repo")
        if not repo:
            repo = "(auto-win)" if not cid else "(unknown)"
        side = "A" if t.get("red_model_id") == a_id else "B"
        slot = counters[(repo, side)]
        counters[(repo, side)] += 1
        out.append((slot, side, repo, t))
    return out


def _render_match(m: dict, idx: int, challenge_idx: dict[str, dict] | None = None) -> str:
    a_label = display_composite_id(m.get("model_a_id") or "")
    b_label = display_composite_id(m.get("model_b_id") or "")
    side_attacker = {"A": a_label, "B": b_label}
    side_defender = {"A": b_label, "B": a_label}

    classified = _classify_turns(m, challenge_idx)

    # slot → side → list[(repo, turn)]
    by_slot: dict[int, dict[str, list[tuple[str, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for slot, side, repo, turn in classified:
        by_slot[slot][side].append((repo, turn))

    turn_blocks = []
    for slot in sorted(by_slot):
        side_blocks = []
        for side in ("A", "B"):
            repos = by_slot[slot].get(side)
            if not repos:
                continue
            repo_leaves = "".join(
                _render_round(turn, label=repo)
                for repo, turn in sorted(repos, key=lambda rt: rt[0])
            )
            side_blocks.append(
                '<details class="subturn"><summary>'
                f'Sub-turn {side} — <span class="mono">{_esc(side_attacker[side])}</span> '
                f'attacks → <span class="mono">{_esc(side_defender[side])}</span> defends '
                f'<span class="subturn-count">({len(repos)} repo{"s" if len(repos) != 1 else ""})</span>'
                '</summary><div class="subturn-body">'
                f"{repo_leaves}</div></details>"
            )
        turn_blocks.append(
            '<details class="turn"><summary>'
            f'Turn {slot + 1} <span class="subturn-count">(target slot {slot + 1})</span>'
            '</summary><div class="turn-body">'
            f"{''.join(side_blocks)}</div></details>"
        )
    turns_html = "".join(turn_blocks) or '<div class="empty">(no turns)</div>'

    outcome = m.get("outcome", "")
    outcome_cls = {
        "model_a_wins": "win-a",
        "model_b_wins": "win-b",
        "draw": "draw",
    }.get(outcome, "")
    in_tok, out_tok = _match_token_totals(m)
    return f"""
    <details class="match">
      <summary>
        Match {idx + 1} · <span class="mono">{_esc(m.get('repo_name'))}</span> ·
        <span class="mono">{_esc(a_label)}</span> ({m.get('model_a_total', 0):.2f})
        vs
        <span class="mono">{_esc(b_label)}</span> ({m.get('model_b_total', 0):.2f})
        · <span class="outcome {outcome_cls}">{_esc(outcome)}</span>
        · {m.get('duration_seconds', 0):.0f}s · ${m.get('total_cost_usd', 0):.4f}
        · {in_tok:,} in / {out_tok:,} out tokens
      </summary>
      <div class="match-body">{turns_html}</div>
    </details>
    """


_HTML_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 24px; background: #f6f8fa; color: #1f2328; }
h1, h2, h3, h4, h5 { margin: 0.6em 0 0.3em; }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.9em; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
        padding: 16px; margin-bottom: 16px; }
.header-meta { color: #57606a; font-size: 0.9em; line-height: 1.6; }
table { border-collapse: collapse; width: 100%; font-size: 0.92em; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eaeef2; }
th { background: #f6f8fa; }
table.elo td:nth-child(1) { width: 3em; color: #57606a; }
table.rubric td:nth-child(n+2):nth-child(-n+6) { text-align: center; font-variant-numeric: tabular-nums; }
table.rubric td:nth-child(5) { color: #0969da; }
details { border: 1px solid #d0d7de; border-radius: 6px; margin: 8px 0; background: #fff; }
summary { padding: 10px 12px; cursor: pointer; user-select: none; }
summary:hover { background: #f6f8fa; }
details.match > summary { font-weight: 600; }
details.turn > summary { font-weight: 600; background: #eef2f6; }
details.turn { border-color: #c4ccd4; }
details.subturn > summary { font-weight: 500; background: #f2f6fa; }
.turn-body, .subturn-body { padding: 6px 12px 10px; }
.subturn-count { color: #8b949e; font-size: 0.82em; font-weight: 400; }
details.round > summary { background: #f6f8fa; }
details.sub { border-color: #eaeef2; margin: 6px 0; }
details.step { border: none; border-top: 1px solid #eaeef2; border-radius: 0; margin: 0; }
details.step > summary { padding: 4px 10px; color: #57606a; font-size: 0.85em; }
.step-block { padding: 4px 10px; }
.step-label { font-size: 0.72em; color: #57606a; text-transform: uppercase; letter-spacing: 0.05em; }
.step-block pre { margin: 2px 0 8px; }
.match-body, .round-body { padding: 8px 14px 14px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 24px; color: #57606a; font-size: 0.9em; margin-bottom: 8px; }
pre { background: #f6f8fa; border: 1px solid #eaeef2; border-radius: 6px;
      padding: 10px; overflow-x: auto; font-family: ui-monospace, Menlo, monospace;
      font-size: 0.82em; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
pre.text { white-space: pre-wrap; }
pre.diff { background: #fff; padding: 0; }
pre.diff span { display: block; white-space: pre-wrap; padding: 0 10px; }
.diff-add { background: #e6ffec; color: #1a7f37; }
.diff-del { background: #ffebe9; color: #b1281f; }
.diff-file { color: #57606a; font-weight: 600; background: #f6f8fa; }
.diff-hunk { color: #8250df; background: #fbefff; }
.empty { color: #8b949e; font-style: italic; padding: 4px 0; }
.score-row { display: inline-flex; gap: 6px; flex-wrap: wrap; margin-left: 8px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
.badge.ok  { background: #dafbe1; color: #116329; }
.badge.mid { background: #fff8c5; color: #7d4e00; }
.badge.bad { background: #ffebe9; color: #82071e; }
.outcome { padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: 600; }
.outcome.win-a { background: #ddf4ff; color: #0969da; }
.outcome.win-b { background: #fbefff; color: #8250df; }
.outcome.draw { background: #eaeef2; color: #57606a; }
.traj-meta { color: #57606a; font-size: 0.85em; padding: 6px 10px; }
.step-cost { color: #8b949e; font-size: 0.85em; font-weight: 400; }
.matrix-embed { max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; display: block; }
.rounds-scroll { display: flex; gap: 12px; overflow-x: auto; padding: 4px 0 12px; align-items: flex-start; }
.round-card { flex: 0 0 auto; min-width: 240px; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px 10px; }
.round-card h3 { margin: 4px 0 8px; text-align: center; font-size: 0.95em; }
.bracket-group { margin-bottom: 8px; padding: 6px 8px; background: #fff; border-radius: 4px; border: 1px solid #eaeef2; }
.bracket-group:last-child { margin-bottom: 0; }
.bracket-label { font-size: 0.7em; color: #57606a; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; text-align: center; }
.pairing { padding: 4px 0; border-bottom: 1px dashed #eaeef2; text-align: center; }
.pairing:last-child { border-bottom: none; }
.bracket-model { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.78em; padding: 2px 6px; background: #ddf4ff; color: #0969da; border-radius: 4px; display: inline-block; margin: 2px 0; }
.bracket-model.winner { background: #dafbe1; color: #116329; }
.bracket-model.loser  { background: #ffebe9; color: #82071e; }
.bracket-model.draw   { background: #fff8c5; color: #7d4e00; }
.bracket-vs { color: #8b949e; font-size: 0.7em; padding: 1px 0; }
.bracket-bye { display: inline-block; background: #fff8c5; color: #7d4e00; font-weight: 600; font-size: 0.72em; padding: 2px 8px; border-radius: 10px; margin-top: 2px; }
.elo-scroll { overflow-x: auto; width: 100%; }
.elo-scroll table.elo { width: max-content; min-width: 100%; }
.elo-scroll th, .elo-scroll td { white-space: nowrap; }
table.sortable th { cursor: pointer; user-select: none; position: relative; }
table.sortable th:hover { background: #eaeef2; }
.sort-arrow { display: inline-block; width: 0.9em; margin-left: 4px; color: #57606a; font-size: 0.85em; }
table.sortable th.sorted-asc .sort-arrow::after { content: "▲"; }
table.sortable th.sorted-desc .sort-arrow::after { content: "▼"; }
"""

_SORT_JS = r"""
<script>
(function () {
  function parseNum(s) {
    if (s == null) return NaN;
    var t = String(s).replace(/[$,\s]/g, "");
    var m = t.match(/-?\d+(?:\.\d+)?/);
    return m ? parseFloat(m[0]) : NaN;
  }
  function sortTable(table, colIdx, kind, asc) {
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var av = a.cells[colIdx].textContent.trim();
      var bv = b.cells[colIdx].textContent.trim();
      if (kind === "num") {
        var an = parseNum(av), bn = parseNum(bv);
        if (isNaN(an) && isNaN(bn)) return 0;
        if (isNaN(an)) return 1;
        if (isNaN(bn)) return -1;
        return asc ? an - bn : bn - an;
      }
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var ths = table.querySelectorAll("thead th");
    ths.forEach(function (th, idx) {
      th.addEventListener("click", function () {
        var kind = th.getAttribute("data-sort") || "str";
        var asc = !th.classList.contains("sorted-asc");
        ths.forEach(function (h) { h.classList.remove("sorted-asc", "sorted-desc"); });
        th.classList.add(asc ? "sorted-asc" : "sorted-desc");
        sortTable(table, idx, kind, asc);
      });
    });
  });
})();
</script>
"""


def _render_scoring_rubric_html() -> str:
    """Per-turn scoring rubric, appended at the bottom of the report for reference.

    Blue earns ``blue_composite`` for a turn it defends; Red (the challenge
    author) earns ``red_composite = 1 - blue_composite``. All three suites are
    HARD multiplicative gates:

        blue_composite = s_regression × s_feature × s_bugfix
    """
    rows = [
        ("Blue fixes the bug; all tests pass", "1", "1", "1", "1", "0",
         "Best case — bug removed and feature retained; Blue wins the turn."),
        ("Blue accepts the PR as-is (bug not fixed)", "1", "1", "0", "0", "1",
         "s_bugfix gate fails: the embedded bug survives, so Red wins the turn."),
        ("Blue fixes the bug but breaks the feature", "1", "0", "1", "0", "1",
         "s_feature gate fails: feature lost is a total loss for Blue (no partial credit)."),
        ("Blue fixes the bug but breaks existing tests", "0", "—", "—", "0", "1",
         "s_regression gate fails: shipping a regression is a total loss for Blue."),
        ("Blue's changes break everything", "0", "—", "—", "0", "1",
         "Regression gate fails before feature/bug suites even run."),
        ("Blue makes no changes (empty fix)", "0", "—", "—", "0", "1",
         "No diff submitted — scored as a total loss for Blue."),
    ]
    body = "".join(
        f"<tr><td>{_esc(case)}</td><td>{reg}</td><td>{feat}</td><td>{bug}</td>"
        f"<td><b>{blue}</b></td><td>{red}</td><td>{_esc(note)}</td></tr>"
        for case, reg, feat, bug, blue, red, note in rows
    )
    return f"""
  <div class="card">
    <h2>Per-turn scoring rubric</h2>
    <p class="header-meta">Each turn, the defending <b>Blue</b> agent earns
    <code>blue_composite</code>; the attacking <b>Red</b> (challenge author) earns
    <code>red_composite = 1 − blue_composite</code>. Sub-scores are binary (1.0 iff
    every test in the suite passes). All three of <code>s_regression</code>,
    <code>s_feature</code>, and <code>s_bugfix</code> are <b>hard multiplicative
    gates</b> — Blue wins the turn only when every suite passes:</p>
    <p class="header-meta"><code>blue_composite = s_regression × s_feature ×
    s_bugfix</code></p>
    <table class="rubric">
      <thead><tr>
        <th>Case</th><th>s_regression</th><th>s_feature</th><th>s_bugfix</th>
        <th>Blue</th><th>Red</th><th>Notes</th>
      </tr></thead>
      <tbody>{body}</tbody>
    </table>
  </div>
"""


def _build_tournament_html(
    tournament: dict,
    source_path: Path,
    matrix_pdf_rel: str | None,
    *,
    matches: list[dict] | None = None,
    ratings: list[dict] | None = None,
    per_model_usage: dict[str, dict] | None = None,
    totals: dict | None = None,
    challenge_idx: dict[str, dict] | None = None,
    round_robin: bool = False,
    ranking_matches: list[dict] | None = None,
) -> str:
    tid = tournament.get("tournament_id", "?")
    ts = tournament.get("timestamp", "")
    if matches is None:
        matches = tournament.get("matches", [])
    if ratings is None:
        ratings = tournament.get("final_ratings", [])
    totals = totals or {}
    gen_cost = float(
        totals.get("gen_cost")
        if totals.get("gen_cost") is not None
        else tournament.get("total_generation_cost_usd", 0.0)
    )
    val_cost = float(totals.get("val_cost") or 0.0)
    eval_cost = float(
        totals.get("eval_cost")
        if totals.get("eval_cost") is not None
        else tournament.get("total_evaluation_cost_usd", 0.0)
    )
    total_in = int(totals.get("in_tokens") or 0)
    total_out = int(totals.get("out_tokens") or 0)
    if not total_in and not total_out:
        for m in matches:
            i, o = _match_token_totals(m)
            total_in += i
            total_out += o

    # Rankings / league / bracket outcomes use ranking_matches when provided
    # (harness-ablation folded pairings); match-list body still uses ``matches``.
    rank_src = ranking_matches if ranking_matches is not None else matches
    league_table = _league_table(rank_src) if round_robin else None

    matches_html = "".join(
        _render_match(m, i, challenge_idx) for i, m in enumerate(matches)
    )

    bracket_html = _render_bracket_html(
        tournament, rank_src, round_robin=round_robin
    )
    if _is_harness_ablation():
        bracket_title = "Harness league rounds"
        bracket_blurb = (
            "One card per round. Participants are harnesses; each pairing "
            "aggregates same-model sub-matches across the fixed model set."
        )
    elif round_robin:
        bracket_title = "League rounds"
        bracket_blurb = (
            "One card per round; pairings are fixed by the circle method (no "
            "bracket re-grouping — that is a Swiss-system concept)."
        )
    else:
        bracket_title = "Swiss bracket progression"
        bracket_blurb = (
            "One card per round; from round 2 onward, pairings are grouped by the "
            "points bracket they entered the round with (scroll horizontally to "
            "see later rounds)."
        )
    bracket_section = f"""
  <div class="card">
    <h2>{bracket_title}</h2>
    <p class="header-meta">{bracket_blurb}</p>
    {bracket_html}
  </div>
"""

    matrix_section = ""
    if matrix_pdf_rel:
        matrix_section = f"""
  <div class="card">
    <h2>Matchup matrix</h2>
    <p class="header-meta">Mean <code>red_composite</code> per (Red attacker, Blue defender).
    Rendered from <span class="mono">{_esc(matrix_pdf_rel)}</span> (scoped to the matches
    claimed by this tournament).</p>
    <img class="matrix-embed" src="{_esc(matrix_pdf_rel)}" alt="Matchup matrix" />
  </div>
"""

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>SWE-Duel Tournament {_esc(tid)}</title>
<style>{_HTML_CSS}</style>
</head><body>
  <div class="card">
    <h1>SWE-Duel Tournament Report</h1>
    <div class="header-meta">
      <div><b>Tournament ID:</b> <span class="mono">{_esc(tid)}</span></div>
      <div><b>Timestamp:</b> {_esc(ts)}</div>
      <div><b>Source:</b> <span class="mono">{_esc(source_path)}</span></div>
      <div><b>Matches:</b> {len(matches)} · <b>Generation cost:</b> ${gen_cost:.4f} · <b>Validation cost:</b> ${val_cost:.4f} · <b>Defence cost:</b> ${eval_cost:.4f} · <b>Total cost:</b> ${gen_cost + val_cost + eval_cost:.4f}</div>
      <div><b>Tokens:</b> {total_in:,} input · {total_out:,} output</div>
    </div>
  </div>

  <div class="card">
    <h2>{"League table" if round_robin else "Overall Rankings"}</h2>
    {_render_elo_html(ratings, per_model_usage, league_table=league_table)}
  </div>
{bracket_section}{matrix_section}
  <div class="card">
    <h2>Matches</h2>
    {matches_html if matches_html else '<div class="empty">(no matches)</div>'}
  </div>
{_render_scoring_rubric_html()}
{_SORT_JS}
</body></html>
"""


def _load_challenge_index(challenges_dir: Path) -> dict[str, dict]:
    """challenge_id → {red_model, gen_cost, val_cost, in_tokens, out_tokens}.

    gen_cost  — Red agent cost only (feature + bug phases).
    val_cost  — Self-review (gate_self_review) cost: the emulated-Blue agent
                that validates the challenge is solvable.
    Token counts span all phases: feature, bug, and self-review.
    """
    out: dict[str, dict] = {}
    if not challenges_dir.exists():
        return out
    for f in challenges_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        ch = d.get("challenge") or {}
        # Sum tokens across all Red agent phases.  feature_trajectory and
        # bug_trajectory (modern two-phase records) each cover one phase;
        # agent_trajectory is the legacy combined form — present on older records
        # that pre-date the split.  Use the split form when available to avoid
        # double-counting, otherwise fall back to the combined form.
        in_tok = out_tok = 0
        ft = ch.get("feature_trajectory") or {}
        bt = ch.get("bug_trajectory") or {}
        if ft or bt:
            for t in (ft, bt):
                in_tok += int(t.get("total_input_tokens") or 0)
                out_tok += int(t.get("total_output_tokens") or 0)
        else:
            t = ch.get("agent_trajectory") or {}
            in_tok += int(t.get("total_input_tokens") or 0)
            out_tok += int(t.get("total_output_tokens") or 0)
        # Self-review (gate_self_review) is the emulated-Blue validation agent.
        # Its tokens count toward generation totals; its cost is reported separately
        # as "Val $ (Emulated Blue)" so it stays distinct from Red agent cost.
        sr_traj = ((d.get("validation") or {}).get("self_review") or {}).get("agent_trajectory") or {}
        in_tok += int(sr_traj.get("total_input_tokens") or 0)
        out_tok += int(sr_traj.get("total_output_tokens") or 0)
        harness = d.get("red_harness_id", "mini-swe-agent")
        out[d["challenge_id"]] = {
            "red_model": d.get("red_model_id"),
            "red_composite": f"{d.get('red_model_id')}#{harness}",
            "repo": d.get("repo_name"),
            "slot": int(d.get("slot", 1)),
            "gen_cost": float(d.get("generation_cost_usd") or 0.0),
            "val_cost": float(sr_traj.get("total_cost_usd") or 0.0),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
        }
    return out


def _load_failed_challenge_index(failed_dir: Path) -> dict[str, dict]:
    """failed_id → {red_composite, repo, slot, gen_cost, in_tokens, out_tokens}.

    A *failed* challenge attempt still burned Red-agent (and possibly
    self-review) tokens. ``FailedChallengeRecord.generation_cost_usd`` already
    rolls up every attempt's cost within that failed effort. These records are
    never referenced by a match turn (an auto-win turn has an empty
    challenge_id), so without this index the cost of a **failed-only slot** —
    one where every generation attempt failed and no challenge was banked — is
    dropped from the report entirely. ``_aggregate_usage`` folds them back in so
    the cost columns cover all attempts under the configured target count.
    """
    out: dict[str, dict] = {}
    if not failed_dir.exists():
        return out
    for f in failed_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        ch = d.get("challenge") or {}
        in_tok = out_tok = 0
        ft = ch.get("feature_trajectory") or {}
        bt = ch.get("bug_trajectory") or {}
        if ft or bt:
            for t in (ft, bt):
                in_tok += int(t.get("total_input_tokens") or 0)
                out_tok += int(t.get("total_output_tokens") or 0)
        else:
            t = ch.get("agent_trajectory") or {}
            in_tok += int(t.get("total_input_tokens") or 0)
            out_tok += int(t.get("total_output_tokens") or 0)
        sr = d.get("self_review_trajectory") or {}
        in_tok += int(sr.get("total_input_tokens") or 0)
        out_tok += int(sr.get("total_output_tokens") or 0)
        harness = d.get("red_harness_id", "mini-swe-agent")
        out[d["challenge_id"]] = {
            "red_composite": f"{d.get('red_model_id')}#{harness}",
            "repo": d.get("repo_name"),
            "slot": int(d.get("slot", 1)),
            "gen_cost": float(d.get("generation_cost_usd") or 0.0),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
        }
    return out


def _load_defense_index(defenses_dir: Path) -> dict[str, dict]:
    """defense_id → {blue_model, blue_composite, eval_cost, in_tokens, out_tokens}."""
    out: dict[str, dict] = {}
    if not defenses_dir.exists():
        return out
    for f in defenses_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        traj = (d.get("blue_fix") or {}).get("agent_trajectory") or {}
        blue_model = d.get("blue_model_id") or ""
        harness = d.get("blue_harness_id", "mini-swe-agent")
        out[d["defense_id"]] = {
            "blue_model": blue_model,
            "blue_composite": (
                f"{blue_model}#{harness}" if blue_model else ""
            ),
            "eval_cost": float(d.get("cost_usd") or 0.0),
            "in_tokens": int(traj.get("total_input_tokens") or 0),
            "out_tokens": int(traj.get("total_output_tokens") or 0),
        }
    return out


def _aggregate_usage(
    matches: list[dict],
    chal_idx: dict[str, dict],
    def_idx: dict[str, dict],
    failed_idx: dict[str, dict] | None = None,
    *,
    scope_competitors: set[str] | None = None,
    scope_repos: set[str] | None = None,
) -> tuple[dict[str, dict], dict]:
    """Per-competitor usage (counting each challenge / defense once) and totals.

    Generation cost covers **all attempts under the configured targets**, not
    just the ones a match turn happens to reference:

    - A **successful** challenge contributes its ``generation_cost_usd`` + val $.
      That figure already rolls up every failed attempt that preceded it within
      the same slot (the generator accumulates across attempts), so in-slot
      failed attempts are never added again — that would double-count.
    - A **failed-only slot** (every attempt failed; nothing banked) is never
      referenced by any turn — an auto-win turn carries an empty challenge_id.
      Its cost is the sum of every attempt in the slot, folded in via
      ``failed_idx`` for competitors/repos in scope.
    """
    per_model: dict[str, dict] = defaultdict(
        lambda: {
            "gen_cost": 0.0,
            "val_cost": 0.0,
            "eval_cost": 0.0,
            "in_tokens": 0,
            "out_tokens": 0,
        }
    )
    seen_challenges: set[str] = set()
    seen_defenses: set[str] = set()
    scoped_comp: set[str] = set(scope_competitors or ())
    scoped_repo: set[str] = set(scope_repos or ())
    # (composite, repo, slot) → success info already banked into per_model
    success_by_slot: dict[tuple[str, str, int], dict] = {}

    for m in matches:
        for k in ("model_a_id", "model_b_id"):
            if m.get(k):
                scoped_comp.add(m[k])
        repo_from_match = m.get("repo_name") or ""
        if repo_from_match:
            for part in str(repo_from_match).split(","):
                part = part.strip()
                if part:
                    scoped_repo.add(part)

        for r in m.get("turns", []) or []:
            if r.get("red_model_id"):
                scoped_comp.add(r["red_model_id"])
            if r.get("blue_model_id"):
                scoped_comp.add(r["blue_model_id"])
            if r.get("repo_name"):
                scoped_repo.add(r["repo_name"])

            cid = r.get("challenge_id")
            did = r.get("defense_id")
            if cid and cid in chal_idx and cid not in seen_challenges:
                seen_challenges.add(cid)
                info = chal_idx[cid]
                # Prefer turn composite id so buckets align with rating rows.
                mid = (
                    r.get("red_model_id")
                    or info.get("red_composite")
                    or info.get("red_model")
                )
                repo = info.get("repo") or r.get("repo_name") or ""
                slot = int(info.get("slot", 1))
                success_by_slot[(str(mid), str(repo), slot)] = info
                bucket = per_model[mid]
                bucket["gen_cost"] += float(info.get("gen_cost") or 0.0)
                bucket["val_cost"] += float(info.get("val_cost") or 0.0)
                bucket["in_tokens"] += int(info.get("in_tokens") or 0)
                bucket["out_tokens"] += int(info.get("out_tokens") or 0)
            if did and did in def_idx and did not in seen_defenses:
                seen_defenses.add(did)
                info = def_idx[did]
                mid = (
                    r.get("blue_model_id")
                    or info.get("blue_composite")
                    or info.get("blue_model")
                )
                bucket = per_model[mid]
                bucket["eval_cost"] += float(info.get("eval_cost") or 0.0)
                bucket["in_tokens"] += int(info.get("in_tokens") or 0)
                bucket["out_tokens"] += int(info.get("out_tokens") or 0)

    # When an explicit competitor/repo scope is provided (tournament field),
    # also bank every success in that field even if no match turn referenced it
    # (e.g. unused extra slots). Failures for those slots are handled below.
    if scope_competitors is not None or scope_repos is not None:
        for cid, info in chal_idx.items():
            if cid in seen_challenges:
                continue
            comp = info.get("red_composite") or info.get("red_model")
            repo = info.get("repo")
            if not comp or not repo:
                continue
            if scoped_comp and str(comp) not in scoped_comp:
                continue
            if scoped_repo and str(repo) not in scoped_repo:
                continue
            seen_challenges.add(cid)
            slot = int(info.get("slot", 1))
            success_by_slot[(str(comp), str(repo), slot)] = info
            bucket = per_model[comp]
            bucket["gen_cost"] += float(info.get("gen_cost") or 0.0)
            bucket["val_cost"] += float(info.get("val_cost") or 0.0)
            bucket["in_tokens"] += int(info.get("in_tokens") or 0)
            bucket["out_tokens"] += int(info.get("out_tokens") or 0)


    # Group failed records by (composite, repo, slot).
    failed_by_slot: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for info in (failed_idx or {}).values():
        comp = info.get("red_composite")
        repo = info.get("repo")
        if not comp or not repo:
            continue
        if scoped_comp and comp not in scoped_comp:
            continue
        if scoped_repo and repo not in scoped_repo:
            continue
        slot = int(info.get("slot", 1))
        failed_by_slot[(str(comp), str(repo), slot)].append(info)

    for key, fails in failed_by_slot.items():
        fail_gen = sum(float(f.get("gen_cost") or 0.0) for f in fails)
        fail_in = sum(int(f.get("in_tokens") or 0) for f in fails)
        fail_out = sum(int(f.get("out_tokens") or 0) for f in fails)
        succ = success_by_slot.get(key)
        comp = key[0]
        bucket = per_model[comp]
        if succ is not None:
            # Slot eventually succeeded: the success's generation_cost_usd
            # already rolls up every in-slot failed attempt — adding these
            # would double-count.
            continue
        # Failed-only slot: sum every attempt's cost.
        bucket["gen_cost"] += fail_gen
        bucket["in_tokens"] += fail_in
        bucket["out_tokens"] += fail_out

    totals = {
        "gen_cost": sum(v["gen_cost"] for v in per_model.values()),
        "val_cost": sum(v["val_cost"] for v in per_model.values()),
        "eval_cost": sum(v["eval_cost"] for v in per_model.values()),
        "in_tokens": sum(v["in_tokens"] for v in per_model.values()),
        "out_tokens": sum(v["out_tokens"] for v in per_model.values()),
    }
    return dict(per_model), totals


def _pairings_with_outcomes(state: dict, matches: list[dict]) -> list[list[tuple]]:
    """For each round, list of (model_a, model_b_or_None, pts_a, pts_b).

    Joins each pairing to its match by sorted-pair key, consuming matches in
    timestamp order so replays line up with successive rounds.
    """
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for m in sorted(matches, key=lambda x: x.get("timestamp", "")):
        key = tuple(sorted([m["model_a_id"], m["model_b_id"]]))
        by_pair[key].append(m)
    used: dict[tuple, int] = defaultdict(int)

    out: list[list[tuple]] = []
    for rnd in state.get("rounds") or []:
        row = []
        for pp in rnd:
            a = pp.get("model_a")
            b = pp.get("model_b")
            if a is None:
                continue
            if b is None:
                # Bye scores a full point for the seated player.
                row.append((a, None, 1.0, None))
                continue
            key = tuple(sorted([a, b]))
            ms = by_pair.get(key, [])
            idx = used[key]
            used[key] += 1
            m = ms[idx] if idx < len(ms) else None
            if m is None:
                row.append((a, b, None, None))
                continue
            outcome = m.get("outcome")
            if outcome == "draw":
                pa = pb = 0.5
            elif outcome == "model_a_wins":
                pa, pb = (1.0, 0.0) if m["model_a_id"] == a else (0.0, 1.0)
            elif outcome == "model_b_wins":
                pa, pb = (1.0, 0.0) if m["model_b_id"] == a else (0.0, 1.0)
            else:
                pa = pb = None
            row.append((a, b, pa, pb))
        out.append(row)
    return out


def _pre_round_points(rounds_w_out: list[list[tuple]]) -> list[dict[str, float]]:
    pts: dict[str, float] = defaultdict(float)
    snapshots: list[dict[str, float]] = []
    for rnd in rounds_w_out:
        snapshots.append(dict(pts))
        for a, b, pa, pb in rnd:
            if pa is not None:
                pts[a] += pa
            if b is not None and pb is not None:
                pts[b] += pb
    return snapshots


def _render_pairing(a: str, b: str | None, pa: float | None, pb: float | None) -> str:
    def cls(p):
        if p is None:
            return ""
        if p >= 1.0:
            return " winner"
        if p > 0:
            return " draw"
        return " loser"
    if b is None:
        return (
            '<div class="pairing">'
            f'<div class="bracket-model winner">{_esc(_participant_label(a))}</div>'
            '<div class="bracket-bye">BYE</div>'
            "</div>"
        )
    return (
        '<div class="pairing">'
        f'<div class="bracket-model{cls(pa)}">{_esc(_participant_label(a))}</div>'
        '<div class="bracket-vs">vs</div>'
        f'<div class="bracket-model{cls(pb)}">{_esc(_participant_label(b))}</div>'
        "</div>"
    )


def _render_bracket_html(state: dict, matches: list[dict], *, round_robin: bool = False) -> str:
    rounds = state.get("rounds") or []
    if not rounds:
        return '<div class="empty">(no rounds recorded)</div>'
    rounds_w_out = _pairings_with_outcomes(state, matches)

    if round_robin:
        # League rounds: one flat card per round, no point-bracket grouping
        # (brackets are a Swiss-system concept — round-robin pairings are fixed
        # up front and don't re-group by standings).
        cards = []
        for r_idx, rnd in enumerate(rounds_w_out):
            round_num = r_idx + 1
            inner = "".join(_render_pairing(a, b, pa, pb) for a, b, pa, pb in rnd)
            body = f'<div class="bracket-group">{inner}</div>'
            cards.append(
                f'<div class="round-card"><h3>Round {round_num}</h3>{body}</div>'
            )
        return f'<div class="rounds-scroll">{"".join(cards)}</div>'

    pre_pts = _pre_round_points(rounds_w_out)

    cards = []
    for r_idx, (rnd, pts) in enumerate(zip(rounds_w_out, pre_pts)):
        round_num = r_idx + 1
        if r_idx == 0:
            inner = "".join(_render_pairing(a, b, pa, pb) for a, b, pa, pb in rnd)
            body = f'<div class="bracket-group">{inner}</div>'
        else:
            buckets: dict[float, list[tuple]] = defaultdict(list)
            for a, b, pa, pb in rnd:
                if b is None:
                    key = pts.get(a, 0.0)
                else:
                    key = (pts.get(a, 0.0) + pts.get(b, 0.0)) / 2.0
                buckets[key].append((a, b, pa, pb))
            blocks = []
            for level in sorted(buckets.keys(), reverse=True):
                inner = "".join(
                    _render_pairing(a, b, pa, pb) for a, b, pa, pb in buckets[level]
                )
                blocks.append(
                    '<div class="bracket-group">'
                    f'<div class="bracket-label">{level:.1f} pts</div>'
                    f"{inner}"
                    "</div>"
                )
            body = "".join(blocks)
        cards.append(
            f'<div class="round-card"><h3>Round {round_num}</h3>{body}</div>'
        )
    return f'<div class="rounds-scroll">{"".join(cards)}</div>'


def _snapshot_to_dict(s) -> dict:
    return {
        "model_id": s.model_id,
        "elo": s.elo,
        "red_elo": s.red_elo,
        "blue_elo": s.blue_elo,
        "trueskill_mu": s.trueskill_mu,
        "trueskill_sigma": s.trueskill_sigma,
        "matches_played": s.matches_played,
        "bradley_terry": getattr(s, "bradley_terry", 0.0),
    }


def _filter_matches_for_tournament(data: dict, all_matches: list[dict]) -> list[dict]:
    """Pick matches that plausibly belong to this Swiss tournament.

    swiss_state_*.json doesn't store a tournament_id on each match, so we
    intersect by repo + selected model set. NOTE: when the same model pair
    appears in multiple tournaments this returns matches from all of them —
    use ``_claim_matches_per_tournament`` for a clean partition.
    """
    repo = data.get("repo_name")
    selected = set(data.get("selected_model_ids") or [])
    if not selected:
        return [m for m in all_matches if m.get("repo_name") == repo]
    return [
        m for m in all_matches
        if m.get("repo_name") == repo
        and m.get("model_a_id") in selected
        and m.get("model_b_id") in selected
    ]


def _claim_matches_per_tournament(
    states: list[dict], all_matches: list[dict]
) -> dict[str, list[dict]]:
    """Partition ``all_matches`` across tournaments by walking each Swiss
    state's pairings in chronological order and claiming the earliest
    unclaimed match for every (model pair, repo). A match can be claimed by
    at most one tournament, so repeated pairs across tournaments stop
    inflating later tournaments' match counts.

    Multi-repo matches store a comma-joined ``repo_name``; tournaments may
    also list ``repo_names``. Prefer exact ``repo_name`` match, then the same
    unordered repo-set, then pair-only so intake / expanded-repo states still
    claim their schedule.

    Returns ``{tournament_id: [match, ...]}``.
    """
    by_pair_repo: dict[tuple, list[dict]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in all_matches:
        pair = tuple(sorted([m["model_a_id"], m["model_b_id"]]))
        by_pair_repo[(pair, m.get("repo_name"))].append(m)
        by_pair[pair].append(m)
    # Newest first so the newest tournament gets the freshest match per pair.
    for v in by_pair_repo.values():
        v.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    for v in by_pair.values():
        v.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    def _repo_set(raw: str | None) -> frozenset[str]:
        if not raw:
            return frozenset()
        return frozenset(p.strip() for p in str(raw).split(",") if p.strip())

    used: set[str] = set()
    result: dict[str, list[dict]] = {}
    ordered = sorted(states, key=lambda s: s.get("timestamp", ""), reverse=True)
    for st in ordered:
        tid = st.get("tournament_id", "?")
        repo = st.get("repo_name")
        names = st.get("repo_names") or []
        want_set = frozenset(names) if names else _repo_set(repo)
        claimed: list[dict] = []
        for rnd in st.get("rounds") or []:
            for pp in rnd:
                a = pp.get("model_a")
                b = pp.get("model_b")
                if not a or not b:
                    continue
                pair = tuple(sorted([a, b]))
                candidates: list[dict] = list(by_pair_repo.get((pair, repo), []))
                if want_set:
                    for cand in by_pair.get(pair, []):
                        if cand in candidates:
                            continue
                        if _repo_set(cand.get("repo_name")) == want_set:
                            candidates.append(cand)
                # Pair-only fallback for intake/expanded schedules.
                for cand in by_pair.get(pair, []):
                    if cand not in candidates:
                        candidates.append(cand)
                for cand in candidates:
                    mid = cand.get("match_id")
                    if mid in used:
                        continue
                    used.add(mid)
                    claimed.append(cand)
                    break
        result[tid] = claimed
    return result


def _expand_harness_ablation_state_for_claim(state: dict) -> dict:
    """Expand harness pairings into composite competitor pairings for claiming.

    State pairings are bare harness ids; on-disk matches use
    ``model#harness`` composites, one sub-match per fixed model.
    """
    fixed = list(state.get("fixed_model_ids") or [])
    if not fixed:
        return state
    expanded_rounds: list[list[dict]] = []
    for rnd in state.get("rounds") or []:
        expanded: list[dict] = []
        for pp in rnd:
            ha = pp.get("model_a")
            hb = pp.get("model_b")
            if not ha or not hb:
                continue
            for mid in fixed:
                expanded.append(
                    {
                        "model_a": composite_id(mid, ha),
                        "model_b": composite_id(mid, hb),
                    }
                )
        expanded_rounds.append(expanded)
    out = dict(state)
    out["rounds"] = expanded_rounds
    return out


def _orient_submatch_to_harnesses(
    m: dict, h_a: str, h_b: str
) -> dict | None:
    """Return a copy of ``m`` with model_a/b forced to ``h_a``/``h_b`` sides.

    Turn red/blue ids are rewritten to bare harness ids so league/ELO tables
    key on harness participants. Returns ``None`` if the match is not a
    same-model duel between ``h_a`` and ``h_b``.
    """
    a_mid, a_h, _a_effort, _a_provider = split_composite_id(m.get("model_a_id") or "")
    b_mid, b_h, _b_effort, _b_provider = split_composite_id(m.get("model_b_id") or "")
    if not a_h or not b_h:
        return None
    if a_mid != b_mid:
        return None
    if {a_h, b_h} != {h_a, h_b}:
        return None
    flip = a_h == h_b and b_h == h_a
    score_a = float(m.get("model_a_total") or 0.0)
    score_b = float(m.get("model_b_total") or 0.0)
    outcome = m.get("outcome", MatchOutcome.DRAW.value)
    if flip:
        score_a, score_b = score_b, score_a
        if outcome == MatchOutcome.MODEL_A_WINS.value:
            outcome = MatchOutcome.MODEL_B_WINS.value
        elif outcome == MatchOutcome.MODEL_B_WINS.value:
            outcome = MatchOutcome.MODEL_A_WINS.value

    turns_out: list[dict] = []
    for t in m.get("turns") or []:
        t2 = dict(t)
        # Prefer explicit harness fields; fall back to splitting composite ids.
        red_h = t.get("red_harness_id") or split_composite_id(
            str(t.get("red_model_id") or "")
        )[1]
        blue_h = t.get("blue_harness_id") or split_composite_id(
            str(t.get("blue_model_id") or "")
        )[1]
        if flip:
            # Turns keep their adversarial roles; only the identity labels used
            # for GF attribution need to be harness ids.
            pass
        t2["red_model_id"] = red_h
        t2["blue_model_id"] = blue_h
        turns_out.append(t2)

    return {
        **m,
        "model_a_id": h_a,
        "model_b_id": h_b,
        "model_a_total": score_a,
        "model_b_total": score_b,
        "outcome": outcome,
        "turns": turns_out,
        "_source_match_id": m.get("match_id"),
        "_fixed_model_id": a_mid,
    }


def _fold_harness_ablation_matches(
    state: dict, sub_matches: list[dict]
) -> list[dict]:
    """Collapse same-model sub-matches into one ranking Bout per harness pairing.

    Mirrors the tournament's ``record_result`` aggregation: totals sum across
    fixed models; turns are concatenated with harness-keyed identities.
    """
    fixed = list(state.get("fixed_model_ids") or [])
    # Index claimed sub-matches by unsorted composite pair + repo set.
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in sub_matches:
        pair = tuple(sorted([m["model_a_id"], m["model_b_id"]]))
        by_pair[pair].append(m)

    folded: list[dict] = []
    used: set[str] = set()
    for rnd in state.get("rounds") or []:
        for pp in rnd:
            h_a = pp.get("model_a")
            h_b = pp.get("model_b")
            if not h_a or not h_b:
                continue
            oriented_subs: list[dict] = []
            for mid in fixed or [None]:
                # When fixed list empty, fall back to any scanned match.
                if mid is not None:
                    want = tuple(
                        sorted(
                            [
                                composite_id(mid, h_a),
                                composite_id(mid, h_b),
                            ]
                        )
                    )
                    candidates = by_pair.get(want, [])
                else:
                    candidates = []
                    for pair, ms in by_pair.items():
                        # Either side's harness half must be {h_a,h_b}
                        a_h = split_composite_id(pair[0])[1]
                        b_h = split_composite_id(pair[1])[1]
                        if {a_h, b_h} == {h_a, h_b}:
                            candidates.extend(ms)
                for cand in sorted(
                    candidates, key=lambda x: x.get("timestamp", "")
                ):
                    cid = cand.get("match_id")
                    if cid in used:
                        continue
                    ori = _orient_submatch_to_harnesses(cand, h_a, h_b)
                    if ori is None:
                        continue
                    used.add(cid)
                    oriented_subs.append(ori)
                    break

            if not oriented_subs:
                continue
            score_a = sum(float(s["model_a_total"]) for s in oriented_subs)
            score_b = sum(float(s["model_b_total"]) for s in oriented_subs)
            if abs(score_a - score_b) < 1e-9:
                outcome = MatchOutcome.DRAW.value
            elif score_a > score_b:
                outcome = MatchOutcome.MODEL_A_WINS.value
            else:
                outcome = MatchOutcome.MODEL_B_WINS.value
            all_turns: list[dict] = []
            for s in oriented_subs:
                all_turns.extend(s.get("turns") or [])
            cost = sum(float(s.get("total_cost_usd") or 0.0) for s in oriented_subs)
            duration = sum(
                float(s.get("duration_seconds") or 0.0) for s in oriented_subs
            )
            folded.append(
                {
                    "match_id": (
                        f"harness:{h_a}|{h_b}:"
                        f"{oriented_subs[0].get('_source_match_id') or oriented_subs[0]['match_id']}"
                    ),
                    "model_a_id": h_a,
                    "model_b_id": h_b,
                    "repo_name": oriented_subs[0].get("repo_name", ""),
                    "turns": all_turns,
                    "model_a_total": score_a,
                    "model_b_total": score_b,
                    "outcome": outcome,
                    "duration_seconds": duration,
                    "total_cost_usd": cost,
                    "timestamp": max(
                        (s.get("timestamp") or "") for s in oriented_subs
                    ),
                    "_sub_match_ids": [
                        s.get("_source_match_id") or s.get("match_id")
                        for s in oriented_subs
                    ],
                    "_fixed_models": [
                        s.get("_fixed_model_id") for s in oriented_subs
                    ],
                }
            )
    return folded


def _render_tournament_html(
    src: Path,
    out_dir: Path,
    matrix_pdf: Path | None,
    *,
    snapshots_by_model: dict,
    all_matches: list[dict],
    challenge_idx: dict[str, dict],
    defense_idx: dict[str, dict],
    failed_idx: dict[str, dict] | None = None,
    round_robin: bool = False,
    ranking_matches: list[dict] | None = None,
) -> Path:
    data = json.loads(src.read_text())
    tid = data.get("tournament_id", "report")
    out = out_dir / f"tournament_{tid}.html"

    def _rel(p: Path | None) -> str | None:
        if not p or not p.exists():
            return None
        try:
            return str(p.relative_to(out.parent))
        except ValueError:
            return str(p)

    matrix_rel = _rel(matrix_pdf)

    # Prefer caller-provided claimed matches (already scoped to this tournament).
    # In particular, harness-ablation states seat bare harness ids in
    # selected_model_ids while match files use composites — the legacy
    # _filter_matches_for_tournament would drop every match.
    if all_matches:
        matches = list(all_matches)
    else:
        matches = data.get("matches") or _filter_matches_for_tournament(
            data, all_matches
        )

    if data.get("final_ratings"):
        ratings = data["final_ratings"]
    else:
        selected = set(
            data.get("selected_harness_ids")
            or data.get("selected_model_ids")
            or []
        )
        ratings = [
            _snapshot_to_dict(snap) for mid, snap in snapshots_by_model.items()
            if not selected or mid in selected
        ]
        # Ensure every seated participant appears even with zero matches yet.
        have = {r["model_id"] for r in ratings}
        for pid in selected:
            if pid not in have:
                ratings.append(
                    {
                        "model_id": pid,
                        "elo": 1500.0,
                        "red_elo": 1500.0,
                        "blue_elo": 1500.0,
                        "trueskill_mu": 25.0,
                        "trueskill_sigma": 8.333,
                        "matches_played": 0,
                        "bradley_terry": 0.0,
                    }
                )

    # Scope generation costs to the tournament field, not only turns that
    # happened to reference a challenge_id (auto-wins omit it).
    scope_comp: set[str] = set()
    scope_repo: set[str] = set(data.get("repo_names") or [])
    if not scope_repo:
        rn = data.get("repo_name") or ""
        scope_repo = {p for p in rn.split(",") if p}
    if _is_harness_ablation():
        harnesses = list(
            data.get("selected_harness_ids")
            or data.get("selected_model_ids")
            or []
        )
        fixed = list(data.get("fixed_model_ids") or [])
        for mid in fixed:
            for h in harnesses:
                scope_comp.add(composite_id(mid, h))
    else:
        scope_comp = set(data.get("selected_model_ids") or [])

    # Usage aggregates over the detailed match list (sub-matches for ablation).
    per_model_usage, totals = _aggregate_usage(
        matches,
        challenge_idx,
        defense_idx,
        failed_idx,
        scope_competitors=scope_comp or None,
        scope_repos=scope_repo or None,
    )
    # Harness-ablation usage is keyed by composite competitors; roll up by
    # harness so the ranking table's cost columns line up with harness rows.
    if _is_harness_ablation() and per_model_usage:
        rolled: dict[str, dict] = {}
        for cid, u in per_model_usage.items():
            _mid, hid, _effort, _provider = split_composite_id(cid)
            if not hid:
                hid = cid
            bucket = rolled.setdefault(
                hid,
                {
                    "gen_cost": 0.0,
                    "val_cost": 0.0,
                    "eval_cost": 0.0,
                    "in_tokens": 0,
                    "out_tokens": 0,
                },
            )
            for k in bucket:
                bucket[k] += u.get(k, 0)
        per_model_usage = rolled


    out.write_text(_build_tournament_html(
        data, src, matrix_rel,
        matches=matches, ratings=ratings,
        per_model_usage=per_model_usage, totals=totals,
        challenge_idx=challenge_idx,
        round_robin=round_robin,
        ranking_matches=ranking_matches,
    ))
    return out


def _find_tournaments(tournaments_dir: Path, specific: Path | None) -> list[Path]:
    if specific:
        return [specific]
    return sorted(p for p in tournaments_dir.glob("*.json") if p.name != "checkpoint.json")


def _pick_active_tournament(
    tournaments_dir: Path, specific: Path | None, *, state_pattern: str
) -> Path | None:
    """Pick the tournament whose matches the global figures should be scoped to.

    Prefers ``--tournament`` if supplied; otherwise the most-recently-modified
    state file matching ``state_pattern`` so the report reflects the latest run
    rather than a union across historical tournaments (which would give models
    unequal match counts).
    """
    if specific:
        return specific
    candidates = sorted(
        tournaments_dir.glob(state_pattern),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _detect_report_format() -> str:
    """Map ``SWE_DUEL_REPORT_FORMAT`` env to an internal format key.

    * ``round-robin`` / ``round_robin`` → league-style RR over composites
    * ``harness-ablation-rr`` / variants → RR over bare harness participants
    * anything else / unset → Swiss (default)
    """
    raw = os.environ.get("SWE_DUEL_REPORT_FORMAT", "").strip().lower()
    if raw in {"round-robin", "round_robin"}:
        return "round-robin"
    if raw in {
        "harness-ablation-rr",
        "harness_ablation_rr",
        "harness-ablation",
        "harness_ablation",
    }:
        return "harness-ablation-rr"
    return "swiss"


def _state_glob_for_format(fmt: str) -> str:
    if fmt == "round-robin":
        return "round_robin_state_*.json"
    if fmt == "harness-ablation-rr":
        return "harness_ablation_rr_state_*.json"
    return "swiss_state_*.json"


def main() -> int:
    global _REPORT_FORMAT

    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-dir", default="data/matches")
    parser.add_argument("--tournaments-dir", default="data/tournaments")
    parser.add_argument("--out-dir", default="analysis/figures")
    parser.add_argument("--tournament", type=Path, default=None,
                        help="Render HTML for just this tournament JSON (default: all).")
    parser.add_argument("--skip-html", action="store_true",
                        help="Skip per-tournament HTML reports.")
    args = parser.parse_args()

    matches_dir = Path(args.matches_dir)
    tournaments_dir = Path(args.tournaments_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_matches(matches_dir)
    if not raw:
        print(f"No matches in {matches_dir}", file=sys.stderr)
        return 1

    kept_all, dropped = _dedupe(raw)
    print(f"[report] loaded {len(raw)} matches, kept {len(kept_all)}, dropped {len(dropped)} redundant")
    for d in dropped:
        print(
            f"  dropped {d['match_id']}  "
            f"({display_composite_id(d['model_a_id'])} vs "
            f"{display_composite_id(d['model_b_id'])}, {d['timestamp']})"
        )

    _REPORT_FORMAT = _detect_report_format()
    state_pattern = _state_glob_for_format(_REPORT_FORMAT)
    round_robin = _REPORT_FORMAT in {"round-robin", "harness-ablation-rr"}
    harness_ablation = _REPORT_FORMAT == "harness-ablation-rr"

    active_tournament = _pick_active_tournament(
        tournaments_dir, args.tournament, state_pattern=state_pattern
    )
    if active_tournament is None:
        print(
            f"[report] no {state_pattern} in {tournaments_dir}; cannot scope report to a tournament",
            file=sys.stderr,
        )
        return 1
    active_state = json.loads(active_tournament.read_text())

    # Partition matches across ALL tournaments first (chronological claim),
    # then take only the ones that belong to the active tournament. This
    # prevents matches from an earlier tournament that happened to use the
    # same model pair from inflating the active tournament's counts.
    all_state_paths = sorted(tournaments_dir.glob(state_pattern))
    all_states_raw = [json.loads(p.read_text()) for p in all_state_paths]
    if harness_ablation:
        claim_states = [
            _expand_harness_ablation_state_for_claim(s) for s in all_states_raw
        ]
    else:
        claim_states = all_states_raw
    claimed_by_tid = _claim_matches_per_tournament(claim_states, kept_all)
    active_tid = active_state.get("tournament_id")
    kept_raw = claimed_by_tid.get(active_tid, [])

    n_participants = len(
        active_state.get("selected_harness_ids")
        or active_state.get("selected_model_ids")
        or []
    )
    print(
        f"[report] format={_REPORT_FORMAT}  tournament={active_tournament.name} "
        f"(repo={active_state.get('repo_name')}, "
        f"participants={n_participants}): "
        f"{len(kept_raw)}/{len(kept_all)} sub-matches in scope"
    )

    # Ranking tables key on tournament participants. For harness ablation,
    # fold same-model sub-matches into one harness-vs-harness bout each.
    if harness_ablation:
        kept = _fold_harness_ablation_matches(active_state, kept_raw)
        print(
            f"[report] folded {len(kept_raw)} model sub-matches → "
            f"{len(kept)} harness pairings for rankings"
        )
        # HTML still lists the collected sub-matches so per-model detail is
        # available; rankings/ELO run on the folded set below.
        html_matches = kept_raw
    else:
        kept = kept_raw
        html_matches = kept_raw

    if not kept:
        print("[report] no matches belong to the active tournament", file=sys.stderr)
        return 1

    match_results = [_to_match_result(m) for m in kept]
    snapshots = compute_all_ratings(match_results)

    table_path = out_dir / "elo_table.tex"
    md_path = out_dir / "elo_table.md"
    pdf_path = out_dir / "elo_table.pdf"
    matrix_pdf_path = out_dir / "matchup_matrix.pdf"
    matrix_png_path = out_dir / "matchup_matrix.png"
    bracket_pdf_path = out_dir / "swiss_brackets.pdf"
    bracket_png_path = out_dir / "swiss_brackets.png"
    snapshot_list = list(snapshots.values())
    elo_by_model = {s.model_id: s.elo for s in snapshot_list}

    write_elo_table(snapshot_list, table_path)
    write_elo_markdown(snapshot_list, md_path)
    write_elo_pdf(snapshot_list, pdf_path)
    write_matchup_figure(
        kept, [matrix_pdf_path, matrix_png_path], elo_by_model=elo_by_model
    )
    swiss_md, combined_md, combined_pdf = write_swiss_tables(
        kept, elo_by_model, out_dir
    )
    write_bracket_figure([active_state], [bracket_pdf_path, bracket_png_path])

    for p in (table_path, md_path, pdf_path, matrix_pdf_path, matrix_png_path,
              swiss_md, combined_md, combined_pdf,
              bracket_pdf_path, bracket_png_path):
        print(f"[report] wrote {p}")

    if not args.skip_html:
        challenge_idx = _load_challenge_index(Path("data/challenge_bank/challenges"))
        defense_idx = _load_defense_index(Path("data/defenses"))
        failed_idx = _load_failed_challenge_index(
            Path("data/challenge_bank/failed_challenges")
        )
        # Rankings (ELO table / league) use folded harness matches; the
        # match-list section still receives the raw sub-matches so each
        # same-model duel stays inspectable. Ratings dict is harness-keyed after
        # fold, which is what the ranking table expects.
        out = _render_tournament_html(
            active_tournament, out_dir, matrix_png_path,
            snapshots_by_model=snapshots,
            all_matches=html_matches if harness_ablation else kept,
            challenge_idx=challenge_idx,
            defense_idx=defense_idx,
            failed_idx=failed_idx,
            round_robin=round_robin,
            ranking_matches=kept if harness_ablation else None,
        )
        print(f"[report] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
