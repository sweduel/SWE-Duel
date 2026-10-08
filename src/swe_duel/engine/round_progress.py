"""Live console UI for a parallel tournament round of Blue defenses.

`DefenseRoundReporter` holds a thread-safe, plain data model describing the
`MATCH → SUB-TURN` hierarchy of an in-flight round. Each sub-turn is one Blue
agent defending one challenge (plus its three follow-up test executions), which
is exactly the unit the round-level ``ThreadPoolExecutor`` runs concurrently
(see ``scripts/run_tournament.py::TournamentConsole._run_round_parallel``).

This mirrors the generation TUI (`swe_duel.challenge_bank.progress`): worker
threads only mutate the data model through a `SubturnHandle`; a
`rich.live.Live` view rebuilds a `rich.tree.Tree` from scratch on a timer, which
is the safe pattern for driving a Live display from background threads. A
running sub-turn shows the Blue agent's step-progress bar advancing toward its
budget, followed by glyphs for the three deterministic scoring phases
(regression / feature / bugfix) as each test suite runs. Finished sub-turns
collapse to a one-line verdict (blue_composite + outcome).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.tree import Tree


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAIL = "fail"
    CACHED = "cached"


# Scoring phases rendered (in order) under a running sub-turn, after the Blue
# agent's own step bar. These match TurnScorer's three test suites.
_PHASE_ORDER = ("regression", "feature", "bugfix")
_PHASE_LABEL = {
    "regression": "regression tests",
    "feature": "feature tests",
    "bugfix": "bug tests",
}

_STATUS_GLYPH = {
    Status.PENDING: ("•", "dim"),
    Status.RUNNING: ("⏳", "yellow"),
    Status.SUCCESS: ("✓", "green"),
    Status.FAIL: ("✗", "red"),
    Status.CACHED: ("⚡", "blue"),
}

_BAR_WIDTH = 24


# ── data model ──────────────────────────────────────────────


@dataclass
class PhaseState:
    name: str
    status: Status = Status.PENDING


@dataclass
class SubturnState:
    label: str               # e.g. "blue=opus repo=flask"
    blue_model: str = ""
    repo: str = ""
    status: Status = Status.PENDING
    step: int = 0
    max_steps: int = 0
    detail: str = ""
    phases: dict[str, PhaseState] = field(default_factory=dict)


@dataclass
class MatchState:
    label: str               # "A vs B"
    status: Status = Status.PENDING
    detail: str = ""
    subturns: dict[str, SubturnState] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


# ── reporter ────────────────────────────────────────────────


class DefenseRoundReporter:
    """Thread-safe state for the live round tree. Pass to `live()`."""

    def __init__(self, round_index: int = 0) -> None:
        self._lock = threading.RLock()
        self._round_index = round_index
        self._matches: dict[str, MatchState] = {}
        self._order: list[str] = []

    # ── registration / handles ──────────────────────────

    def register_match(self, match_key: str, label: str) -> None:
        with self._lock:
            if match_key not in self._matches:
                self._matches[match_key] = MatchState(label=label)
                self._order.append(match_key)

    def register_subturn(
        self,
        match_key: str,
        subturn_id: str,
        *,
        label: str,
        blue_model: str,
        repo: str,
        cached: bool = False,
    ) -> "SubturnHandle":
        """Register one sub-turn under a match. ``cached=True`` marks a defense
        resolved from the cache (no agent will run)."""
        with self._lock:
            m = self._matches.get(match_key)
            if m is None:
                m = MatchState(label=match_key)
                self._matches[match_key] = m
                self._order.append(match_key)
            if subturn_id not in m.subturns:
                st = SubturnState(
                    label=label,
                    blue_model=blue_model,
                    repo=repo,
                    status=Status.CACHED if cached else Status.PENDING,
                )
                m.subturns[subturn_id] = st
                m.order.append(subturn_id)
        return SubturnHandle(self, match_key, subturn_id)

    # ── mutators (called via SubturnHandle) ─────────────

    def _subturn(self, match_key: str, subturn_id: str) -> SubturnState | None:
        m = self._matches.get(match_key)
        if m is None:
            return None
        return m.subturns.get(subturn_id)

    def start_subturn(self, match_key: str, subturn_id: str) -> None:
        with self._lock:
            st = self._subturn(match_key, subturn_id)
            if st is not None and st.status != Status.CACHED:
                st.status = Status.RUNNING
            m = self._matches.get(match_key)
            if m is not None and m.status == Status.PENDING:
                m.status = Status.RUNNING

    def update_step(
        self, match_key: str, subturn_id: str, step: int, max_steps: int
    ) -> None:
        with self._lock:
            st = self._subturn(match_key, subturn_id)
            if st is None:
                return
            st.step = step
            st.max_steps = max_steps
            if st.status not in (Status.SUCCESS, Status.FAIL):
                st.status = Status.RUNNING

    def update_phase(
        self, match_key: str, subturn_id: str, phase: str, status: Status
    ) -> None:
        with self._lock:
            st = self._subturn(match_key, subturn_id)
            if st is None:
                return
            ph = st.phases.get(phase)
            if ph is None:
                ph = PhaseState(name=phase)
                st.phases[phase] = ph
            ph.status = status

    def finish_subturn(
        self,
        match_key: str,
        subturn_id: str,
        status: Status,
        detail: str = "",
    ) -> None:
        with self._lock:
            st = self._subturn(match_key, subturn_id)
            if st is None:
                return
            st.status = status
            st.detail = detail

    def finish_match(
        self, match_key: str, status: Status, detail: str = ""
    ) -> None:
        with self._lock:
            m = self._matches.get(match_key)
            if m is not None:
                m.status = status
                m.detail = detail

    # ── rendering ───────────────────────────────────────

    def __rich__(self) -> Tree:
        with self._lock:
            total = sum(len(m.subturns) for m in self._matches.values())
            done = sum(
                1
                for m in self._matches.values()
                for st in m.subturns.values()
                if st.status in (Status.SUCCESS, Status.FAIL, Status.CACHED)
            )
            root = Tree(
                f"Round {self._round_index} — defenses "
                f"[{done}/{total} sub-turns]",
                guide_style="dim",
            )
            for key in self._order:
                self._render_match(root, self._matches[key])
            return root

    def _render_match(self, root: Tree, match: MatchState) -> None:
        done = sum(
            1
            for st in match.subturns.values()
            if st.status in (Status.SUCCESS, Status.FAIL, Status.CACHED)
        )
        label = Text()
        _append_status(label, match.status)
        label.append(f"{match.label}  ", style="bold cyan")
        label.append(f"[{done}/{len(match.subturns)} sub-turns]", style="dim")
        if match.detail:
            label.append(f"  {match.detail}", style="dim")
        node = root.add(label)
        for sid in match.order:
            self._render_subturn(node, match.subturns[sid])

    def _render_subturn(self, node: Tree, st: SubturnState) -> None:
        label = Text()
        _append_status(label, st.status)
        label.append(f"{st.label}", style="white")
        if st.status in (Status.SUCCESS, Status.FAIL):
            if st.detail:
                label.append(f"  {st.detail}", style=_STATUS_GLYPH[st.status][1])
            node.add(label)
            return
        if st.status == Status.CACHED:
            label.append("  CACHED", style="blue")
            node.add(label)
            return
        if st.status == Status.PENDING:
            label.append("  PENDING", style="dim")
            node.add(label)
            return
        # Running: show the agent step bar + per-phase glyphs.
        snode = node.add(label)
        snode.add(_render_step_bar(st))
        phase_line = Text()
        any_phase = False
        for phase in _PHASE_ORDER:
            ph = st.phases.get(phase)
            if ph is None:
                continue
            any_phase = True
            glyph, style = _STATUS_GLYPH[ph.status]
            phase_line.append(f"{glyph} ", style=style)
            phase_line.append(f"{_PHASE_LABEL[phase]}  ", style="dim")
        if any_phase:
            snode.add(phase_line)


class SubturnHandle:
    """Per-sub-turn handle. Forwards all calls to the parent with keys bound."""

    def __init__(
        self, parent: DefenseRoundReporter, match_key: str, subturn_id: str
    ) -> None:
        self._parent = parent
        self._match_key = match_key
        self._subturn_id = subturn_id

    def start(self) -> None:
        self._parent.start_subturn(self._match_key, self._subturn_id)

    def step(self, step: int, max_steps: int) -> None:
        self._parent.update_step(
            self._match_key, self._subturn_id, step, max_steps
        )

    def phase(self, phase: str, status: "Status | str") -> None:
        if not isinstance(status, Status):
            status = Status(status)
        self._parent.update_phase(
            self._match_key, self._subturn_id, phase, status
        )

    def finish(self, status: Status, detail: str = "") -> None:
        self._parent.finish_subturn(
            self._match_key, self._subturn_id, status, detail
        )


# ── helpers ─────────────────────────────────────────────────


def _append_status(text: Text, status: Status) -> None:
    glyph, style = _STATUS_GLYPH[status]
    text.append(f"{glyph} ", style=style)


def _render_step_bar(st: SubturnState) -> Text:
    out = Text()
    out.append("blue agent steps: ", style="dim")
    max_steps = st.max_steps or 0
    step = max(0, st.step)
    if max_steps > 0:
        filled = min(_BAR_WIDTH, round(_BAR_WIDTH * step / max_steps))
    else:
        filled = 0
    out.append("[", style="dim")
    out.append("█" * filled, style="yellow")
    out.append("░" * (_BAR_WIDTH - filled), style="dim")
    out.append("] ", style="dim")
    out.append(f"{step}/{max_steps or '?'}", style="yellow")
    return out


@contextmanager
def live(
    reporter: DefenseRoundReporter,
    *,
    console: Console | None = None,
    refresh_per_second: int = 8,
) -> Iterator[Live]:
    """Context manager driving a `rich.live.Live` view of the reporter.

    The display auto-refreshes on a timer, so worker threads only mutate the
    reporter's data model and never call Live directly.
    """
    live_console = console or Console()
    with Live(
        reporter,
        console=live_console,
        refresh_per_second=refresh_per_second,
        auto_refresh=True,
        transient=False,
    ) as live_view:
        yield live_view


__all__ = [
    "DefenseRoundReporter",
    "SubturnHandle",
    "Status",
    "live",
]
