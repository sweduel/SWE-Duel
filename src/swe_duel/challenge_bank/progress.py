"""Live console UI for parallel challenge generation.

`GenerationReporter` holds a thread-safe, plain data model describing the
`MODEL → REPO → TARGET → ATTEMPT → PHASE` hierarchy of an in-flight generation
run. Worker threads (one per (model, repo) pool) mutate the model through a
`PoolReporter` handle; a `rich.live.Live` view re-renders the model on a timer.

The renderable rebuilds a `rich.tree.Tree` from scratch each refresh — workers
never touch rich widgets directly, which is the safe pattern for driving a Live
display from background threads. Finished targets/attempts collapse to a single
summary line; the currently-running attempt shows:

- a step-progress bar per agent phase (feature / bug / self-review);
- under it, a DefenseRoundReporter-style glyph row for validation gates
  (⏳ while a gate runs, then ✓ / ✗ as each finishes, accumulating). Gate
  glyphs stay until the next agent phase begins, at which point they
  collapse / are cleared so the UI moves cleanly into the next phase.
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


# Agent phases use a step bar (max_steps > 0).
_AGENT_PHASE_ORDER = ("feature", "bug", "self-review")
# Gate glyph order (matches RedGateValidator sequence). Accumulated row:
#   ✓ existing-test gate  ⏳ feature-test gate  …
# Finished gates stay as ✓/✗ until the next agent phase starts (then cleared).
_GATE_PHASE_ORDER = (
    "gate_diff_valid",
    "gate_existing_tests",
    "gate_feature_tests",
    "gate_bug_tests",
    "gate_complexity",
    "gate_lint",
    "gate_self_review",
)
_PHASE_LABEL = {
    "feature": "feature agent steps",
    "bug": "bug agent steps",
    "self-review": "self-review agent steps",
    "gate_diff_valid": "diff-valid gate",
    "gate_existing_tests": "existing-test gate",
    "gate_feature_tests": "feature-test gate",
    "gate_bug_tests": "bug-test gate",
    "gate_complexity": "complexity gate",
    "gate_lint": "lint gate",
    "gate_self_review": "self-review gate",
    # Reference-reviewer study phases (scripts/run_reference_reviewer.py):
    # one Blue defense run per challenge, scored by the sealed suites.
    "defense": "blue defense steps",
    "gate_regression": "regression-test gate",
    "gate_feature": "feature-retention gate",
    "gate_bugfix": "bug-test gate",
}



_STATUS_GLYPH = {
    Status.PENDING: ("•", "dim"),
    Status.RUNNING: ("⏳", "yellow"),
    Status.SUCCESS: ("✓", "green"),
    Status.FAIL: ("✗", "red"),
}

_BAR_WIDTH = 24


# ── data model ──────────────────────────────────────────────


@dataclass
class PhaseState:
    name: str
    step: int = 0
    max_steps: int = 0
    status: Status = Status.RUNNING


@dataclass
class AttemptState:
    number: int
    status: Status = Status.RUNNING
    detail: str = ""  # e.g. failure kind / gate summary
    phases: dict[str, PhaseState] = field(default_factory=dict)


@dataclass
class TargetState:
    index: int
    status: Status = Status.PENDING
    detail: str = ""
    attempts: list[AttemptState] = field(default_factory=list)


@dataclass
class PoolState:
    model_nick: str
    repo: str
    status: Status = Status.PENDING
    target_total: int = 0
    detail: str = ""
    harness: str = ""
    targets: list[TargetState] = field(default_factory=list)


# ── reporter ────────────────────────────────────────────────


class GenerationReporter:
    """Thread-safe state for the live generation tree. Pass to `live()`.

    ``title`` overrides the rendered root label for reuse by other pool-shaped
    studies (e.g. the Phase C stability re-runs) that drive the same
    pool → target → attempt → phase hierarchy.
    """

    def __init__(self, title: str = "Challenge generation") -> None:
        self._lock = threading.RLock()
        self._title = title
        self._pools: dict[tuple[str, str, str], PoolState] = {}
        self._order: list[tuple[str, str, str]] = []

    # ── registration / handles ──────────────────────────

    def register_pool(
        self, model_nick: str, repo: str, target_total: int, harness: str = ""
    ) -> "PoolReporter":
        """Register a (model, harness, repo) pool up front (rendered PENDING) and
        return a handle the worker thread uses to report progress. The harness is
        part of the key so the same model under different harnesses shows as
        separate rows."""
        key = (model_nick, harness, repo)
        with self._lock:
            if key not in self._pools:
                self._pools[key] = PoolState(
                    model_nick=model_nick, repo=repo,
                    target_total=target_total, harness=harness,
                )
                self._order.append(key)
        return PoolReporter(self, key)

    # ── mutators (called via PoolReporter) ──────────────

    def _pool(self, key: tuple[str, str]) -> PoolState:
        return self._pools[key]

    def set_target_total(self, key: tuple[str, str], total: int) -> None:
        with self._lock:
            self._pool(key).target_total = total

    def start_pool(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._pool(key).status = Status.RUNNING

    def finish_pool(
        self, key: tuple[str, str], status: Status, detail: str = ""
    ) -> None:
        with self._lock:
            p = self._pool(key)
            p.status = status
            p.detail = detail

    def start_target(self, key: tuple[str, str], index: int) -> None:
        with self._lock:
            p = self._pool(key)
            if not any(t.index == index for t in p.targets):
                p.targets.append(TargetState(index=index, status=Status.RUNNING))
            if p.status == Status.PENDING:
                p.status = Status.RUNNING

    def finish_target(
        self, key: tuple[str, str], index: int, status: Status, detail: str = ""
    ) -> None:
        with self._lock:
            t = self._target(key, index)
            if t is not None:
                t.status = status
                t.detail = detail

    def start_attempt(
        self, key: tuple[str, str], target_index: int, number: int
    ) -> None:
        with self._lock:
            t = self._target(key, target_index)
            if t is None:
                return
            if not any(a.number == number for a in t.attempts):
                t.attempts.append(AttemptState(number=number))

    def update_phase(
        self,
        key: tuple[str, str],
        target_index: int,
        attempt_number: int,
        phase: str,
        step: int,
        max_steps: int,
    ) -> None:
        """Update a running phase under an attempt.

        Two modes (same callback signature agents already use):

        - *Agent steps* (``max_steps > 0``): progress bar for feature / bug /
          self-review. Also **clears any gate glyph rows** so the previous
          validation batch collapses when the next agent phase begins —
          mirrors DefenseRoundReporter collapsing scoring phases at
          sub-turn boundaries.
        - *Gate status* (``max_steps == 0``): glyph-mode row under the agent
          bar. ``step`` encodes status like the tournament ``phase(name,
          status)`` API:
            * ``step == 0`` → ⏳ RUNNING
            * ``step == 1`` → ✓ SUCCESS (hourglass turns into tick)
            * ``step == 2`` → ✗ FAIL

          Finished gates stay visible until an agent phase starts again
          (or the attempt finishes / collapses).
        """
        with self._lock:
            a = self._attempt(key, target_index, attempt_number)
            if a is None:
                return

            # Agent step tick: bar + drop previous gate batch.
            if max_steps > 0:
                for name in list(a.phases.keys()):
                    if name.startswith("gate_"):
                        a.phases.pop(name, None)
                ph = a.phases.get(phase)
                if ph is None:
                    ph = PhaseState(name=phase)
                    a.phases[phase] = ph
                ph.step = step
                ph.max_steps = max_steps
                ph.status = Status.RUNNING
                return

            # Gate glyph mode (max_steps == 0).
            ph = a.phases.get(phase)
            if ph is None:
                ph = PhaseState(name=phase, max_steps=0)
                a.phases[phase] = ph
            ph.max_steps = 0
            if step <= 0:
                ph.status = Status.RUNNING
            elif step == 1:
                ph.status = Status.SUCCESS
            else:
                ph.status = Status.FAIL
            ph.step = step



    def finish_attempt(
        self,
        key: tuple[str, str],
        target_index: int,
        attempt_number: int,
        status: Status,
        detail: str = "",
    ) -> None:
        with self._lock:
            a = self._attempt(key, target_index, attempt_number)
            if a is None:
                return
            a.status = status
            a.detail = detail
            for ph in a.phases.values():
                if ph.status == Status.RUNNING:
                    ph.status = status

    # ── lookups (caller holds lock) ─────────────────────

    def _target(
        self, key: tuple[str, str], index: int
    ) -> TargetState | None:
        for t in self._pool(key).targets:
            if t.index == index:
                return t
        return None

    def _attempt(
        self, key: tuple[str, str], target_index: int, number: int
    ) -> AttemptState | None:
        t = self._target(key, target_index)
        if t is None:
            return None
        for a in t.attempts:
            if a.number == number:
                return a
        return None

    # ── rendering ───────────────────────────────────────

    def __rich__(self) -> Tree:
        with self._lock:
            root = Tree(self._title, guide_style="dim")
            for key in self._order:
                self._render_pool(root, self._pools[key])
            return root

    def _render_pool(self, root: Tree, pool: PoolState) -> None:
        done = sum(
            1 for t in pool.targets if t.status in (Status.SUCCESS, Status.FAIL)
        )
        label = Text()
        _append_status(label, pool.status)
        label.append(f"{pool.model_nick} ", style="bold cyan")
        if pool.harness:
            label.append(f"⟨{pool.harness}⟩ ", style="magenta")
        label.append(f"— {pool.repo}  ", style="cyan")
        label.append(f"[{done}/{pool.target_total} targets]", style="dim")
        if pool.detail:
            label.append(f"  {pool.detail}", style="dim")
        node = root.add(label)

        if pool.status == Status.PENDING:
            return
        for target in pool.targets:
            self._render_target(node, target)

    def _render_target(self, node: Tree, target: TargetState) -> None:
        label = Text()
        _append_status(label, target.status)
        label.append(f"Target {target.index}", style="bold")
        if target.status in (Status.SUCCESS, Status.FAIL):
            verdict = "SUCCESS" if target.status == Status.SUCCESS else "FAIL"
            label.append(f"  {verdict}", style=_STATUS_GLYPH[target.status][1])
            if target.detail:
                label.append(f" — {target.detail}", style="dim")
            node.add(label)  # collapsed: no attempt children
            return
        if target.status == Status.PENDING:
            label.append("  PENDING", style="dim")
            node.add(label)
            return
        tnode = node.add(label)
        for attempt in target.attempts:
            self._render_attempt(tnode, attempt)

    def _render_attempt(self, tnode: Tree, attempt: AttemptState) -> None:
        label = Text()
        _append_status(label, attempt.status)
        label.append(f"Attempt {attempt.number}", style="bold")
        if attempt.status in (Status.SUCCESS, Status.FAIL):
            verdict = "SUCCESS" if attempt.status == Status.SUCCESS else "FAIL"
            label.append(f"  {verdict}", style=_STATUS_GLYPH[attempt.status][1])
            if attempt.detail:
                label.append(f" — {attempt.detail}", style="dim")
            tnode.add(label)  # collapsed
            return
        anode = tnode.add(label)
        # Agent step bars (feature → bug → self-review), then any custom agent
        # phases from other pool-shaped studies (e.g. the reference reviewer's
        # "defense" run) in insertion order — same bar renderer, own labels.
        for phase in _AGENT_PHASE_ORDER:
            ph = attempt.phases.get(phase)
            if ph is None or ph.max_steps <= 0:
                continue
            anode.add(_render_agent_phase(ph))
        for phase, ph in attempt.phases.items():
            if phase in _AGENT_PHASE_ORDER or ph.max_steps <= 0:
                continue
            anode.add(_render_agent_phase(ph))
        # Gate glyphs on one line under the bars, DefenseRoundReporter-style:
        #   ✓ existing-test gate  ⏳ feature-test gate  …
        gate_line = Text()
        any_gate = False
        seen: set[str] = set()
        for phase in _GATE_PHASE_ORDER:
            ph = attempt.phases.get(phase)
            if ph is None or ph.max_steps != 0:
                continue
            seen.add(phase)
            any_gate = True
            glyph, style = _STATUS_GLYPH[ph.status]
            gate_line.append(f"{glyph} ", style=style)
            gate_line.append(f"{_PHASE_LABEL.get(phase, phase)}  ", style="dim")
        # Any custom gate names not in the ordered list.
        for phase, ph in attempt.phases.items():
            if phase in seen or ph.max_steps != 0:
                continue
            any_gate = True
            glyph, style = _STATUS_GLYPH[ph.status]
            gate_line.append(f"{glyph} ", style=style)
            gate_line.append(f"{_PHASE_LABEL.get(phase, phase)}  ", style="dim")
        if any_gate:
            anode.add(gate_line)




class PoolReporter:
    """Per-(model, repo) handle. All calls are forwarded to the parent reporter
    with this pool's key bound, so the generator never juggles keys."""

    def __init__(self, parent: GenerationReporter, key: tuple[str, str]) -> None:
        self._parent = parent
        self._key = key

    def set_target_total(self, total: int) -> None:
        self._parent.set_target_total(self._key, total)

    def start_pool(self) -> None:
        self._parent.start_pool(self._key)

    def finish_pool(self, status: Status, detail: str = "") -> None:
        self._parent.finish_pool(self._key, status, detail)

    def start_target(self, index: int) -> None:
        self._parent.start_target(self._key, index)

    def finish_target(self, index: int, status: Status, detail: str = "") -> None:
        self._parent.finish_target(self._key, index, status, detail)

    def start_attempt(self, target_index: int, number: int) -> None:
        self._parent.start_attempt(self._key, target_index, number)

    def update_phase(
        self,
        target_index: int,
        attempt_number: int,
        phase: str,
        step: int,
        max_steps: int,
    ) -> None:
        self._parent.update_phase(
            self._key, target_index, attempt_number, phase, step, max_steps
        )

    def finish_attempt(
        self,
        target_index: int,
        attempt_number: int,
        status: Status,
        detail: str = "",
    ) -> None:
        self._parent.finish_attempt(
            self._key, target_index, attempt_number, status, detail
        )


# ── helpers ─────────────────────────────────────────────────


def _append_status(text: Text, status: Status) -> None:
    glyph, style = _STATUS_GLYPH[status]
    text.append(f"{glyph} ", style=style)


def _render_agent_phase(ph: PhaseState) -> Text:
    """Step progress bar for feature / bug / self-review agent runs."""
    out = Text()
    out.append(f"{_PHASE_LABEL.get(ph.name, ph.name)}: ", style="dim")
    max_steps = ph.max_steps or 0
    step = max(0, ph.step)
    if max_steps > 0:
        filled = min(_BAR_WIDTH, round(_BAR_WIDTH * step / max_steps))
    else:
        filled = 0
    bar_style = {
        Status.RUNNING: "yellow",
        Status.SUCCESS: "green",
        Status.FAIL: "red",
    }.get(ph.status, "yellow")
    out.append("[", style="dim")
    out.append("█" * filled, style=bar_style)
    out.append("░" * (_BAR_WIDTH - filled), style="dim")
    out.append("] ", style="dim")
    out.append(f"{step}/{max_steps or '?'}", style=bar_style)
    return out


def _render_gate_phase(ph: PhaseState) -> Text:
    """Unused standalone helper kept for tests; rendering uses the accum row."""
    out = Text()
    glyph, style = _STATUS_GLYPH[ph.status]
    out.append(f"{glyph} ", style=style)
    out.append(_PHASE_LABEL.get(ph.name, ph.name), style="dim")
    return out



@contextmanager
def live(
    reporter: GenerationReporter,
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
    "GenerationReporter",
    "PoolReporter",
    "Status",
    "live",
]
