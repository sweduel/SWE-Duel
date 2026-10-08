"""Unit tests for the OpenHands harness trajectory parsing.

These exercise the pure event/metrics → AgentTrajectory mapping with synthetic
stand-in objects (no live agent, no Docker, no network), plus the harness
registry wiring. Integration with a real OpenHands run is covered separately
and gated on SWE_DUEL_OPENROUTER_API_KEY + a built repo image.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swe_duel.config import ModelConfig


def _model_config(**overrides) -> ModelConfig:
    base = dict(
        model_id="deepseek/deepseek-v4-flash-0731",
        temperature=0.0,
        max_tokens=2048,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ── registry wiring (no API key needed beyond construction guard) ──────────


def test_registry_returns_distinct_harnesses(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness import HARNESS_IDS, get_harness, harness_display_name

    assert HARNESS_IDS == (
        "mini-swe-agent",
        "openhands",
        "codex",
        "claude-code",
    )
    mc = _model_config()
    mini = get_harness("mini-swe-agent", mc)
    oh = get_harness("openhands", mc)
    assert mini.harness_id == "mini-swe-agent"
    assert oh.harness_id == "openhands"
    assert type(mini) is not type(oh)
    assert harness_display_name("openhands") == "OpenHands"
    assert get_harness("codex", mc).harness_id == "codex"
    assert get_harness("claude-code", mc).harness_id == "claude-code"

    with pytest.raises(ValueError):
        get_harness("nope", mc)


def test_agent_wrapper_shim_is_mini_swe(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.agent_wrapper import AgentWrapper, CONTAINER_WORKSPACE
    from swe_duel.agents.harness.mini_swe import MiniSweHarness

    assert AgentWrapper is MiniSweHarness
    assert CONTAINER_WORKSPACE == "/workspace"
    assert AgentWrapper(_model_config()).harness_id == "mini-swe-agent"


# ── synthetic OpenHands events → AgentTrajectory ───────────────────────────


class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class ActionEvent:
    """Stand-in whose class name matches the real openhands ActionEvent."""

    def __init__(self, eid, ts, thought, command):
        self.id = eid
        self.timestamp = ts
        self.thought = [_FakeText(thought)]
        self.reasoning_content = None
        self.action = SimpleNamespace(command=command)
        self.tool_name = "terminal"


class ObservationEvent:
    def __init__(self, action_id, content):
        self.action_id = action_id
        self.tool_call_id = None
        self.observation = SimpleNamespace(content=content)
        self.error = None


def _action_event(eid, ts, thought, command):
    return ActionEvent(eid, ts, thought, command)


def _obs_event(action_id, content):
    return ObservationEvent(action_id, content)


class _TokenUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.cache_read_tokens = 0


class _Cost:
    def __init__(self, cost):
        self.cost = cost


class _Metrics:
    def __init__(self, usages, costs, accumulated_cost):
        self.token_usages = usages
        self.costs = costs
        self.accumulated_cost = accumulated_cost


class _FakeConversation:
    def __init__(self, events, metrics):
        self.state = SimpleNamespace(events=events, stats=None)
        self._metrics = metrics

    @property
    def conversation_stats(self):
        outer = self

        class _Stats:
            def get_combined_metrics(self):
                return outer._metrics

        return _Stats()


def test_extract_trajectory_maps_events_and_metrics(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    harness = OpenHandsHarness(_model_config())

    events = [
        _action_event("a1", "2026-06-18T12:00:00+00:00", "look around", "ls -la"),
        _obs_event("a1", "file listing here"),
        _action_event("a2", "2026-06-18T12:00:05+00:00", "edit file", "echo hi > f"),
        _obs_event("a2", "done"),
    ]
    metrics = _Metrics(
        usages=[_TokenUsage(100, 20), _TokenUsage(150, 30)],
        costs=[_Cost(0.01), _Cost(0.02)],
        accumulated_cost=0.03,
    )
    conv = _FakeConversation(events, metrics)

    traj = harness._extract_trajectory(
        conv,
        duration_seconds=5.0,
        exit_status="Submitted",
        task_prompt="TASK PROMPT HERE",
    )

    assert traj.total_steps == 2
    assert traj.total_input_tokens == 250
    assert traj.total_output_tokens == 50
    assert traj.total_cost_usd == pytest.approx(0.03)
    assert traj.exit_status == "Submitted"
    assert traj.model_id == "deepseek/deepseek-v4-flash-0731"

    s0, s1 = traj.steps
    assert s0["thought"] == "look around"
    assert s0["action"] == "ls -la"
    # Preceding-observation model (mini-swe): seed step 1 with the task prompt;
    # tool result lands on the next step.
    assert s0["observation"] == "TASK PROMPT HERE"
    assert s1["observation"] == "file listing here"
    assert s1["thought"] == "edit file"
    assert s0["input_tokens"] == 100 and s0["output_tokens"] == 20
    assert s0["cost_usd"] == pytest.approx(0.01)
    # Second step's per-step time is the timestamp delta (5s); first is 0.
    assert s0["step_seconds"] == pytest.approx(0.0)
    assert s1["step_seconds"] == pytest.approx(5.0)
    assert s1["cum_seconds"] == pytest.approx(5.0)


def test_extract_trajectory_falls_back_to_priced_cost(monkeypatch):
    """When metrics carry no costs, per-step cost is derived from the cached
    OpenRouter per-token pricing (stubbed here — no network)."""
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness import cost_tracking
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    monkeypatch.setattr(
        cost_tracking,
        "fetch_openrouter_pricing",
        lambda model_id, provider="": cost_tracking.TokenPricing(
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )

    harness = OpenHandsHarness(_model_config())
    events = [_action_event("a1", "2026-06-18T12:00:00+00:00", "t", "cmd")]
    metrics = _Metrics(
        usages=[_TokenUsage(1000, 500)], costs=[], accumulated_cost=0.0
    )
    traj = harness._extract_trajectory(
        _FakeConversation(events, metrics), duration_seconds=1.0, exit_status=""
    )
    # 1000*1e-6 + 500*2e-6 = 0.001 + 0.001 = 0.002
    assert traj.steps[0]["cost_usd"] == pytest.approx(0.002)
    assert traj.total_cost_usd == pytest.approx(0.002)


class _StatusEnum:
    """Mimic ConversationExecutionStatus members used by _exit_status."""

    FINISHED = "finished"
    ERROR = "error"
    STUCK = "stuck"
    PAUSED = "paused"


def test_exit_status_normalisation(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    def _conv(status):
        return SimpleNamespace(state=SimpleNamespace(execution_status=status))

    es = OpenHandsHarness._exit_status
    enum = _StatusEnum
    # Wall-clock always wins.
    assert es(_conv(enum.FINISHED), enum, True, missing=["x"]) == "WallClockTimeout"
    # Artifact-completeness wins over raw status: all present → Submitted,
    # regardless of the loop's final status.
    assert es(_conv(enum.FINISHED), enum, False, missing=[]) == "Submitted"
    assert es(_conv(enum.ERROR), enum, False, missing=[]) == "Submitted"
    # Artifacts still missing → classify by why we stopped.
    assert es(_conv(enum.ERROR), enum, False, missing=["x"]) == "LimitsExceeded"
    assert es(_conv(enum.STUCK), enum, False, missing=["x"]) == "Stuck"
    # Paused on the per-run budget with work still missing → LimitsExceeded;
    # paused but everything present → Submitted.
    assert (
        es(_conv(enum.PAUSED), enum, False, budget_paused=True, missing=["x"])
        == "LimitsExceeded"
    )
    assert (
        es(_conv(enum.PAUSED), enum, False, budget_paused=True, missing=[])
        == "Submitted"
    )
    # budget_paused flag alone (e.g. status reads back IDLE) still classifies.
    assert (
        es(_conv("idle"), enum, False, budget_paused=True, missing=["x"])
        == "LimitsExceeded"
    )


class _RecoveryConversation:
    """Fake conversation that exhausts its budget on the initial run (raising),
    then completes the missing artifact on the first recovery turn.

    Records how many run() calls happened so the test can assert the recovery
    loop actually ran after the budget-exhausted initial pass.
    """

    def __init__(self, callbacks):
        self._cb = callbacks[0]
        self.runs = 0
        self.messages: list[str] = []
        self.paused = 0
        self.state = SimpleNamespace(events=[], stats=None, execution_status="finished")
        self._artifact_written = False

    @property
    def conversation_stats(self):
        return SimpleNamespace(get_combined_metrics=lambda: None)

    def send_message(self, msg):
        self.messages.append(msg)

    def run(self):
        self.runs += 1
        if self.runs == 1:
            # Emit enough ActionEvents to blow the per-run ceiling, then raise
            # like the SDK's MaxIterationsReached.
            for i in range(30):
                self._cb(ActionEvent(f"i{i}", None, "t", "cmd"))
            raise RuntimeError("MaxIterationsReached: limit (30)")
        # Recovery turn: write the artifact then finish.
        self._artifact_written = True
        self._cb(ActionEvent(f"r{self.runs}", None, "fix", "write metadata"))

    def pause(self):
        self.paused += 1

    def close(self):
        pass


def _install_sdk_stubs(monkeypatch, conv_factory):
    """Stub the heavy SDK/tools imports performed inside run() and route
    Conversation(...) through conv_factory(callbacks)."""
    import swe_duel.agents.harness.openhands as oh_mod

    captured = {}

    def _fake_conversation(agent, workspace, callbacks, **kw):
        conv = conv_factory(callbacks)
        captured["conv"] = conv
        return conv

    fake_sdk = SimpleNamespace(
        LLM=lambda **k: object(),
        Agent=lambda **k: object(),
        Conversation=_fake_conversation,
        Tool=lambda **k: object(),
        ConversationExecutionStatus=_StatusEnum,
    )
    monkeypatch.setitem(__import__("sys").modules, "openhands.sdk", fake_sdk)
    fake_tools = SimpleNamespace(
        FileEditorTool=SimpleNamespace(name="str_replace_editor"),
        TerminalTool=SimpleNamespace(name="terminal"),
    )
    monkeypatch.setitem(__import__("sys").modules, "openhands.tools", fake_tools)
    monkeypatch.setattr(oh_mod, "_silence_openhands_io", lambda: None)
    return captured


def test_initial_budget_exhaustion_still_runs_recovery(monkeypatch, tmp_path):
    """A MaxIterationsReached on the initial run must NOT skip the recovery loop,
    and each recovery turn gets its own fresh per-run budget."""
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    captured = _install_sdk_stubs(monkeypatch, _RecoveryConversation)
    harness = OpenHandsHarness(_model_config())

    def _completion_check():
        conv = captured.get("conv")
        if conv is not None and conv._artifact_written:
            return []
        return ["_swe-duel/metadata.json"]

    traj = harness.run(
        workspace_path=tmp_path,
        task_prompt="do the thing",
        role_label="probe",
        completion_check=_completion_check,
        reminder_builder=lambda missing: f"please write {missing}",
        max_recovery_turns=3,
        max_steps=30,
        docker_image=None,
        verbose=False,
        console_echo=False,
    )

    conv = captured["conv"]
    # Initial run + at least one recovery run actually executed.
    assert conv.runs >= 2, "recovery loop did not run after budget exhaustion"
    # A reminder message was sent for the recovery turn.
    assert any("please write" in m for m in conv.messages)
    # The artifact got written during recovery → exit is a clean Submitted.
    assert traj.exit_status == "Submitted"


class _EarlyFinishConversation:
    """Fake conversation that calls run() and returns 'finished' after only a
    couple of steps WITHOUT raising — i.e. the agent invoked its `finish` tool
    early, before writing artifacts. It keeps doing this until it has been
    re-prompted ``finish_after`` times, then writes the artifact.

    Used to assert the initial pass re-prompts/resumes within the initial
    budget instead of dropping straight into the 6-step recovery loop.
    """

    def __init__(self, callbacks, finish_after=2, steps_per_run=2):
        self._cb = callbacks[0]
        self._finish_after = finish_after
        self._steps_per_run = steps_per_run
        self.runs = 0
        self.messages: list[str] = []
        self.paused = 0
        self.state = SimpleNamespace(events=[], stats=None, execution_status="finished")
        self._artifact_written = False

    @property
    def conversation_stats(self):
        return SimpleNamespace(get_combined_metrics=lambda: None)

    def send_message(self, msg):
        self.messages.append(msg)

    def run(self):
        self.runs += 1
        for i in range(self._steps_per_run):
            self._cb(ActionEvent(f"r{self.runs}s{i}", None, "t", "cmd"))
        # Finish early (no raise). After enough re-prompts, write the artifact.
        if self.runs >= self._finish_after:
            self._artifact_written = True

    def pause(self):
        self.paused += 1

    def close(self):
        pass


def test_early_finish_resumes_within_initial_budget(monkeypatch, tmp_path):
    """If the agent finishes early (finish tool) with artifacts missing, the
    harness must re-prompt and resume on the SAME initial budget — not jump to
    the constrained recovery turns. The recovery loop is a fallback only after
    the initial budget is spent."""
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    captured = _install_sdk_stubs(
        monkeypatch, lambda cbs: _EarlyFinishConversation(cbs, finish_after=3)
    )
    harness = OpenHandsHarness(_model_config())

    def _completion_check():
        conv = captured.get("conv")
        if conv is not None and conv._artifact_written:
            return []
        return ["_swe-duel/metadata.json"]

    reminders: list[str] = []
    denominators: list[int] = []

    traj = harness.run(
        workspace_path=tmp_path,
        task_prompt="do the thing",
        role_label="probe",
        completion_check=_completion_check,
        reminder_builder=lambda missing: reminders.append("r") or "keep going",
        step_callback=lambda step, limit: denominators.append(limit),
        max_recovery_turns=3,
        max_steps=30,
        docker_image=None,
        verbose=False,
        console_echo=False,
    )

    conv = captured["conv"]
    # The agent finished early twice (runs 1,2) then completed on run 3 — all
    # within the initial budget (3*2=6 steps « 30), so NO recovery turn was
    # needed. exit must be a clean Submitted.
    assert conv.runs == 3, f"expected 3 initial-budget runs, got {conv.runs}"
    assert traj.exit_status == "Submitted"
    # Re-prompts happened during the initial pass (2 early finishes).
    assert len(reminders) >= 2
    # Crucially: every step ran under the INITIAL budget (denominator stayed at
    # max_steps). A recovery turn would have grown the denominator past 30, so
    # this asserts the early-finish resumes happened in the initial pass — not
    # in the constrained recovery loop.
    assert denominators, "step_callback was never called"
    assert max(denominators) == 30, (
        f"denominator grew past initial budget ({max(denominators)}); "
        "early finish was wrongly handled as a recovery turn"
    )


def test_event_helpers():
    from swe_duel.agents.harness.openhands import (
        _event_action_str,
        _event_observation,
        _event_thought,
    )

    ev = _action_event("a1", "2026-06-18T12:00:00+00:00", "reason text", "run cmd")
    assert _event_thought(ev) == "reason text"
    assert _event_action_str(ev) == "run cmd"

    obs = _obs_event("a1", "obs body")
    assert _event_observation(obs) == "obs body"
