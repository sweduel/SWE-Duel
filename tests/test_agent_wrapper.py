"""Phase 4: agent wrapper tests.

Integration tests launch real mini-swe-agent sessions and make LLM calls
through OpenRouter. They are gated on `SWE_DUEL_OPENROUTER_API_KEY` being set.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
from minisweagent.exceptions import FormatError, LimitsExceeded

from swe_duel.agents.agent_wrapper import AgentWrapper
from swe_duel.config import ModelConfig

from conftest import load_integration_models, integration_model_id

from conftest import FIXTURES_DIR
MOCK_REPO_DIR = FIXTURES_DIR / "mock_repo"

HAS_API_KEY = bool(os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"))
_skip_no_key = pytest.mark.skipif(not HAS_API_KEY, reason="SWE_DUEL_OPENROUTER_API_KEY not set")

_INTEGRATION_MODELS = load_integration_models() if HAS_API_KEY else []


def _model_config(**overrides) -> ModelConfig:
    # Default to the cheapest model in config/models.yaml; the parameterized
    # integration tests override this with every yaml model anyway.
    base = dict(
        model_id="z-ai/glm-5.3-flash",
        temperature=0.0,
        max_tokens=2048,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    shutil.copytree(MOCK_REPO_DIR, ws)
    return ws


# ── Unit test: no API call ─────────────────────────────────


class TestBuildAgentConfig:
    def test_build_agent_config(self, tmp_path: Path, monkeypatch, record):
        monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy-test-key")
        mc = _model_config(temperature=0.3)
        wrapper = AgentWrapper(mc)

        cfg = wrapper._build_agent_config(tmp_path, max_steps=7)
        record("agent_config", cfg)

        assert set(cfg.keys()) == {"model", "env", "agent"}

        assert cfg["model"]["model_name"] == "openrouter/z-ai/glm-5.3-flash"
        assert cfg["model"]["model_kwargs"]["temperature"] == 0.3
        assert cfg["model"]["model_kwargs"]["max_tokens"] == 2048

        assert cfg["env"]["cwd"] == str(tmp_path)
        assert isinstance(cfg["env"]["env"], dict)
        assert isinstance(cfg["env"]["timeout"], int)

        assert cfg["agent"]["step_limit"] == 7
        assert isinstance(cfg["agent"]["system_template"], str)
        assert cfg["agent"]["system_template"]
        assert isinstance(cfg["agent"]["instance_template"], str)
        assert "{{task}}" in cfg["agent"]["instance_template"]

    def test_build_config_forwards_truncation_aware_format_error_template(
        self, tmp_path: Path, monkeypatch, record
    ):
        """mini.yaml's format_error_template must reach the model config.

        The default ``"{{ error }}"`` template only says "No tool calls found
        in the response", so a response truncated by the output-token limit
        before emitting its tool call (finish_reason="length", tool_calls=null
        — the classic reasoning-model-at-max-effort failure) never teaches the
        model to respond more concisely. The run then dies with three
        consecutive FormatErrors (RepeatedFormatError) long before the step
        limit.
        """
        monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy-test-key")
        wrapper = AgentWrapper(_model_config())

        cfg = wrapper._build_agent_config(tmp_path, max_steps=7)
        record("format_error_template", cfg["model"]["format_error_template"])

        template = cfg["model"]["format_error_template"]
        assert "output token limit" in template
        assert "finish_reason" in template

        # The truncation branch renders actionable feedback for a
        # finish_reason="length" response with no tool calls.
        from jinja2 import StrictUndefined, Template

        rendered = Template(template, undefined=StrictUndefined).render(
            error="No tool calls found in the response.",
            actions=[],
            has_tool_calls=False,
            finish_reason="length",
        )
        assert "reached the output token limit" in rendered
        assert "Respond more concisely" in rendered

    def test_init_requires_api_key(self, monkeypatch, record):
        monkeypatch.delenv("SWE_DUEL_OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SWE_DUEL_OPENROUTER_API_KEY") as exc_info:
            AgentWrapper(_model_config())
        record("error_message", str(exc_info.value))


class _FakeAgentConfig:
    def __init__(self) -> None:
        self.step_limit = 12
        self.cost_limit = 3.0


class _FakeAgentModel:
    def format_message(self, *, role: str, content: str) -> dict:
        return {"role": role, "content": content}


class _FakeAgent:
    """Emulates DefaultAgent's message handling + query() limit checks.

    ``script`` items are either exceptions to raise from step() (mirroring
    mini-swe-agent's InterruptAgentFlow carriers) or strings meaning a
    successful step appending an assistant message.
    """

    def __init__(self, script: list) -> None:
        self.config = _FakeAgentConfig()
        self.model = _FakeAgentModel()
        self.messages: list[dict] = [
            {"role": "exit", "content": "LimitsExceeded", "extra": {}}
        ]
        self.n_calls = 10
        self.cost = 0.5
        self.script = list(script)
        self.step_calls = 0

    def run(self, task: str) -> None:
        pass

    def add_messages(self, *messages: dict) -> None:
        self.messages.extend(messages)

    def step(self) -> list[dict]:
        # Mirror DefaultAgent.query(): raise LimitsExceeded once the step
        # budget is exhausted, BEFORE consuming a model call.
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        self.step_calls += 1
        action = self.script.pop(0) if self.script else "ok"
        if isinstance(action, Exception):
            raise action
        return self.add_messages({"role": "assistant", "content": "ok"})


_MISSING = ["_swe-duel/metadata.json::bug_type"]


class TestRecoveryLoopInterruptHandling:
    """Recovery turns must not discard mini-swe-agent's interrupt messages.

    DefaultAgent.run() adds FormatError feedback / exit notices carried on
    interrupt exceptions to the conversation; _drive_agent drives step()
    directly, so it must do the same or the model never learns why its
    response was rejected and every recovery turn repeats the failure.
    """

    def _drive(self, monkeypatch, script: list, max_recovery_turns: int = 3) -> _FakeAgent:
        monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy-test-key")
        wrapper = AgentWrapper(_model_config())
        agent = _FakeAgent(script)
        wrapper._drive_agent(
            agent,
            "task",
            verbose=False,
            role_label="test",
            completion_check=lambda: _MISSING,
            reminder_builder=lambda items: "Please write: " + ", ".join(items),
            max_recovery_turns=max_recovery_turns,
            max_wall_seconds=None,
            started=time.monotonic(),
            emit=lambda line: None,
        )
        return agent

    def test_format_error_feedback_is_added_and_retried_in_turn(self, monkeypatch, record):
        agent = self._drive(monkeypatch, [FormatError({"role": "user", "content": "FEEDBACK"}), "ok"])
        user_contents = [m.get("content") for m in agent.messages if m["role"] == "user"]
        record("user_contents", user_contents)
        # The FormatError's feedback message joined the conversation …
        assert "FEEDBACK" in user_contents
        # … and the turn kept stepping (the retried step succeeded) rather
        # than discarding the turn on the first format error.
        assert any(m["role"] == "assistant" for m in agent.messages)
        assert agent.step_calls >= 2

    def test_persistent_format_errors_are_bounded_and_leave_exit_status(self, monkeypatch, record):
        agent = self._drive(monkeypatch, [FormatError({"role": "user", "content": "FB"})] * 50)
        record("roles", [m["role"] for m in agent.messages])
        # 3 recovery turns x 6-step budget; each FormatError retried in-turn.
        assert agent.step_calls == 18
        feedback = [m for m in agent.messages if m["role"] == "user" and m["content"] == "FB"]
        assert len(feedback) == 18
        # The run ends with an honest exit status instead of an empty one
        # (which callers report as "agent exit_status=unknown").
        assert agent.messages[-1]["role"] == "exit"
        assert agent.messages[-1]["extra"]["exit_status"] == "LimitsExceeded"

    def test_non_interrupt_exception_ends_turn_without_messages(self, monkeypatch, record):
        agent = self._drive(monkeypatch, [RuntimeError("API exploded")] * 2, max_recovery_turns=2)
        user_contents = [m.get("content") for m in agent.messages if m["role"] == "user"]
        record("user_contents", user_contents)
        # Only the per-turn reminders; no fabricated messages for a plain
        # (non-interrupt) exception. Each turn ended on the first error.
        assert user_contents == ["Please write: " + ", ".join(_MISSING)] * 2
        assert agent.step_calls == 2


# ── Integration tests: real LLM calls ──────────────────────


@pytest.mark.integration
@_skip_no_key
@pytest.mark.parametrize("model_cfg", _INTEGRATION_MODELS, ids=[integration_model_id(m) for m in _INTEGRATION_MODELS])
class TestAgentRun:
    def test_run_simple_task(self, workspace: Path, record, model_cfg):
        mc = _model_config(
            model_id=model_cfg.model_id,
            temperature=0.0,
            max_tokens=model_cfg.max_tokens,
        )
        record("model_id", mc.model_id)
        wrapper = AgentWrapper(mc)

        task = (
            "Inside the current working directory, count how many files end with the "
            ".py extension (search recursively). Then create a directory named _swe-duel "
            "(if it does not already exist) and write that count as a single integer "
            "into _swe-duel/output.txt. When finished, submit by running the exact "
            "command: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        )
        trajectory = wrapper.run(workspace, task, max_steps=15)

        output_file = workspace / "_swe-duel" / "output.txt"
        content = output_file.read_text().strip() if output_file.exists() else ""
        record("task", task)
        record("output_file_exists", output_file.exists())
        record("output_content", content)
        record("trajectory", {
            "model_id": trajectory.model_id,
            "total_steps": trajectory.total_steps,
            "total_input_tokens": trajectory.total_input_tokens,
            "total_output_tokens": trajectory.total_output_tokens,
            "total_cost_usd": trajectory.total_cost_usd,
            "duration_seconds": trajectory.duration_seconds,
            "steps": trajectory.steps,
        })
        assert output_file.exists(), "agent did not create _swe-duel/output.txt"
        assert content, "_swe-duel/output.txt is empty"
        assert trajectory.total_steps >= 1

    def test_run_returns_trajectory(self, workspace: Path, record, model_cfg):
        mc = _model_config(model_id=model_cfg.model_id)
        record("model_id", mc.model_id)
        wrapper = AgentWrapper(mc)

        trajectory = wrapper.run(
            workspace,
            "Run `ls` once, then immediately submit by executing: "
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            max_steps=5,
        )

        record("trajectory", {
            "model_id": trajectory.model_id,
            "total_steps": trajectory.total_steps,
            "total_input_tokens": trajectory.total_input_tokens,
            "total_output_tokens": trajectory.total_output_tokens,
            "total_cost_usd": trajectory.total_cost_usd,
            "duration_seconds": trajectory.duration_seconds,
            "steps": trajectory.steps,
        })
        assert trajectory.model_id == mc.model_id
        assert trajectory.total_steps > 0
        assert trajectory.duration_seconds > 0
        assert isinstance(trajectory.steps, list)
        assert trajectory.total_input_tokens >= 0
        assert trajectory.total_output_tokens >= 0
        assert trajectory.total_cost_usd >= 0.0

    def test_run_respects_max_steps(self, workspace: Path, record, model_cfg):
        mc = _model_config(model_id=model_cfg.model_id)
        record("model_id", mc.model_id)
        wrapper = AgentWrapper(mc)

        max_steps = 3
        task = (
            "Systematically read every Python file in the workspace one file per "
            "command, and for each one print its full contents with `cat`. Do not "
            "submit until you have read every .py file individually."
        )
        trajectory = wrapper.run(workspace, task, max_steps=max_steps)

        record("max_steps_config", max_steps)
        record("trajectory", {
            "model_id": trajectory.model_id,
            "total_steps": trajectory.total_steps,
            "total_input_tokens": trajectory.total_input_tokens,
            "total_output_tokens": trajectory.total_output_tokens,
            "total_cost_usd": trajectory.total_cost_usd,
            "duration_seconds": trajectory.duration_seconds,
            "steps": trajectory.steps,
        })
        assert trajectory.total_steps <= max_steps
