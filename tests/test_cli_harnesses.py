"""Unit tests for Codex and Claude Code harnesses.

Pure JSONL → AgentTrajectory mapping and registry wiring — no Docker, no
network, no real CLI binaries required.
"""

from __future__ import annotations

import json

import pytest

from swe_duel.config import ModelConfig


def _model_config(**overrides) -> ModelConfig:
    base = dict(
        model_id="z-ai/glm-5.3-flash",
        temperature=0.0,
        max_tokens=2048,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(autouse=True)
def _stub_openrouter_pricing(monkeypatch):
    """Keep cost tracking hermetic: unit tests must not query OpenRouter.

    Stubs the cached per-token pricing used by the CLI harnesses' cost
    fallback with $0.000001 / $0.000002 per input/output token (the legacy
    static rates from config/models.yaml), so cost assertions are exact.
    """
    from swe_duel.agents.harness import cost_tracking

    monkeypatch.setattr(
        cost_tracking,
        "fetch_openrouter_pricing",
        lambda model_id, provider="": cost_tracking.TokenPricing(
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )


def test_registry_includes_cli_harnesses(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness import HARNESS_IDS, get_harness, harness_display_name

    assert HARNESS_IDS == (
        "mini-swe-agent",
        "openhands",
        "codex",
        "claude-code",
    )
    mc = _model_config()
    for hid, expected_cls_name in (
        ("codex", "CodexHarness"),
        ("claude-code", "ClaudeCodeHarness"),
    ):
        h = get_harness(hid, mc)
        assert h.harness_id == hid
        assert type(h).__name__ == expected_cls_name
    assert harness_display_name("codex") == "Codex"
    assert harness_display_name("claude-code") == "Claude Code"


def test_codex_jsonl_steps_and_exit(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.cli_container import StreamState
    from swe_duel.agents.harness.codex import CodexHarness

    harness = CodexHarness(_model_config())
    state = StreamState(limit=10)
    prompt = "Please explore the repo and add a feature"
    pending = {
        "thought": "",
        "action": "",
        "next_observation": prompt,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t1"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "r0",
                    "type": "reasoning",
                    "text": "I should list the files first",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls -la",
                    "status": "in_progress",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "i1",
                    "type": "command_execution",
                    "command": "ls -la",
                    "aggregated_output": "file.py\n",
                    "status": "completed",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "i2",
                    "type": "agent_message",
                    "text": "listed the repo; done",
                },
            }
        ),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_input_tokens": 10,
                },
            }
        ),
    ]
    for line in lines:
        harness._handle_json_line(
            line,
            state=state,
            pending=pending,
            role_label="red",
            verbose=False,
            emit=lambda _s: None,
            step_callback=None,
            run_limit=10,
        )
    # Flush trailing agent_message onto the step's thought.
    trailing = str(pending.get("thought") or "").strip()
    if trailing and state.steps and not str(state.steps[-1].get("thought") or "").strip():
        last = dict(state.steps[-1])
        last["thought"] = trailing
        state.steps[-1] = last
    elif trailing:
        # keep parity with harness end-of-run flush path
        pending["thought"] = trailing
        if state.steps and not state.steps[-1].get("thought"):
            state.steps[-1] = {**state.steps[-1], "thought": trailing}

    assert state.step == 1
    assert len(state.steps) == 1
    assert state.steps[0]["observation"] == prompt
    assert "ls -la" in state.steps[0]["action"]
    assert "list" in state.steps[0]["thought"].lower() or "listed" in state.steps[0]["thought"].lower()
    # tool output is queued for the next step, not the current step
    assert "file.py" not in state.steps[0]["observation"]
    assert state.total_input_tokens == 100
    assert state.total_output_tokens == 20
    # turn.completed usage (100 in / 20 out / 10 cached) is priced with the
    # stubbed OpenRouter per-token rates: (100-10)*1e-6 + 10*0 + 20*2e-6.
    assert state.total_cost_usd == pytest.approx(1.3e-4)

    es = CodexHarness._exit_status(state, None, missing=[])
    assert es == "Submitted"
    es2 = CodexHarness._exit_status(state, None, missing=["_swe-duel/feature.md"])
    assert es2 == "LimitsExceeded"
    state.wall_hit = True
    assert CodexHarness._exit_status(state, None, missing=[]) == "WallClockTimeout"


def test_claude_code_stream_json_steps(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.claude_code import ClaudeCodeHarness, _parse_content_blocks
    from swe_duel.agents.harness.cli_container import StreamState

    thoughts, actions = _parse_content_blocks(
        [
            {"type": "text", "text": "I will list files"},
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "ls -la"},
            },
        ]
    )
    assert thoughts == ["I will list files"]
    assert actions == ["Bash: ls -la"]

    harness = ClaudeCodeHarness(_model_config(model_id="z-ai/glm-5.3-flash"))
    state = StreamState(limit=10)
    prompt = "Start: review the project"
    pending: dict = {
        "thought": "",
        "action": "",
        "next_observation": prompt,
        "session_id": None,
    }

    lines = [
        json.dumps(
            {
                "type": "assistant",
                "session_id": "sess-1",
                "message": {
                    "content": [
                        {"type": "text", "text": "exploring"},
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "cat README.md"},
                        },
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 12},
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "session_id": "sess-1",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "README contents here",
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "session_id": "sess-1",
                "total_cost_usd": 0.0123,
                "usage": {"input_tokens": 50, "output_tokens": 12},
                "is_error": False,
                "result": "done",
            }
        ),
    ]
    last_sid = None
    for line in lines:
        sid = harness._handle_json_line(
            line,
            state=state,
            pending=pending,
            role_label="blue",
            verbose=False,
            emit=lambda _s: None,
            step_callback=None,
            run_limit=10,
        )
        if sid:
            last_sid = sid
    assert last_sid == "sess-1"
    assert state.step == 1
    assert state.steps[0]["observation"] == prompt
    assert "Bash: cat README.md" in state.steps[0]["action"]
    assert "exploring" in state.steps[0]["thought"]
    # tool result seeds the *next* step observation, not this step
    assert "README contents" not in state.steps[0]["observation"]
    assert pending["next_observation"] == "README contents here"
    # The CLI's own `total_cost_usd` (its Anthropic price-table estimate,
    # wrong under the OpenRouter gateway) must be IGNORED; the per-message
    # usage (50 in / 12 out) is priced with the stubbed OpenRouter rates:
    # 50*1e-6 + 12*2e-6 = 7.4e-5 — NOT 0.0123.
    assert state.total_cost_usd == pytest.approx(7.4e-5)
    assert state.total_cost_usd != pytest.approx(0.0123)
    assert ClaudeCodeHarness._exit_status(state, None, missing=[]) == "Submitted"


def test_claude_code_prices_result_usage_when_steps_have_no_tokens(monkeypatch):
    """Real Claude Code under the OpenRouter gateway reports 0/0 per-message
    usage and only the result envelope carries aggregate tokens — the total
    must then be priced from the result usage with OpenRouter rates."""
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.claude_code import ClaudeCodeHarness
    from swe_duel.agents.harness.cli_container import StreamState

    harness = ClaudeCodeHarness(_model_config())
    state = StreamState(limit=10)
    pending: dict = {
        "thought": "",
        "action": "",
        "next_observation": "task",
        "session_id": None,
    }
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "session_id": "sess-2",
                "message": {
                    "content": [{"type": "text", "text": "working"}],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "session_id": "sess-2",
                "total_cost_usd": 0.05,  # CLI's own (wrong) estimate — ignored
                "usage": {"input_tokens": 1014, "output_tokens": 16},
                "is_error": False,
                "result": "done",
            }
        ),
    ]
    for line in lines:
        harness._handle_json_line(
            line,
            state=state,
            pending=pending,
            role_label="blue",
            verbose=False,
            emit=lambda _s: None,
            step_callback=None,
            run_limit=10,
        )
    # No per-step token attribution → the cumulative result usage is priced:
    # 1014*1e-6 + 16*2e-6 = 1.046e-3.
    assert state.total_input_tokens == 1014
    assert state.total_output_tokens == 16
    assert state.total_cost_usd == pytest.approx(1.046e-3)


def test_cli_container_host_mode_runs_command(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.cli_container import CliContainer

    with CliContainer(
        docker_image=None,
        host_workspace=tmp_path,
        harness_id="codex",
        env={},
    ) as c:
        result = c.run_command(["bash", "-c", "echo hello && pwd"], timeout=10)
    assert result.return_code == 0
    assert "hello" in result.stdout


def test_is_docker_infra_failure_detects_daemon_errors():
    from swe_duel.agents.harness.cli_container import CliRunResult, is_docker_infra_failure

    assert is_docker_infra_failure(
        CliRunResult(1, "", "Error response from daemon: No such container: swe-duel-x")
    )
    assert is_docker_infra_failure(
        CliRunResult(1, "", "Error response from daemon: container abc is not running")
    )
    assert not is_docker_infra_failure(CliRunResult(1, "", "claude: command not found"))
    assert not is_docker_infra_failure(None)


def test_ensure_running_recreates_dead_container(tmp_path, monkeypatch):
    """Dead sleep-infinity container must be rebuilt (OOM / --rm) before exec."""
    from swe_duel.agents.harness.cli_container import CliContainer

    c = CliContainer(
        docker_image="swe-duel-mock:latest",
        host_workspace=tmp_path,
        harness_id="claude-code",
        env={},
    )
    c.container_name = "swe-duel-mock-claude-code-dead"
    c._started = True

    inspect_calls: list[str] = []
    run_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        run_calls.append(list(cmd))
        if cmd[:3] == ["docker", "inspect", "-f"]:
            inspect_calls.append(cmd[4])
            # First inspect: dead. After start, subsequent inspects report running.
            running = "true" if any(r[:2] == ["docker", "run"] for r in run_calls) else "false"
            class R:
                returncode = 0
                stdout = running + "\n"
                stderr = ""
            return R()
        if cmd[:2] == ["docker", "rm"]:
            class R:
                returncode = 0
                stdout = ""
                stderr = ""
            return R()
        if cmd[:2] == ["docker", "run"]:
            class R:
                returncode = 0
                stdout = "newid\n"
                stderr = ""
            return R()
        # mkdir preflight etc. via docker exec after start
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(
        "swe_duel.agents.harness.cli_container.subprocess.run",
        fake_run,
    )
    # Avoid real docker exec threads in start()'s mkdir helper.
    monkeypatch.setattr(
        CliContainer,
        "run_command",
        lambda self, argv, **kw: type(
            "R",
            (),
            {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "killed_for_budget": False,
            },
        )(),
    )

    assert c.is_running() is False
    restarted = c.ensure_running()
    assert restarted is True
    assert c._started is True
    assert c.container_name is not None
    assert c.container_name != "swe-duel-mock-claude-code-dead"
    assert any(cmd[:2] == ["docker", "run"] for cmd in run_calls)
    # Second call is a no-op while "running".
    assert c.ensure_running() is False


def test_codex_config_toml_uses_openrouter(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness.codex import CodexHarness

    cfg = CodexHarness(_model_config())._config_toml()
    assert 'model_provider = "openrouter"' in cfg
    assert "openrouter.ai/api/v1" in cfg
    assert 'model = "z-ai/glm-5.3-flash"' in cfg
    assert 'wire_api = "responses"' in cfg
    assert 'wire_api = "chat"' not in cfg


# ── participant-identity capability validation ───────────────────────────


def test_identity_capability_tables():
    from swe_duel.agents.harness import (
        HARNESS_EFFORT_SUPPORT,
        harness_supported_efforts,
        harness_supports_provider,
    )

    # litellm-backed harnesses forward any effort string via extra_body.
    assert harness_supported_efforts("mini-swe-agent") is None
    assert harness_supported_efforts("openhands") is None
    # Codex can only express its fixed config enum; Claude Code none at all.
    assert HARNESS_EFFORT_SUPPORT["codex"] == frozenset(
        {"minimal", "low", "medium", "high", "xhigh"}
    )
    assert harness_supported_efforts("claude-code") == frozenset()
    # Provider pinning needs request-body control → litellm harnesses only.
    assert harness_supports_provider("mini-swe-agent")
    assert harness_supports_provider("openhands")
    assert not harness_supports_provider("codex")
    assert not harness_supports_provider("claude-code")


def test_harness_rejects_identity_it_cannot_honour(monkeypatch):
    """A selected effort/provider the harness would silently drop must fail
    fast at construction, not corrupt the recorded competitor identity."""
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness import get_harness

    base = _model_config()
    # codex cannot express "max" (not in its config enum).
    with pytest.raises(ValueError, match="cannot express reasoning effort"):
        get_harness("codex", base.with_selection("max", ""))
    # codex cannot pin providers at all.
    with pytest.raises(ValueError, match="cannot pin an OpenRouter provider"):
        get_harness("codex", base.with_selection("", "fireworks"))
    # claude-code has neither control.
    with pytest.raises(ValueError, match="cannot express reasoning effort"):
        get_harness("claude-code", base.with_selection("high", ""))
    with pytest.raises(ValueError, match="cannot pin an OpenRouter provider"):
        get_harness("claude-code", base.with_selection("", "cloudflare"))
    # mini-swe-agent (litellm extra_body) accepts arbitrary effort strings and
    # provider slugs.
    h = get_harness("mini-swe-agent", base.with_selection("high", "cloudflare"))
    assert h.model_config.reasoning_effort == "high"
    assert h.model_config.provider == "cloudflare"


def test_openrouter_body_extras_builder():
    from swe_duel.agents.harness.base import openrouter_body_extras

    assert openrouter_body_extras(_model_config()) == {}
    sel = _model_config().with_selection("high", "cloudflare")
    assert openrouter_body_extras(sel) == {
        "reasoning": {"effort": "high"},
        "provider": {"order": ["cloudflare"], "allow_fallbacks": False},
    }
    # Provider-only selection sends only the routing object.
    pv_only = _model_config().with_selection("", "fireworks")
    assert openrouter_body_extras(pv_only) == {
        "provider": {"order": ["fireworks"], "allow_fallbacks": False},
    }


def test_codex_config_toml_embeds_reasoning_effort(monkeypatch):
    monkeypatch.setenv("SWE_DUEL_OPENROUTER_API_KEY", "dummy")
    from swe_duel.agents.harness import get_harness

    h = get_harness("codex", _model_config().with_selection("xhigh", ""))
    toml = h._config_toml()
    assert 'model_reasoning_effort = "xhigh"' in toml
    # Default selection sends no effort line.
    h2 = get_harness("codex", _model_config())
    assert "model_reasoning_effort" not in h2._config_toml()
