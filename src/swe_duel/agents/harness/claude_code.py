"""Anthropic Claude Code harness (non-interactive ``claude -p``).

Runs the Claude Code CLI **inside** the per-repo Docker image so the agent
shares the validation-gate environment. Model calls are routed through
OpenRouter by setting ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` (Claude
Code's documented LLM-gateway path) and passing ``--model`` from the arena's
model registry.

Contract: same ``AgentHarness.run`` signature as mini-swe / OpenHands — step
budget + 3×6 recovery turns, live step callback, wall-clock, trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from swe_duel.agents.harness.base import AgentHarness
from swe_duel.agents.harness.cli_container import (
    _RECOVERY_BUDGET_PER_TURN,
    CliContainer,
    CliRunResult,
    StreamState,
    is_docker_infra_failure,
    openrouter_api_key,
    truncate,
)
from swe_duel.agents.harness.cost_tracking import fallback_cost_for, response_reported_cost
from swe_duel.config import ModelConfig
from swe_duel.models import AgentTrajectory


class ClaudeCodeHarness(AgentHarness):
    """Drives ``claude -p`` (headless / print mode) behind AgentHarness."""

    harness_id = "claude-code"

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)

    def run(
        self,
        workspace_path: Path,
        task_prompt: str,
        *,
        verbose: bool = True,
        role_label: str = "agent",
        completion_check: Callable[[], list[str]] | None = None,
        reminder_builder: Callable[[list[str]], str] | None = None,
        max_recovery_turns: int = 3,
        max_wall_seconds: float | None = None,
        max_steps: int = 30,
        log_file: Path | None = None,
        step_callback: Callable[[int, int], None] | None = None,
        console_echo: bool = True,
        docker_image: str | None = None,
        preserve_paths: list[str] | None = None,
        login_shell: bool = True,
    ) -> AgentTrajectory:
        api_key = openrouter_api_key()
        # OpenRouter's Anthropic-compatible endoint. ANTHROPIC_AUTH_TOKEN is the
        # gateway token; ANTHROPIC_API_KEY is also set because some Claude Code
        # code paths still require it to be non-empty under bare mode.
        env = {
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_API_KEY": api_key,
            "OPENROUTER_API_KEY": api_key,
            "SWE_DUEL_OPENROUTER_API_KEY": api_key,
            "NO_COLOR": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

        log_fp = None
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_file, "a", encoding="utf-8", buffering=1)
            log_fp.write(
                f"\n═══ agent start role={role_label} harness=claude-code "
                f"model={self.model_config.model_id} {self._identity_suffix()} "
                f"cwd={workspace_path} "
                f"ts={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )

        def _emit(line: str) -> None:
            if console_echo:
                print(line, flush=True)
            if log_fp is not None:
                try:
                    log_fp.write(line + "\n")
                    log_fp.flush()
                except Exception:
                    pass

        state = StreamState(limit=int(max_steps))
        wall = float(max_wall_seconds) if max_wall_seconds and max_wall_seconds > 0 else None
        started = time.monotonic()
        last_result: CliRunResult | None = None
        session_id: str | None = None

        with CliContainer(
            docker_image=docker_image,
            host_workspace=workspace_path,
            harness_id=self.harness_id,
            preserve_paths=preserve_paths,
            env=env,
            login_shell=login_shell,
        ) as container:
            self._preflight(container, emit=_emit, role_label=role_label)
            if verbose:
                wall_msg = f" wall_limit={wall:.0f}s" if wall else ""
                _emit(
                    f"[{role_label}] ═══ agent start harness=claude-code "
                    f"model={self.model_config.model_id} {self._identity_suffix()} "
                f"cwd={workspace_path} "
                    f"step_limit={max_steps}{wall_msg}"
                )

            last_result, session_id = self._run_claude(
                container,
                prompt=task_prompt,
                resume_session=None,
                state=state,
                role_label=role_label,
                verbose=verbose,
                emit=_emit,
                step_callback=step_callback,
                wall=wall,
                started=started,
                run_limit=int(max_steps),
            )
            # Initial-budget early-finish loop (mirrors OpenHands / Codex): if
            # `claude -p` returns before artefacts exist and budget remains,
            # re-prompt within max_steps before opening the +6 recovery turns.
            if completion_check is not None and reminder_builder is not None:
                stagnant = 0
                while True:
                    if state.wall_hit or (
                        wall is not None and time.monotonic() - started > wall
                    ):
                        state.wall_hit = True
                        break
                    missing = completion_check() or []
                    if not missing:
                        break
                    if state.step >= int(max_steps) or state.budget_hit:
                        break
                    if verbose:
                        _emit(
                            f"[{role_label}] ↺ agent finished early at step "
                            f"{state.step}/{max_steps} with missing={missing}; "
                            "resuming within initial budget"
                        )
                    state.budget_hit = False
                    step_before = state.step
                    last_result, session_id = self._run_claude(
                        container,
                        prompt=reminder_builder(missing),
                        resume_session=session_id,
                        state=state,
                        role_label=role_label,
                        verbose=verbose,
                        emit=_emit,
                        step_callback=step_callback,
                        wall=wall,
                        started=started,
                        run_limit=int(max_steps),
                    )
                    if is_docker_infra_failure(last_result):
                        if verbose:
                            _emit(
                                f"[{role_label}] ✗ container infra failure after resume; "
                                "aborting early-finish loop"
                            )
                        break
                    if state.step <= step_before:
                        stagnant += 1
                        if stagnant >= 2:
                            if verbose:
                                _emit(
                                    f"[{role_label}] ✗ no step progress on resume "
                                    f"(stuck at {state.step}); aborting early-finish loop"
                                )
                            break
                    else:
                        stagnant = 0

            if completion_check is not None and reminder_builder is not None:
                for turn in range(1, max_recovery_turns + 1):
                    if state.wall_hit or (wall is not None and time.monotonic() - started > wall):
                        state.wall_hit = True
                        break
                    missing = completion_check() or []
                    if not missing:
                        break
                    if state.step < int(max_steps) and not state.budget_hit:
                        break
                    if verbose:
                        _emit(
                            f"[{role_label}] ↻ recovery turn {turn}/{max_recovery_turns}: "
                            f"missing={missing}"
                        )
                    state.limit = int(max_steps) + turn * _RECOVERY_BUDGET_PER_TURN
                    state.budget_hit = False
                    last_result, session_id = self._run_claude(
                        container,
                        prompt=reminder_builder(missing),
                        resume_session=session_id,
                        state=state,
                        role_label=role_label,
                        verbose=verbose,
                        emit=_emit,
                        step_callback=step_callback,
                        wall=wall,
                        started=started,
                        run_limit=state.step + _RECOVERY_BUDGET_PER_TURN,
                    )
                    if is_docker_infra_failure(last_result):
                        if verbose:
                            _emit(
                                f"[{role_label}] ✗ container infra failure on recovery; "
                                "stopping"
                            )
                        break

        duration = time.monotonic() - started
        missing = completion_check() if completion_check else []
        exit_status = self._exit_status(state, last_result, missing)

        if verbose:
            tail = f" (missing still: {missing})" if missing else " OK"
            _emit(
                f"[{role_label}] ═══ agent end{tail} duration={duration:.1f}s "
                f"cost=${state.total_cost_usd:.4f} exit={exit_status}"
            )
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass

        return AgentTrajectory(
            steps=list(state.steps),
            total_steps=len(state.steps),
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            total_cost_usd=state.total_cost_usd,
            model_id=self.model_config.model_id,
            duration_seconds=duration,
            exit_status=exit_status,
        )

    # ── internals ──────────────────────────────────────────

    @staticmethod
    def _preflight(
        container: CliContainer,
        *,
        emit: Callable[[str], None],
        role_label: str,
    ) -> None:
        """Fail fast if the Claude Code binary is missing from the image."""
        check = container.run_command(
            ["bash", "-c", "command -v claude && claude --version"],
            timeout=30,
        )
        if check.return_code == 0 and "not found" not in (check.stdout + check.stderr).lower():
            return
        detail = (check.stderr or check.stdout or "").strip() or f"exit={check.return_code}"
        msg = (
            "claude CLI not found in the agent container. Rebuild the per-repo "
            "Docker images so install-cli-agents.sh runs "
            f"(make build-docker). Detail: {detail}"
        )
        emit(f"[{role_label}] ✗ {msg}")
        raise RuntimeError(msg)

    def _run_claude(
        self,
        container: CliContainer,
        *,
        prompt: str,
        resume_session: str | None,
        state: StreamState,
        role_label: str,
        verbose: bool,
        emit: Callable[[str], None],
        step_callback: Callable[[int, int], None] | None,
        wall: float | None,
        started: float,
        run_limit: int,
    ) -> tuple[CliRunResult, str | None]:
        # Ephemeral CLAUDE_CONFIG_DIR lives in the container; a restart voids
        # any prior --resume session id. Check *before* assembling argv.
        if container.ensure_running():
            if verbose:
                emit(
                    f"[{role_label}] ⚠ agent container died; restarted "
                    "(session reset)"
                )
            resume_session = None

        # --bare: skip multi-user config discovery for AI-repeatable CI.
        # --dangerously-skip-permissions / bypassPermissions: fully autonomous
        # in the isolated container (mirrors mini-swe's free shell + OpenHands
        # unrestricted tools).
        argv = [
            "claude",
            "--bare",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            self.model_config.model_id,
            "--allowedTools",
            "Bash,Read,Edit,Write,MultiEdit,Glob,Grep",
        ]
        if resume_session:
            argv.extend(["--resume", resume_session])

        pending: dict[str, Any] = {
            "thought": "",
            "action": "",
            # Preceding observation (mini-swe order). Seed for first step.
            "next_observation": prompt if not resume_session else "",
            "session_id": resume_session,
        }
        last_session = resume_session

        def _on_line(line: str) -> None:
            nonlocal last_session
            sid = self._handle_json_line(
                line,
                state=state,
                pending=pending,
                role_label=role_label,
                verbose=verbose,
                emit=emit,
                step_callback=step_callback,
                run_limit=run_limit,
            )
            if sid:
                last_session = sid

        def _should_stop() -> bool:
            if wall is not None and (time.monotonic() - started) > wall:
                state.wall_hit = True
                return True
            if state.step >= run_limit:
                state.budget_hit = True
                return True
            return False

        remaining = None
        if wall is not None:
            remaining = max(1.0, wall - (time.monotonic() - started))

        result = container.run_command(
            argv,
            timeout=remaining,
            on_stdout_line=_on_line,
            on_stderr_line=(
                lambda ln: emit(f"[{role_label}]     stderr: {truncate(ln, 300)}")
                if verbose and ln.strip()
                else None
            ),
            should_stop=_should_stop,
        )
        # Flush trailing pure-thought without tool calls if any.
        trailing = str(pending.get("thought") or "").strip()
        if trailing and state.steps and not str(state.steps[-1].get("thought") or "").strip():
            last = dict(state.steps[-1])
            last["thought"] = trailing
            state.steps[-1] = last
        elif trailing and not state.steps:
            pending["action"] = ""
            self._finalize_step(
                state,
                pending,
                role_label=role_label,
                verbose=verbose,
                emit=emit,
                step_callback=step_callback,
                run_limit=run_limit,
                advance_observation=None,
            )
        return result, last_session

    def _handle_json_line(
        self,
        line: str,
        *,
        state: StreamState,
        pending: dict[str, Any],
        role_label: str,
        verbose: bool,
        emit: Callable[[str], None],
        step_callback: Callable[[int, int], None] | None,
        run_limit: int,
    ) -> str | None:
        line = line.strip()
        if not line or line[0] != "{":
            return None
        try:
            obj = json.loads(line)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        state.lines.append(line)
        session_id = obj.get("session_id")
        sid = str(session_id) if isinstance(session_id, str) and session_id else None

        etype = str(obj.get("type") or "")

        # stream-json protocol: assistant / user / result / system / stream_event
        if etype == "assistant":
            message = obj.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            thoughts, actions = _parse_content_blocks(content)
            if thoughts:
                prev = str(pending.get("thought") or "").strip()
                joined = "\n".join(thoughts)
                pending["thought"] = f"{prev}\n{joined}".strip() if prev else joined
            if actions:
                # One step per tool_use. Observation = what PRECEDED this action.
                for i, action in enumerate(actions):
                    pending["action"] = action
                    # Only attach remaining thought to first tool of this message.
                    if i > 0:
                        pending["thought"] = ""
                    self._finalize_step(
                        state,
                        pending,
                        role_label=role_label,
                        verbose=verbose,
                        emit=emit,
                        step_callback=step_callback,
                        run_limit=run_limit,
                        # Tool result arrives later on user message; leave
                        # next_observation empty until tool_result arrives.
                        advance_observation="",
                    )
            elif thoughts and not actions:
                # Pure reasoning/final answer with no tools.
                pending["action"] = ""
                self._finalize_step(
                    state,
                    pending,
                    role_label=role_label,
                    verbose=verbose,
                    emit=emit,
                    step_callback=step_callback,
                    run_limit=run_limit,
                    advance_observation=None,
                )
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict) and state.steps:
                in_tok = int(usage.get("input_tokens", 0) or 0)
                out_tok = int(usage.get("output_tokens", 0) or 0)
                cached = int(
                    usage.get("cache_read_input_tokens", 0)
                    or usage.get("cache_read_tokens", 0)
                    or 0
                )
                # Claude Code surfaces the gateway's `usage.cost` verbatim when
                # present (OpenRouter's Anthropic-compatible endpoint reports
                # it); otherwise price the tokens with the cached OpenRouter
                # per-token rates. The CLI's own price-table estimate is never
                # used — under the OpenRouter gateway it bills Anthropic
                # direct rates, not what OpenRouter actually charged.
                reported = response_reported_cost({"usage": usage})
                cost = (
                    float(reported)
                    if reported is not None
                    else fallback_cost_for(self.model_config, in_tok, out_tok, cached)
                )
                last = dict(state.steps[-1])
                last["input_tokens"] = in_tok
                last["output_tokens"] = out_tok
                last["cached_tokens"] = cached
                last["cost_usd"] = cost
                state.steps[-1] = last
                state.total_input_tokens = sum(int(s.get("input_tokens", 0) or 0) for s in state.steps)
                state.total_output_tokens = sum(int(s.get("output_tokens", 0) or 0) for s in state.steps)
                state.total_cached_tokens = sum(int(s.get("cached_tokens", 0) or 0) for s in state.steps)
                state.total_cost_usd = sum(float(s.get("cost_usd", 0.0) or 0.0) for s in state.steps)

        elif etype == "user":
            message = obj.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
            obs_parts: list[str] = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        raw = block.get("content")
                        if isinstance(raw, str):
                            obs_parts.append(raw)
                        elif isinstance(raw, list):
                            for sub in raw:
                                if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                    obs_parts.append(sub["text"])
                                elif isinstance(sub, str):
                                    obs_parts.append(sub)
            if obs_parts:
                # Tool result is the observation for the *next* step (mini-swe).
                pending["next_observation"] = "\n".join(obs_parts)
                if verbose:
                    emit(
                        f"[{role_label}]     observation(next): "
                        f"{truncate(pending['next_observation'], 600)}"
                    )

        elif etype == "result":
            # Final envelope with aggregate usage. `total_cost_usd` is the
            # CLI's own estimate from its bundled (Anthropic) price table —
            # wrong under the OpenRouter gateway, so it is deliberately
            # ignored. The aggregate tokens are priced with the cached
            # OpenRouter per-token rates, but only when per-step attribution
            # produced nothing (the totals are session-cumulative, not
            # additive on top of the per-step costs). Across multiple
            # invocations (early-finish resumes, recovery turns) the last
            # result's cumulative usage wins, mirroring the token max() below.
            usage = obj.get("usage") or {}
            if isinstance(usage, dict):
                in_tok = int(usage.get("input_tokens", 0) or 0)
                out_tok = int(usage.get("output_tokens", 0) or 0)
                if in_tok or out_tok:
                    state.total_input_tokens = max(state.total_input_tokens, in_tok)
                    state.total_output_tokens = max(state.total_output_tokens, out_tok)
                    step_tokens = sum(
                        int(s.get("input_tokens", 0) or 0)
                        + int(s.get("output_tokens", 0) or 0)
                        for s in state.steps
                    )
                    if step_tokens <= 0:
                        # No per-step token attribution: price the cumulative
                        # result tokens (no cache credit — the per-call cache
                        # numbers must not be compounded into session totals).
                        state.total_cost_usd = fallback_cost_for(
                            self.model_config,
                            state.total_input_tokens,
                            state.total_output_tokens,
                        )
            if obj.get("is_error") and verbose:
                emit(
                    f"[{role_label}] ⚠ claude result error: "
                    f"{truncate(str(obj.get('result') or obj.get('error') or ''), 400)}"
                )

        return sid

    def _finalize_step(
        self,
        state: StreamState,
        pending: dict[str, Any],
        *,
        role_label: str,
        verbose: bool,
        emit: Callable[[str], None],
        step_callback: Callable[[int, int], None] | None,
        run_limit: int,
        advance_observation: str | None = "",
    ) -> None:
        now = time.monotonic()
        step_elapsed = now - state.last_step_at
        cum_elapsed = now - state.started
        state.last_step_at = now
        state.step += 1
        observation = str(pending.get("next_observation") or "")
        step = {
            "observation": observation,
            "thought": str(pending.get("thought") or ""),
            "action": str(pending.get("action") or ""),
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "step_seconds": float(step_elapsed),
            "cum_seconds": float(cum_elapsed),
        }
        state.steps.append(step)
        pending["thought"] = ""
        pending["action"] = ""
        if advance_observation is not None:
            pending["next_observation"] = advance_observation

        if step_callback is not None:
            try:
                step_callback(int(state.step), int(state.limit))
            except Exception:
                pass
        if verbose:
            emit(
                f"[{role_label}] ▶ step {state.step}/{state.limit} thought: "
                f"{truncate(str(step['thought']), 400)}"
            )
            if step["action"]:
                emit(f"[{role_label}]     action: {truncate(str(step['action']), 400)}")
            emit(
                f"[{role_label}]     step_time_taken={step_elapsed:.1f}s "
                f"cum_time_taken={cum_elapsed:.1f}s"
            )
        if state.step >= run_limit:
            state.budget_hit = True

    @staticmethod
    def _exit_status(
        state: StreamState,
        result: CliRunResult | None,
        missing: list[str] | None,
    ) -> str:
        if state.wall_hit or (result is not None and result.timed_out):
            return "WallClockTimeout"
        if not missing:
            return "Submitted"
        if state.budget_hit or (result is not None and result.killed_for_budget):
            return "LimitsExceeded"
        if result is not None and result.return_code not in (0, None):
            return "Error"
        return "LimitsExceeded"


def _parse_content_blocks(content: Any) -> tuple[list[str], list[str]]:
    """Split Claude content blocks into thoughts and tool-use actions."""
    thoughts: list[str] = []
    actions: list[str] = []
    if isinstance(content, str):
        if content.strip():
            thoughts.append(content)
        return thoughts, actions
    if not isinstance(content, list):
        return thoughts, actions
    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str) and block.strip():
                thoughts.append(block)
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                thoughts.append(text)
        elif btype in ("thinking", "reasoning"):
            text = block.get("thinking") or block.get("text") or block.get("reasoning")
            if isinstance(text, str) and text.strip():
                thoughts.append(text)
        elif btype == "tool_use":
            name = str(block.get("name") or "tool")
            inp = block.get("input") or {}
            if isinstance(inp, dict):
                # Prefer the executable/command-looking fields.
                for key in ("command", "file_path", "path", "pattern", "query"):
                    if key in inp and inp[key]:
                        actions.append(f"{name}: {inp[key]}")
                        break
                else:
                    actions.append(f"{name}: {json.dumps(inp, ensure_ascii=False)[:300]}")
            else:
                actions.append(f"{name}: {inp}")
    return thoughts, actions
