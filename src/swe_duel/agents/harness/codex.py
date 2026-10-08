"""OpenAI Codex harness (non-interactive ``codex exec``).

Runs the Codex CLI **inside** the per-repo Docker image so the agent shares
the validation-gate environment. Model calls are routed through OpenRouter by
shipping a ephemeral ``config.toml`` that declares a custom ``openrouter``
provider and points ``model`` / ``model_provider`` at the selected model.

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
from swe_duel.agents.harness.cost_tracking import fallback_cost_for
from swe_duel.config import ModelConfig
from swe_duel.models import AgentTrajectory


class CodexHarness(AgentHarness):
    """Drives ``codex exec`` in non-interactive mode behind AgentHarness."""

    harness_id = "codex"

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
        env = {
            "OPENROUTER_API_KEY": api_key,
            "OPENAI_API_KEY": api_key,
            "CODEX_API_KEY": api_key,
            "SWE_DUEL_OPENROUTER_API_KEY": api_key,
            # Quiet CLI analytics / update checks inside the container.
            "CODEX_QUIET_MODE": "1",
            "NO_COLOR": "1",
        }

        log_fp = None
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_file, "a", encoding="utf-8", buffering=1)
            log_fp.write(
                f"\n═══ agent start role={role_label} harness=codex "
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
        # CODEX_HOME must not be under /tmp (Codex refuses tempdirs for helpers).
        config_path = "/var/tmp/swe-duel-codex/config.toml"
        env["CODEX_HOME"] = "/var/tmp/swe-duel-codex"
        env["HOME"] = "/var/tmp/swe-duel-home"

        with CliContainer(
            docker_image=docker_image,
            host_workspace=workspace_path,
            harness_id=self.harness_id,
            preserve_paths=preserve_paths,
            env=env,
            login_shell=login_shell,
        ) as container:
            self._preflight(container, emit=_emit, role_label=role_label)
            container.write_text(config_path, self._config_toml())

            if verbose:
                wall_msg = f" wall_limit={wall:.0f}s" if wall else ""
                _emit(
                    f"[{role_label}] ═══ agent start harness=codex "
                    f"model={self.model_config.model_id} {self._identity_suffix()} "
                f"cwd={workspace_path} "
                    f"step_limit={max_steps}{wall_msg}"
                )

            # ── Initial pass (full max_steps budget) ──
            # Codex can exit os.exit after exploring/planning (or its own
            # internal stop) before writing `_swe-duel/*` artefacts. OpenHands /
            # mini-swe keep going until max_steps within the *initial* budget;
            # re-prompt with reminders here until artefacts exist or the
            # initial 50-step (etc.) ceiling is consumed. Only then start the
            # constrained +6 recovery turns.
            last_result = self._run_exec(
                container,
                prompt=task_prompt,
                state=state,
                role_label=role_label,
                verbose=verbose,
                emit=_emit,
                step_callback=step_callback,
                wall=wall,
                started=started,
                run_limit=int(max_steps),
            )
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
                    last_result = self._run_exec(
                        container,
                        prompt=reminder_builder(missing),
                        state=state,
                        role_label=role_label,
                        verbose=verbose,
                        emit=_emit,
                        step_callback=step_callback,
                        wall=wall,
                        started=started,
                        # Absolute ceiling stays at max_steps for the initial phase.
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

            # ── Recovery turns (only after initial budget is spent) ──
            # Fresh `codex exec` (not `resume`): workspace files already hold
            # partial progress, and `resume` omits several top-level flags.
            if completion_check is not None and reminder_builder is not None:
                for turn in range(1, max_recovery_turns + 1):
                    if state.wall_hit or (wall is not None and time.monotonic() - started > wall):
                        state.wall_hit = True
                        break
                    missing = completion_check() or []
                    if not missing:
                        break
                    # Do not start limited recovery turns until the initial budget
                    # has actually been used (early-finish loop above should have
                    # burnt it). Guard in case completion_check is missing.
                    if state.step < int(max_steps) and not state.budget_hit:
                        break
                    if verbose:
                        _emit(
                            f"[{role_label}] ↻ recovery turn {turn}/{max_recovery_turns}: "
                            f"missing={missing}"
                        )
                    state.limit = int(max_steps) + turn * _RECOVERY_BUDGET_PER_TURN
                    state.budget_hit = False
                    last_result = self._run_exec(
                        container,
                        prompt=reminder_builder(missing),
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
        """Fail fast if the Codex binary is missing from the image.

        Requires a rebuild after install-cli-agents.sh is added
        (``make build-docker``). Without this, every recovery turn also
        fails with ``codex: command not found`` and the generator only
        reports an opaque incomplete-feature.
        """
        check = container.run_command(
            ["bash", "-c", "command -v codex && codex --version"],
            timeout=30,
        )
        if check.return_code == 0 and "not found" not in (check.stdout + check.stderr).lower():
            return
        detail = (check.stderr or check.stdout or "").strip() or f"exit={check.return_code}"
        msg = (
            "codex CLI not found in the agent container. Rebuild the per-repo "
            "Docker images so install-cli-agents.sh runs "
            f"(make build-docker). Detail: {detail}"
        )
        emit(f"[{role_label}] ✗ {msg}")
        raise RuntimeError(msg)

    def _config_toml(self) -> str:
        """Ephemeral Codex config: OpenRouter provider + full-auto permissions.

        Codex 0.144+ dropped ``wire_api = "chat"`` — providers must use the
        Responses API (``wire_api = "responses"``).

        The participant's selected reasoning effort maps onto Codex's
        ``model_reasoning_effort`` config key (Responses-API only; the base
        class already rejects efforts outside Codex's enum). Provider pinning
        is not expressible through codex config (no request-body injection),
        so the base class rejects non-empty provider selections for this
        harness.
        """
        model = self.model_config.model_id
        lines = [
            f'model = "{model}"',
            'model_provider = "openrouter"',
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
        ]
        effort = getattr(self.model_config, "reasoning_effort", "") or ""
        if effort:
            lines.append(f'model_reasoning_effort = "{effort}"')
        lines += [
            "",
            "[model_providers.openrouter]",
            'name = "OpenRouter"',
            'base_url = "https://openrouter.ai/api/v1"',
            'env_key = "OPENROUTER_API_KEY"',
            'wire_api = "responses"',
        ]
        return "\n".join(lines) + "\n"

    def _run_exec(
        self,
        container: CliContainer,
        *,
        prompt: str,
        state: StreamState,
        role_label: str,
        verbose: bool,
        emit: Callable[[str], None],
        step_callback: Callable[[int, int], None] | None,
        wall: float | None,
        started: float,
        run_limit: int,
    ) -> CliRunResult:
        if container.ensure_running() and verbose:
            emit(f"[{role_label}] ⚠ agent container died; restarted")
        # CODEX_HOME is container-local; rewrite provider config after any
        # rebuild (and cheaply on every exec so partial deaths are covered).
        try:
            container.write_text("/var/tmp/swe-duel-codex/config.toml", self._config_toml())
        except Exception as exc:
            if verbose:
                emit(f"[{role_label}] ⚠ failed to write Codex config: {exc}")

        # Isolated CODEX_HOME (set on container env at construction) discovers
        # the config.toml written by run(). Sandbox/approvals already bypassed
        # for this externally-isolated Docker environment.
        argv = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            self.model_config.model_id,
            prompt,
        ]

        # mini-swe ordering: observation PRECEDED the thought/action. Seed the
        # first observation with the task prompt so step 1 mirrors mini-swe.
        pending: dict[str, Any] = {
            "thought": "",
            "action": "",
            "next_observation": prompt,  # becomes this step's observation
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

        def _on_line(line: str) -> None:
            self._handle_json_line(
                line,
                state=state,
                pending=pending,
                role_label=role_label,
                verbose=verbose,
                emit=emit,
                step_callback=step_callback,
                run_limit=run_limit,
            )

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
            on_stderr_line=(lambda ln: emit(f"[{role_label}]     stderr: {truncate(ln, 300)}")
                            if verbose and ln.strip() else None),
            should_stop=_should_stop,
        )
        # Flush a trailing agent_message / reasoning that arrived after the last
        # tool call (common for Codex: tools first, final message last). Prefer
        # attaching to the last step's thought so we don't invent a phantom step.
        trailing = str(pending.get("thought") or "").strip()
        if trailing:
            if state.steps:
                last = dict(state.steps[-1])
                prev = str(last.get("thought") or "").strip()
                last["thought"] = f"{prev}\n{trailing}".strip() if prev else trailing
                state.steps[-1] = last
            else:
                pending["action"] = ""
                self._finalize_step(
                    state,
                    pending,
                    role_label=role_label,
                    verbose=verbose,
                    emit=emit,
                    step_callback=step_callback,
                    run_limit=run_limit,
                    advance_observation="",
                )
        return result

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
    ) -> None:
        line = line.strip()
        if not line or line[0] != "{":
            return
        try:
            obj = json.loads(line)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        state.lines.append(line)
        etype = str(obj.get("type") or "")

        if etype in ("item.started", "item.completed"):
            item = obj.get("item") or {}
            if not isinstance(item, dict):
                return
            itype = str(item.get("type") or "")
            if etype == "item.started" and itype in (
                "command_execution",
                "file_change",
                "mcp_tool_call",
            ):
                # Capture action early; thought may have been buffered already.
                pending["action"] = _codex_item_action(item)
            elif etype == "item.completed":
                if itype in ("agent_message", "reasoning", "message"):
                    text = _codex_item_thought(item)
                    if text:
                        prev = str(pending.get("thought") or "").strip()
                        pending["thought"] = f"{prev}\n{text}".strip() if prev else text
                elif itype in ("command_execution", "file_change", "mcp_tool_call"):
                    pending["action"] = _codex_item_action(item) or pending.get("action") or ""
                    cmd_output = str(
                        item.get("aggregated_output")
                        or item.get("output")
                        or item.get("result")
                        or ""
                    )
                    # Step gets the *preceding* observation; tool output seeds
                    # the next step (mini-swe offset).
                    self._finalize_step(
                        state,
                        pending,
                        role_label=role_label,
                        verbose=verbose,
                        emit=emit,
                        step_callback=step_callback,
                        run_limit=run_limit,
                        advance_observation=cmd_output,
                    )
        elif etype == "turn.completed":
            usage = obj.get("usage") or {}
            if isinstance(usage, dict):
                in_tok = int(usage.get("input_tokens", 0) or 0)
                out_tok = int(usage.get("output_tokens", 0) or 0)
                cached = int(usage.get("cached_input_tokens", 0) or 0)
                if state.steps and (in_tok or out_tok):
                    last = dict(state.steps[-1])
                    last["input_tokens"] = in_tok
                    last["output_tokens"] = out_tok
                    last["cached_tokens"] = cached
                    # Codex reports token counts only (no per-response cost
                    # from the LLM API), so price them with the cached
                    # OpenRouter per-token rates for this (model, provider).
                    cost = fallback_cost_for(self.model_config, in_tok, out_tok, cached)
                    last["cost_usd"] = cost
                    state.steps[-1] = last
                    state.total_input_tokens = sum(
                        int(s.get("input_tokens", 0) or 0) for s in state.steps
                    )
                    state.total_output_tokens = sum(
                        int(s.get("output_tokens", 0) or 0) for s in state.steps
                    )
                    state.total_cached_tokens = sum(
                        int(s.get("cached_tokens", 0) or 0) for s in state.steps
                    )
                    state.total_cost_usd = sum(
                        float(s.get("cost_usd", 0.0) or 0.0) for s in state.steps
                    )

        elif etype in ("turn.failed", "error") and verbose:
            emit(f"[{role_label}] ⚠ codex event: {truncate(line, 400)}")

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
        advance_observation: str | None = None,
    ) -> None:
        now = time.monotonic()
        step_elapsed = now - state.last_step_at
        cum_elapsed = now - state.started
        state.last_step_at = now
        state.step += 1

        in_tok = int(pending.get("input_tokens", 0) or 0)
        out_tok = int(pending.get("output_tokens", 0) or 0)
        cached = int(pending.get("cached_tokens", 0) or 0)
        # Codex never surfaces the LLM API response body (its events carry
        # token counts only), so every step is priced with the cached
        # OpenRouter per-token rates for the participant's (model, provider).
        cost = fallback_cost_for(self.model_config, in_tok, out_tok, cached)
        state.total_input_tokens += in_tok
        state.total_output_tokens += out_tok
        state.total_cached_tokens += cached
        state.total_cost_usd += cost

        # Observation for this step is whatever PRECEDED it.
        observation = str(pending.get("next_observation") or "")
        step = {
            "observation": observation,
            "thought": str(pending.get("thought") or ""),
            "action": str(pending.get("action") or ""),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cached_tokens": cached,
            "cost_usd": cost,
            "step_seconds": float(step_elapsed),
            "cum_seconds": float(cum_elapsed),
        }
        state.steps.append(step)
        pending["thought"] = ""
        pending["action"] = ""
        pending["input_tokens"] = 0
        pending["output_tokens"] = 0
        pending["cached_tokens"] = 0
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
            if observation:
                emit(
                    f"[{role_label}]     observation: {truncate(observation, 300)}"
                )
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


def _codex_item_action(item: dict[str, Any]) -> str:
    for key in ("command", "cmd", "path", "changes"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list):
            parts = [str(v) for v in val if v]
            if parts:
                return "\n".join(parts)
    itype = str(item.get("type") or "item")
    return itype


def _codex_item_thought(item: dict[str, Any]) -> str:
    """Extract reasoning / agent_message text from a Codex item payload."""
    for key in ("text", "content", "reasoning", "summary", "message"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
