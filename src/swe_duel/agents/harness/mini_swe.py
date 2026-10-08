"""mini-swe-agent harness.

Configures a `DefaultAgent` against a `DockerEnvironment` (or `LocalEnvironment`
for host tests) rooted at a workspace, routes model calls through OpenRouter via
litellm, runs the agent with an artifact-recovery loop, and extracts an
`AgentTrajectory` from the finished agent's state.

This is the original `AgentWrapper` implementation, relocated behind the
`AgentHarness` interface. Behavior is unchanged.
"""

from __future__ import annotations

import os
import time
from importlib import resources
from pathlib import Path
from typing import Any, Callable

import yaml

from swe_duel.agents.harness.base import CONTAINER_WORKSPACE, AgentHarness, openrouter_body_extras
from swe_duel.agents.harness.cost_tracking import (
    fallback_cost_for,
    response_reported_cost,
)
from swe_duel.agents.harness.rate_limit import (
    RateLimitedModelProxy,
    install_rate_limit_log_filter,
    set_thread_noise_absorbed,
    throttle_for,
)
from swe_duel.config import ModelConfig
from swe_duel.models import AgentTrajectory
from swe_duel.sandbox.docker_executor import swe_duel_container_name, repo_from_image

# Silence litellm's "Provider List: https://docs.litellm.ai/docs/providers"
# print, which fires every time it fails to identify a provider for a model
# name it doesn't recognize (e.g. openrouter-only IDs like "z-ai/glm-5.1").
# The error is handled downstream; only the noisy print is the problem.
try:
    import litellm  # type: ignore

    litellm.suppress_debug_info = True
except Exception:
    pass

# mini-swe-agent prints a 👋 startup banner to stdout on first import and
# configures its `minisweagent` logger at DEBUG with a RichHandler (→ stderr).
# Both fire inside worker threads mid-TUI and corrupt rich's `Live` cursor
# tracking, so the generate/match progress bar repaints below the screen
# instead of in place. Set the banner-suppression env var at module import
# time (before minisweagent is ever imported — the heavy imports are lazy,
# inside run()) and quiet the logger tree once it has been imported.
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

_IO_SILENCED = False


def _silence_mini_swe_io() -> None:
    """Suppress mini-swe-agent's unconditional console chatter (idempotent).

    Two noise sources are *not* gated by any verbosity flag and would
    otherwise bury the generate/match TUI's `rich.live.Live` progress bar
    (the Live view tracks the cursor on its console; any stray write to
    stdout OR stderr from a worker thread makes the next refresh paint below
    the current screen instead of updating it):

    1. A startup banner ("👋 This is mini-swe-agent version …"). Suppressed
       via the ``MSWEA_SILENT_STARTUP`` env var, set at module import time
       above (it is read at minisweagent import time, which is lazy).
    2. The `minisweagent` logger tree (RichHandler → stderr) runs at DEBUG and
       logs every container start at INFO, plus retry warnings and per-step
       debug lines — one burst per agent run. We quiet `minisweagent` and the
       sibling `agent` / `litellm_model` / `litellm` / `LiteLLM` loggers to
       WARNING. Operators can raise the level by exporting ``LOG_LEVEL``
       (e.g. ``LOG_LEVEL=DEBUG``) before launch, mirroring the OpenHands
       harness.

    Must run AFTER minisweagent is imported: its `utils.log._setup_root_logger`
    calls `logger.setLevel(DEBUG)` at import time and would clobber an
    earlier setting.
    """
    global _IO_SILENCED
    if _IO_SILENCED:
        return
    import logging as _logging

    level_name = os.environ.setdefault("LOG_LEVEL", "WARNING")
    level = _logging.getLevelName(level_name.upper())
    for name in ("minisweagent", "agent", "litellm_model", "litellm", "LiteLLM"):
        _logging.getLogger(name).setLevel(level)
    _IO_SILENCED = True


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ⏎ ")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + f"… ⟨+{len(s) - n} chars⟩"


class MiniSweHarness(AgentHarness):
    """Wraps mini-swe-agent's `DefaultAgent` with OpenRouter routing."""

    harness_id = "mini-swe-agent"

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)

    # ── public ─────────────────────────────────────────────

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
        """Launch the agent against `workspace_path` and return its trajectory.

        See :meth:`swe_duel.agents.harness.base.AgentHarness.run` for the contract.
        """
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models import get_model

        # Quiet the minisweagent logger tree (banner already suppressed via
        # MSWEA_SILENT_STARTUP at module import). Must come AFTER the imports
        # above so minisweagent.utils.log._setup_root_logger's setLevel(DEBUG)
        # doesn't clobber us.
        _silence_mini_swe_io()

        log_fp = None
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_file, "a", encoding="utf-8", buffering=1)
            log_fp.write(
                f"\n═══ agent start role={role_label} "
                f"model={self.model_config.model_id} {self._identity_suffix()} "
                f"cwd={workspace_path} ts={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
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

        # Rate-limit noise absorption: this thread runs under the live TUI
        # when its console output is suppressed — absorb 429/retry chatter
        # into data/logs/model_retries.log instead of letting it reach stderr
        # (a stray write from a worker thread breaks rich's Live cursor).
        install_rate_limit_log_filter()
        set_thread_noise_absorbed(not console_echo)

        # Throttle waits worth mentioning (per-run): only surface waits that
        # actually delay the agent noticeably, so sustained-limit queues
        # don't spam one line per step.
        def _on_throttle_wait(waited: float) -> None:
            if waited >= 5.0:
                _emit(
                    f"[{role_label}] ⏳ rate-limiter: waited {waited:.1f}s for "
                    f"{self.model_config.model_id} @ "
                    f"{self.model_config.provider or 'auto-route'} "
                    f"(shared process-wide budget)"
                )

        cfg = self._build_agent_config(workspace_path, max_steps=max_steps)

        model = get_model(
            input_model_name=cfg["model"]["model_name"],
            config={
                "model_kwargs": cfg["model"]["model_kwargs"],
                # needed for minimax model for some reason
                "cost_tracking": "ignore_errors",
                # mini.yaml's truncation-aware template: when a response is cut
                # off by the output-token limit before emitting a tool call
                # (finish_reason="length", tool_calls=null — the classic
                # reasoning-model-at-max-effort failure), the model is told to
                # respond more concisely instead of seeing only the generic
                # "No tool calls found" error and repeating the truncation.
                "format_error_template": cfg["model"]["format_error_template"],
            },
        )
        throttle = throttle_for(self.model_config)
        if throttle is not None:
            model = RateLimitedModelProxy(model, throttle, on_wait=_on_throttle_wait)
        if docker_image:
            env = self._build_docker_environment(
                DockerEnvironment,
                docker_image=docker_image,
                host_workspace=workspace_path,
                base_env=cfg["env"]["env"],
                timeout=cfg["env"]["timeout"],
                preserve_paths=preserve_paths or [],
                login_shell=login_shell,
            )
        else:
            env = LocalEnvironment(
                cwd=cfg["env"]["cwd"], env=cfg["env"]["env"], timeout=cfg["env"]["timeout"]
            )

        agent_cls = _VerboseDefaultAgent if verbose else _import_default_agent()
        agent_kwargs = dict(
            system_template=cfg["agent"]["system_template"],
            instance_template=cfg["agent"]["instance_template"],
            step_limit=cfg["agent"]["step_limit"],
            cost_limit=cfg["agent"]["cost_limit"],
        )
        if verbose:
            agent_kwargs["role_label"] = role_label
            agent_kwargs["log_fp"] = log_fp
            agent_kwargs["model_config"] = self.model_config
            agent_kwargs["max_wall_seconds"] = max_wall_seconds
            agent_kwargs["step_callback"] = step_callback
            agent_kwargs["console_echo"] = console_echo

        agent = agent_cls(model, env, **agent_kwargs)

        if verbose:
            wall_msg = (
                f" wall_limit={max_wall_seconds:.0f}s" if max_wall_seconds else ""
            )
            _emit(
                f"[{role_label}] ═══ agent start model={self.model_config.model_id} "
                f"{self._identity_suffix()} "
                f"cwd={workspace_path} step_limit={cfg['agent']['step_limit']}{wall_msg}"
            )

        started = time.monotonic()
        try:
            self._drive_agent(
                agent,
                task_prompt,
                verbose=verbose,
                role_label=role_label,
                completion_check=completion_check,
                reminder_builder=reminder_builder,
                max_recovery_turns=max_recovery_turns,
                max_wall_seconds=max_wall_seconds,
                started=started,
                emit=_emit,
            )
        finally:
            # The DockerEnvironment launches a long-lived `sleep` container per
            # agent run; release it now so concurrent generation pools and long
            # tournaments don't accumulate containers. LocalEnvironment has no
            # cleanup() and is a no-op here.
            cleanup = getattr(env, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

        duration = time.monotonic() - started

        if verbose:
            tracked = float(getattr(agent, "cost", 0.0) or 0.0)
            cum = float(getattr(agent, "_cum_cost", 0.0) or 0.0)
            final_cost = tracked if tracked > 0.0 else cum
            final_missing = completion_check() if completion_check else []
            if final_missing:
                _emit(
                    f"[{role_label}] ═══ agent end (missing still: {final_missing}) "
                    f"duration={duration:.1f}s cost=${final_cost:.4f}"
                )
            else:
                _emit(
                    f"[{role_label}] ═══ agent end OK duration={duration:.1f}s "
                    f"cost=${final_cost:.4f}"
                )

        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass

        return self._extract_trajectory(agent, duration)

    def _drive_agent(
        self,
        agent: Any,
        task_prompt: str,
        *,
        verbose: bool,
        role_label: str,
        completion_check: Callable[[], list[str]] | None,
        reminder_builder: Callable[[list[str]], str] | None,
        max_recovery_turns: int,
        max_wall_seconds: float | None,
        started: float,
        emit: Callable[[str], None],
    ) -> None:
        """Run the agent's initial pass plus the artifact-recovery loop.

        Extracted from `run()` so the surrounding `try/finally` can guarantee
        the (Docker) environment is cleaned up regardless of how the agent loop
        terminates.
        """
        try:
            agent.run(task_prompt)
        except Exception as e:
            if verbose:
                emit(f"[{role_label}] ⚠ initial run raised: {type(e).__name__}: {e}")

        # ── Recovery loop: nudge the agent to finish missing artifacts ─────
        if completion_check is not None and reminder_builder is not None:
            recovery_budget_per_turn = 6
            for turn in range(1, max_recovery_turns + 1):
                # Skip recovery entirely if we've already blown the wall-clock
                # budget — re-prompting a stalled run wastes calls.
                if max_wall_seconds is not None and (time.monotonic() - started) > max_wall_seconds:
                    if verbose:
                        emit(
                            f"[{role_label}] ⏱ skipping recovery: wall-clock limit exceeded"
                        )
                    break
                missing = completion_check() or []
                if not missing:
                    break
                if verbose:
                    emit(
                        f"[{role_label}] ↻ recovery turn {turn}/{max_recovery_turns}: "
                        f"missing={missing}"
                    )

                # Extend step + cost budgets so the agent can actually execute
                # the recovery nudge (the initial run typically exited on
                # LimitsExceeded or Submitted).
                try:
                    agent.config.step_limit = int(getattr(agent, "n_calls", 0) or 0) + recovery_budget_per_turn
                except Exception:
                    pass
                try:
                    current_cost = float(getattr(agent, "cost", 0.0) or 0.0)
                    agent.config.cost_limit = max(agent.config.cost_limit, current_cost + 1.0)
                except Exception:
                    pass

                # Drop any trailing "exit" message so the step loop can resume.
                while agent.messages and agent.messages[-1].get("role") == "exit":
                    agent.messages.pop()

                reminder = reminder_builder(missing)
                try:
                    agent.add_messages(
                        agent.model.format_message(role="user", content=reminder)
                    )
                except Exception:
                    break
                try:
                    while True:
                        try:
                            agent.step()
                        except Exception as e:
                            if verbose:
                                emit(
                                    f"[{role_label}] ⚠ recovery step raised: "
                                    f"{type(e).__name__}: {e}"
                                )
                            # Mirror DefaultAgent.run()'s handler: mini-swe-agent's
                            # interrupt exceptions carry their messages (FormatError
                            # feedback / LimitsExceeded exit). Without adding them the
                            # model never sees WHY its response was rejected, so every
                            # recovery turn repeats the same failure blindly — e.g. an
                            # output-token-limit truncation ("No tool calls found")
                            # burns all recovery turns without the model ever learning
                            # to respond more concisely.
                            interrupt_messages = list(getattr(e, "messages", None) or [])
                            if interrupt_messages:
                                try:
                                    agent.add_messages(*interrupt_messages)
                                except Exception:
                                    pass
                            if (
                                not interrupt_messages
                                or interrupt_messages[-1].get("role") == "exit"
                            ):
                                # Exit-typed interrupt (LimitsExceeded / Submitted)
                                # or a non-interrupt error (API failure): end this
                                # turn — the outer loop re-reminds on the next one.
                                break
                            # FormatError feedback was added to the conversation:
                            # keep stepping within the turn (bounded by the extended
                            # step_limit), exactly like run() retries format errors.
                            continue
                        if agent.messages and agent.messages[-1].get("role") == "exit":
                            break
                except Exception:
                    # Keep iterating — completion_check() drives termination.
                    continue

    # ── internals ──────────────────────────────────────────

    def _build_docker_environment(
        self,
        docker_env_cls: type,
        *,
        docker_image: str,
        host_workspace: Path,
        base_env: dict[str, str],
        timeout: int,
        preserve_paths: list[str],
        login_shell: bool,
    ):
        """Construct a mini-swe-agent DockerEnvironment rooted at the repo image.

        The host workspace is bind-mounted at ``CONTAINER_WORKSPACE`` so the
        agent's edits persist to disk (for diffing) while the image's baked
        toolchain and dependencies are available. ``preserve_paths`` are masked
        with anonymous volumes so image-baked dirs (e.g. node_modules) show
        through the bind mount instead of the host clone's empty/absent ones.
        The container runs as the host uid/gid (so written files are
        host-owned).  Network access is left enabled so agents can resolve
        dependencies as needed.
        """
        host_ws = str(Path(host_workspace).resolve())
        uid, gid = os.getuid(), os.getgid()
        run_args = [
            "--rm",
            "--user", f"{uid}:{gid}",
            "-v", f"{host_ws}:{CONTAINER_WORKSPACE}",
            "--label", f"swe_duel.session_pid={os.getpid()}",
            "--label", "swe_duel.managed=true",
            # Override mini-swe-agent's internal `--name minisweagent-<uuid>`
            # so our containers carry the `swe-duel-` prefix and never collide with
            # a co-tenant's `minisweagent-*` containers. `docker run` resolves a
            # duplicate `--name` last-wins, and DockerEnvironment appends our
            # run_args AFTER its own name flag, so this one takes effect.
            "--name", swe_duel_container_name(repo_from_image(docker_image), self.harness_id),
        ]
        for sub in preserve_paths:
            sub = sub.strip().strip("/")
            if sub:
                # Anonymous volume (no host source) → the image's contents at
                # this path remain visible instead of being masked by the bind.
                run_args += ["-v", f"{CONTAINER_WORKSPACE}/{sub}"]

        # Route HOME and language caches to a writable location: the container
        # runs as a non-root uid that may not own the image's default HOME.
        env_vars = dict(base_env)
        env_vars.setdefault("HOME", "/tmp")
        env_vars.setdefault("GOCACHE", "/tmp/.gocache")
        env_vars.setdefault("GOPATH", "/tmp/go")
        env_vars.setdefault("npm_config_cache", "/tmp/.npm")

        interpreter = ["bash", "-lc"] if login_shell else ["bash", "-c"]

        return docker_env_cls(
            image=docker_image,
            cwd=CONTAINER_WORKSPACE,
            env=env_vars,
            run_args=run_args,
            interpreter=interpreter,
            timeout=timeout,
        )

    def _build_agent_config(self, workspace_path: Path, max_steps: int = 30) -> dict[str, Any]:
        """Compose a config dict consumed by `run()`.

        Structure:
            {
              "model": {"model_name", "model_kwargs", "format_error_template"},
              "env":   {"cwd", "env", "timeout"},
              "agent": {"system_template", "instance_template", "step_limit", "cost_limit"},
            }

        The participant's selected reasoning effort / OpenRouter provider are
        forwarded through ``model_kwargs["extra_body"]``: mini-swe-agent passes
        model_kwargs verbatim to ``litellm.completion``, which merges extra_body
        into the OpenRouter request body (``reasoning.effort`` + provider
        routing). An empty selection sends nothing (model default / auto-route).

        ``format_error_template`` is forwarded from the bundled mini.yaml so a
        response truncated by the output-token limit before its tool call
        (``finish_reason == "length"``, no tool calls — frequent for reasoning
        models at deep efforts) produces actionable "respond more concisely"
        feedback instead of the bare default ``"{{ error }}"`` message; without
        it the model never learns why its response was rejected and repeats the
        truncation until ``RepeatedFormatError`` ends the run early.
        """
        mini_cfg = self._load_mini_defaults()

        model_cfg = mini_cfg.get("model", {})
        model_kwargs: dict[str, Any] = dict(model_cfg.get("model_kwargs", {}))
        model_kwargs["temperature"] = self.model_config.temperature
        model_kwargs["max_tokens"] = self.model_config.max_tokens
        extra_body = openrouter_body_extras(self.model_config)
        if extra_body:
            model_kwargs["extra_body"] = extra_body

        env_vars: dict[str, str] = dict(mini_cfg.get("environment", {}).get("env", {}))

        agent_cfg = mini_cfg.get("agent", {})
        system_template = agent_cfg.get("system_template", "You are a helpful assistant that can interact with a computer.\n")
        instance_template = agent_cfg.get("instance_template", "{{task}}")

        return {
            "model": {
                "model_name": self.model_config.openrouter_model_id,
                "model_kwargs": model_kwargs,
                "format_error_template": model_cfg.get("format_error_template", "{{ error }}"),
            },
            "env": {
                "cwd": str(workspace_path),
                "env": env_vars,
                "timeout": 120,
            },
            "agent": {
                "system_template": system_template,
                "instance_template": instance_template,
                "step_limit": max_steps,
                "cost_limit": agent_cfg.get("cost_limit", 3.0),
            },
        }

    def _extract_trajectory(self, agent: Any, duration_seconds: float) -> AgentTrajectory:
        """Parse the finished agent's `messages`, `cost`, `n_calls` into a trajectory."""
        messages: list[dict] = list(getattr(agent, "messages", []) or [])

        steps: list[dict] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        total_step_cost = 0.0

        def _usage_from_extra(extra: dict) -> tuple[int, int, int, float | None]:
            """Extract (input_tokens, output_tokens, cached_tokens, cost_usd)
            from a message's extra payload. Mini-swe-agent's LitellmModel stashes
            the raw litellm response under extra["response"] and the computed
            dollar cost under extra["cost"]; tokens live at
            response.usage.{prompt,completion}_tokens. Older code paths
            sometimes populate input_tokens/output_tokens directly on extra, so
            support both.

            The cost is the OpenRouter-reported per-response cost from the
            response body (``usage.cost`` — reflects the provider that actually
            served, peak pricing, promotions) when present, falling back to
            litellm's computed cost. ``None`` means neither is available so the
            caller can price the token counts via the OpenRouter endpoints API.
            """
            if not extra:
                return 0, 0, 0, None
            in_tok = int(extra.get("input_tokens", 0) or 0)
            out_tok = int(extra.get("output_tokens", 0) or 0)
            cached_tok = int(extra.get("cached_tokens", 0) or 0)
            response = extra.get("response") or {}
            usage = response.get("usage") if isinstance(response, dict) else None
            if usage:
                in_tok = in_tok or int(usage.get("prompt_tokens", 0) or 0)
                out_tok = out_tok or int(usage.get("completion_tokens", 0) or 0)
                ptd = usage.get("prompt_tokens_details") or {}
                if isinstance(ptd, dict):
                    cached_tok = cached_tok or int(ptd.get("cached_tokens", 0) or 0)
            # Priority: OpenRouter's reported response-body cost (dynamic) …
            reported = response_reported_cost(response)
            if reported is not None:
                return in_tok, out_tok, cached_tok, reported
            # … then litellm's computed cost (static model map, 0.0 for
            # openrouter-only ids under cost_tracking="ignore_errors") …
            cost = float(extra.get("cost", 0.0) or 0.0)
            if cost > 0.0:
                return in_tok, out_tok, cached_tok, cost
            # … else None → caller prices the tokens via OpenRouter pricing.
            return in_tok, out_tok, cached_tok, None

        def _coerce_text(value: Any) -> str:
            """Best-effort flatten of message content. Handles plain strings,
            lists of content blocks (Anthropic-style), and None."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for block in value:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        # Common shapes: {"type": "text", "text": "..."},
                        # {"type": "thinking", "thinking": "..."}, etc.
                        for key in ("text", "thinking", "reasoning", "content"):
                            v = block.get(key)
                            if isinstance(v, str) and v:
                                parts.append(v)
                                break
                return "\n".join(p for p in parts if p)
            return str(value)

        def _extract_thought(assistant_msg: dict, extra: dict) -> str:
            """Pull the assistant's reasoning text. Some models leave `content`
            empty when the response is purely tool calls and instead place
            reasoning under `reasoning_content` / `reasoning` on the response
            message."""
            text = _coerce_text(assistant_msg.get("content"))
            if text:
                return text
            response = extra.get("response") or {}
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    rmsg = choices[0].get("message") or {}
                    if isinstance(rmsg, dict):
                        for key in ("reasoning_content", "reasoning", "content"):
                            text = _coerce_text(rmsg.get(key))
                            if text:
                                return text
            return ""

        def _finalize_step(assistant_msg: dict, observation: str) -> None:
            extra = assistant_msg.get("extra") or {}
            in_tok, out_tok, cached_tok, cost = _usage_from_extra(extra)
            if cost is None:
                cost = fallback_cost_for(self.model_config, in_tok, out_tok, cached_tok)
            nonlocal total_input_tokens, total_output_tokens, total_cached_tokens, total_step_cost
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cached_tokens += cached_tok
            total_step_cost += cost
            action_str = extra.get("action", "")
            if not action_str:
                actions = extra.get("actions") or []
                cmds = [
                    (a.get("command") or "").strip()
                    for a in actions
                    if isinstance(a, dict)
                ]
                cmds = [c for c in cmds if c]
                action_str = "\n".join(cmds)
            steps.append({
                "observation": observation,
                "thought": _extract_thought(assistant_msg, extra),
                "action": action_str,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_tokens": cached_tok,
                "cost_usd": cost,
                "step_seconds": float(extra.get("step_seconds", 0.0) or 0.0),
                "cum_seconds": float(extra.get("cum_seconds", 0.0) or 0.0),
            })

        # Pair each assistant message with the observation that PRECEDED it
        # (chronological order: observation → thought → action). The initial
        # user/task message acts as the observation for step 1.
        pending_observation = ""
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                _finalize_step(msg, pending_observation)
                pending_observation = ""
            elif role in ("user", "tool"):
                pending_observation = _coerce_text(msg.get("content"))
            elif role == "exit":
                pending_observation = ""

        # Prefer the dynamic per-step costs (OpenRouter-reported response-body
        # cost first, pricing fallback last). mini-swe's own `agent.cost`
        # accumulates litellm's static-map cost and is only a legacy fallback.
        tracked_cost = float(getattr(agent, "cost", 0.0) or 0.0)
        if total_step_cost > 0.0:
            total_cost_usd = total_step_cost
        elif tracked_cost > 0.0:
            total_cost_usd = tracked_cost
        else:
            total_cost_usd = fallback_cost_for(
                self.model_config, total_input_tokens, total_output_tokens, total_cached_tokens
            )

        total_steps = int(getattr(agent, "n_calls", 0) or 0) or len(steps)

        exit_status = ""
        for msg in reversed(messages):
            if msg.get("role") == "exit":
                extra_exit = msg.get("extra") or {}
                exit_status = str(extra_exit.get("exit_status", "") or "")
                break

        return AgentTrajectory(
            steps=steps,
            total_steps=total_steps,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost_usd,
            model_id=self.model_config.model_id,
            duration_seconds=duration_seconds,
            exit_status=exit_status,
        )

    @staticmethod
    def _load_mini_defaults() -> dict[str, Any]:
        """Load the bundled mini.yaml template shipped with mini-swe-agent."""
        try:
            text = resources.files("minisweagent.config").joinpath("mini.yaml").read_text()
            return yaml.safe_load(text) or {}
        except (FileNotFoundError, ModuleNotFoundError):
            return {}


def _import_default_agent():
    from minisweagent.agents.default import DefaultAgent
    return DefaultAgent


def _make_verbose_agent_class():
    """Build a subclass of mini-swe-agent's DefaultAgent that streams progress."""
    from minisweagent.agents.default import DefaultAgent

    from minisweagent.exceptions import LimitsExceeded

    class _VerboseDefaultAgent(DefaultAgent):
        def __init__(
            self,
            *args,
            role_label: str = "agent",
            log_fp=None,
            model_config: ModelConfig | None = None,
            max_wall_seconds: float | None = None,
            step_callback=None,
            console_echo: bool = True,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._role_label = role_label
            self._printed_count = 0
            self._log_fp = log_fp
            self._step_callback = step_callback
            self._console_echo = console_echo
            self._cum_input_tokens = 0
            self._cum_output_tokens = 0
            self._cum_cost = 0.0
            self._model_config = model_config
            self._started_at = time.monotonic()
            self._last_step_at = self._started_at
            self._max_wall_seconds = (
                float(max_wall_seconds) if max_wall_seconds and max_wall_seconds > 0 else None
            )

        def query(self, *args, **kwargs):
            if self._max_wall_seconds is not None:
                elapsed = time.monotonic() - self._started_at
                if elapsed > self._max_wall_seconds:
                    self._emit(
                        f"[{self._role_label}] ⏱ wall-clock limit hit: "
                        f"elapsed={elapsed:.1f}s > max={self._max_wall_seconds:.1f}s — aborting agent loop"
                    )
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "WallClockTimeout",
                            "extra": {
                                "exit_status": "WallClockTimeout",
                                "submission": "",
                            },
                        }
                    )
            return super().query(*args, **kwargs)

        def _emit(self, line: str) -> None:
            if self._console_echo:
                print(line, flush=True)
            if self._log_fp is not None:
                try:
                    self._log_fp.write(line + "\n")
                    self._log_fp.flush()
                except Exception:
                    pass

        def add_messages(self, *messages: dict) -> list[dict]:
            result = super().add_messages(*messages)
            # Only print messages added after system+instance boot-up
            for msg in messages:
                self._printed_count += 1
                if self._printed_count <= 2:
                    continue
                self._print_message(msg)
            return result

        def _print_message(self, msg: dict) -> None:
            role = msg.get("role")
            content = msg.get("content", "") or ""
            extra = msg.get("extra") or {}
            label = self._role_label

            if role == "assistant":
                in_tok = int(extra.get("input_tokens", 0) or 0)
                out_tok = int(extra.get("output_tokens", 0) or 0)
                response = extra.get("response") or {}
                usage = response.get("usage") if isinstance(response, dict) else None
                if usage:
                    in_tok = in_tok or int(usage.get("prompt_tokens", 0) or 0)
                    out_tok = out_tok or int(usage.get("completion_tokens", 0) or 0)
                # Dynamic cost: OpenRouter's per-response reported cost from
                # the response body is authoritative (even a legit 0.0);
                # litellm's computed cost is the secondary source, and the
                # cached OpenRouter per-token pricing is the last resort.
                step_cost = float(extra.get("cost", 0.0) or 0.0)
                if self._model_config is not None:
                    reported = response_reported_cost(response)
                    if reported is None and step_cost <= 0.0:
                        reported = fallback_cost_for(self._model_config, in_tok, out_tok)
                    if reported is not None:
                        step_cost = reported
                self._cum_input_tokens += in_tok
                self._cum_output_tokens += out_tok
                self._cum_cost += step_cost
                now = time.monotonic()
                step_elapsed = now - self._last_step_at
                cum_elapsed = now - self._started_at
                self._last_step_at = now
                # Stash timing back onto the message so _extract_trajectory
                # can carry it into the AgentTrajectory step records.
                try:
                    extra["step_seconds"] = step_elapsed
                    extra["cum_seconds"] = cum_elapsed
                    msg["extra"] = extra
                except Exception:
                    pass

                # Notify any live-UI progress bar of the current step / budget.
                if self._step_callback is not None:
                    try:
                        self._step_callback(int(self.n_calls), int(self.config.step_limit))
                    except Exception:
                        pass

                thought_preview = _truncate(content, 400)
                self._emit(f"[{label}] ▶ step {self.n_calls} thought: {thought_preview}")
                self._emit(
                    f"[{label}]     tokens: in={in_tok} out={out_tok} "
                    f"(cum in={self._cum_input_tokens} out={self._cum_output_tokens}) "
                    f"cost=${step_cost:.4f} cum=${self._cum_cost:.4f} "
                    f"step_time_taken={step_elapsed:.1f}s cum_time_taken={cum_elapsed:.1f}s"
                )
                actions = extra.get("actions") or []
                for a in actions:
                    cmd = (a.get("command") or "").strip()
                    self._emit(f"[{label}]     action: {cmd}")
            elif role in ("user", "tool"):
                self._emit(f"[{label}]     observation: {content}")
            elif role == "exit":
                exit_status = extra.get("exit_status", "")
                self._emit(f"[{label}] ⏹ exit status={exit_status}")

    return _VerboseDefaultAgent


# Lazy-initialized verbose agent class (imports minisweagent at call time).
class _VerboseDefaultAgentProxy:
    _impl = None

    def __call__(self, *args, **kwargs):
        if _VerboseDefaultAgentProxy._impl is None:
            _VerboseDefaultAgentProxy._impl = _make_verbose_agent_class()
        return _VerboseDefaultAgentProxy._impl(*args, **kwargs)


_VerboseDefaultAgent = _VerboseDefaultAgentProxy()
