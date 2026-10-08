"""OpenHands harness.

Runs the OpenHands agent SDK against a workspace, routing model calls through
OpenRouter, and adapts it onto the shared :class:`AgentHarness` contract so it is
a drop-in alternative to the mini-swe-agent harness for every agent role.

Key differences from mini-swe-agent that this module hides:

- OpenHands executes tools inside a containerised *agent-server* that it talks to
  over HTTP (``DockerWorkspace`` → ``RemoteConversation``). The agent-server must
  be baked into the repo Docker image; we pass that image as ``server_image`` and
  bind-mount the host workspace at :data:`CONTAINER_WORKSPACE` via ``volumes`` so
  edits persist to disk for diffing — matching the validation-gate environment.
- Progress/trajectory come from typed conversation *events* (``ActionEvent`` /
  ``ObservationEvent``) and a ``Metrics`` object, not from chat messages. We map
  these into the same step-dict shape the mini-swe harness emits.
- The step budget is ``max_iteration_per_run``; recovery is done by
  ``send_message`` + ``run`` after a non-FINISHED stop. Wall-clock timeouts are
  enforced from the event callback via ``conversation.pause()``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from swe_duel.agents.harness.base import CONTAINER_WORKSPACE, AgentHarness, openrouter_body_extras
from swe_duel.agents.harness.cost_tracking import fallback_cost_for
from swe_duel.agents.harness.rate_limit import (
    install_rate_limit_log_filter,
    make_rate_limited_llm_class,
    set_thread_noise_absorbed,
    throttle_for,
)
from swe_duel.config import ModelConfig
from swe_duel.models import AgentTrajectory
from swe_duel.sandbox.docker_executor import swe_duel_container_name, repo_from_image

_RECOVERY_BUDGET_PER_TURN = 6

_IO_SILENCED = False


def _silence_openhands_io() -> None:
    """Suppress the SDK's unconditional console chatter (idempotent).

    Two noise sources are *not* gated by ``LOG_LEVEL`` and would otherwise bury
    the generate TUI's progress bar:

    1. A startup banner ("OpenHands SDK v… / Report a bug …"). Hidden via the
       ``OPENHANDS_SUPPRESS_BANNER`` env var the banner itself documents.
    2. ``openhands.sdk.utils.command.execute_command`` echoes every subprocess's
       stdout/stderr straight to ``sys.stdout``/``sys.stderr`` (it defaults
       ``print_output=True``). ``DockerWorkspace`` drives the container lifecycle
       through it — ``docker version``, ``docker run`` (prints the container id),
       ``docker inspect`` health checks (prints ``true``), ``docker stop`` — so
       this floods the console with `docker version` tables and bare hashes. We
       rebind the name in the workspace module to a wrapper that forces
       ``print_output=False``; the SDK still captures the output it needs from
       the returned ``CompletedProcess``.
    """
    global _IO_SILENCED
    if _IO_SILENCED:
        return
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    try:
        from openhands.workspace.docker import workspace as _ws_mod

        _orig = _ws_mod.execute_command

        def _quiet_execute_command(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            kwargs["print_output"] = False
            return _orig(*args, **kwargs)

        _ws_mod.execute_command = _quiet_execute_command  # type: ignore[assignment]
    except Exception:
        pass
    _IO_SILENCED = True


def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ⏎ ")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + f"… ⟨+{len(s) - n} chars⟩"


class _WallClockExceeded(Exception):
    """Internal sentinel raised when the wall-clock budget is blown."""


class OpenHandsHarness(AgentHarness):
    """Drives the OpenHands SDK behind the shared harness interface."""

    harness_id = "openhands"

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
        # The OpenHands SDK logs every container action and HTTP round-trip at
        # INFO via the `openhands` logger tree, which buries the generate TUI's
        # progress bar. Quiet it to WARNING (operators can override by exporting
        # LOG_LEVEL before launch); per-step reasoning still reaches the user via
        # our own step_callback + `_swe-duel/*.log` streaming below. Set both the env
        # var (read at SDK import time) and the live logger level (in case the
        # SDK is already imported) so this holds regardless of import order.
        import logging as _logging

        _oh_level = os.environ.setdefault("LOG_LEVEL", "WARNING")
        _logging.getLogger("openhands").setLevel(
            _logging.getLevelName(_oh_level.upper())
        )

        # Hide the SDK banner + the DockerWorkspace subprocess echo (docker
        # version/run/inspect/stop). Must run before importing the SDK so the
        # banner env var takes effect at import time.
        _silence_openhands_io()

        # Heavy imports are local so selecting mini-swe-agent never pays for them.
        from pydantic import SecretStr

        from openhands.sdk import LLM, Agent, Conversation, Tool
        from openhands.sdk import ConversationExecutionStatus as _Status
        from openhands.tools import FileEditorTool, TerminalTool

        # Rate-limit noise absorption. Must run AFTER the SDK imports above:
        # importing openhands.sdk configures the ROOT logger with a stderr
        # RichHandler, and the filter has to be attached to that handler (its
        # retry-mixin logs the full "Provider returned error 429" body via the
        # openhands.* tree, which would otherwise break the live TUI). This
        # thread absorbs when its console output is suppressed (TUI mode).
        install_rate_limit_log_filter()
        set_thread_noise_absorbed(not console_echo)

        log_fp = None
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_file, "a", encoding="utf-8", buffering=1)
            log_fp.write(
                f"\n═══ agent start role={role_label} harness=openhands "
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

        api_key = os.environ["SWE_DUEL_OPENROUTER_API_KEY"]
        # Selected reasoning effort / OpenRouter provider flow through
        # litellm_extra_body (merged into the OpenRouter request body). When an
        # effort is selected we also suppress the SDK's own reasoning_effort
        # default ("high") so a single reasoning-effort field reaches the API.
        extra_body = openrouter_body_extras(self.model_config)
        llm_kwargs: dict[str, Any] = {
            "model": self._openrouter_model_for_litellm(),
            "api_key": SecretStr(api_key),
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": self.model_config.temperature,
            "max_output_tokens": self.model_config.max_tokens,
            "usage_id": f"{self.harness_id}:{role_label}",
            "drop_params": True,
        }
        if extra_body:
            llm_kwargs["litellm_extra_body"] = extra_body
            if "reasoning" in extra_body:
                llm_kwargs["reasoning_effort"] = None
        # Client-side rate limiting: when the (model, provider) selection has
        # a provider_rate_limits cap, build an LLM subclass whose completion
        # methods acquire the process-wide throttle first. Test stubs may
        # provide a non-class "LLM" — then subclassing is impossible and we
        # fall back to the unwrapped LLM (throttling skipped, never a crash).
        throttle = throttle_for(self.model_config)
        if throttle is not None:

            def _on_throttle_wait(waited: float) -> None:
                if waited >= 5.0:
                    _emit(
                        f"[{role_label}] ⏳ rate-limiter: waited {waited:.1f}s for "
                        f"{self.model_config.model_id} @ "
                        f"{self.model_config.provider or 'auto-route'} "
                        f"(shared process-wide budget)"
                    )

            try:
                llm_cls = make_rate_limited_llm_class(
                    LLM, throttle, _on_throttle_wait
                )
                llm = llm_cls(**llm_kwargs)
            except TypeError:
                llm = LLM(**llm_kwargs)
        else:
            llm = LLM(**llm_kwargs)
        agent = Agent(
            llm=llm,
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        )

        workspace, workspace_ctx = self._build_workspace(
            docker_image=docker_image,
            host_workspace=workspace_path,
            preserve_paths=preserve_paths or [],
        )

        # ── live step counter + per-run/wall-clock guards via the callback ──
        #
        # `step`      = cumulative ActionEvents across the initial run + every
        #               recovery turn (drives the TUI progress + trajectory).
        # `run_step`  = ActionEvents in the *current* run() only. The OpenHands
        #               server enforces `max_iteration_per_run` per run and
        #               resets its counter to 0 on each run(); it is fixed at
        #               conversation creation and not mutable afterwards. To
        #               honour "initial budget + 3×6-step recovery turns" we
        #               enforce the per-run ceiling ourselves: when `run_step`
        #               reaches `run_limit` we `pause()`, which makes the
        #               blocking run() return cleanly so the recovery loop can
        #               take the next turn.
        started = time.monotonic()
        wall = float(max_wall_seconds) if max_wall_seconds and max_wall_seconds > 0 else None
        state = {
            "step": 0,           # cumulative steps (all runs)
            "run_step": 0,       # steps in the current run only
            "run_limit": int(max_steps),  # per-run ceiling (initial = max_steps)
            "limit": int(max_steps),      # cumulative budget shown on the TUI
            "wall_hit": False,
            "budget_paused": False,  # set when we pause for the per-run ceiling
            "last_step_at": started,
        }
        # Capture every event as it streams. For a RemoteConversation (Docker)
        # `conversation.state.events` is a lazily-synced server view that can
        # come back empty after a pause→error→idle sequence; the live callback
        # stream is the authoritative record, so we build the trajectory from
        # this list and fall back to state.events only if it's empty.
        captured_events: list[Any] = []

        def _on_event(event: Any) -> None:
            captured_events.append(event)
            # An ActionEvent is emitted each time the LLM produces a tool call.
            cls_name = type(event).__name__
            if cls_name == "ActionEvent":
                state["step"] += 1
                state["run_step"] += 1
                now = time.monotonic()
                step_elapsed = now - state["last_step_at"]
                cum_elapsed = now - started
                state["last_step_at"] = now
                if step_callback is not None:
                    try:
                        step_callback(int(state["step"]), int(state["limit"]))
                    except Exception:
                        pass
                if verbose:
                    thought = _event_thought(event)
                    action = _event_action_str(event)
                    _emit(
                        f"[{role_label}] ▶ step {state['step']} thought: "
                        f"{_truncate(thought, 400)}"
                    )
                    if action:
                        _emit(f"[{role_label}]     action: {_truncate(action, 400)}")
                    _emit(
                        f"[{role_label}]     step_time_taken={step_elapsed:.1f}s "
                        f"cum_time_taken={cum_elapsed:.1f}s"
                    )
            elif cls_name in ("ObservationEvent", "AgentErrorEvent") and verbose:
                _emit(f"[{role_label}]     observation: {_truncate(_event_observation(event), 600)}")

            # Wall-clock enforcement: pausing makes run() return; we then mark a
            # WallClockTimeout exit_status (mirrors mini-swe semantics).
            if wall is not None and not state["wall_hit"]:
                if (time.monotonic() - started) > wall:
                    state["wall_hit"] = True
                    _emit(
                        f"[{role_label}] ⏱ wall-clock limit hit "
                        f"({wall:.0f}s) — pausing agent loop"
                    )
                    try:
                        conversation.pause()
                    except Exception:
                        pass
                    return

            # Per-run step-budget enforcement: once the current run has used its
            # ceiling, pause so the blocking run() returns and control comes back
            # to the recovery loop. Equivalent to mini-swe's LimitsExceeded exit.
            if (
                cls_name == "ActionEvent"
                and not state["budget_paused"]
                and state["run_step"] >= state["run_limit"]
            ):
                state["budget_paused"] = True
                if verbose:
                    _emit(
                        f"[{role_label}] ⏹ per-run step budget hit "
                        f"({state['run_step']}/{state['run_limit']}) — pausing"
                    )
                try:
                    conversation.pause()
                except Exception:
                    pass

        conversation = Conversation(
            agent,
            workspace=workspace,
            callbacks=[_on_event],
            max_iteration_per_run=int(max_steps),
            stuck_detection=True,
            visualizer=None,
        )

        if verbose:
            wall_msg = f" wall_limit={wall:.0f}s" if wall else ""
            _emit(
                f"[{role_label}] ═══ agent start harness=openhands "
                f"model={self.model_config.model_id} {self._identity_suffix()} "
                f"cwd={workspace_path} "
                f"step_limit={max_steps}{wall_msg}"
            )

        def _wall_exceeded() -> bool:
            if wall is None:
                return False
            if (time.monotonic() - started) > wall:
                state["wall_hit"] = True
                return True
            return False

        def _sync_steps_from_server() -> None:
            """Reconcile the cumulative step count from the server after a run.

            The event WebSocket can silently drop deliveries, so the live
            callback's `state["step"]` may undercount. The server's reconciled
            ActionEvent total is authoritative; trust it when it's higher and
            refresh the TUI so progress reflects work actually done."""
            try:
                total = _count_actions(self._events(conversation))
            except Exception:
                return
            if total > state["step"]:
                state["step"] = total
                if step_callback is not None:
                    try:
                        step_callback(int(state["step"]), int(state["limit"]))
                    except Exception:
                        pass

        exit_status = ""
        try:
            # ── Initial pass (full max_steps budget) ──
            # The OpenHands agent can call its builtin `finish` tool and end a
            # run() early — often after only exploring/planning, before it has
            # written the required `_swe-duel/*` artifacts. mini-swe keeps stepping
            # until its step_limit; to match that, we re-prompt and resume within
            # the SAME initial budget whenever the agent finishes early with
            # artifacts still missing, until either the artifacts exist or the
            # full max_steps initial budget is consumed. send_message() resets a
            # FINISHED conversation to IDLE and run() resumes from there.
            #
            # Hitting the budget surfaces as a ConversationRunError wrapping
            # MaxIterationsReached (or our own pause). Swallow it — exactly like
            # mini-swe swallows LimitsExceeded — so we still fall through to the
            # recovery loop instead of escaping to the outer handler.
            conversation.send_message(task_prompt)
            while True:
                try:
                    self._run_quietly(conversation)
                except Exception as e:
                    if verbose:
                        _emit(
                            f"[{role_label}] ⚠ initial run raised: "
                            f"{type(e).__name__}: {e}"
                        )
                    _sync_steps_from_server()
                    break
                _sync_steps_from_server()
                # Stop the initial pass if: nothing left to nudge, we already
                # spent the whole initial budget, the per-run ceiling tripped
                # (treated as budget exhaustion), or the wall-clock blew.
                if completion_check is None or reminder_builder is None:
                    break
                missing = completion_check() or []
                if not missing:
                    break
                if (
                    state["budget_paused"]
                    or state["step"] >= int(max_steps)
                    or _wall_exceeded()
                ):
                    break
                # Agent finished early with work outstanding — nudge and resume
                # on the remaining initial budget (cap stays at max_steps).
                if verbose:
                    _emit(
                        f"[{role_label}] ↺ agent finished early at step "
                        f"{state['step']}/{max_steps} with missing={missing}; "
                        "resuming within initial budget"
                    )
                try:
                    conversation.send_message(reminder_builder(missing))
                except Exception:
                    break

            # ── Recovery loop: nudge the agent to finish missing artifacts ──
            # Begins only after the initial budget is spent. Each turn grants a
            # fresh _RECOVERY_BUDGET_PER_TURN-step ceiling (enforced by the
            # callback's pause()) and GROWS the displayed denominator by that
            # amount, so the TUI shows the extra budget the recovery turns add.
            # The server resets its own per-run iteration counter on each run(),
            # so a turn always starts from a clean budget even after the previous
            # run hit MaxIterationsReached.
            if completion_check is not None and reminder_builder is not None:
                for turn in range(1, max_recovery_turns + 1):
                    if state["wall_hit"] or _wall_exceeded():
                        break
                    missing = completion_check() or []
                    if not missing:
                        break
                    if verbose:
                        _emit(
                            f"[{role_label}] ↻ recovery turn {turn}/{max_recovery_turns}: "
                            f"missing={missing}"
                        )
                    # Fresh per-run budget; grow the displayed cumulative budget.
                    state["run_step"] = 0
                    state["run_limit"] = _RECOVERY_BUDGET_PER_TURN
                    state["budget_paused"] = False
                    state["limit"] = int(max_steps) + turn * _RECOVERY_BUDGET_PER_TURN
                    try:
                        conversation.send_message(reminder_builder(missing))
                        self._run_quietly(conversation)
                        _sync_steps_from_server()
                    except Exception as e:
                        if verbose:
                            _emit(
                                f"[{role_label}] ⚠ recovery run raised: "
                                f"{type(e).__name__}: {e}"
                            )
                        _sync_steps_from_server()
                        continue

            exit_status = self._exit_status(
                conversation, _Status, state["wall_hit"],
                budget_paused=state["budget_paused"],
                missing=(completion_check() if completion_check else []),
            )
        except Exception as e:
            if verbose:
                _emit(f"[{role_label}] ⚠ run raised: {type(e).__name__}: {e}")
            exit_status = "WallClockTimeout" if state["wall_hit"] else "Error"
        finally:
            trajectory = self._extract_trajectory(
                conversation,
                duration_seconds=time.monotonic() - started,
                exit_status=exit_status,
                captured_events=captured_events,
                task_prompt=task_prompt,
            )
            try:
                conversation.close()
            except Exception:
                pass
            # Stop+remove the agent-server container.
            if workspace_ctx is not None:
                try:
                    workspace_ctx()
                except Exception:
                    pass

        if verbose:
            final_missing = completion_check() if completion_check else []
            tail = (
                f" (missing still: {final_missing})" if final_missing else " OK"
            )
            _emit(
                f"[{role_label}] ═══ agent end{tail} "
                f"duration={trajectory.duration_seconds:.1f}s "
                f"cost=${trajectory.total_cost_usd:.4f} exit={exit_status}"
            )
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass
        return trajectory

    # ── internals ──────────────────────────────────────────

    def _openrouter_model_for_litellm(self) -> str:
        """OpenHands' LLM uses litellm; route through OpenRouter via the prefix."""
        mid = self.model_config.model_id
        if mid.startswith("openrouter/"):
            return mid
        return f"openrouter/{mid}"

    def _build_workspace(
        self,
        *,
        docker_image: str | None,
        host_workspace: Path,
        preserve_paths: list[str],
    ) -> tuple[Any, Callable[[], None] | None]:
        """Return (workspace, cleanup) for the conversation.

        With ``docker_image`` the agent runs inside that image (which must have
        the openhands agent-server baked in) with ``host_workspace`` bind-mounted
        at :data:`CONTAINER_WORKSPACE`. Without it (host unit/integration tests),
        a plain on-disk path is used so the agent runs locally.
        """
        if not docker_image:
            return str(Path(host_workspace).resolve()), None

        from openhands.workspace import DockerWorkspace

        host_ws = str(Path(host_workspace).resolve())
        # Mask each preserve_path with an anonymous volume so the image-baked
        # contents at that path (e.g. node_modules with tsx/supertest/esbuild)
        # show THROUGH the bind mount instead of being hidden by the host
        # clone's empty/absent dir. Without this the agent finds e.g. `tsx: not
        # found` and burns steps running `npm install`. The volume entries are
        # passed verbatim to `docker run -v`; a bare container path = anonymous
        # volume. Mirrors MiniSweHarness._build_docker_environment.
        volumes = [f"{host_ws}:{CONTAINER_WORKSPACE}"]
        for sub in preserve_paths or []:
            sub = sub.strip().strip("/")
            if sub:
                volumes.append(f"{CONTAINER_WORKSPACE}/{sub}")

        # Redirect the agent-server's own persistence (conversation/event state,
        # bash-event logs) OUT of the bind-mounted workspace. The server runs as
        # ROOT inside the container and otherwise writes `workspace/conversations`
        # (relative to its /workspace cwd) into the host bind mount as root-owned
        # files. That (a) corrupts the challenge diff and (b) makes the host-side
        # `get_modified_files` rglob crash with PermissionError when it later
        # walks the workspace. Point these at an in-container path that is NOT
        # under CONTAINER_WORKSPACE so nothing root-owned lands on the host.
        # `forward_env` only forwards vars that exist in our process env, so set
        # them here. The OH_-prefixed names are read by the agent-server's
        # `from_env(Config, "OH")` loader (config.py: conversations_path /
        # bash_events_dir / workspace_path).
        server_state_dir = "/tmp/oh_server_state"
        os.environ["OH_CONVERSATIONS_PATH"] = f"{server_state_dir}/conversations"
        os.environ["OH_BASH_EVENTS_DIR"] = f"{server_state_dir}/bash_events"
        ws = DockerWorkspace(
            server_image=docker_image,
            working_dir=CONTAINER_WORKSPACE,
            volumes=volumes,
            host_port=None,
            forward_env=[
                "SWE_DUEL_OPENROUTER_API_KEY",
                "OH_CONVERSATIONS_PATH",
                "OH_BASH_EVENTS_DIR",
            ],
            # The SDK otherwise spawns a background `docker logs -f` thread that
            # pipes the agent-server's HTTP access log to our stdout (every line
            # prefixed `[DOCKER] … h11_impl.py …`), which buries the TUI progress
            # bar. We capture per-step reasoning from typed events instead, so
            # the raw container log is noise — turn the streamer off.
            detach_logs=False,
        )

        # OpenHands' DockerWorkspace hardcodes `--name agent-server-<uuid>` with
        # no override hook, so rename the started container to our `swe-duel-` prefix
        # for consistency with the mini-swe + sandbox containers (and so the
        # `swe-duel-`-prefix cleanup fallback reaps it). Rename by container id keeps
        # every subsequent SDK operation (which addresses the container by id)
        # working. Best-effort: a failed rename is harmless.
        container_id = getattr(ws, "_container_id", None)
        if container_id:
            try:
                subprocess.run(
                    [
                        "docker", "rename", container_id,
                        swe_duel_container_name(repo_from_image(docker_image), self.harness_id),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                pass

        def _cleanup() -> None:
            for name in ("cleanup", "close", "__exit__"):
                fn = getattr(ws, name, None)
                if callable(fn):
                    try:
                        fn() if name != "__exit__" else fn(None, None, None)
                        return
                    except Exception:
                        continue

        return ws, _cleanup

    @staticmethod
    def _run_quietly(conversation: Any) -> None:
        """Drive one blocking conversation.run().

        The per-run step budget is enforced by the event callback (it calls
        ``conversation.pause()`` once the run's step ceiling is reached), and the
        server resets its own iteration counter on every run(), so we never need
        to pass an iteration argument here — both ``Local`` and ``Remote``
        conversations take ``run()`` with no required args.
        """
        conversation.run()

    @staticmethod
    def _exit_status(
        conversation: Any,
        status_enum: Any,
        wall_hit: bool,
        *,
        budget_paused: bool = False,
        missing: list[str] | None = None,
    ) -> str:
        """Normalise the final conversation status into our exit_status vocab.

        Mirrors mini-swe's semantics, where artifact completeness — not the raw
        loop status — is the authoritative success signal: the recovery loop
        terminates precisely when ``completion_check()`` reports nothing missing.
        So:
          - wall-clock blown          → ``WallClockTimeout`` (always wins)
          - all artifacts present     → ``Submitted`` (the agent did the job,
            regardless of whether the final run ended FINISHED, was paused on
            its per-run budget, or read back as IDLE on the remote path)
          - artifacts still missing   → classify by why we stopped:
            STUCK → ``Stuck``; ERROR / per-run-budget pause → ``LimitsExceeded``;
            otherwise the raw status string.
        """
        if wall_hit:
            return "WallClockTimeout"
        if not bool(missing):
            return "Submitted"
        try:
            st = conversation.state.execution_status
        except Exception:
            st = None
        if st == status_enum.STUCK:
            return "Stuck"
        if (
            st == status_enum.ERROR
            or st == getattr(status_enum, "PAUSED", object())
            or budget_paused
        ):
            return "LimitsExceeded"
        return str(getattr(st, "value", st) or "")

    def _extract_trajectory(
        self,
        conversation: Any,
        duration_seconds: float,
        exit_status: str,
        captured_events: list[Any] | None = None,
        task_prompt: str = "",
    ) -> AgentTrajectory:
        """Build an AgentTrajectory from conversation events + metrics.

        Mini-swe ordering is preserved: each step's ``observation`` is what
        **preceded** the thought/action (step 1 seeds with ``task_prompt``;
        subsequent steps get the prior Action's ObservationEvent text). Thought
        comes from the ActionEvent's reasoning fields; action is the tool call.

        Sources events from both the live callback stream (``captured_events``)
        and the server (``conversation.state.events``, force-reconciled over
        REST) and uses whichever yielded more ActionEvents.
        """
        server_events = self._events(conversation)
        cb_events = list(captured_events or [])
        events = (
            server_events
            if _count_actions(server_events) >= _count_actions(cb_events)
            else cb_events
        )
        metrics = self._metrics(conversation)

        token_usages = list(getattr(metrics, "token_usages", []) or [])
        costs = list(getattr(metrics, "costs", []) or [])

        # Index observations by the action they answer (for the *next* step).
        obs_by_action: dict[str, str] = {}
        for ev in events:
            if type(ev).__name__ in ("ObservationEvent", "AgentErrorEvent"):
                aid = getattr(ev, "action_id", None) or getattr(ev, "tool_call_id", None)
                if aid:
                    obs_by_action[str(aid)] = _event_observation(ev)

        steps: list[dict[str, Any]] = []
        total_in = total_out = total_cached = 0
        total_cost = 0.0
        usage_i = 0
        cost_i = 0
        prev_ts: float | None = None
        first_ts: float | None = None
        # Seed with the task prompt so step 1's observation matches mini-swe.
        pending_observation = task_prompt or ""

        for ev in events:
            if type(ev).__name__ != "ActionEvent":
                continue
            ts = _event_ts(ev)
            if first_ts is None:
                first_ts = ts
            step_seconds = (ts - prev_ts) if (ts is not None and prev_ts is not None) else 0.0
            cum_seconds = (ts - first_ts) if (ts is not None and first_ts is not None) else 0.0
            prev_ts = ts

            in_tok = out_tok = cached = 0
            if usage_i < len(token_usages):
                tu = token_usages[usage_i]
                in_tok = int(getattr(tu, "prompt_tokens", 0) or 0)
                out_tok = int(getattr(tu, "completion_tokens", 0) or 0)
                cached = int(getattr(tu, "cache_read_tokens", 0) or 0)
                usage_i += 1
            cost = 0.0
            if cost_i < len(costs):
                cost = float(getattr(costs[cost_i], "cost", 0.0) or 0.0)
                cost_i += 1
            if cost <= 0.0:
                # OpenHands' telemetry already prefers OpenRouter's reported
                # per-response cost; when it has none, price the tokens with
                # the cached OpenRouter per-token rates.
                cost = fallback_cost_for(self.model_config, in_tok, out_tok, cached)

            total_in += in_tok
            total_out += out_tok
            total_cached += cached
            total_cost += cost

            aid = str(getattr(ev, "id", "") or "")
            steps.append({
                "observation": pending_observation,
                "thought": _event_thought(ev),
                "action": _event_action_str(ev),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_tokens": cached,
                "cost_usd": cost,
                "step_seconds": float(step_seconds or 0.0),
                "cum_seconds": float(cum_seconds or 0.0),
            })
            # Tool result becomes the observation for the following step.
            pending_observation = obs_by_action.get(aid, "")

        # Prefer the metrics' authoritative accumulated cost when present.
        acc_cost = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        total_cost_usd = acc_cost if acc_cost > 0.0 else total_cost

        return AgentTrajectory(
            steps=steps,
            total_steps=len(steps),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost_usd=total_cost_usd,
            model_id=self.model_config.model_id,
            duration_seconds=duration_seconds,
            exit_status=exit_status,
        )

    @staticmethod
    def _events(conversation: Any) -> list[Any]:
        # Force a server-side reconcile first: on the remote path the events
        # list is WebSocket-fed and can lag or miss deliveries, but exposes a
        # reconcile() that re-fetches the full event log over REST.
        try:
            ev_list = conversation.state.events
        except Exception:
            return []
        try:
            reconcile = getattr(ev_list, "reconcile", None)
            if callable(reconcile):
                reconcile()
        except Exception:
            pass
        try:
            return list(ev_list)
        except Exception:
            return []

    @staticmethod
    def _metrics(conversation: Any) -> Any:
        # ConversationStats aggregates per-usage Metrics; combined is what we want.
        for getter in (
            lambda: conversation.conversation_stats.get_combined_metrics(),
            lambda: conversation.state.stats.get_combined_metrics(),
        ):
            try:
                m = getter()
                if m is not None:
                    return m
            except Exception:
                continue
        return None


# ── event field helpers (kept module-level for unit testing) ──────────────


def _count_actions(events: list[Any]) -> int:
    """Number of ActionEvents in an event list (how many agent steps it holds)."""
    return sum(1 for ev in events if type(ev).__name__ == "ActionEvent")


def _event_ts(event: Any) -> float | None:
    """Parse an event's ISO timestamp into epoch seconds; None if unavailable."""
    ts = getattr(event, "timestamp", None)
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _event_thought(event: Any) -> str:
    """Flatten an ActionEvent's reasoning text from all known OH fields."""
    parts: list[str] = []

    def _push(text: Any) -> None:
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    thought = getattr(event, "thought", None)
    if isinstance(thought, (list, tuple)):
        for t in thought:
            txt = getattr(t, "text", None)
            if isinstance(txt, str) and txt:
                _push(txt)
            elif isinstance(t, str) and t:
                _push(t)
            elif isinstance(t, dict):
                for key in ("text", "thinking", "content", "reasoning"):
                    v = t.get(key)
                    if isinstance(v, str) and v.strip():
                        _push(v)
                        break
    elif isinstance(thought, str):
        _push(thought)

    _push(getattr(event, "reasoning_content", None))
    _push(getattr(event, "summary", None))

    for block in getattr(event, "thinking_blocks", None) or []:
        if isinstance(block, dict):
            for key in ("thinking", "text", "content"):
                v = block.get(key)
                if isinstance(v, str) and v.strip():
                    _push(v)
                    break
        else:
            for key in ("thinking", "text", "content"):
                v = getattr(block, key, None)
                if isinstance(v, str) and v.strip():
                    _push(v)
                    break

    ritem = getattr(event, "responses_reasoning_item", None)
    if ritem is not None:
        if isinstance(ritem, dict):
            for key in ("summary", "content", "text", "reasoning"):
                v = ritem.get(key)
                if isinstance(v, str) and v.strip():
                    _push(v)
                elif isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, str) and sub.strip():
                            _push(sub)
                        elif isinstance(sub, dict):
                            for sk in ("text", "content"):
                                sv = sub.get(sk)
                                if isinstance(sv, str) and sv.strip():
                                    _push(sv)
                                    break
        else:
            for key in ("summary", "content", "text", "reasoning"):
                v = getattr(ritem, key, None)
                if isinstance(v, str) and v.strip():
                    _push(v)
                elif isinstance(v, list):
                    for sub in v:
                        stext = getattr(sub, "text", None) if not isinstance(sub, str) else sub
                        if isinstance(stext, str) and stext.strip():
                            _push(stext)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "\n".join(uniq)


def _event_action_str(event: Any) -> str:
    """Render the executable action of an ActionEvent as a human string."""
    action = getattr(event, "action", None)
    if action is None:
        return ""
    # Terminal/bash actions carry `command`; file edits carry path/command.
    for attr in ("command",):
        val = getattr(action, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # FileEditor and others: summarise tool_name + key fields.
    tool = getattr(event, "tool_name", "") or type(action).__name__
    bits: list[str] = [str(tool)]
    for attr in ("command", "path", "file_text", "old_str", "new_str", "insert_line"):
        val = getattr(action, attr, None)
        if val is None:
            continue
        bits.append(f"{attr}={_truncate(str(val), 120)}")
    return " ".join(bits)


def _event_observation(event: Any) -> str:
    """Extract observation text from an ObservationEvent / AgentErrorEvent."""
    err = getattr(event, "error", None)
    if isinstance(err, str) and err:
        return err
    obs = getattr(event, "observation", None)
    if obs is None:
        return ""
    for attr in ("content", "text"):
        val = getattr(obs, attr, None)
        if isinstance(val, str) and val:
            return val
    to_text = getattr(obs, "to_llm_content", None)
    if callable(to_text):
        try:
            return str(to_text())
        except Exception:
            pass
    return str(obs)
