"""Agent-harness abstraction.

A *harness* is a concrete coding-agent runtime (mini-swe-agent, OpenHands, Codex,
Claude Code, …) that, given a model and a task prompt, autonomously explores a
workspace, runs commands, edits files, and returns an :class:`AgentTrajectory`.

The framework treats the harness as a first-class, selectable dimension of a
competitor: a competitor is identified by ``(model_id, harness_id,
reasoning_effort, provider)``. All agent roles (Red feature/bug/self-review,
Blue) go through a harness, so the rest of the codebase never imports a specific
runtime — it asks the registry for one via :func:`get_harness` and calls
:meth:`AgentHarness.run`.

Not every harness can express every identity dimension. mini-swe-agent and
OpenHands call litellm directly, so they forward the OpenRouter request-body
``reasoning`` and ``provider`` objects for any effort string / provider slug.
The Codex CLI only exposes reasoning effort via its ``model_reasoning_effort``
config key (a fixed enum, no "max"/"none") and has no request-body injection at
all, so it can express a subset of efforts and NO provider pinning. The Claude
Code CLI has no reasoning-effort or provider controls. :data:`HARNESS_EFFORT_SUPPORT`
/ :data:`HARNESS_PROVIDER_SUPPORT` declare these capabilities so the selection
TUIs can filter choices and :func:`get_harness` can fail fast on an identity a
harness would silently not honour.

To add a new harness: implement :class:`AgentHarness` in a new module, register
it in :data:`_HARNESS_FACTORIES`, add its id to :data:`HARNESS_IDS` and its
capability entries to the support tables, install any in-image binaries via
``docker/install-*.sh``, and document the steps in ``harnesses.md``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from swe_duel.config import ModelConfig
from swe_duel.models import AgentTrajectory

# Absolute path at which the host workspace is mounted inside the agent's
# container, for every harness. Per-repo Docker images bake their dependencies
# here, so mounting the host clone at this exact path lets the agent use those
# baked deps while its source edits persist back to the host for diffing.
CONTAINER_WORKSPACE = "/workspace"

# Canonical harness identifiers. Order is the display order in selectors.
HARNESS_IDS: tuple[str, ...] = (
    "mini-swe-agent",
    "openhands",
    "codex",
    "claude-code",
)

_HARNESS_DISPLAY = {
    "mini-swe-agent": "mini-swe-agent",
    "openhands": "OpenHands",
    "codex": "Codex",
    "claude-code": "Claude Code",
}

# ── Identity-dimension capabilities ─────────────────────────────────────
#
# A competitor identity is (model_id, harness_id, reasoning_effort, provider).
# Each harness declares which non-empty selections it can actually honour:

# Effort strings the harness can send to OpenRouter. ``None`` = any string is
# forwarded verbatim (litellm extra_body). A frozenset = the closed enum the
# CLI accepts; empty frozenset = no reasoning-effort control at all.
HARNESS_EFFORT_SUPPORT: dict[str, frozenset[str] | None] = {
    "mini-swe-agent": None,  # any effort via request-body reasoning.effort
    "openhands": None,        # any effort via litellm_extra_body
    "codex": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
    "claude-code": frozenset(),
}

# Whether the harness can pin a specific OpenRouter provider slug via the
# request-body provider.order object.
HARNESS_PROVIDER_SUPPORT: dict[str, bool] = {
    "mini-swe-agent": True,
    "openhands": True,
    "codex": False,      # no request-body injection in codex config.toml
    "claude-code": False,  # no routing controls on the Anthropic gateway path
}

# Whether the harness can honour a `provider_rate_limits` throttle from
# config/models.yaml. mini-swe-agent and OpenHands make their LLM calls from
# the host process (litellm), so the process-wide throttle in
# `swe_duel.agents.harness.rate_limit` applies; the Codex / Claude Code CLIs run
# in-container and make their own API calls, so a configured limit cannot be
# enforced there (warned at TUI startup by `warn_unenforceable_rate_limits`,
# NOT fail-fast — traffic shaping is not competitor identity, and the CLIs'
# own retry loops still absorb transient provider 429s).
HARNESS_RATE_LIMIT_SUPPORT: dict[str, bool] = {
    "mini-swe-agent": True,
    "openhands": True,
    "codex": False,
    "claude-code": False,
}


def harness_supported_efforts(harness_id: str) -> frozenset[str] | None:
    """Effort strings this harness can express (None = any string)."""
    return HARNESS_EFFORT_SUPPORT.get(harness_id, frozenset())


def harness_supports_provider(harness_id: str) -> bool:
    """Whether this harness can pin an OpenRouter provider slug."""
    return HARNESS_PROVIDER_SUPPORT.get(harness_id, False)


def openrouter_body_extras(model_config: ModelConfig) -> dict[str, dict[str, Any]]:
    """OpenRouter request-body extras for the participant's selection.

    Returns the dict to merge into the request body (litellm ``extra_body`` /
    OpenHands ``litellm_extra_body``):

    * ``reasoning: {"effort": <e>}`` when a reasoning effort is selected
      (empty selection = model-default reasoning behaviour; nothing is sent).
    * ``provider: {"order": [<slug>], "allow_fallbacks": False}`` when an
      OpenRouter provider is pinned. Fallbacks are disabled so a pinned
      provider is *pinned*: a would-be-silently-re-routed request fails loudly
      instead of recording a defense/generation under a dishonest identity.
    """
    extra_body: dict[str, dict[str, Any]] = {}
    effort = getattr(model_config, "reasoning_effort", "") or ""
    provider = getattr(model_config, "provider", "") or ""
    if effort:
        extra_body["reasoning"] = {"effort": effort}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    return extra_body


class AgentHarness(ABC):
    """One coding-agent runtime bound to a single model.

    Concrete subclasses must set ``harness_id`` and implement :meth:`run` with
    the exact contract documented below — every caller (red/blue/red_gates)
    relies on this signature and on a returned :class:`AgentTrajectory`.
    """

    #: Stable identifier, e.g. ``"mini-swe-agent"`` / ``"openhands"``.
    harness_id: str

    def _identity_suffix(self) -> str:
        """Compact ``effort=… provider=…`` log suffix for the selected identity."""
        effort = getattr(self.model_config, "reasoning_effort", "") or ""
        provider = getattr(self.model_config, "provider", "") or ""
        parts: list[str] = []
        if effort:
            parts.append(f"effort={effort}")
        if provider:
            parts.append(f"provider={provider}")
        return " ".join(parts)

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        if not os.environ.get("SWE_DUEL_OPENROUTER_API_KEY"):
            raise RuntimeError(
                "SWE_DUEL_OPENROUTER_API_KEY environment variable is not set. "
                "Agent harnesses route model calls through OpenRouter."
            )
        # Fail fast when the participant's identity carries a dimension this
        # harness cannot honour — silently ignoring it would record a dishonest
        # (model, harness, effort, provider) identity in the bank/defenses.
        effort = getattr(model_config, "reasoning_effort", "") or ""
        provider = getattr(model_config, "provider", "") or ""
        supported_efforts = HARNESS_EFFORT_SUPPORT.get(self.harness_id, frozenset())
        if effort and supported_efforts is not None and effort not in supported_efforts:
            raise ValueError(
                f"harness {self.harness_id!r} cannot express reasoning effort "
                f"{effort!r} (supported: {sorted(supported_efforts) or 'none'}). "
                f"Select the effort via the request-body-capable harnesses "
                f"(mini-swe-agent, openhands) or clear the selection."
            )
        if provider and not HARNESS_PROVIDER_SUPPORT.get(self.harness_id, False):
            raise ValueError(
                f"harness {self.harness_id!r} cannot pin an OpenRouter provider "
                f"(no request-body routing control). Provider {provider!r} would "
                f"be silently ignored; clear the selection or use "
                f"mini-swe-agent / openhands."
            )

    @abstractmethod
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
        """Launch the agent against ``workspace_path`` and return its trajectory.

        Contract shared by every harness:

        - ``completion_check`` returns the list of still-missing artifacts after
          the agent submits; while non-empty (and recovery turns remain) the
          harness nudges the agent with ``reminder_builder(missing)`` and lets it
          step further. Each recovery turn grants **6** extra steps; up to
          ``max_recovery_turns`` (default 3) turns run.
        - ``step_callback(current_step, max_steps)`` fires once per agent step so
          a live UI can render a progress bar.
        - ``console_echo=False`` suppresses stdout (the live UI owns the
          console); per-agent reasoning is still written to ``log_file``.
        - When ``docker_image`` is given the agent runs INSIDE that image with
          ``workspace_path`` bind-mounted at :data:`CONTAINER_WORKSPACE`, so the
          environment matches the validation gates exactly. ``preserve_paths``
          masks image-baked subpaths (e.g. ``node_modules``) and ``login_shell``
          selects a login shell. When ``docker_image`` is None the agent runs on
          the host (used by host-based unit/integration tests).
        - On exceeding ``max_wall_seconds`` the returned trajectory's
          ``exit_status`` is ``"WallClockTimeout"``.
        """
        raise NotImplementedError


def harness_display_name(harness_id: str) -> str:
    """Human-friendly label for a harness id."""
    return _HARNESS_DISPLAY.get(harness_id, harness_id)


def get_harness(harness_id: str, model_config: ModelConfig) -> AgentHarness:
    """Instantiate the harness named ``harness_id`` bound to ``model_config``.

    ``model_config`` carries the participant's selected ``reasoning_effort`` /
    ``provider`` (see :meth:`swe_duel.config.ModelConfig.with_selection`); the
    constructor rejects selections the harness cannot honour.

    Raises ``ValueError`` for an unknown id — there is intentionally **no
    default**: callers must select a harness explicitly.
    """
    factory = _HARNESS_FACTORIES.get(harness_id)
    if factory is None:
        raise ValueError(
            f"unknown harness {harness_id!r}; known: {sorted(_HARNESS_FACTORIES)}"
        )
    return factory(model_config)


def _mini_swe_factory(model_config: ModelConfig) -> AgentHarness:
    from swe_duel.agents.harness.mini_swe import MiniSweHarness

    return MiniSweHarness(model_config)


def _openhands_factory(model_config: ModelConfig) -> AgentHarness:
    from swe_duel.agents.harness.openhands import OpenHandsHarness

    return OpenHandsHarness(model_config)


def _codex_factory(model_config: ModelConfig) -> AgentHarness:
    from swe_duel.agents.harness.codex import CodexHarness

    return CodexHarness(model_config)


def _claude_code_factory(model_config: ModelConfig) -> AgentHarness:
    from swe_duel.agents.harness.claude_code import ClaudeCodeHarness

    return ClaudeCodeHarness(model_config)


# Registry of harness_id → factory. Lazy imports keep optional heavy runtimes
# (openhands) from being imported unless actually selected.
_HARNESS_FACTORIES: dict[str, Callable[[ModelConfig], AgentHarness]] = {
    "mini-swe-agent": _mini_swe_factory,
    "openhands": _openhands_factory,
    "codex": _codex_factory,
    "claude-code": _claude_code_factory,
}
