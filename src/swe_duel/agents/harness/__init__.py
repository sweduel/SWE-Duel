"""Pluggable coding-agent harnesses (mini-swe-agent, OpenHands, Codex, Claude Code, …)."""

from swe_duel.agents.harness.base import (
    CONTAINER_WORKSPACE,
    HARNESS_EFFORT_SUPPORT,
    HARNESS_IDS,
    HARNESS_PROVIDER_SUPPORT,
    AgentHarness,
    get_harness,
    harness_display_name,
    harness_supported_efforts,
    harness_supports_provider,
)

__all__ = [
    "AgentHarness",
    "CONTAINER_WORKSPACE",
    "HARNESS_EFFORT_SUPPORT",
    "HARNESS_IDS",
    "HARNESS_PROVIDER_SUPPORT",
    "get_harness",
    "harness_display_name",
    "harness_supported_efforts",
    "harness_supports_provider",
]
