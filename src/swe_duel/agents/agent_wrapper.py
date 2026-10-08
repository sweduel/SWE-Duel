"""Backwards-compatibility shim.

The mini-swe-agent runtime moved to :mod:`swe_duel.agents.harness.mini_swe` behind
the :class:`swe_duel.agents.harness.base.AgentHarness` interface, which now also has
an OpenHands implementation. ``AgentWrapper`` remains as an alias for the
mini-swe-agent harness so existing imports keep working; new code should select
a harness via :func:`swe_duel.agents.harness.get_harness`.
"""

from __future__ import annotations

from swe_duel.agents.harness.base import CONTAINER_WORKSPACE
from swe_duel.agents.harness.mini_swe import MiniSweHarness

# Historical name. Identical to the mini-swe-agent harness.
AgentWrapper = MiniSweHarness

__all__ = ["AgentWrapper", "MiniSweHarness", "CONTAINER_WORKSPACE"]
