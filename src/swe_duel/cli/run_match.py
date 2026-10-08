#!/usr/bin/env python
"""Run a single match between two models on one repo (draws from challenge bank)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swe_duel.cli._common import default_prompt_dir
from swe_duel.cli._common import setup
from swe_duel.engine.match import MatchOrchestrator
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument(
        "--harness-a", default="mini-swe-agent",
        help="Agent harness for model A (mini-swe-agent | openhands | codex | claude-code).",
    )
    parser.add_argument(
        "--harness-b", default="mini-swe-agent",
        help="Agent harness for model B (mini-swe-agent | openhands | codex | claude-code).",
    )
    parser.add_argument(
        "--effort-a", default="",
        help="Reasoning effort for competitor A (from the model's models.yaml menu; empty = model default).",
    )
    parser.add_argument(
        "--provider-a", default="",
        help="OpenRouter provider slug for competitor A (models.yaml menu; empty = auto-route).",
    )
    parser.add_argument(
        "--effort-b", default="",
        help="Reasoning effort for competitor B (models.yaml menu; empty = model default).",
    )
    parser.add_argument(
        "--provider-b", default="",
        help="OpenRouter provider slug for competitor B (models.yaml menu; empty = auto-route).",
    )
    parser.add_argument(
        "--repos", nargs="+", required=True,
        help="One or more repos; the match spans all of them.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--turns-per-player", "--turns-per-agent",
        dest="turns_per_player", type=int, default=None,
    )
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--prompt-dir", default=default_prompt_dir())
    args = parser.parse_args()

    ctx = setup(args)
    arena_cfg = ctx["arena_config"]
    if args.turns_per_player is not None:
        arena_cfg.match.turns_per_player = args.turns_per_player

    repo_cfgs = []
    for rn in args.repos:
        if rn in ctx["repo_configs"]:
            repo_cfgs.append(ctx["repo_configs"][rn])
        else:
            print(
                f"[warn] repo {rn!r} not found in config/repos/ — skipping.",
                file=sys.stderr,
            )
    if not repo_cfgs:
        raise SystemExit(f"none of --repos {args.repos!r} matched config/repos/.")

    _runner_cache: dict[str, TestRunner] = {}

    def _test_runner_factory(rc) -> TestRunner:
        tr = _runner_cache.get(rc.name)
        if tr is None:
            ex = DockerExecutor(
                docker_image=rc.docker_image,
                timeout_s=arena_cfg.sandbox.timeout_seconds,
                memory_mb=arena_cfg.sandbox.memory_mb,
            )
            tr = TestRunner(executor=ex, repo_config=rc)
            _runner_cache[rc.name] = tr
        return tr

    model_configs_by_id = {c.model_id: c for c in ctx["model_configs"].values()}

    orchestrator = MatchOrchestrator(
        model_configs=model_configs_by_id,
        challenge_store=ctx["challenge_store"],
        workspace_manager=ctx["workspace_manager"],
        config=arena_cfg,
        artifact_logger=ctx["artifact_logger"],
        prompt_dir=Path(args.prompt_dir),
        test_runner_factory=_test_runner_factory,
    )

    from swe_duel.agents.harness import (
        HARNESS_IDS,
        harness_supported_efforts,
        harness_supports_provider,
    )
    from swe_duel.models import composite_id

    for h in (args.harness_a, args.harness_b):
        if h not in HARNESS_IDS:
            raise SystemExit(f"Unknown harness {h!r}; known: {list(HARNESS_IDS)}")

    def resolve(m: str) -> str:
        return ctx["model_configs"][m].model_id if m in ctx["model_configs"] else m

    for side, h, e, pv in (
        ("A", args.harness_a, args.effort_a, args.provider_a),
        ("B", args.harness_b, args.effort_b, args.provider_b),
    ):
        supported = harness_supported_efforts(h)
        if e and supported is not None and e not in supported:
            raise SystemExit(
                f"harness {h!r} cannot express reasoning effort {e!r} for "
                f"competitor {side} (supported: {sorted(supported) or 'none'})"
            )
        if pv and not harness_supports_provider(h):
            raise SystemExit(
                f"harness {h!r} cannot pin provider {pv!r} for competitor {side}"
            )

    # Competitors are (model, harness, effort, provider) tuples → composite ids.
    result = orchestrator.execute(
        composite_id(resolve(args.model_a), args.harness_a, args.effort_a, args.provider_a),
        composite_id(resolve(args.model_b), args.harness_b, args.effort_b, args.provider_b),
        repo_cfgs,
        seed=args.seed,
    )
    print(f"Match {result.match_id}: {result.outcome.value}")
    print(f"  A={result.model_a_total:.2f}  B={result.model_b_total:.2f}")
    print(f"  cost=${result.total_cost_usd:.4f}  duration={result.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
