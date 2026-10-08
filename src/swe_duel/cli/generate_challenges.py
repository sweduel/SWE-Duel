#!/usr/bin/env python
"""Generation-phase CLI: populate Challenge Bank with Red-generated challenges.

Participant selection is interactive and paged: a participant is the 4-tuple
(model, harness, reasoning_effort, provider), chosen via one page per dimension
(see ``participant_select.py``). After each 4-tuple the wizard returns to a
landing page listing the selected participants, where the user adds another,
removes the last, or starts generation. `--models` (+ `--harnesses`,
`--efforts`, `--providers`) can be passed to bypass the prompts (useful for
non-interactive automation).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swe_duel.cli._common import install_container_cleanup_handlers
from swe_duel.cli._common import default_prompt_dir
from swe_duel.cli._common import setup
from swe_duel.cli.participant_select import (
    Participant,
    bank_status,
    print_pool_table,
    resolve_cli_participants,
    select_participants,
)
from swe_duel.agents.harness import get_harness
from swe_duel.agents.harness.cost_tracking import warn_missing_pricing
from swe_duel.agents.harness.rate_limit import (
    summarize_rate_limits,
    warn_unenforceable_rate_limits,
)
from swe_duel.agents.red import RedAgent
from swe_duel.challenge_bank.generator import ChallengeGenerator
from swe_duel.challenge_bank.progress import (
    GenerationReporter,
    Status as _Status,
    live as reporter_live,
)
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.models import display_composite_id
from swe_duel.sandbox.docker_executor import DockerExecutor
from swe_duel.sandbox.test_runner import TestRunner
from swe_duel.validation.red_gates import RedGateValidator


def _select_participants_interactive(
    all_models: dict,
    store: ChallengeStore,
    repo_names: list[str],
    target_per_repo: int,
) -> list[Participant]:
    """Paged wizard: model → harness → effort → provider, then landing page."""
    print_pool_table(store, repo_names)
    print(
        f"Target = {target_per_repo} slots per repo per participant "
        f"(a slot is one generation effort: up to N attempts ending in one "
        f"success or a fully-failed chain)."
    )
    bank_ids, model_status, _slot_status = bank_status(store, all_models, repo_names)
    return select_participants(
        all_models,
        start_verb="Start challenge generation",
        require_slots=False,
        bank_ids=bank_ids,
        model_status=model_status,
    )


def main() -> int:
    install_container_cleanup_handlers()

    parser = argparse.ArgumentParser(description="Populate Challenge Bank")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional: skip the interactive wizard and use these model nicks/ids.",
    )
    parser.add_argument(
        "--harnesses",
        nargs="+",
        default=None,
        help=(
            "Agent harness(es) to run for the --models (e.g. mini-swe-agent "
            "openhands codex claude-code). Required when --models is given; "
            "ignored interactively (harness is picked per participant). "
            "No default — a harness must always be chosen explicitly."
        ),
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=None,
        help=(
            "Reasoning-effort selection(s) for --models (cross-producted with "
            "--harnesses/--providers). Each must be in the model's "
            "reasoning_efforts menu in config/models.yaml. Default: the model "
            "default (no reasoning parameter sent)."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help=(
            "OpenRouter provider slug(s) for --models (cross-producted with "
            "--harnesses/--efforts). Each must be in the model's providers menu "
            "in config/models.yaml, and the harness must support provider "
            "pinning (mini-swe-agent, openhands). Default: OpenRouter "
            "auto-routing."
        ),
    )
    parser.add_argument("--repos", nargs="+", required=True)
    parser.add_argument("--target-per-repo", type=int, default=5)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--repos-dir", default="repos")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--prompt-dir", default=default_prompt_dir())
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "How many (participant, repo) generation pools to run in parallel. "
            "Defaults to challenge_bank.max_workers in arena.yaml."
        ),
    )
    args = parser.parse_args()

    # Load all models (no model filter) so the interactive UI can show the
    # full registry; we apply the selection ourselves below.
    cli_models = args.models
    args.models = None
    ctx_full = setup(args)
    all_models = ctx_full["model_configs"]
    repo_configs = ctx_full["repo_configs"]
    store = ctx_full["challenge_store"]

    if not repo_configs:
        print(f"[gen] No repos matched {args.repos}; aborting.", file=sys.stderr)
        return 2

    # Participants = (model, harness, reasoning_effort, provider) 4-tuples.
    if cli_models:
        if not args.harnesses:
            print(
                "[gen] --harnesses is required when --models is given; aborting.",
                file=sys.stderr,
            )
            return 2
        participants = resolve_cli_participants(
            all_models,
            models=list(cli_models),
            harnesses=list(args.harnesses),
            efforts=args.efforts,
            providers=args.providers,
        )
    else:
        participants = _select_participants_interactive(
            all_models=all_models,
            store=store,
            repo_names=list(repo_configs.keys()),
            target_per_repo=args.target_per_repo,
        )

    if not participants:
        print("[gen] No participant selected; aborting.", file=sys.stderr)
        return 2

    # Warn up front when OpenRouter has no pricing for a selected (model,
    # provider) — cost tracking for those participants will report $0.00.
    warn_missing_pricing(p.model_config for p in participants)
    # Surface per-(model, provider) request-rate caps before the live TUI
    # starts (the throttle is process-wide; workers divide one budget), and
    # warn when a selected harness cannot enforce a configured cap.
    summarize_rate_limits(p.model_config for p in participants)
    warn_unenforceable_rate_limits(
        (p.model_config, p.harness_id) for p in participants
    )

    arena_cfg = ctx_full["arena_config"]
    workspace_manager = ctx_full["workspace_manager"]

    max_workers = args.max_workers or arena_cfg.challenge_bank.max_workers
    max_workers = max(1, int(max_workers))

    # Flatten the (participant, repo) product into pool tasks. Each task runs
    # one generate_pool() — targets within it stay sequential so previous_gists
    # diversity hints keep working.
    pool_tasks = [
        (participant, repo_name, repo_cfg)
        for participant in participants
        for repo_name, repo_cfg in repo_configs.items()
    ]
    # Don't spin up more workers than there is work.
    max_workers = min(max_workers, len(pool_tasks))

    reporter = GenerationReporter()
    pool_handles = {
        (p.cid, repo_name): reporter.register_pool(
            p.nick, repo_name, args.target_per_repo, harness=p.harness_id
        )
        for p, repo_name, _repo_cfg in pool_tasks
    }

    def _run_pool(task) -> tuple[str, str, str, object, int, int]:
        """Worker: build per-pool sandbox/agent objects and generate one pool.

        Returns (cid, harness_id, repo_name, stats, in_tokens, out_tokens).
        Per-pool objects are constructed inside the worker so nothing mutable is
        shared across threads; the ChallengeStore and WorkspaceManager are the
        only shared objects (the store guards its index with a lock; the
        workspace manager creates uuid-unique dirs)."""
        participant, repo_name, repo_cfg = task
        handle = pool_handles[(participant.cid, repo_name)]
        model_cfg = participant.model_config  # selection already applied
        wrapper = get_harness(participant.harness_id, model_cfg)
        red_agent = RedAgent(
            agent_wrapper=wrapper,
            workspace_manager=workspace_manager,
            prompt_dir=Path(args.prompt_dir),
            feature_wall_seconds=arena_cfg.agent_timeouts.red_feature_seconds,
            bug_wall_seconds=arena_cfg.agent_timeouts.red_bug_seconds,
            feature_steps=arena_cfg.agent_steps.red_feature_steps,
            bug_steps=arena_cfg.agent_steps.red_bug_steps,
            min_diff_lines=arena_cfg.red_gates.min_diff_lines,
            min_test_assertions=arena_cfg.red_gates.min_test_assertions,
            min_test_functions=arena_cfg.red_gates.min_test_functions,
        )
        executor = DockerExecutor(
            docker_image=repo_cfg.docker_image,
            timeout_s=arena_cfg.sandbox.timeout_seconds,
            memory_mb=arena_cfg.sandbox.memory_mb,
        )
        test_runner = TestRunner(executor=executor, repo_config=repo_cfg)
        validator = RedGateValidator(
            test_runner=test_runner,
            config=arena_cfg,
            agent_wrapper=wrapper,
            workspace_manager=workspace_manager,
            prompt_dir=Path(args.prompt_dir),
            self_review_wall_seconds=arena_cfg.agent_timeouts.red_self_review_seconds,
            self_review_steps=arena_cfg.agent_steps.red_self_review_steps,
        )
        generator = ChallengeGenerator(
            store=store,
            red_gate_validator=validator,
            workspace_manager=workspace_manager,
            config=arena_cfg,
        )
        stats = generator.generate_pool(
            red_agent=red_agent,
            repo_config=repo_cfg,
            target_count=args.target_per_repo,
            red_model_id=model_cfg.model_id,
            pool_reporter=handle,
        )
        pool_in = pool_out = 0
        for rec in store.query(
            red_model_id=model_cfg.model_id, repo_name=repo_name,
            red_harness_id=participant.harness_id,
            red_reasoning_effort=participant.reasoning_effort,
            red_provider=participant.provider,
        ):
            ch = rec.challenge
            for traj in (ch.agent_trajectory, ch.feature_trajectory, ch.bug_trajectory):
                if traj is None:
                    continue
                pool_in += int(traj.total_input_tokens or 0)
                pool_out += int(traj.total_output_tokens or 0)
        return participant.cid, participant.harness_id, repo_name, stats, pool_in, pool_out

    results: list[tuple] = []
    print(
        f"[gen] launching {len(pool_tasks)} (participant × repo) pool(s) "
        f"with max_workers={max_workers}:"
    )
    for p in participants:
        print(f"    · {p.label}")
    with reporter_live(reporter):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_pool, t): t for t in pool_tasks}
            try:
                for fut in as_completed(futures):
                    participant, repo_name, _repo_cfg = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:  # noqa: BLE001
                        handle = pool_handles[(participant.cid, repo_name)]
                        handle.finish_pool(
                            _Status.FAIL,
                            # Bounded: a raw provider-429 payload would wrap
                            # across the live tree for pages.
                            detail=f"error: {type(e).__name__}: {str(e)[:80]}",
                        )
                        results.append(
                            (participant.cid, participant.harness_id, repo_name, None, 0, 0)
                        )
            except KeyboardInterrupt:
                print("\n[gen] interrupted — cancelling pending pools...", file=sys.stderr)
                for fut in futures:
                    fut.cancel()
                raise

    # ── Per-pool + grand-total summary (after the live view stops) ──
    grand_in = grand_out = 0
    grand_cost = 0.0
    for cid, harness_id, repo_name, stats, pool_in, pool_out in results:
        if stats is None:
            print(f"[gen] {display_composite_id(cid)} × {repo_name}: FAILED (see above)")
            continue
        grand_in += pool_in
        grand_out += pool_out
        grand_cost += float(stats.total_generation_cost_usd or 0.0)
        print(
            f"[gen] {display_composite_id(cid)} × {repo_name}: "
            f"successful={stats.total_challenges} "
            f"failed={stats.failed_challenges} "
            f"cost=${stats.total_generation_cost_usd:.4f} "
            f"tokens={pool_in:,} in / {pool_out:,} out"
        )

    print(
        f"[gen] GRAND TOTAL: cost=${grand_cost:.4f} "
        f"tokens={grand_in:,} in / {grand_out:,} out"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
