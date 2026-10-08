"""``swe-duel`` — umbrella dispatcher for the per-command CLIs.

The package ships one console script per command (``swe-duel-init``,
``swe-duel-doctor``, …) and this single ``swe-duel`` entry point that forwards
``swe-duel <subcommand> [options]`` to the matching module, so the README's
documented usage works verbatim::

    swe-duel init
    swe-duel setup repos
    swe-duel doctor

Subcommand modules are imported lazily so ``swe-duel init`` does not pay the
import cost of the tournament/report stacks. Dispatch rewrites ``sys.argv`` to
``["swe-duel-<sub>", *rest]`` before calling the module's ``main()`` — every
existing CLI main either accepts ``argv=None`` (argparse then reads
``sys.argv[1:]``) or parses ``sys.argv`` itself, so both signatures work
unchanged.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable, TextIO, cast

Subcommand = tuple[str, str]

SUBCOMMANDS: dict[str, Subcommand] = {
    "init": ("swe_duel.cli.init", "scaffold ./config + ./data for a new arena"),
    "setup": ("swe_duel.cli.setup", "clone target repos / build the Docker images"),
    "doctor": ("swe_duel.cli.doctor", "staged environment validation"),
    "generate": ("swe_duel.cli.generate_challenges", "populate the challenge bank"),
    "match": ("swe_duel.cli.run_match", "run one Red-vs-Blue match"),
    "tournament": ("swe_duel.cli.run_tournament", "interactive Swiss tournament REPL"),
    "tournament-rr": (
        "swe_duel.cli.run_tournament_round_robin",
        "interactive round-robin tournament REPL",
    ),
    "tournament-as": (
        "swe_duel.cli.run_tournament_active_sampling",
        "active-sampled matchups from ./rankings/rankings_<id>.json",
    ),
    "tournament-update": (
        "swe_duel.cli.tournament_update",
        "import a contributed ./data zip: merge matches/defenses/challenges, "
        "update rankings_<id>.json + the index.html leaderboard",
    ),
    "rankings": (
        "swe_duel.cli.export_rankings",
        "export a tournament's rankings to ./rankings/rankings_<id>.json",
    ),
    "ablation": (
        "swe_duel.cli.run_tournament_harness_ablation",
        "round-robin over harnesses with a fixed model set",
    ),
    "evaluate": ("swe_duel.cli.evaluate_blue", "run a Blue evaluation"),
    "report": ("swe_duel.cli.build_report", "Elo/matchup outputs from data/matches"),
    "kill-containers": ("swe_duel.cli.kill_containers", "kill all swe-duel-* containers"),
    "probe-rate-limits": (
        "swe_duel.cli.probe_provider_rate_limits",
        "ground provider rate limits against OpenRouter",
    ),
}


def _usage(out: TextIO) -> None:
    print("usage: swe-duel <subcommand> [options]", file=out)
    print(file=out)
    print("subcommands:", file=out)
    width = max(len(name) for name in SUBCOMMANDS)
    for name, (_module, description) in SUBCOMMANDS.items():
        print(f"  {name:<{width}}  {description}", file=out)
    print(file=out)
    print("Run `swe-duel help <subcommand>` for a subcommand's full options.", file=out)


def _dispatch(name: str, rest: list[str]) -> int:
    module_path, _description = SUBCOMMANDS[name]
    module = import_module(module_path)
    main_fn = cast("Callable[[], int]", getattr(module, "main"))
    saved_argv = sys.argv
    sys.argv = [f"swe-duel-{name}", *rest]
    try:
        return main_fn()
    finally:
        sys.argv = saved_argv


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        _usage(sys.stderr)
        return 2
    name, rest = args[0], args[1:]
    if name in ("help", "--help", "-h"):
        if rest:
            sub = rest[0]
            if sub not in SUBCOMMANDS:
                print(f"swe-duel: unknown subcommand: {sub}", file=sys.stderr)
                return 2
            return _dispatch(sub, ["--help"])
        _usage(sys.stdout)
        return 0
    if name not in SUBCOMMANDS:
        print(f"swe-duel: unknown subcommand: {name}", file=sys.stderr)
        print("Run `swe-duel --help` to list subcommands.", file=sys.stderr)
        return 2
    return _dispatch(name, rest)


if __name__ == "__main__":
    raise SystemExit(main())
