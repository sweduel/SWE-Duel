"""Pytest plugin: auto-parallelize the long validation-gate regression suite.

Registered via the ``pytest11`` entry point so ``pytest_load_initial_conftests``
runs early enough to inject ``-n`` / ``--dist`` before argparse finishes.
``tests/conftest.py`` alone is too late for that particular hook.
"""

from __future__ import annotations

import os

_GATE_REGRESSION = "test_validation_gates_regression"
_DEFAULT_GATE_MAX_WORKERS = 12


def _has_explicit_xdist(args: list[str]) -> bool:
    for a in args:
        if a in ("-n", "--numprocesses", "--num-processes"):
            return True
        if a.startswith("-n") and a != "-n":
            return True
        if a.startswith("--numprocesses=") or a.startswith("--num-processes="):
            return True
    return False


def pytest_load_initial_conftests(early_config: object, parser: object, args: list[str]) -> None:
    """Auto-enable pytest-xdist when targeting the gate regression file.

    Overrides:

      SWE_DUEL_GATE_TEST_WORKERS=8   # explicit worker count
      SWE_DUEL_GATE_TEST_WORKERS=0   # disable auto-parallel
      pytest … -n 4              # manual xdist wins over auto
    """
    del early_config, parser
    if _has_explicit_xdist(args):
        return

    env_workers = os.environ.get("SWE_DUEL_GATE_TEST_WORKERS")
    if env_workers == "0":
        return

    pathish = [a for a in args if not a.startswith("-")]
    targets_gates = any(_GATE_REGRESSION in a for a in pathish)
    only_gates = bool(pathish) and all(_GATE_REGRESSION in a for a in pathish)
    if not targets_gates:
        return
    if not only_gates and env_workers is None:
        # Whole ``tests/`` run — leave scheduling alone.
        return

    if env_workers and env_workers not in ("auto", "0"):
        n = env_workers
    else:
        cpus = os.cpu_count() or 4
        n = str(min(cpus, _DEFAULT_GATE_MAX_WORKERS))
    args[:] = [*args, "-n", n, "--dist=worksteal"]
