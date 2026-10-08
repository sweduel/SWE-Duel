"""Shared setup for CLI scripts."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

import swe_duel
import swe_duel.agents
from swe_duel.challenge_bank.store import ChallengeStore
from swe_duel.cli.preflight import PreflightError, run_preflight
from swe_duel.config import (
    ArenaConfig,
    ModelConfig,
    RepoConfig,
    load_arena_config,
    load_all_repo_configs,
    load_models_config,
)
from swe_duel.logging.artifacts import ArtifactLogger
from swe_duel.sandbox.docker_executor import kill_all_swe_duel_containers
from swe_duel.sandbox.workspace import WorkspaceManager


def default_prompt_dir() -> Path:
    """Directory of the bundled agent prompt templates (shipped as package data).

    Resolves inside the installed ``swe-duel`` package so it works for editable
    checkouts and pip-installed wheels alike; the CLIs expose it as the
    ``--prompt-dir`` default (overridable with a custom directory).
    """
    return Path(swe_duel.agents.__file__).resolve().parent / "prompts"


def resolve_config_dir(args: Any) -> Path:
    """Locate the user's arena configuration directory.

    Resolution order:

    1. an explicit ``--config-dir`` CLI flag,
    2. the ``SWE_DUEL_CONFIG_DIR`` environment variable,
    3. ``./config`` in the working directory (the layout ``swe-duel init``
       scaffolds, and the one a source checkout uses),
    4. otherwise: fail fast with setup instructions.
    """
    explicit = getattr(args, "config_dir", None)
    if explicit:
        return Path(explicit)
    env = os.environ.get("SWE_DUEL_CONFIG_DIR")
    if env:
        return Path(env)
    cwd_config = Path("config")
    if (cwd_config / "arena.yaml").is_file():
        return cwd_config
    print(
        "error: no arena configuration found.\n"
        f"  looked for: {cwd_config.resolve() / 'arena.yaml'}\n"
        " remedies (pick one):\n"
        "   - run `swe-duel init` here to scaffold ./config + ./data\n"
        "   - pass --config-dir <path>\n"
        "   - set SWE_DUEL_CONFIG_DIR=<path>",
        file=sys.stderr,
    )
    raise SystemExit(2)


def resolve_output_dir(args: Any, arena_config: ArenaConfig) -> Path:
    """Locate the output/artifact directory.

    Resolution order:

    1. an explicit ``--output-dir`` CLI flag,
    2. the ``SWE_DUEL_OUTPUT_DIR`` environment variable,
    3. ``paths.output_dir`` in arena.yaml,
    4. fallback default ``./data`` (relative to the working directory).
    """
    explicit = getattr(args, "output_dir", None)
    if explicit:
        return Path(explicit)
    env = os.environ.get("SWE_DUEL_OUTPUT_DIR")
    if env:
        return Path(env)
    return Path(arena_config.paths.output_dir)


def setup(args: Any, *, preflight: bool = True) -> dict[str, Any]:
    config_dir = resolve_config_dir(args)
    arena_config: ArenaConfig = load_arena_config(config_dir)
    output_dir = resolve_output_dir(args, arena_config)
    bank_dir = Path(
        getattr(args, "bank_dir", None) or str(output_dir / "challenge_bank")
    )

    model_configs: dict[str, ModelConfig] = load_models_config(
        config_dir, max_tokens=arena_config.agent_model.max_tokens
    )
    repo_configs: dict[str, RepoConfig] = load_all_repo_configs(config_dir)

    # Filter by CLI selections
    wanted_models = getattr(args, "models", None)
    if wanted_models:
        model_configs = {
            nick: cfg
            for nick, cfg in model_configs.items()
            if nick in wanted_models or cfg.model_id in wanted_models
        }

    wanted_repos = getattr(args, "repos", None)
    if wanted_repos:
        repo_configs = {
            name: cfg
            for name, cfg in repo_configs.items()
            if name in wanted_repos
        }

    repos_dir = Path(getattr(args, "repos_dir", "repos"))

    if preflight:
        skip = bool(getattr(args, "skip_preflight", False)) or (
            os.environ.get("SWE_DUEL_SKIP_PREFLIGHT", "") == "1"
        )
        if not skip:
            try:
                run_preflight(
                    repo_configs=repo_configs, repos_dir=repos_dir, config_dir=config_dir
                )
            except PreflightError:
                # run_preflight already printed the formatted problem report.
                raise SystemExit(2) from None

    store = ChallengeStore(bank_dir=bank_dir)
    logger = ArtifactLogger(data_dir=output_dir, defenses_dir=output_dir / "defenses")
    workspace_manager = WorkspaceManager(
        repos_dir=repos_dir,
        tmp_root=output_dir / "workspaces",
    )

    # Default the model-retry log into the resolved output dir unless the
    # user pinned a location themselves (rate_limit.py reads this env var).
    if not os.environ.get("SWE_DUEL_MODEL_RETRY_LOG"):
        os.environ["SWE_DUEL_MODEL_RETRY_LOG"] = str(output_dir / "logs" / "model_retries.log")

    return {
        "config_dir": config_dir,
        "output_dir": output_dir,
        "bank_dir": bank_dir,
        "arena_config": arena_config,
        "model_configs": model_configs,
        "repo_configs": repo_configs,
        "challenge_store": store,
        "artifact_logger": logger,
        "workspace_manager": workspace_manager,
    }


def install_container_cleanup_handlers() -> None:
    r"""Register signal handlers that kill SWE-Duel Docker containers on exit.

    Catches SIGINT (Ctrl+C), SIGTERM, SIGHUP (terminal closed), SIGQUIT
    (Ctrl+\\) and SIGTSTP (Ctrl+Z) and kills every sandbox/agent container
    tagged with this process's PID before allowing the default handler to run.
    """

    def _handler(signum: int, frame: Any) -> None:
        print(
            f"\n[signal] Caught signal {signum}; killing SWE-Duel containers for PID {os.getpid()}...",
            file=sys.stderr,
            flush=True,
        )
        try:
            kill_all_swe_duel_containers()
        except Exception as e:
            print(
                f"[signal] Container cleanup raised {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
        if signum == signal.SIGINT:
            # Restore the default handler so the normal KeyboardInterrupt flow
            # (and any existing try/except blocks) takes over.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.default_int_handler(signum, frame)
        else:
            # For SIGTERM/SIGHUP/SIGQUIT/SIGTSTP, exit immediately after cleanup.
            sys.exit(1)

    signals = ["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGTSTP"]
    for name in signals:
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass
