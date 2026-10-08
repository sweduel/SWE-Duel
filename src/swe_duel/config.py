"""Configuration loaders for SWE-Duel."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


# ── Pydantic config models ──────────────────────────────


class ScoringConfig(BaseModel):
    lambda_feature: float = 0.5
    lambda_bugfix: float = 0.5


class RedGatesConfig(BaseModel):
    min_lines_added: int = 5
    min_lines_removed: int = 1
    min_hunks: int = 1
    min_assertions_feature: int = 2
    min_assertions_bug: int = 1
    min_diff_lines: int = 10
    min_test_assertions: int = 2
    min_test_functions: int = 1
    enable_diff_gate: bool = True
    enable_lint_gate: bool = True
    enable_self_review_gate: bool = True


class SandboxConfig(BaseModel):
    timeout_seconds: int = 120
    memory_mb: int = 4096


class AgentTimeoutsConfig(BaseModel):
    red_feature_seconds: float = 600.0
    red_bug_seconds: float = 480.0
    red_self_review_seconds: float = 600.0
    blue_seconds: float = 600.0


class AgentStepsConfig(BaseModel):
    """Per-phase step budgets (max agent loop iterations) for each agent role."""

    red_feature_steps: int = 50
    red_bug_steps: int = 50
    red_self_review_steps: int = 50
    blue_steps: int = 50

    @field_validator(
        "red_feature_steps", "red_bug_steps", "red_self_review_steps", "blue_steps"
    )
    @classmethod
    def _validate_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"agent step budget must be positive, got: {v}")
        return v


class AgentModelConfig(BaseModel):
    """Arena-wide LLM request settings applied to every participant model.

    These are uniform across the field on purpose (a shared per-response
    budget keeps harness-ablation / identity comparisons honest), so they live
    in ``arena.yaml`` rather than per-model in ``models.yaml``.
    """

    max_tokens: int = 32768
    """Per-response completion cap (mini-swe ``max_tokens`` / OpenHands
    ``max_output_tokens``). Reasoning tokens count against it: at the deepest
    selectable reasoning effort a truncated response dies BEFORE emitting its
    tool call (``finish_reason="length"``, ``tool_calls=null``), which
    mini-swe-agent reports as ``FormatError("No tool calls found")`` — three
    consecutive kill the run (``RepeatedFormatError``) long before the step
    limit. The cap must cover the deepest selectable reasoning effort."""

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"agent_model.max_tokens must be positive, got: {v}")
        return v


class ChallengeBankConfig(BaseModel):
    target_challenges_per_model_repo: int = 5
    max_generation_attempts: int = 3
    # Number of (model, repo) generation pools to run concurrently. Targets
    # within a single pool stay sequential. CLI --max-workers overrides this.
    max_workers: int = 1

    @field_validator("max_workers")
    @classmethod
    def _validate_max_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_workers must be >= 1, got: {v}")
        return v


class MatchConfig(BaseModel):
    # A "player" is a (model, harness, reasoning_effort, provider) competitor.
    # This is the number of challenges each player must defend against per repo
    # per match (and, symmetrically, the number of its own challenges the
    # opponent defends).
    turns_per_player: int = 3

    # How many defense sub-turns (one Blue agent + its containerized test runs)
    # to run concurrently within a tournament round. Each round flattens every
    # cache-miss defense across all of its matches into a single pool and
    # dispatches that pool through a ThreadPoolExecutor of this width. Defenses
    # are fully containerized and independent (unique workspaces + container
    # names), so they parallelize cleanly. CLI --match-workers overrides this.
    max_workers: int = 5

    @field_validator("max_workers")
    @classmethod
    def _validate_max_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"match.max_workers must be >= 1, got: {v}")
        return v

    @property
    def turns_per_agent(self) -> int:
        """Back-compat alias for the pre-rename field name."""
        return self.turns_per_player


class TournamentConfig(BaseModel):
    schedule: str = "round_robin"
    repos_per_match: int = 1


class RatingConfig(BaseModel):
    elo_k: float = 32
    initial_elo: float = 1500
    trueskill_initial_mu: float = 25.0
    trueskill_initial_sigma: float = 8.333


class PathsConfig(BaseModel):
    """Filesystem locations for run artifacts.

    ``output_dir`` is where the challenge bank, defenses, logs, workspaces,
    matches, and tournament state are written. Resolution order (see
    ``swe_duel.cli._common.resolve_output_dir``): explicit ``--output-dir`` CLI
    flag, then the ``SWE_DUEL_OUTPUT_DIR`` environment variable, then this value,
    then ``./data`` relative to the working directory.
    """

    output_dir: str = "data"

    @field_validator("output_dir")
    @classmethod
    def _validate_output_dir(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("paths.output_dir must be a non-empty path")
        return v


class ArenaConfig(BaseModel):
    scoring: ScoringConfig = ScoringConfig()
    red_gates: RedGatesConfig = RedGatesConfig()
    sandbox: SandboxConfig = SandboxConfig()
    agent_timeouts: AgentTimeoutsConfig = AgentTimeoutsConfig()
    agent_steps: AgentStepsConfig = AgentStepsConfig()
    agent_model: AgentModelConfig = AgentModelConfig()
    challenge_bank: ChallengeBankConfig = ChallengeBankConfig()
    match: MatchConfig = MatchConfig()
    tournament: TournamentConfig = TournamentConfig()
    rating: RatingConfig = RatingConfig()
    paths: PathsConfig = PathsConfig()


class ModelConfig(BaseModel):
    model_id: str
    temperature: float = 0.2
    # Per-response completion cap. The authoritative, arena-wide value lives in
    # `arena.yaml` (`agent_model.max_tokens`) and is stamped onto every entry
    # by `load_models_config` — models.yaml no longer carries it. This default
    # only mirrors that setting for directly-constructed configs (tests).
    max_tokens: int = 32768
    # Menu of selectable reasoning-effort levels for this model (OpenRouter
    # `reasoning.effort` strings, e.g. "high" / "low" / "minimal"). Curated per
    # model in config/models.yaml because every model accepts a different set.
    reasoning_efforts: list[str] = []
    # Menu of selectable OpenRouter provider slugs for this model (e.g.
    # "cloudflare", "fireworks"). Curated per model in config/models.yaml.
    providers: list[str] = []
    # Client-side request-rate throttle (requests per minute) per OpenRouter
    # provider slug, enforced by the harness layer before every LLM call (see
    # src/swe_duel/agents/harness/rate_limit.py). The special key "default" applies
    # to auto-route requests and to pinned slugs without their own entry.
    # Limits are per (model, provider) because upstream pools differ; the
    # limiter is process-wide, so parallel workers share (divide) one budget.
    # Probe grounded values with scripts/probe_provider_rate_limits.py.
    provider_rate_limits: dict[str, int] = {}
    # Selected values for ONE participant: a competitor's identity is the
    # 4-tuple (model_id, harness_id, reasoning_effort, provider), so selection
    # lives on the per-participant copy of this config (see `with_selection`).
    # Empty string = model default effort / OpenRouter auto provider routing.
    reasoning_effort: str = ""
    provider: str = ""

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, v: str) -> str:
        if "/" not in v or v.startswith("/") or v.endswith("/"):
            raise ValueError(
                f"model_id must be an OpenRouter-style 'provider/model' identifier, got: {v!r}"
            )
        return v

    @field_validator("reasoning_effort", "provider")
    @classmethod
    def _validate_identity_segment(cls, v: str) -> str:
        # These become `#`-separated segments of the composite competitor id,
        # so a literal `#` would corrupt identity parsing. Effort strings are
        # lowercase OpenRouter tokens; provider slugs are lowercase OpenRouter
        # slugs — whitespace would never appear in a valid one.
        if "#" in v or any(ch.isspace() for ch in v):
            raise ValueError(
                f"reasoning_effort/provider must be a non-empty slug without "
                f"'#' or whitespace, got: {v!r}"
            )
        return v

    @field_validator("reasoning_efforts", "providers")
    @classmethod
    def _validate_menu(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item or "#" in item or any(
                ch.isspace() for ch in item
            ):
                raise ValueError(
                    f"reasoning_efforts/providers entries must be non-empty "
                    f"slugs without '#' or whitespace, got: {item!r}"
                )
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate entries in reasoning_efforts/providers: {v!r}")
        return v

    @field_validator("provider_rate_limits")
    @classmethod
    def _validate_rate_limits(cls, v: dict[str, int]) -> dict[str, int]:
        for slug, rpm in v.items():
            if not isinstance(slug, str) or not slug or "#" in slug or any(
                ch.isspace() for ch in slug
            ):
                raise ValueError(
                    f"provider_rate_limits keys must be non-empty provider slugs "
                    f"without '#' or whitespace, got: {slug!r}"
                )
            if not isinstance(rpm, int) or isinstance(rpm, bool) or rpm <= 0:
                raise ValueError(
                    f"provider_rate_limits values must be positive integers "
                    f"(requests per minute), got: {slug!r}: {rpm!r}"
                )
        return v

    def rate_limit_rpm(self, provider: str | None = None) -> int | None:
        """Requests/minute cap for this model when routed via ``provider``.

        Resolution order: the pinned provider's own entry, then the "default"
        entry, then no limit. ``provider`` empty/unpinned consults "default"
        only (auto-route spreads across every serving provider).
        """
        limits = self.provider_rate_limits or {}
        slug = (provider if provider is not None else self.provider) or ""
        if slug and slug in limits:
            return limits[slug]
        return limits.get("default") or None

    @property
    def openrouter_model_id(self) -> str:
        """Prepends 'openrouter/' so litellm routes through OpenRouter."""
        if self.model_id.startswith("openrouter/"):
            return self.model_id
        return f"openrouter/{self.model_id}"

    def with_selection(
        self, reasoning_effort: str = "", provider: str = ""
    ) -> "ModelConfig":
        """Copy of this config bound to one participant's effort/provider.

        The returned copy is what gets handed to a harness: `reasoning_effort`
        / `provider` are part of the competitor identity and flow into the
        OpenRouter request body (reasoning effort + provider routing). Empty
        strings keep the model-default / auto-routed behaviour of the base
        config (identity-compatible with records written before selection
        existed).
        """
        return self.model_copy(
            update={
                "reasoning_effort": reasoning_effort or "",
                "provider": provider or "",
            }
        )


class RepoConfig(BaseModel):
    name: str
    url: str
    commit: str
    language: str = "python"
    test_command: str
    docker_image: str

    # Agents run inside `docker_image` (the same image the gates validate in),
    # with the host workspace bind-mounted at the container's /workspace. These
    # two fields tune that container so language toolchains and pre-installed
    # deps are reachable; both default to the values that suit Python/flask so
    # existing repo YAMLs need no changes.

    # Workspace subpaths whose IMAGE-baked contents must show through the bind
    # mount via an anonymous volume (e.g. node deps installed by `npm ci` at
    # build time). Without this the host clone — which has no node_modules —
    # would mask the image's, and the agent would have to reinstall over the
    # network (denied) or leak host binaries into the diff. helmet: ["node_modules"].
    preserve_paths: list[str] = []

    # Whether the agent's command interpreter is a login shell (`bash -lc`). A
    # login shell re-sources /etc/profile and resets PATH, which drops Go's
    # /usr/local/go/bin in the golang image — so jwt sets this False to use
    # `bash -c` and keep the image's PATH intact.
    login_shell: bool = True

    # Extra path patterns to exclude from diff/snapshot computation for this
    # repo, in addition to the global ``_DEFAULT_EXCLUDE``. Patterns match by
    # prefix or ``"/" + pattern`` substring (see ``_is_excluded``). Use this
    # for repo-specific build artifacts that an in-tree build may generate in
    # the workspace (e.g. libexpat's autotools outputs: ``autom4te.cache/``,
    # ``.libs/``, ``Makefile.in`` under ``expat/``).
    exclude_paths: list[str] = []


# ── Loaders ─────────────────────────────────────────────


def load_arena_config(config_dir: Path) -> ArenaConfig:
    """Parse config/arena.yaml into ArenaConfig."""
    path = config_dir / "arena.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ArenaConfig(**data)


def load_models_config(
    config_dir: Path, *, max_tokens: int | None = None
) -> dict[str, ModelConfig]:
    """Parse config/models.yaml, return dict keyed by model nickname.

    ``max_tokens`` is the arena-wide per-response completion cap
    (``arena.yaml: agent_model.max_tokens``); when given it is stamped onto
    every entry — the per-model knob was removed from models.yaml so all
    participants share one setting — and overrides any stale per-model value.
    Callers that load models for actual agent runs (``scripts/_common.py``)
    must pass it; omitting it keeps whatever the file/default carries (used
    by tests that pin their own caps).
    """
    path = config_dir / "models.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    if max_tokens is not None:
        for cfg in data["models"].values():
            cfg["max_tokens"] = max_tokens
    return {nick: ModelConfig(**cfg) for nick, cfg in data["models"].items()}


def load_repo_config(config_path: Path) -> RepoConfig:
    """Parse a single repo YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return RepoConfig(**data["repo"])


def load_all_repo_configs(config_dir: Path) -> dict[str, RepoConfig]:
    """Load all YAML files in config/repos/."""
    repos_dir = config_dir / "repos"
    configs: dict[str, RepoConfig] = {}
    for path in sorted(repos_dir.glob("*.yaml")):
        rc = load_repo_config(path)
        configs[rc.name] = rc
    return configs
