from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════
# Competitor identity = (model_id, harness_id, reasoning_effort, provider)
# ═══════════════════════════════════════════════════════
#
# A competitor in the arena is a (model, agent-harness, reasoning-effort,
# OpenRouter-provider) tuple — e.g. the same model run under mini-swe-agent
# vs. OpenHands, or at reasoning effort "high" vs. "low", or routed via
# "cloudflare" vs. "fireworks", is a distinct entrant each time. The composite
# id string is used as the dict key in the Swiss pairing and rating layers and
# in pool/defense identity.
#
# Format: ``"<model_id>#<harness_id>"`` when both effort and provider are
# empty (byte-identical to the legacy 2-part format, so every persisted
# pool/defense/match/state record written before this dimension existed keeps
# parsing unchanged), and ``"<model_id>#<harness_id>#<effort>#<provider>"``
# as soon as either extension is non-empty. Model ids contain `/`, harness
# ids only `[a-z-]`, and effort/provider slugs `[a-z0-9-]`, so `#` is an
# unambiguous separator; empty segments are preserved positionally.

COMPOSITE_ID_SEP = "#"


def composite_id(
    model_id: str,
    harness_id: str,
    reasoning_effort: str = "",
    provider: str = "",
) -> str:
    """Build the competitor identity string for a participant 4-tuple.

    ``reasoning_effort`` / ``provider`` empty (the model-default selection)
    collapse to the legacy 2-part ``"<model_id>#<harness_id>"`` form.
    """
    if not reasoning_effort and not provider:
        return f"{model_id}{COMPOSITE_ID_SEP}{harness_id}"
    return (
        f"{model_id}{COMPOSITE_ID_SEP}{harness_id}"
        f"{COMPOSITE_ID_SEP}{reasoning_effort}{COMPOSITE_ID_SEP}{provider}"
    )


def split_composite_id(cid: str) -> tuple[str, str, str, str]:
    """Inverse of :func:`composite_id`.

    Returns ``(model_id, harness_id, reasoning_effort, provider)``. Tolerates
    a bare model id (no harness suffix; both extensions empty) and the legacy
    2-part form written before effort/provider selection existed — both yield
    empty strings for the missing segments.
    """
    parts = cid.split(COMPOSITE_ID_SEP)
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], COMPOSITE_ID_SEP.join(parts[3:])
    if len(parts) == 3:
        # Defensive: a malformed 3-part id still parses (empty provider).
        return parts[0], parts[1], parts[2], ""
    if len(parts) == 2:
        return parts[0], parts[1], "", ""
    return cid, "", "", ""


def display_composite_id(cid: str) -> str:
    """Human-friendly form of a competitor identity.

    ``model_id [harness_id]`` for legacy/default selections, extended with
    ``effort=<e>`` / ``provider=<p>`` segments when set.
    """
    model_id, harness_id, effort, provider = split_composite_id(cid)
    out = f"{model_id} [{harness_id}]" if harness_id else model_id
    if effort:
        out += f" effort={effort}"
    if provider:
        out += f" provider={provider}"
    return out


# ═══════════════════════════════════════════════════════
# Phase 1: Enums
# ═══════════════════════════════════════════════════════

# Bug labels are NOT a fixed taxonomy: `bug_type` is a short free-text
# description (about six or seven words) the Red agent writes in its own
# words, e.g. "modulo returns wrong remainder for negative dividends".

class MatchOutcome(str, Enum):
    MODEL_A_WINS = "model_a_wins"
    MODEL_B_WINS = "model_b_wins"
    DRAW = "draw"

class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


# ═══════════════════════════════════════════════════════
# Phase 2: Sandbox execution
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecutionResult:
    """Raw output from running a command in Docker."""
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int

@dataclass(frozen=True)
class TestExecutionResult:
    """Parsed pytest output."""
    passed: bool
    total: int
    passed_count: int
    failed_count: int
    error_count: int
    failure_messages: list[str]
    stdout: str
    stderr: str
    duration_ms: int
    command: str = ""          # the shell command that produced this result


# ═══════════════════════════════════════════════════════
# Phase 3: Workspace and diffs
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class DiffResult:
    success: bool
    patched: str              # patched file content (empty on failure)
    error: str

@dataclass(frozen=True)
class DiffStats:
    lines_added: int
    lines_removed: int
    hunks: int
    files_changed: int

@dataclass
class Workspace:
    """A temporary copy of a repo for an agent session."""
    workspace_id: str         # UUID
    repo_name: str
    path: Path                # path to this workspace's root
    reference_path: Path      # path to the clean original copy (for diff computation)
    # Repo-specific path patterns to exclude from diff/snapshot (merged with
    # the global _DEFAULT_EXCLUDE). Populated from RepoConfig.exclude_paths.
    excludes: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# Phase 4: Agent trajectory
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentTrajectory:
    """Full log of a mini-swe-agent session."""
    steps: list[dict]         # [{"thought": ..., "action": ..., "observation": ...}, ...]
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    model_id: str
    duration_seconds: float
    exit_status: str = ""        # e.g. "Submitted", "LimitsExceeded", "WallClockTimeout"


# ═══════════════════════════════════════════════════════
# Phase 5: Red challenge (feature only, bug fields optional)
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class RedChallenge:
    """
    Complete Red challenge output, extracted from the agent's workspace.
    In Phase 5, bug_* fields are None. Phase 6 populates them.
    """
    # Exploration
    target_files: list[str]               # which file(s) Red chose to modify
    exploration_summary: str              # Red's reasoning about module selection

    # Feature
    feature_spec: str                     # natural language description
    feature_rationale: str                # why this feature belongs here
    pr_diff: str                          # unified diff: original → Red's modified workspace
    modified_file_contents: dict[str, str]  # {relative_path: full content after modification}
    original_file_contents: dict[str, str]  # {relative_path: full content before modification}
    feature_test_code: str                # pytest file testing the new feature

    # Bug (populated in Phase 6; None during Phase 5 feature-only tests)
    # Free-text label of the embedded bug (about 6-7 words, agent's own words).
    bug_type: str | None
    bug_description: str | None           # HIDDEN from Blue
    bug_location: str | None              # HIDDEN from Blue
    bug_test_code: str | None             # HIDDEN from Blue — pytest that FAILS when bug present

    # Agent session metadata (combined across phases for backwards compatibility)
    agent_trajectory: AgentTrajectory

    # Two-phase Red generation artefacts (new)
    feature_only_file_contents: dict[str, str] = field(default_factory=dict)
    feature_trajectory: AgentTrajectory | None = None
    bug_trajectory: AgentTrajectory | None = None

    # Pytest nodeids that failed on the pristine repo before Red made any
    # changes. Recorded by Red in Phase 1 and excluded from validation gates so
    # pre-existing bugs aren't blamed on the generated feature/bug.
    pre_existing_failures: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
# Phase 7: Validation
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class GateResult:
    gate_name: str
    status: GateStatus
    message: str
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    command: str = ""          # the shell command the gate executed (for HTML/debug)

@dataclass(frozen=True)
class RedSelfReview:
    """Result of the Red model emulating Blue to self-review AND fix its own PR.

    Used by the self-review validation gate to confirm the embedded bug is
    genuinely solvable. The Red model is given the PR diff + feature spec +
    feature tests ONLY (no bug_description, bug_location, or bug_test_code) — the
    same view the real Blue agent gets — and asked to FIX any defect it finds
    while retaining the feature. The gate then runs the hidden bug tests against
    the reviewer's modified code: `detected=True` iff all bug tests pass on that
    fixed code (the bug was found and removed). If the bug is NOT fixed, the
    challenge is rejected — this prevents Red from cheating with no-op PRs that
    contain no real defect (which would unfairly score a same-skill Blue as a
    loser) or with bugs a same-skill reviewer cannot realistically fix.
    """
    detected: bool                            # True iff all bug tests pass on the reviewer's fixed code
    findings: list[ReviewFinding]
    fix_explanation: str
    agent_trajectory: AgentTrajectory
    detection_reason: str = ""                # human-readable explanation of the bug-test verdict


@dataclass(frozen=True)
class RedValidationResult:
    passed: bool
    gate_results: list[GateResult]
    attempt_number: int
    self_review: RedSelfReview | None = None


# ═══════════════════════════════════════════════════════
# Phase 8: Challenge Bank
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChallengeRecord:
    """A validated Red challenge persisted in the Challenge Bank."""
    challenge_id: str                     # UUID, primary key
    red_model_id: str
    repo_name: str
    repo_commit_sha: str                  # for staleness detection
    target_files: list[str]
    challenge: RedChallenge
    validation: RedValidationResult
    generated_at: datetime
    generation_cost_usd: float
    generation_retries: int
    # Agent harness that produced this challenge. Part of the competitor
    # identity (model_id, harness_id, reasoning_effort, provider). Defaults to
    # mini-swe-agent for records written before harness selection existed.
    red_harness_id: str = "mini-swe-agent"
    # 1-based generation slot for this (competitor, repo): one slot = one
    # generation effort (up to max_generation_attempts attempts ending in one
    # success or a fully-failed chain). Earlier single-effort runs are slot 1.
    slot: int = 1
    # Selected reasoning effort / OpenRouter provider of the Red competitor.
    # Part of the pool identity; empty = model default / OpenRouter auto-route
    # (also the value for every record written before this dimension existed).
    red_reasoning_effort: str = ""
    red_provider: str = ""

@dataclass(frozen=True)
class FailedChallengeRecord:
    """A persisted record of a Red generation attempt that did NOT pass all gates.

    Stored alongside successful `ChallengeRecord`s in the Challenge Bank so the
    user can inspect every attempt (statistics + agent reasoning for Phase 1,
    Phase 2, and the self-review gate). Tournament/match logic ignores these.

    `kind` indicates the failure mode:
      - "validation"     : agent finished but one or more gates failed
      - "feature-gate"   : Phase A produced a feature that failed feature gates;
                           Phase B skipped to save tokens
      - "timeout-feature": Phase A exceeded wall-clock budget
      - "timeout-bug"    : Phase B exceeded wall-clock budget
      - "incomplete-feature" / "incomplete-bug": agent exited without writing
                           required `_swe-duel/` artefacts
    """
    challenge_id: str                         # UUID, primary key
    red_model_id: str
    repo_name: str
    repo_commit_sha: str
    kind: str
    error_message: str
    attempt_number: int
    target_files: list[str]                   # may be empty if Phase A never produced metadata
    challenge: RedChallenge | None            # partial challenge if extractable
    validation: RedValidationResult | None    # None when agent never produced a valid output
    feature_trajectory: AgentTrajectory | None
    bug_trajectory: AgentTrajectory | None
    self_review_trajectory: AgentTrajectory | None
    generated_at: datetime
    generation_cost_usd: float
    elapsed_seconds: float
    # Agent harness that produced this (failed) attempt. See ChallengeRecord.
    red_harness_id: str = "mini-swe-agent"
    # 1-based generation slot for this (competitor, repo). See ChallengeRecord.
    slot: int = 1
    # Selected reasoning effort / OpenRouter provider. See ChallengeRecord.
    red_reasoning_effort: str = ""
    red_provider: str = ""


@dataclass(frozen=True)
class ChallengePoolStats:
    red_model_id: str
    repo_name: str
    total_challenges: int                     # successful only (back-compat)
    target_file_distribution: dict[str, int]  # target_file → count (successful)
    bug_type_distribution: dict[str, int]
    total_generation_cost_usd: float
    avg_retries: float
    failed_challenges: int = 0                # count of persisted FailedChallengeRecords
    red_harness_id: str = "mini-swe-agent"    # harness this pool was generated under
    red_reasoning_effort: str = ""            # effort this pool was generated under
    red_provider: str = ""                     # OpenRouter provider this pool was generated under


# ═══════════════════════════════════════════════════════
# Phase 9: Blue fix
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReviewFinding:
    location: str
    severity: str
    description: str

@dataclass(frozen=True)
class BlueFix:
    """Blue agent's output, extracted from the agent's workspace."""
    review_findings: list[ReviewFinding]
    fix_explanation: str
    fix_diff: str                         # unified diff: original → Blue's workspace
    modified_file_contents: dict[str, str]
    agent_trajectory: AgentTrajectory


# ═══════════════════════════════════════════════════════
# Phase 10: Scoring
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class TurnScore:
    s_regression: float                   # 1.0 if all existing tests pass, else 0.0
    s_feature: float                      # 1.0 if feature tests pass, else 0.0
    s_bugfix: float                       # 1.0 if ALL bug tests pass (bug removed), else 0.0
    blue_composite: float                 # s_regression × s_feature × s_bugfix (all hard gates)
    red_composite: float                  # 1.0 - blue_composite
    test_details: dict                    # {suite_name: TestExecutionResult serialised}


# ═══════════════════════════════════════════════════════
# Phase 11: Defense, Round, Match
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class DefenseResult:
    """One Blue agent attempting one cached challenge."""
    defense_id: str
    challenge_id: str                     # FK → ChallengeRecord
    blue_model_id: str
    blue_fix: BlueFix
    score: TurnScore
    duration_seconds: float
    cost_usd: float
    timestamp: datetime
    # Harness that ran this Blue defense. Part of Blue's competitor identity and
    # of the defense cache key (challenge_id, blue_model_id, blue_harness_id,
    # blue_reasoning_effort, blue_provider).
    blue_harness_id: str = "mini-swe-agent"
    # Selected reasoning effort / OpenRouter provider of the Blue competitor
    # (empty = model default / OpenRouter auto-route, and the value implicit in
    # every defense recorded before this dimension existed).
    blue_reasoning_effort: str = ""
    blue_provider: str = ""

@dataclass
class TurnResult:
    """Joins a ChallengeRecord with a DefenseResult within a match."""
    turn_id: str
    turn_index: int
    challenge_record: ChallengeRecord
    defense_result: DefenseResult
    red_model_id: str
    blue_model_id: str
    red_harness_id: str = "mini-swe-agent"
    blue_harness_id: str = "mini-swe-agent"
    red_reasoning_effort: str = ""
    red_provider: str = ""
    blue_reasoning_effort: str = ""
    blue_provider: str = ""

@dataclass
class MatchResult:
    match_id: str
    model_a_id: str
    model_b_id: str
    repo_name: str
    turns: list[TurnResult]
    model_a_total: float
    model_b_total: float
    outcome: MatchOutcome
    duration_seconds: float
    total_cost_usd: float
    timestamp: datetime


# ═══════════════════════════════════════════════════════
# Phase 12: Rating, Tournament
# ═══════════════════════════════════════════════════════

@dataclass
class RatingSnapshot:
    model_id: str
    elo: float
    trueskill_mu: float
    trueskill_sigma: float
    red_elo: float
    blue_elo: float
    matches_played: int
    timestamp: datetime
    bradley_terry: float = 0.0   # log-strength from BradleyTerryRating.fit

@dataclass
class TournamentResult:
    tournament_id: str
    generation_stats: dict[str, ChallengePoolStats]
    matches: list[MatchResult]
    final_ratings: list[RatingSnapshot]
    head_to_head: dict[str, dict[str, dict]]
    config_snapshot: dict
    total_generation_cost_usd: float
    total_evaluation_cost_usd: float
    timestamp: datetime