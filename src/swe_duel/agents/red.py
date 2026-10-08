"""Red agent: two-phase challenge generation.

Phase A (feature generation): agent adds a correct feature + feature tests.
Phase B (bug embedding): agent edits its feature to embed a subtle bug + bug tests.
Each phase gets its own independent step budget (config agent_steps).
"""

from __future__ import annotations

import fnmatch
import html
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from typing import Callable

from swe_duel.agents.agent_wrapper import CONTAINER_WORKSPACE
from swe_duel.agents.harness.base import AgentHarness
from swe_duel.config import RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    RedChallenge,
    RedValidationResult,
    Workspace,
)
from swe_duel.sandbox import diff_utils
from swe_duel.sandbox.languages import get_adapter
from swe_duel.sandbox.workspace import WorkspaceManager


# Per-language on-disk names for the agent-authored test artifacts inside
# `_swe-duel/`. Go test files MUST end in `_test.go`; Node tests use `.test.ts` so
# `node:test`/`tsx` recognise them.
_TEST_ARTIFACT_NAMES: dict[str, dict[str, str]] = {
    "python": {"feature": "feature_tests.py", "bug": "bug_tests.py"},
    "go": {"feature": "feature_tests_test.go", "bug": "bug_tests_test.go"},
    "node": {"feature": "feature_tests.test.ts", "bug": "bug_tests.test.ts"},
    "c": {"feature": "feature_tests.c", "bug": "bug_tests.c"},
    "java": {"feature": "TestSweDuelFeature.java", "bug": "TestSweDuelBug.java"},
}


def _artifact_names(language: str | None) -> dict[str, str]:
    key = (language or "python").strip().lower()
    if key in ("javascript", "typescript"):
        key = "node"
    return _TEST_ARTIFACT_NAMES.get(key, _TEST_ARTIFACT_NAMES["python"])


# Language-specific guidance injected into the Red/self-review prompts. The key
# fields the templates consume:
#   lang             : canonical language key (python|go|node)
#   test_framework   : human name of the test framework the agent must use
#   feature_test_filename : where the agent writes its feature test
#   test_instructions: a block describing how to write & run tests natively
#
# The per-language guide is a TEMPLATE that may carry the placeholders below;
# ``_prompt_language_context`` fills them from the adapter's ``prompt_hints()``
# so the same language scales across multiple repos (e.g. jwt + chi for Go,
# helmet + expressjs for Node, cJSON + libexpat for C, owasp + java-jwt for
# Java):
#   <<FEATURE_TEST_FILENAME>>  : the on-disk artifact name for this repo
#   <<GO_IMPORT_PATH>>         : the repo's Go module path (go.mod)
#   <<NODE_IMPORT_EXAMPLE>>    : a relative import line for this repo's sources
#   <<C_INCLUDE_DIRECTIVE>>    : the #include line for this repo's main header
#   <<JAVA_IMPORT_ROOT>>       : the repo's top-level Java package prefix
#   <<JAVA_MODULE>>            : the Maven module / Gradle project path
#   <<JAVA_BUILD_TOOL>>        : "Maven" or "Gradle"
#   <<JAVA_RUN_COMMAND>>       : the exact harness command that runs the test
_PROMPT_LANG_GUIDES: dict[str, dict[str, str]] = {
    "python": {
        "lang": "python",
        "test_framework": "pytest",
        "test_instructions": (
            "Write tests as a standard pytest file. Each test is a "
            "`def test_*():` function using `assert`. Run them with:\n"
            "    python -m pytest _swe-duel/<<FEATURE_TEST_FILENAME>> -q\n"
            "The harness runs these with pytest exactly as written."
        ),
    },
    "go": {
        "lang": "go",
        "test_framework": "the Go testing package (`go test`)",
        "test_instructions": (
            "This is a GO project — DO NOT write Python/pytest. Write a real Go "
            "test file in its OWN standalone package so the harness can compile "
            "and run it in isolation:\n"
            "  - File: `_swe-duel/<<FEATURE_TEST_FILENAME>>` (the name MUST end in "
            "`_test.go`).\n"
            "  - Declare `package swedueltest` at the top.\n"
            "  - Import the module under test by its full import path "
            "(`<<GO_IMPORT_PATH>>`, from `go.mod`). Import sub-packages as "
            "needed.\n"
            "  - Each test is `func TestXxx(t *testing.T) { ... }` and fails via "
            "`t.Error`/`t.Errorf`/`t.Fatal`/`t.Fatalf`.\n"
            "  - The harness runs your tests with: "
            "`go test -buildvcs=false -json ./_swe-duel/`.\n"
            "  - Verify locally the same way before submitting.\n"
            "  - Do NOT add a `_test.go` file next to the source for the harness "
            "tests — keep them under `_swe-duel/` only (the source `*_test.go` files "
            "you may still run to check existing tests)."
        ),
    },
    "node_ts": {
        "lang": "node",
        "test_framework": "Node's built-in test runner via tsx (`node:test`)",
        "test_instructions": (
            "This is a NODE/TYPESCRIPT project — DO NOT write Python/pytest. "
            "Write a real TypeScript test using the built-in `node:test` runner:\n"
            "  - File: `_swe-duel/<<FEATURE_TEST_FILENAME>>` (the name MUST end in "
            "`.test.ts`).\n"
            "  - `import { test } from \"node:test\";` and "
            "`import assert from \"node:assert/strict\";`\n"
            "  - Import the source under test with a RELATIVE path from `_swe-duel/` "
            "(e.g. `<<NODE_IMPORT_EXAMPLE>>`). Mirror how the existing files in "
            "`test/` import the source.\n"
            "  - Each test is `test(\"name\", () => { ... })` (or `it(...)`) using "
            "`assert.*`.\n"
            "  - The harness runs your tests with: "
            "`npx tsx --test _swe-duel/<<FEATURE_TEST_FILENAME>>`.\n"
            "  - Verify locally the same way before submitting."
        ),
    },
    "node_mocha": {
        "lang": "node",
        "test_framework": "mocha (JavaScript)",
        "test_instructions": (
            "This is a NODE/JAVASCRIPT project — DO NOT write Python/pytest or "
            "TypeScript. Write a plain JavaScript (CommonJS) mocha test:\n"
            "  - File: `_swe-duel/<<FEATURE_TEST_FILENAME>>` (the name MUST end in "
            "`.test.js`).\n"
            "  - `const assert = require('assert');` and use "
            "`describe(...)/it(...)` (or `test(...)`).\n"
            "  - Require the source under test with a RELATIVE path from "
            "`_swe-duel/` (e.g. `<<NODE_IMPORT_EXAMPLE>>`). Mirror how the existing "
            "files in `test/` require the source.\n"
            "  - The harness runs your tests with: "
            "`npx mocha --reporter tap _swe-duel/<<FEATURE_TEST_FILENAME>>` "
            "(plus the repo's `--require` bootstrap).\n"
            "  - Verify locally the same way before submitting."
        ),
    },
    "c": {
        "lang": "c",
        "test_framework": "ANSI-C assertion-based tests compiled with gcc",
        "test_instructions": (
            "This is a C project — DO NOT write Python/pytest. Write a "
            "standalone C test program in `_swe-duel/<<FEATURE_TEST_FILENAME>>`.\n"
            "  - The file MUST contain `int main(void) { ... }`.\n"
            "  - Use `<assert.h>` and `assert(...)` for checks.\n"
            "  - Write helper test functions named `test_*` and call them from "
            "`main`.\n"
            "  - Include the library headers (e.g. `<<C_INCLUDE_DIRECTIVE>>`); "
            "do NOT embed the library's `.c` source files directly.\n"
            "  - The harness compiles your test together with the project's C "
            "source files (with the right include flags) and runs the resulting "
            "binary."
        ),
    },
    "java_maven": {
        "lang": "java",
        "test_framework": "JUnit 4 (via Maven Surefire)",
        "test_instructions": (
            "This is a JAVA project — DO NOT write Python/pytest. Write a "
            "JUnit 4 test class in `_swe-duel/<<FEATURE_TEST_FILENAME>>`.\n"
            "  - The class name must be exactly `TestSweDuelFeature` and must have "
            "NO `package` declaration (default package).\n"
            "  - Use `import static org.junit.Assert.*;` and "
            "`@Test public void testXxx() { ... }`.\n"
            "  - Import classes under test from the project, e.g. "
            "`import <<JAVA_IMPORT_ROOT>>.*;`.\n"
            "  - The harness copies the file into the Maven module "
            "`<<JAVA_MODULE>>` and runs: `<<JAVA_RUN_COMMAND>>`."
        ),
    },
    "java_gradle": {
        "lang": "java",
        "test_framework": "JUnit 4 (via Gradle)",
        "test_instructions": (
            "This is a JAVA project — DO NOT write Python/pytest. Write a "
            "JUnit 4 test class in `_swe-duel/<<FEATURE_TEST_FILENAME>>`.\n"
            "  - The class name must be exactly `TestSweDuelFeature` and must have "
            "NO `package` declaration (default package).\n"
            "  - Use `import static org.junit.Assert.*;` and "
            "`@Test public void testXxx() { ... }`.\n"
            "  - Import classes under test from the project, e.g. "
            "`import <<JAVA_IMPORT_ROOT>>.*;`.\n"
            "  - The harness copies the file into the Gradle module's test "
            "source set (`<<JAVA_MODULE>>`) and runs: "
            "`<<JAVA_RUN_COMMAND>>`."
        ),
    },
}


def _lang_key(language: str | None, repo_name: str | None) -> str:
    key = (language or "python").strip().lower()
    if key in ("javascript", "typescript"):
        key = "node"
    return key


def _prompt_language_context(
    repo_config: RepoConfig, feature_test_filename: str
) -> dict[str, str]:
    """Build the language-specific template variables for the Red prompts.

    Repo-specific values (Go import path, Java module/build tool, C header,
    Node runner + import example) come from the language adapter's
    ``prompt_hints()`` so adding a second repo for the same language only
    needs a YAML profile under ``languages/profiles/`` — no prompt edit required.
    """
    lang = _lang_key(repo_config.language, repo_config.name)
    adapter = get_adapter(repo_config.language, repo_config.name)
    hints = adapter.prompt_hints()

    if lang == "node":
        runner = hints.get("test_runner", "node_test")
        guide = _PROMPT_LANG_GUIDES["node_mocha"] if runner == "mocha" else _PROMPT_LANG_GUIDES["node_ts"]
    elif lang == "java":
        tool = hints.get("build_tool", "maven")
        guide = _PROMPT_LANG_GUIDES["java_gradle"] if tool == "gradle" else _PROMPT_LANG_GUIDES["java_maven"]
    elif lang == "c":
        guide = _PROMPT_LANG_GUIDES["c"]
    elif lang == "go":
        guide = _PROMPT_LANG_GUIDES["go"]
    else:
        guide = _PROMPT_LANG_GUIDES["python"]

    # Build the per-repo fill map.
    fill: dict[str, str] = {"<<FEATURE_TEST_FILENAME>>": feature_test_filename}
    if lang == "go":
        fill["<<GO_IMPORT_PATH>>"] = hints.get("import_path", "<go.mod module path>")
    elif lang == "node":
        fill["<<NODE_IMPORT_EXAMPLE>>"] = hints.get(
            "source_import_example", 'import x from "../lib/...";'
        )
    elif lang == "c":
        fill["<<C_INCLUDE_DIRECTIVE>>"] = hints.get(
            "include_directive", '#include "library.h"'
        )
    elif lang == "java":
        tool = hints.get("build_tool", "maven")
        module = hints.get("module", "")
        class_name = "TestSweDuelFeature"
        if tool == "gradle":
            fill["<<JAVA_IMPORT_ROOT>>"] = hints.get("import_root", "<project package>")
            fill["<<JAVA_MODULE>>"] = module
            fill["<<JAVA_BUILD_TOOL>>"] = "Gradle"
            fill["<<JAVA_RUN_COMMAND>>"] = (
                f"./gradlew {module}:test --console=plain --tests {class_name}"
            )
        else:
            fill["<<JAVA_IMPORT_ROOT>>"] = hints.get("import_root", "<project package>")
            fill["<<JAVA_MODULE>>"] = module
            fill["<<JAVA_BUILD_TOOL>>"] = "Maven"
            fill["<<JAVA_RUN_COMMAND>>"] = (
                f"mvn -B -pl {module} -am -Dtest={class_name} "
                f"-DfailIfNoTests=false test"
            )

    instructions = guide["test_instructions"]
    for placeholder, value in fill.items():
        instructions = instructions.replace(placeholder, value)

    return {
        "lang": guide["lang"],
        "test_framework": guide["test_framework"],
        "feature_test_filename": feature_test_filename,
        "test_instructions": instructions,
    }


class RedOutputError(ValueError):
    """Raised when Red's workspace output is missing or malformed."""


class AgentTimeoutError(Exception):
    """Raised when a Red phase hits its wall-clock budget.

    Carries the workspace plus whichever trajectories completed before the
    timeout fired so callers can persist failure artefacts for inspection.
    """

    def __init__(
        self,
        phase: str,
        workspace: Workspace,
        feature_trajectory: AgentTrajectory | None,
        bug_trajectory: AgentTrajectory | None,
        wall_seconds: float | None,
    ) -> None:
        super().__init__(
            f"red agent timed out during phase={phase} "
            f"(wall_limit={wall_seconds}s)"
        )
        self.phase = phase
        self.workspace = workspace
        self.feature_trajectory = feature_trajectory
        self.bug_trajectory = bug_trajectory
        self.wall_seconds = wall_seconds


class RedPhaseIncomplete(Exception):
    """Raised when a Red phase finishes without producing required artefacts.

    Typically fires when the agent hits its step budget (exit_status
    "LimitsExceeded") and never wrote a valid `_swe-duel/metadata.json` or
    `_swe-duel/feature_tests.py`. Carries the workspace and whichever
    trajectories completed so the caller can dump failure artefacts (incl.
    the rendered feature.html) for inspection.
    """

    def __init__(
        self,
        phase: str,
        workspace: Workspace,
        feature_trajectory: AgentTrajectory | None,
        bug_trajectory: AgentTrajectory | None,
        reason: str,
    ) -> None:
        super().__init__(
            f"red agent phase={phase} incomplete: {reason}"
        )
        self.phase = phase
        self.workspace = workspace
        self.feature_trajectory = feature_trajectory
        self.bug_trajectory = bug_trajectory
        self.reason = reason


class FeatureGateFailure(Exception):
    """Raised when the post-Phase-A feature gates fail; aborts before Phase B.

    Carries the partial (feature-only) challenge, the workspace, the feature
    trajectory, and the validation result so callers can log / account for the
    failed attempt without spending tokens on bug embedding.
    """

    def __init__(
        self,
        partial_challenge: RedChallenge,
        validation: RedValidationResult,
        workspace: Workspace,
        feature_trajectory: AgentTrajectory,
    ) -> None:
        super().__init__(
            f"feature gates failed after Phase A "
            f"(attempt {validation.attempt_number}): "
            + ", ".join(
                f"{g.gate_name}={g.status.value}" for g in validation.gate_results
            )
        )
        self.partial_challenge = partial_challenge
        self.validation = validation
        self.workspace = workspace
        self.feature_trajectory = feature_trajectory


_FEATURE_METADATA_KEYS = (
    "target_files",
    "exploration_summary",
    "feature_spec",
    "feature_rationale",
)

_BUG_METADATA_KEYS = (
    "bug_type",
    "bug_description",
    "bug_location",
)

# Files Red agents commonly leave behind that would trivially leak the bug to
# Blue (pre-bug backups of the target file, scratch copies, editor artefacts).
# Anything in the working tree matching these patterns is deleted before the
# challenge diff is computed.
_LEAKY_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "*.backup",
    "*.bak",
    "*.orig",
    "*.original",
    "*.old",
    "*.save",
    "*.prev",
    "*.previous",
    "*.new",
    "*.tmp",
    "*.rej",
    "*.swp",
    "*~",
    "*.py.backup",
    "*.py.bak",
    "*.py.orig",
    "*.py.original",
    "*.py.old",
    "*.py.new",
)


class RedAgent:
    def __init__(
        self,
        agent_wrapper: AgentHarness,
        workspace_manager: WorkspaceManager,
        prompt_dir: Path,
        feature_wall_seconds: float | None = None,
        bug_wall_seconds: float | None = None,
        feature_steps: int = 50,
        bug_steps: int = 50,
        min_diff_lines: int = 10,
        min_test_assertions: int = 2,
        min_test_functions: int = 1,
    ) -> None:
        self.agent_wrapper = agent_wrapper
        self.workspace_manager = workspace_manager
        self.prompt_dir = prompt_dir
        self.feature_wall_seconds = feature_wall_seconds
        self.bug_wall_seconds = bug_wall_seconds
        self.feature_steps = feature_steps
        self.bug_steps = bug_steps
        # Complexity-gate thresholds surfaced to the feature prompt so the agent
        # generates features large enough to pass gate_complexity (see
        # validation/complexity.py::check_thresholds).
        self.min_diff_lines = min_diff_lines
        self.min_test_assertions = min_test_assertions
        self.min_test_functions = min_test_functions
        self._env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self._feature_template = self._env.get_template("red_feature_task.md")
        self._bug_template = self._env.get_template("red_bug_task.md")

    # ── public ─────────────────────────────────────────────

    def generate_challenge(
        self,
        repo_config: RepoConfig,
        previous_gists: list[dict] | None = None,
        feature_validator: Callable[[RedChallenge, int], RedValidationResult] | None = None,
        attempt_number: int = 1,
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> tuple[RedChallenge, Workspace]:
        workspace = self.workspace_manager.create_workspace(
            repo_config,
            model_id=self.agent_wrapper.model_config.model_id,
            role="red",
        )
        swe_duel_dir = workspace.path / "_swe-duel"
        swe_duel_dir.mkdir(exist_ok=True)

        artifact_names = _artifact_names(repo_config.language)

        # ── Phase A: feature generation ───────────────────
        feature_trajectory = self._run_feature_phase(
            repo_config, workspace, previous_gists or [],
            feature_test_name=artifact_names["feature"],
            progress=progress, console_echo=console_echo,
        )

        # Snapshot whatever the agent produced and emit feature.html FIRST,
        # before requiring metadata. This way the diff + reasoning steps are
        # always inspectable, even if the agent ran out of steps
        # ("LimitsExceeded"), wall-timed out, or failed to write metadata.
        self._scrub_leaky_artifacts(workspace)
        feature_only_file_contents = self._snapshot_non_swe_duel_files(workspace)
        feature_original_contents = self.workspace_manager.get_original_files(
            workspace, list(feature_only_file_contents.keys())
        )
        feature_spec_for_html = ""
        try:
            _meta_raw = (workspace.path / "_swe-duel" / "metadata.json").read_text()
            _meta = json.loads(_meta_raw)
            if isinstance(_meta, dict):
                feature_spec_for_html = str(_meta.get("feature_spec", "") or "")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        self._write_feature_html(
            workspace=workspace,
            feature_spec=feature_spec_for_html,
            original_files=feature_original_contents,
            feature_only_files=feature_only_file_contents,
            feature_trajectory=feature_trajectory,
        )

        if feature_trajectory.exit_status == "WallClockTimeout":
            raise AgentTimeoutError(
                phase="feature",
                workspace=workspace,
                feature_trajectory=feature_trajectory,
                bug_trajectory=None,
                wall_seconds=self.feature_wall_seconds,
            )

        try:
            feature_metadata = self._load_feature_metadata(workspace)
            feature_test_code = self._load_feature_tests(
                workspace, artifact_names["feature"], repo_config.language,
                repo_config.name,
            )
        except RedOutputError as e:
            raise RedPhaseIncomplete(
                phase="feature",
                workspace=workspace,
                feature_trajectory=feature_trajectory,
                bug_trajectory=None,
                reason=(
                    f"{e} (agent exit_status="
                    f"{feature_trajectory.exit_status or 'unknown'!s})"
                ),
            ) from e

        # ── Feature-gate checkpoint: fail fast before Phase B ──
        # If the feature regresses existing tests or its own feature tests
        # don't pass, there is no point spending tokens on bug embedding.
        if feature_validator is not None:
            partial_challenge = self._build_feature_only_challenge(
                workspace=workspace,
                metadata=feature_metadata,
                feature_test_code=feature_test_code,
                feature_only_file_contents=feature_only_file_contents,
                feature_trajectory=feature_trajectory,
            )
            feature_validation = feature_validator(partial_challenge, attempt_number)
            if not feature_validation.passed:
                raise FeatureGateFailure(
                    partial_challenge=partial_challenge,
                    validation=feature_validation,
                    workspace=workspace,
                    feature_trajectory=feature_trajectory,
                )

        # ── Phase B: bug embedding ────────────────────────
        bug_trajectory = self._run_bug_phase(
            repo_config, workspace, feature_metadata,
            bug_test_name=artifact_names["bug"],
            feature_test_name=artifact_names["feature"],
            progress=progress, console_echo=console_echo,
        )
        if bug_trajectory.exit_status == "WallClockTimeout":
            raise AgentTimeoutError(
                phase="bug",
                workspace=workspace,
                feature_trajectory=feature_trajectory,
                bug_trajectory=bug_trajectory,
                wall_seconds=self.bug_wall_seconds,
            )
        # And again after Phase B: the agent frequently copies the pre-bug
        # file to a sibling `*.backup` / `*.original` path to sanity-check
        # its edits. Any such file would trivially hand the bug to Blue.
        self._scrub_leaky_artifacts(workspace)
        try:
            full_metadata = self._load_full_metadata(workspace)
            bug_test_code = self._load_bug_tests(
                workspace, artifact_names["bug"], repo_config.language,
                repo_config.name,
            )
        except RedOutputError as e:
            raise RedPhaseIncomplete(
                phase="bug",
                workspace=workspace,
                feature_trajectory=feature_trajectory,
                bug_trajectory=bug_trajectory,
                reason=(
                    f"{e} (agent exit_status="
                    f"{bug_trajectory.exit_status or 'unknown'!s})"
                ),
            ) from e

        # ── Extract final challenge ───────────────────────
        challenge = self._extract_challenge(
            workspace=workspace,
            metadata=full_metadata,
            feature_test_code=feature_test_code,
            bug_test_code=bug_test_code,
            feature_only_file_contents=feature_only_file_contents,
            feature_trajectory=feature_trajectory,
            bug_trajectory=bug_trajectory,
        )
        self._write_diff_visualizations(workspace, challenge)
        return challenge, workspace

    # ── phase runners ──────────────────────────────────────

    def _run_feature_phase(
        self,
        repo_config: RepoConfig,
        workspace: Workspace,
        previous_gists: list[dict],
        feature_test_name: str = "feature_tests.py",
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> AgentTrajectory:
        swe_duel_dir = workspace.path / "_swe-duel"
        lang_ctx = _prompt_language_context(repo_config, feature_test_name)
        task_prompt = self._feature_template.render(
            repo_name=repo_config.name,
            workspace_path=CONTAINER_WORKSPACE,
            previous_gists=previous_gists,
            min_diff_lines=self.min_diff_lines,
            min_test_assertions=self.min_test_assertions,
            min_test_functions=self.min_test_functions,
            **lang_ctx,
        )

        def _missing() -> list[str]:
            missing: list[str] = []
            if not (swe_duel_dir / "metadata.json").exists():
                missing.append("_swe-duel/metadata.json")
            else:
                try:
                    data = json.loads((swe_duel_dir / "metadata.json").read_text())
                except json.JSONDecodeError:
                    missing.append("_swe-duel/metadata.json (invalid JSON)")
                else:
                    if not isinstance(data, dict):
                        missing.append("_swe-duel/metadata.json (not an object)")
                    else:
                        for k in _FEATURE_METADATA_KEYS:
                            if k not in data:
                                missing.append(f"_swe-duel/metadata.json::{k}")
            if not (swe_duel_dir / feature_test_name).exists():
                missing.append(f"_swe-duel/{feature_test_name}")
            return missing

        def _reminder(items: list[str]) -> str:
            bullets = "\n".join(f"  - {p}" for p in items)
            return (
                "Phase 1 (feature generation) is incomplete. Before you submit, "
                "you MUST create the following file(s) / fields:\n"
                f"{bullets}\n\n"
                "metadata.json must include target_files, exploration_summary, "
                "feature_spec, feature_rationale. Write them now, then verify with "
                "`cat _swe-duel/metadata.json`, and only then submit with "
                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
            )

        return self.agent_wrapper.run(
            workspace.path,
            task_prompt,
            verbose=True,
            role_label="red-feature",
            completion_check=_missing,
            reminder_builder=_reminder,
            max_recovery_turns=3,
            max_wall_seconds=self.feature_wall_seconds,
            max_steps=self.feature_steps,
            log_file=swe_duel_dir / "red_feature_agent.log",
            step_callback=(
                (lambda c, m: progress("feature", c, m)) if progress else None
            ),
            console_echo=console_echo,
            docker_image=repo_config.docker_image,
            preserve_paths=repo_config.preserve_paths,
            login_shell=repo_config.login_shell,
        )

    def _run_bug_phase(
        self,
        repo_config: RepoConfig,
        workspace: Workspace,
        feature_metadata: dict,
        bug_test_name: str = "bug_tests.py",
        feature_test_name: str = "feature_tests.py",
        progress: Callable[[str, int, int], None] | None = None,
        console_echo: bool = True,
    ) -> AgentTrajectory:
        swe_duel_dir = workspace.path / "_swe-duel"
        lang_ctx = _prompt_language_context(repo_config, feature_test_name)
        lang_ctx["bug_test_filename"] = bug_test_name
        task_prompt = self._bug_template.render(
            repo_name=repo_config.name,
            workspace_path=CONTAINER_WORKSPACE,
            feature_spec=feature_metadata.get("feature_spec", ""),
            feature_rationale=feature_metadata.get("feature_rationale", ""),
            target_files=feature_metadata.get("target_files", []),
            pre_existing_failures=feature_metadata.get("pre_existing_failures", []),
            **lang_ctx,
        )

        def _missing() -> list[str]:
            missing: list[str] = []
            meta_path = swe_duel_dir / "metadata.json"
            if not meta_path.exists():
                missing.append("_swe-duel/metadata.json")
            else:
                try:
                    data = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    missing.append("_swe-duel/metadata.json (invalid JSON)")
                else:
                    if not isinstance(data, dict):
                        missing.append("_swe-duel/metadata.json (not an object)")
                    else:
                        for k in _BUG_METADATA_KEYS:
                            if k not in data or not str(data.get(k, "")).strip():
                                missing.append(f"_swe-duel/metadata.json::{k}")
            if not (swe_duel_dir / bug_test_name).exists():
                missing.append(f"_swe-duel/{bug_test_name}")
            return missing

        def _reminder(items: list[str]) -> str:
            bullets = "\n".join(f"  - {p}" for p in items)
            return (
                "Phase 2 (bug embedding) is incomplete. Before you submit, you MUST "
                "ensure the following file(s) / fields exist:\n"
                f"{bullets}\n\n"
                "metadata.json must now additionally contain bug_type, "
                "bug_description, bug_location. `bug_type` MUST be a short "
                "free-text categorization of the bug in your own words (up to six or seven "
                "words, e.g. \"Input validation, boundary, or sentinel handling error\"). "
                "bug_tests.py must have at "
                "least 1 test function that FAILS on the bugged code and "
                "PASSES on the original feature code. Write them now, then "
                "verify with `cat _swe-duel/metadata.json`, and only then "
                "submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
            )

        return self.agent_wrapper.run(
            workspace.path,
            task_prompt,
            verbose=True,
            role_label="red-bug",
            completion_check=_missing,
            reminder_builder=_reminder,
            max_recovery_turns=3,
            max_wall_seconds=self.bug_wall_seconds,
            max_steps=self.bug_steps,
            log_file=swe_duel_dir / "red_bug_agent.log",
            step_callback=(
                (lambda c, m: progress("bug", c, m)) if progress else None
            ),
            console_echo=console_echo,
            docker_image=repo_config.docker_image,
            preserve_paths=repo_config.preserve_paths,
            login_shell=repo_config.login_shell,
        )

    # ── loaders ────────────────────────────────────────────

    def _load_feature_metadata(self, workspace: Workspace) -> dict:
        meta_path = workspace.path / "_swe-duel" / "metadata.json"
        if not meta_path.exists():
            raise RedOutputError(
                f"Phase 1 did not produce _swe-duel/metadata.json at {meta_path}"
            )
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise RedOutputError(
                f"Phase 1 metadata.json is not valid JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise RedOutputError("Phase 1 metadata.json root is not an object")
        for k in _FEATURE_METADATA_KEYS:
            if k not in data:
                raise RedOutputError(
                    f"Phase 1 metadata.json missing required key: {k!r}"
                )
        return data

    def _load_feature_tests(
        self,
        workspace: Workspace,
        filename: str = "feature_tests.py",
        language: str | None = None,
        repo_name: str | None = None,
    ) -> str:
        tests_path = workspace.path / "_swe-duel" / filename
        if not tests_path.exists():
            raise RedOutputError(
                f"Phase 1 did not produce _swe-duel/{filename} at {tests_path}"
            )
        code = tests_path.read_text()
        self._validate_tests(
            code, label="feature", min_test_functions=3,
            language=language, repo_name=repo_name,
        )
        return code

    def _load_full_metadata(self, workspace: Workspace) -> dict:
        meta_path = workspace.path / "_swe-duel" / "metadata.json"
        if not meta_path.exists():
            raise RedOutputError(
                f"Phase 2 did not produce _swe-duel/metadata.json at {meta_path}"
            )
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise RedOutputError(
                f"Phase 2 metadata.json is not valid JSON: {e}"
            ) from e
        self._validate_metadata(data)
        return data

    def _load_bug_tests(
        self,
        workspace: Workspace,
        filename: str = "bug_tests.py",
        language: str | None = None,
        repo_name: str | None = None,
    ) -> str:
        tests_path = workspace.path / "_swe-duel" / filename
        if not tests_path.exists():
            raise RedOutputError(
                f"Phase 2 did not produce _swe-duel/{filename} at {tests_path}"
            )
        code = tests_path.read_text()
        self._validate_tests(
            code, label="bug", min_test_functions=1,
            language=language, repo_name=repo_name,
        )
        return code

    def _scrub_leaky_artifacts(self, workspace: Workspace) -> None:
        """Delete agent-created backup/scratch files that would leak the bug.

        Only removes files that (a) match a leaky pattern AND (b) are not
        present in the reference copy. Files that genuinely belong to the
        upstream repo are never touched.
        """
        ref_root = workspace.reference_path
        for path in workspace.path.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace.path)
            rel_str = str(rel)
            if rel_str.startswith("_swe-duel/") or rel_str.startswith("_swe-duel" + str(Path().anchor)):
                continue
            name = path.name
            if not any(fnmatch.fnmatch(name, pat) for pat in _LEAKY_ARTIFACT_PATTERNS):
                continue
            if (ref_root / rel).exists():
                # Upstream file with a suspicious name — leave it alone.
                continue
            try:
                path.unlink()
                print(
                    f"[red] scrubbed leaky workspace artefact: {rel_str}",
                    flush=True,
                )
            except OSError:
                pass

    def _snapshot_non_swe_duel_files(self, workspace: Workspace) -> dict[str, str]:
        modified = self.workspace_manager.get_modified_files(workspace)
        return {
            rel: content
            for rel, content in modified.items()
            if not rel.startswith("_swe-duel/")
            and not rel.startswith("_swe-duel" + str(Path().anchor))
        }

    def _build_feature_only_challenge(
        self,
        workspace: Workspace,
        metadata: dict,
        feature_test_code: str,
        feature_only_file_contents: dict[str, str],
        feature_trajectory: AgentTrajectory,
    ) -> RedChallenge:
        """Construct a partial RedChallenge (bug fields None) for mid-phase gating.

        The PR diff reflects the feature-only changes in the workspace so that
        ``gate_complexity`` can run before spending tokens on Phase B bug embedding.
        """
        target_files = list(metadata.get("target_files") or [])
        pr_diff = self.workspace_manager.compute_diff(workspace)
        return RedChallenge(
            target_files=target_files,
            exploration_summary=str(metadata.get("exploration_summary", "")),
            feature_spec=str(metadata.get("feature_spec", "")),
            feature_rationale=str(metadata.get("feature_rationale", "")),
            pr_diff=pr_diff,
            modified_file_contents=dict(feature_only_file_contents),
            original_file_contents={},
            feature_test_code=feature_test_code,
            bug_type=None,
            bug_description=None,
            bug_location=None,
            bug_test_code=None,
            agent_trajectory=feature_trajectory,
            feature_only_file_contents=dict(feature_only_file_contents),
            feature_trajectory=feature_trajectory,
            bug_trajectory=None,
            pre_existing_failures=list(metadata.get("pre_existing_failures") or []),
        )

    # ── extract / visualise ────────────────────────────────

    def _extract_challenge(
        self,
        workspace: Workspace,
        metadata: dict,
        feature_test_code: str,
        bug_test_code: str,
        feature_only_file_contents: dict[str, str],
        feature_trajectory: AgentTrajectory,
        bug_trajectory: AgentTrajectory,
    ) -> RedChallenge:
        pr_diff = self.workspace_manager.compute_diff(workspace)
        modified_file_contents = self.workspace_manager.get_modified_files(workspace)
        modified_file_contents = {
            rel: content
            for rel, content in modified_file_contents.items()
            if not rel.startswith("_swe-duel/")
            and not rel.startswith("_swe-duel" + str(Path().anchor))
        }
        original_file_contents = self.workspace_manager.get_original_files(
            workspace, list(modified_file_contents.keys())
        )

        # Free-text bug label (validated as a non-empty string by
        # _validate_metadata before this point).
        bug_type = str(metadata["bug_type"])

        combined_trajectory = self._combine_trajectories(
            feature_trajectory, bug_trajectory
        )

        return RedChallenge(
            target_files=list(metadata["target_files"]),
            exploration_summary=str(metadata["exploration_summary"]),
            feature_spec=str(metadata["feature_spec"]),
            feature_rationale=str(metadata["feature_rationale"]),
            pr_diff=pr_diff,
            modified_file_contents=modified_file_contents,
            original_file_contents=original_file_contents,
            feature_test_code=feature_test_code,
            bug_type=bug_type,
            bug_description=str(metadata["bug_description"]),
            bug_location=str(metadata["bug_location"]),
            bug_test_code=bug_test_code,
            agent_trajectory=combined_trajectory,
            feature_only_file_contents=feature_only_file_contents,
            feature_trajectory=feature_trajectory,
            bug_trajectory=bug_trajectory,
            pre_existing_failures=list(metadata.get("pre_existing_failures") or []),
        )

    @staticmethod
    def _combine_trajectories(
        a: AgentTrajectory, b: AgentTrajectory
    ) -> AgentTrajectory:
        return AgentTrajectory(
            steps=list(a.steps) + list(b.steps),
            total_steps=a.total_steps + b.total_steps,
            total_input_tokens=a.total_input_tokens + b.total_input_tokens,
            total_output_tokens=a.total_output_tokens + b.total_output_tokens,
            total_cost_usd=a.total_cost_usd + b.total_cost_usd,
            model_id=a.model_id,
            duration_seconds=a.duration_seconds + b.duration_seconds,
        )

    @staticmethod
    def _write_feature_html(
        workspace: Workspace,
        feature_spec: str,
        original_files: dict[str, str],
        feature_only_files: dict[str, str],
        feature_trajectory: AgentTrajectory | None = None,
    ) -> None:
        """Emit _swe-duel/feature.html (original → feature-only) right after Phase A."""
        swe_duel_dir = workspace.path / "_swe-duel"
        swe_duel_dir.mkdir(exist_ok=True)
        traj_steps = list(feature_trajectory.steps) if feature_trajectory else None
        traj_html = diff_utils._render_trajectory_steps(
            traj_steps,
            "Red agent reasoning — feature generation phase",
            "traj-feature",
        )
        feature_html = diff_utils.render_html_diff(
            original_files=original_files,
            modified_files=feature_only_files,
            title=f"Red — new feature ({workspace.repo_name})",
            annotation_html=(
                f"<div class='banner'><b>Feature:</b> "
                f"{html.escape(feature_spec)}</div>"
            ),
            trailing_html=f"<h2 id='reasoning'>Red agent reasoning</h2>{traj_html}",
        )
        (swe_duel_dir / "feature.html").write_text(feature_html)
        # print(
        #     f"[red] wrote feature diff HTML: {swe_duel_dir / 'feature.html'}",
        #     flush=True,
        # )

    @classmethod
    def _write_diff_visualizations(cls, workspace: Workspace, challenge: RedChallenge) -> None:
        """Emit _swe-duel/feature.html and _swe-duel/bug.html inside the workspace."""
        swe_duel_dir = workspace.path / "_swe-duel"
        swe_duel_dir.mkdir(exist_ok=True)

        feature_only = challenge.feature_only_file_contents or {}
        # Feature HTML: original → feature-only (pre-bug).
        cls._write_feature_html(
            workspace=workspace,
            feature_spec=challenge.feature_spec,
            original_files=challenge.original_file_contents,
            feature_only_files=feature_only or challenge.modified_file_contents,
            feature_trajectory=challenge.feature_trajectory,
        )

        # Bug HTML: feature-only → bugged.
        bug_annotation = (
            f"<div class='banner'><b>Embedded bug ({html.escape(challenge.bug_type or 'unknown')}):</b> "
            f"{html.escape(challenge.bug_description or '')}<br/>"
            f"<b>Location:</b> {html.escape(challenge.bug_location or '')}</div>"
        )
        bug_html = diff_utils.render_html_diff(
            original_files=feature_only or challenge.original_file_contents,
            modified_files=challenge.modified_file_contents,
            title=f"Red — embedded bug ({workspace.repo_name})",
            annotation_html=bug_annotation,
        )
        (swe_duel_dir / "bug.html").write_text(bug_html)

    # ── validators ─────────────────────────────────────────

    @staticmethod
    def _validate_metadata(metadata: dict) -> None:
        if not isinstance(metadata, dict):
            raise RedOutputError("metadata.json root is not an object")
        for key in _FEATURE_METADATA_KEYS + _BUG_METADATA_KEYS:
            if key not in metadata:
                raise RedOutputError(f"metadata.json missing required key: {key!r}")
        target_files = metadata["target_files"]
        if not isinstance(target_files, list) or not target_files:
            raise RedOutputError("target_files must be a non-empty list")
        for tf in target_files:
            if not isinstance(tf, str) or not tf.strip():
                raise RedOutputError(
                    f"target_files entries must be non-empty strings, got: {tf!r}"
                )
        for key in (
            "exploration_summary",
            "feature_spec",
            "feature_rationale",
            "bug_type",
            "bug_description",
            "bug_location",
        ):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise RedOutputError(f"metadata.{key} must be a non-empty string")

    @staticmethod
    def _validate_tests(
        test_code: str,
        label: str,
        min_test_functions: int = 3,
        language: str | None = None,
        repo_name: str | None = None,
    ) -> None:
        if not test_code or not test_code.strip():
            raise RedOutputError(f"{label} test code is empty")
        # Count test functions the way the language defines them.
        fn_count = get_adapter(language, repo_name).count_test_functions(test_code)
        if fn_count < min_test_functions:
            raise RedOutputError(
                f"{label} test code has {fn_count} test functions, "
                f"need at least {min_test_functions}"
            )
