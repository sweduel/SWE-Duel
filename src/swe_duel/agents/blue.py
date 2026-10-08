"""Blue agent: review Red's PR and fix any embedded bug while retaining the feature."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from swe_duel.agents.agent_wrapper import CONTAINER_WORKSPACE, AgentWrapper
from swe_duel.config import RepoConfig
from swe_duel.models import (
    AgentTrajectory,
    BlueFix,
    ChallengeRecord,
    ReviewFinding,
    Workspace,
)
from swe_duel.sandbox import diff_utils
from swe_duel.sandbox.languages import get_adapter
from swe_duel.sandbox.workspace import WorkspaceManager


class BlueAgent:
    def __init__(
        self,
        agent_wrapper: AgentWrapper,
        workspace_manager: WorkspaceManager,
        prompt_dir: Path,
        wall_seconds: float | None = None,
        steps: int = 50,
    ) -> None:
        self.agent_wrapper = agent_wrapper
        self.workspace_manager = workspace_manager
        self.prompt_dir = prompt_dir
        self.wall_seconds = wall_seconds
        self.steps = steps
        self._env = Environment(
            loader=FileSystemLoader(str(prompt_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template("blue_task.md")

    # ── public ─────────────────────────────────────────────

    def review_and_fix(
        self,
        repo_config: RepoConfig,
        challenge_record: ChallengeRecord,
        *,
        step_callback: Callable[[int, int], None] | None = None,
        console_echo: bool = True,
    ) -> tuple[BlueFix, Workspace]:
        """Review Red's PR and fix the embedded bug while retaining the feature.

        ``step_callback(step, max_steps)`` (optional) is forwarded to the harness
        so a live UI can tick the Blue agent's per-step progress bar.
        ``console_echo`` is set False when a live TUI owns the console (per-step
        reasoning still streams to ``_swe-duel/blue.log``).
        """
        workspace = self.workspace_manager.create_workspace(
            repo_config,
            model_id=self.agent_wrapper.model_config.model_id,
            role="blue",
        )

        challenge = challenge_record.challenge
        self.workspace_manager.apply_diff_to_workspace(workspace, challenge.pr_diff)

        swe_duel_dir = workspace.path / "_swe-duel"
        swe_duel_dir.mkdir(exist_ok=True)

        adapter = get_adapter(repo_config.language, repo_config.name)
        feature_extra, feature_command = adapter.prepare_injected(
            "test_swe_duel_feature.py",
            challenge.feature_test_code,
            challenge.target_files,
        )
        existing_command = adapter.build_existing_command(
            repo_config.test_command, []
        )

        task_prompt = self._template.render(
            repo_name=repo_config.name,
            workspace_path=CONTAINER_WORKSPACE,
            feature_spec=challenge.feature_spec,
            pr_diff=challenge.pr_diff,
            feature_test_code=challenge.feature_test_code,
            lang=(repo_config.language or "python"),
            test_framework=getattr(adapter, "name", repo_config.language or "python"),
            existing_test_command=existing_command,
            feature_test_command=feature_command,
            feature_test_file=next(iter(feature_extra.keys())),
        )

        def _missing() -> list[str]:
            missing: list[str] = []
            if not (swe_duel_dir / "review.json").exists():
                missing.append("_swe-duel/review.json")
            return missing

        def _reminder(items: list[str]) -> str:
            bullets = "\n".join(f"  - {p}" for p in items)
            return (
                "You have not written your review yet. Before you submit, you MUST "
                "create the following file(s):\n"
                f"{bullets}\n\n"
                "`_swe-duel/review.json` must contain two top-level keys: `findings` "
                "(a list of {location, severity, description} objects) and "
                "`fix_explanation` (a string). Write it now, then verify with "
                "`cat _swe-duel/review.json`, and only then submit with "
                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
            )

        trajectory = self.agent_wrapper.run(
            workspace.path,
            task_prompt,
            verbose=True,
            role_label="blue",
            completion_check=_missing,
            reminder_builder=_reminder,
            max_recovery_turns=3,
            max_wall_seconds=self.wall_seconds,
            max_steps=self.steps,
            log_file=workspace.path / "_swe-duel" / "blue.log",
            step_callback=step_callback,
            console_echo=console_echo,
            docker_image=repo_config.docker_image,
            preserve_paths=repo_config.preserve_paths,
            login_shell=repo_config.login_shell,
        )
        blue_fix = self._extract_fix(workspace, trajectory)
        self._write_diff_visualizations(workspace, challenge_record, blue_fix)
        return blue_fix, workspace

    @staticmethod
    def _write_diff_visualizations(
        workspace: Workspace,
        challenge_record: ChallengeRecord,
        blue_fix: BlueFix,
    ) -> None:
        """Emit _swe-duel/detected.html and _swe-duel/removal.html inside the workspace."""
        swe_duel_dir = workspace.path / "_swe-duel"
        swe_duel_dir.mkdir(exist_ok=True)

        challenge = challenge_record.challenge

        if blue_fix.review_findings:
            findings_html = "<ul>" + "".join(
                f"<li><b>{html.escape(f.severity)}</b> @ {html.escape(f.location)}: "
                f"{html.escape(f.description)}</li>"
                for f in blue_fix.review_findings
            ) + "</ul>"
        else:
            findings_html = "<p><em>No findings reported.</em></p>"

        detected_html = diff_utils.render_html_diff(
            original_files=challenge.original_file_contents,
            modified_files=challenge.modified_file_contents,
            title=f"Blue — detected bug ({workspace.repo_name})",
            annotation_html=(
                f"<div class='banner'><b>Blue review findings:</b>{findings_html}</div>"
            ),
        )
        (swe_duel_dir / "detected.html").write_text(detected_html)

        # Removal: Red's (buggy) files → Blue's fixed files.
        fixed_files = {
            rel: blue_fix.modified_file_contents.get(
                rel, challenge.modified_file_contents[rel]
            )
            for rel in challenge.modified_file_contents
        }
        for rel, content in blue_fix.modified_file_contents.items():
            fixed_files.setdefault(rel, content)

        removal_html = diff_utils.render_html_diff(
            original_files=challenge.modified_file_contents,
            modified_files=fixed_files,
            title=f"Blue — bug removal ({workspace.repo_name})",
            annotation_html=(
                f"<div class='banner'><b>Fix explanation:</b> "
                f"{html.escape(blue_fix.fix_explanation)}</div>"
            ),
        )
        (swe_duel_dir / "removal.html").write_text(removal_html)

    # ── internals ──────────────────────────────────────────

    def _extract_fix(
        self, workspace: Workspace, trajectory: AgentTrajectory
    ) -> BlueFix:
        fix_diff = self.workspace_manager.compute_diff(workspace)
        modified_file_contents = self.workspace_manager.get_modified_files(workspace)
        modified_file_contents = {
            rel: content
            for rel, content in modified_file_contents.items()
            if not rel.startswith("_swe-duel/") and not rel.startswith("_swe-duel" + str(Path().anchor))
        }

        findings: list[ReviewFinding] = []
        fix_explanation = "no issues found"

        review_path = workspace.path / "_swe-duel" / "review.json"
        if review_path.exists():
            try:
                data = json.loads(review_path.read_text())
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                raw_findings = data.get("findings") or []
                if isinstance(raw_findings, list):
                    for item in raw_findings:
                        if not isinstance(item, dict):
                            continue
                        findings.append(
                            ReviewFinding(
                                location=str(item.get("location", "")),
                                severity=str(item.get("severity", "info")),
                                description=str(item.get("description", "")),
                            )
                        )
                expl = data.get("fix_explanation")
                if isinstance(expl, str) and expl.strip():
                    fix_explanation = expl

        return BlueFix(
            review_findings=findings,
            fix_explanation=fix_explanation,
            fix_diff=fix_diff,
            modified_file_contents=modified_file_contents,
            agent_trajectory=trajectory,
        )
