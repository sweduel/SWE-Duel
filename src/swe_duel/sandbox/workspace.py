"""Workspace lifecycle management for agent sessions."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path

from swe_duel.config import RepoConfig
from swe_duel.models import Workspace
from swe_duel.sandbox import diff_utils

# Directories that are build output, dependency trees, or VCS/tool caches — never
# part of an agent's source change, and (for node_modules/vendor) full of native
# binaries that `read_text(errors="replace")` would corrupt and bloat. Excluded
# from every diff/snapshot so they cannot leak into a challenge's
# modified_file_contents. Agents run in Docker with deps baked into the image, so
# these should not appear at all — this is defense in depth.
# SWE-Duel injected Java test classes are also listed here because Red agents
# sometimes copy them out of `_swe-duel/` into the Maven module tree; the harness
# runs them explicitly and they must not be treated as source changes.
_DEFAULT_EXCLUDE = [
    "_swe-duel/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "node_modules/",
    "vendor/",
    "dist/",
    "build/",
    ".next/",
    ".turbo/",
    "coverage/",
    ".cache/",
    "target/",
    ".m2/",
    # Gradle's project-local cache (written to the workspace root when an
    # agent runs `./gradlew`). Like Maven's `target/` it is pure build
    # output, never a source edit, and balloons the challenge diff with
    # hundreds of `.bin`/`.class`/`.lock` files that make it unparseable.
    ".gradle/",
    "TestSweDuelFeature.java",
    "TestSweDuelBug.java",
    # OpenHands agent-server persistence (redirected out of the workspace via
    # OH_CONVERSATIONS_PATH, but excluded defensively in case it ever lands here
    # — it is written as root and is not an agent edit).
    "conversations/",
    "bash_events/",
]


def _is_probably_binary(path: Path) -> bool:
    """Heuristic: treat a file as binary if its first 8 KiB contain a NUL byte.

    Binaries must never be read with ``read_text`` — the lossy UTF-8 decode
    corrupts them (and can balloon their size). Used to keep binary files out of
    diffs/snapshots even if they slip past the directory excludes.
    """
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return False


class WorkspaceManager:
    def __init__(self, repos_dir: Path, tmp_root: Path | None = None) -> None:
        self.repos_dir = repos_dir
        if tmp_root is None:
            self._owns_tmp = True
            self.tmp_root = Path(tempfile.mkdtemp(prefix="swe_duel_workspaces_"))
        else:
            self._owns_tmp = False
            self.tmp_root = tmp_root
            self.tmp_root.mkdir(parents=True, exist_ok=True)

    def create_workspace(
        self,
        repo_config: RepoConfig,
        model_id: str | None = None,
        role: str | None = None,
    ) -> Workspace:
        """Create a fresh workspace with reference + mutable working copies.

        When `model_id` and `role` are supplied, the workspace is nested as
        `tmp_root/<sanitized_model>/<repo>/<role>/<uuid>/` so each model's
        Red/Blue workspaces are segregated per repo.
        """
        uid = str(uuid.uuid4())
        if model_id and role:
            model_slug = model_id.replace("/", "__").replace(":", "_")
            base = self.tmp_root / model_slug / repo_config.name / role / uid
            workspace_id = f"{model_slug}/{repo_config.name}/{role}/{uid}"
        else:
            base = self.tmp_root / uid
            workspace_id = uid
        base.mkdir(parents=True)

        source = self.repos_dir / repo_config.name
        reference_path = base / "reference"
        working_path = base / "working"

        shutil.copytree(source, reference_path)
        shutil.copytree(source, working_path)

        return Workspace(
            workspace_id=workspace_id,
            repo_name=repo_config.name,
            path=working_path.resolve(),
            reference_path=reference_path.resolve(),
            excludes=list(getattr(repo_config, "exclude_paths", []) or []),
        )

    def compute_diff(self, workspace: Workspace) -> str:
        """Return unified diff of all modifications made in the working copy."""
        return diff_utils.generate_tree_diff(
            workspace.reference_path,
            workspace.path,
            exclude=_DEFAULT_EXCLUDE + list(workspace.excludes or []),
        )

    def get_modified_files(self, workspace: Workspace) -> dict[str, str]:
        """Return {relative_path: content} for every file that differs from reference."""
        modified: dict[str, str] = {}
        excludes = _DEFAULT_EXCLUDE + list(workspace.excludes or [])

        # Use the permission-tolerant walker: a containerised agent that ran as
        # root can leave root-owned dirs (e.g. the agent-server's
        # `conversations`) in the bind-mounted workspace, and a plain rglob would
        # crash with PermissionError. Such files are excluded from diffs anyway.
        all_rels: set[str] = set()
        for p in diff_utils.iter_files_safe(workspace.path):
            all_rels.add(str(p.relative_to(workspace.path)))
        for p in diff_utils.iter_files_safe(workspace.reference_path):
            all_rels.add(str(p.relative_to(workspace.reference_path)))

        for rel in all_rels:
            if _is_excluded(rel, excludes):
                continue
            wp = workspace.path / rel
            rp = workspace.reference_path / rel
            # Never read binaries as text — the lossy decode corrupts them.
            try:
                if (wp.exists() and _is_probably_binary(wp)) or (
                    rp.exists() and _is_probably_binary(rp)
                ):
                    continue
                content_w = wp.read_text(errors="replace") if wp.exists() else ""
                content_r = rp.read_text(errors="replace") if rp.exists() else ""
            except OSError:
                # Unreadable (e.g. root-owned leftover) → not an agent edit.
                continue
            if content_w != content_r:
                modified[rel] = content_w

        return modified

    def get_original_files(self, workspace: Workspace, paths: list[str]) -> dict[str, str]:
        """Read specific files from the reference copy."""
        return {
            rel: (workspace.reference_path / rel).read_text(errors="replace")
            for rel in paths
            if (workspace.reference_path / rel).exists()
        }

    def read_file(self, workspace: Workspace, relative_path: str) -> str:
        """Read a single file from the working copy."""
        return (workspace.path / relative_path).read_text(errors="replace")

    def file_exists(self, workspace: Workspace, relative_path: str) -> bool:
        """Check if a file exists in the working copy."""
        return (workspace.path / relative_path).exists()

    def apply_diff_to_workspace(self, workspace: Workspace, diff: str) -> bool:
        """Apply a unified diff to the working copy. Returns success/failure."""
        try:
            from unidiff import PatchSet
            patch = PatchSet(diff)
        except Exception:
            return False

        for patched_file in patch:
            # Determine target path, stripping a/ b/ prefix if present
            target_rel = patched_file.path
            for prefix in ("b/", "a/"):
                if target_rel.startswith(prefix):
                    target_rel = target_rel[len(prefix):]
                    break

            target_path = workspace.path / target_rel
            original = target_path.read_text(errors="replace") if target_path.exists() else ""
            result = diff_utils.apply_diff(original, diff)

            if not result.success:
                # Re-attempt with the full diff scoped to this file
                single_patch = "\n".join(
                    line for line in diff.splitlines()
                    if not line.startswith("diff --git")
                )
                result = diff_utils.apply_diff(original, single_patch)
                if not result.success:
                    return False

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(result.patched)

        return True

    def cleanup(self, workspace: Workspace) -> None:
        """Delete both reference and working temp directories."""
        _robust_rmtree(workspace.path.parent)

    def cleanup_all(self) -> None:
        """Delete entire temp root."""
        _robust_rmtree(self.tmp_root)


def _robust_rmtree(path: Path) -> None:
    """Remove a directory tree even when it contains root-owned files.

    The OpenHands agent-server runs as **root** inside its container, so any
    `__pycache__/*.pyc`, build output, or persisted state it writes into the
    bind-mounted workspace ends up root-owned. A plain ``shutil.rmtree`` then
    fails with ``PermissionError`` because the non-root host user cannot unlink
    files inside a root-owned directory. We try (1) a normal rmtree, (2) an
    rmtree that chmods what it can on error, and finally (3) a throwaway root
    Docker container that ``rm -rf``s the path over a bind mount — the only way
    to delete files the host user does not own. Cleanup must never abort a run.
    """
    if not path.exists():
        return

    def _onerror(func, p, exc_info):  # type: ignore[no-untyped-def]
        # Best effort: make the entry (and its parent) writable, then retry.
        try:
            os.chmod(p, stat.S_IRWXU)
            parent = os.path.dirname(p)
            if parent:
                os.chmod(parent, os.stat(parent).st_mode | stat.S_IRWXU)
            func(p)
        except Exception:
            pass

    try:
        shutil.rmtree(path)
        return
    except Exception:
        pass
    try:
        # `onexc` (py3.12+) / `onerror` (older) — pass both-compatible callback.
        shutil.rmtree(path, onerror=_onerror)
    except Exception:
        pass
    if not path.exists():
        return
    # Still here → root-owned files remain. Delete them from inside a root
    # container that bind-mounts the parent dir.
    try:
        parent = str(path.parent.resolve())
        name = path.name
        subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{parent}:/host_cleanup",
                "alpine", "sh", "-c", f"rm -rf '/host_cleanup/{name}'",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except Exception:
        pass


def _is_excluded(rel: str, exclude: list[str]) -> bool:
    import fnmatch

    for pattern in exclude:
        if any(ch in pattern for ch in "*?["):
            # Glob pattern: match the full relative path.
            if fnmatch.fnmatch(rel, pattern):
                return True
        elif rel.startswith(pattern) or ("/" + pattern) in rel:
            return True
    return False
