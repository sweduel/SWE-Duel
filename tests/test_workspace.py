"""Phase 3: workspace tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from swe_duel.config import RepoConfig
from swe_duel.sandbox.diff_utils import generate_diff
from swe_duel.sandbox.workspace import WorkspaceManager

from conftest import MOCK_REPO_DIR

MOCK_REPO_CONFIG = RepoConfig(
    name="mock_repo",
    url="file:///dev/null",
    commit="abc123",
    test_command="pytest tests/",
    docker_image="swe-duel-mock",
)


@pytest.fixture
def repos_dir(tmp_path) -> Path:
    d = tmp_path / "repos"
    d.mkdir()
    shutil.copytree(MOCK_REPO_DIR, d / "mock_repo")
    return d


@pytest.fixture
def manager(repos_dir, tmp_path) -> WorkspaceManager:
    return WorkspaceManager(repos_dir=repos_dir, tmp_root=tmp_path / "workspaces")


def test_create_workspace(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    record("workspace", {
        "workspace_id": ws.workspace_id,
        "repo_name": ws.repo_name,
        "path": str(ws.path),
        "reference_path": str(ws.reference_path),
        "path_exists": ws.path.exists(),
        "reference_exists": ws.reference_path.exists(),
    })
    assert ws.path.exists()
    assert ws.reference_path.exists()
    assert ws.repo_name == "mock_repo"
    assert ws.workspace_id


def test_compute_diff_no_changes(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    diff = manager.compute_diff(ws)
    record("diff", diff)
    assert diff == ""


def test_compute_diff_with_changes(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    target = ws.path / "src" / "calculator" / "basic.py"
    target.write_text(target.read_text() + "\n# modification\n")
    diff = manager.compute_diff(ws)
    record("diff", diff)
    assert diff != ""
    assert "basic.py" in diff
    assert "# modification" in diff


def test_get_modified_files(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    (ws.path / "src" / "calculator" / "basic.py").write_text("# changed basic\n")
    (ws.path / "src" / "calculator" / "advanced.py").write_text("# changed advanced\n")

    modified = manager.get_modified_files(ws)
    record("modified_files", modified)
    keys = set(modified.keys())
    assert any("basic.py" in k for k in keys)
    assert any("advanced.py" in k for k in keys)
    assert len(keys) == 2


def test_get_modified_files_excludes_deps_and_binaries(manager, record):
    """Agent-created dependency trees and binaries must never enter the diff.

    Regression guard for the multi-language bug: a Node/Go agent running in the
    workspace can create node_modules/ and native binaries; reading those as
    text corrupts them (and bloats challenge JSONs to hundreds of MB). They must
    be excluded from get_modified_files / compute_diff entirely.

    Also covers Gradle's ``.gradle/`` project cache (an agent running
    ``./gradlew`` writes hundreds of ``.bin``/``.class``/``.lock`` files there,
    which previously ballooned the java-jwt challenge diff to ~900 KB and made
    it unparseable, failing gate_diff_valid).
    """
    ws = manager.create_workspace(MOCK_REPO_CONFIG)

    # A legitimate source edit (should be captured).
    (ws.path / "src" / "calculator" / "basic.py").write_text("# changed basic\n")

    # A dependency tree the agent created (should be excluded by directory).
    nm = ws.path / "node_modules" / "leftpad"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = () => {}\n")

    # A native binary at the repo root (should be excluded by content sniff).
    (ws.path / "a.out").write_bytes(b"\x7fELF\x00\x00\x00binary\x00payload")

    # A Gradle project cache the agent created (should be excluded by directory).
    gradle = ws.path / ".gradle" / "6.9.2" / "fileHashes"
    gradle.mkdir(parents=True)
    (gradle / "fileHashes.bin").write_text("binary cache payload\n")

    modified = manager.get_modified_files(ws)
    record("modified_files_keys", sorted(modified.keys()))

    keys = set(modified.keys())
    assert any("basic.py" in k for k in keys)
    assert not any("node_modules" in k for k in keys)
    assert "a.out" not in keys
    assert not any(".gradle" in k for k in keys)

    diff = manager.compute_diff(ws)
    record("diff", diff)
    assert "node_modules" not in diff
    assert "a.out" not in diff
    assert ".gradle" not in diff
    assert "basic.py" in diff


def test_get_original_files(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    rel = "src/calculator/basic.py"
    original_content = (MOCK_REPO_DIR / rel).read_text()

    (ws.path / rel).write_text("# totally different\n")

    originals = manager.get_original_files(ws, [rel])
    record("requested_paths", [rel])
    record("originals", originals)
    assert originals[rel] == original_content


def test_apply_diff_to_workspace(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    rel = "src/calculator/basic.py"
    original = (ws.path / rel).read_text()
    modified = original + "\n# injected\n"
    diff = generate_diff(original, modified, rel)

    success = manager.apply_diff_to_workspace(ws, diff)
    final_contents = (ws.path / rel).read_text()
    record("diff", diff)
    record("success", success)
    record("final_contents", final_contents)
    assert success
    assert "# injected" in final_contents


def test_cleanup(manager, record):
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    working = ws.path
    reference = ws.reference_path
    manager.cleanup(ws)
    record("working_exists_after_cleanup", working.exists())
    record("reference_exists_after_cleanup", reference.exists())
    assert not working.exists()
    assert not reference.exists()


def test_repo_specific_glob_excludes(repos_dir, tmp_path, record):
    """Repo-specific exclude_paths (globs) keep build artifacts out of the diff.

    Regression guard for the libexpat in-tree-autotools-build bug: an agent
    that runs ``./configure && make`` generates ``autom4te.cache/``,
    ``Makefile``, ``*.o``, etc. in the workspace. These are not source edits
    and must not pollute the challenge diff (a 6 MB diff of generated files
    was unparseable and failed gate_diff_valid).
    """
    shutil.copytree(MOCK_REPO_DIR, repos_dir / "glob_repo")
    cfg = RepoConfig(
        name="glob_repo",
        url="file:///dev/null",
        commit="abc",
        test_command="true",
        docker_image="swe-duel-mock",
        exclude_paths=[
            "generated/autom4te.cache/*",
            "generated/Makefile",
            "generated/*.o",
        ],
    )
    manager = WorkspaceManager(repos_dir=repos_dir, tmp_root=tmp_path / "workspaces")
    ws = manager.create_workspace(cfg)

    # A legitimate source edit (kept).
    (ws.path / "src" / "calculator" / "basic.py").write_text("# changed\n")
    # Autotools-style build artifacts under generated/ (excluded by globs).
    (ws.path / "generated" / "autom4te.cache").mkdir(parents=True)
    (ws.path / "generated" / "autom4te.cache" / "requests").write_text("generated\n")
    (ws.path / "generated" / "Makefile").write_text("generated makefile\n")
    (ws.path / "generated" / "foo.o").write_text("object\n")
    # A tracked source file under generated/ that is NOT matched by the globs
    # (proves the globs are precise, not over-broad).
    (ws.path / "generated" / "real_source.c").write_text("# real edit\n")

    modified = manager.get_modified_files(ws)
    record("modified_files_keys", sorted(modified.keys()))
    keys = set(modified.keys())
    assert "src/calculator/basic.py" in keys
    assert "generated/real_source.c" in keys
    assert "generated/Makefile" not in keys
    assert "generated/foo.o" not in keys
    assert not any("autom4te.cache" in k for k in keys)

    diff = manager.compute_diff(ws)
    record("diff", diff)
    assert "autom4te.cache" not in diff
    assert "generated/Makefile" not in diff
    assert "generated/foo.o" not in diff
    assert "real_source.c" in diff
    assert "basic.py" in diff


def test_exclude_paths_default_empty(manager, record):
    """Workspaces for repos without exclude_paths carry an empty list."""
    ws = manager.create_workspace(MOCK_REPO_CONFIG)
    assert ws.excludes == []
