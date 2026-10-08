"""Phase 3: diff_utils tests."""

from __future__ import annotations

import shutil

import os

import pytest

from swe_duel.sandbox.diff_utils import (
    apply_diff,
    compute_diff_stats,
    generate_diff,
    generate_tree_diff,
    iter_files_safe,
    normalize_diff,
)

from conftest import MOCK_REPO_DIR


def test_generate_diff_basic(record):
    original = "def foo():\n    return 1\n"
    modified = "def foo():\n    return 2\n"
    diff = generate_diff(original, modified, "foo.py")
    record("original", original)
    record("modified", modified)
    record("diff", diff)
    assert "--- " in diff
    assert "+++ " in diff
    assert "-    return 1" in diff
    assert "+    return 2" in diff


def test_apply_diff_roundtrip(record):
    original = "def foo():\n    return 1\n"
    modified = "def foo():\n    return 2\n"
    diff = generate_diff(original, modified, "foo.py")
    result = apply_diff(original, diff)
    record("original", original)
    record("diff", diff)
    record("result", {"success": result.success, "patched": result.patched, "error": result.error})
    assert result.success
    assert result.patched == modified


def test_apply_diff_malformed(record):
    bad_diff = "this is not a valid diff at all!!!"
    result = apply_diff("some content\n", bad_diff)
    record("input_diff", bad_diff)
    record("result", {"success": result.success, "patched": result.patched, "error": result.error})
    assert not result.success
    assert result.patched == ""
    assert result.error != ""


def test_compute_diff_stats(record):
    original = "line1\nline2\nline3\n"
    modified = "line1\nNEW_A\nNEW_B\nNEW_C\nNEW_D\nNEW_E\nline3\n"
    diff = generate_diff(original, modified, "f.py")
    stats = compute_diff_stats(diff)
    record("diff", diff)
    record("stats", {
        "lines_added": stats.lines_added,
        "lines_removed": stats.lines_removed,
        "hunks": stats.hunks,
        "files_changed": stats.files_changed,
    })
    assert stats.lines_added == 5
    assert stats.lines_removed == 1
    assert stats.hunks >= 1
    assert stats.files_changed == 1


def test_generate_tree_diff(tmp_path, record):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    shutil.copytree(MOCK_REPO_DIR, dir_a)
    shutil.copytree(MOCK_REPO_DIR, dir_b)

    target = dir_b / "src" / "calculator" / "basic.py"
    content = target.read_text()
    target.write_text(content + "\n# changed\n")

    diff = generate_tree_diff(dir_a, dir_b)
    record("diff", diff)
    assert "basic.py" in diff
    assert "# changed" in diff
    assert "advanced.py" not in diff


def test_generate_tree_diff_excludes(tmp_path, record):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    shutil.copytree(MOCK_REPO_DIR, dir_a)
    shutil.copytree(MOCK_REPO_DIR, dir_b)

    swe_duel_dir = dir_b / "_swe-duel"
    swe_duel_dir.mkdir()
    (swe_duel_dir / "metadata.json").write_text('{"key": "val"}')

    cache_dir = dir_b / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "basic.cpython-312.pyc").write_bytes(b"\x00\x01")

    diff = generate_tree_diff(
        dir_a, dir_b, exclude=["_swe-duel/", ".git/", "__pycache__/"]
    )
    record("excludes", ["_swe-duel/", ".git/", "__pycache__/"])
    record("diff", diff)
    assert "_swe-duel" not in diff
    assert "__pycache__" not in diff


def test_iter_files_safe_skips_unreadable_dir(tmp_path, record):
    """A directory with no read/execute permission must be skipped, not crash."""
    if os.geteuid() == 0:
        pytest.skip("running as root bypasses directory permissions")

    (tmp_path / "readable.txt").write_text("ok")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.txt").write_text("nope")
    os.chmod(locked, 0o000)
    try:
        files = {p.name for p in iter_files_safe(tmp_path)}
    finally:
        os.chmod(locked, 0o755)  # restore so tmp cleanup works

    record("files", sorted(files))
    # The readable file is found; the locked subtree is skipped without raising.
    assert "readable.txt" in files
    assert "secret.txt" not in files


def test_generate_tree_diff_tolerates_unreadable_dir(tmp_path, record):
    """generate_tree_diff must not crash if one side has an unreadable subtree."""
    if os.geteuid() == 0:
        pytest.skip("running as root bypasses directory permissions")

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "f.txt").write_text("one\n")
    (dir_b / "f.txt").write_text("two\n")
    locked = dir_b / "locked"
    locked.mkdir()
    (locked / "x.txt").write_text("data")
    os.chmod(locked, 0o000)
    try:
        diff = generate_tree_diff(dir_a, dir_b)
    finally:
        os.chmod(locked, 0o755)

    record("diff", diff)
    # The readable change is captured; no PermissionError was raised.
    assert "one" in diff and "two" in diff


def test_normalize_diff(record):
    raw = (
        "--- a/foo.py\t2024-01-01 12:00:00 +0000\n"
        "+++ b/foo.py\t2024-01-01 12:00:01 +0000\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    result = normalize_diff(raw)
    record("input", raw)
    record("normalized", result)
    assert "\t2024" not in result
    assert "a/foo.py" not in result
    assert "b/foo.py" not in result
    assert "foo.py" in result


def _defense_kwargs(**over):
    base = dict(
        defense_id="d1", challenge_id="c1", red_model_id="r", blue_model_id="b",
        repo_name="flask", target_files=["a.py"], feature_spec="feat",
        bug_type="logic_error", bug_description="d", bug_location="l",
        review_findings=[], fix_explanation="fix",
        original_file_contents={"a.py": "1\n"}, red_file_contents={"a.py": "2\n"},
        blue_file_contents={"a.py": "3\n"},
        s_regression=1.0, s_feature=1.0, s_bugfix=0.0, blue_composite=0.0,
        blue_trajectory_steps=[],
    )
    base.update(over)
    return base


def test_defense_html_renders_test_details(record):
    from swe_duel.sandbox.diff_utils import render_defense_html

    td = {
        "regression": {
            "passed": True, "total": 10, "passed_count": 10, "failed_count": 0,
            "error_count": 0, "failure_messages": [], "stdout": "10 passed",
            "stderr": "", "duration_ms": 1200,
            "command": "python -m pytest tests/ -q",
        },
        "bugfix": {
            "passed": False, "total": 1, "passed_count": 0, "failed_count": 1,
            "error_count": 0, "failure_messages": ["assert x==y"],
            "stdout": "1 failed", "stderr": "Traceback...", "duration_ms": 300,
            "command": "pytest test_swe_duel_bug.py",
        },
    }
    h = render_defense_html(**_defense_kwargs(test_details=td))
    record("len", len(h))
    assert "Post-Blue test runs" in h
    assert "python -m pytest tests/ -q" in h        # regression command rendered
    assert "pytest test_swe_duel_bug.py" in h           # bug command rendered
    assert "Traceback" in h                          # bug stderr rendered
    assert "gate-out" in h                           # output block class present
    # feature suite absent from test_details → shown as "not run"
    assert "not run" in h


def test_defense_html_without_test_details(record):
    from swe_duel.sandbox.diff_utils import render_defense_html

    h = render_defense_html(**_defense_kwargs())
    record("len", len(h))
    assert "No test details recorded" in h
