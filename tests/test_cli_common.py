"""Unit tests for the CLI path-resolution chains (no Docker, no LLM).

``swe_duel.cli._common`` resolves three knobs for every run CLI, in a fixed order:

* config dir: ``--config-dir`` > ``SWE_DUEL_CONFIG_DIR`` > ``./config`` > exit(2)
* output dir:  ``--output-dir`` > ``SWE_DUEL_OUTPUT_DIR`` > arena.yaml
  ``paths.output_dir`` > ``./data``

These tests pin that order so pip-installed users get predictable scaffolding
semantics (``swe-duel init`` writes ``./config`` + ``./data``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from swe_duel.cli._common import default_prompt_dir, resolve_config_dir, resolve_output_dir
from swe_duel.config import ArenaConfig, PathsConfig


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _make_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "arena.yaml").write_text("rating:\n  elo_k: 32\n")
    return config_dir


class TestResolveConfigDir:
    def test_cli_flag_wins(self, tmp_path: Path, monkeypatch, record):
        marked = _make_config(tmp_path)
        monkeypatch.setenv("SWE_DUEL_CONFIG_DIR", str(tmp_path / "elsewhere"))
        monkeypatch.chdir(tmp_path)
        got = resolve_config_dir(_ns(config_dir=str(marked)))
        record("resolved", str(got))
        assert got == marked

    def test_env_var_beats_cwd(self, tmp_path: Path, monkeypatch, record):
        _make_config(tmp_path)  # ./config exists in CWD — env must still win
        env_dir = tmp_path / "envcfg"
        env_dir.mkdir()
        (env_dir / "arena.yaml").write_text("rating:\n  elo_k: 32\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SWE_DUEL_CONFIG_DIR", str(env_dir))
        got = resolve_config_dir(_ns(config_dir=None))
        record("resolved", str(got))
        assert got == env_dir

    def test_cwd_config_fallback(self, tmp_path: Path, monkeypatch, record):
        _make_config(tmp_path)
        monkeypatch.delenv("SWE_DUEL_CONFIG_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        got = resolve_config_dir(_ns(config_dir=None))
        record("resolved", str(got))
        assert got.resolve() == (tmp_path / "config").resolve()

    def test_missing_config_exits_with_guidance(
        self, tmp_path: Path, monkeypatch, capsys, record
    ):
        monkeypatch.delenv("SWE_DUEL_CONFIG_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # no ./config here
        with pytest.raises(SystemExit) as exc:
            resolve_config_dir(_ns(config_dir=None))
        stderr = capsys.readouterr().err
        record("exit_code", exc.value.code)
        record("stderr", stderr)
        assert exc.value.code == 2
        assert "swe-duel init" in stderr


class TestResolveOutputDir:
    def test_full_chain(self, tmp_path: Path, monkeypatch, record):
        arena = ArenaConfig(paths=PathsConfig(output_dir=str(tmp_path / "from-yaml")))
        monkeypatch.delenv("SWE_DUEL_OUTPUT_DIR", raising=False)

        # 4. arena.yaml value
        record("yaml", str(resolve_output_dir(_ns(output_dir=None), arena)))
        assert resolve_output_dir(_ns(output_dir=None), arena) == tmp_path / "from-yaml"

        # 3. env var beats arena.yaml
        monkeypatch.setenv("SWE_DUEL_OUTPUT_DIR", str(tmp_path / "from-env"))
        record("env", str(resolve_output_dir(_ns(output_dir=None), arena)))
        assert resolve_output_dir(_ns(output_dir=None), arena) == tmp_path / "from-env"

        # 2. CLI flag beats everything
        record(
            "cli",
            str(resolve_output_dir(_ns(output_dir=str(tmp_path / "from-cli")), arena)),
        )
        assert (
            resolve_output_dir(_ns(output_dir=str(tmp_path / "from-cli")), arena)
            == tmp_path / "from-cli"
        )

    def test_default_is_cwd_data(self, record):
        arena = ArenaConfig()
        os.environ.pop("SWE_DUEL_OUTPUT_DIR", None)
        got = resolve_output_dir(_ns(output_dir=None), arena)
        record("default", str(got))
        assert got == Path("data")


class TestDefaultPromptDir:
    def test_bundled_prompts_present(self, record):
        d = default_prompt_dir()
        record("prompt_dir", str(d))
        names = {p.name for p in d.iterdir()}
        assert {
            "blue_task.md",
            "red_feature_task.md",
            "red_bug_task.md",
            "red_self_review_task.md",
        } <= names
