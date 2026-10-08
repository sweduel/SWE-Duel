"""Tests for the ``swe-duel`` umbrella dispatcher (`swe_duel.cli.main`)."""

from __future__ import annotations

import sys
import types

import pytest

from swe_duel.cli import main as cli_main


def test_registry_covers_every_console_script() -> None:
    expected = {
        "init",
        "setup",
        "doctor",
        "generate",
        "match",
        "tournament",
        "tournament-rr",
        "tournament-as",
        "tournament-update",
        "rankings",
        "ablation",
        "evaluate",
        "report",
        "kill-containers",
        "probe-rate-limits",
    }
    assert set(cli_main.SUBCOMMANDS) == expected


def test_no_args_prints_usage_and_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main.main([]) == 2
    captured = capsys.readouterr()
    assert "usage: swe-duel <subcommand>" in captured.err
    assert captured.out == ""


def test_help_prints_usage_and_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "subcommands:" in captured.out
    for name in cli_main.SUBCOMMANDS:
        assert name in captured.out
    assert captured.err == ""


def test_unknown_subcommand_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main.main(["bogus"]) == 2
    assert "unknown subcommand: bogus" in capsys.readouterr().err


@pytest.fixture
def stub_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    def fake_main() -> int:
        seen["argv"] = list(sys.argv)
        return 7

    stub = types.ModuleType("swe_duel.cli._stub_cmd")
    stub.main = fake_main
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(
        cli_main,
        "SUBCOMMANDS",
        {"stub": ("swe_duel.cli._stub_cmd", "stub description")},
    )
    return seen


def test_dispatch_forwards_rest_as_swe_duel_argv(stub_module: dict[str, object]) -> None:
    saved_argv = list(sys.argv)
    assert cli_main.main(["stub", "--flag", "value"]) == 7
    assert stub_module["argv"] == ["swe-duel-stub", "--flag", "value"]
    assert sys.argv == saved_argv


def test_help_subcommand_runs_submodule_help(stub_module: dict[str, object]) -> None:
    assert cli_main.main(["help", "stub"]) == 7
    assert stub_module["argv"] == ["swe-duel-stub", "--help"]


def test_help_unknown_subcommand_returns_2(stub_module: dict[str, object]) -> None:
    assert cli_main.main(["help", "bogus"]) == 2
    assert "argv" not in stub_module
