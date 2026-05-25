"""Tests for the chisel CLI."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from chisel.cli import _collect_pass_package_config, main


def test_main_no_args_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "chisel" in out
    assert "run" in out
    assert "list" in out


def test_version_flag(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert "chisel" in capsys.readouterr().out.lower()


def test_list_command_prints_passes_and_evaluators(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "list"])
    main()
    out = capsys.readouterr().out
    assert "TorchPruningPass" in out
    assert "FineTunePass" in out
    assert "QuantErrorEvaluator" in out


def test_run_command_positional_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "run", "workflow.yaml"])
    with patch("olive.workflows.run") as mock_run:
        main()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == "workflow.yaml"


def test_run_command_flag_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "run", "--config", "workflow.yaml"])
    with patch("olive.workflows.run") as mock_run:
        main()
    assert mock_run.call_args.args[0] == "workflow.yaml"


def test_run_command_requires_config(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "run"])
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err.lower()
    assert "config" in err


def test_run_registers_chisel_passes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel", "run", "workflow.yaml"])
    with patch("olive.workflows.run") as mock_run:
        main()
    pkg = mock_run.call_args.kwargs["package_config"]
    assert "TorchPruningPass" in pkg["passes"]
    assert "FineTunePass" in pkg["passes"]


def test_collect_pass_package_config_uses_real_module_paths():
    cfg = _collect_pass_package_config()
    assert cfg["TorchPruningPass"]["module_path"].endswith(".TorchPruningPass")
    assert cfg["FineTunePass"]["module_path"].endswith(".FineTunePass")
    assert "chisel.passes" in cfg["TorchPruningPass"]["module_path"]
