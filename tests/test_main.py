import sys

import pytest

from chisel import main


def test_main_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chisel"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert "chisel" in capsys.readouterr().out.lower()
