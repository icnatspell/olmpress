import sys

import pytest

from olmpress import main


def test_main_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["olmpress"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert "olmpress" in capsys.readouterr().out.lower()
