"""Smoke test: package imports cleanly and exposes its subpackages."""

import olmpress
import olmpress.evaluators
import olmpress.passes


def test_olmpress_imports():
    assert hasattr(olmpress, "main")


def test_subpackages_present():
    assert olmpress.evaluators is not None
    assert olmpress.passes is not None
