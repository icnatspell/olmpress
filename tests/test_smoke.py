"""Smoke test: package imports cleanly and exposes its subpackages."""

import chisel
import chisel.evaluators
import chisel.passes


def test_chisel_imports():
    assert hasattr(chisel, "main")


def test_subpackages_present():
    assert chisel.evaluators is not None
    assert chisel.passes is not None
