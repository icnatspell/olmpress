"""Tests for olmpress.mapping."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from olmpress.mapping import (
    MappingDiff,
    MappingError,
    build_mapping,
    diff,
    group_by_block,
    select_view,
)


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)


class TinyModel(nn.Module):
    def __init__(self, dim: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(16, dim)
        self.layers = nn.ModuleList(TinyBlock(dim) for _ in range(n_layers))
        self.lm_head = nn.Linear(dim, 16)


def _identical_pair():
    torch.manual_seed(0)
    return TinyModel(), TinyModel()


def test_build_mapping_identical_topology():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    assert mapping["lm_head"] == "lm_head"
    assert mapping["layers.1.attn"] == "layers.1.attn"
    assert len(mapping) == len([n for n, _ in ref.named_modules() if n])


def test_diff_identical_is_empty():
    ref, tgt = _identical_pair()
    d = diff(ref, tgt)
    assert isinstance(d, MappingDiff)
    assert d.is_empty


def test_diff_reports_asymmetry():
    ref = TinyModel(n_layers=3)
    tgt = TinyModel(n_layers=2)
    d = diff(ref, tgt)
    assert not d.is_empty
    assert any("layers.2" in n for n in d.only_in_reference)
    assert d.only_in_target == ()


def test_strict_mismatch_raises():
    ref = TinyModel(n_layers=3)
    tgt = TinyModel(n_layers=2)
    with pytest.raises(MappingError, match="no target counterpart"):
        build_mapping(ref, tgt)


def test_non_strict_skips_missing():
    ref = TinyModel(n_layers=3)
    tgt = TinyModel(n_layers=2)
    mapping = build_mapping(ref, tgt, strict=False)
    assert "layers.2.attn" not in mapping
    assert "layers.1.attn" in mapping


def test_rename_override():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt, rename={"lm_head": "lm_head"})
    assert mapping["lm_head"] == "lm_head"


def test_view_all_returns_full_mapping():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    assert select_view(mapping, ref, "all") == mapping


def test_view_linears_keeps_only_linear():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    linears = select_view(mapping, ref, "linears")
    assert "layers.0.attn" in linears
    assert "layers.0.norm" not in linears
    assert "embed" not in linears


def test_view_blocks_keeps_only_block_roots():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    blocks = select_view(mapping, ref, "blocks")
    assert set(blocks) == {"layers.0", "layers.1", "layers.2"}


def test_view_logits_is_empty():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    assert select_view(mapping, ref, "logits") == {}


def test_view_unknown_raises():
    ref, tgt = _identical_pair()
    mapping = build_mapping(ref, tgt)
    with pytest.raises(ValueError, match="Unknown view"):
        select_view(mapping, ref, "bogus")  # pyrefly: ignore  # type: ignore[arg-type]


def test_group_by_block():
    names = [
        "embed",
        "layers.0.attn",
        "layers.0.mlp",
        "layers.1.attn",
        "lm_head",
    ]
    groups = group_by_block(names)
    assert groups["layers.0"] == ["layers.0.attn", "layers.0.mlp"]
    assert groups["layers.1"] == ["layers.1.attn"]
    assert set(groups[""]) == {"embed", "lm_head"}
