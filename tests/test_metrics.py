"""Tests for olmpress.metrics."""

from __future__ import annotations

import math

import pytest
import torch

from olmpress.metrics import cosine_similarity, kl_divergence, mse, relative_l2, sqnr

# ---- SQNR ----------------------------------------------------------------


def test_sqnr_identical_inputs_is_very_high():
    x = torch.randn(64)
    result = sqnr(x, x.clone())
    assert result.item() > 100


def test_sqnr_decreases_with_noise():
    torch.manual_seed(0)
    x = torch.randn(1024)
    small = sqnr(x, x + 0.01 * torch.randn_like(x))
    large = sqnr(x, x + 0.5 * torch.randn_like(x))
    assert small > large


def test_sqnr_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        sqnr(torch.zeros(3), torch.zeros(4))


# ---- MSE / relative L2 ---------------------------------------------------


def test_mse_zero_for_identical():
    x = torch.randn(8, 8)
    assert mse(x, x.clone()).item() == pytest.approx(0.0)


def test_mse_value():
    x = torch.zeros(4)
    y = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert mse(x, y).item() == pytest.approx(1.0)


def test_relative_l2_zero_for_identical():
    x = torch.randn(8, 8)
    assert relative_l2(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mse(torch.zeros(3), torch.zeros(4))


def test_relative_l2_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        relative_l2(torch.zeros(3), torch.zeros(4))


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity(torch.zeros(3), torch.zeros(4))


def test_relative_l2_scale_invariant():
    torch.manual_seed(0)
    x = torch.randn(64)
    err = torch.randn(64) * 0.1
    a = relative_l2(x, x + err)
    b = relative_l2(10 * x, 10 * (x + err))
    assert a.item() == pytest.approx(b.item(), rel=1e-5)


# ---- Cosine --------------------------------------------------------------


def test_cosine_identical_is_one():
    x = torch.randn(32)
    assert cosine_similarity(x, x.clone()).item() == pytest.approx(1.0, abs=1e-5)


def test_cosine_opposite_is_minus_one():
    x = torch.randn(32)
    assert cosine_similarity(x, -x).item() == pytest.approx(-1.0, abs=1e-5)


def test_cosine_orthogonal_is_zero():
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([0.0, 1.0])
    assert cosine_similarity(x, y).item() == pytest.approx(0.0, abs=1e-6)


# ---- KL divergence -------------------------------------------------------


def test_kl_identical_logits_is_zero():
    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    assert kl_divergence(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_nonzero_for_different_logits():
    torch.manual_seed(0)
    a = torch.randn(4, 16)
    b = torch.randn(4, 16)
    assert kl_divergence(a, b).item() > 0.0


def test_kl_increases_with_divergence():
    torch.manual_seed(0)
    ref = torch.randn(8, 32)
    small = kl_divergence(ref, ref + 0.05 * torch.randn_like(ref))
    large = kl_divergence(ref, ref + 1.0 * torch.randn_like(ref))
    assert large.item() > small.item()


def test_kl_temperature_softens_distribution():
    """Higher temperature reduces KL between sharply different distributions."""
    torch.manual_seed(0)
    a = torch.randn(8, 32) * 5  # sharpened
    b = torch.randn(8, 32) * 5
    cold = kl_divergence(a, b, temperature=1.0)
    hot = kl_divergence(a, b, temperature=4.0)
    assert hot.item() < cold.item()


def test_kl_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        kl_divergence(torch.zeros(2, 3), torch.zeros(2, 4))


def test_kl_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        kl_divergence(torch.zeros(2, 3), torch.zeros(2, 3), temperature=0.0)


def test_kl_two_point_distribution_matches_closed_form():
    """For 2-class distributions p=(.9,.1), q=(.5,.5): KL = 0.9*log(1.8) + 0.1*log(0.2)."""
    p = torch.tensor([[math.log(0.9), math.log(0.1)]])
    q = torch.tensor([[math.log(0.5), math.log(0.5)]])
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    result = kl_divergence(p, q, reduction="batchmean")
    assert result.item() == pytest.approx(expected, abs=1e-5)
