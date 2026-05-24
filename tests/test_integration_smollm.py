"""Integration tests for DegradationEvaluator using HuggingFaceTB/SmolLM2-135M-Instruct.

Downloads the model on first run (~270 MB) and caches it via HuggingFace Hub.
Run selectively with: pytest tests/test_integration_smollm.py
Skip in fast CI with: pytest -m "not integration"
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from olive.evaluator.metric import Metric, MetricType
from transformers import AutoModelForCausalLM

from olmpress.evaluators.degradation import DegradationEvaluator

pytestmark = pytest.mark.integration

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
VOCAB_SIZE = 49152
SEQ_LEN = 16


# ---------------------------------------------------------------------------
# Helpers


def _val(res, name: str, sub: str) -> float:
    v = res.get_value(name, sub)
    assert v is not None, f"missing metric {name!r}/{sub!r}"
    return float(v)


def _make_metric(sub_type_names: list[str]) -> Metric:
    return Metric(
        name="degradation",
        type=MetricType.CUSTOM,
        sub_types=[{"name": n} for n in sub_type_names],
    )


def _perturb(model: torch.nn.Module, noise: float, seed: int = 0) -> torch.nn.Module:
    """Return a deep copy of *model* with Gaussian noise added to all parameters."""
    noisy = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in noisy.parameters():
            p.add_(noise * torch.randn(p.shape, generator=gen))
    return noisy


# ---------------------------------------------------------------------------
# Fixtures (loaded once per module)


@pytest.fixture(scope="module")
def smollm() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def smollm_inputs() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(42)
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.long, generator=gen)
    }


# ---------------------------------------------------------------------------
# Tests


def test_identical_smollm_perfect_scores(smollm, smollm_inputs):
    """Comparing the model to itself must yield near-perfect per-layer scores."""
    ev = DegradationEvaluator(
        reference_model=lambda: smollm,
        inputs=lambda: smollm_inputs,
    )
    result = ev.evaluate(smollm, [_make_metric(["sqnr_mean", "cosine_mean", "mse_mean", "kl"])])

    assert _val(result, "degradation", "sqnr_mean") > 100
    assert _val(result, "degradation", "cosine_mean") == pytest.approx(1.0, abs=1e-4)
    assert _val(result, "degradation", "mse_mean") == pytest.approx(0.0, abs=1e-8)
    assert _val(result, "degradation", "kl") == pytest.approx(0.0, abs=1e-6)


def test_perturbed_smollm_shows_degradation(smollm, smollm_inputs):
    """Noise-perturbed weights must produce measurable, finite degradation on all metrics."""
    noisy = _perturb(smollm, noise=0.05)
    ev = DegradationEvaluator(
        reference_model=lambda: smollm,
        inputs=lambda: smollm_inputs,
    )
    result = ev.evaluate(noisy, [_make_metric(["sqnr_mean", "mse_mean", "relative_l2_mean", "kl"])])

    assert _val(result, "degradation", "sqnr_mean") < 100
    for sub in ("mse_mean", "relative_l2_mean", "kl"):
        v = _val(result, "degradation", sub)
        assert math.isfinite(v), f"{sub} expected finite, got {v}"
        assert v > 0, f"{sub} expected positive, got {v}"


def test_higher_noise_yields_worse_scores(smollm, smollm_inputs):
    """Larger weight perturbation must monotonically worsen all reported metrics."""
    ev = DegradationEvaluator(
        reference_model=lambda: smollm,
        inputs=lambda: smollm_inputs,
    )
    metric = _make_metric(["sqnr_mean", "mse_mean", "relative_l2_mean", "kl"])
    low = ev.evaluate(_perturb(smollm, noise=0.01, seed=1), [metric])
    high = ev.evaluate(_perturb(smollm, noise=0.1, seed=1), [metric])

    assert _val(low, "degradation", "sqnr_mean") > _val(high, "degradation", "sqnr_mean")
    assert _val(low, "degradation", "mse_mean") < _val(high, "degradation", "mse_mean")
    assert _val(low, "degradation", "relative_l2_mean") < _val(
        high, "degradation", "relative_l2_mean"
    )
    assert _val(low, "degradation", "kl") < _val(high, "degradation", "kl")


def test_view_linears_produces_finite_ordered_scores(smollm, smollm_inputs):
    """view='linears' must limit capture to Linear layers and produce valid, ordered scores."""
    noisy = _perturb(smollm, noise=0.05)
    ev = DegradationEvaluator(
        reference_model=lambda: smollm,
        inputs=lambda: smollm_inputs,
        view="linears",
    )
    result = ev.evaluate(
        noisy, [_make_metric(["sqnr_min", "sqnr_mean", "sqnr_max", "cosine_min", "mse_max"])]
    )

    for sub in ("sqnr_min", "sqnr_mean", "sqnr_max", "cosine_min", "mse_max"):
        v = _val(result, "degradation", sub)
        assert math.isfinite(v), f"{sub} is not finite: {v}"

    # Aggregation ordering invariants
    assert (
        _val(result, "degradation", "sqnr_min")
        <= _val(result, "degradation", "sqnr_mean")
        <= _val(result, "degradation", "sqnr_max")
    )
