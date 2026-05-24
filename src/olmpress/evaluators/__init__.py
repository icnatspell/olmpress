"""Custom Olive evaluators."""

from olmpress.evaluators.degradation import DegradationEvaluator
from olmpress.evaluators.topk_accuracy import ImageNetAccuracyEvaluator

__all__ = ["DegradationEvaluator", "ImageNetAccuracyEvaluator"]
