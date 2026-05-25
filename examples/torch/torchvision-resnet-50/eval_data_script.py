"""Olive data config user script for ImageNet-1k validation evaluation.

Registers two components with Olive's data registry:

  imagenet_val_dataset   -- load_dataset component: streams num_samples examples
                            from the ILSVRC/imagenet-1k validation split, applies
                            standard torchvision val transforms, and returns a list
                            of (tensor, label) pairs that PyTorch's default collate_fn
                            turns into (batch_tensor, label_tensor) batches.

  imagenet_post_process  -- post_process component: converts raw logits [B, C] output
                            from the model into predicted class indices [B] via argmax,
                            which is what Olive's HuggingFace accuracy backend expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from olive.data.registry import Registry as DataRegistry
from torchvision import transforms

if TYPE_CHECKING:
    import torch

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


@DataRegistry.register_dataset(name="imagenet_val_dataset")
def load_imagenet_val(num_samples: int = 200, **kwargs: Any) -> list[tuple[torch.Tensor, int]]:
    """Stream num_samples validation examples and return as a list of (tensor, label)."""
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    data = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        tensor = _VAL_TRANSFORM(sample["image"].convert("RGB"))
        data.append((tensor, sample["label"]))
    return data


@DataRegistry.register_post_process(name="imagenet_post_process")
def imagenet_post_process(output: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Convert logits [B, C] → predicted class indices [B]."""
    return output.argmax(dim=-1)
