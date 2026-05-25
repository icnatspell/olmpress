"""User script for FineTunePass — ResNet50 structured-pruning recovery.

Implements knowledge-distillation fine-tuning (or plain cross-entropy if no
teacher_model is specified) for a pruned torchvision ResNet50 TorchScript model.

config keys (all optional):
    train_samples   int     samples to stream from ImageNet train  (default: 5000)
    eval_samples    int     samples to stream from ImageNet val     (default: 200)
    batch_size      int                                             (default: 32)
    epochs          int                                             (default: 5)
    lr              float   SGD learning rate (cosine-annealed)     (default: 1e-3)
    teacher_model   str     torchvision weights name for KD teacher (default: "IMAGENET1K_V2")
                            set to null/None to use plain CE loss
    temperature     float   KD softmax temperature                  (default: 4.0)
    alpha           float   CE weight in a*CE + (1-a)*t^2*KL       (default: 0.5)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


class _StreamingSubset(IterableDataset):
    def __init__(self, hf_ds, transform, num_samples: int) -> None:
        self._ds = hf_ds
        self._transform = transform
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Generator[tuple[torch.Tensor, int]]:
        for i, sample in enumerate(self._ds):
            if i >= self._num_samples:
                break
            yield self._transform(sample["image"].convert("RGB")), sample["label"]


def finetune(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    """Fine-tune a pruned TorchScript ResNet50.

    Called by FineTunePass with the loaded model and the finetune_config dict
    from the workflow YAML.  Returns the fine-tuned model.
    """
    train_samples = config.get("train_samples", 5000)
    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 32)
    lr = config.get("lr", 1e-3)
    teacher_name = config.get("teacher_model", "IMAGENET1K_V2")
    temperature = config.get("temperature", 4.0)
    alpha = config.get("alpha", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.train().to(device)

    use_kd = teacher_name is not None
    teacher = None
    if use_kd:
        weights = ResNet50_Weights[teacher_name]
        teacher = resnet50(weights=weights).eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(requires_grad=False)

    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    loader = DataLoader(
        _StreamingSubset(ds, _TRAIN_TRANSFORM, train_samples),
        batch_size=batch_size,
        num_workers=0,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        t0 = time.time()
        for _imgs, _labels in loader:
            imgs, labels = _imgs.to(device), _labels.to(device)
            logits = model(imgs)
            if use_kd:
                with torch.no_grad():
                    teacher_logits = teacher(imgs)
                loss = alpha * ce(logits, labels) + (1 - alpha) * (temperature**2) * kl(
                    torch.log_softmax(logits / temperature, dim=-1),
                    torch.softmax(teacher_logits / temperature, dim=-1),
                )
            else:
                loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        mode = f"KD (teacher={teacher_name})" if use_kd else "plain CE"
        print(
            f"epoch {epoch + 1}/{epochs}  loss={avg:.4f}  mode={mode}  "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    return model
