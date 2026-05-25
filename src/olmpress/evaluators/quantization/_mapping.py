"""Layer-name mapping between two PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn


View = Literal["all", "linears", "blocks", "logits"]


@dataclass(frozen=True)
class MappingDiff:
    """Symmetric difference between two models' module-name sets."""

    only_in_reference: tuple[str, ...]
    only_in_target: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """True when both models share the same module names."""
        return not self.only_in_reference and not self.only_in_target


class MappingError(ValueError):
    """Raised when reference modules cannot be matched to a target."""


def _module_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_modules() if name]


def diff(reference: nn.Module, target: nn.Module) -> MappingDiff:
    """Return the symmetric difference of module names between two models."""
    ref = set(_module_names(reference))
    tgt = set(_module_names(target))
    return MappingDiff(
        only_in_reference=tuple(sorted(ref - tgt)),
        only_in_target=tuple(sorted(tgt - ref)),
    )


def build_mapping(
    reference: nn.Module,
    target: nn.Module,
    *,
    rename: dict[str, str] | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Return ``{reference_name: target_name}`` over named modules."""
    rename = rename or {}
    target_names = set(_module_names(target))
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for ref_name in _module_names(reference):
        tgt_name = rename.get(ref_name, ref_name)
        if tgt_name in target_names:
            mapping[ref_name] = tgt_name
        else:
            missing.append(ref_name)

    if strict and missing:
        max_preview = 5
        preview = ", ".join(missing[:max_preview])
        more = f" (+{len(missing) - max_preview} more)" if len(missing) > max_preview else ""
        msg = (
            f"{len(missing)} reference module(s) have no target counterpart: "
            f"{preview}{more}. Pass rename={{...}} to align them or strict=False to skip."
        )
        raise MappingError(msg)

    return mapping


_WEIGHT_BEARING_SUFFIXES: tuple[str, ...] = (
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
)


def _is_weight_bearing(module: nn.Module) -> bool:
    return type(module).__name__ in _WEIGHT_BEARING_SUFFIXES


def _is_block(name: str) -> bool:
    parts = name.split(".")
    if len(parts) < 2:  # noqa: PLR2004
        return False
    return parts[-2] in ("layers", "h") and parts[-1].isdigit()


def select_view(
    mapping: dict[str, str],
    reference: nn.Module,
    view: View,
) -> dict[str, str]:
    """Filter ``mapping`` down to the subset described by ``view``."""
    if view == "all":
        return dict(mapping)
    if view == "logits":
        return {}
    if view == "blocks":
        return {ref: tgt for ref, tgt in mapping.items() if _is_block(ref)}
    if view == "linears":
        ref_modules = dict(reference.named_modules())
        return {
            ref: tgt
            for ref, tgt in mapping.items()
            if ref in ref_modules and _is_weight_bearing(ref_modules[ref])
        }
    msg = f"Unknown view: {view!r}"
    raise ValueError(msg)


def group_by_block(names: Iterable[str]) -> dict[str, list[str]]:
    """Group module names by the transformer block that contains them."""
    groups: dict[str, list[str]] = {}
    for name in names:
        parts = name.split(".")
        block = ""
        for i in range(len(parts) - 1):
            if parts[i] in ("layers", "h") and parts[i + 1].isdigit():
                block = ".".join(parts[: i + 2])
                break
        groups.setdefault(block, []).append(name)
    return groups
