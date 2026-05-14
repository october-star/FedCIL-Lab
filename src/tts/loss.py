from __future__ import annotations

import torch
import torch.nn.functional as F


def split_logits(logits: torch.Tensor, old_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if old_classes <= 0:
        return logits[:, :0], logits
    if old_classes >= logits.size(1):
        return logits, logits[:, 0:0]
    return logits[:, :old_classes], logits[:, old_classes:]


def apply_temperature_split(
    logits: torch.Tensor,
    old_classes: int,
    old_temp: float,
    new_temp: float,
) -> torch.Tensor:
    old_logits, new_logits = split_logits(logits, old_classes)
    chunks = []
    if old_logits.numel() > 0:
        chunks.append(old_logits / old_temp)
    if new_logits.numel() > 0:
        chunks.append(new_logits / new_temp)
    return torch.cat(chunks, dim=1) if chunks else logits


def _weighted_group_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weights is None:
        return values.mean()
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def tts_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_classes: int,
    old_temp: float = 2.0,
    new_temp: float = 1.0,
    old_weight: float = 1.0,
    new_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    scaled_logits = apply_temperature_split(
        logits=logits,
        old_classes=old_classes,
        old_temp=old_temp,
        new_temp=new_temp,
    )
    per_sample = F.cross_entropy(scaled_logits, targets, reduction="none")

    external_weights = (
        sample_weights.to(per_sample.dtype) if sample_weights is not None else None
    )
    if old_classes <= 0 or old_classes >= logits.size(1):
        return _weighted_group_mean(per_sample, external_weights)

    old_mask = targets < old_classes
    new_mask = ~old_mask
    loss = per_sample.new_tensor(0.0)

    if old_mask.any():
        old_losses = per_sample[old_mask]
        old_sample_weights = (
            external_weights[old_mask] if external_weights is not None else None
        )
        loss = loss + old_weight * _weighted_group_mean(old_losses, old_sample_weights)

    if new_mask.any():
        new_losses = per_sample[new_mask]
        new_sample_weights = (
            external_weights[new_mask] if external_weights is not None else None
        )
        loss = loss + new_weight * _weighted_group_mean(new_losses, new_sample_weights)

    return loss
