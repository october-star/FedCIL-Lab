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


def tts_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_classes: int,
    old_temp: float = 0.9,
    new_temp: float = 1.1,
    old_weight: float = 1.2,
    new_weight: float = 0.9,
) -> torch.Tensor:
    scaled_logits = apply_temperature_split(
        logits=logits,
        old_classes=old_classes,
        old_temp=old_temp,
        new_temp=new_temp,
    )
    per_sample = F.cross_entropy(scaled_logits, targets, reduction="none")

    if old_classes <= 0 or old_classes >= logits.size(1):
        return per_sample.mean()

    sample_weights = torch.full_like(per_sample, fill_value=new_weight, dtype=per_sample.dtype)
    old_mask = targets < old_classes
    sample_weights = torch.where(
        old_mask,
        torch.full_like(sample_weights, fill_value=old_weight),
        sample_weights,
    )
    # return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
    return (per_sample * sample_weights).mean()