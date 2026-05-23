# Author: mmj
# DATE: 22.05.2026
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_weighted_tts_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    old_classes: int,
    old_temp: float,
    new_temp: float,
    old_weight: float,
    new_weight: float,
    class_weights: dict[int, float] | None = None,
) -> torch.Tensor:
    old_logits = logits[:, :old_classes] / old_temp
    new_logits = logits[:, old_classes:] / new_temp
    scaled_logits = torch.cat([old_logits, new_logits], dim=1)

    is_old = labels < old_classes

    sample_temp = torch.where(
        is_old,
        torch.full_like(labels, old_temp, dtype=logits.dtype),
        torch.full_like(labels, new_temp, dtype=logits.dtype),
    ).view(-1, 1)

    scaled_logits = scaled_logits / sample_temp

    losses = F.cross_entropy(scaled_logits, labels, reduction="none")

    task_weights = torch.where(
        is_old,
        torch.full_like(labels, old_weight, dtype=logits.dtype),
        torch.full_like(labels, new_weight, dtype=logits.dtype),
    )

    if class_weights:
        cls_weights = torch.tensor(
            [class_weights.get(int(y.item()), 1.0) for y in labels],
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        cls_weights = torch.ones_like(losses)

    return (losses * task_weights * cls_weights).mean()