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


# def tts_cross_entropy(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     old_classes: int,
#     old_temp: float = 2.0,
#     new_temp: float = 1.0,
#     old_weight: float = 1.0,
#     new_weight: float = 1.0,
#     sample_weights: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     scaled_logits = apply_temperature_split(
#         logits=logits,
#         old_classes=old_classes,
#         old_temp=old_temp,
#         new_temp=new_temp,
#     )

#     # Match the original B.py implementation: after scaling the old/new logits
#     # columns, apply a second sample-wise temperature to the full row.
#     if 0 < old_classes < logits.size(1):
#         is_old_sample = targets < old_classes
#         sample_temps = torch.where(
#             is_old_sample,
#             torch.full_like(targets, old_temp, dtype=scaled_logits.dtype),
#             torch.full_like(targets, new_temp, dtype=scaled_logits.dtype),
#         ).unsqueeze(1)
#         scaled_logits = scaled_logits / sample_temps
#         task_weights = torch.where(
#             is_old_sample,
#             torch.full_like(targets, old_weight, dtype=scaled_logits.dtype),
#             torch.full_like(targets, new_weight, dtype=scaled_logits.dtype),
#         )
#     else:
#         task_weights = torch.ones_like(targets, dtype=scaled_logits.dtype)

#     losses = F.cross_entropy(scaled_logits, targets, reduction="none")
#     if sample_weights is not None:
#         task_weights = task_weights * sample_weights.to(task_weights.dtype)
#     return (losses * task_weights).mean()
def tts_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_classes: int,
    old_temp: float = 0.9,
    new_temp: float = 1.1,
    old_weight: float = 1.1,
    new_weight: float = 0.9,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    
    scaled_logits = apply_temperature_split(
        logits=logits,
        old_classes=old_classes,
        old_temp=old_temp,
        new_temp=new_temp,
    )

    
    if 0 < old_classes < logits.size(1):
        is_old_sample = targets < old_classes
        task_weights = torch.where(
            is_old_sample,
            torch.full_like(targets, old_weight, dtype=scaled_logits.dtype),
            torch.full_like(targets, new_weight, dtype=scaled_logits.dtype),
        )
    else:
        task_weights = torch.ones_like(targets, dtype=scaled_logits.dtype)

    losses = F.cross_entropy(scaled_logits, targets, reduction="none")
    if sample_weights is not None:
        task_weights = task_weights * sample_weights.to(task_weights.dtype)
    return (losses * task_weights).mean()