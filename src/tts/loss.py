from __future__ import annotations

import torch
import torch.nn.functional as F

def _as_class_tensor(
    class_ids: list[int] | None,
    *,
    device,
) -> torch.Tensor:
    if not class_ids:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.tensor(class_ids, device=device, dtype=torch.long)

def apply_temperature_split(
    logits: torch.Tensor,
    old_class_ids: list[int] | None,
    new_class_ids: list[int] | None,
    old_temp: float,
    new_temp: float,
) -> torch.Tensor:
    scaled_logits = logits.clone()
    old_idx = _as_class_tensor(old_class_ids, device=logits.device)
    new_idx = _as_class_tensor(new_class_ids, device=logits.device)

    if old_idx.numel() > 0:
        scaled_logits[:, old_idx] = scaled_logits[:, old_idx] / old_temp

    if new_idx.numel() > 0:
        scaled_logits[:, new_idx] = scaled_logits[:, new_idx] / new_temp

    return scaled_logits


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
    old_class_ids: list[int] | None = None,
    new_class_ids: list[int] | None = None,
    old_temp: float = 2.0,
    new_temp: float = 1.0,
    old_weight: float = 1.0,
    new_weight: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    old_class_ids = old_class_ids or []
    new_class_ids = new_class_ids or []

    scaled_logits = apply_temperature_split(
        logits=logits,
        old_class_ids=old_class_ids,
        new_class_ids=new_class_ids,
        old_temp=old_temp,
        new_temp=new_temp,
    )

    losses = F.cross_entropy(scaled_logits, targets, reduction="none")
    if not old_class_ids or not new_class_ids:
        return _weighted_group_mean(losses, sample_weights)

    old_idx = _as_class_tensor(
        old_class_ids,
        device=targets.device,
    ).to(targets.dtype)

    is_old_sample = torch.isin(targets, old_idx)

    total_loss = losses.new_tensor(0.0)

    if is_old_sample.any():
        old_sample_weights = None
        if sample_weights is not None:
            old_sample_weights = sample_weights[is_old_sample]
        total_loss = total_loss + old_weight * _weighted_group_mean(
            losses[is_old_sample],
            old_sample_weights,
        )

    is_new_sample = ~is_old_sample
    if is_new_sample.any():
        new_sample_weights = None
        if sample_weights is not None:
            new_sample_weights = sample_weights[is_new_sample]
        total_loss = total_loss + new_weight * _weighted_group_mean(
            losses[is_new_sample],
            new_sample_weights,
        )
    return total_loss
