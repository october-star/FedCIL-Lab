from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class ClientFeaturePayload:
    client_id: int
    task_id: int
    features: torch.Tensor
    labels: list[int]
    dataset_indices: list[int]


def construct_pseudo_feature(x: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
    """
    Build a lightweight, model-free pseudo feature from an image tensor.

    The feature mixes channel statistics and coarse spatial averages. This keeps
    GDR independent from the classifier while still exposing sample diversity.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(x.shape)}")

    x = x.detach().float().cpu()
    flat = x.flatten(start_dim=1)
    channel_mean = flat.mean(dim=1)
    channel_std = flat.std(dim=1, unbiased=False)
    channel_abs_mean = flat.abs().mean(dim=1)
    pooled = F.adaptive_avg_pool2d(x.unsqueeze(0), (grid_size, grid_size)).flatten()
    return torch.cat([channel_mean, channel_std, channel_abs_mean, pooled])


def build_client_feature_payload(
    dataset: Dataset,
    client_id: int,
    task_id: int,
    max_samples: int | None = None,
    seed: int = 0,
    grid_size: int = 4,
) -> ClientFeaturePayload:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]

    features = []
    labels = []
    for dataset_index in indices:
        x, y = dataset[dataset_index]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        features.append(construct_pseudo_feature(x, grid_size=grid_size))
        labels.append(int(y))

    if features:
        feature_matrix = torch.stack(features)
    else:
        feature_matrix = torch.empty(0, 0)

    return ClientFeaturePayload(
        client_id=client_id,
        task_id=task_id,
        features=feature_matrix,
        labels=labels,
        dataset_indices=indices,
    )
