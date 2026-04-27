from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class ClientFeaturePayload:
    client_id: int
    task_id: int
    features: torch.Tensor
    labels: list[int]
    dataset_indices: list[int]


def build_client_feature_payload(
    dataset: Dataset,
    client_id: int,
    task_id: int,
    feature_extractor: nn.Module,
    device: torch.device | str,
    max_samples: int | None = None,
    seed: int = 0,
    batch_size: int = 128,
) -> ClientFeaturePayload:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]

    labels = [int(dataset[dataset_index][1]) for dataset_index in indices]

    if not indices:
        return ClientFeaturePayload(
            client_id=client_id,
            task_id=task_id,
            features=torch.empty(0, 0),
            labels=labels,
            dataset_indices=indices,
        )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_extractor_device = torch.device(device)
    was_training = feature_extractor.training
    feature_extractor.eval()

    feature_batches = []
    with torch.no_grad():
        for x, _ in loader:
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            x = x.to(feature_extractor_device, non_blocking=True)
            feats = feature_extractor(x)
            feature_batches.append(feats.detach().cpu())

    if was_training:
        feature_extractor.train()

    feature_matrix = torch.cat(feature_batches, dim=0)

    return ClientFeaturePayload(
        client_id=client_id,
        task_id=task_id,
        features=feature_matrix,
        labels=labels,
        dataset_indices=indices,
    )
