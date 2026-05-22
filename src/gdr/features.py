from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.sample_ids import resolve_sample_id


@dataclass
class ClientFeaturePayload:
    client_id: int
    task_id: int
    features: torch.Tensor
    labels: list[int]
    dataset_indices: list[int]
    sample_ids: list[int]


def sample_orthogonal_matrix(
    dim: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim <= 1:
        return torch.eye(max(dim, 1), dtype=dtype)[:dim, :dim]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn(dim, dim, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def build_client_feature_payload(
    dataset: Dataset,
    client_id: int,
    task_id: int,
    feature_extractor: nn.Module,
    device: torch.device | str,
    max_samples: int | None = None,
    seed: int = 0,
    batch_size: int = 128,
    right_transform: torch.Tensor | None = None,
    left_transform_seed: int | None = None,
) -> ClientFeaturePayload:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]

    labels = [int(dataset[dataset_index][1]) for dataset_index in indices]
    sample_ids = [resolve_sample_id(dataset, dataset_index) for dataset_index in indices]

    if not indices:
        return ClientFeaturePayload(
            client_id=client_id,
            task_id=task_id,
            features=torch.empty(0, 0),
            labels=labels,
            dataset_indices=indices,
            sample_ids=sample_ids,
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
        for batch in loader:
            x = batch[0]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            x = x.to(feature_extractor_device, non_blocking=True)
            feats = feature_extractor(x)
            feature_batches.append(feats.detach().cpu())

    if was_training:
        feature_extractor.train()

    feature_matrix = torch.cat(feature_batches, dim=0)
    if right_transform is not None:
        feature_matrix = feature_matrix @ right_transform.to(dtype=feature_matrix.dtype)

    if left_transform_seed is not None:
        left_transform = sample_orthogonal_matrix(
            feature_matrix.size(0),
            seed=left_transform_seed,
            dtype=feature_matrix.dtype,
        )
        feature_matrix = left_transform @ feature_matrix

    return ClientFeaturePayload(
        client_id=client_id,
        task_id=task_id,
        features=feature_matrix,
        labels=labels,
        dataset_indices=indices,
        sample_ids=sample_ids,
    )
