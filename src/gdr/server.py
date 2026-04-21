from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.gdr.features import ClientFeaturePayload


@dataclass
class GDRResult:
    records: list[dict[str, Any]]
    singular_values: list[float]
    rank: int


def compute_leverage_scores(
    payloads: list[ClientFeaturePayload],
    rank: int = 8,
    eps: float = 1e-12,
) -> GDRResult:
    valid_payloads = [payload for payload in payloads if len(payload.labels) > 0]
    if not valid_payloads:
        return GDRResult(records=[], singular_values=[], rank=0)

    feature_matrix = torch.cat([payload.features for payload in valid_payloads], dim=0)
    feature_matrix = feature_matrix.float()
    feature_matrix = feature_matrix - feature_matrix.mean(dim=0, keepdim=True)

    max_rank = min(feature_matrix.shape)
    actual_rank = min(rank, max_rank)
    if actual_rank <= 0:
        return GDRResult(records=[], singular_values=[], rank=0)

    u, singular_values, _ = torch.linalg.svd(feature_matrix, full_matrices=False)
    raw_scores = (u[:, :actual_rank] ** 2).sum(dim=1)
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    normalized = (raw_scores - min_score) / (max_score - min_score + eps)

    records = []
    offset = 0
    for payload in valid_payloads:
        for local_pos, label in enumerate(payload.labels):
            global_pos = offset + local_pos
            records.append(
                {
                    "client_id": payload.client_id,
                    "task_id": payload.task_id,
                    "dataset_index": payload.dataset_indices[local_pos],
                    "label": label,
                    "leverage_score": float(normalized[global_pos].item()),
                    "raw_leverage_score": float(raw_scores[global_pos].item()),
                }
            )
        offset += len(payload.labels)

    return GDRResult(
        records=records,
        singular_values=[float(value.item()) for value in singular_values],
        rank=actual_rank,
    )


def group_records_by_client(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["client_id"]), []).append(record)
    return grouped
