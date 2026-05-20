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
                    "label": int(label),
                    "raw_leverage_score": float(raw_scores[global_pos].item()),
                }
            )
        offset += len(payload.labels)

    # class-wise normalization
    by_class = {}
    for record in records:
        by_class.setdefault(int(record["label"]), []).append(record)

    for class_records in by_class.values():
        total = sum(r["raw_leverage_score"] for r in class_records) + eps
        for r in class_records:
            r["leverage_score"] = r["raw_leverage_score"] / total

    return GDRResult(
        records=records,
        singular_values=[float(value.item()) for value in singular_values],
        rank=actual_rank,
    )

import random
from collections import defaultdict
from typing import Any


def select_global_class_balanced_records(
    records: list[dict[str, Any]],
    total_budget: int,
    seed: int = 0,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    Global class-balanced sampling with leverage-score probabilities.

    This is closer to the paper/original implementation than greedy top-k:
    - first allocate quota per class
    - then sample within each class with probability proportional to leverage score
    """
    if total_budget <= 0 or not records:
        return []

    rng = random.Random(seed)

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[int(record["label"])].append(record)

    classes = sorted(by_class.keys())
    num_classes = len(classes)

    base_quota = total_budget // num_classes
    remainder = total_budget % num_classes

    selected: list[dict[str, Any]] = []

    for i, cls in enumerate(classes):
        class_records = by_class[cls]
        quota = base_quota + (1 if i < remainder else 0)
        quota = min(quota, len(class_records))

        if quota <= 0:
            continue

        weights = [
            max(float(r.get("leverage_score", 0.0)), eps)
            for r in class_records
        ]

        # weighted sampling without replacement
        chosen = []
        candidates = list(class_records)
        candidate_weights = list(weights)

        for _ in range(quota):
            total_w = sum(candidate_weights)
            probs = [w / total_w for w in candidate_weights]

            idx = rng.choices(
                range(len(candidates)),
                weights=probs,
                k=1,
            )[0]

            chosen.append(candidates.pop(idx))
            candidate_weights.pop(idx)

            if not candidates:
                break

        selected.extend(chosen)

    # If some classes had too few samples, fill the remaining budget globally
    if len(selected) < total_budget:
        selected_keys = {
            (
                int(r["client_id"]),
                int(r["task_id"]),
                int(r["dataset_index"]),
                int(r["label"]),
            )
            for r in selected
        }

        remaining = [
            r for r in records
            if (
                int(r["client_id"]),
                int(r["task_id"]),
                int(r["dataset_index"]),
                int(r["label"]),
            )
            not in selected_keys
        ]

        while remaining and len(selected) < total_budget:
            weights = [
                max(float(r.get("leverage_score", 0.0)), eps)
                for r in remaining
            ]
            idx = rng.choices(
                range(len(remaining)),
                weights=weights,
                k=1,
            )[0]
            selected.append(remaining.pop(idx))

    return selected

def group_records_by_client(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["client_id"]), []).append(record)
    return grouped
