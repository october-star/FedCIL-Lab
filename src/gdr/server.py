from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.gdr.features import ClientFeaturePayload


@dataclass
class GDRResult:
    records: list[dict[str, Any]]
    singular_values: list[float]
    rank: int
    class_wise: bool = False
    class_singular_values: dict[int, list[float]] | None = None


@dataclass
class ClientLocalGDRResult:
    records: list[dict[str, Any]]
    client_singular_values: dict[int, list[float]]
    rank: int


def _compute_group_scores(
    feature_matrix: torch.Tensor,
    rank: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], int]:
    feature_matrix = feature_matrix.float()

    num_samples, feature_dim = feature_matrix.shape
    if num_samples == 0 or feature_dim == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, [], 0

    if num_samples <= 1:
        scores = torch.ones(num_samples, dtype=torch.float32)
        return scores, scores, [], 1

    feature_matrix = feature_matrix - feature_matrix.mean(dim=0, keepdim=True)

    max_rank = min(feature_matrix.shape)
    actual_rank = min(rank, max_rank)
    if actual_rank <= 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, [], 0

    u, singular_values, _ = torch.linalg.svd(feature_matrix, full_matrices=False)
    raw_scores = (u[:, :actual_rank] ** 2).sum(dim=1)
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    if float((max_score - min_score).item()) < eps:
        normalized = torch.ones_like(raw_scores)
    else:
        normalized = (raw_scores - min_score) / (max_score - min_score + eps)
    return (
        normalized,
        raw_scores,
        [float(value.item()) for value in singular_values],
        actual_rank,
    )


def compute_leverage_scores(
    payloads: list[ClientFeaturePayload],
    rank: int = 8,
    eps: float = 1e-12,
    class_wise: bool = False,
) -> GDRResult:
    valid_payloads = [payload for payload in payloads if len(payload.labels) > 0]
    if not valid_payloads:
        return GDRResult(records=[], singular_values=[], rank=0, class_wise=class_wise)

    if class_wise and any(payload.left_transform is not None for payload in valid_payloads):
        raise ValueError(
            "Class-wise GDR is not compatible with left-orthogonal encrypted payloads."
        )

    if not class_wise:
        feature_matrix = torch.cat(
            [payload.features for payload in valid_payloads], dim=0
        )
        normalized, _, singular_values, actual_rank = _compute_group_scores(
            feature_matrix,
            rank=rank,
            eps=eps,
        )
        if actual_rank <= 0:
            return GDRResult(
                records=[],
                singular_values=[],
                rank=0,
                class_wise=False,
            )

        u, _, _ = torch.linalg.svd(feature_matrix.float(), full_matrices=False)
        decoded_raw_scores = []
        decoded_segments: list[torch.Tensor] = []
        offset = 0
        for payload in valid_payloads:
            next_offset = offset + len(payload.labels)
            local_u = u[offset:next_offset, :actual_rank]
            if payload.left_transform is not None:
                local_u = payload.left_transform.transpose(0, 1) @ local_u
            decoded_segments.append(local_u)
            decoded_raw_scores.append((local_u**2).sum(dim=1))
            offset = next_offset

        raw_scores = torch.cat(decoded_raw_scores, dim=0)
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if float((max_score - min_score).item()) < eps:
            normalized = torch.ones_like(raw_scores)
        else:
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
                        "source_task_id": payload.task_id,
                        "source_client_id": payload.client_id,
                        "dataset_index": payload.dataset_indices[local_pos],
                        "sample_id": payload.sample_ids[local_pos],
                        "label": label,
                        "leverage_score": float(normalized[global_pos].item()),
                        "raw_leverage_score": float(raw_scores[global_pos].item()),
                    }
                )
            offset += len(payload.labels)

        return GDRResult(
            records=records,
            singular_values=singular_values,
            rank=actual_rank,
            class_wise=False,
        )

    records = []
    items_by_class: dict[int, list[dict[str, Any]]] = {}
    for payload in valid_payloads:
        for local_pos, label in enumerate(payload.labels):
            label = int(label)
            items_by_class.setdefault(label, []).append(
                {
                    "client_id": payload.client_id,
                    "task_id": payload.task_id,
                    "source_task_id": payload.task_id,
                    "source_client_id": payload.client_id,
                    "dataset_index": payload.dataset_indices[local_pos],
                    "sample_id": payload.sample_ids[local_pos],
                    "label": label,
                    "feature": payload.features[local_pos],
                }
            )
    class_singular_values: dict[int, list[float]] = {}
    observed_ranks: list[int] = []

    for label in sorted(items_by_class):
        class_items = items_by_class[label]
        class_features = torch.stack(
            [item["feature"] for item in class_items],
            dim=0,
        )
        normalized, raw_scores, singular_values, actual_rank = _compute_group_scores(
            class_features,
            rank=rank,
            eps=eps,
        )
        class_singular_values[label] = singular_values
        observed_ranks.append(actual_rank)

        for item, normalized_score, raw_score in zip(
            class_items,
            normalized,
            raw_scores,
            strict=True,
        ):
            records.append(
                {
                    "client_id": item["client_id"],
                    "task_id": item["task_id"],
                    "source_task_id": item["source_task_id"],
                    "source_client_id": item["source_client_id"],
                    "dataset_index": item["dataset_index"],
                    "sample_id": item["sample_id"],
                    "label": label,
                    "leverage_score": float(normalized_score.item()),
                    "raw_leverage_score": float(raw_score.item()),
                }
            )

    return GDRResult(
        records=records,
        singular_values=[],
        rank=max(observed_ranks, default=0),
        class_wise=True,
        class_singular_values=class_singular_values,
    )


def group_records_by_client(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["client_id"]), []).append(record)
    return grouped


def compute_client_local_leverage_scores(
    payloads: list[ClientFeaturePayload],
    rank: int = 8,
    eps: float = 1e-12,
) -> ClientLocalGDRResult:
    valid_payloads = [payload for payload in payloads if len(payload.labels) > 0]
    if not valid_payloads:
        return ClientLocalGDRResult(records=[], client_singular_values={}, rank=0)

    records: list[dict[str, Any]] = []
    client_singular_values: dict[int, list[float]] = {}
    observed_ranks: list[int] = []

    for payload in valid_payloads:
        normalized, _, singular_values, actual_rank = _compute_group_scores(
            payload.features,
            rank=rank,
            eps=eps,
        )
        observed_ranks.append(actual_rank)
        client_singular_values[payload.client_id] = singular_values

        # Align row scores back to original sample order when a left transform exists.
        if actual_rank > 0 and payload.left_transform is not None:
            u, _, _ = torch.linalg.svd(payload.features.float(), full_matrices=False)
            decoded_u = payload.left_transform.transpose(0, 1) @ u[:, :actual_rank]
            raw_scores = (decoded_u**2).sum(dim=1)
            total_score = float(raw_scores.sum().item())
            if total_score <= eps:
                normalized = torch.full_like(raw_scores, 1.0 / max(len(raw_scores), 1))
            else:
                normalized = raw_scores / total_score
        else:
            raw_scores = normalized.clone()

        for local_pos, label in enumerate(payload.labels):
            local_probability = float(normalized[local_pos].item()) if len(normalized) else 0.0
            records.append(
                {
                    "client_id": payload.client_id,
                    "task_id": payload.task_id,
                    "source_task_id": payload.task_id,
                    "source_client_id": payload.client_id,
                    "dataset_index": payload.dataset_indices[local_pos],
                    "sample_id": payload.sample_ids[local_pos],
                    "label": int(label),
                    "leverage_score": local_probability,
                    "raw_leverage_score": float(raw_scores[local_pos].item()) if len(raw_scores) else 0.0,
                    "client_sampling_probability": local_probability,
                }
            )

    return ClientLocalGDRResult(
        records=records,
        client_singular_values=client_singular_values,
        rank=max(observed_ranks, default=0),
    )


def attach_sampling_probabilities(
    records: list[dict[str, Any]],
    score_key: str = "raw_leverage_score",
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    if not records:
        return []

    raw_scores = torch.tensor(
        [max(0.0, float(record.get(score_key, 0.0))) for record in records],
        dtype=torch.float64,
    )
    total_score = float(raw_scores.sum().item())
    if total_score <= eps:
        probabilities = torch.full_like(raw_scores, 1.0 / len(records))
    else:
        probabilities = raw_scores / total_score

    annotated = []
    for record, probability in zip(records, probabilities.tolist(), strict=True):
        annotated_record = dict(record)
        annotated_record["sampling_probability"] = float(probability)
        annotated.append(annotated_record)
    return annotated


def sample_records_by_probability(
    records: list[dict[str, Any]],
    total_budget: int,
    seed: int,
    score_key: str = "raw_leverage_score",
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    if total_budget <= 0 or not records:
        return []

    annotated = attach_sampling_probabilities(records, score_key=score_key, eps=eps)
    sample_size = min(total_budget, len(annotated))
    probabilities = torch.tensor(
        [float(record["sampling_probability"]) for record in annotated],
        dtype=torch.float64,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    selected_indices = torch.multinomial(
        probabilities,
        sample_size,
        replacement=False,
        generator=generator,
    )

    selected = []
    for selected_index in selected_indices.tolist():
        record = dict(annotated[selected_index])
        probability = max(float(record["sampling_probability"]), eps)
        record["sampling_weight"] = math.sqrt(1.0 / (sample_size * probability))
        selected.append(record)
    return selected


def sample_records_by_class_probability(
    records: list[dict[str, Any]],
    total_budget: int,
    seed: int,
    score_key: str = "raw_leverage_score",
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    if total_budget <= 0 or not records:
        return []

    records_by_class: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        records_by_class.setdefault(int(record["label"]), []).append(record)

    classes = sorted(records_by_class)
    if not classes:
        return []

    total_budget = min(total_budget, len(records))
    num_classes = len(classes)
    base_quota = total_budget // num_classes
    remainder = total_budget % num_classes

    # First pass: assign a near-uniform per-class quota, capped by class size.
    class_quotas: dict[int, int] = {}
    leftover_budget = 0
    for class_index, label in enumerate(classes):
        target_quota = base_quota + (1 if class_index < remainder else 0)
        class_size = len(records_by_class[label])
        assigned_quota = min(target_quota, class_size)
        class_quotas[label] = assigned_quota
        leftover_budget += target_quota - assigned_quota

    # Second pass: redistribute leftover budget to classes with remaining capacity.
    while leftover_budget > 0:
        progress = False
        for label in classes:
            if leftover_budget <= 0:
                break
            remaining_capacity = len(records_by_class[label]) - class_quotas[label]
            if remaining_capacity <= 0:
                continue
            class_quotas[label] += 1
            leftover_budget -= 1
            progress = True
        if not progress:
            break

    selected: list[dict[str, Any]] = []
    for class_offset, label in enumerate(classes):
        class_records = records_by_class[label]
        class_budget = class_quotas[label]
        if class_budget <= 0:
            continue

        annotated = attach_sampling_probabilities(
            class_records,
            score_key=score_key,
            eps=eps,
        )
        sample_size = min(class_budget, len(annotated))
        probabilities = torch.tensor(
            [float(record["sampling_probability"]) for record in annotated],
            dtype=torch.float64,
        )
        generator = torch.Generator()
        generator.manual_seed(seed + class_offset)
        selected_indices = torch.multinomial(
            probabilities,
            sample_size,
            replacement=False,
            generator=generator,
        )

        for selected_index in selected_indices.tolist():
            record = dict(annotated[selected_index])
            probability = max(float(record["sampling_probability"]), eps)
            record["sampling_probability"] = float(probability)
            record["class_sampling_budget"] = int(sample_size)
            record["sampling_weight"] = math.sqrt(1.0 / (sample_size * probability))
            selected.append(record)

    return selected


def sample_records_official_style(
    records: list[dict[str, Any]],
    total_budget: int,
    num_clients: int,
    seed: int,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    if total_budget <= 0 or not records or num_clients <= 0:
        return []

    records_by_client = group_records_by_client(records)
    min_samples_per_client = total_budget // num_clients
    selected: list[dict[str, Any]] = []

    global_records: list[dict[str, Any]] = []
    global_probabilities: list[float] = []

    for client_offset in range(num_clients):
        client_records = records_by_client.get(client_offset, [])
        if not client_records:
            continue

        client_probs = torch.tensor(
            [
                max(float(record.get("client_sampling_probability", 0.0)), 0.0)
                for record in client_records
            ],
            dtype=torch.float64,
        )
        prob_sum = float(client_probs.sum().item())
        if prob_sum <= eps:
            client_probs = torch.full_like(client_probs, 1.0 / len(client_records))
        else:
            client_probs = client_probs / prob_sum

        sample_size = min_samples_per_client
        if sample_size > 0:
            generator = torch.Generator()
            generator.manual_seed(seed + client_offset)
            sampled_indices = torch.multinomial(
                client_probs,
                sample_size,
                replacement=True,
                generator=generator,
            )
            for sampled_index in sampled_indices.tolist():
                record = dict(client_records[sampled_index])
                record["sampling_probability"] = float(client_probs[sampled_index].item())
                selected.append(record)

        client_weight = 1.0 / num_clients
        for record, probability in zip(client_records, client_probs.tolist(), strict=True):
            global_records.append(record)
            global_probabilities.append(float(probability) * client_weight)

    remaining_budget = total_budget - len(selected)
    if remaining_budget > 0 and global_records:
        probabilities = torch.tensor(global_probabilities, dtype=torch.float64)
        prob_sum = float(probabilities.sum().item())
        if prob_sum <= eps:
            probabilities = torch.full_like(probabilities, 1.0 / len(global_records))
        else:
            probabilities = probabilities / prob_sum

        generator = torch.Generator()
        generator.manual_seed(seed + 99_999)
        sampled_indices = torch.multinomial(
            probabilities,
            remaining_budget,
            replacement=True,
            generator=generator,
        )
        for sampled_index in sampled_indices.tolist():
            record = dict(global_records[sampled_index])
            record["sampling_probability"] = float(probabilities[sampled_index].item())
            selected.append(record)

    return selected
