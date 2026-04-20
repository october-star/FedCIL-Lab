from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FederatedPartitionResult:
    dataset_name: str
    num_tasks: int
    num_clients: int
    beta: float
    seed: int
    task_to_client_train_indices: dict[str, dict[str, list[int]]]
    task_to_client_class_counts: dict[str, dict[str, dict[str, int]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_labels_from_indices(
    full_targets: list[int],
    indices: list[int],
) -> np.ndarray:
    """
    Extract labels corresponding to a list of sample indices.
    """
    return np.array([full_targets[i] for i in indices], dtype=np.int64)


def compute_client_class_counts(
    full_targets: list[int],
    client_to_indices: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    """
    Return nested dict:
    {
        "client_0": {"0": 123, "7": 45},
        "client_1": {"0": 12, "3": 99},
        ...
    }
    """
    result: dict[str, dict[str, int]] = {}

    for client_id, indices in client_to_indices.items():
        counts: dict[str, int] = {}
        for idx in indices:
            label = str(full_targets[idx])
            counts[label] = counts.get(label, 0) + 1
        result[client_id] = counts

    return result


def dirichlet_partition(
    labels: np.ndarray,
    indices: list[int],
    num_clients: int,
    beta: float,
    seed: int,
    min_size_per_client: int = 1,
    max_retry: int = 100,
) -> dict[str, list[int]]:
    """
    Partition one task's sample indices into multiple clients using Dirichlet distribution.

    Args:
        labels: labels for the provided indices, shape [N]
        indices: original dataset indices, shape [N]
        num_clients: number of federated clients
        beta: Dirichlet concentration parameter
        seed: random seed
        min_size_per_client: minimum number of samples each client must receive
        max_retry: retry times if min_size constraint is not satisfied

    Returns:
        {
            "client_0": [...],
            "client_1": [...],
            ...
        }
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if len(indices) == 0:
        raise ValueError("indices cannot be empty")
    if len(labels) != len(indices):
        raise ValueError("labels and indices must have the same length")

    labels = np.asarray(labels)
    indices = np.asarray(indices)
    unique_classes = np.unique(labels)

    for attempt in range(max_retry):
        rng = np.random.default_rng(seed + attempt)

        client_bins: list[list[int]] = [[] for _ in range(num_clients)]

        for cls in unique_classes:
            cls_mask = labels == cls
            cls_indices = indices[cls_mask].copy()
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(alpha=np.repeat(beta, num_clients))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            split_indices = np.split(cls_indices, split_points)

            for client_id, part in enumerate(split_indices):
                client_bins[client_id].extend(part.tolist())

        sizes = [len(v) for v in client_bins]
        if min(sizes) >= min_size_per_client:
            break
    else:
        raise RuntimeError(
            f"Failed to satisfy min_size_per_client={min_size_per_client} "
            f"after {max_retry} retries."
        )

    # shuffle each client's final sample order
    client_to_indices: dict[str, list[int]] = {}
    for client_id, client_indices in enumerate(client_bins):
        client_indices = list(client_indices)
        rng.shuffle(client_indices)
        client_to_indices[f"client_{client_id}"] = client_indices

    return client_to_indices


def build_federated_partition_from_task_split(
    task_split_payload: dict[str, Any],
    full_train_targets: list[int],
    num_clients: int,
    beta: float,
    seed: int,
    min_size_per_client: int = 1,
) -> FederatedPartitionResult:
    """
    Build federated client partition for each task based on saved task split.
    """
    dataset_name = task_split_payload["dataset_name"]
    num_tasks = task_split_payload["num_tasks"]
    task_to_train_indices = task_split_payload["task_to_train_indices"]

    task_to_client_train_indices: dict[str, dict[str, list[int]]] = {}
    task_to_client_class_counts: dict[str, dict[str, dict[str, int]]] = {}

    for task_id in range(num_tasks):
        task_key = f"task_{task_id}"
        task_indices = task_to_train_indices[task_key]
        task_labels = get_labels_from_indices(full_train_targets, task_indices)

        client_partition = dirichlet_partition(
            labels=task_labels,
            indices=task_indices,
            num_clients=num_clients,
            beta=beta,
            seed=seed + task_id * 1000,
            min_size_per_client=min_size_per_client,
        )

        class_counts = compute_client_class_counts(
            full_targets=full_train_targets,
            client_to_indices=client_partition,
        )

        task_to_client_train_indices[task_key] = client_partition
        task_to_client_class_counts[task_key] = class_counts

    return FederatedPartitionResult(
        dataset_name=dataset_name,
        num_tasks=num_tasks,
        num_clients=num_clients,
        beta=beta,
        seed=seed,
        task_to_client_train_indices=task_to_client_train_indices,
        task_to_client_class_counts=task_to_client_class_counts,
    )


def save_federated_partition(
    result: FederatedPartitionResult,
    save_dir: str | Path = "data/processed/federated_partitions",
) -> Path:
    save_dir = ensure_dir(save_dir)
    beta_str = str(result.beta).replace(".", "")
    save_path = save_dir / (
        f"{result.dataset_name}_{result.num_tasks}task_"
        f"{result.num_clients}clients_beta{beta_str}_seed{result.seed}.json"
    )

    save_json(result.to_dict(), save_path)
    return save_path


def load_federated_partition(path: str | Path) -> dict[str, Any]:
    return load_json(path)