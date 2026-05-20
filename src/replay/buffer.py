from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.sample_ids import resolve_sample_id


@dataclass
class ReplaySample:
    x: torch.Tensor
    y: int
    metadata: dict[str, Any]


class ReplayBuffer(Dataset):
    def __init__(
        self,
        capacity: int,
        seed: int = 0,
        balance_classes: bool = True,
        use_scores: bool = False,
    ) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.balance_classes = balance_classes
        self.use_scores = use_scores
        self.samples: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return sample.x, sample.y

    def get_sample_id(self, index: int) -> int:
        return int(self.samples[index].metadata.get("sample_id", index))

    def get_sampling_weight(self, index: int) -> float:
        return float(self.samples[index].metadata.get("sampling_weight", 1.0))

    def add_dataset(
        self,
        dataset: Dataset,
        task_id: int,
        client_id: int,
        max_samples: int | None = None,
    ) -> None:
        if self.capacity == 0 or len(dataset) == 0:
            return

        indices = list(range(len(dataset)))
        self.rng.shuffle(indices)
        if max_samples is not None:
            indices = indices[:max_samples]

        for dataset_index in indices:
            x, y = dataset[dataset_index]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            self.samples.append(
                ReplaySample(
                    x=x.detach().cpu(),
                    y=int(y),
                    metadata={
                        "task_id": task_id,
                        "client_id": client_id,
                        "dataset_index": dataset_index,
                        "sample_id": resolve_sample_id(dataset, dataset_index),
                    },
                )
            )

        self._rebalance()

    def add_scored_dataset(
        self,
        dataset: Dataset,
        records: list[dict[str, Any]],
        task_id: int,
        client_id: int,
        max_samples: int | None = None,
    ) -> None:
        if self.capacity == 0 or len(dataset) == 0 or not records:
            return

        selected_records = (
            records
            if max_samples is None
            else self._class_balanced_records(records, max_samples)
        )

        for record in selected_records:
            dataset_index = int(record["dataset_index"])
            x, y = dataset[dataset_index]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            metadata = {
                "task_id": task_id,
                "client_id": client_id,
                "dataset_index": dataset_index,
                "sample_id": int(
                    record.get("sample_id", resolve_sample_id(dataset, dataset_index))
                ),
                "leverage_score": float(record.get("leverage_score", 0.0)),
                "raw_leverage_score": float(record.get("raw_leverage_score", 0.0)),
                "sampling_probability": float(record.get("sampling_probability", 0.0)),
                "sampling_weight": float(record.get("sampling_weight", 1.0)),
            }

            self.samples.append(
                ReplaySample(
                    x=x.detach().cpu(),
                    y=int(y),
                    metadata=metadata,
                )
            )

        self._rebalance()

    def add_selected_records(
        self,
        dataset: Dataset,
        records: list[dict[str, Any]],
        task_id: int,
        client_id: int,
    ) -> None:
        if self.capacity == 0 or len(dataset) == 0 or not records:
            return

        for record in records:
            dataset_index = int(record["dataset_index"])
            x, y = dataset[dataset_index]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            metadata = {
                "task_id": int(record.get("source_task_id", task_id)),
                "client_id": int(record.get("source_client_id", client_id)),
                "dataset_index": dataset_index,
                "sample_id": int(
                    record.get("sample_id", resolve_sample_id(dataset, dataset_index))
                ),
                "leverage_score": float(
                    record.get("leverage_score", record.get("raw_leverage_score", 0.0))
                ),
                "raw_leverage_score": float(record.get("raw_leverage_score", 0.0)),
                "sampling_probability": float(record.get("sampling_probability", 0.0)),
                "sampling_weight": float(record.get("sampling_weight", 1.0)),
            }

            self.samples.append(
                ReplaySample(
                    x=x.detach().cpu(),
                    y=int(y),
                    metadata=metadata,
                )
            )

        self._rebalance()

    def replace_with_scored_dataset(
        self,
        dataset: Dataset,
        records: list[dict[str, Any]],
        task_id: int,
        client_id: int,
        max_samples: int | None = None,
        class_wise: bool = True,
    ) -> None:
        if self.capacity == 0 or len(dataset) == 0 or not records:
            self.samples = []
            return

        budget = self.capacity if max_samples is None else min(max_samples, self.capacity)
        if budget <= 0:
            self.samples = []
            return

        if class_wise:
            selected_records = self._class_balanced_records(records, budget)
        else:
            selected_records = sorted(
                records,
                key=lambda record: float(record.get("raw_leverage_score", 0.0)),
                reverse=True,
            )[:budget]

        new_samples: list[ReplaySample] = []
        for record in selected_records:
            dataset_index = int(record["dataset_index"])
            x, y = dataset[dataset_index]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            metadata = {
                "task_id": int(record.get("source_task_id", task_id)),
                "client_id": int(record.get("source_client_id", client_id)),
                "dataset_index": dataset_index,
                "sample_id": int(
                    record.get("sample_id", resolve_sample_id(dataset, dataset_index))
                ),
                "leverage_score": float(record.get("leverage_score", 0.0)),
                "raw_leverage_score": float(record.get("raw_leverage_score", 0.0)),
            }

            new_samples.append(
                ReplaySample(
                    x=x.detach().cpu(),
                    y=int(y),
                    metadata=metadata,
                )
            )

        self.samples = new_samples
        self._rebalance()

    def class_counts(self) -> dict[int, int]:
        counts = Counter(sample.y for sample in self.samples)
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "num_samples": len(self.samples),
            "class_counts": self.class_counts(),
            "balance_classes": self.balance_classes,
            "use_scores": self.use_scores,
        }

    def gdr_index_payload(self) -> list[dict[str, Any]]:
        return [sample.metadata for sample in self.samples]

    def _rebalance(self) -> None:
        if len(self.samples) <= self.capacity:
            return

        if not self.balance_classes:
            self.rng.shuffle(self.samples)
            self.samples = self.samples[: self.capacity]
            return

        by_class: dict[int, list[ReplaySample]] = {}
        for sample in self.samples:
            by_class.setdefault(sample.y, []).append(sample)

        for class_samples in by_class.values():
            if self.use_scores:
                class_samples.sort(
                    key=lambda sample: sample.metadata.get(
                        "raw_leverage_score",
                        sample.metadata.get("leverage_score", 0.0),
                    ),
                    reverse=True,
                )
            else:
                self.rng.shuffle(class_samples)

        classes = sorted(by_class)
        kept: list[ReplaySample] = []

        while len(kept) < self.capacity and classes:
            next_classes = []
            for class_id in classes:
                class_samples = by_class[class_id]
                if class_samples and len(kept) < self.capacity:
                    kept.append(class_samples.pop(0))
                if class_samples:
                    next_classes.append(class_id)
            classes = next_classes

        self.samples = kept

    def _class_balanced_records(
        self,
        records: list[dict[str, Any]],
        max_samples: int | None,
    ) -> list[dict[str, Any]]:
        limit = len(records) if max_samples is None else min(max_samples, len(records))
        if limit <= 0:
            return []

        by_class: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_class.setdefault(int(record["label"]), []).append(record)

        for class_records in by_class.values():
            class_records.sort(
                key=lambda record: float(record.get("raw_leverage_score", 0.0)),
                reverse=True,
            )

        classes = sorted(by_class)
        selected: list[dict[str, Any]] = []

        if len(classes) >= limit:
            ranked_heads = sorted(
                (class_records[0] for class_records in by_class.values() if class_records),
                key=lambda record: float(record.get("raw_leverage_score", 0.0)),
                reverse=True,
            )
            return ranked_heads[:limit]

        per_class = max(1, limit // len(classes))

        for class_id in classes:
            selected.extend(by_class[class_id][:per_class])

        if len(selected) < limit:
            selected_keys = {
                (
                    int(record.get("client_id", record.get("source_client_id", -1))),
                    int(record.get("sample_id", record.get("dataset_index", -1))),
                )
                for record in selected
            }

            remaining = [
                record
                for class_id in classes
                for record in by_class[class_id][per_class:]
                if (
                    int(record.get("client_id", record.get("source_client_id", -1))),
                    int(record.get("sample_id", record.get("dataset_index", -1))),
                )
                not in selected_keys
            ]

            remaining.sort(
                key=lambda record: float(record.get("raw_leverage_score", 0.0)),
                reverse=True,
            )
            selected.extend(remaining[: limit - len(selected)])

        return selected[:limit]