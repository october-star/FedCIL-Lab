from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class ReplaySample:
    x: torch.Tensor
    y: int
    metadata: dict[str, Any]


class ReplayBuffer(Dataset):
    """
    Per-client replay buffer with class-balanced capacity management.

    The buffer stores tensors and remapped incremental labels. Metadata is kept
    alongside samples so future GDR code can attach or return sample indices.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.samples: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return sample.x, sample.y

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

        selected_records = self._class_balanced_records(records, max_samples)
        for record in selected_records:
            dataset_index = int(record["dataset_index"])
            x, y = dataset[dataset_index]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)

            metadata = {
                "task_id": task_id,
                "client_id": client_id,
                "dataset_index": dataset_index,
                "leverage_score": float(record["leverage_score"]),
                "raw_leverage_score": float(record["raw_leverage_score"]),
            }
            self.samples.append(
                ReplaySample(
                    x=x.detach().cpu(),
                    y=int(y),
                    metadata=metadata,
                )
            )

        self._rebalance()

    def class_counts(self) -> dict[int, int]:
        counts = Counter(sample.y for sample in self.samples)
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "num_samples": len(self.samples),
            "class_counts": self.class_counts(),
        }

    def gdr_index_payload(self) -> list[dict[str, Any]]:
        """
        Placeholder interface for future GDR index feedback.
        """
        return [sample.metadata for sample in self.samples]

    def _rebalance(self) -> None:
        if len(self.samples) <= self.capacity:
            return

        by_class: dict[int, list[ReplaySample]] = {}
        for sample in self.samples:
            by_class.setdefault(sample.y, []).append(sample)

        for class_samples in by_class.values():
            class_samples.sort(
                key=lambda sample: sample.metadata.get("leverage_score", 0.0),
                reverse=True,
            )

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
                key=lambda record: float(record["leverage_score"]),
                reverse=True,
            )

        selected: list[dict[str, Any]] = []
        classes = sorted(by_class)
        while len(selected) < limit and classes:
            next_classes = []
            for class_id in classes:
                class_records = by_class[class_id]
                if class_records and len(selected) < limit:
                    selected.append(class_records.pop(0))
                if class_records:
                    next_classes.append(class_id)
            classes = next_classes

        return selected
