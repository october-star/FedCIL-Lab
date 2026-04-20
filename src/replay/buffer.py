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
            self.rng.shuffle(class_samples)

        classes = sorted(by_class)
        kept: list[ReplaySample] = []

        while len(kept) < self.capacity and classes:
            next_classes = []
            for class_id in classes:
                class_samples = by_class[class_id]
                if class_samples and len(kept) < self.capacity:
                    kept.append(class_samples.pop())
                if class_samples:
                    next_classes.append(class_id)
            classes = next_classes

        self.samples = kept
