from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SampleWeightDataset(Dataset):
    """
    Wrap a dataset so every sample returns `(x, y, sample_weight)`.
    """

    def __init__(self, dataset: Dataset, default_weight: float | None = None) -> None:
        self.dataset = dataset
        self.default_weight = default_weight

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        x, y = item[0], item[1]
        return x, y, torch.tensor(self.get_sampling_weight(index), dtype=torch.float32)

    def get_sampling_weight(self, index: int) -> float:
        if self.default_weight is not None:
            return float(self.default_weight)

        getter = getattr(self.dataset, "get_sampling_weight", None)
        if callable(getter):
            return float(getter(index))
        return 1.0

    def get_sample_id(self, index: int) -> int:
        getter = getattr(self.dataset, "get_sample_id", None)
        if callable(getter):
            return int(getter(index))
        return int(index)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)
