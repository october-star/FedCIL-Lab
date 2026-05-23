#Author: mmj
#DATE: 22.05.2026
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class IndexedSelectionDataset(Dataset):
    """
    Lightweight wrapper around a base dataset.

    Keeps only selected indices while preserving:
    - original __getitem__
    - labels
    - sample_id access (if exists)

    Used by GDR replay retention.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        real_index = self.indices[index]
        return self.dataset[real_index]

    def get_original_index(self, index: int) -> int:
        return self.indices[index]

    def get_sample_id(self, index: int) -> int:
        """
        Forward sample_id lookup if supported by base dataset.
        """
        real_index = self.indices[index]

        if hasattr(self.dataset, "get_sample_id"):
            return int(self.dataset.get_sample_id(real_index))

        return int(real_index)

    def get_label(self, index: int) -> int:
        """
        Try to retrieve label robustly.
        """
        item = self[index]

        if isinstance(item, (tuple, list)):
            return int(item[-1])

        raise ValueError(
            f"Cannot infer label from item type: {type(item)}"
        )

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}

        for i in range(len(self)):
            label = self.get_label(i)
            counts[label] = counts.get(label, 0) + 1

        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": len(self),
            "class_counts": self.class_counts(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={len(self)}, "
            f"base_dataset={type(self.dataset).__name__})"
        )