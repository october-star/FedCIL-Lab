from __future__ import annotations

from bisect import bisect_right

from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import ConcatDataset

from src.data.federated_dataset import TargetRemapDataset


def resolve_sample_id(dataset: Dataset, index: int) -> int:
    """
    Resolve a dataset-local index back to the original underlying sample id.

    This keeps replay/GDR bookkeeping stable even when the training dataset is
    wrapped by incremental remapping or nested Subset objects.
    """
    sample_id = int(index)
    current = dataset

    while True:
        if isinstance(current, TargetRemapDataset):
            current = current.dataset
            continue
        if isinstance(current, ConcatDataset):
            dataset_pos = bisect_right(current.cumulative_sizes, sample_id)
            prev_cum = 0 if dataset_pos == 0 else current.cumulative_sizes[dataset_pos - 1]
            sample_id = sample_id - prev_cum
            current = current.datasets[dataset_pos]
            continue
        if isinstance(current, Subset):
            sample_id = int(current.indices[sample_id])
            current = current.dataset
            continue
        getter = getattr(current, "get_sample_id", None)
        if callable(getter):
            return int(getter(sample_id))
        return sample_id
