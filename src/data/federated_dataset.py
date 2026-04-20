from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset, Subset


@dataclass
class FederatedTaskData:
    train_subsets: dict[str, Subset]
    test_subset_seen: Subset
    task_classes: list[int]


class FederatedDatasetManager:
    """
    Wraps:
    - full train dataset
    - full test dataset
    - saved task split payload
    - saved federated partition payload

    Provides:
    - per-task per-client train subset
    - per-task seen-class test subset
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        task_split_payload: dict[str, Any],
        federated_partition_payload: dict[str, Any],
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.task_split_payload = task_split_payload
        self.federated_partition_payload = federated_partition_payload

        self.num_tasks = int(task_split_payload["num_tasks"])
        self.task_classes = task_split_payload["task_classes"]
        self.task_to_test_indices = task_split_payload["task_to_test_indices"]
        self.task_to_client_train_indices = federated_partition_payload[
            "task_to_client_train_indices"
        ]

    def get_task_classes(self, task_id: int) -> list[int]:
        return list(self.task_classes[task_id])

    def get_seen_test_indices(self, task_id: int) -> list[int]:
        """
        Return test indices for all classes seen up to current task.
        """
        merged: list[int] = []
        for tid in range(task_id + 1):
            merged.extend(self.task_to_test_indices[f"task_{tid}"])
        return merged

    def get_train_subset(self, task_id: int, client_id: int) -> Subset:
        task_key = f"task_{task_id}"
        client_key = f"client_{client_id}"
        indices = self.task_to_client_train_indices[task_key][client_key]
        return Subset(self.train_dataset, indices)

    def get_all_client_train_subsets(self, task_id: int) -> dict[str, Subset]:
        task_key = f"task_{task_id}"
        result: dict[str, Subset] = {}

        for client_key, indices in self.task_to_client_train_indices[task_key].items():
            result[client_key] = Subset(self.train_dataset, indices)

        return result

    def get_seen_test_subset(self, task_id: int) -> Subset:
        indices = self.get_seen_test_indices(task_id)
        return Subset(self.test_dataset, indices)

    def get_task_data(self, task_id: int) -> FederatedTaskData:
        return FederatedTaskData(
            train_subsets=self.get_all_client_train_subsets(task_id),
            test_subset_seen=self.get_seen_test_subset(task_id),
            task_classes=self.get_task_classes(task_id),
        )

    def summary(self) -> str:
        lines = []
        lines.append(f"Num tasks: {self.num_tasks}")

        for task_id in range(self.num_tasks):
            task_key = f"task_{task_id}"
            lines.append(f"[Task {task_id}] classes={self.get_task_classes(task_id)}")

            for client_key, indices in self.task_to_client_train_indices[task_key].items():
                lines.append(f"  {client_key}: {len(indices)} train samples")

            lines.append(
                f"  seen test samples: {len(self.get_seen_test_indices(task_id))}"
            )

        return "\n".join(lines)