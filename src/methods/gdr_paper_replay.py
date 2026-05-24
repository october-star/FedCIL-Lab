# from __future__ import annotations

# import copy
# import math
# from collections import Counter
# from pathlib import Path

# from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

# from src.data.weighted_dataset import SampleWeightDataset
# from src.federated.aggregator import fedavg
# from src.federated.client import Client
# from src.gdr.features import build_client_feature_payload, sample_orthogonal_matrix
# from src.gdr.server import (
#     compute_client_local_leverage_scores,
#     group_records_by_client,
#     sample_records_by_class_probability,
# )
# from src.methods.base_method import BaseMethod


# class IndexedSelectionDataset(Dataset):
#     """
#     Dataset view backed by an explicit list of indices.

#     Unlike torch.utils.data.Subset, this keeps duplicate indices, which matches
#     the original B.py sampling behavior where official GDR sampling is done with
#     replacement.
#     """

#     def __init__(
#         self,
#         dataset: Dataset,
#         indices: list[int],
#         sampling_weights: list[float] | None = None,
#     ) -> None:
#         self.dataset = dataset
#         self.indices = [int(index) for index in indices]

#         self.sampling_weights = (
#             [float(weight) for weight in sampling_weights]
#             if sampling_weights is not None
#             else None
#         )
#         if self.sampling_weights is not None and len(self.sampling_weights) != len(
#             self.indices
#         ):
#             raise ValueError("sampling_weights must match indices length")

#     def __len__(self) -> int:
#         return len(self.indices)

#     def __getitem__(self, index: int):
#         return self.dataset[self.indices[index]]
    
#     def get_sampling_weight(self, index: int) -> float:
#         if self.sampling_weights is not None:
#             return float(self.sampling_weights[index])

#         getter = getattr(self.dataset, "get_sampling_weight", None)
#         if callable(getter):
#             return float(getter(self.indices[index]))
#         return 1.0



# class LocalReplayGDRPaper(BaseMethod):
#     """
#     GDR-only variant abstracted from the paper TTS branch.
#     """

#     method_name = "local_replay_gdr_paper"

#     def __init__(
#         self,
#         *args,
#         buffer_size: int = 200,
#         samples_per_task: int | None = None,
#         seed: int = 0,
#         gdr_rank: int = 8,
#         gdr_feature_samples: int | None = None,
#         gdr_class_wise: bool = False,
#         figure_dir: str | Path = "outputs/figures/gdr",
#         run_name: str = "local_replay_gdr_paper",
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.buffer_size = buffer_size
#         self.samples_per_task = samples_per_task
#         self.seed = seed
#         self.gdr_rank = gdr_rank
#         self.gdr_feature_samples = gdr_feature_samples
#         self.gdr_class_wise = gdr_class_wise
#         self.figure_dir = Path(figure_dir)
#         self.run_name = run_name
#         self.clients = [Client(self.device) for _ in range(self.num_clients)]
#         self.retained_datasets: list[list[Dataset]] = [[] for _ in range(self.num_clients)]

#     def train(self) -> dict:
#         print("Start Federated Local Replay + GDR (paper branch)...")
#         history = {
#             "method": self.method_name,
#             "buffer_size": self.buffer_size,
#             "samples_per_task": self.samples_per_task,
#             "gdr_rank": self.gdr_rank,
#             "gdr_feature_samples": self.gdr_feature_samples,
#             "gdr_class_wise": self.gdr_class_wise,
#             "gdr_candidate_pool": "current_task_only",
#             "gdr_selection_mode": "per_client_class_balanced_sampling",
#             "gdr_encryption": "P_k_X_Q",
#             "replay_weighting": "sampling_weighted_ce",
#             "tasks": [],
#         }

#         for task_id in range(self.dataset_manager.num_tasks):
#             print(f"\n=== Task {task_id} ===")

#             old_classes = self.model.num_classes
#             task_classes = self.dataset_manager.get_task_classes(task_id)
#             self.model.expand_head(len(task_classes))
#             self.model.to(self.device)
#             task_history = {
#                 "task_id": task_id,
#                 "task_classes": task_classes,
#                 "old_classes": old_classes,
#                 "new_classes": len(task_classes),
#                 "rounds": [],
#                 "buffer_before": self._retained_summaries(),
#             }

#             for round_id in range(self.rounds):
#                 local_states = []
#                 sample_counts = []
#                 losses = []
#                 round_lr = self._round_lr(round_id)
#                 #round_lr = self.lr

#                 for client_id in range(self.num_clients):
#                     current_subset = self.dataset_manager.get_train_subset(
#                         task_id, client_id
#                     )
#                     train_dataset, sample_weights = self._compose_train_dataset(
#                         current_subset,
#                         client_id,
#                     )
#                     if len(train_dataset) < 2:
#                         continue

#                     sampler = None
#                     if sample_weights is not None:
#                         sampler = WeightedRandomSampler(
#                             weights=sample_weights,
#                             num_samples=len(train_dataset),
#                             replacement=True,
#                         )

#                     local_model = copy.deepcopy(self.model)
#                     state, loss = self.clients[client_id].train(
#                         model=local_model,
#                         dataset=train_dataset,
#                         batch_size=self.batch_size,
#                         epochs=self.local_epochs,
#                         lr=round_lr,
#                         sampler=sampler,
#                     )

#                     local_states.append(state)
#                     sample_counts.append(len(train_dataset))
#                     losses.append(loss)

#                 if not local_states:
#                     raise RuntimeError(f"No client data found for task {task_id}.")

#                 total_samples = sum(sample_counts)
#                 weights = [1.0 / len(local_states)] * len(local_states)
#                 self.model.load_state_dict(fedavg(local_states, weights))

#                 test_set = self.dataset_manager.get_seen_test_subset(task_id)
#                 acc = self.evaluate(self.model, test_set)
#                 avg_loss = sum(losses) / len(losses)
#                 print(
#                     f"[Task {task_id}][Round {round_id}] "
#                     f"Loss: {avg_loss:.4f} Acc: {acc:.4f}"
#                 )
#                 task_history["rounds"].append(
#                     {
#                         "round_id": round_id,
#                         "loss": avg_loss,
#                         "seen_acc": acc,
#                         "num_clients": len(local_states),
#                         "num_samples": total_samples,
#                     }
#                 )

#             final_test = self.dataset_manager.get_seen_test_subset(task_id)
#             final_acc = self.evaluate(self.model, final_test)
#             print(f"[Task {task_id}] Final Acc: {final_acc:.4f}")
#             task_history["final_seen_acc"] = final_acc
#             gdr_result = self._update_retained_datasets_with_gdr(
#                 task_id=task_id,
#                 new_classes=len(task_classes),
#             )
#             task_history["buffer_after"] = self._retained_summaries()
#             task_history["gdr"] = gdr_result
#             history["tasks"].append(task_history)

#         history["buffer_stats"] = self._retained_summaries()
#         return history

#     def _compose_train_dataset(
#         self,
#         current_subset: Dataset,
#         client_id: int,
#     ) -> tuple[Dataset, list[float] | None]:
#         retained = self.retained_datasets[client_id]
#         if not retained:
#             return (
#                 SampleWeightDataset(
#                     current_subset,
#                     default_weight=1.0,
#                 ),
#                 None,
#             )
#         full = ConcatDataset(
#             [
#                 SampleWeightDataset(
#                     current_subset,
#                     default_weight=1.0,
#                 ),
#                 *[
#                     SampleWeightDataset(
#                         retained_dataset,
#                     )
#                     for retained_dataset in retained
#                 ],
#             ]
#         )
#         return full, None
#         # retained = self.retained_datasets[client_id]
#         # if not retained:
#         #     return current_subset, None
#         # full = ConcatDataset([current_subset, *retained])
#         # n_cur = len(current_subset)
#         # n_rep = len(full) - n_cur
#         # if n_rep == 0:
#         #     return full, None
#         # replay_ratio = 1.0    
#         # w = [1.0/n_cur]*n_cur + [replay_ratio/n_rep]*n_rep
#         #return full, w

#     def _global_sampling_budget(self, new_classes: int) -> int:
#         if self.samples_per_task is not None:
#             return self.samples_per_task * self.num_clients
#         return new_classes * self.buffer_size

#     def _per_client_sampling_budgets(self, total_budget: int) -> dict[int, int]:
#         total_budget = max(int(total_budget), 0)
#         if self.num_clients <= 0:
#             return {}

#         base_quota = total_budget // self.num_clients
#         remainder = total_budget % self.num_clients
#         return {
#             client_id: base_quota + (1 if client_id < remainder else 0)
#             for client_id in range(self.num_clients)
#         }

#     def _round_lr(self, round_id: int, eta_min: float = 1e-3) -> float:
#         if self.rounds <= 1:
#             return self.lr
#         cosine = (1.0 + math.cos(math.pi * round_id / self.rounds)) / 2.0
#         return eta_min + (self.lr - eta_min) * cosine

#     def _update_retained_datasets_with_gdr(
#         self,
#         task_id: int,
#         new_classes: int,
#     ) -> dict:
#         from src.gdr.visualize import plot_buffer_class_distribution, plot_leverage_scores

#         if self.gdr_class_wise:
#             raise ValueError(
#                 "Paper GDR with P_k / Q encryption currently does not support "
#                 "class-wise leverage computation. Use --no-gdr_class_wise."
#             )

#         payloads = []
#         current_datasets = {}
#         right_transform = sample_orthogonal_matrix(
#             self.model.feature_dim,
#             seed=self.seed + task_id * 1000 + 999_999,
#         )

#         for client_id in range(self.num_clients):
#             current_subset = self.dataset_manager.get_train_subset(task_id, client_id)
#             current_datasets[client_id] = current_subset
#             payloads.append(
#                 build_client_feature_payload(
#                     current_subset,
#                     client_id=client_id,
#                     task_id=task_id,
#                     feature_extractor=self.model.backbone,
#                     device=self.device,
#                     max_samples=self.gdr_feature_samples,
#                     seed=self.seed + task_id * 1000 + client_id,
#                     right_transform=right_transform,
#                     left_transform_seed=self.seed + task_id * 10_000 + client_id,
#                 )
#             )

#         result = compute_client_local_leverage_scores(
#             payloads,
#             rank=self.gdr_rank,
#         )
#         total_budget = self._global_sampling_budget(new_classes)
#         per_client_budgets = self._per_client_sampling_budgets(total_budget)
#         records_by_client = group_records_by_client(result.records)
#         selected_records: list[dict] = []
#         selected_records_by_client: dict[int, list[dict]] = {}

#         for client_id in range(self.num_clients):
#             selected_client_records = sample_records_by_class_probability(
#                 records_by_client.get(client_id, []),
#                 total_budget=per_client_budgets.get(client_id, 0),
#                 seed=self.seed + task_id * 1000 + client_id,
#             )
#             selected_records_by_client[client_id] = selected_client_records
#             selected_records.extend(selected_client_records)

#         for client_id in range(self.num_clients):
#             client_records = selected_records_by_client.get(client_id, [])
#             if not client_records:
#                 continue

#             selected_indices = [
#                 int(record["dataset_index"]) for record in client_records
#             ]
#             selected_weights = [
#                 float(record.get("sampling_weight", 1.0)) for record in client_records
#             ]
#             self.retained_datasets[client_id].append(
#                 IndexedSelectionDataset(
#                     current_datasets[client_id],
#                     selected_indices,
#                     sampling_weights=selected_weights,
#                 )
#             )

#         leverage_plot = self.figure_dir / f"{self.run_name}_task{task_id}_leverage.png"
#         buffer_plot = (
#             self.figure_dir / f"{self.run_name}_task{task_id}_buffer_distribution.png"
#         )
#         leverage_title = f"{self.run_name} task {task_id} leverage scores"
#         try:
#             plot_leverage_scores(
#                 result.records,
#                 leverage_plot,
#                 title=leverage_title,
#                 selected_records=selected_records,
#                 use_raw_score=True,
#                 log_scale=False,
#             )
#         except TypeError as exc:
#             if "unexpected keyword argument" not in str(exc):
#                 raise
#             plot_leverage_scores(
#                 result.records,
#                 leverage_plot,
#                 title=leverage_title,
#             )
#         plot_buffer_class_distribution(
#             self._retained_summaries(),
#             buffer_plot,
#             title=f"{self.run_name} task {task_id} buffer class distribution",
#         )

#         return {
#             "rank": result.rank,
#             "class_wise": False,
#             "singular_values": [],
#             "client_singular_values": result.client_singular_values,
#             "class_singular_values": None,
#             "num_scored_samples": len(result.records),
#             "num_selected_samples": len(selected_records),
#             "global_sampling_budget": total_budget,
#             "per_client_sampling_budgets": {
#                 f"client_{client_id}": budget
#                 for client_id, budget in per_client_budgets.items()
#             },
#             "selection_mode": "per_client_class_balanced_sampling",
#             "candidate_pool": "current_task_only",
#             "encryption": "P_k_X_Q",
#             "sampling_weight_formula": "per_client_class_leverage_probability",
#             "leverage_plot": str(leverage_plot),
#             "buffer_distribution_plot": str(buffer_plot),
#         }

#     def _retained_summaries(self) -> dict[str, dict]:
#         summaries: dict[str, dict] = {}
#         for client_id, datasets in enumerate(self.retained_datasets):
#             counts: Counter[int] = Counter()
#             total = 0
#             for dataset in datasets:
#                 total += len(dataset)
#                 for index in range(len(dataset)):
#                     label = int(dataset[index][1])
#                     counts[label] += 1
#             summaries[f"client_{client_id}"] = {
#                 "capacity": self.buffer_size,
#                 "num_samples": total,
#                 "class_counts": dict(sorted(counts.items())),
#             }
#         return summaries

#     def _buffer_summaries(self) -> dict[str, dict]:
#         return self._retained_summaries()

from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from src.data.weighted_dataset import SampleWeightDataset
from src.federated.aggregator import fedavg
from src.federated.client import Client
from src.gdr.features import build_client_feature_payload, sample_orthogonal_matrix
from src.gdr.server import (
    compute_leverage_scores,
    group_records_by_client,
    sample_records_by_class_probability,
)
from src.methods.base_method import BaseMethod


class IndexedSelectionDataset(Dataset):
    """
    Dataset view backed by an explicit list of indices.

    Unlike torch.utils.data.Subset, this keeps duplicate indices, which matches
    the original B.py sampling behavior where official GDR sampling is done with
    replacement.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        sampling_weights: list[float] | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.sampling_weights = (
            [float(weight) for weight in sampling_weights]
            if sampling_weights is not None
            else None
        )
        self.records = [dict(record) for record in records] if records is not None else None
        if self.sampling_weights is not None and len(self.sampling_weights) != len(
            self.indices
        ):
            raise ValueError("sampling_weights must match indices length")
        if self.records is not None and len(self.records) != len(self.indices):
            raise ValueError("records must match indices length")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def get_sampling_weight(self, index: int) -> float:
        if self.sampling_weights is not None:
            return float(self.sampling_weights[index])

        if self.records is not None:
            weight = self.records[index].get("sampling_weight")
            if weight is not None:
                return float(weight)

        getter = getattr(self.dataset, "get_sampling_weight", None)
        if callable(getter):
            return float(getter(self.indices[index]))
        return 1.0

    def get_label(self, index: int) -> int:
        if self.records is not None:
            label = self.records[index].get("label")
            if label is not None:
                return int(label)
        return int(self.dataset[self.indices[index]][1])

    def get_raw_leverage_score(self, index: int) -> float:
        if self.records is not None:
            record = self.records[index]
            return float(
                record.get(
                    "raw_leverage_score",
                    record.get("leverage_score", 0.0),
                )
            )
        return 0.0

    def get_record(self, index: int) -> dict[str, Any]:
        if self.records is not None:
            return dict(self.records[index])
        return {
            "dataset_index": int(self.indices[index]),
            "label": self.get_label(index),
            "sampling_weight": self.get_sampling_weight(index),
            "raw_leverage_score": self.get_raw_leverage_score(index),
        }


class LocalReplayGDRPaper(BaseMethod):
    """
    GDR-only variant abstracted from the paper TTS branch.
    """

    method_name = "local_replay_gdr_paper"

    def __init__(
        self,
        *args,
        buffer_size: int = 200,
        samples_per_task: int | None = None,
        seed: int = 0,
        gdr_rank: int = 8,
        gdr_feature_samples: int | None = None,
        gdr_class_wise: bool = False,
        figure_dir: str | Path = "outputs/figures/gdr",
        run_name: str = "local_replay_gdr_paper",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.samples_per_task = samples_per_task
        self.seed = seed
        self.gdr_rank = gdr_rank
        self.gdr_feature_samples = gdr_feature_samples
        self.gdr_class_wise = gdr_class_wise
        self.figure_dir = Path(figure_dir)
        self.run_name = run_name
        self.clients = [Client(self.device) for _ in range(self.num_clients)]
        self.retained_datasets: list[list[Dataset]] = [[] for _ in range(self.num_clients)]

    def train(self) -> dict:
        print("Start Federated Local Replay + GDR (paper branch)...")
        history = {
            "method": self.method_name,
            "buffer_size": self.buffer_size,
            "samples_per_task": self.samples_per_task,
            "gdr_rank": self.gdr_rank,
            "gdr_feature_samples": self.gdr_feature_samples,
            "gdr_class_wise": self.gdr_class_wise,
            "gdr_candidate_pool": "current_task_only",
            "gdr_selection_mode": "global_class_balanced_sampling",
            "gdr_encryption": "P_k_X_Q",
            "replay_weighting": "sampling_weighted_ce",
            "tasks": [],
        }

        for task_id in range(self.dataset_manager.num_tasks):
            print(f"\n=== Task {task_id} ===")

            old_classes = self.model.num_classes
            task_classes = self.dataset_manager.get_task_classes(task_id)
            self.model.expand_head(len(task_classes))
            self.model.to(self.device)
            task_history = {
                "task_id": task_id,
                "task_classes": task_classes,
                "old_classes": old_classes,
                "new_classes": len(task_classes),
                "rounds": [],
                "buffer_before": self._retained_summaries(),
            }

            for round_id in range(self.rounds):
                local_states = []
                sample_counts = []
                losses = []
                #round_lr = self._round_lr(round_id)
                round_lr = self.lr

                for client_id in range(self.num_clients):
                    current_subset = self.dataset_manager.get_train_subset(
                        task_id, client_id
                    )
                    train_dataset, sample_weights = self._compose_train_dataset(
                        current_subset,
                        client_id,
                    )
                    if len(train_dataset) < 2:
                        continue

                    sampler = None
                    if sample_weights is not None:
                        sampler = WeightedRandomSampler(
                            weights=sample_weights,
                            num_samples=len(train_dataset),
                            replacement=True,
                        )

                    local_model = copy.deepcopy(self.model)
                    state, loss = self.clients[client_id].train(
                        model=local_model,
                        dataset=train_dataset,
                        batch_size=self.batch_size,
                        epochs=self.local_epochs,
                        lr=round_lr,
                        sampler=sampler,
                    )

                    local_states.append(state)
                    sample_counts.append(len(train_dataset))
                    losses.append(loss)

                if not local_states:
                    raise RuntimeError(f"No client data found for task {task_id}.")

                total_samples = sum(sample_counts)
                weights = [1.0 / len(local_states)] * len(local_states)
                self.model.load_state_dict(fedavg(local_states, weights))

                test_set = self.dataset_manager.get_seen_test_subset(task_id)
                acc = self.evaluate(self.model, test_set)
                avg_loss = sum(losses) / len(losses)
                print(
                    f"[Task {task_id}][Round {round_id}] "
                    f"Loss: {avg_loss:.4f} Acc: {acc:.4f}"
                )
                task_history["rounds"].append(
                    {
                        "round_id": round_id,
                        "loss": avg_loss,
                        "seen_acc": acc,
                        "num_clients": len(local_states),
                        "num_samples": total_samples,
                    }
                )

            final_test = self.dataset_manager.get_seen_test_subset(task_id)
            final_acc = self.evaluate(self.model, final_test)
            print(f"[Task {task_id}] Final Acc: {final_acc:.4f}")
            task_history["final_seen_acc"] = final_acc
            gdr_result = self._update_retained_datasets_with_gdr(
                task_id=task_id,
                new_classes=len(task_classes),
            )
            task_history["buffer_after"] = self._retained_summaries()
            task_history["gdr"] = gdr_result
            history["tasks"].append(task_history)

        history["buffer_stats"] = self._retained_summaries()
        return history

    def _compose_train_dataset(
        self,
        current_subset: Dataset,
        client_id: int,
    ) -> tuple[Dataset, list[float] | None]:
        retained = self.retained_datasets[client_id]
        if not retained:
            return (
                SampleWeightDataset(
                    current_subset,
                    default_weight=1.0,
                ),
                None,
            )
        full = ConcatDataset(
            [
                SampleWeightDataset(
                    current_subset,
                    default_weight=1.0,
                ),
                *[
                    SampleWeightDataset(
                        retained_dataset,
                    )
                    for retained_dataset in retained
                ],
            ]
        )
        return full, None

    def _global_sampling_budget(self, new_classes: int) -> int:
        if self.samples_per_task is not None:
            return self.samples_per_task * self.num_clients
        return new_classes * self.buffer_size

    def _per_client_sampling_budgets(self, total_budget: int) -> dict[int, int]:
        total_budget = max(int(total_budget), 0)
        if self.num_clients <= 0:
            return {}

        base_quota = total_budget // self.num_clients
        remainder = total_budget % self.num_clients
        return {
            client_id: base_quota + (1 if client_id < remainder else 0)
            for client_id in range(self.num_clients)
        }

    def _round_lr(self, round_id: int, eta_min: float = 1e-3) -> float:
        if self.rounds <= 1:
            return self.lr
        cosine = (1.0 + math.cos(math.pi * round_id / self.rounds)) / 2.0
        return eta_min + (self.lr - eta_min) * cosine

    def _update_retained_datasets_with_gdr(
        self,
        task_id: int,
        new_classes: int,
    ) -> dict:
        from src.gdr.visualize import plot_buffer_class_distribution, plot_leverage_scores

        if self.gdr_class_wise:
            raise ValueError(
                "Paper GDR with P_k / Q encryption currently does not support "
                "class-wise leverage computation. Use --no-gdr_class_wise."
            )

        payloads = []
        current_datasets = {}
        right_transform = sample_orthogonal_matrix(
            self.model.feature_dim,
            seed=self.seed + task_id * 1000 + 999_999,
        )

        for client_id in range(self.num_clients):
            current_subset = self.dataset_manager.get_train_subset(task_id, client_id)
            current_datasets[client_id] = current_subset
            payloads.append(
                build_client_feature_payload(
                    current_subset,
                    client_id=client_id,
                    task_id=task_id,
                    feature_extractor=self.model.backbone,
                    device=self.device,
                    max_samples=self.gdr_feature_samples,
                    seed=self.seed + task_id * 1000 + client_id,
                    right_transform=right_transform,
                    left_transform_seed=self.seed + task_id * 10_000 + client_id,
                )
            )

        result = compute_leverage_scores(
            payloads,
            rank=self.gdr_rank,
            class_wise=False,
        )
        total_budget = self._global_sampling_budget(new_classes)
        selected_records = sample_records_by_class_probability(
            result.records,
            total_budget=total_budget,
            seed=self.seed + task_id,
        )
        selected_records_by_client = group_records_by_client(selected_records)

        for client_id in range(self.num_clients):
            client_records = selected_records_by_client.get(client_id, [])
            if not client_records:
                continue

            selected_indices = [
                int(record["dataset_index"]) for record in client_records
            ]
            selected_weights = [
                float(record.get("sampling_weight", 1.0)) for record in client_records
            ]

            # Guard against accidental mismatches when global records are mapped
            # back to each client's local dataset indices.
            for record in client_records:
                dataset_index = int(record["dataset_index"])
                actual_label = int(current_datasets[client_id][dataset_index][1])
                expected_label = int(record["label"])
                if actual_label != expected_label:
                    raise RuntimeError(
                        "Index mismatch while rebuilding retained datasets: "
                        f"client={client_id} idx={dataset_index} "
                        f"actual_label={actual_label} expected={expected_label}"
                    )

            self.retained_datasets[client_id].append(
                IndexedSelectionDataset(
                    current_datasets[client_id],
                    selected_indices,
                    sampling_weights=selected_weights,
                    records=client_records,
                )
            )
            #self._enforce_retained_capacity(client_id)

        leverage_plot = self.figure_dir / f"{self.run_name}_task{task_id}_leverage.png"
        buffer_plot = (
            self.figure_dir / f"{self.run_name}_task{task_id}_buffer_distribution.png"
        )
        leverage_title = f"{self.run_name} task {task_id} leverage scores"
        try:
            plot_leverage_scores(
                result.records,
                leverage_plot,
                title=leverage_title,
                selected_records=selected_records,
                use_raw_score=True,
                log_scale=False,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            plot_leverage_scores(
                result.records,
                leverage_plot,
                title=leverage_title,
            )
        plot_buffer_class_distribution(
            self._retained_summaries(),
            buffer_plot,
            title=f"{self.run_name} task {task_id} buffer class distribution",
        )

        return {
            "rank": result.rank,
            "class_wise": False,
            "singular_values": result.singular_values,
            "client_singular_values": None,
            "class_singular_values": None,
            "num_scored_samples": len(result.records),
            "num_selected_samples": len(selected_records),
            "global_sampling_budget": total_budget,
            "selection_mode": "global_class_balanced_sampling",
            "candidate_pool": "current_task_only",
            "encryption": "P_k_X_Q",
            "sampling_weight_formula": "global_class_leverage_probability",
            "leverage_plot": str(leverage_plot),
            "buffer_distribution_plot": str(buffer_plot),
        }

    def _enforce_retained_capacity(self, client_id: int) -> None:
        capacity = max(int(self.buffer_size), 0)
        if capacity == 0:
            self.retained_datasets[client_id] = []
            return

        flattened: list[dict[str, Any]] = []
        for retained_dataset in self.retained_datasets[client_id]:
            if isinstance(retained_dataset, IndexedSelectionDataset):
                for local_index, dataset_index in enumerate(retained_dataset.indices):
                    flattened.append(
                        {
                            "dataset": retained_dataset.dataset,
                            "dataset_index": int(dataset_index),
                            "label": retained_dataset.get_label(local_index),
                            "sampling_weight": retained_dataset.get_sampling_weight(
                                local_index
                            ),
                            "raw_leverage_score": retained_dataset.get_raw_leverage_score(
                                local_index
                            ),
                            "record": retained_dataset.get_record(local_index),
                        }
                    )
                continue

            getter = getattr(retained_dataset, "get_sampling_weight", None)
            for local_index in range(len(retained_dataset)):
                label = int(retained_dataset[local_index][1])
                sampling_weight = float(getter(local_index)) if callable(getter) else 1.0
                flattened.append(
                    {
                        "dataset": retained_dataset,
                        "dataset_index": local_index,
                        "label": label,
                        "sampling_weight": sampling_weight,
                        "raw_leverage_score": 0.0,
                        "record": {
                            "dataset_index": local_index,
                            "label": label,
                            "sampling_weight": sampling_weight,
                            "raw_leverage_score": 0.0,
                        },
                    }
                )

        if len(flattened) <= capacity:
            return

        by_class: dict[int, list[dict[str, Any]]] = {}
        for entry in flattened:
            by_class.setdefault(int(entry["label"]), []).append(entry)

        for class_entries in by_class.values():
            class_entries.sort(
                key=lambda entry: float(entry["raw_leverage_score"]),
                reverse=True,
            )

        classes = sorted(by_class)
        kept: list[dict[str, Any]] = []
        while len(kept) < capacity and classes:
            next_classes = []
            for label in classes:
                class_entries = by_class[label]
                if class_entries and len(kept) < capacity:
                    kept.append(class_entries.pop(0))
                if class_entries:
                    next_classes.append(label)
            classes = next_classes

        grouped: dict[int, dict[str, Any]] = {}
        group_order: list[int] = []
        for entry in kept:
            dataset = entry["dataset"]
            group_key = id(dataset)
            if group_key not in grouped:
                grouped[group_key] = {
                    "dataset": dataset,
                    "indices": [],
                    "sampling_weights": [],
                    "records": [],
                }
                group_order.append(group_key)

            record = dict(entry["record"])
            record["dataset_index"] = int(entry["dataset_index"])
            record["label"] = int(entry["label"])
            record["sampling_weight"] = float(entry["sampling_weight"])
            record["raw_leverage_score"] = float(entry["raw_leverage_score"])
            grouped[group_key]["indices"].append(int(entry["dataset_index"]))
            grouped[group_key]["sampling_weights"].append(float(entry["sampling_weight"]))
            grouped[group_key]["records"].append(record)

        self.retained_datasets[client_id] = [
            IndexedSelectionDataset(
                grouped[group_key]["dataset"],
                grouped[group_key]["indices"],
                sampling_weights=grouped[group_key]["sampling_weights"],
                records=grouped[group_key]["records"],
            )
            for group_key in group_order
        ]

    def _retained_summaries(self) -> dict[str, dict]:
        summaries: dict[str, dict] = {}
        for client_id, datasets in enumerate(self.retained_datasets):
            counts: Counter[int] = Counter()
            total = 0
            for dataset in datasets:
                total += len(dataset)
                for index in range(len(dataset)):
                    label = int(dataset[index][1])
                    counts[label] += 1
            summaries[f"client_{client_id}"] = {
                "capacity": self.buffer_size,
                "num_samples": total,
                "class_counts": dict(sorted(counts.items())),
            }
        return summaries

    def _buffer_summaries(self) -> dict[str, dict]:
        return self._retained_summaries()
