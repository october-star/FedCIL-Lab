# Author: mmj
# DATE: 24.05.2026
from __future__ import annotations

import copy
import math
from collections import Counter
from functools import partial
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from src.federated.aggregator import fedavg
from src.federated.client import Client
from src.gdr.features import build_client_feature_payload, sample_orthogonal_matrix
from src.gdr.server import (
    compute_client_local_leverage_scores,
    group_records_by_client,
    sample_records_by_class_probability,
)
from src.methods.base_method import BaseMethod
from src.tts.loss import tts_cross_entropy
from src.utils.kl_forgetting_utils import compute_class_kl_forgetting
from src.replay.selection_dataset import IndexedSelectionDataset

class ReplayViewDataset(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dataset, item_index = self.items[index]
        return dataset[item_index]

class CBDRKlAwareAdaptiveReplay(BaseMethod):
    """
    Paper-faithful GDR + TTS replay variant.
    """

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
        run_name: str = "cbdr_kl_aware_adaptive",
        old_temp: float = 0.9,
        new_temp: float = 1.1,
        old_weight: float = 1.1,
        new_weight: float = 0.9,
        adaptive_replay: bool = False,
        adaptive_replay_gamma: float = 1.0,
        adaptive_replay_min_weight: float = 0.5,
        adaptive_replay_max_weight: float = 2.0,
        replay_sampling_mass: float = 0.5,
        kl_temperature: float = 2.0,
        kl_max_samples_per_class: int = 100,
        candidate_pool_multiplier: float = 2.0,
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
        self.old_temp = old_temp
        self.new_temp = new_temp
        self.old_weight = old_weight
        self.new_weight = new_weight
        self.clients = [Client(self.device) for _ in range(self.num_clients)]
        self.retained_datasets: list[list[Dataset]] = [
            [] for _ in range(self.num_clients)
        ]
        self.adaptive_replay = adaptive_replay
        self.adaptive_replay_gamma = adaptive_replay_gamma
        self.adaptive_replay_min_weight = adaptive_replay_min_weight
        self.adaptive_replay_max_weight = adaptive_replay_max_weight
        self.replay_sampling_mass = replay_sampling_mass
        self.kl_temperature = kl_temperature
        self.kl_max_samples_per_class = kl_max_samples_per_class
        self.class_kl_scores: dict[int, float] = {}
        self.candidate_pool_multiplier = candidate_pool_multiplier
        self.candidate_datasets: list[list[Dataset]] = [
            [] for _ in range(self.num_clients)
        ]

    def train(self) -> dict:
        print("Start Federated CBDR Adaptive Replay...")
        history = {
            "method": "cbdr_kl_aware_adaptive",
            "buffer_size": self.buffer_size,
            "samples_per_task": self.samples_per_task,
            "gdr_rank": self.gdr_rank,
            "gdr_feature_samples": self.gdr_feature_samples,
            "gdr_class_wise": self.gdr_class_wise,
            "gdr_candidate_pool": "current_task_only",
            "gdr_selection_mode": "per_client_class_balanced_sampling",
            "gdr_encryption": "P_k_X_Q",
            "replay_weighting": "uniform",
            "tts": {
                "old_temp": self.old_temp,
                "new_temp": self.new_temp,
                "old_weight": self.old_weight,
                "new_weight": self.new_weight,
            },
            "tasks": [],
            "adaptive_replay": {
                "enabled": self.adaptive_replay,
                "gamma": self.adaptive_replay_gamma,
                "min_weight": self.adaptive_replay_min_weight,
                "max_weight": self.adaptive_replay_max_weight,
                "replay_sampling_mass": self.replay_sampling_mass,
            },
            "kl_forgetting": {
                "temperature": self.kl_temperature,
                "max_samples_per_class": self.kl_max_samples_per_class,
            },
        }

        for task_id in range(self.dataset_manager.num_tasks):
            print(f"\n=== Task {task_id} ===")

            old_classes = self.model.num_classes

            # store old model
            teacher_model = None
            if task_id > 0:
                teacher_model = copy.deepcopy(self.model)
                teacher_model.to(self.device)
                teacher_model.eval()

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
                "adaptive_replay": {
                    "enabled": self.adaptive_replay,
                    "class_kl_scores_before_training": dict(self.class_kl_scores),
                    "gamma": self.adaptive_replay_gamma,
                },
            }

            loss_fn = None
            if task_id > 0 :
                loss_fn = partial(
                    tts_cross_entropy,
                    old_classes=old_classes,
                    old_temp=self.old_temp,
                    new_temp=self.new_temp,
                    old_weight=self.old_weight,
                    new_weight=self.new_weight,
                )

            for round_id in range(self.rounds):
                local_states = []
                sample_counts = []
                losses = []
                round_lr = self._round_lr(round_id)

                for client_id in range(self.num_clients):
                    current_subset = self.dataset_manager.get_train_subset(
                        task_id, client_id
                    )
                    train_dataset, sample_weights = self._compose_train_dataset(current_subset, client_id)
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
                        loss_fn=loss_fn,
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
            # add history
            task_history["final_seen_acc"] = final_acc
            kl_result = {
                "class_kl_scores_after_training": {},
                "num_replay_classes": 0,
            }

            if task_id > 0 and teacher_model is not None:
                self.class_kl_scores = compute_class_kl_forgetting(
                    teacher_model=teacher_model,
                    student_model=self.model,
                    retained_datasets=self.retained_datasets,
                    old_classes=old_classes,
                    device=self.device,
                    temperature=self.kl_temperature,
                    max_samples_per_class=self.kl_max_samples_per_class,
                )

                kl_result = {
                    "class_kl_scores_after_training": dict(self.class_kl_scores),
                    "num_replay_classes": len(self.class_kl_scores),
                }

            task_history["adaptive_replay"] = {
                "enabled": self.adaptive_replay,
                "gamma": self.adaptive_replay_gamma,
                "min_weight": self.adaptive_replay_min_weight,
                "max_weight": self.adaptive_replay_max_weight,
                "replay_sampling_mass": self.replay_sampling_mass,
                **kl_result,
            }
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
            return current_subset, None

        return ConcatDataset([current_subset, *retained]), None

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

    def _update_retained_datasets_with_gdr(self, task_id: int, new_classes: int) -> dict:
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

        result = compute_client_local_leverage_scores(
            payloads,
            rank=self.gdr_rank,
        )
        active_budget = self._global_sampling_budget(new_classes)
        candidate_budget = int(
            active_budget * self.candidate_pool_multiplier
        )
        per_client_budgets = self._per_client_sampling_budgets(candidate_budget)
        records_by_client = group_records_by_client(result.records)
        selected_records: list[dict] = []
        selected_records_by_client: dict[int, list[dict]] = {}

        for client_id in range(self.num_clients):
            selected_client_records = sample_records_by_class_probability(
                records_by_client.get(client_id, []),
                total_budget=per_client_budgets.get(client_id, 0),
                seed=self.seed + task_id * 1000 + client_id,
            )
            selected_records_by_client[client_id] = selected_client_records
            selected_records.extend(selected_client_records)

        for client_id in range(self.num_clients):
            client_records = selected_records_by_client.get(client_id, [])
            if not client_records:
                continue

            selected_indices = [
                int(record["dataset_index"]) for record in client_records
            ]
            self.candidate_datasets[client_id].append(
                IndexedSelectionDataset(
                    current_datasets[client_id],
                    selected_indices,
                )
            )

        rebuild_result = self._rebuild_active_retained_from_candidates(
            new_classes=new_classes,
        )

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
            "singular_values": [],
            "client_singular_values": result.client_singular_values,
            "class_singular_values": None,

            "num_scored_samples": len(result.records),

            # candidate pool
            "num_candidate_samples": len(selected_records),
            "candidate_budget": candidate_budget,
            "candidate_pool_multiplier":
                self.candidate_pool_multiplier,

            # active replay buffer
            "num_selected_samples": sum(
                stat["num_selected"]
                for stat in rebuild_result["client_stats"].values()
            ),
            "global_sampling_budget": active_budget,

            "per_client_sampling_budgets": {
                f"client_{client_id}": budget
                for client_id, budget
                in self._per_client_sampling_budgets(
                    active_budget
                ).items()
            },

            "selection_mode":
                "per_client_class_balanced_sampling",

            "candidate_pool":
                "current_task_large_pool",

            "active_buffer":
                "kl_aware_rebuild",

            "adaptive_replay": {
                "enabled": self.adaptive_replay,
                "gamma": self.adaptive_replay_gamma,
                "min_weight":
                    self.adaptive_replay_min_weight,
                "max_weight":
                    self.adaptive_replay_max_weight,
                "replay_sampling_mass":
                    self.replay_sampling_mass,
            },

            "kl_rebuild": rebuild_result,

            "encryption": "P_k_X_Q",

            "sampling_weight_formula":
                "per_client_class_leverage_probability",

            "leverage_plot": str(leverage_plot),
            "buffer_distribution_plot":
                str(buffer_plot),
        }

    def _class_weight(self, class_id: int) -> float:
        score = max(float(self.class_kl_scores.get(class_id, 0.0)), 0.0)
        weight = 1.0 + self.adaptive_replay_gamma * score
        return max(
            self.adaptive_replay_min_weight,
            min(self.adaptive_replay_max_weight, weight),
        )

    def _rebuild_active_retained_from_candidates(self, new_classes: int) -> dict:
        total_budget = self._global_sampling_budget(new_classes)
        per_client_budgets = self._per_client_sampling_budgets(total_budget)

        rebuilt: list[list[Dataset]] = [[] for _ in range(self.num_clients)]
        stats = {}

        for client_id in range(self.num_clients):
            budget = per_client_budgets.get(client_id, 0)
            candidates = []

            for dataset in self.candidate_datasets[client_id]:
                for index in range(len(dataset)):
                    item = dataset[index]
                    label = int(item[-1])
                    candidates.append((dataset, index, label))

            if budget <= 0 or not candidates:
                continue

            by_class: dict[int, list[tuple[Dataset, int, int]]] = {}
            for item in candidates:
                by_class.setdefault(item[2], []).append(item)

            class_weights = {
                class_id: self._class_weight(class_id)
                for class_id in by_class
            }

            total_weight = sum(class_weights.values())
            quotas = {
                class_id: max(
                    1,
                    int(round(budget * class_weights[class_id] / total_weight)),
                )
                for class_id in by_class
            }

            while sum(quotas.values()) > budget:
                class_id = max(quotas, key=quotas.get)
                if quotas[class_id] > 1:
                    quotas[class_id] -= 1
                else:
                    break

            while sum(quotas.values()) < budget:
                class_id = max(class_weights, key=class_weights.get)
                quotas[class_id] += 1

            selected = []
            for class_id, items in by_class.items():
                items = sorted(
                    items,
                    key=lambda x: (
                        x[0].get_sample_id(x[1])
                        if hasattr(x[0], "get_sample_id")
                        else x[1]
                    ),
                )
                quota = min(quotas[class_id], len(items))
                selected.extend(items[:quota])

            rebuilt[client_id].append(
                ReplayViewDataset(
                    [(dataset, index) for dataset, index, _ in selected]
                )
            )

            class_counts = Counter(label for _, _, label in selected)
            stats[f"client_{client_id}"] = {
                "budget": budget,
                "num_candidates": len(candidates),
                "num_selected": len(selected),
                "class_weights": {
                    str(k): float(v) for k, v in class_weights.items()
                },
                "class_quotas": {
                    str(k): int(v) for k, v in quotas.items()
                },
                "class_counts": dict(sorted(class_counts.items())),
            }

        self.retained_datasets = rebuilt

        return {
            "total_active_budget": total_budget,
            "per_client_budgets": {
                f"client_{k}": v for k, v in per_client_budgets.items()
            },
            "client_stats": stats,
        }

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
