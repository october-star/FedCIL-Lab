from __future__ import annotations

import copy
from pathlib import Path

from torch.utils.data import ConcatDataset

from src.data.weighted_dataset import SampleWeightDataset
from src.federated.aggregator import fedavg
from src.federated.client import Client
from src.gdr.features import build_client_feature_payload, sample_orthogonal_matrix
from src.gdr.server import (
    compute_client_local_leverage_scores,
    group_records_by_client,
    sample_records_official_style,
)
from src.methods.base_method import BaseMethod
from src.replay.buffer import ReplayBuffer


class LocalReplayGDRPaper(BaseMethod):
    """
    Paper-faithful GDR replay variant.

    This follows the Algorithm 1 structure more closely than the class-wise
    replacement branch:
    - training uses current task data plus the historical replay buffer
    - GDR scores only the current task subset
    - the server globally samples IDs from the scored current-task pool
    - selected records are appended for future replay
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
        self.buffers = [
            ReplayBuffer(capacity=buffer_size, seed=seed + client_id)
            for client_id in range(self.num_clients)
        ]

    def train(self) -> dict:
        print("Start Federated Local Replay + GDR (paper branch)...")
        history = {
            "method": "local_replay_gdr_paper",
            "buffer_size": self.buffer_size,
            "samples_per_task": self.samples_per_task,
            "gdr_rank": self.gdr_rank,
            "gdr_feature_samples": self.gdr_feature_samples,
            "gdr_class_wise": self.gdr_class_wise,
            "gdr_candidate_pool": "current_task_only",
            "gdr_selection_mode": "official_client_balanced_sampling",
            "gdr_encryption": "P_k_X_Q",
            "replay_weighting": "uniform",
            "tasks": [],
        }

        for task_id in range(self.dataset_manager.num_tasks):
            print(f"\n=== Task {task_id} ===")

            task_classes = self.dataset_manager.get_task_classes(task_id)
            self.model.expand_head(len(task_classes))
            self.model.to(self.device)
            task_history = {
                "task_id": task_id,
                "task_classes": task_classes,
                "rounds": [],
                "buffer_before": self._buffer_summaries(),
            }

            for round_id in range(self.rounds):
                local_states = []
                sample_counts = []
                losses = []

                for client_id in range(self.num_clients):
                    current_subset = self.dataset_manager.get_train_subset(
                        task_id, client_id
                    )
                    train_dataset = self._compose_train_dataset(
                        current_subset=current_subset,
                        buffer=self.buffers[client_id],
                    )
                    if len(train_dataset) < 2:
                        continue

                    local_model = copy.deepcopy(self.model)
                    state, loss = self.clients[client_id].train(
                        model=local_model,
                        dataset=train_dataset,
                        batch_size=self.batch_size,
                        epochs=self.local_epochs,
                        lr=self.lr,
                    )

                    local_states.append(state)
                    sample_counts.append(len(train_dataset))
                    losses.append(loss)

                if not local_states:
                    raise RuntimeError(f"No client data found for task {task_id}.")

                total_samples = sum(sample_counts)
                weights = [count / total_samples for count in sample_counts]
                self.model.load_state_dict(fedavg(local_states, weights))

                if round_id % 5 == 0 or round_id == self.rounds - 1:
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
            gdr_result = self._update_buffers_with_gdr(task_id)
            task_history["buffer_after"] = self._buffer_summaries()
            task_history["gdr"] = gdr_result
            history["tasks"].append(task_history)

        history["buffer_stats"] = self._buffer_summaries()
        return history

    def _compose_train_dataset(self, current_subset, buffer: ReplayBuffer):
        if len(buffer) == 0:
            return SampleWeightDataset(current_subset, default_weight=1.0)
        return ConcatDataset(
            [
                SampleWeightDataset(current_subset, default_weight=1.0),
                SampleWeightDataset(buffer),
            ]
        )

    def _global_sampling_budget(self) -> int:
        if self.samples_per_task is not None:
            return self.samples_per_task * self.num_clients
        return self.buffer_size * self.num_clients

    def _update_buffers_with_gdr(self, task_id: int) -> dict:
        from src.gdr.visualize import plot_buffer_class_distribution, plot_leverage_scores

        if self.gdr_class_wise:
            raise ValueError(
                "Paper GDR with P_k / Q encryption currently supports only global "
                "sampling. Use --no-gdr_class_wise."
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
        selected_records = sample_records_official_style(
            result.records,
            total_budget=self._global_sampling_budget(),
            num_clients=self.num_clients,
            seed=self.seed + task_id,
        )
        records_by_client = group_records_by_client(selected_records)

        for client_id, buffer in enumerate(self.buffers):
            buffer.add_selected_records(
                current_datasets[client_id],
                records=records_by_client.get(client_id, []),
                task_id=task_id,
                client_id=client_id,
            )

        leverage_plot = self.figure_dir / f"{self.run_name}_task{task_id}_leverage.png"
        buffer_plot = (
            self.figure_dir / f"{self.run_name}_task{task_id}_buffer_distribution.png"
        )
        plot_leverage_scores(
            result.records,
            leverage_plot,
            title=f"{self.run_name} task {task_id} leverage scores",
        )
        plot_buffer_class_distribution(
            self._buffer_summaries(),
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
            "num_selected_samples": len(selected_records),
            "global_sampling_budget": self._global_sampling_budget(),
            "selection_mode": "official_client_balanced_sampling",
            "candidate_pool": "current_task_only",
            "encryption": "P_k_X_Q",
            "sampling_weight_formula": "uniform",
            "leverage_plot": str(leverage_plot),
            "buffer_distribution_plot": str(buffer_plot),
        }

    def _buffer_summaries(self) -> dict[str, dict]:
        return {
            f"client_{client_id}": buffer.summary()
            for client_id, buffer in enumerate(self.buffers)
        }
