from __future__ import annotations

import copy

from torch.utils.data import ConcatDataset

from src.federated.aggregator import fedavg
from src.federated.client import Client
from src.methods.base_method import BaseMethod
from src.replay.buffer import ReplayBuffer


class LocalReplay(BaseMethod):
    """
    Federated local replay baseline.

    Each client trains on its current task subset plus its own replay buffer.
    After a task finishes, current-task samples are inserted into that client's
    buffer for future tasks.
    """

    def __init__(
        self,
        *args,
        buffer_size: int = 200,
        samples_per_task: int | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self.samples_per_task = samples_per_task
        self.clients = [Client(self.device) for _ in range(self.num_clients)]
        self.buffers = [
            ReplayBuffer(capacity=buffer_size, seed=seed + client_id)
            for client_id in range(self.num_clients)
        ]

    def train(self) -> dict:
        print("Start Federated Local Replay Baseline...")
        history = {
            "method": "local_replay",
            "buffer_size": self.buffer_size,
            "samples_per_task": self.samples_per_task,
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
                new_state = fedavg(local_states, weights)
                self.model.load_state_dict(new_state)

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

            self._update_buffers(task_id)

            final_test = self.dataset_manager.get_seen_test_subset(task_id)
            final_acc = self.evaluate(self.model, final_test)
            print(f"[Task {task_id}] Final Acc: {final_acc:.4f}")
            task_history["final_seen_acc"] = final_acc
            task_history["buffer_after"] = self._buffer_summaries()
            history["tasks"].append(task_history)

        history["buffer_stats"] = self._buffer_summaries()
        return history

    def _compose_train_dataset(self, current_subset, buffer: ReplayBuffer):
        if len(buffer) == 0:
            return current_subset
        return ConcatDataset([current_subset, buffer])

    def _update_buffers(self, task_id: int) -> None:
        for client_id, buffer in enumerate(self.buffers):
            current_subset = self.dataset_manager.get_train_subset(task_id, client_id)
            buffer.add_dataset(
                current_subset,
                task_id=task_id,
                client_id=client_id,
                max_samples=self.samples_per_task,
            )

    def _buffer_summaries(self) -> dict[str, dict]:
        return {
            f"client_{client_id}": buffer.summary()
            for client_id, buffer in enumerate(self.buffers)
        }
