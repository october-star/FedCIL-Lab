from __future__ import annotations

import copy

from src.federated.aggregator import fedavg
from src.federated.client import Client
from src.methods.base_method import BaseMethod


class Finetune(BaseMethod):
    """
    Federated finetuning baseline.

    Each task trains only on the current task's client data. The classifier head
    is expanded as new classes arrive, and evaluation uses all seen classes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clients = [Client(self.device) for _ in range(self.num_clients)]

    def train(self) -> dict:
        print("Start Federated Finetune Baseline...")
        history = {
            "method": "finetune",
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
            }

            for round_id in range(self.rounds):
                local_states = []
                sample_counts = []
                losses = []

                for client_id in range(self.num_clients):
                    subset = self.dataset_manager.get_train_subset(task_id, client_id)
                    if len(subset) < 2:
                        continue

                    local_model = copy.deepcopy(self.model)

                    state, loss = self.clients[client_id].train(
                        model=local_model,
                        dataset=subset,
                        batch_size=self.batch_size,
                        epochs=self.local_epochs,
                        lr=self.lr,
                    )

                    local_states.append(state)
                    sample_counts.append(len(subset))
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

            final_test = self.dataset_manager.get_seen_test_subset(task_id)
            final_acc = self.evaluate(self.model, final_test)
            print(f"[Task {task_id}] Final Acc: {final_acc:.4f}")
            task_history["final_seen_acc"] = final_acc
            history["tasks"].append(task_history)

        return history
