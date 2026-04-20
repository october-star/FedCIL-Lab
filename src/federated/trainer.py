from __future__ import annotations

import copy
import torch
from torch.utils.data import DataLoader

from src.federated.aggregator import fedavg
from src.federated.client import Client


class FederatedTrainer:
    def __init__(
        self,
        model,
        dataset_manager,
        device,
        num_clients: int,
        batch_size: int = 128,
        local_epochs: int = 1,
        rounds: int = 10,
        lr: float = 0.01,
    ):
        self.model = model
        self.dataset_manager = dataset_manager
        self.device = device
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.lr = lr

        self.clients = [Client(device) for _ in range(num_clients)]

    def evaluate(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=256)

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total

    def train(self):
        print("Start Federated Training...")

        for task_id in range(self.dataset_manager.num_tasks):
            print(f"\n=== Task {task_id} ===")

            task_classes = self.dataset_manager.get_task_classes(task_id)
            self.model.expand_head(len(task_classes))
            self.model.to(self.device)

            for round_id in range(self.rounds):
                local_states = []
                weights = []

                for client_id in range(self.num_clients):
                    subset = self.dataset_manager.get_train_subset(task_id, client_id)

                    local_model = copy.deepcopy(self.model)

                    state, loss = self.clients[client_id].train(
                        model=local_model,
                        dataset=subset,
                        batch_size=self.batch_size,
                        epochs=self.local_epochs,
                        lr=self.lr,
                    )

                    local_states.append(state)
                    weights.append(len(subset))

                weights = [w / sum(weights) for w in weights]

                new_state = fedavg(local_states, weights)
                self.model.load_state_dict(new_state)

                if round_id % 5 == 0:
                    test_set = self.dataset_manager.get_seen_test_subset(task_id)
                    acc = self.evaluate(self.model, test_set)
                    print(f"[Task {task_id}][Round {round_id}] Acc: {acc:.4f}")

            final_test = self.dataset_manager.get_seen_test_subset(task_id)
            final_acc = self.evaluate(self.model, final_test)
            print(f"[Task {task_id}] Final Acc: {final_acc:.4f}")