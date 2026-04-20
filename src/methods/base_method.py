from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BaseMethod(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataset_manager,
        device: torch.device,
        num_clients: int,
        batch_size: int = 128,
        local_epochs: int = 1,
        rounds: int = 10,
        lr: float = 0.01,
    ) -> None:
        self.model = model
        self.dataset_manager = dataset_manager
        self.device = device
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.lr = lr

    def evaluate(self, model: nn.Module, dataset: Dataset) -> float:
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

        if total == 0:
            return 0.0
        return correct / total

    @abstractmethod
    def train(self) -> dict:
        raise NotImplementedError
