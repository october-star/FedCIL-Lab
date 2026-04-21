from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class Client:
    def __init__(self, device: torch.device):
        self.device = device

    def train(
        self,
        model: nn.Module,
        dataset,
        batch_size: int,
        epochs: int,
        lr: float,
        loss_fn=None,
    ):
        if len(dataset) < 2:
            raise ValueError("BatchNorm training requires at least 2 samples.")

        model = model.to(self.device)
        model.train()

        effective_batch_size = min(max(batch_size, 2), len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-5,
        )

        total_loss = 0.0

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = model(x)
                if loss_fn is None:
                    loss = nn.functional.cross_entropy(logits, y)
                else:
                    loss = loss_fn(logits, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return model.state_dict(), total_loss / len(loader)
