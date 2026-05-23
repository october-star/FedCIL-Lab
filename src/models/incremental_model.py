from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class IncrementalNet(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = backbone.fc.in_features
            #backbone.fc = nn.Identity()
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            backbone.maxpool = nn.Identity()
            # ============================================
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.classifier = None
        self.num_classes = 0

    def expand_head(self, num_new_classes: int):
        new_total = self.num_classes + num_new_classes

        new_fc = nn.Linear(self.feature_dim, new_total)

        if self.classifier is not None:
            with torch.no_grad():
                new_fc.weight[: self.num_classes] = self.classifier.weight
                new_fc.bias[: self.num_classes] = self.classifier.bias

        self.classifier = new_fc
        self.num_classes = new_total

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits