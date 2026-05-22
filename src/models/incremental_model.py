from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torchvision.models as models


class IncrementalNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        norm_layer: str = "groupnorm",
        group_norm_groups: int = 32,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = models.resnet18(
                pretrained=pretrained,
                norm_layer=self._build_norm_layer(
                    norm_layer=norm_layer,
                    group_norm_groups=group_norm_groups,
                ),
            )
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone
        self.classifier = None
        self.num_classes = 0
        self.norm_layer_name = norm_layer
        self.group_norm_groups = group_norm_groups

    @staticmethod
    def _build_norm_layer(norm_layer: str, group_norm_groups: int):
        norm_layer = norm_layer.lower().strip()
        if norm_layer == "batchnorm":
            return nn.BatchNorm2d
        if norm_layer == "sync_batchnorm":
            return nn.SyncBatchNorm
        if norm_layer == "groupnorm":
            return partial(nn.GroupNorm, group_norm_groups)

        raise ValueError(
            f"Unsupported norm_layer: {norm_layer}. "
            "Expected 'batchnorm', 'sync_batchnorm', or 'groupnorm'."
        )


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