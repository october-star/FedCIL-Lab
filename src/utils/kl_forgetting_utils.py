from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _unpack_item(item):
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return item[0], item[1]
        return item[-2], item[-1]
    raise ValueError(f"Unsupported dataset item type: {type(item)}")


def _get_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


@torch.no_grad()
def compute_class_kl_forgetting(
    *,
    teacher_model,
    student_model,
    retained_datasets: list[list[Dataset]],
    old_classes: int,
    device,
    temperature: float = 2.0,
    max_samples_per_class: int = 100,
) -> dict[int, float]:
    # 关掉 dropout / BN 更新 只测试，不训练
    teacher_model.eval()
    student_model.eval()

    # 按 class 收集 replay 样本。
    samples_by_class: dict[int, list[torch.Tensor]] = defaultdict(list)

    # 按类划分数据
    # samples_by_class =
    # {
    #     dog: [x1, x2],
    #     cat: [x3],
    #     truck: [x4, x5],
    # }
    for client_datasets in retained_datasets:
        for dataset in client_datasets:
            for i in range(len(dataset)):
                x, y = _unpack_item(dataset[i])
                y = int(y)

                if y >= old_classes:
                    continue

                if len(samples_by_class[y]) >= max_samples_per_class:
                    continue

                samples_by_class[y].append(x.detach().cpu())

    scores: dict[int, float] = {}

    # 计算 temperature是为了变平滑
    for class_id, xs in samples_by_class.items():
        if not xs:
            continue

        batch = torch.stack(xs).to(device)

        teacher_logits = _get_logits(teacher_model(batch))[:, :old_classes]
        student_logits = _get_logits(student_model(batch))[:, :old_classes]

        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=1)

        kl = kl * (temperature ** 2)
        scores[class_id] = float(kl.mean().item())

    return scores