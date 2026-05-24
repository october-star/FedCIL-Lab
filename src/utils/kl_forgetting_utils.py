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
    old_class_ids: list[int],
    device,
    temperature: float = 2.0,
    max_samples_per_class: int = 100,
) -> dict[int, float]:
    teacher_model.eval()
    student_model.eval()

    old_class_ids = [int(c) for c in old_class_ids]
    old_class_set = set(old_class_ids)

    if not old_class_ids:
        return {}

    old_idx = torch.tensor(
        old_class_ids,
        device=device,
        dtype=torch.long,
    )

    samples_by_class: dict[int, list[torch.Tensor]] = defaultdict(list)

    for client_datasets in retained_datasets:
        for dataset in client_datasets:
            for i in range(len(dataset)):
                x, y = _unpack_item(dataset[i])
                y = int(y)

                if y not in old_class_set:
                    continue

                if len(samples_by_class[y]) >= max_samples_per_class:
                    continue

                samples_by_class[y].append(x.detach().cpu())

    scores: dict[int, float] = {}

    for class_id, xs in samples_by_class.items():
        if not xs:
            continue

        batch = torch.stack(xs).to(device)

        teacher_logits_all = _get_logits(teacher_model(batch))
        student_logits_all = _get_logits(student_model(batch))

        teacher_logits = teacher_logits_all.index_select(1, old_idx)
        student_logits = student_logits_all.index_select(1, old_idx)

        teacher_probs = F.softmax(teacher_logits / temperature,dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature,dim=1)

        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=1)

        kl = kl * (temperature ** 2)
        scores[class_id] = float(kl.mean().item())

    return scores