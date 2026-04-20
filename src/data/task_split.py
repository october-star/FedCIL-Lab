from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class TaskSplitResult:
    dataset_name: str
    num_classes: int
    num_tasks: int
    seed: int
    class_order: list[int]
    task_classes: list[list[int]]
    task_to_train_indices: dict[str, list[int]]
    task_to_test_indices: dict[str, list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_class_order(num_classes: int, seed: int) -> list[int]:
    """
    Build a fixed class order using a random seed.

    Example:
        num_classes=10, seed=1 -> [6, 8, 9, 7, ...]
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    class_order = list(range(num_classes))
    rng = random.Random(seed)
    rng.shuffle(class_order)
    return class_order


def split_classes_into_tasks(class_order: list[int], num_tasks: int) -> list[list[int]]:
    """
    Split a class order into num_tasks contiguous tasks.

    If num_classes is not divisible by num_tasks, the first few tasks
    will get one extra class.

    Example:
        10 classes, 3 tasks -> [4, 3, 3]
    """
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")

    num_classes = len(class_order)
    if num_tasks > num_classes:
        raise ValueError(
            f"num_tasks ({num_tasks}) cannot exceed num_classes ({num_classes})"
        )

    base = num_classes // num_tasks
    remainder = num_classes % num_tasks

    task_sizes = []
    for task_id in range(num_tasks):
        size = base + (1 if task_id < remainder else 0)
        task_sizes.append(size)

    task_classes: list[list[int]] = []
    start = 0
    for size in task_sizes:
        end = start + size
        task_classes.append(class_order[start:end])
        start = end

    return task_classes


def build_task_indices(
    targets: list[int],
    task_classes: list[list[int]],
) -> dict[str, list[int]]:
    """
    Build sample indices for each task from dataset targets.

    Returns:
        {
            "task_0": [indices...],
            "task_1": [indices...],
            ...
        }
    """
    task_to_indices: dict[str, list[int]] = {}

    for task_id, classes in enumerate(task_classes):
        class_set = set(classes)
        indices = [idx for idx, label in enumerate(targets) if label in class_set]
        task_to_indices[f"task_{task_id}"] = indices

    return task_to_indices


def summarize_task_split(task_classes: list[list[int]]) -> str:
    lines = []
    for task_id, classes in enumerate(task_classes):
        lines.append(
            f"Task {task_id}: {len(classes)} classes -> {classes}"
        )
    return "\n".join(lines)


def save_class_order(
    dataset_name: str,
    seed: int,
    class_order: list[int],
    save_dir: str | Path = "data/processed/class_orders",
) -> Path:
    save_dir = ensure_dir(save_dir)
    save_path = save_dir / f"{dataset_name}_order_seed{seed}.json"

    payload = {
        "dataset_name": dataset_name,
        "seed": seed,
        "class_order": class_order,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return save_path


def load_class_order(path: str | Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["class_order"])


def save_task_split(
    result: TaskSplitResult,
    save_dir: str | Path = "data/processed/task_splits",
) -> Path:
    save_dir = ensure_dir(save_dir)
    save_path = save_dir / (
        f"{result.dataset_name}_{result.num_tasks}task_seed{result.seed}.json"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return save_path


def load_task_split(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def create_task_split_result(
    dataset_name: str,
    num_classes: int,
    num_tasks: int,
    seed: int,
    train_targets: list[int],
    test_targets: list[int],
) -> TaskSplitResult:
    class_order = build_class_order(num_classes=num_classes, seed=seed)
    task_classes = split_classes_into_tasks(class_order=class_order, num_tasks=num_tasks)

    task_to_train_indices = build_task_indices(
        targets=train_targets,
        task_classes=task_classes,
    )
    task_to_test_indices = build_task_indices(
        targets=test_targets,
        task_classes=task_classes,
    )

    return TaskSplitResult(
        dataset_name=dataset_name,
        num_classes=num_classes,
        num_tasks=num_tasks,
        seed=seed,
        class_order=class_order,
        task_classes=task_classes,
        task_to_train_indices=task_to_train_indices,
        task_to_test_indices=task_to_test_indices,
    )