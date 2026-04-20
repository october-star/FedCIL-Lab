from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import get_cifar_dataset, get_targets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize federated class distribution for each task."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw",
    )
    parser.add_argument(
        "--task_split_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partition_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/figures/data_distribution",
    )
    return parser.parse_args()


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_paths(args):
    if args.task_split_path is None:
        task_split_path = (
            Path("data/processed/task_splits")
            / f"{args.dataset}_{args.num_tasks}task_seed{args.seed}.json"
        )
    else:
        task_split_path = Path(args.task_split_path)

    if args.partition_path is None:
        beta_str = str(args.beta).replace(".", "")
        partition_path = (
            Path("data/processed/federated_partitions")
            / f"{args.dataset}_{args.num_tasks}task_"
              f"{args.num_clients}clients_beta{beta_str}_seed{args.seed}.json"
        )
    else:
        partition_path = Path(args.partition_path)

    return task_split_path, partition_path


def build_count_matrix(
    full_targets: list[int],
    task_classes: list[int],
    task_client_indices: dict[str, list[int]],
    num_clients: int,
) -> np.ndarray:
    """
    Build count matrix with shape [num_clients, num_task_classes]
    rows: clients
    cols: classes in this task
    """
    class_to_col = {cls: i for i, cls in enumerate(task_classes)}
    matrix = np.zeros((num_clients, len(task_classes)), dtype=np.int64)

    for client_id in range(num_clients):
        client_key = f"client_{client_id}"
        indices = task_client_indices[client_key]

        for idx in indices:
            label = full_targets[idx]
            if label in class_to_col:
                matrix[client_id, class_to_col[label]] += 1

    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    task_classes: list[int],
    task_id: int,
    dataset: str,
    beta: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(task_classes) * 0.7), 4.8))

    im = ax.imshow(matrix, aspect="auto")

    ax.set_title(f"{dataset.upper()} | Task {task_id} | beta={beta}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")

    ax.set_xticks(np.arange(len(task_classes)))
    ax.set_xticklabels([str(c) for c in task_classes], rotation=45, ha="right")

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"client_{i}" for i in range(matrix.shape[0])])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sample Count")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_bar(
    matrix: np.ndarray,
    task_classes: list[int],
    task_id: int,
    dataset: str,
    beta: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(matrix.shape[0])
    bottom = np.zeros(matrix.shape[0], dtype=np.int64)

    for col_id, cls in enumerate(task_classes):
        values = matrix[:, col_id]
        ax.bar(x, values, bottom=bottom, label=str(cls))
        bottom += values

    ax.set_title(f"{dataset.upper()} | Task {task_id} | beta={beta}")
    ax.set_xlabel("Client")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(x)
    ax.set_xticklabels([f"client_{i}" for i in range(matrix.shape[0])])
    ax.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    task_split_path, partition_path = resolve_paths(args)

    task_split_payload = load_json(task_split_path)
    partition_payload = load_json(partition_path)

    train_set = get_cifar_dataset(
        name=args.dataset,
        root=args.root,
        train=True,
        download=True,
    )
    train_targets = get_targets(train_set)

    save_dir = ensure_dir(args.save_dir)

    task_classes_all = task_split_payload["task_classes"]
    task_to_client_train_indices = partition_payload["task_to_client_train_indices"]

    print("=" * 80)
    print(f"Dataset         : {args.dataset}")
    print(f"Task split file : {task_split_path}")
    print(f"Partition file  : {partition_path}")
    print(f"Save dir        : {save_dir}")
    print("=" * 80)

    for task_id in range(args.num_tasks):
        task_key = f"task_{task_id}"
        task_classes = list(task_classes_all[task_id])
        client_indices = task_to_client_train_indices[task_key]

        matrix = build_count_matrix(
            full_targets=train_targets,
            task_classes=task_classes,
            task_client_indices=client_indices,
            num_clients=args.num_clients,
        )

        heatmap_path = (
            save_dir
            / f"{args.dataset}_task{task_id}_beta{str(args.beta).replace('.', '')}_heatmap.png"
        )
        bar_path = (
            save_dir
            / f"{args.dataset}_task{task_id}_beta{str(args.beta).replace('.', '')}_stackedbar.png"
        )

        plot_heatmap(
            matrix=matrix,
            task_classes=task_classes,
            task_id=task_id,
            dataset=args.dataset,
            beta=args.beta,
            save_path=heatmap_path,
        )

        plot_stacked_bar(
            matrix=matrix,
            task_classes=task_classes,
            task_id=task_id,
            dataset=args.dataset,
            beta=args.beta,
            save_path=bar_path,
        )

        print(f"[Task {task_id}]")
        print(f"  classes: {task_classes}")
        print(f"  matrix shape: {matrix.shape}")
        print(f"  saved heatmap   -> {heatmap_path}")
        print(f"  saved stackedbar-> {bar_path}")
        print("-" * 80)

    print("Visualization finished successfully.")


if __name__ == "__main__":
    main()