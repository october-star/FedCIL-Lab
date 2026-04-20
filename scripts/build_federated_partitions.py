from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import get_cifar_dataset, get_targets
from src.data.task_split import load_task_split
from src.data.dirichlet_partition import (
    build_federated_partition_from_task_split,
    save_federated_partition,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build federated Dirichlet partitions.")
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
        default=5,
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
        help="Optional explicit path to task split json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task_split_path is None:
        task_split_path = (
            Path("data/processed/task_splits")
            / f"{args.dataset}_{args.num_tasks}task_seed{args.seed}.json"
        )
    else:
        task_split_path = Path(args.task_split_path)

    task_split_payload = load_task_split(task_split_path)

    train_set = get_cifar_dataset(
        name=args.dataset,
        root=args.root,
        train=True,
        download=True,
    )
    train_targets = get_targets(train_set)

    result = build_federated_partition_from_task_split(
        task_split_payload=task_split_payload,
        full_train_targets=train_targets,
        num_clients=args.num_clients,
        beta=args.beta,
        seed=args.seed,
        min_size_per_client=1,
    )

    save_path = save_federated_partition(result)

    print("=" * 70)
    print(f"Dataset     : {args.dataset}")
    print(f"Num tasks   : {args.num_tasks}")
    print(f"Num clients : {args.num_clients}")
    print(f"Beta        : {args.beta}")
    print(f"Seed        : {args.seed}")
    print("-" * 70)

    for task_id in range(result.num_tasks):
        task_key = f"task_{task_id}"
        print(task_key)
        for client_key, indices in result.task_to_client_train_indices[task_key].items():
            print(f"  {client_key}: {len(indices)} samples")
        print("  class counts:")
        for client_key, counts in result.task_to_client_class_counts[task_key].items():
            print(f"    {client_key}: {counts}")
        print("-" * 70)

    print(f"Saved federated partition to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()