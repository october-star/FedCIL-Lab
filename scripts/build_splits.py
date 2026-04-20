from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import (
    get_cifar_dataset,
    get_num_classes,
    get_targets,
)
from src.data.task_split import (
    create_task_split_result,
    save_class_order,
    save_task_split,
    summarize_task_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build task splits for FCIL datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        required=True,
        help="Number of incremental tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for fixed class order.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw",
        help="Dataset root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_set = get_cifar_dataset(
        name=args.dataset,
        root=args.root,
        train=True,
        download=True,
    )
    test_set = get_cifar_dataset(
        name=args.dataset,
        root=args.root,
        train=False,
        download=True,
    )

    train_targets = get_targets(train_set)
    test_targets = get_targets(test_set)
    num_classes = get_num_classes(args.dataset)

    result = create_task_split_result(
        dataset_name=args.dataset,
        num_classes=num_classes,
        num_tasks=args.num_tasks,
        seed=args.seed,
        train_targets=train_targets,
        test_targets=test_targets,
    )

    class_order_path = save_class_order(
        dataset_name=args.dataset,
        seed=args.seed,
        class_order=result.class_order,
    )
    task_split_path = save_task_split(result)

    print("=" * 70)
    print(f"Dataset     : {args.dataset}")
    print(f"Num classes : {num_classes}")
    print(f"Num tasks   : {args.num_tasks}")
    print(f"Seed        : {args.seed}")
    print("-" * 70)
    print("Class order:")
    print(result.class_order)
    print("-" * 70)
    print("Task classes:")
    print(summarize_task_split(result.task_classes))
    print("-" * 70)

    for task_id in range(args.num_tasks):
        task_key = f"task_{task_id}"
        train_count = len(result.task_to_train_indices[task_key])
        test_count = len(result.task_to_test_indices[task_key])
        print(
            f"{task_key}: train_samples={train_count}, test_samples={test_count}"
        )

    print("-" * 70)
    print(f"Saved class order to: {class_order_path}")
    print(f"Saved task split  to: {task_split_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()