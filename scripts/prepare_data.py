from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import (
    get_cifar10,
    get_cifar100,
    get_class_names,
    get_targets,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and verify CIFAR datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cifar10", "cifar100", "all"],
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw",
        help="Root directory to store datasets.",
    )
    return parser.parse_args()


def summarize_dataset(name: str, train_set, test_set) -> None:
    train_targets = get_targets(train_set)
    test_targets = get_targets(test_set)
    class_names = get_class_names(train_set)

    print("=" * 60)
    print(f"Dataset: {name}")
    print(f"Train size: {len(train_set)}")
    print(f"Test size : {len(test_set)}")
    print(f"Num classes: {len(set(train_targets))}")
    print(f"Train label range: [{min(train_targets)}, {max(train_targets)}]")
    print(f"Test  label range: [{min(test_targets)}, {max(test_targets)}]")

    if class_names:
        preview = class_names[:10]
        print(f"Class names preview: {preview}")

    print("Sample check passed.")
    print("=" * 60)


def prepare_cifar10(root: str) -> None:
    train_set = get_cifar10(root=root, train=True, download=True)
    test_set = get_cifar10(root=root, train=False, download=True)
    summarize_dataset("CIFAR10", train_set, test_set)


def prepare_cifar100(root: str) -> None:
    train_set = get_cifar100(root=root, train=True, download=True)
    test_set = get_cifar100(root=root, train=False, download=True)
    summarize_dataset("CIFAR100", train_set, test_set)


def main():
    args = parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["cifar10", "all"]:
        prepare_cifar10(str(root))

    if args.dataset in ["cifar100", "all"]:
        prepare_cifar100(str(root))

    print("Data preparation finished successfully.")


if __name__ == "__main__":
    main()