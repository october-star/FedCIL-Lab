from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import Dataset


@dataclass
class CIFARConfig:
    root: str = "data/raw"
    image_size: int = 32
    mean_cifar10: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std_cifar10: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)
    mean_cifar100: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    std_cifar100: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cifar10_transforms(image_size: int = 32):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    return train_transform, test_transform


def get_cifar100_transforms(image_size: int = 32):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return train_transform, test_transform


def get_cifar10(
    root: str = "data/raw",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """
    Return CIFAR-10 dataset with default train/test transforms.
    """
    ensure_dir(root)
    train_transform, test_transform = get_cifar10_transforms()
    transform = train_transform if train else test_transform

    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
    return dataset


def get_cifar100(
    root: str = "data/raw",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """
    Return CIFAR-100 dataset with default train/test transforms.
    """
    ensure_dir(root)
    train_transform, test_transform = get_cifar100_transforms()
    transform = train_transform if train else test_transform

    dataset = datasets.CIFAR100(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
    return dataset


def get_cifar_dataset(
    name: str,
    root: str = "data/raw",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """
    Unified entry for CIFAR datasets.

    Args:
        name: 'cifar10' or 'cifar100'
    """
    name = name.lower().strip()

    if name == "cifar10":
        return get_cifar10(root=root, train=train, download=download)
    if name == "cifar100":
        return get_cifar100(root=root, train=train, download=download)

    raise ValueError(f"Unsupported dataset: {name}. Expected 'cifar10' or 'cifar100'.")


def get_targets(dataset: Dataset) -> list[int]:
    """
    Extract targets from torchvision CIFAR datasets.

    CIFAR10 / CIFAR100 store labels in `.targets`.
    """
    if hasattr(dataset, "targets"):
        return list(dataset.targets)

    raise AttributeError("Dataset does not provide `.targets` attribute.")


def get_num_classes(dataset_name: str) -> int:
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def get_class_names(dataset: Dataset) -> list[str]:
    """
    Return class names if available.
    """
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    return []