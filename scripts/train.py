import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar import get_cifar_dataset
from src.data.federated_dataset import FederatedDatasetManager
from src.methods.finetune import Finetune
from src.methods.gdr_replay import LocalReplayGDR
from src.methods.replay import LocalReplay
from src.models.incremental_model import IncrementalNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FCIL baselines.")
    parser.add_argument(
        "--method",
        default="finetune",
        choices=["finetune", "local_replay", "local_replay_gdr"],
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--task_split_path",
        default="data/processed/task_splits/cifar10_5task_seed1.json",
    )
    parser.add_argument(
        "--partition_path",
        default=(
            "data/processed/federated_partitions/"
            "cifar10_5task_5clients_beta05_seed1.json"
        ),
    )
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--output_dir", default="outputs/results")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=200)
    parser.add_argument("--samples_per_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gdr_rank", type=int, default=8)
    parser.add_argument("--gdr_feature_samples", type=int, default=None)
    parser.add_argument("--figure_dir", default="outputs/figures/gdr")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.method}_{args.dataset}_{timestamp}"

    with open(args.task_split_path, "r", encoding="utf-8") as f:
        task_split = json.load(f)

    with open(args.partition_path, "r", encoding="utf-8") as f:
        partition = json.load(f)

    train_set = get_cifar_dataset(
        args.dataset,
        root=args.data_root,
        train=True,
        download=not args.no_download,
    )
    test_set = get_cifar_dataset(
        args.dataset,
        root=args.data_root,
        train=False,
        download=not args.no_download,
    )

    manager = FederatedDatasetManager(train_set, test_set, task_split, partition)
    device = get_device()
    print(f"Device: {device}")

    model = IncrementalNet(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    )

    method_kwargs = {
        "model": model,
        "dataset_manager": manager,
        "device": device,
        "num_clients": args.num_clients,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "rounds": args.rounds,
        "lr": args.lr,
    }

    if args.method == "finetune":
        method = Finetune(**method_kwargs)
    elif args.method == "local_replay":
        method = LocalReplay(
            **method_kwargs,
            buffer_size=args.buffer_size,
            samples_per_task=args.samples_per_task,
            seed=args.seed,
        )
    elif args.method == "local_replay_gdr":
        method = LocalReplayGDR(
            **method_kwargs,
            buffer_size=args.buffer_size,
            samples_per_task=args.samples_per_task,
            seed=args.seed,
            gdr_rank=args.gdr_rank,
            gdr_feature_samples=args.gdr_feature_samples,
            figure_dir=args.figure_dir,
            run_name=run_name,
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    history = method.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{run_name}.json"

    payload = {
        "run_name": run_name,
        "config": vars(args),
        "device": str(device),
        "results": history,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to: {result_path}")

    if args.save_checkpoint:
        checkpoint_path = output_dir / f"{run_name}.pt"
        torch.save(
            {
                "run_name": run_name,
                "config": vars(args),
                "model_state": model.state_dict(),
                "num_classes": model.num_classes,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
