import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar import get_cifar_dataset
from src.data.federated_dataset import FederatedDatasetManager

task_split_path = "data/processed/task_splits/cifar10_5task_seed1.json"
partition_path = "data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json"

with open(task_split_path, "r", encoding="utf-8") as f:
    task_split_payload = json.load(f)

with open(partition_path, "r", encoding="utf-8") as f:
    federated_partition_payload = json.load(f)

train_dataset = get_cifar_dataset("cifar10", root="data/raw", train=True, download=True)
test_dataset = get_cifar_dataset("cifar10", root="data/raw", train=False, download=True)

manager = FederatedDatasetManager(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    task_split_payload=task_split_payload,
    federated_partition_payload=federated_partition_payload,
)

print(manager.summary())

subset = manager.get_train_subset(task_id=0, client_id=0)
print(f"\nTask 0, Client 0 train subset size: {len(subset)}")

test_subset = manager.get_seen_test_subset(task_id=2)
print(f"Seen test subset after task 2: {len(test_subset)}")