from pathlib import Path
import json

task_split_path = Path("data/processed/task_splits/cifar10_5task_seed1.json")
partition_path = Path("data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json")

with open(task_split_path, "r", encoding="utf-8") as f:
    split_payload = json.load(f)

with open(partition_path, "r", encoding="utf-8") as f:
    part_payload = json.load(f)

task_classes = split_payload["task_classes"]
task_train_indices = split_payload["task_to_train_indices"]
task_client_indices = part_payload["task_to_client_train_indices"]

for task_id in range(split_payload["num_tasks"]):
    task_key = f"task_{task_id}"

    total_task_samples = len(task_train_indices[task_key])
    total_client_samples = sum(
        len(v) for v in task_client_indices[task_key].values()
    )

    print(f"{task_key}")
    print(f"  task classes       : {task_classes[task_id]}")
    print(f"  total task samples : {total_task_samples}")
    print(f"  total client sum   : {total_client_samples}")
    print(f"  match              : {total_task_samples == total_client_samples}")