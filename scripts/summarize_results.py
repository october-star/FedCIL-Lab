from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training result JSON files.")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_csv", default="outputs/results/summary.csv")
    parser.add_argument("--pattern", default="*.json")
    return parser.parse_args()


def extract_row(result_path: Path) -> dict[str, object]:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    results = payload.get("results", {})
    tasks = results.get("tasks", [])
    last_task = tasks[-1] if tasks else {}

    return {
        "run_name": payload.get("run_name", result_path.stem),
        "method": results.get("method", config.get("method", "")),
        "dataset": config.get("dataset", ""),
        "seed": config.get("seed", ""),
        "num_clients": config.get("num_clients", ""),
        "rounds": config.get("rounds", ""),
        "buffer_size": config.get("buffer_size", ""),
        "beta": infer_beta_from_partition(config.get("partition_path", "")),
        "final_task_id": last_task.get("task_id", ""),
        "final_seen_acc": last_task.get("final_seen_acc", ""),
        "num_tasks": len(tasks),
        "result_path": str(result_path),
    }


def infer_beta_from_partition(partition_path: str) -> str:
     if "beta100" in partition_path:
        return "10.0"
    if "beta01" in partition_path:
        return "0.1"
    if "beta05" in partition_path:
        return "0.5"
    if "beta10" in partition_path:
        return "1.0"
    return ""


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob(args.pattern))
    rows = [extract_row(path) for path in result_files]

    fieldnames = [
        "run_name",
        "method",
        "dataset",
        "seed",
        "num_clients",
        "rounds",
        "buffer_size",
        "beta",
        "final_task_id",
        "final_seen_acc",
        "num_tasks",
        "result_path",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scanned {len(result_files)} result files.")
    print(f"Saved summary to: {output_csv}")


if __name__ == "__main__":
    main()
