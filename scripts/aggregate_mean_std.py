from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate result JSON files into mean/std summaries."
    )
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--output_csv", default="outputs/results/mean_std_summary.csv")
    parser.add_argument("--pattern", default="*.json")
    return parser.parse_args()


def infer_beta_from_partition(partition_path: str) -> str:
    if "beta01" in partition_path:
        return "0.1"
    if "beta05" in partition_path:
        return "0.5"
    if "beta10" in partition_path:
        return "1.0"
    return ""


def infer_variant(run_name: str, method: str, results: dict[str, Any]) -> str:
    lower_name = run_name.lower()

    if method in {"local_replay_gdr_paper", "local_replay_gdr_tts_paper"}:
        return "paper"
    if results.get("gdr_candidate_pool") == "current_task_only":
        return "paper"
    if results.get("gdr_class_wise") is True:
        return "classwise"
    if "backbonegdr" in lower_name or "backbonecheck" in lower_name:
        return "backbone"
    if "papertts" in lower_name and "classwise" in lower_name:
        return "classwise_papertts"
    if "papertts" in lower_name:
        return "papertts"
    if method in {"local_replay_gdr", "local_replay_gdr_tts"}:
        return "original"
    return "default"


def format_num(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return ""
    return str(value)


def extract_row(result_path: Path) -> dict[str, Any]:
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    results = payload.get("results", {})
    tasks = results.get("tasks", [])
    last_task = tasks[-1] if tasks else {}
    tts = results.get("tts", {})

    run_name = payload.get("run_name", result_path.stem)
    method = results.get("method", config.get("method", ""))

    return {
        "run_name": run_name,
        "method": method,
        "variant": infer_variant(run_name, method, results),
        "dataset": config.get("dataset", ""),
        "seed": config.get("seed", ""),
        "num_clients": config.get("num_clients", ""),
        "rounds": config.get("rounds", ""),
        "buffer_size": config.get("buffer_size", ""),
        "samples_per_task": config.get("samples_per_task", ""),
        "gdr_rank": config.get("gdr_rank", results.get("gdr_rank", "")),
        "gdr_feature_samples": config.get(
            "gdr_feature_samples", results.get("gdr_feature_samples", "")
        ),
        "gdr_class_wise": results.get("gdr_class_wise", config.get("gdr_class_wise", "")),
        "gdr_candidate_pool": results.get("gdr_candidate_pool", ""),
        "gdr_selection_mode": results.get("gdr_selection_mode", ""),
        "gdr_encryption": results.get("gdr_encryption", ""),
        "replay_weighting": results.get("replay_weighting", ""),
        "tts_old_temp": tts.get("old_temp", config.get("tts_old_temp", "")),
        "tts_new_temp": tts.get("new_temp", config.get("tts_new_temp", "")),
        "tts_old_weight": tts.get("old_weight", config.get("tts_old_weight", "")),
        "tts_new_weight": tts.get("new_weight", config.get("tts_new_weight", "")),
        "beta": infer_beta_from_partition(config.get("partition_path", "")),
        "final_task_id": last_task.get("task_id", ""),
        "final_seen_acc": last_task.get("final_seen_acc", ""),
        "num_tasks": len(tasks),
        "result_path": str(result_path),
    }


GROUP_FIELDS = [
    "method",
    "variant",
    "dataset",
    "num_clients",
    "rounds",
    "buffer_size",
    "samples_per_task",
    "gdr_rank",
    "gdr_feature_samples",
    "gdr_class_wise",
    "gdr_candidate_pool",
    "gdr_selection_mode",
    "gdr_encryption",
    "replay_weighting",
    "tts_old_temp",
    "tts_new_temp",
    "tts_old_weight",
    "tts_new_weight",
    "beta",
    "num_tasks",
]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    def sort_key(entry: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[str, ...]:
        key, _ = entry
        return tuple("" if value is None else str(value) for value in key)

    for key, items in sorted(grouped.items(), key=sort_key):
        accs = [
            float(item["final_seen_acc"])
            for item in items
            if item.get("final_seen_acc", "") not in ("", None)
        ]
        seeds = sorted(
            str(item["seed"])
            for item in items
            if item.get("seed", "") not in ("", None, "")
        )
        mean = sum(accs) / len(accs) if accs else float("nan")
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0

        record = {field: items[0].get(field, "") for field in GROUP_FIELDS}
        record.update(
            {
                "num_runs": len(items),
                "seeds": ",".join(seeds),
                "final_seen_acc_mean": format_num(mean),
                "final_seen_acc_std": format_num(std),
                "final_seen_acc_mean_std": (
                    f"{mean * 100:.2f}±{std * 100:.2f}" if accs else ""
                ),
                "run_names": ";".join(str(item["run_name"]) for item in items),
                "result_paths": ";".join(str(item["result_path"]) for item in items),
            }
        )
        aggregated.append(record)

    return aggregated


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob(args.pattern))
    rows = [extract_row(path) for path in result_files]
    aggregated = aggregate_rows(rows)

    fieldnames = GROUP_FIELDS + [
        "num_runs",
        "seeds",
        "final_seen_acc_mean",
        "final_seen_acc_std",
        "final_seen_acc_mean_std",
        "run_names",
        "result_paths",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)

    print(f"Scanned {len(result_files)} result files.")
    print(f"Aggregated into {len(aggregated)} groups.")
    print(f"Saved mean/std summary to: {output_csv}")


if __name__ == "__main__":
    main()
