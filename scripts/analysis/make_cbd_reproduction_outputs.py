# Author: mmj
# DATE: 22.05.2026
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "finetune": "Finetune",
    "local_replay": "Exemplar-based Replay",
    "local_replay_tts": "+TTS",
    "local_replay_gdr_paper": "+GDR",
    "local_replay_gdr_tts_paper": "+GDR+TTS",
    "cbdr_adaptive_reply": "CBDR+Adaptive",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output_dir", default="outputs/analysis")
    return parser.parse_args()


def load_results(results_dir: Path, prefix: str):
    files = sorted(results_dir.glob(f"{prefix}*.json"))
    rows = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        config = payload.get("config", {})
        results = payload.get("results", {})
        tasks = results.get("tasks", [])

        if not tasks:
            continue

        method = results.get("method", config.get("method", ""))
        rows.append(
            {
                "path": path,
                "run_name": payload.get("run_name", path.stem),
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "dataset": config.get("dataset", ""),
                "seed": str(config.get("seed", "")),
                "beta": infer_beta(config.get("partition_path", "")),
                "num_tasks": len(tasks),
                "tasks": tasks,
                "final_acc": float(tasks[-1]["final_seen_acc"]),
            }
        )

    return rows


def infer_beta(partition_path: str) -> str:
    if "beta01" in partition_path:
        return "0.1"
    if "beta05" in partition_path:
        return "0.5"
    if "beta10" in partition_path:
        return "1.0"
    if "beta100" in partition_path:
        return "10.0"
    return ""

def write_table4_markdown(grouped, output_path: Path) -> None:
    rows = []

    for key, items in sorted(grouped.items()):
        dataset, num_tasks, beta, label = key
        accs = [item["final_acc"] for item in items]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0

        rows.append(
            {
                "dataset": dataset,
                "num_tasks": num_tasks,
                "beta": beta,
                "method": label,
                "num_runs": len(items),
                "seeds": ",".join(sorted(item["seed"] for item in items)),
                "mean_std": f"{mean * 100:.2f}±{std * 100:.2f}",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Table 4 Summary\n\n")
        f.write("| Dataset | Tasks | Beta | Method | Runs | Seeds | Final Acc |\n")
        f.write("|---|---:|---:|---|---:|---|---:|\n")

        for row in rows:
            f.write(
                f"| {row['dataset']} "
                f"| {row['num_tasks']} "
                f"| {row['beta']} "
                f"| {row['method']} "
                f"| {row['num_runs']} "
                f"| {row['seeds']} "
                f"| {row['mean_std']} |\n"
            )


def make_table4(rows, output_dir: Path):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["num_tasks"],
            row["beta"],
            row["label"],
        )
        grouped[key].append(row)

    out_path = output_dir / "table4_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "num_tasks",
                "beta",
                "method",
                "num_runs",
                "seeds",
                "mean",
                "std",
                "mean_std_percent",
                "runs",
            ],
        )
        writer.writeheader()

        for key, items in sorted(grouped.items()):
            dataset, num_tasks, beta, label = key
            accs = [item["final_acc"] for item in items]
            mean = statistics.mean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0

            writer.writerow(
                {
                    "dataset": dataset,
                    "num_tasks": num_tasks,
                    "beta": beta,
                    "method": label,
                    "num_runs": len(items),
                    "seeds": ",".join(sorted(item["seed"] for item in items)),
                    "mean": f"{mean:.4f}",
                    "std": f"{std:.4f}",
                    "mean_std_percent": f"{mean * 100:.2f}±{std * 100:.2f}",
                    "runs": ";".join(item["run_name"] for item in items),
                }
            )

    print(f"Saved Table 4 summary: {out_path}")

    # markdown_path = output_dir / "table4_summary.md"
    # write_table4_markdown(grouped, markdown_path)
    # print(f"Saved Table 4 markdown: {markdown_path}")


def plot_figure3_variant(
    by_setting,
    output_dir: Path,
    *,
    include_adaptive: bool,
) -> None:
    suffix = "with_adaptive" if include_adaptive else "no_adaptive"

    for setting, method_curves in sorted(by_setting.items()):
        dataset, num_tasks, beta = setting

        plot_curves = dict(method_curves)
        if not include_adaptive:
            plot_curves.pop("CBDR+Adaptive", None)

        if not plot_curves:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, curve in sorted(plot_curves.items()):
            ax.plot(range(len(curve)), curve, marker="o", label=label)

        ax.set_title(f"{dataset} {num_tasks}-task beta={beta} ({suffix})")
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Test Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = (
            output_dir
            / f"figure3_{suffix}_{dataset}_{num_tasks}task_beta{beta}.png"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved Figure 3 curve: {out_path}")


def make_figure3(rows, output_dir: Path):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["num_tasks"],
            row["beta"],
            row["label"],
        )
        curve = [float(task["final_seen_acc"]) * 100 for task in row["tasks"]]
        grouped[key].append(curve)

    by_setting = defaultdict(dict)
    for key, curves in grouped.items():
        dataset, num_tasks, beta, label = key
        max_len = max(len(c) for c in curves)
        mean_curve = []
        for i in range(max_len):
            values = [c[i] for c in curves if i < len(c)]
            mean_curve.append(statistics.mean(values))
        by_setting[(dataset, num_tasks, beta)][label] = mean_curve

    plot_figure3_variant(
        by_setting,
        output_dir,
        include_adaptive=False,
    )
    plot_figure3_variant(
        by_setting,
        output_dir,
        include_adaptive=True,
    )


def merge_class_counts(buffer_after: dict[str, Any]) -> dict[int, int]:
    merged = defaultdict(int)

    for client_stats in buffer_after.values():
        class_counts = client_stats.get("class_counts", {})
        for class_id, count in class_counts.items():
            merged[int(class_id)] += int(count)

    return dict(sorted(merged.items()))


def build_figure5_setting_counts(rows, candidate_labels):
    grouped = defaultdict(list)

    for row in rows:
        if row["label"] not in candidate_labels:
            continue

        key = (
            row["dataset"],
            row["num_tasks"],
            row["beta"],
            row["label"],
        )
        grouped[key].append(row)

    by_setting = defaultdict(dict)

    for key, items in grouped.items():
        dataset, num_tasks, beta, label = key

        task_counts_by_seed = []
        for item in items:
            task_counts = []
            for task in item["tasks"]:
                buffer_after = task.get("buffer_after", {})
                task_counts.append(merge_class_counts(buffer_after))
            task_counts_by_seed.append(task_counts)

        num_task_steps = max(len(x) for x in task_counts_by_seed)
        averaged_task_counts = []

        for task_id in range(num_task_steps):
            class_values = defaultdict(list)

            for seed_counts in task_counts_by_seed:
                if task_id >= len(seed_counts):
                    continue

                for class_id, count in seed_counts[task_id].items():
                    class_values[class_id].append(count)

            averaged = {
                class_id: statistics.mean(values)
                for class_id, values in class_values.items()
            }
            averaged_task_counts.append(averaged)

        by_setting[(dataset, num_tasks, beta)][label] = averaged_task_counts

    return by_setting


def plot_figure5_variant(
    by_setting,
    output_dir: Path,
    *,
    include_adaptive: bool,
) -> None:
    suffix = "with_adaptive" if include_adaptive else "no_adaptive"

    required = {"Exemplar-based Replay", "+GDR+TTS"}
    labels = ["Exemplar-based Replay", "+GDR+TTS"]

    if include_adaptive:
        labels.append("CBDR+Adaptive")

    for setting, label_to_task_counts in sorted(by_setting.items()):
        dataset, num_tasks, beta = setting

        if not required.issubset(label_to_task_counts):
            continue

        available_labels = [
            label for label in labels
            if label in label_to_task_counts
        ]

        if include_adaptive and "CBDR+Adaptive" not in available_labels:
            continue

        num_task_steps = min(
            len(label_to_task_counts[label])
            for label in available_labels
        )

        for task_id in range(num_task_steps):
            counts_by_label = {
                label: label_to_task_counts[label][task_id]
                for label in available_labels
            }

            classes = sorted(
                set().union(
                    *[
                        set(counts.keys())
                        for counts in counts_by_label.values()
                    ]
                )
            )

            if not classes:
                continue

            x = list(range(len(classes)))
            width = 0.8 / len(available_labels)

            fig, ax = plt.subplots(
                figsize=(max(8, len(classes) * 0.35), 4)
            )

            for idx, label in enumerate(available_labels):
                offset = (
                    idx - (len(available_labels) - 1) / 2
                ) * width

                display_label = (
                    "FedCBDR"
                    if label == "+GDR+TTS"
                    else label
                )

                ax.bar(
                    [i + offset for i in x],
                    [
                        counts_by_label[label].get(c, 0)
                        for c in classes
                    ],
                    width=width,
                    label=display_label,
                )

            cbd_counts = counts_by_label.get("+GDR+TTS", {})
            avg = (
                sum(cbd_counts.get(c, 0) for c in classes)
                / max(len(classes), 1)
            )
            ax.axhline(avg, linewidth=2, label="FedCBDR Avg")

            ax.set_title(
                f"{dataset} {num_tasks}-task beta={beta} "
                f"task {task_id} ({suffix})"
            )
            ax.set_xlabel("Class Index")
            ax.set_ylabel("Number")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in classes], rotation=90)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

            out_path = (
                output_dir
                / f"figure5_{suffix}_{dataset}_{num_tasks}task_beta{beta}_task{task_id}.png"
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

            print(f"Saved Figure 5 buffer plot: {out_path}")


def make_figure5(rows, output_dir: Path):
    candidate_labels = {
        "Exemplar-based Replay",
        "+GDR+TTS",
        "CBDR+Adaptive",
    }

    by_setting = build_figure5_setting_counts(
        rows,
        candidate_labels,
    )

    plot_figure5_variant(
        by_setting,
        output_dir,
        include_adaptive=False,
    )

    plot_figure5_variant(
        by_setting,
        output_dir,
        include_adaptive=True,
    )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_dir, args.prefix)

    print(f"Loaded {len(rows)} result files with prefix: {args.prefix}")

    if not rows:
        return

    make_table4(rows, output_dir)
    make_figure3(rows, output_dir)
    make_figure5(rows, output_dir)


if __name__ == "__main__":
    main()