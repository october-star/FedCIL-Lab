from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "outputs/.matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def plot_buffer_class_distribution(
    buffer_stats: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clients = sorted(buffer_stats)
    classes = sorted(
        {
            int(class_id)
            for stats in buffer_stats.values()
            for class_id in stats.get("class_counts", {})
        }
    )
    matrix = np.zeros((len(clients), len(classes)), dtype=np.int64)

    for row, client_key in enumerate(clients):
        counts = buffer_stats[client_key].get("class_counts", {})
        for col, class_id in enumerate(classes):
            matrix[row, col] = int(counts.get(class_id, counts.get(str(class_id), 0)))

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7), 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Class index")
    ax.set_ylabel("Client")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels([str(class_id) for class_id in classes])
    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels(clients)
    fig.colorbar(im, ax=ax, label="Buffer samples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_leverage_scores(
    records: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Leverage scores",
    selected_records: list[dict[str, Any]] | None = None,
    use_raw_score: bool = True,
    log_scale: bool = False,
) -> None:
    """
    Plot leverage scores and optionally highlight selected replay samples.

    Gray/colored small points: all scored samples
    Larger points: selected replay samples
    """
    if not records:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_key = "raw_leverage_score" if use_raw_score else "leverage_score"

    labels = np.array([int(r["label"]) for r in records])
    scores = np.array([float(r.get(scores_key, 0.0)) for r in records])

    if log_scale:
        scores = np.log1p(scores)

    # Sort by class, then by score for clearer visualization
    order = np.lexsort((scores, labels))
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    # Map original record identity to sorted x position
    record_to_x = {}
    for x_pos, original_idx in enumerate(order):
        r = records[int(original_idx)]
        key = (
            int(r["client_id"]),
            int(r["task_id"]),
            int(r["dataset_index"]),
            int(r["label"]),
        )
        record_to_x[key] = x_pos

    x_all = np.arange(len(records))

    plt.figure(figsize=(12, 7))

    scatter = plt.scatter(
        x_all,
        scores_sorted,
        color="lightgray",
        s=5,
        alpha=0.15,
    )

    if selected_records:
        selected_x = []
        selected_y = []
        selected_labels = []

        for r in selected_records:
            key = (
                int(r["client_id"]),
                int(r["task_id"]),
                int(r["dataset_index"]),
                int(r["label"]),
            )
            if key not in record_to_x:
                continue

            x_pos = record_to_x[key]
            selected_x.append(x_pos)

            score = float(r.get(scores_key, 0.0))
            if log_scale:
                score = np.log1p(score)
            selected_y.append(score)
            selected_labels.append(int(r["label"]))

        if selected_x:
            plt.scatter(
                selected_x,
                selected_y,
                c=selected_labels,
                cmap="tab10",
                s=40,
                marker="x",
                linewidths=1.8,
                label="selected",
            )

    plt.colorbar(scatter, label="Class index")
    plt.xlabel("Samples sorted by class")
    ylabel = "Raw leverage score" if use_raw_score else "Normalized leverage score"
    if log_scale:
        ylabel = f"log(1 + {ylabel})"
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
