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
    title: str,
    selected_records: list[dict[str, Any]] | None = None,
    use_raw_score: bool = True,
    log_scale: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No records", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    score_key = (
        "raw_leverage_score"
        if use_raw_score
        else "leverage_score"
    )

    scores = np.array(
        [
            float(record.get(score_key, 0.0))
            for record in records
        ],
        dtype=float,
    )

    labels = np.array(
        [int(record["label"]) for record in records],
        dtype=int,
    )

    order = np.argsort(labels)

    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # build selected lookup
    selected_keys = set()
    if selected_records is not None:
        selected_keys = {
            (
                int(r["client_id"]),
                int(r["sample_id"]),
            )
            for r in selected_records
        }

    selected_mask = np.array(
        [
            (
                int(records[idx]["client_id"]),
                int(records[idx]["sample_id"]),
            )
            in selected_keys
            for idx in order
        ]
    )

    x = np.arange(len(sorted_scores))

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter = ax.scatter(
        x,
        sorted_scores,
        c=sorted_labels,
        cmap="tab20",
        s=16,
        alpha=0.55,
    )

    # overlay selected samples
    if selected_mask.any():
        ax.scatter(
            x[selected_mask],
            sorted_scores[selected_mask],
            facecolors="none",
            edgecolors="red",
            linewidths=1.3,
            s=60,
            label="Selected samples",
        )
        ax.legend()

    if log_scale:
        ax.set_yscale("log")

    fig.colorbar(
        scatter,
        ax=ax,
        label="Class index",
    )

    ax.set_title(title)
    ax.set_xlabel("Samples sorted by class")

    ylabel = (
        "Raw leverage score"
        if use_raw_score
        else "Normalized leverage score"
    )
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path