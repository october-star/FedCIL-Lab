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
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = np.array([record["leverage_score"] for record in records], dtype=float)
    labels = np.array([record["label"] for record in records], dtype=int)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(scores) > 0:
        order = np.argsort(labels)
        scatter = ax.scatter(
            np.arange(len(scores)),
            scores[order],
            c=labels[order],
            s=16,
            cmap="tab10",
            alpha=0.8,
        )
        fig.colorbar(scatter, ax=ax, label="Class index")
    ax.set_title(title)
    ax.set_xlabel("Samples sorted by class")
    ax.set_ylabel("Normalized leverage score")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
