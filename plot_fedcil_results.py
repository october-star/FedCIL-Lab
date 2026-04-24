#!/usr/bin/env python3
"""
FedCIL-Lab — Results Visualization Script
Run locally to generate clean charts without watermarks.
Requires: matplotlib, numpy
  pip install matplotlib numpy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os

os.makedirs("output", exist_ok=True)

methods      = ["Finetune", "Local Replay", "GDR", "TTS", "FedCBDR"]
tasks        = ["T0", "T1", "T2", "T3", "T4"]
colors       = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

# ── β=0.1 results (mean / std across seeds 1,2,3) ──────────────────────────
b01_means = np.array([
    [94.25, 17.34, 12.51, 11.10, 10.40],   # Finetune
    [94.27, 28.00, 23.33, 19.05, 15.84],   # Local Replay
    [94.23, 30.10, 24.66, 20.89, 17.58],   # GDR
    [94.68, 32.21, 27.79, 22.91, 20.08],   # TTS
    [94.68, 34.24, 29.62, 25.29, 21.79],   # FedCBDR
])
b01_stds = np.array([
    [0.20, 0.26, 0.40, 0.54, 0.50],
    [0.31, 1.54, 1.55, 0.87, 1.42],
    [0.14, 1.51, 2.35, 1.21, 1.08],
    [0.16, 1.68, 1.67, 1.18, 1.31],
    [0.16, 1.90, 1.89, 1.36, 1.27],
])

# ── β=1.0 results (mean / std across seeds 1,2,3) ──────────────────────────
b10_means = np.array([
    [94.63, 19.68, 16.17, 14.28, 12.01],   # Finetune
    [94.62, 36.09, 28.59, 19.23, 18.03],   # Local Replay
    [93.77, 35.40, 27.54, 20.92, 18.60],   # GDR
    [95.57, 42.25, 35.28, 26.04, 22.17],   # TTS
    [94.42, 39.76, 33.59, 22.78, 19.86],   # FedCBDR
])
b10_stds = np.array([
    [0.31,  1.21, 0.42, 1.42, 0.96],
    [0.36,  8.98, 3.64, 4.59, 0.66],
    [0.87,  4.87, 6.06, 3.12, 0.48],
    [0.52, 10.70, 0.46, 3.92, 2.28],
    [0.58,  7.45, 3.15, 3.84, 0.87],
])

avg01 = [29.12, 36.10, 37.49, 39.53, 41.13]
avg10 = [31.35, 39.31, 39.24, 44.26, 42.08]


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1 — Average Accuracy grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 5.2))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("white")

x = np.arange(len(methods))
w = 0.35
bars1 = ax1.bar(x - w/2, avg01, width=w, color="#636EFA", label="β=0.1", zorder=3)
bars2 = ax1.bar(x + w/2, avg10, width=w, color="#EF553B", label="β=1.0", zorder=3)

for bar, val in zip(list(bars1) + list(bars2), avg01 + avg10):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.6,
             f"{val:.1f}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)
ax1.set_ylim(0, 52)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
ax1.grid(axis="y", color="#E8E8E8", linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(loc="upper left", fontsize=11, frameon=False)

# Title below chart
fig1.text(0.5, -0.04,
          "Average Accuracy by Method & Heterogeneity — FedCIL-Lab · CIFAR-10 · 5 Tasks · 3 Seeds",
          ha="center", va="top", fontsize=12, fontweight="bold", color="#222")

fig1.subplots_adjust(top=0.96, left=0.09, right=0.97, bottom=0.12)
fig1.savefig("output/avg_acc.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("Saved: output/avg_acc.png")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2 — Per-task accuracy line chart (2 subplots)
# ═══════════════════════════════════════════════════════════════════════════
fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
fig2.patch.set_facecolor("white")

for ax, means, stds, beta_label in [
    (ax_l, b01_means, b01_stds, "β=0.1 — High Heterogeneity"),
    (ax_r, b10_means, b10_stds, "β=1.0 — Low Heterogeneity"),
]:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E8E8E8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for i, m in enumerate(methods):
        ax.errorbar(tasks, means[i], yerr=stds[i],
                    label=m, color=colors[i], linewidth=2.2,
                    marker="o", markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2, zorder=3)
    ax.set_title(beta_label, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylim(0, 108)

ax_l.set_ylabel("Accuracy (%)", fontsize=11)
ax_l.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))

handles, labels = ax_l.get_legend_handles_labels()
fig2.legend(handles, labels,
            loc="center right", bbox_to_anchor=(1.0, 0.5),
            fontsize=11, frameon=False)

# Title below chart
fig2.text(0.42, -0.04,
          "Per-Task Final Accuracy — Mean ± Std (3 Seeds, CIFAR-10)",
          ha="center", va="top", fontsize=12, fontweight="bold", color="#222")

fig2.subplots_adjust(top=0.93, left=0.07, right=0.84, bottom=0.12, wspace=0.06)
fig2.savefig("output/per_task.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("Saved: output/per_task.png")
