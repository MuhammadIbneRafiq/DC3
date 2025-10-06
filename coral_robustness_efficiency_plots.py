#!/usr/bin/env python3
"""
Robustness vs Metrics & GPU Hours - Minimal consolidated plotting script

Generates exactly two plots:
1) Accuracy vs Robustness (with/without robust image processing)
2) Accuracy vs Training Time (GPU hours) highlighting robustness impact

This script is intentionally minimal and self-contained. Provide your
measured values below or leave the defaults as placeholders.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_robustness(models, accuracy_pct, robustness_score):

    fig, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(len(models))

    ax1.bar(x - 0.2, accuracy_pct, width=0.4, label="Accuracy (%)", color="#4E79A7")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=0)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, robustness_score, width=0.4, label="Robustness (↑)", color="#F28E2B")
    ax2.set_ylabel("Robustness score (normalized)")

    lines, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        lines += h
        labels += l
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title("Does robust image processing improve evaluation metrics?")
    plt.tight_layout()
    plt.savefig("accuracy_vs_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_gpu_hours(models, accuracy_pct, gpu_hours):

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(gpu_hours, accuracy_pct, s=140, c=np.arange(len(models)), cmap="viridis")
    for i, name in enumerate(models):
        ax.annotate(name, (gpu_hours[i], accuracy_pct[i]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Training time (GPU hours)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("How much does robustness help reduce GPU hours?")
    ax.grid(True, alpha=0.25)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Model index")

    plt.tight_layout()
    plt.savefig("accuracy_vs_gpu_hours.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():

    # Replace these with your measured results.
    # Provide exactly two variants if you want a clean A/B comparison.
    models = ["Baseline", "Robust"]

    # Example placeholders:
    # - accuracy_pct: final evaluation accuracy
    # - robustness_score: normalized robustness (e.g., avg acc under corruptions, 0-1)
    # - gpu_hours: total training time
    accuracy_pct = [88.5, 89.3]
    robustness_score = [0.62, 0.78]
    gpu_hours = [40.0, 38.5]

    plot_accuracy_vs_robustness(models, accuracy_pct, robustness_score)
    plot_accuracy_vs_gpu_hours(models, accuracy_pct, gpu_hours)

    print("Generated files:")
    print("- accuracy_vs_robustness.png")
    print("- accuracy_vs_gpu_hours.png")


if __name__ == "__main__":
    main()


