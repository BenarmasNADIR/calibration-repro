"""Reliability diagram in the style of Guo et al. (2017), Figure 1 / Figure 4.

Blue bars: accuracy per confidence bin. Red hatched bars: gap to the diagonal ("Gap").
The ECE is printed in the lower-right corner.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration_metrics import ece, reliability_bins


def plot_reliability(ax, probs, labels, title="", n_bins=15):
    counts, acc, conf = reliability_bins(probs, labels, n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    left, width = edges[:-1], 1.0 / n_bins
    # Empty bins are drawn at height 0 with no gap.
    out = np.where(counts > 0, acc, 0.0)
    target = np.where(counts > 0, conf, 0.0)
    ax.bar(left, out, width=width, align="edge", color="#1f4fd1", edgecolor="black",
           linewidth=0.8, label="Outputs")
    ax.bar(left, target - out, bottom=out, width=width, align="edge", color="#ff9999",
           edgecolor="red", alpha=0.6, hatch="//", linewidth=0.8, label="Gap")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.95, 0.05, f"ECE={100 * ece(probs, labels, n_bins):.2f}%", ha="right", va="bottom",
            fontsize=11, bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
    return ax


def save_reliability_grid(panels, labels, path, suptitle=None, n_bins=15):
    """panels: list of (title, probs). Saves a 1 x len(panels) grid to `path`."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (title, probs) in zip(axes, panels):
        plot_reliability(ax, probs, labels, title, n_bins)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
