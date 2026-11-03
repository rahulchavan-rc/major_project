"""
Generate Confusion Matrix and Training/Validation Curves for MobileNetV3.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/rahulchavan3134gmail.com/Desktop/pro copy 2")
TRAINING_LOG  = PROJECT_ROOT / "mobilenetv3_runs" / "training_log.csv"
PREDICTIONS   = PROJECT_ROOT / "mobilenetv3_runs" / "test_predictions.csv"
CLASS_NAMES   = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]

OUTPUT_DIR = Path("/Users/rahulchavan3134gmail.com/Desktop/imp")
CM_OUT     = OUTPUT_DIR / "confusion_matrix.png"
CURVE_OUT  = OUTPUT_DIR / "training_validation_curves.png"

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
SURFACE      = "#1c1f2e"
ACCENT_BLUE  = "#4f8ef7"
ACCENT_PINK  = "#f7658e"
ACCENT_GREEN = "#3dd68c"
ACCENT_GOLD  = "#f7c948"
GRID_COLOR   = "#2a2d3e"
TEXT_COLOR   = "#e8eaf6"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  GRID_COLOR,
})


# ── Load training log ─────────────────────────────────────────────────────────
def load_training_log(path: Path):
    epochs, acc, loss, val_acc, val_loss = [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]) + 1)
            acc.append(float(row["accuracy"]))
            loss.append(float(row["loss"]))
            val_acc.append(float(row["val_accuracy"]))
            val_loss.append(float(row["val_loss"]))
    return epochs, acc, loss, val_acc, val_loss


# ── Load predictions → confusion matrix ──────────────────────────────────────
def load_confusion_matrix(path: Path, class_names: list[str]) -> np.ndarray:
    label_map = {name.lower(): i for i, name in enumerate(class_names)}
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true  = row["true_label"].strip().lower()
            pred  = row["predicted_label"].strip().lower()
            ti = label_map.get(true)
            pi = label_map.get(pred)
            if ti is not None and pi is not None:
                cm[ti][pi] += 1
    return cm


# ── Plot 1: Training / Validation Curves ─────────────────────────────────────
def plot_curves(epochs, acc, loss, val_acc, val_loss, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MobileNetV3 — Training vs Validation", fontsize=18, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax1.set_facecolor(SURFACE)
    ax1.plot(epochs, acc, color=ACCENT_BLUE, linewidth=2.5, marker="o",
             markersize=5, label="Train Accuracy", zorder=3)
    ax1.plot(epochs, val_acc, color=ACCENT_PINK, linewidth=2.5, marker="s",
             markersize=5, linestyle="--", label="Val Accuracy", zorder=3)
    ax1.fill_between(epochs, acc, val_acc,
                     color=ACCENT_BLUE, alpha=0.08)

    # best val accuracy marker
    best_epoch = epochs[int(np.argmax(val_acc))]
    best_val   = max(val_acc)
    ax1.axvline(x=best_epoch, color=ACCENT_GREEN, linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.scatter([best_epoch], [best_val], color=ACCENT_GREEN, s=120, zorder=5,
                label=f"Best Val Acc: {best_val*100:.1f}% (epoch {best_epoch})")

    ax1.set_title("Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.legend(framealpha=0.85, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax2.set_facecolor(SURFACE)
    ax2.plot(epochs, loss, color=ACCENT_BLUE, linewidth=2.5, marker="o",
             markersize=5, label="Train Loss", zorder=3)
    ax2.plot(epochs, val_loss, color=ACCENT_PINK, linewidth=2.5, marker="s",
             markersize=5, linestyle="--", label="Val Loss", zorder=3)
    ax2.fill_between(epochs, loss, val_loss,
                     color=ACCENT_PINK, alpha=0.08)

    best_loss_epoch = epochs[int(np.argmin(val_loss))]
    best_val_loss   = min(val_loss)
    ax2.axvline(x=best_loss_epoch, color=ACCENT_GREEN, linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.scatter([best_loss_epoch], [best_val_loss], color=ACCENT_GREEN, s=120, zorder=5,
                label=f"Best Val Loss: {best_val_loss:.4f} (epoch {best_loss_epoch})")

    ax2.set_title("Loss", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(framealpha=0.85, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✅ Saved curves  →  {out_path}")


# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    # Normalised version (row-wise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MobileNetV3 — Confusion Matrix (Test Set)", fontsize=18,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)

    _cmap = sns.color_palette("Blues", as_cmap=True)

    for ax, data, title, fmt in [
        (ax_raw,  cm,      "Counts",      "d"),
        (ax_norm, cm_norm, "Normalised",  ".2%"),
    ]:
        ax.set_facecolor(SURFACE)
        sns.heatmap(
            data, ax=ax,
            annot=True, fmt=fmt, cmap=_cmap,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor=DARK_BG,
            cbar_kws={"shrink": 0.85},
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

        # colour the colorbar background
        ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)

    # Per-class accuracy text box
    per_class = np.diag(cm_norm)
    summary = "\n".join(
        f"{name}: {acc*100:.1f}%"
        for name, acc in zip(class_names, per_class)
    )
    overall = np.trace(cm) / cm.sum()
    summary = f"Overall: {overall*100:.2f}%\n" + "─" * 20 + "\n" + summary
    fig.text(0.5, -0.03, summary, ha="center", va="top",
             fontsize=11, color=TEXT_COLOR,
             bbox=dict(facecolor=SURFACE, edgecolor=GRID_COLOR, boxstyle="round,pad=0.6"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"✅ Saved confusion matrix  →  {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training log …")
    epochs, acc, loss, val_acc, val_loss = load_training_log(TRAINING_LOG)
    print(f"   {len(epochs)} epochs found")

    print("Loading test predictions …")
    cm = load_confusion_matrix(PREDICTIONS, [c.lower() for c in CLASS_NAMES])
    print(f"   Total predictions: {cm.sum()}")

    print("Plotting training/validation curves …")
    plot_curves(epochs, acc, loss, val_acc, val_loss, CURVE_OUT)

    print("Plotting confusion matrix …")
    plot_confusion_matrix(cm, CLASS_NAMES, CM_OUT)

    print("\nDone! 🎉")
