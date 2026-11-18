"""
Evaluation and visualization module for the Two-Stage IDS pipeline.
Supports performance metrics, confusion matrices, class distribution plots,
and statistical significance tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from src.utils import get_logger, load_config, save_figure

logger = get_logger("evaluate")


def plot_class_distribution(
    y: Union[pd.Series, np.ndarray, list],
    title: str,
    config: Optional[Dict[str, Any]] = None,
    log_scale: bool = True,
    filename: Optional[str] = None,
) -> Path:
    """
    Generate and save a publication-quality bar chart showing class counts.
    Includes exact counts and percentages printed above each bar.
    Supports logarithmic y-axis to visualize severe minority classes (e.g. U2R, R2L).

    Args:
        y: Class labels (array-like of categories or binary integers).
        title: Title of the plot.
        config: Optional configuration dictionary containing output directories.
        log_scale: Whether to use a logarithmic scale for the y-axis (default: True).
        filename: Optional base filename for the saved figure. If None, derived from title.

    Returns:
        Path to the saved 300 DPI PNG figure.
    """
    # Convert input to pandas Series
    series = pd.Series(y)

    # Format binary numeric labels if needed
    if set(series.unique()).issubset({0, 1}):
        series = series.map({0: "Normal (0)", 1: "Attack (1)"})

    counts = series.value_counts()
    total = len(series)

    # Order standard NSL-KDD 5-class if present, otherwise descending
    standard_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    if all(cat in counts.index for cat in standard_order if cat in counts.index):
        ordered_indices = [cat for cat in standard_order if cat in counts.index]
        # Include any remaining classes
        ordered_indices += [cat for cat in counts.index if cat not in ordered_indices]
        counts = counts.reindex(ordered_indices)

    # Set up matplotlib style
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Use a curated, harmonious color palette
    palette = sns.color_palette("mako", n_colors=len(counts))
    bars = ax.bar(
        [str(idx) for idx in counts.index],
        counts.values,
        color=palette,
        edgecolor="#222222",
        linewidth=1.0,
        width=0.55,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)", fontsize=12, fontweight="bold", labelpad=8)
    else:
        ax.set_ylabel("Count", fontsize=12, fontweight="bold", labelpad=8)

    ax.set_xlabel("Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    # Add count and percentage labels above each bar
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        label_text = f"{int(height):,}\n({pct:.2f}%)"

        # Position text slightly above bar
        if log_scale:
            y_pos = height * 1.25
        else:
            y_pos = height + (total * 0.015)

        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="semibold",
            color="#111111",
        )

    # Adjust y-limit to avoid clipping labels
    if log_scale:
        ax.set_ylim(bottom=max(0.5, counts.min() * 0.3), top=counts.max() * 6)
    else:
        ax.set_ylim(top=counts.max() * 1.18)

    plt.tight_layout()

    # Determine filename
    if filename is None:
        safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filename = f"class_distribution_{safe_name}"

    output_path = save_figure(fig, filename, config=config, dpi=300)
    plt.close(fig)

    logger.info(f"Class distribution figure saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Task 6.1: Evaluation Metrics Suite
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_proba: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    classes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for binary or multiclass IDS evaluation:
        - Accuracy
        - Macro Precision, Recall, F1
        - Weighted F1
        - Matthews Correlation Coefficient (MCC)
        - Macro Area Under Precision-Recall Curve (AUC-PR, one-vs-rest via y_proba)
        - Per-class Precision, Recall, F1
        - False Positive Rate on Normal class (FPR_Normal)

    Args:
        y_true: Ground truth labels (strings or integers).
        y_pred: Predicted class labels.
        y_proba: Optional probability estimates of shape (N, n_classes).
        classes: Optional ordered list of class names. If None, derived from union.

    Returns:
        Dictionary containing overall and per-class performance metrics.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if classes is None:
        classes = sorted(list(np.unique(np.concatenate([y_t, y_p]))))

    # 1. Overall / Aggregated Metrics
    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, labels=classes, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p))

    # 2. Macro AUC-PR (One-vs-Rest)
    auc_pr_macro = float("nan")
    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=np.float32)
        if proba_arr.ndim == 2 and proba_arr.shape[1] == len(classes):
            try:
                # Binarize y_true for multi-class PR curve
                if len(classes) == 2:
                    # Binary classification
                    # Pos class is classes[1]
                    y_bin_pos = (y_t == classes[1]).astype(int)
                    auc_pr_macro = float(average_precision_score(y_bin_pos, proba_arr[:, 1]))
                else:
                    y_bin = label_binarize(y_t, classes=classes)
                    auc_pr_macro = float(
                        average_precision_score(y_bin, proba_arr, average="macro")
                    )
            except Exception as e:
                logger.warning(f"Could not compute macro AUC-PR: {e}")

    # 3. Per-class metrics
    precisions = precision_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    recalls = recall_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    f1s = f1_score(y_t, y_p, labels=classes, average=None, zero_division=0)

    per_class_dict: Dict[str, Dict[str, float]] = {}
    for cls, p, r, f in zip(classes, precisions, recalls, f1s):
        per_class_dict[str(cls)] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }

    # 4. FPR on Normal class (False alarms)
    normal_labels = [c for c in classes if str(c).strip().lower() in ("normal", "0", "benign")]
    fpr_normal = float("nan")
    if normal_labels:
        norm_label = normal_labels[0]
        norm_mask = y_t == norm_label
        total_normal = int(np.sum(norm_mask))
        if total_normal > 0:
            fp_normal = int(np.sum(norm_mask & (y_p != norm_label)))
            fpr_normal = float(fp_normal / total_normal)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "auc_pr_macro": auc_pr_macro,
        "fpr_normal": fpr_normal,
        "per_class": per_class_dict,
    }

    # Add flattened keys for direct DataFrame logging
    for cls, vals in per_class_dict.items():
        metrics[f"precision_{cls}"] = vals["precision"]
        metrics[f"recall_{cls}"] = vals["recall"]
        metrics[f"f1_{cls}"] = vals["f1"]

    return metrics


# ---------------------------------------------------------------------------
# Task 6.1: Row-Normalized Confusion Matrix Plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    classes: List[Any],
    title: str = "Confusion Matrix",
    config: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Generate and save a publication-quality confusion matrix heatmap.
    Displays row-normalized values (percentages) along with raw counts in each cell.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        classes: Ordered list of class labels for matrix axes.
        title: Figure title.
        config: Optional configuration dict containing output directory paths.
        filename: Base filename for the saved PNG.
        dpi: Resolution (default 300).

    Returns:
        Path to the saved PNG figure.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    cm = confusion_matrix(y_t, y_p, labels=classes)

    # Compute row-normalized proportions (Recall per cell)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Construct custom text annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=[str(c) for c in classes],
        yticklabels=[str(c) for c in classes],
        cbar=True,
        cbar_kws={"label": "Recall (Row-Normalized)"},
        linewidths=0.5,
        linecolor="#dddddd",
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Category", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()

    safe_name = filename or f"confusion_matrix_{title.lower().replace(' ', '_')}"
    output_path = save_figure(fig, safe_name, config=config, dpi=dpi)
    plt.close(fig)

    logger.info(f"Confusion matrix plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    from src.preprocess import load_nslkdd, map_attack_categories

    config_path = Path("configs/nslkdd.yaml")
    config = load_config(config_path) if config_path.exists() else None

    train_path = Path("dataset folder/archive/KDDTrain+.txt")
    test_path = Path("dataset folder/archive/KDDTest+.txt")

    if train_path.exists() and test_path.exists():
        train_df, _ = load_nslkdd(train_path, test_path)
        train_df = map_attack_categories(train_df)

        # 1. 5-Class Distribution Plot (Log Scale)
        plot_5class = plot_class_distribution(
            train_df["category"],
            title="NSL-KDD Training Set: 5-Class Distribution",
            config=config,
            log_scale=True,
            filename="nslkdd_train_5class_distribution",
        )

        # 2. Binary Distribution Plot (Linear Scale)
        plot_binary = plot_class_distribution(
            train_df["is_attack"],
            title="NSL-KDD Training Set: Binary Distribution (Normal vs Attack)",
            config=config,
            log_scale=False,
            filename="nslkdd_train_binary_distribution",
        )

        print(f"5-Class plot generated: {plot_5class}")
        print(f"Binary plot generated: {plot_binary}")
