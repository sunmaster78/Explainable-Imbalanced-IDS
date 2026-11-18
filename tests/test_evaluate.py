"""
Unit tests for evaluation and visualization in src/evaluate.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import plot_class_distribution


def test_plot_class_distribution(tmp_path: Path):
    """Test generating and saving class distribution bar charts."""
    config = {"output_dirs": {"figures": str(tmp_path / "figures")}}

    # Multiclass test series
    y_multi = pd.Series(["Normal"] * 100 + ["DoS"] * 50 + ["Probe"] * 15 + ["R2L"] * 5 + ["U2R"] * 1)
    fig_path = plot_class_distribution(
        y_multi,
        title="Test Multiclass Distribution",
        config=config,
        log_scale=True,
        filename="test_multi_dist",
    )

    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
    assert fig_path.suffix == ".png"

    # Binary test series
    y_bin = pd.Series([0] * 80 + [1] * 20)
    fig_bin_path = plot_class_distribution(
        y_bin,
        title="Test Binary Distribution",
        config=config,
        log_scale=False,
        filename="test_bin_dist",
    )

    assert fig_bin_path.exists()
    assert fig_bin_path.stat().st_size > 0


def test_compute_metrics():
    """Test compute_metrics returns all required metrics."""
    from src.evaluate import compute_metrics

    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    y_true = np.array(["Normal", "DoS", "Probe", "R2L", "U2R", "Normal", "DoS", "Normal"])
    y_pred = np.array(["Normal", "DoS", "Probe", "R2L", "Normal", "Normal", "DoS", "DoS"])

    # Probabilities dummy
    rng = np.random.RandomState(42)
    y_proba = rng.dirichlet(np.ones(len(classes)), size=len(y_true))

    metrics = compute_metrics(y_true, y_pred, y_proba=y_proba, classes=classes)

    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "mcc" in metrics
    assert "auc_pr_macro" in metrics
    assert "fpr_normal" in metrics
    assert "per_class" in metrics

    # Normal FPR check: 3 normal samples, 1 misclassified as DoS -> FPR = 1/3
    assert metrics["fpr_normal"] == pytest.approx(1.0 / 3.0)

    # Flattened per-class keys
    assert "f1_Normal" in metrics
    assert "recall_U2R" in metrics


def test_plot_confusion_matrix(tmp_path: Path):
    """Test generating and saving row-normalized confusion matrix heatmap."""
    from src.evaluate import plot_confusion_matrix

    config = {"output_dirs": {"figures": str(tmp_path / "figures")}}
    classes = ["Normal", "DoS", "Probe"]
    y_true = ["Normal"] * 10 + ["DoS"] * 10 + ["Probe"] * 10
    y_pred = ["Normal"] * 8 + ["DoS"] * 2 + ["DoS"] * 9 + ["Probe"] * 1 + ["Probe"] * 10

    cm_path = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        title="Test CM",
        config=config,
        filename="test_cm",
    )

    assert cm_path.exists()
    assert cm_path.stat().st_size > 0
    assert cm_path.suffix == ".png"

