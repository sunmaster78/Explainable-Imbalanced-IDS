"""
GAN Quality Evaluation Module.

Provides empirical evaluation metrics and diagnostic visualizations to evaluate
the quality, diversity, and realism of synthetic attack samples produced by the
Conditional WGAN-GP generator:

1. TSTR (Train on Synthetic, Test on Real): Trains a classifier exclusively on
   synthetic attack data and evaluates generalization on real test attacks.
2. Nearest-Neighbor Distance: Evaluates sample diversity and memorization risk
   by measuring synthetic-to-real vs. real-to-real nearest neighbor distances.
3. Feature-wise Kolmogorov-Smirnov (KS) Test: Compares marginal distributions of
   synthetic and real samples across all features.
4. 2D PCA Scatter Visualization: Real vs. synthetic distribution alignment.

CRITICAL: Test data is only used for TSTR evaluation; real nearest neighbors and
KS statistics are computed strictly against the training partition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import NearestNeighbors

from src.utils import get_logger, save_figure, save_table

logger = get_logger("gan_quality")


# ---------------------------------------------------------------------------
# 1. TSTR (Train on Synthetic, Test on Real) Score
# ---------------------------------------------------------------------------

def tstr_score(
    X_syn: Union[np.ndarray, pd.DataFrame],
    y_syn: Union[np.ndarray, pd.Series],
    X_real_test: Union[np.ndarray, pd.DataFrame],
    y_real_test: Union[np.ndarray, pd.Series],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    n_estimators: int = 100,
    table_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a RandomForest classifier on synthetic data only and evaluate on real test data.

    High TSTR score confirms that synthetic data captures predictive decision
    boundaries rather than arbitrary noise.

    Args:
        X_syn: Synthetic features array of shape (N_syn, D).
        y_syn: Synthetic labels array of shape (N_syn,).
        X_real_test: Real test features array of shape (N_test, D).
        y_real_test: Real test labels array of shape (N_test,).
        config: Optional configuration dict.
        seed: Random seed for classifier initialization.
        n_estimators: Number of trees in the RandomForest.
        table_name: Optional filename to save per-class metrics table.

    Returns:
        Dict containing:
            - 'macro_f1': Macro-averaged F1 score on real test set.
            - 'per_class_recall': Dict mapping class -> recall on real test set.
            - 'per_class_f1': Dict mapping class -> F1 on real test set.
            - 'model': Fitted RandomForestClassifier.
    """
    X_s = np.asarray(X_syn, dtype=np.float32)
    y_s = np.asarray(y_syn)
    X_t = np.asarray(X_real_test, dtype=np.float32)
    y_t = np.asarray(y_real_test)

    if len(X_s) == 0:
        raise ValueError("X_syn cannot be empty for TSTR evaluation.")

    # Unique classes present in synthetic training set
    syn_classes = np.unique(y_s)
    test_classes = np.unique(y_t)
    all_eval_classes = np.unique(np.concatenate([syn_classes, test_classes]))

    logger.info(
        f"Evaluating TSTR: {len(X_s)} synthetic train samples, "
        f"{len(X_t)} real test samples across classes {list(all_eval_classes)}"
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_s, y_s)
    y_pred = clf.predict(X_t)

    # Compute macro-F1 and per-class recall
    macro_f1 = float(f1_score(y_t, y_pred, average="macro", zero_division=0))
    recalls = recall_score(y_t, y_pred, labels=all_eval_classes, average=None, zero_division=0)
    f1s = f1_score(y_t, y_pred, labels=all_eval_classes, average=None, zero_division=0)

    per_class_recall = {cls: float(r) for cls, r in zip(all_eval_classes, recalls)}
    per_class_f1 = {cls: float(f) for cls, f in zip(all_eval_classes, f1s)}

    logger.info(f"TSTR Macro-F1: {macro_f1:.4f}")
    for cls in all_eval_classes:
        logger.info(f"  Class '{cls}' Recall: {per_class_recall[cls]:.4f}, F1: {per_class_f1[cls]:.4f}")

    if table_name and config:
        rows = [
            {"class": str(c), "recall": per_class_recall[c], "f1": per_class_f1[c]}
            for c in all_eval_classes
        ]
        save_table(pd.DataFrame(rows), table_name, config)

    return {
        "macro_f1": macro_f1,
        "tstr_macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "model": clf,
    }


# ---------------------------------------------------------------------------
# 2. Nearest-Neighbor Distance (Memorization vs. Generalization)
# ---------------------------------------------------------------------------

def nearest_neighbor_distance(
    X_syn: Union[np.ndarray, pd.DataFrame],
    X_real_train: Union[np.ndarray, pd.DataFrame],
    threshold: float = 1e-3,
    class_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute Euclidean nearest-neighbor distances from synthetic to real training samples.

    Checks whether the generator is truly generating novel samples or merely
    memorizing and duplicating training points:
    - Distance < threshold (1e-3) indicates potential memorization.
    - Compares synthetic-to-real NN distance with real-to-real NN distance.

    Args:
        X_syn: Synthetic samples of shape (N_syn, D).
        X_real_train: Real training samples of shape (N_real, D).
        threshold: Distance cutoff for considering a sample memorized (default 1e-3).
        class_name: Optional attack class name for logging warnings (e.g. 'U2R').

    Returns:
        Dict with keys:
            - 'syn_mean_dist': Mean distance from each synthetic sample to closest real sample.
            - 'syn_min_dist': Minimum distance from any synthetic sample to real data.
            - 'syn_memorized_pct': Percentage of synthetic samples with distance < threshold.
            - 'real_mean_dist': Mean distance from each real sample to its closest neighbor.
            - 'real_min_dist': Minimum distance between distinct real samples.
            - 'real_memorized_pct': Percentage of real samples with distance < threshold.
    """
    X_s = np.asarray(X_syn, dtype=np.float32)
    X_r = np.asarray(X_real_train, dtype=np.float32)

    if len(X_s) == 0:
        raise ValueError("X_syn cannot be empty.")
    if len(X_r) == 0:
        raise ValueError("X_real_train cannot be empty.")

    # 1. Synthetic -> Real 1-NN
    nn_syn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_syn.fit(X_r)
    distances_syn, _ = nn_syn.kneighbors(X_s)
    dists_syn = distances_syn.flatten()

    syn_mean = float(np.mean(dists_syn))
    syn_min = float(np.min(dists_syn))
    syn_memorized = float(np.mean(dists_syn < threshold) * 100.0)

    # 2. Real -> Real 2-NN (first neighbor is self, second is nearest distinct sample)
    if len(X_r) > 1:
        nn_real = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn_real.fit(X_r)
        distances_real, _ = nn_real.kneighbors(X_r)
        dists_real = distances_real[:, 1]  # distance to closest distinct real point

        real_mean = float(np.mean(dists_real))
        real_min = float(np.min(dists_real))
        real_memorized = float(np.mean(dists_real < threshold) * 100.0)
    else:
        # Edge case: single training sample
        real_mean = float("nan")
        real_min = float("nan")
        real_memorized = float("nan")

    tag = f" [{class_name}]" if class_name else ""
    logger.info(
        f"NN Distance{tag}: Syn-to-Real mean={syn_mean:.4f}, min={syn_min:.6f}, "
        f"memorized (<{threshold})={syn_memorized:.2f}% | "
        f"Real-to-Real mean={real_mean:.4f}, min={real_min:.6f}"
    )

    if syn_memorized > 0.0 or syn_min < threshold:
        logger.warning(
            f"Potential memorization detected{tag}! {syn_memorized:.2f}% of synthetic "
            f"samples are within {threshold} Euclidean distance of training data."
        )

    return {
        "syn_mean_dist": syn_mean,
        "syn_min_dist": syn_min,
        "syn_memorized_pct": syn_memorized,
        "memorization_rate_pct": syn_memorized,
        "real_mean_dist": real_mean,
        "real_min_dist": real_min,
        "real_memorized_pct": real_memorized,
    }


# ---------------------------------------------------------------------------
# 3. Kolmogorov-Smirnov (KS) Test per Feature
# ---------------------------------------------------------------------------

def ks_test_per_feature(
    X_syn: Union[np.ndarray, pd.DataFrame],
    X_real: Union[np.ndarray, pd.DataFrame],
    alpha: float = 0.05,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform a two-sample Kolmogorov-Smirnov test per feature comparing real vs synthetic.

    Evaluates whether the continuous and categorical marginal distributions
    align closely with real data:
    - Lower KS statistic indicates higher fidelity.
    - Features with p < alpha have statistically significant distributional divergence.

    Args:
        X_syn: Synthetic samples of shape (N_syn, D).
        X_real: Real samples of shape (N_real, D).
        alpha: Significance level threshold (default 0.05).
        feature_names: Optional list of feature names of length D.

    Returns:
        Dict with keys:
            - 'mean_ks_stat': Mean KS statistic across all features.
            - 'p_less_alpha_count': Count of features where p < alpha.
            - 'total_features': Total number of features evaluated.
            - 'p_less_alpha_pct': Percentage of features where p < alpha.
            - 'feature_stats': DataFrame with per-feature stats ('feature', 'ks_stat', 'p_value').
    """
    X_s = np.asarray(X_syn, dtype=np.float32)
    X_r = np.asarray(X_real, dtype=np.float32)

    if X_s.shape[1] != X_r.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: synthetic has {X_s.shape[1]}, "
            f"real has {X_r.shape[1]}."
        )

    n_features = X_s.shape[1]
    rows = []

    for j in range(n_features):
        feat_name = feature_names[j] if feature_names and j < len(feature_names) else f"feature_{j}"
        val_syn = X_s[:, j]
        val_real = X_r[:, j]

        # Edge case: both constant
        if np.all(val_syn == val_syn[0]) and np.all(val_real == val_real[0]) and val_syn[0] == val_real[0]:
            ks_stat = 0.0
            p_val = 1.0
        else:
            res = stats.ks_2samp(val_syn, val_real)
            ks_stat = float(res.statistic)
            p_val = float(res.pvalue)

        rows.append({
            "feature": feat_name,
            "ks_stat": ks_stat,
            "p_value": p_val,
            "p_less_alpha": p_val < alpha,
        })

    df = pd.DataFrame(rows)
    mean_ks = float(df["ks_stat"].mean())
    p_count = int(df["p_less_alpha"].sum())
    p_pct = float(p_count / n_features * 100.0)

    logger.info(
        f"KS Test: Mean KS statistic = {mean_ks:.4f}, "
        f"{p_count}/{n_features} features ({p_pct:.1f}%) have p < {alpha}."
    )

    return {
        "mean_ks_stat": mean_ks,
        "p_less_alpha_count": p_count,
        "total_features": n_features,
        "p_less_alpha_pct": p_pct,
        "feature_stats": df,
    }


# ---------------------------------------------------------------------------
# 4. 2D PCA Visualization (Real vs. Synthetic)
# ---------------------------------------------------------------------------

def pca_plot(
    X_real: Union[np.ndarray, pd.DataFrame],
    X_syn: Union[np.ndarray, pd.DataFrame],
    class_name: str,
    config: Optional[Dict[str, Any]] = None,
    n_components: int = 2,
    save_name: Optional[str] = None,
    max_plot_points: int = 2000,
    seed: int = 42,
) -> Tuple[matplotlib.figure.Figure, Optional[str]]:
    """
    Generate and save a 2D PCA scatter plot comparing real vs. synthetic distributions.

    PCA projection is fitted strictly on the real samples of the class and then
    used to transform both real and synthetic points for unbiased visual alignment.

    Args:
        X_real: Real samples of shape (N_real, D).
        X_syn: Synthetic samples of shape (N_syn, D).
        class_name: Attack class label (e.g. 'DoS', 'Probe', 'R2L', 'U2R').
        config: Optional configuration dict for output directories.
        n_components: Number of PCA dimensions (default 2).
        save_name: Optional filename for the saved figure.
        max_plot_points: Maximum number of points to plot per group to avoid overplotting.
        seed: Random seed for sampling if data exceeds max_plot_points.

    Returns:
        Tuple of (matplotlib.figure.Figure, saved_path_str).
    """
    X_r = np.asarray(X_real, dtype=np.float32)
    X_s = np.asarray(X_syn, dtype=np.float32)

    # Subsample if necessary to keep plot clean and fast
    rng = np.random.RandomState(seed)
    if len(X_r) > max_plot_points:
        idx_r = rng.choice(len(X_r), size=max_plot_points, replace=False)
        X_r_plot = X_r[idx_r]
    else:
        X_r_plot = X_r

    if len(X_s) > max_plot_points:
        idx_s = rng.choice(len(X_s), size=max_plot_points, replace=False)
        X_s_plot = X_s[idx_s]
    else:
        X_s_plot = X_s

    # Fit PCA on real training data (or combined if real has fewer samples than n_components)
    pca = PCA(n_components=n_components, random_state=seed)
    if len(X_r) >= n_components:
        pca.fit(X_r)
    else:
        # Fall back to fitting on combined real + synthetic
        pca.fit(np.vstack([X_r, X_s]))

    r_proj = pca.transform(X_r_plot)
    s_proj = pca.transform(X_s_plot)

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        r_proj[:, 0],
        r_proj[:, 1],
        c="#1f77b4",
        alpha=0.6,
        s=30,
        label=f"Real {class_name} (n={len(X_r_plot)})",
        edgecolors="none",
    )
    ax.scatter(
        s_proj[:, 0],
        s_proj[:, 1],
        c="#ff7f0e",
        alpha=0.6,
        s=35,
        marker="^",
        label=f"Synthetic {class_name} (n={len(X_s_plot)})",
        edgecolors="none",
    )

    ax.set_title(
        f"2D PCA Projection: Real vs. Synthetic ({class_name})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(f"PC 1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var2:.1f}% variance)", fontsize=11)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    filename = save_name or f"pca_{class_name.lower()}"
    saved_path = None
    if config:
        saved_path = str(save_figure(fig, filename, config))
        logger.info(f"Saved PCA plot for '{class_name}' to {saved_path}")

    return fig, saved_path


# ---------------------------------------------------------------------------
# 5. Full Evaluation Orchestrator: gan_quality Summary Table
# ---------------------------------------------------------------------------

def evaluate_gan_quality(
    X_train_dict: Dict[str, np.ndarray],
    X_syn_dict: Dict[str, np.ndarray],
    X_real_test: np.ndarray,
    y_real_test: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    table_name: str = "gan_quality",
    make_plots: bool = True,
) -> pd.DataFrame:
    """
    Run all four GAN quality checks per attack class and compile a master 'gan_quality' table.

    1. Computes TSTR across the synthetic attack pool vs real test attacks.
    2. Computes nearest-neighbor distance (syn-to-real vs real-to-real) per class.
    3. Computes Kolmogorov-Smirnov test per class.
    4. Generates 2D PCA plots per class.
    5. Saves the final summary table to results/tables/gan_quality.csv.
    6. Verifies U2R memorization metrics.

    Args:
        X_train_dict: Dict mapping class_name -> real training feature array.
        X_syn_dict: Dict mapping class_name -> synthetic feature array.
        X_real_test: Real test features array.
        y_real_test: Real test labels array.
        config: Optional configuration dict.
        table_name: Name of CSV table (default 'gan_quality').
        make_plots: Whether to generate and save PCA plots (default True).

    Returns:
        Summary DataFrame containing the empirical quality metrics per class.
    """
    classes = [c for c in X_syn_dict if len(X_syn_dict[c]) > 0]
    logger.info(f"Starting comprehensive GAN quality evaluation for classes: {classes}")

    # --- 1. TSTR evaluation across all classes ---
    all_X_syn = []
    all_y_syn = []
    for cls in classes:
        all_X_syn.append(X_syn_dict[cls])
        all_y_syn.append(np.full(len(X_syn_dict[cls]), cls))

    if all_X_syn:
        X_syn_full = np.vstack(all_X_syn)
        y_syn_full = np.concatenate(all_y_syn)
        tstr_res = tstr_score(
            X_syn=X_syn_full,
            y_syn=y_syn_full,
            X_real_test=X_real_test,
            y_real_test=y_real_test,
            config=config,
            table_name=f"{table_name}_tstr_details",
        )
    else:
        tstr_res = {"macro_f1": 0.0, "per_class_recall": {}, "per_class_f1": {}}

    # --- 2. Per-class metrics: NN distance, KS test, and PCA plots ---
    summary_rows = []

    for cls in classes:
        x_syn = X_syn_dict[cls]
        x_real = X_train_dict.get(cls)

        if x_real is None or len(x_real) == 0:
            logger.warning(f"No real training data provided for class '{cls}', skipping NN and KS.")
            continue

        # Nearest neighbor analysis
        nn_res = nearest_neighbor_distance(
            X_syn=x_syn,
            X_real_train=x_real,
            threshold=1e-3,
            class_name=cls,
        )

        # KS test
        ks_res = ks_test_per_feature(
            X_syn=x_syn,
            X_real=x_real,
            alpha=0.05,
        )

        # PCA plot
        if make_plots:
            fig, _ = pca_plot(
                X_real=x_real,
                X_syn=x_syn,
                class_name=cls,
                config=config,
            )
            plt.close(fig)

        # Compile row
        cls_recall = tstr_res["per_class_recall"].get(cls, float("nan"))
        cls_f1 = tstr_res["per_class_f1"].get(cls, float("nan"))

        summary_rows.append({
            "class": cls,
            "n_real_train": len(x_real),
            "n_synthetic": len(x_syn),
            "tstr_recall": cls_recall,
            "tstr_f1": cls_f1,
            "tstr_macro_f1": tstr_res["macro_f1"],
            "syn_mean_nn_dist": nn_res["syn_mean_dist"],
            "syn_min_nn_dist": nn_res["syn_min_dist"],
            "syn_memorized_pct": nn_res["syn_memorized_pct"],
            "real_mean_nn_dist": nn_res["real_mean_dist"],
            "ks_mean_stat": ks_res["mean_ks_stat"],
            "ks_diff_feature_count": ks_res["p_less_alpha_count"],
            "ks_total_features": ks_res["total_features"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save summary table
    if config:
        saved_path = save_table(summary_df, table_name, config)
        logger.info(f"Saved GAN quality summary table to {saved_path}")

    # Specific check for U2R memorization
    u2r_rows = summary_df[summary_df["class"].str.upper() == "U2R"]
    if not u2r_rows.empty:
        u2r_row = u2r_rows.iloc[0]
        u2r_mem = u2r_row["syn_memorized_pct"]
        u2r_min = u2r_row["syn_min_nn_dist"]
        logger.info(
            f"=== U2R Memorization Audit ===\n"
            f"Synthetic U2R samples: {u2r_row['n_synthetic']}\n"
            f"Minimum distance to real U2R: {u2r_min:.6f}\n"
            f"Memorized percentage (<1e-3): {u2r_mem:.2f}%\n"
            f"Conclusion: {'PASSED (No exact memorization detected)' if u2r_mem == 0.0 else 'WARNING (Potential memorization)'}"
        )

    return summary_df
