"""
Publication-Quality Figure Generation Module (Phase 9).

Regenerates all paper figures from saved CSV results and model artifacts (not by retraining),
enforcing consistent academic styling:
- Font size: 10 pt
- Palettes: Colour-blind friendly (e.g. Wong palette / Seaborn colorblind)
- Export formats: Both 300 DPI PNG and vector PDF
- Layout: Clean axis labels, no clutter, suitable for two-column IEEE/ACM paper formatting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger, load_config

logger = get_logger("make_figures")

# Global academic style configuration
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colour-blind friendly palette (Wong 2011)
CB_COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "dark_blue": "#56B4E9",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}


def _save_dual_format(fig: plt.Figure, base_path: Path) -> Tuple[Path, Path]:
    """Save figure as both 300 DPI PNG and vector PDF."""
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# 1. Class Distribution (Before and After Oversampling)
# ---------------------------------------------------------------------------

def plot_class_distribution_comparison(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "gan_oversample_class_counts.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate before-and-after class distribution bar chart from oversampling table."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"Class count table not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    required = {"class", "count_before", "count_after"}
    if not required.issubset(df.columns):
        logger.warning(f"CSV {csv_file} missing required columns {required}")
        return None

    classes = df["class"].astype(str)
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    b1 = ax.bar(x - width / 2, df["count_before"], width, label="Real (Before)", color=CB_COLORS["blue"], edgecolor="#222222", linewidth=0.7)
    b2 = ax.bar(x + width / 2, df["count_after"], width, label="Augmented (After WGAN-GP)", color=CB_COLORS["orange"], edgecolor="#222222", linewidth=0.7)

    ax.set_ylabel("Sample Count (Log Scale)")
    ax.set_xlabel("Attack Class")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Value annotations on bars
    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig1_class_distribution_comparison")
    plt.close(fig)
    logger.info(f"Saved class distribution comparison to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# 2. WGAN-GP Loss Curves
# ---------------------------------------------------------------------------

def plot_wgan_loss_curves(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "wgan_loss_history.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate publication WGAN-GP training loss curves from saved history CSV."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"WGAN loss history not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=300)

    # Critic loss
    axes[0].plot(epochs, df["critic_loss"], color=CB_COLORS["red"], linewidth=1.4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Critic Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Generator loss
    axes[1].plot(epochs, df["generator_loss"], color=CB_COLORS["green"], linewidth=1.4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Generator Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Estimated Wasserstein distance
    axes[2].plot(epochs, df["wasserstein_dist"], color=CB_COLORS["blue"], linewidth=1.4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Wasserstein Distance (Est.)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig2_wgan_loss_curves")
    plt.close(fig)
    logger.info(f"Saved WGAN loss curves to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# 3. Per-Class Recall across All Methods
# ---------------------------------------------------------------------------

def plot_per_class_recall_benchmark(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "raw_results.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate per-class recall grouped bar chart across all evaluated benchmark methods."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"Raw results file not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    recall_cols = [c for c in df.columns if c.startswith("recall_") and c != "recall_macro"]
    if not recall_cols:
        logger.warning("No per-class recall columns found in raw_results.csv")
        return None

    # Group by method and compute mean recall
    mean_df = df.groupby("method")[recall_cols].mean()
    class_names = [c.replace("recall_", "") for c in recall_cols]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    methods = list(mean_df.index)
    n_methods = len(methods)
    x = np.arange(len(class_names))
    width = 0.8 / max(n_methods, 1)

    palette = sns.color_palette("colorblind", n_colors=n_methods)

    for i, method in enumerate(methods):
        y_vals = mean_df.loc[method, recall_cols].values
        ax.bar(
            x + (i - (n_methods - 1) / 2) * width,
            y_vals,
            width=width,
            label=str(method),
            color=palette[i],
            edgecolor="#222222",
            linewidth=0.6,
        )

    ax.set_ylabel("Test Recall")
    ax.set_xlabel("Traffic Category")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig3_per_class_recall_benchmark")
    plt.close(fig)
    logger.info(f"Saved per-class recall benchmark to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# 4. SHAP Fidelity Comparison (Novelty Claim 1)
# ---------------------------------------------------------------------------

def plot_shap_fidelity_summary(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "shap_fidelity.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate grouped bar chart comparing SHAP explanation fidelity across synthetic generators."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"SHAP fidelity file not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    classes = df["class"].unique()
    metrics = ["spearman", "jaccard_top10", "cosine"]
    metric_labels = ["Spearman Rank Corr.", "Top-10 Jaccard", "Cosine Similarity"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.5), dpi=300, sharey=True)

    methods = df["method"].unique()
    palette = {"WGAN-GP": CB_COLORS["blue"], "SMOTE": CB_COLORS["orange"], "ROS": CB_COLORS["green"]}

    for ax_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        x = np.arange(len(classes))
        n_m = len(methods)
        width = 0.8 / n_m

        for m_idx, method in enumerate(methods):
            sub = df[df["method"] == method].set_index("class").reindex(classes)
            vals = sub[metric].values
            color = palette.get(method, "#666666")

            # Error bars from bootstrap CIs if available
            ci_low_col = f"{metric}_ci_low"
            ci_high_col = f"{metric}_ci_high"
            if ci_low_col in sub.columns and ci_high_col in sub.columns:
                err_low = vals - sub[ci_low_col].values
                err_high = sub[ci_high_col].values - vals
                yerr = np.vstack([np.maximum(0, err_low), np.maximum(0, err_high)])
            else:
                yerr = None

            ax.bar(
                x + (m_idx - (n_m - 1) / 2) * width,
                vals,
                width=width,
                label=method if ax_idx == 0 else "",
                color=color,
                yerr=yerr,
                capsize=3,
                edgecolor="#222222",
                linewidth=0.6,
            )

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_xlabel("Minority Attack Class")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax_idx == 0:
            ax.set_ylabel("Fidelity Score")
            ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig4_shap_fidelity_summary")
    plt.close(fig)
    logger.info(f"Saved SHAP fidelity summary to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# 5. Stage-Wise SHAP Feature Roles (Novelty Claim 2)
# ---------------------------------------------------------------------------

def plot_stagewise_feature_roles(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "stagewise_feature_roles.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate bar chart illustrating feature role partition: Stage 1 only vs Stage 2 only vs Shared."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"Feature roles table not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    if "status" not in df.columns:
        return None

    counts = df["status"].value_counts()
    role_order = ["Stage1_Only", "Stage2_Only", "Both"]
    ordered_counts = [counts.get(r, 0) for r in role_order]
    labels = ["Stage 1 Only\n(Macro Anomaly)", "Stage 2 Only\n(Attack Family)", "Both Stages\n(Core Traffic)"]
    colors = [CB_COLORS["blue"], CB_COLORS["orange"], CB_COLORS["green"]]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    bars = ax.bar(labels, ordered_counts, color=colors, edgecolor="#222222", linewidth=0.8, width=0.5)

    ax.set_ylabel("Feature Count (Top-10 Unions)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)} ({h / sum(ordered_counts) * 100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig5_stagewise_feature_roles")
    plt.close(fig)
    logger.info(f"Saved stage-wise feature roles to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# 6. Novelty Claim 3 Comparison: Flat GAN vs Attack-Only GAN
# ---------------------------------------------------------------------------

def plot_flat_vs_attack_only_comparison(
    tables_dir: Path,
    figures_dir: Path,
    csv_name: str = "flat_vs_attack_only.csv",
) -> Optional[Tuple[Path, Path]]:
    """Generate comparative summary between Flat WGAN-GP and Attack-Only WGAN-GP."""
    csv_file = tables_dir / csv_name
    if not csv_file.exists():
        logger.warning(f"Flat vs Attack-Only file not found at: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    conditions = df["condition"].tolist()

    metrics = [
        ("class_imbalance_ratio", "Imbalance Ratio (Lower is Better)"),
        ("critic_loss_stability_std", "Critic Std (Lower is Better)"),
        ("tstr_macro_f1", "TSTR Macro-F1 (Higher is Better)"),
        ("two_stage_macro_f1", "Pipeline Macro-F1 (Higher is Better)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), dpi=300)
    colors = [CB_COLORS["red"], CB_COLORS["blue"]]

    for ax_idx, (col, title) in enumerate(metrics):
        ax = axes[ax_idx]
        if col in df.columns:
            vals = df[col].values
            bars = ax.bar(conditions, vals, color=colors[:len(conditions)], edgecolor="#222222", linewidth=0.6, width=0.55)
            ax.set_title(title, fontsize=8.5, fontweight="bold")
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(["Flat", "Proposed"], fontsize=8.5)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.3f}" if h < 10 else f"{h:.1f}",
                            xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out = _save_dual_format(fig, figures_dir / "fig6_flat_vs_attack_only")
    plt.close(fig)
    logger.info(f"Saved flat vs attack-only comparison to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# Task 9.2: Complete Framework Architecture Diagram (Pure Matplotlib)
# ---------------------------------------------------------------------------

def draw_framework_architecture(
    figures_dir: Path,
    filename: str = "framework_architecture",
) -> Tuple[Path, Path]:
    """
    Generate an academic publication-ready system architecture diagram (no external graphviz required).
    Flow: Raw Traffic -> Preprocessing -> Stage 1 (Binary) -> Normal: Pass
                                                          -> Attack: Stage 2 (Multiclass)
                                                                     ^ WGAN-GP oversampling (Train)
                                                          -> Stage-Wise SHAP Explanations
    """
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, header=None, subtext=None, border_color="#111111"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.2,rounding_size=2.0",
            facecolor=color,
            edgecolor=border_color,
            linewidth=1.2,
        )
        ax.add_patch(box)
        cy = y + h / 2
        if header:
            ax.text(x + w / 2, y + h - 4, header, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#111111")
            ax.text(x + w / 2, cy - 1, text, ha="center", va="center", fontsize=8.5, color="#222222")
        else:
            ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=9, fontweight="bold", color="#111111")
        if subtext:
            ax.text(x + w / 2, y + 3, subtext, ha="center", va="bottom", fontsize=7.5, fontstyle="italic", color="#444444")

    def draw_arrow(x1, y1, x2, y2, label=None, color="#222222"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 1.8, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    # 1. Raw Traffic
    draw_box(4, 45, 16, 16, "Raw Network\nTraffic Flows", "#E8F0FE", subtext="(NSL / UNSW / CIC)")

    # 2. Preprocessing
    draw_box(26, 45, 18, 16, "Leak-Free Scaling &\nCategorical OHE", "#E8F0FE", header="Preprocessing", subtext="Train Fit Strictly")
    draw_arrow(20, 53, 26, 53)

    # 3. Stage 1 Binary Classifier
    draw_box(50, 45, 20, 16, "Soft-Voting Ensemble\n(RF + XGB + LGBM)", "#D2E3FC", header="STAGE 1: Macro Detection", subtext="Normal vs Attack")
    draw_arrow(44, 53, 50, 53)

    # Normal Exit (top right)
    draw_box(78, 68, 18, 12, "Normal Traffic\n(Allowed / Passed)", "#CEEAD6", border_color="#137333")
    draw_arrow(60, 61, 78, 74, label="P(Att) < θ (Normal)", color="#137333")

    # 4. Stage 2 Multiclass Classifier (bottom right)
    draw_box(50, 10, 20, 16, "Attack-Only Multiclass\n(RF + XGB + LGBM)", "#FCE8E6", header="STAGE 2: Attack Family", subtext="DoS / Probe / R2L / U2R")
    draw_arrow(60, 45, 60, 26, label="P(Att) ≥ θ (Attack)", color="#C5221F")

    # Stage 2 Output
    draw_box(78, 12, 18, 12, "Intrusion Alert &\nFamily Classification", "#FAD2CF", border_color="#C5221F")
    draw_arrow(70, 18, 78, 18)

    # 5. Conditional WGAN-GP (Oversampling loop strictly into Stage 2)
    draw_box(16, 10, 24, 16, "Conditional WGAN-GP\n(Attack Rows Only)", "#FEF7E0", header="Minority Oversampling", subtext="Training Phase Only", border_color="#E37400")
    draw_arrow(40, 18, 50, 18, label="Synthetic Minority", color="#E37400")

    # 6. Stage-Wise SHAP Explainability (spanning right)
    draw_box(78, 38, 18, 20, "Stage 1: Macro Beeswarm\nStage 2: Family Bar Plots\nLocal Waterfalls", "#F1F3F4", header="Stage-Wise SHAP", subtext="TreeExplainer Unified", border_color="#5F6368")
    draw_arrow(70, 53, 78, 50, color="#5F6368")
    draw_arrow(70, 24, 78, 44, color="#5F6368")

    out = _save_dual_format(fig, figures_dir / filename)
    plt.close(fig)
    logger.info(f"Saved publication framework architecture diagram to: {out[0]}")
    return out


# ---------------------------------------------------------------------------
# Master Regenerator
# ---------------------------------------------------------------------------

def generate_all_figures(config_path: Union[str, Path] = "configs/nslkdd.yaml") -> Dict[str, Any]:
    """Regenerate all paper figures from saved results and artifacts."""
    config = load_config(config_path)
    tbl_dir = Path(config.get("output_dirs", {}).get("tables", "results/tables"))
    fig_dir = Path(config.get("output_dirs", {}).get("figures", "results/figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("Regenerating Publication Figures (Phase 9)")
    logger.info(f"Tables Source: {tbl_dir}")
    logger.info(f"Figures Target: {fig_dir}")
    logger.info("============================================================")

    generated = {}

    # 1. Architecture diagram
    generated["architecture"] = draw_framework_architecture(fig_dir)

    # 2. Class distribution comparison
    generated["class_distribution"] = plot_class_distribution_comparison(tbl_dir, fig_dir)

    # 3. WGAN-GP loss curves
    generated["wgan_loss"] = plot_wgan_loss_curves(tbl_dir, fig_dir)

    # 4. Per-class recall
    generated["recall_benchmark"] = plot_per_class_recall_benchmark(tbl_dir, fig_dir)

    # 5. SHAP fidelity summary
    generated["shap_fidelity"] = plot_shap_fidelity_summary(tbl_dir, fig_dir)

    # 6. Stage-wise feature roles
    generated["feature_roles"] = plot_stagewise_feature_roles(tbl_dir, fig_dir)

    # 7. Flat vs Attack-Only comparison
    generated["flat_vs_attack_only"] = plot_flat_vs_attack_only_comparison(tbl_dir, fig_dir)

    logger.info("All paper figures regenerated successfully!")
    return generated


if __name__ == "__main__":
    generate_all_figures()
