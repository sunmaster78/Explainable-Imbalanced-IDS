"""
Unit and Integration Tests for Phase 9: Publication Figures Module.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.make_figures import (
    draw_framework_architecture,
    plot_class_distribution_comparison,
    plot_flat_vs_attack_only_comparison,
    plot_per_class_recall_benchmark,
    plot_shap_fidelity_summary,
    plot_stagewise_feature_roles,
    plot_wgan_loss_curves,
)


@pytest.fixture
def mock_tables(tmp_path: Path):
    tbl_dir = tmp_path / "tables"
    fig_dir = tmp_path / "figures"
    tbl_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)

    # 1. Class count table
    df_counts = pd.DataFrame([
        {"class": "DoS", "count_before": 1000, "count_after": 1000},
        {"class": "Probe", "count_before": 500, "count_after": 500},
        {"class": "R2L", "count_before": 50, "count_after": 500},
        {"class": "U2R", "count_before": 10, "count_after": 500},
    ])
    df_counts.to_csv(tbl_dir / "gan_oversample_class_counts.csv", index=False)

    # 2. WGAN Loss History
    df_loss = pd.DataFrame([
        {"epoch": 1, "critic_loss": 5.0, "generator_loss": 0.5, "wasserstein_dist": -0.2},
        {"epoch": 2, "critic_loss": 3.0, "generator_loss": 0.8, "wasserstein_dist": -0.1},
        {"epoch": 3, "critic_loss": 2.0, "generator_loss": 1.1, "wasserstein_dist": -0.05},
    ])
    df_loss.to_csv(tbl_dir / "wgan_loss_history.csv", index=False)

    # 3. Raw results
    df_raw = pd.DataFrame([
        {"method": "B1", "recall_Normal": 0.95, "recall_DoS": 0.88, "recall_R2L": 0.15, "recall_U2R": 0.05},
        {"method": "Proposed", "recall_Normal": 0.94, "recall_DoS": 0.90, "recall_R2L": 0.75, "recall_U2R": 0.65},
    ])
    df_raw.to_csv(tbl_dir / "raw_results.csv", index=False)

    # 4. SHAP fidelity
    df_fid = pd.DataFrame([
        {"class": "R2L", "method": "WGAN-GP", "spearman": 0.82, "jaccard_top10": 0.70, "cosine": 0.91,
         "spearman_ci_low": 0.75, "spearman_ci_high": 0.88, "jaccard_ci_low": 0.6, "jaccard_ci_high": 0.8, "cosine_ci_low": 0.85, "cosine_ci_high": 0.95},
        {"class": "R2L", "method": "SMOTE", "spearman": 0.52, "jaccard_top10": 0.40, "cosine": 0.68,
         "spearman_ci_low": 0.45, "spearman_ci_high": 0.60, "jaccard_ci_low": 0.3, "jaccard_ci_high": 0.5, "cosine_ci_low": 0.60, "cosine_ci_high": 0.75},
        {"class": "U2R", "method": "WGAN-GP", "spearman": 0.78, "jaccard_top10": 0.60, "cosine": 0.86,
         "spearman_ci_low": 0.70, "spearman_ci_high": 0.85, "jaccard_ci_low": 0.5, "jaccard_ci_high": 0.7, "cosine_ci_low": 0.80, "cosine_ci_high": 0.92},
        {"class": "U2R", "method": "SMOTE", "spearman": 0.44, "jaccard_top10": 0.30, "cosine": 0.55,
         "spearman_ci_low": 0.35, "spearman_ci_high": 0.52, "jaccard_ci_low": 0.2, "jaccard_ci_high": 0.4, "cosine_ci_low": 0.45, "cosine_ci_high": 0.62},
    ])
    df_fid.to_csv(tbl_dir / "shap_fidelity.csv", index=False)

    # 5. Feature roles
    df_roles = pd.DataFrame([
        {"feature": f"f_{i}", "status": "Stage1_Only" if i < 4 else ("Stage2_Only" if i < 8 else "Both")}
        for i in range(12)
    ])
    df_roles.to_csv(tbl_dir / "stagewise_feature_roles.csv", index=False)

    # 6. Flat vs Attack-Only
    df_flat = pd.DataFrame([
        {"condition": "Flat GAN (All Classes)", "class_imbalance_ratio": 120.0, "critic_loss_stability_std": 0.45, "tstr_macro_f1": 0.62, "two_stage_macro_f1": 0.70},
        {"condition": "Attack-Only (Proposed)", "class_imbalance_ratio": 15.0, "critic_loss_stability_std": 0.12, "tstr_macro_f1": 0.79, "two_stage_macro_f1": 0.84},
    ])
    df_flat.to_csv(tbl_dir / "flat_vs_attack_only.csv", index=False)

    return {"tbl_dir": tbl_dir, "fig_dir": fig_dir}


def test_make_all_figures_dual_format(mock_tables):
    tbl_dir = mock_tables["tbl_dir"]
    fig_dir = mock_tables["fig_dir"]

    # 1. Architecture diagram
    arch_res = draw_framework_architecture(fig_dir)
    assert arch_res[0].exists() and arch_res[1].exists()
    assert arch_res[0].suffix == ".png" and arch_res[1].suffix == ".pdf"

    # 2. Class distribution
    cd_res = plot_class_distribution_comparison(tbl_dir, fig_dir)
    assert cd_res is not None and cd_res[0].exists() and cd_res[1].exists()

    # 3. WGAN loss
    loss_res = plot_wgan_loss_curves(tbl_dir, fig_dir)
    assert loss_res is not None and loss_res[0].exists() and loss_res[1].exists()

    # 4. Recall benchmark
    rec_res = plot_per_class_recall_benchmark(tbl_dir, fig_dir)
    assert rec_res is not None and rec_res[0].exists() and rec_res[1].exists()

    # 5. SHAP fidelity
    fid_res = plot_shap_fidelity_summary(tbl_dir, fig_dir)
    assert fid_res is not None and fid_res[0].exists() and fid_res[1].exists()

    # 6. Feature roles
    roles_res = plot_stagewise_feature_roles(tbl_dir, fig_dir)
    assert roles_res is not None and roles_res[0].exists() and roles_res[1].exists()

    # 7. Flat vs Attack-Only
    flat_res = plot_flat_vs_attack_only_comparison(tbl_dir, fig_dir)
    assert flat_res is not None and flat_res[0].exists() and flat_res[1].exists()
