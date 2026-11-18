"""
Unit and Integration Tests for Phase 8: SHAP Explainability Suite.

Validates:
1. get_mean_abs_shap: Unified tree ensemble SHAP importance extraction.
2. compute_attribution_fidelity: Spearman, Jaccard, Cosine fidelity scores.
3. bootstrap_fidelity_ci: 95% confidence intervals from bootstrap resampling.
4. run_fidelity: Novelty Claim 1 (WGAN-GP vs SMOTE vs Real attribution alignment).
5. run_stagewise: Novelty Claim 2 (Stage 1 beeswarm & Stage 2 bar plots, feature roles, overlap).
6. explain_sample: Local explanations and SHAP waterfall plots for canonical instances.
7. compare_flat_vs_attack_only: Novelty Claim 3 (Flat GAN vs Attack-Only GAN comparison).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.baselines import compare_flat_vs_attack_only
from src.ensemble import build_voting_ensemble
from src.pipeline import TwoStageIDS
from src.shap_fidelity import (
    bootstrap_fidelity_ci,
    compute_attribution_fidelity,
    get_mean_abs_shap,
    get_sample_shap_matrix,
    run_fidelity,
)
from src.shap_stagewise import explain_sample, run_local_explanations_suite, run_stagewise
from src.wgan_gp import Generator, train_wgan_gp


@pytest.fixture
def mock_shap_data(tmp_path: Path) -> Dict[str, Any]:
    """Generate small synthetic dataset fixture for fast SHAP testing."""
    rng = np.random.RandomState(42)
    n_train = 60
    n_val = 20
    n_test = 30
    n_features = 8

    # 4 classes: Normal (0), DoS (1), Probe (2), R2L (3)
    classes = ["Normal", "DoS", "Probe", "R2L"]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Distribute classes: Normal is majority, R2L is minority
    y_train_cat = rng.choice(classes, size=n_train, p=[0.5, 0.3, 0.15, 0.05])
    # Ensure at least 2 R2L samples in train
    y_train_cat[0] = "R2L"
    y_train_cat[1] = "R2L"
    y_train_cat[2] = "Probe"

    y_val_cat = rng.choice(classes, size=n_val, p=[0.5, 0.3, 0.15, 0.05])
    y_val_cat[0] = "R2L"

    y_test_cat = rng.choice(classes, size=n_test, p=[0.5, 0.3, 0.15, 0.05])
    y_test_cat[0] = "R2L"

    X_train = rng.randn(n_train, n_features).astype(np.float32)
    X_val = rng.randn(n_val, n_features).astype(np.float32)
    X_test = rng.randn(n_test, n_features).astype(np.float32)

    y_train_bin = (y_train_cat != "Normal").astype(int)
    y_val_bin = (y_val_cat != "Normal").astype(int)
    y_test_bin = (y_test_cat != "Normal").astype(int)

    return {
        "X_train": X_train,
        "y_train_cat": y_train_cat,
        "y_train_bin": y_train_bin,
        "X_val": X_val,
        "y_val_cat": y_val_cat,
        "y_val_bin": y_val_bin,
        "X_test": X_test,
        "y_test_cat": y_test_cat,
        "y_test_bin": y_test_bin,
        "feature_names": feature_names,
        "onehot_column_indices": {},
        "tmp_path": tmp_path,
    }


@pytest.fixture
def mock_shap_config(tmp_path: Path) -> Dict[str, Any]:
    """Test configuration dictionary with temporary output paths and tiny epoch count."""
    return {
        "dataset": {"name": "Mock-NSL-KDD", "target_column": "category"},
        "experiment": {"default_seed": 42, "validation_size": 0.2},
        "output_dirs": {
            "tables": str(tmp_path / "tables"),
            "figures": str(tmp_path / "figures"),
            "models": str(tmp_path / "models"),
            "logs": str(tmp_path / "logs"),
        },
        "gan": {
            "latent_dim": 16,
            "gen_hidden_layers": [32],
            "critic_hidden_layers": [32],
            "batch_size": 16,
            "lr": 0.0002,
            "betas": [0.5, 0.9],
            "n_critic": 2,
            "lambda_gp": 10.0,
            "epochs": 2,
        },
        "ensemble": {"n_estimators": 5, "voting": "soft"},
        "pipeline": {"stage1_threshold": 0.5, "target_recall": 0.85},
        "resampling": {"target_ratio": 0.5},
        "classes": ["Normal", "DoS", "Probe", "R2L"],
        "attack_classes": ["DoS", "Probe", "R2L"],
        "normal_class": "Normal",
        "use_gpu": False,
    }


# ===========================================================================
# Task 8.1 Tests: SHAP Helper
# ===========================================================================

def test_get_mean_abs_shap_ensemble(mock_shap_data):
    data = mock_shap_data
    X = data["X_train"]
    y = data["y_train_bin"]

    rf = RandomForestClassifier(n_estimators=3, random_state=42)
    xgb = XGBClassifier(n_estimators=3, random_state=42, eval_metric="logloss")
    lgbm = LGBMClassifier(n_estimators=3, random_state=42, verbose=-1)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft",
    )
    ensemble.fit(X, y)

    ser = get_mean_abs_shap(
        models=ensemble,
        X=X,
        class_index=1,
        max_samples=20,
        feature_names=data["feature_names"],
        seed=42,
    )

    assert isinstance(ser, pd.Series)
    assert len(ser) == len(data["feature_names"])
    assert list(ser.index) == data["feature_names"]
    assert np.all(ser.values >= 0.0)
    assert np.isclose(ser.sum(), 1.0, atol=1e-4)


def test_get_sample_shap_matrix_shape(mock_shap_data):
    data = mock_shap_data
    X = data["X_train"][:20]
    y = data["y_train_bin"][:20]

    rf = RandomForestClassifier(n_estimators=3, random_state=42).fit(X, y)
    M = get_sample_shap_matrix(rf, X, class_index=1, max_samples=15, seed=42)

    assert isinstance(M, np.ndarray)
    assert M.shape == (15, X.shape[1])
    assert np.all(M >= 0.0)


# ===========================================================================
# Task 8.2 Tests: Claim 1 Fidelity Metrics and CI
# ===========================================================================

def test_compute_attribution_fidelity_metrics():
    v_real = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    v_syn_ident = np.array([0.4, 0.3, 0.2, 0.1, 0.0])

    f_ident = compute_attribution_fidelity(v_real, v_syn_ident, top_k=3)
    assert np.isclose(f_ident["spearman"], 1.0, atol=1e-4)
    assert np.isclose(f_ident["jaccard_top10"], 1.0, atol=1e-4)
    assert np.isclose(f_ident["cosine"], 1.0, atol=1e-4)

    # Inverted ranking
    v_syn_inv = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    f_inv = compute_attribution_fidelity(v_real, v_syn_inv, top_k=2)
    assert f_inv["spearman"] < 0.0
    assert f_inv["jaccard_top10"] == 0.0  # Top 2 sets are disjoint


def test_bootstrap_fidelity_ci():
    rng = np.random.RandomState(42)
    M_real = rng.rand(20, 5).astype(np.float32)
    M_syn = rng.rand(20, 5).astype(np.float32)

    ci = bootstrap_fidelity_ci(M_real, M_syn, n_bootstrap=50, top_k=3, seed=42)

    assert "spearman_ci" in ci
    assert "jaccard_ci" in ci
    assert "cosine_ci" in ci

    assert ci["spearman_ci"][0] <= ci["spearman_ci"][1]
    assert ci["cosine_ci"][0] <= ci["cosine_ci"][1]


def test_run_fidelity_end_to_end(mock_shap_data, mock_shap_config):
    data = mock_shap_data
    cfg = mock_shap_config

    # Build a tiny trained generator for testing
    attack_mask = data["y_train_bin"] == 1
    X_att = data["X_train"][attack_mask]
    y_att = data["y_train_cat"][attack_mask]

    attack_classes = sorted(list(np.unique(y_att)))
    y_onehot = np.zeros((len(y_att), len(attack_classes)), dtype=np.float32)
    for i, c in enumerate(y_att):
        y_onehot[i, attack_classes.index(c)] = 1.0

    train_res = train_wgan_gp(
        X_attack=X_att,
        y_attack_onehot=y_onehot,
        config=cfg,
        device="cpu",
        save_plots=False,
    )
    generator = train_res["generator"]

    df_fidelity, importances = run_fidelity(
        data=data,
        generator=generator,
        config=cfg,
        seed=42,
        n_bootstrap=10,  # Small bootstrap for test speed
        max_eval_samples=25,
    )

    assert isinstance(df_fidelity, pd.DataFrame)
    assert not df_fidelity.empty
    expected_cols = {"class", "method", "spearman", "jaccard_top10", "cosine", "spearman_ci_low", "spearman_ci_high"}
    assert expected_cols.issubset(df_fidelity.columns)

    # Verify saved artifacts
    fig_dir = Path(cfg["output_dirs"]["figures"])
    tbl_dir = Path(cfg["output_dirs"]["tables"])
    assert (tbl_dir / "shap_fidelity.csv").exists()
    assert any(fig_dir.glob("shap_fidelity_*.png"))


# ===========================================================================
# Task 8.3 & 8.4 Tests: Stage-Wise & Local Explanations
# ===========================================================================

def test_run_stagewise_and_local_explanations(mock_shap_data, mock_shap_config):
    data = mock_shap_data
    cfg = mock_shap_config

    pipeline = TwoStageIDS(config=cfg, seed=42)
    pipeline.fit_stage1(data["X_train"], data["y_train_bin"])
    pipeline.tune_stage1_threshold(data["X_val"], data["y_val_bin"])

    # Fit stage 2 on attacks
    attack_mask = data["y_train_bin"] == 1
    pipeline.fit_stage2(
        data["X_train"][attack_mask],
        data["y_train_cat"][attack_mask],
        use_gan=False,
    )

    # 1. Run Stage-Wise SHAP Audit
    res_stagewise = run_stagewise(
        pipeline=pipeline,
        data=data,
        config=cfg,
        max_samples=25,
        seed=42,
    )

    assert "top_features" in res_stagewise
    assert "feature_roles" in res_stagewise
    assert "stage_importance_overlap" in res_stagewise

    tbl_dir = Path(cfg["output_dirs"]["tables"])
    fig_dir = Path(cfg["output_dirs"]["figures"])
    assert (tbl_dir / "stagewise_top_features.csv").exists()
    assert (tbl_dir / "stagewise_feature_roles.csv").exists()
    assert (tbl_dir / "stage_importance_overlap.csv").exists()
    assert (fig_dir / "shap_stage1_beeswarm.png").exists()

    # 2. Local Sample Explanation
    sample = data["X_test"][0]
    res_sample = explain_sample(
        pipeline=pipeline,
        x=sample,
        config=cfg,
        sample_id="test_sample_0",
        feature_names=data["feature_names"],
        true_label=data["y_test_cat"][0],
    )
    assert res_sample["p_attack"] >= 0.0
    assert len(res_sample["stage1_top5"]) <= 5
    assert Path(res_sample["stage1_waterfall"]).exists()

    # 3. Local explanations suite
    df_local = run_local_explanations_suite(pipeline=pipeline, data=data, config=cfg, seed=42)
    assert isinstance(df_local, pd.DataFrame)
    assert (tbl_dir / "local_explanations_top5.csv").exists()


# ===========================================================================
# Task 8.5 Tests: Claim 3 (Flat GAN vs Attack-Only GAN)
# ===========================================================================

def test_compare_flat_vs_attack_only(mock_shap_data, mock_shap_config):
    data = mock_shap_data
    cfg = mock_shap_config

    df_comp = compare_flat_vs_attack_only(data=data, config=cfg, seed=42)

    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_comp) == 2
    conditions = list(df_comp["condition"])
    assert "Flat GAN (All Classes)" in conditions
    assert "Attack-Only (Proposed)" in conditions

    for col in [
        "class_imbalance_ratio",
        "critic_loss_stability_std",
        "tstr_macro_f1",
        "nn_memorization_rate_pct",
        "two_stage_macro_f1",
        "minority_recall",
        "gan_train_time_sec",
    ]:
        assert col in df_comp.columns
        assert not df_comp[col].isna().any()

    # Claim 3 verification: Attack-Only GAN sees lower imbalance ratio than Flat GAN
    flat_ratio = float(df_comp.loc[df_comp["condition"] == "Flat GAN (All Classes)", "class_imbalance_ratio"].iloc[0])
    att_ratio = float(df_comp.loc[df_comp["condition"] == "Attack-Only (Proposed)", "class_imbalance_ratio"].iloc[0])
    assert flat_ratio >= att_ratio, f"Flat ratio ({flat_ratio}) should be >= attack-only ratio ({att_ratio})"
