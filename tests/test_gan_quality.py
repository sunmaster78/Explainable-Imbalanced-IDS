"""
Unit tests for GAN quality evaluation module (src/gan_quality.py, Task 2.6).

Tests:
    - TSTR score calculation (macro-F1 and per-class recall)
    - Nearest-neighbor distance and memorization detection
    - Kolmogorov-Smirnov test per feature
    - 2D PCA scatter plotting
    - Full evaluate_gan_quality orchestrator and U2R memorization check
"""

import numpy as np
import pandas as pd
import pytest

from src.gan_quality import (
    evaluate_gan_quality,
    ks_test_per_feature,
    nearest_neighbor_distance,
    pca_plot,
    tstr_score,
)
from src.utils import set_seed

N_FEATURES = 10


@pytest.fixture
def synthetic_and_real_data():
    """Create controlled synthetic and real data for testing."""
    set_seed(42)
    n_real_train = 50
    n_syn = 60
    n_real_test = 40

    # Real train: mean 0, std 1
    X_real_train = np.random.randn(n_real_train, N_FEATURES).astype(np.float32)
    # Synthetic: close to real train (mean 0.1, std 1.0)
    X_syn = (np.random.randn(n_syn, N_FEATURES) + 0.1).astype(np.float32)
    # Real test: mean 0, std 1
    X_real_test = np.random.randn(n_real_test, N_FEATURES).astype(np.float32)

    # Binary labels for simple TSTR test: class 0 and 1
    y_syn = np.array([0] * (n_syn // 2) + [1] * (n_syn // 2))
    y_real_test = np.array([0] * (n_real_test // 2) + [1] * (n_real_test // 2))

    return X_real_train, X_syn, X_real_test, y_syn, y_real_test


class TestTstrScore:
    def test_tstr_basic(self, synthetic_and_real_data, tmp_path):
        _, X_syn, X_real_test, y_syn, y_real_test = synthetic_and_real_data
        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        res = tstr_score(
            X_syn=X_syn,
            y_syn=y_syn,
            X_real_test=X_real_test,
            y_real_test=y_real_test,
            config=config,
            seed=42,
            n_estimators=20,
            table_name="tstr_test",
        )

        assert "macro_f1" in res
        assert "per_class_recall" in res
        assert 0.0 <= res["macro_f1"] <= 1.0
        assert 0 in res["per_class_recall"]
        assert 1 in res["per_class_recall"]
        assert (tmp_path / "tables" / "tstr_test.csv").exists()

    def test_tstr_empty_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            tstr_score(
                X_syn=np.empty((0, 5)),
                y_syn=np.array([]),
                X_real_test=np.ones((10, 5)),
                y_real_test=np.ones(10),
            )


class TestNearestNeighborDistance:
    def test_distinct_samples_no_memorization(self, synthetic_and_real_data):
        X_real_train, X_syn, _, _, _ = synthetic_and_real_data

        res = nearest_neighbor_distance(X_syn, X_real_train, threshold=1e-3)
        assert res["syn_mean_dist"] > 0.1
        assert res["syn_min_dist"] > 0.01
        assert res["syn_memorized_pct"] == 0.0
        assert res["real_mean_dist"] > 0.0

    def test_exact_memorization_detection(self):
        # Create synthetic data identical to real train data
        X_real = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        X_syn = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        res = nearest_neighbor_distance(X_syn, X_real, threshold=1e-3, class_name="U2R")
        assert res["syn_min_dist"] == pytest.approx(0.0, abs=1e-6)
        assert res["syn_memorized_pct"] == 100.0


class TestKsTestPerFeature:
    def test_identical_distributions(self):
        set_seed(42)
        # Sample both from standard normal
        X_real = np.random.randn(500, 5).astype(np.float32)
        X_syn = np.random.randn(500, 5).astype(np.float32)

        res = ks_test_per_feature(X_syn, X_real, alpha=0.01)
        assert res["total_features"] == 5
        # KS statistic should be relatively small
        assert res["mean_ks_stat"] < 0.15
        assert len(res["feature_stats"]) == 5

    def test_divergent_distributions(self):
        set_seed(42)
        # Real is standard normal, synthetic is shifted heavily
        X_real = np.random.randn(200, 3).astype(np.float32)
        X_syn = (np.random.randn(200, 3) + 10.0).astype(np.float32)

        res = ks_test_per_feature(X_syn, X_real, alpha=0.05)
        # High KS statistic and all features rejected
        assert res["mean_ks_stat"] > 0.9
        assert res["p_less_alpha_count"] == 3

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            ks_test_per_feature(np.zeros((10, 3)), np.zeros((10, 4)))


class TestPcaPlot:
    def test_pca_plot_generation(self, synthetic_and_real_data, tmp_path):
        X_real_train, X_syn, _, _, _ = synthetic_and_real_data
        config = {"output_dirs": {"figures": str(tmp_path / "figures")}}

        fig, saved_path = pca_plot(
            X_real=X_real_train,
            X_syn=X_syn,
            class_name="Probe",
            config=config,
            save_name="test_pca_probe",
        )

        assert fig is not None
        assert saved_path is not None
        assert (tmp_path / "figures" / "test_pca_probe.png").exists()


class TestEvaluateGanQuality:
    def test_full_pipeline(self, tmp_path):
        set_seed(42)
        # Prepare multi-class attack dictionaries
        classes = ["DoS", "Probe", "U2R"]
        X_train_dict = {}
        X_syn_dict = {}

        for cls in classes:
            X_train_dict[cls] = np.random.randn(30, N_FEATURES).astype(np.float32)
            X_syn_dict[cls] = (np.random.randn(25, N_FEATURES) + 0.05).astype(np.float32)

        X_real_test = np.random.randn(60, N_FEATURES).astype(np.float32)
        y_real_test = np.array(["DoS"] * 20 + ["Probe"] * 20 + ["U2R"] * 20)

        config = {
            "output_dirs": {
                "tables": str(tmp_path / "tables"),
                "figures": str(tmp_path / "figures"),
            }
        }

        summary_df = evaluate_gan_quality(
            X_train_dict=X_train_dict,
            X_syn_dict=X_syn_dict,
            X_real_test=X_real_test,
            y_real_test=y_real_test,
            config=config,
            table_name="gan_quality_test",
            make_plots=True,
        )

        assert len(summary_df) == 3
        assert "tstr_macro_f1" in summary_df.columns
        assert "syn_mean_nn_dist" in summary_df.columns
        assert "syn_memorized_pct" in summary_df.columns
        assert "ks_mean_stat" in summary_df.columns

        # Verify summary CSV table saved
        assert (tmp_path / "tables" / "gan_quality_test.csv").exists()
        # Verify PCA plot saved for U2R
        assert (tmp_path / "figures" / "pca_u2r.png").exists()
