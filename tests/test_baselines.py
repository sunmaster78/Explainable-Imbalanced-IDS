"""
Unit tests for baseline models (src/baselines.py, Tasks 5.2 & 5.3).

Tests:
    - run_single_stage: 'none', 'ros', 'smote'
    - run_two_stage: 'none', 'ros', 'smote'
    - Timing metrics and prediction validation
"""

import numpy as np
import pytest

from src.baselines import run_single_stage, run_two_stage
from src.utils import set_seed

N_FEATURES = 10


@pytest.fixture
def dummy_benchmark_data():
    """Create a minimal synthetic benchmark dataset."""
    set_seed(42)

    def _make_dataset(n_samples):
        X = np.random.randn(n_samples, N_FEATURES).astype(np.float32)
        # 4 classes: Normal, DoS, Probe, U2R
        y_cat = []
        y_bin = []
        for i in range(n_samples):
            mod = i % 10
            if mod < 5:
                y_cat.append("Normal")
                y_bin.append(0)
            elif mod < 8:
                y_cat.append("DoS")
                y_bin.append(1)
            elif mod < 9:
                y_cat.append("Probe")
                y_bin.append(1)
            else:
                y_cat.append("U2R")
                y_bin.append(1)
        return X, np.array(y_cat), np.array(y_bin)

    X_train, y_train_cat, y_train_bin = _make_dataset(80)
    X_val, y_val_cat, y_val_bin = _make_dataset(40)
    X_test, y_test_cat, y_test_bin = _make_dataset(40)

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
        "onehot_column_indices": {},
    }


@pytest.fixture
def benchmark_config(tmp_path):
    return {
        "experiment": {"default_seed": 42},
        "pipeline": {"stage1_threshold": 0.5, "target_recall": 0.85},
        "ensemble": {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
            "weights": [1.0, 1.0, 1.0],
        },
        "resampling": {"target_ratio": 0.5},
        "output_dirs": {
            "tables": str(tmp_path / "tables"),
            "figures": str(tmp_path / "figures"),
            "models": str(tmp_path / "models"),
            "logs": str(tmp_path / "logs"),
        },
    }


class TestSingleStageBaselines:
    @pytest.mark.parametrize("method", ["none", "ros", "smote"])
    def test_single_stage_methods(self, method, dummy_benchmark_data, benchmark_config):
        res = run_single_stage(
            method=method,
            data=dummy_benchmark_data,
            config=benchmark_config,
            seed=42,
        )

        assert len(res["y_pred"]) == len(dummy_benchmark_data["X_test"])
        assert res["y_proba"].shape == (len(dummy_benchmark_data["X_test"]), len(res["classes"]))
        assert np.allclose(res["y_proba"].sum(axis=1), 1.0)
        assert res["training_time"] > 0.0
        assert res["inference_time"] >= 0.0

    def test_invalid_method(self, dummy_benchmark_data, benchmark_config):
        with pytest.raises(ValueError, match="Unknown single-stage method"):
            run_single_stage("invalid_method", dummy_benchmark_data, benchmark_config)


class TestTwoStageBaselines:
    @pytest.mark.parametrize("method", ["none", "ros", "smote"])
    def test_two_stage_methods(self, method, dummy_benchmark_data, benchmark_config):
        res = run_two_stage(
            method=method,
            data=dummy_benchmark_data,
            config=benchmark_config,
            seed=42,
        )

        assert len(res["y_pred"]) == len(dummy_benchmark_data["X_test"])
        assert 0.05 <= res["stage1_threshold"] <= 0.95
        assert res["training_time"] > 0.0
        assert res["inference_time"] >= 0.0
        assert res["pipeline"] is not None
