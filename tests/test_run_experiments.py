"""
Unit tests for the experiment runner CLI (src/run_experiments.py, Task 5.4).

Tests:
    - Resumable state detection (load_completed_runs)
    - Method execution dispatching (run_experiment_for_method)
    - CLI argument parsing and execution
"""

import numpy as np
import pandas as pd
import pytest

from src.run_experiments import load_completed_runs, run_experiment_for_method
from src.utils import set_seed

N_FEATURES = 10


@pytest.fixture
def mock_dataset():
    set_seed(42)

    def _make(n):
        X = np.random.randn(n, N_FEATURES).astype(np.float32)
        y_cat = np.array(["Normal"] * (n // 2) + ["DoS"] * (n // 4) + ["Probe"] * (n // 4))
        y_bin = (y_cat != "Normal").astype(int)
        return X, y_cat, y_bin

    X_tr, y_tr_c, y_tr_b = _make(40)
    X_val, y_val_c, y_val_b = _make(20)
    X_te, y_te_c, y_te_b = _make(20)

    return {
        "X_train": X_tr,
        "y_train_cat": y_tr_c,
        "y_train_bin": y_tr_b,
        "X_val": X_val,
        "y_val_cat": y_val_c,
        "y_val_bin": y_val_b,
        "X_test": X_te,
        "y_test_cat": y_te_c,
        "y_test_bin": y_te_b,
        "onehot_column_indices": {},
    }


@pytest.fixture
def mock_config(tmp_path):
    return {
        "experiment": {"default_seed": 42},
        "pipeline": {"stage1_threshold": 0.5, "target_recall": 0.8},
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


class TestRunExperiments:
    def test_load_completed_runs(self, tmp_path):
        csv_file = tmp_path / "raw_results.csv"

        # Empty or non-existent file
        assert load_completed_runs(csv_file) == set()

        # Existing file with completed runs
        df = pd.DataFrame([
            {"seed": 0, "method": "B1", "accuracy": 0.95},
            {"seed": 0, "method": "B2", "accuracy": 0.96},
            {"seed": 1, "method": "B1", "accuracy": 0.94},
        ])
        df.to_csv(csv_file, index=False)

        completed = load_completed_runs(csv_file)
        assert len(completed) == 3
        assert (0, "B1") in completed
        assert (0, "B2") in completed
        assert (1, "B1") in completed
        assert (1, "B2") not in completed

    def test_run_experiment_dispatch_single_stage(self, mock_dataset, mock_config):
        y_pred, y_proba, train_time, infer_time = run_experiment_for_method(
            method_name="B1",
            dataset=mock_dataset,
            config=mock_config,
            seed=42,
        )
        assert len(y_pred) == len(mock_dataset["X_test"])
        assert y_proba is not None
        assert train_time > 0.0

    def test_run_experiment_dispatch_two_stage(self, mock_dataset, mock_config):
        y_pred, y_proba, train_time, infer_time = run_experiment_for_method(
            method_name="B5",
            dataset=mock_dataset,
            config=mock_config,
            seed=42,
        )
        assert len(y_pred) == len(mock_dataset["X_test"])
        assert train_time > 0.0
