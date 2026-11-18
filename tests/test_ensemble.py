"""
Unit tests for the ensemble classifier module (src/ensemble.py, Phase 3).

Tests:
    - Base models configuration for binary and multiclass tasks
    - Soft-voting ensemble (build_voting_ensemble, Task 3.1)
    - Stacking ensemble (build_stacking_ensemble, Task 3.2)
    - Factory function (get_ensemble, Task 3.2)
    - Validation-only hyperparameter tuning (tune_ensemble, Task 3.3)
"""

import numpy as np
import pytest
import yaml

from src.ensemble import (
    build_base_models,
    build_stacking_ensemble,
    build_voting_ensemble,
    get_ensemble,
    tune_ensemble,
)
from src.utils import set_seed

N_FEATURES = 8


@pytest.fixture
def dummy_binary_data():
    """Create a small binary classification dataset."""
    set_seed(42)
    n = 60
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


@pytest.fixture
def dummy_multiclass_data():
    """Create a small 4-class classification dataset."""
    set_seed(42)
    n = 80
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
    return X, y


@pytest.fixture
def base_config():
    return {
        "experiment": {"default_seed": 42},
        "ensemble": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "weights": [1.0, 1.0, 1.0],
        },
    }


# ===========================================================================
# Base Model Builder Tests
# ===========================================================================

class TestBaseModels:
    def test_binary_objectives(self, base_config):
        models = dict(build_base_models(base_config, n_classes=2))
        assert models["xgb"].objective == "binary:logistic"
        assert models["lgbm"].objective == "binary"

    def test_multiclass_objectives(self, base_config):
        models = dict(build_base_models(base_config, n_classes=4))
        assert models["xgb"].objective == "multi:softprob"
        assert models["lgbm"].objective == "multiclass"


# ===========================================================================
# Voting Ensemble Tests (Task 3.1)
# ===========================================================================

class TestVotingEnsemble:
    def test_voting_binary_predict_proba(self, dummy_binary_data, base_config):
        X, y = dummy_binary_data
        ensemble = build_voting_ensemble(base_config, n_classes=2)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

        preds = ensemble.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_voting_multiclass_predict_proba(self, dummy_multiclass_data, base_config):
        X, y = dummy_multiclass_data
        ensemble = build_voting_ensemble(base_config, n_classes=4)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 4)
        assert np.allclose(proba.sum(axis=1), 1.0)

        preds = ensemble.predict(X)
        assert len(preds) == len(X)

    def test_voting_custom_weights(self, dummy_binary_data):
        X, y = dummy_binary_data
        config = {
            "experiment": {"default_seed": 42},
            "ensemble": {
                "n_estimators": 5,
                "weights": [2.0, 1.0, 0.5],
            },
        }
        ensemble = build_voting_ensemble(config, n_classes=2)
        ensemble.fit(X, y)
        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ===========================================================================
# Stacking Ensemble & Factory Tests (Task 3.2)
# ===========================================================================

class TestStackingEnsemble:
    def test_stacking_binary_predict_proba(self, dummy_binary_data, base_config):
        X, y = dummy_binary_data
        ensemble = build_stacking_ensemble(base_config, n_classes=2, cv=2)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_get_ensemble_factory(self, base_config):
        voting = get_ensemble("voting", base_config, n_classes=2)
        assert voting.__class__.__name__ == "VotingClassifier"

        stacking = get_ensemble("stacking", base_config, n_classes=2)
        assert stacking.__class__.__name__ == "StackingClassifier"

        with pytest.raises(ValueError, match="Unknown ensemble kind"):
            get_ensemble("invalid_kind", base_config, n_classes=2)


# ===========================================================================
# Validation Hyperparameter Tuning Tests (Task 3.3)
# ===========================================================================

class TestTuneEnsemble:
    def test_tune_ensemble_validation_only(self, tmp_path):
        set_seed(42)
        n_train = 50
        n_val = 30
        X_train = np.random.randn(n_train, 6).astype(np.float32)
        y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

        X_val = np.random.randn(n_val, 6).astype(np.float32)
        y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2))

        config = {
            "experiment": {"default_seed": 42},
            "ensemble": {"tune_n_iter": 2},
            "output_dirs": {"models": str(tmp_path / "models")},
        }

        best_params = tune_ensemble(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            n_iter=2,
            save_name="test_best_params.yaml",
        )

        assert "rf" in best_params
        assert "xgb" in best_params
        assert "lgbm" in best_params

        # Verify YAML file was created and is valid
        yaml_file = tmp_path / "models" / "test_best_params.yaml"
        assert yaml_file.exists()
        with open(yaml_file, "r") as f:
            loaded_params = yaml.safe_load(f)
        assert loaded_params == best_params
