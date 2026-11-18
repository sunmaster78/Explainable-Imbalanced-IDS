"""
Unit and integration tests for TwoStageIDS pipeline (src/pipeline.py, Phase 4).

Tests:
    - Stage 1 binary fitting and P(attack) prediction (Task 4.1)
    - Threshold tuning on validation data (Task 4.2)
    - Stage 2 multiclass fitting with and without GAN oversampling (Task 4.3)
    - Prevention of Normal leakage into Stage 2 training data (Task 4.3)
    - End-to-end two-stage prediction (Task 4.4)
    - Error propagation evaluation and R2L/U2R loss audit (Task 4.4)
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import TwoStageIDS
from src.utils import set_seed
from src.wgan_gp import Generator

N_FEATURES = 10
LATENT_DIM = 16
N_ATTACK_CLASSES = 3  # e.g. DoS, Probe, U2R


@pytest.fixture
def synthetic_generator():
    """Lightweight Generator module for Stage 2 testing."""
    set_seed(42)
    return Generator(
        latent_dim=LATENT_DIM,
        n_classes=N_ATTACK_CLASSES,
        output_dim=N_FEATURES,
        hidden_layers=[32, 32],
    )


@pytest.fixture
def dummy_pipeline_data():
    """
    Synthesize complete multi-stage dataset:
        - Train (Normal + 3 attack classes)
        - Validation (Normal + 3 attack classes)
        - Test (Normal + 3 attack classes)
    """
    set_seed(42)

    def _make_split(n_samples):
        X = np.random.randn(n_samples, N_FEATURES).astype(np.float32)
        # Class distribution: 50% Normal, 30% DoS, 15% Probe, 5% U2R
        y_cat = []
        y_bin = []
        for i in range(n_samples):
            mod = i % 20
            if mod < 10:
                y_cat.append("Normal")
                y_bin.append(0)
            elif mod < 16:
                y_cat.append("DoS")
                y_bin.append(1)
            elif mod < 19:
                y_cat.append("Probe")
                y_bin.append(1)
            else:
                y_cat.append("U2R")
                y_bin.append(1)

        return X, np.array(y_cat), np.array(y_bin)

    X_train, y_train_cat, y_train_bin = _make_split(100)
    X_val, y_val_cat, y_val_bin = _make_split(60)
    X_test, y_test_cat, y_test_bin = _make_split(80)

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
    }


@pytest.fixture
def pipeline_config(tmp_path):
    return {
        "experiment": {"default_seed": 42},
        "pipeline": {
            "stage1_threshold": 0.5,
            "target_recall": 0.90,
        },
        "ensemble": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "weights": [1.0, 1.0, 1.0],
        },
        "gan": {
            "latent_dim": LATENT_DIM,
            "batch_size": 32,
        },
        "resampling": {
            "target_ratio": 0.5,
        },
        "attack_classes": ["DoS", "Probe", "U2R"],
        "output_dirs": {
            "tables": str(tmp_path / "tables"),
            "figures": str(tmp_path / "figures"),
            "models": str(tmp_path / "models"),
            "logs": str(tmp_path / "logs"),
        },
    }


# ===========================================================================
# Stage 1 Tests (Task 4.1 & 4.2)
# ===========================================================================

class TestStage1:
    def test_fit_and_predict_proba(self, dummy_pipeline_data, pipeline_config):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config, ensemble_kind="voting")

        pipeline.fit_stage1(data["X_train"], data["y_train_bin"])
        p_attack = pipeline.predict_proba_stage1(data["X_test"])

        assert len(p_attack) == len(data["X_test"])
        assert ((p_attack >= 0.0) & (p_attack <= 1.0)).all()

    def test_tune_threshold(self, dummy_pipeline_data, pipeline_config, tmp_path):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config, ensemble_kind="voting")
        pipeline.fit_stage1(data["X_train"], data["y_train_bin"])

        best_th = pipeline.tune_stage1_threshold(
            data["X_val"],
            data["y_val_bin"],
            target_recall=0.85,
        )

        assert 0.05 <= best_th <= 0.95
        assert pipeline.stage1_threshold == best_th
        assert (tmp_path / "tables" / "stage1_threshold_tuning.csv").exists()
        assert (tmp_path / "figures" / "stage1_threshold_tuning.png").exists()


# ===========================================================================
# Stage 2 Tests (Task 4.3)
# ===========================================================================

class TestStage2:
    def test_normal_leakage_rejection(self, dummy_pipeline_data, pipeline_config):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config)

        # Attempting to train Stage 2 with Normal traffic must raise ValueError
        with pytest.raises(ValueError, match="Normal traffic detected"):
            pipeline.fit_stage2(
                data["X_train"],
                data["y_train_cat"],  # contains "Normal"
                use_gan=False,
            )

    def test_fit_stage2_without_gan(self, dummy_pipeline_data, pipeline_config):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config)

        # Filter strictly attack samples
        attack_mask = data["y_train_bin"] == 1
        X_att = data["X_train"][attack_mask]
        y_att = data["y_train_cat"][attack_mask]

        pipeline.fit_stage2(X_att, y_att, use_gan=False)

        assert pipeline.stage2_model is not None
        assert pipeline.label_encoder_stage2 is not None
        assert not np.any(pipeline.is_synthetic_stage2)

    def test_fit_stage2_with_gan(self, dummy_pipeline_data, pipeline_config, synthetic_generator):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config)

        attack_mask = data["y_train_bin"] == 1
        X_att = data["X_train"][attack_mask]
        y_att = data["y_train_cat"][attack_mask]

        pipeline.fit_stage2(
            X_att,
            y_att,
            use_gan=True,
            generator=synthetic_generator,
            onehot_groups={},
        )

        assert pipeline.stage2_model is not None
        assert pipeline.is_synthetic_stage2 is not None
        assert np.any(pipeline.is_synthetic_stage2)


# ===========================================================================
# End-to-End Prediction and Error Propagation (Task 4.4)
# ===========================================================================

class TestFullPipeline:
    def test_end_to_end_predict_and_evaluate(
        self,
        dummy_pipeline_data,
        pipeline_config,
        synthetic_generator,
        tmp_path,
    ):
        data = dummy_pipeline_data
        pipeline = TwoStageIDS(config=pipeline_config, ensemble_kind="voting")

        # 1. Fit Stage 1
        pipeline.fit_stage1(data["X_train"], data["y_train_bin"])

        # 2. Tune threshold
        pipeline.tune_stage1_threshold(data["X_val"], data["y_val_bin"], target_recall=0.85)

        # 3. Fit Stage 2
        attack_mask = data["y_train_bin"] == 1
        X_att = data["X_train"][attack_mask]
        y_att = data["y_train_cat"][attack_mask]
        pipeline.fit_stage2(
            X_att,
            y_att,
            use_gan=True,
            generator=synthetic_generator,
            onehot_groups={},
        )

        # 4. Predict
        preds = pipeline.predict(data["X_test"])
        assert len(preds) == len(data["X_test"])
        assert "Normal" in preds

        # 5. Evaluate Pipeline & Error Propagation
        eval_report = pipeline.evaluate_pipeline(
            data["X_test"],
            data["y_test_cat"],
            data["y_test_bin"],
            table_name="test_error_prop",
        )

        assert "accuracy" in eval_report
        assert "macro_f1" in eval_report
        assert "stage1_recall" in eval_report
        assert "loss_df" in eval_report

        # Verify tables created
        assert (tmp_path / "tables" / "test_error_prop.csv").exists()
        assert (tmp_path / "tables" / "test_error_prop_stage1_loss_by_class.csv").exists()

        # Check U2R loss tracking in loss_df
        loss_df = eval_report["loss_df"]
        assert "U2R" in loss_df["attack_class"].values
