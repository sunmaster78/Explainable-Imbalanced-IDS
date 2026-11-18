"""
Unit tests for the resampling module (src/resamplers.py).

Tests:
    - Target calculation logic (compute_oversampling_targets)
    - WGAN-GP oversampling (gan_oversample, Task 2.5)
    - Random oversampling baseline (random_oversample, Task 5.1)
    - SMOTE oversampling baseline (smote_oversample, Task 5.1)
    - Tiny class safety handling
    - Before/after class count table generation
"""

import numpy as np
import pandas as pd
import pytest

from src.resamplers import (
    compute_oversampling_targets,
    gan_oversample,
    random_oversample,
    smote_oversample,
)
from src.utils import set_seed
from src.wgan_gp import Generator

LATENT_DIM = 16
N_CLASSES = 4
N_FEATURES = 20


@pytest.fixture
def dummy_generator():
    """Create a lightweight Generator for testing."""
    set_seed(42)
    return Generator(
        latent_dim=LATENT_DIM,
        n_classes=N_CLASSES,
        output_dim=N_FEATURES,
        hidden_layers=[32, 64],
    )


@pytest.fixture
def sample_attack_data():
    """
    Create synthetic attack data with severe class imbalance:
        DoS:   100 samples
        Probe:  40 samples
        R2L:    10 samples
        U2R:     2 samples
    """
    set_seed(42)
    counts = {"DoS": 100, "Probe": 40, "R2L": 10, "U2R": 2}
    total = sum(counts.values())

    X = np.random.uniform(-1.0, 1.0, size=(total, N_FEATURES)).astype(np.float32)

    # Make the last 4 columns simulate a one-hot categorical feature
    for i in range(total):
        active_col = 16 + (i % 4)
        X[i, 16:20] = 0.0
        X[i, active_col] = 1.0

    y_list = []
    for cls, cnt in counts.items():
        y_list.extend([cls] * cnt)

    y = np.array(y_list)
    return X, y


@pytest.fixture
def onehot_groups():
    return {
        "cat_feature": [16, 17, 18, 19],
    }


# ===========================================================================
# Target Calculation Tests
# ===========================================================================

class TestTargetCalculation:
    def test_standard_targets(self):
        y = ["DoS"] * 100 + ["Probe"] * 40 + ["R2L"] * 10 + ["U2R"] * 2
        counts_before, targets, missing = compute_oversampling_targets(y, target_ratio=0.5)

        # Largest is 100 -> target = 50
        assert counts_before == {"DoS": 100, "Probe": 40, "R2L": 10, "U2R": 2}
        assert targets["DoS"] == 100  # already >= 50, not reduced
        assert targets["Probe"] == 50
        assert targets["R2L"] == 50
        assert targets["U2R"] == 50

        assert missing["DoS"] == 0
        assert missing["Probe"] == 10
        assert missing["R2L"] == 40
        assert missing["U2R"] == 48

    def test_full_parity_target(self):
        y = ["A"] * 80 + ["B"] * 20
        _, targets, missing = compute_oversampling_targets(y, target_ratio=1.0)
        assert targets["A"] == 80
        assert targets["B"] == 80
        assert missing["B"] == 60

    def test_invalid_target_ratio(self):
        with pytest.raises(ValueError, match="target_ratio must be positive"):
            compute_oversampling_targets(["A", "B"], target_ratio=0.0)

    def test_empty_labels(self):
        with pytest.raises(ValueError, match="Cannot compute targets on empty"):
            compute_oversampling_targets([], target_ratio=0.5)


# ===========================================================================
# GAN Oversample Tests (Task 2.5)
# ===========================================================================

class TestGanOversample:
    def test_gan_oversample_string_labels(self, dummy_generator, sample_attack_data, onehot_groups, tmp_path):
        X, y = sample_attack_data
        config = {
            "gan": {"latent_dim": LATENT_DIM, "batch_size": 32},
            "resampling": {"target_ratio": 0.5},
            "attack_classes": ["DoS", "Probe", "R2L", "U2R"],
            "output_dirs": {"tables": str(tmp_path / "tables")},
        }

        X_aug, y_aug, is_synthetic = gan_oversample(
            X_train_attack=X,
            y_train_attack=y,
            generator=dummy_generator,
            config=config,
            onehot_groups=onehot_groups,
            table_name="test_gan_oversample_counts",
        )

        # Largest was 100 -> target is 50 for Probe, R2L, U2R
        # Expected counts after: DoS=100, Probe=50, R2L=50, U2R=50 -> Total = 250
        assert len(y_aug) == 250
        assert len(X_aug) == 250
        assert len(is_synthetic) == 250

        # Real samples are preserved at the beginning
        assert not np.any(is_synthetic[: len(X)])
        # Synthetic samples are at the end
        assert np.all(is_synthetic[len(X) :])

        # Verify class counts
        unique, counts = np.unique(y_aug, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        assert counts_dict["DoS"] == 100
        assert counts_dict["Probe"] == 50
        assert counts_dict["R2L"] == 50
        assert counts_dict["U2R"] == 50

        # Verify one-hot validity of synthetic rows
        syn_rows = X_aug[is_synthetic]
        cat_slice = syn_rows[:, 16:20]
        row_sums = cat_slice.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
        assert set(np.unique(cat_slice)).issubset({0.0, 1.0})

        # Verify table was saved
        saved_csv = tmp_path / "tables" / "test_gan_oversample_counts.csv"
        assert saved_csv.exists()
        saved_df = pd.read_csv(saved_csv)
        assert "count_before" in saved_df.columns
        assert "count_after" in saved_df.columns

    def test_gan_oversample_integer_labels(self, dummy_generator, tmp_path):
        set_seed(42)
        X = np.random.randn(50, N_FEATURES).astype(np.float32)
        # Class 0: 30, Class 1: 15, Class 2: 5
        y = np.array([0] * 30 + [1] * 15 + [2] * 5)

        config = {
            "gan": {"latent_dim": LATENT_DIM, "batch_size": 16},
            "target_ratio": 0.5,
            "output_dirs": {"tables": str(tmp_path / "tables")},
        }

        X_aug, y_aug, is_syn = gan_oversample(
            X_train_attack=X,
            y_train_attack=y,
            generator=dummy_generator,
            config=config,
            onehot_groups={},
        )

        # Largest is 30 -> target is 15
        # Class 0: 30, Class 1: 15 (no change), Class 2: 15 (+10 synthetic)
        assert len(y_aug) == 30 + 15 + 15
        assert is_syn.sum() == 10
        assert (y_aug[is_syn] == 2).all()

    def test_gan_oversample_no_synthetic_needed(self, dummy_generator, tmp_path):
        """When all classes are already >= target, no synthetic samples are added."""
        X = np.random.randn(40, N_FEATURES).astype(np.float32)
        y = np.array(["A"] * 20 + ["B"] * 20)

        config = {
            "gan": {"latent_dim": LATENT_DIM, "batch_size": 16},
            "target_ratio": 0.5,
            "output_dirs": {"tables": str(tmp_path / "tables")},
        }

        X_aug, y_aug, is_syn = gan_oversample(
            X_train_attack=X,
            y_train_attack=y,
            generator=dummy_generator,
            config=config,
            onehot_groups={},
        )

        assert len(X_aug) == 40
        assert not np.any(is_syn)


# ===========================================================================
# Baseline Resampler Tests (Task 5.1)
# ===========================================================================

class TestBaselineResamplers:
    def test_random_oversample(self, sample_attack_data, tmp_path):
        X, y = sample_attack_data
        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        X_aug, y_aug, is_syn = random_oversample(
            X=X,
            y=y,
            target_ratio=0.5,
            config=config,
            seed=42,
        )

        # Expected counts: DoS: 100, Probe: 50, R2L: 50, U2R: 50
        assert len(y_aug) == 250
        assert is_syn.sum() == 250 - len(X)
        assert (tmp_path / "tables" / "ros_oversample_class_counts.csv").exists()

    def test_smote_oversample_standard(self, tmp_path):
        set_seed(42)
        # Sufficient samples for default k_neighbors=5
        X = np.random.randn(80, 5).astype(np.float32)
        y = np.array(["A"] * 60 + ["B"] * 20)

        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        X_aug, y_aug, is_syn = smote_oversample(
            X=X,
            y=y,
            target_ratio=0.5,
            k_neighbors=5,
            config=config,
            seed=42,
        )

        # Target = 30 for B
        unique, counts = np.unique(y_aug, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        assert counts_dict["A"] == 60
        assert counts_dict["B"] == 30
        assert is_syn.sum() == 10

    def test_smote_oversample_tiny_class_reduction(self, tmp_path):
        """When minority class has fewer samples than k_neighbors + 1, k is reduced."""
        set_seed(42)
        # Class B has only 3 samples (<= k_neighbors=5)
        X = np.random.randn(33, 4).astype(np.float32)
        y = np.array(["A"] * 30 + ["B"] * 3)

        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        # Should not raise; reduces k_neighbors to 2
        X_aug, y_aug, is_syn = smote_oversample(
            X=X,
            y=y,
            target_ratio=0.5,
            k_neighbors=5,
            config=config,
            seed=42,
        )

        # Target = 15 for B
        unique, counts = np.unique(y_aug, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        assert counts_dict["A"] == 30
        assert counts_dict["B"] == 15
        assert is_syn.sum() == 12

    def test_smote_oversample_single_sample_class(self, tmp_path):
        """When minority class has only 1 sample, duplicates first, then applies SMOTE."""
        set_seed(42)
        X = np.random.randn(21, 4).astype(np.float32)
        y = np.array(["A"] * 20 + ["B"] * 1)

        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        X_aug, y_aug, is_syn = smote_oversample(
            X=X,
            y=y,
            target_ratio=0.5,
            k_neighbors=5,
            config=config,
            seed=42,
        )

        # Target = 10 for B
        unique, counts = np.unique(y_aug, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        assert counts_dict["A"] == 20
        assert counts_dict["B"] == 10
        assert is_syn.sum() == 9
