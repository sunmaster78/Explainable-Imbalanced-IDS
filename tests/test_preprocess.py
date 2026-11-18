"""
Unit tests for preprocessing functions in src/preprocess.py.
Validates loading, cleaning, category mapping, distribution logging,
leak-free splitting/encoding/scaling, and artifact persistence.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    NSL_KDD_CATEGORIES,
    NSL_KDD_CATEGORICAL_FEATURES,
    NSL_KDD_FEATURES,
    NSL_KDD_NUMERIC_FEATURES,
    build_dataset,
    load_nslkdd,
    log_and_save_class_distribution,
    map_attack_categories,
)
from src.utils import load_config


def test_load_nslkdd_cleaning(tmp_path: Path):
    """Test that load_nslkdd drops difficulty and removes duplicates."""
    dummy_data = []
    dummy_data.append([0] * 41 + ["normal", 20])
    dummy_data.append([0] * 41 + ["normal", 20])
    dummy_data.append([1] * 41 + ["neptune", 21])

    train_file = tmp_path / "dummy_train.txt"
    test_file = tmp_path / "dummy_test.txt"

    df_dummy = pd.DataFrame(dummy_data)
    df_dummy.to_csv(train_file, header=False, index=False)
    df_dummy.to_csv(test_file, header=False, index=False)

    train_df, test_df = load_nslkdd(train_file, test_file)

    assert "difficulty" not in train_df.columns
    assert "difficulty" not in test_df.columns
    assert "label" in train_df.columns
    assert "label" in test_df.columns
    assert len(train_df) == 2
    assert len(test_df) == 2
    assert train_df.duplicated().sum() == 0
    assert test_df.duplicated().sum() == 0
    assert train_df.shape[1] == 42
    assert list(train_df.columns[:41]) == NSL_KDD_FEATURES


def test_load_nslkdd_file_not_found():
    """Test that FileNotFoundError is raised when file paths are invalid."""
    with pytest.raises(FileNotFoundError):
        load_nslkdd("non_existent_train.txt", "non_existent_test.txt")


def test_map_attack_categories():
    """Test mapping raw labels to 5 categories and binary attack flag."""
    df = pd.DataFrame(
        {
            "label": ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"]
        }
    )
    mapped = map_attack_categories(df)

    assert list(mapped["category"]) == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert list(mapped["is_attack"]) == [0, 1, 1, 1, 1]


def test_map_attack_categories_unmapped_error():
    """Test that unknown attack raises ValueError listing the unmapped label."""
    df = pd.DataFrame({"label": ["normal", "alien_attack_xyz"]})
    with pytest.raises(ValueError) as excinfo:
        map_attack_categories(df)
    assert "alien_attack_xyz" in str(excinfo.value)


def test_log_and_save_class_distribution(tmp_path: Path):
    """Test saving class distribution table."""
    config = {"output_dirs": {"tables": str(tmp_path / "tables")}}
    train_df = pd.DataFrame(
        {"category": ["Normal", "Normal", "DoS", "Probe", "R2L", "U2R"]}
    )
    test_df = pd.DataFrame(
        {"category": ["Normal", "DoS", "Probe", "R2L", "U2R"]}
    )

    dist_df = log_and_save_class_distribution(train_df, test_df, config=config)
    assert list(dist_df["category"]) == NSL_KDD_CATEGORIES
    assert dist_df.loc[dist_df["category"] == "Normal", "train_count"].values[0] == 2
    assert dist_df.loc[dist_df["category"] == "Normal", "test_count"].values[0] == 1

    saved_file = tmp_path / "tables" / "class_distribution_nslkdd.csv"
    assert saved_file.exists()


def test_build_dataset_no_leakage(tmp_path: Path):
    """
    STRICT NO-LEAKAGE TEST:
    Verify scaler.data_min_ and data_max_ strictly reflect the training split min/max only,
    uninfluenced by extreme values in the test set.
    """
    num_cols = NSL_KDD_NUMERIC_FEATURES
    cat_cols = NSL_KDD_CATEGORICAL_FEATURES

    # Synthetic training set with known min (10.0) and max (100.0)
    train_records = []
    for i in range(20):
        row = {}
        for c in num_cols:
            row[c] = 10.0 if i == 0 else (100.0 if i == 1 else 50.0)
        row["protocol_type"] = "tcp"
        row["service"] = "http"
        row["flag"] = "SF"
        row["label"] = "normal" if i % 2 == 0 else "neptune"
        train_records.append(row)
    train_df = pd.DataFrame(train_records)

    # Test set containing extreme values (min=-9999.0, max=+9999.0)
    test_records = []
    for i in range(10):
        row = {}
        for c in num_cols:
            row[c] = -9999.0 if i == 0 else 9999.0
        row["protocol_type"] = "udp"
        row["service"] = "unseen_service"
        row["flag"] = "SF"
        row["label"] = "normal"
        test_records.append(row)
    test_df = pd.DataFrame(test_records)

    config = {
        "validation_size": 0.2,
        "seed": 42,
        "output_dirs": {"models": str(tmp_path / "models")},
    }

    dataset = build_dataset(train_df, test_df, config=config)
    scaler = dataset["scaler"]

    # Scaler min must strictly equal 10.0 (from train), NEVER -9999.0 (from test)
    assert np.allclose(scaler.data_min_, 10.0), (
        f"Data leakage detected! Scaler min is {scaler.data_min_}, expected 10.0 from train."
    )
    # Scaler max must strictly equal 100.0 (from train), NEVER 9999.0 (from test)
    assert np.allclose(scaler.data_max_, 100.0), (
        f"Data leakage detected! Scaler max is {scaler.data_max_}, expected 100.0 from train."
    )


def test_build_dataset_shapes_and_artifacts(tmp_path: Path):
    """Verify that dataset splits have consistent feature dimensions and artifacts are saved."""
    num_cols = NSL_KDD_NUMERIC_FEATURES
    cat_cols = NSL_KDD_CATEGORICAL_FEATURES

    records = []
    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    for i in range(50):
        row = {}
        for c in num_cols:
            row[c] = float(i)
        row["protocol_type"] = "tcp" if i % 2 == 0 else "udp"
        row["service"] = "http" if i % 3 == 0 else "smtp"
        row["flag"] = "SF" if i % 2 == 0 else "S0"
        row["label"] = "normal" if i % 5 == 0 else ("neptune" if i % 5 == 1 else "satan")
        records.append(row)

    df = pd.DataFrame(records)
    config = {
        "validation_size": 0.2,
        "seed": 42,
        "output_dirs": {"models": str(tmp_path / "models")},
    }

    dataset = build_dataset(df, df.copy(), config=config)

    # 1. Feature dimension matching
    n_features = len(dataset["feature_names"])
    assert dataset["X_train"].shape[1] == n_features
    assert dataset["X_val"].shape[1] == n_features
    assert dataset["X_test"].shape[1] == n_features

    # 2. Sample size alignment
    assert len(dataset["X_train"]) == len(dataset["y_train_cat"]) == len(dataset["y_train_bin"])
    assert len(dataset["X_val"]) == len(dataset["y_val_cat"]) == len(dataset["y_val_bin"])
    assert len(dataset["X_test"]) == len(dataset["y_test_cat"]) == len(dataset["y_test_bin"])

    # 3. Value scaling bounds in train: [-1, 1]
    assert np.all(dataset["X_train"][:, : len(num_cols)] >= -1.0001)
    assert np.all(dataset["X_train"][:, : len(num_cols)] <= 1.0001)

    # 4. Persistence of models
    assert (tmp_path / "models" / "encoder.joblib").exists()
    assert (tmp_path / "models" / "scaler.joblib").exists()
    assert (tmp_path / "models" / "feature_names.joblib").exists()

    # 5. One-hot column indices coverage
    indices_covered = []
    for indices in dataset["onehot_column_indices"].values():
        indices_covered.extend(indices)
    expected_cat_indices = list(range(len(num_cols), n_features))
    assert sorted(indices_covered) == expected_cat_indices


def test_build_dataset_real_nslkdd():
    """Integration test validating build_dataset on real NSL-KDD dataset."""
    train_path = Path("dataset folder/archive/KDDTrain+.txt")
    test_path = Path("dataset folder/archive/KDDTest+.txt")

    if not train_path.exists() or not test_path.exists():
        pytest.skip("Raw NSL-KDD dataset not found at default location.")

    config = load_config("configs/nslkdd.yaml")
    train_df, test_df = load_nslkdd(train_path, test_path)
    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)

    dataset = build_dataset(train_df, test_df, config=config)

    # KDDTrain+ has 125,973 rows. With 0.2 val split:
    # Train: 100,778 rows; Val: 25,195 rows; Test: 22,544 rows
    assert dataset["X_train"].shape[0] == 100778
    assert dataset["X_val"].shape[0] == 25195
    assert dataset["X_test"].shape[0] == 22544

    # Check zero NaNs across all splits
    assert not np.isnan(dataset["X_train"]).any()
    assert not np.isnan(dataset["X_val"]).any()
    assert not np.isnan(dataset["X_test"]).any()

    # Check feature dimension
    n_features = len(dataset["feature_names"])
    assert dataset["X_train"].shape[1] == n_features
    assert dataset["X_val"].shape[1] == n_features
    assert dataset["X_test"].shape[1] == n_features
    assert len(dataset["onehot_column_indices"]) == 3
