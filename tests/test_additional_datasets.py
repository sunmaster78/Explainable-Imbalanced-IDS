"""
Unit tests for additional datasets: UNSW-NB15 and CIC-IDS2017 (Phase 7).

Tests:
    - load_unsw: schema cleaning, id dropping, minority reporting (Task 7.1)
    - load_cicids: daily CSV merging, whitespace strip, inf/NaN clean, rare grouping (Task 7.2)
    - Generalized build_dataset for non-NSL-KDD datasets (all-numeric and custom categorical)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_dataset, load_cicids, load_unsw


class TestLoadUnsw:
    def test_load_unsw_mock(self, tmp_path):
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"

        df_tr = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "proto": ["tcp", "udp", "tcp", "tcp"],
            "service": ["http", "-", "dns", "http"],
            "state": ["FIN", "CON", "FIN", "FIN"],
            "dur": [0.1, 0.2, 0.05, 0.3],
            "spkts": [10, 20, 15, 30],
            "attack_cat": ["Normal", "Generic", "Worms", "Normal"],
            "label": [0, 1, 1, 0],
        })
        df_te = pd.DataFrame({
            "id": [10, 11],
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["FIN", "CON"],
            "dur": [0.12, 0.22],
            "spkts": [12, 22],
            "attack_cat": ["Normal", "Worms"],
            "label": [0, 1],
        })
        df_tr.to_csv(train_file, index=False)
        df_te.to_csv(test_file, index=False)

        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}
        train_c, test_c = load_unsw(train_file, test_file, config=config)

        assert "id" not in train_c.columns
        assert "category" in train_c.columns
        assert "is_attack" in train_c.columns
        assert len(train_c) == 4
        assert len(test_c) == 2
        assert (tmp_path / "tables" / "class_distribution_unsw.csv").exists()


class TestLoadCicids:
    def test_load_cicids_mock(self, tmp_path):
        csv_dir = tmp_path / "cicids_csvs"
        csv_dir.mkdir()

        # Day 1 CSV with leading whitespace in column names and an inf value
        df1 = pd.DataFrame({
            " Flow Duration": [100, 200, 300],
            " Total Fwd Packets": [2, 4, 6],
            "Flow Bytes/s": [1.5, np.inf, 2.5],
            " Label": ["BENIGN", "BENIGN", "DDoS"],
        })
        # Day 2 CSV with rare class
        df2 = pd.DataFrame({
            " Flow Duration": [150, 250, 350, 450],
            " Total Fwd Packets": [3, 5, 7, 9],
            "Flow Bytes/s": [1.8, 2.8, 3.8, 4.8],
            " Label": ["BENIGN", "Heartbleed", "DDoS", "DDoS"],
        })

        df1.to_csv(csv_dir / "day1.csv", index=False)
        df2.to_csv(csv_dir / "day2.csv", index=False)

        config = {
            "dataset": {
                "sample_fraction": 1.0,
                "min_class_samples": 2,  # Heartbleed has 1 sample -> grouped into Other_Rare
                "rare_class_name": "Other_Rare",
                "test_size": 0.4,
            },
            "output_dirs": {"tables": str(tmp_path / "tables")},
        }

        train_df, test_df = load_cicids(csv_dir, sample_fraction=1.0, config=config, seed=42)

        # Inf row should be dropped -> 7 - 1 = 6 total rows
        assert len(train_df) + len(test_df) == 6
        assert "Flow Duration" in train_df.columns  # leading space stripped
        # Heartbleed grouped into Other_Rare
        all_cats = set(train_df["category"]).union(set(test_df["category"]))
        assert "Heartbleed" not in all_cats
        assert "Other_Rare" in all_cats
        assert (tmp_path / "tables" / "class_distribution_cicids.csv").exists()


class TestGeneralizedBuildDataset:
    def test_all_numeric_dataset(self, tmp_path):
        """Test build_dataset with zero categorical columns (like CIC-IDS)."""
        train_df = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "category": ["BENIGN"] * 30 + ["DDoS"] * 20,
            "is_attack": [0] * 30 + [1] * 20,
        })
        test_df = pd.DataFrame({
            "feat1": np.random.randn(20),
            "feat2": np.random.randn(20),
            "category": ["BENIGN"] * 12 + ["DDoS"] * 8,
            "is_attack": [0] * 12 + [1] * 8,
        })

        config = {
            "validation_size": 0.2,
            "categorical_features": [],
            "output_dirs": {"models": str(tmp_path / "models")},
        }

        dataset = build_dataset(train_df, test_df, config=config)

        assert dataset["X_train"].shape == (40, 2)
        assert dataset["X_val"].shape == (10, 2)
        assert dataset["X_test"].shape == (20, 2)
        assert dataset["encoder"] is None
        assert dataset["onehot_column_indices"] == {}
        assert dataset["scaler"] is not None
