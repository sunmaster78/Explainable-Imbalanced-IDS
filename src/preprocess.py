"""
Preprocessing module for Network Intrusion Detection System (NIDS) datasets.
Supports loading, cleaning, encoding, splitting, and scaling for NSL-KDD, UNSW-NB15, and CIC-IDS2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils import get_logger, save_table

logger = get_logger("preprocess")

# Standard 41 NSL-KDD network traffic features (in exact dataset order)
NSL_KDD_FEATURES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Complete raw column list including label and difficulty
NSL_KDD_RAW_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]

# Categorical and Numeric feature subsets
NSL_KDD_CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NSL_KDD_NUMERIC_FEATURES = [
    col for col in NSL_KDD_FEATURES if col not in NSL_KDD_CATEGORICAL_FEATURES
]

# 5 High-level categories
NSL_KDD_CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Exhaustive attack-to-category mapping for NSL-KDD (including test-only attacks)
NSL_KDD_ATTACK_MAP: Dict[str, str] = {
    # Normal
    "normal": "Normal",
    # Denial of Service (DoS)
    "apache2": "DoS",
    "back": "DoS",
    "land": "DoS",
    "mailbomb": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "processtable": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "mscan": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "saint": "Probe",
    "satan": "Probe",
    # Remote to Local (R2L)
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "httptunnel": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "named": "R2L",
    "phf": "R2L",
    "sendmail": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    # User to Root (U2R)
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "ps": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}


def load_nslkdd(
    train_path: Union[str, Path],
    test_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the NSL-KDD dataset from raw files.

    Operations:
    1. Reads KDDTrain+.txt and KDDTest+.txt with 41 standard feature names plus 'label' and 'difficulty'.
    2. Drops the 'difficulty' column.
    3. Drops exact duplicate rows (train and test processed separately).
    4. Logs dataset shapes before and after cleaning.

    Args:
        train_path: Path to the training file (KDDTrain+.txt).
        test_path: Path to the testing file (KDDTest+.txt).

    Returns:
        Tuple of (train_df, test_df) as cleaned pandas DataFrames.

    Raises:
        FileNotFoundError: If train_path or test_path does not exist.
    """
    train_file = Path(train_path)
    test_file = Path(test_path)

    if not train_file.is_file():
        raise FileNotFoundError(f"NSL-KDD train file not found at: {train_file}")
    if not test_file.is_file():
        raise FileNotFoundError(f"NSL-KDD test file not found at: {test_file}")

    logger.info("Loading raw NSL-KDD datasets...")
    train_df = pd.read_csv(train_file, header=None, names=NSL_KDD_RAW_COLUMNS)
    test_df = pd.read_csv(test_file, header=None, names=NSL_KDD_RAW_COLUMNS)

    train_raw_shape = train_df.shape
    test_raw_shape = test_df.shape

    logger.info(f"Raw KDDTrain+ shape: {train_raw_shape}")
    logger.info(f"Raw KDDTest+ shape: {test_raw_shape}")

    # Drop 'difficulty' column
    if "difficulty" in train_df.columns:
        train_df = train_df.drop(columns=["difficulty"])
    if "difficulty" in test_df.columns:
        test_df = test_df.drop(columns=["difficulty"])

    # Drop duplicates separately
    train_dups = train_df.duplicated().sum()
    test_dups = test_df.duplicated().sum()

    train_cleaned = train_df.drop_duplicates().reset_index(drop=True)
    test_cleaned = test_df.drop_duplicates().reset_index(drop=True)

    logger.info(
        f"Dropped {train_dups} duplicate row(s) from train. Shape: {train_cleaned.shape}"
    )
    logger.info(
        f"Dropped {test_dups} duplicate row(s) from test. Shape: {test_cleaned.shape}"
    )

    # Verification: ensure zero duplicate rows remain
    assert (
        train_cleaned.duplicated().sum() == 0
    ), "Duplicates still found in cleaned train data"
    assert (
        test_cleaned.duplicated().sum() == 0
    ), "Duplicates still found in cleaned test data"

    return train_cleaned, test_cleaned


def map_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw NSL-KDD attack labels to 5 broad categories:
    Normal, DoS, Probe, R2L, U2R.

    Adds two new columns:
    - 'category': 5-class string ('Normal', 'DoS', 'Probe', 'R2L', 'U2R')
    - 'is_attack': binary integer (0 for Normal, 1 otherwise)

    Args:
        df: DataFrame containing a 'label' column.

    Returns:
        DataFrame with 'category' and 'is_attack' columns added.

    Raises:
        KeyError: If 'label' column is missing from df.
        ValueError: If any attack label in df is not present in NSL_KDD_ATTACK_MAP.
    """
    if "label" not in df.columns:
        raise KeyError("Expected 'label' column not found in DataFrame.")

    # Check for unmapped attack names
    unique_labels = set(df["label"].unique())
    unmapped = unique_labels - set(NSL_KDD_ATTACK_MAP.keys())
    if unmapped:
        raise ValueError(
            f"Unmapped attack type(s) found in dataset: {sorted(list(unmapped))}"
        )

    df_mapped = df.copy()
    df_mapped["category"] = df_mapped["label"].map(NSL_KDD_ATTACK_MAP)
    df_mapped["is_attack"] = (df_mapped["category"] != "Normal").astype(int)

    return df_mapped


def log_and_save_class_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Calculate, log, and save class distribution for train and test sets to results/tables/class_distribution_nslkdd.csv.

    Args:
        train_df: Training DataFrame with 'category' column.
        test_df: Testing DataFrame with 'category' column.
        config: Optional configuration dictionary for output directories.

    Returns:
        DataFrame summarizing counts and percentages for each class.
    """
    if "category" not in train_df.columns or "category" not in test_df.columns:
        raise KeyError(
            "Both train_df and test_df must contain 'category' column before computing distribution."
        )

    categories = NSL_KDD_CATEGORIES
    train_counts = train_df["category"].value_counts().reindex(categories, fill_value=0)
    test_counts = test_df["category"].value_counts().reindex(categories, fill_value=0)

    train_total = len(train_df)
    test_total = len(test_df)

    dist_df = pd.DataFrame(
        {
            "category": categories,
            "train_count": train_counts.values,
            "train_pct": (train_counts.values / train_total * 100).round(2),
            "test_count": test_counts.values,
            "test_pct": (test_counts.values / test_total * 100).round(2),
        }
    )

    logger.info("=== NSL-KDD Class Distribution Summary ===")
    for _, row in dist_df.iterrows():
        logger.info(
            f"Category: {row['category']:<8} | "
            f"Train: {int(row['train_count']):>6} ({row['train_pct']:>5.2f}%) | "
            f"Test: {int(row['test_count']):>5} ({row['test_pct']:>5.2f}%)"
        )
    logger.info(
        f"Total Samples | Train: {train_total} | Test: {test_total}"
    )

    output_path = save_table(dist_df, "class_distribution_nslkdd", config=config)
    logger.info(f"Class distribution saved to: {output_path}")

    return dist_df


def build_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Split, encode, and scale NSL-KDD dataset with strict leak-free preprocessing.

    Operations:
    1. Splits train_df into train and validation splits (stratified on 'category').
    2. One-hot encodes ['protocol_type', 'service', 'flag'] fitted on TRAIN only,
       handling unseen categories in val/test with handle_unknown='ignore'.
    3. Scales numeric features with MinMaxScaler to [-1, 1], fitted on TRAIN only.
    4. Records feature names and one-hot categorical column group index ranges.
    5. Saves fitted encoder and scaler objects with joblib to results/models.

    Args:
        train_df: Cleaned training DataFrame (with 'category' and 'is_attack').
        test_df: Cleaned testing DataFrame (with 'category' and 'is_attack').
        config: Optional configuration dictionary containing validation_size, seed,
                and output_dirs.

    Returns:
        Dict containing:
            - X_train, X_val, X_test (np.ndarray, float32)
            - y_train_cat, y_val_cat, y_test_cat (np.ndarray of strings)
            - y_train_bin, y_val_bin, y_test_bin (np.ndarray of ints)
            - feature_names (List[str])
            - encoder (OneHotEncoder)
            - scaler (MinMaxScaler)
            - onehot_column_indices (Dict[str, List[int]])
    """
    cfg = config or {}

    # Extract configuration parameters with fallbacks
    val_size = cfg.get("validation_size", cfg.get("experiment", {}).get("validation_size", 0.2))
    seed = cfg.get("seed", cfg.get("experiment", {}).get("default_seed", 42))

    train_data = train_df.copy()
    test_data = test_df.copy()

    # Resolve target columns
    cfg_target = cfg.get("target_column") or cfg.get("dataset", {}).get("target_column")
    if cfg_target and cfg_target in train_data.columns:
        cat_target = cfg_target
    elif "category" in train_data.columns:
        cat_target = "category"
    elif "attack_cat" in train_data.columns:
        cat_target = "attack_cat"
    elif "Label" in train_data.columns:
        cat_target = "Label"
    elif "label" in train_data.columns:
        cat_target = "label"
    else:
        cat_target = "category"

    if "is_attack" in train_data.columns:
        bin_target = "is_attack"
    elif "label" in train_data.columns and set(train_data["label"].dropna().unique()).issubset({0, 1}):
        bin_target = "label"
    else:
        train_data["is_attack"] = (
            ~train_data[cat_target].astype(str).str.strip().str.lower().isin(["normal", "benign"])
        ).astype(int)
        test_data["is_attack"] = (
            ~test_data[cat_target].astype(str).str.strip().str.lower().isin(["normal", "benign"])
        ).astype(int)
        bin_target = "is_attack"

    if "category" not in train_data.columns:
        train_data["category"] = train_data[cat_target]
    if "category" not in test_data.columns:
        test_data["category"] = test_data[cat_target]
    if "is_attack" not in train_data.columns:
        train_data["is_attack"] = train_data[bin_target]
    if "is_attack" not in test_data.columns:
        test_data["is_attack"] = test_data[bin_target]

    logger.info(
        f"Splitting training data: validation_size={val_size}, seed={seed}, stratified on '{cat_target}'..."
    )
    train_split, val_split = train_test_split(
        train_data,
        test_size=val_size,
        stratify=train_data[cat_target],
        random_state=seed,
    )

    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_split = test_data.reset_index(drop=True)

    # Determine categorical features
    cfg_cat = cfg.get("dataset", {}).get(
        "categorical_features", cfg.get("categorical_features", None)
    )
    if cfg_cat is not None:
        cat_cols = [c for c in cfg_cat if c in train_split.columns]
    elif all(c in train_split.columns for c in NSL_KDD_CATEGORICAL_FEATURES):
        cat_cols = NSL_KDD_CATEGORICAL_FEATURES
    else:
        cat_cols = [
            c for c in train_split.columns
            if c not in (cat_target, bin_target, "category", "is_attack", "id", "difficulty", "label")
            and (train_split[c].dtype == object or isinstance(train_split[c].dtype, pd.CategoricalDtype))
        ]

    exclude_cols = set(cat_cols) | {
        cat_target,
        bin_target,
        "category",
        "is_attack",
        "id",
        "difficulty",
        "label",
        "attack_cat",
        "Label",
    }
    num_cols = [
        c for c in train_split.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_split[c])
    ]

    # 1. Fit OneHotEncoder strictly on train_split if categorical columns exist
    if cat_cols:
        logger.info(f"Fitting OneHotEncoder on train split for categorical features: {cat_cols}")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(train_split[cat_cols])

        X_train_cat = encoder.transform(train_split[cat_cols])
        X_val_cat = encoder.transform(val_split[cat_cols])
        X_test_cat = encoder.transform(test_split[cat_cols])
        ohe_names = list(encoder.get_feature_names_out(cat_cols))
    else:
        encoder = None
        X_train_cat = np.empty((len(train_split), 0), dtype=np.float32)
        X_val_cat = np.empty((len(val_split), 0), dtype=np.float32)
        X_test_cat = np.empty((len(test_split), 0), dtype=np.float32)
        ohe_names = []

    # 2. Fit MinMaxScaler strictly on train_split to [-1, 1]
    logger.info(f"Fitting MinMaxScaler[-1, 1] on train split for {len(num_cols)} numeric features")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_split[num_cols])

    X_train_num = scaler.transform(train_split[num_cols])
    X_val_num = scaler.transform(val_split[num_cols])
    X_test_num = scaler.transform(test_split[num_cols])

    # 3. Stack numeric + categorical features as float32
    if cat_cols:
        X_train = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
        X_val = np.hstack([X_val_num, X_val_cat]).astype(np.float32)
        X_test = np.hstack([X_test_num, X_test_cat]).astype(np.float32)
    else:
        X_train = X_train_num.astype(np.float32)
        X_val = X_val_num.astype(np.float32)
        X_test = X_test_num.astype(np.float32)

    feature_names = list(num_cols) + ohe_names

    # 4. Map one-hot column groups to their exact column index ranges
    onehot_column_indices: Dict[str, List[int]] = {}
    if encoder is not None:
        curr_idx = len(num_cols)
        for col_name, cat_levels in zip(cat_cols, encoder.categories_):
            num_levels = len(cat_levels)
            onehot_column_indices[col_name] = list(range(curr_idx, curr_idx + num_levels))
            curr_idx += num_levels

    # 5. Extract target arrays
    y_train_cat = train_split["category"].to_numpy()
    y_val_cat = val_split["category"].to_numpy()
    y_test_cat = test_split["category"].to_numpy()

    y_train_bin = train_split["is_attack"].to_numpy().astype(int)
    y_val_bin = val_split["is_attack"].to_numpy().astype(int)
    y_test_bin = test_split["is_attack"].to_numpy().astype(int)

    # 6. Save fitted preprocessing artifacts to results/models
    models_dir = Path(
        cfg.get("output_dirs", {}).get("models", "results/models")
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    encoder_save_path = models_dir / "encoder.joblib"
    scaler_save_path = models_dir / "scaler.joblib"
    meta_save_path = models_dir / "feature_names.joblib"

    joblib.dump(encoder, encoder_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(feature_names, meta_save_path)
    logger.info(f"Fitted preprocessing artifacts saved to: {models_dir}")

    logger.info(
        f"build_dataset complete. Shapes: X_train={X_train.shape}, "
        f"X_val={X_val.shape}, X_test={X_test.shape}, Total Features={len(feature_names)}"
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_cat": y_train_cat,
        "y_val_cat": y_val_cat,
        "y_test_cat": y_test_cat,
        "y_train_bin": y_train_bin,
        "y_val_bin": y_val_bin,
        "y_test_bin": y_test_bin,
        "feature_names": feature_names,
        "encoder": encoder,
        "scaler": scaler,
        "onehot_column_indices": onehot_column_indices,
    }


# ---------------------------------------------------------------------------
# Task 7.1: UNSW-NB15 Dataset Loader
# ---------------------------------------------------------------------------

def load_unsw(
    train_path: Union[str, Path] = "dataset folder/archive (1)/UNSW_NB15_training-set.parquet",
    test_path: Union[str, Path] = "dataset folder/archive (1)/UNSW_NB15_testing-set.parquet",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, and validate official UNSW-NB15 dataset (parquet or csv).

    Operations:
    1. Reads train and test partitions.
    2. Drops 'id' column if present.
    3. Retains 'attack_cat' as multiclass target, 'label' as binary target.
    4. Drops exact duplicate rows (separately from train and test).
    5. Logs class distribution and identifies classes with < 100 training samples.
    6. Saves class distribution table to results/tables/class_distribution_unsw.csv.

    Args:
        train_path: Path to training file (.parquet or .csv).
        test_path: Path to testing file (.parquet or .csv).
        config: Optional configuration dictionary.

    Returns:
        Tuple of (cleaned_train_df, cleaned_test_df).
    """
    t_path = Path(train_path)
    te_path = Path(test_path)

    logger.info(f"Loading UNSW-NB15: train={t_path}, test={te_path}")

    # Read train
    if t_path.suffix == ".parquet":
        train_df = pd.read_parquet(t_path)
    else:
        train_df = pd.read_csv(t_path)

    # Read test
    if te_path.suffix == ".parquet":
        test_df = pd.read_parquet(te_path)
    else:
        test_df = pd.read_csv(te_path)

    logger.info(f"Raw UNSW-NB15 shapes: Train={train_df.shape}, Test={test_df.shape}")

    # Drop 'id' column if present
    for df in (train_df, test_df):
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    # Standardize column strings
    train_df["attack_cat"] = train_df["attack_cat"].astype(str).str.strip()
    test_df["attack_cat"] = test_df["attack_cat"].astype(str).str.strip()

    # Fill any empty or missing attack_cat as 'Normal' if label is 0
    train_df["attack_cat"] = train_df["attack_cat"].replace({"": "Normal", "nan": "Normal"})
    test_df["attack_cat"] = test_df["attack_cat"].replace({"": "Normal", "nan": "Normal"})

    # Ensure alias columns
    train_df["category"] = train_df["attack_cat"]
    test_df["category"] = test_df["attack_cat"]
    train_df["is_attack"] = train_df["label"].astype(int)
    test_df["is_attack"] = test_df["label"].astype(int)

    # Drop duplicates
    train_cleaned = train_df.drop_duplicates().reset_index(drop=True)
    test_cleaned = test_df.drop_duplicates().reset_index(drop=True)

    logger.info(
        f"UNSW-NB15 after deduplication: Train={train_cleaned.shape} "
        f"({len(train_df) - len(train_cleaned)} dropped), "
        f"Test={test_cleaned.shape} ({len(test_df) - len(test_cleaned)} dropped)"
    )

    # Log class distribution and check minority classes (< 100 samples)
    train_counts = train_cleaned["attack_cat"].value_counts()
    test_counts = test_cleaned["attack_cat"].value_counts()

    minority_classes = [cls for cls, cnt in train_counts.items() if cnt < 100]
    if minority_classes:
        logger.info(f"Minority attack classes (< 100 training samples): {minority_classes}")
    else:
        logger.info("All UNSW-NB15 classes in train set have >= 100 samples.")

    # Save class distribution table
    all_cats = sorted(list(set(train_counts.index) | set(test_counts.index)))
    dist_rows = []
    for c in all_cats:
        c_tr = int(train_counts.get(c, 0))
        c_te = int(test_counts.get(c, 0))
        dist_rows.append({
            "category": c,
            "train_count": c_tr,
            "train_pct": round(c_tr / len(train_cleaned) * 100.0, 2),
            "test_count": c_te,
            "test_pct": round(c_te / len(test_cleaned) * 100.0, 2),
            "is_minority_under_100": c_tr < 100,
        })
    dist_df = pd.DataFrame(dist_rows)
    save_table(dist_df, "class_distribution_unsw", config)

    return train_cleaned, test_cleaned


# ---------------------------------------------------------------------------
# Task 7.2: CIC-IDS2017 Dataset Loader
# ---------------------------------------------------------------------------

def load_cicids(
    folder: Union[str, Path] = "dataset folder/archive (2)/MachineLearningCVE",
    sample_fraction: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, merge, and split the 8 daily PCAP CSVs from CIC-IDS2017.

    Operations:
    1. Reads all daily CSVs, strips column-name whitespace, and concatenates.
    2. Replaces inf with NaN, drops NaN rows and duplicate rows.
    3. Optionally takes a stratified sample (sample_fraction from config or 0.1) for speed.
    4. Groups classes with fewer than min_class_samples (default 100) into 'Other_Rare'.
    5. Makes a stratified 70/30 train/test split.
    6. Returns (train_df, test_df).

    Args:
        folder: Path to directory containing PCAP CSV files.
        sample_fraction: Fraction of total data to sample (default from config or 0.1).
        config: Optional configuration dict.
        seed: Random seed for reproducible sampling and train/test split.

    Returns:
        Tuple of (train_df, test_df).
    """
    p_folder = Path(folder)
    if not p_folder.exists():
        raise FileNotFoundError(f"CIC-IDS2017 folder not found: {p_folder}")

    csv_files = sorted(list(p_folder.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {p_folder}")

    logger.info(f"Loading {len(csv_files)} CIC-IDS2017 CSV files from: {p_folder}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        # Strip column-name whitespace
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Raw merged CIC-IDS2017 shape: {merged_df.shape}")

    # Standardize label column name
    if "Label" in merged_df.columns:
        label_col = "Label"
    elif "label" in merged_df.columns:
        label_col = "label"
    else:
        raise KeyError("Could not find 'Label' column in CIC-IDS2017 data.")

    # Clean label values (normalize non-ASCII characters to standard hyphens)
    merged_df[label_col] = (
        merged_df[label_col]
        .astype(str)
        .str.replace(r"[^\x00-\x7F]+", " - ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Replace infinite values with NaN and drop NaNs
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before = len(merged_df)
    merged_df.dropna(inplace=True)
    n_after_nan = len(merged_df)

    # Drop duplicates
    merged_df.drop_duplicates(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    logger.info(
        f"Cleaned CIC-IDS2017: {n_before - n_after_nan} NaN rows dropped, "
        f"{n_after_nan - len(merged_df)} duplicates dropped. Remaining: {len(merged_df)}"
    )

    # Sample fraction handling
    cfg = config or {}
    if sample_fraction is None:
        sample_fraction = float(
            cfg.get("dataset", {}).get(
                "sample_fraction",
                cfg.get("sample_fraction", 0.1),
            )
        )

    if 0.0 < sample_fraction < 1.0:
        logger.info(f"Subsampling {sample_fraction * 100:.1f}% stratified by '{label_col}' (seed={seed})...")
        sampled_df, _ = train_test_split(
            merged_df,
            train_size=sample_fraction,
            stratify=merged_df[label_col],
            random_state=seed,
        )
        working_df = sampled_df.reset_index(drop=True)
    else:
        working_df = merged_df

    # Group rare classes with fewer than min_class_samples (default 100)
    min_samples = int(cfg.get("dataset", {}).get("min_class_samples", cfg.get("min_class_samples", 100)))
    rare_label = str(cfg.get("dataset", {}).get("rare_class_name", cfg.get("rare_class_name", "Other_Rare")))

    class_counts = working_df[label_col].value_counts()
    rare_classes = [c for c, cnt in class_counts.items() if cnt < min_samples]

    if rare_classes:
        logger.info(
            f"Grouping {len(rare_classes)} rare class(es) (< {min_samples} samples) into '{rare_label}': {rare_classes}"
        )
        working_df[label_col] = working_df[label_col].replace({c: rare_label for c in rare_classes})

    # Add uniform category and is_attack columns
    working_df["category"] = working_df[label_col]
    working_df["is_attack"] = (
        ~working_df["category"].str.lower().isin(["benign", "normal"])
    ).astype(int)

    # Stratified 70/30 train/test split
    test_size = float(cfg.get("dataset", {}).get("test_size", cfg.get("test_size", 0.3)))
    min_cat_count = working_df["category"].value_counts().min()
    strat_col = working_df["category"] if min_cat_count >= 2 else None

    train_split, test_split = train_test_split(
        working_df,
        test_size=test_size,
        stratify=strat_col,
        random_state=seed,
    )

    train_split = train_split.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)

    logger.info(
        f"CIC-IDS2017 splits: Train={train_split.shape}, Test={test_split.shape} "
        f"across {working_df['category'].nunique()} classes"
    )

    # Save class distribution table
    dist_df = pd.DataFrame({
        "category": train_split["category"].value_counts().index,
        "train_count": train_split["category"].value_counts().values,
        "test_count": test_split["category"].value_counts().reindex(train_split["category"].value_counts().index, fill_value=0).values,
    })
    save_table(dist_df, "class_distribution_cicids", config)

    return train_split, test_split

