"""
Resampling module for network intrusion detection datasets.

Provides oversampling strategies to address extreme class imbalance in attack
data (e.g. U2R and R2L in NSL-KDD):
    - gan_oversample: Oversampling using a trained Conditional WGAN-GP generator
      with categorical constraint enforcement.
    - random_oversample: Baseline Random Oversampling (ROS).
    - smote_oversample: Synthetic Minority Over-sampling Technique (SMOTE) with
      safe handling for tiny classes.

All resamplers follow a unified target calculation rule:
    target_count = target_ratio * largest_class_count
and return a consistent tuple:
    (X_aug, y_aug, is_synthetic)

CRITICAL: Resamplers must be applied strictly to the training split. Never apply
to validation or test sets.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE

from src.utils import get_logger, save_table
from src.wgan_gp import Generator, generate_samples

logger = get_logger("resamplers")

# Default NSL-KDD attack class ordering when mapping strings to generator class IDs
DEFAULT_ATTACK_CLASSES: List[str] = ["DoS", "Probe", "R2L", "U2R"]


# ---------------------------------------------------------------------------
# Helper: Target Calculation & Table Builder
# ---------------------------------------------------------------------------

def compute_oversampling_targets(
    y: Union[np.ndarray, pd.Series, List[Any]],
    target_ratio: float = 0.5,
) -> Tuple[Dict[Any, int], Dict[Any, int], Dict[Any, int]]:
    """
    Calculate target sample counts and missing counts for each class.

    Target logic:
        target_count = ceil(target_ratio * largest_class_count)
        missing_count = max(0, target_count - current_count)

    Args:
        y: Class labels array or Series.
        target_ratio: Desired ratio relative to the largest class (0.0 to 1.0).

    Returns:
        Tuple of (counts_before, target_counts, missing_counts) dictionaries.
    """
    if target_ratio <= 0:
        raise ValueError(f"target_ratio must be positive, got {target_ratio}")

    counts_before: Dict[Any, int] = dict(Counter(y))
    if not counts_before:
        raise ValueError("Cannot compute targets on empty label array.")

    max_count = max(counts_before.values())
    target_threshold = int(np.ceil(target_ratio * max_count))

    target_counts: Dict[Any, int] = {}
    missing_counts: Dict[Any, int] = {}

    for cls, count in counts_before.items():
        # A class never gets reduced; only raised to target_threshold
        t_count = max(count, target_threshold)
        target_counts[cls] = t_count
        missing_counts[cls] = max(0, t_count - count)

    return counts_before, target_counts, missing_counts


def build_count_table(
    counts_before: Dict[Any, int],
    synthetic_generated: Dict[Any, int],
    target_counts: Dict[Any, int],
) -> pd.DataFrame:
    """
    Create a before/after summary DataFrame for oversampling verification.

    Args:
        counts_before: Count of real samples per class before resampling.
        synthetic_generated: Number of synthetic samples generated per class.
        target_counts: Target count set for each class.

    Returns:
        DataFrame with columns: ['class', 'count_before', 'target_count',
        'synthetic_generated', 'count_after'].
    """
    rows = []
    for cls in counts_before:
        c_before = counts_before[cls]
        c_syn = synthetic_generated.get(cls, 0)
        c_target = target_counts.get(cls, c_before)
        c_after = c_before + c_syn
        rows.append({
            "class": str(cls),
            "count_before": c_before,
            "target_count": c_target,
            "synthetic_generated": c_syn,
            "count_after": c_after,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Task 2.5: WGAN-GP Oversampling
# ---------------------------------------------------------------------------

def gan_oversample(
    X_train_attack: Union[np.ndarray, pd.DataFrame],
    y_train_attack: Union[np.ndarray, pd.Series],
    generator: Generator,
    config: Dict[str, Any],
    onehot_groups: Optional[Dict[str, List[int]]] = None,
    class_map: Optional[Dict[Any, int]] = None,
    table_name: str = "gan_oversample_class_counts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversample minority attack classes using a trained Conditional WGAN-GP generator.

    1. Counts samples per attack class.
    2. Sets a target per class = target_ratio * (largest attack class count),
       where target_ratio comes from config (default 0.5).
    3. Generates only the missing amount for classes below the target using
       ``generate_samples()`` which enforces categorical one-hot validity via argmax.
    4. Returns the augmented (X, y) plus a boolean mask 'is_synthetic'.
    5. Saves a before/after class count table to results/tables/.

    CRITICAL: Never pass validation or test data to this function.

    Args:
        X_train_attack: Training attack features array of shape (N, D).
        y_train_attack: Training attack labels array of shape (N,).
        generator: Trained Generator PyTorch module.
        config: YAML-loaded configuration dict.
        onehot_groups: Dict mapping categorical column group name to list of indices.
                       If None, read from config["onehot_groups"] or config["onehot_column_indices"]
                       or defaults to empty dict.
        class_map: Optional mapping from label values in y_train_attack to 0-indexed
                   class IDs expected by the generator's one-hot condition.
        table_name: Filename (without .csv) to save the before/after class count table.

    Returns:
        Tuple of:
            - X_aug: Augmented feature array of shape (N_aug, D), dtype float32.
            - y_aug: Augmented label array of shape (N_aug,), matching input type.
            - is_synthetic: Boolean array of shape (N_aug,), True for synthetic rows.
    """
    # Convert inputs to standard numpy format while preserving label dtype
    if isinstance(X_train_attack, pd.DataFrame):
        X_real = X_train_attack.values.astype(np.float32)
    else:
        X_real = np.asarray(X_train_attack, dtype=np.float32)

    if isinstance(y_train_attack, (pd.Series, pd.Index)):
        y_real = y_train_attack.values
    else:
        y_real = np.asarray(y_train_attack)

    if len(X_real) != len(y_real):
        raise ValueError(
            f"X_train_attack length ({len(X_real)}) does not match "
            f"y_train_attack length ({len(y_real)})."
        )

    # Resolve target_ratio from config (default 0.5)
    target_ratio: float = float(
        config.get("resampling", {}).get(
            "target_ratio",
            config.get("target_ratio", 0.5),
        )
    )

    # Resolve onehot_groups
    if onehot_groups is None:
        onehot_groups = config.get(
            "onehot_groups",
            config.get("onehot_column_indices", {}),
        )
    if onehot_groups is None:
        onehot_groups = {}

    counts_before, target_counts, missing_counts = compute_oversampling_targets(
        y_real, target_ratio=target_ratio
    )

    logger.info(
        f"GAN Oversampling: Target ratio = {target_ratio:.2f}. "
        f"Class counts before: {counts_before}"
    )

    # Determine class mapping for the generator condition
    unique_classes = list(counts_before.keys())
    mapping: Dict[Any, int] = {}

    if class_map is not None:
        mapping = dict(class_map)
    else:
        # Check if labels are already integers within [0, generator.n_classes - 1]
        try:
            int_classes = [int(c) for c in unique_classes]
            if all(0 <= c < generator.n_classes for c in int_classes):
                mapping = {c: int(c) for c in unique_classes}
        except (ValueError, TypeError):
            pass

        if not mapping:
            # Check if classes match known NSL-KDD attack categories
            configured_classes = config.get("attack_classes", DEFAULT_ATTACK_CLASSES)
            if all(c in configured_classes for c in unique_classes):
                mapping = {c: configured_classes.index(c) for c in unique_classes}
            else:
                # Fall back to sorted order
                sorted_classes = sorted(unique_classes, key=lambda x: str(x))
                mapping = {c: i for i, c in enumerate(sorted_classes)}

    logger.info(f"Class mapping to generator condition ID: {mapping}")

    # Generate synthetic samples for classes below target
    synthetic_X_list: List[np.ndarray] = []
    synthetic_y_list: List[Any] = []
    synthetic_generated: Dict[Any, int] = {}

    for cls, missing in missing_counts.items():
        if missing <= 0:
            synthetic_generated[cls] = 0
            continue

        class_id = mapping.get(cls)
        if class_id is None:
            raise KeyError(
                f"Class '{cls}' not found in class mapping {mapping}. "
                f"Cannot determine condition vector for generator."
            )
        if class_id >= generator.n_classes or class_id < 0:
            raise ValueError(
                f"Mapped class_id {class_id} for '{cls}' is out of bounds for "
                f"generator with {generator.n_classes} classes."
            )

        logger.info(f"Generating {missing} synthetic samples for class '{cls}' (class_id={class_id})...")
        syn_x, _ = generate_samples(
            generator=generator,
            class_id=class_id,
            n=missing,
            config=config,
            onehot_groups=onehot_groups,
        )

        synthetic_X_list.append(syn_x)
        # Create corresponding label array of original label type
        syn_y = np.full(shape=(missing,), fill_value=cls, dtype=y_real.dtype)
        synthetic_y_list.append(syn_y)
        synthetic_generated[cls] = missing

    # Assemble augmented dataset
    if synthetic_X_list:
        X_syn = np.vstack(synthetic_X_list)
        y_syn = np.concatenate(synthetic_y_list)
        X_aug = np.vstack([X_real, X_syn]).astype(np.float32)
        y_aug = np.concatenate([y_real, y_syn])
        is_synthetic = np.zeros(len(y_aug), dtype=bool)
        is_synthetic[len(X_real):] = True
    else:
        X_aug = X_real.copy()
        y_aug = y_real.copy()
        is_synthetic = np.zeros(len(y_aug), dtype=bool)

    # Build and save before/after table
    table_df = build_count_table(counts_before, synthetic_generated, target_counts)
    saved_path = save_table(table_df, table_name, config)
    logger.info(f"Saved oversampling class counts table to: {saved_path}")

    # Verify minority classes reached target
    counts_after = Counter(y_aug)
    for cls, target in target_counts.items():
        if counts_after[cls] < target:
            raise AssertionError(
                f"Class '{cls}' count after resampling ({counts_after[cls]}) "
                f"is less than target ({target})!"
            )

    return X_aug, y_aug, is_synthetic


# ---------------------------------------------------------------------------
# Task 5.1: Baseline Resamplers (RandomOversampler & SMOTE)
# ---------------------------------------------------------------------------

def random_oversample(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    target_ratio: float = 0.5,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    table_name: str = "ros_oversample_class_counts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random Oversampling (ROS) with the exact same target logic as gan_oversample.

    Target count per class = max(current_count, ceil(target_ratio * largest_class_count)).
    Returns (X_aug, y_aug, is_synthetic).

    Args:
        X: Feature array or DataFrame of shape (N, D).
        y: Labels array or Series of shape (N,).
        target_ratio: Target ratio relative to largest class (default 0.5).
        config: Optional configuration dict for saving output tables.
        seed: Random state for reproducibility.
        table_name: Name of CSV table to save before/after class counts.

    Returns:
        Tuple of (X_aug, y_aug, is_synthetic).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)

    if isinstance(y, (pd.Series, pd.Index)):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    counts_before, target_counts, missing_counts = compute_oversampling_targets(
        y_arr, target_ratio=target_ratio
    )

    sampling_strategy = {cls: target_counts[cls] for cls in counts_before}

    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
    X_res, y_res = ros.fit_resample(X_arr, y_arr)

    # Imbalanced-learn appends new samples at the end
    n_original = len(X_arr)
    is_synthetic = np.zeros(len(X_res), dtype=bool)
    is_synthetic[n_original:] = True

    synthetic_generated = {cls: missing_counts[cls] for cls in counts_before}
    table_df = build_count_table(counts_before, synthetic_generated, target_counts)
    if config is not None:
        save_table(table_df, table_name, config)

    return X_res.astype(np.float32), y_res, is_synthetic


def smote_oversample(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    target_ratio: float = 0.5,
    k_neighbors: int = 5,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    table_name: str = "smote_oversample_class_counts",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SMOTE oversampling with the exact same target logic as gan_oversample.

    Handles tiny classes safely:
    If a class has fewer samples than k_neighbors + 1, reduces k_neighbors
    dynamically and logs a warning. If a class has only 1 sample, performs
    random replication for that class and SMOTE on the rest.

    Target count per class = max(current_count, ceil(target_ratio * largest_class_count)).

    Args:
        X: Feature array or DataFrame of shape (N, D).
        y: Labels array or Series of shape (N,).
        target_ratio: Target ratio relative to largest class (default 0.5).
        k_neighbors: Number of nearest neighbors to use for synthetic sample generation.
        config: Optional configuration dict for saving output tables.
        seed: Random state for reproducibility.
        table_name: Name of CSV table to save before/after class counts.

    Returns:
        Tuple of (X_aug, y_aug, is_synthetic).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)

    if isinstance(y, (pd.Series, pd.Index)):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    counts_before, target_counts, missing_counts = compute_oversampling_targets(
        y_arr, target_ratio=target_ratio
    )

    # Check for tiny classes
    classes_needing_oversample = [
        cls for cls, missing in missing_counts.items() if missing > 0
    ]

    if not classes_needing_oversample:
        # All classes already at or above target
        is_synthetic = np.zeros(len(y_arr), dtype=bool)
        return X_arr.copy(), y_arr.copy(), is_synthetic

    # Find the smallest class that needs oversampling
    min_samples_in_minority = min(
        counts_before[cls] for cls in classes_needing_oversample
    )

    effective_k = k_neighbors
    # Tiny class handling: SMOTE requires n_samples > k_neighbors
    if min_samples_in_minority <= k_neighbors:
        effective_k = max(1, min_samples_in_minority - 1)
        logger.warning(
            f"Tiny class detected with {min_samples_in_minority} samples. "
            f"Reducing SMOTE k_neighbors from {k_neighbors} to {effective_k}."
        )

    # If min_samples_in_minority == 1, SMOTE cannot interpolate with k=0
    # In this case, we first duplicate 1-sample classes to 2 samples via RandomOverSampler
    X_work = X_arr
    y_work = y_arr

    if min_samples_in_minority == 1:
        logger.warning(
            "Found minority class with exactly 1 sample. "
            "Performing initial single-sample duplication before SMOTE interpolation."
        )
        pre_strategy = {
            cls: max(counts_before[cls], 2)
            for cls in counts_before
        }
        ros_pre = RandomOverSampler(sampling_strategy=pre_strategy, random_state=seed)
        X_work, y_work = ros_pre.fit_resample(X_arr, y_arr)
        effective_k = 1

    sampling_strategy = {cls: target_counts[cls] for cls in counts_before}

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=effective_k,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(X_work, y_work)

    n_original = len(X_arr)
    is_synthetic = np.zeros(len(X_res), dtype=bool)
    is_synthetic[n_original:] = True

    synthetic_generated = {cls: missing_counts[cls] for cls in counts_before}
    table_df = build_count_table(counts_before, synthetic_generated, target_counts)
    if config is not None:
        save_table(table_df, table_name, config)

    return X_res.astype(np.float32), y_res, is_synthetic
