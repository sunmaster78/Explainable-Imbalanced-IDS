"""
Baseline Models and Experimental Benchmarks.

Provides benchmark methods against which the proposed Two-Stage WGAN-GP pipeline
is empirically evaluated:

Single-Stage Baselines (Task 5.2):
    - B1: 'none'    - Flat 5-class ensemble without resampling
    - B2: 'ros'     - Flat 5-class ensemble with Random Oversampling
    - B3: 'smote'   - Flat 5-class ensemble with SMOTE oversampling
    - B4: 'gan_all' - Flat 5-class ensemble with WGAN-GP trained on ALL classes (including Normal)

Two-Stage Baselines (Task 5.3):
    - B5: 'ros'     - Two-stage architecture with Stage 2 Random Oversampling
    - B6: 'smote'   - Two-stage architecture with Stage 2 SMOTE oversampling
    - Proposed: 'wgan' - Two-stage architecture with Stage 2 Conditional WGAN-GP oversampling
    - Two-stage 'none' - Two-stage architecture with no resampling in Stage 2

Both functions record training and inference execution times for efficiency benchmarking.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.ensemble import get_ensemble
from src.evaluate import compute_metrics
from src.gan_quality import nearest_neighbor_distance, tstr_score
from src.pipeline import TwoStageIDS
from src.resamplers import gan_oversample, random_oversample, smote_oversample
from src.utils import get_logger, save_table, set_seed
from src.wgan_gp import generate_samples, train_wgan_gp

logger = get_logger("baselines")


# ---------------------------------------------------------------------------
# Task 5.2: Single-Stage Baselines
# ---------------------------------------------------------------------------

def run_single_stage(
    method: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a single-stage (flat 5-class) baseline model on the dataset.

    Args:
        method: Resampling method: 'none' (B1), 'ros' (B2), 'smote' (B3), or 'gan_all' (B4).
        data: Dict containing 'X_train', 'y_train_cat', 'X_val', 'y_val_cat',
              'X_test', 'y_test_cat', and optional 'onehot_column_indices'.
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Dict containing:
            - 'y_pred': 1D array of predicted class strings on test set.
            - 'y_proba': 2D array of class probabilities (N_test, n_classes).
            - 'classes': List of class names corresponding to y_proba columns.
            - 'training_time': Elapsed training time in seconds.
            - 'inference_time': Elapsed test inference time in seconds.
            - 'model': Fitted VotingClassifier or StackingClassifier.
    """
    set_seed(seed)
    m = method.strip().lower()

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train_cat"])
    X_test = np.asarray(data["X_test"], dtype=np.float32)

    target_ratio = float(
        config.get("resampling", {}).get("target_ratio", config.get("target_ratio", 0.5))
    )
    onehot_groups = data.get("onehot_column_indices", {})

    logger.info(f"--- Running Single-Stage Baseline [{m.upper()}] (seed={seed}) ---")
    start_train_time = time.perf_counter()

    # 1. Apply Resampling if specified
    if m in ("none", "b1"):
        X_res, y_res = X_train, y_train
        is_syn = np.zeros(len(y_res), dtype=bool)

    elif m in ("ros", "b2"):
        X_res, y_res, is_syn = random_oversample(
            X=X_train,
            y=y_train,
            target_ratio=target_ratio,
            config=config,
            seed=seed,
            table_name=f"ros_single_stage_seed{seed}",
        )

    elif m in ("smote", "b3"):
        X_res, y_res, is_syn = smote_oversample(
            X=X_train,
            y=y_train,
            target_ratio=target_ratio,
            k_neighbors=5,
            config=config,
            seed=seed,
            table_name=f"smote_single_stage_seed{seed}",
        )

    elif m in ("gan_all", "b4"):
        # Novelty Claim 3 Comparison: WGAN-GP trained across ALL classes including Normal
        logger.info("Training Flat WGAN-GP across all classes including Normal...")
        all_classes = sorted(list(np.unique(y_train)))
        class_to_id = {c: i for i, c in enumerate(all_classes)}
        y_train_onehot = np.zeros((len(y_train), len(all_classes)), dtype=np.float32)
        for i, c in enumerate(y_train):
            y_train_onehot[i, class_to_id[c]] = 1.0

        gan_cfg = config.copy()
        gan_train_res = train_wgan_gp(
            X_attack=X_train,
            y_attack_onehot=y_train_onehot,
            config=gan_cfg,
        )
        generator = gan_train_res["generator"]

        X_res, y_res, is_syn = gan_oversample(
            X_train_attack=X_train,
            y_train_attack=y_train,
            generator=generator,
            config=config,
            onehot_groups=onehot_groups,
            class_map=class_to_id,
            table_name=f"gan_all_single_stage_seed{seed}",
        )
    else:
        raise ValueError(
            f"Unknown single-stage method: '{method}'. Expected 'none', 'ros', 'smote', or 'gan_all'."
        )

    # 2. Encode labels
    label_encoder = LabelEncoder()
    y_res_encoded = label_encoder.fit_transform(y_res)
    n_classes = len(label_encoder.classes_)

    # 3. Fit ensemble classifier
    ensemble_kind = config.get("ensemble", {}).get("kind", "voting")
    model = get_ensemble(
        kind=ensemble_kind,
        config=config,
        n_classes=n_classes,
        seed=seed,
    )
    model.fit(X_res, y_res_encoded)
    training_time = time.perf_counter() - start_train_time

    # 4. Evaluate inference time on test set
    start_infer_time = time.perf_counter()
    y_proba = model.predict_proba(X_test)
    y_pred_encoded = np.argmax(y_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    inference_time = time.perf_counter() - start_infer_time

    logger.info(
        f"Single-Stage [{m.upper()}] completed: Train time={training_time:.2f}s, "
        f"Infer time={inference_time:.4f}s"
    )

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": list(label_encoder.classes_),
        "training_time": training_time,
        "inference_time": inference_time,
        "model": model,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# Task 5.3: Two-Stage Baselines
# ---------------------------------------------------------------------------

def run_two_stage(
    method: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a two-stage IDS model reusing TwoStageIDS.

    Stage 1: Binary classifier (Normal vs Attack) with validation threshold calibration.
    Stage 2: Multiclass attack classifier trained ONLY on attack samples:
        - 'none':   No oversampling in Stage 2
        - 'ros':    Random Oversampling in Stage 2 (B5)
        - 'smote':  SMOTE oversampling in Stage 2 (B6)
        - 'wgan':   Proposed Conditional WGAN-GP oversampling in Stage 2

    Args:
        method: 'ros' (B5), 'smote' (B6), 'wgan' (Proposed), or 'none'.
        data: Dict containing 'X_train', 'y_train_cat', 'y_train_bin',
              'X_val', 'y_val_bin', 'X_test', 'y_test_cat', 'y_test_bin',
              and optional 'onehot_column_indices'.
        config: Configuration dictionary.
        seed: Random seed.

    Returns:
        Dict containing:
            - 'y_pred': 1D array of predicted class strings on test set.
            - 'stage1_threshold': Calibrated threshold for Stage 1.
            - 'training_time': Total training time (Stage 1 + Stage 2 + GAN if applicable).
            - 'inference_time': Total test inference time.
            - 'pipeline': Fitted TwoStageIDS instance.
    """
    set_seed(seed)
    m = method.strip().lower()

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train_cat = np.asarray(data["y_train_cat"])
    y_train_bin = np.asarray(data["y_train_bin"], dtype=int)

    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val_bin = np.asarray(data["y_val_bin"], dtype=int)

    X_test = np.asarray(data["X_test"], dtype=np.float32)

    target_ratio = float(
        config.get("resampling", {}).get("target_ratio", config.get("target_ratio", 0.5))
    )
    onehot_groups = data.get("onehot_column_indices", {})

    logger.info(f"--- Running Two-Stage Pipeline [{m.upper()}] (seed={seed}) ---")
    start_train_time = time.perf_counter()

    ensemble_kind = config.get("ensemble", {}).get("kind", "voting")
    pipeline = TwoStageIDS(config=config, ensemble_kind=ensemble_kind, seed=seed)

    # 1. Fit Stage 1 (Binary)
    pipeline.fit_stage1(X_train, y_train_bin)

    # 2. Tune Stage 1 threshold on validation partition
    pipeline.tune_stage1_threshold(X_val, y_val_bin)

    # 3. Prepare Stage 2 Attack-Only training set
    attack_mask = y_train_bin == 1
    X_train_att = X_train[attack_mask]
    y_train_att = y_train_cat[attack_mask]

    if m in ("wgan", "proposed"):
        # Train Stage-Aware Conditional WGAN-GP strictly on attack samples
        logger.info("Training Stage-Aware Conditional WGAN-GP on attack-only data...")
        unique_attacks = sorted(list(np.unique(y_train_att)))
        att_to_id = {c: i for i, c in enumerate(unique_attacks)}

        y_att_onehot = np.zeros((len(y_train_att), len(unique_attacks)), dtype=np.float32)
        for i, c in enumerate(y_train_att):
            y_att_onehot[i, att_to_id[c]] = 1.0

        gan_cfg = config.copy()
        gan_train_res = train_wgan_gp(
            X_attack=X_train_att,
            y_attack_onehot=y_att_onehot,
            config=gan_cfg,
        )
        generator = gan_train_res["generator"]

        pipeline.fit_stage2(
            X_train_attack=X_train_att,
            y_train_attack=y_train_att,
            use_gan=True,
            generator=generator,
            onehot_groups=onehot_groups,
        )

    elif m in ("ros", "b5"):
        X_att_res, y_att_res, is_syn = random_oversample(
            X=X_train_att,
            y=y_train_att,
            target_ratio=target_ratio,
            config=config,
            seed=seed,
            table_name=f"ros_two_stage_seed{seed}",
        )
        pipeline.fit_stage2(
            X_train_attack=X_att_res,
            y_train_attack=y_att_res,
            use_gan=False,
        )
        pipeline.is_synthetic_stage2 = is_syn

    elif m in ("smote", "b6"):
        X_att_res, y_att_res, is_syn = smote_oversample(
            X=X_train_att,
            y=y_train_att,
            target_ratio=target_ratio,
            k_neighbors=5,
            config=config,
            seed=seed,
            table_name=f"smote_two_stage_seed{seed}",
        )
        pipeline.fit_stage2(
            X_train_attack=X_att_res,
            y_train_attack=y_att_res,
            use_gan=False,
        )
        pipeline.is_synthetic_stage2 = is_syn

    elif m in ("none", "no_resample"):
        pipeline.fit_stage2(
            X_train_attack=X_train_att,
            y_train_attack=y_train_att,
            use_gan=False,
        )

    else:
        raise ValueError(
            f"Unknown two-stage method: '{method}'. Expected 'ros', 'smote', 'wgan', or 'none'."
        )

    training_time = time.perf_counter() - start_train_time

    # 4. Measure inference time on test set
    start_infer_time = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_time = time.perf_counter() - start_infer_time

    logger.info(
        f"Two-Stage [{m.upper()}] completed: Threshold={pipeline.stage1_threshold:.4f}, "
        f"Train time={training_time:.2f}s, Infer time={inference_time:.4f}s"
    )

    return {
        "y_pred": y_pred,
        "stage1_threshold": pipeline.stage1_threshold,
        "training_time": training_time,
        "inference_time": inference_time,
        "pipeline": pipeline,
    }


# ---------------------------------------------------------------------------
# Task 8.5: Claim 3 - Flat GAN vs Attack-Only GAN Comparison
# ---------------------------------------------------------------------------

def compare_flat_vs_attack_only(
    data: Dict[str, Any],
    config: Dict[str, Any],
    seed: int = 42,
    table_name: str = "flat_vs_attack_only",
) -> pd.DataFrame:
    """
    Execute Novelty Claim 3: Compare Flat WGAN-GP (All Classes) vs Attack-Only WGAN-GP.

    Trains two WGAN-GPs under identical architectures:
    (a) Flat GAN: Trained across ALL classes including Normal traffic.
    (b) Attack-Only GAN (Proposed): Trained strictly on attack rows only.

    For each condition, records:
    1. Class imbalance ratio seen by the GAN (largest class / smallest class).
    2. Critic-loss stability (standard deviation of critic loss across last 20% epochs).
    3. TSTR macro-F1 (Train on Synthetic, Test on Real).
    4. Nearest-neighbor memorization rate (% synthetic samples within distance < 1e-3).
    5. Final Two-Stage Macro-F1.
    6. Average minority class recall (e.g. R2L and U2R).
    7. GAN training time (seconds).

    Args:
        data: Preprocessed dataset dictionary.
        config: Configuration dictionary.
        seed: Random seed.
        table_name: Name for saved results table.

    Returns:
        pd.DataFrame containing empirical comparison metrics for both conditions.
    """
    set_seed(seed)
    logger.info("============================================================")
    logger.info("Executing Novelty Claim 3: Flat GAN vs Attack-Only GAN Audit")
    logger.info("============================================================")

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train_cat = np.asarray(data["y_train_cat"])
    y_train_bin = np.asarray(data["y_train_bin"])
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val_bin = np.asarray(data["y_val_bin"])
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test_cat = np.asarray(data["y_test_cat"])
    onehot_groups = data.get("onehot_column_indices", {})
    device = "cuda" if (torch.cuda.is_available() and config.get("use_gpu", True)) else "cpu"

    # Separate attack rows strictly for Attack-Only and evaluation
    attack_mask_train = y_train_bin == 1
    X_train_att = X_train[attack_mask_train]
    y_train_att = y_train_cat[attack_mask_train]

    normal_labels = ("normal", "benign")
    test_att_mask = ~pd.Series(y_test_cat).str.lower().isin(normal_labels)
    X_test_att = X_test[test_att_mask]
    y_test_att = y_test_cat[test_att_mask]

    attack_classes = sorted(list(np.unique(y_train_att)))
    target_ratio = float(
        config.get("resampling", {}).get("target_ratio", config.get("target_ratio", 0.5))
    )
    class_counts_att = pd.Series(y_train_att).value_counts()
    target_count = int(class_counts_att.max() * target_ratio)
    minority_classes = [c for c, cnt in class_counts_att.items() if cnt < target_count]
    if not minority_classes:
        minority_classes = list(class_counts_att.tail(2).index)

    records = []

    # -----------------------------------------------------------------------
    # Condition (a): Flat WGAN-GP (Trained on ALL classes including Normal)
    # -----------------------------------------------------------------------
    logger.info("--- [1/2] Training Condition (a): Flat WGAN-GP (All Classes) ---")
    all_classes = sorted(list(np.unique(y_train_cat)))
    flat_counts = pd.Series(y_train_cat).value_counts()
    flat_imbalance_ratio = float(flat_counts.max() / max(flat_counts.min(), 1))

    class_to_id_flat = {c: i for i, c in enumerate(all_classes)}
    y_flat_onehot = np.zeros((len(y_train_cat), len(all_classes)), dtype=np.float32)
    for i, c in enumerate(y_train_cat):
        y_flat_onehot[i, class_to_id_flat[c]] = 1.0

    t0_flat = time.perf_counter()
    flat_train_res = train_wgan_gp(
        X_attack=X_train,
        y_attack_onehot=y_flat_onehot,
        config=config,
        device=device,
        save_plots=False,
    )
    t_flat_gan = time.perf_counter() - t0_flat
    gen_flat = flat_train_res["generator"]

    # Critic stability (std of last 20% epochs)
    loss_df_flat = flat_train_res["loss_history"]
    n_last_flat = max(1, int(len(loss_df_flat) * 0.2))
    critic_std_flat = float(loss_df_flat["critic_loss"].iloc[-n_last_flat:].std(ddof=0))

    # Generate synthetic samples for attack classes
    syn_flat_list_X = []
    syn_flat_list_y = []
    for cls in attack_classes:
        cls_id = class_to_id_flat[cls]
        s, _ = generate_samples(gen_flat, cls_id, n=100, config=config, onehot_groups=onehot_groups, device=device)
        syn_flat_list_X.append(s)
        syn_flat_list_y.extend([cls] * len(s))
    X_syn_flat = np.vstack(syn_flat_list_X)
    y_syn_flat = np.array(syn_flat_list_y)

    # TSTR & Memorization
    tstr_flat = tstr_score(X_syn_flat, y_syn_flat, X_test_att, y_test_att)["tstr_macro_f1"]
    nn_flat = nearest_neighbor_distance(X_syn_flat, X_train)["memorization_rate_pct"]

    # Downstream Two-Stage pipeline with Flat generator
    pipe_flat = TwoStageIDS(config=config, seed=seed)
    pipe_flat.fit_stage1(X_train, y_train_bin)
    pipe_flat.tune_stage1_threshold(X_val, y_val_bin)
    # Augment Stage 2 with flat generator
    X_att_flat_aug, y_att_flat_aug, _ = gan_oversample(
        X_train_attack=X_train_att,
        y_train_attack=y_train_att,
        generator=gen_flat,
        config=config,
        onehot_groups=onehot_groups,
        class_map=class_to_id_flat,
    )
    pipe_flat.fit_stage2(X_att_flat_aug, y_att_flat_aug, use_gan=False)
    y_pred_flat = pipe_flat.predict(X_test)
    m_flat = compute_metrics(y_test_cat, y_pred_flat, classes=all_classes)

    min_recall_flat = float(np.mean([
        m_flat["per_class"][c]["recall"] for c in minority_classes if c in m_flat["per_class"]
    ])) if minority_classes else 0.0

    records.append({
        "condition": "Flat GAN (All Classes)",
        "class_imbalance_ratio": round(flat_imbalance_ratio, 2),
        "critic_loss_stability_std": round(critic_std_flat, 4),
        "tstr_macro_f1": round(tstr_flat, 4),
        "nn_memorization_rate_pct": round(nn_flat, 2),
        "two_stage_macro_f1": round(m_flat["f1_macro"], 4),
        "minority_recall": round(min_recall_flat, 4),
        "gan_train_time_sec": round(t_flat_gan, 2),
    })

    # -----------------------------------------------------------------------
    # Condition (b): Attack-Only WGAN-GP (Proposed Architecture)
    # -----------------------------------------------------------------------
    logger.info("--- [2/2] Training Condition (b): Attack-Only WGAN-GP (Proposed) ---")
    att_imbalance_ratio = float(class_counts_att.max() / max(class_counts_att.min(), 1))

    class_to_id_att = {c: i for i, c in enumerate(attack_classes)}
    y_att_onehot = np.zeros((len(y_train_att), len(attack_classes)), dtype=np.float32)
    for i, c in enumerate(y_train_att):
        y_att_onehot[i, class_to_id_att[c]] = 1.0

    t0_att = time.perf_counter()
    att_train_res = train_wgan_gp(
        X_attack=X_train_att,
        y_attack_onehot=y_att_onehot,
        config=config,
        device=device,
        save_plots=False,
    )
    t_att_gan = time.perf_counter() - t0_att
    gen_att = att_train_res["generator"]

    loss_df_att = att_train_res["loss_history"]
    n_last_att = max(1, int(len(loss_df_att) * 0.2))
    critic_std_att = float(loss_df_att["critic_loss"].iloc[-n_last_att:].std(ddof=0))

    # Generate synthetic samples for attack classes
    syn_att_list_X = []
    syn_att_list_y = []
    for cls in attack_classes:
        cls_id = class_to_id_att[cls]
        s, _ = generate_samples(gen_att, cls_id, n=100, config=config, onehot_groups=onehot_groups, device=device)
        syn_att_list_X.append(s)
        syn_att_list_y.extend([cls] * len(s))
    X_syn_att = np.vstack(syn_att_list_X)
    y_syn_att = np.array(syn_att_list_y)

    tstr_att = tstr_score(X_syn_att, y_syn_att, X_test_att, y_test_att)["tstr_macro_f1"]
    nn_att = nearest_neighbor_distance(X_syn_att, X_train_att)["memorization_rate_pct"]

    # Downstream Two-Stage pipeline with Proposed Attack-Only generator
    pipe_att = TwoStageIDS(config=config, seed=seed)
    pipe_att.fit_stage1(X_train, y_train_bin)
    pipe_att.tune_stage1_threshold(X_val, y_val_bin)
    pipe_att.fit_stage2(X_train_att, y_train_att, use_gan=True, generator=gen_att, onehot_groups=onehot_groups)
    y_pred_att = pipe_att.predict(X_test)
    m_att = compute_metrics(y_test_cat, y_pred_att, classes=all_classes)

    min_recall_att = float(np.mean([
        m_att["per_class"][c]["recall"] for c in minority_classes if c in m_att["per_class"]
    ])) if minority_classes else 0.0

    records.append({
        "condition": "Attack-Only (Proposed)",
        "class_imbalance_ratio": round(att_imbalance_ratio, 2),
        "critic_loss_stability_std": round(critic_std_att, 4),
        "tstr_macro_f1": round(tstr_att, 4),
        "nn_memorization_rate_pct": round(nn_att, 2),
        "two_stage_macro_f1": round(m_att["f1_macro"], 4),
        "minority_recall": round(min_recall_att, 4),
        "gan_train_time_sec": round(t_att_gan, 2),
    })

    df_comp = pd.DataFrame(records)
    save_table(df_comp, table_name, config=config)
    logger.info(
        f"Claim 3 audit complete! Table saved to: results/tables/{table_name}.csv"
    )
    return df_comp

