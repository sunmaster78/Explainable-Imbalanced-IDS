"""
Evaluate two-stage models (B5, B6, Proposed) across Stage 1 thresholds on the actual test set.
This directly diagnoses and solves Issue 1, 2, 3 from the review plan.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.preprocess import build_dataset, load_nslkdd, map_attack_categories
from src.baselines import run_two_stage
from src.pipeline import TwoStageIDS
from src.evaluate import compute_metrics
from src.utils import get_logger, load_config, set_seed
from src.wgan_gp import Generator
from src.resamplers import smote_oversample

logger = get_logger("threshold_sweep")

def evaluate_at_threshold(pipeline: TwoStageIDS, X_test: np.ndarray, y_test_cat: np.ndarray, threshold: float, classes: list):
    """Evaluate pipeline at a specific Stage 1 threshold."""
    pipeline.stage1_threshold = threshold
    y_pred = pipeline.predict(X_test)
    m = compute_metrics(y_test_cat, y_pred, classes=classes)
    
    # Also compute test-set Stage 1 FPR
    p_attack = pipeline.predict_proba_stage1(X_test)
    normal_mask = (pd.Series(y_test_cat).str.lower() == "normal").values
    test_fpr = float(np.mean(p_attack[normal_mask] >= threshold))
    test_attack_recall = float(np.mean(p_attack[~normal_mask] >= threshold))
    
    res = {
        "threshold": threshold,
        "accuracy": m["accuracy"],
        "macro_f1": m["f1_macro"],
        "mcc": m["mcc"],
        "test_fpr": test_fpr,
        "test_attack_recall": test_attack_recall,
    }
    for c in classes:
        res[f"{c}_recall"] = m["per_class"][c]["recall"]
        res[f"{c}_precision"] = m["per_class"][c]["precision"]
        res[f"{c}_f1"] = m["per_class"][c]["f1"]
    return res

def main():
    config = load_config("configs/nslkdd.yaml")
    seed = 0
    set_seed(seed)
    
    logger.info("Loading NSL-KDD dataset...")
    train_raw, test_raw = load_nslkdd(
        Path(config["dataset"]["raw_train_path"]),
        Path(config["dataset"]["raw_test_path"])
    )
    train_mapped = map_attack_categories(train_raw)
    test_mapped = map_attack_categories(test_raw)
    dataset = build_dataset(train_mapped, test_mapped, config)
    
    X_train = dataset["X_train"]
    y_train_cat = dataset["y_train_cat"]
    y_train_bin = dataset["y_train_bin"]
    X_test = dataset["X_test"]
    y_test_cat = dataset["y_test_cat"]
    classes = config["classes"]
    
    # 1. Fit Stage 1 once
    logger.info("Fitting Stage 1 Binary Model...")
    pipe_base = TwoStageIDS(config=config, seed=seed)
    pipe_base.fit_stage1(X_train, y_train_bin)
    
    attack_mask = y_train_bin == 1
    X_train_att = X_train[attack_mask]
    y_train_att = y_train_cat[attack_mask]
    onehot_groups = dataset.get("onehot_column_indices", {})
    
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.95]
    all_records = []
    
    # --- Model 1: B5 (No Resampling in Stage 2) ---
    logger.info("Setting up B5 (No Resampling)...")
    pipe_b5 = TwoStageIDS(config=config, seed=seed)
    pipe_b5.stage1_model = pipe_base.stage1_model
    pipe_b5.fit_stage2(X_train_att, y_train_att, use_gan=False)
    for th in thresholds:
        r = evaluate_at_threshold(pipe_b5, X_test, y_test_cat, th, classes)
        r["method"] = "B5 (None)"
        all_records.append(r)
        
    # --- Model 2: B6 (SMOTE in Stage 2) ---
    logger.info("Setting up B6 (SMOTE)...")
    pipe_b6 = TwoStageIDS(config=config, seed=seed)
    pipe_b6.stage1_model = pipe_base.stage1_model
    X_att_smote, y_att_smote, _ = smote_oversample(
        X=X_train_att, y=y_train_att, target_ratio=0.5, k_neighbors=5, config=config, seed=seed
    )
    pipe_b6.fit_stage2(X_att_smote, y_att_smote, use_gan=False)
    for th in thresholds:
        r = evaluate_at_threshold(pipe_b6, X_test, y_test_cat, th, classes)
        r["method"] = "B6 (SMOTE)"
        all_records.append(r)
        
    # --- Model 3: Proposed (WGAN-GP in Stage 2) ---
    logger.info("Setting up Proposed (WGAN-GP)...")
    pipe_wgan = TwoStageIDS(config=config, seed=seed)
    pipe_wgan.stage1_model = pipe_base.stage1_model
    
    generator = Generator(
        latent_dim=config["gan"]["latent_dim"],
        n_classes=len(config["attack_classes"]),
        output_dim=X_train.shape[1],
        hidden_layers=config["gan"]["gen_hidden_layers"],
    )
    ckpt = torch.load("results/models/wgan_generator_best.pt", map_location="cpu", weights_only=False)
    generator.load_state_dict(ckpt)
    generator.eval()
    
    pipe_wgan.fit_stage2(
        X_train_attack=X_train_att,
        y_train_attack=y_train_att,
        use_gan=True,
        generator=generator,
        onehot_groups=onehot_groups,
    )
    for th in thresholds:
        r = evaluate_at_threshold(pipe_wgan, X_test, y_test_cat, th, classes)
        r["method"] = "Proposed (WGAN-GP)"
        all_records.append(r)
        
    # --- Also evaluate Stage 2 ORACLE (Issue B3: Stage 2 given true attacks) ---
    logger.info("Evaluating Stage 2 Oracle (Given True Attacks)...")
    normal_labels = ("normal", "benign")
    test_att_mask = ~pd.Series(y_test_cat).str.lower().isin(normal_labels)
    X_test_att = X_test[test_att_mask]
    y_test_att = y_test_cat[test_att_mask]
    attack_classes = config["attack_classes"]
    
    oracle_records = []
    for name, pipe in [("B5 (None)", pipe_b5), ("B6 (SMOTE)", pipe_b6), ("Proposed (WGAN-GP)", pipe_wgan)]:
        s2_encoded = pipe.stage2_model.predict(X_test_att)
        s2_pred = pipe.label_encoder_stage2.inverse_transform(s2_encoded)
        m_s2 = compute_metrics(y_test_att, s2_pred, classes=attack_classes)
        rec = {
            "method": name,
            "accuracy": m_s2["accuracy"],
            "macro_f1": m_s2["f1_macro"],
        }
        for ac in attack_classes:
            rec[f"{ac}_recall"] = m_s2["per_class"][ac]["recall"]
            rec[f"{ac}_precision"] = m_s2["per_class"][ac]["precision"]
            rec[f"{ac}_f1"] = m_s2["per_class"][ac]["f1"]
        oracle_records.append(rec)
        
    df_results = pd.DataFrame(all_records)
    df_results.to_csv("results/tables/threshold_sweep_test.csv", index=False)
    logger.info("Saved results/tables/threshold_sweep_test.csv")
    
    df_oracle = pd.DataFrame(oracle_records)
    df_oracle.to_csv("results/tables/stage2_oracle_test.csv", index=False)
    logger.info("Saved results/tables/stage2_oracle_test.csv")
    
    # Print summary
    print("\n=== THRESHOLD SWEEP ON TEST SET ===")
    cols_to_show = ["method", "threshold", "macro_f1", "test_fpr", "test_attack_recall", "R2L_recall", "U2R_recall"]
    print(df_results[cols_to_show].to_string())
    
    print("\n=== STAGE 2 ORACLE (PERFECT STAGE 1) ===")
    print(df_oracle.to_string())

if __name__ == "__main__":
    main()

