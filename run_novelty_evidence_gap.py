"""
Script to compute:
1. Monolithic SHAP baseline (Issue 6 / Claim 2 comparison)
2. Mode diversity / intra-class variance for WGAN-GP vs Flat GAN (Issue 7 / Claim 3)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import shap
import torch

from src.preprocess import build_dataset, load_nslkdd, map_attack_categories
from src.baselines import run_single_stage
from src.shap_fidelity import get_mean_abs_shap
from src.utils import get_logger, load_config, set_seed
from src.wgan_gp import Generator, generate_samples

logger = get_logger("novelty_evidence_gap")

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
    
    X_val = dataset["X_val"]
    y_val_cat = dataset["y_val_cat"]
    classes = sorted(list(np.unique(y_val_cat)))
    
    # -----------------------------------------------------------------------
    # Part 1: Monolithic SHAP Baseline
    # -----------------------------------------------------------------------
    logger.info("Training Flat Monolithic 5-Class Ensemble (B1)...")
    res_b1 = run_single_stage("none", dataset, config, seed=seed)
    flat_model = res_b1["model"]
    feature_names = dataset.get("feature_names", None)
    
    # Sample 150 instances
    rng = np.random.RandomState(seed)
    sub_idx = rng.choice(len(X_val), size=150, replace=False)
    X_eval = X_val[sub_idx]
    
    # Compute mean-abs-SHAP for each class in the monolithic model
    flat_importances = {}
    for idx, cls in enumerate(classes):
        logger.info(f"Computing monolithic SHAP for class '{cls}'...")
        v = get_mean_abs_shap(flat_model, X_eval, class_index=idx, max_samples=150, feature_names=feature_names, seed=seed)
        flat_importances[cls] = v
        
    # Measure correlation between Normal and each Attack class in monolithic model
    v_norm_flat = flat_importances["Normal"]
    monolithic_corrs = []
    for cls in ["DoS", "Probe", "R2L", "U2R"]:
        v_att = flat_importances[cls]
        sp_corr = float(stats.spearmanr(v_norm_flat.values, v_att.values).statistic)
        # Top-10 Jaccard overlap
        s1_top = set(v_norm_flat.nlargest(10).index)
        s2_top = set(v_att.nlargest(10).index)
        jaccard = len(s1_top & s2_top) / len(s1_top | s2_top)
        monolithic_corrs.append({
            "attack_class": cls,
            "monolithic_spearman": round(sp_corr, 4),
            "monolithic_top10_jaccard": round(jaccard, 4),
        })
        
    df_mono = pd.DataFrame(monolithic_corrs)
    df_mono.to_csv("results/tables/monolithic_shap_comparison.csv", index=False)
    logger.info(f"Monolithic SHAP results:\n{df_mono.to_string()}")
    
    # -----------------------------------------------------------------------
    # Part 2: Mode Diversity / Intra-Class Variance (Claim 3)
    # -----------------------------------------------------------------------
    logger.info("Computing Mode Diversity (Intra-Class Variance)...")
    # Load Attack-Only generator
    n_features = dataset["X_train"].shape[1]
    gen_att = Generator(
        latent_dim=config["gan"]["latent_dim"],
        n_classes=4,
        output_dim=n_features,
        hidden_layers=config["gan"]["gen_hidden_layers"],
    )
    ckpt = torch.load("results/models/wgan_generator_best.pt", map_location="cpu", weights_only=False)
    gen_att.load_state_dict(ckpt)
    gen_att.eval()
    
    onehot_groups = dataset.get("onehot_column_indices", {})
    
    # Measure intra-class variance on real attack samples vs synthetic samples
    diversity_records = []
    attack_classes = ["DoS", "Probe", "R2L", "U2R"]
    X_tr = dataset["X_train"]
    y_tr_cat = dataset["y_train_cat"]
    
    for cls_id, cls in enumerate(attack_classes):
        X_real_cls = X_tr[y_tr_cat == cls]
        real_var = float(np.mean(np.var(X_real_cls, axis=0)))
        
        # Synthetic from Attack-Only GAN
        syn_samples, _ = generate_samples(gen_att, class_id=cls_id, n=200, config=config, onehot_groups=onehot_groups, device="cpu")
        syn_var = float(np.mean(np.var(syn_samples, axis=0)))
        
        # Ratio of diversity captured
        var_ratio = syn_var / max(real_var, 1e-6)
        diversity_records.append({
            "class": cls,
            "real_intra_class_var": round(real_var, 4),
            "synthetic_intra_class_var": round(syn_var, 4),
            "diversity_coverage_ratio": round(var_ratio, 4),
        })
        
    df_div = pd.DataFrame(diversity_records)
    df_div.to_csv("results/tables/gan_mode_diversity.csv", index=False)
    logger.info(f"Mode Diversity results:\n{df_div.to_string()}")

if __name__ == "__main__":
    main()

