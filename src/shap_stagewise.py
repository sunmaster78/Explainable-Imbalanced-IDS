"""
Stage-Wise Explainability and Local Attribution Module (Novelty Claim 2).

Deconstructs IDS decision-making into two decoupled stages:
1. Stage 1: Macro-level attack detection (Normal vs Anomaly/Attack)
2. Stage 2: Micro-level attack differentiation (DoS vs Probe vs R2L vs U2R)

Demonstrates that stage-wise SHAP attribution provides domain-consistent insights
without conflating attack family characteristics with normalcy boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import shap

from src.shap_fidelity import _extract_base_models, get_mean_abs_shap
from src.utils import get_logger, save_figure, save_table, set_seed

logger = get_logger("shap_stagewise")


# ---------------------------------------------------------------------------
# Helper: Ensemble Explanation Builder
# ---------------------------------------------------------------------------

def _build_ensemble_explanation(
    ensemble: Any,
    X: np.ndarray,
    class_index: int = 1,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """
    Build a unified shap.Explanation object from an ensemble of tree models.
    Averages SHAP values and expected values across RandomForest, XGBoost, and LightGBM.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X_arr.shape
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    base_models = _extract_base_models(ensemble)
    all_shap_values = []
    all_base_values = []

    for model in base_models:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_arr)

            # Extract base value / expected value
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                if class_index < len(ev):
                    base_val = float(ev[class_index])
                else:
                    base_val = float(ev[0])
            elif ev is not None:
                base_val = float(ev)
            else:
                base_val = 0.0

            # Extract class-specific SHAP values
            if isinstance(sv, list):
                if class_index < len(sv):
                    class_sv = sv[class_index]
                elif len(sv) == 1:
                    class_sv = sv[0] if class_index == 1 else -sv[0]
                else:
                    class_sv = sv[0]
            elif isinstance(sv, np.ndarray):
                if sv.ndim == 3:
                    if sv.shape[-1] > class_index:
                        class_sv = sv[:, :, class_index]
                    elif sv.shape[1] > class_index:
                        class_sv = sv[:, class_index, :]
                    else:
                        class_sv = sv[:, :, 0]
                elif sv.ndim == 2:
                    class_sv = sv if class_index == 1 else -sv
                else:
                    class_sv = sv
            else:
                continue

            all_shap_values.append(class_sv)
            all_base_values.append(base_val)

        except Exception as e:
            logger.warning(f"Could not extract explanation for {type(model).__name__}: {e}")
            continue

    if not all_shap_values:
        # Fallback explanation if explainers failed
        avg_sv = np.zeros((n_samples, n_features), dtype=np.float32)
        avg_bv = np.zeros(n_samples, dtype=np.float32)
    else:
        avg_sv = np.mean(all_shap_values, axis=0)
        avg_bv = np.full(n_samples, float(np.mean(all_base_values)))

    return shap.Explanation(
        values=avg_sv,
        base_values=avg_bv,
        data=X_arr,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Task 8.3: Stage-Wise SHAP Audit (Claim 2)
# ---------------------------------------------------------------------------

def run_stagewise(
    pipeline: Any,
    data: Dict[str, Any],
    config: Dict[str, Any],
    max_samples: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute Novelty Claim 2: Stage-Wise Explainability Audit.

    1. Stage 1: SHAP beeswarm summary plot (top 15 features) for Normal vs Attack.
    2. Stage 2: SHAP summary bar plots for each attack family.
    3. Table 'stagewise_top_features': Top-10 features for Stage 1 and Stage 2 classes,
       with categorical flags: 'Stage1_Only', 'Stage2_Only', 'Both'.
    4. Table 'stage_importance_overlap': Spearman correlation between Stage 1 importance
       and each Stage 2 class importance vector.

    Args:
        pipeline: Fitted TwoStageIDS instance.
        data: Preprocessed dataset dictionary.
        config: Configuration dictionary.
        max_samples: Maximum samples per evaluation for computational efficiency.
        seed: Random seed.

    Returns:
        Dict containing top features DataFrame and overlap correlation DataFrame.
    """
    set_seed(seed)
    logger.info("============================================================")
    logger.info("Executing Novelty Claim 2: Stage-Wise SHAP Deconvolution")
    logger.info("============================================================")

    feature_names = data.get("feature_names", None)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val_cat = np.asarray(data["y_val_cat"])
    y_val_bin = np.asarray(data["y_val_bin"])

    rng = np.random.RandomState(seed)
    n_val = len(X_val)
    if n_val > max_samples:
        sub_idx = rng.choice(n_val, size=max_samples, replace=False)
        X_eval = X_val[sub_idx]
        y_cat_eval = y_val_cat[sub_idx]
    else:
        X_eval = X_val
        y_cat_eval = y_val_cat

    # 1. Stage 1 Explanation & Beeswarm Plot
    logger.info("Generating Stage 1 SHAP explanation (Normal vs Attack)...")
    exp_stage1 = _build_ensemble_explanation(
        pipeline.stage1_model,
        X_eval,
        class_index=1,
        feature_names=feature_names,
    )

    fig_s1 = plt.figure(figsize=(10, 7), dpi=300)
    shap.plots.beeswarm(exp_stage1, max_display=15, show=False)
    plt.title("Stage 1 Decision Boundary: Normal vs Attack (Ensemble SHAP)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    s1_plot_path = save_figure(fig_s1, "shap_stage1_beeswarm", config=config, dpi=300)
    plt.close(fig_s1)
    logger.info(f"Stage 1 beeswarm plot saved to: {s1_plot_path}")

    # Stage 1 Mean-Abs-SHAP importance vector
    v_stage1 = get_mean_abs_shap(
        pipeline.stage1_model,
        X_eval,
        class_index=1,
        max_samples=max_samples,
        feature_names=feature_names,
        seed=seed,
    )

    # 2. Stage 2 Explanations & Bar Plots per Attack Class
    le_stage2 = pipeline.label_encoder_stage2
    if le_stage2 is None:
        raise RuntimeError("Pipeline Stage 2 LabelEncoder is not available.")

    stage2_classes = list(le_stage2.classes_)
    stage2_importances: Dict[str, pd.Series] = {}

    for cls in stage2_classes:
        cls_idx = int(le_stage2.transform([cls])[0])
        # Find rows belonging to this attack class
        cls_mask = y_cat_eval == cls
        if np.sum(cls_mask) >= 5:
            X_cls = X_eval[cls_mask]
        else:
            # Fall back to training attack split for rare classes
            X_tr = np.asarray(data["X_train"], dtype=np.float32)
            y_tr_cat = np.asarray(data["y_train_cat"])
            X_cls = X_tr[y_tr_cat == cls]
            if len(X_cls) > max_samples:
                X_cls = X_cls[:max_samples]

        if len(X_cls) == 0:
            continue

        exp_s2_cls = _build_ensemble_explanation(
            pipeline.stage2_model,
            X_cls,
            class_index=cls_idx,
            feature_names=feature_names,
        )

        fig_s2 = plt.figure(figsize=(9, 5), dpi=300)
        shap.plots.bar(exp_s2_cls, max_display=15, show=False)
        plt.title(f"Stage 2 Attack Attribution: {cls} (Ensemble SHAP)", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        clean_name = cls.lower().replace(" ", "_").replace("-", "_")
        s2_plot_path = save_figure(fig_s2, f"shap_stage2_bar_{clean_name}", config=config, dpi=300)
        plt.close(fig_s2)

        v_s2_cls = get_mean_abs_shap(
            pipeline.stage2_model,
            X_cls,
            class_index=cls_idx,
            max_samples=max_samples,
            feature_names=feature_names,
            seed=seed,
        )
        stage2_importances[cls] = v_s2_cls

    # 3. Construct Table: 'stagewise_top_features'
    top_k = min(10, len(v_stage1))
    top10_stage1 = set(v_stage1.nlargest(top_k).index)
    top_records = []

    s1_top = v_stage1.nlargest(top_k)
    for rank in range(1, top_k + 1):
        rec: Dict[str, Any] = {"rank": rank}
        rec["Stage1_Feature"] = s1_top.index[rank - 1]
        rec["Stage1_SHAP"] = round(float(s1_top.iloc[rank - 1]), 4)

        for cls in stage2_classes:
            if cls in stage2_importances:
                ser = stage2_importances[cls]
                s2_top = ser.nlargest(top_k)
                if rank - 1 < len(s2_top):
                    rec[f"{cls}_Feature"] = s2_top.index[rank - 1]
                    rec[f"{cls}_SHAP"] = round(float(s2_top.iloc[rank - 1]), 4)
                else:
                    rec[f"{cls}_Feature"] = ""
                    rec[f"{cls}_SHAP"] = 0.0

        top_records.append(rec)

    df_top_features = pd.DataFrame(top_records)

    # Add Categorization Column: Which features are Stage 1 only vs Stage 2 only vs Both
    all_s2_top10 = set()
    for cls, ser in stage2_importances.items():
        all_s2_top10.update(set(ser.nlargest(10).index))

    overlap_records = []
    all_union_features = sorted(list(top10_stage1 | all_s2_top10))
    for feat in all_union_features:
        in_s1 = feat in top10_stage1
        in_s2 = feat in all_s2_top10
        if in_s1 and in_s2:
            status = "Both"
        elif in_s1:
            status = "Stage1_Only"
        else:
            status = "Stage2_Only"

        overlap_records.append({
            "feature": feat,
            "status": status,
            "in_stage1_top10": in_s1,
            "in_stage2_top10": in_s2,
        })
    df_feature_roles = pd.DataFrame(overlap_records)

    save_table(df_top_features, "stagewise_top_features", config=config)
    save_table(df_feature_roles, "stagewise_feature_roles", config=config)
    logger.info("Saved 'stagewise_top_features' and 'stagewise_feature_roles' tables.")

    # 4. Table: 'stage_importance_overlap' (Spearman correlation between Stage 1 & Stage 2)
    overlap_corrs = []
    for cls, v_s2 in stage2_importances.items():
        res = stats.spearmanr(v_stage1.values, v_s2.values)
        sp_corr = float(res.correlation if hasattr(res, "correlation") else res.statistic)
        p_val = float(res.pvalue if hasattr(res, "pvalue") else 1.0)

        # Top-10 Jaccard overlap
        s1_top = set(v_stage1.nlargest(10).index)
        s2_top = set(v_s2.nlargest(10).index)
        jaccard = len(s1_top & s2_top) / len(s1_top | s2_top) if len(s1_top | s2_top) > 0 else 0.0

        overlap_corrs.append({
            "stage2_class": cls,
            "spearman_correlation": round(sp_corr, 4),
            "p_value": round(p_val, 6),
            "top10_jaccard_overlap": round(jaccard, 4),
        })

    df_overlap = pd.DataFrame(overlap_corrs)
    save_table(df_overlap, "stage_importance_overlap", config=config)
    logger.info("Saved 'stage_importance_overlap' table.")

    return {
        "top_features": df_top_features,
        "feature_roles": df_feature_roles,
        "stage_importance_overlap": df_overlap,
    }


# ---------------------------------------------------------------------------
# Task 8.4: Local Instance Explanations (Waterfall Plots)
# ---------------------------------------------------------------------------

def explain_sample(
    pipeline: Any,
    x: Union[np.ndarray, pd.Series],
    config: Dict[str, Any],
    sample_id: str = "sample",
    feature_names: Optional[List[str]] = None,
    true_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate two-stage local explanations with SHAP waterfall plots for an individual sample:
    1. Stage 1 prediction (P(attack)) and Stage 1 SHAP waterfall plot.
    2. If flagged as attack, Stage 2 multiclass prediction and Stage 2 SHAP waterfall plot.
    3. Tabulates top 5 contributing features for each stage.

    Args:
        pipeline: Fitted TwoStageIDS instance.
        x: Single input feature vector (shape (D,) or (1, D)).
        config: Configuration dictionary.
        sample_id: Identifier used in plot filenames (e.g. 'r2l_detected', 'u2r_detected', 'miss_fn').
        feature_names: Optional list of feature names.
        true_label: Optional ground truth class label for annotation.

    Returns:
        Dict containing predictions, paths to saved figures, and top contributing features.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)

    n_features = x_arr.shape[1]
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    # 1. Stage 1 Prediction & Waterfall Plot
    p_attack = float(pipeline.predict_proba_stage1(x_arr)[0])
    is_attack = p_attack >= pipeline.stage1_threshold

    exp_s1 = _build_ensemble_explanation(
        pipeline.stage1_model,
        x_arr,
        class_index=1,
        feature_names=feature_names,
    )

    fig_w1 = plt.figure(figsize=(9, 5), dpi=300)
    shap.plots.waterfall(exp_s1[0], max_display=10, show=False)
    plt.title(
        f"Stage 1 Local Explanation [{sample_id}] (P(Attack)={p_attack:.3f} | Threshold={pipeline.stage1_threshold:.3f})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    w1_path = save_figure(fig_w1, f"local_waterfall_{sample_id}_stage1", config=config, dpi=300)
    plt.close(fig_w1)

    # Top 5 features in Stage 1
    s1_vals = exp_s1.values[0]
    top5_s1_indices = np.argsort(np.abs(s1_vals))[-5:][::-1]
    top5_s1 = [
        {"feature": feature_names[i], "shap_value": round(float(s1_vals[i]), 4), "feature_value": round(float(x_arr[0, i]), 4)}
        for i in top5_s1_indices
    ]

    # 2. Stage 2 (if routed as Attack)
    predicted_class = pipeline.normal_class
    stage2_path = None
    top5_s2 = []

    if is_attack:
        s2_encoded = int(pipeline.stage2_model.predict(x_arr)[0])
        predicted_class = str(pipeline.label_encoder_stage2.inverse_transform([s2_encoded])[0])

        exp_s2 = _build_ensemble_explanation(
            pipeline.stage2_model,
            x_arr,
            class_index=s2_encoded,
            feature_names=feature_names,
        )

        fig_w2 = plt.figure(figsize=(9, 5), dpi=300)
        shap.plots.waterfall(exp_s2[0], max_display=10, show=False)
        plt.title(
            f"Stage 2 Local Explanation [{sample_id}] (Predicted: {predicted_class} | True: {true_label})",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        stage2_path = save_figure(fig_w2, f"local_waterfall_{sample_id}_stage2", config=config, dpi=300)
        plt.close(fig_w2)

        s2_vals = exp_s2.values[0]
        top5_s2_indices = np.argsort(np.abs(s2_vals))[-5:][::-1]
        top5_s2 = [
            {"feature": feature_names[i], "shap_value": round(float(s2_vals[i]), 4), "feature_value": round(float(x_arr[0, i]), 4)}
            for i in top5_s2_indices
        ]

    logger.info(
        f"Sample [{sample_id}] -> True={true_label} | Pred={predicted_class} | "
        f"P(Att)={p_attack:.4f} | S1 Plot={w1_path.name} | S2 Plot={stage2_path.name if stage2_path else 'None'}"
    )

    return {
        "sample_id": sample_id,
        "true_label": true_label,
        "predicted_label": predicted_class,
        "p_attack": p_attack,
        "is_attack_stage1": is_attack,
        "stage1_waterfall": str(w1_path),
        "stage2_waterfall": str(stage2_path) if stage2_path else None,
        "stage1_top5": top5_s1,
        "stage2_top5": top5_s2,
    }


def run_local_explanations_suite(
    pipeline: Any,
    data: Dict[str, Any],
    config: Dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run local explanations suite across 3 canonical representative instances:
    1. Correctly detected R2L sample (True Positive R2L)
    2. Correctly detected U2R sample (True Positive U2R)
    3. Stage 1 miss (False Negative: true attack classified as Normal)
    """
    set_seed(seed)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test_cat = np.asarray(data["y_test_cat"])
    feature_names = data.get("feature_names", None)

    # Get predictions
    y_pred = pipeline.predict(X_test)
    p_attack = pipeline.predict_proba_stage1(X_test)

    cases: List[Tuple[str, Optional[int], str]] = []

    # 1. Correctly detected R2L
    r2l_hits = np.where((y_test_cat == "R2L") & (y_pred == "R2L"))[0]
    if len(r2l_hits) > 0:
        cases.append(("r2l_detected", r2l_hits[0], "R2L"))
    else:
        r2l_all = np.where(y_test_cat == "R2L")[0]
        if len(r2l_all) > 0:
            cases.append(("r2l_detected", r2l_all[0], "R2L"))

    # 2. Correctly detected U2R
    u2r_hits = np.where((y_test_cat == "U2R") & (y_pred == "U2R"))[0]
    if len(u2r_hits) > 0:
        cases.append(("u2r_detected", u2r_hits[0], "U2R"))
    else:
        u2r_all = np.where(y_test_cat == "U2R")[0]
        if len(u2r_all) > 0:
            cases.append(("u2r_detected", u2r_all[0], "U2R"))

    # 3. Stage 1 miss (true attack missed as Normal/BENIGN)
    normal_labels = ("normal", "benign")
    fn_hits = np.where(
        (~pd.Series(y_test_cat).str.lower().isin(normal_labels))
        & (pd.Series(y_pred).str.lower().isin(normal_labels))
    )[0]

    if len(fn_hits) > 0:
        cases.append(("stage1_miss_fn", fn_hits[0], str(y_test_cat[fn_hits[0]])))
    else:
        # Lowest P(attack) attack sample
        attack_indices = np.where(~pd.Series(y_test_cat).str.lower().isin(normal_labels))[0]
        if len(attack_indices) > 0:
            min_p_idx = attack_indices[np.argmin(p_attack[attack_indices])]
            cases.append(("stage1_miss_fn", min_p_idx, str(y_test_cat[min_p_idx])))

    table_rows = []
    for sample_id, idx, true_lbl in cases:
        if idx is None:
            continue
        x_sample = X_test[idx]
        res = explain_sample(
            pipeline=pipeline,
            x=x_sample,
            config=config,
            sample_id=sample_id,
            feature_names=feature_names,
            true_label=true_lbl,
        )

        for rank, item in enumerate(res["stage1_top5"], start=1):
            table_rows.append({
                "case": sample_id,
                "stage": "Stage 1",
                "true_label": true_lbl,
                "pred_label": res["predicted_label"],
                "p_attack": round(res["p_attack"], 4),
                "rank": rank,
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "feature_value": item["feature_value"],
            })

        for rank, item in enumerate(res["stage2_top5"], start=1):
            table_rows.append({
                "case": sample_id,
                "stage": "Stage 2",
                "true_label": true_lbl,
                "pred_label": res["predicted_label"],
                "p_attack": round(res["p_attack"], 4),
                "rank": rank,
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "feature_value": item["feature_value"],
            })

    df_local = pd.DataFrame(table_rows)
    save_table(df_local, "local_explanations_top5", config=config)
    logger.info("Saved 'local_explanations_top5' table.")
    return df_local
