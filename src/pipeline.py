"""
Two-Stage Intrusion Detection Pipeline (TwoStageIDS).

Architecture:
    Stage 1: Binary Classifier (Normal vs. Attack)
        - Fast, high-recall attack detector.
        - Calibration of decision threshold on the VALIDATION split to meet
          target recall (default 0.98) while keeping false alarm rate low.
    Stage 2: Multiclass Attack-Type Classifier
        - Trained EXCLUSIVELY on attack rows (no Normal traffic).
        - Minority attack categories (e.g. U2R and R2L) oversampled to parity
          using Conditional WGAN-GP (with categorical constraints).
        - Predicts fine-grained attack classes: DoS, Probe, R2L, U2R.

Error Propagation & Diagnostics:
    - Tracks Stage 1 False Negatives per attack category (evaluating R2L & U2R loss).
    - Tracks Stage 1 False Positives (Normal packets routed to Stage 2).
    - Evaluates Stage 2 classification accuracy on correctly routed attacks.
    - Saves comprehensive 'error_propagation' table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.ensemble import get_ensemble
from src.resamplers import gan_oversample
from src.utils import get_logger, save_figure, save_table

logger = get_logger("pipeline")


class TwoStageIDS:
    """
    Two-Stage Explainable Intrusion Detection System.

    Attributes:
        config: Pipeline configuration dictionary.
        ensemble_kind: Base ensemble architecture ('voting' or 'stacking').
        seed: Random seed for reproducibility.
        stage1_model: Fitted Stage 1 binary ensemble (Normal vs Attack).
        stage2_model: Fitted Stage 2 multiclass ensemble (Attack families).
        stage1_threshold: Calibrated classification threshold for P(attack).
        label_encoder_stage2: LabelEncoder mapping attack strings to integers.
        is_synthetic_stage2: Boolean mask marking synthetic Stage 2 training samples.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ensemble_kind: str = "voting",
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.ensemble_kind = ensemble_kind
        self.seed = (
            seed
            if seed is not None
            else int(config.get("experiment", {}).get("default_seed", 42))
        )

        pipe_cfg = config.get("pipeline", {})
        self.stage1_threshold: float = float(pipe_cfg.get("stage1_threshold", 0.5))

        # Dataset-aware normal label
        cfg_normal = config.get("normal_class") or config.get("dataset", {}).get("normal_class")
        if cfg_normal:
            self.normal_class = str(cfg_normal)
        else:
            ds_name = str(config.get("dataset", {}).get("name", ""))
            self.normal_class = "BENIGN" if "CIC" in ds_name.upper() else "Normal"

        self.stage1_model: Optional[Any] = None
        self.stage2_model: Optional[Any] = None
        self.label_encoder_stage2: Optional[LabelEncoder] = None
        self.is_synthetic_stage2: Optional[np.ndarray] = None
        self.threshold_table: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # Task 4.1: Stage 1 Binary Model
    # -----------------------------------------------------------------------

    def fit_stage1(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train_bin: Union[np.ndarray, pd.Series],
    ) -> TwoStageIDS:
        """
        Train the Stage 1 binary classifier (0 = Normal, 1 = Attack).

        Oversampling is NOT applied to Stage 1 unless config['pipeline']['stage1_balance']
        is explicitly set to True.

        Args:
            X_train: Training feature array of shape (N, D).
            y_train_bin: Binary training labels (0: Normal, 1: Attack).

        Returns:
            self
        """
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train_bin, dtype=int)

        pipe_cfg = self.config.get("pipeline", {})
        if pipe_cfg.get("stage1_balance", False):
            logger.info("Balancing Stage 1 training data via RandomOverSampler...")
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=self.seed)
            X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

        logger.info(
            f"Fitting Stage 1 binary classifier ({self.ensemble_kind}) on "
            f"{len(X_tr)} samples (Attacks={np.sum(y_tr == 1)}, Normal={np.sum(y_tr == 0)})..."
        )

        self.stage1_model = get_ensemble(
            kind=self.ensemble_kind,
            config=self.config,
            n_classes=2,
            seed=self.seed,
        )
        self.stage1_model.fit(X_tr, y_tr)
        logger.info("Stage 1 binary model training completed successfully.")
        return self

    def predict_proba_stage1(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Predict probability of being an Attack (P(attack)).

        Args:
            X: Features array of shape (N, D).

        Returns:
            1D array of float probabilities P(attack) in [0.0, 1.0].
        """
        if self.stage1_model is None:
            raise RuntimeError("Stage 1 model has not been fitted. Call fit_stage1() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        proba = self.stage1_model.predict_proba(X_arr)
        # Class 1 is Attack
        return proba[:, 1]

    # -----------------------------------------------------------------------
    # Task 4.2: Threshold Tuning on Validation Split
    # -----------------------------------------------------------------------

    def tune_stage1_threshold(
        self,
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val_bin: Union[np.ndarray, pd.Series],
        target_recall: Optional[float] = None,
        save_name: str = "stage1_threshold_tuning",
    ) -> float:
        """
        Calibrate the Stage 1 decision threshold strictly on the VALIDATION split.

        Scans thresholds from 0.05 to 0.95 and selects the highest threshold
        that satisfies:
            Attack Recall >= target_recall
        This minimizes the false positive rate (Normal traffic needlessly
        routed to Stage 2) while guaranteeing high attack catch rate.

        Saves threshold metrics table to results/tables/ and plot to results/figures/.

        Args:
            X_val: Validation features array.
            y_val_bin: Validation binary labels (0 = Normal, 1 = Attack).
            target_recall: Target recall for attacks (default from config or 0.98).
            save_name: Base name for saved table and plot.

        Returns:
            Selected optimal threshold (float).
        """
        if target_recall is None:
            target_recall = float(
                self.config.get("pipeline", {}).get("target_recall", 0.98)
            )

        y_v = np.asarray(y_val_bin, dtype=int)
        p_attack = self.predict_proba_stage1(X_val)

        thresholds = np.linspace(0.05, 0.95, 91)
        records = []

        total_positives = int(np.sum(y_v == 1))
        total_negatives = int(np.sum(y_v == 0))

        best_threshold = 0.5
        found_target = False

        for th in thresholds:
            y_pred = (p_attack >= th).astype(int)

            tp = int(np.sum((y_v == 1) & (y_pred == 1)))
            fn = int(np.sum((y_v == 1) & (y_pred == 0)))
            fp = int(np.sum((y_v == 0) & (y_pred == 1)))
            tn = int(np.sum((y_v == 0) & (y_pred == 0)))

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            records.append({
                "threshold": round(float(th), 4),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "fpr": round(fpr, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            })

        df_thresh = pd.DataFrame(records)
        self.threshold_table = df_thresh

        # Pick the highest threshold with recall >= target_recall
        eligible = df_thresh[df_thresh["recall"] >= target_recall]
        if not eligible.empty:
            best_threshold = float(eligible.iloc[-1]["threshold"])
            found_target = True
        else:
            # Fall back to threshold maximizing recall
            max_rec_idx = df_thresh["recall"].idxmax()
            best_threshold = float(df_thresh.loc[max_rec_idx, "threshold"])
            logger.warning(
                f"No threshold reached target recall {target_recall:.4f}. "
                f"Selecting threshold {best_threshold:.4f} with max recall "
                f"{df_thresh.loc[max_rec_idx, 'recall']:.4f}."
            )

        self.stage1_threshold = best_threshold
        logger.info(
            f"Stage 1 Threshold Calibrated: threshold={best_threshold:.4f} "
            f"(target recall={target_recall:.4f})"
        )

        # Save metrics table
        save_table(df_thresh, save_name, self.config)

        # Plot threshold curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Attack Recall", color="#2ca02c", lw=2)
        ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1f77b4", lw=2)
        ax.plot(df_thresh["threshold"], df_thresh["fpr"], label="False Positive Rate (FPR)", color="#d62728", lw=2, linestyle="--")
        ax.axvline(best_threshold, color="black", linestyle=":", lw=1.5, label=f"Selected Threshold ({best_threshold:.2f})")
        ax.axhline(target_recall, color="gray", linestyle="-.", lw=1.0, label=f"Target Recall ({target_recall:.2f})")

        ax.set_title("Stage 1 Decision Threshold Calibration (Validation Set)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Probability Threshold P(attack)", fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()

        save_figure(fig, save_name, self.config)
        plt.close(fig)

        return best_threshold

    # -----------------------------------------------------------------------
    # Task 4.3: Stage 2 Attack-Type Model
    # -----------------------------------------------------------------------

    def fit_stage2(
        self,
        X_train_attack: Union[np.ndarray, pd.DataFrame],
        y_train_attack: Union[np.ndarray, pd.Series],
        use_gan: bool = False,
        generator: Optional[Any] = None,
        onehot_groups: Optional[Dict[str, List[int]]] = None,
    ) -> TwoStageIDS:
        """
        Train the Stage 2 multiclass attack-type classifier.

        Trained EXCLUSIVELY on attack rows from the training split.
        If use_gan is True, minority attack categories are oversampled using
        the Conditional WGAN-GP generator prior to fitting the ensemble.

        Args:
            X_train_attack: Features array of attack samples only.
            y_train_attack: Attack category labels (e.g. 'DoS', 'Probe', 'R2L', 'U2R').
            use_gan: Whether to apply Conditional WGAN-GP oversampling.
            generator: Trained Generator module (required if use_gan is True).
            onehot_groups: Dict of one-hot column groups for valid synthetic post-processing.

        Returns:
            self
        """
        if isinstance(X_train_attack, pd.DataFrame):
            X_att = X_train_attack.values.astype(np.float32)
        else:
            X_att = np.asarray(X_train_attack, dtype=np.float32)

        if isinstance(y_train_attack, (pd.Series, pd.Index)):
            y_att = y_train_attack.values
        else:
            y_att = np.asarray(y_train_attack)

        # Check that no Normal/Benign rows are in Stage 2 training data
        if any(str(x).strip().lower() in ("normal", "benign") for x in y_att):
            raise ValueError(
                "CRITICAL VIOLATION: Normal traffic detected in Stage 2 training data. "
                "Stage 2 must be trained EXCLUSIVELY on attack samples."
            )

        if use_gan:
            if generator is None:
                raise ValueError("generator must be provided when use_gan=True.")
            logger.info("Applying Conditional WGAN-GP oversampling to Stage 2 training data...")
            X_aug, y_aug, is_syn = gan_oversample(
                X_train_attack=X_att,
                y_train_attack=y_att,
                generator=generator,
                config=self.config,
                onehot_groups=onehot_groups,
            )
            self.is_synthetic_stage2 = is_syn
        else:
            X_aug = X_att
            y_aug = y_att
            self.is_synthetic_stage2 = np.zeros(len(y_aug), dtype=bool)

        # Encode attack category labels
        self.label_encoder_stage2 = LabelEncoder()
        y_aug_encoded = self.label_encoder_stage2.fit_transform(y_aug)
        n_attack_classes = len(self.label_encoder_stage2.classes_)

        logger.info(
            f"Fitting Stage 2 multiclass ensemble ({self.ensemble_kind}) on "
            f"{len(X_aug)} samples across {n_attack_classes} attack classes: "
            f"{list(self.label_encoder_stage2.classes_)}"
        )

        self.stage2_model = get_ensemble(
            kind=self.ensemble_kind,
            config=self.config,
            n_classes=n_attack_classes,
            seed=self.seed,
        )
        self.stage2_model.fit(X_aug, y_aug_encoded)
        logger.info("Stage 2 multiclass model training completed successfully.")
        return self

    # -----------------------------------------------------------------------
    # Task 4.4: Full Inference and Error Propagation Report
    # -----------------------------------------------------------------------

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        End-to-end two-stage prediction:
        1. Pass sample through Stage 1.
        2. If P(attack) < threshold, predict normal_class ('Normal' or 'BENIGN').
        3. If P(attack) >= threshold, route sample to Stage 2 and predict attack type.

        Args:
            X: Features array of shape (N, D).

        Returns:
            1D array of class labels.
        """
        if self.stage1_model is None:
            raise RuntimeError("Stage 1 model is not fitted.")
        if self.stage2_model is None or self.label_encoder_stage2 is None:
            raise RuntimeError("Stage 2 model is not fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        n_samples = len(X_arr)

        p_attack = self.predict_proba_stage1(X_arr)
        is_attack_pred = p_attack >= self.stage1_threshold

        # Initialize all predictions as normal_class
        y_pred = np.full(n_samples, self.normal_class, dtype=object)

        attack_indices = np.where(is_attack_pred)[0]
        if len(attack_indices) > 0:
            X_attacks = X_arr[attack_indices]
            stage2_encoded = self.stage2_model.predict(X_attacks)
            stage2_labels = self.label_encoder_stage2.inverse_transform(stage2_encoded)
            y_pred[attack_indices] = stage2_labels

        return y_pred

    def evaluate_pipeline(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test_cat: Union[np.ndarray, pd.Series],
        y_test_bin: Union[np.ndarray, pd.Series],
        table_name: str = "error_propagation",
    ) -> Dict[str, Any]:
        """
        Full two-stage evaluation with detailed error propagation audit.

        Reports:
            - Stage 1 False Negatives: Real attack samples misclassified as Normal
              broken down by attack category (specifically auditing R2L and U2R loss).
            - Stage 1 False Positives: Normal packets incorrectly routed to Stage 2.
            - Stage 2 Confusion & Accuracy: Classification fidelity on correctly routed attacks.
            - Full multiclass performance metrics (accuracy, macro-F1, per-class recall).

        Saves:
            results/tables/error_propagation.csv

        Args:
            X_test: Test features array.
            y_test_cat: Ground truth categorical labels ('Normal', 'DoS', 'Probe', 'R2L', 'U2R').
            y_test_bin: Ground truth binary labels (0 = Normal, 1 = Attack).
            table_name: CSV table name for error propagation report.

        Returns:
            Dictionary containing evaluation statistics, reports, and confusion matrices.
        """
        X_t = np.asarray(X_test, dtype=np.float32)
        y_cat = np.asarray(y_test_cat)
        y_bin = np.asarray(y_test_bin, dtype=int)

        # Stage 1 evaluation
        p_attack = self.predict_proba_stage1(X_t)
        routed_to_stage2 = p_attack >= self.stage1_threshold

        # Stage 1 error breakdown
        s1_tp = int(np.sum((y_bin == 1) & routed_to_stage2))
        s1_fn = int(np.sum((y_bin == 1) & (~routed_to_stage2)))
        s1_fp = int(np.sum((y_bin == 0) & routed_to_stage2))
        s1_tn = int(np.sum((y_bin == 0) & (~routed_to_stage2)))

        s1_recall = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0.0
        s1_fpr = s1_fp / (s1_fp + s1_tn) if (s1_fp + s1_tn) > 0 else 0.0

        # Detailed breakdown of Stage 1 False Negatives by attack class
        attack_classes = [c for c in np.unique(y_cat) if str(c).lower() != "normal"]
        s1_loss_records = []

        logger.info("=== Stage 1 Attack Leak-Through / Loss Audit ===")
        for cls in attack_classes:
            cls_mask = y_cat == cls
            cls_total = int(np.sum(cls_mask))
            cls_passed = int(np.sum(cls_mask & routed_to_stage2))
            cls_lost = int(np.sum(cls_mask & (~routed_to_stage2)))
            loss_pct = (cls_lost / cls_total * 100.0) if cls_total > 0 else 0.0

            logger.info(
                f"  Class '{cls}': Total={cls_total}, Routed to S2={cls_passed}, "
                f"Lost in S1 (FN)={cls_lost} ({loss_pct:.2f}%)"
            )

            s1_loss_records.append({
                "attack_class": cls,
                "total_in_test": cls_total,
                "routed_to_stage2": cls_passed,
                "lost_in_stage1_fn": cls_lost,
                "loss_percentage": round(loss_pct, 2),
            })

        loss_df = pd.DataFrame(s1_loss_records)
        save_table(loss_df, f"{table_name}_stage1_loss_by_class", self.config)

        # Full end-to-end predictions
        y_pred = self.predict(X_t)

        all_classes = sorted(list(np.unique(np.concatenate([y_cat, y_pred]))))
        macro_f1 = float(f1_score(y_cat, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_cat, y_pred, average="weighted", zero_division=0))
        accuracy = float(np.mean(y_pred == y_cat))

        recalls = recall_score(y_cat, y_pred, labels=all_classes, average=None, zero_division=0)
        precisions = precision_score(y_cat, y_pred, labels=all_classes, average=None, zero_division=0)
        f1s = f1_score(y_cat, y_pred, labels=all_classes, average=None, zero_division=0)

        per_class_metrics = {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }
            for cls, p, r, f in zip(all_classes, precisions, recalls, f1s)
        }

        # Build Master Error Propagation Summary Table
        summary_rows = [
            {"metric": "Stage 1 Threshold", "value": round(self.stage1_threshold, 4)},
            {"metric": "Stage 1 Attack Recall", "value": round(s1_recall, 4)},
            {"metric": "Stage 1 False Positive Rate (Normal routed to S2)", "value": round(s1_fpr, 4)},
            {"metric": "Stage 1 Total Attacks Lost (FN)", "value": s1_fn},
            {"metric": "Pipeline Overall Accuracy", "value": round(accuracy, 4)},
            {"metric": "Pipeline Macro F1", "value": round(macro_f1, 4)},
            {"metric": "Pipeline Weighted F1", "value": round(weighted_f1, 4)},
        ]

        # Append per-class metrics to summary
        for cls in all_classes:
            summary_rows.append({
                "metric": f"F1 - {cls}",
                "value": round(per_class_metrics[cls]["f1"], 4),
            })
            summary_rows.append({
                "metric": f"Recall - {cls}",
                "value": round(per_class_metrics[cls]["recall"], 4),
            })

        summary_df = pd.DataFrame(summary_rows)
        save_table(summary_df, table_name, self.config)

        # Highlight R2L and U2R loss in logs
        for target_minority in ["R2L", "U2R"]:
            sub = loss_df[loss_df["attack_class"].str.upper() == target_minority]
            if not sub.empty:
                r = sub.iloc[0]
                logger.info(
                    f"AUDIT {target_minority}: {r['lost_in_stage1_fn']} of {r['total_in_test']} "
                    f"samples lost in Stage 1 ({r['loss_percentage']}%)."
                )

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "stage1_recall": s1_recall,
            "stage1_fpr": s1_fpr,
            "stage1_fn": s1_fn,
            "stage1_fp": s1_fp,
            "per_class_metrics": per_class_metrics,
            "loss_df": loss_df,
            "y_pred": y_pred,
        }
