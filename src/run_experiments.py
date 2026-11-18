"""
Experiment Runner CLI for Multi-Seed Benchmarking.

Orchestrates the complete benchmark grid across methods and seeds:
    Single-Stage Baselines:
        - B1: No resampling ('none')
        - B2: Random Oversampling ('ros')
        - B3: SMOTE ('smote')
        - B4: Flat Conditional WGAN-GP across all classes ('gan_all')
    Two-Stage Baselines:
        - B5: Two-Stage + Stage 2 ROS ('ros')
        - B6: Two-Stage + Stage 2 SMOTE ('smote')
        - Proposed: Two-Stage + Stage-Aware Attack Conditional WGAN-GP ('wgan')

Features:
    - Fully resumable: checks results/tables/raw_results.csv and skips completed runs.
    - Leak-free: strictly rebuilds splits and fitted scalers/encoders per seed.
    - Empirical logging: records training time, test inference latency, and all macro/per-class metrics.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path for direct CLI invocation
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.baselines import run_single_stage, run_two_stage
from src.evaluate import compute_metrics
from src.preprocess import (
    build_dataset,
    load_cicids,
    load_nslkdd,
    load_unsw,
    map_attack_categories,
)
from src.utils import ensure_dirs, get_logger, load_config, set_seed

logger = get_logger("run_experiments")

ALL_METHODS = ["B1", "B2", "B3", "B4", "B5", "B6", "Proposed"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run intrusion detection benchmark experiments across methods and seeds."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nslkdd.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        help="List of methods to evaluate: B1, B2, B3, B4, B5, B6, Proposed (or space-separated).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of random seeds to evaluate. If not specified, reads from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. Defaults to results/tables/raw_results.csv.",
    )
    return parser.parse_args()


def load_completed_runs(output_csv: Path) -> Set[Tuple[str, int, str]]:
    """
    Read output CSV to identify already completed (dataset, seed, method) triplets.
    Allows resuming interrupted experiment sweeps without repeating work.
    """
    if not output_csv.exists():
        return set()

    try:
        df = pd.read_csv(output_csv)
        if "seed" in df.columns and "method" in df.columns:
            completed = set(
                zip(
                    df["seed"].astype(int),
                    df["method"].astype(str).str.strip(),
                )
            )
            logger.info(f"Loaded {len(completed)} already completed run(s) from {output_csv}")
            return completed
    except Exception as e:
        logger.warning(f"Could not parse existing results table: {e}")

    return set()


def run_experiment_for_method(
    method_name: str,
    dataset: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[Any, Optional[Any], float, float]:
    """
    Dispatch and execute a single experiment method.

    Returns:
        Tuple of (y_pred, y_proba, training_time, inference_time)
    """
    m = method_name.strip()
    norm_m = m.upper()

    # Single-Stage Baselines
    if norm_m in ("B1", "NONE"):
        res = run_single_stage("none", dataset, config, seed=seed)
        return res["y_pred"], res["y_proba"], res["training_time"], res["inference_time"]

    elif norm_m in ("B2", "ROS"):
        res = run_single_stage("ros", dataset, config, seed=seed)
        return res["y_pred"], res["y_proba"], res["training_time"], res["inference_time"]

    elif norm_m in ("B3", "SMOTE"):
        res = run_single_stage("smote", dataset, config, seed=seed)
        return res["y_pred"], res["y_proba"], res["training_time"], res["inference_time"]

    elif norm_m in ("B4", "GAN_ALL"):
        res = run_single_stage("gan_all", dataset, config, seed=seed)
        return res["y_pred"], res["y_proba"], res["training_time"], res["inference_time"]

    # Two-Stage Baselines & Proposed
    elif norm_m in ("B5", "TWO_STAGE_ROS"):
        res = run_two_stage("ros", dataset, config, seed=seed)
        return res["y_pred"], None, res["training_time"], res["inference_time"]

    elif norm_m in ("B6", "TWO_STAGE_SMOTE"):
        res = run_two_stage("smote", dataset, config, seed=seed)
        return res["y_pred"], None, res["training_time"], res["inference_time"]

    elif norm_m in ("PROPOSED", "WGAN", "TWO_STAGE_WGAN"):
        res = run_two_stage("wgan", dataset, config, seed=seed)
        return res["y_pred"], None, res["training_time"], res["inference_time"]

    else:
        raise ValueError(
            f"Unknown method '{method_name}'. Supported methods: {ALL_METHODS}"
        )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    ensure_dirs(config)

    dataset_name = config.get("dataset", {}).get("name", "NSL-KDD")

    # Resolve output CSV path
    if args.output:
        output_csv = Path(args.output)
    else:
        tables_dir = Path(config.get("output_dirs", {}).get("tables", "results/tables"))
        clean_ds_name = dataset_name.lower().replace("-", "_").replace(" ", "_")
        output_csv = tables_dir / f"raw_results_{clean_ds_name}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resolve seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = config.get("experiment", {}).get("seeds", [0])

    methods = args.methods

    logger.info("============================================================")
    logger.info(f"Starting Benchmark Experiment Sweep")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output File: {output_csv}")
    logger.info("============================================================")

    completed_runs = load_completed_runs(output_csv)

    # Raw dataset loading (dataset-aware)
    logger.info(f"Loading raw dataset for benchmark: {dataset_name}")

    if dataset_name.upper() in ("UNSW", "UNSW-NB15", "UNSW_NB15"):
        raw_train_path = Path(config["dataset"]["raw_train_path"])
        raw_test_path = Path(config["dataset"]["raw_test_path"])
        train_df_mapped, test_df_mapped = load_unsw(raw_train_path, raw_test_path, config=config)
    elif dataset_name.upper() in ("CICIDS", "CIC-IDS2017", "CICIDS2017"):
        raw_folder = Path(config["dataset"]["folder"])
        sample_frac = config.get("dataset", {}).get("sample_fraction", 0.1)
        train_df_mapped, test_df_mapped = load_cicids(
            raw_folder, sample_fraction=sample_frac, config=config, seed=seeds[0] if seeds else 42
        )
    else:
        raw_train_path = Path(config["dataset"]["raw_train_path"])
        raw_test_path = Path(config["dataset"]["raw_test_path"])
        logger.info(f"Loading NSL-KDD from {raw_train_path} and {raw_test_path}...")
        train_df_raw, test_df_raw = load_nslkdd(raw_train_path, raw_test_path)
        train_df_mapped = map_attack_categories(train_df_raw)
        test_df_mapped = map_attack_categories(test_df_raw)

    # Derive classes from config or dynamically from dataset
    classes = config.get("classes", None)
    if classes is None:
        if "category" in train_df_mapped.columns:
            cats = list(train_df_mapped["category"].unique())
            norm = [c for c in cats if str(c).lower() in ("normal", "benign")]
            attacks = [c for c in cats if str(c).lower() not in ("normal", "benign")]
            classes = norm + sorted(attacks)
        else:
            classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    logger.info(f"Evaluating {len(classes)} classes: {classes}")

    total_runs = len(seeds) * len(methods)
    current_run_idx = 0

    for seed in seeds:
        logger.info(f"\n>>>>>>>>>>>>>>> Processing Seed: {seed} <<<<<<<<<<<<<<<")
        set_seed(seed)

        # Build seed-specific split and preprocessing pipeline
        seed_cfg = copy.deepcopy(config)
        seed_cfg["seed"] = seed
        dataset = build_dataset(train_df_mapped, test_df_mapped, seed_cfg)

        for method in methods:
            current_run_idx += 1
            pair = (str(dataset_name).strip(), int(seed), str(method).strip())

            # Resumable skip check
            if pair in completed_runs:
                logger.info(
                    f"[{current_run_idx}/{total_runs}] Skipping ({dataset_name}, Seed={seed}, Method={method}) - already in {output_csv.name}"
                )
                continue

            logger.info(
                f"[{current_run_idx}/{total_runs}] Executing {dataset_name}: Seed={seed}, Method={method}..."
            )

            try:
                y_pred, y_proba, train_time, infer_time = run_experiment_for_method(
                    method_name=method,
                    dataset=dataset,
                    config=seed_cfg,
                    seed=seed,
                )

                # Compute comprehensive metrics
                metrics = compute_metrics(
                    y_true=dataset["y_test_cat"],
                    y_pred=y_pred,
                    y_proba=y_proba,
                    classes=classes,
                )

                n_test = len(dataset["X_test"])
                infer_per_sample_ms = (infer_time / n_test) * 1000.0 if n_test > 0 else 0.0

                row: Dict[str, Any] = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "method": method,
                    "training_time_sec": round(train_time, 4),
                    "inference_time_sec": round(infer_time, 4),
                    "inference_time_per_sample_ms": round(infer_per_sample_ms, 4),
                    "accuracy": round(metrics["accuracy"], 5),
                    "precision_macro": round(metrics["precision_macro"], 5),
                    "recall_macro": round(metrics["recall_macro"], 5),
                    "f1_macro": round(metrics["f1_macro"], 5),
                    "f1_weighted": round(metrics["f1_weighted"], 5),
                    "mcc": round(metrics["mcc"], 5),
                    "auc_pr_macro": round(metrics["auc_pr_macro"], 5) if not pd.isna(metrics["auc_pr_macro"]) else "",
                    "fpr_normal": round(metrics["fpr_normal"], 5) if not pd.isna(metrics["fpr_normal"]) else "",
                }

                # Record per-class F1, Recall, and Precision
                for cls in classes:
                    if str(cls) in metrics["per_class"]:
                        c_vals = metrics["per_class"][str(cls)]
                        row[f"f1_{cls}"] = round(c_vals["f1"], 5)
                        row[f"recall_{cls}"] = round(c_vals["recall"], 5)
                        row[f"precision_{cls}"] = round(c_vals["precision"], 5)

                # Append immediately to CSV to ensure live persistence
                df_row = pd.DataFrame([row])
                write_header = not output_csv.exists()
                df_row.to_csv(output_csv, mode="a", header=write_header, index=False)

                # Also sync NSL-KDD runs to standard raw_results.csv
                if dataset_name.upper() in ("NSL-KDD", "NSLKDD") and output_csv.name != "raw_results.csv":
                    default_raw = tables_dir / "raw_results.csv"
                    write_h = not default_raw.exists()
                    df_row.to_csv(default_raw, mode="a", header=write_h, index=False)

                completed_runs.add(pair)
                logger.info(
                    f"  Finished: Seed={seed}, Method={method} | Macro-F1={metrics['f1_macro']:.4f} | "
                    f"Train={train_time:.1f}s | Infer/sample={infer_per_sample_ms:.4f}ms"
                )

            except Exception as e:
                logger.error(
                    f"Failed experiment for Seed={seed}, Method={method}: {e}",
                    exc_info=True,
                )

    logger.info("============================================================")
    logger.info(f"Experiment sweep completed! All results recorded in: {output_csv}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
