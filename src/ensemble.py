"""
Ensemble classification module for Network Intrusion Detection.

Combines three high-performing gradient-boosted and tree-based learners:
    1. Random Forest (scikit-learn)
    2. XGBoost (XGBClassifier)
    3. LightGBM (LGBMClassifier)

Provides:
    - build_voting_ensemble: Soft-voting ensemble with configurable weights.
    - build_stacking_ensemble: Stacking classifier with LogisticRegression meta-learner.
    - get_ensemble: Unified factory function for 'voting' and 'stacking'.
    - tune_ensemble: PredefinedSplit validation tuning with macro-F1 scoring.

Strictly adheres to:
    - Unified random seed from config.
    - Leak-free hyperparameter tuning on validation split only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from src.utils import get_logger

logger = get_logger("ensemble")


# ---------------------------------------------------------------------------
# Base Estimator Builders
# ---------------------------------------------------------------------------

def build_base_models(
    config: Dict[str, Any],
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> List[Tuple[str, Any]]:
    """
    Construct the three base classifiers (RF, XGBoost, LightGBM) configured
    for either binary (n_classes=2) or multiclass (n_classes > 2) tasks.

    Args:
        config: Configuration dictionary.
        n_classes: Number of target classes (2 for Stage 1, >2 for Stage 2).
        seed: Random seed. If None, read from config["experiment"]["default_seed"].

    Returns:
        List of tuples: [('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)].
    """
    if seed is None:
        seed = int(config.get("experiment", {}).get("default_seed", 42))

    ens_cfg = config.get("ensemble", {})
    n_estimators = int(ens_cfg.get("n_estimators", 100))
    max_depth = ens_cfg.get("max_depth", None)
    learning_rate = float(ens_cfg.get("learning_rate", 0.1))

    # 1. Random Forest
    rf_kwargs: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "random_state": seed,
        "n_jobs": -1,
    }
    if max_depth is not None:
        rf_kwargs["max_depth"] = max_depth
    rf = RandomForestClassifier(**rf_kwargs)

    # 2. XGBoost
    xgb_kwargs: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if max_depth is not None:
        xgb_kwargs["max_depth"] = max_depth

    if n_classes == 2:
        xgb_kwargs["objective"] = "binary:logistic"
        xgb_kwargs["eval_metric"] = "logloss"
    else:
        xgb_kwargs["objective"] = "multi:softprob"
        xgb_kwargs["eval_metric"] = "mlogloss"
        xgb_kwargs["num_class"] = n_classes

    xgb_model = xgb.XGBClassifier(**xgb_kwargs)

    # 3. LightGBM
    lgb_kwargs: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if max_depth is not None:
        lgb_kwargs["max_depth"] = max_depth

    if n_classes == 2:
        lgb_kwargs["objective"] = "binary"
        lgb_kwargs["metric"] = "binary_logloss"
    else:
        lgb_kwargs["objective"] = "multiclass"
        lgb_kwargs["num_class"] = n_classes
        lgb_kwargs["metric"] = "multi_logloss"

    lgb_model = lgb.LGBMClassifier(**lgb_kwargs)

    return [("rf", rf), ("xgb", xgb_model), ("lgbm", lgb_model)]


# ---------------------------------------------------------------------------
# Task 3.1: Soft-Voting Ensemble
# ---------------------------------------------------------------------------

def build_voting_ensemble(
    config: Dict[str, Any],
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> VotingClassifier:
    """
    Construct a soft-voting ensemble of Random Forest, XGBoost, and LightGBM.

    Args:
        config: Configuration dictionary containing ensemble settings.
        n_classes: Number of output classes (2 for binary, >2 for multiclass).
        seed: Random seed.

    Returns:
        scikit-learn VotingClassifier(voting='soft').
    """
    estimators = build_base_models(config=config, n_classes=n_classes, seed=seed)

    ens_cfg = config.get("ensemble", {})
    weights: Optional[List[float]] = ens_cfg.get("weights", None)
    if weights is not None:
        if len(weights) != 3:
            raise ValueError(f"Ensemble weights must have length 3, got {len(weights)}")
        weights = [float(w) for w in weights]

    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
    logger.info(
        f"Built soft-voting ensemble: n_classes={n_classes}, weights={weights}, "
        f"estimators={[name for name, _ in estimators]}"
    )
    return voting_ensemble


# ---------------------------------------------------------------------------
# Task 3.2: Stacking Ensemble
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    config: Dict[str, Any],
    n_classes: int = 2,
    seed: Optional[int] = None,
    cv: int = 5,
) -> StackingClassifier:
    """
    Construct a stacking ensemble using RF, XGBoost, and LightGBM as base learners
    and LogisticRegression as the meta-learner.

    Args:
        config: Configuration dictionary.
        n_classes: Number of target classes.
        seed: Random seed.
        cv: Cross-validation generator or number of folds for out-of-fold predictions.

    Returns:
        scikit-learn StackingClassifier.
    """
    if seed is None:
        seed = int(config.get("experiment", {}).get("default_seed", 42))

    estimators = build_base_models(config=config, n_classes=n_classes, seed=seed)

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        n_jobs=-1,
    )

    stacking_ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    logger.info(
        f"Built stacking ensemble: n_classes={n_classes}, meta=LogisticRegression, "
        f"cv={cv}, stack_method='predict_proba'"
    )
    return stacking_ensemble


def get_ensemble(
    kind: str,
    config: Dict[str, Any],
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> Union[VotingClassifier, StackingClassifier]:
    """
    Factory function returning either a soft-voting or stacking ensemble.

    Args:
        kind: 'voting' or 'stacking' (case-insensitive).
        config: Configuration dictionary.
        n_classes: Number of target classes.
        seed: Random seed.

    Returns:
        VotingClassifier or StackingClassifier.
    """
    k = kind.strip().lower()
    if k in ("voting", "soft_voting", "soft-voting"):
        return build_voting_ensemble(config=config, n_classes=n_classes, seed=seed)
    elif k in ("stacking", "stack"):
        return build_stacking_ensemble(config=config, n_classes=n_classes, seed=seed)
    else:
        raise ValueError(
            f"Unknown ensemble kind: '{kind}'. Expected 'voting' or 'stacking'."
        )


# ---------------------------------------------------------------------------
# Task 3.3: Validation-Only Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune_ensemble(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_val: Union[np.ndarray, pd.DataFrame],
    y_val: Union[np.ndarray, pd.Series],
    config: Dict[str, Any],
    n_iter: Optional[int] = None,
    seed: Optional[int] = None,
    save_name: str = "best_ensemble_params.yaml",
) -> Dict[str, Dict[str, Any]]:
    """
    Tune hyperparameters for each base model separately on the validation split.

    Strict No Data Leakage Rule:
    - Uses PredefinedSplit with train=-1 and val=0, ensuring no test data is touched
      and models are evaluated purely on the validation partition.
    - Optimizes for macro-averaged F1 score ('f1_macro').

    Args:
        X_train: Training features array.
        y_train: Training labels array.
        X_val: Validation features array.
        y_val: Validation labels array.
        config: Configuration dictionary.
        n_iter: Number of parameter combinations to sample per model (default from config or 4).
        seed: Random seed.
        save_name: Filename to save the best parameters YAML in results/models/.

    Returns:
        Dict mapping model name ('rf', 'xgb', 'lgbm') -> best parameters dict.
    """
    if seed is None:
        seed = int(config.get("experiment", {}).get("default_seed", 42))

    if n_iter is None:
        n_iter = int(
            config.get("ensemble", {}).get(
                "tune_n_iter",
                config.get("tune_n_iter", 4),
            )
        )

    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train)
    X_v = np.asarray(X_val, dtype=np.float32)
    y_v = np.asarray(y_val)

    # Combine train and validation data for PredefinedSplit
    X_all = np.vstack([X_tr, X_v])
    y_all = np.concatenate([y_tr, y_v])

    # PredefinedSplit: -1 for training indices, 0 for validation indices
    test_fold = np.full(len(X_all), -1, dtype=int)
    test_fold[len(X_tr):] = 0
    ps = PredefinedSplit(test_fold)

    n_classes = len(np.unique(y_tr))

    logger.info(
        f"Starting validation-only tuning: {len(X_tr)} train, {len(X_v)} val, "
        f"n_classes={n_classes}, n_iter={n_iter}"
    )

    # Parameter search spaces (compact for fast, reliable execution)
    rf_param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [6, 12, None],
        "min_samples_split": [2, 5],
    }

    xgb_param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }

    lgb_param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [4, 8, -1],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
    }

    # Instantiate fresh base estimators
    base_models = dict(build_base_models(config=config, n_classes=n_classes, seed=seed))

    best_params: Dict[str, Dict[str, Any]] = {}

    search_configs = [
        ("rf", base_models["rf"], rf_param_dist),
        ("xgb", base_models["xgb"], xgb_param_dist),
        ("lgbm", base_models["lgbm"], lgb_param_dist),
    ]

    for name, estimator, param_dist in search_configs:
        logger.info(f"Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=ps,
            random_state=seed,
            n_jobs=-1,
            refit=False,  # We only need the best parameters
        )
        search.fit(X_all, y_all)
        best_score = float(search.best_score_)
        best_p = search.best_params_
        logger.info(f"  {name} Best Val Macro-F1: {best_score:.4f} with params: {best_p}")
        best_params[name] = best_p

    # Save best parameters to results/models
    models_dir = Path("results/models")
    if config and "output_dirs" in config and "models" in config["output_dirs"]:
        models_dir = Path(config["output_dirs"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    save_path = models_dir / save_name
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)

    logger.info(f"Saved tuned ensemble parameters to: {save_path}")
    return best_params
