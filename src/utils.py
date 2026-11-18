"""
Utility functions for the Two-Stage IDS pipeline.
Includes random seed configuration, YAML configuration loading, logging setup,
directory management, and empirical artifact persistence (tables & figures).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.figure
import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set unified random seed across random, numpy, and torch (CPU & CUDA).
    Configures deterministic algorithms where feasible to ensure full reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Attempt to enable deterministic algorithms with warn_only to avoid hard crashes on unsupported ops
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if isinstance(config, dict) else {}


def get_logger(
    name: str = "IDS",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a standard logger that writes to both console and log file.
    Prevents adding duplicate handlers on multiple invocations.

    Args:
        name: Name of the logger.
        log_file: Optional file path for file logging. If None, defaults to results/logs/{name}.log.
        level: Logging level (e.g., logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            default_log_dir = Path("results/logs")
            default_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = default_log_dir / f"{name}.log"
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dirs(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Create all required project directories for tables, figures, models, and logs.
    Also creates any additional output directories specified in the config.

    Args:
        config: Optional configuration dictionary that may contain 'output_dirs' mapping.
    """
    default_dirs = [
        Path("results/tables"),
        Path("results/figures"),
        Path("results/models"),
        Path("results/logs"),
        Path("data/raw"),
        Path("data/processed"),
    ]

    for d in default_dirs:
        d.mkdir(parents=True, exist_ok=True)

    if config and "output_dirs" in config and isinstance(config["output_dirs"], dict):
        for path_str in config["output_dirs"].values():
            Path(path_str).mkdir(parents=True, exist_ok=True)


def save_table(
    df: pd.DataFrame,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a pandas DataFrame as a CSV table to results/tables/{name}.csv.

    Args:
        df: DataFrame to save.
        name: Name of the table (with or without .csv extension).
        config: Optional configuration dictionary.

    Returns:
        Path to the saved CSV file.
    """
    if config and "output_dirs" in config and "tables" in config["output_dirs"]:
        tables_dir = Path(config["output_dirs"]["tables"])
    else:
        tables_dir = Path("results/tables")

    tables_dir.mkdir(parents=True, exist_ok=True)

    filename = name if name.endswith(".csv") else f"{name}.csv"
    output_path = tables_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    config: Optional[Dict[str, Any]] = None,
    dpi: int = 300,
) -> Path:
    """
    Save a matplotlib Figure as a high-resolution 300 DPI PNG to results/figures/{name}.png.

    Args:
        fig: matplotlib.figure.Figure object to save.
        name: Name of the figure (with or without .png extension).
        config: Optional configuration dictionary.
        dpi: Dots per inch resolution (default 300).

    Returns:
        Path to the saved PNG file.
    """
    if config and "output_dirs" in config and "figures" in config["output_dirs"]:
        figures_dir = Path(config["output_dirs"]["figures"])
    else:
        figures_dir = Path("results/figures")

    figures_dir.mkdir(parents=True, exist_ok=True)

    filename = name if name.endswith(".png") else f"{name}.png"
    output_path = figures_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
