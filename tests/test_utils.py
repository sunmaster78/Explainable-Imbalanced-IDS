"""
Unit tests for src/utils.py.
Validates repeatable pseudorandom number generation across random, numpy, and torch,
as well as directory creation and artifact persistence.
"""

import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.utils import (
    ensure_dirs,
    get_logger,
    load_config,
    save_figure,
    save_table,
    set_seed,
)


def test_set_seed_repeatability():
    """Verify that set_seed produces identical random sequences across runs."""
    seed = 42

    # Run 1
    set_seed(seed)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.rand(5)

    # Run 2 with same seed
    set_seed(seed)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert r1 == r2, f"random.random() not repeatable: {r1} != {r2}"
    np.testing.assert_allclose(np1, np2, err_msg="numpy.random not repeatable")
    assert torch.equal(t1, t2), "torch.rand not repeatable"


def test_load_config_and_ensure_dirs(tmp_path: Path):
    """Test load_config reads YAML and ensure_dirs creates configured paths."""
    sample_config = {
        "dataset": "NSL-KDD",
        "output_dirs": {
            "tables": str(tmp_path / "custom_tables"),
            "figures": str(tmp_path / "custom_figures"),
        },
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)

    loaded = load_config(config_file)
    assert loaded["dataset"] == "NSL-KDD"

    ensure_dirs(loaded)
    assert Path(sample_config["output_dirs"]["tables"]).is_dir()
    assert Path(sample_config["output_dirs"]["figures"]).is_dir()


def test_save_table_and_figure(tmp_path: Path):
    """Test save_table and save_figure write valid files to disk."""
    config = {
        "output_dirs": {
            "tables": str(tmp_path / "tbl"),
            "figures": str(tmp_path / "fig"),
        }
    }

    # Test save_table
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    csv_path = save_table(df, "test_table", config=config)
    assert csv_path.exists()
    loaded_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(df, loaded_df)

    # Test save_figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_path = save_figure(fig, "test_fig", config=config)
    plt.close(fig)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_get_logger(tmp_path: Path):
    """Test logger writes messages to log file."""
    log_path = tmp_path / "test.log"
    logger = get_logger(name="test_logger", log_file=log_path)
    logger.info("Test message for logger")

    # Flush handlers
    for handler in logger.handlers:
        handler.flush()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "Test message for logger" in content
