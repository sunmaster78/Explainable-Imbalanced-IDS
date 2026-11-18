"""
Unit tests for statistical tests and LaTeX export (src/stats_tests.py, Phase 6).

Tests:
    - summarize: mean ± std aggregation (Task 6.2)
    - wilcoxon_vs_best: paired Wilcoxon signed-rank test and warning checks (Task 6.2)
    - make_efficiency_table: training and inference profiling (Task 6.3)
    - to_latex_tables: booktabs LaTeX conversion and bolding (Task 6.4)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.stats_tests import (
    make_efficiency_table,
    summarize,
    to_latex_tables,
    wilcoxon_vs_best,
)


@pytest.fixture
def mock_results_df():
    """Create a mock raw_results DataFrame across 6 seeds and 3 methods."""
    np.random.seed(42)
    rows = []
    methods = ["Proposed", "B1", "B6"]

    for seed in range(6):
        for m in methods:
            base_f1 = 0.85 if m == "Proposed" else (0.75 if m == "B6" else 0.65)
            f1 = base_f1 + np.random.normal(0, 0.02)
            acc = f1 + 0.05
            train_sec = 20.0 if "Proposed" in m else 10.0
            infer_ms = 0.015 if "Proposed" in m else 0.010

            rows.append({
                "seed": seed,
                "method": m,
                "accuracy": acc,
                "f1_macro": f1,
                "f1_weighted": f1 + 0.02,
                "mcc": f1 - 0.05,
                "fpr_normal": 0.02 if m == "Proposed" else 0.05,
                "training_time_sec": train_sec + np.random.uniform(0, 1),
                "inference_time_per_sample_ms": infer_ms + np.random.uniform(0, 0.002),
            })

    return pd.DataFrame(rows)


class TestStatsTests:
    def test_summarize(self, mock_results_df, tmp_path):
        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        sum_df = summarize(mock_results_df, config=config, table_name="test_summary")

        assert len(sum_df) == 3
        assert "method" in sum_df.columns
        assert "f1_macro" in sum_df.columns
        assert "±" in str(sum_df.iloc[0]["f1_macro"])
        assert (tmp_path / "tables" / "test_summary.csv").exists()

    def test_wilcoxon_vs_best(self, mock_results_df, tmp_path):
        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        sig_df = wilcoxon_vs_best(
            raw_results_path=mock_results_df,
            metric="f1_macro",
            proposed_name="Proposed",
            config=config,
            table_name="test_significance",
        )

        assert len(sig_df) == 2  # B1 and B6
        assert "baseline" in sig_df.columns
        assert "p_value" in sig_df.columns
        assert (tmp_path / "tables" / "test_significance.csv").exists()

    def test_wilcoxon_few_pairs_warning(self, mock_results_df, caplog):
        # Subset to only 3 seeds (< 6)
        few_seeds_df = mock_results_df[mock_results_df["seed"] < 3]

        sig_df = wilcoxon_vs_best(
            raw_results_path=few_seeds_df,
            metric="f1_macro",
            proposed_name="Proposed",
        )

        assert len(sig_df) == 2
        # Check that warning was logged
        assert any("At least 6 paired observations are required" in rec.message for rec in caplog.records)

    def test_make_efficiency_table(self, mock_results_df, tmp_path):
        config = {"output_dirs": {"tables": str(tmp_path / "tables")}}

        eff_df = make_efficiency_table(
            raw_results_path=mock_results_df,
            config=config,
            table_name="test_efficiency",
        )

        assert len(eff_df) == 3
        assert "training_time_sec" in eff_df.columns
        assert "inference_time_ms_per_sample" in eff_df.columns
        assert "throughput_samples_per_sec" in eff_df.columns
        assert (tmp_path / "tables" / "test_efficiency.csv").exists()

    def test_to_latex_tables(self, mock_results_df, tmp_path):
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        config = {"output_dirs": {"tables": str(tables_dir)}}

        # Save summary, significance, and efficiency tables first
        summarize(mock_results_df, config=config, table_name="summary_mean_std")
        wilcoxon_vs_best(mock_results_df, config=config, table_name="significance")
        make_efficiency_table(mock_results_df, config=config, table_name="efficiency")

        tex_files = to_latex_tables(config=config, tables_dir=tables_dir)

        assert len(tex_files) == 3
        for tf in tex_files:
            assert tf.exists()
            content = tf.read_text(encoding="utf-8")
            assert r"\begin{table}" in content
            assert r"\toprule" in content
            assert r"\bottomrule" in content
            assert r"\textbf{" in content
