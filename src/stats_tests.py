"""
Statistical Significance and Evaluation Reporting Module.

Provides post-experiment aggregation, significance hypothesis testing,
efficiency profiling, and publication-ready LaTeX table export:

1. summarize: Computes mean ± std per method across experimental seeds.
2. wilcoxon_vs_best: Paired Wilcoxon signed-rank tests comparing Proposed
   against all baselines with minimum sample size warnings.
3. make_efficiency_table: Profiles training runtime and per-sample inference latency.
4. to_latex_tables: Exports booktabs-formatted LaTeX tables with column-wise
   best performers highlighted in bold.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger, save_table

logger = get_logger("stats_tests")


# ---------------------------------------------------------------------------
# Task 6.2: Aggregation & Summary (mean ± std)
# ---------------------------------------------------------------------------

def summarize(
    raw_results_path: Union[str, Path, pd.DataFrame] = "results/tables/raw_results.csv",
    config: Optional[Dict[str, Any]] = None,
    table_name: str = "summary_mean_std",
) -> pd.DataFrame:
    """
    Compute mean ± std across seeds for each evaluated method and metric.

    Args:
        raw_results_path: Path to raw_results.csv or an existing DataFrame.
        config: Optional configuration dict.
        table_name: Base filename to save the resulting summary table.

    Returns:
        DataFrame with rows as methods and columns formatted as 'mean ± std'.
    """
    if isinstance(raw_results_path, pd.DataFrame):
        df = raw_results_path.copy()
    else:
        p = Path(raw_results_path)
        if not p.exists():
            raise FileNotFoundError(f"Raw results file not found at: {p}")
        df = pd.read_csv(p)

    if "method" not in df.columns:
        raise ValueError("raw_results DataFrame must contain a 'method' column.")

    # Identify numeric metric columns to summarize
    exclude_cols = {"seed", "method"}
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    summary_rows = []
    methods = df["method"].unique()

    for m in methods:
        sub = df[df["method"] == m]
        row_dict: Dict[str, Any] = {"method": str(m), "n_seeds": len(sub)}

        for col in numeric_cols:
            vals = sub[col].dropna()
            if len(vals) == 0:
                row_dict[col] = "N/A"
            elif len(vals) == 1:
                row_dict[col] = f"{vals.iloc[0]:.4f}"
            else:
                mean_val = vals.mean()
                std_val = vals.std()
                row_dict[col] = f"{mean_val:.4f} ± {std_val:.4f}"

            # Also retain raw float means for automated LaTeX bolding
            row_dict[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan

        summary_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)

    if config is not None:
        save_table(summary_df, table_name, config)
        logger.info(f"Saved summary (mean ± std) table to {table_name}.csv")

    return summary_df


# ---------------------------------------------------------------------------
# Task 6.2: Paired Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

def wilcoxon_vs_best(
    raw_results_path: Union[str, Path, pd.DataFrame] = "results/tables/raw_results.csv",
    metric: str = "f1_macro",
    proposed_name: str = "Proposed",
    config: Optional[Dict[str, Any]] = None,
    table_name: str = "significance",
) -> pd.DataFrame:
    """
    Perform paired Wilcoxon signed-rank tests between Proposed and every baseline.

    Statistical Integrity Safeguards:
        - Pairs results strictly on matching random seeds.
        - Warns if number of paired seeds < 6 (Wilcoxon cannot achieve p < 0.05
          with fewer than 6 non-zero differences).

    Args:
        raw_results_path: Path to raw_results.csv or a DataFrame.
        metric: Performance metric column to test (default: 'f1_macro').
        proposed_name: Identifier of the proposed method in the table.
        config: Optional configuration dict.
        table_name: Base filename to save the resulting significance table.

    Returns:
        DataFrame with columns: ['baseline', 'n_pairs', 'proposed_mean',
        'baseline_mean', 'mean_diff', 'statistic', 'p_value', 'p_less_05'].
    """
    if isinstance(raw_results_path, pd.DataFrame):
        df = raw_results_path.copy()
    else:
        p = Path(raw_results_path)
        if not p.exists():
            raise FileNotFoundError(f"Raw results file not found at: {p}")
        df = pd.read_csv(p)

    if "seed" not in df.columns or "method" not in df.columns:
        raise ValueError("raw_results must contain 'seed' and 'method' columns.")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in raw_results columns.")

    # Locate proposed method (case-insensitive search)
    unique_methods = df["method"].unique()
    match_proposed = [m for m in unique_methods if str(m).strip().lower() == proposed_name.lower()]
    if not match_proposed:
        raise ValueError(
            f"Proposed method '{proposed_name}' not found among methods: {list(unique_methods)}"
        )
    prop_key = match_proposed[0]

    prop_sub = df[df["method"] == prop_key].set_index("seed")[metric]

    results = []
    baselines = [m for m in unique_methods if m != prop_key]

    for base in baselines:
        base_sub = df[df["method"] == base].set_index("seed")[metric]
        common_seeds = prop_sub.index.intersection(base_sub.index)
        n_pairs = len(common_seeds)

        if n_pairs == 0:
            logger.warning(f"No matching seeds found between '{prop_key}' and '{base}'.")
            continue

        prop_vals = prop_sub.loc[common_seeds].values
        base_vals = base_sub.loc[common_seeds].values
        diffs = prop_vals - base_vals

        if n_pairs < 6:
            logger.warning(
                f"Wilcoxon signed-rank test for '{base}' has only {n_pairs} paired seed(s). "
                f"Statistical warning: At least 6 paired observations are required to reach "
                f"significance at p < 0.05."
            )

        # Wilcoxon requires non-zero differences
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) == 0:
            stat_val = float("nan")
            p_val = 1.0
        elif len(nonzero_diffs) < 2:
            stat_val = float("nan")
            p_val = 1.0
        else:
            try:
                res = stats.wilcoxon(prop_vals, base_vals, alternative="two-sided")
                stat_val = float(res.statistic)
                p_val = float(res.pvalue)
            except Exception as e:
                logger.warning(f"Wilcoxon computation failed for {base}: {e}")
                stat_val = float("nan")
                p_val = float("nan")

        results.append({
            "baseline": str(base),
            "metric": metric,
            "n_pairs": n_pairs,
            "proposed_mean": round(float(np.mean(prop_vals)), 5),
            "baseline_mean": round(float(np.mean(base_vals)), 5),
            "mean_diff": round(float(np.mean(diffs)), 5),
            "statistic": round(stat_val, 4) if not np.isnan(stat_val) else np.nan,
            "p_value": round(p_val, 5) if not np.isnan(p_val) else np.nan,
            "p_less_05": (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    sig_df = pd.DataFrame(results)

    if config is not None:
        save_table(sig_df, table_name, config)
        logger.info(f"Saved significance table to {table_name}.csv")

    return sig_df


# ---------------------------------------------------------------------------
# Task 6.3: Efficiency Profiling Table
# ---------------------------------------------------------------------------

def make_efficiency_table(
    raw_results_path: Union[str, Path, pd.DataFrame] = "results/tables/raw_results.csv",
    config: Optional[Dict[str, Any]] = None,
    table_name: str = "efficiency",
) -> pd.DataFrame:
    """
    Summarize computational training time and per-sample inference latency per method.

    Args:
        raw_results_path: Path to raw_results.csv or a DataFrame.
        config: Optional configuration dict.
        table_name: Name of CSV table to save.

    Returns:
        DataFrame containing mean ± std of training time and inference latency.
    """
    if isinstance(raw_results_path, pd.DataFrame):
        df = raw_results_path.copy()
    else:
        p = Path(raw_results_path)
        if not p.exists():
            raise FileNotFoundError(f"Raw results file not found at: {p}")
        df = pd.read_csv(p)

    if "method" not in df.columns:
        raise ValueError("raw_results must contain 'method' column.")

    train_col = "training_time_sec"
    infer_col = "inference_time_per_sample_ms"

    if train_col not in df.columns or infer_col not in df.columns:
        raise ValueError(
            f"Columns '{train_col}' and '{infer_col}' required in raw_results for efficiency profiling."
        )

    rows = []
    for m in df["method"].unique():
        sub = df[df["method"] == m]
        t_vals = sub[train_col].dropna()
        i_vals = sub[infer_col].dropna()

        t_mean, t_std = (t_vals.mean(), t_vals.std()) if len(t_vals) > 0 else (np.nan, np.nan)
        i_mean, i_std = (i_vals.mean(), i_vals.std()) if len(i_vals) > 0 else (np.nan, np.nan)

        throughput = (1000.0 / i_mean) if i_mean > 0 else np.nan

        rows.append({
            "method": str(m),
            "n_runs": len(sub),
            "training_time_sec": (
                f"{t_mean:.2f} ± {t_std:.2f}" if not np.isnan(t_std) else f"{t_mean:.2f}"
            ),
            "inference_time_ms_per_sample": (
                f"{i_mean:.4f} ± {i_std:.4f}" if not np.isnan(i_std) else f"{i_mean:.4f}"
            ),
            "throughput_samples_per_sec": (
                f"{throughput:,.1f}" if not np.isnan(throughput) else "N/A"
            ),
            # Raw floats for LaTeX bolding
            "raw_train_sec_mean": t_mean,
            "raw_infer_ms_mean": i_mean,
        })

    eff_df = pd.DataFrame(rows)

    if config is not None:
        save_table(eff_df, table_name, config)
        logger.info(f"Saved efficiency table to {table_name}.csv")

    return eff_df


# ---------------------------------------------------------------------------
# Task 6.4: Publication-Ready LaTeX Tables
# ---------------------------------------------------------------------------

def to_latex_tables(
    config: Optional[Dict[str, Any]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Convert summary_mean_std, significance, and efficiency CSV tables into
    publication-ready LaTeX tables (booktabs style) with the best value in each
    column in bold.

    Saves .tex files in results/tables/latex/.

    Args:
        config: Optional configuration dict.
        tables_dir: Directory containing input CSVs (default: results/tables).
        output_dir: Directory to save .tex files (default: results/tables/latex).

    Returns:
        List of Path objects to the generated .tex files.
    """
    if tables_dir is None:
        tables_dir = Path("results/tables")
    else:
        tables_dir = Path(tables_dir)

    if output_dir is None:
        out_dir = tables_dir / "latex"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_tex_files: List[Path] = []

    # 1. Summary Performance LaTeX Table
    sum_csv = tables_dir / "summary_mean_std.csv"
    if sum_csv.exists():
        df_sum = pd.read_csv(sum_csv)
        tex_path = out_dir / "table_performance_summary.tex"

        display_cols = ["method", "accuracy", "f1_macro", "f1_weighted", "mcc", "fpr_normal"]
        cols_in_df = [c for c in display_cols if c in df_sum.columns]

        # Extract numeric values to find column-wise best performers
        best_indices: Dict[str, int] = {}
        for c in cols_in_df:
            if c == "method":
                continue
            mean_col = f"{c}_mean"
            if mean_col in df_sum.columns:
                series = df_sum[mean_col]
            else:
                # Parse numeric mean from '0.1234 ± 0.0056'
                series = df_sum[c].astype(str).str.split("±").str[0].astype(float)

            if c in ("fpr_normal", "fpr"):
                best_indices[c] = int(series.idxmin())  # lower is better for FPR
            else:
                best_indices[c] = int(series.idxmax())  # higher is better for F1/acc

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Empirical Benchmark Performance Across Evaluated Methods (Mean $\pm$ Std).}",
            r"\label{tab:performance_summary}",
            r"\begin{tabular}{l" + "c" * (len(cols_in_df) - 1) + r"}",
            r"\toprule",
            " & ".join([c.replace("_", " ").title() for c in cols_in_df]) + r" \\",
            r"\midrule",
        ]

        for i, row in df_sum.iterrows():
            row_items = []
            for c in cols_in_df:
                val = str(row[c])
                if c != "method" and c in best_indices and i == best_indices[c]:
                    val = r"\textbf{" + val + r"}"
                row_items.append(val)
            tex_lines.append(" & ".join(row_items) + r" \\")

        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines) + "\n")
        generated_tex_files.append(tex_path)
        logger.info(f"Generated LaTeX performance table at: {tex_path}")

    # 2. Significance LaTeX Table
    sig_csv = tables_dir / "significance.csv"
    if sig_csv.exists():
        df_sig = pd.read_csv(sig_csv)
        tex_path = out_dir / "table_statistical_significance.tex"

        display_cols = ["baseline", "n_pairs", "proposed_mean", "baseline_mean", "mean_diff", "p_value"]
        cols_in_df = [c for c in display_cols if c in df_sig.columns]

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Paired Wilcoxon Signed-Rank Test: Proposed Architecture vs. Baselines.}",
            r"\label{tab:statistical_significance}",
            r"\begin{tabular}{l" + "c" * (len(cols_in_df) - 1) + r"}",
            r"\toprule",
            " & ".join([c.replace("_", " ").title() for c in cols_in_df]) + r" \\",
            r"\midrule",
        ]

        for _, row in df_sig.iterrows():
            row_items = []
            for c in cols_in_df:
                val = str(row[c])
                if c == "p_value" and float(row[c]) < 0.05:
                    val = r"\textbf{" + val + r"}*"
                row_items.append(val)
            tex_lines.append(" & ".join(row_items) + r" \\")

        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines) + "\n")
        generated_tex_files.append(tex_path)
        logger.info(f"Generated LaTeX significance table at: {tex_path}")

    # 3. Efficiency LaTeX Table
    eff_csv = tables_dir / "efficiency.csv"
    if eff_csv.exists():
        df_eff = pd.read_csv(eff_csv)
        tex_path = out_dir / "table_computational_efficiency.tex"

        display_cols = ["method", "training_time_sec", "inference_time_ms_per_sample", "throughput_samples_per_sec"]
        cols_in_df = [c for c in display_cols if c in df_eff.columns]

        # Fast training and low latency are best
        best_indices = {}
        if "raw_train_sec_mean" in df_eff.columns:
            best_indices["training_time_sec"] = int(df_eff["raw_train_sec_mean"].idxmin())
        if "raw_infer_ms_mean" in df_eff.columns:
            best_indices["inference_time_ms_per_sample"] = int(df_eff["raw_infer_ms_mean"].idxmin())

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Computational Efficiency Profile: Training Runtime and Inference Throughput.}",
            r"\label{tab:efficiency}",
            r"\begin{tabular}{l" + "c" * (len(cols_in_df) - 1) + r"}",
            r"\toprule",
            " & ".join([c.replace("_", " ").title() for c in cols_in_df]) + r" \\",
            r"\midrule",
        ]

        for i, row in df_eff.iterrows():
            row_items = []
            for c in cols_in_df:
                val = str(row[c])
                if c in best_indices and i == best_indices[c]:
                    val = r"\textbf{" + val + r"}"
                row_items.append(val)
            tex_lines.append(" & ".join(row_items) + r" \\")

        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines) + "\n")
        generated_tex_files.append(tex_path)
        logger.info(f"Generated LaTeX efficiency table at: {tex_path}")

    return generated_tex_files
