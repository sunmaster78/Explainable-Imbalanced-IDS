#!/usr/bin/env bash
# ============================================================================
# Two-Stage IDS Full Experimental Pipeline Runner (Bash)
# ============================================================================
set -e

CONFIG=${1:-"configs/nslkdd.yaml"}
SEEDS=${2:-"0"}
METHODS=${3:-"B1,B2,B3,B4,B5,B6,Proposed"}

echo "============================================================"
echo "Starting Full Two-Stage IDS Pipeline"
echo "Config:  $CONFIG"
echo "Seeds:   $SEEDS"
echo "Methods: $METHODS"
echo "============================================================"

# 1. Run Benchmark Experiments Sweep
brmenv/Scripts/python.exe src/run_experiments.py --config "$CONFIG" --seeds "$SEEDS" --methods "$METHODS"

# 2. Run Significance and Evaluation Reporting
brmenv/Scripts/python.exe -c "
from src.stats_tests import summarize, wilcoxon_vs_best, make_efficiency_table, to_latex_tables
from src.utils import load_config
cfg = load_config('$CONFIG')
summarize(config=cfg)
wilcoxon_vs_best(config=cfg)
make_efficiency_table(config=cfg)
to_latex_tables(config=cfg)
print('Evaluation tables and LaTeX exports completed.')
"

# 3. Regenerate Publication Figures
brmenv/Scripts/python.exe src/make_figures.py

# 4. Run Pytest Verification
brmenv/Scripts/python.exe -m pytest

echo "============================================================"
echo "Pipeline execution completed successfully!"
echo "Artifacts saved in results/tables/ and results/figures/"
echo "============================================================"
