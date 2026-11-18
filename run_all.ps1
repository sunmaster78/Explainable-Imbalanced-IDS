# ============================================================================
# Two-Stage IDS Full Experimental Pipeline Runner (PowerShell)
# ============================================================================
param (
    [string]$Config = "configs/nslkdd.yaml",
    [string]$Seeds = "0",
    [string]$Methods = "B1,B2,B3,B4,B5,B6,Proposed"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Starting Full Two-Stage IDS Pipeline"
Write-Host "Config:  $Config"
Write-Host "Seeds:   $Seeds"
Write-Host "Methods: $Methods"
Write-Host "============================================================"

# 1. Run Benchmark Experiments Sweep
brmenv\Scripts\python.exe src/run_experiments.py --config $Config --seeds $Seeds --methods $Methods

# 2. Run Significance and Evaluation Reporting
brmenv\Scripts\python.exe -c "
from src.stats_tests import summarize, wilcoxon_vs_best, make_efficiency_table, to_latex_tables
from src.utils import load_config
cfg = load_config('$Config')
summarize(config=cfg)
wilcoxon_vs_best(config=cfg)
make_efficiency_table(config=cfg)
to_latex_tables(config=cfg)
print('Evaluation tables and LaTeX exports completed.')
"

# 3. Regenerate Publication Figures
brmenv\Scripts\python.exe src/make_figures.py

# 4. Run Pytest Verification
brmenv\Scripts\python.exe -m pytest

Write-Host "============================================================"
Write-Host "Pipeline execution completed successfully!"
Write-Host "Artifacts saved in results/tables/ and results/figures/"
Write-Host "============================================================"
