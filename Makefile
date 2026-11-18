# ============================================================================
# Makefile: Two-Stage Explainable Intrusion Detection System (IDS)
# ============================================================================

PYTHON = brmenv\Scripts\python.exe
CONFIG ?= configs/nslkdd.yaml
SEEDS ?= 0
METHODS ?= B1,B2,B3,B4,B5,B6,Proposed

.PHONY: help setup preprocess train evaluate shap figures test all

help:
	@echo Two-Stage GAN + Ensemble + SHAP IDS Pipeline
	@echo Targets:
	@echo   setup       - Install pinned dependencies
	@echo   preprocess  - Run data loading and preprocessing check
	@echo   train       - Run benchmark experiment sweep across methods and seeds
	@echo   evaluate    - Compute aggregate statistics, Wilcoxon tests, and LaTeX tables
	@echo   shap        - Run SHAP fidelity and stage-wise explainability audit
	@echo   figures     - Regenerate publication figures (300 DPI PNG and vector PDF)
	@echo   test        - Run complete pytest test suite
	@echo   all         - Run preprocess, train, evaluate, shap, figures, and test

setup:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) -c "from src.utils import load_config; from src.preprocess import load_nslkdd, map_attack_categories, build_dataset; cfg = load_config('$(CONFIG)'); tr, te = load_nslkdd(cfg['dataset']['raw_train_path'], cfg['dataset']['raw_test_path']); d = build_dataset(map_attack_categories(tr), map_attack_categories(te), cfg); print('Preprocessing successful!')"

train:
	$(PYTHON) src/run_experiments.py --config $(CONFIG) --seeds $(SEEDS) --methods $(METHODS)

evaluate:
	$(PYTHON) -c "from src.stats_tests import summarize, wilcoxon_vs_best, make_efficiency_table, to_latex_tables; summarize(); wilcoxon_vs_best(); make_efficiency_table(); to_latex_tables(); print('Evaluation tables exported successfully!')"

shap:
	$(PYTHON) -c "from src.utils import load_config; from src.preprocess import load_nslkdd, map_attack_categories, build_dataset; from src.shap_fidelity import run_fidelity; from src.shap_stagewise import run_stagewise, run_local_explanations_suite; from src.pipeline import TwoStageIDS; cfg = load_config('$(CONFIG)'); tr, te = load_nslkdd(cfg['dataset']['raw_train_path'], cfg['dataset']['raw_test_path']); d = build_dataset(map_attack_categories(tr), map_attack_categories(te), cfg); pipe = TwoStageIDS(cfg); pipe.fit_stage1(d['X_train'], d['y_train_bin']); pipe.tune_stage1_threshold(d['X_val'], d['y_val_bin']); pipe.fit_stage2(d['X_train'][d['y_train_bin']==1], d['y_train_cat'][d['y_train_bin']==1], use_gan=False); run_stagewise(pipe, d, cfg, max_samples=50); run_local_explanations_suite(pipe, d, cfg); print('SHAP audit complete!')"

figures:
	$(PYTHON) src/make_figures.py

test:
	$(PYTHON) -m pytest

all: preprocess train evaluate shap figures test
