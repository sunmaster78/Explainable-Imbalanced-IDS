# Two-Stage GAN + Ensemble + SHAP Intrusion Detection: AI Prompt Playbook

This README is a list of **small, ready-to-paste prompts** for an AI code writer (Claude Code, Cursor, Copilot Chat, ChatGPT, etc.). Each task is small, so the AI gives better code and you can test each step before moving on.

## Project in one paragraph

A two-stage IDS for imbalanced network traffic.
- **Stage 1:** binary classifier (Normal vs Attack).
- **Stage 2:** attack-type classifier (DoS, Probe, R2L, U2R ...), trained on attack-only data balanced with a **conditional WGAN-GP**.
- **Classifier:** Random Forest + XGBoost + LightGBM (soft-voting or stacking).
- **Explainability:** SHAP TreeExplainer.

**Three novelty claims to prove**
1. **SHAP fidelity:** synthetic minority samples (WGAN-GP) are explained by the same features as real ones, better than SMOTE.
2. **Stage-wise SHAP:** separate explanations for Stage 1 and Stage 2.
3. **Stage-aware GAN:** a GAN trained only on attack rows works better than a GAN trained on all classes.

Datasets: NSL-KDD (first), then UNSW-NB15, then CICIDS2017.

---

## How to use this file

1. Start a new AI chat or session. Paste the **Master Context Prompt** (below) first.
2. Paste **one task prompt at a time**, in order.
3. Run the code and check the **"Done when"** line before going to the next task.
4. Commit to Git after every working task.
5. If the AI gives broken code, paste the error back and say: `Fix this error. Do not change anything else.`
6. Never let the AI invent results. All numbers in your paper must come from your own saved CSV files.

---

## Master Context Prompt (paste this first, every new session)

```text
You are helping me build a research project: a two-stage network intrusion
detection system for imbalanced data.

Pipeline:
- Stage 1: binary classifier (Normal vs Attack).
- Stage 2: attack-type classifier trained ONLY on attack rows, with minority
  classes oversampled using a conditional WGAN-GP (PyTorch).
- Classifiers: RandomForest + XGBoost + LightGBM combined by soft-voting or stacking.
- Explainability: SHAP TreeExplainer.
- Datasets: NSL-KDD first, later UNSW-NB15 and CICIDS2017.

Strict rules for ALL code you write:
1. Python 3.10+, PyTorch, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap,
   pandas, numpy, matplotlib, seaborn, pyyaml.
2. Set random seeds (random, numpy, torch) from a single function set_seed(seed).
3. NO DATA LEAKAGE: split first, then fit scalers/encoders on training data only,
   and oversample ONLY the training data. Never touch the test set for tuning.
4. Tune thresholds and hyperparameters on a validation split only.
5. Save every result as a CSV in results/tables and every plot as PNG (300 dpi)
   in results/figures.
6. Write clean, modular code with type hints and short docstrings. No notebooks
   unless I ask. Use functions, not giant scripts.
7. Read settings from a YAML config file. No hard-coded paths.
8. Use logging, not print.
9. After writing code, tell me exactly how to run it and what output I should see.
10. Do not invent numbers or results. If something is uncertain, say so.

Reply "ready" and wait for my first task.
```

---

## Target repository structure

```text
ids-gan-shap/
├── README.md
├── requirements.txt
├── configs/
│   ├── nslkdd.yaml
│   ├── unsw.yaml
│   └── cicids.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── utils.py
│   ├── preprocess.py
│   ├── wgan_gp.py
│   ├── gan_quality.py
│   ├── resamplers.py
│   ├── ensemble.py
│   ├── pipeline.py
│   ├── baselines.py
│   ├── evaluate.py
│   ├── stats_tests.py
│   ├── shap_fidelity.py
│   ├── shap_stagewise.py
│   └── run_experiments.py
├── results/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── logs/
└── tests/
```

---

# PHASE 0: Setup

### Task 0.1: Project scaffold

```text
Create the repository scaffold shown below (folders and empty module files
with docstrings). Also create:
- requirements.txt with pinned-style entries for the libraries in the context.
- .gitignore (ignore data/, results/models/, __pycache__, .ipynb_checkpoints).
- configs/nslkdd.yaml containing: dataset name, raw train/test file paths,
  seed list [0..9], validation_size 0.2, output directories, and a section
  placeholder for gan, ensemble and pipeline settings.

[paste the repository structure from this README here]

Do not write any model code yet.
```
**Done when:** folders exist and `pip install -r requirements.txt` works.

### Task 0.2: Utilities

```text
In src/utils.py write:
1. set_seed(seed): seeds random, numpy, torch (and cuda if available), and sets
   torch deterministic flags where possible.
2. load_config(path): loads YAML into a dict.
3. get_logger(name, log_file): logger writing to console and results/logs.
4. ensure_dirs(config): creates all output directories.
5. save_table(df, name, config): saves a DataFrame to results/tables/{name}.csv.
6. save_figure(fig, name, config): saves PNG at 300 dpi to results/figures.
Add a tiny test in tests/test_utils.py checking set_seed gives repeatable numbers.
```
**Done when:** `pytest tests/test_utils.py` passes.

---

# PHASE 1: Preprocessing (NSL-KDD)

### Task 1.1: Load and clean

```text
In src/preprocess.py write load_nslkdd(train_path, test_path) that:
1. Reads KDDTrain+.txt and KDDTest+.txt with the 41 standard feature names plus
   'label' and 'difficulty'.
2. Drops the 'difficulty' column.
3. Drops exact duplicate rows (train and test separately).
4. Returns two DataFrames and logs shapes before and after cleaning.
Include the feature name list as a constant at the top of the file.
```
**Done when:** shapes printed; no duplicates remain.

### Task 1.2: Label mapping

```text
In src/preprocess.py add map_attack_categories(df) that maps the raw 'label'
column to 5 categories: Normal, DoS, Probe, R2L, U2R using the standard
NSL-KDD attack-to-category mapping (include the full dictionary, including
attacks that appear only in the test set). Add columns:
- 'category' (5-class string)
- 'is_attack' (0 for Normal, 1 otherwise)
Log the class counts for train and test and save them with save_table as
'class_distribution_nslkdd'. If an attack name is not in the mapping, raise a
clear error listing it.
```
**Done when:** class-count table shows Normal, DoS, Probe, R2L, U2R for train and test.

### Task 1.3: Encode, split and scale

```text
In src/preprocess.py write build_dataset(train_df, test_df, config) that:
1. Splits the training data into train and validation (stratified on 'category',
   size from config, using the given seed).
2. One-hot encodes protocol_type, service, flag. Fit the encoder on TRAIN only
   and handle unseen categories in val/test with handle_unknown='ignore'.
3. Scales numeric features with MinMaxScaler to the range [-1, 1], fit on TRAIN only.
4. Returns a dict with X_train, X_val, X_test, y_train_cat, y_val_cat,
   y_test_cat, y_train_bin, y_val_bin, y_test_bin, feature_names, and the
   fitted encoder and scaler objects, and the indices of the one-hot column groups.
5. Saves fitted objects with joblib to results/models.
Add a test that checks no leakage: scaler.data_min_ equals train min only.
```
**Done when:** the leakage test passes and shapes match across splits.

### Task 1.4: Class distribution plot

```text
Write a function plot_class_distribution(y, title, config) in src/evaluate.py
that makes a bar chart (log y-axis option) of class counts with counts written
above the bars, and saves it with save_figure. Call it for the NSL-KDD train
set (5-class and binary).
```
**Done when:** PNG shows the heavy imbalance (U2R is tiny).

---

# PHASE 2: Conditional WGAN-GP

### Task 2.1: Model definitions

```text
In src/wgan_gp.py write PyTorch classes:
- Generator: input = noise (latent_dim=100) concatenated with a one-hot class
  vector; layers Linear 256 -> 512 -> output_dim, LeakyReLU(0.2) with
  BatchNorm in hidden layers, final Tanh.
- Critic: input = sample concatenated with the one-hot class vector; layers
  Linear 512 -> 256 -> 1, LeakyReLU(0.2), NO sigmoid, no batchnorm (use
  LayerNorm or none).
Make layer sizes configurable. Add a shape test for both models.
```
**Done when:** the shape test passes.

### Task 2.2: Gradient penalty and training step

```text
In src/wgan_gp.py add:
1. gradient_penalty(critic, real, fake, labels, device, lambda_gp=10).
2. train_step: 5 critic updates per 1 generator update.
   Critic loss = mean(critic(fake)) - mean(critic(real)) + gradient penalty.
   Generator loss = -mean(critic(fake)).
Use Adam with lr=1e-4 and betas=(0.5, 0.9) for both.
Do not write the full training loop yet. Show the code for one iteration and a
small test using random data.
```
**Done when:** one iteration runs without errors and the losses are finite.

### Task 2.3: Full training loop

```text
In src/wgan_gp.py write train_wgan_gp(X_attack, y_attack_onehot, config) that:
1. Trains for config epochs with a DataLoader (batch size from config), on
   GPU if available.
2. Logs critic loss, generator loss, and estimated Wasserstein distance per epoch.
3. Saves the loss history as a CSV and a loss-curve plot.
4. Saves the generator and critic weights (best and last).
5. Supports early stop on NaN.
X_attack contains ONLY attack rows (no Normal). Use the training split only.
```
**Done when:** loss curves are smooth (no explosion), and weights are saved.

### Task 2.4: Sample generation with valid features

```text
In src/wgan_gp.py write generate_samples(generator, class_id, n, config,
onehot_groups) that:
1. Generates n samples for the given class.
2. For each one-hot column group (protocol_type, service, flag), converts the
   generated values to a valid one-hot vector using argmax.
3. Returns a numpy array in the same scaled space as the training data and the
   label array.
Add a check that every one-hot group has exactly one active column.
```
**Done when:** the validity check passes.

### Task 2.5: Oversampling target logic

```text
In src/resamplers.py write gan_oversample(X_train_attack, y_train_attack,
generator, config) that:
1. Counts samples per attack class.
2. Sets a target per class = target_ratio * (largest attack class count),
   where target_ratio comes from config (default 0.5).
3. Generates only the missing amount for classes below the target.
4. Returns the augmented X, y plus a boolean mask 'is_synthetic'.
5. Saves a before/after class count table.
Never use validation or test data here.
```
**Done when:** the table shows minority classes raised to the target.

### Task 2.6: GAN quality checks

```text
In src/gan_quality.py write these functions, each saving a table or figure:
1. tstr_score(X_syn, y_syn, X_real_test, y_real_test): train a RandomForest on
   synthetic data only, test on real data; return macro-F1 and per-class recall.
2. nearest_neighbor_distance(X_syn, X_real_train): for every synthetic sample,
   the distance to the closest real training sample; report mean, min and the
   percentage with distance < 1e-3 (possible memorization). Compare with the
   real-to-real nearest-neighbor distance.
3. ks_test_per_feature(X_syn, X_real): KS statistic per feature, report the
   mean and the count of features with p < 0.05.
4. pca_plot(X_real, X_syn, class_name): 2D PCA scatter, real vs synthetic.
Run all four per attack class and write one summary table 'gan_quality'.
```
**Done when:** the table is saved; check U2R for memorization.

---

# PHASE 3: Ensemble classifier

### Task 3.1: Soft-voting ensemble

```text
In src/ensemble.py write build_voting_ensemble(config, n_classes) returning a
scikit-learn VotingClassifier(voting='soft') with RandomForestClassifier,
XGBClassifier and LGBMClassifier. Take parameters from config, set random_state
from the seed, use n_jobs=-1, and handle both binary and multi-class cases
(for XGBoost use the right objective and eval_metric). Add a quick test fitting
on synthetic data.
```

### Task 3.2: Stacking ensemble

```text
In src/ensemble.py add build_stacking_ensemble(config, n_classes) returning a
StackingClassifier with the same three base models and a LogisticRegression
meta-learner, cv=5, stack_method='predict_proba'. Add a function
get_ensemble(kind, config, n_classes) that returns either 'voting' or 'stacking'.
```

### Task 3.3: Hyperparameter tuning on validation data

```text
In src/ensemble.py write tune_ensemble(X_train, y_train, X_val, y_val, config)
that runs RandomizedSearchCV-style tuning for each base model separately
(small search spaces for n_estimators, max_depth, learning_rate, num_leaves),
scoring with macro-F1 on the VALIDATION set only (use a PredefinedSplit).
Return the best params and save them as YAML in results/models.
Keep the search small (n_iter from config) so it runs in reasonable time.
```
**Done when:** best parameters are saved and reused by the other tasks.

---

# PHASE 4: Two-stage pipeline

### Task 4.1: Stage 1 binary model

```text
In src/pipeline.py write a class TwoStageIDS with:
- fit_stage1(X_train, y_train_bin): trains the binary ensemble.
- predict_proba_stage1(X): returns P(attack).
Use the get_ensemble function. Do not add oversampling to Stage 1 unless
config.stage1_balance is set (then use class_weight or random oversampling).
```

### Task 4.2: Threshold tuning

```text
In src/pipeline.py add tune_stage1_threshold(X_val, y_val_bin, target_recall)
that scans thresholds from 0.05 to 0.95 on the VALIDATION set and picks the
highest threshold that still gives attack recall >= target_recall (from config,
default 0.98). Save the full threshold-vs-recall/precision/FPR table and a plot.
Store the chosen threshold in the object.
```

### Task 4.3: Stage 2 attack-type model

```text
In src/pipeline.py add fit_stage2(X_train_attack, y_train_attack, use_gan,
generator=None): if use_gan is True, first augment with gan_oversample, then
train the multi-class ensemble. Encode class labels with a LabelEncoder saved
in the object. Record which training rows are synthetic.
Real training rows only come from the training split.
```

### Task 4.4: Full prediction and error propagation report

```text
In src/pipeline.py add:
1. predict(X): Stage 1 with the tuned threshold, then Stage 2 only on rows
   flagged as attack; returns string labels.
2. error_propagation_report(X, y_true_cat): table with, per true attack class,
   how many samples Stage 1 sent to Normal (missed), how many Stage 2
   misclassified, and how many were correct end to end.
Save the report as 'error_propagation'.
```
**Done when:** the report shows how many R2L and U2R samples are lost in Stage 1.

---

# PHASE 5: Baselines and experiment runner

### Task 5.1: Resamplers

```text
In src/resamplers.py add random_oversample(X, y, target_ratio) and
smote_oversample(X, y, target_ratio, k_neighbors) with the same target logic as
gan_oversample (target = ratio * largest class). Handle tiny classes safely:
if a class has fewer samples than k_neighbors + 1, reduce k_neighbors and log it.
All three resamplers must return the same output format
(X_aug, y_aug, is_synthetic).
```

### Task 5.2: Single-stage baselines

```text
In src/baselines.py write run_single_stage(method, data, config, seed) where
method is one of: 'none' (B1), 'ros' (B2), 'smote' (B3), 'gan_all' (B4).
It trains one 5-class ensemble on the training data (oversampled where needed;
for 'gan_all' the GAN is trained on ALL classes including Normal) and returns
predictions on the test set. Keep the ensemble settings identical across methods.
```

### Task 5.3: Two-stage baselines

```text
In src/baselines.py write run_two_stage(method, data, config, seed) where
method is 'ros' (B5), 'smote' (B6) or 'wgan' (Proposed). Reuse TwoStageIDS.
Return predictions on the test set and the saved Stage 1 threshold.
```

### Task 5.4: Experiment runner

```text
In src/run_experiments.py write a CLI (argparse) with --config, --methods and
--seeds. For each seed and each method (B1, B2, B3, B4, B5, B6, Proposed):
1. set_seed, rebuild the split, run the method.
2. Compute metrics with evaluate.compute_metrics.
3. Append one row per (seed, method) to results/tables/raw_results.csv,
   including training time and inference time.
Make it resumable: skip rows already in raw_results.csv. Log progress.
```
**Done when:** `python src/run_experiments.py --config configs/nslkdd.yaml --seeds 0` works for a single seed.

---

# PHASE 6: Evaluation and statistics

### Task 6.1: Metrics

```text
In src/evaluate.py write compute_metrics(y_true, y_pred, y_proba, classes)
returning a dict with: accuracy, macro precision/recall/F1, weighted F1, MCC,
macro AUC-PR (one-vs-rest, using y_proba), per-class precision/recall/F1,
and false positive rate on the Normal class. Also write
plot_confusion_matrix(y_true, y_pred, classes, title, config) with row-normalized
values and counts, and save it as PNG.
```

### Task 6.2: Aggregation and significance tests

```text
In src/stats_tests.py write:
1. summarize(raw_results.csv): mean ± std per method for each metric, saved as
   'summary_mean_std'.
2. wilcoxon_vs_best(raw_results.csv, metric): paired Wilcoxon signed-rank test
   between Proposed and every other method across seeds, with p-values saved
   as 'significance'. Warn if the number of paired seeds is below 6, because the
   test cannot reach p < 0.05 with so few pairs.
```

### Task 6.3: Efficiency table

```text
Write make_efficiency_table() in src/stats_tests.py that reads training time
and inference time (per sample, in ms) from raw_results.csv and makes a table
per method with mean ± std. Save it as 'efficiency'.
```

### Task 6.4: Paper-ready tables

```text
Write to_latex_tables() that converts summary_mean_std, significance and
efficiency CSV files into LaTeX tables (booktabs style) with the best value
in each column in bold. Save the .tex files in results/tables/latex.
```

---

# PHASE 7: More datasets

### Task 7.1: UNSW-NB15

```text
Create configs/unsw.yaml and add load_unsw(train_path, test_path) in
src/preprocess.py using the official UNSW_NB15_training-set.csv and
UNSW_NB15_testing-set.csv. Drop 'id', keep 'attack_cat' as the multi-class
label and 'label' as the binary label. One-hot encode proto, service, state,
scale numeric columns, and reuse build_dataset logic (generalize it so it works
from the config, not only NSL-KDD). Log the class distribution, and list classes
with fewer than 100 training samples (Worms, Shellcode, etc.).
```

### Task 7.2: CICIDS2017

```text
Create configs/cicids.yaml and add load_cicids(folder, sample_fraction) in
src/preprocess.py that:
1. Reads all daily CSV files, strips column-name spaces, and merges them.
2. Replaces inf with NaN and drops NaN rows and duplicate rows.
3. Optionally keeps a stratified sample (sample_fraction from config) for speed.
4. Groups classes with fewer than 100 samples into one class 'Other_Rare'
   (config-controlled) and logs exactly which classes were merged.
5. Makes a stratified 70/30 train/test split, then the validation split from train.
Generalize build_dataset to work for this dataset as well.
```
**Done when:** all three datasets run through the same experiment runner with only the config changed.

---

# PHASE 8: SHAP analysis (your novelty)

### Task 8.1: SHAP helper

```text
In src/shap_fidelity.py write get_mean_abs_shap(models, X, class_index, max_samples)
that, for a fitted ensemble, runs shap.TreeExplainer separately on the
RandomForest, XGBoost and LightGBM base models, handles the different output
shapes across SHAP versions (list of arrays vs 3D array), takes the mean
absolute SHAP per feature for the requested class, normalizes each model's
vector to sum to 1, and averages across the three models. Return a pandas
Series indexed by feature name. Use a random subsample of max_samples rows
for speed.
```

### Task 8.2: Claim 1, SHAP fidelity of synthetic data

```text
In src/shap_fidelity.py write run_fidelity(data, generator, config, seed):
1. Train a REFERENCE Stage 2 ensemble on REAL training attack data only
   (no synthetic samples), so the test is not circular.
2. For each minority attack class (R2L, U2R, and any other class below the
   target), build three sets: real held-out samples (use validation or test
   real samples of that class), WGAN-GP synthetic samples, SMOTE synthetic
   samples (and Random Oversampling as a trivial reference).
3. Compute mean-abs-SHAP importance vectors for each set with the reference model.
4. Compute fidelity scores between real and each synthetic set:
   Spearman rank correlation, top-10 Jaccard overlap, cosine similarity.
5. Use 1000 bootstrap resamples to give 95% confidence intervals (important
   because U2R is tiny).
6. Save the table 'shap_fidelity' (class x method x metric with CI) and a
   side-by-side bar plot of top-10 features for real vs WGAN-GP vs SMOTE for
   R2L and U2R.
Repeat over all seeds and report mean ± std.
```
**Done when:** you have a table and a figure that compare WGAN-GP and SMOTE.

### Task 8.3: Claim 2, stage-wise SHAP

```text
In src/shap_stagewise.py write run_stagewise(pipeline, data, config):
1. Stage 1: SHAP beeswarm summary plot (top 15 features) for Normal vs Attack,
   using the averaged tree-model SHAP values.
2. Stage 2: one SHAP summary bar plot per attack class.
3. A table 'stagewise_top_features' with the top-10 features of Stage 1 and
   of each Stage 2 class side by side, plus a column marking features that are
   top-10 in Stage 1 only or in Stage 2 only.
4. Spearman correlation between Stage 1 and each Stage 2 class importance
   vector, saved as 'stage_importance_overlap'.
```

### Task 8.4: Local explanations

```text
In src/shap_stagewise.py add explain_sample(pipeline, x, config) that, for one
sample, returns the Stage 1 prediction with a SHAP waterfall plot, and if it
is flagged as attack, the Stage 2 prediction with a SHAP waterfall plot.
Run it for: one correctly detected R2L sample, one correctly detected U2R sample,
and one Stage 1 miss (an attack classified as Normal). Save the figures and a
short text table of the top 5 contributing features for each case.
```

### Task 8.5: Claim 3, flat GAN vs attack-only GAN

```text
In src/baselines.py write compare_flat_vs_attack_only(data, config, seed) that
trains two WGAN-GPs: (a) on ALL classes including Normal (flat), (b) on attack
rows only. For each, record: class imbalance ratio seen by the GAN (largest :
smallest class), critic-loss stability (std of the last 20% of epochs),
TSTR macro-F1, nearest-neighbor memorization rate, final two-stage Macro-F1 and
minority-class recall, and GAN training time. Save a comparison table
'flat_vs_attack_only'. Run over all seeds.
```

---

# PHASE 9: Figures

### Task 9.1: Figure scripts

```text
Write src/make_figures.py that regenerates every paper figure from saved CSV
files (not by retraining):
1. Class distribution before and after oversampling.
2. WGAN-GP loss curves.
3. PCA of real vs synthetic for R2L and U2R.
4. Confusion matrices: Proposed vs best baseline.
5. Bar chart of per-class recall for all methods.
6. SHAP fidelity bar plot.
7. Stage-wise SHAP summaries.
Use a consistent style (font size 10, colour-blind-friendly palette, clear axis
labels, no titles inside figures because captions go in the paper).
Save as both PNG (300 dpi) and PDF.
```

### Task 9.2: Architecture diagram code

```text
Write a Graphviz or matplotlib script that draws the framework diagram:
Raw traffic -> Preprocessing -> Stage 1 (Normal/Attack) -> Normal: allow;
Attack -> Stage 2 (attack type) <- WGAN-GP oversampling (training only)
-> SHAP explanations for both stages. Clean boxes and arrows, white background.
```

---

# PHASE 10: Paper writing prompts

> Always give the AI your **real result tables** (paste the CSV contents). Tell it: *"Use only the numbers I give you."*

### Task 10.1: Methodology section

```text
Write the "Proposed Framework" section of an IEEE-style journal paper using
the description below. Include: overview of the two-stage design, the WGAN-GP
objective with equations in LaTeX (Wasserstein loss and gradient penalty),
how the conditional GAN is trained on attack rows only, the ensemble
(soft-voting/stacking), threshold selection for Stage 1, and the SHAP fidelity
metric definition (Spearman, top-k Jaccard, cosine similarity).
Formal academic tone, no bullet points in the final text, no claims that are
not in the description.

[paste your method description here]
```

### Task 10.2: Experimental setup

```text
Write the "Experimental Setup" section using these facts only: datasets and
their splits, preprocessing, hyperparameter table, baselines B1-B6 and Proposed,
metrics (Macro-F1, MCC, AUC-PR, per-class recall, FPR), 10 seeds, statistical
test, hardware.

[paste dataset stats, config values, and hardware here]
```

### Task 10.3: Results and discussion

```text
Write the "Results and Discussion" section using ONLY the tables below.
Describe the main comparison, the ablation (what each component adds),
per-class results for R2L and U2R, significance test outcomes, SHAP fidelity,
stage-wise explanations, and efficiency. If a baseline wins on any metric, say
so honestly. Do not add numbers that are not in the tables.

[paste summary_mean_std, significance, shap_fidelity, flat_vs_attack_only, efficiency tables]
```

### Task 10.4: Related work

```text
Write the "Related Work" section in three groups:
(1) GAN-based oversampling for IDS, (2) ensemble learning for IDS,
(3) hierarchical / two-stage IDS, ending with a short paragraph on the gap:
none of them combine a two-stage design, GAN oversampling, tree-ensemble
classification and SHAP-based validation of synthetic data.
Use the papers I list. Use "to the best of our knowledge". Do NOT add any
citation that is not in my list.

[paste your verified reference list with one-line summaries]
```

### Task 10.5: Introduction, abstract, conclusion

```text
Using the finished sections below, write: (a) the Introduction (problem,
gap, contributions list of three items, paper organization), (b) an Abstract
of at most 200 words with one headline result taken from the results table,
(c) a Conclusion with limitations and future work (unseen attacks, real-time
deployment, small-class GAN overfitting).

[paste your methodology and results sections]
```

### Task 10.6: Self-review

```text
Act as a strict reviewer for Computers & Security. Read the paper below and list:
(1) the 10 weakest points, (2) missing experiments, (3) claims not supported by
the results, (4) unclear writing. Be specific and give a fix for each point.

[paste full paper]
```

---

# PHASE 11: Release

### Task 11.1: Reproducibility

```text
Create: (1) a Makefile with targets setup, preprocess, train, evaluate, shap,
figures, all; (2) a run_all.sh that runs the full pipeline for one dataset
given a config; (3) a short README section with exact commands, expected
runtime, and the Python and library versions. Add a seed list and say that all
reported numbers are mean ± std over those seeds.
```

### Task 11.2: Code review pass

```text
Review the whole repository for: data leakage (scaling or oversampling before
the split, tuning on the test set), hard-coded paths, missing seeds, silent
exceptions, and inconsistent hyperparameters between baselines. List each issue
with the file and line, then fix them one by one without changing behaviour
otherwise.
```

---

## Checklist

- [ ] Scaffold, utils, config
- [ ] NSL-KDD preprocessing, no leakage
- [ ] WGAN-GP stable, quality checks done
- [ ] Ensemble tuned on validation
- [ ] Two-stage pipeline and error propagation report
- [ ] Baselines B1-B6 and Proposed, 10 seeds
- [ ] Metrics, significance tests, efficiency table
- [ ] UNSW-NB15 and CICIDS2017
- [ ] SHAP fidelity (Claim 1)
- [ ] Stage-wise SHAP (Claim 2)
- [ ] Flat vs attack-only GAN (Claim 3)
- [ ] Figures generated from CSV
- [ ] Paper drafted, references verified from the original sources
- [ ] Plagiarism check, supervisor review
- [ ] Code on GitHub

## Common problems and quick prompts

| Problem | Prompt to use |
|---|---|
| Code crashes | `Here is the full error: [paste]. Fix only this error, show the changed lines.` |
| GAN loss explodes | `My WGAN-GP critic loss diverges. Here are my losses and code: [paste]. Suggest fixes in order of likelihood and apply the first one.` |
| SHAP shape errors | `shap_values has shape [..] for my installed version [x]. Make the code handle both list and 3D array outputs.` |
| Too slow | `Profile this function and make it faster without changing results. Suggest subsampling or n_jobs changes.` |
| Suspect leakage | `Check this code for data leakage and tell me exactly which line is wrong.` |
| AI changed too much | `Undo the unrelated changes. Only modify the function I asked about.` |

## Important rules for your paper

1. Do not call the two-stage design new. Cite Tauscher et al. (2021) and similar work, and claim the combination plus the SHAP-based contributions.
2. Use "to the best of our knowledge" for every novelty claim, and search Google Scholar before writing it.
3. If a claim is not supported by your results, change the claim. Do not change the results.
4. Verify every reference (title, venue, year, pages) from the original source.
5. U2R has very few training samples, so check the GAN for memorization and report U2R results with confidence intervals.
