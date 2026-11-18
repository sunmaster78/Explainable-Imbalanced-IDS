"""
Script to generate quantitative evidence and publication figures for Novelty Claim 2:
Stage-Wise SHAP Deconvolution.
"""

from pathlib import Path
import torch

from src.preprocess import build_dataset, load_nslkdd, map_attack_categories
from src.pipeline import TwoStageIDS
from src.shap_stagewise import run_stagewise
from src.make_figures import plot_stagewise_feature_roles
from src.utils import get_logger, load_config, set_seed
from src.wgan_gp import Generator

logger = get_logger("claim2_evidence")

def main():
    config = load_config("configs/nslkdd.yaml")
    seed = 0
    set_seed(seed)

    logger.info("Loading and preprocessing NSL-KDD dataset...")
    train_raw, test_raw = load_nslkdd(
        Path(config["dataset"]["raw_train_path"]),
        Path(config["dataset"]["raw_test_path"])
    )
    train_mapped = map_attack_categories(train_raw)
    test_mapped = map_attack_categories(test_raw)
    dataset = build_dataset(train_mapped, test_mapped, config)

    logger.info("Initializing TwoStageIDS pipeline...")
    pipeline = TwoStageIDS(config=config, ensemble_kind="voting", seed=seed)

    # 1. Fit Stage 1
    logger.info("Fitting Stage 1 binary classifier...")
    pipeline.fit_stage1(dataset["X_train"], dataset["y_train_bin"])

    # 2. Tune Stage 1 threshold
    logger.info("Tuning Stage 1 threshold on validation set...")
    pipeline.tune_stage1_threshold(dataset["X_val"], dataset["y_val_bin"])

    # 3. Load pre-trained WGAN-GP generator
    logger.info("Loading pre-trained WGAN-GP generator...")
    n_features = dataset["X_train"].shape[1]
    n_attack_classes = len(config["attack_classes"])
    generator = Generator(
        latent_dim=config["gan"]["latent_dim"],
        n_classes=n_attack_classes,
        output_dim=n_features,
        hidden_layers=config["gan"]["gen_hidden_layers"],
    )
    ckpt = torch.load("results/models/wgan_generator_best.pt", map_location="cpu", weights_only=False)
    generator.load_state_dict(ckpt)
    generator.eval()

    # 4. Fit Stage 2 with WGAN generator
    logger.info("Fitting Stage 2 multiclass attack classifier...")
    attack_mask = dataset["y_train_bin"] == 1
    X_train_att = dataset["X_train"][attack_mask]
    y_train_att = dataset["y_train_cat"][attack_mask]

    pipeline.fit_stage2(
        X_train_attack=X_train_att,
        y_train_attack=y_train_att,
        use_gan=True,
        generator=generator,
        onehot_groups=dataset.get("onehot_column_indices", {}),
    )

    logger.info("Pipeline ready. Running Novelty Claim 2 Stage-Wise SHAP audit...")
    stagewise_res = run_stagewise(
        pipeline=pipeline,
        data=dataset,
        config=config,
        max_samples=150,
        seed=seed,
    )

    logger.info("Generating publication figure: fig5_stagewise_feature_roles...")
    plot_stagewise_feature_roles(
        tables_dir=Path("results/tables"),
        figures_dir=Path("results/figures"),
    )

    logger.info("Claim 2 evidence generation completed successfully!")

if __name__ == "__main__":
    main()

