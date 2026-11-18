"""
Script to generate quantitative evidence and publication figures for Novelty Claim 3:
Stage-Aware GAN (Attack-Only) vs Flat GAN (All Classes).
"""

from pathlib import Path
import pandas as pd

from src.preprocess import build_dataset, load_nslkdd, map_attack_categories
from src.baselines import compare_flat_vs_attack_only
from src.make_figures import plot_flat_vs_attack_only_comparison
from src.utils import get_logger, load_config, set_seed

logger = get_logger("claim3_evidence")

def main():
    config = load_config("configs/nslkdd.yaml")
    seed = 0
    set_seed(seed)

    # Use 15 epochs and batch_size 256 for balanced training and stability audit
    config["gan"]["epochs"] = 15
    config["gan"]["batch_size"] = 256

    logger.info("Loading and preprocessing NSL-KDD dataset...")
    train_raw, test_raw = load_nslkdd(
        Path(config["dataset"]["raw_train_path"]),
        Path(config["dataset"]["raw_test_path"])
    )
    train_mapped = map_attack_categories(train_raw)
    test_mapped = map_attack_categories(test_raw)
    dataset = build_dataset(train_mapped, test_mapped, config)

    logger.info("Running Claim 3 Comparison: Flat GAN vs Attack-Only GAN...")
    df_comp = compare_flat_vs_attack_only(
        data=dataset,
        config=config,
        seed=seed,
        table_name="flat_vs_attack_only"
    )

    logger.info(f"Comparison results:\n{df_comp.to_string()}")

    logger.info("Generating publication figure: fig6_flat_vs_attack_only...")
    plot_flat_vs_attack_only_comparison(
        tables_dir=Path("results/tables"),
        figures_dir=Path("results/figures"),
        csv_name="flat_vs_attack_only.csv"
    )

    logger.info("Claim 3 evidence generation completed successfully!")

if __name__ == "__main__":
    main()

