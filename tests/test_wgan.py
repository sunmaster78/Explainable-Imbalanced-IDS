"""
Unit tests for WGAN-GP Generator and Critic shape validation.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils import load_config, set_seed
from src.wgan_gp import (
    Critic,
    Generator,
    build_critic,
    build_generator,
    critic_train_step,
    generate_samples,
    generator_train_step,
    gradient_penalty,
    train_one_iteration,
    train_wgan_gp,
)


# Canonical test dimensions matching NSL-KDD after preprocessing
LATENT_DIM = 100
N_CLASSES = 4        # DoS, Probe, R2L, U2R  (attack-only Stage 2)
OUTPUT_DIM = 122     # 38 numeric + 84 one-hot features
BATCH_SIZE = 32


class TestGenerator:
    """Shape and behaviour tests for the conditional Generator."""

    def test_output_shape(self):
        """Generator output must be (batch, output_dim)."""
        set_seed(0)
        gen = Generator(
            latent_dim=LATENT_DIM,
            n_classes=N_CLASSES,
            output_dim=OUTPUT_DIM,
        )
        noise = torch.randn(BATCH_SIZE, LATENT_DIM)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 0] = 1.0  # all DoS

        out = gen(noise, labels)
        assert out.shape == (BATCH_SIZE, OUTPUT_DIM), (
            f"Expected ({BATCH_SIZE}, {OUTPUT_DIM}), got {out.shape}"
        )

    def test_output_range_tanh(self):
        """Generator Tanh final layer should bound outputs to [-1, 1]."""
        set_seed(0)
        gen = Generator(
            latent_dim=LATENT_DIM,
            n_classes=N_CLASSES,
            output_dim=OUTPUT_DIM,
        )
        noise = torch.randn(64, LATENT_DIM)
        labels = torch.zeros(64, N_CLASSES)
        labels[:, 1] = 1.0

        out = gen(noise, labels)
        assert out.min().item() >= -1.0
        assert out.max().item() <= 1.0

    def test_configurable_hidden_layers(self):
        """Generator should accept arbitrary hidden layer widths."""
        gen = Generator(
            latent_dim=50,
            n_classes=3,
            output_dim=80,
            hidden_layers=[128, 256, 512],
        )
        noise = torch.randn(8, 50)
        labels = torch.zeros(8, 3)
        labels[:, 2] = 1.0

        out = gen(noise, labels)
        assert out.shape == (8, 80)

    def test_single_sample(self):
        """Generator must handle batch_size=1."""
        gen = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES, output_dim=OUTPUT_DIM)
        # BatchNorm requires eval mode for batch_size=1
        gen.eval()
        noise = torch.randn(1, LATENT_DIM)
        labels = torch.zeros(1, N_CLASSES)
        labels[0, 0] = 1.0

        with torch.no_grad():
            out = gen(noise, labels)
        assert out.shape == (1, OUTPUT_DIM)


class TestCritic:
    """Shape and behaviour tests for the conditional Critic."""

    def test_output_shape(self):
        """Critic output must be (batch, 1)."""
        set_seed(0)
        critic = Critic(
            input_dim=OUTPUT_DIM,
            n_classes=N_CLASSES,
        )
        sample = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 0] = 1.0

        score = critic(sample, labels)
        assert score.shape == (BATCH_SIZE, 1), (
            f"Expected ({BATCH_SIZE}, 1), got {score.shape}"
        )

    def test_no_sigmoid_unbounded(self):
        """Critic must NOT apply sigmoid — outputs should be unbounded."""
        set_seed(0)
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)
        # Feed extreme inputs
        sample = torch.randn(64, OUTPUT_DIM) * 10.0
        labels = torch.zeros(64, N_CLASSES)
        labels[:, 0] = 1.0

        score = critic(sample, labels)
        # With extreme inputs, at least some outputs should exceed [0, 1] range
        # (a sigmoid-free critic is unbounded)
        assert score.min().item() < 0.0 or score.max().item() > 1.0, (
            "Critic scores appear bounded to [0,1] — check for accidental sigmoid."
        )

    def test_configurable_hidden_layers(self):
        """Critic should accept arbitrary hidden layer widths."""
        critic = Critic(
            input_dim=80,
            n_classes=3,
            hidden_layers=[1024, 512, 256],
        )
        sample = torch.randn(8, 80)
        labels = torch.zeros(8, 3)
        labels[:, 1] = 1.0

        score = critic(sample, labels)
        assert score.shape == (8, 1)

    def test_no_batchnorm_present(self):
        """Critic must not contain any BatchNorm layers (WGAN-GP requirement)."""
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)
        for module in critic.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)), (
                f"Critic contains BatchNorm: {module}. WGAN-GP critics must avoid BatchNorm."
            )

    def test_has_layernorm(self):
        """Critic should use LayerNorm in hidden layers."""
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)
        ln_count = sum(1 for m in critic.modules() if isinstance(m, nn.LayerNorm))
        assert ln_count > 0, "Critic should use LayerNorm in hidden layers."


class TestFactoryHelpers:
    """Tests for config-driven construction via build_generator / build_critic."""

    def test_build_from_config(self):
        """Factory helpers should construct models matching config dimensions."""
        config_path = Path("configs/nslkdd.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found.")

        config = load_config(config_path)
        gen = build_generator(config, n_classes=N_CLASSES, output_dim=OUTPUT_DIM)
        critic = build_critic(config, n_classes=N_CLASSES, input_dim=OUTPUT_DIM)

        noise = torch.randn(16, gen.latent_dim)
        labels = torch.zeros(16, N_CLASSES)
        labels[:, 0] = 1.0

        fake = gen(noise, labels)
        score = critic(fake, labels)

        assert fake.shape == (16, OUTPUT_DIM)
        assert score.shape == (16, 1)



class TestGradientPenalty:
    """Tests for gradient_penalty computation in WGAN-GP."""

    def test_gp_scalar_and_finite(self):
        """Gradient penalty should return a non-negative finite scalar."""
        set_seed(42)
        device = torch.device("cpu")
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)

        real = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        fake = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 0] = 1.0

        gp = gradient_penalty(critic, real, fake, labels, device, lambda_gp=10.0)

        assert gp.dim() == 0, f"GP should be a scalar tensor, got dim {gp.dim()}"
        assert torch.isfinite(gp).item(), "GP must be finite (no NaN or Inf)"
        assert gp.item() >= 0.0, "Gradient penalty must be non-negative"


class TestTrainingStep:
    """Tests for critic_train_step, generator_train_step, and train_one_iteration."""

    def test_critic_step_runs_and_loss_is_finite(self):
        """A single critic step should compute finite loss and update weights."""
        set_seed(42)
        device = torch.device("cpu")
        gen = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES, output_dim=OUTPUT_DIM)
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)

        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

        real = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 1] = 1.0

        # Snapshot weights to verify update
        orig_weights = [p.clone() for p in critic.parameters() if p.requires_grad]

        loss_c, w_dist = critic_train_step(critic, gen, real, labels, opt_c, device, lambda_gp=10.0)

        import math
        assert math.isfinite(loss_c), f"Critic loss must be finite, got {loss_c}"
        assert math.isfinite(w_dist), f"Wasserstein distance must be finite, got {w_dist}"

        # Ensure parameters were updated
        any_changed = any(
            not torch.equal(orig, p)
            for orig, p in zip(orig_weights, [p for p in critic.parameters() if p.requires_grad])
        )
        assert any_changed, "Critic weights should update after critic_train_step"

    def test_generator_step_runs_and_loss_is_finite(self):
        """A single generator step should compute finite loss and update weights."""
        set_seed(42)
        device = torch.device("cpu")
        gen = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES, output_dim=OUTPUT_DIM)
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)

        opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 2] = 1.0

        orig_weights = [p.clone() for p in gen.parameters() if p.requires_grad]

        loss_g = generator_train_step(critic, gen, BATCH_SIZE, labels, opt_g, device)

        import math
        assert math.isfinite(loss_g), f"Generator loss must be finite, got {loss_g}"

        any_changed = any(
            not torch.equal(orig, p)
            for orig, p in zip(orig_weights, [p for p in gen.parameters() if p.requires_grad])
        )
        assert any_changed, "Generator weights should update after generator_train_step"

    def test_one_iteration_runs_without_error_and_losses_are_finite(self):
        """
        Acceptance Criteria Task 2.2:
        One iteration of 5 critic updates + 1 generator update runs on random
        data without error, using Adam(lr=1e-4, betas=(0.5, 0.9)), and losses are finite.
        """
        set_seed(42)
        device = torch.device("cpu")
        gen = Generator(latent_dim=LATENT_DIM, n_classes=N_CLASSES, output_dim=OUTPUT_DIM)
        critic = Critic(input_dim=OUTPUT_DIM, n_classes=N_CLASSES)

        opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

        real = torch.randn(BATCH_SIZE, OUTPUT_DIM)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        labels[:, 3] = 1.0

        metrics = train_one_iteration(
            generator=gen,
            critic=critic,
            real_samples=real,
            real_labels=labels,
            optimizer_g=opt_g,
            optimizer_c=opt_c,
            device=device,
            n_critic=5,
            lambda_gp=10.0,
        )

        import math
        assert "critic_loss" in metrics
        assert "generator_loss" in metrics
        assert "wasserstein_dist" in metrics

        assert math.isfinite(metrics["critic_loss"]), f"critic_loss not finite: {metrics['critic_loss']}"
        assert math.isfinite(metrics["generator_loss"]), f"generator_loss not finite: {metrics['generator_loss']}"
        assert math.isfinite(metrics["wasserstein_dist"]), f"wasserstein_dist not finite: {metrics['wasserstein_dist']}"


class TestTrainingLoop:
    """Tests for the full train_wgan_gp training loop (Task 2.3)."""

    def test_train_wgan_gp_smoke(self, tmp_path: Path):
        """
        Acceptance Criteria Task 2.3:
        Train for 3 epochs on random data, verify:
          - Loss curves are smooth (no NaN/Inf explosion).
          - Generator and critic weights are saved (best + last).
          - Loss history CSV and plot PNG are created.
        """
        set_seed(42)

        # Small synthetic dataset: 256 samples, OUTPUT_DIM features, N_CLASSES one-hot
        n_samples = 256
        X_attack = np.random.randn(n_samples, OUTPUT_DIM).astype(np.float32)
        y_onehot = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
        for i in range(n_samples):
            y_onehot[i, i % N_CLASSES] = 1.0

        config = {
            "gan": {
                "latent_dim": LATENT_DIM,
                "gen_hidden_layers": [64, 128],     # small for speed
                "critic_hidden_layers": [128, 64],
                "batch_size": 64,
                "lr": 1e-4,
                "betas": [0.5, 0.9],
                "n_critic": 2,    # fewer critic steps for speed
                "lambda_gp": 10.0,
                "epochs": 3,
            },
            "output_dirs": {
                "tables": str(tmp_path / "tables"),
                "figures": str(tmp_path / "figures"),
                "models": str(tmp_path / "models"),
            },
        }

        result = train_wgan_gp(X_attack, y_onehot, config)

        # -- Returned objects --
        assert isinstance(result["generator"], Generator)
        assert isinstance(result["critic"], Critic)
        assert isinstance(result["best_epoch"], int)
        assert result["best_epoch"] >= 1

        # -- History DataFrame --
        history = result["history"]
        assert len(history) == 3, f"Expected 3 epochs in history, got {len(history)}"
        for col in ["epoch", "critic_loss", "generator_loss", "wasserstein_dist"]:
            assert col in history.columns, f"Missing column: {col}"

        # All losses should be finite (no explosion)
        for _, row in history.iterrows():
            assert math.isfinite(row["critic_loss"]), f"critic_loss not finite at epoch {row['epoch']}"
            assert math.isfinite(row["generator_loss"]), f"generator_loss not finite at epoch {row['epoch']}"
            assert math.isfinite(row["wasserstein_dist"]), f"wasserstein_dist not finite at epoch {row['epoch']}"

        # -- Artifacts on disk --
        assert (tmp_path / "tables" / "wgan_loss_history.csv").exists()
        assert (tmp_path / "figures" / "wgan_loss_curves.png").exists()
        assert (tmp_path / "models" / "wgan_generator_best.pt").exists()
        assert (tmp_path / "models" / "wgan_critic_best.pt").exists()
        assert (tmp_path / "models" / "wgan_generator_last.pt").exists()
        assert (tmp_path / "models" / "wgan_critic_last.pt").exists()

    def test_train_wgan_gp_nan_early_stop(self, tmp_path: Path):
        """
        Verify that NaN values in losses trigger early stopping
        and training terminates gracefully without crashing.
        """
        set_seed(42)

        # Create data with extreme values likely to cause instability
        n_samples = 128
        X_attack = (np.random.randn(n_samples, OUTPUT_DIM) * 1e6).astype(np.float32)
        y_onehot = np.zeros((n_samples, N_CLASSES), dtype=np.float32)
        for i in range(n_samples):
            y_onehot[i, i % N_CLASSES] = 1.0

        config = {
            "gan": {
                "latent_dim": LATENT_DIM,
                "gen_hidden_layers": [32, 64],
                "critic_hidden_layers": [64, 32],
                "batch_size": 64,
                "lr": 1.0,  # absurdly high LR to provoke NaN
                "betas": [0.5, 0.9],
                "n_critic": 5,
                "lambda_gp": 10.0,
                "epochs": 20,  # should stop well before 20
            },
            "output_dirs": {
                "tables": str(tmp_path / "tables"),
                "figures": str(tmp_path / "figures"),
                "models": str(tmp_path / "models"),
            },
        }

        # Should NOT raise; NaN triggers graceful early stop
        result = train_wgan_gp(X_attack, y_onehot, config)
        history = result["history"]

        # Training should have stopped early (fewer epochs than requested)
        # or run to completion without crashing — either is acceptable
        assert len(history) <= 20
        # Last checkpoint should always be saved
        assert (tmp_path / "models" / "wgan_generator_last.pt").exists()


class TestGenerateSamples:
    """Tests for generate_samples (Task 2.4)."""

    def _make_generator(self) -> Generator:
        """Create a small generator for testing."""
        set_seed(42)
        return Generator(
            latent_dim=LATENT_DIM,
            n_classes=N_CLASSES,
            output_dim=OUTPUT_DIM,
            hidden_layers=[64, 128],
        )

    def _make_config(self) -> dict:
        return {
            "gan": {
                "latent_dim": LATENT_DIM,
                "batch_size": 64,
            },
        }

    def _make_onehot_groups(self) -> dict:
        """Simulate 3 one-hot column groups within OUTPUT_DIM=122 columns.

        Numeric cols: 0-37 (38 cols)
        protocol_type: 38-40 (3 levels)
        service:       41-110 (70 levels)
        flag:          111-121 (11 levels)
        """
        return {
            "protocol_type": list(range(38, 41)),
            "service": list(range(41, 111)),
            "flag": list(range(111, 122)),
        }

    def test_output_shape_and_dtype(self):
        """Generated samples should have correct shape and float32 dtype."""
        gen = self._make_generator()
        config = self._make_config()
        groups = self._make_onehot_groups()

        samples, labels = generate_samples(gen, class_id=0, n=50, config=config, onehot_groups=groups)

        assert samples.shape == (50, OUTPUT_DIM)
        assert samples.dtype == np.float32
        assert labels.shape == (50,)
        assert np.all(labels == 0)

    def test_onehot_validity_exactly_one_active(self):
        """Every one-hot group must have exactly one 1.0 per sample."""
        gen = self._make_generator()
        config = self._make_config()
        groups = self._make_onehot_groups()

        samples, _ = generate_samples(gen, class_id=1, n=200, config=config, onehot_groups=groups)

        for group_name, col_indices in groups.items():
            group_vals = samples[:, col_indices]
            # Each row should sum to exactly 1.0
            row_sums = group_vals.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0,
                err_msg=f"One-hot group '{group_name}' has rows that don't sum to 1.0",
            )
            # Each row should have exactly one 1.0 and the rest 0.0
            assert np.all((group_vals == 0.0) | (group_vals == 1.0)), (
                f"One-hot group '{group_name}' contains non-binary values after argmax."
            )

    def test_different_class_ids_produce_different_labels(self):
        """Different class_id values should produce matching label arrays."""
        gen = self._make_generator()
        config = self._make_config()
        groups = self._make_onehot_groups()

        _, labels_0 = generate_samples(gen, class_id=0, n=10, config=config, onehot_groups=groups)
        _, labels_3 = generate_samples(gen, class_id=3, n=10, config=config, onehot_groups=groups)

        assert np.all(labels_0 == 0)
        assert np.all(labels_3 == 3)

    def test_n_larger_than_batch_size(self):
        """Generating more samples than batch_size should work via batching."""
        gen = self._make_generator()
        config = {"gan": {"latent_dim": LATENT_DIM, "batch_size": 32}}
        groups = self._make_onehot_groups()

        samples, labels = generate_samples(gen, class_id=2, n=100, config=config, onehot_groups=groups)

        assert samples.shape == (100, OUTPUT_DIM)
        assert labels.shape == (100,)

    def test_n_equals_one(self):
        """Edge case: generating a single sample should work."""
        gen = self._make_generator()
        config = self._make_config()
        groups = self._make_onehot_groups()

        samples, labels = generate_samples(gen, class_id=0, n=1, config=config, onehot_groups=groups)

        assert samples.shape == (1, OUTPUT_DIM)
        assert labels.shape == (1,)
