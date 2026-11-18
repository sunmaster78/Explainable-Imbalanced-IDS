"""
Conditional WGAN-GP (Wasserstein GAN with Gradient Penalty) for the
Two-Stage IDS pipeline.

This module provides the Generator and Critic (Discriminator) architectures
used to synthesise realistic minority-class attack samples. Both models accept
a class-conditioning vector (one-hot encoded attack category) concatenated to
their primary input.

Architecture choices follow WGAN-GP best practices:
  - Generator: BatchNorm in hidden layers, Tanh final activation.
  - Critic: NO BatchNorm (LayerNorm used instead), NO sigmoid output,
            produces an unbounded scalar score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils import get_logger, save_figure, save_table

logger = get_logger("wgan_gp")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Conditional Generator for WGAN-GP.

    Input:  noise vector (latent_dim) concatenated with a one-hot class vector (n_classes).
    Output: synthetic sample of dimension `output_dim`.

    Hidden layers use LeakyReLU(0.2) + BatchNorm.  Final layer uses Tanh
    (matching the [-1, 1] range of MinMaxScaler-scaled features).

    Args:
        latent_dim:    Dimension of the noise input vector.
        n_classes:     Number of conditioning classes (one-hot width).
        output_dim:    Dimension of the generated feature vector.
        hidden_layers: List of hidden-layer widths (default [256, 512]).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_classes: int = 4,
        output_dim: int = 122,
        hidden_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 512]

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = latent_dim + n_classes

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h_dim

        # Final projection -> output_dim with Tanh
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

        # Weight initialisation (Xavier uniform)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            noise:  (batch, latent_dim)  random noise tensor.
            labels: (batch, n_classes)   one-hot class condition tensor.

        Returns:
            Generated samples of shape (batch, output_dim).
        """
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)


# ---------------------------------------------------------------------------
# Critic (Discriminator)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Conditional Critic (Discriminator) for WGAN-GP.

    Input:  feature sample (input_dim) concatenated with a one-hot class vector (n_classes).
    Output: unbounded scalar realness score (NO sigmoid).

    Hidden layers use LeakyReLU(0.2) + LayerNorm.  BatchNorm is deliberately
    avoided because the gradient penalty requires per-sample gradients.

    Args:
        input_dim:     Dimension of the input feature vector.
        n_classes:     Number of conditioning classes (one-hot width).
        hidden_layers: List of hidden-layer widths (default [512, 256]).
    """

    def __init__(
        self,
        input_dim: int = 122,
        n_classes: int = 4,
        hidden_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256]

        self.input_dim = input_dim
        self.n_classes = n_classes

        layers: list[nn.Module] = []
        in_dim = input_dim + n_classes

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h_dim

        # Final projection -> scalar, NO activation
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sample: (batch, input_dim)  feature vector.
            labels: (batch, n_classes)  one-hot class condition tensor.

        Returns:
            Critic scores of shape (batch, 1).
        """
        x = torch.cat([sample, labels], dim=1)
        return self.model(x)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_generator(config: Dict[str, Any], n_classes: int, output_dim: int) -> Generator:
    """
    Construct a Generator from a YAML-loaded config dictionary.

    Expected config keys (under 'gan'):
        latent_dim, gen_hidden_layers
    """
    gan_cfg = config.get("gan", {})
    return Generator(
        latent_dim=gan_cfg.get("latent_dim", 100),
        n_classes=n_classes,
        output_dim=output_dim,
        hidden_layers=gan_cfg.get("gen_hidden_layers", [256, 512]),
    )


def build_critic(config: Dict[str, Any], n_classes: int, input_dim: int) -> Critic:
    """
    Construct a Critic from a YAML-loaded config dictionary.

    Expected config keys (under 'gan'):
        critic_hidden_layers
    """
    gan_cfg = config.get("gan", {})
    return Critic(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_layers=gan_cfg.get("critic_hidden_layers", [512, 256]),
    )


# ---------------------------------------------------------------------------
# Gradient Penalty (WGAN-GP)
# ---------------------------------------------------------------------------

def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute the gradient penalty for WGAN-GP to enforce the 1-Lipschitz constraint.

    Interpolates between real and fake samples, passes through the critic, and
    penalises gradient norms that deviate from 1.

    Args:
        critic:    The Critic network.
        real:      Real samples, shape (batch, feature_dim).
        fake:      Generated (fake) samples, shape (batch, feature_dim).
        labels:    One-hot class conditioning, shape (batch, n_classes).
        device:    Torch device (cpu / cuda).
        lambda_gp: Gradient penalty coefficient (default 10.0).

    Returns:
        Scalar gradient penalty loss.
    """
    batch_size = real.size(0)

    # Random interpolation coefficient per sample
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # Critic score on interpolated samples
    d_interpolated = critic(interpolated, labels)

    # Compute gradients w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # L2 norm per sample, then penalise deviation from 1
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()

    return gp


# ---------------------------------------------------------------------------
# Single Training Step Functions
# ---------------------------------------------------------------------------

def critic_train_step(
    critic: Critic,
    generator: Generator,
    real_samples: torch.Tensor,
    real_labels: torch.Tensor,
    optimizer_c: optim.Optimizer,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tuple[float, float]:
    """
    Perform one critic training step.

    Critic loss = E[critic(fake)] - E[critic(real)] + gradient_penalty

    The estimated Wasserstein distance is E[critic(real)] - E[critic(fake)]
    (negation of the first two terms of the loss, before GP).

    Args:
        critic:        Critic network.
        generator:     Generator network.
        real_samples:  Real feature samples (batch, feature_dim).
        real_labels:   One-hot class labels (batch, n_classes).
        optimizer_c:   Critic optimizer.
        device:        Torch device.
        lambda_gp:     GP coefficient.

    Returns:
        Tuple of (critic_loss, wasserstein_distance_estimate).
    """
    batch_size = real_samples.size(0)

    # Generate fake samples (detached from generator graph)
    noise = torch.randn(batch_size, generator.latent_dim, device=device)
    fake_samples = generator(noise, real_labels).detach()

    # Critic scores
    score_real = critic(real_samples, real_labels)
    score_fake = critic(fake_samples, real_labels)

    # Gradient penalty
    gp = gradient_penalty(critic, real_samples, fake_samples, real_labels, device, lambda_gp)

    # Critic loss: maximize E[D(real)] - E[D(fake)] → minimize E[D(fake)] - E[D(real)] + GP
    loss_c = score_fake.mean() - score_real.mean() + gp

    optimizer_c.zero_grad()
    loss_c.backward()
    optimizer_c.step()

    # Wasserstein distance estimate (without GP)
    w_dist = score_real.mean().item() - score_fake.mean().item()

    return loss_c.item(), w_dist


def generator_train_step(
    critic: Critic,
    generator: Generator,
    batch_size: int,
    labels: torch.Tensor,
    optimizer_g: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Perform one generator training step.

    Generator loss = -E[critic(G(z, y), y)]

    Args:
        critic:      Critic network.
        generator:   Generator network.
        batch_size:  Number of samples to generate.
        labels:      One-hot class labels (batch, n_classes).
        optimizer_g: Generator optimizer.
        device:      Torch device.

    Returns:
        Generator loss (float).
    """
    noise = torch.randn(batch_size, generator.latent_dim, device=device)
    fake_samples = generator(noise, labels)

    score_fake = critic(fake_samples, labels)
    loss_g = -score_fake.mean()

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    return loss_g.item()


def train_one_iteration(
    generator: Generator,
    critic: Critic,
    real_samples: torch.Tensor,
    real_labels: torch.Tensor,
    optimizer_g: optim.Optimizer,
    optimizer_c: optim.Optimizer,
    device: torch.device,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
) -> Dict[str, float]:
    """
    Execute one full WGAN-GP iteration: n_critic critic updates followed by
    one generator update.

    Args:
        generator:    Generator network.
        critic:       Critic network.
        real_samples: Real feature samples (batch, feature_dim).
        real_labels:  One-hot class labels (batch, n_classes).
        optimizer_g:  Generator optimizer.
        optimizer_c:  Critic optimizer.
        device:       Torch device.
        n_critic:     Number of critic updates per generator update (default 5).
        lambda_gp:    Gradient penalty coefficient (default 10.0).

    Returns:
        Dict with keys 'critic_loss', 'generator_loss', 'wasserstein_dist'.
    """
    # --- Critic updates ---
    critic_losses = []
    w_dists = []
    for _ in range(n_critic):
        c_loss, w_dist = critic_train_step(
            critic, generator, real_samples, real_labels,
            optimizer_c, device, lambda_gp,
        )
        critic_losses.append(c_loss)
        w_dists.append(w_dist)

    # --- Generator update ---
    batch_size = real_samples.size(0)
    g_loss = generator_train_step(
        critic, generator, batch_size, real_labels, optimizer_g, device,
    )

    return {
        "critic_loss": sum(critic_losses) / len(critic_losses),
        "generator_loss": g_loss,
        "wasserstein_dist": sum(w_dists) / len(w_dists),
    }


# ---------------------------------------------------------------------------
# Full Training Loop
# ---------------------------------------------------------------------------

def train_wgan_gp(
    X_attack: np.ndarray,
    y_attack_onehot: np.ndarray,
    config: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Train a conditional WGAN-GP on attack-only data.

    This function implements the complete training pipeline:
      1. Wraps data in a PyTorch DataLoader (batch size from config).
      2. Moves models to GPU if available.
      3. Trains for ``config['gan']['epochs']`` epochs, each consisting of
         full passes over the DataLoader with ``n_critic`` critic updates per
         generator update.
      4. Logs critic loss, generator loss, and estimated Wasserstein distance
         per epoch.
      5. Saves loss history as a CSV (``results/tables/wgan_loss_history.csv``)
         and a loss-curve plot (``results/figures/wgan_loss_curves.png``).
      6. Saves generator and critic weights — both "best" (lowest critic loss)
         and "last" checkpoints — to ``results/models/``.
      7. Stops early if any loss becomes NaN.

    Args:
        X_attack:        Feature matrix (numpy) containing ONLY attack rows
                         (no Normal class). Shape ``(n_samples, n_features)``.
        y_attack_onehot: One-hot encoded attack category labels (numpy).
                         Shape ``(n_samples, n_classes)``.
        config:          YAML-loaded configuration dictionary. Expected keys
                         under ``'gan'``: ``latent_dim``, ``gen_hidden_layers``,
                         ``critic_hidden_layers``, ``batch_size``, ``lr``,
                         ``betas``, ``n_critic``, ``lambda_gp``, ``epochs``.
        device:          Optional torch device or device string ('cpu', 'cuda').
        save_plots:      Whether to generate and save training loss curve plots.

    Returns:
        Dictionary with keys:
            - ``'generator'``: trained Generator module (on CPU).
            - ``'critic'``:    trained Critic module (on CPU).
            - ``'history'``:   pandas DataFrame of per-epoch losses.
            - ``'loss_history'``: alias for history DataFrame.
            - ``'best_epoch'``: epoch index with the lowest average critic loss.
    """
    gan_cfg = config.get("gan", {})

    # --- Hyperparameters from config ---
    latent_dim: int = gan_cfg.get("latent_dim", 100)
    batch_size: int = gan_cfg.get("batch_size", 128)
    lr: float = gan_cfg.get("lr", 1e-4)
    betas: Tuple[float, float] = tuple(gan_cfg.get("betas", [0.5, 0.9]))  # type: ignore[assignment]
    n_critic: int = gan_cfg.get("n_critic", 5)
    lambda_gp: float = gan_cfg.get("lambda_gp", 10.0)
    epochs: int = gan_cfg.get("epochs", 50)

    # --- Derived dimensions ---
    n_features = X_attack.shape[1]
    n_classes = y_attack_onehot.shape[1]

    # --- Device ---
    if device is None:
        use_gpu = config.get("use_gpu", True)
        device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
    else:
        device = torch.device(device)
    logger.info("Training WGAN-GP on device: %s", device)
    logger.info(
        "Config: epochs=%d, batch_size=%d, lr=%.1e, n_critic=%d, "
        "lambda_gp=%.1f, latent_dim=%d",
        epochs, batch_size, lr, n_critic, lambda_gp, latent_dim,
    )
    logger.info(
        "Data: %d attack samples, %d features, %d classes",
        X_attack.shape[0], n_features, n_classes,
    )

    # --- Build models ---
    generator = build_generator(config, n_classes=n_classes, output_dim=n_features).to(device)
    critic = build_critic(config, n_classes=n_classes, input_dim=n_features).to(device)

    # --- Optimizers (Adam with specified lr and betas) ---
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_c = optim.Adam(critic.parameters(), lr=lr, betas=betas)

    # --- DataLoader ---
    dataset = TensorDataset(
        torch.tensor(X_attack, dtype=torch.float32),
        torch.tensor(y_attack_onehot, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- Output directories ---
    models_dir = Path(config.get("output_dirs", {}).get("models", "results/models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Training history ---
    history: List[Dict[str, float]] = []
    best_critic_loss = float("inf")
    best_epoch = 0

    logger.info("Starting WGAN-GP training for %d epochs ...", epochs)

    for epoch in range(1, epochs + 1):
        generator.train()
        critic.train()

        epoch_c_losses: List[float] = []
        epoch_g_losses: List[float] = []
        epoch_w_dists: List[float] = []

        for real_batch, label_batch in loader:
            real_batch = real_batch.to(device)
            label_batch = label_batch.to(device)

            metrics = train_one_iteration(
                generator=generator,
                critic=critic,
                real_samples=real_batch,
                real_labels=label_batch,
                optimizer_g=optimizer_g,
                optimizer_c=optimizer_c,
                device=device,
                n_critic=n_critic,
                lambda_gp=lambda_gp,
            )

            epoch_c_losses.append(metrics["critic_loss"])
            epoch_g_losses.append(metrics["generator_loss"])
            epoch_w_dists.append(metrics["wasserstein_dist"])

        # --- Epoch-level averages ---
        avg_c_loss = sum(epoch_c_losses) / len(epoch_c_losses)
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_w_dist = sum(epoch_w_dists) / len(epoch_w_dists)

        history.append({
            "epoch": epoch,
            "critic_loss": avg_c_loss,
            "generator_loss": avg_g_loss,
            "wasserstein_dist": avg_w_dist,
        })

        logger.info(
            "Epoch %3d/%d  |  C_loss: %+.4f  |  G_loss: %+.4f  |  W_dist: %+.4f",
            epoch, epochs, avg_c_loss, avg_g_loss, avg_w_dist,
        )

        # --- NaN early stopping ---
        if math.isnan(avg_c_loss) or math.isnan(avg_g_loss) or math.isnan(avg_w_dist):
            logger.warning(
                "NaN detected at epoch %d — stopping training early.", epoch,
            )
            break

        # --- Best checkpoint (lowest critic loss) ---
        if avg_c_loss < best_critic_loss:
            best_critic_loss = avg_c_loss
            best_epoch = epoch
            torch.save(generator.state_dict(), models_dir / "wgan_generator_best.pt")
            torch.save(critic.state_dict(), models_dir / "wgan_critic_best.pt")
            logger.info("  ↳ New best model saved (epoch %d).", epoch)

    # --- Save last checkpoint ---
    torch.save(generator.state_dict(), models_dir / "wgan_generator_last.pt")
    torch.save(critic.state_dict(), models_dir / "wgan_critic_last.pt")
    logger.info("Last-epoch weights saved to %s", models_dir)

    # --- Save loss history CSV ---
    history_df = pd.DataFrame(history)
    save_table(history_df, "wgan_loss_history", config=config)
    logger.info("Loss history saved to results/tables/wgan_loss_history.csv")

    # --- Plot loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Critic loss
    axes[0].plot(history_df["epoch"], history_df["critic_loss"], color="#e74c3c", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Critic Loss")
    axes[0].set_title("Critic Loss")
    axes[0].grid(True, alpha=0.3)

    # Generator loss
    axes[1].plot(history_df["epoch"], history_df["generator_loss"], color="#2ecc71", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Generator Loss")
    axes[1].set_title("Generator Loss")
    axes[1].grid(True, alpha=0.3)

    # Wasserstein distance
    axes[2].plot(history_df["epoch"], history_df["wasserstein_dist"], color="#3498db", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Wasserstein Distance")
    axes[2].set_title("Estimated Wasserstein Distance")
    axes[2].grid(True, alpha=0.3)

    if save_plots:
        fig.suptitle("WGAN-GP Training Curves", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, "wgan_loss_curves", config=config)
        plt.close(fig)
        logger.info("Loss curve plot saved to results/figures/wgan_loss_curves.png")
    else:
        plt.close(fig)

    # --- Move models back to CPU for downstream use ---
    generator.cpu()
    critic.cpu()

    logger.info(
        "WGAN-GP training complete.  Best epoch: %d  (critic_loss=%.4f)",
        best_epoch, best_critic_loss,
    )

    return {
        "generator": generator,
        "critic": critic,
        "history": history_df,
        "loss_history": history_df,
        "best_epoch": best_epoch,
    }


# ---------------------------------------------------------------------------
# Sample Generation with Valid One-Hot Features
# ---------------------------------------------------------------------------

def generate_samples(
    generator: Generator,
    class_id: int,
    n: int,
    config: Dict[str, Any],
    onehot_groups: Optional[Dict[str, List[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate *n* synthetic samples for a single attack class and post-process
    categorical one-hot columns to ensure validity.

    The generator produces continuous outputs in [-1, 1] (Tanh).  For each
    one-hot column group (e.g. protocol_type, service, flag) the raw
    continuous values are converted to a hard one-hot vector via argmax:
    the column with the highest activation is set to 1.0, all others to 0.0.

    After correction, every one-hot group is validated to ensure exactly one
    active (1.0) column per sample.

    Args:
        generator:     Trained Generator model (on CPU or GPU — will be
                       moved to the appropriate device internally).
        class_id:      Integer class index for the conditioning one-hot
                       vector (0-indexed within the attack categories).
        n:             Number of samples to generate.
        config:        YAML-loaded config dict (needs ``gan.latent_dim`` and
                       ``gan.batch_size``).
        onehot_groups: Dict mapping categorical feature name → list of
                       column indices in the final feature array.
        device:        Optional torch device or device string.

    Returns:
        Tuple of:
            - ``samples``: numpy array of shape ``(n, n_features)`` in the
              same scaled space as the training data (float32).
            - ``labels``:  numpy array of shape ``(n,)`` containing the
              integer ``class_id`` repeated *n* times.
    """
    if onehot_groups is None:
        onehot_groups = {}

    gan_cfg = config.get("gan", {})
    latent_dim: int = gan_cfg.get("latent_dim", 100)
    batch_size: int = gan_cfg.get("batch_size", 128)

    if device is None:
        device = next(generator.parameters()).device
    else:
        device = torch.device(device)
        generator.to(device)
    n_classes = generator.n_classes

    generator.eval()

    # Build the one-hot class conditioning vector for all samples
    # class_id is the 0-indexed position in the attack-only one-hot encoding
    class_onehot = torch.zeros(1, n_classes, device=device)
    class_onehot[0, class_id] = 1.0

    all_samples: List[np.ndarray] = []
    remaining = n

    with torch.no_grad():
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            noise = torch.randn(current_batch, latent_dim, device=device)
            labels_batch = class_onehot.expand(current_batch, -1)

            fake = generator(noise, labels_batch)  # (current_batch, n_features)
            all_samples.append(fake.cpu().numpy())
            remaining -= current_batch

    samples = np.concatenate(all_samples, axis=0)[:n].astype(np.float32)

    # --- Post-process one-hot column groups via argmax ---
    for group_name, col_indices in onehot_groups.items():
        group_vals = samples[:, col_indices]          # (n, n_levels)
        argmax_idx = np.argmax(group_vals, axis=1)    # (n,)

        # Zero out the group and set only the argmax column to 1.0
        samples[:, col_indices] = 0.0
        for i, idx in enumerate(argmax_idx):
            samples[i, col_indices[idx]] = 1.0

    # --- Validity check: every one-hot group must have exactly one 1.0 ---
    for group_name, col_indices in onehot_groups.items():
        row_sums = samples[:, col_indices].sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            bad_rows = np.where(~np.isclose(row_sums, 1.0))[0]
            raise ValueError(
                f"One-hot validity check failed for group '{group_name}': "
                f"{len(bad_rows)} sample(s) do not have exactly one active "
                f"column (e.g. row {bad_rows[0]} has sum={row_sums[bad_rows[0]]:.4f})."
            )

    logger.info(
        "Generated %d samples for class_id=%d. One-hot groups validated: %s",
        n, class_id, list(onehot_groups.keys()),
    )

    labels = np.full(n, class_id, dtype=np.int64)
    return samples, labels
