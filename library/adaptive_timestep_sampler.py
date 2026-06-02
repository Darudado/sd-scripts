"""
Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training.

Implements the method from Kim et al. (arXiv:2411.09998) which adaptively samples
timesteps by tracking the impact of gradient updates on the objective for each timestep,
prioritizing timesteps that are most likely to minimize the objective effectively.

Supports both DDPM-style discrete timesteps (SD1.5, SDXL) and flow-matching
continuous timesteps (Flux, SD3, Anima, Lumina).
"""

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TimestepSamplerNetwork(nn.Module):
    """
    Neural network π_φ that parameterizes a Beta distribution for timestep sampling.

    Takes a latent representation x_0 and outputs two positive scalars (a, b)
    which parameterize a Beta(a, b) distribution from which timesteps are sampled.

    Architecture follows the paper's design:
    - Adaptive average pooling to reduce spatial dimensions
    - Flatten
    - MLP with SiLU activations
    - Softplus outputs to ensure a, b > 0
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        hidden_depth: int = 2,
    ):
        super().__init__()

        # Adaptive average pooling to fixed spatial size (reduces computation)
        self.pool = nn.AdaptiveAvgPool2d(4)

        # Calculate flattened size after pooling
        flat_size = in_channels * 4 * 4

        # Build MLP layers
        layers = []
        in_dim = flat_size
        for i in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_channels))
            layers.append(nn.SiLU())
            in_dim = hidden_channels

        self.mlp = nn.Sequential(*layers)

        # Output heads: each produces a single positive scalar
        self.head_a = nn.Linear(hidden_channels, 1)
        self.head_b = nn.Linear(hidden_channels, 1)

        # Initialize biases for softplus to give reasonable initial Beta parameters
        # Initial Beta(2, 2) is symmetric around 0.5, close to uniform
        nn.init.constant_(self.head_a.bias, 1.2)  # softplus(1.2) ≈ 1.55
        nn.init.constant_(self.head_b.bias, 1.2)
        nn.init.zeros_(self.head_a.weight)
        nn.init.zeros_(self.head_b.weight)

    def forward(self, x_0_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_0_latent: Latent representation of x_0, shape (B, C, H, W)

        Returns:
            a, b: Positive scalars for Beta distribution, each shape (B,)
        """
        # Pool and flatten
        h = self.pool(x_0_latent)
        h = h.reshape(h.shape[0], -1)

        # MLP
        h = self.mlp(h)

        # Output heads with softplus to ensure positivity
        a = F.softplus(self.head_a(h).squeeze(-1)) + 1e-6
        b = F.softplus(self.head_b(h).squeeze(-1)) + 1e-6

        return a, b


def select_timesteps_f_statistic(
    queue_data: torch.Tensor,
    num_selected: int,
) -> torch.Tensor:
    """
    Feature selection using F-statistic from linear regression.

    Given a queue of shape (Q_size, T) containing per-timestep delta values,
    identifies the top |num_selected| timesteps that best explain the overall
    delta (mean across timesteps) using the F-statistic.

    For each timestep tau, the F-statistic measures how well delta_{k,tau}
    predicts the mean delta across all timesteps, based on a simple linear
    regression: F = r^2 / (1 - r^2) * (n - 2), where r is the Pearson
    correlation coefficient.

    Args:
        queue_data: Tensor of shape (Q_size, T) with historical delta values
        num_selected: Number of top timesteps to select

    Returns:
        Tensor of shape (num_selected,) with indices of selected timesteps
    """
    Q_size, T = queue_data.shape

    if Q_size < 2:
        # Not enough data for correlation, select timesteps with highest mean abs delta
        mean_abs = queue_data.mean(dim=0).abs()
        return torch.topk(mean_abs, min(num_selected, T)).indices

    # Compute the target: mean delta across timesteps for each queue entry
    # This represents the overall impact of each gradient update
    target = queue_data.mean(dim=1)  # shape (Q_size,)

    # Compute Pearson correlation between each timestep's delta and the target
    # r = cov(X, Y) / (std(X) * std(Y))
    target_centered = target - target.mean()
    target_std = target_centered.norm()

    if target_std < 1e-10:
        # Target has near-zero variance, fall back to mean absolute delta
        mean_abs = queue_data.mean(dim=0).abs()
        return torch.topk(mean_abs, min(num_selected, T)).indices

    # Center and normalize each timestep column
    data_centered = queue_data - queue_data.mean(dim=0, keepdim=True)  # (Q, T)
    data_std = data_centered.norm(dim=0)  # (T,)

    # Avoid division by zero
    valid_mask = data_std > 1e-10

    # Compute correlation
    correlations = torch.zeros(T, device=queue_data.device)
    if valid_mask.any():
        # cov(X_tau, Y) = X_tau^T @ Y / n
        cov = (data_centered[:, valid_mask].T @ target_centered) / Q_size
        # r = cov / (std_x * std_y)
        correlations[valid_mask] = cov / (data_std[valid_mask] * target_std)

    # F-statistic: F = r^2 / (1 - r^2) * (n - 2)
    r_sq = correlations ** 2
    # Clamp to avoid division by zero
    r_sq = torch.clamp(r_sq, max=1.0 - 1e-6)
    f_stats = r_sq / (1.0 - r_sq) * max(Q_size - 2, 1)

    # Select top |num_selected| timesteps by F-statistic
    actual_selected = min(num_selected, T)
    return torch.topk(f_stats, actual_selected).indices


class AdaptiveTimestepManager:
    """
    Manages adaptive non-uniform timestep sampling for diffusion model training.

    Implements Algorithm 1 (Training DM with Timestep Sampler) and Algorithm 2
    (Approximation of Delta_k^t) from the paper.

    The manager:
    1. Uses a TimestepSamplerNetwork (π_φ) to generate Beta distribution parameters
    2. Samples timesteps from the Beta distribution
    3. Every f_S gradient steps, computes the impact of the gradient update on
       per-timestep losses (Algorithm 2)
    4. Updates the sampler using policy gradient (REINFORCE) with entropy regularization
    """

    def __init__(
        self,
        sampler_network: TimestepSamplerNetwork,
        noise_scheduler,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        # Hyperparameters from the paper
        learning_rate: float = 1e-2,
        entropy_coeff: float = 1e-2,
        update_freq: int = 40,  # f_S
        queue_size: int = 20,  # |Q|
        num_selected: int = 3,  # |S|
        v_parameterization: bool = False,
        # Flow-matching support
        model_type: str = "ddpm",  # "ddpm" or "flow_matching"
        compute_loss_fn=None,  # Optional: (model_output, x_0, noise, t_indices) -> per_sample_losses
    ):
        self.sampler_network = sampler_network.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.f_s = update_freq
        self.queue_size = queue_size
        self.num_selected = num_selected

        # Noise scheduler reference for Algorithm 2
        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps

        # Optimizer for the sampler network (SGD as used in the paper)
        self.optimizer = torch.optim.SGD(
            self.sampler_network.parameters(),
            lr=learning_rate,
        )

        # Queue Q for storing historical delta values (Algorithm 2, line 3)
        self.queue: deque = deque(maxlen=queue_size)

        # Cached Beta distribution parameters from the last sampling
        self._cached_a: Optional[torch.Tensor] = None
        self._cached_b: Optional[torch.Tensor] = None
        # Cached sampled t (continuous, in [0,1]) for REINFORCE update
        self._cached_t_continuous: Optional[torch.Tensor] = None
        # Track the currently selected |S| timesteps for the next call to Algorithm 2.
        # On the first call, we use the top-|S| timesteps with highest absolute delta.
        # On subsequent calls, we use the selection from the previous call (so that
        # we can compute losses at those timesteps for the full batch BEFORE the step).
        self._current_selected_indices: Optional[torch.Tensor] = None
        # Cache for the previous full-batch losses at |S| timesteps (computed before
        # the optimizer step). Used in Algorithm 2 line 7 to compute the delta for
        # the full batch at the selected timesteps.
        self._prev_batch_losses_at_S: Optional[torch.Tensor] = None
        # Cached previous selection's latents/noise for computing prev losses
        self._prev_batch_latents: Optional[torch.Tensor] = None
        self._prev_batch_noise: Optional[torch.Tensor] = None

        # Model type: "ddpm" or "flow_matching"
        self.model_type = model_type

        # Precompute alphas_cumprod on device for noise addition (DDPM only)
        if model_type == "ddpm":
            self._alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        else:
            self._alphas_cumprod = None

        # Loss target type: epsilon prediction vs v-prediction (DDPM only)
        self.v_parameterization = v_parameterization

        # Custom loss function for flow-matching models
        # Signature: (model_output, x_0, noise, t_indices) -> per_sample_losses
        # When provided, overrides the default DDPM/v-pred loss computation.
        self._compute_loss_fn = compute_loss_fn

        logger.info(
            f"AdaptiveTimestepManager initialized: lr={learning_rate}, "
            f"entropy_coeff={entropy_coeff}, f_S={update_freq}, "
            f"|Q|={queue_size}, |S|={num_selected}, v_pred={v_parameterization}, "
            f"model_type={model_type}, custom_loss_fn={compute_loss_fn is not None}"
        )

    def should_update(self, global_step: int) -> bool:
        """Returns True if the sampler should be updated this step."""
        return global_step > 0 and global_step % self.f_s == 0

    def sample_timesteps(
        self,
        x_0_latent: torch.Tensor,
        num_timesteps: int,
    ) -> torch.Tensor:
        """
        Sample timesteps using the adaptive Beta distribution sampler.

        Args:
            x_0_latent: Latent representations, shape (B, C, H, W)
            num_timesteps: Total number of discrete timesteps (T)

        Returns:
            timesteps: Sampled timesteps, shape (B,), dtype long, in range [0, num_timesteps)
        """
        self.sampler_network.eval()
        with torch.no_grad():
            a, b = self.sampler_network(x_0_latent)

        # Cache the parameters and sampled t for later policy gradient update
        self._cached_a = a.detach()
        self._cached_b = b.detach()

        # Sample from Beta distribution
        beta_dist = torch.distributions.Beta(a, b)
        t_continuous = beta_dist.sample()  # shape (B,), values in (0, 1)

        # Cache the sampled t for REINFORCE update (Algorithm 1, line 8)
        self._cached_t_continuous = t_continuous.detach()

        # Map to discrete timesteps [0, num_timesteps)
        timesteps = (t_continuous * num_timesteps).long()
        timesteps = torch.clamp(timesteps, 0, num_timesteps - 1)

        return timesteps

    def compute_per_timestep_losses(
        self,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
        chunk_size: int = 100,
    ) -> torch.Tensor:
        """
        Compute the diffusion loss at each timestep for a single x_0.

        This is used by Algorithm 2 to evaluate the impact of gradient updates.
        For efficiency, processes timesteps in chunks to avoid OOM.

        Supports both DDPM-style and flow-matching noise addition.
        For flow-matching models, a custom compute_loss_fn can be provided
        at initialization to handle model-specific loss computation.

        Args:
            x_0_latent: Single latent, shape (1, C, H, W)
            noise: Corresponding noise, shape (1, C, H, W)
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            weight_dtype: Data type for model inference
            chunk_size: Number of timesteps to process at once

        Returns:
            losses: Per-timestep losses, shape (T,)
        """
        T = self.num_train_timesteps

        # Expand x_0 and noise to single sample
        x_0 = x_0_latent[:1]  # (1, C, H, W)
        eps = noise[:1]  # (1, C, H, W)

        losses = torch.zeros(T, device=self.device, dtype=torch.float32)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            t_indices = torch.arange(start, end, device=self.device, dtype=torch.long)
            chunk_len = end - start

            # Create chunk of timesteps
            t_chunk = t_indices  # (chunk_len,)

            if self.model_type == "flow_matching":
                # Flow-matching noise addition: x_t = sigma * noise + (1 - sigma) * x_0
                # where sigma = t / T (continuous in [0, 1])
                sigmas = t_indices.float().to(self.device) / T  # (chunk_len,)
                sigmas_view = sigmas.view(-1, 1, 1, 1)  # (chunk_len, 1, 1, 1)

                x_0_expanded = x_0.expand(chunk_len, -1, -1, -1)
                eps_expanded = eps.expand(chunk_len, -1, -1, -1)

                x_t = sigmas_view * eps_expanded + (1.0 - sigmas_view) * x_0_expanded
                x_t = x_t.to(weight_dtype)

                # Forward pass through the model
                with torch.no_grad():
                    model_output = model_fn(x_t, t_chunk, weight_dtype)

                # Compute loss
                model_output = model_output.to(torch.float32)
                if self._compute_loss_fn is not None:
                    # Custom loss function for model-specific loss computation
                    chunk_losses = self._compute_loss_fn(model_output, x_0_expanded, eps_expanded, t_indices)
                else:
                    # Default flow-matching: velocity prediction target v = noise - x_0
                    target = (eps_expanded - x_0_expanded).to(torch.float32)
                    chunk_losses = F.mse_loss(model_output, target, reduction="none")
                    chunk_losses = chunk_losses.mean(dim=list(range(1, chunk_losses.ndim)))
            else:
                # DDPM noise addition: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
                alphas_cumprod = self._alphas_cumprod
                alpha_bar_t = alphas_cumprod[t_indices].view(-1, 1, 1, 1)  # (chunk_len, 1, 1, 1)
                sqrt_alpha = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

                # Expand x_0 and noise to match chunk size
                x_0_expanded = x_0.expand(chunk_len, -1, -1, -1)
                eps_expanded = eps.expand(chunk_len, -1, -1, -1)

                x_t = sqrt_alpha * x_0_expanded + sqrt_one_minus_alpha * eps_expanded
                x_t = x_t.to(weight_dtype)

                # Forward pass through the model
                with torch.no_grad():
                    model_output = model_fn(x_t, t_chunk, weight_dtype)

                # Compute MSE loss for each timestep
                model_output = model_output.to(torch.float32)
                if self._compute_loss_fn is not None:
                    chunk_losses = self._compute_loss_fn(model_output, x_0_expanded, eps_expanded, t_indices)
                elif self.v_parameterization:
                    # v-prediction target: v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
                    target = sqrt_alpha * eps_expanded - sqrt_one_minus_alpha * x_0_expanded
                    target = target.to(torch.float32)
                    chunk_losses = F.mse_loss(model_output, target, reduction="none")
                    chunk_losses = chunk_losses.mean(dim=list(range(1, chunk_losses.ndim)))
                else:
                    # epsilon prediction target
                    target = eps_expanded.to(torch.float32)
                    chunk_losses = F.mse_loss(model_output, target, reduction="none")
                    chunk_losses = chunk_losses.mean(dim=list(range(1, chunk_losses.ndim)))

            losses[start:end] = chunk_losses

        return losses

    def compute_delta_approximation(
        self,
        model_fn,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        weight_dtype: torch.dtype,
        losses_before: torch.Tensor,
        full_batch_latents: Optional[torch.Tensor] = None,
        full_batch_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 2: Approximate Delta_k^t.

        Given per-timestep losses with θ_k (before optimizer step) and θ_{k+1}
        (after optimizer step), computes the approximated delta.

        When full batch data is provided AND we have a previous |S| selection from
        a prior call, the returned delta is computed over the full mini-batch at the
        |S| selected timesteps (per Algorithm 2, line 7: "for x_0s in current mini-batch").
        This requires losses_before_batch_at_S to have been cached via
        `cache_batch_losses_at_S()` BEFORE the optimizer step.

        Otherwise, falls back to computing the delta for a single x_0 at the
        |S| selected timesteps.

        Args:
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            x_0_latent: Single latent for the queue, shape (1, C, H, W)
            noise: Corresponding noise for the queue, shape (1, C, H, W)
            weight_dtype: Data type for model inference
            losses_before: Per-timestep losses with θ_k for the single x_0, shape (T,)
            full_batch_latents: Optional full batch latents, shape (B, C, H, W)
            full_batch_noise: Optional full batch noise, shape (B, C, H, W)

        Returns:
            delta_approx: Scalar approximation of Δ̃_k^t
            selected_indices: The timestep indices selected by feature selection
        """
        # Step 2: Compute per-timestep losses with θ_{k+1} (after optimizer step)
        losses_after = self.compute_per_timestep_losses(
            x_0_latent, noise, model_fn, weight_dtype
        )

        # Compute delta for each timestep: δ_{k,τ} = L_τ(θ_k) - L_τ(θ_{k+1})
        deltas = losses_before - losses_after  # shape (T,)

        # Step 3: Push into queue Q
        self.queue.append(deltas.detach().cpu())

        # Step 4-5: Feature selection if queue has enough data
        if len(self.queue) > 1:
            queue_tensor = torch.stack(list(self.queue), dim=0).to(self.device)  # (Q, T)
            new_selected_indices = select_timesteps_f_statistic(
                queue_tensor, self.num_selected
            )
        else:
            # First iteration: select timesteps with highest absolute delta
            new_selected_indices = torch.topk(deltas.abs(), self.num_selected).indices

        # Step 7: Compute approximation for the current mini-batch.
        # Prefer the full batch at the PREVIOUS |S| selection (if available),
        # since we already have losses_before for the full batch at those timesteps.
        # Otherwise, fall back to the single x_0 at the new |S| selection.
        if (
            full_batch_latents is not None
            and full_batch_noise is not None
            and self._current_selected_indices is not None
            and self._prev_batch_losses_at_S is not None
        ):
            prev_S = self._current_selected_indices.to(self.device)
            # Compute losses after the step for the full batch at the PREVIOUS |S| timesteps
            losses_after_batch_at_S = self.compute_per_timestep_losses_for_batch(
                full_batch_latents, full_batch_noise, model_fn, weight_dtype, prev_S
            )
            # Delta for the full batch: mean over batch and |S| timesteps
            delta_approx = (self._prev_batch_losses_at_S - losses_after_batch_at_S).mean()
            # Use the previous selection as the returned indices (these are the ones
            # for which the delta was actually computed)
            selected_indices = prev_S
        else:
            # Fallback: single x_0 at the new |S| selection
            delta_approx = deltas[new_selected_indices].mean()
            selected_indices = new_selected_indices

        # Update the current selection for the NEXT call to Algorithm 2
        self._current_selected_indices = selected_indices.detach().cpu()
        # Clear the cached prev batch losses (will be set by the next before-step call)
        self._prev_batch_losses_at_S = None
        self._prev_batch_latents = None
        self._prev_batch_noise = None

        return delta_approx, selected_indices

    def cache_batch_losses_at_S(
        self,
        full_batch_latents: torch.Tensor,
        full_batch_noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
    ):
        """
        Cache per-timestep losses for the full batch at the current |S| selection.

        This should be called BEFORE the optimizer step, so that after the step
        we can compute the full-batch delta at those |S| timesteps.

        If no |S| selection exists yet (first call), this is a no-op.
        """
        if self._current_selected_indices is None:
            return
        indices = self._current_selected_indices.to(self.device)
        self._prev_batch_losses_at_S = self.compute_per_timestep_losses_for_batch(
            full_batch_latents, full_batch_noise, model_fn, weight_dtype, indices
        ).detach()
        self._prev_batch_latents = full_batch_latents
        self._prev_batch_noise = full_batch_noise

    def compute_per_timestep_losses_for_batch(
        self,
        x_0_latent: torch.Tensor,
        noise: torch.Tensor,
        model_fn,
        weight_dtype: torch.dtype,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-timestep losses for the full batch at selected timesteps only.

        This is used to compute the final Δ̃_k^t for the full mini-batch
        (Algorithm 2, line 7: "for x_0s in current mini-batch").

        Args:
            x_0_latent: Latent representations for the batch, shape (B, C, H, W)
            noise: Corresponding noise, shape (B, C, H, W)
            model_fn: Function(noisy_latents, timesteps, weight_dtype) -> noise_pred
            weight_dtype: Data type for model inference
            selected_indices: Timestep indices to evaluate, shape (|S|,)

        Returns:
            delta_approx: Scalar approximation of Δ̃_k^t averaged over batch and timesteps
        """
        T = self.num_train_timesteps
        B = x_0_latent.shape[0]
        S = selected_indices.shape[0]

        # Expand: (B, C, H, W) -> (B*S, C, H, W) by repeating
        x_0_exp = x_0_latent.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, *x_0_latent.shape[1:])
        eps_exp = noise.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, *noise.shape[1:])

        timesteps_exp = selected_indices.unsqueeze(0).expand(B, -1).reshape(B * S)

        if self.model_type == "flow_matching":
            # Flow-matching noise addition: x_t = sigma * noise + (1 - sigma) * x_0
            sigmas = selected_indices.float().to(self.device) / T  # (S,)
            sigmas_exp = sigmas.unsqueeze(0).expand(B, -1).reshape(B * S)  # (B*S,)
            sigmas_view = sigmas_exp.view(-1, 1, 1, 1)  # (B*S, 1, 1, 1)

            x_t = sigmas_view * eps_exp + (1.0 - sigmas_view) * x_0_exp
            x_t = x_t.to(weight_dtype)

            with torch.no_grad():
                model_output = model_fn(x_t, timesteps_exp, weight_dtype)

            model_output = model_output.to(torch.float32)
            if self._compute_loss_fn is not None:
                # Custom loss function
                losses = self._compute_loss_fn(model_output, x_0_exp, eps_exp, selected_indices)
                if losses.ndim > 1:
                    losses = losses.mean(dim=list(range(1, losses.ndim)))
            else:
                # Default flow-matching: velocity prediction target v = noise - x_0
                target = (eps_exp - x_0_exp).to(torch.float32)
                losses = F.mse_loss(model_output, target, reduction="none")
                losses = losses.mean(dim=list(range(1, losses.ndim)))
        else:
            # DDPM noise addition
            alphas_cumprod = self._alphas_cumprod
            alpha_bar_t = alphas_cumprod[selected_indices].view(S, 1, 1, 1)
            sqrt_alpha = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar_t)

            sqrt_alpha_exp = sqrt_alpha.expand(-1, -1, -1, -1).repeat(B, 1, 1, 1).reshape(B * S, 1, 1, 1)
            sqrt_one_minus_exp = sqrt_one_minus_alpha.expand(-1, -1, -1, -1).repeat(B, 1, 1, 1).reshape(B * S, 1, 1, 1)

            x_t = sqrt_alpha_exp * x_0_exp + sqrt_one_minus_exp * eps_exp
            x_t = x_t.to(weight_dtype)

            with torch.no_grad():
                model_output = model_fn(x_t, timesteps_exp, weight_dtype)

            model_output = model_output.to(torch.float32)
            if self._compute_loss_fn is not None:
                losses = self._compute_loss_fn(model_output, x_0_exp, eps_exp, selected_indices)
                if losses.ndim > 1:
                    losses = losses.mean(dim=list(range(1, losses.ndim)))
            elif self.v_parameterization:
                # v-prediction target: v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
                target = sqrt_alpha_exp * eps_exp - sqrt_one_minus_exp * x_0_exp
                target = target.to(torch.float32)
                losses = F.mse_loss(model_output, target, reduction="none")
                losses = losses.mean(dim=list(range(1, losses.ndim)))
            else:
                # epsilon prediction target
                target = eps_exp.to(torch.float32)
                losses = F.mse_loss(model_output, target, reduction="none")
                losses = losses.mean(dim=list(range(1, losses.ndim)))

        # Reshape to (B, S) and average
        losses = losses.reshape(B, S)

        return losses.mean()

    def update_sampler(
        self,
        delta_k_t: torch.Tensor,
        x_0_latent: torch.Tensor,
    ):
        """
        Update the timestep sampler π_φ using policy gradient (REINFORCE).

        Implements Algorithm 1, line 8:
        φ_{k+1} = φ_k + γ · Δ̃_k^t · ∇_{φ_k} log π_{φ_k}(a, b | x_0)

        With entropy regularization to prevent premature convergence.

        Args:
            delta_k_t: Approximated Δ̃_k^t, scalar tensor
            x_0_latent: The x_0 latent used for sampling, shape (B, C, H, W)
        """
        self.sampler_network.train()

        # Forward pass to get current (a, b) with gradients
        a, b = self.sampler_network(x_0_latent)

        # Compute log probability of the cached Beta distribution parameters
        # We use the NEW forward pass (with grad) but evaluate at the CACHED (a, b)
        # Actually, we need to compute log_prob of the ACTUAL timesteps that were sampled
        # using the current network parameters. Since we need gradients through φ,
        # we re-sample or use the cached continuous values.

        # For the REINFORCE estimator, we need:
        # ∇_φ E[Δ̃] = E[Δ̃ · ∇_φ log π_φ(a,b|x_0)]
        # This is equivalent to minimizing: -Δ̃ · log Beta(t; a, b)

        # Sample from the current Beta distribution (with gradients)
        beta_dist = torch.distributions.Beta(a, b)

        # Use the cached sampled t (the action actually taken in the training step)
        # This is the correct REINFORCE estimator: gradient of log_prob evaluated
        # at the action that was sampled, per Eq. 13 of the paper.
        if self._cached_t_continuous is not None:
            t_continuous = self._cached_t_continuous
        else:
            t_continuous = a / (a + b)  # fallback to mean

        # Compute log probability
        log_prob = beta_dist.log_prob(t_continuous.clamp(1e-6, 1.0 - 1e-6))

        # Policy gradient loss: -Δ̃ * log_prob (we want to maximize Δ̃)
        # delta_k_t is positive when the update was beneficial
        policy_loss = -(delta_k_t.detach() * log_prob).mean()

        # Entropy regularization: H(Beta(a,b)) = log B(a,b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b)
        entropy = beta_dist.entropy().mean()
        entropy_loss = -self.entropy_coeff * entropy

        total_loss = policy_loss + entropy_loss

        # Update sampler
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        logger.debug(
            f"Sampler update: delta={delta_k_t.item():.6f}, "
            f"policy_loss={policy_loss.item():.6f}, "
            f"entropy={entropy.item():.6f}, "
            f"a_mean={a.mean().item():.3f}, b_mean={b.mean().item():.3f}"
        )

    def state_dict(self) -> Dict:
        """Save sampler state for checkpointing."""
        return {
            "sampler_network": self.sampler_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "queue": list(self.queue),
            "learning_rate": self.learning_rate,
            "entropy_coeff": self.entropy_coeff,
            "f_s": self.f_s,
            "queue_size": self.queue_size,
            "num_selected": self.num_selected,
            "v_parameterization": self.v_parameterization,
            "model_type": self.model_type,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load sampler state from checkpoint."""
        self.sampler_network.load_state_dict(state_dict["sampler_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.queue = deque(state_dict["queue"], maxlen=self.queue_size)
