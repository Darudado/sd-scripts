"""
Standalone Independent Patch Self-Similarity Topology Loss Module (PatchTopologyLoss).

Computes VAE-free spatial topology matching between predicted representations (latents or noise predictions)
and target representations (ground-truth latents or vision backbone patch features like DINOv3).
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d_spatial(x: torch.Tensor) -> torch.Tensor:
    """Reshapes input to (N, C, H, W) format if passed as sequence (N, L, C)."""
    if x.ndim == 4:
        return x
    elif x.ndim == 3:
        n, l, c = x.shape
        side = int(math.isqrt(l))
        if side * side == l:
            return x.transpose(1, 2).reshape(n, c, side, side)
        else:
            raise ValueError(f"Cannot infer square spatial dimensions (H, W) from sequence length L={l}.")
    else:
        raise ValueError(f"Expected 3D or 4D tensor for PatchTopologyLoss, got shape {x.shape}.")


class PatchTopologyLoss(nn.Module):
    """
    Standalone VAE-Free Patch Self-Similarity Topology Loss.

    Compares spatial patch attention distributions of predicted representations
    against target representations across spatial pyramid scale octaves.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        tau_latent: float = 0.1,
        tau_target: float = 0.1,
        scale_levels: int = 2,
        loss_type: str = "kl",
        apply_channel_norm: bool = True,
        apply_timestep_weight: bool = True,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.tau_latent = tau_latent
        self.tau_target = tau_target
        self.scale_levels = max(1, scale_levels)
        self.loss_type = loss_type.lower()
        self.apply_channel_norm = apply_channel_norm
        self.apply_timestep_weight = apply_timestep_weight

    def _compute_spatial_attention_matrix(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
            tau: Softmax temperature
        Returns:
            Spatial attention map (B, N, N) where N = H * W
        """
        b, c, h, w = x.shape
        n = h * w

        if self.apply_channel_norm:
            # Channel-wise L2 normalization to eliminate single-channel variance dominance
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        else:
            x_norm = x

        x_flat = x_norm.view(b, c, n).transpose(1, 2)  # (B, N, C)

        # Spatial inner-product affinity matrix (B, N, N)
        sim_matrix = torch.bmm(x_flat, x_flat.transpose(1, 2)) / float(max(tau, 1e-6))
        attn_matrix = F.softmax(sim_matrix, dim=-1)
        return attn_matrix

    def _compute_distribution_loss(self, p_pred: torch.Tensor, p_target: torch.Tensor) -> torch.Tensor:
        """
        Computes distance between predicted and target spatial probability matrices.
        Args:
            p_pred: (B, N, N) Softmax distribution
            p_target: (B, N, N) Softmax distribution
        Returns:
            Per-sample loss tensor (B,)
        """
        eps = 1e-8
        if self.loss_type == "kl":
            # KL(P_target || P_pred) = sum(P_target * (log(P_target) - log(P_pred)))
            kl_elem = p_target * (torch.log(p_target + eps) - torch.log(p_pred + eps))
            return kl_elem.sum(dim=-1).mean(dim=-1)
        elif self.loss_type == "ce":
            # Cross-Entropy = -sum(P_target * log(P_pred))
            ce_elem = -p_target * torch.log(p_pred + eps)
            return ce_elem.sum(dim=-1).mean(dim=-1)
        elif self.loss_type == "cosine":
            # 1.0 - Cosine Similarity between spatial distribution vectors
            cos_sim = F.cosine_similarity(p_pred, p_target, dim=-1)  # (B, N)
            return (1.0 - cos_sim).mean(dim=-1)
        else:  # l2
            diff_sq = (p_pred - p_target) ** 2
            return diff_sq.sum(dim=-1).mean(dim=-1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C_p, H, W) or (B, L, C_p)
            target: (B, C_t, H_t, W_t) or (B, L_t, C_t)
            timesteps: Optional diffusion/flow timesteps tensor for time decay weighting
        Returns:
            Per-sample loss tensor (B,) or scalar mean
        """
        pred_4d = _ensure_4d_spatial(pred)
        target_4d = _ensure_4d_spatial(target).detach()

        orig_dtype = pred_4d.dtype
        p = pred_4d.to(torch.float32)
        t = target_4d.to(torch.float32)
        batch_size = p.shape[0]

        total_loss = torch.zeros(batch_size, device=pred_4d.device, dtype=torch.float32)

        curr_p = p
        curr_t = t

        for scale in range(self.scale_levels):
            if scale > 0:
                curr_p = F.avg_pool2d(curr_p, kernel_size=2, stride=2)
                curr_t = F.avg_pool2d(curr_t, kernel_size=2, stride=2)

            # Spatial dimension alignment if target and pred spatial resolutions differ
            if curr_t.shape[2:] != curr_p.shape[2:]:
                curr_t = F.interpolate(curr_t, size=curr_p.shape[2:], mode="bilinear", align_corners=False)

            attn_p = self._compute_spatial_attention_matrix(curr_p, self.tau_latent)
            with torch.no_grad():
                attn_t = self._compute_spatial_attention_matrix(curr_t, self.tau_target).detach()

            level_loss = self._compute_distribution_loss(attn_p, attn_t)
            scale_weight = 1.0 / (2.0 ** scale)
            total_loss = total_loss + scale_weight * level_loss

        if self.apply_timestep_weight and timesteps is not None:
            # Normalizes timesteps to continuous t in [0, 1] (handles both [0, 1] and [0, 1000] formats)
            t_float = timesteps.to(torch.float32).view(batch_size)
            t_norm = torch.where(t_float > 1.0, t_float / 1000.0, t_float).clamp(min=0.0, max=1.0)
            t_weight = (1.0 - t_norm).clamp(min=0.0, max=1.0)
            total_loss = total_loss * t_weight

        return (total_loss * self.loss_weight).to(orig_dtype)
