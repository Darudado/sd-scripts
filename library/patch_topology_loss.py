"""
Standalone Independent Patch Self-Similarity Topology Loss Module (PatchTopologyLoss).

Computes VAE-free spatial topology matching between predicted representations (latents or noise predictions)
and target representations (ground-truth latents or vision backbone patch features like DINOv3).

Uses chunked query processing, fused log_softmax, and mixed precision BMM for memory-efficient
computation of spatial patch affinity distributions.
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

    Uses chunked query processing (configurable ``chunk_size``) to reduce peak VRAM
    from O(B·N²) to O(B·K·N), fused ``log_softmax``, and native-precision BMM with
    FP32 upcast only for softmax/log_softmax normalization.
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
        chunk_size: int = 512,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.tau_latent = tau_latent
        self.tau_target = tau_target
        self.scale_levels = max(1, scale_levels)
        self.loss_type = loss_type.lower()
        self.apply_channel_norm = apply_channel_norm
        self.apply_timestep_weight = apply_timestep_weight
        self.chunk_size = chunk_size

    def _compute_octave_loss(
        self,
        curr_p: torch.Tensor,
        curr_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes spatial topology distribution loss for a single scale octave using
        chunked query processing, fused log_softmax, and mixed precision BMM.

        Args:
            curr_p: (B, C_p, H, W) tensor
            curr_t: (B, C_t, H, W) tensor
        Returns:
            Per-sample loss tensor (B,)
        """
        b, c_p, h, w = curr_p.shape
        c_t = curr_t.shape[1]
        n = h * w
        eps = 1e-8

        if self.apply_channel_norm:
            curr_p_norm = F.normalize(curr_p, p=2, dim=1, eps=1e-8)
            curr_t_norm = F.normalize(curr_t, p=2, dim=1, eps=1e-8)
        else:
            curr_p_norm = curr_p
            curr_t_norm = curr_t

        x_p = curr_p_norm.view(b, c_p, n).transpose(1, 2)  # (B, N, C_p)
        x_t = curr_t_norm.view(b, c_t, n).transpose(1, 2)  # (B, N, C_t)

        chunk_size = self.chunk_size if (self.chunk_size is not None and self.chunk_size > 0) else n

        accumulated_loss = torch.zeros(b, device=curr_p.device, dtype=torch.float32)

        k_p_t = x_p.transpose(1, 2)  # (B, C_p, N)
        k_t_t = x_t.transpose(1, 2)  # (B, C_t, N)

        tau_p = max(self.tau_latent, 1e-6)
        tau_t = max(self.tau_target, 1e-6)

        for start_idx in range(0, n, chunk_size):
            end_idx = min(start_idx + chunk_size, n)

            # Optimization 1: Chunked query slices (B, K, C)
            q_p = x_p[:, start_idx:end_idx, :]
            q_t = x_t[:, start_idx:end_idx, :]

            # Optimization 3: Native precision BMM
            sim_p = torch.bmm(q_p, k_p_t) / float(tau_p)

            with torch.no_grad():
                sim_t = torch.bmm(q_t, k_t_t) / float(tau_t)
                p_target_chunk = F.softmax(sim_t.to(torch.float32), dim=-1)

            # Optimization 2: Fused log_softmax & direct loss computation in FP32
            sim_p_f32 = sim_p.to(torch.float32)

            if self.loss_type == "kl":
                # KL(P_t || P_p) = sum(P_t * (log(P_t) - log(P_p)))
                log_p_pred_chunk = F.log_softmax(sim_p_f32, dim=-1)
                with torch.no_grad():
                    log_p_target_chunk = torch.log(p_target_chunk + eps)
                kl_elem = p_target_chunk * (log_p_target_chunk - log_p_pred_chunk)
                chunk_loss = kl_elem.sum(dim=-1).sum(dim=-1)
            elif self.loss_type == "ce":
                # Cross-Entropy = -sum(P_t * log(P_p))
                log_p_pred_chunk = F.log_softmax(sim_p_f32, dim=-1)
                ce_elem = -p_target_chunk * log_p_pred_chunk
                chunk_loss = ce_elem.sum(dim=-1).sum(dim=-1)
            elif self.loss_type == "cosine":
                p_pred_chunk = F.softmax(sim_p_f32, dim=-1)
                cos_sim = F.cosine_similarity(p_pred_chunk, p_target_chunk, dim=-1)  # (B, K)
                chunk_loss = (1.0 - cos_sim).sum(dim=-1)
            else:  # l2
                p_pred_chunk = F.softmax(sim_p_f32, dim=-1)
                diff_sq = (p_pred_chunk - p_target_chunk) ** 2
                chunk_loss = diff_sq.sum(dim=-1).sum(dim=-1)

            accumulated_loss += chunk_loss

        return accumulated_loss / float(n)

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
        batch_size = pred_4d.shape[0]

        total_loss = torch.zeros(batch_size, device=pred_4d.device, dtype=torch.float32)

        curr_p = pred_4d
        curr_t = target_4d

        for scale in range(self.scale_levels):
            if scale > 0:
                curr_p = F.avg_pool2d(curr_p, kernel_size=2, stride=2)
                curr_t = F.avg_pool2d(curr_t, kernel_size=2, stride=2)

            # Spatial dimension alignment if target and pred spatial resolutions differ
            if curr_t.shape[2:] != curr_p.shape[2:]:
                curr_t = F.interpolate(curr_t, size=curr_p.shape[2:], mode="bilinear", align_corners=False)

            level_loss = self._compute_octave_loss(curr_p, curr_t)
            scale_weight = 1.0 / (2.0 ** scale)
            total_loss += scale_weight * level_loss

        if self.apply_timestep_weight and timesteps is not None:
            # Normalizes timesteps to continuous t in [0, 1] (handles both [0, 1] and [0, 1000] formats)
            t_float = timesteps.to(torch.float32).view(batch_size)
            t_norm = torch.where(t_float > 1.0, t_float / 1000.0, t_float).clamp(min=0.0, max=1.0)
            t_weight = (1.0 - t_norm).clamp(min=0.0, max=1.0)
            total_loss = total_loss * t_weight

        return (total_loss * self.loss_weight).to(orig_dtype)
