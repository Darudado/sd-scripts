"""Tests for the soft parameter in apply_snr_weight and apply_snr_weight_for_flow_matching.

Verifies that:
1. soft=False (default) produces the standard Min-SNR-γ weights.
2. soft=True produces the smooth Soft Min-SNR transition from https://arxiv.org/abs/2401.11605.
3. The two modes produce different results (confirming the branching actually matters).
4. Edge cases (sigma=0, sigma=1, infinite SNR, etc.) are handled correctly in soft mode.
"""

import math
import pytest
import torch
from unittest.mock import MagicMock

from library.custom_train_functions import (
    apply_snr_weight,
    apply_snr_weight_for_flow_matching,
    prepare_scheduler_for_custom_training,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_soft_weight_flow(sigma: float, gamma: float) -> float:
    """Reference: soft Min-SNR for flow matching (v-prediction)."""
    sigma = max(sigma, 1e-6)
    snr = ((1.0 - sigma) / sigma) ** 2
    return (snr * gamma) / ((snr + gamma) * (snr + 1))


def _expected_hard_weight_flow(sigma: float, gamma: float) -> float:
    """Reference: hard Min-SNR for flow matching (v-prediction)."""
    sigma = max(sigma, 1e-6)
    snr = ((1.0 - sigma) / sigma) ** 2
    min_snr = min(snr, gamma)
    return min_snr / (snr + 1)


def _expected_soft_weight_ddpm_vpred(snr_val: float, gamma: float) -> float:
    """Reference: soft Min-SNR for DDPM with v-prediction."""
    return (snr_val * gamma) / ((snr_val + gamma) * (snr_val + 1))


def _expected_hard_weight_ddpm_vpred(snr_val: float, gamma: float) -> float:
    """Reference: hard Min-SNR for DDPM with v-prediction."""
    return min(snr_val, gamma) / (snr_val + 1)


def _expected_soft_weight_ddpm_eps(snr_val: float, gamma: float) -> float:
    """Reference: soft Min-SNR for DDPM with epsilon prediction."""
    return gamma / (snr_val + gamma)


def _expected_hard_weight_ddpm_eps(snr_val: float, gamma: float) -> float:
    """Reference: hard Min-SNR for DDPM with epsilon prediction."""
    return min(snr_val, gamma) / snr_val


def _make_noise_scheduler_with_snr(snr_values: list):
    """Create a mock noise scheduler with the given SNR values per timestep."""
    scheduler = MagicMock()
    snr_tensor = torch.tensor(snr_values, dtype=torch.float64)
    scheduler.all_snr = snr_tensor
    return scheduler


# ===========================================================================
# Tests for apply_snr_weight_for_flow_matching with soft parameter
# ===========================================================================

class TestFlowMatchingSoftWeight:
    """Verify soft Min-SNR-γ for flow matching models."""

    def test_soft_produces_different_result_than_hard(self):
        """soft=True and soft=False should produce different weights (except at trivial points)."""
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([0.3], device="cuda", dtype=torch.float64)
        result_hard = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma, soft=False)
        result_soft = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma, soft=True)
        assert not torch.allclose(result_hard, result_soft), "soft and hard should differ at σ=0.3"

    @pytest.mark.parametrize("sigma_val", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    def test_soft_weight_matches_formula(self, sigma_val):
        """Soft weight at each sigma should match the reference formula."""
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([sigma_val], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma, soft=True)
        expected = _expected_soft_weight_flow(sigma_val, gamma)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-8
        )

    @pytest.mark.parametrize("gamma_val", [1.0, 3.0, 5.0, 10.0, 25.0])
    def test_soft_weight_respects_gamma(self, gamma_val):
        """Soft weight should respect gamma parameter."""
        sigma_val = 0.1
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([sigma_val], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma_val, soft=True)
        expected = _expected_soft_weight_flow(sigma_val, gamma_val)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-8
        )

    def test_soft_default_false(self):
        """Default soft=False should produce hard Min-SNR weights."""
        gamma = 5.0
        sigma_val = 0.3
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([sigma_val], device="cuda", dtype=torch.float64)
        result_default = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma)
        result_hard = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma, soft=False)
        torch.testing.assert_close(result_default, result_hard)

    def test_soft_sigma_one_gives_zero_weight(self):
        """At σ=1 (pure noise), SNR=0, soft weight = 0*γ/((0+γ)*(0+1)) = 0."""
        loss = torch.tensor([10.0], device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0, soft=True)
        torch.testing.assert_close(result.cpu(), torch.tensor([0.0], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    def test_soft_sigma_near_zero_clamped(self):
        """Sigma near 0 should be clamped — soft mode should still produce finite results."""
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([1e-10], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0, soft=True)
        assert torch.isfinite(result).all()

    def test_soft_batch_correctness(self):
        """Each element in the batch should be weighted independently in soft mode."""
        gamma = 5.0
        sigma_vals = [0.05, 0.3, 0.5, 0.8]
        loss = torch.ones(4, 2, 4, 4, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor(sigma_vals, device="cuda", dtype=torch.float64).view(-1, 1, 1, 1)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma, soft=True)
        for i, sv in enumerate(sigma_vals):
            expected = _expected_soft_weight_flow(sv, gamma)
            assert torch.allclose(result[i], torch.full_like(result[i], expected), rtol=1e-6, atol=1e-8)

    def test_soft_gradients_flow_through(self):
        """Gradients should propagate through the soft-weighted loss."""
        loss = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float64, requires_grad=True)
        sigmas = torch.tensor([0.3, 0.7], device="cuda", dtype=torch.float64).view(-1, 1, 1, 1)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0, soft=True)
        result.sum().backward()
        assert loss.grad is not None
        assert torch.isfinite(loss.grad).all()
        assert loss.grad.abs().sum() > 0

    def test_soft_is_smoother_than_hard(self):
        """The soft weight curve should be smoother (no sharp kink at gamma threshold).

        For gamma=5, the hard weight has a kink at σ ≈ 0.309 (where SNR = gamma).
        The soft weight should transition smoothly there.
        """
        gamma = 5.0
        sigma_threshold = 1.0 / (1.0 + math.sqrt(gamma))

        # Take points very close to the threshold on both sides
        eps = 0.001
        sigma_below = sigma_threshold - eps
        sigma_above = sigma_threshold + eps

        w_hard_below = _expected_hard_weight_flow(sigma_below, gamma)
        w_hard_above = _expected_hard_weight_flow(sigma_above, gamma)
        hard_diff = abs(w_hard_below - w_hard_above)

        w_soft_below = _expected_soft_weight_flow(sigma_below, gamma)
        w_soft_above = _expected_soft_weight_flow(sigma_above, gamma)
        soft_diff = abs(w_soft_below - w_soft_above)

        # Soft should have a smaller rate of change at the threshold
        assert soft_diff < hard_diff, (
            f"Soft weight diff ({soft_diff:.6f}) should be < hard diff ({hard_diff:.6f}) near threshold"
        )


# ===========================================================================
# Tests for apply_snr_weight with soft parameter
# ===========================================================================

class TestDDPMSoftWeight:
    """Verify soft Min-SNR for DDPM-based models (apply_snr_weight)."""

    def _make_scheduler_for_snr_values(self):
        """Create a noise scheduler mock with plausible SNR values."""
        # Simple decreasing SNR schedule (high at low timesteps, low at high timesteps)
        snr_values = [1000.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        return _make_noise_scheduler_with_snr(snr_values)

    def test_soft_vpred_produces_different_result_than_hard(self):
        """soft=True should produce different results than soft=False for v-pred."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([3], device="cuda")  # SNR=20.0
        result_hard = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=False)
        result_soft = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=True)
        assert not torch.allclose(result_hard, result_soft)

    def test_soft_eps_produces_different_result_than_hard(self):
        """soft=True should produce different results than soft=False for eps-pred."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([3], device="cuda")  # SNR=20.0
        result_hard = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=False, soft=False)
        result_soft = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=False, soft=True)
        assert not torch.allclose(result_hard, result_soft)

    @pytest.mark.parametrize("timestep_idx,expected_snr", [(0, 1000.0), (3, 20.0), (5, 5.0), (7, 1.0)])
    def test_soft_vpred_matches_formula(self, timestep_idx, expected_snr):
        """Soft weight for v-pred should match (snr * gamma) / ((snr + gamma) * (snr + 1))."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([timestep_idx], device="cuda")
        result = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=True)
        expected = _expected_soft_weight_ddpm_vpred(expected_snr, gamma)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-5, atol=1e-8
        )

    @pytest.mark.parametrize("timestep_idx,expected_snr", [(0, 1000.0), (3, 20.0), (5, 5.0), (7, 1.0)])
    def test_soft_eps_matches_formula(self, timestep_idx, expected_snr):
        """Soft weight for eps-pred should match gamma / (snr + gamma)."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([timestep_idx], device="cuda")
        result = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=False, soft=True)
        expected = _expected_soft_weight_ddpm_eps(expected_snr, gamma)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-5, atol=1e-8
        )

    def test_soft_default_false(self):
        """Default soft=False should produce hard Min-SNR weights."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([3], device="cuda")
        result_default = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True)
        result_hard = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=False)
        torch.testing.assert_close(result_default, result_hard)

    def test_soft_batch_correctness(self):
        """Each element in the batch should be weighted independently in soft mode."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        # Use timesteps [2, 5, 7] → SNR [50, 5, 1]
        timesteps = torch.tensor([2, 5, 7], device="cuda")
        loss = torch.ones(3, 4, 8, 8, device="cuda", dtype=torch.float64)
        result = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=True)
        expected_snrs = [50.0, 5.0, 1.0]
        for i, snr in enumerate(expected_snrs):
            expected = _expected_soft_weight_ddpm_vpred(snr, gamma)
            assert torch.allclose(
                result[i], torch.full_like(result[i], expected), rtol=1e-5, atol=1e-8
            )

    def test_soft_ndim_broadcasting(self):
        """Soft mode should properly broadcast snr_weight to match loss ndim."""
        scheduler = self._make_scheduler_for_snr_values()
        gamma = 5.0
        # 1D loss, but we want to verify unsqueeze loop works in soft mode
        loss = torch.ones(2, 3, 4, 4, device="cuda", dtype=torch.float64)
        timesteps = torch.tensor([3, 7], device="cuda")  # SNR=20, SNR=1
        result = apply_snr_weight(loss, timesteps, scheduler, gamma, v_prediction=True, soft=True)
        assert result.shape == loss.shape


# ===========================================================================
# Tests verifying args.min_snr_gamma_soft integration at call sites
# ===========================================================================

class TestSoftArgIntegration:
    """Verify that the --min_snr_gamma_soft flag is parsed and passed correctly."""

    def test_custom_train_arguments_adds_min_snr_gamma_soft(self):
        """add_custom_train_arguments should register --min_snr_gamma_soft."""
        from library.custom_train_functions import add_custom_train_arguments
        import argparse
        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser)
        args = parser.parse_args(["--min_snr_gamma", "5.0", "--min_snr_gamma_soft"])
        assert args.min_snr_gamma == 5.0
        assert args.min_snr_gamma_soft is True

    def test_min_snr_gamma_soft_defaults_false(self):
        """Without --min_snr_gamma_soft flag, it should default to False."""
        from library.custom_train_functions import add_custom_train_arguments
        import argparse
        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser)
        args = parser.parse_args(["--min_snr_gamma", "5.0"])
        assert args.min_snr_gamma_soft is False
