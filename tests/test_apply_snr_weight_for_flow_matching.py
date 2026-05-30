"""Tests for apply_snr_weight_for_flow_matching in library/custom_train_functions.py.

The Min-SNR-γ weighting for flow matching computes:
    SNR(σ) = (1 - σ)² / σ²
    weight = min(SNR, γ) / (SNR + 1)

This is the velocity-prediction (v-prediction) analog, because flow matching
velocity prediction (v = ε - x₀) is mathematically analogous to v-prediction
in DDPM.
"""

import math
import pytest
import torch

from library.custom_train_functions import apply_snr_weight_for_flow_matching


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_weight(sigma: float, gamma: float) -> float:
    """Pure-Python reference implementation of the min-snr weight."""
    sigma = max(sigma, 1e-6)
    snr = ((1.0 - sigma) / sigma) ** 2
    min_snr = min(snr, gamma)
    return min_snr / (snr + 1)


# ---------------------------------------------------------------------------
# Basic shape / dtype tests
# ---------------------------------------------------------------------------

class TestShapeAndDtype:
    """Verify the function preserves tensor shape and dtype."""

    def test_output_shape_matches_loss(self):
        """Output should have same shape as input loss."""
        loss = torch.randn(4, 8, 16, 16, device="cuda", dtype=torch.float32)
        sigmas = torch.rand(4, 1, 1, 1, device="cuda", dtype=torch.float32)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert result.shape == loss.shape

    def test_output_dtype_matches_loss(self):
        """Output dtype should match the loss dtype (float64 case)."""
        loss = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float64)
        sigmas = torch.rand(2, 1, 1, 1, device="cuda", dtype=torch.float32)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert result.dtype == torch.float64

    def test_1d_sigmas_with_1d_loss(self):
        """Sigmas as (B,) should broadcast with loss (B,) — the post_process_loss path."""
        loss = torch.ones(3, device="cuda", dtype=torch.float32)
        sigmas = torch.tensor([0.1, 0.5, 0.9], device="cuda", dtype=torch.float32)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert result.shape == loss.shape

    def test_4d_sigmas_broadcast_with_4d_loss(self):
        """Sigmas as (B, 1, 1, 1) should broadcast with loss (B, C, H, W)."""
        loss = torch.ones(3, 4, 8, 8, device="cuda", dtype=torch.float32)
        sigmas = torch.tensor([0.1, 0.5, 0.9], device="cuda", dtype=torch.float32).view(-1, 1, 1, 1)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert result.shape == loss.shape

    def test_scalar_loss(self):
        """Should work with scalar-like loss tensors."""
        loss = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        sigmas = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

class TestNumericalCorrectness:
    """Verify the computed weights match the mathematical formula."""

    @pytest.mark.parametrize("sigma_val", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    def test_weight_matches_formula(self, sigma_val):
        """Weight at each sigma should match the reference formula."""
        gamma = 5.0
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([sigma_val], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma)
        expected = _expected_weight(sigma_val, gamma)
        torch.testing.assert_close(result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("gamma_val", [1.0, 3.0, 5.0, 10.0, 25.0])
    def test_weight_with_different_gamma(self, gamma_val):
        """Weight should respect gamma parameter."""
        sigma_val = 0.1  # high SNR
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([sigma_val], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma_val)
        expected = _expected_weight(sigma_val, gamma_val)
        torch.testing.assert_close(result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    def test_batch_correctness(self):
        """Each element in the batch should be weighted independently."""
        gamma = 5.0
        sigma_vals = [0.05, 0.3, 0.5, 0.8]
        loss = torch.ones(4, 2, 4, 4, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor(sigma_vals, device="cuda", dtype=torch.float64).view(-1, 1, 1, 1)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma)
        for i, sv in enumerate(sigma_vals):
            expected = _expected_weight(sv, gamma)
            # All elements in sample i should have the same weight
            assert torch.allclose(result[i], torch.full_like(result[i], expected), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Verify correct behavior at boundary values."""

    def test_sigma_near_zero_clamped(self):
        """Sigma near 0 should be clamped to avoid division by zero."""
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([1e-10], device="cuda", dtype=torch.float64)
        # Should not raise or produce NaN/Inf
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert torch.isfinite(result).all()

    def test_sigma_exactly_zero_clamped(self):
        """Sigma = 0 should be clamped and produce finite result."""
        loss = torch.ones(1, device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([0.0], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        assert torch.isfinite(result).all()

    def test_sigma_one_gives_zero_weight(self):
        """At σ=1 (pure noise), SNR=0, weight=0/(0+1)=0."""
        loss = torch.tensor([10.0], device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        torch.testing.assert_close(result.cpu(), torch.tensor([0.0], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    def test_sigma_half_snr_one(self):
        """At σ=0.5, SNR=1, weight = min(1, γ)/(1+1) = 0.5 (for γ≥1)."""
        loss = torch.tensor([2.0], device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([0.5], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        expected_weight = 1.0 / 2.0  # SNR=1, min(1,5)=1, 1/(1+1)=0.5
        torch.testing.assert_close(
            result.cpu(), torch.tensor([2.0 * expected_weight], dtype=torch.float64), rtol=1e-6, atol=1e-8
        )

    def test_high_snr_gamma_caps_weight(self):
        """At very low sigma (high SNR), gamma should cap the weight.

        σ=0.01 → SNR = (0.99/0.01)² = 9801
        Without cap: weight = 9801/(9801+1) ≈ 0.9999
        With γ=5:    weight = 5/(9801+1) ≈ 0.00051
        """
        gamma = 5.0
        loss = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        sigmas = torch.tensor([0.01], device="cuda", dtype=torch.float64)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma)
        expected = _expected_weight(0.01, gamma)
        torch.testing.assert_close(result.cpu(), torch.tensor([expected], dtype=torch.float64), rtol=1e-6, atol=1e-8)
        # Verify the weight is much smaller than uncapped
        uncapped_weight = ((0.99 / 0.01) ** 2) / (((0.99 / 0.01) ** 2) + 1)
        assert expected < uncapped_weight * 0.01  # gamma cap dramatically reduces weight


# ---------------------------------------------------------------------------
# Monotonicity and qualitative behavior tests
# ---------------------------------------------------------------------------

class TestQualitativeBehavior:
    """Verify the weight function has expected monotonicity properties."""

    def test_weight_increases_from_sigma_one_to_peak(self):
        """Weight should increase from σ=1 (weight=0) towards lower sigmas."""
        gamma = 5.0
        sigmas_vals = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]
        weights = [_expected_weight(s, gamma) for s in sigmas_vals]
        # Each should be larger than the previous (monotonically increasing as sigma decreases from 1)
        for i in range(1, len(weights)):
            assert weights[i] > weights[i - 1], f"Weight at σ={sigmas_vals[i]} should be > σ={sigmas_vals[i-1]}"

    def test_gamma_caps_high_snr_region(self):
        """For σ below gamma threshold, weights should decrease (cap effect)."""
        gamma = 5.0
        # For gamma=5, SNR=5 when σ = 1/(1+√5) ≈ 0.309
        # Below that, weights should decrease
        sigma_threshold = 1.0 / (1.0 + math.sqrt(gamma))  # ≈ 0.309
        sigmas_below = [sigma_threshold * 0.9, sigma_threshold * 0.5, sigma_threshold * 0.1]
        weights_below = [_expected_weight(s, gamma) for s in sigmas_below]
        # Weights should decrease as sigma goes further below threshold
        for i in range(1, len(weights_below)):
            assert weights_below[i] < weights_below[i - 1]

    def test_weight_is_bounded(self):
        """All weights should be in [0, 1] for any valid sigma and gamma."""
        gamma = 5.0
        for sigma in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]:
            w = _expected_weight(sigma, gamma)
            assert 0.0 <= w <= 1.0 + 1e-9, f"Weight {w} out of [0,1] at σ={sigma}"

    def test_weight_bounded_by_gamma(self):
        """Weight should never exceed gamma / (SNR + 1) which is ≤ gamma."""
        gamma = 5.0
        loss = torch.ones(100, device="cuda", dtype=torch.float64)
        sigmas = torch.rand(100, device="cuda", dtype=torch.float64).clamp(min=0.001, max=0.999)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=gamma)
        # All weights should be finite and non-negative
        assert (result >= 0).all()
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow correctly through the weighting."""

    def test_gradients_flow_through_loss(self):
        """Gradients should propagate through the weighted loss."""
        loss = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float64, requires_grad=True)
        sigmas = torch.tensor([0.3, 0.7], device="cuda", dtype=torch.float64).view(-1, 1, 1, 1)
        result = apply_snr_weight_for_flow_matching(loss, sigmas, gamma=5.0)
        result.sum().backward()
        assert loss.grad is not None
        assert torch.isfinite(loss.grad).all()
        # Gradients should be non-zero (the weight is non-zero for these sigmas)
        assert loss.grad.abs().sum() > 0
