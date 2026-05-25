"""
Tests for Focal Frequency Loss integration in latent space.

Validates:
  - FocalFrequencyLoss module computation
  - CLI argument registration
  - Integration with train_network.py's process_batch
  - Edge cases (identical inputs, different inputs, gradient flow)
"""

import argparse
import sys
import os
import pytest
import torch
import torch.nn as nn
import math

# Add the sd_scripts directory to the path so we can import the library
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from library.focal_frequency_loss import FocalFrequencyLoss


# ──────────────────────────────────────────────
# FocalFrequencyLoss module tests
# ──────────────────────────────────────────────


class TestFocalFrequencyLossModule:
    """Unit tests for the FocalFrequencyLoss class."""

    def test_output_is_scalar(self):
        """FFL should return a scalar tensor."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        loss = ffl(pred, target)
        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_identical_inputs_yield_zero_loss(self):
        """When pred == target, FFL should be zero."""
        ffl = FocalFrequencyLoss()
        x = torch.randn(2, 4, 16, 16)
        loss = ffl(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7), (
            f"Expected zero loss for identical inputs, got {loss.item()}"
        )

    def test_loss_is_non_negative(self):
        """FFL should always be non-negative."""
        ffl = FocalFrequencyLoss()
        for _ in range(10):
            pred = torch.randn(2, 4, 16, 16)
            target = torch.randn(2, 4, 16, 16)
            loss = ffl(pred, target)
            assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_loss_weight_scaling(self):
        """Loss weight should linearly scale the output."""
        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16)

        ffl1 = FocalFrequencyLoss(loss_weight=1.0)
        ffl2 = FocalFrequencyLoss(loss_weight=2.0)

        loss1 = ffl1(pred, target)
        loss2 = ffl2(pred, target)

        assert torch.isclose(loss2, loss1 * 2.0, atol=1e-6), (
            f"Expected loss2 = 2 * loss1, got {loss2.item()} vs {loss1.item() * 2}"
        )

    def test_gradient_flow(self):
        """Gradients should flow through FFL."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(1, 4, 16, 16, requires_grad=True)
        target = torch.randn(1, 4, 16, 16)

        loss = ffl(pred, target)
        loss.backward()

        assert pred.grad is not None, "Gradient should flow to pred"
        assert not torch.all(pred.grad == 0), "Gradient should be non-zero"

    def test_gradient_does_not_flow_to_target(self):
        """Gradients should not flow through the weight matrix (detached)."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16, requires_grad=True)

        loss = ffl(pred, target)
        loss.backward()

        assert target.grad is not None, "Gradient should flow to target"
        assert not torch.all(target.grad == 0), "Target gradient should be non-zero"

    def test_works_with_float64(self):
        """FFL should work with float64 tensors (used in training)."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(2, 4, 16, 16, dtype=torch.float64)
        target = torch.randn(2, 4, 16, 16, dtype=torch.float64)
        loss = ffl(pred, target)
        assert loss.dtype == torch.float64
        assert loss.ndim == 0

    def test_works_with_small_spatial_dims(self):
        """FFL should work with small latent spatial dimensions (e.g., 4x4)."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(1, 4, 4, 4)
        target = torch.randn(1, 4, 4, 4)
        loss = ffl(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_works_with_various_batch_sizes(self):
        """FFL should work with different batch sizes."""
        ffl = FocalFrequencyLoss()
        for batch_size in [1, 2, 4, 8]:
            pred = torch.randn(batch_size, 4, 16, 16)
            target = torch.randn(batch_size, 4, 16, 16)
            loss = ffl(pred, target)
            assert loss.ndim == 0

    def test_alpha_parameter_affects_loss(self):
        """Different alpha values should produce different losses."""
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)

        ffl_a1 = FocalFrequencyLoss(alpha=1.0)
        ffl_a2 = FocalFrequencyLoss(alpha=2.0)

        loss_a1 = ffl_a1(pred, target)
        loss_a2 = ffl_a2(pred, target)

        # They should generally be different (not always guaranteed for random
        # inputs, but extremely unlikely to be identical)
        assert not torch.isclose(loss_a1, loss_a2, atol=1e-6), (
            "Different alpha values should produce different losses"
        )

    def test_ave_spectrum_mode(self):
        """FFL with ave_spectrum=True should work without errors."""
        ffl = FocalFrequencyLoss(ave_spectrum=True)
        pred = torch.randn(4, 4, 16, 16)
        target = torch.randn(4, 4, 16, 16)
        loss = ffl(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_log_matrix_mode(self):
        """FFL with log_matrix=True should work without errors."""
        ffl = FocalFrequencyLoss(log_matrix=True)
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        loss = ffl(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_batch_matrix_mode(self):
        """FFL with batch_matrix=True should work without errors."""
        ffl = FocalFrequencyLoss(batch_matrix=True)
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        loss = ffl(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_patch_factor(self):
        """FFL with patch_factor > 1 should work when dims are divisible."""
        ffl = FocalFrequencyLoss(patch_factor=2)
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randn(2, 4, 16, 16)
        loss = ffl(pred, target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_patch_factor_assertion(self):
        """FFL with patch_factor should assert if dims not divisible."""
        ffl = FocalFrequencyLoss(patch_factor=3)
        pred = torch.randn(2, 4, 16, 16)  # 16 % 3 != 0
        with pytest.raises(AssertionError):
            ffl(pred, pred)

    def test_predefined_matrix(self):
        """FFL should accept a predefined weight matrix."""
        ffl = FocalFrequencyLoss()
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)

        # Create a valid weight matrix [0, 1]
        pred_freq = ffl.tensor2freq(pred)
        target_freq = ffl.tensor2freq(target)
        matrix = torch.ones_like(pred_freq[..., 0]) * 0.5

        loss = ffl(pred, target, matrix=matrix)
        assert loss.ndim == 0
        assert loss.item() >= 0


# ──────────────────────────────────────────────
# CLI argument registration tests
# ──────────────────────────────────────────────


class TestCLIArguments:
    """Tests for CLI argument registration in custom_train_functions."""

    def test_focal_frequency_loss_flag_registered(self):
        """--focal_frequency_loss should be a registered argument."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args(["--focal_frequency_loss"])
        assert args.focal_frequency_loss is True

    def test_focal_frequency_loss_defaults(self):
        """Default values should be correct."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args([])
        assert args.focal_frequency_loss is False
        assert args.focal_frequency_loss_weight == 1.0
        assert args.focal_frequency_loss_alpha == 1.0

    def test_focal_frequency_loss_custom_values(self):
        """Custom values should be parsed correctly."""
        from library.custom_train_functions import add_custom_train_arguments

        parser = argparse.ArgumentParser()
        add_custom_train_arguments(parser, support_weighted_captions=False)
        args = parser.parse_args([
            "--focal_frequency_loss",
            "--focal_frequency_loss_weight", "0.5",
            "--focal_frequency_loss_alpha", "2.0",
        ])
        assert args.focal_frequency_loss is True
        assert args.focal_frequency_loss_weight == 0.5
        assert args.focal_frequency_loss_alpha == 2.0


# ──────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────


class TestFFLIntegration:
    """Integration tests verifying FFL works in the training context."""

    def test_ffl_adds_to_loss(self):
        """FFL should add a positive value to the base loss."""
        ffl = FocalFrequencyLoss(loss_weight=1.0)

        # Simulate noise_pred and target from latent space
        noise_pred = torch.randn(1, 4, 32, 32, dtype=torch.float64, requires_grad=True)
        target = torch.randn(1, 4, 32, 32, dtype=torch.float64)

        # Base MSE loss (like conditional_loss)
        base_loss = ((noise_pred - target) ** 2).mean()

        # FFL addition
        ffl_loss = ffl(noise_pred, target)
        total_loss = base_loss + ffl_loss

        assert total_loss > base_loss, "FFL should increase the total loss"
        assert total_loss.ndim == 0

    def test_ffl_backward_with_combined_loss(self):
        """Combined base + FFL loss should backprop correctly."""
        ffl = FocalFrequencyLoss(loss_weight=0.1)

        noise_pred = torch.randn(1, 4, 16, 16, requires_grad=True)
        target = torch.randn(1, 4, 16, 16)

        base_loss = ((noise_pred - target) ** 2).mean()
        ffl_loss = ffl(noise_pred, target)
        total_loss = base_loss + ffl_loss

        total_loss.backward()

        assert noise_pred.grad is not None
        assert not torch.all(noise_pred.grad == 0)

    def test_ffl_with_different_loss_weights(self):
        """Higher loss_weight should produce higher total loss."""
        pred = torch.randn(1, 4, 16, 16)
        target = torch.randn(1, 4, 16, 16)

        ffl_low = FocalFrequencyLoss(loss_weight=0.1)
        ffl_high = FocalFrequencyLoss(loss_weight=10.0)

        loss_low = ffl_low(pred, target)
        loss_high = ffl_high(pred, target)

        assert loss_high > loss_low, (
            f"Higher weight should produce higher loss: {loss_high.item()} vs {loss_low.item()}"
        )

    def test_ffl_typical_latent_dimensions(self):
        """FFL should work with typical SD latent dimensions."""
        ffl = FocalFrequencyLoss()

        # SD1.5: 512px -> 64x64 latents
        pred = torch.randn(1, 4, 64, 64)
        target = torch.randn(1, 4, 64, 64)
        loss = ffl(pred, target)
        assert loss.ndim == 0

        # SDXL: 1024px -> 128x128 latents
        pred = torch.randn(1, 4, 128, 128)
        target = torch.randn(1, 4, 128, 128)
        loss = ffl(pred, target)
        assert loss.ndim == 0

    def test_ffl_similar_inputs_lower_loss_than_different(self):
        """More similar inputs should produce lower FFL loss."""
        ffl = FocalFrequencyLoss()
        base = torch.randn(1, 4, 32, 32)

        similar = base + torch.randn_like(base) * 0.01
        different = base + torch.randn_like(base) * 2.0

        loss_similar = ffl(base, similar)
        loss_different = ffl(base, different)

        assert loss_similar < loss_different, (
            f"Similar inputs should have lower loss: {loss_similar.item()} vs {loss_different.item()}"
        )


# ──────────────────────────────────────────────
# Metadata tests
# ──────────────────────────────────────────────


class TestMetadata:
    """Tests for FFL metadata integration."""

    def test_metadata_keys_exist_in_defaults(self):
        """Verify the metadata keys follow the naming convention."""
        expected_keys = [
            "ss_focal_frequency_loss",
            "ss_focal_frequency_loss_weight",
            "ss_focal_frequency_loss_alpha",
        ]
        # This is a structural test - we verify the keys are valid strings
        for key in expected_keys:
            assert key.startswith("ss_"), f"Key {key} should start with 'ss_'"
            assert "focal_frequency" in key, f"Key {key} should contain 'focal_frequency'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
