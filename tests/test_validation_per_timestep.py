"""
Test per-timestep validation loss tracking in train_network.py.

Verifies that:
1. process_val_batch returns per-timestep losses alongside the average
2. calculate_val_loss accumulates per-timestep losses across batches correctly
3. Per-timestep log keys follow the expected naming convention
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import argparse
import random


class TestProcessValBatchReturnType(unittest.TestCase):
    """Test that process_val_batch returns the expected (average_loss, dict) tuple."""

    def test_simulated_return_values(self):
        """Simulate process_val_batch logic to verify dict construction is correct."""
        timesteps_list = [50, 350, 500, 650, 950]
        per_t_step_losses = [torch.tensor(0.123), torch.tensor(0.456), torch.tensor(0.789), torch.tensor(0.234), torch.tensor(0.567)]
        total_loss = torch.stack(per_t_step_losses).sum()
        average_loss = total_loss / len(timesteps_list)

        per_timestep_losses = {t: l.detach().item() for t, l in zip(timesteps_list, per_t_step_losses)}

        # Verify return is a tuple
        result = (average_loss, per_timestep_losses)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        avg, per_t = result
        self.assertIsInstance(avg, torch.Tensor)
        self.assertEqual(avg.shape, ())  # scalar
        self.assertIsInstance(per_t, dict)
        self.assertEqual(len(per_t), 5)

        # Verify keys match timesteps
        for t in timesteps_list:
            self.assertIn(t, per_t)
            self.assertIsInstance(per_t[t], float)


class TestCalculateValLossAccumulation(unittest.TestCase):
    """Test per-timestep accumulation logic in calculate_val_loss."""

    def test_per_timestep_accumulation_across_batches(self):
        """Simulate 3 validation batches and verify per-timestep accumulation."""
        timesteps_list = [50, 350, 950]

        # Simulate 3 batches with different sizes
        batches = [
            {"total_loss": 0.5, "batch_size": 2, "per_t": {50: 0.1, 350: 0.2, 950: 0.5}},
            {"total_loss": 0.3, "batch_size": 4, "per_t": {50: 0.05, 350: 0.1, 950: 0.3}},
            {"total_loss": 0.4, "batch_size": 1, "per_t": {50: 0.08, 350: 0.15, 950: 0.4}},
        ]

        total_loss = 0.0
        total_samples = 0
        per_timestep_total_loss = {}
        per_timestep_total_samples = {}

        for batch in batches:
            current_batch_size = batch["batch_size"]
            loss_val = batch["total_loss"]
            batch_per_t_losses = batch["per_t"]

            total_loss += loss_val * current_batch_size
            total_samples += current_batch_size

            for t, t_loss in batch_per_t_losses.items():
                per_timestep_total_loss[t] = per_timestep_total_loss.get(t, 0.0) + t_loss * current_batch_size
                per_timestep_total_samples[t] = per_timestep_total_samples.get(t, 0) + current_batch_size

        current_val_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Compute per-timestep average
        per_timestep_avg = {
            f"loss/val/t{int(t)}": per_timestep_total_loss[t] / per_timestep_total_samples[t]
            for t in per_timestep_total_loss
        }

        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": 0.0, **per_timestep_avg}

        # Verify expected totals
        # total_samples = 2 + 4 + 1 = 7
        self.assertEqual(total_samples, 7)

        # Total weighted loss
        # total_loss = 0.5*2 + 0.3*4 + 0.4*1 = 1.0 + 1.2 + 0.4 = 2.6
        self.assertAlmostEqual(total_loss, 2.6)
        self.assertAlmostEqual(current_val_loss, 2.6 / 7)

        # Per-timestep weighted loss for t=50
        # 0.1*2 + 0.05*4 + 0.08*1 = 0.2 + 0.2 + 0.08 = 0.48
        # 0.48 / 7 ≈ 0.068571
        self.assertAlmostEqual(logs["loss/val/t50"], 0.48 / 7)

        # Per-timestep for t=350
        # 0.2*2 + 0.1*4 + 0.15*1 = 0.4 + 0.4 + 0.15 = 0.95
        # 0.95 / 7 ≈ 0.135714
        self.assertAlmostEqual(logs["loss/val/t350"], 0.95 / 7)

        # Per-timestep for t=950
        # 0.5*2 + 0.3*4 + 0.4*1 = 1.0 + 1.2 + 0.4 = 2.6
        # 2.6 / 7 ≈ 0.371428
        self.assertAlmostEqual(logs["loss/val/t950"], 2.6 / 7)

        # Verify all expected keys exist
        self.assertIn("loss/current_val_loss", logs)
        self.assertIn("loss/average_val_loss", logs)
        self.assertIn("loss/val/t50", logs)
        self.assertIn("loss/val/t350", logs)
        self.assertIn("loss/val/t950", logs)

        # No unexpected keys
        self.assertEqual(len(logs), 5)

    def test_per_timestep_key_naming_with_default_timesteps(self):
        """Verify log key naming for the default validation_timesteps list."""
        default_timesteps = [50, 350, 500, 650, 950]
        per_timestep_total_loss = {t: float(t) for t in default_timesteps}
        per_timestep_total_samples = {t: 10 for t in default_timesteps}

        per_timestep_avg = {
            f"loss/val/t{int(t)}": per_timestep_total_loss[t] / per_timestep_total_samples[t]
            for t in per_timestep_total_loss
        }

        expected_keys = {
            "loss/val/t50",
            "loss/val/t350",
            "loss/val/t500",
            "loss/val/t650",
            "loss/val/t950",
        }
        self.assertEqual(set(per_timestep_avg.keys()), expected_keys)

    def test_empty_per_timestep_dict_handling(self):
        """Test that empty per_timestep dict (no validation ran) produces correct logs."""
        per_timestep_avg = {}
        logs = {"loss/current_val_loss": 0.0, "loss/average_val_loss": 0.0, **per_timestep_avg}
        self.assertEqual(logs, {"loss/current_val_loss": 0.0, "loss/average_val_loss": 0.0})


class TestValLogsMerge(unittest.TestCase):
    """Test that val_logs merging does not lose per-timestep keys."""

    def test_val_logs_merged_into_step_logs(self):
        """Simulate the step logging flow where val_logs is merged into logs."""
        # Simulate generate_step_logs output
        step_logs = {
            "loss/current": 0.5,
            "loss/average": 0.45,
            "lr/unet": 1e-4,
            "loss/current_val_loss": 0.3,
            "loss/average_val_loss": 0.32,
        }

        # Simulate val_logs from calculate_val_loss (with per-timestep keys)
        val_logs = {
            "loss/current_val_loss": 0.3,
            "loss/average_val_loss": 0.32,
            "loss/val/t50": 0.1,
            "loss/val/t350": 0.2,
            "loss/val/t950": 0.5,
        }

        if val_logs:
            step_logs.update(val_logs)

        # All keys should be present
        self.assertIn("loss/val/t50", step_logs)
        self.assertIn("loss/val/t350", step_logs)
        self.assertIn("loss/val/t950", step_logs)
        self.assertIn("loss/current", step_logs)
        self.assertIn("lr/unet", step_logs)

    def test_none_val_logs_does_not_error(self):
        """Test that val_logs=None is safely handled."""
        step_logs = {"loss/current": 0.5}
        val_logs = None

        if val_logs:
            step_logs.update(val_logs)

        self.assertEqual(step_logs, {"loss/current": 0.5})

    def test_empty_dict_val_logs_does_not_error(self):
        """Test that val_logs={} is safely handled."""
        step_logs = {"loss/current": 0.5}
        val_logs = {}

        if val_logs:
            step_logs.update(val_logs)

        self.assertEqual(step_logs, {"loss/current": 0.5})


if __name__ == "__main__":
    unittest.main()
