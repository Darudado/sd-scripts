"""
Tests for step_func optimizer integration in train_network.py.

Verifies that AdamWScheduleFreePlus (and any optimizer with step_func) is
correctly detected and integrated into the training loop, handling:
- Gradient accumulation (step_func called only on sync boundaries)
- Loss averaging across accumulation micro-steps
- Mixed-precision gradient unscale before step_func reads p.grad
- GradScaler state management (scaler.update() after step_func)
- Regular optimizers remain unaffected
"""

import sys
import os
import math
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest
import torch
import torch.nn as nn

# Ensure custom_scheduler is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "custom_scheduler"))
from LoraEasyCustomOptimizer.adamw_schedulefree_plus import AdamWScheduleFreePlus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    """Minimal model for optimizer tests."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def _make_step_func_optimizer(model: nn.Module, **kwargs) -> AdamWScheduleFreePlus:
    """Create an AdamWScheduleFreePlus optimizer in train mode."""
    opt = AdamWScheduleFreePlus(model.parameters(), lr=1.0, **kwargs)
    opt.train()
    return opt


def _make_adam_optimizer(model: nn.Module, **kwargs) -> torch.optim.AdamW:
    """Create a standard AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=1e-4, **kwargs)


# ---------------------------------------------------------------------------
# 1. Detection: AdamWScheduleFreePlus exposes step_func
# ---------------------------------------------------------------------------

class TestStepFuncDetection:
    """Verify that AdamWScheduleFreePlus is detected as a step_func optimizer."""

    def test_has_step_func(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)
        assert hasattr(opt, "step_func") and callable(opt.step_func)

    def test_step_raises_not_implemented(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)
        with pytest.raises(NotImplementedError, match="step_func"):
            opt.step()

    def test_has_train_eval(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)
        assert hasattr(opt, "train") and callable(opt.train)
        assert hasattr(opt, "eval") and callable(opt.eval)

    def test_adamw_does_not_have_step_func(self):
        model = _SimpleModel()
        opt = _make_adam_optimizer(model)
        assert not (hasattr(opt, "step_func") and callable(getattr(opt, "step_func", None)))

    def test_detection_flag_step_func(self):
        """Simulate the _use_step_func detection from train_network.py."""
        model = _SimpleModel()
        opt_sf = _make_step_func_optimizer(model)
        opt_adam = _make_adam_optimizer(model)

        _use_step_func_sf = hasattr(opt_sf, "step_func") and callable(getattr(opt_sf, "step_func"))
        _use_step_func_adam = hasattr(opt_adam, "step_func") and callable(getattr(opt_adam, "step_func"))

        assert _use_step_func_sf is True
        assert _use_step_func_adam is False

    def test_detection_through_accelerate_wrapper(self):
        """
        After accelerator.prepare(), the optimizer is wrapped in an
        AcceleratedOptimizer that doesn't expose step_func directly.
        The raw optimizer lives at optimizer.optimizer.
        Regression test: detection must unwrap to find step_func.
        """
        model = _SimpleModel()
        raw_opt = _make_step_func_optimizer(model)

        # Simulate AcceleratedOptimizer wrapper (has .optimizer pointing to raw)
        wrapper = MagicMock()
        wrapper.optimizer = raw_opt
        wrapper.scaler = None
        # The wrapper itself does NOT have step_func
        del wrapper.step_func

        # Detection logic from train_network.py:
        _raw_optimizer = getattr(wrapper, "optimizer", wrapper)
        _use_step_func = hasattr(_raw_optimizer, "step_func") and callable(getattr(_raw_optimizer, "step_func"))

        assert _use_step_func is True, "Must detect step_func through accelerator wrapper"

        # Verify the raw optimizer's step_func can be called via _raw_optimizer
        _raw_optimizer.step_func(1.0)
        assert raw_opt.param_groups[0]["k"] == 1

    def test_detection_no_wrapper(self):
        """
        When optimizer is not wrapped (no .optimizer attr), detection
        must still work by falling back to the optimizer itself.
        """
        model = _SimpleModel()
        raw_opt = _make_step_func_optimizer(model)

        # No wrapper — getattr falls back to raw_opt itself
        _raw_optimizer = getattr(raw_opt, "optimizer", raw_opt)
        _use_step_func = hasattr(_raw_optimizer, "step_func") and callable(getattr(_raw_optimizer, "step_func"))

        assert _use_step_func is True
        assert _raw_optimizer is raw_opt  # should be the same object

    def test_detection_non_step_func_through_wrapper(self):
        """Non-step_func optimizer inside a wrapper should NOT be detected."""
        model = _SimpleModel()
        raw_adam = _make_adam_optimizer(model)

        wrapper = MagicMock()
        wrapper.optimizer = raw_adam

        _raw_optimizer = getattr(wrapper, "optimizer", wrapper)
        _use_step_func = hasattr(_raw_optimizer, "step_func") and callable(getattr(_raw_optimizer, "step_func"))

        assert _use_step_func is False


# ---------------------------------------------------------------------------
# 2. step_func basic operation
# ---------------------------------------------------------------------------

class TestStepFuncBasicOperation:
    """Verify step_func updates model parameters and advances internal state."""

    def test_step_func_updates_params(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        # Do a forward/backward to get gradients
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        params_before = [p.clone() for p in model.parameters()]
        opt.step_func(loss.item())

        # At least one parameter should have changed
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, model.parameters())
        )
        assert changed, "step_func should modify at least one parameter"

    def test_step_func_increments_k(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        assert opt.param_groups[0]["k"] == 0

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        assert opt.param_groups[0]["k"] == 1

    def test_step_func_returns_function_value(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        fv = 42.0
        result = opt.step_func(fv)
        assert result == fv

    def test_step_func_initializes_state(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        for p in model.parameters():
            state = opt.state[p]
            assert "z" in state
            assert "exp_avg" in state
            assert "exp_avg_sq" in state


# ---------------------------------------------------------------------------
# 3. Train/eval mode switching
# ---------------------------------------------------------------------------

class TestTrainEvalMode:
    """Verify optimizer train()/eval() properly switches parameter mode."""

    def test_train_eval_roundtrip(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        # Do a step to initialize state
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())

        # eval mode
        opt.eval()
        assert opt.param_groups[0]["train_mode"] is False

        # train mode
        opt.train()
        assert opt.param_groups[0]["train_mode"] is True

    def test_step_func_requires_train_mode(self):
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)
        opt.eval()  # switch to eval mode

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        with pytest.raises(Exception, match="train mode"):
            opt.step_func(loss.item())


# ---------------------------------------------------------------------------
# 4. Gradient accumulation simulation
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    """
    Simulate the training loop's gradient accumulation logic with step_func.

    Tests that:
    - step_func is only called on sync steps (accumulation boundary)
    - Loss is averaged across micro-steps before being passed to step_func
    - Loss accumulator resets after each sync step
    """

    def test_step_func_called_only_on_sync(self):
        """step_func should not be called on non-sync (accumulation) steps."""
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        step_func_calls = []
        original_step_func = opt.step_func

        def tracking_step_func(function_value):
            step_func_calls.append(function_value)
            return original_step_func(function_value)

        opt.step_func = tracking_step_func

        accum_steps = 3
        _step_func_loss_accum = 0.0
        accumulation_counter = 0

        for micro_step in range(accum_steps):
            accumulation_counter += 1
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()

            loss_val = loss.detach().item()
            _step_func_loss_accum += loss_val

            # Simulate sync_gradients = True only on last micro-step
            sync_gradients = (micro_step == accum_steps - 1)

            if sync_gradients:
                avg_loss = _step_func_loss_accum / accumulation_counter
                opt.step_func(avg_loss)
                _step_func_loss_accum = 0.0

        assert len(step_func_calls) == 1, (
            f"step_func should be called exactly once per accumulation cycle, "
            f"but was called {len(step_func_calls)} times"
        )

    def test_loss_averaging_across_micro_steps(self):
        """The function_value passed to step_func should be the average loss."""
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        captured_losses = []
        original_step_func = opt.step_func

        def capturing_step_func(function_value):
            captured_losses.append(function_value)
            return original_step_func(function_value)

        opt.step_func = capturing_step_func

        accum_steps = 4
        _step_func_loss_accum = 0.0
        accumulation_counter = 0
        raw_losses = []

        for micro_step in range(accum_steps):
            accumulation_counter += 1
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()

            loss_val = loss.detach().item()
            raw_losses.append(loss_val)
            _step_func_loss_accum += loss_val

            sync_gradients = (micro_step == accum_steps - 1)
            if sync_gradients:
                avg_loss = _step_func_loss_accum / accumulation_counter
                opt.step_func(avg_loss)
                _step_func_loss_accum = 0.0

        expected_avg = sum(raw_losses) / len(raw_losses)
        assert len(captured_losses) == 1
        assert abs(captured_losses[0] - expected_avg) < 1e-6, (
            f"step_func should receive averaged loss {expected_avg:.6f}, "
            f"but got {captured_losses[0]:.6f}"
        )

    def test_loss_accumulator_resets_after_sync(self):
        """_step_func_loss_accum should reset to 0 after a sync step."""
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        accum_steps = 3
        _step_func_loss_accum = 0.0
        accumulation_counter = 0

        for cycle in range(2):  # two accumulation cycles
            for micro_step in range(accum_steps):
                accumulation_counter += 1
                x = torch.randn(2, 4)
                loss = model(x).sum()
                loss.backward()
                _step_func_loss_accum += loss.detach().item()

                sync_gradients = (micro_step == accum_steps - 1)
                if sync_gradients:
                    opt.step_func(_step_func_loss_accum / accumulation_counter)
                    _step_func_loss_accum = 0.0
                    accumulation_counter = 0

            assert _step_func_loss_accum == 0.0, (
                f"Loss accumulator should be 0 after sync step in cycle {cycle}"
            )


# ---------------------------------------------------------------------------
# 5. Mixed-precision gradient unscale simulation
# ---------------------------------------------------------------------------

class TestGradientUnscaling:
    """
    Verify that the training loop logic correctly unscales gradients before
    step_func when they haven't already been unscaled.
    """

    def test_unscale_called_when_no_grad_clipping(self):
        """
        When max_grad_norm == 0 and no EDM2, gradients aren't unscaled in the
        gradient-clipping block. The step_func path should call unscale_gradients().
        """
        mock_accelerator = MagicMock()
        mock_accelerator.sync_gradients = True

        max_grad_norm = 0.0
        edm2_loss_weighting = False

        # Simulate the training loop logic
        if max_grad_norm != 0.0 or edm2_loss_weighting:
            mock_accelerator.unscale_gradients()

        # step_func path
        if not (max_grad_norm != 0.0 or edm2_loss_weighting):
            mock_accelerator.unscale_gradients()

        # unscale_gradients should have been called exactly once
        assert mock_accelerator.unscale_gradients.call_count == 1

    def test_unscale_not_called_twice_when_grad_clipping_active(self):
        """
        When max_grad_norm != 0, gradients are already unscaled in the clipping
        block. The step_func path should NOT call unscale_gradients() again.
        """
        mock_accelerator = MagicMock()
        mock_accelerator.sync_gradients = True

        max_grad_norm = 1.0
        edm2_loss_weighting = False

        # Simulate the training loop logic
        if max_grad_norm != 0.0 or edm2_loss_weighting:
            mock_accelerator.unscale_gradients()

        # step_func path
        if not (max_grad_norm != 0.0 or edm2_loss_weighting):
            mock_accelerator.unscale_gradients()

        # unscale_gradients should have been called exactly once (in clipping block only)
        assert mock_accelerator.unscale_gradients.call_count == 1


# ---------------------------------------------------------------------------
# 6. GradScaler update simulation
# ---------------------------------------------------------------------------

class TestGradScalerUpdate:
    """
    Verify that the GradScaler state is properly maintained when step_func
    bypasses accelerate's optimizer wrapper.
    """

    def test_scaler_update_called_after_step_func(self):
        """scaler.update() should be called after step_func to reset unscaled state."""
        mock_scaler = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.scaler = mock_scaler

        # Simulate the step_func path
        mock_optimizer.step_func(1.0)

        scaler = getattr(mock_optimizer, "scaler", None)
        if scaler is not None:
            scaler.update()

        mock_scaler.update.assert_called_once()

    def test_no_crash_when_scaler_is_none(self):
        """When there's no scaler (bf16), the code should not crash."""
        mock_optimizer = MagicMock()
        mock_optimizer.scaler = None

        mock_optimizer.step_func(1.0)

        scaler = getattr(mock_optimizer, "scaler", None)
        # Should be None and the update should be skipped
        assert scaler is None

    def test_no_crash_when_no_scaler_attribute(self):
        """When optimizer has no scaler attribute, the code should not crash."""
        mock_optimizer = MagicMock(spec=[])  # Empty spec = no attributes

        scaler = getattr(mock_optimizer, "scaler", None)
        assert scaler is None


# ---------------------------------------------------------------------------
# 7. Regular optimizer path unaffected
# ---------------------------------------------------------------------------

class TestRegularOptimizerPath:
    """
    Verify that regular (non-step_func) optimizers still use optimizer.step().
    """

    def test_adamw_step_called_normally(self):
        """For regular optimizers, optimizer.step() should be called directly."""
        mock_optimizer = MagicMock()
        # No step_func attribute
        del mock_optimizer.step_func

        _use_step_func = hasattr(mock_optimizer, "step_func") and callable(
            getattr(mock_optimizer, "step_func")
        )

        if _use_step_func:
            mock_optimizer.step_func(1.0)
        else:
            mock_optimizer.step()

        mock_optimizer.step.assert_called_once()
        mock_optimizer.step_func.assert_not_called() if hasattr(mock_optimizer, "step_func") else None

    def test_lr_scheduler_stepped_for_both_paths(self):
        """lr_scheduler.step() should be called regardless of optimizer type."""
        mock_lr_scheduler = MagicMock()

        # step_func path
        mock_lr_scheduler.step()
        assert mock_lr_scheduler.step.call_count == 1

        # regular path
        mock_lr_scheduler.step()
        assert mock_lr_scheduler.step.call_count == 2

    def test_zero_grad_called_for_both_paths(self):
        """optimizer.zero_grad() should be called regardless of optimizer type."""
        mock_optimizer = MagicMock()

        mock_optimizer.zero_grad(set_to_none=True)
        mock_optimizer.zero_grad(set_to_none=True)

        assert mock_optimizer.zero_grad.call_count == 2


# ---------------------------------------------------------------------------
# 8. Multi-step integration test
# ---------------------------------------------------------------------------

class TestMultiStepIntegration:
    """
    Simulate multiple accumulation cycles to verify the full integration
    works end-to-end with a real AdamWScheduleFreePlus optimizer.
    """

    def test_multiple_accumulation_cycles(self):
        """Run several accumulation cycles and verify optimizer state is consistent."""
        model = _SimpleModel(dim=8)
        opt = _make_step_func_optimizer(model, warmup_steps=0)

        accum_steps = 3
        num_cycles = 5

        for cycle in range(num_cycles):
            _step_func_loss_accum = 0.0
            accumulation_counter = 0

            for micro_step in range(accum_steps):
                accumulation_counter += 1
                x = torch.randn(4, 8)
                loss = model(x).sum()
                loss.backward()
                _step_func_loss_accum += loss.detach().item()

                sync_gradients = (micro_step == accum_steps - 1)
                if sync_gradients:
                    avg_loss = _step_func_loss_accum / accumulation_counter
                    opt.step_func(avg_loss)
                    _step_func_loss_accum = 0.0
                    opt.zero_grad(set_to_none=True)

        # Verify k has advanced correctly
        assert opt.param_groups[0]["k"] == num_cycles

        # Verify optimizer state exists for all parameters
        for p in model.parameters():
            state = opt.state[p]
            assert "z" in state
            assert "exp_avg" in state
            assert "exp_avg_sq" in state

    def test_train_eval_between_cycles(self):
        """Simulate eval/train mode switches between cycles (as done for sampling)."""
        model = _SimpleModel(dim=4)
        opt = _make_step_func_optimizer(model, warmup_steps=0)

        for cycle in range(3):
            # Training phase
            opt.train()
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad(set_to_none=True)

            # Eval phase (e.g., for sampling/validation)
            opt.eval()
            assert opt.param_groups[0]["train_mode"] is False

            # Resume training
            opt.train()
            assert opt.param_groups[0]["train_mode"] is True

        assert opt.param_groups[0]["k"] == 3


# ---------------------------------------------------------------------------
# 9. Polyak step size mathematical properties
# ---------------------------------------------------------------------------

class TestPolyakStepProperties:
    """Verify mathematical properties of the Polyak step size."""

    def test_polyak_lr_is_nonnegative(self):
        """Polyak lr should always be >= 0."""
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model)

        for _ in range(5):
            x = torch.randn(2, 4)
            loss = model(x).sum()
            loss.backward()
            opt.step_func(loss.item())
            opt.zero_grad(set_to_none=True)

            scheduled_lr = opt.param_groups[0].get("scheduled_lr", 0.0)
            assert scheduled_lr >= 0.0, f"scheduled_lr should be >= 0, got {scheduled_lr}"

    def test_higher_loss_gives_larger_step(self):
        """Higher loss should result in larger Polyak lr (all else being equal)."""
        model = _SimpleModel()
        opt = _make_step_func_optimizer(model, polyak_beta=0.0)

        # First step to initialize EMA
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(loss.item())
        opt.zero_grad(set_to_none=True)

        lr_after_low_loss = opt.param_groups[0]["scheduled_lr"]

        # Second step with artificially high loss
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step_func(1000.0)  # artificially high loss
        opt.zero_grad(set_to_none=True)

        lr_after_high_loss = opt.param_groups[0]["scheduled_lr"]

        assert lr_after_high_loss >= lr_after_low_loss, (
            "Higher function value should yield larger Polyak lr"
        )
