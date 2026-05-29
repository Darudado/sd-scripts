"""
Test zero_lr_warmup with custom schedulers (RexAnnealingWarmRestarts, CosineAnnealingWarmRestarts).

Verifies that when zero_lr_warmup is enabled and a custom scheduler has warmup_steps
in lr_scheduler_kwargs, the LR is forced to 0 during the warmup phase.

This tests the fix for: "zero lr warmup is not being applied to RexAnnealingWarmRestarts warmup"
"""

import sys
import os
import argparse
import torch
import math

# Add paths so we can import the modules we need
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "custom_scheduler"))

from LoraEasyCustomOptimizer.RexAnnealingWarmRestarts import RexAnnealingWarmRestarts
from LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts import CosineAnnealingWarmRestarts


def build_args(
    lr_scheduler_type: str = "",
    lr_warmup_steps=0,
    zero_lr_warmup: bool = False,
    lr_scheduler_args: list = None,
    lr_scheduler_num_cycles: int = 1,
    max_train_steps: int = 1000,
    lr_scheduler="constant",
    lr_decay_steps=None,
    lr_scheduler_power=1.0,
    lr_scheduler_timescale=None,
    lr_scheduler_min_lr_ratio=None,
    optimizer_type="AdamW",
    validation_split=0.0,
):
    """Build a minimal args namespace that mimics what get_scheduler_fix expects."""
    args = argparse.Namespace(
        lr_scheduler_type=lr_scheduler_type,
        lr_warmup_steps=lr_warmup_steps,
        zero_lr_warmup=zero_lr_warmup,
        lr_scheduler_args=lr_scheduler_args or [],
        lr_scheduler_num_cycles=lr_scheduler_num_cycles,
        max_train_steps=max_train_steps,
        lr_scheduler=lr_scheduler,
        lr_decay_steps=lr_decay_steps,
        lr_scheduler_power=lr_scheduler_power,
        lr_scheduler_timescale=lr_scheduler_timescale,
        lr_scheduler_min_lr_ratio=lr_scheduler_min_lr_ratio,
        optimizer_type=optimizer_type,
        validation_split=validation_split,
    )
    return args


def test_rex_annealing_zero_lr_warmup_enabled():
    """When zero_lr_warmup=True and warmup_steps is in lr_scheduler_args,
    the returned scheduler should wrap with ZeroLRWarmupScheduler and
    return LR=0 during warmup steps."""
    base_lr = 1e-4
    warmup_steps = 50
    total_steps = 500

    model = torch.nn.Linear(10, 10, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    args = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
        lr_warmup_steps=0,  # must be 0 for custom schedulers
        zero_lr_warmup=True,
        lr_scheduler_args=[f"warmup_steps={warmup_steps}", f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9", "d=0.9"],
        max_train_steps=total_steps,
    )

    # Import the actual get_scheduler_fix function
    from library.train_util import get_scheduler_fix

    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    # The scheduler should be a ZeroLRWarmupScheduler wrapper
    from library.train_util import ZeroLRWarmupScheduler
    assert isinstance(scheduler, ZeroLRWarmupScheduler), (
        f"Expected ZeroLRWarmupScheduler, got {type(scheduler).__name__}"
    )
    assert scheduler.warmup_steps == warmup_steps, (
        f"Expected warmup_steps={warmup_steps}, got {scheduler.warmup_steps}"
    )

    # During warmup (steps 1 to warmup_steps), LR should be 0
    for step in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        for lr in last_lr:
            assert lr == 0.0, f"Step {step}: Expected LR=0 during warmup, got {lr}"

    # After warmup, LR should be non-zero (following Rex annealing curve)
    optimizer.step()
    scheduler.step()
    last_lr = scheduler.get_last_lr()
    for lr in last_lr:
        assert lr > 0.0, f"Step {warmup_steps + 1}: Expected LR > 0 after warmup, got {lr}"

    print("  PASS: RexAnnealingWarmRestarts with zero_lr_warmup=True")


def test_rex_annealing_zero_lr_warmup_disabled():
    """When zero_lr_warmup=False, the returned scheduler should NOT be wrapped
    and LR should ramp from min_lr during warmup (not zero)."""
    base_lr = 1e-4
    warmup_steps = 50
    total_steps = 500

    model = torch.nn.Linear(10, 10, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    args = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
        lr_warmup_steps=0,
        zero_lr_warmup=False,
        lr_scheduler_args=[f"warmup_steps={warmup_steps}", f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9", "d=0.9"],
        max_train_steps=total_steps,
    )

    from library.train_util import get_scheduler_fix, ZeroLRWarmupScheduler

    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    # Should NOT be wrapped with ZeroLRWarmupScheduler
    assert not isinstance(scheduler, ZeroLRWarmupScheduler), (
        f"Should not be ZeroLRWarmupScheduler when zero_lr_warmup=False, got {type(scheduler).__name__}"
    )
    # Should be the raw RexAnnealingWarmRestarts
    assert isinstance(scheduler, RexAnnealingWarmRestarts), (
        f"Expected RexAnnealingWarmRestarts, got {type(scheduler).__name__}"
    )

    # During warmup, LR should ramp from min_lr (not zero)
    optimizer.step()
    scheduler.step()
    last_lr = scheduler.get_last_lr()
    for lr in last_lr:
        # At step 1 of warmup, lr should be approximately min_lr (not zero)
        assert lr > 0.0, f"Step 1: Expected LR > 0 (min_lr ramp) when zero_lr_warmup=False, got {lr}"

    print("  PASS: RexAnnealingWarmRestarts with zero_lr_warmup=False (no wrapping)")


def test_cosine_annealing_zero_lr_warmup_enabled():
    """CosineAnnealingWarmRestarts should also benefit from the fix."""
    base_lr = 1e-4
    warmup_steps = 50
    total_steps = 500

    model = torch.nn.Linear(10, 10, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    args = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts",
        lr_warmup_steps=0,
        zero_lr_warmup=True,
        lr_scheduler_args=[f"warmup_steps={warmup_steps}", f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9"],
        max_train_steps=total_steps,
    )

    from library.train_util import get_scheduler_fix, ZeroLRWarmupScheduler

    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    assert isinstance(scheduler, ZeroLRWarmupScheduler), (
        f"Expected ZeroLRWarmupScheduler, got {type(scheduler).__name__}"
    )

    # During warmup, LR should be 0
    for step in range(1, warmup_steps + 1):
        optimizer.step()
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        for lr in last_lr:
            assert lr == 0.0, f"Step {step}: Expected LR=0 during warmup, got {lr}"

    # After warmup, LR should be non-zero
    optimizer.step()
    scheduler.step()
    last_lr = scheduler.get_last_lr()
    for lr in last_lr:
        assert lr > 0.0, f"Step {warmup_steps + 1}: Expected LR > 0 after warmup, got {lr}"

    print("  PASS: CosineAnnealingWarmRestarts with zero_lr_warmup=True")


def test_custom_scheduler_no_warmup_steps_zero_lr_warmup_ignored():
    """If a custom scheduler has no warmup_steps in kwargs, zero_lr_warmup should be a no-op."""
    base_lr = 1e-4
    total_steps = 500

    model = torch.nn.Linear(10, 10, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # RexAnnealingWarmRestarts with warmup_steps=0 (default)
    args = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
        lr_warmup_steps=0,
        zero_lr_warmup=True,
        lr_scheduler_args=[f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9", "d=0.9"],  # no warmup_steps
        max_train_steps=total_steps,
    )

    from library.train_util import get_scheduler_fix, ZeroLRWarmupScheduler

    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    # Should NOT be wrapped since warmup_steps defaults to 0
    assert not isinstance(scheduler, ZeroLRWarmupScheduler), (
        "Should not wrap with ZeroLRWarmupScheduler when warmup_steps not in kwargs"
    )

    print("  PASS: Custom scheduler without warmup_steps - zero_lr_warmup ignored")


def test_rex_warmup_then_annealing_curve_unaffected():
    """Verify that after the zero-warmup phase, the Rex scheduler's annealing
    curve is identical to what it would produce without wrapping (just shifted)."""
    base_lr = 1e-4
    warmup_steps = 30
    total_steps = 200

    # Run WITHOUT zero_lr_warmup to get the reference curve
    model_ref = torch.nn.Linear(10, 10, device="cuda")
    opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=base_lr)
    args_ref = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
        lr_warmup_steps=0,
        zero_lr_warmup=False,
        lr_scheduler_args=[f"warmup_steps={warmup_steps}", f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9", "d=0.9"],
        max_train_steps=total_steps,
    )
    from library.train_util import get_scheduler_fix
    sched_ref = get_scheduler_fix(args_ref, opt_ref, num_processes=1)

    ref_lrs = []
    for step in range(1, total_steps + 1):
        opt_ref.step()
        sched_ref.step()
        ref_lrs.append(sched_ref.get_last_lr()[0])

    # Run WITH zero_lr_warmup
    model_test = torch.nn.Linear(10, 10, device="cuda")
    opt_test = torch.optim.AdamW(model_test.parameters(), lr=base_lr)
    args_test = build_args(
        lr_scheduler_type="LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
        lr_warmup_steps=0,
        zero_lr_warmup=True,
        lr_scheduler_args=[f"warmup_steps={warmup_steps}", f"first_cycle_max_steps={total_steps}", "min_lr=1e-6", "gamma=0.9", "d=0.9"],
        max_train_steps=total_steps,
    )
    sched_test = get_scheduler_fix(args_test, opt_test, num_processes=1)

    test_lrs = []
    for step in range(1, total_steps + 1):
        opt_test.step()
        sched_test.step()
        test_lrs.append(sched_test.get_last_lr()[0])

    # During warmup: test should be 0, ref should be > 0
    for i in range(warmup_steps):
        assert test_lrs[i] == 0.0, f"Step {i+1}: Expected 0, got {test_lrs[i]}"
        assert ref_lrs[i] > 0.0, f"Step {i+1}: Reference should be > 0"

    # After warmup: the Rex scheduler's internal state has been stepped the same number
    # of times, so the annealing curve values should match (since ZeroLRWarmupScheduler
    # still steps the inner scheduler every call)
    for i in range(warmup_steps, total_steps):
        # Allow small floating point differences
        assert abs(test_lrs[i] - ref_lrs[i]) < 1e-10, (
            f"Step {i+1}: LR mismatch after warmup. "
            f"test={test_lrs[i]}, ref={ref_lrs[i]}, diff={abs(test_lrs[i] - ref_lrs[i])}"
        )

    print("  PASS: Annealing curve is identical after zero-warmup phase")


if __name__ == "__main__":
    print("Testing zero_lr_warmup with custom schedulers...")
    print()

    test_rex_annealing_zero_lr_warmup_enabled()
    test_rex_annealing_zero_lr_warmup_disabled()
    test_cosine_annealing_zero_lr_warmup_enabled()
    test_custom_scheduler_no_warmup_steps_zero_lr_warmup_ignored()
    test_rex_warmup_then_annealing_curve_unaffected()

    print()
    print("All tests passed!")
