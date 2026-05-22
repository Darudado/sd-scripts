"""
Test zero_lr_warmup with WARMUP_STABLE_DECAY scheduler.

Investigates reported bug: "with warmup stable decay, it decays far too soon 
and sits at zero LR for an extended period."

Key scenarios tested:
1. Basic zero_lr_warmup + WSD without gradient accumulation (no accelerate)
2. With gradient_accumulation_steps=2 (no accelerate, manual sync_gradients)
3. Full AcceleratedOptimizer + AcceleratedScheduler wrapping simulation
4. Epoch-based max_train_steps calculation
"""

import torch
import math
from functools import partial
from accelerate.scheduler import AcceleratedScheduler
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.state import GradientState


# -------- Replicated from transformers.optimization --------
def _get_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_type: str,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        if warmup_type == "linear":
            factor = progress
        elif warmup_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)

    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0

    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        if decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)
    return min_lr_ratio


# -------- Replicated from train_util.py --------
class ZeroLRWarmupScheduler:
    """Scheduler wrapper that returns LR=0 during warmup, then delegates."""

    def __init__(self, optimizer: torch.optim.Optimizer, inner_scheduler, warmup_steps: int):
        self.optimizer = optimizer
        self.inner_scheduler = inner_scheduler
        self.warmup_steps = warmup_steps
        self._step_count = 0

    def step(self, epoch=None):
        self.inner_scheduler.step(epoch)
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0

    def get_last_lr(self):
        if self._step_count <= self.warmup_steps:
            return [0.0 for _ in self.optimizer.param_groups]
        return self.inner_scheduler.get_last_lr()


# -------- Helper: build scheduler WITHOUT accelerate wrapping --------
def build_simple_scheduler_chain(
    base_lr: float = 1e-3,
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    num_decay_steps: int = 100,
    zero_lr_warmup: bool = True,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    """Build: optimizer -> inner LambdaLR -> [ZeroLRWarmupScheduler]. No accelerate wrapping."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        warmup_type="linear",
        decay_type="cosine",
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    zero_lr_steps = num_warmup_steps if zero_lr_warmup and num_warmup_steps else 0
    wrapper = ZeroLRWarmupScheduler(optimizer, inner_scheduler, warmup_steps=zero_lr_steps)

    outer_scheduler = wrapper  # outermost is ZeroLRWarmupScheduler (or inner if no zero_lr)

    meta = {
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "num_stable_steps": num_stable_steps,
        "num_decay_steps": num_decay_steps,
        "zero_lr_warmup": zero_lr_warmup,
        "zero_lr_warmup_steps": zero_lr_steps,
    }
    return optimizer, model, wrapper, outer_scheduler, inner_scheduler, meta


# -------- Helper: build scheduler WITH full accelerate wrapping --------
def build_accelerated_scheduler_chain(
    base_lr: float = 1e-3,
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    num_decay_steps: int = 100,
    zero_lr_warmup: bool = True,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    gradient_accumulation_steps: int = 1,
):
    """Build the full chain with AcceleratedOptimizer + AcceleratedScheduler wrapping."""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

    num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        warmup_type="linear",
        decay_type="cosine",
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    inner_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    zero_lr_steps = num_warmup_steps if zero_lr_warmup and num_warmup_steps else 0
    wrapper = ZeroLRWarmupScheduler(optimizer, inner_scheduler, warmup_steps=zero_lr_steps)

    # Wrap with accelerate: optimizer first, then scheduler
    accelerated_optimizer = AcceleratedOptimizer(
        optimizer,
        device_placement=False,
        scaler=None,
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_accumulation_kwargs={},
    )
    accelerated_scheduler = AcceleratedScheduler(
        wrapper,
        accelerated_optimizer,
        step_with_optimizer=True,
        split_batches=False,
    )

    meta = {
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "num_stable_steps": num_stable_steps,
        "num_decay_steps": num_decay_steps,
        "zero_lr_warmup": zero_lr_warmup,
        "zero_lr_warmup_steps": zero_lr_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    return model, accelerated_optimizer, wrapper, accelerated_scheduler, inner_scheduler, meta


def format_lr(lr):
    if lr == 0.0:
        return "0.000000"
    return f"{lr:.6f}"


# ============ TEST 1 ============
def test_basic_zero_lr_warmup_wsd():
    """Test basic zero_lr_warmup + WSD without gradient accumulation or accelerate wrapping."""
    print("=" * 70)
    print("TEST 1: Basic zero_lr_warmup + WSD (no accelerate, no grad accumulation)")
    print("=" * 70)

    num_training_steps = 1000
    num_warmup_steps = 100
    num_decay_steps = 100

    for zero_lr in [False, True]:
        print(f"\n--- zero_lr_warmup={zero_lr} ---")
        optimizer, model, wrapper, outer_sched, inner_sched, meta = build_simple_scheduler_chain(
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=num_decay_steps,
            zero_lr_warmup=zero_lr,
        )
        print(f"  Config: {meta}")

        lr_values = []
        for step in range(num_training_steps):
            optimizer.step()
            outer_sched.step()
            optimizer.zero_grad(set_to_none=True)
            lr_values.append(optimizer.param_groups[0]["lr"])

        warmup_end = num_warmup_steps
        stable_mid = num_warmup_steps + meta["num_stable_steps"] // 2
        decay_start = num_warmup_steps + meta["num_stable_steps"]
        decay_mid = num_warmup_steps + meta["num_stable_steps"] + num_decay_steps // 2
        final = num_training_steps - 1

        print(f"  Step {warmup_end:4d} (warmup end):   LR={format_lr(lr_values[warmup_end])}")
        print(f"  Step {stable_mid:4d} (stable mid):   LR={format_lr(lr_values[stable_mid])}")
        print(f"  Step {decay_start:4d} (decay start):  LR={format_lr(lr_values[decay_start])}")
        print(f"  Step {decay_mid:4d} (decay mid):    LR={format_lr(lr_values[decay_mid])}")
        print(f"  Step {final:4d} (final):        LR={format_lr(lr_values[final])}")

        # Verify basic expectations
        assert abs(lr_values[stable_mid] - 1e-3) < 1e-8, f"FAIL: Stable LR mismatch at step {stable_mid}: {lr_values[stable_mid]}"
        assert lr_values[final] < 1e-5, f"FAIL: Final LR should be ~0, got {lr_values[final]}"

        if zero_lr:
            assert lr_values[0] == 0.0, f"FAIL: Warmup step 0 should be 0, got {lr_values[0]}"
            assert lr_values[num_warmup_steps - 1] == 0.0, f"FAIL: Last warmup step should be 0"
            assert abs(lr_values[num_warmup_steps] - 1e-3) < 1e-8, f"FAIL: First post-warmup should be max_lr, got {lr_values[num_warmup_steps]}"

    print("\n  TEST 1 PASSED\n")


# ============ TEST 2 ============
def test_with_gradient_accumulation_manual():
    """Test zero_lr_warmup + WSD simulating gradient accumulation by only stepping on sync boundaries."""
    print("=" * 70)
    print("TEST 2: zero_lr_warmup + WSD with manual gradient accumulation (no accelerate)")
    print("=" * 70)

    num_training_steps = 500
    num_warmup_steps = 50
    num_decay_steps = 50
    ga_steps = 2

    for zero_lr in [False, True]:
        print(f"\n--- zero_lr_warmup={zero_lr} ---")
        optimizer, model, wrapper, outer_sched, inner_sched, meta = build_simple_scheduler_chain(
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=num_decay_steps,
            zero_lr_warmup=zero_lr,
        )
        print(f"  Config: {meta}")

        # Simulate the training loop correctly:
        # Only step optimizer+scheduler on sync boundaries
        lr_values = []
        optimizer_step = 0
        inner_step_counts = []
        zero_step_counts = []

        for batch in range(num_training_steps * ga_steps):
            is_sync = ((batch + 1) % ga_steps == 0)
            if is_sync:
                # Only call optimizer.step() + scheduler.step() on sync steps
                # (this is what AcceleratedOptimizer + AcceleratedScheduler does)
                optimizer.step()
                outer_sched.step()
                optimizer.zero_grad(set_to_none=True)
                lr_values.append(optimizer.param_groups[0]["lr"])
                optimizer_step += 1
                inner_step_counts.append(inner_sched.last_epoch)
                zero_step_counts.append(wrapper._step_count)

            if optimizer_step >= num_training_steps:
                break

        print(f"  Total backward passes: {batch + 1}")
        print(f"  Total optimizer steps: {optimizer_step}")
        print(f"  Inner scheduler last_epoch: {inner_sched.last_epoch}")
        print(f"  ZeroLRWarmupScheduler._step_count: {wrapper._step_count}")

        warmup_end = num_warmup_steps
        decay_start = num_warmup_steps + meta["num_stable_steps"]
        final = num_training_steps - 1

        if warmup_end < len(lr_values):
            print(f"  Step {warmup_end:4d} (warmup end):   LR={format_lr(lr_values[warmup_end])}")
        if decay_start < len(lr_values):
            print(f"  Step {decay_start:4d} (decay start):  LR={format_lr(lr_values[decay_start])}")
        if final < len(lr_values):
            print(f"  Step {final:4d} (final):        LR={format_lr(lr_values[final])}")

        # Verify step counts match
        assert inner_sched.last_epoch == optimizer_step - 1, \
            f"FAIL: Inner scheduler last_epoch ({inner_sched.last_epoch}) != optimizer_steps - 1 ({optimizer_step - 1})"
        assert wrapper._step_count == optimizer_step, \
            f"FAIL: ZeroLRWarmupScheduler._step_count ({wrapper._step_count}) != optimizer_steps ({optimizer_step})"

        if zero_lr and len(lr_values) > num_warmup_steps:
            assert lr_values[0] == 0.0, f"FAIL: First warmup step should be 0"
            assert abs(lr_values[num_warmup_steps] - 1e-3) < 1e-8, f"FAIL: First post-warmup should be max_lr"

    print("\n  TEST 2 PASSED\n")


# ============ TEST 3 ============
def test_accelerated_wrapping_behavior():
    """Test 3: Full AcceleratedOptimizer + AcceleratedScheduler wrapping with gradient accumulation."""
    print("=" * 70)
    print("TEST 3: Full accelerate wrapping (AcceleratedOptimizer + AcceleratedScheduler)")
    print("=" * 70)

    num_training_steps = 100
    num_warmup_steps = 10
    num_decay_steps = 10
    ga_steps = 4  # batch_size=4, grad_accum is implicitly handled

    model, accel_opt, wrapper, accel_sched, inner_sched, meta = build_accelerated_scheduler_chain(
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        zero_lr_warmup=True,
        gradient_accumulation_steps=ga_steps,
    )
    print(f"  Config: {meta}")
    print(f"  Gradient accumulation steps: {ga_steps}")

    # Track actual calls to inner scheduler
    inner_step_calls = []
    original_inner_step = inner_sched.step
    def tracked_inner_step(epoch=None):
        inner_step_calls.append({
            "epoch": epoch,
            "last_epoch_before": inner_sched.last_epoch,
            "zero_step_count": wrapper._step_count,
        })
        result = original_inner_step(epoch)
        inner_step_calls[-1]["last_epoch_after"] = inner_sched.last_epoch
        return result
    inner_sched.step = tracked_inner_step

    # Track ZeroLRWarmupScheduler.step
    zero_step_calls = []
    original_zero_step = wrapper.step
    def tracked_zero_step(epoch=None):
        zero_step_calls.append({
            "epoch": epoch,
            "zero_step_count_before": wrapper._step_count,
        })
        result = original_zero_step(epoch)
        zero_step_calls[-1]["zero_step_count_after"] = wrapper._step_count
        return result
    wrapper.step = tracked_zero_step

    # Track AcceleratedScheduler.step
    accel_step_calls = []
    original_accel_step = accel_sched.step
    def tracked_accel_step(*args, **kwargs):
        accel_step_calls.append({
            "sync_gradients": accel_sched.gradient_state.sync_gradients,
            "adjust_scheduler": accel_sched.gradient_state.adjust_scheduler,
        })
        return original_accel_step(*args, **kwargs)
    accel_sched.step = tracked_accel_step

    # Simulate training loop
    lr_per_opt_step = []
    optimizer_step = 0

    for batch in range(ga_steps * num_training_steps):
        is_sync = ((batch + 1) % ga_steps == 0)
        accel_sched.gradient_state.sync_gradients = is_sync

        # In the real training loop, these are called on every backward pass
        # But AcceleratedOptimizer only actually steps when sync_gradients is True
        accel_opt.step()
        accel_sched.step()
        accel_opt.zero_grad(set_to_none=True)

        if is_sync:
            lr_per_opt_step.append(model.parameters().__next__().data.clone().item() if False else accel_opt.param_groups[0]["lr"])
            optimizer_step += 1

        if optimizer_step >= num_training_steps:
            break

    # Get LR values from the model's param_groups
    lr_values = []
    # Re-simulate to get clean LR values
    model2, accel_opt2, wrapper2, accel_sched2, inner_sched2, meta2 = build_accelerated_scheduler_chain(
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_decay_steps=num_decay_steps,
        zero_lr_warmup=True,
        gradient_accumulation_steps=ga_steps,
    )

    lr_values = []
    optimizer_step = 0
    for batch in range(ga_steps * num_training_steps):
        is_sync = ((batch + 1) % ga_steps == 0)
        accel_sched2.gradient_state.sync_gradients = is_sync
        accel_opt2.step()
        accel_sched2.step()
        accel_opt2.zero_grad(set_to_none=True)
        if is_sync:
            lr_values.append(accel_opt2.param_groups[0]["lr"])
            optimizer_step += 1
        if optimizer_step >= num_training_steps:
            break

    print(f"\n  Total backward passes: {batch + 1}")
    print(f"  Total optimizer steps: {optimizer_step}")
    print(f"  AcceleratedScheduler.step() calls: {len(accel_step_calls)}")
    print(f"    - sync steps: {sum(1 for c in accel_step_calls if c['sync_gradients'])}")
    print(f"    - non-sync steps: {sum(1 for c in accel_step_calls if not c['sync_gradients'])}")
    print(f"  ZeroLRWarmupScheduler.step() calls: {len(zero_step_calls)}")
    print(f"  Inner scheduler.step() calls: {len(inner_step_calls)}")

    print(f"\n  First 8 AcceleratedScheduler calls:")
    for i, call in enumerate(accel_step_calls[:8]):
        print(f"    Call {i}: sync={call['sync_gradients']}, adjust={call['adjust_scheduler']}")

    print(f"\n  ZeroLRWarmupScheduler calls (should only be on sync steps):")
    for i, call in enumerate(zero_step_calls[:8]):
        print(f"    Call {i}: count {call['zero_step_count_before']}->{call['zero_step_count_after']}")

    print(f"\n  Inner scheduler calls (should only be on sync steps):")
    for i, call in enumerate(inner_step_calls[:8]):
        print(f"    Call {i}: last_epoch {call['last_epoch_before']}->{call['last_epoch_after']}")

    print(f"\n  LR at key points:")
    if len(lr_values) > num_warmup_steps:
        print(f"    Step {num_warmup_steps} (warmup end):     LR={format_lr(lr_values[num_warmup_steps])}")
    decay_start = num_warmup_steps + meta["num_stable_steps"]
    if len(lr_values) > decay_start:
        print(f"    Step {decay_start} (decay start):    LR={format_lr(lr_values[decay_start])}")
    if len(lr_values) > 0:
        print(f"    Step {len(lr_values) - 1} (final):          LR={format_lr(lr_values[-1])}")

    # KEY VERIFICATIONS:
    sync_count = sum(1 for c in accel_step_calls if c["sync_gradients"])
    
    # Verify: inner scheduler only stepped on sync steps
    assert len(inner_step_calls) == sync_count, \
        f"FAIL: Inner scheduler stepped {len(inner_step_calls)} times but {sync_count} sync steps"
    
    # Verify: ZeroLRWarmupScheduler only stepped on sync steps
    assert len(zero_step_calls) == sync_count, \
        f"FAIL: ZeroLRWarmupScheduler stepped {len(zero_step_calls)} times but {sync_count} sync steps"
    
    # Verify: inner scheduler last_epoch matches optimizer steps
    expected_last_epoch = optimizer_step - 1
    assert inner_sched2.last_epoch == expected_last_epoch, \
        f"FAIL: Inner scheduler last_epoch ({inner_sched2.last_epoch}) != expected ({expected_last_epoch})"

    # Verify: ZeroLRWarmupScheduler._step_count matches optimizer steps
    assert wrapper2._step_count == optimizer_step, \
        f"FAIL: ZeroLRWarmupScheduler._step_count ({wrapper2._step_count}) != optimizer_steps ({optimizer_step})"

    # Verify: warmup LR is 0
    if len(lr_values) > 0:
        assert lr_values[0] == 0.0, f"FAIL: First step should be LR=0, got {lr_values[0]}"
    if len(lr_values) > num_warmup_steps:
        assert abs(lr_values[num_warmup_steps] - 1e-3) < 1e-8, \
            f"FAIL: Post-warmup should be max_lr, got {lr_values[num_warmup_steps]}"

    print(f"\n  *** VERIFIED: AcceleratedScheduler correctly gates inner step() behind sync_gradients ***")
    print(f"  *** VERIFIED: _step_count matches optimizer steps (no double-counting) ***")

    print("\n  TEST 3 PASSED\n")


# ============ TEST 4 ============
def test_epoch_based_max_steps():
    """Test epoch-based max_train_steps calculation with zero_lr_warmup."""
    print("=" * 70)
    print("TEST 4: Epoch-based max_train_steps with zero_lr_warmup")
    print("=" * 70)

    dataloader_len = 100
    epochs = 10
    warmup_ratio = 0.1
    decay_ratio = 0.1

    for num_processes in [1, 2]:
        ga_steps = 1

        # As computed in train_network.py line 1354
        max_train_steps = epochs * math.ceil(dataloader_len / num_processes / ga_steps)
        # As computed in get_scheduler_fix line 5697
        num_training_steps = max_train_steps * num_processes

        print(f"\n--- num_processes={num_processes} ---")
        print(f"  max_train_steps (from epochs): {max_train_steps}")
        print(f"  num_training_steps (in scheduler): {num_training_steps}")
        print(f"  Match? {max_train_steps == num_training_steps}")

        num_warmup_steps = int(warmup_ratio * num_training_steps)
        num_decay_steps = int(decay_ratio * num_training_steps)
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

        print(f"  num_warmup_steps: {num_warmup_steps}")
        print(f"  num_decay_steps: {num_decay_steps}")
        print(f"  num_stable_steps: {num_stable_steps}")

        # Build and run for the ACTUAL number of optimizer steps (max_train_steps per GPU)
        optimizer, model, wrapper, outer_sched, inner_sched, meta = build_simple_scheduler_chain(
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            num_decay_steps=num_decay_steps,
            zero_lr_warmup=True,
        )

        lr_values = []
        for step in range(max_train_steps):  # Only step max_train_steps times (per GPU)
            optimizer.step()
            outer_sched.step()
            optimizer.zero_grad(set_to_none=True)
            lr_values.append(optimizer.param_groups[0]["lr"])

        print(f"  Inner scheduler last_epoch: {inner_sched.last_epoch}")
        print(f"  Inner scheduler expects total: {num_training_steps}")

        # Determine which phase we end in
        if max_train_steps <= num_warmup_steps:
            phase = "WARMUP"
        elif max_train_steps <= num_warmup_steps + num_stable_steps:
            phase = "STABLE"
        elif max_train_steps <= num_training_steps:
            phase = "DECAY"
        else:
            phase = "BEYOND_SCHEDULE (min_lr)"
        print(f"  End phase: {phase}")

        print(f"  LR at key points:")
        print(f"    Step 0:              LR={format_lr(lr_values[0])}")
        if len(lr_values) > num_warmup_steps:
            print(f"    Step {num_warmup_steps} (warmup end):     LR={format_lr(lr_values[num_warmup_steps])}")
        decay_start = num_warmup_steps + num_stable_steps
        if len(lr_values) > decay_start:
            print(f"    Step {decay_start} (decay start):    LR={format_lr(lr_values[decay_start])}")
        print(f"    Step {len(lr_values) - 1} (final):          LR={format_lr(lr_values[-1])}")

        if num_processes > 1 and num_training_steps != max_train_steps:
            print(f"\n  *** NOTE: With num_processes={num_processes}, num_training_steps ({num_training_steps})")
            print(f"  *** differs from actual optimizer steps ({max_train_steps}).")
            print(f"  *** This means the scheduler will not complete its full schedule.")
            print(f"  *** The decay phase may start later than expected or never happen.")

    print("\n  TEST 4 COMPLETE\n")


if __name__ == "__main__":
    test_basic_zero_lr_warmup_wsd()
    test_with_gradient_accumulation_manual()
    test_accelerated_wrapping_behavior()
    test_epoch_based_max_steps()
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
