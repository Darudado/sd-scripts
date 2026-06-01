"""Tests for Adaptive Non-uniform Timestep Sampling (arXiv:2411.09998).

Verifies:
1. TimestepSamplerNetwork forward pass, output shapes, and positivity constraints
2. F-statistic feature selection correctness
3. AdaptiveTimestepManager initialization, timestep sampling, and state dict round-trip
4. Per-timestep loss computation and delta approximation
5. Policy gradient sampler update
"""

import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from collections import deque

from library.adaptive_timestep_sampler import (
    TimestepSamplerNetwork,
    select_timesteps_f_statistic,
    AdaptiveTimestepManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def noise_scheduler(device):
    """Create a mock DDPMScheduler-like object with alphas_cumprod."""
    scheduler = MagicMock()
    T = 1000
    # Create a realistic cosine-like schedule
    t = torch.linspace(0, 1, T, device=device)
    alphas_cumprod = torch.cos(t * math.pi / 2) ** 2
    scheduler.alphas_cumprod = alphas_cumprod
    scheduler.config = MagicMock()
    scheduler.config.num_train_timesteps = T
    return scheduler


@pytest.fixture
def sampler_network(device):
    """Create a small TimestepSamplerNetwork for testing."""
    return TimestepSamplerNetwork(
        in_channels=4,
        hidden_channels=32,
        hidden_depth=2,
    ).to(device)


@pytest.fixture
def adaptive_manager(sampler_network, noise_scheduler, device):
    """Create an AdaptiveTimestepManager for testing."""
    return AdaptiveTimestepManager(
        sampler_network=sampler_network,
        noise_scheduler=noise_scheduler,
        device=device,
        dtype=torch.float32,
        learning_rate=1e-2,
        entropy_coeff=1e-2,
        update_freq=5,  # Small for testing
        queue_size=10,
        num_selected=3,
    )


# ---------------------------------------------------------------------------
# Tests: TimestepSamplerNetwork
# ---------------------------------------------------------------------------

class TestTimestepSamplerNetwork:

    def test_forward_output_shapes(self, sampler_network, device):
        """Output (a, b) should each have shape (batch_size,)."""
        sampler_network.eval()
        x = torch.randn(4, 4, 8, 8, device=device)
        a, b = sampler_network(x)
        assert a.shape == (4,), f"Expected shape (4,), got {a.shape}"
        assert b.shape == (4,), f"Expected shape (4,), got {b.shape}"

    def test_outputs_positive(self, sampler_network, device):
        """a and b must always be positive (Beta distribution requirement)."""
        sampler_network.eval()
        # Test with various inputs including zeros and large values
        for _ in range(10):
            x = torch.randn(8, 4, 16, 16, device=device) * 10
            a, b = sampler_network(x)
            assert (a > 0).all(), f"Found non-positive a values: {a}"
            assert (b > 0).all(), f"Found non-positive b values: {b}"

    def test_different_spatial_sizes(self, sampler_network, device):
        """Network should handle different spatial dimensions via AdaptiveAvgPool."""
        sampler_network.eval()
        for h, w in [(4, 4), (8, 8), (16, 32), (64, 64)]:
            x = torch.randn(2, 4, h, w, device=device)
            a, b = sampler_network(x)
            assert a.shape == (2,)
            assert b.shape == (2,)

    def test_initial_bias_near_uniform(self, sampler_network, device):
        """Initial parameters should produce a roughly symmetric Beta distribution."""
        sampler_network.eval()
        x = torch.zeros(1, 4, 8, 8, device=device)
        a, b = sampler_network(x)
        # With zero input and zero weights, softplus(bias) should give symmetric a, b
        ratio = a / b
        assert 0.5 < ratio.item() < 2.0, f"Initial a/b ratio {ratio.item()} too far from 1.0"

    def test_gradient_flow(self, sampler_network, device):
        """Gradients should flow through the network."""
        sampler_network.train()
        x = torch.randn(2, 4, 8, 8, device=device, requires_grad=True)
        a, b = sampler_network(x)
        loss = a.sum() + b.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed through the network"
        for param in sampler_network.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter has no gradient"


# ---------------------------------------------------------------------------
# Tests: Feature Selection (F-statistic)
# ---------------------------------------------------------------------------

class TestFeatureSelection:

    def test_selects_correct_timesteps(self, device):
        """When one timestep has a strong linear relationship with the target,
        it should be selected."""
        Q, T = 20, 100
        queue = torch.randn(Q, T, device=device) * 0.1
        # Make timestep 42 highly correlated with the row mean
        target = queue.mean(dim=1)  # (Q,)
        queue[:, 42] = target * 5.0 + torch.randn(Q, device=device) * 0.01
        # Make timestep 77 also correlated but weaker
        queue[:, 77] = target * 2.0 + torch.randn(Q, device=device) * 0.01

        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert 42 in selected.tolist(), f"Timestep 42 should be selected, got {selected}"
        assert 77 in selected.tolist(), f"Timestep 77 should be selected, got {selected}"

    def test_returns_correct_count(self, device):
        """Should return exactly num_selected indices."""
        queue = torch.randn(10, 50, device=device)
        for n in [1, 3, 5, 10]:
            selected = select_timesteps_f_statistic(queue, num_selected=n)
            assert selected.shape == (n,), f"Expected {n} indices, got {selected.shape}"

    def test_single_queue_entry(self, device):
        """With only one queue entry, should fall back to highest absolute delta."""
        queue = torch.randn(1, 100, device=device)
        queue[0, 10] = 100.0  # Largest absolute value
        selected = select_timesteps_f_statistic(queue, num_selected=1)
        assert selected[0].item() == 10

    def test_all_zeros(self, device):
        """All-zero queue should not crash."""
        queue = torch.zeros(10, 50, device=device)
        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert selected.shape == (3,)

    def test_uniform_queue(self, device):
        """Uniform queue (no variance) should not crash."""
        queue = torch.ones(10, 50, device=device) * 0.5
        selected = select_timesteps_f_statistic(queue, num_selected=3)
        assert selected.shape == (3,)


# ---------------------------------------------------------------------------
# Tests: AdaptiveTimestepManager
# ---------------------------------------------------------------------------

class TestAdaptiveTimestepManager:

    def test_initialization(self, adaptive_manager):
        """Manager should initialize with correct hyperparameters."""
        assert adaptive_manager.f_s == 5
        assert adaptive_manager.queue_size == 10
        assert adaptive_manager.num_selected == 3
        assert adaptive_manager.num_train_timesteps == 1000
        assert len(adaptive_manager.queue) == 0

    def test_should_update(self, adaptive_manager):
        """should_update should return True at multiples of f_s, False otherwise."""
        assert adaptive_manager.should_update(0) == False  # Step 0 is never updated
        assert adaptive_manager.should_update(1) == False
        assert adaptive_manager.should_update(4) == False
        assert adaptive_manager.should_update(5) == True
        assert adaptive_manager.should_update(6) == False
        assert adaptive_manager.should_update(10) == True
        assert adaptive_manager.should_update(15) == True

    def test_sample_timesteps_shape(self, adaptive_manager, device):
        """sample_timesteps should return integer timesteps in [0, T)."""
        x = torch.randn(4, 4, 8, 8, device=device)
        timesteps = adaptive_manager.sample_timesteps(x, num_timesteps=1000)
        assert timesteps.shape == (4,), f"Expected shape (4,), got {timesteps.shape}"
        assert timesteps.dtype == torch.int64 or timesteps.dtype == torch.long
        assert (timesteps >= 0).all() and (timesteps < 1000).all()

    def test_sample_timesteps_varies(self, adaptive_manager, device):
        """Sampled timesteps should vary across different inputs (not all identical)."""
        x = torch.randn(16, 4, 8, 8, device=device)
        timesteps = adaptive_manager.sample_timesteps(x, num_timesteps=1000)
        # With 16 samples, we expect at least some variation
        assert timesteps.unique().numel() > 1, "All timesteps are identical"

    def test_state_dict_round_trip(self, adaptive_manager, device):
        """Saving and loading state dict should restore the manager's state."""
        # Add some data to the queue
        adaptive_manager.queue.append(torch.randn(1000, device=device))
        adaptive_manager.queue.append(torch.randn(1000, device=device))

        state = adaptive_manager.state_dict()
        assert "sampler_network" in state
        assert "optimizer" in state
        assert "queue" in state
        assert len(state["queue"]) == 2

        # Create a new manager and load state
        new_manager = AdaptiveTimestepManager(
            sampler_network=TimestepSamplerNetwork(in_channels=4, hidden_channels=32, hidden_depth=2).to(device),
            noise_scheduler=adaptive_manager.noise_scheduler,
            device=device,
            queue_size=10,
        )
        new_manager.load_state_dict(state)
        assert len(new_manager.queue) == 2

    def test_compute_per_timestep_losses_shape(self, adaptive_manager, device):
        """compute_per_timestep_losses should return shape (T,) with non-negative values."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Simple mock model that returns random predictions
        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        losses = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=200
        )
        assert losses.shape == (T,), f"Expected shape ({T},), got {losses.shape}"
        assert (losses >= 0).all(), "Losses should be non-negative (MSE)"

    def test_compute_delta_approximation(self, adaptive_manager, device):
        """Delta approximation should return a scalar and selected indices."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        # Create mock losses_before (random)
        losses_before = torch.rand(T, device=device)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )
        assert delta_approx.ndim == 0 or delta_approx.shape == (), "delta_approx should be scalar"
        assert selected.shape == (adaptive_manager.num_selected,)
        # Queue should now have one entry
        assert len(adaptive_manager.queue) == 1

    def test_update_sampler(self, adaptive_manager, device):
        """update_sampler should not crash and should update network parameters."""
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        delta_k_t = torch.tensor(0.5, device=device)

        # Store initial parameters
        initial_params = {
            name: param.clone() for name, param in adaptive_manager.sampler_network.named_parameters()
        }

        adaptive_manager.update_sampler(delta_k_t, x_0)

        # At least one parameter should have changed
        changed = False
        for name, param in adaptive_manager.sampler_network.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-7):
                changed = True
                break
        assert changed, "No parameters changed after update_sampler"


    def test_cached_t_continuous_used_in_update(self, adaptive_manager, device):
        """update_sampler should use the cached sampled t, not a new one."""
        x_0 = torch.randn(2, 4, 8, 8, device=device)
        # Sample timesteps to populate the cache
        timesteps = adaptive_manager.sample_timesteps(x_0, num_timesteps=1000)
        # Verify _cached_t_continuous is set
        assert adaptive_manager._cached_t_continuous is not None
        assert adaptive_manager._cached_t_continuous.shape == (2,)
        # The cached t should be in (0, 1) range
        cached = adaptive_manager._cached_t_continuous
        assert (cached > 0).all() and (cached < 1).all()

        # Now update - the gradient should flow through the cached t
        delta_k_t = torch.tensor(0.5, device=device)
        initial_params = {
            name: param.clone() for name, param in adaptive_manager.sampler_network.named_parameters()
        }
        adaptive_manager.update_sampler(delta_k_t, x_0)
        # Parameters should have changed
        changed = any(
            not torch.allclose(param, initial_params[name], atol=1e-7)
            for name, param in adaptive_manager.sampler_network.named_parameters()
        )
        assert changed

    def test_f_statistic_matches_paper_formula(self, device):
        """F-statistic should match the standard formula."""
        # Create a queue where timestep 0 is perfectly correlated with target
        Q, T = 50, 20
        target_signal = torch.randn(Q, device=device)
        queue = torch.randn(Q, T, device=device) * 0.01
        queue[:, 0] = target_signal * 2.0
        queue[:, 0] += torch.randn(Q, device=device) * 0.001

        selected = select_timesteps_f_statistic(queue, num_selected=1)
        # Timestep 0 should have the highest F-statistic
        assert selected[0].item() == 0, f"Expected timestep 0, got {selected[0].item()}"



    def test_cache_batch_losses_at_S_noop_without_selection(self, adaptive_manager, device):
        """cache_batch_losses_at_S should be a no-op when no |S| selection exists yet."""
        assert adaptive_manager._current_selected_indices is None
        x = torch.randn(4, 4, 8, 8, device=device)
        noise = torch.randn_like(x)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Should be a no-op since no selection exists
        adaptive_manager.cache_batch_losses_at_S(x, noise, mock_model_fn, torch.float32)
        assert adaptive_manager._prev_batch_losses_at_S is None

    def test_cache_batch_losses_at_S_after_selection(self, adaptive_manager, device):
        """After a delta approximation, the selection should be cached and
        cache_batch_losses_at_S should store the full batch losses at those timesteps."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # First call: run a full delta approximation to establish a selection
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        # After this call, _current_selected_indices should be set
        assert adaptive_manager._current_selected_indices is not None
        assert adaptive_manager._current_selected_indices.shape == (adaptive_manager.num_selected,)

        # Now cache batch losses at the current |S|
        adaptive_manager.cache_batch_losses_at_S(
            full_batch, full_noise, mock_model_fn, torch.float32
        )
        # The cached losses should have shape (B, |S|) but mean gives scalar
        assert adaptive_manager._prev_batch_losses_at_S is not None
        assert adaptive_manager._prev_batch_losses_at_S.ndim == 0  # scalar (mean)

    def test_full_batch_delta_approximation(self, adaptive_manager, device):
        """When a previous |S| selection exists, compute_delta_approximation should
        use the full batch at those timesteps to compute the delta."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # ---- First call: establish a selection ----
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_first, selected_first = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        # The first call uses single x_0 (no previous selection)
        assert adaptive_manager._current_selected_indices is not None
        first_selection = adaptive_manager._current_selected_indices.clone()

        # ---- Cache full batch losses at the current |S| ----
        adaptive_manager.cache_batch_losses_at_S(
            full_batch, full_noise, mock_model_fn, torch.float32
        )
        assert adaptive_manager._prev_batch_losses_at_S is not None

        # ---- Second call: should use full batch at the previous |S| ----
        losses_before_2 = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_second, selected_second = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before_2,
            full_batch_latents=full_batch, full_batch_noise=full_noise,
        )
        # The selected indices should equal the first selection (the one used for the full batch)
        assert torch.equal(selected_second.cpu(), first_selection),             "Second call should return the previous |S| selection (the one used for full batch)"
        # The delta_approx should be a scalar
        assert delta_approx_second.ndim == 0 or delta_approx_second.shape == ()

    def test_full_batch_falls_back_to_single_when_no_cache(self, adaptive_manager, device):
        """If cache_batch_losses_at_S was not called, compute_delta_approximation
        should fall back to the single x_0 path even if full batch is provided."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)
        full_batch = torch.randn(4, 4, 8, 8, device=device)
        full_noise = torch.randn_like(full_batch)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # First call: establish a selection (but DON'T cache full batch losses)
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_first, _ = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before,
        )
        first_selection = adaptive_manager._current_selected_indices.clone()
        # No cache_batch_losses_at_S call! So _prev_batch_losses_at_S is None

        # Second call with full batch provided but no cached losses
        losses_before_2 = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )
        delta_approx_second, selected_second = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before_2,
            full_batch_latents=full_batch, full_batch_noise=full_noise,
        )
        # Should fall back to single x_0 path -> selected indices = new selection (not first)
        # The selected indices are the new ones from this call
        assert selected_second.shape == (adaptive_manager.num_selected,)

    def test_full_algorithm_2_cycle(self, adaptive_manager, device):
        """Full cycle: sample timesteps, compute losses before/after, update."""
        T = adaptive_manager.num_train_timesteps
        x_0 = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn_like(x_0)

        def mock_model_fn(noisy_latents, timesteps, weight_dtype):
            return torch.randn_like(noisy_latents)

        # Step 1: Compute losses before (simulating theta_k)
        losses_before = adaptive_manager.compute_per_timestep_losses(
            x_0, noise, mock_model_fn, torch.float32, chunk_size=500
        )

        # Step 2: Compute delta approximation (simulating theta_{k+1} after optimizer step)
        delta_approx, selected = adaptive_manager.compute_delta_approximation(
            mock_model_fn, x_0, noise, torch.float32, losses_before
        )

        # Step 3: Update sampler
        adaptive_manager.update_sampler(delta_approx, x_0)

        # Verify queue was populated
        assert len(adaptive_manager.queue) >= 1
