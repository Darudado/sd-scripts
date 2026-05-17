"""Unit tests for the RateTracker class (Welford-based step timing)."""
import time
import pytest
from library.train_util import RateTracker


class TestRateTrackerBasic:
    """Test basic RateTracker functionality."""

    def test_initial_state(self):
        r = RateTracker(skip_first=False)
        assert r.it_per_sec == 0.0
        assert r.step_time_std == 0.0
        assert r.step_count == 0
        assert r.display_rate == "0.0it/s"
        assert r.mean_step_time == 0.0

    def test_single_tick_no_rate(self):
        """First tick just starts timing, no rate yet."""
        r = RateTracker(skip_first=False)
        r.tick()
        assert r.step_count == 0
        assert r.it_per_sec == 0.0

    def test_two_ticks_produce_rate(self):
        r = RateTracker(skip_first=False)
        r.tick()
        time.sleep(0.1)
        r.tick()
        assert r.step_count == 1
        # 1 / ~0.1s ≈ 10 it/s
        assert 5.0 < r.it_per_sec < 20.0

    def test_display_rate_format_fast(self):
        """When rate >= 1.0, display as 'X.XXit/s'."""
        r = RateTracker(skip_first=False)
        r.tick()
        time.sleep(0.05)
        r.tick()
        rate_str = r.display_rate
        assert rate_str.endswith("it/s")
        assert "." in rate_str
        # Parse the numeric part to confirm it's valid
        parts = rate_str.split("it/s")
        assert len(parts) == 2
        assert float(parts[0]) > 0

    def test_display_rate_format_slow(self):
        """When rate < 1.0, display as 'X.XXs/it'."""
        r = RateTracker(skip_first=False)
        r.tick()
        time.sleep(1.5)
        r.tick()
        rate_str = r.display_rate
        assert "s/it" in rate_str, f"Expected s/it for slow rate, got: {rate_str}"

    def test_multiple_ticks_convergence(self):
        """After many ticks with consistent timing, it/s should converge."""
        r = RateTracker(skip_first=False)
        for _ in range(20):
            r.tick()
            time.sleep(0.05)
        assert r.step_count == 19  # 20 ticks = 19 intervals
        # After many ticks of 0.05s, should be close to 20 it/s
        assert 15.0 < r.it_per_sec < 25.0


class TestRateTrackerWelford:
    """Test that Welford statistics work correctly."""

    def test_std_zero_for_identical_timings(self):
        r = RateTracker(skip_first=False)
        # Mock identical timings by directly calling _add
        for _ in range(10):
            r._add(0.05)
        assert r.step_count == 10
        assert r.it_per_sec == pytest.approx(20.0, rel=0.01)
        assert r.step_time_std == pytest.approx(0.0, abs=0.001)

    def test_std_increases_with_variance(self):
        r = RateTracker(skip_first=False)
        for v in [0.04, 0.06, 0.04, 0.06, 0.04, 0.06]:
            r._add(v)
        # Mean = 0.05, std > 0
        assert r.step_time_std > 0.005

    def test_outlier_resistance(self):
        """A single slow step should not wildly distort the mean."""
        r = RateTracker(skip_first=False)
        for _ in range(10):
            r._add(0.05)
        # Add one very slow step
        r._add(1.0)
        # Mean should shift but not dramatically (0.05*10 + 1.0) / 11 ≈ 0.136
        # So it/s should be around 1/0.136 ≈ 7.3
        # std should be large
        assert r.step_time_std > 0.2
        assert 5.0 < r.it_per_sec < 20.0  # Still in a reasonable range


class TestRateTrackerIntegration:
    """Test realistic training-loop integration pattern."""

    def test_typical_usage_pattern(self):
        r = RateTracker(skip_first=False)
        # Simulate training loop
        rates = []
        for i in range(30):
            r.tick()
            if i == 0:
                continue  # First tick just sets baseline
            rate = r.it_per_sec
            rates.append(rate)
            time.sleep(0.02)
        # After warmup, rate should stabilize
        last_rates = rates[-10:]
        avg_last = sum(last_rates) / len(last_rates)
        # Should be around 50 it/s (1/0.02)
        assert 35.0 < avg_last < 65.0

    def test_step_count_matches_intervals(self):
        r = RateTracker(skip_first=False)
        for _ in range(50):
            r.tick()
        # 50 ticks = 49 intervals
        assert r.step_count == 49


class TestRateTrackerSkipFirst:
    """Test that skip_first=True discards the initial outlier interval."""

    def test_skip_first_default(self):
        """Default constructor should skip the first interval."""
        r = RateTracker()
        assert r.step_count == 0

    def test_two_ticks_no_rate_when_skipping(self):
        """With skip_first=True, 2 ticks should record 0 intervals."""
        r = RateTracker(skip_first=True)
        r.tick()
        time.sleep(0.1)
        r.tick()
        assert r.step_count == 0  # First interval discarded
        assert r.it_per_sec == 0.0

    def test_three_ticks_one_interval(self):
        """3 ticks with skip_first=True: interval 0 discarded, interval 1 kept."""
        r = RateTracker(skip_first=True)
        r.tick()
        time.sleep(0.1)
        r.tick()
        time.sleep(0.1)
        r.tick()
        assert r.step_count == 1
        assert 5.0 < r.it_per_sec < 20.0

    def test_skip_first_excludes_init_overhead(self):
        """Simulate training: first step takes 5s (torch.compile), rest 0.05s."""
        r = RateTracker(skip_first=True)
        # Step 0→1: includes torch.compile (outlier)
        r.tick()
        time.sleep(0.3)  # Simulated compile overhead
        r.tick()
        # Step 1→2 and onward: normal training speed
        for _ in range(10):
            time.sleep(0.05)
            r.tick()
        # After skipping the outlier, it/s should be near 20 (1/0.05)
        assert 15.0 < r.it_per_sec < 25.0, f"Expected ~20 it/s, got {r.it_per_sec:.1f}"

    def test_skip_first_disabled(self):
        """With skip_first=False, the first interval IS recorded."""
        r = RateTracker(skip_first=False)
        r.tick()
        time.sleep(0.3)  # Slow first step
        r.tick()
        assert r.step_count == 1
        # With the slow step included, it/s should be low
        assert r.it_per_sec < 10.0
