"""Unit tests for the enhanced EMARecorder class."""
import pytest
from library.train_util import EMARecorder


class TestEMARecorderBasic:
    """Test basic EMA functionality with the new default smoothing=0.25."""

    def test_converges_to_constant(self):
        r = EMARecorder(smoothing=0.25)
        for _ in range(20):
            r.add(1.0)
        assert abs(r.average - 1.0) < 0.01

    def test_adapts_to_change(self):
        r = EMARecorder(smoothing=0.25)
        for _ in range(15):
            r.add(1.0)
        for _ in range(15):
            r.add(0.5)
        # After 15 steps of 0.5, should be well below 0.7
        assert r.average < 0.65, f"Expected <0.65, got {r.average:.4f}"

    def test_bias_corrected_startup(self):
        """Bias correction ensures early values aren't dragged toward zero."""
        r = EMARecorder(smoothing=0.25)
        r.add(2.0)
        r.add(2.0)
        r.add(2.0)
        # With bias correction, should be close to 2.0, not 0.5
        assert r.average > 1.5

    def test_legacy_kwargs_support(self):
        r = EMARecorder(smoothing=0.25)
        r.add(epoch=0, step=0, loss=2.0)
        r.add(epoch=0, step=1, loss=1.0)
        assert abs(r.moving_average - r.average) < 0.001

    def test_add_with_none_does_nothing(self):
        r = EMARecorder(smoothing=0.25)
        r.add(1.0)
        r.add()  # No value, no kwargs
        assert r.num_updates == 1


class TestEMARecorderOutlierClipping:
    """Test that outlier_sigma clips extreme values."""

    def test_outlier_clipped(self):
        r = EMARecorder(smoothing=0.25, outlier_sigma=3.0)
        for _ in range(20):
            r.add(1.0)
        r.add(100.0)  # Extreme outlier
        # With 20% minimum std fallback: min_std = 0.2 * 1.0 = 0.2
        # bounds = 1.0 ± 3 * 0.2 = [0.4, 1.6]
        # 100.0 clipped to 1.6, then EMA: 0.75*1.0 + 0.25*1.6 = 1.15
        assert r.average < 5.0, f"Outlier should be clipped, got {r.average:.4f}"

    def test_no_clipping_when_disabled(self):
        r = EMARecorder(smoothing=0.25, outlier_sigma=0)
        for _ in range(20):
            r.add(1.0)
        r.add(100.0)
        assert r.average > 10.0, f"Should not clip when disabled, got {r.average:.4f}"

    def test_moderate_noise_not_clipped(self):
        r = EMARecorder(smoothing=0.25, outlier_sigma=3.0)
        for _ in range(20):
            r.add(1.0)
        for v in [0.5, 1.5, 0.8, 1.2, 0.7, 1.3]:
            r.add(v)
        # After moderate noise, average should have adapted somewhat
        assert 0.7 < r.average < 1.3

    def test_std_property(self):
        r = EMARecorder(smoothing=0.25)
        assert r.std == 0.0
        r.add(1.0)
        r.add(2.0)
        r.add(3.0)
        # With values [1,2,3], std ~ 1.0
        assert r.std > 0.5


class TestEMARecorderEdgeCases:
    """Test edge cases."""

    def test_zero_updates(self):
        r = EMARecorder(smoothing=0.25)
        assert r.average == 0.0
        assert r.std == 0.0

    def test_single_update(self):
        r = EMARecorder(smoothing=0.25)
        r.add(5.0)
        assert r.average > 0

    def test_invalid_smoothing(self):
        with pytest.raises(ValueError):
            EMARecorder(smoothing=1.5)
        with pytest.raises(ValueError):
            EMARecorder(smoothing=-0.1)
