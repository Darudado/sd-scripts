"""Unit tests for library.image_utils.to_srgb."""

import io
import os
import sys
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image, ImageCms

# Allow running from repo root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.image_utils import to_srgb, _TRANSFORM_ATTEMPTS, _log_profile_failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid(mode: str, size=(8, 8), color=128) -> Image.Image:
    """Create a solid-colour image in *mode*."""
    return Image.new(mode, size, color)


def _make_srgb_profile_bytes() -> bytes:
    """Return raw ICC bytes for sRGB."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _make_adobe_rgb_profile_bytes() -> bytes:
    """Return raw ICC bytes for AdobeRGB (wide-gamut)."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToSrgb:
    """Test suite for the to_srgb colour-space conversion utility."""

    # -- basic mode conversion -----------------------------------------------

    def test_rgb_no_profile_returns_unchanged(self):
        """An RGB image with no ICC profile should be returned as-is (fast path)."""
        im = _make_solid("RGB", color=(200, 100, 50))
        result = to_srgb(im)
        assert result.mode == "RGB"
        # Fast path: same object returned
        assert result is im

    def test_greyscale_converted_to_rgb(self):
        """A greyscale (L) image should be returned as RGB."""
        im = _make_solid("L", color=100)
        result = to_srgb(im)
        assert result.mode == "RGB"
        assert result.size == im.size

    def test_rgba_converted_to_rgb(self):
        """An RGBA image should be returned as RGB (alpha dropped)."""
        im = _make_solid("RGBA", color=(100, 150, 200, 255))
        result = to_srgb(im)
        assert result.mode == "RGB"
        assert result.size == im.size

    def test_p_mode_converted_to_rgb(self):
        """A palette-mode image should be returned as RGB."""
        im = _make_solid("P", color=0)
        result = to_srgb(im)
        assert result.mode == "RGB"

    def test_cmyk_converted_to_rgb(self):
        """A CMYK image should be returned as RGB."""
        im = _make_solid("CMYK", color=(0, 0, 0, 0))
        result = to_srgb(im)
        assert result.mode == "RGB"
        assert result.size == im.size

    # -- ICC profile handling ------------------------------------------------

    def test_rgb_with_srgb_profile_unchanged(self):
        """An RGB image already in sRGB should pass through with no visible change."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()
        result = to_srgb(im)
        assert result.mode == "RGB"
        # Pixel values should be essentially the same (sRGB -> sRGB is identity).
        np.testing.assert_allclose(
            np.array(result), np.array(im), atol=1,
        )

    def test_result_has_srgb_icc_tag_after_conversion(self):
        """The returned image should carry an sRGB ICC profile tag after conversion."""
        im = _make_solid("RGB", color=(100, 100, 100))
        im.info["icc_profile"] = _make_srgb_profile_bytes()
        result = to_srgb(im)
        icc = result.info.get("icc_profile")
        assert icc is not None
        # Verify it parses back as sRGB.
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        assert "sRGB" in ImageCms.getProfileName(profile)

    def test_profile_to_profile_called_with_icc_bytes(self):
        """When an embedded ICC profile is present, profileToProfile is invoked."""
        im = _make_solid("RGB", color=(128, 128, 128))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        with patch("library.image_utils.ImageCms.profileToProfile", wraps=ImageCms.profileToProfile) as mock_ptp:
            result = to_srgb(im)

        mock_ptp.assert_called_once()
        # Verify the source profile was created from the embedded bytes.
        call_args = mock_ptp.call_args
        src_profile = call_args[0][1]
        assert isinstance(src_profile, ImageCms.ImageCmsProfile)

    def test_no_profile_uses_assume(self):
        """Without embedded ICC bytes, the assume parameter is used as source."""
        im = _make_solid("RGB", color=(200, 100, 50))

        with patch("library.image_utils.ImageCms.createProfile") as mock_create:
            # Make createProfile return the real sRGB profile so conversion works.
            mock_create.return_value = ImageCms.createProfile("sRGB")
            result = to_srgb(im, assume="sRGB")

        mock_create.assert_called_once_with("sRGB")

    # -- fallback behaviour -------------------------------------------------

    def test_fallback_on_corrupt_profile(self):
        """A corrupt ICC profile should fall back to plain .convert('RGB')."""
        im = _make_solid("L", color=128)
        im.info["icc_profile"] = b"corrupt-garbage-bytes"
        result = to_srgb(im)
        assert result.mode == "RGB"
        assert result.size == im.size

    def test_fallback_returns_correct_pixels(self):
        """The fallback path should still produce correct pixel values."""
        im = _make_solid("L", color=128)
        im.info["icc_profile"] = b"bad"
        result = to_srgb(im)
        expected = im.convert("RGB")
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    # -- size preservation ---------------------------------------------------

    def test_size_preserved(self):
        """Output dimensions must match input."""
        im = _make_solid("RGB", size=(37, 53))
        result = to_srgb(im)
        assert result.size == (37, 53)

    # -- pixel content sanity ------------------------------------------------

    def test_solid_image_remains_solid(self):
        """A solid-colour sRGB image should remain solid after conversion."""
        im = _make_solid("RGB", size=(16, 16), color=(128, 64, 32))
        result = to_srgb(im)
        arr = np.array(result)
        # Every pixel should be the same value.
        assert (arr == arr[0, 0]).all()

    # -- idempotency ---------------------------------------------------------

    def test_idempotent_for_rgb_no_profile(self):
        """Calling to_srgb twice on an RGB image with no profile returns the same object."""
        im = _make_solid("RGB", color=(50, 100, 150))
        r1 = to_srgb(im)
        r2 = to_srgb(r1)
        assert r1 is im
        assert r2 is im

    # -- graduated fallback --------------------------------------------------

    def test_graduated_fallback_retries_with_less_flags(self):
        """When the first attempt (BPC + perceptual) fails, the function should
        retry with less strict flag combinations before giving up."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        # Capture the real function reference before patching.
        _real_ptp = ImageCms.profileToProfile
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            flags = kwargs.get("flags", 0)
            # Fail only when BLACKPOINTCOMPENSATION is requested.
            if flags & ImageCms.Flags.BLACKPOINTCOMPENSATION:
                raise ValueError("cannot build transform")
            return _real_ptp(*args, **kwargs)

        with patch("library.image_utils.ImageCms.profileToProfile", side_effect=side_effect):
            result = to_srgb(im)

        assert result.mode == "RGB"
        # At least 2 calls: first with BPC (fails), second without BPC (succeeds).
        assert call_count[0] >= 2

    def test_graduated_fallback_tries_all_intent_combinations(self):
        """When all BPC variants fail, the function should also try different
        rendering intents before falling back."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        call_args_log = []

        def side_effect(*args, **kwargs):
            call_args_log.append(kwargs)
            raise ValueError("cannot build transform")

        with patch("library.image_utils.ImageCms.profileToProfile", side_effect=side_effect):
            result = to_srgb(im)

        # Should have tried all 4 combinations from _TRANSFORM_ATTEMPTS.
        assert len(call_args_log) == len(_TRANSFORM_ATTEMPTS)
        # Verify each attempt uses a different (flags, intent) pair.
        seen = {(a.get("flags", 0), a.get("renderingIntent", 0)) for a in call_args_log}
        assert len(seen) == len(_TRANSFORM_ATTEMPTS)

    def test_graduated_fallback_ultimate_fallback_to_convert(self):
        """When ALL ICC transform attempts fail, plain .convert('RGB') is used."""
        im = _make_solid("L", color=128)
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        with patch("library.image_utils.ImageCms.profileToProfile", side_effect=ValueError("cannot build transform")):
            result = to_srgb(im)

        assert result.mode == "RGB"
        expected = im.convert("RGB")
        np.testing.assert_array_equal(np.array(result), np.array(expected))

    def test_graduated_fallback_succeeds_on_first_attempt(self):
        """When the first attempt succeeds, no retries should be made."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        with patch("library.image_utils.ImageCms.profileToProfile", wraps=ImageCms.profileToProfile) as mock_ptp:
            result = to_srgb(im)

        mock_ptp.assert_called_once()
        assert result.mode == "RGB"

    def test_graduated_fallback_succeeds_on_last_attempt(self):
        """Even if only the last (flags, intent) combo works, conversion should succeed."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        # Capture the real function reference before patching.
        _real_ptp = ImageCms.profileToProfile
        last_flags, last_intent = _TRANSFORM_ATTEMPTS[-1]
        attempt_count = [0]

        def side_effect(*args, **kwargs):
            attempt_count[0] += 1
            flags = kwargs.get("flags", 0)
            intent = kwargs.get("renderingIntent", 0)
            # Only succeed on the very last combination.
            if flags == last_flags and intent == last_intent:
                return _real_ptp(*args, **kwargs)
            raise ValueError("cannot build transform")

        with patch("library.image_utils.ImageCms.profileToProfile", side_effect=side_effect):
            result = to_srgb(im)

        assert result.mode == "RGB"
        assert attempt_count[0] == len(_TRANSFORM_ATTEMPTS)

    def test_graduated_fallback_result_tagged_srgb(self):
        """Successful conversion on a retry should still tag the result with sRGB ICC."""
        im = _make_solid("RGB", color=(128, 64, 32))
        im.info["icc_profile"] = _make_srgb_profile_bytes()

        # Capture the real function reference before patching.
        _real_ptp = ImageCms.profileToProfile
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            flags = kwargs.get("flags", 0)
            if flags & ImageCms.Flags.BLACKPOINTCOMPENSATION:
                raise ValueError("cannot build transform")
            return _real_ptp(*args, **kwargs)

        with patch("library.image_utils.ImageCms.profileToProfile", side_effect=side_effect):
            result = to_srgb(im)

        icc = result.info.get("icc_profile")
        assert icc is not None
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        assert "sRGB" in ImageCms.getProfileName(profile)


class TestLogProfileFailure:
    """Test suite for the _log_profile_failure helper."""

    def test_logs_image_mode_and_size(self, caplog):
        """Log message should include the image mode and size."""
        im = _make_solid("CMYK", size=(64, 48), color=(0, 0, 0, 0))
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im)
        assert "mode=CMYK" in caplog.text
        assert "size=(64, 48)" in caplog.text

    def test_logs_format_and_filename(self, caplog):
        """Log message should include image format and filename when available."""
        im = _make_solid("RGB", color=(128, 128, 128))
        im.format = "JPEG"
        im.filename = "/tmp/test_image.jpg"
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im)
        assert "format=JPEG" in caplog.text
        assert "test_image.jpg" in caplog.text

    def test_logs_format_unknown_when_not_set(self, caplog):
        """Log message should show <unknown> when format/filename are not set."""
        im = _make_solid("RGB", color=(128, 128, 128))
        # _make_solid doesn't set format or filename
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im)
        assert "format=<unknown>" in caplog.text
        assert "filename=<unknown>" in caplog.text

    def test_logs_profile_details_when_available(self, caplog):
        """Log message should include profile name, class, colour-space, and byte size."""
        im = _make_solid("RGB", color=(128, 128, 128))
        src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im, src)
        assert "sRGB" in caplog.text
        assert "bytes=" in caplog.text
        assert "class=" in caplog.text
        assert "colour-space=" in caplog.text

    def test_logs_mismatch_reason_for_greyscale_with_rgb_profile(self, caplog):
        """Should detect and log the mode/profile mismatch for L mode + sRGB profile."""
        im = _make_solid("L", color=128)
        src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        # Attach ICC bytes so mismatch detection triggers
        im.info["icc_profile"] = _make_srgb_profile_bytes()
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im, src)
        assert "likely cause" in caplog.text
        assert "mode=L" in caplog.text
        assert "does not match" in caplog.text

    def test_handles_none_profile_gracefully(self, caplog):
        """Should not raise when src_profile is None."""
        im = _make_solid("L", color=128)
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im, None)
        assert "unavailable" in caplog.text

    def test_handles_broken_profile_gracefully(self, caplog):
        """Should not raise when src_profile methods throw."""
        im = _make_solid("RGB", color=(128, 128, 128))
        broken_profile = MagicMock()
        broken_profile.tobytes.side_effect = Exception("broken")
        # ImageCms functions may raise on a mock object — that's fine.
        with caplog.at_level("WARNING", logger="library.image_utils"):
            _log_profile_failure(im, broken_profile)
        assert "falling back" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
