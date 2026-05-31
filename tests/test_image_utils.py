"""Unit tests for library.image_utils.to_srgb."""

import io
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageCms

# Allow running from repo root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.image_utils import to_srgb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid(mode: str, size=(8, 8), color=128) -> Image.Image:
    """Create a solid-colour image in *mode*."""
    return Image.new(mode, size, color)


def _make_srgb_profile_bytes() -> bytes:
    """Return raw ICC bytes for sRGB."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
