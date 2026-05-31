"""Image loading and color-space utilities for sd_scripts.

Provides :func:`to_srgb`, an ICC-profile-aware replacement for bare
``Image.convert("RGB")`` calls.  When an image carries an embedded ICC
profile the conversion honours it; otherwise a sensible default
(``"sRGB"``) is assumed.
"""

import io
import logging
from typing import Optional

from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

# Cache the sRGB profile at module level so it is created exactly once.
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_CMS = ImageCms.ImageCmsProfile(_SRGB_PROFILE)


def to_srgb(im: Image.Image, assume: str = "sRGB") -> Image.Image:
    """Convert *im* to sRGB, respecting any embedded ICC profile.

    Parameters
    ----------
    im : PIL.Image.Image
        Source image in any mode (RGB, RGBA, CMYK, L, …).
    assume : str
        Colour-space name to use when the image carries **no** embedded
        ICC profile.  Must be a string accepted by
        ``ImageCms.createProfile`` (e.g. ``"sRGB"``, ``"AdobeRGB"``).
        Default is ``"sRGB"`` — meaning no conversion is needed for
        typical web images.

    Returns
    -------
    PIL.Image.Image
        An ``"RGB"`` image in the sRGB colour space.

    Notes
    -----
    * CMYK images are converted to RGB through the ICC pipeline when a
      CMYK profile is present (or the *assume* fallback).  PIL's
      ``profileToProfile`` handles the CMYK → RGB channel reduction
      natively when ``outputMode="RGB"``.
    * On any failure a plain ``im.convert("RGB")`` is returned so
      callers never need to handle exceptions.
    """
    # Fast path: already RGB and sRGB — nothing to do.
    if im.mode == "RGB" and not im.info.get("icc_profile"):
        return im

    try:
        # --- determine source profile ---
        icc_bytes = im.info.get("icc_profile")
        if icc_bytes:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        else:
            src = ImageCms.createProfile(assume)

        # --- convert via the ICC pipeline ---
        im = ImageCms.profileToProfile(
            im,
            src,
            _SRGB_PROFILE,
            outputMode="RGB",
            renderingIntent=0,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )

        # Tag the result so downstream consumers know the colour space.
        im.info["icc_profile"] = _SRGB_CMS.tobytes()
        return im

    except Exception:
        logger.warning(
            "ICC-aware colour conversion failed; falling back to plain .convert('RGB')",
            exc_info=True,
        )
        return im.convert("RGB")
