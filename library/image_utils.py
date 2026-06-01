"""Image loading and color-space utilities for sd_scripts.

Provides :func:`to_srgb`, an ICC-profile-aware replacement for bare
``Image.convert("RGB")`` calls.  When an image carries an embedded ICC
profile the conversion honours it; otherwise a sensible default
(``"sRGB"``) is assumed.
"""

import io
import logging

from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

# Cache the sRGB profile at module level so it is created exactly once.
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_CMS = ImageCms.ImageCmsProfile(_SRGB_PROFILE)

# Progressively less strict (flags, rendering-intent) pairs.  Some ICC
# profiles (V4, wide-gamut, CMYK, etc.) reject BLACKPOINTCOMPENSATION
# or only define certain rendering intents, so we try several combos
# before giving up.
_TRANSFORM_ATTEMPTS = [
    (ImageCms.Flags.BLACKPOINTCOMPENSATION, ImageCms.Intent.PERCEPTUAL),
    (0, ImageCms.Intent.PERCEPTUAL),
    (ImageCms.Flags.BLACKPOINTCOMPENSATION, ImageCms.Intent.RELATIVE_COLORIMETRIC),
    (0, ImageCms.Intent.RELATIVE_COLORIMETRIC),
]


def _log_profile_failure(im: Image.Image, src_profile=None) -> None:
    """Log diagnostic information about a failed ICC conversion.

    Extracts as much detail as possible from both the PIL image object
    (format, filename, mode, size) and the ICC source profile (name,
    class, colour-space, byte-size) so that the root cause — typically a
    mode/profile mismatch such as a greyscale image carrying an sRGB
    profile — can be identified without needing to reproduce the exact
    image.
    """
    # --- image metadata ---
    img_format = getattr(im, "format", None) or "<unknown>"
    img_filename = getattr(im, "filename", None) or "<unknown>"

    # --- ICC profile details (each call independent so one failure
    #     doesn't suppress the others) ---
    profile_name = profile_class = color_space = "<unavailable>"
    icc_size = 0
    if src_profile is not None:
        try:
            profile_name = ImageCms.getProfileName(src_profile).strip()
        except Exception:
            pass
        try:
            profile_class = ImageCms.getProfileClass(src_profile).strip()
        except Exception:
            pass
        try:
            color_space = ImageCms.getColorSpace(src_profile).strip()
        except Exception:
            pass
        try:
            icc_size = len(src_profile.tobytes())
        except Exception:
            pass

    # --- detect the likely cause ---
    reason = ""
    icc_bytes = im.info.get("icc_profile")
    if icc_bytes:
        icc_size = icc_size or len(icc_bytes)

        # Fallback: if the ImageCms API didn't yield a colour-space,
        # read it directly from the ICC header (bytes 16-19 are the
        # colour-space signature in the ICC spec).
        if color_space == "<unavailable>" and len(icc_bytes) >= 20:
            try:
                color_space = icc_bytes[16:20].decode("ascii", errors="replace")
            except Exception:
                pass

        # Check for a common mismatch: e.g. greyscale/LA image carrying
        # an RGB profile (or vice-versa).  littlecms cannot build a
        # transform when the image mode and the ICC colour-space are
        # fundamentally incompatible.
        _CS_TO_MODES = {
            "RGB ": ("RGB", "RGBA"),
            "GRAY": ("L", "LA"),
            "CMYK": ("CMYK",),
        }
        expected_modes = _CS_TO_MODES.get(color_space, ())
        if color_space != "<unavailable>" and im.mode not in expected_modes:
            reason = (
                f" — likely cause: image mode={im.mode} does not match "
                f"ICC profile colour-space={color_space!r}; "
                f"littlecms cannot build a transform across these types"
            )

    logger.warning(
        "ICC-aware colour conversion failed for image mode=%s, size=%s, "
        "format=%s, filename=%s; "
        "ICC profile: name=%r, class=%r, colour-space=%r, bytes=%d%s — "
        "falling back to plain .convert('RGB')",
        im.mode,
        im.size,
        img_format,
        img_filename,
        profile_name,
        profile_class,
        color_space,
        icc_size,
        reason,
    )


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
    * The function tries several (flags, rendering-intent) combinations
      before falling back, so that images carrying unusual ICC profiles
      (V4, wide-gamut, etc.) still get correct colour conversion when
      possible.
    * On any failure a plain ``im.convert("RGB")`` is returned so
      callers never need to handle exceptions.
    """
    # Fast path: already RGB and sRGB — nothing to do.
    if im.mode == "RGB" and not im.info.get("icc_profile"):
        return im

    src = None  # ensure defined for the except clause
    try:
        # --- determine source profile ---
        icc_bytes = im.info.get("icc_profile")
        if icc_bytes:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        else:
            src = ImageCms.createProfile(assume)

        # --- convert via the ICC pipeline ---
        for flags, intent in _TRANSFORM_ATTEMPTS:
            try:
                result = ImageCms.profileToProfile(
                    im,
                    src,
                    _SRGB_PROFILE,
                    outputMode="RGB",
                    renderingIntent=intent,
                    flags=flags,
                )
                # Tag the result so downstream consumers know the colour space.
                result.info["icc_profile"] = _SRGB_CMS.tobytes()
                return result
            except Exception:
                continue

        # All ICC transform attempts failed — fall back.
        _log_profile_failure(im, src)
        return im.convert("RGB")

    except Exception:
        _log_profile_failure(im, src)
        return im.convert("RGB")
