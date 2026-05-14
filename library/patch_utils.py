# Patch Training Utilities for Anima LoRA
#
# Provides on-the-fly patch extraction from unscaled source images,
# variance-based validation (rejects solid/flat patches), feathered
# alpha mask generation, and disk-based caching of patch latents.

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Patch size helpers
# ---------------------------------------------------------------------------

def get_random_patch_size(min_size: int, max_size: int, align: int = 16) -> int:
    """Pick a random square dimension between min_size and max_size, aligned to *align*.

    The Anima VAE has spatial downscale=8 and patch_size=2, so dimensions must
    be divisible by 16.  The returned value is guaranteed to satisfy that.
    """
    if min_size > max_size:
        min_size, max_size = max_size, min_size
    min_aligned = ((min_size + align - 1) // align) * align
    max_aligned = (max_size // align) * align
    if min_aligned > max_aligned:
        return min_aligned
    steps = (max_aligned - min_aligned) // align
    return min_aligned + random.randint(0, steps) * align


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_random_patch(image_path: str, size: int) -> Optional[Image.Image]:
    """Open *image_path* at original resolution and crop a random *size*x*size* square.

    Returns ``None`` when the source image is smaller than *size* in either
    dimension (the image cannot contain a patch of the requested size).
    """
    try:
        img = Image.open(image_path)
        img.load()
    except Exception as exc:
        logger.warning(f"[Patch] Failed to open {image_path}: {exc}")
        return None

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if w < size or h < size:
        return None

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


# ---------------------------------------------------------------------------
# Patch validation
# ---------------------------------------------------------------------------

def validate_patch(patch: Image.Image, variance_threshold: float = 50.0) -> Tuple[bool, float]:
    """Check whether a patch has enough texture to be useful for training.

    Computes the pixel-to-pixel variance of the grayscale patch.  Patches
    below *variance_threshold* are considered near-solid (sky, blank wall,
    solid background) and should be rejected.

    Returns ``(is_valid, variance_value)``.
    """
    gray = np.array(patch.convert("L"), dtype=np.float32)
    variance = float(np.var(gray))
    return (variance >= variance_threshold, variance)


# ---------------------------------------------------------------------------
# Feathered alpha mask
# ---------------------------------------------------------------------------

def create_feathered_alpha_mask(
    latent_h: int,
    latent_w: int,
    feather_px: int = 16,
    vae_scale: int = 8,
) -> torch.Tensor:
    """Create a feathered alpha mask **directly at latent resolution**.

    The mask has value 1.0 in the centre and ramps linearly to 0.0 at the
    borders.  *feather_px* is specified in **pixel** space and is converted to
    latent space internally.  A minimum of 1 latent pixel of feathering is
    always applied.

    Returns a ``(latent_h, latent_w)`` float32 tensor.
    """
    feather_latent = max(1, feather_px // vae_scale)

    mask = torch.ones(latent_h, latent_w, dtype=torch.float32)

    for i in range(feather_latent):
        weight = (i + 1) / (feather_latent + 1)
        # Top / bottom rows
        if i < latent_h:
            mask[i, :] = torch.minimum(mask[i, :], torch.tensor(weight))
        if latent_h - 1 - i >= 0:
            mask[latent_h - 1 - i, :] = torch.minimum(
                mask[latent_h - 1 - i, :], torch.tensor(weight)
            )
        # Left / right columns
        if i < latent_w:
            mask[:, i] = torch.minimum(mask[:, i], torch.tensor(weight))
        if latent_w - 1 - i >= 0:
            mask[:, latent_w - 1 - i] = torch.minimum(
                mask[:, latent_w - 1 - i], torch.tensor(weight)
            )

    return mask


# ---------------------------------------------------------------------------
# Single patch extraction with retries
# ---------------------------------------------------------------------------

def extract_valid_patch(
    image_paths: List[str],
    size: int,
    variance_threshold: float = 50.0,
    max_retries: int = 10,
) -> Optional[Tuple[Image.Image, float, str]]:
    """Try up to *max_retries* times to extract a patch passing validation.

    Each attempt picks a random image and crops a random region.
    Returns ``(patch, variance, source_path)`` on success, or ``None``.
    """
    for _ in range(max_retries):
        img_path = random.choice(image_paths)
        patch = extract_random_patch(img_path, size)
        if patch is None:
            continue  # image too small, try another
        is_valid, variance = validate_patch(patch, variance_threshold)
        if is_valid:
            return (patch, variance, img_path)
    return None


# ---------------------------------------------------------------------------
# Batch building (pixel-space, before VAE)
# ---------------------------------------------------------------------------

def build_patch_batch(
    image_paths: List[str],
    batch_size: int,
    patch_size: int,
    variance_threshold: float,
    max_retries: int,
    feather_px: int,
    vae_scale: int = 8,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
    """Build a batch of validated patches, **all the same size**.

    Returns
    -------
    images : Tensor, shape ``(B, 3, H, W)``, float32 in ``[-1, 1]``
        Ready for VAE encoding.
    masks : Tensor, shape ``(B, H/vae_scale, W/vae_scale)``, float32
        Feathered alpha masks at latent resolution.
    source_paths : list of str
        Which image each patch came from (for logging).
    ``None`` if we cannot fill the batch.
    """
    images = []
    source_paths = []

    for _ in range(batch_size):
        result = extract_valid_patch(image_paths, patch_size, variance_threshold, max_retries)
        if result is None:
            logger.warning(f"[Patch] Could not fill batch (got {len(images)}/{batch_size})")
            return None
        patch_pil, _var, src_path = result
        images.append(patch_pil)
        source_paths.append(os.path.basename(src_path))

    # PIL -> tensor [B, 3, H, W] in [-1, 1]
    tensors = []
    for img in images:
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        t = t * 2.0 - 1.0
        tensors.append(t)
    images_tensor = torch.stack(tensors)

    # Feathered mask at latent resolution
    latent_h = patch_size // vae_scale
    latent_w = patch_size // vae_scale
    single_mask = create_feathered_alpha_mask(latent_h, latent_w, feather_px, vae_scale)
    masks = single_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

    return images_tensor, masks, source_paths


# ---------------------------------------------------------------------------
# Disk-based patch caching
# ---------------------------------------------------------------------------

def _get_patch_dir(dataset_dir: str, size: int) -> str:
    """Return the on-disk path for cached patches of a given size."""
    return os.path.join(dataset_dir, "patches", f"{size}x{size}")


def get_existing_patches(
    dataset_dirs: List[str],
    min_size: int,
    max_size: int,
    align: int = 16,
) -> Dict[int, List[str]]:
    """Scan ``dataset_dirs/patches/`` and return a map of size -> list of .npz paths.

    Only sizes that are aligned and within [min_size, max_size] are included.
    """
    pools: Dict[int, List[str]] = {}
    min_aligned = ((min_size + align - 1) // align) * align
    max_aligned = (max_size // align) * align

    for ddir in dataset_dirs:
        patches_root = os.path.join(ddir, "patches")
        if not os.path.isdir(patches_root):
            continue
        for folder_name in os.listdir(patches_root):
            # Expect folder names like "256x256", "384x384"
            parts = folder_name.lower().split("x")
            if len(parts) != 2:
                continue
            try:
                size = int(parts[0])
            except ValueError:
                continue
            if size < min_aligned or size > max_aligned or size % align != 0:
                continue
            folder_path = os.path.join(patches_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith(".npz"):
                    pools.setdefault(size, []).append(os.path.join(folder_path, fname))

    return pools


def generate_and_cache_patches(
    image_paths: List[str],
    dataset_dirs: List[str],
    target_count: int,
    min_size: int,
    max_size: int,
    variance_threshold: float,
    max_retries: int,
    feather_px: int,
    vae,
    vae_dtype,
    accelerator,
    encode_fn,
    shift_scale_fn,
    vae_scale: int = 8,
    align: int = 16,
) -> Dict[int, List[str]]:
    """Generate patch images, VAE-encode them, and save to disk.

    Patches are saved as:
      - ``{dataset_dir}/patches/{size}x{size}/{basename}_p{idx}.png``  (visual)
      - ``{dataset_dir}/patches/{size}x{size}/{basename}_p{idx}.npz``  (latents)

    Parameters
    ----------
    encode_fn : callable
        ``trainer.encode_images_to_latents(args, vae, images)``
    shift_scale_fn : callable
        ``trainer.shift_scale_latents(args, latents)``
    target_count : int
        Total number of individual patches to generate.

    Returns
    -------
    pools : dict mapping ``size -> list of .npz paths`` (existing + newly generated)
    """
    # Use the first dataset dir for storage
    primary_dir = dataset_dirs[0]

    # Gather existing patches first
    pools = get_existing_patches(dataset_dirs, min_size, max_size, align)
    existing_count = sum(len(v) for v in pools.values())

    if existing_count >= target_count:
        logger.info(
            f"[Patch Cache] Found {existing_count} cached patches on disk "
            f"(need {target_count}). Reusing existing patches."
        )
        return pools

    needed = target_count - existing_count
    logger.info(
        f"[Patch Cache] Found {existing_count} cached, need {target_count}. "
        f"Generating {needed} more patches..."
    )

    # Track how many patches we've created per source image to avoid filename collisions
    patch_counters: Dict[str, int] = {}
    generated = 0
    failures = 0
    max_total_failures = needed * 5  # safety valve

    pbar = tqdm(total=needed, desc="Caching patches")
    while generated < needed and failures < max_total_failures:
        size = get_random_patch_size(min_size, max_size, align)
        result = extract_valid_patch(image_paths, size, variance_threshold, max_retries)

        if result is None:
            failures += 1
            continue

        patch_pil, variance, src_path = result
        src_basename = os.path.splitext(os.path.basename(src_path))[0]

        # Determine patch index for this source
        counter_key = f"{src_basename}_{size}"
        idx = patch_counters.get(counter_key, 0)
        patch_counters[counter_key] = idx + 1

        # Save PNG
        patch_dir = _get_patch_dir(primary_dir, size)
        os.makedirs(patch_dir, exist_ok=True)
        png_name = f"{src_basename}_p{idx}.png"
        png_path = os.path.join(patch_dir, png_name)
        patch_pil.save(png_path)

        # VAE encode
        arr = np.array(patch_pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        t = t * 2.0 - 1.0  # [-1, 1]
        with torch.no_grad():
            latents = encode_fn(vae, t.to(accelerator.device, dtype=vae_dtype))
            latents = shift_scale_fn(latents)

        # Save NPZ (latents on CPU)
        npz_name = f"{src_basename}_p{idx}.npz"
        npz_path = os.path.join(patch_dir, npz_name)
        np.savez(npz_path, latents=latents.cpu().float().numpy())

        pools.setdefault(size, []).append(npz_path)
        generated += 1
        pbar.update(1)

    pbar.close()

    if generated < needed:
        logger.warning(
            f"[Patch Cache] Only generated {generated}/{needed} patches "
            f"({failures} failures). Consider lowering --patch_variance_threshold."
        )
    else:
        logger.info(f"[Patch Cache] Successfully cached {generated} new patches.")

    return pools


# ---------------------------------------------------------------------------
# Debug: save patches for visual inspection
# ---------------------------------------------------------------------------

def save_debug_patches(
    image_paths: List[str],
    output_dir: str,
    count: int = 50,
    min_size: int = 256,
    max_size: int = 512,
    variance_threshold: float = 50.0,
    max_retries: int = 10,
):
    """Extract sample patches and save them as images for visual inspection.

    Saves both PASS and FAIL patches so the user can tune the variance
    threshold.  Writes a ``summary.txt`` alongside the images.
    """
    debug_dir = os.path.join(output_dir, "patch_debug")
    os.makedirs(debug_dir, exist_ok=True)

    stats = {"total": 0, "pass": 0, "fail": 0, "too_small": 0}
    summary_lines = []

    logger.info(f"[Patch Debug] Extracting {count} patches to {debug_dir}")
    logger.info(
        f"[Patch Debug] Size range: {min_size}-{max_size}, "
        f"variance threshold: {variance_threshold}"
    )

    for i in tqdm(range(count), desc="Extracting debug patches"):
        img_path = random.choice(image_paths)
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        size = get_random_patch_size(min_size, max_size)

        patch = extract_random_patch(img_path, size)
        if patch is None:
            stats["too_small"] += 1
            summary_lines.append(f"{i:04d} | {img_basename} | {size}x{size} | TOO_SMALL")
            continue

        stats["total"] += 1
        is_valid, variance = validate_patch(patch, variance_threshold)

        status = "PASS" if is_valid else "FAIL"
        if is_valid:
            stats["pass"] += 1
        else:
            stats["fail"] += 1

        filename = f"{i:04d}_{img_basename}_{size}x{size}_var{variance:.1f}_{status}.png"
        patch.save(os.path.join(debug_dir, filename))
        summary_lines.append(
            f"{i:04d} | {img_basename} | {size}x{size} | var={variance:.1f} | {status}"
        )

    # Write summary
    summary_path = os.path.join(debug_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Patch Debug Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Variance threshold: {variance_threshold}\n")
        f.write(f"Size range: {min_size}-{max_size}\n")
        f.write(f"Total extracted: {stats['total']}\n")
        f.write(f"  PASS: {stats['pass']}\n")
        f.write(f"  FAIL: {stats['fail']}\n")
        f.write(f"  TOO_SMALL: {stats['too_small']}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    logger.info(
        f"[Patch Debug] Done: {stats['pass']} PASS, {stats['fail']} FAIL, "
        f"{stats['too_small']} too small. See {summary_path}"
    )


# ---------------------------------------------------------------------------
# Collect image paths from dataset TOML config (for debug mode)
# ---------------------------------------------------------------------------

def collect_image_paths_from_toml(config_path: str) -> List[str]:
    """Read image directory paths from a dataset TOML config and glob images.

    Used by ``--patch_debug`` mode which runs before the full dataset is
    constructed.
    """
    try:
        import toml
    except ImportError:
        try:
            import tomllib as toml
        except ImportError:
            import tomli as toml

    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.loads(f.read()) if hasattr(toml, "loads") else toml.load(f)

    image_paths = []
    for dataset in config.get("datasets", []):
        for subset in dataset.get("subsets", []):
            image_dir = subset.get("image_dir", None)
            if image_dir and os.path.isdir(image_dir):
                for root, _dirs, files in os.walk(image_dir):
                    for fname in files:
                        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                            image_paths.append(os.path.join(root, fname))

    return image_paths
