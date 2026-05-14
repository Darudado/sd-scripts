========================================================
             ANIMA LORA PATCH TRAINING
========================================================

Patch-based training teaches the model fine textures and details by training
on random square crops extracted directly from the unscaled source images. 
These patches are pre-cached, VAE-encoded, and masked with a feathered border 
before training begins, resulting in zero performance degradation during the run.

--- ARGUMENTS ---

--enable_patch_training
    Enables patch-based training interleaved with normal training steps.

--patch_ratio (default: 0.3)
    Fraction of total training steps to replace with patch batches.
    For example, 0.3 means roughly every 3rd step will be a patch step.

--patch_min_size (default: 256)
    Minimum allowed crop size in pixels. Must be divisible by 16.

--patch_max_size (default: 512)
    Maximum allowed crop size in pixels. Must be divisible by 16.

--patch_min_timestep (default: 0)
    Minimum timestep for patch training steps. 

--patch_max_timestep (default: 300)
    Maximum timestep for patch training steps. Kept low (e.g. 300) to ensure 
    the model learns fine details/textures rather than global structures.

--patch_variance_threshold (default: 50.0)
    Minimum grayscale pixel variance required to accept a cropped patch. 
    Prevents the model from training on completely blank/flat patches (e.g. solid sky).

--patch_feather_px (default: 16)
    Thickness of the soft border around patches (in pixels). Soft borders 
    prevent the model from learning harsh artificial crop edges.

--patch_caption_trigger 
    (REQUIRED) A caption or trigger word applied to all patches 
    (e.g., "sks style").

--patch_max_retries (default: 10)
    Number of times the script will try cropping a valid patch before giving up.

--patch_regenerate
    Flag to force the clearing and regeneration of existing cached patch files.

--patch_debug
    Flag that skips training entirely. Instead, it extracts sample patches 
    and saves them as readable .png files so you can visually verify what the
    model will learn from, and tune the variance threshold.

--patch_debug_count (default: 50)
    Number of patches to extract when --patch_debug is enabled.


--- EXAMPLES ---

1) Visualize Patches (No Training):
python anima_train_network.py --dataset_config config.toml --output_dir ./out \
    --patch_debug --patch_debug_count 100 \
    --patch_min_size 256 --patch_max_size 512 \
    --patch_variance_threshold 50.0

*(Check the output_dir/patch_debug/ folder for images ending in _PASS or _FAIL)*

2) Standard Patch Training integration:
python anima_train_network.py \
    --enable_patch_training \
    --patch_ratio 0.3 \
    --patch_caption_trigger "a photo of a person" \
    --patch_min_size 256 \
    --patch_max_size 512 \
    --patch_variance_threshold 50.0
