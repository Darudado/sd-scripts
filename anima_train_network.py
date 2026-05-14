# Anima LoRA training script

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    patch_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
import train_network
from library.utils import setup_logging
from library.ramtorch_util import apply_ramtorch_to_module

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self._ot_logged = False    # fires a one-time first-batch OT log
        self._cfm_logged = False   # fires a one-time first-batch CFM log

        # Patch training state
        self._patch_image_paths: List[str] = []
        self._patch_dataset_dirs: List[str] = []
        self._patch_pools: Dict[int, List[str]] = {}  # size -> shuffled list of .npz paths
        self._patch_accumulator: float = 0.0
        self._patch_step_count: int = 0
        self._current_fixed_timesteps: Optional[torch.Tensor] = None
        self._cached_patch_te_outputs: Optional[list] = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        # --- Flow matching feature ---
        if getattr(args, "flow_use_ot", False):
            logger.info("[Anima] Cosine Optimal Transport (OT): ENABLED -- noise vectors will be batch-reassigned each step.")

        if getattr(args, "contrastive_flow_matching", False):
            logger.info(
                f"[Anima] Contrastive Flow Matching (\u0394FM): ENABLED -- "
                f"cfm_lambda={getattr(args, 'cfm_lambda', 0.05)}. "
                "A negative contrastive loss term will be subtracted from the main loss each step."
            )

        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported. / fp8_baseとfp8_base_unetはサポートされていません。")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False  # Anima DiT does not support fp8_scaled

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

        # --- Patch training setup ---
        if getattr(args, "enable_patch_training", False):
            assert args.patch_caption_trigger, (
                "--patch_caption_trigger is required when --enable_patch_training is set"
            )
            # Collect image paths and dataset directories from loaded datasets
            seen_dirs = set()
            for dataset in train_dataset_group.datasets:
                for info in dataset.image_data.values():
                    self._patch_image_paths.append(info.absolute_path)
                    parent = os.path.dirname(info.absolute_path)
                    if parent not in seen_dirs:
                        seen_dirs.add(parent)
                        self._patch_dataset_dirs.append(parent)

            logger.info(
                f"[Patch Training] ENABLED: ratio={args.patch_ratio}, "
                f"size={args.patch_min_size}-{args.patch_max_size}, "
                f"timesteps={args.patch_min_timestep}-{args.patch_max_timestep}, "
                f"trigger='{args.patch_caption_trigger}', "
                f"images={len(self._patch_image_paths)}, "
                f"dirs={len(self._patch_dataset_dirs)}"
            )

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        if args.use_ramtorch and not args.cache_text_encoder_outputs:
            qwen3_text_encoder= apply_ramtorch_to_module(qwen3_text_encoder, "qwen3_text_encoder", accelerator.device, weight_dtype)

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
        )
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
        )

        # Store unsloth preference so that when the base NetworkTrainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        if args.use_ramtorch:
            logger.info("Applying RamTorch to Anima model dit.")
            model = apply_ramtorch_to_module(model, "unet/dit", accelerator.device, model.dtype)

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        # --- Patch latent caching (while VAE is fresh from normal latent caching) ---
        if getattr(args, "enable_patch_training", False) and self._patch_image_paths:
            self._cache_patch_latents(args, vae, weight_dtype, accelerator, dataset)

        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # Cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            # Cache patch trigger caption TE outputs if patch training is enabled
            if getattr(args, "enable_patch_training", False) and args.patch_caption_trigger:
                logger.info(f"[Patch Training] Caching TE outputs for trigger: '{args.patch_caption_trigger}'")
                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
                with accelerator.autocast(), torch.no_grad():
                    tokens_and_masks = tokenize_strategy.tokenize(args.patch_caption_trigger)
                    self._cached_patch_te_outputs = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, text_encoders, tokens_and_masks
                    )
                logger.info(f"[Patch Training] Cached TE outputs: {len(self._cached_patch_te_outputs)} tensors")

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        text_encoder_masks,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
    ):
        # Consume patch timesteps if set by _build_patch_batch
        if self._current_fixed_timesteps is not None:
            fixed_timesteps = self._current_fixed_timesteps
            self._current_fixed_timesteps = None
        anima: anima_models.Anima = unet

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
            with torch.no_grad():
                b_size = latents.size(0)
                lat_flat = latents.view(b_size, -1)
                noise_flat = noise.view(b_size, -1)
                _, (_, col_indices) = train_util.cosine_optimal_transport(lat_flat, noise_flat)
                noise = noise[col_indices.squeeze(0)]
            if not self._ot_logged:
                logger.info(
                    f"[Anima OT] First batch: noise reordered by cosine OT. "
                    f"New noise assignment indices: {col_indices.squeeze(0).tolist()}"
                )
                self._ot_logged = True

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, 
            noise_scheduler, 
            latents, 
            noise, 
            accelerator.device, 
            weight_dtype,
            fixed_timesteps=fixed_timesteps, 
            is_train=is_train,
        )
        # Set T-LoRA timestep mask before timestep scaling (mask expects [0, max_timestep] range)
        self.apply_tlora_mask(timesteps)

        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
            :4
        ]  # ignore caption_dropout_rate which is not needed for training step

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Clear T-LoRA mask after the forward pass
        self.clear_tlora_mask_if_needed()

        # Upcast for grokking
        latents = latents.to(torch.float64)
        noise = noise.to(torch.float64)

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting, noise

    def process_batch(
        self, 
        batch, 
        text_encoders, 
        unet, 
        network, 
        vae, 
        noise_scheduler,
        vae_dtype, 
        weight_dtype, 
        accelerator, 
        args,
        text_encoding_strategy, 
        tokenize_strategy,
        is_train=True, 
        train_text_encoder=True, 
        train_unet=True,
        edm2_model=None,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs.

        When patch training is enabled, this method also handles patch step
        scheduling: on selected steps, the normal batch is replaced with a
        synthetic batch built from pre-cached patch latents.
        """

        # --- Patch step decision ---
        if is_train and getattr(args, "enable_patch_training", False) and self._patch_pools:
            self._patch_accumulator += args.patch_ratio
            if self._patch_accumulator >= 1.0:
                self._patch_accumulator -= 1.0
                patch_batch = self._build_patch_batch(
                    batch, args, accelerator, text_encoders,
                    text_encoding_strategy, tokenize_strategy, weight_dtype,
                )
                if patch_batch is not None:
                    batch = patch_batch

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            # Skip caption dropout processing for patch batches (no dropout rates)
            if not batch.get("_is_patch_batch", False):
                caption_dropout_rates = text_encoder_outputs_list[-1]
                text_encoder_outputs_list = text_encoder_outputs_list[:-1]

                # Apply caption dropout to cached outputs
                text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                    *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
                )
                # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
                batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        if is_train and not self._cfm_logged and getattr(args, "contrastive_flow_matching", False):
            logger.info(
                f"[Anima CFM] First batch: Contrastive Flow Matching is active. "
                f"Negative rolled targets will be computed and subtracted with lambda={getattr(args, 'cfm_lambda', 0.05)}."
            )
            self._cfm_logged = True

        return super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
            edm2_model,
        )

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        model = unet
        model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    # -----------------------------------------------------------------------
    # Patch Training: Caching & Batch Building
    # -----------------------------------------------------------------------

    def _cache_patch_latents(self, args, vae, weight_dtype, accelerator, dataset_group):
        """Pre-generate patch images, VAE-encode, and save to disk.

        Called during the latent caching phase while the VAE is still on GPU
        (or can be cheaply moved there).  Patches are stored as .npz files in
        ``{dataset_dir}/patches/{size}x{size}/`` alongside .png previews.
        """
        # Determine maximum batch size from the dataset group config
        batch_size = args.train_batch_size
        if getattr(dataset_group, "datasets", None):
            max_ds_batch = max([getattr(d, "batch_size", 1) for d in dataset_group.datasets])
            batch_size = max(batch_size, max_ds_batch)
        
        # Actually generate an extra 10% just to be extremely safe against rounding and exhaustion issues
        num_patch_steps = int(args.max_train_steps * args.patch_ratio)
        target = int(num_patch_steps * batch_size * 1.1)

        logger.info(
            f"[Patch Cache] Need {num_patch_steps} patch steps * {batch_size} batch = {target} patches (including 10% buffer)"
        )

        # Bring VAE to GPU for encoding
        vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype)
        org_device = vae.device
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        # Regenerate: clean existing patches
        if getattr(args, "patch_regenerate", False):
            for ddir in self._patch_dataset_dirs:
                patches_root = os.path.join(ddir, "patches")
                if os.path.isdir(patches_root):
                    import shutil
                    logger.info(f"[Patch Cache] Regenerating — removing {patches_root}")
                    shutil.rmtree(patches_root)

        self._patch_pools = patch_utils.generate_and_cache_patches(
            image_paths=self._patch_image_paths,
            dataset_dirs=self._patch_dataset_dirs,
            target_count=target,
            min_size=args.patch_min_size,
            max_size=args.patch_max_size,
            variance_threshold=args.patch_variance_threshold,
            max_retries=args.patch_max_retries,
            feather_px=args.patch_feather_px,
            vae=vae,
            vae_dtype=vae_dtype,
            accelerator=accelerator,
            encode_fn=lambda v, imgs: self.encode_images_to_latents(args, v, imgs),
            shift_scale_fn=lambda lat: self.shift_scale_latents(args, lat),
        )

        # Move VAE back
        vae.to(org_device)
        clean_memory_on_device(accelerator.device)

        # Shuffle each size pool for non-duplicate usage
        for size in self._patch_pools:
            random.shuffle(self._patch_pools[size])

        total_cached = sum(len(v) for v in self._patch_pools.values())
        sizes_str = ", ".join(f"{s}x{s}: {len(v)}" for s, v in sorted(self._patch_pools.items()))
        logger.info(f"[Patch Cache] Ready: {total_cached} patches in {len(self._patch_pools)} size pools [{sizes_str}]")

    def _build_patch_batch(
        self, original_batch, args, accelerator,
        text_encoders, text_encoding_strategy, tokenize_strategy, weight_dtype,
    ):
        """Build a synthetic batch from pre-cached patch latents.

        Selects a random size pool, pops ``batch_size`` items without
        replacement (reshuffling only when the pool is exhausted), loads
        the cached ``.npz`` latents, generates the feathered alpha mask,
        and constructs a batch dict compatible with ``super().process_batch()``.
        """
        batch_size = (
            original_batch["latents"].shape[0]
            if "latents" in original_batch
            else original_batch["images"].shape[0]
        )

        # Find a size pool with enough items
        valid_sizes = [s for s, pool in self._patch_pools.items() if len(pool) >= batch_size]

        if not valid_sizes:
            # Recycle all pools: reshuffle
            for s in self._patch_pools:
                random.shuffle(self._patch_pools[s])
            valid_sizes = [s for s, pool in self._patch_pools.items() if len(pool) >= batch_size]
            if not valid_sizes:
                logger.warning("[Patch] No pools have enough patches for a full batch. Skipping patch step.")
                return None

        size = random.choice(valid_sizes)
        pool = self._patch_pools[size]

        # Pop without replacement
        npz_paths = [pool.pop() for _ in range(batch_size)]

        # Load latents from disk
        latent_list = []
        source_names = []
        for p in npz_paths:
            data = np.load(p)
            latent_list.append(torch.from_numpy(data["latents"]))
            source_names.append(os.path.basename(p).replace(".npz", ""))

        latents = torch.cat(latent_list, dim=0).to(accelerator.device)  # [B, C, H, W]

        # Generate feathered alpha mask at latent resolution
        latent_h = size // 8
        latent_w = size // 8
        single_mask = patch_utils.create_feathered_alpha_mask(
            latent_h, latent_w, args.patch_feather_px, vae_scale=8
        )
        masks = single_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Uniform timesteps within patch range
        patch_timesteps = torch.randint(
            args.patch_min_timestep,
            args.patch_max_timestep + 1,
            (batch_size,),
            device=accelerator.device,
            dtype=torch.float32,
        )
        self._current_fixed_timesteps = patch_timesteps

        # Build batch dict
        patch_batch = {
            "latents": latents,
            "alpha_masks": masks.to(accelerator.device),
            "loss_weights": torch.ones(batch_size, device=accelerator.device),
            "_is_patch_batch": True,  # flag for caption dropout bypass
        }

        # Text encoder outputs — handle both cached and non-cached paths
        if self._cached_patch_te_outputs is not None:
            # Cached TE path: repeat the single-sample outputs for the batch
            repeated = []
            for t in self._cached_patch_te_outputs:
                if isinstance(t, torch.Tensor):
                    # Repeat along batch dimension
                    rep = t.expand(batch_size, *t.shape[1:]).contiguous().to(accelerator.device)
                    repeated.append(rep)
                else:
                    repeated.append(t)
            patch_batch["text_encoder_outputs_list"] = repeated
        else:
            # Non-cached: provide captions and let process_batch encode on the fly
            patch_batch["captions"] = [args.patch_caption_trigger] * batch_size
            tokens_and_masks = tokenize_strategy.tokenize(args.patch_caption_trigger)
            # tokens_and_masks is a list of tensors; repeat each for the batch
            patch_batch["input_ids_list"] = [
                t.repeat(batch_size, *([1] * (t.ndim - 1))).to(accelerator.device)
                if isinstance(t, torch.Tensor) else t
                for t in tokens_and_masks
            ]

        self._patch_step_count += 1
        logger.info(
            f"[Patch Step {self._patch_step_count}] size={size}x{size}, "
            f"timesteps=[{args.patch_min_timestep}-{args.patch_max_timestep}], "
            f"sources=[{', '.join(source_names[:3])}{'...' if len(source_names) > 3 else ''}], "
            f"pool_remaining={len(pool)}"
        )

        return patch_batch

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    # parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )

    # --- Patch training arguments ---
    patch_group = parser.add_argument_group("Patch Training", "Train on random crops from unscaled images to learn fine details.")
    patch_group.add_argument("--enable_patch_training", action="store_true",
        help="Enable patch-based training interleaved with normal steps.")
    patch_group.add_argument("--patch_ratio", type=float, default=0.3,
        help="Fraction of steps replaced with patch steps (default: 0.3 = ~every 3rd step).")
    patch_group.add_argument("--patch_min_size", type=int, default=256,
        help="Minimum patch crop size in pixels (default: 256). Must be divisible by 16.")
    patch_group.add_argument("--patch_max_size", type=int, default=512,
        help="Maximum patch crop size in pixels (default: 512). Must be divisible by 16.")
    patch_group.add_argument("--patch_min_timestep", type=int, default=0,
        help="Minimum timestep for patch steps (default: 0).")
    patch_group.add_argument("--patch_max_timestep", type=int, default=300,
        help="Maximum timestep for patch steps (default: 300). Low values focus on fine details.")
    patch_group.add_argument("--patch_variance_threshold", type=float, default=50.0,
        help="Minimum grayscale pixel variance to accept a patch (default: 50.0). Rejects solid colors.")
    patch_group.add_argument("--patch_feather_px", type=int, default=16,
        help="Feather width in pixels for border alpha mask (default: 16).")
    patch_group.add_argument("--patch_caption_trigger", type=str, default=None,
        help="Caption/trigger word used for all patches. Required when patch training is enabled.")
    patch_group.add_argument("--patch_max_retries", type=int, default=10,
        help="Max retries to extract a valid patch before skipping (default: 10).")
    patch_group.add_argument("--patch_regenerate", action="store_true",
        help="Force regeneration of cached patches even if enough exist on disk.")
    patch_group.add_argument("--patch_debug", action="store_true",
        help="Extract sample patches, save to output_dir/patch_debug/, and exit (no training).")
    patch_group.add_argument("--patch_debug_count", type=int, default=50,
        help="Number of debug patches to extract (default: 50).")

    # Anima-specific default: lower cfm_lambda than the SDXL default of 0.05
    parser.set_defaults(cfm_lambda=0.02)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # --- Patch debug mode: extract samples and exit ---
    if getattr(args, "patch_debug", False):
        # Resolve dataset config path
        config_path = getattr(args, "dataset_config", None)
        if config_path is None:
            print("ERROR: --patch_debug requires --dataset_config to locate images.")
            sys.exit(1)
        output_dir = getattr(args, "output_dir", ".")
        image_paths = patch_utils.collect_image_paths_from_toml(config_path)
        if not image_paths:
            print(f"ERROR: No images found in dataset config: {config_path}")
            sys.exit(1)
        print(f"[Patch Debug] Found {len(image_paths)} images. Extracting patches...")
        patch_utils.save_debug_patches(
            image_paths,
            output_dir,
            count=getattr(args, "patch_debug_count", 50),
            min_size=getattr(args, "patch_min_size", 256),
            max_size=getattr(args, "patch_max_size", 512),
            variance_threshold=getattr(args, "patch_variance_threshold", 50.0),
            max_retries=getattr(args, "patch_max_retries", 10),
        )
        sys.exit(0)

    # Automatically switch to Anima-specific LoRA module if generic one is provided
    if args.network_module == "networks.lora":
        print("Override network module: networks.lora -> networks.lora_anima")
        args.network_module = "networks.lora_anima"

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    # Anima is a Rectified Flow model. Use a private flag to bypass the CFM guard
    # in train_network.py WITHOUT triggering the generic "Using Rectified Flow" log block.
    args._anima_model = True

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
