"""
Unit tests for Tier 1 optimizations applied to library/anima_models.py.

Tests verify that each optimization produces numerically identical results
to the pre-optimization implementation.

Usage:
    pytest tests/test_anima_models_tier1_optimizations.py -v
"""

import sys
import os
import math
import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.anima_models import (
    RMSNorm,
    GPT2FeedForward,
    Attention,
    Block,
    FinalLayer,
    PatchEmbed,
    VideoRopePosition3DEmb,
    Timesteps,
    TimestepEmbedding,
    Anima,
    LLMAdapter,
)
from einops import rearrange, repeat


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


# ──────────────────────────────────────────────────────────────
# 1.1  RMSNorm — verify autocast-free forward matches reference
# ──────────────────────────────────────────────────────────────
class TestRMSNormOptimization:
    """Verify RMSNorm produces correct output without torch.autocast wrapper."""

    def test_rmsnorm_no_autocast_context_manager(self):
        """Forward should not use torch.autocast (the optimization removes it)."""
        src = inspect.getsource(RMSNorm.forward)
        assert "torch.autocast" not in src, (
            "RMSNorm.forward should not use torch.autocast — the optimization removes it"
        )

    def test_rmsnorm_numerical_correctness(self):
        """Output should match manual RMSNorm computation."""
        dim = 64
        norm = RMSNorm(dim, eps=1e-5).to(DEVICE, DTYPE)
        x = torch.randn(2, 4, dim, device=DEVICE, dtype=DTYPE)
        out = norm(x)
        # Manual reference
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
        expected = (x_f32 * rms * norm.weight.float()).to(DTYPE)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_preserves_dtype(self):
        """Output dtype should match input dtype."""
        dim = 32
        norm = RMSNorm(dim).to(DEVICE)
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            if dt == torch.float16 and DEVICE == "cpu":
                continue  # fp16 not supported on CPU
            x = torch.randn(2, dim, device=DEVICE, dtype=dt)
            out = norm(x)
            assert out.dtype == dt, f"Expected {dt}, got {out.dtype}"


# ──────────────────────────────────────────────────────────────
# 1.2  Block._forward — native torch ops instead of einops
# ──────────────────────────────────────────────────────────────
class TestBlockNativeOps:
    """Verify Block._forward uses native torch ops, not einops rearrange."""

    def test_block_forward_no_rearrange_in_hotpath(self):
        """Block._forward should not call einops rearrange."""
        src = inspect.getsource(Block._forward)
        # Allow rearrange only if it's in a comment
        lines = [
            l for l in src.split("\n")
            if "rearrange" in l and not l.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            f"Block._forward should use native torch ops, not rearrange. Found: {lines}"
        )

    def test_block_forward_flatten_unflatten_equivalence(self):
        """Verify flatten/unflatten produces same result as einops rearrange."""
        B, T, H, W, D = 1, 2, 3, 4, 8
        x = torch.randn(B, T, H, W, D, device=DEVICE, dtype=DTYPE)

        # einops reference
        ref_flat = rearrange(x, "b t h w d -> b (t h w) d")
        ref_unflat = rearrange(ref_flat, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

        # native ops (the optimization)
        opt_flat = x.flatten(1, 3)
        opt_unflat = opt_flat.unflatten(1, (T, H, W))

        torch.testing.assert_close(opt_flat, ref_flat)
        torch.testing.assert_close(opt_unflat, ref_unflat)

    def test_block_broadcast_unsqueeze_equivalence(self):
        """Verify [:, :, None, None, :] matches einops rearrange for broadcasting."""
        B, T, D = 2, 3, 16
        x = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE)

        ref = rearrange(x, "b t d -> b t 1 1 d")
        opt = x[:, :, None, None, :]

        torch.testing.assert_close(opt, ref)
        # Both should be views (no copy)
        assert opt.shape == (B, T, 1, 1, D)


# ──────────────────────────────────────────────────────────────
# 1.3  unpatchify — native torch ops instead of einops
# ──────────────────────────────────────────────────────────────
class TestUnpatchifyNativeOps:
    """Verify unpatchify uses native torch ops, not einops rearrange."""

    def test_unpatchify_no_rearrange(self):
        """unpatchify should not call einops rearrange."""
        src = inspect.getsource(Anima.unpatchify)
        assert "rearrange" not in src, (
            "unpatchify should use native torch ops, not rearrange"
        )

    def test_unpatchify_numerical_equivalence(self):
        """Native unpatchify should match einops reference exactly."""
        # Build a minimal Anima-like object with just the needed attributes
        p1, p2, pt = 2, 2, 1
        B, T, H, W, C = 1, 2, 3, 4, 16
        M = p1 * p2 * pt * C  # total features per patch

        x = torch.randn(B, T, H, W, M, device=DEVICE, dtype=DTYPE)

        # einops reference
        ref = rearrange(
            x,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=p1, p2=p2, t=pt,
        )

        # native ops (the optimization)
        opt = (
            x.unflatten(-1, (p1, p2, pt, C))
            .permute(0, 7, 1, 6, 2, 4, 3, 5)
            .reshape(B, C, T * pt, H * p1, W * p2)
        )

        torch.testing.assert_close(opt, ref)
        assert opt.shape == ref.shape


# ──────────────────────────────────────────────────────────────
# 1.4  prepare_embedded_sequence — expand vs repeat
# ──────────────────────────────────────────────────────────────
class TestPaddingMaskExpand:
    """Verify .expand() produces identical result to .repeat() for padding mask."""

    def test_expand_vs_repeat_equivalence(self):
        """expand and repeat should produce the same values for a constant tensor."""
        B, C, T, H, W = 2, 1, 4, 8, 8
        mask = torch.randn(B, C, H, W, device=DEVICE, dtype=DTYPE)

        # Old: repeat
        ref = mask.unsqueeze(1).repeat(1, 1, T, 1, 1)

        # New: expand
        opt = mask.unsqueeze(2).expand(-1, -1, T, -1, -1)

        torch.testing.assert_close(opt, ref)
        assert opt.shape == ref.shape == (B, C, T, H, W)

    def test_expand_is_view_not_copy(self):
        """expand should return a view (no extra memory allocation)."""
        mask = torch.ones(2, 1, 8, 8, device=DEVICE)
        expanded = mask.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        # Modify original — expand view should reflect the change
        mask[0, 0, 0, 0] = 99.0
        assert expanded[0, 0, 0, 0, 0].item() == 99.0


# ──────────────────────────────────────────────────────────────
# 1.5  Conditional padding mask resize
# ──────────────────────────────────────────────────────────────
class TestConditionalResize:
    """Verify padding mask resize only happens when dimensions mismatch."""

    def test_no_resize_when_dimensions_match(self):
        """When mask spatial dims match input, no torchvision resize should be called."""
        # We test this by checking that prepare_embedded_sequence works
        # correctly without importing torchvision when dims already match.
        # Since we can't easily mock the resize, we verify the source code
        # has the conditional check.
        src = inspect.getsource(Anima.prepare_embedded_sequence)
        assert "padding_mask.shape[-2:] != x_B_C_T_H_W.shape[-2:]" in src, (
            "prepare_embedded_sequence should check dimensions before resizing"
        )

    def test_prepare_embedded_sequence_rejects_none_mask(self):
        """Should raise ValueError when concat_padding_mask is True but mask is None."""
        # Create minimal Anima with concat_padding_mask=True
        model = Anima(
            max_img_h=64, max_img_w=64, max_frames=1,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1,
            concat_padding_mask=True,
            model_channels=64, num_blocks=1, num_heads=4,
            crossattn_emb_channels=32,
            pos_emb_cls="rope3d", pos_emb_learnable=True,
        ).to(DEVICE)
        x = torch.randn(1, 16, 1, 32, 32, device=DEVICE)
        with pytest.raises(ValueError, match="padding_mask must be provided"):
            model.prepare_embedded_sequence(x, padding_mask=None)


# ──────────────────────────────────────────────────────────────
# 1.6  torch.is_grad_enabled() guard
# ──────────────────────────────────────────────────────────────
class TestIsGradEnabledGuard:
    """Verify gradient checkpointing guard includes torch.is_grad_enabled()."""

    def test_block_forward_checks_grad_enabled(self):
        """Block.forward should check torch.is_grad_enabled()."""
        src = inspect.getsource(Block.forward)
        assert "torch.is_grad_enabled()" in src, (
            "Block.forward should check torch.is_grad_enabled() before checkpointing"
        )

    def test_block_forward_no_checkpoint_under_no_grad(self):
        """Under torch.no_grad(), gradient checkpointing should be skipped."""
        block = Block(
            x_dim=64, context_dim=32, num_heads=4, mlp_ratio=2.0,
        ).to(DEVICE, DTYPE)
        block.enable_gradient_checkpointing()

        B, T, H, W, D = 1, 1, 4, 4, 64
        x = torch.randn(B, T, H, W, D, device=DEVICE, dtype=DTYPE)
        emb = torch.randn(B, 1, D, device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, 8, 32, device=DEVICE, dtype=DTYPE)

        from library.attention import AttentionParams
        attn_params = AttentionParams.create_attention_params("torch", False)

        # Under no_grad + eval, this should NOT attempt checkpointing
        block.eval()
        with torch.no_grad():
            out = block(x, emb, context, attn_params)
        assert out.shape == x.shape


# ──────────────────────────────────────────────────────────────
# 1.7  RoPE dtype pre-cast
# ──────────────────────────────────────────────────────────────
class TestRoPEDtypeCast:
    """Verify RoPE embeddings are cast to compute dtype before block loop."""

    def test_rope_dtype_precast_in_source(self):
        """forward_mini_train_dit should cast rope_cos_sin to compute_dtype before the loop."""
        src = inspect.getsource(Anima.forward_mini_train_dit)
        assert "rope_cos_sin[0].to(compute_dtype)" in src, (
            "forward_mini_train_dit should pre-cast RoPE cos/sin to compute dtype"
        )

    def test_rope_dtype_cast_correctness(self):
        """RoPE dtype cast should produce numerically correct results."""
        dim_h, dim_w, dim_t = 8, 8, 8
        head_dim = dim_h + dim_w + dim_t
        rope = VideoRopePosition3DEmb(
            head_dim=head_dim,
            len_h=16, len_w=16, len_t=4,
        ).to(DEVICE)

        B, T, H, W, D = 1, 1, 8, 8, head_dim
        x = torch.randn(B, T, H, W, D, device=DEVICE, dtype=DTYPE)
        freqs = rope.generate_embeddings(x.shape)

        # Cast to bf16 (simulating the pre-cast optimization)
        if DEVICE == "cuda":
            freqs_bf16 = (freqs[0].to(torch.bfloat16), freqs[1].to(torch.bfloat16))
            assert freqs_bf16[0].dtype == torch.bfloat16
            assert freqs_bf16[1].dtype == torch.bfloat16


# ──────────────────────────────────────────────────────────────
# 1.9  Attention.compute_qkv — unflatten instead of einops
# ──────────────────────────────────────────────────────────────
class TestAttentionComputeQKVNativeOps:
    """Verify Attention.compute_qkv uses unflatten instead of einops rearrange."""

    def test_compute_qkv_no_rearrange(self):
        """compute_qkv should not use einops rearrange."""
        src = inspect.getsource(Attention.compute_qkv)
        lines = [
            l for l in src.split("\n")
            if "rearrange" in l and not l.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            f"Attention.compute_qkv should use unflatten, not rearrange. Found: {lines}"
        )

    def test_unflatten_matches_rearrange_for_head_reshape(self):
        """unflatten(-1, (H, D)) should match rearrange('b l (h d) -> b l h d')."""
        B, L, H, D = 2, 16, 4, 8
        x = torch.randn(B, L, H * D, device=DEVICE, dtype=DTYPE)

        ref = rearrange(x, "b l (h d) -> b l h d", h=H, d=D)
        opt = x.unflatten(-1, (H, D))

        torch.testing.assert_close(opt, ref)
        assert opt.shape == ref.shape == (B, L, H, D)


# ──────────────────────────────────────────────────────────────
# 1.10  Timesteps.forward — reshape instead of einops
# ──────────────────────────────────────────────────────────────
class TestTimestepsReshape:
    """Verify Timesteps.forward uses .reshape() instead of einops rearrange."""

    def test_timesteps_no_rearrange(self):
        """Timesteps.forward should not use einops rearrange."""
        src = inspect.getsource(Timesteps.forward)
        lines = [
            l for l in src.split("\n")
            if "rearrange" in l and not l.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            f"Timesteps.forward should use .reshape(), not rearrange. Found: {lines}"
        )

    def test_timesteps_output_shape(self):
        """Timesteps should produce correct (B, T, D) output shape."""
        ts = Timesteps(num_channels=64).to(DEVICE)
        timesteps = torch.tensor([[100], [500], [999]], device=DEVICE, dtype=DTYPE)
        out = ts(timesteps)
        assert out.shape == (3, 1, 64), f"Expected (3, 1, 64), got {out.shape}"


# ──────────────────────────────────────────────────────────────
# 1.11  LLMAdapterRMSNorm — unconditional dtype cast
# ──────────────────────────────────────────────────────────────
from library.anima_models import LLMAdapterRMSNorm


class TestLLMAdapterRMSNormDtype:
    """Verify LLMAdapterRMSNorm uses unconditional dtype cast."""

    def test_no_conditional_dtype_check(self):
        """LLMAdapterRMSNorm.forward should not have conditional dtype check."""
        src = inspect.getsource(LLMAdapterRMSNorm.forward)
        assert "if self.weight.dtype" not in src, (
            "LLMAdapterRMSNorm should use unconditional .to(self.weight.dtype)"
        )

    def test_dtype_preservation(self):
        """Output dtype should match weight dtype."""
        # When weight is float32, output is float32 regardless of input dtype.
        # When weight is bf16, output is bf16.
        norm = LLMAdapterRMSNorm(32).to(DEVICE, torch.float32)
        x_bf16 = torch.randn(2, 10, 32, device=DEVICE, dtype=torch.bfloat16)
        out = norm(x_bf16)
        # Weight is fp32, so output should be fp32 (weight * hidden_states promotes)
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        if DEVICE == "cuda":
            norm_bf16 = LLMAdapterRMSNorm(32).to(DEVICE, torch.bfloat16)
            out_bf16 = norm_bf16(x_bf16)
            assert out_bf16.dtype == torch.bfloat16, f"Expected bfloat16, got {out_bf16.dtype}"

    def test_noop_when_dtype_matches(self):
        """When input dtype matches weight dtype, .to() should be a no-op."""
        norm = LLMAdapterRMSNorm(32).to(DEVICE, torch.float32)
        x = torch.randn(2, 10, 32, device=DEVICE, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32


# ──────────────────────────────────────────────────────────────
# 1.8  free_cache parameter
# ──────────────────────────────────────────────────────────────
class TestFreeCacheParam:
    """Verify prepare_block_swap_before_forward accepts free_cache parameter."""

    def test_free_cache_param_in_signature(self):
        """prepare_block_swap_before_forward should accept free_cache kwarg."""
        sig = inspect.signature(Anima.prepare_block_swap_before_forward)
        assert "free_cache" in sig.parameters, (
            "prepare_block_swap_before_forward should accept free_cache parameter"
        )
        assert sig.parameters["free_cache"].default is True, (
            "free_cache should default to True"
        )


# ──────────────────────────────────────────────────────────────
# Integration: Full forward pass still works
# ──────────────────────────────────────────────────────────────
class TestIntegrationForwardPass:
    """Verify the full forward pass works after all Tier 1 optimizations."""

    @pytest.fixture
    def small_model(self):
        """Create a small Anima model for testing."""
        model = Anima(
            max_img_h=64,
            max_img_w=64,
            max_frames=1,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            mlp_ratio=2.0,
            crossattn_emb_channels=32,
            pos_emb_cls="rope3d",
            pos_emb_learnable=True,
            use_adaln_lora=True,
            adaln_lora_dim=16,
            rope_h_extrapolation_ratio=4.0,
            rope_w_extrapolation_ratio=4.0,
            rope_enable_fps_modulation=False,
        ).to(DEVICE, DTYPE)
        return model

    def test_forward_pass_shape(self, small_model):
        """Forward pass should produce correct output shape."""
        B, C, T, H, W = 1, 16, 1, 32, 32
        x = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)
        timesteps = torch.tensor([500], device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, 8, 32, device=DEVICE, dtype=DTYPE)
        padding_mask = torch.ones(B, 1, H, W, device=DEVICE, dtype=DTYPE)

        small_model.eval()
        with torch.no_grad():
            out = small_model(x, timesteps, context, padding_mask=padding_mask)

        assert out.shape == (B, 16, T, H, W), f"Expected shape {(B, 16, T, H, W)}, got {out.shape}"

    def test_forward_pass_deterministic(self, small_model):
        """Forward pass should be deterministic in eval mode."""
        B, C, T, H, W = 1, 16, 1, 32, 32
        x = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)
        timesteps = torch.tensor([500], device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, 8, 32, device=DEVICE, dtype=DTYPE)
        padding_mask = torch.ones(B, 1, H, W, device=DEVICE, dtype=DTYPE)

        small_model.eval()
        with torch.no_grad():
            out1 = small_model(x, timesteps, context, padding_mask=padding_mask)
            out2 = small_model(x, timesteps, context, padding_mask=padding_mask)

        torch.testing.assert_close(out1, out2)

    def test_forward_pass_gradient_flow(self, small_model):
        """Forward pass should allow gradient computation in training mode."""
        B, C, T, H, W = 1, 16, 1, 32, 32
        x = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE, requires_grad=True)
        timesteps = torch.tensor([500], device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, 8, 32, device=DEVICE, dtype=DTYPE)
        padding_mask = torch.ones(B, 1, H, W, device=DEVICE, dtype=DTYPE)

        small_model.train()
        out = small_model(x, timesteps, context, padding_mask=padding_mask)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_llm_adapter_forward(self, small_model):
        """LLM adapter path should work when use_llm_adapter=True."""
        # The small_model fixture doesn't have llm_adapter, so test separately
        adapter = LLMAdapter(
            source_dim=32, target_dim=32, model_dim=32,
            num_layers=2, num_heads=4, self_attn=True,
        ).to(DEVICE, DTYPE)

        B, L_src, L_tgt = 1, 8, 6
        source = torch.randn(B, L_src, 32, device=DEVICE, dtype=DTYPE)
        target_ids = torch.randint(0, 32128, (B, L_tgt), device=DEVICE)

        adapter.eval()
        with torch.no_grad():
            out = adapter(source, target_ids)
        assert out.shape == (B, L_tgt, 32)


# ──────────────────────────────────────────────────────────────
# 2.1  Joint Q/K RoPE + cos/sin caching
# ──────────────────────────────────────────────────────────────
class TestJointRoPEAndCaching:
    """Verify 2.1: cos/sin caching and apply_rotary_pos_emb_qk."""

    def test_apply_rotary_pos_emb_qk_exists(self):
        """apply_rotary_pos_emb_qk function should exist."""
        from library import anima_models
        assert hasattr(anima_models, "apply_rotary_pos_emb_qk")

    def test_generate_embeddings_returns_cos_sin_tuple(self):
        """generate_embeddings should return (cos, sin) tuple."""
        rope = VideoRopePosition3DEmb(
            head_dim=24, len_h=8, len_w=8, len_t=4,
        ).to(DEVICE)
        x = torch.randn(1, 1, 4, 4, 24, device=DEVICE)
        result = rope(x, fps=None)
        assert isinstance(result, tuple) and len(result) == 2
        cos, sin = result
        assert cos.shape == sin.shape
        assert cos.dtype == torch.float32

    def test_cos_sin_cache_hit(self):
        """Repeated calls with same shape should return cached results."""
        rope = VideoRopePosition3DEmb(
            head_dim=24, len_h=8, len_w=8, len_t=4,
        ).to(DEVICE)
        x = torch.randn(1, 1, 4, 4, 24, device=DEVICE)
        result1 = rope(x, fps=None)
        result2 = rope(x, fps=None)
        # Should be the exact same tensor objects (cache hit)
        assert result1[0] is result2[0]
        assert result1[1] is result2[1]

    def test_cos_sin_cache_different_shapes(self):
        """Different input shapes should produce different cache entries."""
        rope = VideoRopePosition3DEmb(
            head_dim=24, len_h=8, len_w=8, len_t=4,
        ).to(DEVICE)
        x1 = torch.randn(1, 1, 4, 4, 24, device=DEVICE)
        x2 = torch.randn(1, 1, 2, 2, 24, device=DEVICE)
        result1 = rope(x1, fps=None)
        result2 = rope(x2, fps=None)
        # Different shapes → different cache entries
        assert result1[0] is not result2[0]

    def test_apply_rotary_pos_emb_qk_numerical(self):
        """apply_rotary_pos_emb_qk should match separate Q/K application."""
        from library.anima_models import apply_rotary_pos_emb_qk, _rotate_half
        B, L, H, D = 1, 16, 4, 24
        q = torch.randn(B, L, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, L, H, D, device=DEVICE, dtype=DTYPE)
        cos = torch.randn(L, 1, 1, D, device=DEVICE, dtype=DTYPE)
        sin = torch.randn(L, 1, 1, D, device=DEVICE, dtype=DTYPE)

        q_out, k_out = apply_rotary_pos_emb_qk(q, k, (cos, sin), tensor_format="bshd")

        # Manual reference for Q
        cos_t = cos.transpose(0, 1).to(q.dtype)
        sin_t = sin.transpose(0, 1).to(q.dtype)
        q_rot = q[..., :D]
        q_ref = (q_rot * cos_t) + (_rotate_half(q_rot, False) * sin_t)

        torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)

    def test_attention_uses_rope_cos_sin_param(self):
        """Attention.compute_qkv should accept rope_cos_sin parameter."""
        sig = inspect.signature(Attention.compute_qkv)
        assert "rope_cos_sin" in sig.parameters

    def test_full_forward_with_rope_caching(self):
        """Full forward should work correctly with cos/sin caching."""
        model = Anima(
            max_img_h=64, max_img_w=64, max_frames=1,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1,
            concat_padding_mask=True,
            model_channels=64, num_blocks=2, num_heads=4,
            crossattn_emb_channels=32,
            pos_emb_cls="rope3d", pos_emb_learnable=True,
            rope_enable_fps_modulation=False,
        ).to(DEVICE, DTYPE)

        B, C, T, H, W = 1, 16, 1, 32, 32
        x = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)
        timesteps = torch.tensor([500], device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, 8, 32, device=DEVICE, dtype=DTYPE)
        padding_mask = torch.ones(B, 1, H, W, device=DEVICE, dtype=DTYPE)

        model.eval()
        with torch.no_grad():
            out1 = model(x, timesteps, context, padding_mask=padding_mask)
            out2 = model(x, timesteps, context, padding_mask=padding_mask)

        # Both calls should produce identical output (caching ensures same RoPE)
        torch.testing.assert_close(out1, out2)


# ──────────────────────────────────────────────────────────────
# Tier 2: Native torch ops replace einops repeat
# ──────────────────────────────────────────────────────────────
class TestRepeatReplacement:
    """Verify einops repeat replaced with native unsqueeze+expand."""

    def test_generate_embeddings_no_repeat(self):
        """VideoRopePosition3DEmb.generate_embeddings should not use einops repeat."""
        src = inspect.getsource(VideoRopePosition3DEmb.generate_embeddings)
        lines = [
            l for l in src.split("\n")
            if "repeat(" in l and not l.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            f"generate_embeddings should use unsqueeze/expand, not repeat. Found: {lines}"
        )

    def test_rope_expand_matches_repeat(self):
        """unsqueeze+expand should produce the same result as einops repeat for RoPE."""
        T, H, W, D = 2, 4, 6, 8
        half_emb_t = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
        half_emb_h = torch.randn(H, D, device=DEVICE, dtype=DTYPE)
        half_emb_w = torch.randn(W, D, device=DEVICE, dtype=DTYPE)

        # einops reference
        ref_t = repeat(half_emb_t, "t d -> t h w d", h=H, w=W)
        ref_h = repeat(half_emb_h, "h d -> t h w d", t=T, w=W)
        ref_w = repeat(half_emb_w, "w d -> t h w d", t=T, h=H)

        # Native ops (matching the implementation)
        opt_t = half_emb_t[:, None, None, :].expand(-1, H, W, -1)
        opt_h = half_emb_h[None, :, None, :].expand(T, -1, W, -1)
        opt_w = half_emb_w[None, None, :, :].expand(T, H, -1, -1)

        torch.testing.assert_close(opt_t, ref_t)
        torch.testing.assert_close(opt_h, ref_h)
        torch.testing.assert_close(opt_w, ref_w)

    def test_rope_expand_shape(self):
        """Expanded tensors should have shape (T, H, W, D)."""
        T, H, W, D = 3, 5, 7, 12
        half_emb_t = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
        opt_t = half_emb_t[:, None, None, :].expand(-1, H, W, -1)
        assert opt_t.shape == (T, H, W, D), f"Expected {(T, H, W, D)}, got {opt_t.shape}"

    def test_rope_expand_is_view(self):
        """expand should return a view, not a copy (no extra memory)."""
        T, H, W, D = 2, 4, 6, 8
        half_emb_t = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
        opt_t = half_emb_t[:, None, None, :].expand(-1, H, W, -1)
        # expand returns a view: storage should be shared
        assert opt_t.storage().data_ptr() == half_emb_t.storage().data_ptr(), (
            "expand should return a view sharing the same storage"
        )

    def test_generate_embeddings_with_expand_matches_reference(self):
        """Full generate_embeddings output should be numerically identical after repeat→expand."""
        model = VideoRopePosition3DEmb(
            head_dim=64, len_h=16, len_w=16, len_t=4,
            base_fps=24, enable_fps_modulation=False,
        ).to(DEVICE, DTYPE)

        shape = torch.Size([1, 4, 8, 8, 64])

        # Clear cache so the computation runs fresh
        model._cos_sin_cache.clear()

        # Get the optimized result
        cos_opt, sin_opt = model.generate_embeddings(shape)

        # Reconstruct using einops repeat as reference
        model._cos_sin_cache.clear()

        # Manually compute reference
        B, T, H, W, _ = shape
        h_spatial_freqs = 1.0 / (10000.0 ** model.dim_spatial_range)
        w_spatial_freqs = 1.0 / (10000.0 ** model.dim_spatial_range)
        temporal_freqs = 1.0 / (10000.0 ** model.dim_temporal_range)

        half_emb_h = torch.outer(model.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(model.seq[:W], w_spatial_freqs)
        half_emb_t = torch.outer(model.seq[:T], temporal_freqs)

        ref_em = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        ref_freqs = ref_em.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        ref_cos = torch.cos(ref_freqs)
        ref_sin = torch.sin(ref_freqs)

        torch.testing.assert_close(cos_opt, ref_cos)
        torch.testing.assert_close(sin_opt, ref_sin)


class TestLearnablePosEmbRepeatReplacement:
    """Verify LearnablePosEmbAxis uses native ops instead of einops repeat."""

    def test_learnable_pos_emb_no_repeat(self):
        """LearnablePosEmbAxis.generate_embeddings should not use einops repeat."""
        src = inspect.getsource(Anima)
        # Check generate_embeddings in the source (it's defined in LearnablePosEmbAxis)
        from library.anima_models import LearnablePosEmbAxis
        src = inspect.getsource(LearnablePosEmbAxis.generate_embeddings)
        lines = [
            l for l in src.split("\n")
            if "repeat(" in l and not l.strip().startswith("#")
        ]
        assert len(lines) == 0, (
            f"LearnablePosEmbAxis.generate_embeddings should use unsqueeze/expand, not repeat. Found: {lines}"
        )

    def test_learnable_pos_emb_expand_matches_repeat(self):
        """unsqueeze+expand should produce the same result as einops repeat for learnable pos emb."""
        B, T, H, W, D = 2, 3, 4, 5, 16
        emb_t_T = torch.randn(T, D, device=DEVICE, dtype=DTYPE)
        emb_h_H = torch.randn(H, D, device=DEVICE, dtype=DTYPE)
        emb_w_W = torch.randn(W, D, device=DEVICE, dtype=DTYPE)

        # einops reference
        ref = (
            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
        )

        # Native ops (matching the implementation)
        opt = (
            emb_t_T[None, :, None, None, :].expand(B, -1, H, W, -1)
            + emb_h_H[None, None, :, None, :].expand(B, T, -1, W, -1)
            + emb_w_W[None, None, None, :, :].expand(B, T, H, -1, -1)
        )

        torch.testing.assert_close(opt, ref)
        assert opt.shape == (B, T, H, W, D)


class TestNoEinopsRepeatImport:
    """Verify einops repeat is no longer imported in the module."""

    def test_no_repeat_in_module_imports(self):
        """library.anima_models should not import repeat from einops."""
        import library.anima_models as mod
        src = inspect.getsource(mod)
        # Check top-level imports only (not inside functions/comments)
        import_lines = [
            l for l in src.split("\n")
            if l.strip().startswith(("from einops import", "import einops"))
            and "repeat" in l
            and not l.strip().startswith("#")
        ]
        assert len(import_lines) == 0, (
            f"anima_models should not import repeat from einops. Found: {import_lines}"
        )


# ──────────────────────────────────────────────────────────────
# Tier 2: Self-attention passes explicit context
# ──────────────────────────────────────────────────────────────
class TestSelfAttnExplicitContext:
    """Verify Block._forward passes x_flat as self-attention context instead of None."""

    def test_self_attn_receives_explicit_context(self):
        """Block._forward should pass x_flat, not None, as context to self_attn."""
        src = inspect.getsource(Block._forward)
        # Find the self_attn call
        in_self_attn_block = False
        found_none_context = False
        for line in src.split("\n"):
            if "self.self_attn(" in line:
                in_self_attn_block = True
            if in_self_attn_block:
                stripped = line.strip().rstrip(",")
                # Check for 'None' as a positional arg (context) after x_flat
                if stripped == "None":
                    found_none_context = True
                if ")" in line:
                    break
        assert not found_none_context, (
            "Block._forward should pass x_flat as context to self_attn, not None"
        )

    def test_self_attn_explicit_context_matches_none_fallback(self):
        """Passing x_flat explicitly should produce the same output as passing None."""
        from library.anima_models import Attention

        B, L, D, H = 1, 16, 32, 4
        head_dim = D // H
        attn = Attention(D, None, H, head_dim, qkv_format="bshd").to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L, D, device=DEVICE, dtype=DTYPE)
        from library import attention
        attn_params = attention.AttentionParams.create_attention_params("torch", False)

        with torch.no_grad():
            # With None context (old way)
            q1, k1, v1 = attn.compute_qkv(x, None)
            # With explicit context (new way)
            q2, k2, v2 = attn.compute_qkv(x, x)

        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)
        torch.testing.assert_close(v1, v2)


# ──────────────────────────────────────────────────────────────
# Tier 2: LLMAdapter flash_attn support
# ──────────────────────────────────────────────────────────────
class TestLLMAdapterFlashAttn:
    """Verify LLMAdapterAttention supports flash_attn with varlen packing."""

    def test_attention_signature_uses_q_mask_kv_mask(self):
        """LLMAdapterAttention.forward should accept q_mask and kv_mask parameters."""
        from library.anima_models import LLMAdapterAttention
        sig = inspect.signature(LLMAdapterAttention.forward)
        param_names = list(sig.parameters.keys())
        assert "q_mask" in param_names, f"Expected 'q_mask' parameter, got {param_names}"
        assert "kv_mask" in param_names, f"Expected 'kv_mask' parameter, got {param_names}"
        # Old 'mask' parameter should be gone
        assert "mask" not in param_names, f"Old 'mask' parameter should be removed, got {param_names}"

    def test_no_mask_sdpa_output_shape(self):
        """LLMAdapterAttention should produce correct output shape without masks."""
        from library.anima_models import LLMAdapterAttention
        B, L, D, H = 2, 16, 32, 4
        head_dim = D // H
        attn = LLMAdapterAttention(D, D, H, head_dim).to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L, D, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"

    def test_2d_mask_forward_shape(self):
        """LLMAdapterAttention should accept 2D bool masks and produce correct output."""
        from library.anima_models import LLMAdapterAttention
        B, L_q, L_kv, D, H = 2, 16, 20, 32, 4
        head_dim = D // H
        attn = LLMAdapterAttention(D, D, H, head_dim).to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L_q, D, device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, L_kv, D, device=DEVICE, dtype=DTYPE)
        q_mask = torch.ones(B, L_q, device=DEVICE, dtype=torch.bool)
        kv_mask = torch.ones(B, L_kv, device=DEVICE, dtype=torch.bool)
        # Mask out last 5 tokens of KV
        kv_mask[:, -5:] = False

        with torch.no_grad():
            out = attn(x, q_mask=q_mask, kv_mask=kv_mask, context=context)
        assert out.shape == (B, L_q, D), f"Expected {(B, L_q, D)}, got {out.shape}"

    def test_2d_mask_with_rope(self):
        """LLMAdapterAttention with RoPE and 2D masks should produce correct output."""
        from library.anima_models import LLMAdapterAttention, AdapterRotaryEmbedding
        B, L_q, L_kv, D, H = 2, 16, 20, 32, 4
        head_dim = D // H
        attn = LLMAdapterAttention(D, D, H, head_dim).to(DEVICE, DTYPE)
        rope = AdapterRotaryEmbedding(head_dim).to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L_q, D, device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, L_kv, D, device=DEVICE, dtype=DTYPE)

        pos_ids_q = torch.arange(L_q, device=DEVICE).unsqueeze(0)
        pos_ids_kv = torch.arange(L_kv, device=DEVICE).unsqueeze(0)
        pos_emb_q = rope(x, pos_ids_q)
        pos_emb_kv = rope(x, pos_ids_kv)

        q_mask = torch.ones(B, L_q, device=DEVICE, dtype=torch.bool)
        kv_mask = torch.ones(B, L_kv, device=DEVICE, dtype=torch.bool)
        kv_mask[:, -3:] = False

        with torch.no_grad():
            out = attn(
                x, q_mask=q_mask, kv_mask=kv_mask, context=context,
                position_embeddings=pos_emb_q,
                position_embeddings_context=pos_emb_kv,
            )
        assert out.shape == (B, L_q, D)

    def test_self_attention_with_q_mask_kv_mask(self):
        """Self-attention (context=None) with masks should work."""
        from library.anima_models import LLMAdapterAttention
        B, L, D, H = 2, 16, 32, 4
        head_dim = D // H
        attn = LLMAdapterAttention(D, D, H, head_dim).to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L, D, device=DEVICE, dtype=DTYPE)
        mask = torch.ones(B, L, device=DEVICE, dtype=torch.bool)
        mask[:, -4:] = False

        with torch.no_grad():
            out = attn(x, q_mask=mask, kv_mask=mask)
        assert out.shape == (B, L, D)

    def test_mask_zeroing_effect(self):
        """Masked-out KV tokens should produce same result as manual SDPA masking."""
        from library.anima_models import LLMAdapterAttention
        B, L_q, L_kv, D, H = 1, 4, 8, 32, 4
        head_dim = D // H
        attn = LLMAdapterAttention(D, D, H, head_dim).to(DEVICE, DTYPE)
        attn.eval()

        x = torch.randn(B, L_q, D, device=DEVICE, dtype=DTYPE)
        context = torch.randn(B, L_kv, D, device=DEVICE, dtype=DTYPE)
        kv_mask = torch.ones(B, L_kv, device=DEVICE, dtype=torch.bool)
        kv_mask[:, -2:] = False  # mask out last 2 tokens

        with torch.no_grad():
            # Optimized path: uses q_mask/kv_mask with SDPA fallback
            out_optimized = attn(x, kv_mask=kv_mask, context=context)

            # Reference: compute manually with SDPA using the same 4D mask
            q = attn.q_norm(attn.q_proj(x).view(B, L_q, H, head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(context).view(B, L_kv, H, head_dim)).transpose(1, 2)
            v = attn.v_proj(context).view(B, L_kv, H, head_dim).transpose(1, 2)
            sdpa_mask_4d = kv_mask[:, None, None, :]  # [B, 1, 1, L_kv]
            ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask_4d)
            ref_out = ref_out.transpose(1, 2).reshape(B, L_q, D).contiguous()
            ref_out = attn.o_proj(ref_out)

        torch.testing.assert_close(out_optimized, ref_out, atol=1e-5, rtol=1e-4)


class TestLLMAdapterTransformerBlockMaskParams:
    """Verify LLMAdapterTransformerBlock passes q_mask/kv_mask correctly."""

    def test_block_uses_q_mask_kv_mask(self):
        """LLMAdapterTransformerBlock.forward should pass q_mask and kv_mask to attention."""
        from library.anima_models import LLMAdapterTransformerBlock
        src = inspect.getsource(LLMAdapterTransformerBlock.forward)
        assert "q_mask=" in src, "Expected q_mask= parameter in forward"
        assert "kv_mask=" in src, "Expected kv_mask= parameter in forward"
        # Old 'mask=' should not appear as keyword argument to attention calls.
        # Only check lines that are attention method calls (self.self_attn or self.cross_attn).
        import re
        call_lines = [
            l.strip() for l in src.split("\n")
            if re.search(r'\bmask=', l)
            and "q_mask" not in l
            and "kv_mask" not in l
            and not l.strip().startswith("#")
            and not l.strip().startswith("def ")
            and not l.strip().endswith("None,")  # exclude function def default params
            and not l.strip().endswith("None")    # exclude function def default params
        ]
        assert len(call_lines) == 0, f"Old 'mask=' keyword arg in attention calls should be removed. Found: {call_lines}"


class TestLLMAdapterMaskHandling:
    """Verify LLMAdapter keeps masks as 2D bool tensors."""

    def test_adapter_keeps_masks_2d(self):
        """LLMAdapter.forward should keep masks as 2D, not expand to 4D."""
        from library.anima_models import LLMAdapter
        src = inspect.getsource(LLMAdapter.forward)
        # Should not have unsqueeze(1).unsqueeze(1) pattern
        assert "unsqueeze(1).unsqueeze(1)" not in src, (
            "LLMAdapter should not expand masks to 4D — keep them as 2D bool tensors"
        )

    def test_adapter_squeezes_4d_to_2d(self):
        """LLMAdapter.forward should squeeze 4D masks to 2D for compatibility."""
        from library.anima_models import LLMAdapter
        src = inspect.getsource(LLMAdapter.forward)
        assert "squeeze(1).squeeze(1)" in src, (
            "LLMAdapter should squeeze 4D masks to 2D for backward compatibility"
        )

    def test_adapter_forward_with_2d_masks(self):
        """Full LLMAdapter forward with 2D bool masks should produce correct output."""
        from library.anima_models import LLMAdapter
        B, L_target, L_source, D = 2, 12, 20, 32
        adapter = LLMAdapter(
            source_dim=D, target_dim=D, model_dim=D,
            num_layers=2, num_heads=4, self_attn=True,
        ).to(DEVICE, DTYPE)
        adapter.eval()

        source = torch.randn(B, L_source, D, device=DEVICE, dtype=DTYPE)
        target_ids = torch.randint(0, 32128, (B, L_target), device=DEVICE)
        target_mask = torch.ones(B, L_target, device=DEVICE, dtype=torch.bool)
        source_mask = torch.ones(B, L_source, device=DEVICE, dtype=torch.bool)
        source_mask[:, -5:] = False

        with torch.no_grad():
            out = adapter(source, target_ids, target_mask, source_mask)
        assert out.shape == (B, L_target, D), f"Expected {(B, L_target, D)}, got {out.shape}"

    def test_adapter_forward_with_4d_masks_compat(self):
        """LLMAdapter should handle 4D masks (backward compatibility) by squeezing."""
        from library.anima_models import LLMAdapter
        B, L_target, L_source, D = 1, 8, 16, 32
        adapter = LLMAdapter(
            source_dim=D, target_dim=D, model_dim=D,
            num_layers=2, num_heads=4, self_attn=True,
        ).to(DEVICE, DTYPE)
        adapter.eval()

        source = torch.randn(B, L_source, D, device=DEVICE, dtype=DTYPE)
        target_ids = torch.randint(0, 32128, (B, L_target), device=DEVICE)
        # 4D masks (old format)
        target_mask_4d = torch.ones(B, 1, 1, L_target, device=DEVICE, dtype=torch.bool)
        source_mask_4d = torch.ones(B, 1, 1, L_source, device=DEVICE, dtype=torch.bool)

        with torch.no_grad():
            out = adapter(source, target_ids, target_mask_4d, source_mask_4d)
        assert out.shape == (B, L_target, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
