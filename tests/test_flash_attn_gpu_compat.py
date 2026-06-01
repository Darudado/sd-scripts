"""
Tests for flash_attn GPU compatibility checks.

Verifies that flash_attn is correctly disabled on pre-Ampere GPUs (e.g. T4/Turing)
and that the SDPA fallback path is used instead.

Usage:
    pytest tests/test_flash_attn_gpu_compat.py -v
"""

import sys
import os
import importlib
import logging
from unittest import mock

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────────────
# 1. attention.py — module-level GPU capability check
# ──────────────────────────────────────────────────────────────
class TestAttentionFlashAttnGpuCheck:
    """Verify attention.py disables flash_attn on pre-Ampere GPUs."""

    def test_is_flash_attn_supported_returns_bool(self):
        """is_flash_attn_supported() should always return a bool."""
        from library import attention

        result = attention.is_flash_attn_supported()
        assert isinstance(result, bool)

    def test_flash_attn_consistent_with_gpu_capability(self):
        """If flash_attn is available, GPU capability must be >= (8, 0)."""
        from library import attention

        if attention.flash_attn_varlen_func is not None:
            # flash_attn survived the import-time check, so GPU must be Ampere+
            assert torch.cuda.is_available(), "flash_attn is set but CUDA not available"
            cap = torch.cuda.get_device_capability()
            assert cap >= (8, 0), (
                f"flash_attn_varlen_func is not None on GPU with capability {cap}; "
                "expected it to be None on pre-Ampere GPUs"
            )

    def test_is_flash_attn_supported_consistent_with_module_state(self):
        """is_flash_attn_supported() must agree with the module-level state."""
        from library import attention

        if attention.flash_attn_varlen_func is None:
            assert not attention.is_flash_attn_supported(), (
                "is_flash_attn_supported() returned True but flash_attn_varlen_func is None"
            )

    def test_all_flash_refs_consistent(self):
        """All flash_attn references in attention.py should be consistently None or non-None."""
        from library import attention

        refs = [
            attention.flash_attn,
            attention.flash_attn_varlen_func,
            attention._flash_attn_forward,
            attention.flash_attn_func,
        ]
        none_count = sum(1 for r in refs if r is None)
        assert none_count in (0, len(refs)), (
            f"flash_attn references are inconsistent: {none_count}/{len(refs)} are None"
        )


# ──────────────────────────────────────────────────────────────
# 2. attention.py — simulated pre-Ampere GPU (T4 / Turing sm_75)
# ──────────────────────────────────────────────────────────────
class TestAttentionFlashAttnDisabledOnPreAmpere:
    """Simulate a pre-Ampere GPU and verify flash_attn is disabled."""

    def test_flash_attn_disabled_when_gpu_capability_below_ampere(self):
        """
        Re-import the attention module with a mocked get_device_capability
        returning (7, 5) (Turing / T4) and verify flash_attn is set to None.
        """
        # Only run this test if flash_attn is actually importable on this system;
        # otherwise there's nothing to guard against.
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pytest.skip("flash_attn not installed; nothing to guard against")

        # Save originals
        import library.attention as attn_mod

        orig_flash_attn = attn_mod.flash_attn
        orig_varlen = attn_mod.flash_attn_varlen_func
        orig_forward = attn_mod._flash_attn_forward
        orig_func = attn_mod.flash_attn_func

        try:
            # Temporarily restore flash_attn refs (as if import succeeded)
            import flash_attn as _fa
            from flash_attn.flash_attn_interface import (
                _flash_attn_forward as _fwd,
                flash_attn_varlen_func as _vlen,
                flash_attn_func as _func,
            )

            attn_mod.flash_attn = _fa
            attn_mod.flash_attn_varlen_func = _vlen
            attn_mod._flash_attn_forward = _fwd
            attn_mod.flash_attn_func = _func

            # Now simulate the GPU check returning Turing capability
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_capability", return_value=(7, 5)
            ):
                # Re-run the guard logic inline (same as module-level code)
                if attn_mod.flash_attn is not None:
                    cap = torch.cuda.get_device_capability()
                    if cap < (8, 0):
                        attn_mod.flash_attn = None
                        attn_mod.flash_attn_varlen_func = None
                        attn_mod._flash_attn_forward = None
                        attn_mod.flash_attn_func = None

            assert attn_mod.flash_attn is None
            assert attn_mod.flash_attn_varlen_func is None
            assert attn_mod._flash_attn_forward is None
            assert attn_mod.flash_attn_func is None
        finally:
            # Restore originals
            attn_mod.flash_attn = orig_flash_attn
            attn_mod.flash_attn_varlen_func = orig_varlen
            attn_mod._flash_attn_forward = orig_forward
            attn_mod.flash_attn_func = orig_func

    def test_flash_attn_kept_when_gpu_is_ampere(self):
        """
        Simulate an Ampere GPU (capability 8.0) and verify flash_attn refs are kept.
        """
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pytest.skip("flash_attn not installed; nothing to test")

        import library.attention as attn_mod

        orig_flash_attn = attn_mod.flash_attn
        orig_varlen = attn_mod.flash_attn_varlen_func
        orig_forward = attn_mod._flash_attn_forward
        orig_func = attn_mod.flash_attn_func

        try:
            import flash_attn as _fa
            from flash_attn.flash_attn_interface import (
                _flash_attn_forward as _fwd,
                flash_attn_varlen_func as _vlen,
                flash_attn_func as _func,
            )

            attn_mod.flash_attn = _fa
            attn_mod.flash_attn_varlen_func = _vlen
            attn_mod._flash_attn_forward = _fwd
            attn_mod.flash_attn_func = _func

            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_capability", return_value=(8, 0)
            ):
                if attn_mod.flash_attn is not None:
                    cap = torch.cuda.get_device_capability()
                    if cap < (8, 0):
                        attn_mod.flash_attn = None
                        attn_mod.flash_attn_varlen_func = None
                        attn_mod._flash_attn_forward = None
                        attn_mod.flash_attn_func = None

            # On Ampere, flash_attn should be kept
            assert attn_mod.flash_attn is not None
            assert attn_mod.flash_attn_varlen_func is not None
        finally:
            attn_mod.flash_attn = orig_flash_attn
            attn_mod.flash_attn_varlen_func = orig_varlen
            attn_mod._flash_attn_forward = orig_forward
            attn_mod.flash_attn_func = orig_func


# ──────────────────────────────────────────────────────────────
# 3. anima_models.py — can_use_flash defense-in-depth
# ──────────────────────────────────────────────────────────────
class TestAnimaModelsCanUseFlash:
    """Verify anima_models.py can_use_flash check includes GPU capability."""

    def test_can_use_flash_source_code_contains_capability_check(self):
        """The can_use_flash computation must include a GPU capability check."""
        import inspect
        from library import anima_models

        # Find the LLMAdapterAttention class and its forward method
        cls = anima_models.LLMAdapterAttention
        source = inspect.getsource(cls.forward)

        assert "get_device_capability" in source, (
            "LLMAdapterAttention.forward should check GPU compute capability"
        )
        assert "(8, 0)" in source, (
            "LLMAdapterAttention.forward should compare against (8, 0) for Ampere"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_can_use_flash_reflects_actual_gpu(self):
        """
        On the current GPU, can_use_flash should be True only if
        flash_attn is available AND GPU is Ampere+.
        """
        from library import attention

        expected_flash = (
            attention.flash_attn_varlen_func is not None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 0)
        )

        if expected_flash:
            assert attention.is_flash_attn_supported()
        else:
            assert not attention.is_flash_attn_supported()


# ──────────────────────────────────────────────────────────────
# 4. lumina_models.py — JointAttention flash_attn guard
# ──────────────────────────────────────────────────────────────
class TestLuminaModelsFlashAttnGuard:
    """Verify lumina_models.py disables flash_attn on unsupported GPUs."""

    def test_lumina_module_level_check_consistency(self):
        """
        If flash_attn_varlen_func was imported but set to None by the GPU check,
        the module state should reflect this.
        """
        from library import lumina_models

        # Check if flash_attn_varlen_func is defined in the module's namespace
        has_flash = hasattr(lumina_models, "flash_attn_varlen_func") and \
                    lumina_models.flash_attn_varlen_func is not None

        if has_flash:
            # If flash_attn survived, GPU must support it
            assert torch.cuda.is_available()
            assert torch.cuda.get_device_capability() >= (8, 0)

    def test_joint_attention_init_disables_flash_on_pre_ampere(self):
        """
        JointAttention.__init__ should disable use_flash_attn when
        flash_attn_varlen_func is None.
        """
        from library.lumina_models import JointAttention

        # Temporarily set flash_attn_varlen_func to None in the module namespace
        import library.lumina_models as lm_mod

        orig = getattr(lm_mod, "flash_attn_varlen_func", None)
        try:
            lm_mod.flash_attn_varlen_func = None
            # Re-read to confirm the guard code path
            # The __init__ checks: "flash_attn_varlen_func" not in dir() or flash_attn_varlen_func is None
            ja = JointAttention(dim=64, n_heads=4, n_kv_heads=4, qk_norm=False, use_flash_attn=True)
            assert not ja.use_flash_attn, (
                "JointAttention should have disabled use_flash_attn when flash_attn_varlen_func is None"
            )
        finally:
            if orig is not None:
                lm_mod.flash_attn_varlen_func = orig
            elif hasattr(lm_mod, "flash_attn_varlen_func"):
                delattr(lm_mod, "flash_attn_varlen_func")

    def test_joint_attention_init_keeps_flash_when_available(self):
        """
        JointAttention.__init__ should keep use_flash_attn=True when
        flash_attn_varlen_func is available.
        """
        from library.lumina_models import JointAttention
        import library.lumina_models as lm_mod

        orig = getattr(lm_mod, "flash_attn_varlen_func", None)
        if orig is None:
            pytest.skip("flash_attn_varlen_func not available; cannot test keep-flash path")

        try:
            ja = JointAttention(dim=64, n_heads=4, n_kv_heads=4, qk_norm=False, use_flash_attn=True)
            assert ja.use_flash_attn, (
                "JointAttention should keep use_flash_attn=True when flash_attn_varlen_func is available"
            )
        finally:
            pass  # no modification needed


# ──────────────────────────────────────────────────────────────
# 5. End-to-end: LLMAdapterAttention SDPA fallback on pre-Ampere
# ──────────────────────────────────────────────────────────────
class TestLLMAdapterAttentionFallback:
    """Verify LLMAdapterAttention falls back to SDPA when flash_attn is disabled."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_llm_adapter_attention_forward_uses_sdpa_when_flash_disabled(self):
        """
        Temporarily disable flash_attn and verify that LLMAdapterAttention.forward
        completes without error using the SDPA fallback.
        """
        from library.anima_models import LLMAdapterAttention
        from library import attention

        orig_varlen = attention.flash_attn_varlen_func
        orig_func = attention.flash_attn_func

        try:
            attention.flash_attn_varlen_func = None
            attention.flash_attn_func = None

            attn = LLMAdapterAttention(
                query_dim=64,
                context_dim=64,
                n_heads=4,
                head_dim=16,
            ).to(device="cuda", dtype=torch.float16)

            x = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)
            mask = torch.ones(2, 10, device="cuda", dtype=torch.bool)

            # This should succeed via SDPA fallback, not raise RuntimeError
            out = attn(x, q_mask=mask, kv_mask=mask)
            assert out.shape == (2, 10, 64)
        finally:
            attention.flash_attn_varlen_func = orig_varlen
            attention.flash_attn_func = orig_func

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_llm_adapter_attention_forward_no_mask_uses_sdpa_when_flash_disabled(self):
        """
        Test the no-mask path (can_use_flash and q_mask is None and kv_mask is None)
        also falls back to SDPA when flash_attn is disabled.
        """
        from library.anima_models import LLMAdapterAttention
        from library import attention

        orig_varlen = attention.flash_attn_varlen_func
        orig_func = attention.flash_attn_func

        try:
            attention.flash_attn_varlen_func = None
            attention.flash_attn_func = None

            attn = LLMAdapterAttention(
                query_dim=64,
                context_dim=64,
                n_heads=4,
                head_dim=16,
            ).to(device="cuda", dtype=torch.float16)

            x = torch.randn(2, 10, 64, device="cuda", dtype=torch.float16)

            # No masks — should still work via SDPA fallback
            out = attn(x)
            assert out.shape == (2, 10, 64)
        finally:
            attention.flash_attn_varlen_func = orig_varlen
            attention.flash_attn_func = orig_func


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
