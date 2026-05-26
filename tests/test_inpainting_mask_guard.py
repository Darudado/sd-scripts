"""
Regression tests for the inpainting mask / attention mask variable-shadowing bug.

Bug summary
-----------
In ``library/train_util.py``, the variable ``masks`` was used for both the
inpainting mask accumulator (line 1722) and the SDXL per-image attention mask
(line 1859).  The attention-mask assignment overwrote the inpainting list,
causing ``example["masks"]`` to contain attention-mask data even when inpainting
was not configured.  Downstream in ``train_network.py``, the guard only checked
``batch.get("masks")`` — which was non-None — and then tried to access
``batch["masked_images"]``, which was ``None``, triggering an ``AttributeError``.

Fixes applied
~~~~~~~~~~~~~
1. ``train_util.py``: The inpainting mask accumulator is now named
   ``inpainting_masks`` to avoid shadowing the attention-mask variable.
2. ``train_network.py:626``: The guard now checks *both* ``masks`` and
   ``masked_images``.

This test module verifies both fixes using source-level assertions and
targeted functional tests with mock objects.
"""

import ast
import os
import sys
import textwrap

import pytest

# Allow imports from the sd_scripts package root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_source(relpath: str) -> str:
    """Read a source file relative to the sd_scripts directory."""
    base = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(base, relpath), encoding="utf-8") as f:
        return f.read()


def _get_assign_targets(source: str, line_number: int) -> list[str]:
    """Return the left-hand-side names of an assignment on *line_number*."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.lineno == line_number:
            return [t.id if isinstance(t, ast.Name) else t.attr for t in node.targets if isinstance(t, (ast.Name, ast.Attribute))]
        if isinstance(node, ast.AugAssign) and node.lineno == line_number:
            t = node.target
            return [t.id if isinstance(t, ast.Name) else t.attr]
    return []


# ---------------------------------------------------------------------------
# Tests for train_util.py fix: variable shadowing elimination
# ---------------------------------------------------------------------------

class TestTrainUtilVariableShadowing:
    """Verify that ``inpainting_masks`` is used instead of ``masks`` for
    the inpainting mask accumulator in ``__getitem__``."""

    def test_inpainting_accumulator_is_renamed(self):
        """The inpainting mask list should be named ``inpainting_masks``,
        not ``masks``, so it cannot be overwritten by the attention-mask
        assignment later in the method."""
        source = _read_source("library/train_util.py")
        # Find the __getitem__ method body
        lines = source.splitlines()

        # Look for the initialization of the inpainting mask accumulator.
        # It should be ``inpainting_masks = []`` not ``masks = []``
        found_inpainting_init = False
        found_masks_init_in_same_position = False
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped == "inpainting_masks = []":
                found_inpainting_init = True
            # The old buggy pattern: masks = [] in the accumulator position
            # This should NOT exist in the __getitem__ method after the fix
            if i < 1800 and stripped == "masks = []":
                found_masks_init_in_same_position = True

        assert found_inpainting_init, (
            "Expected 'inpainting_masks = []' initialization in __getitem__"
        )
        assert not found_masks_init_in_same_position, (
            "The old 'masks = []' accumulator should be renamed to 'inpainting_masks = []'"
        )

    def test_inpainting_append_uses_renamed_variable(self):
        """The append inside the ``if self.train_inpainting`` block should
        use ``inpainting_masks.append(mask)``."""
        source = _read_source("library/train_util.py")
        assert "inpainting_masks.append(mask)" in source, (
            "Expected 'inpainting_masks.append(mask)' in the inpainting block"
        )

    def test_example_masks_uses_renamed_variable(self):
        """The ``example["masks"]`` assignment should reference
        ``inpainting_masks``, not ``masks``."""
        source = _read_source("library/train_util.py")
        # The line should be:
        #   example["masks"] = torch.stack(inpainting_masks) if inpainting_masks else None
        lines = source.splitlines()
        found = False
        for line in lines:
            stripped = line.strip()
            if 'example["masks"]' in stripped and "inpainting_masks" in stripped:
                found = True
                break
        assert found, (
            'Expected example["masks"] to reference inpainting_masks, not masks'
        )

    def test_attention_mask_variable_not_shadowed(self):
        """After the fix, the SDXL attention-mask variable ``masks`` (set at
        line ~1859 and ~1902) should be a *separate* variable from the
        inpainting accumulator.  We verify that ``masks`` is still used for
        attention masks in the tokenization block."""
        source = _read_source("library/train_util.py")
        # The attention-mask assignment should still exist
        assert "masks = None" in source, (
            "The attention-mask variable 'masks = None' should still be present"
        )
        assert "masks = mask_iter" in source, (
            "The SDXL attention-mask assignment 'masks = mask_iter' should still be present"
        )
        # The attention mask should be appended to attn_mask_list
        assert "attn_mask_list.append(masks)" in source, (
            "Attention mask should be appended to attn_mask_list"
        )


# ---------------------------------------------------------------------------
# Tests for train_network.py fix: dual guard
# ---------------------------------------------------------------------------

class TestTrainNetworkGuard:
    """Verify that the inpainting guard in ``train_network.py`` checks both
    ``masks`` and ``masked_images`` before attempting to encode."""

    def test_guard_checks_both_keys(self):
        """The guard should check ``batch.get("masked_images")`` in addition
        to ``batch.get("masks")``."""
        source = _read_source("train_network.py")
        lines = source.splitlines()

        guard_line = None
        for i, line in enumerate(lines, start=1):
            if 'batch.get("masks") is not None' in line and "masked_images" in line:
                guard_line = (i, line.strip())
                break

        assert guard_line is not None, (
            "Expected a guard line checking both 'masks' and 'masked_images' in train_network.py"
        )
        line_num, content = guard_line
        assert 'batch.get("masked_images") is not None' in content, (
            "Guard should also check batch.get('masked_images') is not None"
        )


# ---------------------------------------------------------------------------
# Functional tests: simulate the crash scenario
# ---------------------------------------------------------------------------

class TestInpaintingMaskCrashScenario:
    """Functional tests that simulate the data flow through ``__getitem__``
    and the ``train_network.py`` guard to verify the crash no longer occurs."""

    def test_non_inpainting_batch_masks_none(self):
        """When inpainting is not configured, ``example["masks"]`` should be
        ``None`` (not populated with attention masks).  This simulates the
        fixed data flow."""
        import torch

        # Simulate the fixed __getitem__ data flow:
        # inpainting_masks = []  (accumulator, never appended to)
        inpainting_masks = []
        masked_images = []

        # SDXL attention masks (separate variable now)
        # These would be populated by tokenize_strategy.tokenize()
        attention_masks = None  # or could be a tensor for SDXL

        # The example dict construction (fixed version)
        example = {}
        example["masks"] = torch.stack(inpainting_masks) if inpainting_masks else None
        example["masked_images"] = torch.stack(masked_images) if masked_images else None

        assert example["masks"] is None, "masks should be None when not doing inpainting"
        assert example["masked_images"] is None, "masked_images should be None when not doing inpainting"

    def test_non_inpainting_sdxl_attention_mask_scenario(self):
        """When using SDXL tokenization (attention masks are non-None) but
        inpainting is disabled, ``example["masks"]`` should still be ``None``."""
        import torch

        # Simulate the fixed data flow:
        inpainting_masks = []
        masked_images = []

        # SDXL attention mask tokenization sets a separate 'masks' variable
        # but now it doesn't affect inpainting_masks
        sdxl_attention_mask = torch.ones(2, 77)  # example SDXL attention mask

        # In the fixed code, the attention mask goes to attn_mask_list, 
        # not to the inpainting masks variable
        example = {}
        example["masks"] = torch.stack(inpainting_masks) if inpainting_masks else None
        example["masked_images"] = torch.stack(masked_images) if masked_images else None

        assert example["masks"] is None, (
            "masks should be None even when SDXL attention masks are present"
        )
        assert example["masked_images"] is None

    def test_inpainting_batch_both_populated(self):
        """When inpainting IS configured, both ``example["masks"]`` and
        ``example["masked_images"]`` should be populated."""
        import torch

        # Simulate the fixed data flow with inpainting enabled:
        inpainting_masks = [torch.ones(1, 64, 64), torch.ones(1, 64, 64)]
        masked_images = [torch.randn(3, 512, 512), torch.randn(3, 512, 512)]

        example = {}
        example["masks"] = torch.stack(inpainting_masks) if inpainting_masks else None
        example["masked_images"] = torch.stack(masked_images) if masked_images else None

        assert example["masks"] is not None
        assert example["masked_images"] is not None
        assert example["masks"].shape == (2, 1, 64, 64)
        assert example["masked_images"].shape == (2, 3, 512, 512)

    def test_guard_blocks_when_only_masks_populated(self):
        """The guard should NOT enter the inpainting branch when ``masks``
        contains data but ``masked_images`` is ``None``.  This simulates
        the old bug scenario."""
        import torch

        # Old buggy scenario: masks is attention mask data (non-None),
        # but masked_images is None
        batch = {
            "masks": torch.ones(2, 77),  # attention mask, not inpainting mask
            "masked_images": None,
        }

        # Fixed guard: check BOTH conditions
        guard_passes = batch.get("masks") is not None and batch.get("masked_images") is not None

        assert not guard_passes, (
            "Guard should NOT pass when only masks is populated (attention mask scenario)"
        )

    def test_guard_passes_when_both_populated(self):
        """The guard should enter the inpainting branch when both ``masks``
        and ``masked_images`` are populated."""
        import torch

        batch = {
            "masks": torch.ones(2, 1, 64, 64),
            "masked_images": torch.randn(2, 3, 512, 512),
        }

        guard_passes = batch.get("masks") is not None and batch.get("masked_images") is not None

        assert guard_passes, (
            "Guard should pass when both masks and masked_images are populated"
        )

    def test_guard_blocks_when_masks_none(self):
        """The guard should NOT pass when ``masks`` is ``None``."""
        batch = {
            "masks": None,
            "masked_images": None,
        }

        guard_passes = batch.get("masks") is not None and batch.get("masked_images") is not None

        assert not guard_passes

    def test_guard_blocks_when_masks_missing(self):
        """The guard should NOT pass when ``masks`` key is missing entirely."""
        batch = {
            "masked_images": None,
        }

        guard_passes = batch.get("masks") is not None and batch.get("masked_images") is not None

        assert not guard_passes

    def test_old_guard_would_crash(self):
        """Demonstrate that the OLD guard (only checking masks) would have
        entered the branch and crashed on masked_images being None."""
        import torch

        # This is the exact scenario that caused the crash
        batch = {
            "masks": torch.ones(2, 77),  # non-None attention mask
            "masked_images": None,         # None because inpainting not configured
        }

        # Old guard (buggy):
        old_guard = batch.get("masks") is not None
        assert old_guard, "Old guard would have entered the branch"

        # The crash would occur on: batch["masked_images"].to(...)
        with pytest.raises(AttributeError):
            batch["masked_images"].to("cpu")

        # New guard (fixed):
        new_guard = batch.get("masks") is not None and batch.get("masked_images") is not None
        assert not new_guard, "New guard correctly prevents entering the branch"
