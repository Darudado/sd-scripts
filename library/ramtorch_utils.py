import torch
import torch.nn as nn
from library.utils import setup_logging

try:
    from ramtorch.modules.linear import Linear as RamTorchLinear
    RAMTORCH_AVAILABLE = True
except ImportError:
    RAMTORCH_AVAILABLE = False

setup_logging()
import logging

logger = logging.getLogger(__name__)

def replace_linear_with_ramtorch_linear(module: nn.Module, device="cuda", recursive: bool = True):
    """
    Recursively replaces all `torch.nn.Linear` layers with `ramtorch.modules.Linear`.

    Args:
        module (nn.Module): The module to modify.
        device (str): The target device for computation ('cuda', 'cpu', etc.).
        recursive (bool): Whether to recurse into submodules.
    """
    if not RAMTORCH_AVAILABLE:
        logger.error("RamTorch is not installed. Please install it with 'pip install ramtorch'")
        raise ImportError("RamTorch is not installed.")

    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Create a new RamTorch Linear layer with the same parameters
            ramtorch_linear = RamTorchLinear(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                bias=child_module.bias is not None,
                device=device,
            )

            # Copy weights and bias from the original layer
            ramtorch_linear.weight.data.copy_(child_module.weight.data.to("cpu"))
            if child_module.bias is not None:
                ramtorch_linear.bias.data.copy_(child_module.bias.data.to("cpu"))
            
            # Replace the original layer
            setattr(module, name, ramtorch_linear)
            logger.info(f"Replaced {name} with RamTorch Linear layer.")
        elif recursive and len(list(child_module.children())) > 0:
            replace_linear_with_ramtorch_linear(child_module, device, recursive=True)

def apply_ramtorch(args, unet, text_encoders, accelerator):
    # Apply ramtorch
    if args.use_ramtorch:
        try:
            from library.ramtorch_utils import replace_linear_with_ramtorch_linear
            logger.info("Applying RamTorch to U-Net and Text Encoders for memory efficiency...")
            replace_linear_with_ramtorch_linear(unet, accelerator.device)
            for text_encoder in text_encoders:
                replace_linear_with_ramtorch_linear(text_encoder, accelerator.device)
            logger.info("RamTorch applied successfully.")
        except ImportError as e:
            logger.error(f"Failed to apply RamTorch: {e}")