import torch
import torch.nn.functional as F

def gaussian_kernel_2d(size=5, sigma=1.0):
    coords = torch.arange(size) - size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def mse_pyramid_loss_2d(pred, target, levels=5):
    kernel = gaussian_kernel_2d(size=5, sigma=1).repeat(3, 1, 1, 1).to(pred.device)
    
    levels_pred = []
    levels_target = []
    
    working_pred = pred.to(torch.float64)
    working_target = target.to(torch.float64)
    
    for i in range(levels - 1):
        blurred_pred = F.conv2d(working_pred, weight=kernel, padding="same", groups=3)
        blurred_target = F.conv2d(working_target, weight=kernel, padding="same", groups=3)
        
        levels_pred.append(working_pred - blurred_pred)
        levels_target.append(working_target - blurred_target)
        
        working_pred = F.interpolate(blurred_pred, scale_factor=0.5, mode="area")
        working_target = F.interpolate(blurred_target, scale_factor=0.5, mode="area")
    
    levels_pred.append(working_pred)
    levels_target.append(working_target)
    
    loss = torch.zeros(pred.shape[0], device=pred.device)
    
    for l_pred, l_target in zip(levels_pred, levels_target):
        l_loss = F.mse_loss(l_pred, l_target, reduction="none").mean(dim=(1, 2, 3))
        loss += l_loss / l_target.std(dim=(1, 2, 3))
    
    return loss / levels # shape = [B]