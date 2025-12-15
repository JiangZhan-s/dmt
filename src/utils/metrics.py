import torch
import torch.nn.functional as F
from math import log10

def psnr(a, b):
    mse = F.mse_loss(a, b)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / mse.sqrt())

# MS-SSIM implementation can be complex, usually we use pytorch_msssim or compressai's implementation
# Here is a placeholder or wrapper if compressai is installed
try:
    from pytorch_msssim import ms_ssim
    def compute_msssim(a, b):
        return ms_ssim(a, b, data_range=1.0, size_average=True)
except ImportError:
    def compute_msssim(a, b):
        return 0.0 # Placeholder
