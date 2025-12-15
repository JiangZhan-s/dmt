import torch
import torch.nn as nn
from .motion import SPyNet, flow_warp
from .compression import MotionCompressor, ResidualCompressor

class VideoCompressor(nn.Module):
    def __init__(self, use_attention=True):
        super(VideoCompressor, self).__init__()
        self.optic_flow = SPyNet()
        self.mv_encoder = MotionCompressor()
        self.res_encoder = ResidualCompressor(use_attention=use_attention)
        
    def forward(self, x_cur, x_ref):
        """
        x_cur: Current frame [B, 3, H, W]
        x_ref: Reference frame (reconstructed) [B, 3, H, W]
        """
        # 1. Motion Estimation
        # Estimate flow from x_cur to x_ref
        flow = self.optic_flow(x_cur, x_ref)
        
        # 2. Motion Compression
        flow_hat, mv_likelihoods = self.mv_encoder(flow)
        
        # 3. Motion Compensation
        # Warp reference frame using compressed flow
        x_pred = flow_warp(x_ref, flow_hat)
        
        # 4. Residual Calculation
        residual = x_cur - x_pred
        
        # 5. Residual Compression
        res_hat, res_likelihoods = self.res_encoder(residual)
        
        # 6. Reconstruction
        x_rec = x_pred + res_hat
        
        return {
            "x_rec": x_rec,
            "x_pred": x_pred,
            "flow_hat": flow_hat,
            "likelihoods": {
                "mv": mv_likelihoods,
                "res": res_likelihoods
            }
        }

    def load_weights(self, path):
        # Placeholder for loading weights
        pass
