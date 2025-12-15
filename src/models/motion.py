import torch
import torch.nn as nn
import torch.nn.functional as F

def flow_warp(x, flow):
    """
    Warp an image or feature map with optical flow.
    x: [B, C, H, W] (image)
    flow: [B, 2, H, W] (flow field)
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    
    vgrid = grid + flow
    
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super(SPyNetBasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

class SPyNet(nn.Module):
    """
    SPyNet: Spatial Pyramid Network for Optical Flow Estimation
    """
    def __init__(self, pretrained=None):
        super(SPyNet, self).__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        if pretrained:
            # Load pretrained weights logic here
            pass

    def forward(self, ref, supp):
        """
        Estimate flow from ref to supp
        """
        B, C, H, W = ref.size()
        
        # Downsample images for pyramid
        ref_pyramid = [ref]
        supp_pyramid = [supp]
        for i in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], 2, stride=2))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], 2, stride=2))
            
        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]

        flow = None

        for i in range(6):
            if flow is None:
                B, _, H_new, W_new = ref_pyramid[i].size()
                flow = torch.zeros(B, 2, H_new, W_new).type_as(ref)
            else:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            
            warped_supp = flow_warp(supp_pyramid[i], flow)
            flow_res = self.basic_module[i](torch.cat([ref_pyramid[i], warped_supp, flow], 1))
            flow = flow + flow_res
            
        return flow
