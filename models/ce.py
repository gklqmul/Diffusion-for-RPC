# # ce.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from blocks import MPSBlock, Downsample, Upsample, BasicConv
# from typing import List, Tuple

# class ContourEncoderUNet(nn.Module):
#     """
#     U-Net style Contour Encoder that returns a feature map.
#     Input: (B, C_in, H, W) (C_in usually 1)
#     Output: (B, latent_ch, H, W)  (same spatial size as some decoder level)
#     """
#     def __init__(self, in_ch: int = 1, base_ch: int = 32, latent_ch: int = 128):
#         super().__init__()
#         self.enc1 = MPSBlock(in_ch, base_ch)          # level 1
#         self.down1 = Downsample(base_ch, base_ch*2)   # H/2
#         self.enc2 = MPSBlock(base_ch*2, base_ch*2)
#         self.down2 = Downsample(base_ch*2, base_ch*4) # H/4
#         self.enc3 = MPSBlock(base_ch*4, base_ch*4)
#         self.down3 = Downsample(base_ch*4, base_ch*8) # H/8
#         self.bottleneck = MPSBlock(base_ch*8, base_ch*8)

#         # Decoder
#         self.up3 = Upsample(base_ch*8, base_ch*4)
#         self.dec3 = MPSBlock(base_ch*8, base_ch*4)   # concat skip -> 8 -> 4
#         self.up2 = Upsample(base_ch*4, base_ch*2)
#         self.dec2 = MPSBlock(base_ch*4, base_ch*2)
#         self.up1 = Upsample(base_ch*2, base_ch)
#         self.dec1 = MPSBlock(base_ch*2, base_ch)

#         # final projection to latent channels (keeps spatial)
#         self.out_conv = BasicConv(base_ch, latent_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, C_in, H, W)
#         returns: (B, latent_ch, H, W)  (same H,W as input)
#         """
#         # Encoder
#         e1 = self.enc1(x)        # (B, base_ch, H, W)
#         e2 = self.enc2(self.down1(e1))   # (B, base_ch*2, H/2, W/2)
#         e3 = self.enc3(self.down2(e2))   # (B, base_ch*4, H/4, W/4)
#         b = self.bottleneck(self.down3(e3))  # (B, base_ch*8, H/8, W/8)

#         # Decoder with skip connections
#         d3 = self.up3(b)                       # (B, base_ch*4, H/4, W/4)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))
#         d2 = self.up2(d3)                      # (B, base_ch*2, H/2, W/2)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))
#         d1 = self.up1(d2)                      # (B, base_ch, H, W)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))

#         out = self.out_conv(d1)        # (B, latent_ch, H, W)
#         out = torch.sigmoid(out)       # Sigmoid

#         # Hadamard product with input (broadcasting if needed)
#         out = out * x                  # (B, latent_ch, H, W)

#         # Concatenate original input along channel dim
#         out = torch.cat([out, x], dim=1)  # (B, latent_ch + C_in, H, W)

#         return out

import torch
from torch import nn
import torch.nn.functional as F

class PointCE(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        # PointNet-like 结构
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # 将全局特征降维到所需的 latent_channels
        self.fc = nn.Sequential(
            nn.Linear(1024, latent_channels),
            nn.BatchNorm1d(latent_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # 输入 x 的形状是 (B, N, C)，需要转换为 (B, C, N) 以适应 Conv1d
        x = x.permute(0, 2, 1)
        
        # 逐点特征提取
        x = self.mlp1(x)
        
        # 全局特征提取
        x = self.mlp2(x)
        
        # 最大池化，得到全局特征向量 (B, 1024, 1) -> (B, 1024)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # 降维并输出条件向量
        x = self.fc(x)
        return x