# blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BasicConv(nn.Module):
    """Basic Conv2d -> BN -> ReLU block"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
       

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    """
    Spatial attention: channel pooling (avg & max) -> conv -> sigmoid -> scale
    Works on (B, C, H, W) -> returns scaled (B, C, H, W)
    """
    def __init__(self, in_ch, out_ch,kernel_size: int = 3):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.basic = BasicConv(in_ch, out_ch, kernel_size=3, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)      # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # (B,1,H,W)
        cat = torch.cat([avg_out, max_out], dim=1)        # (B,2,H,W)
        y = self.basic(cat)                      # (B,1,H,W)
        y = self.relu(y)
        return x * y

class MPSBlock(nn.Module):
    """
    Multi-Path Scale Block:
    - parallel conv3x3 and conv5x5 (BasicConv)
    - fuse by addition
    - spatial attention
    - final basic conv fuse
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv3 = BasicConv(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv5 = BasicConv(in_ch, out_ch, kernel_size=5, padding=2)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.fuse = BasicConv(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv3(x)
        out2 = self.conv5(x)
        out3 = out1 + out2
        out4 = self.relu(out3)
        out = self.spatial_attn(out4)
        out = out3 * out4
        out = self.fuse(out)
        out = self.relu(out)
        return out

class Downsample(nn.Module):
    """Downsample via conv stride=2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = BasicConv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Upsample(nn.Module):
    """Upsample via ConvTranspose2d"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))
