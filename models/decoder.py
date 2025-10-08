# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudDecoder(nn.Module):
    """
    Convert latent feature map (B, C, H, W) into point cloud of size (B, num_points, 3)
    Strategy:
      - global pooling (avg+max) -> vector (B, 2C)
      - MLP -> (B, num_points * 3)
      - reshape -> (B, num_points, 3)
    """
    def __init__(self, in_ch: int, num_points: int = 128, hidden: int = 512):
        super().__init__()
        self.num_points = num_points
        self.pool_fc = nn.Sequential(
            nn.Linear(in_ch * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True)
        )
        self.out_fc = nn.Sequential(
            nn.Linear(hidden, num_points * 3)
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: (B, C, H, W)
        returns: (B, num_points, 3)
        """
        B, C, H, W = feat_map.shape
        # global avg and max pool
        avgp = torch.mean(feat_map.view(B, C, -1), dim=-1)  # (B, C)
        maxp, _ = torch.max(feat_map.view(B, C, -1), dim=-1)  # (B, C)
        pooled = torch.cat([avgp, maxp], dim=1)  # (B, 2C)
        hidden = self.pool_fc(pooled)            # (B, hidden)
        out = self.out_fc(hidden)                # (B, num_points*3)
        out = out.view(B, self.num_points, 3)
        return out
