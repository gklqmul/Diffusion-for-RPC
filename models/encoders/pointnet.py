import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
    


class TransformerEncoder(nn.Module):
    """
    基于 Transformer 的编码器，用于将骨骼点云 (B, N, 3) 编码为
    潜在向量 z (均值 m 和对数方差 v)。
    """
    def __init__(self, zdim, input_dim=3, model_dim=256, num_heads=4, num_layers=3):
        super().__init__()
        self.zdim = zdim
        self.model_dim = model_dim
        
        # 1. 初始特征投影: 3D 坐标 -> Transformer 维度 (model_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(model_dim)
        )

        # 2. Transformer 编码器层
        # N=32 个点作为 N 个 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4, 
            dropout=0.1, 
            batch_first=True  # 确保输入格式是 (B, N, C)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 全局特征聚合 (Set Pooling)
        # 替换 PointNet 的 Max Pooling: 使用线性层将所有点特征聚合为全局特征
        # 聚合后特征维度与 model_dim 相同
        self.global_pool = nn.AdaptiveMaxPool1d(1) 
        
        # 4. 潜在向量映射 (Mapping to z)
        # 将全局特征 (model_dim) 映射到潜在空间 zdim (均值 m)
        self.fc_mean = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, zdim)
        )

        # 5. 潜在向量映射 (Mapping to z)
        # 将全局特征 (model_dim) 映射到潜在空间 zdim (对数方差 v)
        self.fc_logvar = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(model_dim // 2),
            nn.Linear(model_dim // 2, zdim)
        )

    def forward(self, x):
        """
        Args:
            x: 骨骼点云 (B, N, input_dim=3)
        Returns:
            m: 潜在向量均值 (B, zdim)
            v: 潜在向量对数方差 (B, zdim)
        """
        B, N, C = x.shape
        
        # 1. 初始投影: (B, N, 3) -> (B, N, model_dim)
        feat = self.input_proj(x)
        
        # 2. Transformer 编码: 学习全局依赖
        # (B, N, model_dim) -> (B, N, model_dim)
        att_output = self.transformer_encoder(feat)
        
        # 3. 全局池化/聚合: 将 N 个点的特征压缩为 1 个全局向量
        # (B, N, model_dim) -> (B, model_dim, N)
        # x_transposed = att_output.transpose(1, 2)
        # Max Pool over the N points: (B, model_dim, N) -> (B, model_dim, 1)
        global_feat = torch.mean(att_output, dim=1)  # (B, model_dim)

        # 4. 映射到 VAE 潜在空间
        m = self.fc_mean(global_feat)    # (B, zdim)
        v = self.fc_logvar(global_feat)  # (B, zdim)

        # 返回均值和对数方差
        return (m, v)

