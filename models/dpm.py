import torch
from torch import nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, use_attention=False):
        super().__init__()
        # 时间步嵌入层，用于将时间信息融入网络
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
        
        # 主卷积路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        # 如果输入和输出通道数不一致，需要一个快捷连接
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # 可选的注意力机制
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)

    def forward(self, x, time_emb):
        h = self.conv1(x)
        # 将时间嵌入与特征相加
        time_emb_out = self.time_mlp(self.act1(time_emb))
        h = self.norm1(h) + time_emb_out.view(x.shape[0], -1, 1, 1)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        h = h + self.shortcut(x)
        h = self.act2(h)

        if self.use_attention:
            b, c, h_dim, w_dim = h.shape
            h_flat = h.view(b, c, -1).permute(0, 2, 1) # (B, H*W, C)
            h_attention, _ = self.attention(h_flat, h_flat, h_flat)
            h = h_attention.permute(0, 2, 1).view(b, c, h_dim, w_dim) + h

        return h

# 时间步嵌入函数
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class DPMEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, time_embedding_dim, num_blocks, use_attention_at_downsample):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        for i in range(num_blocks):
            out_channels = current_channels * 2 if i > 0 else current_channels
            self.downs.append(nn.ModuleList([
                ResidualBlock(current_channels, out_channels, time_embedding_dim, use_attention=use_attention_at_downsample),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            ]))
            channels.append(out_channels)
            current_channels = out_channels
        
        self.channels = channels
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, 4 * time_embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embedding_dim, time_embedding_dim),
        )

    def forward(self, x, t):
        time_emb = self.time_embedding(timestep_embedding(t, 256))
        
        h = self.initial_conv(x)
        skips = [h]
        for down_block, downsample in self.downs:
            h = down_block(h, time_emb)
            skips.append(h)
            h = downsample(h)
        
        return h, skips
    

class DPMDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, time_embedding_dim, num_blocks, use_attention_at_upsample):
        super().__init__()
        self.ups = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        for i in range(num_blocks):
            out_channels_up = current_channels // 2 if i < num_blocks - 1 else out_channels
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(current_channels, out_channels_up, kernel_size=2, stride=2),
                ResidualBlock(out_channels_up * 2, out_channels_up, time_embedding_dim, use_attention=use_attention_at_upsample),
            ]))
            channels.append(out_channels_up)
            current_channels = out_channels_up
        
        self.final_conv = nn.Conv2d(out_channels_up, out_channels, kernel_size=1)
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, 4 * time_embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embedding_dim, time_embedding_dim),
        )

    def forward(self, x, skips, t):
        time_emb = self.time_embedding(timestep_embedding(t, 256))
        
        h = x
        for i, (upsample, up_block) in enumerate(self.ups):
            h = upsample(h)
            # 跳跃连接
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, time_emb)
        
        h = self.final_conv(h)
        return h
    


class DiffRadarModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=32, num_blocks=3):
        super().__init__()
        self.time_embedding_dim = 256
        
        # DPM Encoder (U-Net 的下采样部分)
        self.encoder = DPMEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            time_embedding_dim=self.time_embedding_dim,
            num_blocks=num_blocks,
            use_attention_at_downsample=True
        )

        # DPM Decoder (U-Net 的上采样部分)
        self.decoder = DPMDecoder(
            out_channels=out_channels,
            base_channels=self.encoder.channels[-1],
            time_embedding_dim=self.time_embedding_dim,
            num_blocks=num_blocks,
            use_attention_at_upsample=True
        )

    def forward(self, x, t):
        # 这里的x是拼接了条件信息Xc的带噪图像Yt
        # t是时间步
        
        # 1. 编码器将输入压缩为潜在特征，并返回跳跃连接
        latent, skips = self.encoder(x, t)
        
        # 2. 解码器利用潜在特征和跳跃连接重构输出
        output = self.decoder(latent, skips, t)
        
        return output
    

class KeypointDiffusionModel(nn.Module):
    def __init__(self, in_channels, time_embedding_dim, context_embedding_dim):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        
        # 时间步嵌入层
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        # 条件嵌入层
        self.context_embedding = nn.Sequential(
            nn.Linear(context_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )
        
        # PointNet-like 去噪网络
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(256, in_channels, 1), # 最终输出预测的噪声
        )

    def forward(self, x, context, t):
        # x: 带噪声的骨骼点云 (B, 32, 3)
        # context: CE模块生成的条件向量 (B, latent_channels)
        # t: 时间步 (B,)
        
        # 1. 生成时间嵌入
        time_emb = self.time_embedding(timestep_embedding(t, self.time_embedding_dim))
        
        # 2. 将条件嵌入到与时间嵌入相同的维度
        context_emb = self.context_embedding(context)
        
        # 3. 将时间嵌入和条件嵌入相加，并扩展以适应点云数据
        emb = (time_emb + context_emb).unsqueeze(2)
        
        # 4. 将输入转换为 (B, C, N) 格式
        x = x.permute(0, 2, 1)
        
        # 5. 点云去噪网络
        x = self.mlp1(x)
        # 将嵌入向量加入到网络中
        x = x + emb
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        
        # 将输出转换回 (B, N, C)
        x = x.permute(0, 2, 1)
        
        return x