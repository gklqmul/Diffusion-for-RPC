import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import torch_geometric.nn as PyG
import numpy as np
from torch_scatter import scatter_max

from .common import *


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, cond_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        total_ctx_dim = context_dim + cond_dim + 3  # 原 context + 外部条件 + time emb
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, total_ctx_dim),
            ConcatSquashLinear(128, 256, total_ctx_dim),
            ConcatSquashLinear(256, 512, total_ctx_dim),
            ConcatSquashLinear(512, 256, total_ctx_dim),
            ConcatSquashLinear(256, 128, total_ctx_dim),
            ConcatSquashLinear(128, 3, total_ctx_dim)
        ])

    def forward(self, x, beta, context, cond):
        """
        Args:
            x:      (B,N,3)
            beta:   (B,)
            context:(B,F1) 原 context
            cond:   (B,F2) 外部条件
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)         # (B,1,1)
        context = context.view(batch_size, 1, -1)  # (B,1,F1)
        cond = cond.view(batch_size, 1, -1)        # (B,1,F2)

        # 拼接 time + context + cond
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B,1,3)
        ctx_emb = torch.cat([time_emb, context, cond], dim=-1)                   # (B,1,F1+F2+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers)-1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.action_encoder = ActionConditionEmbedding(num_classes=21, embed_dim=128)
        self.radar_encoder = DGCNNEncoder(in_dim=5, out_dim=256)
        self.cond_fusion = ConditionalFusion(radar_dim=256, action_dim=128, out_dim=512)

    def get_loss(self, x_0, context, radar_cond, action_cond, t=None): #forward
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        radar_feat = self.radar_encoder(radar_cond)         # (B, D1)
        action_feat = self.action_encoder(action_cond)      # (B, D2)
        cond = self.cond_fusion(radar_feat, action_feat)           # (B, C)
        batch_size, _, point_dim = x_0.size()

        # sample t randomly
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        # alpha_bar means 累积的衰减量, beta means 当前的衰减量
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        # c0 control original data percentage, c1 control noise percentage
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        # random noise from standard Gaussian
        e_rand = torch.randn_like(x_0)  # (B, N, d)
        cond_empty = torch.zeros_like(cond)
        # add noise
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context, cond=cond_empty)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, radar_cond, action_cond, point_dim=3, flexibility=0.0, ret_traj=False):
        """
        Conditional diffusion sampling.
        Args:
            num_points: number of points to generate
            context: original context (noisy target, for compatibility)
            cond: external condition (radar+label embedding), shape (B, C)
            point_dim: dimensionality of generated points
            flexibility: scaling for added Gaussian noise (controls stochasticity)
            ret_traj: return full trajectory if True
        """
        batch_size = context.size(0)
        device = context.device
        radar_feat = self.radar_encoder(radar_cond)         # (B, D1)
        action_feat = self.action_encoder(action_cond)      # (B, D2)
        cond = self.cond_fusion(radar_feat, action_feat)           # (B, C)
        # start from pure Gaussian noise
        x_T = torch.randn([batch_size, num_points, point_dim], device=device)
        traj = {self.var_sched.num_steps: x_T}


        for t in range(self.var_sched.num_steps, 0, -1):

            # random noise for reverse step
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]

            # now net takes both original context and external condition
            cond_empty = torch.zeros_like(cond)
            e_theta = self.net(x_t, beta=beta, context=context, cond=cond_empty)

            # reverse update
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z

            traj[t - 1] = x_next.detach()  # no gradient
            traj[t] = traj[t].cpu()        # save history
            if not ret_traj:
                del traj[t]

        return traj if ret_traj else traj[0]


class TransformerDenoiser(nn.Module):
    """
    基于 Transformer 的去噪网络（解码器）。
    它接收含噪的 3D 骨骼点 (B, N, 3)，并预测 3D 噪声。
    使用 Transformer 的 Self-Attention 机制替代 DGCNN 的局部聚合。
    
    参数说明 (基于原始 DGCNN 的输入和推测)：
    point_dim: 3 (骨骼点坐标维度)
    context_dim: F1 (来自 ConditionalFusion/雷达特征等)
    cond_dim: F2 (来自 DGCNNEncoder/动作标签等)
    """
    def __init__(self, point_dim, context_dim, cond_dim, residual, k=8, 
                 model_dim=256, num_heads=4, num_layers=3):
        super().__init__()
        self.residual = residual
        self.point_dim = point_dim  # 3 (输入输出维度)
        self.model_dim = model_dim  # Transformer 内部特征维度
        
        # 1. 初始特征编码：将 (3) 维坐标编码为 (model_dim)
        # 类似原 DGCNN 的 init_mlp
        self.init_linear = nn.Sequential(
            nn.Linear(point_dim, model_dim), 
            nn.LeakyReLU(0.2),
            nn.LayerNorm(model_dim)
        )
        
        # 2. 条件/时间步嵌入
        # Fused Condition (F1 + F2): 用于融合雷达和动作特征
        # 原始：total_ctx_dim = context_dim + cond_dim + 3
        # 现在：将 F1 + F2 映射到 model_dim，时间步单独处理
        self.fused_cond_proj = nn.Linear(context_dim+cond_dim, model_dim)
        
        # Time Embedding Projection: 将时间步 (beta) 映射到 model_dim
        # 使用简单的 MLP 替代原 DGCNN 中的 sin/cos 手动拼接
        self.time_proj = nn.Sequential(
            nn.Linear(1, model_dim),  # beta 是 (B,)
            nn.LeakyReLU(0.2), 
            nn.Linear(model_dim, model_dim)
        )
        
        # 3. Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4, 
            dropout=0.1, 
            batch_first=True # (B, N, C) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 最终输出 MLP (降维到 3 维噪声预测)
        # 类似原 DGCNN 的 final_mlp
        self.final_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(model_dim // 2, point_dim) # 输出 3 维噪声
        )

    def forward(self, x, beta, context, cond):
        """
        Args:
            x:       含噪骨骼点 (B, N, 3), N=32
            beta:    时间步 (B,)
            context: 原全局 context (B, F1)
            cond:    DGCNNEncoder 输出的条件向量 (B, F2)
        """
        B, N, C = x.shape 
        
        # 1. 点特征初始化 (B, N, 3) -> (B, N, model_dim)
        # 替换 DGCNN 的 init_mlp 
        feat = self.init_linear(x)

        # 2. 条件嵌入和调制
        
        # 融合雷达/动作条件 (B, F1+F2) -> (B, model_dim)
        cond_empty = torch.zeros_like(cond)
        fused_cond = torch.cat([context, cond_empty], dim=-1)
        C_cond = self.fused_cond_proj(fused_cond)
        
        # 时间步嵌入 (B,) -> (B, model_dim)
        # 我们使用 beta 的 unsqueeze(1) 来保持 (B, 1) 的输入维度
        t_emb = self.time_proj(beta.view(B, 1))

        # 条件注入 (加性偏置/上下文): 
        # 将条件信息作为全局偏置，加到每个 token 的特征上
        # (B, model_dim) + (B, model_dim) -> (B, model_dim)
        global_bias = C_cond + t_emb 
        # 扩展到所有 N 个点 (B, N, model_dim)
        global_bias_expanded = global_bias.unsqueeze(1).expand(-1, N, -1) 
        
        # 特征 + 全局偏置
        feat = feat + global_bias_expanded 

        # 3. Transformer 编码 (自注意力)
        # 捕获 32 个骨骼点之间的全局和拓扑关系
        # (B, N, model_dim) -> (B, N, model_dim)
        att_output = self.transformer_encoder(feat)
        
        # 4. 最终降维 (Pointwise MLP)
        # 从 (B, N, model_dim) 降到 (B, N, 3)
        noise_pred = self.final_mlp(att_output)

        # 5. 残差连接（预测噪声 epsilon）
        if self.residual:
            # 返回预测的噪声 epsilon
            return noise_pred 
        else:
            # 如果预测的是 x0，则返回 x - epsilon_pred
            # 在 DDPM 中，通常预测 epsilon 或 x0
            return x - noise_pred  
        

class ActionConditionEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels):
        return self.embedding(labels)  # (B, embed_dim)


class RadarPointEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=128, k=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.k = k

    def forward(self, points):
        # points: (B, N, 5)   radar point cloud
        feat = self.mlp(points)  # (B, N, H)

        # Self-attention denoising
        denoised, _ = self.attn(feat, feat, feat)  # (B, N, H)

        # 局部平滑 (可选, kNN-based)
        # -> 简单写法：用全局 mean，或者用邻域均值 (实现上可以 knn_graph + scatter_mean)
        clean_feat = feat + denoised  # residual denoising
        out = self.fc_out(clean_feat) # (B, N, out_dim)

        # 输出既可以是每个点的特征，也可以做 pooling 得到全局条件
        global_feat = out.mean(dim=1)  # (B, out_dim)
        return global_feat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

class DGCNNPointwiseNet(nn.Module):
    """
    DGCNN 去噪网络（解码器），融合了 DGCNN 局部聚合。
    它接受含噪的 3D 点云，并预测 3D 噪声。
    """
    def __init__(self, point_dim, context_dim, cond_dim, residual, k=8):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.k = k
        self.point_dim = point_dim # 3 (输入输出维度)
        
        # 拼接后的总条件维度：context(F1) + cond(F2) + time_emb(3)
        self.total_ctx_dim = context_dim + cond_dim + 3 

        # 初始 MLP，将 X 编码为特征 (B, N, 3) -> (B, N, 64)
        self.init_mlp = nn.Sequential(
            nn.Linear(point_dim, 64), 
            nn.LeakyReLU(0.2),
            nn.LayerNorm(64)
        )
        
        # 核心：EdgeConv 层。EdgeConv 的 MLP 输入需要是：
        # (中心点特征 C_in) + (边缘差值 C_in) + (展平后的条件 C_cond)
        
        # Conv 1: (64*2) + C_cond -> 128
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(64*2 + self.total_ctx_dim, 128), nn.LeakyReLU(0.2)), aggr='max')
        # Conv 2: (128*2) + C_cond -> 256
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128*2 + self.total_ctx_dim, 256), nn.LeakyReLU(0.2)), aggr='max')
        # Conv 3: (256*2) + C_cond -> 512
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256*2 + self.total_ctx_dim, 512), nn.LeakyReLU(0.2)), aggr='max')
        
        # 最终 MLP (降维到 3 维噪声预测)
        self.final_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, point_dim) # 输出 3 维噪声
        )

    def forward(self, x, beta, context, cond):
        """
        Args:
            x:       含噪点云 (B, N, 3)
            beta:    时间步 (B,)
            context: 原全局 context (B, F1)
            cond:    DGCNNEncoder 输出的条件向量 (B, F2)
        """
        B, N, C = x.shape 
        device = x.device
        N_total = B * N # 总点数

        # --- 辅助函数：条件 EdgeConv 聚合逻辑 ---
        # 定义在 forward 内部以访问 self.convX 的 MLP 
        def conditional_edge_conv_forward(conv_layer, x_in, edge_idx, ctx_in_flat):
            # x_in: (N_total, C_in)
            # edge_idx: (2, E)
            # ctx_in_flat: (N_total, C_cond)
            
            row, col = edge_idx
            
            # 1. EdgeConv 的基本特征：x_i || (x_j - x_i)
            edge_feature = torch.cat([x_in[row], x_in[col] - x_in[row]], dim=1) 
            
            # 2. 拼接条件：将中心点 x_i 对应的条件向量拼接到 Edge Feature
            # (E, C_in*2 + C_cond)
            edge_feature_with_cond = torch.cat([edge_feature, ctx_in_flat[row]], dim=1)
            
            x_out = conv_layer.nn(edge_feature_with_cond) # <-- Use .nn instead
            
            # 4. Max Aggregation (EdgeConv 的核心步骤)
            # dim_size 应该是总的点数 N_total
            x_out, _ = scatter_max(x_out, row, dim=0, dim_size=N_total)
            return x_out

        # 1. 条件/时间步嵌入准备
        time_emb = torch.cat([beta.view(B, 1, 1), 
                            torch.sin(beta).view(B, 1, 1), 
                            torch.cos(beta).view(B, 1, 1)], dim=-1) # (B, 1, 3)

        ctx_emb = torch.cat([time_emb, context.view(B, 1, -1), cond.view(B, 1, -1)], dim=-1) 
        ctx_emb_expanded = ctx_emb.expand(-1, N, -1) # (B, N, total_ctx_dim)

        # 4. 展平条件 (FIXED: 使用 .reshape() 解决内存视图问题)
        ctx_flat = ctx_emb_expanded.reshape(N_total, self.total_ctx_dim) # (B*N, C_cond)
        
        # 2. PyG 数据转换和图构建
        
        # 坐标和特征 (含噪点云)
        pos_flat = x.view(N_total, C)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # 动态 k-NN 图构建 (基于含噪的 X, Y, Z)
        edge_index = knn_graph(pos_flat, k=self.k, batch=batch)
        
        # 3. 逐点初始化特征
        feat = self.init_mlp(x) # (B, N, 64)
        feat_flat = feat.view(N_total, 64) # (B*N, 64)

        # 5. 空间聚合 (EdgeConv + 条件拼接)
        
        # EdgeConv layers (使用内部定义的辅助函数)
        x1 = conditional_edge_conv_forward(self.conv1, feat_flat, edge_index, ctx_flat) # (N_total, 128)
        x2 = conditional_edge_conv_forward(self.conv2, x1, edge_index, ctx_flat) # (N_total, 256)
        x3 = conditional_edge_conv_forward(self.conv3, x2, edge_index, ctx_flat) # (N_total, 512)

        # 6. 最终降维 (Pointwise MLP)
        
        out = self.final_mlp(x3) # (N_total, 3)
        out = out.view(B, N, self.point_dim) # (B, N, 3)

        # 7. 残差连接
        if self.residual:
            # 预测的是噪声 epsilon，残差连接返回 x + epsilon
            return x + out 
        else:
            # 如果预测的是 x0，则直接返回 out
            return out
        
class DGCNNEncoder(nn.Module):

    def __init__(self, in_dim=5, out_dim=128, k=8):
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        
        self.conv1 = EdgeConv(nn.Sequential(
            nn.Linear(in_dim * 2, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64)
        ), aggr='max')
        
        # Conv 2: 64 -> 128
        self.conv2 = EdgeConv(nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128)
        ), aggr='max')
        
        # Conv 3: 128 -> 256
        self.conv3 = EdgeConv(nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256)
        ), aggr='max')
        
        # 全局特征聚合和降维 MLP
        self.fc_out = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512),  # 聚合多层特征
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, out_dim)
        )

    def forward(self, points):

        B, N, C = points.shape
        device = points.device
        
        x = points.view(B * N, C)
        
        pos = points[..., :3].view(B * N, 3) 
        
        batch = torch.arange(B, device=device).repeat_interleave(N)

        edge_index = knn_graph(pos, k=self.k, batch=batch)
        
        # 3. DGCNN 层次特征提取
        
        # Conv 1: (B*N, 5) -> (B*N, 64)
        x1 = self.conv1(x, edge_index)
        
        # Conv 2: (B*N, 64) -> (B*N, 128)
        x2 = self.conv2(x1, edge_index)
        
        # Conv 3: (B*N, 128) -> (B*N, 256)
        x3 = self.conv3(x2, edge_index)

        x_stacked = torch.cat([x1, x2, x3], dim=1) # (B*N, 448)

        global_feat = global_max_pool(x_stacked, batch) # (B, 448)
        
        # 6. 降维到输出维度
        final_feat = self.fc_out(global_feat) # (B, out_dim)
        
        return final_feat
    
class ConditionalFusion(nn.Module):
    def __init__(self, radar_dim=256, action_dim=128, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(radar_dim + action_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, radar_feat, action_feat):
        fused = torch.cat([radar_feat, action_feat], dim=-1)  # (B, radar_dim+action_dim)
        return self.fc(fused)  # (B, out_dim)
