import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = TransformerEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = TransformerDenoiser(point_dim=3, context_dim=args.latent_dim, cond_dim=512, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        self.kl_weight = args.kl_weight if hasattr(args, 'kl_weight') else 1e-4

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        # code, _ = self.encoder(x)
        m, v = self.encoder(x) # TransformerEncoder 返回 m 和 v
        return m, v
        # return code

    def decode(self, code, radar_cond, action_cond, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, radar_cond, action_cond, flexibility=flexibility, ret_traj=ret_traj)

    # def get_loss(self, x, radar_cond, action_cond):
    #     code = self.encode(x)
    #     loss = self.diffusion.get_loss(x, code, radar_cond, action_cond)
    #     return loss
    
    # def get_loss(self, x, radar_cond, action_cond):
        
    #     # 1. 编码 (Encoder)
    #     m, v = self.encode(x) # m: 均值, v: 对数方差 (log(sigma^2))
        
    #     # 2. VAE 重参数化技巧 (Reparameterization Trick)
    #     # 从 N(m, exp(v)) 中采样 z
    #     std = torch.exp(0.5 * v) # 标准差
    #     eps = torch.randn_like(std)
    #     code = m + eps * std # code (z) 作为 Diffusion 的 context
        
    #     # 3. 扩散损失 (Diffusion Loss): 预测噪声的 MSE
    #     # x: x_0 (GT), code: z (潜在向量)
    #     loss_diffusion = self.diffusion.get_loss(x, code, radar_cond, action_cond)
        
    #     # 4. KL 散度损失 (KL Divergence Loss)
    #     # L_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     # v 是 log(sigma^2), exp(v) 是 sigma^2
    #     loss_kl = -0.5 * torch.sum(1 + v - m.pow(2) - torch.exp(v))
    #     loss_kl = loss_kl / m.size(0) # 对 Batch 求平均

    #     loss_total = loss_diffusion + self.kl_weight * loss_kl

    #     self.latest_diffusion_loss = loss_diffusion.item()
    #     self.latest_kl_loss = loss_kl.item()

    #     return loss_total
    def get_loss(self, x, radar_cond, action_cond, cond_prob=0.5):
        """
        Args:
            x: Ground truth skeleton (B, N, D)
            radar_cond: Radar condition
            action_cond: Action condition
            cond_prob: Probability to drop latent and train condition-only mode
        """
        # 1. 编码
        m, v = self.encode(x)
        std = torch.exp(0.5 * v)
        eps = torch.randn_like(std)
        code = m + eps * std  # VAE reparameterization

        # 2. 随机选择模式
        if torch.rand(1).item() < cond_prob:
            # 模拟推理模式 —— 不使用latent
            code_used = torch.randn_like(code)
        else:
            # 正常训练模式 —— 使用encoder生成的latent
            code_used = code

        # 3. 扩散损失 (预测噪声MSE)
        loss_diffusion = self.diffusion.get_loss(x, code_used, radar_cond, action_cond)

        # 4. KL散度损失
        loss_kl = -0.5 * torch.sum(1 + v - m.pow(2) - torch.exp(v)) / m.size(0)

        # 5. 合并
        loss_total = loss_diffusion + self.kl_weight * loss_kl

        # logging
        self.latest_diffusion_loss = loss_diffusion.item()
        self.latest_kl_loss = loss_kl.item()

        return loss_total
