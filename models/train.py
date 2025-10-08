import torch
from torch.utils.data import DataLoader
import os
import glob
from tqdm import tqdm

from ce import ContourEncoderUNet
from decoder import PointCloudDecoder
from dpm import DiffRadarModel


def main():
    args = parse_args()
    os.makedirs(args.save, exist_ok=True)
    files = glob.glob(args.data)
    if len(files) == 0:
        raise RuntimeError("No data files found. Please set --data to point to your .npy/.npz files")

    # 假设 Radar2DDataset 现在返回雷达图像和对应的 LiDAR 真值
    dataset = Radar2DDataset(files, H=args.H, W=args.W)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device)

    # 1. 初始化模型
    # CE模块用于从雷达输入中提取条件信息
    ce = ContourEncoderUNet(in_ch=1, base_ch=32, latent_ch=args.latent_ch).to(device)
    # 扩散模型部分，包含编码器和解码器
    dpm = DiffRadarModel(in_channels=args.latent_ch + 2, out_channels=1, base_channels=128).to(device)
    # PointCloudDecoder用于从扩散模型的最终输出（LiDAR BEV）中生成点云
    decoder = PointCloudDecoder(in_ch=1, num_points=args.num_points, hidden=512).to(device)
    
    # 2. 定义优化器
    # 优化器需要同时更新所有可训练的模型参数
    optim = torch.optim.Adam(list(ce.parameters()) + list(dpm.parameters()) + list(decoder.parameters()), lr=args.lr)

    # 3. 定义噪声调度
    # 这部分通常放在DDPM或DPMModel类中，但为了清晰，这里也展示
    timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        running_loss = 0.0
        # 假设 loader 返回雷达图像和相应的LiDAR真值
        for radar_imgs, lidar_gts in pbar:
            radar_imgs = radar_imgs.to(device)  # 雷达输入 (B, C, H, W)
            lidar_gts = lidar_gts.to(device)    # LiDAR 真值 (B, C, H, W)
            B = radar_imgs.shape[0]

            # 4. 前向过程：从LiDAR真值中添加噪声
            t = torch.randint(0, timesteps, (B,), device=device, dtype=torch.long)
            # 从tensor中提取对应时间步t的参数
            sqrt_alpha_t = sqrt_alphas_cumprod[t, None, None, None]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t, None, None, None]
            
            noise = torch.randn_like(lidar_gts)
            # 这是前向过程的核心公式：yt = sqrt(alpha_cumprod_t) * y0 + sqrt(1 - alpha_cumprod_t) * noise
            noisy_lidar_gt = sqrt_alpha_t * lidar_gts + sqrt_one_minus_alpha_t * noise

            # 5. 生成条件输入 xc
            # 从雷达图像中提取轮廓信息作为条件
            x_contour = ce(radar_imgs)
            # 根据论文，需要对x_contour进行二值化等操作以构建完整的xc
            # 这是一个简化的实现，实际需要根据论文细节来构建xc
            xc = torch.cat([noisy_lidar_gt, x_contour], dim=1) 
            # 注意：这里的noisy_lidar_gt是需要去噪的，x_contour是指导信息

            # 6. 反向过程：去噪网络预测噪声
            # DPMModel现在需要接收带噪的图像和条件信息xc
            predicted_noise = dpm(xc, t)

            # 7. 计算DDPM损失
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # 8. 反向传播和优化
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1e-12))
        
        # 保存检查点
        ckpt = {
            'ce': ce.state_dict(),
            'dpm': dpm.state_dict(),
            'decoder': decoder.state_dict(),
            'optim': optim.state_dict(),
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join(args.save, f'ckpt_epoch_{epoch}.pth'))
        print(f"Saved checkpoint epoch {epoch} loss {running_loss / len(loader):.6f}")

if __name__ == '__main__':
    # 假设有一个简单的参数解析器
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--save", type=str, default="checkpoints")
        parser.add_argument("--data", type=str, default="data/*.npy")
        parser.add_argument("--H", type=int, default=256)
        parser.add_argument("--W", type=int, default=256)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--latent_ch", type=int, default=3)
        parser.add_argument("--num_points", type=int, default=2048)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        return parser.parse_args()
        
    main()