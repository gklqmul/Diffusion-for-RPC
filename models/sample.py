# sample.py
import torch
import argparse
import numpy as np
from ce import ContourEncoderUNet
from dpm import UNet2D, DDPM
from decoder import PointCloudDecoder
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint .pth')
    parser.add_argument('--num', type=int, default=4, help='number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--W', type=int, default=128)
    parser.add_argument('--latent_ch', type=int, default=128)
    parser.add_argument('--num_points', type=int, default=128)
    args = parser.parse_args()
    return args

def load_models(ckpt_path, device, latent_ch, num_points, H, W):
    # instantiate models (CE architecture needed only for shape info)
    ce = ContourEncoderUNet(in_ch=1, base_ch=32, latent_ch=latent_ch).to(device)
    unet2d = UNet2D(in_ch=latent_ch, base_ch=128, time_emb_dim=256).to(device)
    ddpm = DDPM(unet2d, timesteps=1000, device=device).to(device)
    decoder = PointCloudDecoder(in_ch=latent_ch, num_points=num_points, hidden=512).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    ce.load_state_dict(ck['ce'])
    unet2d.load_state_dict(ck['unet2d'])
    decoder.load_state_dict(ck['decoder'])
    ce.eval(); unet2d.eval(); decoder.eval(); ddpm.eval()
    return ce, ddpm, decoder

def sample_and_decode(ddpm, decoder, device, num_samples, latent_ch, H, W):
    # sample feature maps from ddpm
    shape = (num_samples, latent_ch, H, W)
    feat_maps = ddpm.sample(shape, device=device, progress=True)  # (B, C, H, W)
    # decode into point clouds
    pcs = decoder(feat_maps)  # (B, num_points, 3)
    return pcs.cpu().numpy()

def main():
    args = parse_args()
    ce, ddpm, decoder = load_models(args.ckpt, args.device, args.latent_ch, args.num_points, args.H, args.W)
    pcs = sample_and_decode(ddpm, decoder, torch.device(args.device), args.num, args.latent_ch, args.H, args.W)
    save_path = os.path.splitext(args.ckpt)[0] + f'_samples.npy'
    np.save(save_path, pcs)
    print(f"Saved generated point clouds to {save_path}")

if __name__ == '__main__':
    main()
