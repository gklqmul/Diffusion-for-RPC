import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoencoder import AutoEncoder
from utils.dataset import RadarDiffusionDataset
from utils.misc import seed_all


parser = argparse.ArgumentParser()

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./dataset')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)


args = parser.parse_args()

# ----------------------------
# Configuration
# ----------------------------
ckpt_path = './pretrained/AE_alls.pt'
dataset_path = './dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = './latent_tsne_results'
os.makedirs(save_dir, exist_ok=True)

print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

# Init model
model = AutoEncoder(ckpt['args']).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
seed_all(getattr(ckpt['args'], 'seed', 42))


# Load val set
val_dset = RadarDiffusionDataset(root_dir=dataset_path, split='val')
val_loader = DataLoader(val_dset, batch_size=32, num_workers=0, shuffle=False)

# ----------------------------
# Helper: encode whole val set
# ----------------------------
def encode_latent(use_zero_cond=False):
    all_codes, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Encoding ({'zero cond' if use_zero_cond else 'with cond'})"):
            x_val = batch['pointcloud'].to(device)
            radar_cond = batch['radar_cond'].to(device)
            action_cond = batch['action_cond'].to(device)
            if use_zero_cond:
                radar_cond = torch.zeros_like(radar_cond)
                action_cond = torch.zeros_like(action_cond)

            code = model.encode(x_val)
            if isinstance(code, tuple):
                code = code[0]
            all_codes.append(code.cpu().numpy())
            all_labels.append(batch['action_cond'].cpu().numpy())

    code_np = np.concatenate(all_codes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return code_np, labels

# ----------------------------
# Encode two settings
# ----------------------------
code_with, labels = encode_latent(use_zero_cond=False)
code_zero, _ = encode_latent(use_zero_cond=True)

# ----------------------------
# Visualisation (t-SNE)
# ----------------------------
def plot_tsne(code_np, labels, title, save_path):
    print(f"Computing t-SNE for: {title}")
    code_scaled = StandardScaler().fit_transform(code_np)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    code_2d = tsne.fit_transform(code_scaled)

    plt.figure(figsize=(8, 7))
    for action_id in np.unique(labels):
        mask = labels == action_id
        plt.scatter(code_2d[mask, 0], code_2d[mask, 1], label=f"Action {action_id}", alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

# Plot and save
epoch_id = ckpt_path.split('_')[-1].replace('.pt', '')
plot_tsne(code_with, labels, f"t-SNE with Condition (Epoch {epoch_id})",
          os.path.join(save_dir, f"latent_withcond_epoch{epoch_id}.png"))
plot_tsne(code_zero, labels, f"t-SNE with Zero Condition (Epoch {epoch_id})",
          os.path.join(save_dir, f"latent_zerocond_epoch{epoch_id}.png"))
