import os
import time
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import plotly.graph_objects as go

from evaluation.evaluation_metrics import calculate_permutation_aware_mpje
from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD

BONES = [ (0, 1), (1, 2), (2, 25), # Torso 
         (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (7, 9), # Left arm 
         (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (14, 16), # Right arm 
         (0, 17), (17, 18), (18, 19), (19, 20), # Left leg 
         (0, 21), (21, 22), (22, 23), (23, 24) # Right leg 
       ]

def save_skeleton_compare_html(gt_points, pred_points, BONES, html_path="skeleton_compare.html", title="Skeleton Comparison"):
    # === Convert to numpy ===
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.detach().cpu().numpy()
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.detach().cpu().numpy()

    # === Prepare line segments ===
    def make_bone_lines(points, bones, color, name):
        lines = []
        for (i, j) in bones:
            if i < len(points) and j < len(points):
                lines.append(
                    go.Scatter3d(
                        x=[points[i, 0], points[j, 0]],
                        y=[points[i, 1], points[j, 1]],
                        z=[points[i, 2], points[j, 2]],
                        mode='lines',
                        line=dict(color=color, width=5),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
        return lines

    # === Create traces ===
    traces = []

    # Ground truth joints
    traces.append(go.Scatter3d(
        x=gt_points[:, 0],
        y=gt_points[:, 1],
        z=gt_points[:, 2],
        mode='markers+text',
        text=[str(i) for i in range(len(gt_points))],
        textposition='top center',
        marker=dict(size=5, color='black', symbol='circle'),
        name='GT joints'
    ))
    traces += make_bone_lines(gt_points, BONES, 'gray', name='GT bones')

    # Predicted joints
    traces.append(go.Scatter3d(
        x=pred_points[:, 0],
        y=pred_points[:, 1],
        z=pred_points[:, 2],
        mode='markers+text',
        text=[str(i) for i in range(len(pred_points))],
        textposition='top center',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Pred joints'
    ))

    # === Define layout ===
    all_points = np.vstack([gt_points, pred_points])
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[all_points[:, 0].min(), all_points[:, 0].max()]),
            yaxis=dict(range=[all_points[:, 1].min(), all_points[:, 1].max()]),
            zaxis=dict(range=[all_points[:, 2].min(), all_points[:, 2].max()]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )

    fig = go.Figure(data=traces, layout=layout)

    # === Save to HTML ===
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive skeleton comparison to {html_path}")


def save_radar_pointcloud_html(points, html_path="radar_cloud.html", title="Radar Point Cloud"):
    assert points.shape[1] == 5, "Input must have shape (N, 5): [x, y, z, doppler, snr]"
    x, y, z, doppler, snr = points.T

    # Normalize doppler and snr for visual mapping
    doppler_norm = (doppler - doppler.min()) / (doppler.max() - doppler.min() + 1e-8)
    snr_norm = (snr - snr.min()) / (snr.max() - snr.min() + 1e-8)

    # Colour by Doppler (blue to red), size by SNR
    colours = [f"hsl({240 - 240*d}, 100%, 50%)" for d in doppler_norm]
    sizes = 4 + 6 * snr_norm

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colours,
                opacity=0.8
            ),
            text=[f"Doppler: {d:.2f}<br>SNR: {s:.2f}" for d, s in zip(doppler, snr)],
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Save as HTML only (no display)
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"✅ Saved interactive radar point cloud to {html_path}")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_alls.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./dataset')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--latent_dim', type=int, default=256)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = RadarDiffusionDataset(
    root_dir=args.dataset_path,
    split='test'
)

test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
total_mpje = 0
total_samples = 0
log_dir = 'test_logs'
os.makedirs(log_dir, exist_ok=True)

for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    radar_cond = batch['radar_cond'].to(args.device)
    action_cond = batch['action_cond'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    B, N, _ = ref.shape
    B, N, _ = ref.shape
    latent_dim = ckpt['args'].latent_dim
    z = torch.randn(B, latent_dim).to(args.device)
   
    # Decode only from radar/action conditions (no GT x)
    # recons = model.decode(z, radar_cond, action_cond, N, flexibility=ckpt['args'].flexibility)
    model.eval()
    with torch.no_grad():
        m, v = model.encode(ref)  # VAE returns mean and log-variance
        recons = model.decode(m, radar_cond, action_cond, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref_world = ref * scale + shift
    recons_world = recons * scale + shift


    batch_mpje = calculate_permutation_aware_mpje(recons_world, ref_world)
    total_mpje += batch_mpje * B

    all_ref.append(ref_world)
    all_recons.append(recons_world)

        # --- 可视化前2个样本 ---
    if i < 2:
        for b in range(min(2, B)):
            save_skeleton_compare_html(
                    ref_world[b].cpu().numpy(),
                    recons_world[b].cpu().numpy(),
                    BONES,
                    html_path=os.path.join(log_dir, f"test_batch{i}_sample{b}_skeleton.html"),
            )
            save_radar_pointcloud_html(
                    radar_cond[b].cpu().numpy(),
                    html_path=os.path.join(log_dir, f"test_batch{i}_sample{b}_radar.html"),
            )


all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.cpu().numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.cpu().numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
mpje = total_mpje / all_ref.size(0)
print('MPJE: %.12f' % mpje)
print('CD:  %.12f' % cd)
logger.info('MPJE: %.12f' % mpje)
logger.info('CD:  %.12f' % cd)
logger.info('EMD: %.12f' % emd)
