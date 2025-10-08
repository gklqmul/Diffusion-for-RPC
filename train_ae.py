import os
import argparse
from matplotlib import pyplot as plt
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np
import plotly.graph_objects as go

from evaluation.evaluation_metrics import calculate_permutation_aware_mpje
from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD

# BONES = [ (0, 1), (1, 2), (2, 3), (3, 26), # Torso 
#          (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10), # Left arm 
#          (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17), # Right arm 
#          (0, 18), (18, 19), (19, 20), (20, 21), # Left leg 
#          (0, 22), (22, 23), (23, 24), (24, 25) # Right leg 
#        ]

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
# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.1)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./dataset')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=5*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=10*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')
train_dset = RadarDiffusionDataset(
    root_dir=args.dataset_path,
    split='train'
)
val_dset = RadarDiffusionDataset(
    root_dir=args.dataset_path,
    split='val'
)   

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    radar_cond = batch['radar_cond'].to(args.device)
    action_cond = batch['action_cond'].to(args.device)
    radar_cond_zero = torch.zeros_like(radar_cond)   # (B, radar_feat_dim)
    action_cond_zero = torch.zeros_like(action_cond) # (B, action_feat_dim)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x, radar_cond, action_cond)

    # Backward and optimize 
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):
    all_refs = []
    all_recons = []
    total_mpje = 0.0

    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break

        ref = batch['pointcloud'].to(args.device)
        radar_cond = batch['radar_cond'].to(args.device)
        action_cond = batch['action_cond'].to(args.device)
        radar_cond_zero = torch.zeros_like(radar_cond)   # (B, radar_feat_dim)
        action_cond_zero = torch.zeros_like(action_cond) # (B, action_feat_dim)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        B, N, _ = ref.shape

        with torch.no_grad():
            model.eval()
            m, v = model.encode(ref)
            code = m
            recons = model.decode(code, radar_cond, action_cond, ref.size(1), flexibility=args.flexibility)

        # 坐标还原
        ref_world = ref * scale + shift
        recons_world = recons * scale + shift

        # MPJPE
        # batch_mpje = calculate_mpje(recons_world, ref_world)
        batch_mpje = calculate_permutation_aware_mpje(recons_world, ref_world)
        total_mpje += batch_mpje * B

        all_refs.append(ref_world)
        all_recons.append(recons_world)

        # --- 可视化前2个样本 ---
        if i < 2:
            for b in range(min(2, B)):
                save_skeleton_compare_html(
                    ref_world[b].cpu().numpy(),
                    recons_world[b].cpu().numpy(),
                    BONES,
                    html_path=os.path.join(log_dir, f"iter_{it}_batch{i}_sample{b}_skeleton.html"),
                )
                save_radar_pointcloud_html(
                    radar_cond[b].cpu().numpy(),
                    html_path=os.path.join(log_dir, f"iter_{it}_batch{i}_sample{b}_radar.html"),
                )

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    mpje = total_mpje / all_refs.size(0)

    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f | MPJPE %.6f' % (it, cd, emd, mpje))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.add_scalar('val/mpje', mpje, it)
    writer.flush()

    return cd  # Return CD loss as the primary metric for checkpointing

def validate_inspect(it):
    sum_n = 0
    
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        radar_cond = batch['radar_cond'].to(args.device)
        action_cond = batch['action_cond'].to(args.device)
        radar_cond_zero = torch.zeros_like(radar_cond)   # (B, radar_feat_dim)
        action_cond_zero = torch.zeros_like(action_cond) # (B, action_feat_dim)
        model.eval()
        m, v = model.encode(x)  # VAE returns mean and log-variance
        recons = model.decode(m, radar_cond, action_cond, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')