import os
import random
from copy import copy
from typing import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm

# 为了简单起见，删除10，17，2，28，30，31，29，27， 20，24，
class RadarDiffusionDataset(Dataset):

    def __init__(self, root_dir, split, scale_mode='global_unit', transform=None):
        super().__init__()
        assert split in ('train', 'val', 'test')
        assert scale_mode in ('global_unit', None), 'Only global_unit mode is implemented.'
        self.root_dir = root_dir
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds_data = []
        self.stats = None
        self.label_counter = Counter()

        self._load()
        self._get_statistics()
        self._split_dataset()
        self.data_by_split = self._get_split_data()

    def _collect_data_paths(self):
        datapath = []
        for env in ['env1', 'env2']:
            env_path = os.path.join(self.root_dir, env, 'subjects')
            if not os.path.exists(env_path):
                continue
            for subject in os.listdir(env_path):
                aligned_path = os.path.join(env_path, subject, 'aligned')
                if not os.path.exists(aligned_path):
                    continue
                for action in os.listdir(aligned_path):
                    action_path = os.path.join(aligned_path, action)
                    if not os.path.isdir(action_path):
                        continue
                    radar_files = [f for f in os.listdir(action_path) if f.endswith('.h5')]
                    skeleton_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
                    if radar_files and skeleton_files:
                        datapath.append((os.path.join(action_path, radar_files[0]),
                                         os.path.join(action_path, skeleton_files[0])))
        return datapath

    def _process_single_sample(self, radar_path, skeleton_path):
        raw_radar_frames = []
        with h5py.File(radar_path, 'r') as f:
            for name in sorted(f["frames"].keys()):
                frame = np.array(f["frames"][name])
                raw_radar_frames.append(frame)
        skeleton_data = np.load(skeleton_path).astype(np.float32)
        assert len(raw_radar_frames) == len(skeleton_data)
        
        processed_data = []
        for i in range(len(raw_radar_frames)):
         
            stacked_frames = raw_radar_frames[max(0, i-5):min(len(raw_radar_frames), i+5)]
            stacked_points = np.vstack(stacked_frames)
            
            stacked_points = self._clean_and_expend(stacked_points)[:, [5, 1, 6, 3, 7]]
            if len(stacked_points) > 0:
                stacked_points = np.unique(stacked_points, axis=0)

            processed_points = self._process_point_cloud(stacked_points)
            
            label = int(radar_path.split('action')[-1][:2]) - 1
            root_coord_mm = (skeleton_data[i][18].copy() + skeleton_data[i][22].copy()) / 2
    
            # 确保 root_coord 是 (1, 3) 形状，用于后续的归一化和存储
            root_coord_mm = root_coord_mm.reshape(1, 3) 

            # 2. 关节裁剪
            # skeleton_30x3 形状为 (30, 3)
            skeleton_data[i][0] = root_coord_mm
            delete_indices = [2, 28, 29, 27, 30, 31]
            skeleton_data26= np.delete(skeleton_data[i], delete_indices, axis=0)
           
    
            pointcloud = skeleton_data26
            processed_data.append({
                'radar_cond': torch.from_numpy(processed_points).float(),
                'action_cond': torch.tensor(label).long(),
                'pointcloud': torch.from_numpy(pointcloud).float(),
                'id': f'{os.path.basename(radar_path)}_{i}',
                'root-shift': torch.from_numpy(root_coord_mm).float(),
            })
            self.label_counter[label] += 1
        return processed_data
    
    def _clean_and_expend(self, radar_data):
        if radar_data.shape[0] == 0 or len(radar_data.shape) != 2 or radar_data.shape[1] < 8:
            return np.zeros((1, 8))
        x = radar_data[:, 5]
        y = radar_data[:, 6]
        z = radar_data[:, 1]
        valid_mask = (x >= -1.5) & (x <= 1.5) & (y >= 1.0) & (y <= 4.5) & (z >= -1) & (z <= 2.0)
        if not np.any(valid_mask):
            return np.zeros((1, 8))
        return radar_data[valid_mask]

    def _process_point_cloud(self, points, max_points=512):
        if points.size == 0:
            points = np.zeros((0, 5))
        
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            return points[indices]
        
        if len(points) < max_points:
            pad_size = max_points - len(points)
            padding = np.tile(points[-1:], (pad_size, 1)) if len(points) > 0 else np.zeros((pad_size, 3))
            return np.vstack([points, padding])
        return points

    def _load(self):
        datapath = self._collect_data_paths()
        print('Loading data...')
        for radar_path, skeleton_path in tqdm(datapath):
            self.pointclouds_data.extend(self._process_single_sample(radar_path, skeleton_path))
        print(f'Total {len(self.pointclouds_data)} samples loaded.')
        
    def _get_statistics(self):
        cache_path = os.path.join(os.path.dirname(self.root_dir), 'stats_cache.pt')
        if os.path.exists(cache_path):
            self.stats = torch.load(cache_path)
            return
        
        # 1. 骨骼点统计 (Pointcloud Stats)
        # 注意：需要将所有点云展平以计算全局统计
        all_sk_points = torch.cat([d['pointcloud'].view(-1, 3) for d in self.pointclouds_data], dim=0)
        sk_mean = all_sk_points.mean(dim=0)  # 形状为 [3]
        sk_std = all_sk_points.std(dim=0)    # 形状为 [3]
        
        # 2. 雷达点云统计 (Radar Cond Stats)
        # 注意：雷达点云是 (N_points, D) 形状，需要将其展平以计算全局统计
        all_radar_points = torch.cat([d['radar_cond'] for d in self.pointclouds_data], dim=0)
   
        radar_mean = all_radar_points.mean(dim=0)
        radar_std = all_radar_points.std(dim=0)
        
        self.stats = {
            'sk_mean': sk_mean, 
            'sk_std': sk_std,
            'radar_mean': radar_mean, # 新增
            'radar_std': radar_std     # 新增
        }
        torch.save(self.stats, cache_path)

    def _split_dataset(self):
        random.seed(42)
        random.shuffle(self.pointclouds_data)
        
        total_size = len(self.pointclouds_data)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        self.train_data = self.pointclouds_data[:train_size]
        self.val_data = self.pointclouds_data[train_size:train_size + val_size]
        self.test_data = self.pointclouds_data[train_size + val_size:]

    def _get_split_data(self):
        if self.split == 'train':
            return self.train_data
        elif self.split == 'val':
            return self.val_data
        else:
            return self.test_data

    def __len__(self):
        return len(self.data_by_split)

    def __getitem__(self, idx):
        data = self.data_by_split[idx]
        
        # sk_mean和sk_std现在是形状为[3]的向量，需要广播到点云形状
        sk_shift = self.stats['sk_mean'].unsqueeze(0)  # [1, 3]
        sk_scale = self.stats['sk_std'].unsqueeze(0)   # [1, 3]
        normalized_sk = (data['pointcloud'] - sk_shift) / sk_scale 
        
        # 2. 雷达点云标准化 (Standardization)
        # 从 stats 中获取雷达的 mean/std
        radar_mean = self.stats['radar_mean'].reshape(1, 5) 
        radar_std = self.stats['radar_std'].reshape(1, 5)
        normalized_radar = (data['radar_cond'] - radar_mean) / radar_std

        # 确保所有tensors都是可以调整大小的（解决storage问题）
        normalized_sk = normalized_sk.clone()
        normalized_radar = normalized_radar.clone()

        return {
            'pointcloud': normalized_sk,
            'action_cond': data['action_cond'],
            'radar_cond': normalized_radar,
            'id': data['id'],
            'shift': sk_shift.clone(),  # [1, 3]
            'scale': sk_scale.clone(),  # [1, 3]
        }

