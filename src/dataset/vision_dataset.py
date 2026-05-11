import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class VisionTransitionDataset(Dataset):
    def __init__(self, h5_path, device):
        self.device = device
        print("Loading entire dataset into RAM... This will take a minute.")
        with h5py.File(h5_path, 'r') as hf:
            self.images = hf['images'][:]
            self.q = hf['q'][:]
            self.qd = hf['qd'][:]
            self.u = hf['u'][:]           # NEW: Action taken
            self.min_dist = hf['min_dist'][:]
            self.ncon = hf['ncon'][:]     # NEW: Collision flag
            
        self.N = self.q.shape[0] - 1 

        # Compute normalization stats
        subset_idx = min(10000, self.N)
        x_raw = np.concatenate([self.q[:subset_idx], self.qd[:subset_idx]], axis=1)
        self.x_mean = torch.tensor(x_raw.mean(axis=0), dtype=torch.float32).to(self.device)
        self.x_std = torch.tensor(x_raw.std(axis=0) + 1e-6, dtype=torch.float32).to(self.device)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Current timestep
        img_t = torch.from_numpy(self.images[idx])
        q_t = torch.from_numpy(self.q[idx])
        qd_t = torch.from_numpy(self.qd[idx])
        u_t = torch.from_numpy(self.u[idx])
        dist_t = torch.from_numpy(self.min_dist[idx])
        ncon_t = torch.from_numpy(self.ncon[idx])

        # Next timestep target
        img_next = torch.from_numpy(self.images[idx+1])

        # Returns exactly what train_vision_vocbf.py expects
        return img_t, q_t, qd_t, u_t, dist_t, ncon_t, img_next