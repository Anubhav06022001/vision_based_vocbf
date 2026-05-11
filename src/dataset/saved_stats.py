import h5py
import numpy as np
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
h5_path = PROJECT_ROOT / "data" / "vision_vocbf" / "transition_dataset.h5"
models_dir = PROJECT_ROOT / "models"
os.makedirs(models_dir, exist_ok=True)

print("Extracting normalization stats from HDF5...")
with h5py.File(h5_path, 'r') as hf:
    # Use the exact same subset logic from the training script
    N = hf['q'].shape[0] - 1
    subset_idx = min(10000, N)
    
    qs = hf['q'][:subset_idx]
    qds = hf['qd'][:subset_idx]
    
    x_raw = np.concatenate([qs, qds], axis=1)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)

np.save(models_dir / "dyn_x_mean.npy", x_mean)
np.save(models_dir / "dyn_x_std.npy", x_std)

print("Saved dyn_x_mean.npy and dyn_x_std.npy to models/ directory.")