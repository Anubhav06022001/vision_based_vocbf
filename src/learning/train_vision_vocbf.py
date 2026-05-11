import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.vocbf.vision_dynamics_net import VisionDynamicsBarrierNet 
from src.learning.expectile import expectile_loss
from src.safety.safety_function import ell_from_distance
from src.dataset.vision_dataset import VisionTransitionDataset

# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
H5_PATH = PROJECT_ROOT / "data" / "vision_vocbf" / "transition_dataset.h5"

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, running on CPU")

gamma = 0.99
tau = 0.9 
lr = 0.001 
epochs = 75
batch_size = 512
d_safe = 0.1

print("Loading dataset...")
dataset = VisionTransitionDataset(H5_PATH, DEVICE)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ================= MODEL =================
# Now instantiating the 3-part network
B_net = VisionDynamicsBarrierNet(latent_dim=64, state_dim=14, action_dim=7).to(DEVICE)
optimizer = torch.optim.Adam(B_net.parameters(), lr=lr)

# ================= TRAIN =================
print("Starting training...")

for epoch in range(epochs):
    losses = []
    dyn_losses = []
    safe_losses = []
    
    for img_t, q_t, qd_t, u_t, dist_t, ncon_t, img_next in dataloader:
        
        # 1. Preprocess Images
        img_t = img_t.permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
        img_next = img_next.permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
        
        # 2. Preprocess States & Actions
        x_t = torch.cat([q_t, qd_t], dim=1).to(DEVICE)
        x_t = (x_t - dataset.x_mean) / dataset.x_std
        u_t = u_t.float().to(DEVICE)
        
        dist_t = dist_t.to(DEVICE)
        ncon_t = ncon_t.to(DEVICE)

        # 3. Compute Safety Signal l(x) and ANCHOR COLLISIONS
        ell = ell_from_distance(dist_t, d_safe).float()
        ell[ncon_t.squeeze() > 0] = -1.0 
        
        # --- FORWARD PASS: LATENT DYNAMICS ---
        z_t = B_net.encode(img_t)
        
        with torch.no_grad():
            z_next_actual = B_net.encode(img_next)
            
        z_next_pred = B_net.predict_next_z(z_t, x_t, u_t)
        
        # --- FORWARD PASS: SAFETY BARRIER ---
        B_current = B_net.get_barrier(z_t).squeeze()
        B_current = torch.clamp(B_current, -10, 10)
        
        with torch.no_grad():
            B_next = B_net.get_barrier(z_next_actual).squeeze()
            B_next = torch.clamp(B_next, -10, 10)
            
        # --- LOSS COMPUTATION ---
        
        # A. Dynamics Loss
        loss_dyn = F.mse_loss(z_next_pred, z_next_actual)
        
        # B. Value-Guided Expectile Loss
        target = (1 - gamma) * ell + gamma * torch.minimum(ell, B_next)
        loss_safe = expectile_loss(B_current, target, tau)
        
        # C. Anchor Loss
        unsafe = (ell < 0).squeeze()
        safe = (ell > 0).squeeze()
        
        anchor = torch.tensor(0.0, device=DEVICE)
        if unsafe.any():
            anchor += 0.05 * (B_current[unsafe] + 1).pow(2).mean()
        if safe.any():
            anchor += 0.01 * (B_current[safe] - 1).pow(2).mean()
            
        # Total Loss (weight dynamics highly so it learns physics first)
        loss = (10.0 * loss_dyn) + loss_safe + anchor
        
        # --- OPTIMIZE ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(B_net.parameters(), 5.0)
        optimizer.step()
        
        losses.append(loss.item())
        dyn_losses.append(loss_dyn.item())
        safe_losses.append(loss_safe.item())

    print(f"Epoch {epoch:03d} | Total: {np.mean(losses):.4f} | Dyn: {np.mean(dyn_losses):.4f} | Safe: {np.mean(safe_losses):.4f}")

    if (epoch + 1) % 10 == 0:
        os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
        checkpoint_path = PROJECT_ROOT / "models" / f"vision_vocbf_epoch_{epoch+1}.pt"
        torch.save(B_net.state_dict(), checkpoint_path)

# ================= SAVE =================
os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
torch.save(B_net.state_dict(), PROJECT_ROOT / "models" / "vision_vocbf.pt")
print("Saved learned Vision V-OCBF to models/vision_vocbf.pt")