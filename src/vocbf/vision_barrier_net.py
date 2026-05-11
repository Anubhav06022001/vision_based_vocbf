import torch
import torch.nn as nn
from src.perception.encoder_net import VisionEncoder



class VisionBarrierNet(nn.Module):
    def __init__(self, latent_dim=64, state_dim=14):
        super().__init__()
        self.encoder = VisionEncoder(latent_dim=latent_dim)
        
        # Input: latent (64) + q (7) + qd (7) = 78
        self.barrier_mlp = nn.Sequential(
            nn.Linear(latent_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, img, x_state):
        """
        img: (Batch, 3, 128, 128)
        x_state: (Batch, 14) representing concatenated q and qd
        """
        z = self.encoder(img)
        combined = torch.cat([z, x_state], dim=1)
        return self.barrier_mlp(combined)