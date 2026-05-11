import torch
import torch.nn as nn
from src.perception.encoder_net import VisionEncoder

class VisionDynamicsBarrierNet(nn.Module):
    def __init__(self, latent_dim=64, state_dim=14, action_dim=7):
        super().__init__()
        
        # 1. Image Encoder (img_t -> z_t)
        self.encoder = VisionEncoder(latent_dim=latent_dim)
        
        # 2. Latent Dynamics Model (z_t, x_t, u_t -> z_next)
        # Input size: 64 (latent) + 14 (state) + 7 (action) = 85
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim) # Outputs a predicted change in z
        )
        
        # 3. Barrier Function B(z_next) -> Safety Margin
        # The barrier now only needs to look at the visual features 
        # to determine if an obstacle is taking up the camera frame.
        self.barrier_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode(self, img):
        """Extracts visual features from the camera."""
        return self.encoder(img)

    def predict_next_z(self, z_t, x_t, u_t):
        """Predicts what the camera will see AFTER we take action u_t."""
        dyn_input = torch.cat([z_t, x_t, u_t], dim=1)
        
        # We predict the residual (delta_z) instead of absolute z_next.
        # This is a standard trick in deep learning that makes dynamics 
        # much more stable to train!
        z_delta = self.dynamics(dyn_input)
        return z_t + z_delta

    def get_barrier(self, z):
        """Calculates safety value B from visual features."""
        return self.barrier_mlp(z)

    def forward(self, img_t, x_t, u_t):
        """
        Full forward pass used during QP Inference.
        This provides the unbroken gradient path from B all the way back to u_t.
        """
        z_t = self.encode(img_t)
        z_next = self.predict_next_z(z_t, x_t, u_t)
        B_val = self.get_barrier(z_next)
        return B_val