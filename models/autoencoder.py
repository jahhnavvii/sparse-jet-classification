import torch
import torch.nn as nn
from models.pointnet import PointNetPPEncoder, square_distance

class FoldingDecoder(nn.Module):

    def __init__(self, latent_dim=128, num_points=512):
        super().__init__()
        self.num_points = num_points
        import math
        g = math.ceil(num_points**0.5)
        xs, ys = torch.meshgrid(
            torch.linspace(0, 1, g),
            torch.linspace(0, 1, g),
            indexing='ij'
        )
        grid = torch.stack([xs.flatten(), ys.flatten()], 1)
        self.register_buffer('grid', grid[:num_points])
        self.fold = nn.Sequential(
            nn.Linear(latent_dim + 2, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128),             nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, z):
        B    = z.shape[0]
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)
        z_   = z.unsqueeze(1).expand(-1, self.num_points, -1)
        return self.fold(torch.cat([z_, grid], dim=-1))

class JetAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, num_points=512):
        super().__init__()
        self.encoder = PointNetPPEncoder(latent_dim)
        self.decoder = FoldingDecoder(latent_dim, num_points)

    def forward(self, points):
        z = self.encoder(points)
        return self.decoder(z), z

def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = square_distance(pred, target)
    return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
