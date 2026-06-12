import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from phase2.quadtree import build_quadtree, get_active_points

class SparseMLP(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(4,  32), nn.LayerNorm(32), nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(64,128), nn.LayerNorm(128), nn.ReLU(inplace=True)
        )

        self.proj = nn.Sequential(
            nn.Linear(32+64+128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, points):

        x1 = self.mlp1(points)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)
        f1 = x1.max(dim=1)[0]
        f2 = x2.max(dim=1)[0]
        f3 = x3.max(dim=1)[0]
        return self.proj(torch.cat([f1, f2, f3], dim=1))

def edge_mlp(in_channels, out_channels):

    return nn.Sequential(
        nn.Linear(in_channels * 2, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class DGCNNEncoder(nn.Module):

    def __init__(self, latent_dim=128, k=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.k = k

        self.conv1 = DynamicEdgeConv(edge_mlp(4,  32),  k=k, aggr='max')
        self.conv2 = DynamicEdgeConv(edge_mlp(32, 64),  k=k, aggr='max')
        self.conv3 = DynamicEdgeConv(edge_mlp(64, 128), k=k, aggr='max')

        self.proj = nn.Sequential(
            nn.Linear(32+64+128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, points):

        B, N, _ = points.shape
        device = points.device

        x     = points.view(B * N, 4)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        mask  = x[:, 2] > 0
        if mask.sum() < self.k:
            return torch.zeros(B, self.latent_dim, device=device)

        x_active     = x[mask]
        batch_active = batch[mask]

        x1 = self.conv1(x_active,     batch_active)
        x2 = self.conv2(x1,           batch_active)
        x3 = self.conv3(x2,           batch_active)

        f1 = global_max_pool(x1, batch_active)
        f2 = global_max_pool(x2, batch_active)
        f3 = global_max_pool(x3, batch_active)

        feat = torch.cat([f1, f2, f3], dim=1)
        return self.proj(feat)
