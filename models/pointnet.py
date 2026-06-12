import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:

    dist  = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist +=  (src**2).sum(-1, keepdim=True)
    dist +=  (dst**2).sum(-1).unsqueeze(1)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

    B   = points.shape[0]
    vs  = [B] + [1]*(idx.dim()-1)
    rs  = [1] + list(idx.shape[1:])
    bi  = torch.arange(B, device=points.device).view(vs).repeat(rs)
    return points[bi, idx, :]

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:

    B, N, _   = xyz.shape
    device    = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c        = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist     = ((xyz - c)**2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)
    return centroids

def ball_query(radius: float, nsample: int,
               xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:

    B, N, _ = xyz.shape
    S       = new_xyz.shape[1]
    device  = xyz.device
    idx     = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
    sq      = square_distance(new_xyz, xyz)
    idx[sq > radius**2] = N
    idx     = idx.sort(dim=2)[0][:, :, :nsample]
    first   = idx[:, :, 0:1].expand_as(idx)
    idx[idx == N] = first[idx == N]
    return idx

class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample
        layers, last = [], in_channel + 3
        for out_c in mlp:
            layers += [nn.Conv2d(last, out_c, 1, bias=False),
                       nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            last = out_c
        self.mlp_convs  = nn.Sequential(*layers)
        self.out_channels = last

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape
        fps_idx  = farthest_point_sample(xyz, self.npoint)
        new_xyz  = index_points(xyz, fps_idx)
        idx      = ball_query(self.radius, self.nsample, xyz, new_xyz)
        g_xyz    = index_points(xyz, idx)
        g_xyz   -= new_xyz.unsqueeze(2)
        if points is not None:
            grouped = torch.cat([g_xyz, index_points(points, idx)], -1)
        else:
            grouped = g_xyz
        grouped  = self.mlp_convs(grouped.permute(0, 3, 2, 1))
        new_pts  = grouped.max(dim=2)[0].permute(0, 2, 1)
        return new_xyz, new_pts

class PointNetPPEncoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=64, radius=0.2, nsample=32, in_channel=1, mlp=[32, 32, 64])

        self.sa2 = PointNetSetAbstraction(
            npoint=16, radius=0.4, nsample=64, in_channel=64, mlp=[64, 64, 128])
        self.proj = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True)
        )
        self.latent_dim = latent_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        xyz   = points[..., :3]
        feat  = points[..., 3:4]

        xyz1, feat1 = self.sa1(xyz, feat)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        global_feat = feat2.max(dim=1)[0]
        return self.proj(global_feat)
