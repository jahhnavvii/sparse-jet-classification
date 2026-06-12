import torch

def image_to_points(img: torch.Tensor, max_points: int = 512) -> torch.Tensor:

    energy_map = img.max(dim=-1).values
    track_pt   = img[:, :, 1]

    H, W = energy_map.shape
    rows, cols = torch.where(energy_map > 0)

    if len(rows) == 0:
        return torch.zeros(max_points, 4)

    energy  = energy_map[rows, cols]
    track   = track_pt[rows, cols]

    rows_n   = rows.float()  / (H - 1)
    cols_n   = cols.float()  / (W - 1)
    energy_n = energy        / (energy.max() + 1e-8)
    track_n  = track         / (track.max()  + 1e-8) if track.max() > 0 else track

    points = torch.stack([rows_n, cols_n, energy_n, track_n], dim=1)

    N = points.shape[0]
    if N >= max_points:
        idx    = torch.argsort(energy_n, descending=True)[:max_points]
        points = points[idx]
    else:
        pad    = torch.zeros(max_points - N, 4)
        points = torch.cat([points, pad], dim=0)

    return points
def augment_points(points: torch.Tensor, training: bool = True) -> torch.Tensor:

    if not training:
        return points

    active = points[points[:, 2] > 0]
    if len(active) == 0:
        return points

    if torch.rand(1) > 0.5:
        angle = torch.rand(1) * 2 * 3.14159
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        cx = active[:, 0] - 0.5
        cy = active[:, 1] - 0.5
        active[:, 0] = (cos_a * cx - sin_a * cy + 0.5).clamp(0, 1)
        active[:, 1] = (sin_a * cx + cos_a * cy + 0.5).clamp(0, 1)

    if torch.rand(1) > 0.5:
        noise = 1.0 + (torch.rand_like(active[:, 2]) - 0.5) * 0.1
        active[:, 2] = (active[:, 2] * noise).clamp(0, 1)

    if torch.rand(1) > 0.5:
        keep = torch.rand(len(active)) < (0.9 + torch.rand(1) * 0.1)
        active = active[keep]

    N = points.shape[0]
    if len(active) == 0:
        return points
    if len(active) >= N:
        return active[:N]
    pad = torch.zeros(N - len(active), 4)
    return torch.cat([active, pad], dim=0)
