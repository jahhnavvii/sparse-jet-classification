import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QNode:
    x: int
    y: int
    size: int
    depth: int = 0
    children: List = field(default_factory=list)
    patch: Optional[torch.Tensor] = None

def build_quadtree(img, x, y, size, min_size=8, depth=0):

    h, w = img.shape
    x_end = min(x + size, w)
    y_end = min(y + size, h)
    patch = img[y:y_end, x:x_end]

    if patch.numel() == 0 or patch.max() == 0:
        return None

    if size <= min_size or depth >= 4:
        return QNode(x, y, size, depth,
                     patch=patch)

    half = size // 2
    children = [
        build_quadtree(img, x,      y,      half, min_size, depth+1),
        build_quadtree(img, x+half, y,      half, min_size, depth+1),
        build_quadtree(img, x,      y+half, half, min_size, depth+1),
        build_quadtree(img, x+half, y+half, half, min_size, depth+1),
    ]
    children = [c for c in children if c is not None]
    return QNode(x, y, size, depth, children=children)

def get_leaves(node):

    if node is None: return []
    if node.patch is not None: return [node]
    leaves = []
    for child in node.children:
        leaves.extend(get_leaves(child))
    return leaves

def get_active_points(node):

    leaves = get_leaves(node)
    if not leaves:
        return torch.zeros(1, 3)

    all_rows, all_cols, all_energy = [], [], []
    H = W = 125.0

    for leaf in leaves:
        patch = leaf.patch
        rows, cols = torch.where(patch > 0)
        energy = patch[rows, cols]

        global_rows = (rows + leaf.y).float() / (H - 1)
        global_cols = (cols + leaf.x).float() / (W - 1)
        energy_n    = energy / (energy.max() + 1e-8)
        all_rows.append(global_rows)
        all_cols.append(global_cols)
        all_energy.append(energy_n)

    rows   = torch.cat(all_rows)
    cols   = torch.cat(all_cols)
    energy = torch.cat(all_energy)
    return torch.stack([rows, cols, energy], dim=1)
