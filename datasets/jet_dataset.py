import torch
import h5py
import os
from torch.utils.data import Dataset
from utils.sparse import image_to_points

MAX_UNLABELLED = 60_000
MAX_LABELLED   = 10_000

class UnlabelledJetDataset(Dataset):

    def __init__(self, data_dir: str, max_points: int = 512):
        self.max_points = max_points
        self.index = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(('.h5', '.hdf5')): continue
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as f:
                n = min(f['jet'].shape[0], MAX_UNLABELLED)
                self.index.extend([(fpath, i) for i in range(n)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, i = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            img = torch.tensor(f['jet'][i], dtype=torch.float32)
        return image_to_points(img, self.max_points)

class LabelledJetDataset(Dataset):

    def __init__(self, data_dir: str, split: str = 'train', max_points: int = 512):
        assert split in ('train', 'val', 'test')
        self.max_points = max_points
        self.index = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(('.h5', '.hdf5')): continue
            fpath = os.path.join(data_dir, fname)
            with h5py.File(fpath, 'r') as f:
                n  = min(f['jet'].shape[0], MAX_LABELLED)
                lo = {'train': 0,          'val': int(.8*n), 'test': int(.9*n)}[split]
                hi = {'train': int(.8*n),  'val': int(.9*n), 'test': n        }[split]
                self.index.extend([(fpath, i) for i in range(lo, hi)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, i = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            img   = torch.tensor(f['jet'][i], dtype=torch.float32)
            label = int(f['Y'][i][0])
        return image_to_points(img, self.max_points), label

class CachedLabelledDataset(Dataset):

    def __init__(self, data_dir: str, split: str = 'train', max_points: int = 512):
        base = LabelledJetDataset(data_dir, split, max_points)
        print(f'Pre-computing {len(base)} samples for {split}...')
        self.points = []
        self.labels = []
        for i in range(len(base)):
            pts, lbl = base[i]
            self.points.append(pts)
            self.labels.append(lbl)
            if i % 1000 == 0:
                print(f'  {i}/{len(base)}')
        self.points = torch.stack(self.points)
        self.labels = torch.tensor(self.labels)
        print(f'Done! {len(self.points)} samples cached.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx].item()
