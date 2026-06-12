import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.jet_dataset import CachedLabelledDataset
from models.classifier import JetClassifier
from phase2.sparse_encoder import SparseMLP, DGCNNEncoder

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    tr_ld = DataLoader(CachedLabelledDataset(args.data_dir, 'train', 512),
                       batch_size=args.batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(CachedLabelledDataset(args.data_dir, 'val',   512),
                       batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == 'sparse_mlp':
        encoder = SparseMLP(latent_dim=128)
        print('Model: SparseMLP (Pure PyTorch Sparse)')
    else:
        encoder = DGCNNEncoder(latent_dim=128)
        print('Model: DGCNN (PyTorch Geometric)')

    if args.checkpoint and os.path.exists(args.checkpoint):

        state = torch.load(args.checkpoint, map_location='cpu')
        try:
            encoder.load_state_dict(state, strict=False)
            print(f'Loaded pretrained weights from {args.checkpoint}')
        except:
            print('Could not load pretrained weights — training from scratch')

    model     = JetClassifier(encoder, latent_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(),    'lr': 1e-3},
    ], weight_decay=1e-4)

    best_auc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):

        model.train()
        for pts, labels in tr_ld:
            pts, labels = pts.to(device), labels.to(device)
            loss = criterion(model(pts), labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for pts, labels in va_ld:
                probs = model(pts.to(device)).softmax(-1)[:, 1].cpu()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.tolist())

        auc = roc_auc_score(all_labels, all_probs)
        print(f'Epoch {epoch+1:3d}/{args.epochs}  val_AUC={auc:.4f}')

        if auc > best_auc:
            best_auc = auc
            ckpt = os.path.join(args.ckpt_dir, f'{args.model}_best.pt')
            torch.save(model.state_dict(), ckpt)
            print(f'  → saved (AUC={best_auc:.4f})')

    print(f'Done! Best AUC: {best_auc:.4f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',      choices=['sparse_mlp','dgcnn'], default='sparse_mlp')
    p.add_argument('--data_dir',   required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--ckpt_dir',   default='./checkpoints')
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    train(p.parse_args())
