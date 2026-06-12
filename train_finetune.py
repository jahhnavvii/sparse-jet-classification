import torch, os, argparse
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from datasets.jet_dataset import LabelledJetDataset
from models.pointnet import PointNetPPEncoder
from models.classifier import JetClassifier
from utils.sparse import augment_points

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Fine-tuning on {device}')

    tr_ld = DataLoader(LabelledJetDataset(args.data_dir, 'train', args.max_points),
                       batch_size=args.batch_size, shuffle=True,  num_workers=4)
    va_ld = DataLoader(LabelledJetDataset(args.data_dir, 'val',   args.max_points),
                       batch_size=args.batch_size, shuffle=False, num_workers=4)

    encoder = PointNetPPEncoder(args.latent_dim)
    if args.checkpoint:
        encoder.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        print(f'Loaded pretrained encoder from {args.checkpoint}')

    model = JetClassifier(encoder, args.latent_dim,
                          freeze_encoder=args.freeze_encoder).to(device)

    opt = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(),    'lr': 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    best_auc  = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for pts, labels in tr_ld:

            pts = torch.stack([augment_points(p, training=True) for p in pts])
            pts, labels = pts.to(device), labels.to(device)
            loss = criterion(model(pts), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

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
            torch.save(model.state_dict(), f'{args.ckpt_dir}/classifier_best.pt')
            print(f'  → saved (AUC={best_auc:.4f})')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       required=True)
    p.add_argument('--checkpoint',     default=None)
    p.add_argument('--ckpt_dir',       default='./checkpoints')
    p.add_argument('--epochs',         type=int,  default=30)
    p.add_argument('--batch_size',     type=int,  default=32)
    p.add_argument('--max_points',     type=int,  default=512)
    p.add_argument('--latent_dim',     type=int,  default=128)
    p.add_argument('--freeze_encoder', action='store_true')
    train(p.parse_args())
