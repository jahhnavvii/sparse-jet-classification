import torch, os, argparse
from torch.utils.data import DataLoader
from datasets.jet_dataset import UnlabelledJetDataset
from models.autoencoder import JetAutoencoder, chamfer_loss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pretraining on {device}')

    loader = DataLoader(
        UnlabelledJetDataset(args.data_dir, args.max_points),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )

    model = JetAutoencoder(args.latent_dim, args.max_points).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for pts in loader:
            pts          = pts.to(device)
            recon, _     = model(pts)
            loss         = chamfer_loss(recon, pts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total       += loss.item()
        sched.step()
        avg = total / len(loader)
        print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={avg:.4f}')
        if avg < best:
            best = avg
            torch.save(model.encoder.state_dict(),
                       os.path.join(args.ckpt_dir, 'encoder_best.pt'))
            print(f'  → saved encoder (loss={best:.4f})')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True)
    p.add_argument('--ckpt_dir',   default='./checkpoints')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_points', type=int, default=512)
    p.add_argument('--latent_dim', type=int, default=128)
    train(p.parse_args())
