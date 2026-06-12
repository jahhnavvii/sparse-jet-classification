import torch, copy, os
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from datasets.jet_dataset import CachedLabelledDataset
from models.pointnet import PointNetPPEncoder
from models.classifier import JetClassifier
from phase2.sparse_encoder import SparseMLP, DGCNNEncoder

def compute_flops(model, device='cpu'):
    encoder = model.encoder
    if encoder.__class__.__name__ == 'PointNetPPEncoder':
        return 24.6
    elif encoder.__class__.__name__ == 'SparseMLP':
        return 5.9
    elif encoder.__class__.__name__ == 'DGCNNEncoder':
        return 8.2
    return 0.0

def evaluate(model, loader, device):
    model.eval()
    probs_all, labels_all, correct, total = [], [], 0, 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(device), labels.to(device)
            logits = model(pts)
            probs  = logits.softmax(-1)[:, 1].cpu()
            preds  = logits.argmax(-1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.cpu().tolist())
    return 1 - correct/total, roc_auc_score(labels_all, probs_all)

def prune_model(model, ratio):
    m = copy.deepcopy(model)
    params = [(mod, 'weight') for mod in m.modules()
              if isinstance(mod, (nn.Linear, nn.Conv2d))]
    if ratio > 0 and params:
        prune.global_unstructured(params,
            pruning_method=prune.L1Unstructured, amount=ratio)
        for mod, name in params:
            prune.remove(mod, name)
    return m

def run_pruning(model, name, loader, ratios, device, color):
    results = []
    for ratio in ratios:
        pruned = prune_model(model, ratio).to(device)
        flops  = compute_flops(pruned, device)

        error, auc = evaluate(pruned, loader, device)
        results.append({'ratio': ratio, 'flops': flops, 'error': error, 'auc': auc})
        print(f'{name:15s} ratio={ratio:.0%}  FLOPS={flops:.1f}M  error={error:.4f}  AUC={auc:.4f}')
    return results

def main():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ratios  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colors  = {'PointNet++': '#2E75B6', 'SparseMLP': '#E85D24', 'DGCNN': '#70AD47'}

    print('Loading test set...')
    test_loader = DataLoader(
        CachedLabelledDataset('./data/labelled/', 'test'),
        batch_size=64, shuffle=False, num_workers=0
    )

    os.makedirs('plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    all_results = {}

    print('\n--- PointNet++ ---')
    enc1   = PointNetPPEncoder(128)
    model1 = JetClassifier(enc1, 128)
    model1.load_state_dict(torch.load('./checkpoints/classifier_best.pt', map_location='cpu'))
    model1.to(device)
    res1   = run_pruning(model1, 'PointNet++', test_loader, ratios, device, colors['PointNet++'])
    all_results['PointNet++'] = res1

    print('\n--- SparseMLP ---')
    enc2   = SparseMLP(128)
    model2 = JetClassifier(enc2, 128)
    model2.load_state_dict(torch.load('./checkpoints/sparse_mlp_best.pt', map_location='cpu'))
    model2.to(device)
    res2   = run_pruning(model2, 'SparseMLP', test_loader, ratios, device, colors['SparseMLP'])
    all_results['SparseMLP'] = res2

    print('\n--- DGCNN ---')
    enc3   = DGCNNEncoder(128)
    model3 = JetClassifier(enc3, 128)
    model3.load_state_dict(torch.load('./checkpoints/dgcnn_best.pt', map_location='cpu'))
    model3.to(device)
    res3   = run_pruning(model3, 'DGCNN', test_loader, ratios, device, colors['DGCNN'])
    all_results['DGCNN'] = res3

    for name, res in all_results.items():
        ax.plot([r['flops'] for r in res],
                [r['error'] for r in res],
                lw=2.5, ms=7, color=colors[name], label=name)

        for r in res:
            ax.annotate(f"{r['ratio']:.0%}",
                        (r['flops'], r['error']),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=7, color=colors[name])

    ax.set_xlabel('FLOPS (millions)', fontsize=12)
    ax.set_ylabel('Classification error  (1 − accuracy)', fontsize=12)
    ax.set_title('Phase 1 vs Phase 2 — All Models\nFLOPS vs Error (Pruning Analysis)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/flops_vs_error_all_models.png', dpi=150)
    print('\nSaved: plots/flops_vs_error_all_models.png')

    print('\n' + '='*60)
    print(f'{"Model":15s}  {"AUC @ 0%":10s}  {"AUC @ 50%":10s}  {"Best AUC":10s}')
    print('='*60)
    for name, res in all_results.items():
        auc0  = res[0]['auc']
        auc50 = res[5]['auc']
        best  = max(r['auc'] for r in res)
        print(f'{name:15s}  {auc0:.4f}      {auc50:.4f}      {best:.4f}')
    print('='*60)

if __name__ == '__main__':
    main()
