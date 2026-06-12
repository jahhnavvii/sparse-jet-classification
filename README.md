# End-to-End Event Classification with Sparse Neural Networks
### GSoC 2026 — ML4SCI / CMS — Task 2d
**Devarpalli Jahnavi** · [jahhnavvii@gmail.com](mailto:jahhnavvii@gmail.com) · [github.com/jahhnavvii](https://github.com/jahhnavvii)

---

## Overview

CMS calorimeter images are 125×125 pixels across 8 detector channels — but on average only 2–3% of pixels carry any energy signal. Dense neural networks waste 97% of their computation on empty regions.

This project builds a two-phase classification pipeline for quark/gluon jet discrimination that only processes where data actually exists. The result: a sparse architecture that is simultaneously faster **and** more accurate than the dense baseline.

---

## Results

| Model | Test AUC | FLOPS | vs Baseline |
|---|---|---|---|
| PointNet++ (Phase 1) | 0.7648 | 24.6M | baseline |
| **SparseMLP (Phase 2)** | **0.7955** | **5.9M** | **4.2× fewer FLOPS, +3.9% AUC** |
| DGCNN (Phase 2) | 0.7738 | 8.2M | 3.0× fewer FLOPS, +1.2% AUC |

**Key finding:** SparseMLP at 50% pruning (AUC 0.7969) still outperforms unpruned PointNet++ (AUC 0.7648).

---

## Repository Structure

```
E2E_SparseNN_Jahhnavvii/
│
├── utils/
│   └── sparse.py               # image_to_points() — converts image to (512, 4) point cloud
│
├── datasets/
│   └── jet_dataset.py          # UnlabelledJetDataset, LabelledJetDataset, CachedLabelledDataset
│
├── models/
│   ├── pointnet.py             # FPS, ball query, Set Abstraction, PointNetPPEncoder
│   ├── autoencoder.py          # FoldingDecoder, JetAutoencoder, chamfer_loss
│   └── classifier.py          # JetClassifier — plug-and-play head for any encoder
│
├── phase2/
│   ├── quadtree.py             # Recursive quadtree — prunes empty image regions
│   ├── sparse_encoder.py       # SparseMLP + DGCNNEncoder — both output (B, 128)
│   └── train_hybrid.py         # Phase 2 training script
│
├── notebooks/
│   ├── Phase1_Results.ipynb    # Phase 1 evaluation — ROC, confusion matrix, pruning
│   └── Phase2_Results.ipynb    # Phase 2 evaluation — all 3 models compared
│
├── plots/
│   ├── flops_vs_error_all_models.png   # Required deliverable
│   ├── roc_comparison_all_models.png
│   └── confusion_all_models.png
│
├── train_pretrain.py           # Unsupervised pretraining — Chamfer Distance loss
├── train_finetune.py           # Supervised fine-tuning — augmentation + cosine LR
└── prune_and_plot.py           # Pruning sweep + FLOPS vs error plot
```

---

## How It Works

### Input Representation
Every 125×125 image is converted to a point cloud before entering any model. Only non-zero pixels are extracted. Each active pixel becomes a 4-dimensional point:

```
(row_norm, col_norm, energy_norm, track_pT_norm)
```

- **energy** = max across all 8 detector channels
- **track_pT** = channel 1 separately — captures charged particle momentum, physically distinct from energy deposition

This reduces input from 125,000 values to ~900 values — 140× compression, zero information loss.

### Phase 1 — PointNet++ Baseline
1. **Unsupervised pretraining** on 60,000 unlabelled events using Chamfer Distance loss. FoldingDecoder ([1]) reconstructs the point cloud from a 128-d latent vector — no labels needed.
2. **Supervised fine-tuning** on 8,000 labelled events with data augmentation (rotation, energy jitter, point dropout), BatchNorm regularisation, and cosine LR scheduling.

### Phase 2 — Quadtree-Guided Sparse Architecture
The quadtree recursively subdivides the image and immediately prunes empty quadrants. This is physically motivated — particle jet decays follow hierarchical tree-like structures, consistent with the anti-kt jet clustering algorithm used as the standard benchmark in CMS analyses.

**SparseMLP** — three-layer shared MLP applied only to active pixel coordinates. Multi-scale max-pooling at three depths. Implements the core principle of [2]: compute only where data exists.

**DGCNN** — dynamic graph CNN that builds k-nearest-neighbour edges between active pixels. Graph rebuilt in feature space after every layer.

Both encoders share identical interface: `(B, N, 4) → (B, 128)`.

---

## Reproduce

### Setup
```bash
conda create -n cms_sparse python=3.10
conda activate cms_sparse
pip install torch==2.6.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip install h5py scikit-learn matplotlib seaborn pandas fvcore
```

### Data
Place datasets in:
```
data/unlabelled/Dataset_Specific_Unlabelled.h5
data/labelled/Dataset_Specific_labelled_full_only_for_2i.h5
```

### Training
```bash
# Phase 1 — pretrain encoder
python train_pretrain.py --data_dir ./data/unlabelled --epochs 50 --batch_size 32

# Phase 1 — fine-tune classifier
python train_finetune.py --data_dir ./data/labelled \
    --checkpoint ./checkpoints/encoder_best.pt \
    --epochs 50 --batch_size 16

# Phase 2 — SparseMLP
python phase2/train_hybrid.py --model sparse_mlp \
    --data_dir ./data/labelled --epochs 50 --batch_size 16

# Phase 2 — DGCNN
python phase2/train_hybrid.py --model dgcnn \
    --data_dir ./data/labelled --epochs 50 --batch_size 16

# Pruning plot (required deliverable)
python prune_and_plot.py
```

### Notebooks
```bash
jupyter notebook notebooks/Phase1_Results.ipynb
jupyter notebook notebooks/Phase2_Results.ipynb
```

---

## Completed Tasks

| Task | Status |
|---|---|
| Train model on unlabelled dataset |  PointNet++ autoencoder — Chamfer loss 0.053 → 0.011 |
| Fine-tune encoder for classification |  AUC 0.765 (PointNet++), 0.796 (SparseMLP) |
| Prune model + FLOPS vs error plot |  3 models × 10 pruning ratios |
| Bonus: sparse autoencoder benchmark |  SparseMLP 4.2× fewer FLOPS at higher accuracy |

---

## References

- **[1]** FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation — Yang et al., [arxiv 1712.07262](https://arxiv.org/abs/1712.07262)
  → implemented as `FoldingDecoder` in `models/autoencoder.py`

- **[2]** Submanifold Sparse Convolutional Networks — Graham & van der Maaten, [arxiv 1706.01307](https://arxiv.org/abs/1706.01307)
  → implemented as `SparseMLP` in `phase2/sparse_encoder.py`

---

*Branch: `sparse-nn-jahhnavvii` · Folder: `E2E_SparseNN_Jahhnavvii/` · Do not submit PR to main repo.*
