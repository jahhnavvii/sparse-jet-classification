import torch.nn as nn

class JetClassifier(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_classes=2, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, points):
        return self.head(self.encoder(points))
