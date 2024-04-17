import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Define the Autoencoder Network
class Autoencoder(pl.LightningModule):
    def __init__(self, n_features, reduced_dim):
        super().__init__()
        self.n_features = n_features
        self.reduced_dim = reduced_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 12),
            nn.ReLU(),
            nn.Linear(12, reduced_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(reduced_dim, 12),
            nn.ReLU(),
            nn.Linear(12, n_features)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def transform(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            x_enc = self.encoder(x)
            
        return x_enc.numpy()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create the Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

def fit(observations, autoencoder):
    dataset = CustomDataset(observations)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the autoencoder and trainer
    # autoencoder = Autoencoder()
    trainer = pl.Trainer(max_epochs=10)

    # Train the autoencoder
    trainer.fit(autoencoder, train_loader)
    
