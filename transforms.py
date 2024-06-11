import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Define the Autoencoder Network
class Autoencoder(pl.LightningModule):
    """
    Autoencoder neural network for dimensionality reduction.
    
    Parameters:
    n_features (int): The number of features in the input data.
    reduced_dim (int): The number of dimensions for the reduced representation.
    """
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
        """
        Forward pass through the autoencoder.
        
        Parameters:
        x (torch.Tensor): The input data.
        
        Returns:
        torch.Tensor: The reconstructed data.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def transform(self, x):
        """
        Transform the input data to its encoded representation.
        
        Parameters:
        x (numpy.ndarray): The input data.
        
        Returns:
        numpy.ndarray: The encoded data.
        """
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            x_enc = self.encoder(x)
        return x_enc.numpy()

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        
        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): The index of the batch.
        
        Returns:
        torch.Tensor: The loss value for the batch.
        """
        x, _ = batch
        reconstructed = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
        torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create the Dataset
class CustomDataset(Dataset):
    """
    Custom dataset for loading observations.
    
    Parameters:
    data (numpy.ndarray): The observations data.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
        int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Parameters:
        idx (int): The index of the sample.
        
        Returns:
        tuple: A tuple containing the sample and its corresponding label (the same sample).
        """
        return self.data[idx], self.data[idx]

def fit(observations, autoencoder):
    """
    Fit the autoencoder to the observations data.
    
    Parameters:
    observations (numpy.ndarray): The observations data.
    autoencoder (Autoencoder): The autoencoder model to be trained.
    """
    dataset = CustomDataset(observations)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=30, accelerator="cpu")

    # Train the autoencoder
    trainer.fit(autoencoder, train_loader)


class CascadingAutoEncoder(pl.LightningModule):
    def __init__(self, n_features, layers_sizes):
        super().__init__()
        self.layers_sizes =  [n_features] + layers_sizes
        encoder_layers = []
        decoder_layers = []

        """
        creating encoder layers
        """
        for i in range(len(self.layers_sizes) - 1):
            encoder_layers.append(nn.Linear(self.layers_sizes[i],self.layers_sizes[i+1]))
            encoder_layers.append(nn.ReLU())


        """
        create the decoder layers in reverse
        """

        for i in range(len(self.layers_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(self.layers_sizes[i], self.layers_sizes[i - 1]))
            decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers[:-1])
        self.decoder = nn.Sequential(*decoder_layers[:-1])

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    
    

def fit_cascading_autoencoder(observations, autoencoder):
    dataset = CustomDataset(observations)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    trainer = pl.Trainer(max_epochs=30, accelerator="cpu")
    trainer.fit(autoencoder, train_loader)
