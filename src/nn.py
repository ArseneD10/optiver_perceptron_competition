import torch 
import torch.nn as nn
from src.aggregation import ManualAggregation

class Decoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, output_dim),
        )

        self.json_kwargs = {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim}

    def forward(self, x):
        return self.decoder(x)


class VAEEncoder(nn.Module):

    def __init__(self):
        super().__init__()    
        self.json_kwargs = {}

        self.cls = nn.Parameter(torch.randn())



class AggregationAE(nn.Module):

    def __init__(self, encoder: ManualAggregation | VAEEncoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent, mu, logvar = self.encoder(x)
        latent = latent.transpose(1, 2)
        return self.decoder(latent).transpose(1, 2), mu, logvar
    
    def encode(self, x):
        latent, _, _ = self.encoder(x)
        return latent


def train_manual_encoder(model: AggregationAE, optimizer: torch.optim.Optimizer, loader, device: str='mps'):
    model.train()
    model.to(device)
    mse = nn.MSELoss()
    running_loss = torch.tensor([0.], device=device)

    for (X, ) in loader:
        X = X.to(device)
        recon, _, _ = model(X)
        loss = mse(recon, X)
        running_loss += loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.item() / len(loader)

@torch.no_grad()
def test_manual_encoder(model: AggregationAE, loader, device: str='mps'):
    model.train()
    model.to(device)
    mse = nn.MSELoss()
    running_loss = torch.tensor([0.], device=device)
    for (X, ) in loader:
        X = X.to(device)
        recon, _, _ = model(X)
        loss = mse(recon, X)
        running_loss += loss 
    return running_loss.item() / len(loader)