import os
import json
import torch 
import torch.nn as nn
from src.aggregation import ManualAggregation, AggregationFunction
import numpy as np


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


class ResidualAttentionLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class VAEEncoder(nn.Module):

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.json_kwargs = {"dim": dim, "depth": depth}
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.layers = nn.ModuleList([ResidualAttentionLayer(dim) for _ in range(depth)])
        self.logvar = nn.Linear(dim, dim)
        self.mu = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        x = self.reparameterize(mu, logvar)
        return x, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_vae_encoder(model: AggregationAE, optimizer: torch.optim.Optimizer, loader, device: str='mps', beta= 0.7):
    model.train()
    model.to(device)
    mse = nn.MSELoss()
    running_loss = torch.tensor([0.], device=device)

    for (X, ) in loader:
        X = X.to(device)
        recon, mu, logvar = model(X)
        loss = mse(recon, X) + kl_divergence(mu, logvar) * beta
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.item() / len(loader)

@torch.no_grad()
def test_vae_encoder(model: AggregationAE, loader, device: str='mps', beta= 0.7):
    model.train()
    model.to(device)
    mse = nn.MSELoss()
    running_loss = torch.tensor([0.], device=device)
    for (X, ) in loader:
        X = X.to(device)
        recon, mu, logvar = model(X)
        loss = mse(recon, X) + kl_divergence(mu, logvar) * beta
        running_loss += loss
    return running_loss.item() / len(loader)


def evaluate_and_register_phase_1_1_vae(model, train_loader, valid_loader, _dir: str = "evaluation_1_1_vae", weight_index: int = 0, verbose=True, beta=0.7):
    os.makedirs(f"{_dir}/weights", exist_ok=True)
    weight_path = f"{_dir}/weights/vae_{weight_index}.pt"
    OPTIM = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCH = 10
    DEVICE = "mps"
    BEST_LOSS = torch.inf
    BEST_WEIGHT = model.state_dict()
    if verbose:
        print("--" * 40)
        print("RUN FOR INDEX - ", weight_index)

    for e in range(1, EPOCH + 1):
        train_loss = train_vae_encoder(model, OPTIM, train_loader, DEVICE, beta)
        valid_loss = test_vae_encoder(model, valid_loader, DEVICE, beta)
        if valid_loss < BEST_LOSS:
            BEST_LOSS = valid_loss
            BEST_WEIGHT = model.state_dict()
        if verbose and e % 10 == 0:
            print(f"EPOCH {e} - TRAIN LOSS: {train_loss} | VALID LOSS: {valid_loss}")

    torch.save(BEST_WEIGHT, weight_path)
    result = {
        "score": BEST_LOSS,
        "encoder_dim": model.encoder.json_kwargs,
        "decoder_dim": model.decoder.json_kwargs,
        "weight_path": weight_path,
    }

    with open(os.path.join(_dir, f"candidate_{weight_index}.json"), "w") as f:
        json.dump(result, f)

    return result, BEST_WEIGHT


def run_vae_selection_phase_1_1(candidates, train_loader, valid_loader, k: int = 5, _dir: str = "evaluation_1_1_vae", verbose=True, hidden_dim: int = 32, beta=0.7):
    sample_x = next(iter(train_loader))[0]
    window = sample_x.shape[1]
    results = []
    for index in range(len(candidates)):
        candidate = candidates[index]
        kwargs = candidate if isinstance(candidate, dict) else {"dim": candidate[0], "depth": candidate[1]}
        decoder = Decoder(input_dim=window + 1, hidden_dim=hidden_dim, output_dim=window)
        encoder = VAEEncoder(**kwargs)
        model = AggregationAE(encoder, decoder)
        result, _ = evaluate_and_register_phase_1_1_vae(model, train_loader, valid_loader, _dir, index, verbose, beta)
        results.append(result)

    results = sorted(results, key=lambda x: x["score"])
    return results[:k]


def load_manual_autoencoder(file: str):
    info = json.load(open(file, "r"))
    decoder = Decoder(**info["decoder_dim"])
    encoder = ManualAggregation([AggregationFunction(**f) for f in info["function"]])
    model = AggregationAE(encoder, decoder)
    model.load_state_dict(torch.load(info["weight_path"]))
    return model

def load_vae_autoencoder(file: str):
    info = json.load(open(file, "r"))
    decoder = Decoder(**info["decoder_dim"])
    encoder = VAEEncoder(**info["encoder_dim"])
    model = AggregationAE(encoder, decoder)
    model.load_state_dict(torch.load(info["weight_path"]))
    return model

@torch.inference_mode()
def get_latent(model: AggregationAE, loader, device: str = "cpu"):
    latents = []
    model.eval()
    model.to(device)
    for (X, ) in loader:
        X = X.to(device)
        latent = model.encode(X).squeeze(1)
        latents.append(latent.cpu().numpy())
    return np.concat(latents)