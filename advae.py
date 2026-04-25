import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import PARAM_KEYS, PARAM_RANGES

ANATOMICAL_DIM = 25
LATENT_DIM     = 48


def normalize_params(params):
    vals = []
    for k in PARAM_KEYS:
        lo, hi = PARAM_RANGES[k]
        vals.append((params[k] - lo) / (hi - lo) * 2 - 1)
    return torch.tensor(vals, dtype=torch.float32)


def denormalize_params(tensor):
    out = {}
    for i, k in enumerate(PARAM_KEYS):
        lo, hi = PARAM_RANGES[k]
        v = float((tensor[i].item() + 1) / 2 * (hi - lo) + lo)
        out[k] = v
    return out


class ADVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, anatomical_dim=ANATOMICAL_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.anatomical_dim = anatomical_dim

        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv3d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(64*4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(64*4*4*4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(16, 1,  4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 4, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def advae_loss(recon, x, mu, logvar, true_params, beta=0.5, lambda_anat=30.0):
    B = x.size(0)
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / B
    k_all = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_anat = k_all[:, :true_params.size(1)].sum() / B
    kld_free = k_all[:,  true_params.size(1):].sum() / B
    kld = 0.05 * kld_anat + kld_free
    anat_loss = F.mse_loss(mu[:, :true_params.size(1)], true_params)
    return recon_loss + beta*kld + lambda_anat*anat_loss, recon_loss, kld, anat_loss