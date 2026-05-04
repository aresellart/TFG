import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from .base_vae import BaseVAE


# ── Residual block ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 256,
                 hidden_dims: List[int] = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim  = latent_dim
        self.hidden_dims = list(hidden_dims) if hidden_dims else [32, 64, 128, 256, 512, 512]

        # ── Encoder ──────────────────────────────────────────────────────────
        modules = []
        ch = in_channels
        for h_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(h_dim),   # ← residual refinement at each scale
            ))
            ch = h_dim
        self.encoder = nn.Sequential(*modules)

        # 6 stride-2 convs on 128×128 → 2×2 feature map
        self._flat = self.hidden_dims[-1] * 2 * 2
        self.fc_mu  = nn.Linear(self._flat, latent_dim)
        self.fc_var = nn.Linear(self._flat, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder_input = nn.Linear(latent_dim, self._flat)

        dec_dims = list(reversed(self.hidden_dims))
        modules  = []
        for i in range(len(dec_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i+1],
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(dec_dims[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(dec_dims[i+1]),   # ← residual refinement at each scale
            ))
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[-1], dec_dims[-1],
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(dec_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dec_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    # ── Forward ───────────────────────────────────────────────────────────────
    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return [self.fc_mu(h), self.fc_var(h)]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.hidden_dims[-1], 2, 2)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    # ── Loss ──────────────────────────────────────────────────────────────────
    def loss_function(self, *args, **kwargs) -> dict:
        recon, x, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = kwargs.get('M_N', 1.0)

        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kld_loss   = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )
        loss = recon_loss + kld_weight * kld_loss
        return {
            'loss':                loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD':                 kld_loss.detach(),
        }