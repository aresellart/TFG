import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from models.vanilla_vae_v2 import VanillaVAE
from datasets_local.stamps_dataset import get_dataloader
from autoencoder_v3.config_v3 import DATA_PATH, LATENT_DIM, IMAGE_SIZE

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "vae_gan_finetune_best.pt")

device = torch.device("cpu")

model = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded {MODEL_PATH}")

loader = get_dataloader(
    DATA_PATH, batch_size=8,
    image_size=IMAGE_SIZE, shuffle=True, num_workers=0
)
x, _ = next(iter(loader))

# ── 1. Reconstructions ────────────────────────────────────────────────────────
with torch.no_grad():
    recon, _, _, _ = model(x)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original",      fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle("VAE-GAN — reconstructions", fontsize=12)
plt.tight_layout()
plt.savefig("reconstructions_vae_gan.png", dpi=150)
print("Saved reconstructions_vae_gan.png")

# ── 2. Random sampling ────────────────────────────────────────────────────────
with torch.no_grad():
    z      = torch.randn(16, LATENT_DIM)
    samples = model.decode(z)

fig2, axes2 = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes2[i//8, i%8].imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes2[i//8, i%8].axis("off")
plt.suptitle("VAE-GAN — random sampling from N(0,I)", fontsize=12)
plt.tight_layout()
plt.savefig("sampling_vae_gan.png", dpi=150)
print("Saved sampling_vae_gan.png")

# ── 3. Interpolation ──────────────────────────────────────────────────────────
with torch.no_grad():
    z1     = torch.randn(1, LATENT_DIM)
    z2     = torch.randn(1, LATENT_DIM)
    alphas = torch.linspace(0, 1, steps=10)
    interps = torch.cat([model.decode((1-a)*z1 + a*z2) for a in alphas])

fig3, axes3 = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes3[i].imshow(interps[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes3[i].axis("off")
plt.suptitle("VAE-GAN — latent interpolation", fontsize=12)
plt.tight_layout()
plt.savefig("interpolation_vae_gan.png", dpi=150)
print("Saved interpolation_vae_gan.png")