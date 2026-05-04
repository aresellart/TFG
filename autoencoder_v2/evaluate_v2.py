import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from models.vanilla_vae_v2 import VanillaVAE
from datasets_local.stamps_dataset import get_dataloader
from autoencoder_v2.config_v2 import DATA_PATH, LATENT_DIM, IMAGE_SIZE

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "vae_high_kl_best.pt")

device = torch.device("cpu")

model = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

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
axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle("High KL VAE — reconstructions", fontsize=12)
plt.tight_layout()
plt.savefig("reconstructions_high_kl.png", dpi=150)
print("Saved reconstructions_high_kl.png")

# ── 2. Random sampling from N(0,I) ────────────────────────────────────────────
with torch.no_grad():
    z_random = torch.randn(16, LATENT_DIM)
    samples  = model.decode(z_random)

fig2, axes2 = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes2[i//8, i%8].imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes2[i//8, i%8].axis("off")
plt.suptitle("High KL VAE — random sampling from N(0,I)", fontsize=12)
plt.tight_layout()
plt.savefig("sampling_high_kl.png", dpi=150)
print("Saved sampling_high_kl.png")

# ── 3. Latent space interpolation ─────────────────────────────────────────────
with torch.no_grad():
    z1 = torch.randn(1, LATENT_DIM)
    z2 = torch.randn(1, LATENT_DIM)
    alphas = torch.linspace(0, 1, steps=10)
    interps = torch.cat([model.decode((1-a)*z1 + a*z2) for a in alphas])

fig3, axes3 = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes3[i].imshow(interps[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes3[i].axis("off")
plt.suptitle("High KL VAE — latent interpolation between two random stamps", fontsize=12)
plt.tight_layout()
plt.savefig("interpolation_high_kl.png", dpi=150)
print("Saved interpolation_high_kl.png")

# ── 4. Side by side comparison summary ───────────────────────────────────────
print("\n=== Summary ===")
print("Check these three images:")
print("  reconstructions_high_kl.png  — are reconstructions blurrier than your v1?")
print("  sampling_high_kl.png         — are random samples diverse and stamp-like?")
print("  interpolation_high_kl.png    — is the interpolation smooth?")
print("\nIf sampling looks good → the latent space is now Gaussian")
print("If reconstructions are blurry → expected tradeoff with high KL")