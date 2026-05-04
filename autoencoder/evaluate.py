import torch
import matplotlib.pyplot as plt
from models.vanilla_vae import VanillaVAE
from datasets.stamps_dataset import get_dataloader
import tfg_stamps.autoencoder.config as config


device = torch.device("cpu")

model = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
model.load_state_dict(torch.load("vanilla_vae_stamps.pt", map_location=device))
model.eval()

loader = get_dataloader(config.DATA_PATH, batch_size=8, image_size=config.IMAGE_SIZE, shuffle=True, num_workers=0)
x, _ = next(iter(loader))

with torch.no_grad():
    recon, _, _, _ = model(x)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(x[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original", loc="left")
axes[1, 0].set_title("Reconstructed", loc="left")
plt.tight_layout()
plt.savefig("reconstructions.png")
print("Saved reconstructions.png")

# ---- Generate NEW stamps (sampling) ----
num_samples = 16

with torch.no_grad():
    z = torch.randn(num_samples, config.LATENT_DIM).to(device)
    samples = model.decode(z)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    axes[i // 8, i % 8].imshow(samples[i].squeeze(), cmap="gray")
    axes[i // 8, i % 8].axis("off")

plt.suptitle("Generated Stamps (VAE Samples)")
plt.tight_layout()
plt.savefig("generated_stamps.png")
print("Saved generated_stamps.png")

# ---- Latent interpolation ----
z1 = torch.randn(1, config.LATENT_DIM).to(device)
z2 = torch.randn(1, config.LATENT_DIM).to(device)

alphas = torch.linspace(0, 1, steps=10).to(device)
interpolations = []

with torch.no_grad():
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        img = model.decode(z)
        interpolations.append(img)

interpolations = torch.cat(interpolations)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(interpolations[i].squeeze(), cmap="gray")
    axes[i].axis("off")

plt.suptitle("Latent Space Interpolation")
plt.savefig("latent_interpolation.png")