import torch
import matplotlib.pyplot as plt

from models.vanilla_vae import VanillaVAE
from datasets.stamps_dataset import get_dataloader
import tfg_stamps.autoencoder.config as config

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
model.load_state_dict(torch.load("vanilla_vae_stamps.pt", map_location=device))
model.eval()

# Load data
loader = get_dataloader(
    config.DATA_PATH,
    batch_size=8,
    image_size=config.IMAGE_SIZE,
    shuffle=True,
    num_workers=0
)

# ---- Get batch ----
x, _ = next(iter(loader))
x = x.to(device)

# ---- Local sampling ----
with torch.no_grad():
    mu, logvar = model.encode(x)

    eps = torch.randn_like(mu) * 0.5   # 🔑 control variation
    z = mu + eps

    samples = model.decode(z)

# ---- Plot ----
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

for i in range(8):
    axes[0, i].imshow(x[i].squeeze().cpu(), cmap="gray")
    axes[0, i].axis("off")

    axes[1, i].imshow(samples[i].squeeze().cpu(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original", loc="left")
axes[1, 0].set_title("Local Samples", loc="left")

plt.tight_layout()
plt.savefig("local_samples.png")
print("Saved local_samples.png")