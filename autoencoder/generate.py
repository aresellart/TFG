import torch
import matplotlib.pyplot as plt
from models.vanilla_vae import VanillaVAE
from datasets.stamps_dataset import get_dataloader
import tfg_stamps.autoencoder.config as config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# At the top of generate.py, replace config references with hardcoded values
LATENT_DIM = 256
IMAGE_SIZE  = 128

model = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load("vanilla_vae_stamps.pt", map_location=device))

device = torch.device("cpu")

model = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
model.load_state_dict(torch.load("vanilla_vae_stamps.pt", map_location=device))
model.eval()

loader = get_dataloader(
    config.DATA_PATH, batch_size=16,
    image_size=config.IMAGE_SIZE, shuffle=True, num_workers=0
)

# Encode real stamps to get meaningful z vectors
x, _ = next(iter(loader))
x = x.to(device)

with torch.no_grad():
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)   # z from real stamps
    generated = model.decode(z)            # decode them back

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.suptitle("Generated stamps (encoded from real stamps)", y=1.01)
plt.tight_layout()
plt.savefig("generated_samples.png", dpi=150)
print("Saved generated_samples.png")