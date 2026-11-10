"""
train_gan.py

Plain GAN trainer — Phase 1 ablation baseline.

Replaces the VAE-GAN (train_v3.py) with a simpler architecture:
  - Generator: only the DECODER of VanillaVAE (no encoder, no KL, no reconstruction loss)
  - Discriminator: same as VAE-GAN (models/discriminator.py)

Input to generator: z ~ N(0,1)  (same distribution used in Phase 2)
Output: 128x128 grayscale stamp image

Saved weights:
  gan_generator_best.pt   ← use instead of vae_gan_best.pt in Phase 2
  gan_disc_best.pt        ← use instead of disc_gan_best.pt in Phase 2

Usage:
    cd /home/asellart/tfg_stamps
    python3 phase1_generative/plain_gan/train_gan.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from models.vanilla_vae_v2 import VanillaVAE
from models.discriminator import Discriminator
from common.stamps_dataset import get_dataloader
from phase1_generative.vaegan.config_v3 import (
    DATA_PATH, BATCH_SIZE, LATENT_DIM, IMAGE_SIZE,
    LR_VAE, LR_DISC, EPOCHS,
    WANDB_PROJECT
)

WANDB_RUN_NAME = "plain-gan-v2"
SAVE_DIR       = "/home/asellart/tfg_stamps"


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "latent_dim":  LATENT_DIM,
            "image_size":  IMAGE_SIZE,
            "batch_size":  BATCH_SIZE,
            "lr_gen":      LR_VAE,
            "lr_disc":     LR_DISC,
            "epochs":      EPOCHS,
            "architecture": "plain_gan",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    loader = get_dataloader(
        DATA_PATH, BATCH_SIZE,
        image_size=IMAGE_SIZE, num_workers=0,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    # Generator: we reuse VanillaVAE but only ever call .decode()
    # The encoder weights are initialised but never updated — only decoder trains
    vae  = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    disc = Discriminator(in_channels=1).to(device)

    # Only optimise the DECODER parameters
    decoder_params = list(vae.decoder_input.parameters()) + \
                     list(vae.decoder.parameters()) + \
                     list(vae.final_layer.parameters())

    opt_gen  = optim.Adam(decoder_params,    lr=1e-4,  betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=5e-5, betas=(0.5, 0.999))

    best_gen_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        vae.train()
        disc.train()

        total_gen_loss  = 0.0
        total_disc_loss = 0.0
        disc_updates    = 0

        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            B = x.size(0)

            z = torch.randn(B, LATENT_DIM, device=device)

            # ── Step 1: Train discriminator (every 5 batches) ─────────────
            if batch_idx % 3 == 0:
                opt_disc.zero_grad()

                with torch.no_grad():
                    fake = vae.decode(z)

                real_preds = disc(x)
                fake_preds = disc(fake.detach())

                loss_real = F.binary_cross_entropy(
                    real_preds, torch.full((B,), 0.9, device=device)  # label smoothing
                )
                loss_fake = F.binary_cross_entropy(
                    fake_preds, torch.zeros(B, device=device)
                )
                disc_loss = (loss_real + loss_fake) / 2

                disc_loss.backward()
                opt_disc.step()

                total_disc_loss += disc_loss.item()
                disc_updates    += 1

            # ── Step 2: Train generator (decoder only) ────────────────────
            opt_gen.zero_grad()

            fake     = vae.decode(z)
            gen_loss = F.binary_cross_entropy(
                disc(fake), torch.ones(B, device=device)
            )

            gen_loss.backward()
            opt_gen.step()

            total_gen_loss += gen_loss.item()

        nb       = len(loader)
        avg_gen  = total_gen_loss  / nb
        avg_disc = total_disc_loss / max(disc_updates, 1)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"gen={avg_gen:.4f}  disc={avg_disc:.4f}"
        )

        wandb.log({
            "epoch":     epoch,
            "gen_loss":  avg_gen,
            "disc_loss": avg_disc,
        })

        if avg_gen < best_gen_loss:
            best_gen_loss = avg_gen
            torch.save(vae.state_dict(),
                       os.path.join(SAVE_DIR, "gan_generator_best.pt"))
            torch.save(disc.state_dict(),
                       os.path.join(SAVE_DIR, "gan_disc_best.pt"))

    torch.save(vae.state_dict(),  os.path.join(SAVE_DIR, "gan_generator.pt"))
    torch.save(disc.state_dict(), os.path.join(SAVE_DIR, "gan_disc.pt"))
    wandb.finish()
    print(f"Done — best generator loss: {best_gen_loss:.4f}")


if __name__ == "__main__":
    main()