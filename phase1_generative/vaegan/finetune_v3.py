import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from models.vanilla_vae_v2 import VanillaVAE
from models.discriminator import Discriminator
from datasets_local.stamps_dataset import get_dataloader

# ── Fine tune config ──────────────────────────────────────────────────────────
DATA_PATH  = "/home/asellart/tfg_stamps/stamp_dataset_grayscale"
BATCH_SIZE = 64
LATENT_DIM = 256
IMAGE_SIZE  = 128

LR_VAE   = 5e-5    # lower than original — finer updates
LR_DISC  = 1e-6    # same as before
EPOCHS   = 200     # additional epochs on top of 300

KLD_WEIGHT = 1.0   # already at max — no annealing needed

LAMBDA_KL  = 1.0
LAMBDA_ADV = 0.1   # stronger than before (was 0.05)

DISC_UPDATE_FREQ = 5  # same as before

WANDB_PROJECT  = "tfg-stamps-vae"
WANDB_RUN_NAME = "vae-gan-finetune"

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAE_CKPT   = os.path.join(ROOT, "vae_gan_best.pt")
DISC_CKPT  = os.path.join(ROOT, "disc_gan_best.pt")


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "latent_dim":    LATENT_DIM,
            "image_size":    IMAGE_SIZE,
            "batch_size":    BATCH_SIZE,
            "lr_vae":        LR_VAE,
            "lr_disc":       LR_DISC,
            "epochs":        EPOCHS,
            "kld_weight":    KLD_WEIGHT,
            "lambda_kl":     LAMBDA_KL,
            "lambda_adv":    LAMBDA_ADV,
            "finetune_from": VAE_CKPT,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    loader = get_dataloader(
        DATA_PATH, BATCH_SIZE,
        image_size=IMAGE_SIZE, num_workers=4,
    )

    # ── Load models from checkpoint ───────────────────────────────────────────
    vae  = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    disc = Discriminator(in_channels=1).to(device)

    vae.load_state_dict(torch.load(VAE_CKPT,  map_location=device))
    disc.load_state_dict(torch.load(DISC_CKPT, map_location=device))
    print(f"Loaded VAE  from {VAE_CKPT}")
    print(f"Loaded Disc from {DISC_CKPT}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_vae  = optim.Adam(vae.parameters(),  lr=LR_VAE,  betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    best_vae_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        vae.train()
        disc.train()

        total_vae_loss  = 0.0
        total_recon     = 0.0
        total_kld       = 0.0
        total_adv       = 0.0
        total_disc_loss = 0.0
        disc_updates    = 0

        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            B = x.size(0)

            # ── Step 1: Train discriminator (every 5 batches) ─────────────────
            if batch_idx % DISC_UPDATE_FREQ == 0:
                opt_disc.zero_grad()

                real_labels = torch.ones(B, device=device)
                real_preds  = disc(x)
                loss_real   = F.binary_cross_entropy(real_preds, real_labels)

                with torch.no_grad():
                    recon, _, mu, logvar = vae(x)
                    z_random = torch.randn(B, LATENT_DIM, device=device)
                    fake     = vae.decode(z_random)

                fake_labels      = torch.zeros(B, device=device)
                fake_preds_recon = disc(recon.detach())
                fake_preds_rand  = disc(fake.detach())
                loss_fake = (
                    F.binary_cross_entropy(fake_preds_recon, fake_labels) +
                    F.binary_cross_entropy(fake_preds_rand,  fake_labels)
                ) / 2

                disc_loss = (loss_real + loss_fake) / 2
                disc_loss.backward()
                opt_disc.step()

                total_disc_loss += disc_loss.item()
                disc_updates    += 1

            # ── Step 2: Train VAE ─────────────────────────────────────────────
            opt_vae.zero_grad()

            recon, _, mu, logvar = vae(x)

            recon_loss = F.binary_cross_entropy(
                recon, x, reduction='sum'
            ) / B

            kld_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            z_random    = torch.randn(B, LATENT_DIM, device=device)
            fake_rand   = vae.decode(z_random)
            real_labels = torch.ones(B, device=device)

            adv_loss = (
                F.binary_cross_entropy(disc(recon),     real_labels) +
                F.binary_cross_entropy(disc(fake_rand), real_labels)
            ) / 2

            vae_loss = (
                recon_loss +
                KLD_WEIGHT * LAMBDA_KL * kld_loss +
                LAMBDA_ADV * adv_loss
            )

            vae_loss.backward()
            opt_vae.step()

            total_vae_loss += vae_loss.item()
            total_recon    += recon_loss.item()
            total_kld      += kld_loss.item()
            total_adv      += adv_loss.item()

        nb = len(loader)
        avg_vae   = total_vae_loss  / nb
        avg_recon = total_recon     / nb
        avg_kld   = total_kld       / nb
        avg_adv   = total_adv       / nb
        avg_disc  = total_disc_loss / max(disc_updates, 1)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"vae={avg_vae:.4f}  recon={avg_recon:.4f}  "
            f"kld={avg_kld:.4f}  adv={avg_adv:.4f}  "
            f"disc={avg_disc:.4f}"
        )

        wandb.log({
            "epoch":      epoch,
            "vae_loss":   avg_vae,
            "recon_loss": avg_recon,
            "kld_loss":   avg_kld,
            "adv_loss":   avg_adv,
            "disc_loss":  avg_disc,
            "lr_vae":     opt_vae.param_groups[0]["lr"],
        })

        # Save best and latest
        if avg_vae < best_vae_loss:
            best_vae_loss = avg_vae
            torch.save(vae.state_dict(),  "vae_gan_finetune_best.pt")
            torch.save(disc.state_dict(), "disc_gan_finetune_best.pt")
            print(f"  ✓ New best: {best_vae_loss:.4f}")

    torch.save(vae.state_dict(),  "vae_gan_finetune.pt")
    torch.save(disc.state_dict(), "disc_gan_finetune.pt")
    wandb.finish()
    print(f"Done — best VAE loss: {best_vae_loss:.4f}")
    print("Saved vae_gan_finetune_best.pt")


if __name__ == "__main__":
    main()