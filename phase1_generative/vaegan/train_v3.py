import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from models.vanilla_vae_v2 import VanillaVAE
from models.discriminator import Discriminator
from datasets_local.stamps_dataset import get_dataloader
from autoencoder_v3.config_v3 import (
    DATA_PATH, BATCH_SIZE, LATENT_DIM, IMAGE_SIZE,
    LR_VAE, LR_DISC, EPOCHS,
    KLD_MAX_WEIGHT, KLD_ANNEAL_EPOCHS,
    LAMBDA_KL, LAMBDA_ADV,
    WANDB_PROJECT, WANDB_RUN_NAME
)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "latent_dim":         LATENT_DIM,
            "image_size":         IMAGE_SIZE,
            "batch_size":         BATCH_SIZE,
            "lr_vae":             LR_VAE,
            "lr_disc":            LR_DISC,
            "epochs":             EPOCHS,
            "kld_max_weight":     KLD_MAX_WEIGHT,
            "kld_anneal_epochs":  KLD_ANNEAL_EPOCHS,
            "lambda_kl":          LAMBDA_KL,
            "lambda_adv":         LAMBDA_ADV,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    loader = get_dataloader(
        DATA_PATH, BATCH_SIZE,
        image_size=IMAGE_SIZE, num_workers=0,
    )

    # MODELS --------------------------------------------------
    vae  = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    disc = Discriminator(in_channels=1).to(device)

    #OPTIMIZERS --------------------------------------------------
    opt_vae  = optim.Adam(vae.parameters(),  lr=LR_VAE,  betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    best_vae_loss = float("inf") #we'll save the best model based on the lowest VAE loss (reconstruction + KL) during training

    # TRAINING LOOP --------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        vae.train()
        disc.train()

        total_vae_loss  = 0.0
        total_recon     = 0.0
        total_kld       = 0.0
        total_adv       = 0.0
        total_disc_loss = 0.0
        disc_updates    = 0

        # KL annealing
        kld_weight = min(KLD_MAX_WEIGHT,
                         KLD_MAX_WEIGHT * epoch / KLD_ANNEAL_EPOCHS)

        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            B = x.size(0)

            # ── Step 1: Train discriminator (every 5 batches) ─────────────────
            if batch_idx % 5 == 0:
                opt_disc.zero_grad()

                # Real stamps → discriminator should output 1
                real_labels = torch.ones(B, device=device)
                real_preds  = disc(x)
                loss_real   = F.binary_cross_entropy(real_preds, real_labels)

                # Fake stamps → discriminator should output 0
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

            # ── Step 2: Train VAE (encoder + decoder) ────────────────────────
            opt_vae.zero_grad()

            recon, _, mu, logvar = vae(x)

            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(
                recon, x, reduction='sum'
            ) / B

            # KL loss
            kld_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            # Adversarial loss — fool the discriminator
            z_random    = torch.randn(B, LATENT_DIM, device=device)
            fake_rand   = vae.decode(z_random)
            real_labels = torch.ones(B, device=device)

            adv_loss = (
                F.binary_cross_entropy(disc(recon),     real_labels) +
                F.binary_cross_entropy(disc(fake_rand), real_labels)
            ) / 2

            # Total VAE loss
            vae_loss = (
                recon_loss +
                kld_weight * LAMBDA_KL * kld_loss +
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
            f"disc={avg_disc:.4f}  kld_w={kld_weight:.3f}"
        )

        wandb.log({
            "epoch":      epoch,
            "vae_loss":   avg_vae,
            "recon_loss": avg_recon,
            "kld_loss":   avg_kld,
            "adv_loss":   avg_adv,
            "disc_loss":  avg_disc,
            "kld_weight": kld_weight,
            "lr_vae":     opt_vae.param_groups[0]["lr"],
        })

        if avg_vae < best_vae_loss:
            best_vae_loss = avg_vae
            torch.save(vae.state_dict(),  "vae_gan_best.pt")
            torch.save(disc.state_dict(), "disc_gan_best.pt")

    torch.save(vae.state_dict(),  "vae_gan.pt")
    torch.save(disc.state_dict(), "disc_gan.pt")
    wandb.finish()
    print(f"Done — best VAE loss: {best_vae_loss:.4f}")


if __name__ == "__main__":
    main()