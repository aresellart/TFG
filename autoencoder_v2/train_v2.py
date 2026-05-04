import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import wandb

from models.vanilla_vae_v2 import VanillaVAE
from datasets_local.stamps_dataset import get_dataloader
from autoencoder_v2.config_v2 import (
    DATA_PATH, BATCH_SIZE, LATENT_DIM, IMAGE_SIZE,
    LR, EPOCHS, KLD_MAX_WEIGHT, KLD_ANNEAL_EPOCHS,
    WANDB_PROJECT, WANDB_RUN_NAME
)


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "latent_dim":        LATENT_DIM,
            "image_size":        IMAGE_SIZE,
            "batch_size":        BATCH_SIZE,
            "lr":                LR,
            "epochs":            EPOCHS,
            "kld_max_weight":    KLD_MAX_WEIGHT,
            "kld_anneal_epochs": KLD_ANNEAL_EPOCHS,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    print(f"KLD max weight: {KLD_MAX_WEIGHT}  anneal epochs: {KLD_ANNEAL_EPOCHS}")

    loader = get_dataloader(
        DATA_PATH, BATCH_SIZE,
        image_size=IMAGE_SIZE, num_workers=0,
    )

    model = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_recon = 0.0
        total_kld   = 0.0

        # Slow linear annealing up to KLD_MAX_WEIGHT over KLD_ANNEAL_EPOCHS
        kld_weight = min(KLD_MAX_WEIGHT, KLD_MAX_WEIGHT * epoch / KLD_ANNEAL_EPOCHS)

        for x, _ in loader:
            x = x.to(device)
            recon, _, mu, logvar = model(x)

            loss_dict = model.loss_function(
                recon, x, mu, logvar, M_N=kld_weight
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss_dict["loss"].item()
            total_recon += loss_dict["Reconstruction_Loss"].item()
            total_kld   += loss_dict["KLD"].item()

        nb = len(loader)
        avg_loss  = total_loss  / nb
        avg_recon = total_recon / nb
        avg_kld   = total_kld   / nb

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  "
            f"kld={avg_kld:.4f}  kld_w={kld_weight:.4f}"
        )

        wandb.log({
            "epoch":               epoch,
            "loss":                avg_loss,
            "reconstruction_loss": avg_recon,
            "kld":                 avg_kld,
            "kld_weight":          kld_weight,
            "lr":                  optimizer.param_groups[0]["lr"],
        })

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "vae_high_kl_best.pt")

    torch.save(model.state_dict(), "vae_high_kl.pt")
    wandb.save("vae_high_kl.pt")
    wandb.finish()
    print(f"Done — best loss: {best_loss:.4f}")
    print("Saved vae_high_kl.pt and vae_high_kl_best.pt")


if __name__ == "__main__":
    main()