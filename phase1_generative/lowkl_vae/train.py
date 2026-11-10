import torch
import torch.optim as optim
import wandb

from models.vanilla_vae import VanillaVAE
from datasets.stamps_dataset import get_dataloader
from tfg_stamps.autoencoder.utils import vae_loss
import tfg_stamps.autoencoder.config as config

"""""
def main():
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "latent_dim": config.LATENT_DIM,
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LR,
            "epochs": config.EPOCHS,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    loader = get_dataloader(
        config.DATA_PATH,
        config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=0,
    )

    model = VanillaVAE(
        in_channels=1,
        latent_dim=config.LATENT_DIM,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    kld_weight = config.BATCH_SIZE / len(loader.dataset)

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

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

        n = len(loader.dataset)
        avg_loss  = total_loss  / n
        avg_recon = total_recon / n
        avg_kld   = total_kld   / n

        print(f"Epoch {epoch:3d}/{config.EPOCHS}  loss={avg_loss:.4f}  recon={avg_recon:.4f}  kld={avg_kld:.4f}")

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "reconstruction_loss": avg_recon,
            "kld": avg_kld,
            "lr": optimizer.param_groups[0]["lr"],
        })

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "vanilla_vae_stamps.pt")
    wandb.save("vanilla_vae_stamps.pt")
    wandb.finish()
    print("Done — model saved and logged to WandB")


if __name__ == "__main__":
    main()

"""
import torch
import torch.optim as optim
import wandb

from models.vanilla_vae import VanillaVAE
from datasets.stamps_dataset import get_dataloader
from tfg_stamps.autoencoder.utils import vae_loss
import tfg_stamps.autoencoder.config as config


def main():
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "latent_dim": config.LATENT_DIM,
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LR,
            "epochs": config.EPOCHS,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    loader = get_dataloader(
        config.DATA_PATH,
        config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=0,
    )

    model = VanillaVAE(
        in_channels=1,
        latent_dim=config.LATENT_DIM,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        # ✅ KL annealing (NEW)
        kld_weight = min(0.5, epoch / 30) #abans estave a 1.0
        #kld_weight *= 0.1

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

            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["Reconstruction_Loss"].item()
            total_kld += loss_dict["KLD"].item()

        num_batches = len(loader)
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches

        print(
            f"Epoch {epoch:3d}/{config.EPOCHS}  "
            f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  kld={avg_kld:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "reconstruction_loss": avg_recon,
            "kld": avg_kld,
            "kld_weight": kld_weight,  
            "lr": optimizer.param_groups[0]["lr"],
        })

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "vanilla_vae_stamps.pt")
    wandb.save("vanilla_vae_stamps.pt")
    wandb.finish()
    print("Done — model saved and logged to WandB")


if __name__ == "__main__":
    main()