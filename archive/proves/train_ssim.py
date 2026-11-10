import torch
import torch.optim as optim
import wandb

from models.vanilla_vae_ssim import VanillaVAE
from datasets.stamps_dataset import get_dataloader
import config_ssim as config


def main():
    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "latent_dim": config.LATENT_DIM,
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "lr":         config.LR,
            "epochs":     config.EPOCHS,
            "ssim_weight": 8.0,
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
        total_loss  = 0.0
        total_recon = 0.0
        total_bce   = 0.0
        total_ssim  = 0.0
        total_kld   = 0.0

        kld_weight = min(1.0, epoch / 30) * 0.1

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
            total_bce   += loss_dict["BCE"].item()
            total_ssim  += loss_dict["SSIM"].item()
            total_kld   += loss_dict["KLD"].item()

        nb = len(loader)
        avg_loss  = total_loss  / nb
        avg_recon = total_recon / nb
        avg_bce   = total_bce   / nb
        avg_ssim  = total_ssim  / nb
        avg_kld   = total_kld   / nb

        print(
            f"Epoch {epoch:3d}/{config.EPOCHS}  "
            f"loss={avg_loss:.4f}  bce={avg_bce:.4f}  "
            f"ssim={avg_ssim:.4f}  kld={avg_kld:.4f}  "
            f"kld_w={kld_weight:.4f}"
        )

        wandb.log({
            "epoch":               epoch,
            "loss":                avg_loss,
            "reconstruction_loss": avg_recon,
            "bce":                 avg_bce,
            "ssim":                avg_ssim,
            "kld":                 avg_kld,
            "kld_weight":          kld_weight,
            "lr":                  optimizer.param_groups[0]["lr"],
        })

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "vanilla_vae_ssim.pt")
    wandb.save("vanilla_vae_ssim.pt")
    wandb.finish()
    print("Done — model saved as vanilla_vae_ssim.pt")


if __name__ == "__main__":
    main()