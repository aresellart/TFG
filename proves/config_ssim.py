DATA_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/stamp_dataset_grayscale"

BATCH_SIZE = 32
LATENT_DIM = 256
IMAGE_SIZE = 128

LR = 1e-4
EPOCHS = 200

DEVICE = "cuda"

# WandB
WANDB_PROJECT = "tfg-stamps-vae"
WANDB_RUN_NAME = "vae-ssim-128px"