import os

DATA_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/stamp_dataset_grayscale"

BATCH_SIZE = 32
LATENT_DIM = 256
IMAGE_SIZE = 128

LR     = 1e-4
EPOCHS = 200

KLD_MAX_WEIGHT    = 1.0   # much stronger than before (was 0.1)
KLD_ANNEAL_EPOCHS = 100    # slower annealing for stability

DEVICE = "cuda"

WANDB_PROJECT  = "tfg-stamps-vae"
WANDB_RUN_NAME = "vae-high-kl-v2-resblocks"