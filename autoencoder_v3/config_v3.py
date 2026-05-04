DATA_PATH = "/home/asellart/tfg_stamps/stamp_dataset_grayscale"

BATCH_SIZE = 64
LATENT_DIM = 256
IMAGE_SIZE = 128

LR_VAE   = 1e-4
LR_DISC  = 1e-6
EPOCHS   = 300

KLD_MAX_WEIGHT    = 1.0
KLD_ANNEAL_EPOCHS = 50

# Loss weights
LAMBDA_KL  = 1.0    # weight on KL loss
LAMBDA_ADV = 0.05    # weight on adversarial loss — start small, tune if needed

WANDB_PROJECT  = "tfg-stamps-vae"
WANDB_RUN_NAME = "vae-gan-v3"