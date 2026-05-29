DATA_PATH = "/home/asellart/tfg_stamps/stamp_dataset_grayscale" #SPODS Dataset path (grayscale version)

BATCH_SIZE = 64 #number of stamps processed per training iteration
LATENT_DIM = 256 #size of the latent vector
IMAGE_SIZE = 128 #input images are 128x128 pixels (after resizing)

LR_VAE   = 1e-4
LR_DISC  = 1e-6
EPOCHS   = 300

KLD_MAX_WEIGHT    = 1.0
KLD_ANNEAL_EPOCHS = 50

# Loss weights
LAMBDA_KL  = 1.0    # weight on KL loss
LAMBDA_ADV = 0.05    # weight on adversarial loss 

WANDB_PROJECT  = "tfg-stamps-vae"
WANDB_RUN_NAME = "vae-gan-v3"