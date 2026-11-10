# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT          = "/home/asellart/tfg_stamps"
DATASET_PATH  = "/home/asellart/tfg_stamps/rvl_cdip_full"
CORRECT_JSON  = "/home/asellart/tfg_stamps/correct_images.json"
VAE_PATH      = "/home/asellart/tfg_stamps/vae_gan_best.pt"
DISC_PATH     = "/home/asellart/tfg_stamps/disc_gan_best.pt"
RESNET_PATH   = "/home/asellart/tfg_stamps/models/rvl-resnet50.model"
RESULTS_DIR   = "/home/asellart/tfg_stamps/attack_results_endtoend"

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
LATENT_DIM    = 256       # size of z vector — must match Phase 1
IMAGE_SIZE    = 128       # stamp size — must match Phase 1
PATCH_SIZE    = 64        # stamp resized to 64×64 before pasting
PATCH_POS     = "bottom-right"

# ── TRAINING ──────────────────────────────────────────────────────────────────
N_ITERATIONS  = 2000      # training iterations per generator
BATCH_SIZE    = 128       # documents per iteration
LR_G          = 1e-4      # generator learning rate
LAMBDA_REAL   = 0.05      # weight on realism loss — same as Phase 1 λ_adv

# ── CLASSES ───────────────────────────────────────────────────────────────────
CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]