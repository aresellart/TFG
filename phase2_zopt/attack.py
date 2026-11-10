import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from datasets import load_from_disk
from tqdm import tqdm

from models.model_resnet50 import Model
from models.vanilla_vae import VanillaVAE
#from attack.patch_utils import preprocess_document, paste_patch
from patch_utils import preprocess_document, paste_patch
import tfg_stamps.autoencoder.config as config

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
CORRECT_JSON   = "correct_images.json"
ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

# Attack hyperparameters
BATCH_SIZE     = 8       # documents per iteration
N_ITERATIONS   = 300     # optimization steps per target class
LR             = 0.05    # learning rate for z
LAMBDA_Z       = 0.1     # weight of L2 constraint on z
PATCH_SIZE     = 64      # stamp patch size in pixels
PATCH_POSITION = "bottom-right"

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Load frozen ResNet-50
resnet = Model(device=device)
resnet.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"),
    map_location=device
))
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False
print("ResNet-50 loaded and frozen")

# Load frozen VAE
vae = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"),
    map_location=device
))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
print("VAE loaded and frozen")

# Load dataset and correct image list
dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, CORRECT_JSON)) as f:
    correct_images = json.load(f)
print("Dataset and correct image list loaded")


# ── Helper: get a batch of document images excluding target class ──────────────
def get_document_batch(target_class, batch_size):
    """
    Sample batch_size images from all classes except target_class.
    Returns a (B, 3, 224, 224) tensor ready for ResNet.
    """
    # Collect indices from all classes except target
    candidates = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        candidates.extend(correct_images[str(cls_idx)])

    # Sample randomly
    sampled = random.sample(candidates, batch_size)

    tensors = []
    for item in sampled:
        pil_img = dataset[item["idx"]]["image"]
        tensors.append(preprocess_document(pil_img))

    return torch.stack(tensors).to(device)  # (B, 3, 224, 224)


# ── Helper: get initial z from a real stamp ───────────────────────────────────
def get_initial_z():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stamps_dataset",
        os.path.join(ROOT, "datasets_local", "stamps_dataset.py")
)
    stamps_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stamps_module)
    
    stamp_loader = stamps_module.get_dataloader(
        config.DATA_PATH, batch_size=1,
        image_size=config.IMAGE_SIZE, shuffle=True, num_workers=0
    )
    stamp_img, _ = next(iter(stamp_loader))
    stamp_img = stamp_img.to(device)
    with torch.no_grad():
        mu, logvar = vae.encode(stamp_img)
        z = vae.reparameterize(mu, logvar)
    return z


# ── Main attack loop ───────────────────────────────────────────────────────────
def attack_one_class(target_class):
    print(f"\n{'='*50}")
    print(f"Attacking target class: {CLASSES[target_class]} (idx {target_class})")
    print(f"{'='*50}")

    # Initialize z from a real stamp
    z_init = get_initial_z()
    z = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=LR)

    best_loss    = float("inf")
    best_z       = z.detach().clone()

    for iteration in tqdm(range(N_ITERATIONS)):
        optimizer.zero_grad()

        # Decode z → stamp patch
        stamp = vae.decode(z)  # (1, 1, 128, 128)

        # Get a batch of non-target documents
        doc_batch = get_document_batch(target_class, BATCH_SIZE)

        # Paste stamp onto documents
        patched = paste_patch(doc_batch, stamp,
                              patch_size=PATCH_SIZE,
                              position=PATCH_POSITION)

        # Forward pass through frozen ResNet
        _, logits = resnet(patched)  # (B, 16)

        # Attack loss — push toward target class
        target_tensor = torch.full((BATCH_SIZE,), target_class,
                                   dtype=torch.long, device=device)
        attack_loss = F.cross_entropy(logits, target_tensor)

        # Generation loss — keep z realistic
        generation_loss = torch.norm(z)

        # Total loss
        loss = attack_loss + LAMBDA_Z * generation_loss

        loss.backward()
        optimizer.step()

        # Track best z
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z    = z.detach().clone()

        # Log every 50 iterations
        if (iteration + 1) % 50 == 0:
            # Check how many images are predicted as target
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                success_rate = (preds == target_class).float().mean().item()
            print(f"  iter {iteration+1:3d}  loss={loss.item():.4f}  "
                  f"attack={attack_loss.item():.4f}  "
                  f"gen={generation_loss.item():.4f}  "
                  f"success={success_rate*100:.1f}%")

    # Save best z and decoded stamp
    out_dir = os.path.join(ROOT, "attack_results")
    os.makedirs(out_dir, exist_ok=True)

    # Save z
    torch.save(best_z, os.path.join(out_dir, f"z_class{target_class}_{CLASSES[target_class]}.pt"))

    # Save decoded stamp as image
    with torch.no_grad():
        final_stamp = vae.decode(best_z)
    import torchvision
    torchvision.utils.save_image(
        final_stamp,
        os.path.join(out_dir, f"stamp_class{target_class}_{CLASSES[target_class]}.png")
    )

    print(f"Saved stamp and z for class '{CLASSES[target_class]}'")
    return best_z


# ── Run attack for all 16 classes ─────────────────────────────────────────────
if __name__ == "__main__":
    all_z = {}
    for target_class in range(16): #CANVIAR DESPRES A 16 QUAN TINGUI LA GPU
        best_z = attack_one_class(target_class)
        all_z[target_class] = best_z

    print("\nAttack complete for all 16 classes!")
    print(f"Results saved in attack_results/")