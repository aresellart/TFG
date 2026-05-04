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
from patch_utils import paste_patch, preprocess_document
import config
# WE ARE GOING TO TRY THE ATTACK WITH SAMPLING Z RANDOMLY FROM THE AUTOENCODER LATENT SPACE, INSTEAD OF USING THE ENCODED Z OF THE ORIGINAL IMAGE. THIS SHOULD GIVE US A BASELINE OF HOW WELL THE ATTACK CAN PERFORM WITHOUT ANY GUIDANCE FROM THE ORIGINAL IMAGE.import sys, os

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

DATASET_PATH   = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR    = os.path.join(ROOT, "attack_results_random_z")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Only run 4 representative classes — one per success tier
TARGET_CLASSES = [4, 1, 9, 14]  # form (high), memo (medium), news article (low), scientific report (failed)

BATCH_SIZE     = 8
N_ITERATIONS   = 300
LR             = 0.05
LAMBDA_Z       = 0.1
PATCH_SIZE     = 64
PATCH_POSITION = "bottom-right"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── Load models ───────────────────────────────────────────────────────────────
resnet = Model(device=device)
resnet.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"), map_location=device
))
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False

vae = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"), map_location=device
))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

print("Models loaded and frozen")

# ── Load dataset and correct image list ───────────────────────────────────────
dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, "correct_images.json")) as f:
    correct_images = json.load(f)
print("Dataset loaded")


# ── z initialization — N(0,I) sampling ───────────────────────────────────────
def get_initial_z():
    """Sample z directly from the prior N(0,I) — no real stamp used."""
    return torch.randn(1, config.LATENT_DIM, device=device)


# ── Get batch of documents ────────────────────────────────────────────────────
def get_document_batch(target_class, batch_size):
    candidates = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        candidates.extend(correct_images[str(cls_idx)])
    sampled = random.sample(candidates, batch_size)
    tensors = []
    for item in sampled:
        pil_img = dataset[item["idx"]]["image"]
        tensors.append(preprocess_document(pil_img))
    return torch.stack(tensors).to(device)


# ── Attack loop ───────────────────────────────────────────────────────────────
def attack_one_class(target_class):
    print(f"\n{'='*50}")
    print(f"Attacking target class: {CLASSES[target_class]} (idx {target_class})")
    print(f"z initialization: N(0,I) random sampling")
    print(f"{'='*50}")

    z_init = get_initial_z()
    z      = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=LR)

    best_loss = float("inf")
    best_z    = z.detach().clone()

    for iteration in tqdm(range(N_ITERATIONS)):
        optimizer.zero_grad()

        stamp     = vae.decode(z)
        doc_batch = get_document_batch(target_class, BATCH_SIZE)
        patched   = paste_patch(doc_batch, stamp,
                                patch_size=PATCH_SIZE,
                                position=PATCH_POSITION)

        _, logits = resnet(patched)

        target_tensor = torch.full((BATCH_SIZE,), target_class,
                                   dtype=torch.long, device=device)
        attack_loss     = F.cross_entropy(logits, target_tensor)
        generation_loss = torch.norm(z)
        loss            = attack_loss + LAMBDA_Z * generation_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z    = z.detach().clone()

        if (iteration + 1) % 50 == 0:
            with torch.no_grad():
                preds        = logits.argmax(dim=1)
                success_rate = (preds == target_class).float().mean().item()
            print(f"  iter {iteration+1:3d}  loss={loss.item():.4f}  "
                  f"attack={attack_loss.item():.4f}  "
                  f"gen={generation_loss.item():.4f}  "
                  f"success={success_rate*100:.1f}%")

    # Save z and stamp image
    cls_name = CLASSES[target_class]
    torch.save(best_z, os.path.join(RESULTS_DIR,
               f"z_class{target_class}_{cls_name}.pt"))
    import torchvision
    with torch.no_grad():
        final_stamp = vae.decode(best_z)
    torchvision.utils.save_image(
        final_stamp,
        os.path.join(RESULTS_DIR, f"stamp_class{target_class}_{cls_name}.png")
    )
    print(f"Saved stamp and z for class '{cls_name}'")
    return best_z


# ── Run attack for selected classes ───────────────────────────────────────────
if __name__ == "__main__":
    for target_class in TARGET_CLASSES:
        attack_one_class(target_class)
    print("\nDone! Now run evaluate_random_z.py to get the success rates.")