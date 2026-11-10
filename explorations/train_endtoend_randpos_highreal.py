"""
train_endtoend_randpos_highreal.py

TEST SCRIPT — Phase 2 with higher realism weight (lambda_real).

Identical to train_endtoend_randpos.py except:
  - LAMBDA_REAL is overridden to a higher value (test multiple values)
  - Only trains a SMALL SUBSET of classes (default: letter, advertisement, resume)
    to quickly check whether stamps regain semantic/category structure
    before committing to retraining all 16 generators.

Why these 3 classes: "advertisement" and "resume" were the classes that
collapsed hardest under the plain GAN ablation (47.7% and 45.8% ASR),
so they are good stress-test cases. "letter" is a stable, easy class
used as a control.

Results saved to: attack_results_endtoend_highreal_test/

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/train_endtoend_randpos_highreal.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
import json
import random
from tqdm import tqdm
from datasets import load_from_disk

from models.vanilla_vae_v2 import VanillaVAE
from models.discriminator  import Discriminator
from models.model_resnet50 import Model
from attack.patch_utils    import preprocess_document
from endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    VAE_PATH, DISC_PATH, RESNET_PATH,
    LATENT_DIM, PATCH_SIZE,
    N_ITERATIONS, BATCH_SIZE, LR_G,
    CLASSES
)

# ── TEST CONFIGURATION ────────────────────────────────────────────────────
LAMBDA_REAL_TEST = 0.3   # <-- try 0.3 first; original was 0.05
TEST_CLASSES     = [0, 7, 13]   # letter, advertisement, resume (indices)

RESULTS_DIR_TEST = os.path.join(ROOT, "attack_results_endtoend_highreal_test")
os.makedirs(RESULTS_DIR_TEST, exist_ok=True)

DOC_SIZE = 224
MAX_POS  = DOC_SIZE - PATCH_SIZE


def paste_patch_random_pos(doc_batch, stamp, device):
    patched = doc_batch.clone()

    stamp_resized = F.interpolate(stamp, size=(PATCH_SIZE, PATCH_SIZE),
                                   mode='bilinear', align_corners=False)
    stamp_3ch = stamp_resized.repeat(1, 3, 1, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    stamp_norm = (stamp_3ch - mean) / std

    B = doc_batch.size(0)
    for i in range(B):
        row = random.randint(0, MAX_POS)
        col = random.randint(0, MAX_POS)
        patched[i, :, row:row+PATCH_SIZE, col:col+PATCH_SIZE] = stamp_norm[0]

    return patched


def get_document_batch(dataset, correct_images, target_class, batch_size, device):
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


def train_one_generator(target_class, dataset, correct_images,
                         resnet, disc, device, lambda_real):
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"lambda_real = {lambda_real}  (TEST — original was 0.05)")
    print(f"Iterations: {N_ITERATIONS}  Batch: {BATCH_SIZE}")
    print(f"{'='*55}")

    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(VAE_PATH, map_location=device,
                                          weights_only=False))
    generator.train()

    opt_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))

    best_loss  = float("inf")
    best_state = None

    for iteration in tqdm(range(N_ITERATIONS)):
        opt_G.zero_grad()

        z     = torch.randn(1, LATENT_DIM, device=device)
        stamp = generator.decode(z)

        doc_batch = get_document_batch(
            dataset, correct_images, target_class, BATCH_SIZE, device
        )

        patched = paste_patch_random_pos(doc_batch, stamp, device)

        _, logits = resnet(patched)
        target_tensor = torch.full((BATCH_SIZE,), target_class,
                                    dtype=torch.long, device=device)
        L_attack = F.cross_entropy(logits, target_tensor)

        L_realism = F.binary_cross_entropy(
            disc(stamp), torch.ones(1, device=device)
        )

        # ── KEY CHANGE: higher lambda_real ──────────────────────────────
        L_total = L_attack + lambda_real * L_realism

        L_total.backward()
        opt_G.step()

        if L_total.item() < best_loss:
            best_loss  = L_total.item()
            best_state = {k: v.clone() for k, v in generator.state_dict().items()}

        if (iteration + 1) % 500 == 0:
            with torch.no_grad():
                preds        = logits.argmax(dim=1)
                success_rate = (preds == target_class).float().mean().item()
            print(
                f"  iter {iteration+1:4d}  "
                f"loss={L_total.item():.4f}  "
                f"attack={L_attack.item():.4f}  "
                f"realism={L_realism.item():.4f}  "
                f"asr={success_rate*100:.1f}%"
            )

    return best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"TEST RUN — lambda_real = {LAMBDA_REAL_TEST}")
    print(f"Training classes: {[CLASSES[i] for i in TEST_CLASSES]}")

    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)
    print("Dataset loaded")

    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device,
                                       weights_only=False))
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet loaded and frozen")

    disc = Discriminator(in_channels=1).to(device)
    disc.load_state_dict(torch.load(DISC_PATH, map_location=device,
                                     weights_only=False))
    disc.eval()
    for p in disc.parameters():
        p.requires_grad = False
    print("Discriminator loaded and frozen")

    for target_class in TEST_CLASSES:
        cls_name  = CLASSES[target_class]
        save_path = os.path.join(RESULTS_DIR_TEST,
                                  f"G_{target_class}_{cls_name}_lr{LAMBDA_REAL_TEST}.pt")

        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, device, LAMBDA_REAL_TEST
        )

        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print(f"\nDone! Test generators trained with lambda_real={LAMBDA_REAL_TEST}.")
    print(f"Results saved in {RESULTS_DIR_TEST}/")
    print("Next: evaluate ASR + visually inspect stamps before deciding whether")
    print("to retrain all 16 classes with this lambda_real value.")


if __name__ == "__main__":
    main()