"""
train_endtoend_randpos_diversity.py

TEST SCRIPT — Phase 2 with a DIVERSITY LOSS to fix mode collapse.

Problem: after standard Phase 2 training, different z vectors decode to
nearly identical stamps for the same generator — the generator ignores
z and converges to one dominant adversarial pattern.

Fix: at each iteration, sample TWO different z vectors (z1, z2), decode
both into stamps, and penalize the generator if the two stamps are too
similar relative to how different z1 and z2 are. This is a standard
"mode-seeking" regularization (used e.g. in MSGAN, Mao et al. 2019):

    L_diversity = - mean(|stamp1 - stamp2|) / mean(|z1 - z2|)

Maximizing the ratio of output distance to latent distance means we want
DIFFERENT z's to produce stamps that are FAR APART in pixel space. We
minimize the NEGATIVE of this ratio (so the optimizer pushes it up).

L_total = L_attack + lambda_real * L_realism + lambda_div * L_diversity

Only trains the same 3 test classes as before for quick comparison.

Results saved to: attack_results_endtoend_diversity_test/

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/train_endtoend_randpos_diversity.py
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
    N_ITERATIONS, BATCH_SIZE, LR_G, LAMBDA_REAL,
    CLASSES
)

LAMBDA_DIV_TEST = 10.0
EPS             = 1e-5
TEST_CLASSES    = [0, 13]   # letter, resume — the two that failed to diversify
                            # (advertisement already showed real diversity at div=1.0)

RESULTS_DIR_TEST = os.path.join(ROOT, "attack_results_endtoend_diversity_test")
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
                         resnet, disc, device, lambda_real, lambda_div):
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"lambda_real = {lambda_real}  lambda_div = {lambda_div}")
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

        z1 = torch.randn(1, LATENT_DIM, device=device)
        z2 = torch.randn(1, LATENT_DIM, device=device)

        stamp1 = generator.decode(z1)
        stamp2 = generator.decode(z2)

        stamp_dist = torch.mean(torch.abs(stamp1 - stamp2))
        z_dist     = torch.mean(torch.abs(z1 - z2))
        L_diversity = -stamp_dist / (z_dist + EPS)

        doc_batch = get_document_batch(
            dataset, correct_images, target_class, BATCH_SIZE, device
        )
        patched = paste_patch_random_pos(doc_batch, stamp1, device)

        _, logits = resnet(patched)
        target_tensor = torch.full((BATCH_SIZE,), target_class,
                                    dtype=torch.long, device=device)
        L_attack = F.cross_entropy(logits, target_tensor)

        L_realism = (
            F.binary_cross_entropy(disc(stamp1), torch.ones(1, device=device)) +
            F.binary_cross_entropy(disc(stamp2), torch.ones(1, device=device))
        ) / 2

        L_total = L_attack + lambda_real * L_realism + lambda_div * L_diversity

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
                f"div={L_diversity.item():.4f}  "
                f"stamp_dist={stamp_dist.item():.4f}  "
                f"asr={success_rate*100:.1f}%"
            )

    return best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"TEST RUN — lambda_real={LAMBDA_REAL}  lambda_div={LAMBDA_DIV_TEST}")
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
        save_path = os.path.join(
            RESULTS_DIR_TEST,
            f"G_{target_class}_{cls_name}_div{LAMBDA_DIV_TEST}.pt"
        )

        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, device, LAMBDA_REAL, LAMBDA_DIV_TEST
        )

        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print(f"\nDone! Test generators trained with diversity loss.")
    print(f"Results saved in {RESULTS_DIR_TEST}/")
    print("Next: evaluate ASR + check if different z's now produce")
    print("visually different stamps (the mode collapse fix).")


if __name__ == "__main__":
    main()