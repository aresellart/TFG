"""
train_endtoend_randpos_anchor.py

TEST SCRIPT — Phase 2 with WEIGHT ANCHORING to the Phase 1 decoder.

Rationale: raising lambda_realism alone did not recover semantic/category
structure in the stamps — the generator just found a different texture
pattern (a regular grid) that still satisfies the discriminator's narrow
realism check. The discriminator only measures low-level texture
statistics (sharpness, contrast, edges), not whether the output looks
like an actual stamp category.

This script instead adds a direct regularization term that penalizes
the L2 distance between the CURRENT generator weights and the ORIGINAL
Phase 1 VAE-GAN decoder weights. This anchors the generator near the
region of weight-space where Phase 1 random samples were demonstrably
category-coherent (circular seals, text-like stamps), regardless of
what the discriminator does or doesn't penalize.

L_total = L_attack + lambda_real * L_realism + lambda_anchor * L_anchor
where L_anchor = mean squared difference between current and original
decoder weights (computed only over decoder parameters, since the
generator IS the decoder — same architecture as VanillaVAE.decode()).

Only trains the same 3 test classes as before (letter, advertisement,
resume) for quick comparison before committing to all 16.

Results saved to: attack_results_endtoend_anchor_test/

Usage:
    cd /home/asellart/tfg_stamps
    python3 explorations/train_endtoend_randpos_anchor.py
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
from common.patch_utils    import preprocess_document
from phase3_endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    VAE_PATH, DISC_PATH, RESNET_PATH,
    LATENT_DIM, PATCH_SIZE,
    N_ITERATIONS, BATCH_SIZE, LR_G,
    CLASSES
)

# ── TEST CONFIGURATION ────────────────────────────────────────────────────
LAMBDA_REAL_TEST   = 0.05   # back to original — anchoring replaces the
                            # need to push realism weight higher
LAMBDA_ANCHOR_TEST = 1.0    # weight on weight-space drift penalty
TEST_CLASSES       = [0, 7, 13]   # letter, advertisement, resume

RESULTS_DIR_TEST = os.path.join(ROOT, "attack_results_endtoend_anchor_test")
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


def compute_anchor_loss(generator, original_state_dict, device):
    """
    L2 distance between current generator weights and the original
    Phase 1 VAE-GAN decoder weights. Computed over ALL generator
    parameters (the generator IS the decoder architecture, so the
    keys match directly).
    """
    loss = 0.0
    n_params = 0
    for name, param in generator.named_parameters():
        if name in original_state_dict:
            orig = original_state_dict[name].to(device)
            loss = loss + F.mse_loss(param, orig, reduction='sum')
            n_params += param.numel()
    return loss / n_params   # mean squared difference, normalized


def train_one_generator(target_class, dataset, correct_images,
                         resnet, disc, original_state_dict, device,
                         lambda_real, lambda_anchor):
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"lambda_real = {lambda_real}  lambda_anchor = {lambda_anchor}")
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

        # ── KEY ADDITION: anchor loss ────────────────────────────────────
        L_anchor = compute_anchor_loss(generator, original_state_dict, device)

        L_total = L_attack + lambda_real * L_realism + lambda_anchor * L_anchor

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
                f"anchor={L_anchor.item():.6f}  "
                f"asr={success_rate*100:.1f}%"
            )

    return best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"TEST RUN — lambda_real={LAMBDA_REAL_TEST}  lambda_anchor={LAMBDA_ANCHOR_TEST}")
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

    # load the ORIGINAL Phase 1 decoder weights — kept separately,
    # never updated, used only as the anchor reference point
    original_state_dict = torch.load(VAE_PATH, map_location=device,
                                      weights_only=False)
    print("Original Phase 1 decoder weights loaded as anchor reference")

    for target_class in TEST_CLASSES:
        cls_name  = CLASSES[target_class]
        save_path = os.path.join(
            RESULTS_DIR_TEST,
            f"G_{target_class}_{cls_name}_anchor{LAMBDA_ANCHOR_TEST}.pt"
        )

        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, original_state_dict, device,
            LAMBDA_REAL_TEST, LAMBDA_ANCHOR_TEST
        )

        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print(f"\nDone! Test generators trained with anchor regularization.")
    print(f"Results saved in {RESULTS_DIR_TEST}/")
    print("Next: evaluate ASR + visually inspect stamps.")


if __name__ == "__main__":
    main()