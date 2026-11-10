"""
train_endtoend_randpos_morph.py

TEST SCRIPT — Phase 2 with a MORPHOLOGICAL OPENING loss to encourage
coherent, blob-like stamp shapes instead of scattered/fragmented noise.

Rationale (Dimos's feedback): instead of asking "how do we preserve
stamp meaning" (there is nothing to preserve, z is random noise), ask
"how do we INDUCE meaning" by penalizing the specific structural
properties that make current stamps look wrong — scattered blobs,
disconnected fragments, noisy speckled texture.

Morphological opening = erosion followed by dilation. It is implemented
here via differentiable max/min-pooling:
    erosion(x)  = -maxpool(-x)      (shrinks bright regions, kills small
                                      isolated speckles/fragments)
    dilation(x) =  maxpool(x)       (grows bright regions back)
    opening(x)  =  dilation(erosion(x))

A solid, coherent shape survives opening almost unchanged. A noisy,
fragmented shape loses its small disconnected pieces under erosion and
they do NOT come back under dilation, so opening(x) differs
significantly from x. Penalizing this difference pushes the generator
toward solid, coherent shapes.

    L_morph = mean(|stamp - opening(stamp)|)

L_total = L_attack + lambda_real * L_realism + lambda_morph * L_morph

Only trains the same 3 test classes as before for quick comparison.

Results saved to: attack_results_endtoend_morph_test/

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/train_endtoend_randpos_morph.py
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

# ── TEST CONFIGURATION ────────────────────────────────────────────────────
LAMBDA_MORPH_TEST = 3.0    # weight on morphological opening loss — increased from 1.0
MORPH_KERNEL      = 5      # pooling kernel size for erosion/dilation
TEST_CLASSES      = [0, 7, 13]   # letter, advertisement, resume

RESULTS_DIR_TEST = os.path.join(ROOT, "attack_results_endtoend_morph_test")
os.makedirs(RESULTS_DIR_TEST, exist_ok=True)

DOC_SIZE = 224
MAX_POS  = DOC_SIZE - PATCH_SIZE


def morphological_opening(x, kernel_size=MORPH_KERNEL):
    """
    Differentiable approximation of morphological opening via
    min-pooling (erosion) followed by max-pooling (dilation).
    x: (B, 1, H, W) in [0,1]
    """
    pad = kernel_size // 2
    # erosion: shrink bright regions = negative max-pool of negative input
    eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    # dilation: grow bright regions back
    opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=pad)
    return opened


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
                         resnet, disc, device, lambda_real, lambda_morph):
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"lambda_real = {lambda_real}  lambda_morph = {lambda_morph}")
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

        # ── KEY ADDITION: morphological opening loss ─────────────────────
        opened  = morphological_opening(stamp)
        L_morph = torch.mean(torch.abs(stamp - opened))

        L_total = L_attack + lambda_real * L_realism + lambda_morph * L_morph

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
                f"morph={L_morph.item():.4f}  "
                f"asr={success_rate*100:.1f}%"
            )

    return best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"TEST RUN — lambda_real={LAMBDA_REAL}  lambda_morph={LAMBDA_MORPH_TEST}")
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
            f"G_{target_class}_{cls_name}_morph{LAMBDA_MORPH_TEST}.pt"
        )

        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, device, LAMBDA_REAL, LAMBDA_MORPH_TEST
        )

        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print(f"\nDone! Test generators trained with morphological opening loss.")
    print(f"Results saved in {RESULTS_DIR_TEST}/")
    print("Next: evaluate ASR + visually inspect stamps for coherent,")
    print("non-fragmented shapes.")


if __name__ == "__main__":
    main()