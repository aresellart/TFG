"""
train_endtoend_randpos.py

Phase 2 — end-to-end training with RANDOM stamp position.

Identical to train_endtoend.py except the stamp is pasted at a
different random position at every iteration instead of fixed bottom-right.

This forces the generator to produce stamps that fool ResNet
regardless of where they appear on the document — a truly
position-invariant adversarial attack.

Results saved to: attack_results_endtoend_randpos/

Usage:
    python3 endtoend/train_endtoend_randpos.py
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

# ── NEW RESULTS DIRECTORY ─────────────────────────────────────────────────────
RESULTS_DIR_RANDPOS = os.path.join(ROOT, "attack_results_endtoend_randpos")
os.makedirs(RESULTS_DIR_RANDPOS, exist_ok=True)

DOC_SIZE = 224
MAX_POS  = DOC_SIZE - PATCH_SIZE   # 160 — maximum top-left corner coordinate


def paste_patch_random_pos(doc_batch, stamp, device):
    """
    Paste stamp at a DIFFERENT random position for each document in the batch.
    Called at every training iteration — position changes each time.

    This is the KEY difference from train_endtoend.py where position
    was always fixed at bottom-right (row=160, col=160).
    """
    patched = doc_batch.clone()
    # clone: creates new tensor so in-place modification doesn't break autograd

    # resize stamp from 128×128 to 64×64
    stamp_resized = F.interpolate(stamp, size=(PATCH_SIZE, PATCH_SIZE),
                                   mode='bilinear', align_corners=False)
    # bilinear interpolation is differentiable — gradients flow back to stamp

    # grayscale → 3 channels
    stamp_3ch = stamp_resized.repeat(1, 3, 1, 1)

    # normalize with ImageNet stats to match document normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    stamp_norm = (stamp_3ch - mean) / std

    B = doc_batch.size(0)
    for i in range(B):
        # independent random position for each document in the batch
        row = random.randint(0, MAX_POS)
        col = random.randint(0, MAX_POS)
        patched[i, :, row:row+PATCH_SIZE, col:col+PATCH_SIZE] = stamp_norm[0]
        # gradient flows through stamp_norm[0] → stamp_3ch → stamp_resized
        # → stamp → generator.decode → generator weights

    return patched


def get_document_batch(dataset, correct_images, target_class, batch_size, device):
    """Sample batch of documents from all classes except target_class."""
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
                         resnet, disc, device):
    """
    Train one generator G_c with random stamp position at every iteration.
    """
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"Position: RANDOM (changes every iteration)")
    print(f"Iterations: {N_ITERATIONS}  Batch: {BATCH_SIZE}")
    print(f"{'='*55}")

    # initialize generator from pretrained VAE-GAN decoder
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(VAE_PATH, map_location=device,
                                          weights_only=False))
    generator.train()

    opt_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))

    best_loss  = float("inf")
    best_state = None

    for iteration in tqdm(range(N_ITERATIONS)):
        opt_G.zero_grad()

        # sample random z — not optimized, just input noise
        z = torch.randn(1, LATENT_DIM, device=device)

        # decode z → stamp
        stamp = generator.decode(z)

        # get document batch
        doc_batch = get_document_batch(
            dataset, correct_images, target_class, BATCH_SIZE, device
        )

        # ── KEY DIFFERENCE: random position instead of fixed bottom-right ──
        patched = paste_patch_random_pos(doc_batch, stamp, device)
        # each document in the batch gets the stamp at a different random position
        # this forces the generator to find patterns that work everywhere

        # frozen ResNet → L_attack
        _, logits = resnet(patched)
        target_tensor = torch.full((BATCH_SIZE,), target_class,
                                    dtype=torch.long, device=device)
        L_attack = F.cross_entropy(logits, target_tensor)

        # frozen discriminator → L_realism
        L_realism = F.binary_cross_entropy(
            disc(stamp), torch.ones(1, device=device)
        )

        # total loss
        L_total = L_attack + LAMBDA_REAL * L_realism

        # backprop → generator weights only
        L_total.backward()
        opt_G.step()

        # track best
        if L_total.item() < best_loss:
            best_loss  = L_total.item()
            best_state = {k: v.clone() for k, v in generator.state_dict().items()}

        # log every 500 iterations
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

    # load dataset
    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)
    print("Dataset loaded")

    # frozen ResNet
    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device,
                                       weights_only=False))
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet loaded and frozen")

    # frozen discriminator
    disc = Discriminator(in_channels=1).to(device)
    disc.load_state_dict(torch.load(DISC_PATH, map_location=device,
                                     weights_only=False))
    disc.eval()
    for p in disc.parameters():
        p.requires_grad = False
    print("Discriminator loaded and frozen")

    # train 16 generators
    for target_class in range(16):
        cls_name  = CLASSES[target_class]
        save_path = os.path.join(RESULTS_DIR_RANDPOS,
                                  f"G_{target_class}_{cls_name}.pt")

        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, device
        )

        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print("\nDone! All 16 generators trained with random position.")
    print(f"Results saved in {RESULTS_DIR_RANDPOS}/")


if __name__ == "__main__":
    main()