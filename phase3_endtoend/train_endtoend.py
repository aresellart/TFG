"""
train_endtoend.py

Phase 2 — end-to-end adversarial stamp generation.

For each of the 16 target classes:
  - Initialize a generator G_c from the pretrained VAE-GAN decoder weights
  - Train G_c so that any random z → stamp that fools ResNet into class c
  - Realism enforced by frozen discriminator from Phase 1

Nothing gets updated except G_c weights.
ResNet and Discriminator are completely frozen.

Usage:
    python3 attack/train_endtoend.py
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
from attack.patch_utils    import paste_patch, preprocess_document
from endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    VAE_PATH, DISC_PATH, RESNET_PATH, RESULTS_DIR,
    LATENT_DIM, PATCH_SIZE, PATCH_POS,
    N_ITERATIONS, BATCH_SIZE, LR_G, LAMBDA_REAL,
    CLASSES
)


def get_document_batch(dataset, correct_images, target_class, batch_size, device):
    """
    Sample a random batch of documents from all classes EXCEPT target_class.
    These are the documents we want to fool into being classified as target_class.
    """
    candidates = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        # collect image indices from all other classes
        candidates.extend(correct_images[str(cls_idx)])

    # randomly sample batch_size documents from candidates
    sampled = random.sample(candidates, batch_size)

    tensors = []
    for item in sampled:
        pil_img = dataset[item["idx"]]["image"]
        tensors.append(preprocess_document(pil_img))

    return torch.stack(tensors).to(device)
    # shape: (batch_size, 3, 224, 224)
    # normalized with ImageNet stats, grayscale replicated to 3 channels


def train_one_generator(target_class, dataset, correct_images,
                         resnet, disc, device):
    """
    Train one generator G_c for a specific target class.

    Steps each iteration:
        1. Sample random z ~ N(0,I)
        2. Decode z → stamp via G_c (trainable)
        3. Paste stamp on document batch
        4. Forward through frozen ResNet → L_attack
        5. Forward stamp through frozen discriminator → L_realism
        6. L_total = L_attack + λ * L_realism
        7. Backprop → update G_c weights only
    """
    cls_name = CLASSES[target_class]
    print(f"\n{'='*55}")
    print(f"Training generator for: {cls_name} (class {target_class})")
    print(f"Iterations: {N_ITERATIONS}  Batch: {BATCH_SIZE}")
    print(f"{'='*55}")

    # ── Initialize generator from pretrained VAE-GAN decoder ──────────────────
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(VAE_PATH, map_location=device, weights_only= False))
    # load ALL VAE weights (encoder + decoder)
    # we only use the decoder part (generator.decode) during training
    # encoder weights are loaded but never used or updated
    # could be optimized to only load decoder but this is simpler

    generator.train()
    # set to training mode so BatchNorm uses batch statistics
    # and any Dropout layers are active

    # ── Optimizer for generator only ──────────────────────────────────────────
    opt_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    # Adam with same betas as Phase 1 training
    # generator.parameters(): includes encoder too but encoder never
    # receives gradients in Phase 2 (no reconstruction loss, no KL loss)
    # so encoder weights stay frozen implicitly even though not explicitly frozen
    # only decoder weights get meaningful gradient updates

    best_loss = float("inf")
    best_state = None
    # track best generator state to save at the end

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════════

    for iteration in tqdm(range(N_ITERATIONS)):

        opt_G.zero_grad()
        # clear gradients from previous iteration

        # ── Step 1: sample random z from N(0,I) ───────────────────────────────
        z = torch.randn(1, LATENT_DIM, device=device)
        # shape: (1, 256) — one random latent vector
        # this is the KEY difference from Phase 1 attack:
        # z is NOT optimized — it's freshly sampled each iteration
        # the generator learns to handle ANY random z

        # ── Step 2: decode z → stamp ──────────────────────────────────────────
        stamp = generator.decode(z)
        # shape: (1, 1, 128, 128) — grayscale stamp image
        # generator.decode uses ONLY the decoder part of the VAE
        # gradients will flow back through this operation to decoder weights

        # ── Step 3: get document batch ────────────────────────────────────────
        doc_batch = get_document_batch(
            dataset, correct_images, target_class, BATCH_SIZE, device
        )
        # shape: (128, 3, 224, 224)
        # 128 documents from any class except target_class

        # ── Step 4: paste stamp on documents ──────────────────────────────────
        patched = paste_patch(
            doc_batch, stamp,
            patch_size=PATCH_SIZE,
            position=PATCH_POS
        )
        # shape: (128, 3, 224, 224)
        # stamp pasted at bottom-right corner of each document
        # gradient flows: patched → stamp pixels → decoder → generator weights
        # document pixels are dead ends — no gradient flows through them

        # ── Step 5: frozen ResNet → L_attack ──────────────────────────────────
        _, logits = resnet(patched)
        # logits shape: (128, 16) — one score per class per document
        # resnet is completely frozen — no gradient updates

        target_tensor = torch.full(
            (BATCH_SIZE,), target_class,
            dtype=torch.long, device=device
        )
        # tensor of target class indices: [c, c, c, ..., c] shape (128,)

        L_attack = F.cross_entropy(logits, target_tensor)
        # CrossEntropy between logits and target class
        # minimizing this pushes ResNet to predict target_class for all documents
        # gradient flows back through Resnet → patched image → stamp → generator

        # ── Step 6: frozen discriminator → L_realism ──────────────────────────
        L_realism = F.binary_cross_entropy(disc(stamp), torch.ones(1, device=device))
        # disc(stamp): discriminator evaluates the generated stamp
        # returns P(real) — probability that stamp looks real
        # torch.ones(1): target is 1 — we want disc to think stamp is REAL
        # disc is completely frozen — gradients flow through it into generator
        # but disc weights never update
        # this encourages generator to produce visually realistic stamps

        # ── Step 7: combine losses ────────────────────────────────────────────
        L_total = L_attack + LAMBDA_REAL * L_realism
        # L_attack: push prediction toward target class (main objective)
        # LAMBDA_REAL=0.05 * L_realism: keep stamps looking realistic (constraint)
        # same weighting as λ_adv in Phase 1

        # ── Step 8: backprop → update generator weights ───────────────────────
        L_total.backward()
        # gradients flow backwards:
        # L_attack → ResNet (frozen) → patched image → stamp pixels → decoder weights
        # L_realism → discriminator (frozen) → stamp pixels → decoder weights
        # encoder weights receive no gradient (no reconstruction, no KL)

        opt_G.step()
        # update generator (decoder) weights using Adam
        # lr=1e-4 — same as VAE learning rate in Phase 1

        # ── Logging ───────────────────────────────────────────────────────────
        if L_total.item() < best_loss:
            best_loss  = L_total.item()
            best_state = {k: v.clone() for k, v in generator.state_dict().items()}
            # save a copy of the generator weights when loss improves
            # .clone(): deep copy — not a reference to the current weights

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
    # return the best generator weights found during training


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)
    print("Dataset loaded")

    # ── Load frozen models ────────────────────────────────────────────────────

    # ResNet — completely frozen
    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device, weights_only= False))
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    # requires_grad=False: no gradient computation through ResNet
    # .eval(): BatchNorm uses running stats, Dropout disabled
    print("ResNet loaded and frozen")

    # Discriminator — completely frozen (already trained in Phase 1)
    disc = Discriminator(in_channels=1).to(device)
    disc.load_state_dict(torch.load(DISC_PATH, map_location=device, weights_only= False))
    disc = disc.to(device)
    disc.eval()
    for p in disc.parameters():
        p.requires_grad = False
    # requires_grad=False: gradients flow THROUGH disc but don't update it
    # this is the key difference from Phase 1 where disc was trainable
    # disc acts as a fixed realism evaluator
    # .eval(): BatchNorm uses running stats
    print("Discriminator loaded and frozen")

    # ── Create output directory ────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Train one generator per class ─────────────────────────────────────────
    for target_class in range(16):
        cls_name = CLASSES[target_class]

        # skip if already trained
        save_path = os.path.join(RESULTS_DIR, f"G_{target_class}_{cls_name}.pt")
        if os.path.exists(save_path):
            print(f"Skipping {cls_name} — already trained")
            continue

        # train generator for this class
        best_state = train_one_generator(
            target_class, dataset, correct_images,
            resnet, disc, device
        )

        # save best generator weights
        torch.save(best_state, save_path)
        print(f"Saved generator for '{cls_name}' → {save_path}")

    print("\nDone! All 16 generators trained.")
    print(f"Results saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()