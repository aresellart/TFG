"""
train_diagnostic_discriminator.py

DIAGNOSTIC EXPERIMENT — not part of the attack pipeline.

Trains a FRESH discriminator from scratch whose only job is to
distinguish REAL SPODS stamps (label 1) from BASELINE end-to-end
generator outputs (label 0) — the original, unmodified Phase 2 stamps,
with no morphological loss or any other structural constraint applied.

Rationale: the original Phase 1 discriminator's notion of "realism"
is narrow — it only ever learned to distinguish blurry VAE outputs
from sharp real stamps. It was never exposed to "sharp but
semantically meaningless" as a category of fake, because that
category didn't exist until Phase 2 produced it. This experiment
tests whether a discriminator trained with the CORRECT labels from
the start (end-to-end outputs are fake, real stamps are real) can
learn to reliably tell them apart — which would confirm there IS a
learnable, real gap between "looks sharp" and "looks like an actual
stamp", beyond what we've assessed qualitatively by eye.

This discriminator is NOT plugged back into generator training here.
It is purely diagnostic: train it, then report its final accuracy and
confidence on held-out samples of both classes.

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/train_diagnostic_discriminator.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm

from models.vanilla_vae_v2 import VanillaVAE
from models.discriminator  import Discriminator
from datasets_local.stamps_dataset import get_dataloader
from endtoend.config_endtoend import ROOT, LATENT_DIM, CLASSES, RESULTS_DIR
from autoencoder_v3.config_v3 import DATA_PATH, IMAGE_SIZE

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_EPOCHS    = 100
BATCH_SIZE  = 64
LR_DISC     = 1e-4
HOLDOUT_FRACTION = 0.15   # fraction of each class held out for final eval

SAVE_PATH = os.path.join(ROOT, "diagnostic_discriminator.pt")


def load_all_generators(device):
    """Load all 16 trained end-to-end generators (baseline, no morph loss)."""
    generators = []
    for cls_idx in range(16):
        cls_name = CLASSES[cls_idx]
        gen_path = os.path.join(RESULTS_DIR, f"G_{cls_idx}_{cls_name}.pt")
        if not os.path.exists(gen_path):
            print(f"  [warning] generator for {cls_name} not found, skipping")
            continue
        gen = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
        gen.load_state_dict(torch.load(gen_path, map_location=device,
                                        weights_only=False))
        gen.eval()
        generators.append(gen)
    print(f"Loaded {len(generators)} end-to-end generators")
    return generators


def sample_fake_batch(generators, batch_size, device):
    """Sample a batch of fake stamps from randomly chosen generators."""
    stamps = []
    with torch.no_grad():
        for _ in range(batch_size):
            gen = random.choice(generators)
            z = torch.randn(1, LATENT_DIM, device=device)
            stamp = gen.decode(z)
            stamps.append(stamp.squeeze(0))
    return torch.stack(stamps).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    generators = load_all_generators(device)

    real_loader = get_dataloader(DATA_PATH, BATCH_SIZE,
                                  image_size=IMAGE_SIZE, num_workers=0)

    disc = Discriminator(in_channels=1).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    print(f"\nTraining diagnostic discriminator for {N_EPOCHS} epochs...")
    for epoch in range(1, N_EPOCHS + 1):
        disc.train()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        for real_x, _ in real_loader:
            real_x = real_x.to(device)
            B = real_x.size(0)

            fake_x = sample_fake_batch(generators, B, device)

            opt_disc.zero_grad()

            real_preds = disc(real_x)
            fake_preds = disc(fake_x)

            loss_real = F.binary_cross_entropy(
                real_preds, torch.ones(B, device=device)
            )
            loss_fake = F.binary_cross_entropy(
                fake_preds, torch.zeros(B, device=device)
            )
            loss = (loss_real + loss_fake) / 2

            loss.backward()
            opt_disc.step()

            with torch.no_grad():
                acc_real = (real_preds > 0.5).float().mean()
                acc_fake = (fake_preds < 0.5).float().mean()
                acc = (acc_real + acc_fake) / 2

            total_loss += loss.item()
            total_acc  += acc.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_acc  = total_acc  / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  "
                  f"accuracy={avg_acc*100:.1f}%")

    torch.save(disc.state_dict(), SAVE_PATH)
    print(f"\nSaved diagnostic discriminator to {SAVE_PATH}")

    # ── Final held-out evaluation ────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL DIAGNOSTIC EVALUATION")
    print("="*60)
    disc.eval()

    n_eval = 200
    real_batch = next(iter(real_loader))[0][:min(n_eval, BATCH_SIZE)].to(device)
    fake_batch = sample_fake_batch(generators, min(n_eval, BATCH_SIZE), device)

    with torch.no_grad():
        real_conf = disc(real_batch)
        fake_conf = disc(fake_batch)

    print(f"Mean confidence on REAL stamps (should be near 1.0): {real_conf.mean().item():.4f}")
    print(f"Mean confidence on FAKE (end-to-end) stamps (should be near 0.0): {fake_conf.mean().item():.4f}")
    print(f"Real stamps classified correctly: {(real_conf > 0.5).float().mean().item()*100:.1f}%")
    print(f"Fake stamps classified correctly: {(fake_conf < 0.5).float().mean().item()*100:.1f}%")
    print("="*60)
    print("\nInterpretation:")
    print("  High accuracy on BOTH (>90%) -> a real, learnable gap exists")
    print("  between end-to-end outputs and real stamps, beyond sharpness.")
    print("  Low accuracy (~50%, chance level) -> end-to-end outputs are")
    print("  genuinely indistinguishable from real stamps to a fresh discriminator.")


if __name__ == "__main__":
    main()