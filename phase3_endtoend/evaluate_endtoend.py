"""
evaluate_endtoend.py

Evaluates the 16 trained generators from the end-to-end pipeline.

For each target class:
  - Load the trained generator G_c
  - Generate N_EVAL stamps by sampling random z vectors
  - Paste each stamp on evaluation documents
  - Measure attack success rate (ASR)

Usage:
    python3 attack/evaluate_endtoend.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from tqdm import tqdm
from datasets import load_from_disk
import torchvision

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from attack.patch_utils    import paste_patch, preprocess_document
from endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    VAE_PATH, RESNET_PATH, RESULTS_DIR,
    LATENT_DIM, PATCH_SIZE, PATCH_POS,
    CLASSES
)

N_EVAL       = 5
# number of different random z vectors to evaluate per class
# we sample N_EVAL different stamps and average their ASR
# this measures the AVERAGE performance across diverse stamps
# (unlike current pipeline which evaluates ONE specific optimized stamp)

DOCS_PER_CLASS = 134
# same as current pipeline evaluation for fair comparison


def evaluate_generator(target_class, dataset, correct_images,
                        resnet, device):
    """
    Evaluate a trained generator G_c on the full evaluation set.
    Samples N_EVAL different random z vectors and averages ASR.
    """
    cls_name  = CLASSES[target_class]
    save_path = os.path.join(RESULTS_DIR, f"G_{target_class}_{cls_name}.pt")

    if not os.path.exists(save_path):
        print(f"  {cls_name}: no trained generator found — skipping")
        return None

    # ── Load trained generator ────────────────────────────────────────────────
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(save_path, map_location=device))
    generator.eval()
    # .eval(): important for evaluation
    # BatchNorm uses running statistics (not batch statistics)
    # gives consistent outputs regardless of batch size

    # ── Build evaluation document set ─────────────────────────────────────────
    all_correct = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        # take exactly DOCS_PER_CLASS from each source class
        all_correct.extend(correct_images[str(cls_idx)][:DOCS_PER_CLASS])
    # total evaluation documents: 15 classes × 134 = 2010 documents

    success_rates = []

    for eval_idx in range(N_EVAL):
        # sample a fresh random z for this evaluation
        z = torch.randn(1, LATENT_DIM, device=device)
        # each iteration uses a completely different random stamp
        # this tests whether the generator produces good stamps
        # across the ENTIRE N(0,I) distribution — not just one lucky z

        with torch.no_grad():
            # no_grad: we're evaluating, not training — no gradients needed
            # saves memory and speeds up evaluation significantly
            stamp = generator.decode(z)
            # generate stamp from random z
            # shape: (1, 1, 128, 128)

        # save stamp image for visual inspection
        stamp_path = os.path.join(
            RESULTS_DIR,
            f"stamp_eval_{target_class}_{cls_name}_z{eval_idx}.png"
        )
        torchvision.utils.save_image(stamp, stamp_path)

        # evaluate on all documents
        total    = 0
        successes = 0

        # process in batches for efficiency
        batch_size = 64
        for i in range(0, len(all_correct), batch_size):
            batch_items = all_correct[i:i+batch_size]

            tensors = []
            for item in batch_items:
                pil_img = dataset[item["idx"]]["image"]
                tensors.append(preprocess_document(pil_img))
            doc_batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                patched   = paste_patch(doc_batch, stamp,
                                        patch_size=PATCH_SIZE,
                                        position=PATCH_POS)
                _, logits = resnet(patched)
                preds     = logits.argmax(dim=1)
                successes += (preds == target_class).sum().item()
                total     += len(batch_items)

        asr = successes / total * 100
        success_rates.append(asr)
        print(f"    z{eval_idx}: ASR = {asr:.1f}%")

    avg_asr = sum(success_rates) / len(success_rates)
    print(f"  {cls_name:<25}: avg ASR = {avg_asr:.1f}%  "
          f"(min={min(success_rates):.1f}%  max={max(success_rates):.1f}%)")

    return avg_asr


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)
    print("Dataset loaded")

    # ── Load frozen ResNet ────────────────────────────────────────────────────
    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet loaded and frozen")

    # ── Evaluate all 16 generators ────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION — end-to-end generators")
    print("="*60)

    results = {}
    for target_class in range(16):
        asr = evaluate_generator(
            target_class, dataset, correct_images, resnet, device
        )
        if asr is not None:
            results[target_class] = asr

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Class':<25} {'Avg ASR':>10}")
    print("-"*40)
    for cls_idx, asr in results.items():
        print(f"{CLASSES[cls_idx]:<25} {asr:>9.1f}%")
    print("-"*40)
    overall = sum(results.values()) / len(results)
    print(f"{'OVERALL':<25} {overall:>9.1f}%")
    print("="*60)

    # save results to json
    results_path = os.path.join(RESULTS_DIR, "evaluation_endtoend.json")
    with open(results_path, "w") as f:
        json.dump({
            "per_class": {CLASSES[k]: v for k, v in results.items()},
            "overall": overall
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()