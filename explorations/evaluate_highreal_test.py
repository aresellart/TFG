"""
evaluate_highreal_test.py

Evaluates the 3 test generators trained with raised lambda_real (0.3)
in attack_results_endtoend_highreal_test/.

For each test class:
  - Load the trained generator (lambda_real=0.3 version)
  - Generate N_EVAL stamps by sampling random z vectors
  - Paste each stamp at a RANDOM position (matching training distribution)
  - Measure attack success rate (ASR)
  - Save stamp images for visual comparison against the original
    lambda_real=0.05 stamps

Usage:
    cd /home/asellart/tfg_stamps
    python3 explorations/evaluate_highreal_test.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from datasets import load_from_disk
import torchvision

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from common.patch_utils    import preprocess_document
from phase3_endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    RESNET_PATH,
    LATENT_DIM, PATCH_SIZE,
    CLASSES
)

RESULTS_DIR_TEST = os.path.join(ROOT, "attack_results_endtoend_highreal_test")

LAMBDA_REAL_TEST = 0.3
TEST_CLASSES     = [0, 7, 13]   # letter, advertisement, resume

N_EVAL         = 5
DOCS_PER_CLASS = 134
DOC_SIZE       = 224
MAX_POS        = DOC_SIZE - PATCH_SIZE


def paste_patch_random_pos_eval(doc_batch, stamp, device):
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


def evaluate_generator(target_class, dataset, correct_images,
                        resnet, device):
    cls_name  = CLASSES[target_class]
    save_path = os.path.join(
        RESULTS_DIR_TEST,
        f"G_{target_class}_{cls_name}_lr{LAMBDA_REAL_TEST}.pt"
    )

    if not os.path.exists(save_path):
        print(f"  {cls_name}: no trained generator found at {save_path} — skipping")
        return None

    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(save_path, map_location=device,
                                          weights_only=False))
    generator.eval()

    all_correct = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        all_correct.extend(correct_images[str(cls_idx)][:DOCS_PER_CLASS])

    success_rates = []

    for eval_idx in range(N_EVAL):
        z = torch.randn(1, LATENT_DIM, device=device)

        with torch.no_grad():
            stamp = generator.decode(z).to(device)

        stamp_path = os.path.join(
            RESULTS_DIR_TEST,
            f"stamp_highreal_{target_class}_{cls_name}_lr{LAMBDA_REAL_TEST}_z{eval_idx}.png"
        )
        torchvision.utils.save_image(stamp, stamp_path)

        total     = 0
        successes = 0
        batch_size = 64

        for i in range(0, len(all_correct), batch_size):
            batch_items = all_correct[i:i+batch_size]

            tensors = []
            for item in batch_items:
                pil_img = dataset[item["idx"]]["image"]
                tensors.append(preprocess_document(pil_img))
            doc_batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                patched   = paste_patch_random_pos_eval(doc_batch, stamp, device)
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
    print(f"Evaluating lambda_real={LAMBDA_REAL_TEST} test generators")

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

    print("\n" + "="*60)
    print(f"EVALUATION — high realism test (lambda_real={LAMBDA_REAL_TEST})")
    print("="*60)

    results = {}
    for target_class in TEST_CLASSES:
        cls_name = CLASSES[target_class]
        print(f"\n{cls_name}:")
        asr = evaluate_generator(
            target_class, dataset, correct_images, resnet, device
        )
        if asr is not None:
            results[target_class] = asr

    print("\n" + "="*60)
    print(f"RESULTS SUMMARY — lambda_real={LAMBDA_REAL_TEST}")
    print("="*60)
    print(f"{'Class':<25} {'Avg ASR':>10}   {'Original (lr=0.05)':>20}")
    print("-"*60)

    original_asr = {
        0:  98.8,
        7:  99.7,
        13: 99.8,
    }

    for cls_idx, asr in results.items():
        orig = original_asr.get(cls_idx, None)
        orig_str = f"{orig:.1f}%" if orig is not None else "n/a"
        print(f"{CLASSES[cls_idx]:<25} {asr:>9.1f}%   {orig_str:>20}")
    print("="*60)

    results_path = os.path.join(RESULTS_DIR_TEST, "evaluation_highreal_test.json")
    with open(results_path, "w") as f:
        json.dump({
            "lambda_real": LAMBDA_REAL_TEST,
            "per_class": {CLASSES[k]: v for k, v in results.items()},
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")
    print(f"Stamp images saved in {RESULTS_DIR_TEST}/ as stamp_highreal_*.png")
    print("\nNext: visually compare stamp_highreal_*.png against the original")
    print("stamp_eval_*.png images from attack_results_endtoend_randpos/")


if __name__ == "__main__":
    main()