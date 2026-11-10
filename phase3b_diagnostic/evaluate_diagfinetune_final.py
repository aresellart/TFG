"""
evaluate_diagfinetune_final.py

Evaluates the 16 diagnostic-discriminator fine-tuned generators in
attack_results_endtoend_diagfinetune_final/.

For each target class: load the fine-tuned generator, sample 5 random
z vectors, paste each stamp at a RANDOM position, measure ASR, and
save stamp images for qualitative inspection.

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/evaluate_diagfinetune_final.py
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
from attack.patch_utils    import preprocess_document
from endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    RESNET_PATH,
    LATENT_DIM, PATCH_SIZE,
    CLASSES
)

RESULTS_DIR_FINAL = os.path.join(ROOT, "attack_results_endtoend_diagfinetune_final")

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


def evaluate_generator(target_class, dataset, correct_images, resnet, device):
    cls_name  = CLASSES[target_class]
    gen_path  = os.path.join(RESULTS_DIR_FINAL,
                              f"G_{target_class}_{cls_name}_diagfinetune.pt")

    if not os.path.exists(gen_path):
        print(f"  {cls_name}: generator not found — skipping")
        return None

    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device,
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
            RESULTS_DIR_FINAL,
            f"stamp_eval_{target_class}_{cls_name}_z{eval_idx}.png"
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

    avg_asr = sum(success_rates) / len(success_rates)
    print(f"  {cls_name:<25}: avg ASR = {avg_asr:.1f}%  "
          f"(min={min(success_rates):.1f}%  max={max(success_rates):.1f}%)")

    return avg_asr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

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
    print("EVALUATION — diagnostic-discriminator fine-tuned model (16 classes)")
    print("="*60)

    results = {}
    for target_class in range(16):
        cls_name = CLASSES[target_class]
        asr = evaluate_generator(target_class, dataset, correct_images,
                                  resnet, device)
        if asr is not None:
            results[target_class] = asr

    print("\n" + "="*60)
    print("RESULTS SUMMARY — diagnostic fine-tuned model")
    print("="*60)
    print(f"{'Class':<25} {'Avg ASR':>10}")
    print("-"*40)
    for cls_idx, asr in results.items():
        print(f"{CLASSES[cls_idx]:<25} {asr:>9.1f}%")
    print("-"*40)
    overall = sum(results.values()) / len(results)
    print(f"{'OVERALL':<25} {overall:>9.1f}%")
    print("="*60)

    results_path = os.path.join(RESULTS_DIR_FINAL, "evaluation_diagfinetune.json")
    with open(results_path, "w") as f:
        json.dump({
            "per_class": {CLASSES[k]: v for k, v in results.items()},
            "overall": overall
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")
    print(f"Stamp images saved in {RESULTS_DIR_FINAL}/ as stamp_eval_*.png")


if __name__ == "__main__":
    main()