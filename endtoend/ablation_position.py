"""
ablation_position.py

Ablation study: effect of stamp position on attack success rate.
Tests 6 positions using the already-trained end-to-end generators.
No retraining needed — pure evaluation experiment.

Positions tested:
  - top-left
  - top-right
  - bottom-left
  - bottom-right (current baseline)
  - center
  - random (different random position per document)

Usage:
    python3 endtoend/ablation_position.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from attack.patch_utils    import preprocess_document
from endtoend.config_endtoend import (
    DATASET_PATH, CORRECT_JSON,
    RESNET_PATH, RESULTS_DIR,
    LATENT_DIM, PATCH_SIZE,
    CLASSES
)

# ── POSITIONS ─────────────────────────────────────────────────────────────────
# document is 224×224, stamp is PATCH_SIZE×PATCH_SIZE (64×64)
# position = top-left corner of stamp in the document

DOC_SIZE = 224

POSITIONS = {
    "top-left":     (0, 0),
    "top-right":    (0, DOC_SIZE - PATCH_SIZE),
    "bottom-left":  (DOC_SIZE - PATCH_SIZE, 0),
    "bottom-right": (DOC_SIZE - PATCH_SIZE, DOC_SIZE - PATCH_SIZE),  # current baseline
    "center":       ((DOC_SIZE - PATCH_SIZE) // 2, (DOC_SIZE - PATCH_SIZE) // 2),
    "random":       None,  # computed per document
}

N_EVAL         = 5    # random z samples per class per position
DOCS_PER_CLASS = 134  # same as main evaluation
ABL_DIR        = os.path.join(RESULTS_DIR, "ablation_position")
os.makedirs(ABL_DIR, exist_ok=True)


def paste_patch_position(doc_batch, stamp, row, col):
    """
    Paste stamp at a specific (row, col) position in the document.
    row, col: top-left corner of the stamp region.
    Differentiable — gradients flow through stamp pixels.
    """
    patched = doc_batch.clone()
    # resize stamp from 128×128 to PATCH_SIZE×PATCH_SIZE
    stamp_resized = F.interpolate(stamp, size=(PATCH_SIZE, PATCH_SIZE),
                                   mode='bilinear', align_corners=False)
    # convert grayscale stamp to 3 channels
    stamp_3ch = stamp_resized.repeat(1, 3, 1, 1)
    # normalize with ImageNet stats (mean and std for each channel)
    mean = torch.tensor([0.485, 0.456, 0.406],
                         device=doc_batch.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225],
                         device=doc_batch.device).view(1, 3, 1, 1)
    stamp_norm = (stamp_3ch - mean) / std
    # paste at specified position
    patched[:, :, row:row+PATCH_SIZE, col:col+PATCH_SIZE] = stamp_norm
    return patched


def paste_patch_random(doc_batch, stamp, device):
    """
    Paste stamp at a DIFFERENT random position for each document in the batch.
    Tests whether position matters or if the stamp works anywhere.
    """
    patched = doc_batch.clone()
    stamp_resized = F.interpolate(stamp, size=(PATCH_SIZE, PATCH_SIZE),
                                   mode='bilinear', align_corners=False)
    stamp_3ch  = stamp_resized.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    stamp_norm = (stamp_3ch - mean) / std

    B = doc_batch.size(0)
    max_pos = DOC_SIZE - PATCH_SIZE  # 160

    for i in range(B):
        # different random position for each document
        row = random.randint(0, max_pos)
        col = random.randint(0, max_pos)
        patched[i, :, row:row+PATCH_SIZE, col:col+PATCH_SIZE] = stamp_norm[0]

    return patched


def evaluate_position(target_class, position_name, position_coords,
                       dataset, correct_images, resnet, device):
    """
    Evaluate one generator at one position.
    Returns average ASR across N_EVAL random z samples.
    """
    cls_name  = CLASSES[target_class]
    save_path = os.path.join(RESULTS_DIR, f"G_{target_class}_{cls_name}.pt")

    if not os.path.exists(save_path):
        return None

    # load generator
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(save_path, map_location=device,
                                          weights_only=False))
    generator.eval()

    # build evaluation document set
    all_docs = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        all_docs.extend(correct_images[str(cls_idx)][:DOCS_PER_CLASS])

    success_rates = []
    torch.manual_seed(42 + target_class)

    for _ in range(N_EVAL):
        z = torch.randn(1, LATENT_DIM, device=device)

        with torch.no_grad():
            stamp = generator.decode(z)

        total     = 0
        successes = 0
        batch_size = 64

        for i in range(0, len(all_docs), batch_size):
            batch_items = all_docs[i:i+batch_size]
            tensors = []
            for item in batch_items:
                pil_img = dataset[item["idx"]]["image"]
                tensors.append(preprocess_document(pil_img))
            doc_batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                if position_name == "random":
                    patched = paste_patch_random(doc_batch, stamp, device)
                else:
                    row, col = position_coords
                    patched  = paste_patch_position(doc_batch, stamp, row, col)

                _, logits = resnet(patched)
                preds     = logits.argmax(dim=1)
                successes += (preds == target_class).sum().item()
                total     += len(batch_items)

        asr = successes / total * 100
        success_rates.append(asr)

    return sum(success_rates) / len(success_rates)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}\n")

    # load dataset
    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)

    # load frozen ResNet
    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device,
                                       weights_only=False))
    resnet = resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet loaded and frozen\n")

    # ── RUN ABLATION ──────────────────────────────────────────────────────────
    # results[position][class] = ASR
    results = {pos: {} for pos in POSITIONS}

    for pos_name, pos_coords in POSITIONS.items():
        print(f"\n{'='*55}")
        print(f"Position: {pos_name}")
        print(f"{'='*55}")

        class_asrs = []
        for target_class in range(16):
            cls_name = CLASSES[target_class]
            asr = evaluate_position(
                target_class, pos_name, pos_coords,
                dataset, correct_images, resnet, device
            )
            if asr is not None:
                results[pos_name][target_class] = asr
                class_asrs.append(asr)
                print(f"  {cls_name:<25}: {asr:.1f}%")

        overall = sum(class_asrs) / len(class_asrs)
        results[pos_name]["overall"] = overall
        print(f"  {'OVERALL':<25}: {overall:.1f}%")

    # ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────
    print("\n\n" + "="*75)
    print("ABLATION STUDY — STAMP POSITION")
    print("="*75)
    header = f"{'Class':<25}" + "".join(f"{p:>13}" for p in POSITIONS)
    print(header)
    print("-"*75)

    for cls_idx in range(16):
        row = f"{CLASSES[cls_idx]:<25}"
        for pos_name in POSITIONS:
            asr = results[pos_name].get(cls_idx, 0)
            row += f"{asr:>12.1f}%"
        print(row)

    print("-"*75)
    overall_row = f"{'OVERALL':<25}"
    for pos_name in POSITIONS:
        overall_row += f"{results[pos_name].get('overall', 0):>12.1f}%"
    print(overall_row)
    print("="*75)

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    save_path = os.path.join(ABL_DIR, "ablation_position_results.json")
    with open(save_path, "w") as f:
        # convert int keys to strings for JSON
        json_results = {
            pos: {CLASSES[k] if isinstance(k, int) else k: v
                  for k, v in d.items()}
            for pos, d in results.items()
        }
        json.dump(json_results, f, indent=2)
    print(f"\nSaved results to {save_path}")


if __name__ == "__main__":
    main()