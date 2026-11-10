"""
ablation_size.py

Stamp SIZE ablation study using the fixed bottom-right generators
(from attack_results_endtoend/).

Tests 4 patch sizes: 16x16, 32x32, 64x64 (baseline), 96x96
Position: always bottom-right (same as training)
Generators: attack_results_endtoend/ (trained with fixed bottom-right)

The 64x64 result should match the original 97.7% — use it as a sanity check.

Usage:
    python3 ablations/ablation_size.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
from datasets import load_from_disk
import torchvision.transforms.functional as TF

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from common.patch_utils import preprocess_document
from phase3_endtoend.config_endtoend import (
    ROOT, DATASET_PATH, CORRECT_JSON,
    VAE_PATH, RESNET_PATH, RESULTS_DIR,
    LATENT_DIM, CLASSES
)

# ── Configuration ────────────────────────────────────────────────────────────

PATCH_SIZES    = [16, 32, 64, 96]   # pixels — 64 is the baseline
POSITION       = "bottom-right"      # same as training
DOC_SIZE       = 224                 # ResNet input size
DOCS_PER_CLASS = 134                 # same as main evaluation
N_STAMPS       = 5                   # random z vectors per class per size

# ── Paste helper ─────────────────────────────────────────────────────────────

def paste_patch_size(doc_batch, stamp, patch_size, device):
    """
    Resize stamp to patch_size x patch_size and paste at bottom-right.
    stamp shape: (1, 1, 128, 128)  grayscale [0,1]
    doc_batch shape: (B, 3, 224, 224)  ImageNet-normalised RGB
    returns: (B, 3, 224, 224) patched documents
    """
    # resize stamp to target size
    stamp_resized = TF.resize(stamp, [patch_size, patch_size],
                               interpolation=TF.InterpolationMode.BILINEAR,
                               antialias=True)
    # convert grayscale → RGB by repeating channel
    stamp_rgb = stamp_resized.repeat(1, 3, 1, 1).to(device)   # (1, 3, ps, ps)  in [0,1]

    # ImageNet normalise the stamp (same as preprocess_document)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    stamp_norm = (stamp_rgb - mean) / std              # (1, 3, ps, ps)

    # paste at bottom-right corner
    row = DOC_SIZE - patch_size
    col = DOC_SIZE - patch_size

    patched = doc_batch.clone()
    patched[:, :, row:row+patch_size, col:col+patch_size] = stamp_norm

    return patched


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_size(target_class, patch_size, dataset, correct_images,
                  resnet, device):
    """
    Load the trained generator for target_class, generate N_STAMPS stamps,
    paste at the given patch_size bottom-right, and return average ASR.
    """
    cls_name  = CLASSES[target_class]
    gen_path  = os.path.join(RESULTS_DIR, f"G_{target_class}_{cls_name}.pt")

    if not os.path.exists(gen_path):
        return None

    # load generator
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    generator.eval()

    # build evaluation document set (same 134 per class as main eval)
    all_docs = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        all_docs.extend(correct_images[str(cls_idx)][:DOCS_PER_CLASS])

    success_rates = []

    for _ in range(N_STAMPS):
        z = torch.randn(1, LATENT_DIM, device=device)

        with torch.no_grad():
            stamp = generator.decode(z).to(device)   # (1, 1, 128, 128)

        successes = 0
        total     = 0
        batch_sz  = 64

        for i in range(0, len(all_docs), batch_sz):
            items = all_docs[i:i+batch_sz]

            tensors = []
            for item in items:
                pil_img = dataset[item["idx"]]["image"]
                tensors.append(preprocess_document(pil_img))
            doc_batch = torch.stack(tensors).to(device)

            with torch.no_grad():
                patched   = paste_patch_size(doc_batch, stamp, patch_size, device)
                _, logits = resnet(patched)
                preds     = logits.argmax(dim=1)
                successes += (preds == target_class).sum().item()
                total     += len(items)

        success_rates.append(successes / total * 100)

    return sum(success_rates) / len(success_rates)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load dataset and correct images
    dataset = load_from_disk(DATASET_PATH)
    with open(CORRECT_JSON) as f:
        correct_images = json.load(f)
    print("Dataset loaded")

    # load frozen ResNet
    resnet = Model(device=device)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    resnet.to(device)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet loaded and frozen\n")

    # results[patch_size][cls_name] = avg_asr
    results = {ps: {} for ps in PATCH_SIZES}

    for patch_size in PATCH_SIZES:
        print(f"\n{'='*60}")
        print(f"PATCH SIZE: {patch_size}x{patch_size}  (position: {POSITION})")
        print(f"{'='*60}")

        asrs = []
        for target_class in range(16):
            cls_name = CLASSES[target_class]
            asr = evaluate_size(
                target_class, patch_size,
                dataset, correct_images,
                resnet, device
            )
            if asr is None:
                print(f"  {cls_name:<25}: generator not found — skip")
                continue

            results[patch_size][cls_name] = asr
            asrs.append(asr)
            print(f"  {cls_name:<25}: {asr:.1f}%")

        overall = sum(asrs) / len(asrs) if asrs else 0
        results[patch_size]["_overall"] = overall
        print(f"  {'OVERALL':<25}: {overall:.1f}%")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("SUMMARY — overall ASR by patch size")
    print(f"{'='*60}")
    print(f"{'Patch size':<15} {'Overall ASR':>12}")
    print("-"*30)
    for ps in PATCH_SIZES:
        tag = "  ← baseline" if ps == 64 else ""
        print(f"{str(ps)+'x'+str(ps):<15} {results[ps].get('_overall', 0):>11.1f}%{tag}")

    # save results
    out_path = os.path.join(RESULTS_DIR, "ablation_size.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()