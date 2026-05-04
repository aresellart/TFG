import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from datasets import load_from_disk
from tqdm import tqdm

from models.model_resnet50 import Model
from models.vanilla_vae import VanillaVAE
from patch_utils import paste_patch, preprocess_document
import config

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

TARGET_CLASSES = [4, 1, 9, 14]  # same 4 classes

DATASET_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(ROOT, "attack_results_random_z")
PATCH_SIZE   = 64
POSITION     = "bottom-right"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

resnet = Model(device=device)
resnet.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"), map_location=device
))
resnet.eval()
for p in resnet.parameters():
    p.requires_grad = False

vae = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"), map_location=device
))
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, "correct_images.json")) as f:
    correct_images = json.load(f)
print("Models and dataset loaded")

results = {}

for target_class in TARGET_CLASSES:
    target_name = CLASSES[target_class]
    print(f"\nEvaluating: {target_name}")

    z_path = os.path.join(RESULTS_DIR, f"z_class{target_class}_{target_name}.pt")
    if not os.path.exists(z_path):
        print(f"  No z found, skipping")
        continue

    z = torch.load(z_path, map_location=device)
    with torch.no_grad():
        stamp = vae.decode(z)

    results[target_class] = {}
    class_successes = []

    for src_class in range(16):
        if src_class == target_class:
            continue
        candidates = correct_images[str(src_class)]
        total      = len(candidates)
        successes  = 0

        for item in tqdm(candidates, desc=f"  {CLASSES[src_class]}", leave=False):
            try:
                pil_img = dataset[item["idx"]]["image"]
                tensor  = preprocess_document(pil_img).unsqueeze(0).to(device)
                patched = paste_patch(tensor, stamp,
                                      patch_size=PATCH_SIZE,
                                      position=POSITION)
                with torch.no_grad():
                    _, logits = resnet(patched)
                if logits.argmax(dim=1).item() == target_class:
                    successes += 1
            except Exception:
                continue

        rate = successes / total * 100
        results[target_class][src_class] = rate
        class_successes.append(rate)

    avg = sum(class_successes) / len(class_successes)
    results[target_class]["average"] = avg
    print(f"  → Average success rate: {avg:.1f}%")

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("COMPARISON: Real stamp encoding vs N(0,I) random sampling")
print("="*65)

# Results from your previous run (real stamp encoding)
real_stamp_results = {
    4:  97.5,   # form
    1:  66.4,   # memo
    9:  56.8,   # news article
    14:  0.9,   # scientific report
}

print(f"{'Class':<25} {'Real stamp z':>14} {'Random N(0,I)':>14}")
print("-"*55)
for target_class in TARGET_CLASSES:
    cls_name   = CLASSES[target_class]
    real_rate  = real_stamp_results[target_class]
    rand_rate  = results[target_class]["average"] if target_class in results else 0.0
    print(f"{cls_name:<25} {real_rate:>13.1f}%  {rand_rate:>13.1f}%")

print("-"*55)
real_avg = sum(real_stamp_results.values()) / len(real_stamp_results)
rand_avg = sum(results[c]["average"] for c in results if "average" in results[c]) / len(TARGET_CLASSES)
print(f"{'Average':<25} {real_avg:>13.1f}%  {rand_avg:>13.1f}%")

with open(os.path.join(RESULTS_DIR, "comparison_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved comparison_results.json")