import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from datasets import load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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

DATASET_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(ROOT, "attack_results")
PATCH_SIZE   = 64
POSITION     = "bottom-right"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── Load models ───────────────────────────────────────────────────────────────
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

print("Models loaded")

# ── Load dataset and correct image list ───────────────────────────────────────
dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, "correct_images.json")) as f:
    correct_images = json.load(f)
print("Dataset loaded")

# ── Evaluation loop ───────────────────────────────────────────────────────────
results = {}

for target_class in range(16):
    target_name = CLASSES[target_class]
    print(f"\nEvaluating stamp for target class: {target_name}")

    # Load optimized z for this target class
    z_path = os.path.join(RESULTS_DIR, f"z_class{target_class}_{target_name}.pt")
    if not os.path.exists(z_path):
        print(f"  No z found for {target_name}, skipping")
        continue

    z = torch.load(z_path, map_location=device)
    with torch.no_grad():
        stamp = vae.decode(z)  # (1, 1, 128, 128)

    results[target_class] = {}
    class_successes = []

    # Test on ALL available documents from each non-target class
    for src_class in range(16):
        if src_class == target_class:
            continue

        # Use ALL available images for this source class
        candidates = correct_images[str(src_class)]
        total      = len(candidates)

        successes = 0
        for item in tqdm(candidates, desc=f"  src={CLASSES[src_class]} ({total} imgs)", leave=False):
            try:
                pil_img = dataset[item["idx"]]["image"]
                tensor  = preprocess_document(pil_img).unsqueeze(0).to(device)

                # Paste stamp
                patched = paste_patch(tensor, stamp,
                                      patch_size=PATCH_SIZE,
                                      position=POSITION)

                # Classify
                with torch.no_grad():
                    _, logits = resnet(patched)
                pred = logits.argmax(dim=1).item()

                if pred == target_class:
                    successes += 1
            except Exception:
                continue

        rate = successes / total * 100
        results[target_class][src_class] = {
            "success_rate": rate,
            "successes":    successes,
            "total":        total,
            "src_class":    CLASSES[src_class],
        }
        class_successes.append(rate)
        print(f"  {CLASSES[src_class]:25s}: {successes}/{total} = {rate:.1f}%")

    avg = sum(class_successes) / len(class_successes)
    results[target_class]["average"] = avg
    print(f"  → Average: {avg:.1f}%")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL EVALUATION RESULTS (full test subset)")
print("="*60)
print(f"{'Class':<25} {'Total docs tested':>18} {'Avg Success Rate':>16}")
print("-"*60)

for target_class in range(16):
    if target_class not in results:
        continue
    avg   = results[target_class]["average"]
    total = sum(
        results[target_class][src]["total"]
        for src in range(16)
        if src != target_class and src in results[target_class]
    )
    print(f"{CLASSES[target_class]:<25} {total:>18} {avg:>15.1f}%")

overall = sum(results[c]["average"] for c in results) / len(results)
print("-"*60)
print(f"{'OVERALL AVERAGE':<25} {'':>18} {overall:>15.1f}%")

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS_DIR, "evaluation_results_full.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved evaluation_results_full.json")

# ── Bar chart ─────────────────────────────────────────────────────────────────
avg_rates = [results[c]["average"] if c in results else 0 for c in range(16)]
colors    = ["green"  if r >= 75 else
             "orange" if r >= 50 else
             "red"    if r >  0  else "black"
             for r in avg_rates]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(CLASSES, avg_rates, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylim(0, 110)
ax.axhline(y=75, color="green",  linestyle="--", alpha=0.3)
ax.axhline(y=50, color="orange", linestyle="--", alpha=0.3)
ax.set_ylabel("Attack success rate (%)", fontsize=12)
ax.set_title("Attack success rate per target class\n(full test subset evaluation)", fontsize=13)
ax.set_xticks(range(16))
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)

for bar, val in zip(bars, avg_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evaluation_chart_full.png"), dpi=150, bbox_inches="tight")
print("Saved evaluation_chart_full.png")