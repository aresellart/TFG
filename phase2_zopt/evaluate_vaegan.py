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
from patch_utils import paste_patch, preprocess_document

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

DATASET_PATH = "/home/asellart/tfg_stamps/rvl_cdip_full"
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORRECT_JSON = os.path.join(ROOT, "correct_images.json")

LATENT_DIM   = 256
PATCH_SIZE   = 64
POSITION     = "bottom-right"

VARIANTS = [
    {
        "name":        "gpu_vaegan_random",
        "results_dir": os.path.join(ROOT, "attack_results_gpu_vaegan_random"),
        "model_path":  os.path.join(ROOT, "vae_gan_best.pt"),
        "use_v2":      True,
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── Load ResNet ───────────────────────────────────────────────────────────────
resnet = Model(device=device)
resnet.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"),
    map_location=device
))
resnet.eval()
resnet.to(device)
for p in resnet.parameters():
    p.requires_grad = False
print("ResNet-50 loaded")

# ── Load dataset and correct images ──────────────────────────────────────────
dataset = load_from_disk(DATASET_PATH)
with open(CORRECT_JSON) as f:
    correct_images = json.load(f)

N_DOCS = min(len(correct_images[str(i)]) for i in range(16))
print(f"Using {N_DOCS} documents per source class")
print("Dataset loaded")


# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_variant(variant):
    name        = variant["name"]
    results_dir = variant["results_dir"]
    model_path  = variant["model_path"]
    use_v2      = variant["use_v2"]

    print(f"\n{'='*60}")
    print(f"Evaluating variant: {name}")
    print(f"{'='*60}")

    if use_v2:
        from models.vanilla_vae_v2 import VanillaVAE
    else:
        from models.vanilla_vae import VanillaVAE

    vae = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    results = {}

    for target_class in range(16):
        target_name = CLASSES[target_class]
        z_path      = os.path.join(results_dir,
                                   f"z_class{target_class}_{target_name}.pt")

        if not os.path.exists(z_path):
            print(f"  No z found for {target_name}, skipping")
            continue

        z = torch.load(z_path, map_location=device)
        with torch.no_grad():
            stamp = vae.decode(z)

        results[target_class] = {}
        class_successes       = []

        for src_class in range(16):
            if src_class == target_class:
                continue

            candidates = correct_images[str(src_class)]
            sampled    = random.sample(candidates,
                                       min(N_DOCS, len(candidates)))
            total      = len(sampled)
            successes  = 0

            for item in tqdm(sampled,
                             desc=f"  {target_name} ← {CLASSES[src_class]}",
                             leave=False):
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
            results[target_class][src_class] = {
                "success_rate": rate,
                "successes":    successes,
                "total":        total,
            }
            class_successes.append(rate)

        avg = sum(class_successes) / len(class_successes)
        results[target_class]["average"] = avg
        print(f"  {target_name:<25}: {avg:.1f}%")

    # Save results
    out_path = os.path.join(ROOT, f"evaluation_{name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    return results


# ── Run evaluation ────────────────────────────────────────────────────────────
all_results = {}
for variant in VARIANTS:
    all_results[variant["name"]] = evaluate_variant(variant)

# ── Print results table ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"RESULTS — gpu_vanilla_encoded ({N_DOCS} docs per class)")
print("="*60)
print(f"{'Class':<25} {'Success Rate':>14}")
print("-"*42)

gpu_vanilla_avgs = []
for i, cls in enumerate(CLASSES):
    avg = all_results["gpu_vanilla_encoded"].get(i, {}).get("average", 0)
    gpu_vanilla_avgs.append(avg)
    print(f"{cls:<25} {avg:>13.1f}%")

print("-"*42)
overall = sum(gpu_vanilla_avgs) / len(gpu_vanilla_avgs)
print(f"{'OVERALL AVERAGE':<25} {overall:>13.1f}%")

# ── Bar chart ─────────────────────────────────────────────────────────────────
colors = ["green"  if r >= 75 else
          "orange" if r >= 50 else
          "red"    if r >  0  else "black"
          for r in gpu_vanilla_avgs]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(CLASSES, gpu_vanilla_avgs, color=colors,
              edgecolor="white", linewidth=0.5)
ax.set_ylim(0, 110)
ax.axhline(y=75,   color="green", linestyle="--", alpha=0.3)
ax.axhline(y=6.25, color="red",   linestyle="--", alpha=0.4,
           label="Random chance (6.25%)")
ax.set_ylabel("Attack success rate (%)", fontsize=12)
ax.set_title(
    f"VAE low-KL — encoded stamp (GPU, 2000 iter, batch 128)\n"
    f"{N_DOCS} documents per source class",
    fontsize=13
)
ax.set_xticks(range(16))
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
ax.legend(fontsize=10)

for bar, val in zip(bars, gpu_vanilla_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "evaluation_gpu_vanilla_chart.png"),
            dpi=150, bbox_inches="tight")
print("Saved evaluation_gpu_vanilla_chart.png")