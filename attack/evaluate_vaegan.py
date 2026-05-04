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

from models.vanilla_vae_v2 import VanillaVAE
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

LATENT_DIM = 256
PATCH_SIZE  = 64
POSITION    = "bottom-right"

VARIANTS = [
    {
        "name":        "vaegan_encoded_stamp",
        "results_dir": os.path.join(ROOT, "attack_results_vaegan_encoded_stamp"),
        "model_path":  os.path.join(ROOT, "vae_gan_best.pt"),
        "use_v2":      True,
    },
    {
        "name":        "vaegan_random_z",
        "results_dir": os.path.join(ROOT, "attack_results_vaegan_random_z"),
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

# Use minimum available docs per class for fair comparison
N_DOCS = min(len(correct_images[str(i)]) for i in range(16))
print(f"Documents per source class: {N_DOCS} (minimum available across all classes)")
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<25}: {len(correct_images[str(i)])} available → using {min(N_DOCS, len(correct_images[str(i)]))}")

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
        from models.vanilla_vae_v2 import VanillaVAE as VAE
    else:
        from models.vanilla_vae import VanillaVAE as VAE

    vae = VAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
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

# ── Comparison table ──────────────────────────────────────────────────────────
# Baseline results from original VAE low-KL evaluation (full test subset)
baseline_encoded = {
    0: 65.2, 1: 66.4, 2: 64.4, 3: 69.8, 4: 97.5,
    5: 60.3, 6: 70.2, 7: 65.0, 8: 94.9, 9: 56.8,
    10: 94.6, 11: 82.8, 12: 60.2, 13: 80.4, 14: 0.9, 15: 84.2
}

print("\n" + "="*80)
print(f"COMPARISON TABLE — {N_DOCS} docs per source class (equal evaluation)")
print("="*80)
print(f"{'Class':<25} {'VAE encoded':>14} {'VAEGAN encoded':>14} {'VAEGAN random':>14}")
print("-"*70)

vaegan_enc_avgs = []
vaegan_rnd_avgs = []

for i, cls in enumerate(CLASSES):
    base     = baseline_encoded.get(i, 0)
    vgan_enc = all_results["vaegan_encoded_stamp"].get(i, {}).get("average", 0)
    vgan_rnd = all_results["vaegan_random_z"].get(i, {}).get("average", 0)
    vaegan_enc_avgs.append(vgan_enc)
    vaegan_rnd_avgs.append(vgan_rnd)
    print(f"{cls:<25} {base:>13.1f}%  {vgan_enc:>13.1f}%  {vgan_rnd:>13.1f}%")

print("-"*70)
base_avg     = sum(baseline_encoded.values()) / 16
vgan_enc_avg = sum(vaegan_enc_avgs) / len(vaegan_enc_avgs)
vgan_rnd_avg = sum(vaegan_rnd_avgs) / len(vaegan_rnd_avgs)
print(f"{'OVERALL AVERAGE':<25} {base_avg:>13.1f}%  {vgan_enc_avg:>13.1f}%  {vgan_rnd_avg:>13.1f}%")
print(f"\nNote: VAE encoded baseline used full test subset; VAE-GAN variants use {N_DOCS} docs per class")

# ── Bar chart ─────────────────────────────────────────────────────────────────
x     = np.arange(len(CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 7))
ax.bar(x - width, list(baseline_encoded.values()),
       width, label="VAE low-KL (encoded stamp)", color="#378ADD", alpha=0.85)
ax.bar(x,         vaegan_enc_avgs,
       width, label="VAE-GAN (encoded stamp)", color="#1D9E75", alpha=0.85)
ax.bar(x + width, vaegan_rnd_avgs,
       width, label="VAE-GAN (random z)", color="#BA7517", alpha=0.85)

ax.set_ylabel("Attack success rate (%)", fontsize=12)
ax.set_title(
    f"Attack success rate — VAE low-KL vs VAE-GAN\n"
    f"({N_DOCS} documents per source class, bottom-right patch, 64×64px)",
    fontsize=13
)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 110)
ax.axhline(y=6.25, color="red",   linestyle="--", alpha=0.4, label="Random chance (6.25%)")
ax.axhline(y=75,   color="green", linestyle="--", alpha=0.2)
ax.legend(fontsize=10)

plt.tight_layout()
out_chart = os.path.join(ROOT, "comparison_chart.png")
plt.savefig(out_chart, dpi=150, bbox_inches="tight")
print(f"\nSaved comparison_chart.png")

# ── Scatter plot: ResNet confidence vs attack success rate ────────────────────
n_correct = [len(correct_images[str(i)]) for i in range(16)]
vaegan_enc_rates = vaegan_enc_avgs

fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.scatter(n_correct, vaegan_enc_rates, color="#1D9E75", s=100, zorder=5)

for i, cls in enumerate(CLASSES):
    ax2.annotate(cls, (n_correct[i], vaegan_enc_rates[i]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)

# Trend line
z_fit = np.polyfit(n_correct, vaegan_enc_rates, 1)
p_fit = np.poly1d(z_fit)
x_line = np.linspace(min(n_correct), max(n_correct), 100)
ax2.plot(x_line, p_fit(x_line), "r--", alpha=0.5, label="Trend line")

ax2.set_xlabel("Number of correctly classified images (ResNet-50 confidence proxy)", fontsize=11)
ax2.set_ylabel("Attack success rate (%)", fontsize=11)
ax2.set_title("Correlation: ResNet class confidence vs attack success rate\n(VAE-GAN encoded stamp variant)", fontsize=12)
ax2.axhline(y=6.25, color="gray", linestyle="--", alpha=0.4, label="Random chance")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
out_scatter = os.path.join(ROOT, "scatter_confidence_vs_success.png")
plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
print(f"Saved scatter_confidence_vs_success.png")