import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import torch
import os
import sys
sys.path.append("/home/asellart/tfg_stamps")

from models.vanilla_vae import VanillaVAE
from models.vanilla_vae_v2 import VanillaVAE as VanillaVAEv2

ROOT = "/home/asellart/tfg_stamps"

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

N_CORRECT = [500,500,500,273,500,500,139,500,500,500,500,500,500,500,134,492]

# ── Results from all methods ──────────────────────────────────────────────────
RESULTS = {
    "VAE low-KL\nencoded (CPU)": {
        "color": "#94A3B8",
        "avg": 69.6,
        "rates": [65.2,66.4,64.4,69.8,97.5,60.3,70.2,65.0,94.9,56.8,94.6,82.8,60.2,80.4,0.9,84.2]
    },
    "VAE low-KL\nencoded (GPU)": {
        "color": "#378ADD",
        "avg": 88.6,
        "rates": [85.8,92.8,68.6,96.0,98.6,97.8,99.0,94.3,96.0,97.2,98.8,98.5,89.5,89.6,17.4,97.2]
    },
    "VAE-GAN\nencoded (GPU)": {
        "color": "#1D9E75",
        "avg": 84.6,
        "rates": [83.8,80.0,71.1,92.4,98.9,94.7,91.4,96.2,95.3,78.3,64.2,96.2,87.1,89.2,16.5,58.9]
    },
    "VAE-GAN\nrandom z (GPU)": {
        "color": "#BA7517",
        "avg": 80.5,
        "rates": [52.7,80.5,50.0,93.9,98.9,60.5,62.4,86.9,61.0,97.9,97.4,94.1,93.0,75.8,93.8,85.2]
    },
}

# ── Adaptive attack (best per class) ─────────────────────────────────────────
adaptive = []
for i in range(16):
    best = max(v["rates"][i] for v in RESULTS.values())
    adaptive.append(best)
adaptive_avg = sum(adaptive) / 16

device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Bar chart comparing all 4 methods per class
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(20, 8))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("#FAFAFA")

x     = np.arange(16)
width = 0.2
names = list(RESULTS.keys())

for i, (name, data) in enumerate(RESULTS.items()):
    offset = (i - 1.5) * width
    bars   = ax1.bar(x + offset, data["rates"], width,
                     label=f"{name} (avg {data['avg']:.1f}%)",
                     color=data["color"], alpha=0.85, edgecolor="white", linewidth=0.5)

# Adaptive line
ax1.plot(x, adaptive, color="#D85A30", linewidth=2,
         linestyle="--", marker="o", markersize=5,
         label=f"Adaptive best-per-class (avg {adaptive_avg:.1f}%)", zorder=5)

ax1.axhline(y=6.25, color="red", linestyle=":", alpha=0.5, linewidth=1, label="Random chance (6.25%)")
ax1.axhline(y=75,   color="gray", linestyle="--", alpha=0.2, linewidth=1)

ax1.set_ylabel("Attack success rate (%)", fontsize=12)
ax1.set_title("Attack success rate per class — all methods comparison\n(134 documents per source class, equal evaluation)",
              fontsize=13, pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
ax1.set_ylim(0, 112)
ax1.legend(fontsize=9, loc="lower left", framealpha=0.9)
ax1.grid(axis="y", alpha=0.2)
ax1.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "comparison_bar_chart.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(ROOT, "comparison_bar_chart.pdf"),
            bbox_inches="tight", facecolor="white")
print("Saved comparison_bar_chart.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter plot: ResNet confidence vs attack success for all methods
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("#FAFAFA")

markers = ["o", "s", "^", "D"]
x_nc    = np.array(N_CORRECT)

for (name, data), marker in zip(RESULTS.items(), markers):
    y   = np.array(data["rates"])
    col = data["color"]

    ax2.scatter(x_nc, y, color=col, marker=marker, s=70,
                label=name.replace("\n", " "), zorder=5, alpha=0.9)

    slope, intercept, r, p, _ = stats.linregress(x_nc, y)
    x_line = np.linspace(80, 540, 100)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, color=col, linestyle="--",
             linewidth=1.5, alpha=0.6)
    ax2.text(545, slope*545+intercept, f"r={r:.2f}",
             fontsize=8, color=col, va="center")

# Annotate notable points for GPU encoded
notable = {14: "sci. report", 6: "invoice", 4: "form"}
for i, label in notable.items():
    ax2.annotate(label,
                 xy=(N_CORRECT[i], RESULTS["VAE low-KL\nencoded (GPU)"]["rates"][i]),
                 fontsize=8, color="#185FA5",
                 xytext=(6, 4), textcoords="offset points")

ax2.axhline(y=6.25, color="red", linestyle=":", alpha=0.5,
            linewidth=1, label="Random chance")
ax2.set_xlabel("Number of correctly classified images (ResNet confidence proxy)",
               fontsize=11)
ax2.set_ylabel("Attack success rate (%)", fontsize=11)
ax2.set_title("ResNet class confidence vs attack success rate\nall methods — correlation analysis",
              fontsize=12, pad=10)
ax2.set_xlim(80, 560)
ax2.set_ylim(-5, 108)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax2.grid(alpha=0.2)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "comparison_scatter.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(ROOT, "comparison_scatter.pdf"),
            bbox_inches="tight", facecolor="white")
print("Saved comparison_scatter.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Stamp grid: all methods for 4 representative classes
# ══════════════════════════════════════════════════════════════════════════════
SHOWCASE_CLASSES = [
    (4,  "form",            "high — all methods"),
    (14, "scientific report","weak — methods diverge"),
    (10, "presentation",    "strong — encoded wins"),
    (7,  "advertisement",   "mixed results"),
]

STAMP_SOURCES = [
    ("VAE low-KL\nencoded (GPU)", "attack_results_gpu_vanilla",   False),
    ("VAE-GAN\nencoded (GPU)",    "attack_results_gpu_vaegan_encoded", True),
    ("VAE-GAN\nrandom z (GPU)",   "attack_results_gpu_vaegan_random",  True),
]

fig3, axes3 = plt.subplots(len(STAMP_SOURCES), len(SHOWCASE_CLASSES),
                            figsize=(14, 9))
fig3.patch.set_facecolor("white")

# Load VAE models once
vae_v1 = VanillaVAE(in_channels=1, latent_dim=256)
vae_v1.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"), map_location=device))
vae_v1.eval()

vae_v2 = VanillaVAEv2(in_channels=1, latent_dim=256)
vae_v2.load_state_dict(torch.load(
    os.path.join(ROOT, "vae_gan_best.pt"), map_location=device))
vae_v2.eval()

method_colors = ["#378ADD", "#1D9E75", "#BA7517"]

for row, (method_name, results_dir, use_v2) in enumerate(STAMP_SOURCES):
    vae = vae_v2 if use_v2 else vae_v1
    color = method_colors[row]

    for col, (cls_idx, cls_name, description) in enumerate(SHOWCASE_CLASSES):
        ax = axes3[row, col]
        z_path = os.path.join(ROOT, results_dir,
                              f"z_class{cls_idx}_{cls_name}.pt")

        if os.path.exists(z_path):
            z = torch.load(z_path, map_location=device)
            with torch.no_grad():
                stamp = vae.decode(z).squeeze()
            ax.imshow(stamp.cpu(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, "not found", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)

        # Color border by method
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.axis("off")

        # Success rate annotation
        rate = RESULTS[method_name.replace("\n"," ") if method_name.replace("\n"," ") in RESULTS
                       else method_name]["rates"][cls_idx]

        # Get rate from correct key
        for key, data in RESULTS.items():
            if key.replace("\n","") == method_name.replace("\n",""):
                rate = data["rates"][cls_idx]
                break

        ax.set_title(f"{rate:.1f}%", fontsize=9, color=color,
                     fontweight="bold", pad=3)

        if row == 0:
            ax.text(0.5, 1.18, f"{cls_name}\n({description})",
                    ha="center", va="bottom", fontsize=8,
                    transform=ax.transAxes, color="#333333")

    # Row label
    axes3[row, 0].set_ylabel(method_name.replace("\n", " "),
                              fontsize=9, color=color,
                              fontweight="bold", rotation=90, labelpad=8)

fig3.suptitle("Adversarial stamps — all methods × representative classes\n(success rate shown per stamp)",
              fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "comparison_stamps_grid.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(ROOT, "comparison_stamps_grid.pdf"),
            bbox_inches="tight", facecolor="white")
print("Saved comparison_stamps_grid.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print(f"{'Class':<25} {'CPU enc':>9} {'GPU enc':>9} {'GAN enc':>9} {'GAN rnd':>9} {'Adaptive':>9}")
print("-"*70)

for i, cls in enumerate(CLASSES):
    rates = [data["rates"][i] for data in RESULTS.values()]
    best  = adaptive[i]
    print(f"{cls:<25} {rates[0]:>8.1f}% {rates[1]:>8.1f}% {rates[2]:>8.1f}% {rates[3]:>8.1f}% {best:>8.1f}%")

print("-"*70)
avgs = [data["avg"] for data in RESULTS.values()]
print(f"{'OVERALL AVERAGE':<25} {avgs[0]:>8.1f}% {avgs[1]:>8.1f}% {avgs[2]:>8.1f}% {avgs[3]:>8.1f}% {adaptive_avg:>8.1f}%")
print("="*70)