import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models.vanilla_vae import VanillaVAE
import config

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

# Final success rates from your output
SUCCESS_RATES = [
    75.0,   # letter
    87.5,   # memo
    100.0,  # email
    100.0,  # filefolder
    100.0,  # form
    87.5,   # handwritten
    100.0,  # invoice
    62.5,   # advertisement
    100.0,  # budget
    62.5,   # news article
    100.0,  # presentation
    75.0,   # scientific publication
    62.5,   # questionnaire
    87.5,   # resume
    0.0,    # scientific report
    75.0,   # specification
]

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "attack_results")

device = torch.device("cpu")

# Load VAE
vae = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"), map_location=device
))
vae.eval()

# ── Plot 1: all 16 stamps in a grid ──────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle("Adversarial stamps — one per target class", fontsize=16, y=1.01)

for idx, (ax, cls_name, success) in enumerate(zip(axes.flat, CLASSES, SUCCESS_RATES)):
    z_path = os.path.join(OUT_DIR, f"z_class{idx}_{cls_name}.pt")

    if os.path.exists(z_path):
        z = torch.load(z_path, map_location=device)
        with torch.no_grad():
            stamp = vae.decode(z).squeeze()
        ax.imshow(stamp.cpu(), cmap="gray", vmin=0, vmax=1)
    else:
        ax.text(0.5, 0.5, "not found", ha="center", va="center")

    # Color border by success rate
    if success == 100.0:
        color = "green"
    elif success >= 75.0:
        color = "orange"
    elif success > 0.0:
        color = "red"
    else:
        color = "black"

    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    ax.set_title(f"{cls_name}\n{success:.0f}% success", fontsize=9)
    ax.axis("off")

# Legend
legend_elements = [
    mpatches.Patch(color="green",  label="100% success"),
    mpatches.Patch(color="orange", label="75-87.5% success"),
    mpatches.Patch(color="red",    label="< 75% success"),
    mpatches.Patch(color="black",  label="0% — failed"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "all_stamps_grid.png"), dpi=150, bbox_inches="tight")
print("Saved all_stamps_grid.png")

# ── Plot 2: success rate bar chart ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))

colors = ["green" if s == 100 else "orange" if s >= 75 else "red" if s > 0 else "black"
          for s in SUCCESS_RATES]

bars = ax2.bar(CLASSES, SUCCESS_RATES, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_ylim(0, 110)
ax2.axhline(y=100, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax2.set_ylabel("Attack success rate (%)", fontsize=12)
ax2.set_title("Attack success rate per target class", fontsize=14)
ax2.set_xticks(range(len(CLASSES)))
ax2.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)

# Add value labels on bars
for bar, val in zip(bars, SUCCESS_RATES):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "success_rates_chart.png"), dpi=150, bbox_inches="tight")
print("Saved success_rates_chart.png")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== ATTACK SUMMARY ===")
avg = sum(SUCCESS_RATES) / len(SUCCESS_RATES)
perfect = sum(1 for s in SUCCESS_RATES if s == 100.0)
failed  = sum(1 for s in SUCCESS_RATES if s == 0.0)
print(f"Average success rate: {avg:.1f}%")
print(f"Classes with 100% success: {perfect}/16")
print(f"Classes that failed (0%): {failed}/16")
print(f"\nBest classes:  {[CLASSES[i] for i, s in enumerate(SUCCESS_RATES) if s == 100.0]}")
print(f"Worst classes: {[CLASSES[i] for i, s in enumerate(SUCCESS_RATES) if s < 75.0]}")