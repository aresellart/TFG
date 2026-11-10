"""
make_stamp_grid.py

Generates a single horizontal row of example stamps, one per selected
class, sampled from the trained end-to-end generators in
attack_results_endtoend/ (fixed bottom-right model).

Usage:
    cd /home/asellart/tfg_stamps
    python3 make_stamp_grid.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.vanilla_vae_v2 import VanillaVAE
from phase3_endtoend.config_endtoend import ROOT, LATENT_DIM, CLASSES, RESULTS_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Pick a representative subset of classes to display — edit as desired.
# Using class indices from CLASSES list (0=letter ... 15=specification)
SELECTED_CLASSES = [0, 2, 7, 9, 13, 14]   # letter, email, advertisement, news article, resume, sci. report

OUTPUT_PATH = os.path.join(ROOT, "stamp_grid_endtoend.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    n_cols = len(SELECTED_CLASSES)

    fig = plt.figure(figsize=(n_cols * 2, 2.6))
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.08)

    for col, cls_idx in enumerate(SELECTED_CLASSES):
        cls_name = CLASSES[cls_idx]
        gen_path = os.path.join(RESULTS_DIR, f"G_{cls_idx}_{cls_name}.pt")

        ax = fig.add_subplot(gs[0, col])
        ax.axis('off')
        ax.set_title(cls_name, fontsize=10, fontweight='bold', pad=6)

        if not os.path.exists(gen_path):
            print(f"  [{cls_name}] generator not found at {gen_path} — skipping")
            ax.text(0.5, 0.5, "not found", fontsize=9, color='red',
                     ha='center', va='center', transform=ax.transAxes)
            continue

        generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device,
                                              weights_only=False))
        generator.eval()

        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM, device=device)
            stamp = generator.decode(z).squeeze().cpu().numpy()

        ax.imshow(stamp, cmap='gray')
        print(f"  [{cls_name}] done")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\nSaved grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()