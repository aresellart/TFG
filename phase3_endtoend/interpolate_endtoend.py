"""
interpolate_endtoend.py

Latent space interpolation experiment for the end-to-end generators.

For each of the 16 trained generators:
  - Sample two random z vectors z1 and z2 from N(0,I)
  - Linearly interpolate between them: z = (1-α)*z1 + α*z2
  - Decode each interpolated z → stamp
  - Save as a grid showing the smooth (or not) transition

If the latent space is organized → smooth visual transition between stamps
If completely random → abrupt changes or nonsensical intermediates

Usage:
    python3 endtoend/interpolate_endtoend.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models.vanilla_vae_v2 import VanillaVAE
from endtoend.config_endtoend import (
    LATENT_DIM, RESULTS_DIR, CLASSES
)

N_STEPS    = 9     # number of interpolation steps (including endpoints)
N_PAIRS    = 3     # number of z1,z2 pairs to show per class
OUT_DIR    = os.path.join(RESULTS_DIR, "interpolation")
os.makedirs(OUT_DIR, exist_ok=True)


def interpolate_generator(target_class, device):
    """
    For one generator G_c:
    - Sample N_PAIRS pairs of random z vectors
    - Interpolate between each pair in N_STEPS
    - Save grid of decoded stamps
    """
    cls_name  = CLASSES[target_class]
    save_path = os.path.join(RESULTS_DIR, f"G_{target_class}_{cls_name}.pt")

    if not os.path.exists(save_path):
        print(f"  {cls_name}: no trained generator — skipping")
        return

    # load generator
    generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(save_path, map_location=device,
                                          weights_only=False))
    generator.eval()

    # alphas for interpolation: 0.0, 0.125, 0.25, ..., 1.0
    alphas = np.linspace(0, 1, N_STEPS)

    fig, axes = plt.subplots(N_PAIRS, N_STEPS,
                              figsize=(N_STEPS * 1.4, N_PAIRS * 1.4 + 0.6))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Latent space interpolation — G{target_class} ({cls_name})',
                 fontsize=11, fontweight='bold', color='#222222', y=1.01)

    torch.manual_seed(42 + target_class)

    for row in range(N_PAIRS):
        # sample two random z vectors
        z1 = torch.randn(1, LATENT_DIM, device=device)
        z2 = torch.randn(1, LATENT_DIM, device=device)

        for col, alpha in enumerate(alphas):
            # linear interpolation in latent space
            z_interp = (1 - alpha) * z1 + alpha * z2
            # z_interp smoothly moves from z1 (alpha=0) to z2 (alpha=1)
            # if latent space is organized → smooth visual transition
            # if random → abrupt changes

            with torch.no_grad():
                stamp = generator.decode(z_interp)
                # shape: (1, 1, 128, 128)
                stamp_np = stamp.squeeze().cpu().numpy()
                # shape: (128, 128) — grayscale values in [0,1]

            ax = axes[row, col] if N_PAIRS > 1 else axes[col]
            ax.imshow(stamp_np, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # label endpoints and midpoint
            if row == 0:
                if col == 0:
                    ax.set_title('z₁', fontsize=8, color='#1D9E75',
                                 fontweight='bold')
                elif col == N_STEPS - 1:
                    ax.set_title('z₂', fontsize=8, color='#BA7517',
                                 fontweight='bold')
                elif col == N_STEPS // 2:
                    ax.set_title('mid', fontsize=8, color='#888780')

            # color border by position
            for spine in ax.spines.values():
                t = alpha
                r = int(186 + (29 - 186) * t)
                g = int(117 + (158 - 117) * t)
                b = int(23 + (117 - 23) * t)
                spine.set_edgecolor(f'#{r:02x}{g:02x}{b:02x}')
                spine.set_linewidth(1.2)
                spine.set_visible(True)

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(OUT_DIR, f"interp_{target_class}_{cls_name}.png")
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved: interp_{target_class}_{cls_name}.png")


def make_summary_grid(device):
    """
    Makes one big summary figure showing interpolation for
    4 representative classes side by side.
    """
    showcase = [
        (4,  "form"),           # high ASR — all methods
        (14, "scientific report"), # hardest class
        (1,  "memo"),           # good ASR
        (12, "questionnaire"),  # interesting class
    ]

    fig, big_axes = plt.subplots(len(showcase), N_STEPS,
                                  figsize=(N_STEPS * 1.5, len(showcase) * 1.8))
    fig.patch.set_facecolor('white')
    fig.suptitle('Latent Space Interpolation — End-to-End Generators\n'
                 'Each row: smooth transition from z₁ to z₂ via linear interpolation',
                 fontsize=11, fontweight='bold', color='#222222')

    alphas = np.linspace(0, 1, N_STEPS)

    for row_idx, (cls_idx, cls_name) in enumerate(showcase):
        save_path = os.path.join(RESULTS_DIR,
                                  f"G_{cls_idx}_{cls_name}.pt")
        if not os.path.exists(save_path):
            continue

        generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
        generator.load_state_dict(torch.load(save_path, map_location=device,
                                              weights_only=False))
        generator.eval()

        torch.manual_seed(99 + cls_idx)
        z1 = torch.randn(1, LATENT_DIM, device=device)
        z2 = torch.randn(1, LATENT_DIM, device=device)

        for col, alpha in enumerate(alphas):
            z_interp = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                stamp = generator.decode(z_interp).squeeze().cpu().numpy()

            ax = big_axes[row_idx, col]
            ax.imshow(stamp, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_edgecolor('#cccccc')

            if col == 0:
                ax.set_ylabel(cls_name, fontsize=8, fontweight='bold',
                              color='#1D9E75', rotation=0,
                              labelpad=55, va='center')

        # add z1 and z2 labels on top row
        if row_idx == 0:
            big_axes[0, 0].set_title('z₁', fontsize=9, color='#1D9E75',
                                      fontweight='bold')
            big_axes[0, N_STEPS//2].set_title('←  interpolation  →',
                                               fontsize=8, color='#888780')
            big_axes[0, N_STEPS-1].set_title('z₂', fontsize=9,
                                              color='#BA7517', fontweight='bold')

    plt.tight_layout(pad=0.4)
    out_path = os.path.join(OUT_DIR, "interpolation_summary.png")
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved summary grid: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"Output directory: {OUT_DIR}\n")

    # individual interpolation for each class
    print("Generating interpolation grids for all 16 classes...")
    for target_class in range(16):
        interpolate_generator(target_class, device)

    # summary grid with 4 representative classes
    print("\nGenerating summary grid...")
    make_summary_grid(device)

    print("\nDone! Check the interpolation/ folder.")
    print("If transitions are smooth → latent space is organized")
    print("If transitions are abrupt → latent space is more random")


if __name__ == "__main__":
    main()