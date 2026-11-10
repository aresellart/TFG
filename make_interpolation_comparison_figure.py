"""
make_interpolation_comparison_figure.py

Combines two existing latent interpolation strip images (High-KL VAE
and VAE-GAN) into a single stacked comparison figure for the thesis.

Expects two image files already saved on disk:
    - high_kl_interpolation.png   (the "High KL VAE — latent interpolation..." strip)
    - vae_gan_interpolation.png   (the "VAE-GAN — latent interpolation" strip)

If your files have different names, edit HIGH_KL_PATH and VAE_GAN_PATH below.

Usage:
    cd /home/asellart/tfg_stamps
    python3 make_interpolation_comparison_figure.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = "/home/asellart/tfg_stamps"

# ── EDIT THESE if your filenames differ ──────────────────────────────────────
HIGH_KL_PATH = os.path.join(ROOT, "interpolation_high_kl.png")
VAE_GAN_PATH = os.path.join(ROOT, "interpolation_vae_gan.png")

OUTPUT_PATH = os.path.join(ROOT, "generative_model_comparison.pdf")


def main():
    if not os.path.exists(HIGH_KL_PATH):
        print(f"NOT FOUND: {HIGH_KL_PATH}")
        print("Edit HIGH_KL_PATH in this script to point to the correct file.")
        return
    if not os.path.exists(VAE_GAN_PATH):
        print(f"NOT FOUND: {VAE_GAN_PATH}")
        print("Edit VAE_GAN_PATH in this script to point to the correct file.")
        return

    img_high_kl = mpimg.imread(HIGH_KL_PATH)
    img_vae_gan = mpimg.imread(VAE_GAN_PATH)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    axes[0].imshow(img_high_kl)
    axes[0].axis('off')
    axes[0].set_title("High-KL VAE + ResBlocks — latent interpolation",
                       fontsize=11, fontweight='bold', loc='left')

    axes[1].imshow(img_vae_gan)
    axes[1].axis('off')
    axes[1].set_title("VAE-GAN — latent interpolation",
                       fontsize=11, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()