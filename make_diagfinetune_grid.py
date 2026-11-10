"""
make_diagfinetune_grid.py

Combines 4 specific stamp images from the diagnostic-discriminator
fine-tuned model (attack_results_endtoend_diagfinetune_final/) into
a single horizontal grid figure for the thesis.

Usage:
    cd /home/asellart/tfg_stamps
    python3 make_diagfinetune_grid.py
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

RESULTS_DIR = "/home/asellart/tfg_stamps/attack_results_endtoend_diagfinetune_final"

# (filename, display label)
IMAGES = [
    ("stamp_eval_0_letter_z0.png",         "Letter"),
    ("stamp_eval_4_form_z0.png",           "Form"),
    ("stamp_eval_7_advertisement_z0.png",  "Advertisement"),
    ("stamp_eval_12_questionnaire_z0.png", "Questionnaire"),
]

OUTPUT_PATH = os.path.join(RESULTS_DIR, "diagfinetune_grid_selected.png")


def main():
    n = len(IMAGES)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.6))

    for ax, (filename, label) in zip(axes, IMAGES):
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"NOT FOUND: {path}")
            ax.text(0.5, 0.5, "not found", ha='center', va='center',
                    fontsize=9, color='red', transform=ax.transAxes)
            ax.axis('off')
            ax.set_title(label, fontsize=10, fontweight='bold')
            continue

        img = mpimg.imread(path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(label, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()