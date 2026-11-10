"""
make_spods_examples.py

Generates a small grid of real stamp examples from the SPODS dataset
(after preprocessing: grayscale, 128x128), for use in the Dataset
Description section of the thesis.

Usage:
    cd /home/asellart/tfg_stamps
    python3 make_spods_examples.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from common.stamps_dataset import get_dataloader
from autoencoder_v3.config_v3 import DATA_PATH, IMAGE_SIZE

N_EXAMPLES = 6
OUTPUT_PATH = "/home/asellart/tfg_stamps/spods_real_examples.png"


def main():
    loader = get_dataloader(DATA_PATH, batch_size=N_EXAMPLES,
                             image_size=IMAGE_SIZE, num_workers=0)

    images, _ = next(iter(loader))
    # images shape: (N_EXAMPLES, 1, 128, 128)

    fig = plt.figure(figsize=(N_EXAMPLES * 2, 2.4))
    gs = gridspec.GridSpec(1, N_EXAMPLES, figure=fig, wspace=0.08)

    for i in range(N_EXAMPLES):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved {N_EXAMPLES} real stamp examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()