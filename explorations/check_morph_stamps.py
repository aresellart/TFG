"""
check_morph_stamps.py

Generates and saves sample stamps from the morphological-opening-loss
test generators (letter, advertisement, resume), so we can visually
check whether the stamps look more solid/coherent and less fragmented
compared to the original (no morph loss) stamps.

Usage:
    cd /home/asellart/tfg_stamps
    python3 explorations/check_morph_stamps.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision

from models.vanilla_vae_v2 import VanillaVAE
from phase3_endtoend.config_endtoend import ROOT, LATENT_DIM, CLASSES

LAMBDA_MORPH_TEST = 1.0
TEST_CLASSES      = [0, 7, 13]
N_SAMPLES         = 8

RESULTS_DIR_MORPH = os.path.join(ROOT, "attack_results_endtoend_morph_test")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    for target_class in TEST_CLASSES:
        cls_name = CLASSES[target_class]
        gen_path = os.path.join(
            RESULTS_DIR_MORPH,
            f"G_{target_class}_{cls_name}_morph{LAMBDA_MORPH_TEST}.pt"
        )

        if not os.path.exists(gen_path):
            print(f"  {cls_name}: generator not found — skipping")
            continue

        generator = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
        generator.load_state_dict(torch.load(gen_path, map_location=device,
                                              weights_only=False))
        generator.eval()

        stamps = []
        with torch.no_grad():
            for i in range(N_SAMPLES):
                z = torch.randn(1, LATENT_DIM, device=device)
                stamp = generator.decode(z)
                stamps.append(stamp.squeeze(0))

        grid = torchvision.utils.make_grid(stamps, nrow=4, padding=4)
        out_path = os.path.join(
            RESULTS_DIR_MORPH,
            f"morph_grid_{target_class}_{cls_name}.png"
        )
        torchvision.utils.save_image(grid, out_path)
        print(f"  {cls_name}: saved {N_SAMPLES} samples to {out_path}")

    print("\nDone! Open the morph_grid_*.png files to visually check")
    print("whether stamps look more solid/coherent vs the original fragmented look.")


if __name__ == "__main__":
    main()