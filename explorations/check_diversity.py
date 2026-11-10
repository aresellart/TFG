"""
check_diversity.py

Generates N samples per test-class generator (diversity-loss version)
and saves them as a single combined grid image per class, so you can
visually check whether mode collapse has been fixed — i.e. whether
different z vectors now produce VISUALLY DIFFERENT stamps.

Usage:
    cd /home/asellart/tfg_stamps
    python3 endtoend/check_diversity.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision

from models.vanilla_vae_v2 import VanillaVAE
from endtoend.config_endtoend import ROOT, LATENT_DIM, CLASSES

LAMBDA_DIV_TEST  = 10.0
TEST_CLASSES     = [0, 13]   # letter, resume
N_SAMPLES        = 8

RESULTS_DIR_DIV = os.path.join(ROOT, "attack_results_endtoend_diversity_test")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    for target_class in TEST_CLASSES:
        cls_name = CLASSES[target_class]
        gen_path = os.path.join(
            RESULTS_DIR_DIV,
            f"G_{target_class}_{cls_name}_div{LAMBDA_DIV_TEST}.pt"
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
                stamps.append(stamp.squeeze(0))   # (1, 128, 128)

        grid = torchvision.utils.make_grid(stamps, nrow=4, padding=4)
        out_path = os.path.join(
            RESULTS_DIR_DIV,
            f"diversity_grid_{target_class}_{cls_name}.png"
        )
        torchvision.utils.save_image(grid, out_path)
        print(f"  {cls_name}: saved {N_SAMPLES} samples to {out_path}")

    print("\nDone! Open the diversity_grid_*.png files to visually compare")
    print(f"{N_SAMPLES} different z samples side by side for each class.")
    print("If mode collapse is fixed, the 8 stamps in each grid should look")
    print("visibly different from each other (not all near-identical).")


if __name__ == "__main__":
    main()