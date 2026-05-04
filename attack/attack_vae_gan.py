import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
from datasets import load_from_disk
from tqdm import tqdm
import importlib.util

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from patch_utils import paste_patch, preprocess_document

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

DATASET_PATH   = "/home/asellart/tfg_stamps/rvl_cdip_full"
ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAE_MODEL_PATH = os.path.join(ROOT, "vae_gan_best.pt")
RESNET_PATH    = os.path.join(ROOT, "models", "rvl-resnet50.model")
CORRECT_JSON   = os.path.join(ROOT, "correct_images.json")

LATENT_DIM     = 256
BATCH_SIZE     = 64
N_ITERATIONS   = 1000
LR             = 0.05
LAMBDA_Z       = 0.1
PATCH_SIZE     = 64
PATCH_POSITION = "bottom-right"

# encoded_stamp is fully done — only run random_z
VARIANTS = ["random_z"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── Load models ───────────────────────────────────────────────────────────────
resnet = Model(device=device)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
resnet.eval()
resnet.to(device)
for p in resnet.parameters():
    p.requires_grad = False
print("ResNet-50 loaded and frozen")

vae = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
vae.eval()
for p in vae.parameters():
    p.requires_grad = False
print(f"VAE-GAN loaded from {VAE_MODEL_PATH} and frozen")

# ── Load dataset ──────────────────────────────────────────────────────────────
dataset = load_from_disk(DATASET_PATH)
with open(CORRECT_JSON) as f:
    correct_images = json.load(f)
print("Dataset loaded")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_document_batch(target_class, batch_size):
    candidates = []
    for cls_idx in range(16):
        if cls_idx == target_class:
            continue
        candidates.extend(correct_images[str(cls_idx)])
    sampled = random.sample(candidates, batch_size)
    tensors = []
    for item in sampled:
        pil_img = dataset[item["idx"]]["image"]
        tensors.append(preprocess_document(pil_img))
    return torch.stack(tensors).to(device)


def get_encoded_z():
    """Initialize z from a real encoded stamp."""
    spec = importlib.util.spec_from_file_location(
        "stamps_dataset",
        os.path.join(ROOT, "datasets_local", "stamps_dataset.py")
    )
    stamps_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stamps_module)
    stamp_loader = stamps_module.get_dataloader(
        "/home/asellart/tfg_stamps/stamp_dataset_grayscale",
        batch_size=1, image_size=128, shuffle=True, num_workers=0
    )
    stamp_img, _ = next(iter(stamp_loader))
    stamp_img = stamp_img.to(device)
    with torch.no_grad():
        mu, logvar = vae.encode(stamp_img)
        z = vae.reparameterize(mu, logvar)
    return z


def get_random_z():
    """Sample z from prior N(0,I)."""
    return torch.randn(1, LATENT_DIM, device=device)


# ── Attack loop ───────────────────────────────────────────────────────────────
def attack_one_class(target_class, variant):
    print(f"\n{'='*55}")
    print(f"Target: {CLASSES[target_class]} | Variant: {variant}")
    print(f"{'='*55}")

    if variant == "encoded_stamp":
        z_init = get_encoded_z()
    else:
        z_init = get_random_z()

    z         = z_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=LR)

    best_loss = float("inf")
    best_z    = z.detach().clone()

    for iteration in tqdm(range(N_ITERATIONS)):
        optimizer.zero_grad()

        stamp     = vae.decode(z)
        doc_batch = get_document_batch(target_class, BATCH_SIZE)
        patched   = paste_patch(doc_batch, stamp,
                                patch_size=PATCH_SIZE,
                                position=PATCH_POSITION)

        _, logits = resnet(patched)

        target_tensor   = torch.full((BATCH_SIZE,), target_class,
                                     dtype=torch.long, device=device)
        attack_loss     = F.cross_entropy(logits, target_tensor)
        generation_loss = torch.norm(z)
        loss            = attack_loss + LAMBDA_Z * generation_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z    = z.detach().clone()

        if (iteration + 1) % 100 == 0:
            with torch.no_grad():
                preds        = logits.argmax(dim=1)
                success_rate = (preds == target_class).float().mean().item()
            print(f"  iter {iteration+1:4d}  loss={loss.item():.4f}  "
                  f"attack={attack_loss.item():.4f}  "
                  f"success={success_rate*100:.1f}%")

    # Save results
    cls_name = CLASSES[target_class]
    out_dir  = os.path.join(ROOT, f"attack_results_vaegan_{variant}")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(best_z, os.path.join(out_dir,
               f"z_class{target_class}_{cls_name}.pt"))

    import torchvision
    with torch.no_grad():
        final_stamp = vae.decode(best_z)
    torchvision.utils.save_image(
        final_stamp,
        os.path.join(out_dir, f"stamp_class{target_class}_{cls_name}.png")
    )
    print(f"Saved z and stamp for '{cls_name}' — variant: {variant}")
    return best_z


# ── Run attack — skip already completed classes ───────────────────────────────
if __name__ == "__main__":
    for variant in VARIANTS:
        print(f"\n{'#'*55}")
        print(f"RUNNING VARIANT: {variant}")
        print(f"{'#'*55}")
        out_dir = os.path.join(ROOT, f"attack_results_vaegan_{variant}")
        os.makedirs(out_dir, exist_ok=True)

        for target_class in range(16):
            cls_name = CLASSES[target_class]
            z_path   = os.path.join(out_dir,
                                    f"z_class{target_class}_{cls_name}.pt")
            if os.path.exists(z_path):
                print(f"Skipping {cls_name} — already done")
                continue
            attack_one_class(target_class, variant)

    print("\nAll done! Results saved in:")
    print(f"  attack_results_vaegan_random_z/")