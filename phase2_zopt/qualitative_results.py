import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_from_disk
from torchvision import transforms

from models.model_resnet50 import Model
from models.vanilla_vae import VanillaVAE
from patch_utils import paste_patch, preprocess_document
import config

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

# One representative per tier
# high=form(4), medium=memo(1), low=news article(9), failed=scientific report(14)
SHOWCASE = [
    {"target": 4,  "label": "form",             "rate": 97.5, "tier": "high"},
    {"target": 1,  "label": "memo",             "rate": 66.4, "tier": "medium"},
    {"target": 9,  "label": "news article",     "rate": 56.8, "tier": "low"},
    {"target": 14, "label": "scientific report","rate": 0.9,  "tier": "failed"},
]

TIER_COLORS = {
    "high":   "#22C55E",
    "medium": "#F59E0B",
    "low":    "#EF4444",
    "failed": "#1E1E1E",
}

DATASET_PATH = "/mnt/c/Users/aress/OneDrive/Escritorio/tfg_stamps/tfg_stamps/rvl_cdip_full"
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(ROOT, "attack_results")
PATCH_SIZE   = 64
POSITION     = "bottom-right"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── Load models ───────────────────────────────────────────────────────────────
resnet = Model(device=device)
resnet.load_state_dict(torch.load(
    os.path.join(ROOT, "models", "rvl-resnet50.model"), map_location=device
))
resnet.eval()
for p in resnet.parameters():
    p.requires_grad = False

vae = VanillaVAE(in_channels=1, latent_dim=config.LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vanilla_vae_stamps.pt"), map_location=device
))
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, "correct_images.json")) as f:
    correct_images = json.load(f)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def unnormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0, 1)

def get_topk_preds(logits, k=3):
    probs = F.softmax(logits, dim=1)
    topk_probs, topk_idxs = probs.topk(k, dim=1)
    return [(CLASSES[i], p.item()) for i, p in
            zip(topk_idxs[0], topk_probs[0])]

# ── Build figure: 4 columns (one per tier), 4 rows ───────────────────────────
# Row 0: stamp patch
# Row 1: original document
# Row 2: patched document
# Row 3: prediction bar (before vs after)

fig, axes = plt.subplots(4, 4, figsize=(18, 20))
fig.patch.set_facecolor("#F8F9FA")

col_titles = []

for col, showcase in enumerate(SHOWCASE):
    target_class = showcase["target"]
    target_name  = showcase["label"]
    tier         = showcase["tier"]
    rate         = showcase["rate"]
    color        = TIER_COLORS[tier]

    # Load z and decode stamp
    z_path = os.path.join(RESULTS_DIR,
                          f"z_class{target_class}_{target_name}.pt")
    z = torch.load(z_path, map_location=device)
    with torch.no_grad():
        stamp = vae.decode(z)  # (1,1,128,128)

    # Pick a random source document (not from target class)
    src_class = random.choice([c for c in range(16) if c != target_class])
    candidates = correct_images[str(src_class)]
    item = random.choice(candidates)

    pil_img = dataset[item["idx"]]["image"]
    doc_tensor = preprocess_document(pil_img).unsqueeze(0).to(device)

    # Classify original
    with torch.no_grad():
        _, logits_orig = resnet(doc_tensor)
    preds_orig = get_topk_preds(logits_orig)

    # Paste and classify patched
    patched_tensor = paste_patch(doc_tensor, stamp,
                                 patch_size=PATCH_SIZE,
                                 position=POSITION)
    with torch.no_grad():
        _, logits_patch = resnet(patched_tensor)
    preds_patch = get_topk_preds(logits_patch)

    # ── Row 0: Stamp patch ────────────────────────────────────────────────────
    ax = axes[0, col]
    ax.imshow(stamp.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f'"{target_name}" stamp\n({rate}% success)',
                 fontsize=11, fontweight="bold", color=color, pad=6)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    # ── Row 1: Original document ──────────────────────────────────────────────
    ax = axes[1, col]
    orig_display = unnormalize(doc_tensor.squeeze().cpu())
    ax.imshow(orig_display.permute(1, 2, 0), cmap="gray")
    ax.set_title(f'Original: "{CLASSES[src_class]}"',
                 fontsize=10, color="#333333", pad=4)
    ax.axis("off")

    # ── Row 2: Patched document ───────────────────────────────────────────────
    ax = axes[2, col]
    patch_display = unnormalize(patched_tensor.squeeze().cpu())
    ax.imshow(patch_display.permute(1, 2, 0), cmap="gray")
    pred_label = preds_patch[0][0]
    fooled     = pred_label == target_name
    title_col  = color if fooled else "#888888"
    ax.set_title(f'Predicted: "{pred_label}"',
                 fontsize=10, color=title_col, fontweight="bold", pad=4)
    ax.axis("off")

    # ── Row 3: Top-3 predictions before/after ────────────────────────────────
    ax = axes[3, col]
    ax.set_facecolor("#F0F0F0")

    labels_orig  = [f"{p[0][:12]}" for p in preds_orig]
    values_orig  = [p[1]*100 for p in preds_orig]
    labels_patch = [f"{p[0][:12]}" for p in preds_patch]
    values_patch = [p[1]*100 for p in preds_patch]

    x = [0, 1, 2]
    bars_orig  = ax.barh([i + 0.2 for i in x], values_orig,
                          height=0.35, color="#AAAAAA", label="Before")
    bars_patch = ax.barh([i - 0.2 for i in x], values_patch,
                          height=0.35, color=color, alpha=0.85, label="After")

    ax.set_yticks(x)
    ax.set_yticklabels(labels_patch, fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=8)
    ax.axvline(x=50, color="#CCCCCC", linestyle="--", linewidth=0.8)

    # Add value labels
    for bar in bars_patch:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%", va="center", fontsize=7, color="#333333")

    if col == 0:
        ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Top-3 predictions: before vs after",
                 fontsize=8, color="#555555", pad=3)

# Row labels
row_labels = ["Adversarial stamp", "Original document", "Patched document",
              "Classifier predictions"]
for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold",
                             rotation=90, labelpad=10, color="#333333")

# Column tier labels at top
tier_labels = ["HIGH (97.5%)", "MEDIUM (66.4%)", "LOW (56.8%)", "FAILED (0.9%)"]
for col, (label, showcase) in enumerate(zip(tier_labels, SHOWCASE)):
    color = TIER_COLORS[showcase["tier"]]
    fig.text(0.13 + col * 0.22, 0.97, label,
             ha="center", fontsize=12, fontweight="bold", color=color)

fig.suptitle(
    "Qualitative results — adversarial stamp attack on document classification\n"
    "One example per success tier  ·  stamp pasted bottom-right  ·  ResNet-50 frozen",
    fontsize=13, y=0.995, color="#222222"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(RESULTS_DIR, "qualitative_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out_path}")