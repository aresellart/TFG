import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from datasets import load_from_disk

from models.vanilla_vae_v2 import VanillaVAE
from models.model_resnet50 import Model
from patch_utils import paste_patch, preprocess_document

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "scientific publication",
    "questionnaire", "resume", "scientific report", "specification"
]

DATASET_PATH = "/home/asellart/tfg_stamps/rvl_cdip_full"
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(ROOT, "attack_results_vaegan_random_z")
LATENT_DIM   = 256
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
resnet.to(device)
for p in resnet.parameters():
    p.requires_grad = False

vae = VanillaVAE(in_channels=1, latent_dim=LATENT_DIM).to(device)
vae.load_state_dict(torch.load(
    os.path.join(ROOT, "vae_gan_best.pt"), map_location=device
))
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

dataset = load_from_disk(DATASET_PATH)
with open(os.path.join(ROOT, "correct_images.json")) as f:
    correct_images = json.load(f)

normalize   = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0, 1)

def get_topk_preds(logits, k=3):
    probs = F.softmax(logits, dim=1)
    topk_probs, topk_idxs = probs.topk(k, dim=1)
    return [(CLASSES[i], round(p.item()*100, 1))
            for i, p in zip(topk_idxs[0], topk_probs[0])]

def get_example(target_class):
    """Get stamp, a source document, classify before and after."""
    target_name = CLASSES[target_class]
    z_path = os.path.join(RESULTS_DIR,
                          f"z_class{target_class}_{target_name}.pt")
    z = torch.load(z_path, map_location=device)
    with torch.no_grad():
        stamp = vae.decode(z)

    # Pick a random source document (not target class)
    src_class  = random.choice([c for c in range(16) if c != target_class])
    candidates = correct_images[str(src_class)]
    item       = random.choice(candidates)
    pil_img    = dataset[item["idx"]]["image"]

    doc_tensor = preprocess_document(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits_orig = resnet(doc_tensor)
    preds_orig = get_topk_preds(logits_orig)

    patched = paste_patch(doc_tensor, stamp,
                          patch_size=PATCH_SIZE, position=POSITION)
    with torch.no_grad():
        _, logits_pat = resnet(patched)
    preds_pat = get_topk_preds(logits_pat)

    fooled = preds_pat[0][0] == target_name

    return {
        "target_name": target_name,
        "src_class":   CLASSES[src_class],
        "stamp":       stamp.squeeze().cpu(),
        "doc":         unnormalize(doc_tensor.squeeze().cpu()),
        "patched":     unnormalize(patched.squeeze().cpu()),
        "preds_orig":  preds_orig,
        "preds_pat":   preds_pat,
        "fooled":      fooled,
    }

# ── Examples to show ──────────────────────────────────────────────────────────
# Tier examples + notable cases
SHOWCASE = [
    # (target_class, label, success_rate, tier)
    (4,  "form",             94.3,  "high"),      # best tier
    (1,  "memo",             83.1,  "medium"),    # medium tier
    (9,  "news article",     44.1,  "low"),       # low tier
    (14, "scientific report",91.9,  "notable"),   # most dramatic recovery
    (7,  "advertisement",    82.6,  "notable"),   # recovered with random z
    (5,  "handwritten",      64.6,  "low"),       # still struggling
]

TIER_COLORS = {
    "high":    "#22C55E",
    "medium":  "#F59E0B",
    "low":     "#EF4444",
    "notable": "#8B5CF6",
}

TIER_LABELS = {
    "high":    "HIGH",
    "medium":  "MEDIUM",
    "low":     "LOW",
    "notable": "NOTABLE",
}

# ── Build figure ──────────────────────────────────────────────────────────────
n_cols = len(SHOWCASE)
fig    = plt.figure(figsize=(n_cols * 3.2, 18))
fig.patch.set_facecolor("#F8F9FA")

gs = gridspec.GridSpec(5, n_cols, figure=fig,
                       hspace=0.4, wspace=0.3,
                       height_ratios=[0.15, 1, 1, 1, 1.4])

for col, (target_class, label, rate, tier) in enumerate(SHOWCASE):
    color = TIER_COLORS[tier]
    ex    = get_example(target_class)

    # ── Row 0: tier badge ─────────────────────────────────────────────────────
    ax_badge = fig.add_subplot(gs[0, col])
    ax_badge.set_facecolor(color)
    ax_badge.text(0.5, 0.5, f"{TIER_LABELS[tier]}  {rate:.1f}%",
                  ha="center", va="center", fontsize=9,
                  fontweight="bold", color="white",
                  transform=ax_badge.transAxes)
    ax_badge.axis("off")

    # ── Row 1: stamp ──────────────────────────────────────────────────────────
    ax_stamp = fig.add_subplot(gs[1, col])
    ax_stamp.imshow(ex["stamp"], cmap="gray", vmin=0, vmax=1)
    ax_stamp.set_title(f'"{ex["target_name"]}" stamp',
                       fontsize=9, fontweight="bold", color=color, pad=4)
    ax_stamp.axis("off")
    for spine in ax_stamp.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

    # ── Row 2: original document ──────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[2, col])
    ax_orig.imshow(ex["doc"].permute(1,2,0), cmap="gray")
    ax_orig.set_title(f'Original: "{ex["src_class"]}"',
                      fontsize=8, color="#555555", pad=3)
    ax_orig.axis("off")

    # ── Row 3: patched document ───────────────────────────────────────────────
    ax_pat = fig.add_subplot(gs[3, col])
    ax_pat.imshow(ex["patched"].permute(1,2,0), cmap="gray")
    pred_label = ex["preds_pat"][0][0]
    pred_conf  = ex["preds_pat"][0][1]
    title_col  = color if ex["fooled"] else "#888888"
    ax_pat.set_title(f'→ "{pred_label}" ({pred_conf:.0f}%)',
                     fontsize=8, color=title_col,
                     fontweight="bold", pad=3)
    ax_pat.axis("off")

    # ── Row 4: confidence bars ────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[4, col])
    ax_bar.set_facecolor("#F0F0F0")

    labels_o = [p[0][:10] for p in ex["preds_orig"]]
    vals_o   = [p[1] for p in ex["preds_orig"]]
    labels_p = [p[0][:10] for p in ex["preds_pat"]]
    vals_p   = [p[1] for p in ex["preds_pat"]]

    y = [2, 1, 0]
    ax_bar.barh([i+0.2 for i in y], vals_o,   height=0.35,
                color="#AAAAAA", label="Before")
    ax_bar.barh([i-0.2 for i in y], vals_p,   height=0.35,
                color=color, alpha=0.85, label="After")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels_p, fontsize=7)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel("Confidence (%)", fontsize=7)
    ax_bar.axvline(x=50, color="#CCCCCC", linestyle="--", linewidth=0.8)

    for bar in ax_bar.patches[-3:]:
        w = bar.get_width()
        if w > 3:
            ax_bar.text(w+1, bar.get_y()+bar.get_height()/2,
                        f"{w:.0f}%", va="center", fontsize=6.5,
                        color="#333333")

    if col == 0:
        ax_bar.legend(fontsize=7, loc="lower right")
    ax_bar.set_title("Top-3: before vs after",
                     fontsize=7, color="#555555", pad=2)

# Row labels
row_labels = ["", "Adversarial stamp", "Original document",
              "Patched document", "Predictions"]
for row_idx, rl in enumerate(row_labels):
    if rl:
        fig.text(0.01, 1 - (row_idx * 0.21) - 0.08, rl,
                 fontsize=9, fontweight="bold", color="#333333",
                 rotation=90, va="center")

fig.suptitle(
    "VAE-GAN qualitative results — random z initialization\n"
    "Tier examples + notable cases  ·  stamp pasted bottom-right  ·  ResNet-50 frozen",
    fontsize=12, y=0.98, color="#222222"
)

out_path = os.path.join(ROOT, "qualitative_vaegan_random_z.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved {out_path}")

# ── Also save PDF ─────────────────────────────────────────────────────────────
pdf_path = os.path.join(ROOT, "qualitative_vaegan_random_z.pdf")
plt.savefig(pdf_path, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved {pdf_path}")