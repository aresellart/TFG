import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

CLASSES = [
    "letter", "memo", "email", "filefolder", "form",
    "handwritten", "invoice", "advertisement", "budget",
    "news article", "presentation", "sci. publication",
    "questionnaire", "resume", "sci. report", "specification"
]

# Number of correctly classified images per class
N_CORRECT = [500, 500, 500, 273, 500, 500, 139, 500, 500, 500,
             500, 500, 500, 500, 134, 492]

# Attack success rates
VAE_ENCODED  = [65.2, 66.4, 64.4, 69.8, 97.5, 60.3, 70.2, 65.0,
                94.9, 56.8, 94.6, 82.8, 60.2, 80.4,  0.9, 84.2]
VAEGAN_ENC   = [70.4, 69.5, 82.8, 90.3, 97.0, 78.7, 85.2, 29.0,
                94.8, 80.4, 90.0, 84.9, 80.0, 94.7, 35.8, 39.7]
VAEGAN_RND   = [85.1, 83.1, 91.1, 94.8, 94.3, 64.6, 97.0, 82.6,
                88.3, 44.1, 97.5, 96.4, 94.1, 74.1, 91.9, 89.8]

METHODS = [
    {"label": "VAE low-KL — encoded stamp", "data": VAE_ENCODED,  "color": "#378ADD", "marker": "o"},
    {"label": "VAE-GAN — encoded stamp",    "data": VAEGAN_ENC,   "color": "#1D9E75", "marker": "s"},
    {"label": "VAE-GAN — random z",         "data": VAEGAN_RND,   "color": "#BA7517", "marker": "^"},
]

# ── Labels to annotate (avoid cluttering — only notable ones) ────────────────
ANNOTATE = {
    "sci. report", "form", "budget", "advertisement",
    "invoice", "filefolder", "sci. publication"
}

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

x = np.array(N_CORRECT)

for method in METHODS:
    y    = np.array(method["data"])
    col  = method["color"]
    mark = method["marker"]

    # Scatter points
    ax.scatter(x, y, color=col, marker=mark, s=80,
               label=method["label"], zorder=5, alpha=0.9)

    # Trend line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(min(x) - 20, max(x) + 20, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=col, linestyle="--",
            linewidth=1.5, alpha=0.6, zorder=3)

    # R value annotation on trend line
    r_label = f"r = {r:.2f}"
    mid_x   = np.mean(x_line)
    mid_y   = slope * mid_x + intercept
    ax.annotate(r_label, xy=(mid_x, mid_y),
                fontsize=9, color=col, fontstyle="italic",
                xytext=(0, 10), textcoords="offset points")

# Annotate notable points (only for VAE low-KL to avoid clutter)
for i, cls in enumerate(CLASSES):
    if cls in ANNOTATE:
        ax.annotate(
            cls,
            xy=(N_CORRECT[i], VAE_ENCODED[i]),
            fontsize=8, color="#444441",
            xytext=(6, 4), textcoords="offset points"
        )

# Random chance baseline
ax.axhline(y=6.25, color="#E24B4A", linestyle=":",
           linewidth=1.2, alpha=0.7, label="Random chance (6.25%)")

# Averages
for method in METHODS:
    avg = np.mean(method["data"])
    ax.axhline(y=avg, color=method["color"], linestyle="-",
               linewidth=0.8, alpha=0.3)
    ax.text(545, avg + 1.5, f"avg {avg:.1f}%",
            fontsize=8, color=method["color"], alpha=0.8)

ax.set_xlabel("Number of correctly classified images\n(proxy for ResNet class confidence)",
              fontsize=12, color="#444441")
ax.set_ylabel("Attack success rate (%)", fontsize=12, color="#444441")
ax.set_title(
    "ResNet class confidence vs attack success rate\n"
    "Comparison across three methods",
    fontsize=13, color="#2C2C2A", pad=14
)

ax.set_xlim(80, 560)
ax.set_ylim(-5, 108)
ax.grid(alpha=0.25, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#B4B2A9")

ax.legend(fontsize=10, framealpha=0.9, loc="lower right",
          edgecolor="#D3D1C7")

plt.tight_layout()
plt.savefig("scatter_comparison.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.savefig("scatter_comparison.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved scatter_comparison.png and scatter_comparison.pdf")