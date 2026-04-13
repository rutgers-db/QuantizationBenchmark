import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.text(0.5, 0.5, "TBD", transform=ax.transAxes,
        fontsize=48, ha="center", va="center", color="#cccccc")
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
out = os.path.join(FIGURES_DIR, "placeholder")
fig.savefig(out + ".pdf", format="pdf", bbox_inches="tight")
print(f"[OK] Saved: {out}.pdf")
plt.close(fig)