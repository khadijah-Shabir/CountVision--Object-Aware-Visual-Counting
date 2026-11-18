"""Shared figure style: colour-blind-safe palette (Okabe-Ito), same colour per method everywhere."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COL = {"base": "#7f7f7f", "naive": "#E69F00", "lvlm_count": "#0072B2", "specialist": "#009E73",
       "mask_count": "#CC79A7", "accent": "#D55E00", "light": "#56B4E9"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = Path(__file__).resolve().parents[2] / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
