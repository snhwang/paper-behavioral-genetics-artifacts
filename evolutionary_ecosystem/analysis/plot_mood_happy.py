#!/usr/bin/env python3
"""Plot [!mood(happy)] tag frequency over births for Mendelian and Diploid."""

import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy import stats

SIMS = [
    ("sim_log_mendelian_haploid.json",  "Mendelian haploid",   "#20808D"),
    ("sim_log_diploid_codominant.json", "Diploid co-dominant", "#A84B2F"),
]

WINDOW = 80
TAG = "!mood(happy)"
OUT_FILE = "mood_happy_frequency.png"

def rolling(x, w):
    if len(x) < w:
        return np.array(x)
    return uniform_filter1d(np.array(x, dtype=float), size=w, mode="nearest")

def main():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for fname, label, color in SIMS:
        path = Path("/home/user/workspace/bear_repo") / fname
        with open(path) as f:
            data = json.load(f)
        blog = [e for e in data.get("birth_log", [])
                if e.get("child_genes", {}).get("mating")]
        n = len(blog)

        presence = np.array([1.0 if TAG in e["child_genes"]["mating"] else 0.0
                             for e in blog])
        birth_idx = np.arange(n)

        r, p = stats.pearsonr(birth_idx, presence)
        slope, intercept, _, _, _ = stats.linregress(birth_idx, presence)

        ax.scatter(birth_idx, presence, color=color, alpha=0.10, s=6, zorder=2)
        rm = rolling(presence, WINDOW)
        ax.plot(birth_idx, rm, color=color, linewidth=2.2, zorder=4,
                label=f"{label}  (r={r:.2f}, p={'≈0' if p < 0.001 else f'{p:.3f}'})")
        ax.plot(birth_idx, np.poly1d([slope, intercept])(birth_idx),
                color=color, linewidth=1.0, linestyle="--", alpha=0.6, zorder=3)

    ax.set_xlabel("Birth number", fontsize=13)
    ax.set_ylabel("Fraction carrying [!mood(happy)]", fontsize=13)
    ax.set_title("Diploid Protects Fitness-Critical Action Tag Against Mutational Erosion",
                 fontsize=12, pad=10)
    ax.set_ylim(-0.05, 0.85)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, framealpha=0.88)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=180)
    plt.close(fig)
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
