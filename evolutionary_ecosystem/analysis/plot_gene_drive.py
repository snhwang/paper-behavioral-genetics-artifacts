#!/usr/bin/env python3
"""Plot BEAR retrieval strength (bear_strength) for all behavioral dimensions
over births/generations for Mendelian haploid, Diploid codominant, and LLM blend sims."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(".")))
from examples.evolutionary_ecosystem.eval.harness import get_embedder

# ---------------------------------------------------------------------------
# Situation queries — one representative per gene category
# ---------------------------------------------------------------------------
SITUATIONS = {
    "food_seeking":      "I'm hungry and need to find food to survive",
    "combat":            "Another creature is nearby and could be a threat",
    "breeding":          "A potential mate is nearby, should I reproduce",
    "survival":          "Weather is harsh and conditions are dangerous",
    "territory":         "Something has entered my area",
    "exploration":       "I'm wandering and discovering new things in unfamiliar territory",
    "stealth":           "I need to hide and avoid being detected by a nearby predator",
    "detection":         "I need to scan my surroundings for approaching threats or food sources",
    "cooperation":       "Another creature is struggling with a task nearby and might need help",
}

# Gene category each situation probes
SIT_GENE = {
    "food_seeking": "foraging",
    "combat": "predator_defense",
    "breeding": "mating",
    "survival": "climate_survival",
    "territory": "territorial",
    "exploration": "personality",
    "stealth": "stealth",
    "detection": "sensory",
    "cooperation": "social_style",
}

SIMS = [
    ("sim_log_mendelian_haploid.json",    "Mendelian haploid",    "#20808D"),
    ("sim_log_diploid_codominant.json",   "Diploid co-dominant",  "#A84B2F"),
    ("sim_log_llm_synthesis_free_epoch.json", "LLM blend (free epoch)", "#7A39BB"),
]

WINDOW = 30   # rolling mean window (births)
OUT_DIR = Path("gene_drive_plots")
OUT_DIR.mkdir(exist_ok=True)

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_drive(blog, embedder, sit_query, gene_cat):
    """For each birth compute bear_strength of child's gene_cat text against sit_query."""
    query_emb = embedder.embed([sit_query], is_query=True)[0]
    texts = [e.get("child_genes", {}).get(gene_cat, "") for e in blog]
    valid = [(i, t) for i, t in enumerate(texts) if t]
    if not valid:
        return np.array([]), np.array([])
    idxs, txts = zip(*valid)
    doc_embs = embedder.embed(list(txts), is_query=False)
    scores = np.array([cosine(query_emb, e) for e in doc_embs])
    births = np.array([blog[i].get("tick", j) for j, i in enumerate(idxs)])
    return births, scores

def rolling(x, w):
    if len(x) < w:
        return x
    return uniform_filter1d(x.astype(float), size=w, mode="nearest")

def main():
    print("Loading embedder...")
    embedder = get_embedder()

    # Load all sim data
    sim_data = []
    for fname, label, color in SIMS:
        path = Path(".") / fname
        if not path.exists():
            print(f"  Missing: {fname}")
            continue
        with open(path) as f:
            data = json.load(f)
        blog = [e for e in data.get("birth_log", []) if e.get("child_genes")]
        print(f"  {label}: {len(blog)} births")
        sim_data.append((blog, label, color))

    sit_names = list(SITUATIONS.keys())
    n_sit = len(sit_names)

    # --- Per-situation plots (one figure per situation, all 3 sims) ---
    print("\nGenerating per-situation plots...")
    for sit in sit_names:
        query = SITUATIONS[sit]
        gene = SIT_GENE[sit]
        fig, ax = plt.subplots(figsize=(8, 4))

        for blog, label, color in sim_data:
            births, scores = compute_drive(blog, embedder, query, gene)
            if len(scores) == 0:
                continue
            birth_idx = np.arange(len(scores))
            ax.scatter(birth_idx, scores, color=color, alpha=0.25, s=12, zorder=2)
            if len(scores) >= WINDOW:
                rm = rolling(scores, WINDOW)
                ax.plot(birth_idx, rm, color=color, linewidth=2.0,
                        label=f"{label} (n={len(scores)})", zorder=3)
            else:
                ax.plot(birth_idx, scores, color=color, linewidth=1.5,
                        label=f"{label} (n={len(scores)})", zorder=3)

        ax.set_xlabel("Birth number", fontsize=13)
        ax.set_ylabel("BEAR retrieval strength", fontsize=13)
        ax.set_title(f"{sit.replace('_',' ').title()} Drive Over Births\n({gene} gene)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10, framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        out = OUT_DIR / f"drive_{sit}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out.name}")

    # --- Combined overview: all situations for each sim ---
    print("\nGenerating per-sim overview plots...")
    for blog, label, color in sim_data:
        n_cols = 3
        n_rows = (n_sit + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        axes = axes.flatten()

        for i, sit in enumerate(sit_names):
            query = SITUATIONS[sit]
            gene = SIT_GENE[sit]
            births, scores = compute_drive(blog, embedder, query, gene)
            ax = axes[i]
            if len(scores) == 0:
                ax.set_visible(False)
                continue
            birth_idx = np.arange(len(scores))
            ax.scatter(birth_idx, scores, color=color, alpha=0.25, s=10, zorder=2)
            if len(scores) >= WINDOW:
                ax.plot(birth_idx, rolling(scores, WINDOW), color=color,
                        linewidth=1.8, zorder=3)
            # trend
            if len(birth_idx) > 2:
                z = np.polyfit(birth_idx, scores, 1)
                ax.plot(birth_idx, np.poly1d(z)(birth_idx),
                        color="#1B474D", linewidth=1.0, linestyle="--", zorder=4)
            ax.set_title(f"{sit.replace('_',' ').title()}\n({gene})", fontsize=10)
            ax.set_xlabel("Birth #", fontsize=9)
            ax.set_ylabel("Strength", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", alpha=0.3)

        # Hide unused axes
        for j in range(n_sit, len(axes)):
            axes[j].set_visible(False)

        slug = label.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        fig.suptitle(f"{label}: Behavioral Drive Over Births", fontsize=14, y=1.01)
        fig.tight_layout()
        out = OUT_DIR / f"overview_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
