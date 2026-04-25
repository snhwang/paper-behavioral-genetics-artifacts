#!/usr/bin/env python3
"""Plot pairwise cosine distance distributions for climate_survival gene
across 4 time windows for Mendelian haploid and Diploid codominant sims."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path("/home/user/workspace/bear_repo")))
from examples.evolutionary_ecosystem.eval.harness import get_embedder

GENE = "climate_survival"
N_WINDOWS = 4
OUT_FILE = "gene_diversity_climate.png"

def cosine_dist(a, b):
    a, b = np.array(a), np.array(b)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return 1.0 - sim

def pairwise_distances(embs):
    """All pairwise cosine distances for a set of embeddings."""
    dists = []
    for i, j in combinations(range(len(embs)), 2):
        dists.append(cosine_dist(embs[i], embs[j]))
    return np.array(dists)

def main():
    print("Loading embedder...")
    embedder = get_embedder()

    sims = [
        ("sim_log_mendelian_haploid.json",   "Mendelian haploid",   "#20808D"),
        ("sim_log_diploid_codominant.json",  "Diploid co-dominant", "#A84B2F"),
    ]

    fig, axes = plt.subplots(2, N_WINDOWS, figsize=(14, 7), sharey="row")

    for row, (fname, label, color) in enumerate(sims):
        path = Path("/home/user/workspace/bear_repo") / fname
        with open(path) as f:
            data = json.load(f)
        blog = [e for e in data.get("birth_log", [])
                if e.get("child_genes", {}).get(GENE)]
        n = len(blog)
        print(f"{label}: {n} births")

        # Split into N_WINDOWS equal windows
        window_size = n // N_WINDOWS
        windows = [blog[i*window_size:(i+1)*window_size] for i in range(N_WINDOWS)]

        # Embed all gene texts in one batch
        all_texts = [e["child_genes"][GENE] for e in blog]
        print(f"  Embedding {len(all_texts)} texts...")
        all_embs = embedder.embed(all_texts, is_query=False)

        for col, (win, win_entries) in enumerate(zip(
                [all_embs[i*window_size:(i+1)*window_size] for i in range(N_WINDOWS)],
                windows)):
            ax = axes[row, col]

            dists = pairwise_distances(win)
            birth_start = col * window_size + 1
            birth_end = (col + 1) * window_size

            ax.hist(dists, bins=30, color=color, alpha=0.75, edgecolor="white",
                    linewidth=0.4)
            ax.axvline(dists.mean(), color="#1B474D", linewidth=1.5,
                       linestyle="--", label=f"mean={dists.mean():.3f}")
            ax.set_title(f"Births {birth_start}–{birth_end}\nmean={dists.mean():.3f}  std={dists.std():.3f}",
                         fontsize=10)
            ax.set_xlabel("Cosine distance", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{label}\nCount", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(f"Pairwise Gene Diversity: {GENE.replace('_',' ').title()} Over Time",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=160, bbox_inches="tight")
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
