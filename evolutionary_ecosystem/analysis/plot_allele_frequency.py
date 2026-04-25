#!/usr/bin/env python3
"""Track allele frequency over time by assigning each creature's gene
to its nearest founding archetype, then plotting frequency over births."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(".")))
from examples.evolutionary_ecosystem.eval.harness import get_embedder, GENE_BANK

ARCHETYPE_NAMES = ["Aggressive", "Timid", "Curious", "Calm",
                   "Energetic", "Nurturing", "Cunning", "Moody"]

GENE_CATS = ["climate_survival", "predator_defense", "mating", "foraging", "personality"]

SIMS = [
    ("sim_log_mendelian_haploid.json",  "Mendelian haploid"),
    ("sim_log_diploid_codominant.json", "Diploid co-dominant"),
]

WINDOW = 50   # births per frequency window
COLORS = ["#20808D","#A84B2F","#7A39BB","#D19900","#437A22","#006494","#944454","#6E522B"]
OUT_DIR = Path("gene_drive_plots")
OUT_DIR.mkdir(exist_ok=True)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("Loading embedder...")
    embedder = get_embedder()

    for gene_cat in GENE_CATS:
        print(f"\n=== {gene_cat} ===")

        # Embed founding archetype alleles for this gene
        archetype_texts = [g[gene_cat] for g in GENE_BANK]
        archetype_embs = embedder.embed(archetype_texts, is_query=False)

        fig, axes = plt.subplots(1, len(SIMS), figsize=(14, 5), sharey=True)

        for ax, (fname, label) in zip(axes, SIMS):
            path = Path(".") / fname
            with open(path) as f:
                data = json.load(f)
            blog = [e for e in data.get("birth_log", [])
                    if e.get("child_genes", {}).get(gene_cat)]
            n = len(blog)
            print(f"  {label}: {n} births")

            # Embed all child genes
            texts = [e["child_genes"][gene_cat] for e in blog]
            doc_embs = embedder.embed(texts, is_query=False)

            # Assign each to nearest archetype
            assignments = []
            for emb in doc_embs:
                sims = [cosine_sim(emb, ae) for ae in archetype_embs]
                assignments.append(int(np.argmax(sims)))

            # Compute frequency in sliding windows
            n_windows = n // WINDOW
            window_centers = []
            freqs = {i: [] for i in range(8)}

            for w in range(n_windows):
                chunk = assignments[w*WINDOW:(w+1)*WINDOW]
                window_centers.append(w * WINDOW + WINDOW // 2)
                counts = np.bincount(chunk, minlength=8)
                for i in range(8):
                    freqs[i].append(counts[i] / WINDOW)

            window_centers = np.array(window_centers)

            # Stacked area chart
            freq_matrix = np.array([freqs[i] for i in range(8)])  # 8 x n_windows
            ax.stackplot(window_centers, freq_matrix,
                         labels=ARCHETYPE_NAMES, colors=COLORS, alpha=0.82)

            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Birth number", fontsize=11)
            if ax == axes[0]:
                ax.set_ylabel("Allele frequency", fontsize=11)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Single legend outside
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   fontsize=9, framealpha=0.85,
                   bbox_to_anchor=(0.5, -0.08))

        fig.suptitle(f"Allele Frequency Over Time: {gene_cat.replace('_',' ').title()}",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        out = OUT_DIR / f"allele_freq_{gene_cat}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
