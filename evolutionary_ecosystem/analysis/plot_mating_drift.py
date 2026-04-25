#!/usr/bin/env python3
"""Plot mating gene drift over births in the LLM blend abundance sim.

Measures cosine similarity between each offspring's mating gene and
the mean of the founding population's mating genes (generation 0),
showing directional drift away from the baseline.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, str(Path("/home/user/workspace/bear_repo")))
from examples.evolutionary_ecosystem.eval.harness import get_embedder
import examples.evolutionary_ecosystem.eval.harness as harness

# Archetype list is a module-level list of dicts
ARCHETYPES = [x for x in dir(harness) if not x.startswith('_')]
# The archetypes are the list assigned before _NAMES
# Extract directly from module source
import inspect
src = inspect.getsource(harness)
# Just grab mating texts from birth log generation 0 entries as the baseline

LOG_FILE = "/home/user/workspace/bear_repo/sim_log_llm_synthesis_abundance.json"
OUT_FILE = "mating_drift.png"

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    with open(LOG_FILE) as f:
        data = json.load(f)

    blog = data.get("birth_log", [])
    print(f"Loaded {len(blog)} births")

    embedder = get_embedder()

    # Build baseline from generation 0 parent genes (founding population)
    gen0_mating = []
    for entry in blog:
        if entry.get("generation", 1) == 1:
            # gen 1 children have gen 0 parents — grab parent genes as baseline
            pa = entry.get("pa_genes", {}).get("mating", "")
            pb = entry.get("pb_genes", {}).get("mating", "")
            if pa: gen0_mating.append(pa)
            if pb: gen0_mating.append(pb)
    # Deduplicate
    gen0_mating = list(dict.fromkeys(gen0_mating))
    print(f"Gen-0 mating samples: {len(gen0_mating)}")
    if not gen0_mating:
        print("No gen-0 data, using child mating texts from first 5 births as baseline")
        gen0_mating = [blog[i].get("child_genes", {}).get("mating", "") for i in range(min(5, len(blog)))]
        gen0_mating = [t for t in gen0_mating if t]
    archetype_embs = embedder.embed(gen0_mating, is_query=False)
    baseline_emb = np.mean(archetype_embs, axis=0)

    # Embed each birth's mating gene
    mating_texts = [e.get("child_genes", {}).get("mating", "") for e in blog]
    valid = [(e, t) for e, t in zip(blog, mating_texts) if t]
    entries, texts = zip(*valid)

    print(f"Embedding {len(texts)} mating genes...")
    doc_embs = embedder.embed(list(texts), is_query=False)

    ticks = np.array([e.get("tick", i) for i, e in enumerate(entries)])
    scores = np.array([cosine(baseline_emb, emb) for emb in doc_embs])
    generations = np.array([e.get("generation", 0) for e in entries])

    print(f"Score range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Mean: {scores.mean():.3f}  Std: {scores.std():.3f}")

    # Rolling mean
    window = 5
    if len(scores) >= window:
        rolling = np.convolve(scores, np.ones(window)/window, mode='valid')
        rolling_x = np.arange(window-1, len(scores))
    else:
        rolling, rolling_x = scores, np.arange(len(scores))

    birth_idx = np.arange(len(scores))

    # Trend line
    z = np.polyfit(birth_idx, scores, 1)
    p = np.poly1d(z)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(birth_idx, scores, color="#20808D", alpha=0.55, s=28, zorder=3,
               label="Per-birth similarity to baseline")
    if len(scores) >= window:
        ax.plot(rolling_x, rolling, color="#A84B2F", linewidth=2.0, zorder=4,
                label=f"Rolling mean (n={window})")
    ax.plot(birth_idx, p(birth_idx), color="#1B474D", linewidth=1.3,
            linestyle="--", zorder=2, label="Linear trend")

    ax.set_xlabel("Birth number", fontsize=13)
    ax.set_ylabel("Cosine similarity to\nfounding mating archetype", fontsize=13)
    ax.set_title("LLM Blend: Mating Gene Drift Over Generations", fontsize=14, pad=10)
    ax.tick_params(labelsize=11)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    y_pad = max(0.05, scores.std())
    ax.set_ylim(scores.min() - y_pad, scores.max() + y_pad)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=180)
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
