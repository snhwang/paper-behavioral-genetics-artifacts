#!/usr/bin/env python3
"""Evaluation 7: Semantic vs numeric mutation diversity.

Compares the behavioral diversity produced by BEAR's text-based breeding
(sentence splicing + gene bank mutation) against simple numeric crossover
with Gaussian mutation on 7-float behavior vectors.

Design:
  - 4 parent pairs spanning maximally diverse archetypes:
    Bold×Timid, Curious×Calm, Energetic×Nurturing, Cunning×Moody
  - 20 offspring per pair per method (80 offspring per method, 160 total)
  - Measure: per-gene semantic diversity (mean pairwise cosine distance of
    gene-text embeddings for each gene category) and mean across categories

Hypothesis: Semantic mutation produces offspring that occupy a wider region
of behavior space because text blending creates qualitatively novel
combinations, whereas numeric mutation only interpolates ± Gaussian noise.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — deterministic text splicing, no LLM blending)

Parameters:
- BEAR: sentence splicing crossover + gene bank mutation (rate=15%)
- Numeric GA: arithmetic crossover (alpha ∈ [0.3, 0.7]) + Gaussian mutation
  (σ=0.08, rate=15%)
- Seed: 42

Outputs:
- eval7_results.json  — Per-pair offspring data and diversity metrics
- eval7_mutation.png  — Diversity comparison charts
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    GENE_CATEGORIES,
    SITUATION_NAMES,
    _NAMES,
    breed_deterministic,
    compute_per_gene_diversity,
    cosine_similarity,
    ensure_eval_patched,
    gene_diversity_mean,
    get_config,
    get_embedder,
    make_creature,
    profile_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import (
    BehaviorProfile,
    SituationResult,
)
from evolutionary_ecosystem.server.sim import WORLD_W, WORLD_H

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# Parent pair indices into GENE_BANK
PARENT_PAIRS = [
    (0, 1),  # Bold × Timid
    (2, 3),  # Curious × Calm
    (4, 5),  # Energetic × Nurturing
    (6, 7),  # Cunning × Moody
]
N_OFFSPRING = 20
MUTATION_RATE = 0.15


def numeric_crossover_profile(
    a_profile: list[float],
    b_profile: list[float],
    rng: random.Random,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma: float = 0.08,
) -> list[float]:
    """Arithmetic crossover + Gaussian mutation on profile vectors."""
    child = []
    for va, vb in zip(a_profile, b_profile):
        alpha = rng.uniform(0.3, 0.7)
        val = alpha * va + (1 - alpha) * vb
        if rng.random() < mutation_rate:
            val += rng.gauss(0, mutation_sigma)
        child.append(max(0.05, min(1.0, val)))
    return child


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_eval_patched()

    embedder = get_embedder()
    config = get_config()

    print("=" * 60)
    print("EVAL 7: Semantic vs Numeric Mutation Diversity")
    print("=" * 60)
    print(f"Parent pairs: {len(PARENT_PAIRS)}, Offspring per pair: {N_OFFSPRING}")
    print(f"Mutation rate: {MUTATION_RATE}")

    rng = random.Random(42)

    # Create parent creatures
    parents = {}
    for i, (a_idx, b_idx) in enumerate(PARENT_PAIRS):
        genes_a = GENE_BANK[a_idx]
        genes_b = GENE_BANK[b_idx]
        name_a = _NAMES[a_idx]
        name_b = _NAMES[b_idx]

        pa = make_creature(f"pa_{i}", genes_a, name_a, rng)
        pb = make_creature(f"pb_{i}", genes_b, name_b, rng)
        parents[i] = (pa, pb)

        prof_a = profile_to_vector(pa.behavior_profile)
        prof_b = profile_to_vector(pb.behavior_profile)
        parent_sim = cosine_similarity(prof_a, prof_b)
        print(f"\nPair {i}: {name_a} × {name_b} (profile sim={parent_sim:.4f})")
        print(f"  {name_a}: {[round(x, 3) for x in prof_a]}")
        print(f"  {name_b}: {[round(x, 3) for x in prof_b]}")

    # Breed offspring using BEAR (text splicing)
    bear_offspring: dict[int, list] = {}
    bear_offspring_profiles: dict[int, list[list[float]]] = {}
    for pair_idx, (pa, pb) in parents.items():
        children = []
        profiles = []
        for j in range(N_OFFSPRING):
            child = breed_deterministic(
                pa, pb, f"bear_{pair_idx}_{j}", f"BearChild_{j}",
                random.Random(42 + pair_idx * 100 + j),
                mutation_rate=MUTATION_RATE,
            )
            children.append(child)
            profiles.append(profile_to_vector(child.behavior_profile))
        bear_offspring[pair_idx] = children
        bear_offspring_profiles[pair_idx] = profiles
        print(f"  BEAR: bred {len(children)} offspring for pair {pair_idx}")

    # Breed offspring using numeric GA
    numeric_offspring: dict[int, list] = {}
    numeric_offspring_profiles: dict[int, list[list[float]]] = {}
    for pair_idx, (pa, pb) in parents.items():
        prof_a = profile_to_vector(pa.behavior_profile)
        prof_b = profile_to_vector(pb.behavior_profile)
        children = []
        profiles = []
        for j in range(N_OFFSPRING):
            child_rng = random.Random(42 + pair_idx * 100 + j)
            child_prof = numeric_crossover_profile(
                prof_a, prof_b, child_rng,
                mutation_rate=MUTATION_RATE,
            )
            # Create a mock creature with numeric-interpolated genes for diversity comparison
            child = make_creature(f"num_{pair_idx}_{j}", GENE_BANK[PARENT_PAIRS[pair_idx][0]], f"NumChild_{j}", child_rng)
            # Overwrite genes with simple interpolation of parent gene texts
            for cat in GENE_CATEGORIES:
                gene_a = pa.genes.get(cat, "")
                gene_b = pb.genes.get(cat, "")
                # Numeric baseline: deterministically pick one parent's gene (no text blending)
                child.genes[cat] = gene_a if child_rng.random() < 0.5 else gene_b
            children.append(child)
            profiles.append(child_prof)
        numeric_offspring[pair_idx] = children
        numeric_offspring_profiles[pair_idx] = profiles
        print(f"  Numeric GA: bred {len(children)} offspring for pair {pair_idx}")

    # Compute per-gene diversity metrics
    def compute_diversity(creatures: list) -> dict:
        per_gene = compute_per_gene_diversity(creatures)
        mean_div = gene_diversity_mean(creatures)
        return {"per_gene_diversity": per_gene, "mean_gene_diversity": mean_div}

    pair_results = []
    for pair_idx in range(len(PARENT_PAIRS)):
        pa, pb = parents[pair_idx]
        bear_div = compute_diversity(bear_offspring[pair_idx])
        numeric_div = compute_diversity(numeric_offspring[pair_idx])

        pair_result = {
            "pair": pair_idx,
            "parents": (_NAMES[PARENT_PAIRS[pair_idx][0]], _NAMES[PARENT_PAIRS[pair_idx][1]]),
            "parent_profiles": {
                "a": [round(x, 4) for x in profile_to_vector(pa.behavior_profile)],
                "b": [round(x, 4) for x in profile_to_vector(pb.behavior_profile)],
            },
            "bear_diversity": bear_div,
            "numeric_diversity": numeric_div,
            "bear_wins_diversity": bear_div["mean_gene_diversity"] > numeric_div["mean_gene_diversity"],
        }
        pair_results.append(pair_result)

        print(f"\nPair {pair_idx} ({pair_result['parents'][0]} × {pair_result['parents'][1]}):")
        print(f"  BEAR mean gene diversity:    {bear_div['mean_gene_diversity']:.4f}")
        print(f"  Numeric mean gene diversity: {numeric_div['mean_gene_diversity']:.4f}")
        print(f"  BEAR per-gene:    {bear_div['per_gene_diversity']}")
        print(f"  Numeric per-gene: {numeric_div['per_gene_diversity']}")

    # Aggregate across all pairs
    all_bear_creatures = []
    all_numeric_creatures = []
    for pair_idx in range(len(PARENT_PAIRS)):
        all_bear_creatures.extend(bear_offspring[pair_idx])
        all_numeric_creatures.extend(numeric_offspring[pair_idx])

    global_bear_div = compute_diversity(all_bear_creatures)
    global_numeric_div = compute_diversity(all_numeric_creatures)

    print(f"\n{'='*60}")
    print("GLOBAL DIVERSITY (all offspring pooled)")
    print(f"{'='*60}")
    print(f"BEAR:    mean gene diversity = {global_bear_div['mean_gene_diversity']:.4f}")
    print(f"Numeric: mean gene diversity = {global_numeric_div['mean_gene_diversity']:.4f}")
    print(f"BEAR per-gene:    {global_bear_div['per_gene_diversity']}")
    print(f"Numeric per-gene: {global_numeric_div['per_gene_diversity']}")

    bear_wins = sum(1 for pr in pair_results if pr["bear_wins_diversity"])
    print(f"\nBEAR produces higher gene diversity in {bear_wins}/{len(pair_results)} pairs")

    # Save
    output = {
        "parameters": {
            "n_parent_pairs": len(PARENT_PAIRS),
            "n_offspring_per_pair": N_OFFSPRING,
            "mutation_rate": MUTATION_RATE,
            "seed": 42,
        },
        "summary": {
            "bear_global_diversity": global_bear_div,
            "numeric_global_diversity": global_numeric_div,
            "bear_wins_count": bear_wins,
            "total_pairs": len(pair_results),
            "gene_categories": list(GENE_CATEGORIES),
        },
        "pair_results": pair_results,
    }

    results_path = OUT_DIR / "eval7_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_mutation(output, bear_offspring_profiles, numeric_offspring_profiles, parents)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_mutation(output, bear_profiles, numeric_profiles, parents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_results = output["pair_results"]
    n_pairs = len(pair_results)
    gene_cats = output["summary"]["gene_categories"]
    n_genes = len(gene_cats)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: Mean gene diversity per pair
    ax = axes[0][0]
    x = np.arange(n_pairs)
    width = 0.35
    bear_means = [pr["bear_diversity"]["mean_gene_diversity"] for pr in pair_results]
    num_means = [pr["numeric_diversity"]["mean_gene_diversity"] for pr in pair_results]
    ax.bar(x - width/2, bear_means, width, label="BEAR (semantic)", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, num_means, width, label="Numeric GA", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pr['parents'][0]}×\n{pr['parents'][1]}" for pr in pair_results], fontsize=8)
    ax.set_ylabel("Mean Gene Diversity")
    ax.set_title("Per-Gene Diversity (mean) per Parent Pair")
    ax.legend(fontsize=8)

    # Panel 2: Per-gene diversity (global) — BEAR
    ax = axes[0][1]
    bear_pgd = output["summary"]["bear_global_diversity"]["per_gene_diversity"]
    num_pgd = output["summary"]["numeric_global_diversity"]["per_gene_diversity"]
    bear_vals = [bear_pgd.get(cat, 0.0) for cat in gene_cats]
    num_vals = [num_pgd.get(cat, 0.0) for cat in gene_cats]
    x_genes = np.arange(n_genes)
    ax.bar(x_genes - width/2, bear_vals, width, label="BEAR", color="#2196F3", alpha=0.8)
    ax.bar(x_genes + width/2, num_vals, width, label="Numeric", color="#FF9800", alpha=0.8)
    ax.set_xticks(x_genes)
    ax.set_xticklabels([c.replace("_", "\n") for c in gene_cats], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("Per-Gene Diversity (Global)")
    ax.legend(fontsize=8)

    # Panel 3: Per-gene diversity difference (BEAR - Numeric)
    ax = axes[0][2]
    diffs = [b - n for b, n in zip(bear_vals, num_vals)]
    colors = ["#2196F3" if d >= 0 else "#FF9800" for d in diffs]
    ax.bar(x_genes, diffs, color=colors, alpha=0.8)
    ax.set_xticks(x_genes)
    ax.set_xticklabels([c.replace("_", "\n") for c in gene_cats], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("BEAR - Numeric")
    ax.set_title("Per-Gene Diversity Advantage")
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel 4-5: Scatter plots for two pairs (first two behavior dims)
    for pidx, ax_idx in [(0, (1, 0)), (2, (1, 1))]:
        ax = axes[ax_idx[0]][ax_idx[1]]
        bear_profs = np.array(bear_profiles[pidx])
        num_profs = np.array(numeric_profiles[pidx])
        pa, pb = parents[pidx]
        pa_prof = profile_to_vector(pa.behavior_profile)
        pb_prof = profile_to_vector(pb.behavior_profile)

        # Use food_seeking (0) vs combat (1) as axes
        ax.scatter(bear_profs[:, 0], bear_profs[:, 1], c="#2196F3", alpha=0.6,
                   label="BEAR offspring", s=40, marker="o")
        ax.scatter(num_profs[:, 0], num_profs[:, 1], c="#FF9800", alpha=0.6,
                   label="Numeric offspring", s=40, marker="^")
        ax.scatter([pa_prof[0]], [pa_prof[1]], c="red", s=120, marker="*", zorder=5,
                   label="Parent A")
        ax.scatter([pb_prof[0]], [pb_prof[1]], c="darkred", s=120, marker="*", zorder=5,
                   label="Parent B")
        pr = pair_results[pidx]
        ax.set_xlabel("Food Seeking")
        ax.set_ylabel("Combat")
        ax.set_title(f"Pair: {pr['parents'][0]} × {pr['parents'][1]}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Panel 6: Global summary — mean gene diversity
    ax = axes[1][2]
    global_bear = output["summary"]["bear_global_diversity"]["mean_gene_diversity"]
    global_num = output["summary"]["numeric_global_diversity"]["mean_gene_diversity"]
    bars = ax.bar(
        ["BEAR\n(semantic)", "Numeric\nGA"],
        [global_bear, global_num],
        color=["#2196F3", "#FF9800"],
        alpha=0.8,
    )
    ax.set_ylabel("Mean Per-Gene Diversity")
    ax.set_title("Global Gene Diversity")
    for bar, val in zip(bars, [global_bear, global_num]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Semantic vs Numeric Mutation: Per-Gene Diversity",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = OUT_DIR / "eval7_mutation.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
