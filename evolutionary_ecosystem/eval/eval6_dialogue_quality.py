#!/usr/bin/env python3
"""Evaluation 6: Genome-conditioned retrieval diversity.

Tests whether BEAR retrieval produces meaningfully different behavioral
directives for creatures with different genomes.  This is BEAR's unique
advantage over numeric gene systems: the same genome that drives fast-path
movement also conditions slow-path natural language generation through
retrieval conditioning.

Design:
  - Select 4 diverse creature genomes from GENE_BANK (Bold, Timid, Curious, Calm)
  - For each creature, construct BEAR corpus and retrieve guidance for
    the same 5 standardized trigger scenarios (idle, predator, social,
    hungry, enraged)
  - Measure inter-creature guidance diversity (do different genomes produce
    different retrieved directives?) via embedding cosine similarity
  - Measure intra-creature cross-scenario consistency

Since this eval runs headless without an LLM, we measure at the retrieval
level: what behavioral directive does BEAR surface for each creature × scenario
combination?  Different directives → different LLM conditioning → different
dialogue.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — retrieval-level measurement only)

Outputs:
- eval6_results.json  — Per-creature retrieval data and diversity metrics
- eval6_dialogue.png  — Heatmaps of inter-creature similarity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.retriever import Retriever
from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    compute_per_gene_diversity,
    cosine_similarity,
    ensure_eval_patched,
    get_config,
    get_embedder,
)
from evolutionary_ecosystem.server.gene_engine import (
    build_corpus,
)
from bear.models import Context

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# Standardized trigger scenarios (same for all creatures)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "idle_exploration",
        "query": "I'm feeling content with happiness 72/100. What should I do? [Epoch: abundance, Weather: clear]",
        "tags": ["idle", "mood_happy", "happy", "epoch:abundance", "weather:clear"],
    },
    {
        "name": "predator_attack",
        "query": "A predator is attacking! (predator at dist 4.2) Do I flee or rally others to defend? [Epoch: predator_bloom, Weather: storm]",
        "tags": ["predator", "danger", "flee", "epoch:predator_bloom", "weather:storm"],
    },
    {
        "name": "social_greeting",
        "query": "I'm meeting Omega for the first time (distance 2.3). How do I react given my personality? [Epoch: abundance, Weather: clear]",
        "tags": ["greeting", "social", "epoch:abundance", "weather:clear"],
    },
    {
        "name": "hungry_survival",
        "query": "I'm feeling cautious with happiness 35/100. Energy: 22/100. What should I do? [Epoch: famine, Weather: rain]",
        "tags": ["idle", "mood_cautious", "unhappy", "hunger", "survival", "epoch:famine", "weather:rain"],
    },
    {
        "name": "enraged_combat",
        "query": "I am ENRAGED (rage=92/100). I must challenge the nearest creature. What do I do? [Epoch: ice_age, Weather: snow]",
        "tags": ["enraged", "aggression", "nearby", "epoch:ice_age", "weather:snow"],
    },
]

# Select 4 maximally diverse genomes from the bank
CREATURE_INDICES = [0, 1, 2, 3]  # Bold, Timid, Curious, Calm
CREATURE_LABELS = ["Bold", "Timid", "Curious", "Calm"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_eval_patched()

    embedder = get_embedder()
    config = get_config()

    print("=" * 60)
    print("EVAL 6: Slow-Path Dialogue Quality (Retrieval Conditioning)")
    print("=" * 60)

    # Build corpus + retriever for each creature
    creature_data = []
    for idx, label in zip(CREATURE_INDICES, CREATURE_LABELS):
        genes = GENE_BANK[idx]
        name = label
        corpus = build_corpus(name, genes)

        retriever = Retriever(corpus=corpus, config=config)
        retriever.build_index()

        creature_data.append({
            "label": label,
            "genes": genes,
            "retriever": retriever,
            "corpus_size": len(corpus.instructions),
        })
        print(f"\n{label}: corpus={len(corpus.instructions)} instructions")

    # Run each creature through each scenario
    all_guidances: dict[str, dict[str, str]] = {}  # creature -> scenario -> guidance text
    all_scores: dict[str, dict[str, float]] = {}   # creature -> scenario -> top score
    all_embeddings: dict[str, dict[str, list[float]]] = {}  # creature -> scenario -> embedding

    for cd in creature_data:
        label = cd["label"]
        retriever = cd["retriever"]
        all_guidances[label] = {}
        all_scores[label] = {}
        all_embeddings[label] = {}

        for scenario in SCENARIOS:
            ctx = Context(tags=scenario["tags"], domain="evolutionary_ecosystem")
            scored = retriever.retrieve(
                query=scenario["query"],
                context=ctx,
                top_k=3,
                threshold=0.3,
            )

            if scored:
                guidance = scored[0].instruction.content
                score = float(scored[0].final_score)
                # Combine top-3 for richer representation
                full_guidance = " | ".join(s.instruction.content[:100] for s in scored[:3])
            else:
                guidance = "(no retrieval)"
                score = 0.0
                full_guidance = "(no retrieval)"

            all_guidances[label][scenario["name"]] = full_guidance
            all_scores[label][scenario["name"]] = round(score, 4)

            # Embed the guidance text for cross-creature comparison
            emb_arr = embedder.embed_single(full_guidance)
            all_embeddings[label][scenario["name"]] = emb_arr.tolist()

            print(f"  {label} × {scenario['name']}: score={score:.3f} → {guidance[:80]}...")

    # Compute inter-creature diversity per scenario
    scenario_names = [s["name"] for s in SCENARIOS]
    inter_creature_sim = {}  # scenario -> pairwise similarity matrix
    for sname in scenario_names:
        embs = [all_embeddings[cd["label"]][sname] for cd in creature_data]
        n = len(embs)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i][j] = cosine_similarity(embs[i], embs[j])
        inter_creature_sim[sname] = sim_matrix.tolist()

        # Mean off-diagonal similarity
        off_diag = [sim_matrix[i][j] for i in range(n) for j in range(n) if i != j]
        mean_sim = float(np.mean(off_diag))
        print(f"\n  {sname}: mean inter-creature guidance similarity = {mean_sim:.4f}")

    # Compute per-creature cross-scenario consistency
    creature_consistency = {}
    for cd in creature_data:
        label = cd["label"]
        embs = [all_embeddings[label][sname] for sname in scenario_names]
        n = len(embs)
        sims = [cosine_similarity(embs[i], embs[j])
                for i in range(n) for j in range(n) if i != j]
        creature_consistency[label] = round(float(np.mean(sims)), 4)
        print(f"  {label} cross-scenario consistency: {creature_consistency[label]:.4f}")

    # Compute per-gene diversity across the creature population
    # Wrap gene dicts as simple objects with .genes attribute
    class _GeneHolder:
        def __init__(self, genes: dict[str, str]):
            self.genes = genes
    gene_holders = [_GeneHolder(cd["genes"]) for cd in creature_data]
    per_gene_div = compute_per_gene_diversity(gene_holders)
    mean_gene_diversity = float(np.mean(list(per_gene_div.values())))

    # Compute guidance diversity (across all creature × scenario)
    all_emb_pairs = []
    for sname in scenario_names:
        embs = [all_embeddings[cd["label"]][sname] for cd in creature_data]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                all_emb_pairs.append(cosine_similarity(embs[i], embs[j]))
    mean_guidance_sim = float(np.mean(all_emb_pairs))

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Mean per-gene diversity (fast-path):        {mean_gene_diversity:.4f}")
    for cat, div in per_gene_div.items():
        print(f"  {cat}: {div:.4f}")
    print(f"Mean guidance text similarity (slow-path):  {mean_guidance_sim:.4f}")
    print(f"Guidance diversity complements gene diversity? "
          f"{'YES' if mean_guidance_sim < 1.0 - mean_gene_diversity else 'NO'}")

    # Save results
    output = {
        "parameters": {
            "n_creatures": len(CREATURE_INDICES),
            "n_scenarios": len(SCENARIOS),
            "creature_labels": CREATURE_LABELS,
            "scenario_names": scenario_names,
        },
        "summary": {
            "gene_diversity": round(mean_gene_diversity, 4),
            "per_gene_diversity": {cat: round(div, 4) for cat, div in per_gene_div.items()},
            "mean_guidance_text_similarity": round(mean_guidance_sim, 4),
            "creature_consistency": creature_consistency,
        },
        "per_scenario_inter_creature_similarity": {
            sname: {
                "matrix": inter_creature_sim[sname],
                "mean_off_diagonal": round(float(np.mean([
                    inter_creature_sim[sname][i][j]
                    for i in range(len(CREATURE_LABELS))
                    for j in range(len(CREATURE_LABELS)) if i != j
                ])), 4),
            }
            for sname in scenario_names
        },
        "per_creature_data": {
            cd["label"]: {
                "genes": cd["genes"],
                "corpus_size": cd["corpus_size"],
                "guidances": all_guidances[cd["label"]],
                "scores": all_scores[cd["label"]],
            }
            for cd in creature_data
        },
    }

    results_path = OUT_DIR / "eval6_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_dialogue(output, inter_creature_sim)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_dialogue(output, inter_creature_sim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_names = output["parameters"]["scenario_names"]
    creature_labels = output["parameters"]["creature_labels"]
    n_scenarios = len(scenario_names)
    n_creatures = len(creature_labels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1-5: Heatmap per scenario
    for idx, sname in enumerate(scenario_names):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        matrix = np.array(inter_creature_sim[sname])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(n_creatures))
        ax.set_yticks(range(n_creatures))
        ax.set_xticklabels(creature_labels, fontsize=8, rotation=45)
        ax.set_yticklabels(creature_labels, fontsize=8)
        ax.set_title(sname.replace("_", " ").title(), fontsize=10)
        # Add values
        for i in range(n_creatures):
            for j in range(n_creatures):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if matrix[i][j] > 0.7 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 6: Summary bar chart — per-gene diversity + guidance similarity
    ax = axes[1][2]
    gene_div = output["summary"]["gene_diversity"]
    guidance_sim = output["summary"]["mean_guidance_text_similarity"]
    bars = ax.bar(
        ["Gene Diversity\n(per-gene mean)", "Guidance Text\n(semantic sim)"],
        [gene_div, guidance_sim],
        color=["#FF9800", "#2196F3"],
        alpha=0.8,
    )
    ax.set_ylabel("Score")
    ax.set_title("Gene Diversity vs Guidance Similarity")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, [gene_div, guidance_sim]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Genome-Conditioned Retrieval: Inter-Creature Guidance Diversity",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = OUT_DIR / "eval6_dialogue.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
