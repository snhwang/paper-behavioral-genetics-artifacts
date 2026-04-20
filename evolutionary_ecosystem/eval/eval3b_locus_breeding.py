#!/usr/bin/env python3
"""Evaluation 3b: Locus-based vs legacy breeding comparison.

Compares three breeding mechanisms using the same parent pairs:
1. Text-splicing (current harness default) — blends gene text per category
2. Legacy BEAR recombination — random per-instruction sampling at rate 0.5
3. Locus-based BEAR recombination — one parent per gene_category locus

Metrics:
- Behavioral coverage (fraction of 7 situations with strength > threshold)
- Parent-offspring behavior profile similarity
- Gene embedding similarity to midparent
- Inter-offspring diversity (mean pairwise cosine distance)
- Corpus size and locus inheritance balance

Statistics:
- 95% confidence intervals on all reported means
- Pairwise significance tests (Shapiro-Wilk, Welch's t, Mann-Whitney U, Cohen's d)
- Results aggregated across multiple independent seeds

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode)

Parameters:
- 8 parent pairs from diverse gene bank archetypes
- 10 offspring per pair per method per seed
- Seeds: 42, 1042, 2042, 3042, 4042

Outputs:
- eval3b_results.json   — Raw data (including per-seed data and statistical tests)
- eval3b_breeding.png   — Comparison charts
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.evolution import BreedingConfig, breed as bear_breed

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    GENE_CATEGORIES,
    SITUATION_NAMES,
    _NAMES,
    breed_deterministic,
    cosine_similarity,
    get_embedder,
    make_creature,
    profile_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import (
    build_corpus,
    compute_behavior_profile,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gene_embedding(creature, embedder) -> np.ndarray:
    """Mean embedding of all gene texts."""
    vecs = [embedder.embed_single(t) for t in creature.genes.values()]
    return np.mean(vecs, axis=0)


def behavior_coverage(profile, threshold: float = 0.3) -> float:
    """Fraction of situations with strength above threshold."""
    vec = profile_to_vector(profile)
    return sum(1 for v in vec if v > threshold) / len(vec)


def gene_category_coverage(corpus) -> float:
    """Fraction of the 7 behavior gene categories represented in the corpus.

    This is the key metric: legacy recombination can lose entire gene
    categories, while locus-based breeding guarantees every category
    present in either parent appears in the offspring.
    """
    from evolutionary_ecosystem.server.gene_engine import (
        BEHAVIOR_CATEGORIES,
    )
    present = set()
    for inst in corpus:
        cat = inst.metadata.get("gene_category")
        if cat:
            present.add(cat)
    return len(present & set(BEHAVIOR_CATEGORIES)) / len(BEHAVIOR_CATEGORIES)


def pairwise_cosine_distances(vectors: list[list[float]]) -> list[float]:
    """All pairwise cosine distances (1 - similarity) among vectors."""
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dists.append(1.0 - cosine_similarity(vectors[i], vectors[j]))
    return dists


def breed_bear_legacy(parent_a, parent_b, child_id, child_name, rng):
    """Breed using BEAR's legacy per-instruction recombination (no locus)."""
    config = BreedingConfig(
        crossover_rate=0.5,
        scope_to_child=False,  # preserve parent scope tags for retrieval
        seed=rng.randint(0, 2**31),
    )
    result = bear_breed(
        parent_a.corpus, parent_b.corpus,
        child_name, parent_a.name, parent_b.name,
        config=config,
    )
    # Rebuild creature from the child corpus
    # Extract genes from parent blend (same as text-splicing for gene text)
    child_genes = {}
    for cat in GENE_CATEGORIES:
        ga = parent_a.genes.get(cat, "")
        gb = parent_b.genes.get(cat, "")
        child_genes[cat] = ga if rng.random() < 0.5 else gb
    gen = max(parent_a.generation, parent_b.generation) + 1
    c = make_creature(child_id, child_genes, child_name, rng, generation=gen)
    # Replace corpus with the BEAR-bred one
    c.corpus = result.child
    # Recompute behavior profile from the BEAR-bred corpus
    bear_config = Config(
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_backend=EmbeddingBackend.NUMPY,
        priority_weight=0.3,
        default_threshold=0.3,
        default_top_k=3,
    )
    c.behavior_profile = compute_behavior_profile(
        result.child, bear_config,
    )
    return c, result


def breed_bear_locus(parent_a, parent_b, child_id, child_name, rng):
    """Breed using BEAR's locus-based recombination (gene_category as locus)."""
    config = BreedingConfig(
        crossover_rate=0.5,
        locus_key="gene_category",
        scope_to_child=False,  # preserve parent scope tags for retrieval
        seed=rng.randint(0, 2**31),
    )
    result = bear_breed(
        parent_a.corpus, parent_b.corpus,
        child_name, parent_a.name, parent_b.name,
        config=config,
    )
    # Extract genes from parent blend
    child_genes = {}
    for cat in GENE_CATEGORIES:
        ga = parent_a.genes.get(cat, "")
        gb = parent_b.genes.get(cat, "")
        child_genes[cat] = ga if rng.random() < 0.5 else gb
    gen = max(parent_a.generation, parent_b.generation) + 1
    c = make_creature(child_id, child_genes, child_name, rng, generation=gen)
    # Replace corpus with the BEAR-bred one
    c.corpus = result.child
    # Recompute behavior profile from the BEAR-bred corpus
    bear_config = Config(
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_backend=EmbeddingBackend.NUMPY,
        priority_weight=0.3,
        default_threshold=0.3,
        default_top_k=3,
    )
    c.behavior_profile = compute_behavior_profile(
        result.child, bear_config,
    )
    return c, result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

SEEDS = [42, 1042, 2042, 3042, 4042]


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Return (ci_low, ci_high) for the mean using a t-distribution."""
    n = len(values)
    if n < 2:
        m = float(np.mean(values))
        return (m, m)
    m = float(np.mean(values))
    se = float(stats.sem(values))
    ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return (round(ci[0], 6), round(ci[1], 6))


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def pairwise_stat_tests(
    data_by_method: dict[str, list[float]],
    methods: list[str],
) -> list[dict]:
    """Run pairwise significance tests between all method pairs.

    Returns a list of dicts, one per pair, containing:
    - shapiro_a, shapiro_b (W-stat, p-value)
    - welch_t (t-stat, p-value)
    - mann_whitney_u (U-stat, p-value)
    - cohens_d
    """
    pairs = [
        ("text_splice", "bear_legacy"),
        ("text_splice", "bear_locus"),
        ("bear_legacy", "bear_locus"),
    ]
    results = []
    for ma, mb in pairs:
        a = data_by_method[ma]
        b = data_by_method[mb]
        entry: dict = {"pair": f"{ma} vs {mb}"}

        # Shapiro-Wilk normality test (needs n >= 3)
        if len(a) >= 3:
            sw_a = stats.shapiro(a)
            entry["shapiro_a"] = {"W": round(sw_a.statistic, 6),
                                  "p": round(sw_a.pvalue, 6)}
        else:
            entry["shapiro_a"] = {"W": None, "p": None}

        if len(b) >= 3:
            sw_b = stats.shapiro(b)
            entry["shapiro_b"] = {"W": round(sw_b.statistic, 6),
                                  "p": round(sw_b.pvalue, 6)}
        else:
            entry["shapiro_b"] = {"W": None, "p": None}

        # Welch's t-test (unequal variance)
        if len(a) >= 2 and len(b) >= 2:
            t_res = stats.ttest_ind(a, b, equal_var=False)
            entry["welch_t"] = {"t": round(float(t_res.statistic), 6),
                                "p": round(float(t_res.pvalue), 6)}
        else:
            entry["welch_t"] = {"t": None, "p": None}

        # Mann-Whitney U test
        if len(a) >= 1 and len(b) >= 1:
            u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
            entry["mann_whitney_u"] = {"U": round(float(u_res.statistic), 6),
                                       "p": round(float(u_res.pvalue), 6)}
        else:
            entry["mann_whitney_u"] = {"U": None, "p": None}

        # Cohen's d
        entry["cohens_d"] = round(cohens_d(a, b), 6)

        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_single_seed(seed: int, embedder, N_PAIRS: int, N_OFFSPRING: int,
                    methods: list[str]) -> dict[str, list[dict]]:
    """Run the full breeding evaluation for one seed.

    Returns all_results: {method: [pair_data, ...]}
    """
    rng = random.Random(seed)

    # Create parents from gene bank
    parents = []
    for i in range(len(GENE_BANK)):
        genes = GENE_BANK[i]
        name = _NAMES[i]
        c = make_creature(f"p{i}", genes, name, rng, generation=0)
        parents.append(c)

    # Pair parents for maximum diversity (adjacent in gene bank)
    pairs = []
    for i in range(N_PAIRS):
        pa = parents[i % len(parents)]
        pb = parents[(i + 1) % len(parents)]
        pairs.append((pa, pb))

    all_results: dict[str, list[dict]] = {m: [] for m in methods}

    cid = 0
    for pair_idx, (pa, pb) in enumerate(pairs):
        pa_profile = profile_to_vector(pa.behavior_profile)
        pb_profile = profile_to_vector(pb.behavior_profile)
        midparent_profile = [(a + b) / 2 for a, b in zip(pa_profile, pb_profile)]
        pa_gene_emb = gene_embedding(pa, embedder)
        pb_gene_emb = gene_embedding(pb, embedder)
        midparent_gene = (pa_gene_emb + pb_gene_emb) / 2

        for method in methods:
            method_rng = random.Random(seed + pair_idx * 100)
            offspring_profiles = []
            pair_data = {
                "pair": f"{pa.name} x {pb.name}",
                "method": method,
                "offspring": [],
            }

            for j in range(N_OFFSPRING):
                cid += 1
                child_name = f"{method}_{pair_idx}_{j}"

                if method == "text_splice":
                    child = breed_deterministic(
                        pa, pb, f"c{cid}", child_name, method_rng,
                        mutation_rate=0.0,  # no mutation for fair comparison
                    )
                    locus_choices = {}
                    corpus_size = len(child.corpus)
                elif method == "bear_legacy":
                    child, result = breed_bear_legacy(
                        pa, pb, f"c{cid}", child_name, method_rng,
                    )
                    locus_choices = {}
                    corpus_size = len(result.child)
                else:  # bear_locus
                    child, result = breed_bear_locus(
                        pa, pb, f"c{cid}", child_name, method_rng,
                    )
                    locus_choices = result.locus_choices
                    corpus_size = len(result.child)

                child_profile = profile_to_vector(child.behavior_profile)
                child_gene_emb = gene_embedding(child, embedder)

                beh_sim = cosine_similarity(child_profile, midparent_profile)
                gene_sim = float(np.dot(child_gene_emb, midparent_gene) /
                                 (np.linalg.norm(child_gene_emb) *
                                  np.linalg.norm(midparent_gene) + 1e-9))
                cov = behavior_coverage(child.behavior_profile)

                # For BEAR methods, measure gene category coverage from
                # the bred corpus; for text_splice, it's always 1.0
                if method == "text_splice":
                    gene_cov = 1.0
                else:
                    gene_cov = gene_category_coverage(
                        result.child if method != "text_splice" else child.corpus
                    )

                offspring_profiles.append(child_profile)

                pair_data["offspring"].append({
                    "name": child_name,
                    "corpus_size": corpus_size,
                    "behavior_sim": round(beh_sim, 4),
                    "gene_sim": round(gene_sim, 4),
                    "coverage": round(cov, 3),
                    "gene_category_coverage": round(gene_cov, 3),
                    "locus_choices": locus_choices,
                    "profile": {SITUATION_NAMES[k]: round(child_profile[k], 4)
                                for k in range(len(SITUATION_NAMES))},
                })

            # Inter-offspring diversity
            dists = pairwise_cosine_distances(offspring_profiles)
            pair_data["mean_offspring_diversity"] = (
                round(float(np.mean(dists)), 4) if dists else 0.0
            )

            all_results[method].append(pair_data)

    return all_results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_PAIRS = 8
    N_OFFSPRING = 10  # more offspring to capture stochastic variation

    print("=" * 60)
    print("EVAL 3b: Locus-Based vs Legacy Breeding Comparison")
    print("=" * 60)
    print(f"Parent pairs: {N_PAIRS}")
    print(f"Offspring per pair per method: {N_OFFSPRING}")
    print(f"Seeds: {SEEDS}")
    print(f"Methods: text-splicing, BEAR legacy, BEAR locus")
    print()

    embedder = get_embedder()
    methods = ["text_splice", "bear_legacy", "bear_locus"]

    # ------------------------------------------------------------------
    # Run all seeds
    # ------------------------------------------------------------------
    per_seed_results: dict[int, dict[str, list[dict]]] = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        print("Creating parent population...")
        seed_results = run_single_seed(
            seed, embedder, N_PAIRS, N_OFFSPRING, methods,
        )
        per_seed_results[seed] = seed_results
        n_offspring = sum(len(p["offspring"])
                         for p in seed_results[methods[0]])
        print(f"  Completed: {n_offspring} offspring per method")

    # ------------------------------------------------------------------
    # Aggregate across all seeds (use the union of all seed data)
    # ------------------------------------------------------------------
    # Build merged all_results for backward-compatible summary / plotting
    # (concatenates pair data from all seeds)
    all_results: dict[str, list[dict]] = {m: [] for m in methods}
    for seed in SEEDS:
        for m in methods:
            all_results[m].extend(per_seed_results[seed][m])

    # ======================================================================
    # Summary (aggregate across all seeds)
    # ======================================================================
    print("\n" + "=" * 60)
    print("SUMMARY (aggregated across all seeds)")
    print("=" * 60)

    summary = {}
    # Collect flat arrays per method for statistical tests later
    flat_data: dict[str, dict[str, list[float]]] = {m: {} for m in methods}

    for method in methods:
        covs = [o["coverage"]
                for pair in all_results[method]
                for o in pair["offspring"]]
        beh_sims = [o["behavior_sim"]
                    for pair in all_results[method]
                    for o in pair["offspring"]]
        gene_sims = [o["gene_sim"]
                     for pair in all_results[method]
                     for o in pair["offspring"]]
        corpus_sizes = [o["corpus_size"]
                        for pair in all_results[method]
                        for o in pair["offspring"]]
        gene_covs = [o["gene_category_coverage"]
                     for pair in all_results[method]
                     for o in pair["offspring"]]
        diversities = [pair["mean_offspring_diversity"]
                       for pair in all_results[method]]

        flat_data[method] = {
            "coverage": covs,
            "behavior_sim": beh_sims,
            "gene_sim": gene_sims,
            "offspring_diversity": diversities,
        }

        summary[method] = {
            "mean_coverage": round(float(np.mean(covs)), 3),
            "std_coverage": round(float(np.std(covs)), 3),
            "ci95_coverage": confidence_interval_95(covs),
            "mean_gene_cat_coverage": round(float(np.mean(gene_covs)), 3),
            "std_gene_cat_coverage": round(float(np.std(gene_covs)), 3),
            "mean_behavior_sim": round(float(np.mean(beh_sims)), 4),
            "ci95_behavior_sim": confidence_interval_95(beh_sims),
            "mean_gene_sim": round(float(np.mean(gene_sims)), 4),
            "ci95_gene_sim": confidence_interval_95(gene_sims),
            "mean_corpus_size": round(float(np.mean(corpus_sizes)), 1),
            "mean_offspring_diversity": round(float(np.mean(diversities)), 4),
            "ci95_offspring_diversity": confidence_interval_95(diversities),
            "n_seeds": len(SEEDS),
            "n_total_offspring": len(covs),
        }

    header = f"{'Metric':<28s}"
    for m in methods:
        header += f"  {m:>14s}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Mean coverage", "mean_coverage"),
        ("  95% CI", "ci95_coverage"),
        ("Gene category coverage", "mean_gene_cat_coverage"),
        ("Std gene cat coverage", "std_gene_cat_coverage"),
        ("Behavior sim (midparent)", "mean_behavior_sim"),
        ("  95% CI", "ci95_behavior_sim"),
        ("Gene sim (midparent)", "mean_gene_sim"),
        ("  95% CI", "ci95_gene_sim"),
        ("Mean corpus size", "mean_corpus_size"),
        ("Inter-offspring diversity", "mean_offspring_diversity"),
        ("  95% CI", "ci95_offspring_diversity"),
    ]
    for label, key in metrics:
        row = f"{label:<28s}"
        for m in methods:
            val = summary[m][key]
            if isinstance(val, tuple):
                row += f"  {f'[{val[0]:.4f},{val[1]:.4f}]':>14s}"
            else:
                row += f"  {val:>14}"
        print(row)

    # Per-pair breakdown (first seed only, for brevity)
    first_seed_results = per_seed_results[SEEDS[0]]
    print(f"\n{'Pair':<25s}", end="")
    for m in methods:
        print(f"  {m:>14s}", end="")
    print()
    print("-" * 70)

    for pair_idx in range(N_PAIRS):
        pair_name = first_seed_results["text_splice"][pair_idx]["pair"]
        row = f"{pair_name:<25s}"
        for m in methods:
            gene_covs = [o["gene_category_coverage"]
                         for o in first_seed_results[m][pair_idx]["offspring"]]
            row += f"  {np.mean(gene_covs):>14.3f}"
        print(row)

    # ======================================================================
    # Statistical tests (pairwise between methods)
    # ======================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (pairwise)")
    print("=" * 60)

    metric_names_for_tests = [
        ("coverage", "Coverage"),
        ("behavior_sim", "Behavior Similarity"),
        ("gene_sim", "Gene Embedding Similarity"),
        ("offspring_diversity", "Inter-Offspring Diversity"),
    ]

    statistical_tests: dict[str, list[dict]] = {}
    for metric_key, metric_label in metric_names_for_tests:
        data_by_method = {m: flat_data[m][metric_key] for m in methods}
        tests = pairwise_stat_tests(data_by_method, methods)
        statistical_tests[metric_key] = tests

        print(f"\n  {metric_label}:")
        for t in tests:
            pair_label = t["pair"]
            welch_p = t["welch_t"]["p"]
            mw_p = t["mann_whitney_u"]["p"]
            d = t["cohens_d"]
            sig = ""
            if welch_p is not None and welch_p < 0.05:
                sig = " *"
            if welch_p is not None and welch_p < 0.01:
                sig = " **"
            if welch_p is not None and welch_p < 0.001:
                sig = " ***"
            print(f"    {pair_label:<30s}  Welch p={welch_p}  "
                  f"MW-U p={mw_p}  d={d:.3f}{sig}")

    # ======================================================================
    # Save JSON
    # ======================================================================
    # Build per-seed raw data for the JSON output
    per_seed_json: dict[str, dict[str, list[dict]]] = {}
    for seed in SEEDS:
        per_seed_json[str(seed)] = {
            m: per_seed_results[seed][m] for m in methods
        }

    output = {
        "config": {
            "n_pairs": N_PAIRS,
            "n_offspring_per_pair": N_OFFSPRING,
            "seeds": SEEDS,
            "seed": SEEDS[0],  # backward compat
            "methods": methods,
            "embedding_model": "BAAI/bge-base-en-v1.5",
        },
        "summary": summary,
        "statistical_tests": statistical_tests,
        "per_pair": {m: all_results[m] for m in methods},
        "per_seed": per_seed_json,
    }
    json_path = OUT_DIR / "eval3b_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ======================================================================
    # Plot
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Eval 3b: Locus-Based vs Legacy Breeding", fontsize=14)

        colors = {"text_splice": "#4CAF50", "bear_legacy": "#FF9800",
                  "bear_locus": "#2196F3"}
        labels = {"text_splice": "Text Splice", "bear_legacy": "BEAR Legacy",
                  "bear_locus": "BEAR Locus"}

        # Panel 1: Coverage per method
        ax = axes[0, 0]
        for i, m in enumerate(methods):
            covs = [o["coverage"]
                    for pair in all_results[m]
                    for o in pair["offspring"]]
            ax.bar(i, np.mean(covs), color=colors[m], label=labels[m],
                   yerr=np.std(covs), capsize=5, alpha=0.8)
        ax.set_ylabel("Behavioral Coverage")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_title("Mean Behavioral Coverage")
        ax.set_ylim(0, 1.05)

        # Panel 2: Parent-offspring similarity
        ax = axes[0, 1]
        width = 0.25
        x = np.arange(len(methods))
        beh_vals = [summary[m]["mean_behavior_sim"] for m in methods]
        gene_vals = [summary[m]["mean_gene_sim"] for m in methods]
        ax.bar(x - width / 2, beh_vals, width, label="Behavior", alpha=0.8)
        ax.bar(x + width / 2, gene_vals, width, label="Gene Embed", alpha=0.8)
        ax.set_ylabel("Cosine Similarity to Midparent")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_title("Parent-Offspring Similarity")
        ax.legend()

        # Panel 3: Per-pair coverage (first seed for clarity)
        ax = axes[1, 0]
        x = np.arange(N_PAIRS)
        width = 0.25
        for i, m in enumerate(methods):
            pair_covs = [np.mean([o["coverage"]
                                  for o in first_seed_results[m][p]["offspring"]])
                         for p in range(N_PAIRS)]
            ax.bar(x + i * width, pair_covs, width, color=colors[m],
                   label=labels[m], alpha=0.8)
        ax.set_ylabel("Coverage")
        ax.set_xlabel("Parent Pair")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"P{i+1}" for i in range(N_PAIRS)], fontsize=8)
        ax.set_title("Coverage per Parent Pair")
        ax.legend(fontsize=8)

        # Panel 4: Inter-offspring diversity
        ax = axes[1, 1]
        for i, m in enumerate(methods):
            divs = [pair["mean_offspring_diversity"]
                    for pair in all_results[m]]
            ax.bar(i, np.mean(divs), color=colors[m], label=labels[m],
                   yerr=np.std(divs), capsize=5, alpha=0.8)
        ax.set_ylabel("Mean Pairwise Cosine Distance")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_title("Inter-Offspring Diversity")

        plt.tight_layout()
        fig_path = OUT_DIR / "eval3b_breeding.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
