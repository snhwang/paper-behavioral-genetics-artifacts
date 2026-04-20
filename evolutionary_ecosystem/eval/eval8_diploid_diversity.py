#!/usr/bin/env python3
"""Evaluation 8: Diploid vs Haploid Inheritance — Diversity Preservation.

Compares diploid and haploid breeding across multiple generations to test
whether diploid inheritance (carrying both parents' alleles) preserves
higher genetic diversity than haploid inheritance (selecting one allele
per locus).

Three conditions:
1. Haploid (default): One parent's allele per locus
2. Diploid-dominant: Both alleles carried; parent A's expressed
3. Diploid-codominant: Both alleles carried and expressed together

Metrics:
- Genotype diversity (mean pairwise cosine distance of raw gene embeddings)
- Phenotype diversity (mean pairwise cosine distance of expressed behavior profiles)
- Allele retention (fraction of founding alleles still present at each generation)
- Heterozygosity rate (fraction of loci with distinct alleles per individual)
- Inheritance fidelity (cosine similarity to midparent)

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode)

Parameters:
- 8 founders from gene bank
- 5 generations of breeding
- 10 breeding events per generation
- 5 trials per condition
- Seed: 42

Outputs:
- eval8_diploid_results.json  — Raw data
- eval8_diploid_diversity.png — Comparison charts
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
from bear.models import Dominance, GeneLocus, LocusRegistry
from bear.evolution import BreedingConfig, CrossoverMethod, breed as bear_breed, express

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    GENE_CATEGORIES,
    SITUATION_NAMES,
    _NAMES,
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

BEAR_CONFIG = Config(
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_backend=EmbeddingBackend.NUMPY,
    priority_weight=0.3,
    default_threshold=0.3,
    default_top_k=3,
)

# ---------------------------------------------------------------------------
# Locus registries for each condition
# ---------------------------------------------------------------------------

HAPLOID_REGISTRY = LocusRegistry(loci=[
    GeneLocus(name=cat, position=i, dominance=Dominance.HAPLOID)
    for i, cat in enumerate(GENE_CATEGORIES)
])

DOMINANT_REGISTRY = LocusRegistry(loci=[
    GeneLocus(name=cat, position=i, dominance=Dominance.DOMINANT)
    for i, cat in enumerate(GENE_CATEGORIES)
])

CODOMINANT_REGISTRY = LocusRegistry(loci=[
    GeneLocus(name=cat, position=i, dominance=Dominance.CODOMINANT)
    for i, cat in enumerate(GENE_CATEGORIES)
])

CONDITIONS = {
    "haploid": HAPLOID_REGISTRY,
    "diploid_dominant": DOMINANT_REGISTRY,
    "diploid_codominant": CODOMINANT_REGISTRY,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gene_embedding(creature, embedder) -> np.ndarray:
    """Mean embedding of all gene texts from CORPUS instructions (not creature.genes dict).

    BUG FIX: Previously measured creature.genes dict which is set by random
    coin flip independently of BEAR breeding. Now measures the actual corpus
    instructions which contain the real bred genotype (including both alleles
    for diploid corpora).
    """
    vecs = []
    for inst in creature.corpus:
        vecs.append(embedder.embed_single(inst.content))
    if not vecs:
        # Fallback to creature.genes if corpus is empty
        vecs = [embedder.embed_single(t) for t in creature.genes.values()]
    return np.mean(vecs, axis=0)


def per_gene_embeddings(creature, embedder, locus_key: str = "gene_category") -> dict[str, np.ndarray]:
    """Embedding for each gene category from CORPUS instructions.

    BUG FIX: Previously measured creature.genes dict. Now measures corpus
    instructions grouped by gene_category metadata.
    """
    by_locus: dict[str, list] = {}
    for inst in creature.corpus:
        locus = inst.metadata.get(locus_key)
        if locus:
            by_locus.setdefault(locus, []).append(embedder.embed_single(inst.content))
    result = {}
    for cat in GENE_CATEGORIES:
        if cat in by_locus and by_locus[cat]:
            result[cat] = np.mean(by_locus[cat], axis=0)
        elif cat in creature.genes:
            result[cat] = embedder.embed_single(creature.genes[cat])
    return result


def pairwise_cosine_distances(vectors: list[np.ndarray]) -> list[float]:
    """All pairwise cosine distances among vectors."""
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = float(np.dot(vectors[i], vectors[j]) /
                        (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-9))
            dists.append(1.0 - sim)
    return dists


def heterozygosity(corpus, locus_key: str = "gene_category") -> float:
    """Fraction of loci where allele a and allele b have different content."""
    by_locus: dict[str, dict[str, str]] = {}
    for inst in corpus:
        locus = inst.metadata.get(locus_key)
        allele = inst.metadata.get("allele")
        if locus and allele in ("a", "b"):
            by_locus.setdefault(locus, {})[allele] = inst.content

    if not by_locus:
        return 0.0

    n_het = 0
    n_total = 0
    for locus, alleles in by_locus.items():
        if "a" in alleles and "b" in alleles:
            n_total += 1
            if alleles["a"] != alleles["b"]:
                n_het += 1
    return n_het / n_total if n_total > 0 else 0.0


def breed_with_registry(
    parent_a, parent_b, child_id: str, child_name: str,
    registry: LocusRegistry, rng: random.Random,
):
    """Breed two creatures using BEAR with the given locus registry."""
    config = BreedingConfig(
        crossover_rate=0.5,
        locus_key="gene_category",
        locus_registry=registry,
        crossover_method=CrossoverMethod.TAGGED,
        scope_to_child=False,
        seed=rng.randint(0, 2**31),
    )
    result = bear_breed(
        parent_a.corpus, parent_b.corpus,
        child_name, parent_a.name, parent_b.name,
        config=config,
    )

    # Extract genes from parents for the creature object
    child_genes = {}
    for cat in GENE_CATEGORIES:
        ga = parent_a.genes.get(cat, "")
        gb = parent_b.genes.get(cat, "")
        child_genes[cat] = ga if rng.random() < 0.5 else gb

    gen = max(parent_a.generation, parent_b.generation) + 1
    c = make_creature(child_id, child_genes, child_name, rng, generation=gen)
    c.corpus = result.child

    # For diploid corpora, express before computing behavior profile
    is_diploid = any(loc.dominance != Dominance.HAPLOID for loc in registry.loci)
    if is_diploid:
        expressed = express(result.child, registry, locus_key="gene_category")
        # Build a temporary corpus with expressed instructions for profile computation
        from bear import Corpus
        expressed_corpus = Corpus()
        for inst in expressed:
            expressed_corpus.add(inst)
        c.behavior_profile = compute_behavior_profile(expressed_corpus, BEAR_CONFIG)
    else:
        c.behavior_profile = compute_behavior_profile(result.child, BEAR_CONFIG)

    return c, result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_trial(condition: str, registry: LocusRegistry, seed: int, embedder):
    """Run one trial of multi-generation breeding."""
    rng = random.Random(seed)
    N_GENERATIONS = 5
    N_OFFSPRING_PER_GEN = 10

    # Create founders
    population = []
    for i in range(len(GENE_BANK)):
        genes = GENE_BANK[i]
        name = _NAMES[i]
        c = make_creature(f"f{i}", genes, name, rng, generation=0)
        population.append(c)

    # Track founding allele embeddings for retention measurement (from corpus, not genes dict)
    founding_gene_embs = {}
    for c in population:
        for inst in c.corpus:
            locus = inst.metadata.get("gene_category")
            if locus:
                key = f"{c.name}:{locus}:{inst.content[:30]}"
                founding_gene_embs[key] = embedder.embed_single(inst.content)
    n_founding = len(founding_gene_embs)

    gen_data = []
    cid_counter = 100

    for gen in range(N_GENERATIONS):
        # Breed offspring
        offspring = []
        for j in range(N_OFFSPRING_PER_GEN):
            pa = rng.choice(population)
            pb = rng.choice([c for c in population if c is not pa] or population)
            cid_counter += 1
            child_name = f"{condition}_g{gen+1}_{j}"
            try:
                child, result = breed_with_registry(
                    pa, pb, f"c{cid_counter}", child_name, registry, rng,
                )
                offspring.append((child, result))
            except Exception as e:
                print(f"  Warning: breeding failed for {child_name}: {e}")
                continue

        if not offspring:
            print(f"  Generation {gen+1}: no successful offspring, stopping")
            break

        # Compute metrics for this generation
        children = [c for c, _ in offspring]
        results = [r for _, r in offspring]

        # Gene diversity (genotype level — raw corpus embeddings)
        gene_embs = [gene_embedding(c, embedder) for c in children]
        genotype_diversity = float(np.mean(pairwise_cosine_distances(gene_embs))) if len(gene_embs) > 1 else 0.0

        # Phenotype diversity (behavior profile level)
        profiles = [profile_to_vector(c.behavior_profile) for c in children]
        profile_vecs = [np.array(p) for p in profiles]
        phenotype_diversity = float(np.mean(pairwise_cosine_distances(profile_vecs))) if len(profile_vecs) > 1 else 0.0

        # Heterozygosity (diploid only)
        het_rates = [heterozygosity(r.child) for r in results]
        mean_het = float(np.mean(het_rates))

        # Allele retention: fraction of founding alleles with > 0.8 cosine sim
        # to at least one current corpus instruction (BUG FIX: was using creature.genes)
        retained = 0
        for key, founding_emb in founding_gene_embs.items():
            best_sim = 0.0
            for child in children:
                for inst in child.corpus:
                    child_emb = embedder.embed_single(inst.content)
                    sim = float(np.dot(founding_emb, child_emb) /
                                (np.linalg.norm(founding_emb) * np.linalg.norm(child_emb) + 1e-9))
                    best_sim = max(best_sim, sim)
            if best_sim > 0.8:
                retained += 1
        allele_retention = retained / n_founding if n_founding > 0 else 0.0

        # Inheritance fidelity (cosine sim to midparent for each child)
        fidelities = []
        for child in children:
            # Find parents in population
            child_emb = gene_embedding(child, embedder)
            # Compare to population average as rough midparent
            pop_embs = [gene_embedding(p, embedder) for p in population]
            mid = np.mean(pop_embs, axis=0)
            sim = float(np.dot(child_emb, mid) /
                        (np.linalg.norm(child_emb) * np.linalg.norm(mid) + 1e-9))
            fidelities.append(sim)
        mean_fidelity = float(np.mean(fidelities))

        # Corpus size
        corpus_sizes = [len(r.child) for r in results]
        mean_corpus_size = float(np.mean(corpus_sizes))

        gen_data.append({
            "generation": gen + 1,
            "n_offspring": len(offspring),
            "genotype_diversity": round(genotype_diversity, 4),
            "phenotype_diversity": round(phenotype_diversity, 4),
            "heterozygosity": round(mean_het, 4),
            "allele_retention": round(allele_retention, 4),
            "inheritance_fidelity": round(mean_fidelity, 4),
            "mean_corpus_size": round(mean_corpus_size, 1),
        })

        print(f"  Gen {gen+1}: diversity={genotype_diversity:.4f}, "
              f"phenotype={phenotype_diversity:.4f}, "
              f"het={mean_het:.4f}, retention={allele_retention:.3f}")

        # Replace population: keep top half by diversity + all offspring
        population = children + population[:len(population)//2]
        # Cap population
        if len(population) > 20:
            population = population[:20]

    return gen_data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_seed = 42
    N_TRIALS = 5

    print("=" * 60)
    print("EVAL 8: Diploid vs Haploid Diversity Preservation")
    print("=" * 60)
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Trials per condition: {N_TRIALS}")
    print(f"Generations per trial: 5")
    print(f"Offspring per generation: 10")
    print()

    embedder = get_embedder()

    all_results: dict[str, list[list[dict]]] = {c: [] for c in CONDITIONS}

    for condition, registry in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"Dominance: {[loc.dominance.value for loc in registry.loci[:3]]}...")
        print(f"{'='*60}")

        for trial in range(N_TRIALS):
            seed = base_seed + trial * 1000
            print(f"\n  Trial {trial+1}/{N_TRIALS} (seed={seed})")
            gen_data = run_trial(condition, registry, seed, embedder)
            all_results[condition].append(gen_data)

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Mean metrics across trials at each generation")
    print("=" * 60)

    summary: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        summary[condition] = {
            "genotype_diversity": [],
            "phenotype_diversity": [],
            "heterozygosity": [],
            "allele_retention": [],
            "inheritance_fidelity": [],
        }
        for gen_idx in range(5):
            for metric in summary[condition]:
                vals = []
                for trial_data in all_results[condition]:
                    if gen_idx < len(trial_data):
                        vals.append(trial_data[gen_idx][metric])
                summary[condition][metric].append(
                    round(float(np.mean(vals)), 4) if vals else 0.0
                )

    # Print comparison table
    for metric in ["genotype_diversity", "phenotype_diversity", "heterozygosity",
                    "allele_retention"]:
        print(f"\n{metric}:")
        header = f"  {'Gen':>4s}"
        for c in CONDITIONS:
            header += f"  {c:>20s}"
        print(header)
        for gen in range(5):
            row = f"  {gen+1:>4d}"
            for c in CONDITIONS:
                row += f"  {summary[c][metric][gen]:>20.4f}"
            print(row)

    # ======================================================================
    # Helper: confidence interval
    # ======================================================================
    def _ci95(vals):
        """Return (lo, hi) for 95% CI using t-distribution."""
        a = np.asarray(vals, dtype=float)
        n = len(a)
        if n < 2:
            m = float(np.mean(a))
            return (m, m)
        se = float(stats.sem(a))
        h = se * float(stats.t.ppf(0.975, n - 1))
        m = float(np.mean(a))
        return (round(m - h, 6), round(m + h, 6))

    # ======================================================================
    # Helper: full statistical test battery
    # ======================================================================
    def _run_statistical_tests(a, b, label_a, label_b, metric_name):
        """Run Shapiro-Wilk, Welch's t, Mann-Whitney U, Cohen's d on two arrays.

        Returns a dict of results and prints to console.
        """
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        result = {
            "metric": metric_name,
            "group_a": label_a,
            "group_b": label_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": round(float(np.mean(a)), 6),
            "mean_b": round(float(np.mean(b)), 6),
            "std_a": round(float(np.std(a, ddof=1)), 6) if len(a) > 1 else 0.0,
            "std_b": round(float(np.std(b, ddof=1)), 6) if len(b) > 1 else 0.0,
            "ci95_a": list(_ci95(a)),
            "ci95_b": list(_ci95(b)),
        }

        if len(a) < 2 or len(b) < 2:
            print(f"    {metric_name}: insufficient samples (n_a={len(a)}, n_b={len(b)})")
            result.update({
                "shapiro_a": None, "shapiro_b": None,
                "welch_t": None, "welch_p": None,
                "mann_whitney_U": None, "mann_whitney_p": None,
                "cohens_d": None,
            })
            return result

        # Shapiro-Wilk normality (requires n >= 3)
        if len(a) >= 3:
            sw_a_stat, sw_a_p = stats.shapiro(a)
            result["shapiro_a"] = {"W": round(float(sw_a_stat), 6), "p": round(float(sw_a_p), 6)}
        else:
            result["shapiro_a"] = None
        if len(b) >= 3:
            sw_b_stat, sw_b_p = stats.shapiro(b)
            result["shapiro_b"] = {"W": round(float(sw_b_stat), 6), "p": round(float(sw_b_p), 6)}
        else:
            result["shapiro_b"] = None

        # Welch's t-test (equal_var=False)
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        result["welch_t"] = round(float(t_stat), 6)
        result["welch_p"] = round(float(t_p), 6)

        # Mann-Whitney U test
        try:
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
            result["mann_whitney_U"] = round(float(u_stat), 6)
            result["mann_whitney_p"] = round(float(u_p), 6)
        except ValueError:
            result["mann_whitney_U"] = None
            result["mann_whitney_p"] = None

        # Cohen's d (pooled std)
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
        d = float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0
        result["cohens_d"] = round(d, 6)

        # Console output
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "n.s."
        print(f"    {metric_name}: {label_a}={np.mean(a):.4f}\u00b1{np.std(a, ddof=1):.4f} "
              f"CI{result['ci95_a']} vs "
              f"{label_b}={np.mean(b):.4f}\u00b1{np.std(b, ddof=1):.4f} "
              f"CI{result['ci95_b']}")
        if result["shapiro_a"] is not None:
            print(f"      Shapiro-Wilk: {label_a} W={result['shapiro_a']['W']:.4f} p={result['shapiro_a']['p']:.4f}, "
                  f"{label_b} W={result['shapiro_b']['W']:.4f} p={result['shapiro_b']['p']:.4f}")
        print(f"      Welch's t={t_stat:.3f}, p={t_p:.4f} {sig}")
        mw_u_str = f"{result['mann_whitney_U']:.1f}" if result["mann_whitney_U"] is not None else "N/A"
        mw_p_str = f"{result['mann_whitney_p']:.4f}" if result["mann_whitney_p"] is not None else "N/A"
        print(f"      Mann-Whitney U={mw_u_str}, p={mw_p_str}")
        print(f"      Cohen's d={d:.3f}")

        return result

    # ======================================================================
    # Statistical significance (final generation) — full pairwise battery
    # ======================================================================
    print("\n" + "=" * 60)
    print("Statistical Tests (final generation, all pairwise comparisons)")
    print("=" * 60)

    stat_test_results = []
    pairwise_comparisons = [
        ("haploid", "diploid_dominant"),
        ("haploid", "diploid_codominant"),
        ("diploid_dominant", "diploid_codominant"),
    ]
    test_metrics = ["genotype_diversity", "phenotype_diversity",
                    "allele_retention", "heterozygosity"]

    for cond_a, cond_b in pairwise_comparisons:
        print(f"\n  {cond_a} vs {cond_b} (final generation):")
        for metric in test_metrics:
            vals_a = [trial[-1][metric] for trial in all_results[cond_a] if trial]
            vals_b = [trial[-1][metric] for trial in all_results[cond_b] if trial]
            test_result = _run_statistical_tests(
                vals_a, vals_b, cond_a, cond_b, metric
            )
            stat_test_results.append(test_result)

    # ======================================================================
    # 95% confidence intervals for summary means
    # ======================================================================
    summary_ci: dict[str, dict[str, list]] = {}
    for condition in CONDITIONS:
        summary_ci[condition] = {}
        for metric in summary[condition]:
            ci_per_gen = []
            for gen_idx in range(5):
                vals = []
                for trial_data in all_results[condition]:
                    if gen_idx < len(trial_data):
                        vals.append(trial_data[gen_idx][metric])
                ci_per_gen.append(list(_ci95(vals)) if vals else [0.0, 0.0])
            summary_ci[condition][metric] = ci_per_gen

    # ======================================================================
    # Collect raw per-trial final-generation values
    # ======================================================================
    raw_per_trial: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        raw_per_trial[condition] = {}
        for metric in test_metrics:
            raw_per_trial[condition][metric] = [
                trial[-1][metric] for trial in all_results[condition] if trial
            ]

    # ======================================================================
    # Save JSON
    # ======================================================================
    seeds_used = [base_seed + t * 1000 for t in range(N_TRIALS)]
    output = {
        "config": {
            "n_trials": N_TRIALS,
            "n_generations": 5,
            "n_offspring_per_gen": 10,
            "base_seed": base_seed,
            "seeds": seeds_used,
            "conditions": list(CONDITIONS.keys()),
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "gene_categories": GENE_CATEGORIES,
        },
        "summary": summary,
        "summary_ci95": summary_ci,
        "statistical_tests": stat_test_results,
        "raw_final_generation": raw_per_trial,
        "per_trial": {c: all_results[c] for c in CONDITIONS},
    }
    json_path = OUT_DIR / "eval8_diploid_results.json"
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
        fig.suptitle(
            "Eval 8: Diploid vs Haploid Diversity Preservation",
            fontsize=14,
        )

        colors = {
            "haploid": "#4CAF50",
            "diploid_dominant": "#FF9800",
            "diploid_codominant": "#2196F3",
        }
        labels = {
            "haploid": "Haploid",
            "diploid_dominant": "Diploid (Dominant)",
            "diploid_codominant": "Diploid (Co-dominant)",
        }
        gens = list(range(1, 6))

        # Panel 1: Genotype diversity over generations
        ax = axes[0, 0]
        for cond in CONDITIONS:
            vals = summary[cond]["genotype_diversity"]
            ax.plot(gens, vals, "o-", color=colors[cond], label=labels[cond],
                    linewidth=2, markersize=6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean Pairwise Cosine Distance")
        ax.set_title("Genotype Diversity")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)

        # Panel 2: Phenotype diversity over generations
        ax = axes[0, 1]
        for cond in CONDITIONS:
            vals = summary[cond]["phenotype_diversity"]
            ax.plot(gens, vals, "o-", color=colors[cond], label=labels[cond],
                    linewidth=2, markersize=6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean Pairwise Cosine Distance")
        ax.set_title("Phenotype Diversity")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)

        # Panel 3: Heterozygosity over generations
        ax = axes[1, 0]
        for cond in CONDITIONS:
            vals = summary[cond]["heterozygosity"]
            ax.plot(gens, vals, "o-", color=colors[cond], label=labels[cond],
                    linewidth=2, markersize=6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Heterozygosity Rate")
        ax.set_title("Heterozygosity (Diploid Only)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(-0.05, 1.05)

        # Panel 4: Allele retention over generations
        ax = axes[1, 1]
        for cond in CONDITIONS:
            vals = summary[cond]["allele_retention"]
            ax.plot(gens, vals, "o-", color=colors[cond], label=labels[cond],
                    linewidth=2, markersize=6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fraction Retained (sim > 0.8)")
        ax.set_title("Founding Allele Retention")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        fig_path = OUT_DIR / "eval8_diploid_diversity.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
