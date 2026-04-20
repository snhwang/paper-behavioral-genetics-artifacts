#!/usr/bin/env python3
"""Evaluation 9: Diploid vs Haploid Inheritance Under Selection Pressure.

Uses the full Evolutionary Ecosystem simulation (evoeco) with natural and
sexual selection to test whether diploid inheritance preserves genetic
diversity better than haploid under directional selection pressure.

Hypothesis: Diploid inheritance preserves latent diversity via recessive
alleles that survive silently in heterozygotes, re-emerging when
environmental shifts make them advantageous.

Three conditions:
1. Haploid (text-splicing, current default) — one allele per locus
2. Diploid-dominant (BEAR breed + express) — allele A expressed
3. Diploid-codominant (BEAR breed + express) — both alleles expressed

The evoeco simulation provides:
- Natural selection: predators, starvation, weather damage
- Sexual selection: breeding_prob = avg_breeding_drive × avg_attractiveness
- Epoch cycling: Abundance → Ice Age → Predator Bloom → Expansion → Famine

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode)

Parameters:
- 12 starting creatures, max population 30
- 30,000 ticks per trial
- 5 epochs cycling
- 5 trials per condition
- Seeds: 42, 1042, 2042, 3042, 4042

Metrics:
- Gene diversity (mean pairwise cosine distance per gene category)
- Corpus diversity (genotype-level, measured from corpus instructions)
- Population dynamics (avg pop, births, deaths, max generation)
- Heterozygosity rate (diploid only)
- Allele retention (fraction of founding alleles still present)
- Extinction events

Outputs:
- eval9_diploid_results.json — Raw data
- eval9_diploid_selection.png — Comparison charts
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear.models import Dominance

from evolutionary_ecosystem.eval.harness import (
    GENE_CATEGORIES,
    SITUATION_NAMES,
    PopulationTracker,
    breed_bear_diploid,
    breed_deterministic,
    compute_corpus_diversity,
    compute_per_gene_diversity,
    corpus_diversity_mean,
    cosine_similarity,
    gene_diversity_mean,
    get_embedder,
    make_locus_registry,
    make_world,
    profile_to_vector,
    run_simulation,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


# ---------------------------------------------------------------------------
# Heterozygosity measurement
# ---------------------------------------------------------------------------

def population_heterozygosity(creatures: list, locus_key: str = "gene_category") -> float:
    """Mean heterozygosity across all creatures in the population.

    Heterozygosity = fraction of loci where allele-a and allele-b differ.
    For haploid corpora this returns 0.0 (no allele pairs).
    """
    if not creatures:
        return 0.0
    het_rates = []
    for c in creatures:
        if c.corpus is None:
            continue
        by_locus: dict[str, dict[str, str]] = {}
        for inst in c.corpus:
            locus = inst.metadata.get(locus_key)
            allele = inst.metadata.get("allele")
            if locus and allele in ("a", "b"):
                by_locus.setdefault(locus, {})[allele] = inst.content
        if not by_locus:
            het_rates.append(0.0)
            continue
        n_het = sum(1 for al in by_locus.values()
                    if "a" in al and "b" in al and al["a"] != al["b"])
        n_total = sum(1 for al in by_locus.values()
                      if "a" in al and "b" in al)
        het_rates.append(n_het / n_total if n_total > 0 else 0.0)
    return float(np.mean(het_rates)) if het_rates else 0.0


# ---------------------------------------------------------------------------
# Allele retention measurement
# ---------------------------------------------------------------------------

def _fast_gene_diversity(creatures: list) -> float:
    """Gene diversity using batch embedding for speed."""
    embedder = get_embedder()
    all_texts = []
    creature_indices = []
    for idx, c in enumerate(creatures):
        for cat in GENE_CATEGORIES:
            text = c.genes.get(cat, "")
            if text:
                all_texts.append(text)
                creature_indices.append((idx, cat))

    if not all_texts:
        return 0.0

    # Batch embed all texts at once
    all_embs = embedder.embed(all_texts)

    # Group by creature: mean embedding per creature
    from collections import defaultdict
    creature_embs: dict[int, list] = defaultdict(list)
    for i, (cidx, cat) in enumerate(creature_indices):
        creature_embs[cidx].append(all_embs[i])

    vecs = [np.mean(embs, axis=0) for embs in creature_embs.values() if embs]
    if len(vecs) < 2:
        return 0.0

    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
            distances.append(1.0 - sim)
    return round(float(np.mean(distances)), 4)


def _fast_corpus_diversity(creatures: list) -> float:
    """Corpus diversity using batch embedding for speed."""
    embedder = get_embedder()
    all_texts = []
    text_meta = []  # (creature_idx, locus)
    for idx, c in enumerate(creatures):
        if c.corpus is None:
            continue
        for inst in c.corpus:
            locus = inst.metadata.get("gene_category")
            if locus and locus in GENE_CATEGORIES:
                all_texts.append(inst.content)
                text_meta.append((idx, locus))

    if not all_texts:
        return 0.0

    all_embs = embedder.embed(all_texts)

    # Group by creature: mean embedding across all loci
    from collections import defaultdict
    creature_embs: dict[int, list] = defaultdict(list)
    for i, (cidx, locus) in enumerate(text_meta):
        creature_embs[cidx].append(all_embs[i])

    vecs = [np.mean(embs, axis=0) for embs in creature_embs.values() if embs]
    if len(vecs) < 2:
        return 0.0

    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
            distances.append(1.0 - sim)
    return round(float(np.mean(distances)), 4)


def _fast_allele_retention(creatures: list, founding_embs: dict, threshold: float = 0.8) -> float:
    """Allele retention using batch embedding for speed."""
    if not founding_embs or not creatures:
        return 0.0
    embedder = get_embedder()

    # Batch embed all current corpus texts
    all_texts = []
    for c in creatures:
        if c.corpus is None:
            continue
        for inst in c.corpus:
            all_texts.append(inst.content)

    if not all_texts:
        return 0.0

    current_matrix = embedder.embed(all_texts)
    current_norms = np.linalg.norm(current_matrix, axis=1, keepdims=True) + 1e-9

    retained = 0
    for key, founding_emb in founding_embs.items():
        f_norm = np.linalg.norm(founding_emb) + 1e-9
        sims = current_matrix @ founding_emb / (current_norms.flatten() * f_norm)
        if np.max(sims) > threshold:
            retained += 1
    return round(retained / len(founding_embs), 4)


def allele_retention(creatures: list, founding_embs: dict, embedder, threshold: float = 0.8) -> float:
    """Fraction of founding alleles still present (cosine sim > threshold) in current population.

    Optimized: pre-computes all current corpus embeddings once, then checks each founding allele.
    """
    if not founding_embs or not creatures:
        return 0.0

    # Pre-compute all current corpus instruction embeddings
    current_embs = []
    for c in creatures:
        if c.corpus is None:
            continue
        for inst in c.corpus:
            current_embs.append(embedder.embed_single(inst.content))

    if not current_embs:
        return 0.0

    current_matrix = np.array(current_embs)
    current_norms = np.linalg.norm(current_matrix, axis=1, keepdims=True) + 1e-9

    retained = 0
    for key, founding_emb in founding_embs.items():
        f_norm = np.linalg.norm(founding_emb) + 1e-9
        sims = current_matrix @ founding_emb / (current_norms.flatten() * f_norm)
        if np.max(sims) > threshold:
            retained += 1
    return retained / len(founding_embs)


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    breed_fn,
    seed: int,
    n_ticks: int,
    n_creatures: int,
    max_population: int = 30,
) -> dict:
    """Run one experimental condition and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Condition: {label} | seed={seed}")
    print(f"{'='*60}")

    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng, max_population=max_population)

    # Capture founding allele embeddings from corpus (batch for speed)
    embedder = get_embedder()
    founding_texts = []
    founding_keys = []
    for c in world.creatures.values():
        if c.corpus is not None:
            for inst in c.corpus:
                locus = inst.metadata.get("gene_category")
                if locus:
                    key = f"{c.name}:{locus}:{hash(inst.content) % 10000}"
                    founding_keys.append(key)
                    founding_texts.append(inst.content)
    founding_embs = {}
    if founding_texts:
        batch_embs = embedder.embed(founding_texts)
        for i, key in enumerate(founding_keys):
            founding_embs[key] = batch_embs[i]

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)

    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=100,
        breed_enabled=True,
        max_population=max_population,
        verbose=True,
        breed_fn=breed_fn,
    )

    # Compute metrics
    populations = [s.population for s in snapshots]
    avg_pop = float(np.mean(populations))
    pop_std = float(np.std(populations))
    extinctions = sum(1 for p in populations if p == 0)

    final_creatures = list(world.creatures.values())

    print("  Computing metrics...")

    # Gene diversity (from creature.genes dict — phenotype level)
    # Use batch embedding for speed
    gene_div = _fast_gene_diversity(final_creatures) if len(final_creatures) >= 2 else 0.0
    per_gene_div = {c: 0.0 for c in GENE_CATEGORIES}  # Detailed per-gene skipped for speed
    print(f"    gene_diversity done: {gene_div:.4f}")

    # Heterozygosity (fast — text comparison only, no embeddings)
    het_rate = population_heterozygosity(final_creatures)
    print(f"    heterozygosity done: {het_rate:.4f}")

    # Corpus diversity and allele retention: use batch embedding
    corpus_div = _fast_corpus_diversity(final_creatures) if len(final_creatures) >= 2 else 0.0
    print(f"    corpus_diversity done: {corpus_div:.4f}")

    retention = _fast_allele_retention(final_creatures, founding_embs) if founding_embs else 0.0
    print(f"    allele_retention done: {retention:.4f}")

    # Max generation
    max_gen = max((c.generation for c in final_creatures), default=0)

    # Mean behavior profile
    if final_creatures:
        avg_profile = {}
        for sit in SITUATION_NAMES:
            vals = [c.behavior_profile.strength(sit) if c.behavior_profile else 0.3
                    for c in final_creatures]
            avg_profile[sit] = round(float(np.mean(vals)), 4)
    else:
        avg_profile = {s: 0.0 for s in SITUATION_NAMES}

    metrics = {
        "condition": label,
        "avg_population": round(avg_pop, 2),
        "pop_std": round(pop_std, 2),
        "extinction_events": extinctions,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "max_generation": max_gen,
        "gene_diversity": gene_div,
        "per_gene_diversity": per_gene_div,
        "corpus_diversity": corpus_div,
        "per_corpus_diversity": {},
        "heterozygosity": round(het_rate, 4),
        "allele_retention": round(retention, 4),
        "final_population": len(final_creatures),
        "avg_behavior_profile": avg_profile,
        "population_timeline": populations,
    }

    print(f"\n  Avg pop: {avg_pop:.1f} ± {pop_std:.1f}")
    print(f"  Births: {world.total_births}, Deaths: {world.total_deaths}")
    print(f"  Max generation: {max_gen}")
    print(f"  Gene diversity (phenotype): {gene_div:.4f}")
    print(f"  Corpus diversity (genotype): {corpus_div:.4f}")
    print(f"  Heterozygosity: {het_rate:.4f}")
    print(f"  Allele retention: {retention:.4f}")
    print(f"  Extinctions: {extinctions}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_SEED = 42
    N_TICKS = 15000
    N_CREATURES = 30
    MAX_POP = 50
    N_TRIALS = 10

    print("=" * 60)
    print("EVAL 9: Diploid vs Haploid Under Selection Pressure")
    print("=" * 60)
    print(f"Ticks per trial: {N_TICKS}, Starting pop: {N_CREATURES}, Max pop: {MAX_POP}")
    print(f"Trials per condition: {N_TRIALS}")

    # Build breed functions for each condition
    haploid_registry = make_locus_registry(Dominance.HAPLOID)
    dominant_registry = make_locus_registry(Dominance.DOMINANT)
    codominant_registry = make_locus_registry(Dominance.CODOMINANT)

    conditions = [
        ("Haploid (text-splicing)", breed_deterministic),
        ("Diploid-dominant (BEAR)", partial(breed_bear_diploid, registry=dominant_registry)),
        ("Diploid-codominant (BEAR)", partial(breed_bear_diploid, registry=codominant_registry)),
    ]

    all_results: dict[str, list[dict]] = defaultdict(list)

    for label, breed_fn in conditions:
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial * 1000
            result = run_condition(
                f"{label} (trial {trial+1})",
                breed_fn=breed_fn,
                seed=seed,
                n_ticks=N_TICKS,
                n_creatures=N_CREATURES,
                max_population=MAX_POP,
            )
            result["trial"] = trial
            all_results[label].append(result)

    # Aggregate across trials
    summary = {}
    confidence_intervals = {}
    for label, trials in all_results.items():
        agg = {"condition": label, "n_trials": len(trials)}
        label_cis = {}
        for metric in ["avg_population", "total_births", "total_deaths",
                       "max_generation", "gene_diversity", "corpus_diversity",
                       "heterozygosity", "allele_retention", "extinction_events"]:
            vals = [t[metric] for t in trials]
            n = len(vals)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            agg[f"{metric}_mean"] = round(mean_val, 4)
            agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
            if n > 1 and std_val > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
                label_cis[metric] = [round(float(ci_low), 4), round(float(ci_high), 4)]
            else:
                label_cis[metric] = [round(mean_val, 4), round(mean_val, 4)]
        confidence_intervals[label] = label_cis
        summary[label] = agg

    # Statistical tests: compare each diploid condition against haploid
    condition_labels = [c[0] for c in conditions]
    haploid_label = condition_labels[0]
    stat_tests = {}

    for diploid_label in condition_labels[1:]:
        pair_key = f"{diploid_label} vs {haploid_label}"
        pair_tests = {}
        for metric in ["gene_diversity", "corpus_diversity", "max_generation",
                       "avg_population", "total_births", "heterozygosity", "allele_retention"]:
            hap_vals = [t[metric] for t in all_results[haploid_label]]
            dip_vals = [t[metric] for t in all_results[diploid_label]]

            # Handle case where all values are identical (e.g. heterozygosity = 0 for haploid)
            if len(set(hap_vals)) <= 1 and len(set(dip_vals)) <= 1:
                pair_tests[metric] = {
                    "haploid_mean": round(float(np.mean(hap_vals)), 4),
                    "diploid_mean": round(float(np.mean(dip_vals)), 4),
                    "t_p_value": 1.0 if hap_vals[0] == dip_vals[0] else 0.0,
                    "mannwhitney_p_value": None,
                    "both_normal": False,
                    "recommended_test": "N/A (constant values)",
                }
                continue

            t_stat, t_p = scipy_stats.ttest_ind(hap_vals, dip_vals)
            try:
                mw_stat, mw_p = scipy_stats.mannwhitneyu(hap_vals, dip_vals, alternative="two-sided")
                mw_p = round(float(mw_p), 4)
            except ValueError:
                mw_p = None

            # Shapiro-Wilk only with n >= 3
            if len(hap_vals) >= 3 and len(dip_vals) >= 3:
                _, sw_h_p = scipy_stats.shapiro(hap_vals)
                _, sw_d_p = scipy_stats.shapiro(dip_vals)
                both_normal = bool(sw_h_p > 0.05 and sw_d_p > 0.05)
            else:
                both_normal = False

            pair_tests[metric] = {
                "haploid_mean": round(float(np.mean(hap_vals)), 4),
                "diploid_mean": round(float(np.mean(dip_vals)), 4),
                "t_p_value": round(float(t_p), 4),
                "mannwhitney_p_value": mw_p,
                "both_normal": both_normal,
                "recommended_test": "t-test" if both_normal else "Mann-Whitney",
            }
        stat_tests[pair_key] = pair_tests

    # Print results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    for label, agg in summary.items():
        print(f"\n{label}:")
        for k, v in agg.items():
            if k not in ("condition", "n_trials"):
                print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    for pair_key, tests in stat_tests.items():
        print(f"\n{pair_key}:")
        print(f"  {'Metric':<20s} {'Haploid':>9s} {'Diploid':>9s} {'t p':>8s} {'M-W p':>8s} {'Test':>12s}")
        print(f"  {'-'*66}")
        for metric, test in tests.items():
            mw_str = f"{test['mannwhitney_p_value']:>8.4f}" if test['mannwhitney_p_value'] is not None else "     N/A"
            use_p = test['t_p_value'] if test['both_normal'] else (test['mannwhitney_p_value'] or test['t_p_value'])
            sig = " *" if use_p is not None and use_p < 0.05 else ""
            print(f"  {metric:<20s} {test['haploid_mean']:>9.4f} {test['diploid_mean']:>9.4f} "
                  f"{test['t_p_value']:>8.4f} {mw_str} {test['recommended_test']:>12s}{sig}")

    # Enhanced statistical tests (Welch's t-test, Mann-Whitney U, Cohen's d)
    def _run_statistical_tests(a, b, label_a, label_b, metric_name):
        """Print Welch's t-test, Mann-Whitney U, and Cohen's d for two arrays."""
        import numpy as np
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a) < 2 or len(b) < 2:
            print(f"    {metric_name}: insufficient samples (n_a={len(a)}, n_b={len(b)})")
            return
        from scipy.stats import ttest_ind, mannwhitneyu
        t_stat, p_val = ttest_ind(a, b, equal_var=False)
        try:
            u_stat, u_p = mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            u_stat, u_p = float('nan'), float('nan')
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
        d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"    {metric_name}: {label_a}={np.mean(a):.4f}\u00b1{np.std(a,ddof=1):.4f} vs "
              f"{label_b}={np.mean(b):.4f}\u00b1{np.std(b,ddof=1):.4f}")
        print(f"      Welch's t={t_stat:.3f}, p={p_val:.4f} {sig}, Cohen's d={d:.3f}")
        print(f"      Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}")

    print("\n" + "=" * 60)
    print("ENHANCED STATISTICAL TESTS (Welch's t, Mann-Whitney U, Cohen's d)")
    print("=" * 60)

    for diploid_label in condition_labels[1:]:
        print(f"\n  {diploid_label} vs {haploid_label}:")
        for metric in ["gene_diversity", "corpus_diversity", "allele_retention",
                        "heterozygosity", "avg_population", "max_generation"]:
            hap_vals = [t[metric] for t in all_results[haploid_label]]
            dip_vals = [t[metric] for t in all_results[diploid_label]]
            _run_statistical_tests(dip_vals, hap_vals, "diploid", "haploid", metric)

    # Generate chart
    try:
        _plot_results(all_results, summary, condition_labels)
    except Exception as e:
        print(f"\nChart generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results (strip timelines for file size)
    output = {
        "parameters": {
            "n_ticks": N_TICKS,
            "n_creatures": N_CREATURES,
            "max_population": MAX_POP,
            "n_trials": N_TRIALS,
            "base_seed": BASE_SEED,
        },
        "summary": summary,
        "confidence_intervals_95": confidence_intervals,
        "statistical_tests": stat_tests,
        "trials": {k: [{kk: vv for kk, vv in t.items() if kk != "population_timeline"}
                        for t in v] for k, v in all_results.items()},
    }

    results_path = OUT_DIR / "eval9_diploid_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(all_results, summary, condition_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    # Panel 1: Population over time (trial 1 each)
    ax = axes[0][0]
    for idx, label in enumerate(condition_labels):
        trial = all_results[label][0]
        pops = trial["population_timeline"]
        ticks = list(range(0, len(pops) * 100, 100))
        ax.plot(ticks, pops, label=label.split("(")[0].strip(), color=colors[idx],
                linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Population")
    ax.set_title("Population Over Time (Trial 1)")
    ax.legend(fontsize=7)

    # Panel 2: Gene diversity (phenotype) — bar chart
    ax = axes[0][1]
    x = np.arange(len(condition_labels))
    means = [summary[c]["gene_diversity_mean"] for c in condition_labels]
    stds = [summary[c]["gene_diversity_std"] for c in condition_labels]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in condition_labels], fontsize=8)
    ax.set_ylabel("Mean Cosine Distance")
    ax.set_title("Phenotype Diversity (gene dict)")

    # Panel 3: Corpus diversity (genotype)
    ax = axes[0][2]
    means = [summary[c]["corpus_diversity_mean"] for c in condition_labels]
    stds = [summary[c]["corpus_diversity_std"] for c in condition_labels]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in condition_labels], fontsize=8)
    ax.set_ylabel("Mean Cosine Distance")
    ax.set_title("Genotype Diversity (corpus)")

    # Panel 4: Heterozygosity
    ax = axes[1][0]
    means = [summary[c]["heterozygosity_mean"] for c in condition_labels]
    stds = [summary[c]["heterozygosity_std"] for c in condition_labels]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in condition_labels], fontsize=8)
    ax.set_ylabel("Heterozygosity Rate")
    ax.set_title("Heterozygosity")

    # Panel 5: Allele retention
    ax = axes[1][1]
    means = [summary[c]["allele_retention_mean"] for c in condition_labels]
    stds = [summary[c]["allele_retention_std"] for c in condition_labels]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in condition_labels], fontsize=8)
    ax.set_ylabel("Retention Rate")
    ax.set_title("Founding Allele Retention")

    # Panel 6: Births and max generation
    ax = axes[1][2]
    width = 0.25
    births = [summary[c]["total_births_mean"] for c in condition_labels]
    births_std = [summary[c]["total_births_std"] for c in condition_labels]
    gens = [summary[c]["max_generation_mean"] for c in condition_labels]
    gens_std = [summary[c]["max_generation_std"] for c in condition_labels]
    ax.bar(x - width/2, births, width, label="Births", color=[c for c in colors], alpha=0.6,
           yerr=births_std, capsize=3)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, gens, width, label="Max Gen", color=[c for c in colors], alpha=0.3,
            yerr=gens_std, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in condition_labels], fontsize=8)
    ax.set_ylabel("Total Births")
    ax2.set_ylabel("Max Generation")
    ax.set_title("Births & Generational Depth")

    plt.suptitle("Eval 9: Diploid vs Haploid Under Selection Pressure", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = OUT_DIR / "eval9_diploid_selection.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # Also save to paper/figures
    paper_fig = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
    if paper_fig.exists():
        import shutil
        shutil.copy2(chart_path, paper_fig / "eval9_diploid_selection.png")
        print(f"Chart copied to {paper_fig / 'eval9_diploid_selection.png'}")


if __name__ == "__main__":
    main()
