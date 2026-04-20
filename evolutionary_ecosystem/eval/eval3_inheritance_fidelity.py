#!/usr/bin/env python3
"""Evaluation 3: Cross-generational inheritance fidelity.

Measures whether BEAR breeding actually transmits behavioral tendencies
from parents to offspring by computing:
1. Parent-offspring behavior profile cosine similarity
2. Random-pair behavior profile cosine similarity (baseline)
3. Gene embedding cosine similarity (parent-child vs random pairs)
4. Multi-generation lineage chains over 5 successive breeding events

If inheritance works, parent-offspring similarity should be significantly
higher than random-pair similarity.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — deterministic text splicing)

Parameters:
- 10 parent pairs, 5 offspring per pair (50 offspring total)
- 3 lineage chains over 5 generations
- Seeds: 42, 1042, 2042, 3042, 4042

Outputs:
- eval3_results.json     — Raw data
- eval3_inheritance.png  — Comparison charts
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    _NAMES,
    breed_bear_diploid,
    cosine_similarity,
    get_embedder,
    make_creature,
    profile_to_vector,
    stats_to_vector,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

SEEDS = [42, 1042, 2042, 3042, 4042]


def gene_embedding(creature, embedder) -> np.ndarray:
    """Compute mean gene embedding for a creature."""
    vecs = [embedder.embed_single(t) for t in creature.genes.values()]
    return np.mean(vecs, axis=0)


def per_gene_embeddings(creature, embedder) -> dict[str, np.ndarray]:
    """Return {category: embedding} for each gene."""
    return {cat: embedder.embed_single(text)
            for cat, text in creature.genes.items()}


def per_gene_cosine_sim(embs_a: dict[str, np.ndarray],
                        embs_b: dict[str, np.ndarray]) -> float:
    """Mean cosine similarity across shared gene categories."""
    shared = set(embs_a) & set(embs_b)
    if not shared:
        return 0.0
    sims = []
    for cat in shared:
        a, b = embs_a[cat], embs_b[cat]
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(dot / (norm + 1e-9))
    return float(np.mean(sims))


def hausdorff_gene_distance(embs_a: dict[str, np.ndarray],
                            embs_b: dict[str, np.ndarray]) -> float:
    """Hausdorff distance over per-gene embeddings.

    max(max_a min_b d(a,b), max_b min_a d(b,a)) where d is cosine distance.
    """
    vecs_a = list(embs_a.values())
    vecs_b = list(embs_b.values())
    if not vecs_a or not vecs_b:
        return 1.0

    def _cos_dist(u: np.ndarray, v: np.ndarray) -> float:
        dot = float(np.dot(u, v))
        norm = float(np.linalg.norm(u) * np.linalg.norm(v))
        return 1.0 - dot / (norm + 1e-9)

    # max over a of min over b
    sup_a = max(min(_cos_dist(a, b) for b in vecs_b) for a in vecs_a)
    sup_b = max(min(_cos_dist(b, a) for a in vecs_a) for b in vecs_b)
    return max(sup_a, sup_b)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def confidence_interval_95(data: list[float]) -> tuple[float, float]:
    """Compute 95% CI for the mean using the t-distribution."""
    n = len(data)
    if n < 2:
        m = float(np.mean(data)) if data else 0.0
        return (m, m)
    m = float(np.mean(data))
    se = float(stats.sem(data))
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return (float(lo), float(hi))


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size (pooled std denominator)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def run_significance_tests(po_vals: list[float],
                           rand_vals: list[float],
                           label: str) -> dict:
    """Run normality, t-test, Mann-Whitney, and effect-size tests."""
    result: dict = {"metric": label}

    # Shapiro-Wilk normality (requires n >= 3)
    if len(po_vals) >= 3:
        sw_po = stats.shapiro(po_vals)
        result["shapiro_wilk_parent_offspring"] = {
            "statistic": round(float(sw_po.statistic), 6),
            "p_value": round(float(sw_po.pvalue), 6),
        }
    else:
        result["shapiro_wilk_parent_offspring"] = {"note": "n < 3, skipped"}

    if len(rand_vals) >= 3:
        sw_rand = stats.shapiro(rand_vals)
        result["shapiro_wilk_random"] = {
            "statistic": round(float(sw_rand.statistic), 6),
            "p_value": round(float(sw_rand.pvalue), 6),
        }
    else:
        result["shapiro_wilk_random"] = {"note": "n < 3, skipped"}

    # Welch's t-test (unequal variance)
    if len(po_vals) >= 2 and len(rand_vals) >= 2:
        t_stat, t_p = stats.ttest_ind(po_vals, rand_vals, equal_var=False)
        result["welch_t_test"] = {
            "t_statistic": round(float(t_stat), 6),
            "p_value": round(float(t_p), 6),
        }
    else:
        result["welch_t_test"] = {"note": "insufficient samples"}

    # Mann-Whitney U
    if len(po_vals) >= 1 and len(rand_vals) >= 1:
        u_stat, u_p = stats.mannwhitneyu(
            po_vals, rand_vals, alternative="two-sided"
        )
        result["mann_whitney_u"] = {
            "U_statistic": round(float(u_stat), 6),
            "p_value": round(float(u_p), 6),
        }
    else:
        result["mann_whitney_u"] = {"note": "insufficient samples"}

    # Cohen's d
    result["cohens_d"] = round(cohens_d(po_vals, rand_vals), 6)

    return result


def run_single_seed(seed: int, embedder, N_PARENT_PAIRS: int,
                    N_OFFSPRING_PER_PAIR: int, N_GENERATIONS: int):
    """Run the full experiment for a single seed. Returns raw data dict."""
    rng = random.Random(seed)

    # Create parent creatures from diverse gene bank
    parents: list = []
    for i in range(min(len(GENE_BANK), N_PARENT_PAIRS * 2)):
        genes = GENE_BANK[i % len(GENE_BANK)]
        name = _NAMES[i]
        c = make_creature(f"s{seed}_p{i}", genes, name, rng, generation=0)
        parents.append(c)

    # --- Experiment 1: Single-generation parent-offspring similarity ---
    parent_offspring_behavior_sims: list[float] = []
    parent_offspring_gene_sims: list[float] = []
    parent_offspring_stat_sims: list[float] = []
    parent_offspring_pergene_sims: list[float] = []
    parent_offspring_hausdorff: list[float] = []

    all_offspring: list = []
    pairings: list[dict] = []

    cid_counter = 0
    for pair_idx in range(N_PARENT_PAIRS):
        pa = parents[pair_idx % len(parents)]
        pb = parents[(pair_idx + 1) % len(parents)]

        pair_data = {
            "parent_a": pa.name,
            "parent_b": pb.name,
            "offspring": [],
        }

        pa_profile = profile_to_vector(pa.behavior_profile)
        pb_profile = profile_to_vector(pb.behavior_profile)
        pa_gene_emb = gene_embedding(pa, embedder)
        pb_gene_emb = gene_embedding(pb, embedder)
        midparent_profile = [(a + b) / 2 for a, b in zip(pa_profile, pb_profile)]
        midparent_gene_emb = (pa_gene_emb + pb_gene_emb) / 2

        # Per-gene embeddings for midparent
        pa_per = per_gene_embeddings(pa, embedder)
        pb_per = per_gene_embeddings(pb, embedder)
        midparent_per = {
            cat: (pa_per[cat] + pb_per[cat]) / 2
            for cat in set(pa_per) & set(pb_per)
        }

        for j in range(N_OFFSPRING_PER_PAIR):
            cid_counter += 1
            child_name = f"Child_{pair_idx}_{j}"
            child = breed_bear_diploid(pa, pb, f"s{seed}_c{cid_counter}", child_name, rng)
            all_offspring.append(child)

            child_profile = profile_to_vector(child.behavior_profile)
            child_gene_emb = gene_embedding(child, embedder)
            child_stats = stats_to_vector(child.stats)
            child_per = per_gene_embeddings(child, embedder)

            # Similarity to midparent
            beh_sim = cosine_similarity(child_profile, midparent_profile)
            gene_sim = float(np.dot(child_gene_emb, midparent_gene_emb) /
                           (np.linalg.norm(child_gene_emb) * np.linalg.norm(midparent_gene_emb) + 1e-9))
            pergene_sim = per_gene_cosine_sim(child_per, midparent_per)
            hausd = hausdorff_gene_distance(child_per, midparent_per)

            parent_offspring_behavior_sims.append(beh_sim)
            parent_offspring_gene_sims.append(gene_sim)
            parent_offspring_pergene_sims.append(pergene_sim)
            parent_offspring_hausdorff.append(hausd)

            # Stat similarity to midparent stats
            pa_stats = stats_to_vector(pa.stats)
            pb_stats = stats_to_vector(pb.stats)
            midparent_stats = [(a + b) / 2 for a, b in zip(pa_stats, pb_stats)]
            stat_sim = cosine_similarity(child_stats, midparent_stats)
            parent_offspring_stat_sims.append(stat_sim)

            pair_data["offspring"].append({
                "name": child_name,
                "behavior_sim_to_midparent": round(beh_sim, 4),
                "gene_sim_to_midparent": round(gene_sim, 4),
                "pergene_sim_to_midparent": round(pergene_sim, 4),
                "hausdorff_to_midparent": round(hausd, 4),
                "stat_sim_to_midparent": round(stat_sim, 4),
                "behavior_profile": {SITUATION_NAMES[i]: round(child_profile[i], 4)
                                     for i in range(len(SITUATION_NAMES))},
            })

        pairings.append(pair_data)

    # --- Baseline: Random-pair similarity ---
    all_creatures = parents + all_offspring
    random_behavior_sims: list[float] = []
    random_gene_sims: list[float] = []
    random_stat_sims: list[float] = []
    random_pergene_sims: list[float] = []
    random_hausdorff: list[float] = []

    n_random = len(parent_offspring_behavior_sims)
    for _ in range(n_random):
        i, j = rng.sample(range(len(all_creatures)), 2)
        a, b = all_creatures[i], all_creatures[j]
        random_behavior_sims.append(cosine_similarity(
            profile_to_vector(a.behavior_profile),
            profile_to_vector(b.behavior_profile),
        ))
        a_emb = gene_embedding(a, embedder)
        b_emb = gene_embedding(b, embedder)
        random_gene_sims.append(float(np.dot(a_emb, b_emb) /
                                     (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-9)))
        random_stat_sims.append(cosine_similarity(
            stats_to_vector(a.stats), stats_to_vector(b.stats),
        ))
        a_per = per_gene_embeddings(a, embedder)
        b_per = per_gene_embeddings(b, embedder)
        random_pergene_sims.append(per_gene_cosine_sim(a_per, b_per))
        random_hausdorff.append(hausdorff_gene_distance(a_per, b_per))

    # --- Experiment 2: Multi-generation chains ---
    chain_results: list[dict] = []
    for chain_idx in range(3):  # 3 lineage chains
        pa = parents[chain_idx * 2 % len(parents)]
        pb = parents[(chain_idx * 2 + 1) % len(parents)]

        gen0_profile = [(a + b) / 2 for a, b in zip(
            profile_to_vector(pa.behavior_profile),
            profile_to_vector(pb.behavior_profile),
        )]
        # Gen0 per-gene midparent embeddings
        pa_per0 = per_gene_embeddings(pa, embedder)
        pb_per0 = per_gene_embeddings(pb, embedder)
        gen0_per = {
            cat: (pa_per0[cat] + pb_per0[cat]) / 2
            for cat in set(pa_per0) & set(pb_per0)
        }

        current_a, current_b = pa, pb
        gen_profiles: list[dict] = [{
            "generation": 0,
            "profile": {SITUATION_NAMES[i]: round(gen0_profile[i], 4)
                       for i in range(len(SITUATION_NAMES))},
            "sim_to_gen0": 1.0,
            "pergene_sim_to_gen0": 1.0,
            "hausdorff_to_gen0": 0.0,
        }]

        for gen in range(1, N_GENERATIONS + 1):
            cid_counter += 1
            child = breed_bear_diploid(current_a, current_b,
                                       f"s{seed}_chain{chain_idx}_g{gen}",
                                       f"Chain{chain_idx}_Gen{gen}", rng)
            child_profile = profile_to_vector(child.behavior_profile)
            sim_to_gen0 = cosine_similarity(child_profile, gen0_profile)

            child_per = per_gene_embeddings(child, embedder)
            pergene_to_gen0 = per_gene_cosine_sim(child_per, gen0_per)
            hausd_to_gen0 = hausdorff_gene_distance(child_per, gen0_per)

            gen_profiles.append({
                "generation": gen,
                "profile": {SITUATION_NAMES[i]: round(child_profile[i], 4)
                           for i in range(len(SITUATION_NAMES))},
                "sim_to_gen0": round(sim_to_gen0, 4),
                "pergene_sim_to_gen0": round(pergene_to_gen0, 4),
                "hausdorff_to_gen0": round(hausd_to_gen0, 4),
            })

            # For next generation, breed child with a random parent
            other = parents[rng.randint(0, len(parents) - 1)]
            current_a, current_b = child, other

        chain_results.append({
            "chain": chain_idx,
            "initial_parents": [pa.name, pb.name],
            "generations": gen_profiles,
        })

    return {
        "seed": seed,
        "parent_offspring_behavior_sims": parent_offspring_behavior_sims,
        "parent_offspring_gene_sims": parent_offspring_gene_sims,
        "parent_offspring_stat_sims": parent_offspring_stat_sims,
        "parent_offspring_pergene_sims": parent_offspring_pergene_sims,
        "parent_offspring_hausdorff": parent_offspring_hausdorff,
        "random_behavior_sims": random_behavior_sims,
        "random_gene_sims": random_gene_sims,
        "random_stat_sims": random_stat_sims,
        "random_pergene_sims": random_pergene_sims,
        "random_hausdorff": random_hausdorff,
        "pairings": pairings,
        "chain_results": chain_results,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_PARENT_PAIRS = 10  # Number of breeding pairs
    N_OFFSPRING_PER_PAIR = 5  # Offspring per pair
    N_GENERATIONS = 5  # Multi-generation chains

    print("=" * 60)
    print("EVAL 3: Cross-Generational Inheritance Fidelity")
    print("=" * 60)
    print(f"Parent pairs: {N_PARENT_PAIRS}")
    print(f"Offspring per pair: {N_OFFSPRING_PER_PAIR}")
    print(f"Generational chains: {N_GENERATIONS}")
    print(f"Seeds: {SEEDS}")
    print()

    embedder = get_embedder()

    # ------------------------------------------------------------------
    # Run experiment for each seed and collect per-seed data
    # ------------------------------------------------------------------
    all_seed_data: list[dict] = []
    # Aggregated across all seeds
    agg_po_beh: list[float] = []
    agg_po_gene: list[float] = []
    agg_po_stat: list[float] = []
    agg_po_pergene: list[float] = []
    agg_po_hausd: list[float] = []
    agg_rand_beh: list[float] = []
    agg_rand_gene: list[float] = []
    agg_rand_stat: list[float] = []
    agg_rand_pergene: list[float] = []
    agg_rand_hausd: list[float] = []
    # Collect all chain results across seeds
    all_chain_results: list[dict] = []

    for seed in SEEDS:
        print(f"\n--- Running seed {seed} ---")
        seed_data = run_single_seed(
            seed, embedder, N_PARENT_PAIRS, N_OFFSPRING_PER_PAIR, N_GENERATIONS
        )
        all_seed_data.append(seed_data)

        agg_po_beh.extend(seed_data["parent_offspring_behavior_sims"])
        agg_po_gene.extend(seed_data["parent_offspring_gene_sims"])
        agg_po_stat.extend(seed_data["parent_offspring_stat_sims"])
        agg_po_pergene.extend(seed_data["parent_offspring_pergene_sims"])
        agg_po_hausd.extend(seed_data["parent_offspring_hausdorff"])
        agg_rand_beh.extend(seed_data["random_behavior_sims"])
        agg_rand_gene.extend(seed_data["random_gene_sims"])
        agg_rand_stat.extend(seed_data["random_stat_sims"])
        agg_rand_pergene.extend(seed_data["random_pergene_sims"])
        agg_rand_hausd.extend(seed_data["random_hausdorff"])
        all_chain_results.extend(seed_data["chain_results"])

    # For backward-compatible pairings output, use seed=42 (first seed)
    pairings = all_seed_data[0]["pairings"]
    # Use seed=42 chain results for the main multi_generation_chains entry
    chain_results = all_seed_data[0]["chain_results"]

    # ------------------------------------------------------------------
    # Aggregated summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (aggregated across all seeds)")
    print("=" * 60)

    po_beh_mean = float(np.mean(agg_po_beh))
    po_beh_std = float(np.std(agg_po_beh))
    po_beh_ci = confidence_interval_95(agg_po_beh)
    rand_beh_mean = float(np.mean(agg_rand_beh))
    rand_beh_std = float(np.std(agg_rand_beh))
    rand_beh_ci = confidence_interval_95(agg_rand_beh)

    po_gene_mean = float(np.mean(agg_po_gene))
    po_gene_std = float(np.std(agg_po_gene))
    po_gene_ci = confidence_interval_95(agg_po_gene)
    rand_gene_mean = float(np.mean(agg_rand_gene))
    rand_gene_std = float(np.std(agg_rand_gene))
    rand_gene_ci = confidence_interval_95(agg_rand_gene)

    po_stat_mean = float(np.mean(agg_po_stat))
    po_stat_std = float(np.std(agg_po_stat))
    po_stat_ci = confidence_interval_95(agg_po_stat)
    rand_stat_mean = float(np.mean(agg_rand_stat))
    rand_stat_std = float(np.std(agg_rand_stat))
    rand_stat_ci = confidence_interval_95(agg_rand_stat)

    po_pergene_mean = float(np.mean(agg_po_pergene))
    po_pergene_std = float(np.std(agg_po_pergene))
    po_pergene_ci = confidence_interval_95(agg_po_pergene)
    rand_pergene_mean = float(np.mean(agg_rand_pergene))
    rand_pergene_std = float(np.std(agg_rand_pergene))
    rand_pergene_ci = confidence_interval_95(agg_rand_pergene)

    po_hausd_mean = float(np.mean(agg_po_hausd))
    po_hausd_std = float(np.std(agg_po_hausd))
    po_hausd_ci = confidence_interval_95(agg_po_hausd)
    rand_hausd_mean = float(np.mean(agg_rand_hausd))
    rand_hausd_std = float(np.std(agg_rand_hausd))
    rand_hausd_ci = confidence_interval_95(agg_rand_hausd)

    def _fmt_ci(ci):
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    print(f"\nBehavior Profile Similarity (to midparent):")
    print(f"  Parent-offspring: {po_beh_mean:.4f} +/- {po_beh_std:.4f}  95% CI {_fmt_ci(po_beh_ci)}")
    print(f"  Random pairs:    {rand_beh_mean:.4f} +/- {rand_beh_std:.4f}  95% CI {_fmt_ci(rand_beh_ci)}")
    print(f"  Advantage:       {po_beh_mean - rand_beh_mean:+.4f}")

    print(f"\nMean Gene Embedding Similarity (to midparent):")
    print(f"  Parent-offspring: {po_gene_mean:.4f} +/- {po_gene_std:.4f}  95% CI {_fmt_ci(po_gene_ci)}")
    print(f"  Random pairs:    {rand_gene_mean:.4f} +/- {rand_gene_std:.4f}  95% CI {_fmt_ci(rand_gene_ci)}")
    print(f"  Advantage:       {po_gene_mean - rand_gene_mean:+.4f}")

    print(f"\nPer-Gene Cosine Similarity (to midparent):")
    print(f"  Parent-offspring: {po_pergene_mean:.4f} +/- {po_pergene_std:.4f}  95% CI {_fmt_ci(po_pergene_ci)}")
    print(f"  Random pairs:    {rand_pergene_mean:.4f} +/- {rand_pergene_std:.4f}  95% CI {_fmt_ci(rand_pergene_ci)}")
    print(f"  Advantage:       {po_pergene_mean - rand_pergene_mean:+.4f}")

    print(f"\nHausdorff Gene Distance (to midparent):")
    print(f"  Parent-offspring: {po_hausd_mean:.4f} +/- {po_hausd_std:.4f}  95% CI {_fmt_ci(po_hausd_ci)}")
    print(f"  Random pairs:    {rand_hausd_mean:.4f} +/- {rand_hausd_std:.4f}  95% CI {_fmt_ci(rand_hausd_ci)}")
    print(f"  Advantage:       {po_hausd_mean - rand_hausd_mean:+.4f}  (lower = closer)")

    print(f"\nEntity Stats Similarity (to midparent):")
    print(f"  Parent-offspring: {po_stat_mean:.4f} +/- {po_stat_std:.4f}  95% CI {_fmt_ci(po_stat_ci)}")
    print(f"  Random pairs:    {rand_stat_mean:.4f} +/- {rand_stat_std:.4f}  95% CI {_fmt_ci(rand_stat_ci)}")

    # ------------------------------------------------------------------
    # Statistical significance tests (aggregated data)
    # ------------------------------------------------------------------
    stat_tests = []
    stat_tests.append(run_significance_tests(agg_po_beh, agg_rand_beh, "behavior"))
    stat_tests.append(run_significance_tests(agg_po_gene, agg_rand_gene, "gene_embedding"))
    stat_tests.append(run_significance_tests(agg_po_stat, agg_rand_stat, "entity_stats"))
    stat_tests.append(run_significance_tests(agg_po_pergene, agg_rand_pergene, "per_gene_cosine"))
    stat_tests.append(run_significance_tests(agg_po_hausd, agg_rand_hausd, "hausdorff_distance"))

    print("\n--- Statistical Significance Tests ---")
    for t in stat_tests:
        label = t["metric"]
        welch = t.get("welch_t_test", {})
        mwu = t.get("mann_whitney_u", {})
        cd = t.get("cohens_d", 0)
        print(f"\n  {label}:")
        if "t_statistic" in welch:
            print(f"    Welch's t:  t={welch['t_statistic']:.4f}, p={welch['p_value']:.6f}")
        if "U_statistic" in mwu:
            print(f"    Mann-Whitney U: U={mwu['U_statistic']:.1f}, p={mwu['p_value']:.6f}")
        print(f"    Cohen's d:  {cd:.4f}")

    # ------------------------------------------------------------------
    # Multi-generation decay with 95% CI across seeds
    # ------------------------------------------------------------------
    print(f"\nMulti-generation decay (behavior profile) — seed 42:")
    for chain in chain_results:
        sims = [g["sim_to_gen0"] for g in chain["generations"]]
        print(f"  Chain {chain['chain']}: " + " -> ".join(f"{s:.3f}" for s in sims))

    print(f"\nMulti-generation decay (per-gene cosine similarity) — seed 42:")
    for chain in chain_results:
        sims = [g["pergene_sim_to_gen0"] for g in chain["generations"]]
        print(f"  Chain {chain['chain']}: " + " -> ".join(f"{s:.3f}" for s in sims))

    print(f"\nMulti-generation decay (Hausdorff gene distance) — seed 42:")
    for chain in chain_results:
        dists = [g["hausdorff_to_gen0"] for g in chain["generations"]]
        print(f"  Chain {chain['chain']}: " + " -> ".join(f"{d:.3f}" for d in dists))

    # Aggregate multi-gen decay across all seeds for CI
    # Each seed has 3 chains x (N_GENERATIONS+1) generations.
    # Compute mean sim_to_gen0 per generation across all chains and seeds, with CI.
    n_gens = N_GENERATIONS + 1
    gen_sims_by_gen: dict[int, list[float]] = defaultdict(list)
    gen_pergene_by_gen: dict[int, list[float]] = defaultdict(list)
    gen_hausd_by_gen: dict[int, list[float]] = defaultdict(list)
    for chain in all_chain_results:
        for g in chain["generations"]:
            gen_sims_by_gen[g["generation"]].append(g["sim_to_gen0"])
            gen_pergene_by_gen[g["generation"]].append(g["pergene_sim_to_gen0"])
            gen_hausd_by_gen[g["generation"]].append(g["hausdorff_to_gen0"])

    lineage_decay_summary: list[dict] = []
    print(f"\nMulti-generation decay — aggregated across {len(SEEDS)} seeds, "
          f"{len(all_chain_results)} chains:")
    print(f"  {'Gen':>4}  {'Beh sim':>10}  {'95% CI':>22}  "
          f"{'PerGene':>10}  {'95% CI':>22}  "
          f"{'Hausdorff':>10}  {'95% CI':>22}")
    for gen_i in range(n_gens):
        beh_vals = gen_sims_by_gen[gen_i]
        pg_vals = gen_pergene_by_gen[gen_i]
        hd_vals = gen_hausd_by_gen[gen_i]
        beh_m, beh_ci = float(np.mean(beh_vals)), confidence_interval_95(beh_vals)
        pg_m, pg_ci = float(np.mean(pg_vals)), confidence_interval_95(pg_vals)
        hd_m, hd_ci = float(np.mean(hd_vals)), confidence_interval_95(hd_vals)
        print(f"  {gen_i:>4}  {beh_m:>10.4f}  {_fmt_ci(beh_ci):>22}  "
              f"{pg_m:>10.4f}  {_fmt_ci(pg_ci):>22}  "
              f"{hd_m:>10.4f}  {_fmt_ci(hd_ci):>22}")
        lineage_decay_summary.append({
            "generation": gen_i,
            "behavior_sim_mean": round(beh_m, 4),
            "behavior_sim_ci_95": [round(beh_ci[0], 4), round(beh_ci[1], 4)],
            "pergene_sim_mean": round(pg_m, 4),
            "pergene_sim_ci_95": [round(pg_ci[0], 4), round(pg_ci[1], 4)],
            "hausdorff_mean": round(hd_m, 4),
            "hausdorff_ci_95": [round(hd_ci[0], 4), round(hd_ci[1], 4)],
        })

    # ------------------------------------------------------------------
    # Build per-seed breakdown for JSON
    # ------------------------------------------------------------------
    per_seed_breakdown: list[dict] = []
    for sd in all_seed_data:
        s = sd["seed"]
        per_seed_breakdown.append({
            "seed": s,
            "parent_offspring_similarity": {
                "behavior": {
                    "mean": round(float(np.mean(sd["parent_offspring_behavior_sims"])), 4),
                    "std": round(float(np.std(sd["parent_offspring_behavior_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["parent_offspring_behavior_sims"])],
                    "values": [round(v, 4) for v in sd["parent_offspring_behavior_sims"]],
                },
                "gene_embedding": {
                    "mean": round(float(np.mean(sd["parent_offspring_gene_sims"])), 4),
                    "std": round(float(np.std(sd["parent_offspring_gene_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["parent_offspring_gene_sims"])],
                    "values": [round(v, 4) for v in sd["parent_offspring_gene_sims"]],
                },
                "entity_stats": {
                    "mean": round(float(np.mean(sd["parent_offspring_stat_sims"])), 4),
                    "std": round(float(np.std(sd["parent_offspring_stat_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["parent_offspring_stat_sims"])],
                    "values": [round(v, 4) for v in sd["parent_offspring_stat_sims"]],
                },
            },
            "random_pair_baseline": {
                "behavior": {
                    "mean": round(float(np.mean(sd["random_behavior_sims"])), 4),
                    "std": round(float(np.std(sd["random_behavior_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["random_behavior_sims"])],
                    "values": [round(v, 4) for v in sd["random_behavior_sims"]],
                },
                "gene_embedding": {
                    "mean": round(float(np.mean(sd["random_gene_sims"])), 4),
                    "std": round(float(np.std(sd["random_gene_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["random_gene_sims"])],
                    "values": [round(v, 4) for v in sd["random_gene_sims"]],
                },
                "entity_stats": {
                    "mean": round(float(np.mean(sd["random_stat_sims"])), 4),
                    "std": round(float(np.std(sd["random_stat_sims"])), 4),
                    "ci_95": [round(v, 4) for v in confidence_interval_95(sd["random_stat_sims"])],
                    "values": [round(v, 4) for v in sd["random_stat_sims"]],
                },
            },
            "pairings": sd["pairings"],
            "multi_generation_chains": sd["chain_results"],
        })

    # ------------------------------------------------------------------
    # Save results (backward compatible + new fields)
    # ------------------------------------------------------------------
    results = {
        "parameters": {
            "seeds": SEEDS,
            "seed": SEEDS[0],  # backward compat
            "n_seeds": len(SEEDS),
            "n_parent_pairs": N_PARENT_PAIRS,
            "n_offspring_per_pair": N_OFFSPRING_PER_PAIR,
            "n_generations": N_GENERATIONS,
        },
        "parent_offspring_similarity": {
            "behavior": {
                "mean": round(po_beh_mean, 4),
                "std": round(po_beh_std, 4),
                "ci_95": [round(po_beh_ci[0], 4), round(po_beh_ci[1], 4)],
                "values": [round(v, 4) for v in agg_po_beh],
            },
            "gene_embedding": {
                "mean": round(po_gene_mean, 4),
                "std": round(po_gene_std, 4),
                "ci_95": [round(po_gene_ci[0], 4), round(po_gene_ci[1], 4)],
                "values": [round(v, 4) for v in agg_po_gene],
            },
            "entity_stats": {
                "mean": round(po_stat_mean, 4),
                "std": round(po_stat_std, 4),
                "ci_95": [round(po_stat_ci[0], 4), round(po_stat_ci[1], 4)],
                "values": [round(v, 4) for v in agg_po_stat],
            },
        },
        "random_pair_baseline": {
            "behavior": {
                "mean": round(rand_beh_mean, 4),
                "std": round(rand_beh_std, 4),
                "ci_95": [round(rand_beh_ci[0], 4), round(rand_beh_ci[1], 4)],
            },
            "gene_embedding": {
                "mean": round(rand_gene_mean, 4),
                "std": round(rand_gene_std, 4),
                "ci_95": [round(rand_gene_ci[0], 4), round(rand_gene_ci[1], 4)],
            },
            "entity_stats": {
                "mean": round(rand_stat_mean, 4),
                "std": round(rand_stat_std, 4),
                "ci_95": [round(rand_stat_ci[0], 4), round(rand_stat_ci[1], 4)],
            },
        },
        "statistical_tests": stat_tests,
        "pairings": pairings,
        "multi_generation_chains": chain_results,
        "lineage_decay_summary": lineage_decay_summary,
        "per_seed_breakdown": per_seed_breakdown,
    }

    results_path = OUT_DIR / "eval3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_inheritance(results)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_inheritance(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Parent-offspring vs random behavior similarity
    ax1 = axes[0][0]
    po_vals = results["parent_offspring_similarity"]["behavior"]["values"]
    rand_mean = results["random_pair_baseline"]["behavior"]["mean"]
    ax1.hist(po_vals, bins=15, alpha=0.7, color="#4CAF50", label="Parent-Offspring")
    ax1.axvline(rand_mean, color="#F44336", linestyle="--", linewidth=2,
                label=f"Random baseline ({rand_mean:.3f})")
    po_mean = results["parent_offspring_similarity"]["behavior"]["mean"]
    ax1.axvline(po_mean, color="#4CAF50", linestyle="-", linewidth=2,
                label=f"P-O mean ({po_mean:.3f})")
    # CI shading
    ci = results["parent_offspring_similarity"]["behavior"]["ci_95"]
    ax1.axvspan(ci[0], ci[1], alpha=0.15, color="#4CAF50", label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("Behavior Profile: Parent-Offspring vs Random")
    ax1.legend(fontsize=8)

    # Panel 2: Gene embedding similarity
    ax2 = axes[0][1]
    po_gene = results["parent_offspring_similarity"]["gene_embedding"]["values"]
    rand_gene = results["random_pair_baseline"]["gene_embedding"]["mean"]
    ax2.hist(po_gene, bins=15, alpha=0.7, color="#2196F3", label="Parent-Offspring")
    ax2.axvline(rand_gene, color="#F44336", linestyle="--", linewidth=2,
                label=f"Random baseline ({rand_gene:.3f})")
    po_gene_mean = results["parent_offspring_similarity"]["gene_embedding"]["mean"]
    ax2.axvline(po_gene_mean, color="#2196F3", linestyle="-", linewidth=2,
                label=f"P-O mean ({po_gene_mean:.3f})")
    ci_g = results["parent_offspring_similarity"]["gene_embedding"]["ci_95"]
    ax2.axvspan(ci_g[0], ci_g[1], alpha=0.15, color="#2196F3", label=f"95% CI [{ci_g[0]:.3f}, {ci_g[1]:.3f}]")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Count")
    ax2.set_title("Gene Embedding: Parent-Offspring vs Random")
    ax2.legend(fontsize=8)

    # Panel 3: Comparison bar chart
    ax3 = axes[1][0]
    metrics = ["Behavior\nProfile", "Gene\nEmbedding", "Entity\nStats"]
    po_means = [
        results["parent_offspring_similarity"]["behavior"]["mean"],
        results["parent_offspring_similarity"]["gene_embedding"]["mean"],
        results["parent_offspring_similarity"]["entity_stats"]["mean"],
    ]
    rand_means = [
        results["random_pair_baseline"]["behavior"]["mean"],
        results["random_pair_baseline"]["gene_embedding"]["mean"],
        results["random_pair_baseline"]["entity_stats"]["mean"],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, po_means, width, label="Parent-Offspring", color="#4CAF50", alpha=0.8)
    ax3.bar(x + width/2, rand_means, width, label="Random Pairs", color="#F44336", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel("Cosine Similarity")
    ax3.set_title("Inheritance Fidelity: Parent-Offspring vs Random")
    ax3.legend()
    ax3.set_ylim(0, 1)

    # Panel 4: Multi-generation decay with 95% CI ribbon
    ax4 = axes[1][1]
    decay = results.get("lineage_decay_summary", [])
    if decay:
        gens = [d["generation"] for d in decay]
        beh_means = [d["behavior_sim_mean"] for d in decay]
        beh_lo = [d["behavior_sim_ci_95"][0] for d in decay]
        beh_hi = [d["behavior_sim_ci_95"][1] for d in decay]
        ax4.plot(gens, beh_means, "o-", color="#4CAF50", label="Behavior sim (mean)")
        ax4.fill_between(gens, beh_lo, beh_hi, alpha=0.2, color="#4CAF50", label="95% CI")
    else:
        # Fallback: plot individual chains from seed 42
        chains = results["multi_generation_chains"]
        chain_colors = ["#4CAF50", "#2196F3", "#FF9800"]
        for chain in chains:
            gens = [g["generation"] for g in chain["generations"]]
            sims = [g["sim_to_gen0"] for g in chain["generations"]]
            ax4.plot(gens, sims, "o-", color=chain_colors[chain["chain"]],
                    label=f"Chain {chain['chain']}: {' x '.join(chain['initial_parents'])}")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Cosine Similarity to Gen 0")
    ax4.set_title("Behavioral Similarity Decay Over Generations")
    ax4.legend(fontsize=7)
    ax4.set_ylim(0, 1.05)

    plt.tight_layout()
    chart_path = OUT_DIR / "eval3_inheritance.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
