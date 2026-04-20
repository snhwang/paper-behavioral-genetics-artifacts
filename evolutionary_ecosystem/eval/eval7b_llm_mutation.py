#!/usr/bin/env python3
"""Evaluation 7b: LLM-mediated mutation diversity.

Extends eval7 (deterministic text splicing vs numeric GA) by adding a third
condition: actual LLM-mediated gene blending using blend_gene() and
mutate_gene() from gene_engine.py.

Design:
  - Same 4 parent pairs and 20 offspring per pair as eval7
  - Three conditions:
    1. Numeric GA: arithmetic crossover + Gaussian mutation (same as eval7)
    2. BEAR deterministic: text splicing + gene bank mutation (same as eval7)
    3. BEAR LLM: LLM-mediated blending + mutation (NEW)
  - Two temperature conditions: 0.0 and 0.7
  - Two diversity metrics:
    a. Behavior profile diversity (7-dim retrieval scores) — favors numeric GA
       because retrieval scores saturate near 1.0 for well-formed gene text
    b. Gene embedding diversity (768-dim semantic embeddings) — captures actual
       semantic differences in gene text, where LLM should show advantage

Hypothesis: LLM-mediated mutation produces offspring that occupy a wider region
of *semantic* gene space than deterministic text splicing, because the LLM
generates qualitatively novel trait descriptions rather than recombining
existing text fragments. The behavior profile (retrieval scores) saturates
for text-based conditions due to normalization ceiling effects, so gene
embedding diversity is the primary metric for comparing text-based methods.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: mlx-community/mistral-nemo-instruct-2407:3 (default, configurable)

Parameters:
- BEAR LLM: blend_gene() for crossover + mutate_gene() at 15% rate
- BEAR deterministic: sentence splicing + gene bank mutation (rate=15%)
- Numeric GA: arithmetic crossover (alpha ∈ [0.3, 0.7]) + Gaussian (σ=0.08, rate=15%)
- Seeds: [42, 1042, 2042] (3 seeds for statistical rigor)

Outputs:
- eval7b_results.json  — Per-pair offspring data and diversity metrics (3 conditions)
- eval7b_llm_mutation.png  — Diversity comparison charts

Usage:
    python eval7b_llm_mutation.py                          # auto-detect backend
    python eval7b_llm_mutation.py --backend local          # LM Studio
    python eval7b_llm_mutation.py --model MODEL_ID         # specific model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.config import LLMBackend
from bear.llm import LLM

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    _NAMES,
    breed_deterministic,
    cosine_similarity,
    ensure_eval_patched,
    get_config,
    get_embedder,
    make_creature,
    profile_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import (
    GENE_CATEGORIES,
    BehaviorProfile,
    SituationResult,
    blend_gene,
    mutate_gene,
    build_corpus,
    compute_behavior_profile,
    _FALLBACK_GENES,
)
from evolutionary_ecosystem.server.sim import WORLD_W, WORLD_H

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# LLM backend detection (follows eval_output_divergence.py pattern)
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "mistral-nemo-instruct-2407"
try:
    from bear.utils import detect_local_llm_url
    DEFAULT_LOCAL_URL = detect_local_llm_url()
except ImportError:
    DEFAULT_LOCAL_URL = "http://127.0.0.1:1234/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def detect_backend(args):
    """Auto-detect backend, model, and base_url from args + environment."""
    if args.backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --backend anthropic requires ANTHROPIC_API_KEY")
            sys.exit(1)
        model = args.model or DEFAULT_ANTHROPIC_MODEL
        return "anthropic", model, ""

    # Any explicit --base-url routes to the OpenAI-compatible local backend
    # (works for Ollama, LM Studio, or any compatible server)
    if args.backend == "local" or (args.backend == "auto" and getattr(args, "base_url", None) and args.base_url != DEFAULT_LOCAL_URL):
        model = args.model or DEFAULT_LOCAL_MODEL
        return "local", model, args.base_url


    # Auto-detect
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = args.model or DEFAULT_ANTHROPIC_MODEL
        return "anthropic", model, ""

    model = args.model or DEFAULT_LOCAL_MODEL
    try:
        import urllib.request
        urllib.request.urlopen(f"{args.base_url}/models", timeout=3)
        return "local", model, args.base_url
    except Exception:
        pass

    print("ERROR: No LLM backend available.")
    print("  Option 1: export ANTHROPIC_API_KEY=sk-...")
    print(f"  Option 2: start a local LLM server at {args.base_url}")
    sys.exit(1)


def make_llm(backend: str, model: str, base_url: str) -> LLM:
    """Create an LLM instance for gene_engine functions."""
    if backend == "anthropic":
        return LLM(backend=LLMBackend.ANTHROPIC, model=model)
    else:
        return LLM(backend=LLMBackend.OPENAI, model=model, base_url=base_url)


# ---------------------------------------------------------------------------
# Parent pairs and parameters (same as eval7)
# ---------------------------------------------------------------------------

PARENT_PAIRS = [
    (0, 1),  # Bold × Timid
    (2, 3),  # Curious × Calm
    (4, 5),  # Energetic × Nurturing
    (6, 7),  # Cunning × Moody
]
N_OFFSPRING = 20
MUTATION_RATE = 0.15
SEEDS = [42, 1042, 2042]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two samples."""
    a_arr, b_arr = np.array(a), np.array(b)
    na, nb = len(a_arr), len(b_arr)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a_arr, ddof=1) + (nb - 1) * np.var(b_arr, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def _ci_95(values: list[float]) -> tuple[float, float]:
    """Compute 95% confidence interval for the mean (t-based)."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr)) if n == 1 else 0.0
        return (m, m)
    mean = float(np.mean(arr))
    se = float(scipy_stats.sem(arr))
    h = se * scipy_stats.t.ppf(0.975, n - 1)
    return (round(mean - h, 8), round(mean + h, 8))


def _pairwise_stat_tests(
    values_a: list[float], values_b: list[float],
    label_a: str, label_b: str, metric_name: str,
) -> dict:
    """Run pairwise statistical tests between two conditions."""
    result: dict = {
        "metric": metric_name,
        "conditions": [label_a, label_b],
        "n_a": len(values_a),
        "n_b": len(values_b),
        "mean_a": round(float(np.mean(values_a)), 8),
        "mean_b": round(float(np.mean(values_b)), 8),
    }

    # Shapiro-Wilk normality
    if len(values_a) >= 3:
        sw_a = scipy_stats.shapiro(values_a)
        result["shapiro_a"] = {"statistic": round(float(sw_a.statistic), 6),
                               "p": round(float(sw_a.pvalue), 6)}
    if len(values_b) >= 3:
        sw_b = scipy_stats.shapiro(values_b)
        result["shapiro_b"] = {"statistic": round(float(sw_b.statistic), 6),
                               "p": round(float(sw_b.pvalue), 6)}

    # Welch's t-test
    if len(values_a) >= 2 and len(values_b) >= 2:
        t_stat, t_p = scipy_stats.ttest_ind(values_a, values_b, equal_var=False)
        result["welch_t"] = {"statistic": round(float(t_stat), 6),
                             "p": round(float(t_p), 6)}

    # Mann-Whitney U
    if len(values_a) >= 1 and len(values_b) >= 1:
        try:
            u_stat, u_p = scipy_stats.mannwhitneyu(
                values_a, values_b, alternative="two-sided")
            result["mann_whitney_u"] = {"statistic": round(float(u_stat), 6),
                                        "p": round(float(u_p), 6)}
        except ValueError:
            result["mann_whitney_u"] = {"statistic": None, "p": None,
                                        "note": "identical distributions"}

    # Cohen's d
    result["cohens_d"] = round(_cohens_d(values_a, values_b), 6)

    # 95% CIs
    result["ci_95_a"] = list(_ci_95(values_a))
    result["ci_95_b"] = list(_ci_95(values_b))

    return result


# ---------------------------------------------------------------------------
# Numeric GA crossover (same as eval7)
# ---------------------------------------------------------------------------

def numeric_crossover_profile(
    a_profile: list[float],
    b_profile: list[float],
    rng: random.Random,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma: float = 0.08,
) -> list[float]:
    child = []
    for va, vb in zip(a_profile, b_profile):
        alpha = rng.uniform(0.3, 0.7)
        val = alpha * va + (1 - alpha) * vb
        if rng.random() < mutation_rate:
            val += rng.gauss(0, mutation_sigma)
        child.append(max(0.05, min(1.0, val)))
    return child


# ---------------------------------------------------------------------------
# LLM-mediated breeding
# ---------------------------------------------------------------------------

async def breed_llm_single(
    parent_a_genes: dict[str, str],
    parent_b_genes: dict[str, str],
    child_name: str,
    llm: LLM,
    rng: random.Random,
    mutation_rate: float = MUTATION_RATE,
) -> dict[str, str]:
    """Breed a single offspring using LLM blending + mutation."""
    child_genes: dict[str, str] = {}

    for cat in GENE_CATEGORIES:
        gene_a = parent_a_genes.get(cat, "")
        gene_b = parent_b_genes.get(cat, "")

        if gene_a and gene_b:
            # LLM blending
            blended = await blend_gene(llm, cat, gene_a, gene_b)
            child_genes[cat] = blended if blended else gene_a
        elif gene_a:
            child_genes[cat] = gene_a
        elif gene_b:
            child_genes[cat] = gene_b
        else:
            child_genes[cat] = _FALLBACK_GENES.get(cat, "")

        # LLM mutation
        if rng.random() < mutation_rate:
            mutated = await mutate_gene(llm, cat, child_genes[cat])
            if mutated:
                child_genes[cat] = mutated

    return child_genes


async def breed_llm_batch(
    parent_a_genes: dict[str, str],
    parent_b_genes: dict[str, str],
    llm: LLM,
    n_offspring: int,
    pair_idx: int,
    rng_seed: int,
    mutation_rate: float = MUTATION_RATE,
) -> list[dict[str, str]]:
    """Breed n offspring sequentially using LLM (to avoid rate limits)."""
    all_genes = []
    for j in range(n_offspring):
        child_rng = random.Random(rng_seed + pair_idx * 100 + j)
        child_name = f"LLMChild_{pair_idx}_{j}"
        print(f"    Breeding LLM offspring {j+1}/{n_offspring}...",
              end=" ", flush=True)
        t0 = time.time()
        genes = await breed_llm_single(
            parent_a_genes, parent_b_genes, child_name,
            llm, child_rng, mutation_rate)
        elapsed = time.time() - t0
        all_genes.append(genes)
        print(f"({elapsed:.1f}s)")
    return all_genes


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------

def compute_diversity(profiles_list: list[list[float]]) -> dict:
    """Behavior profile diversity (7-dim retrieval scores)."""
    n = len(profiles_list)
    ndim = len(profiles_list[0]) if profiles_list else 7
    if n < 2:
        return {"mean_pairwise_distance": 0.0,
                "per_dim_range": [0.0] * ndim,
                "per_dim_std": [0.0] * ndim}

    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1.0 - cosine_similarity(
                profiles_list[i], profiles_list[j]))
    mean_dist = float(np.mean(dists))

    arr = np.array(profiles_list)
    per_dim_range = (arr.max(axis=0) - arr.min(axis=0)).tolist()
    per_dim_std = arr.std(axis=0).tolist()

    return {
        "mean_pairwise_distance": round(mean_dist, 6),
        "per_dim_range": [round(x, 4) for x in per_dim_range],
        "per_dim_std": [round(x, 4) for x in per_dim_std],
    }


def _hausdorff_distance(embs_a: np.ndarray, embs_b: np.ndarray) -> float:
    """Directed Hausdorff: max over a of min distance to any b, symmetrized."""
    from numpy.linalg import norm

    def _directed(src, tgt):
        max_min = 0.0
        for v in src:
            sims = np.dot(tgt, v) / (norm(tgt, axis=1) * norm(v) + 1e-9)
            min_dist = 1.0 - float(np.max(sims))
            max_min = max(max_min, min_dist)
        return max_min

    return max(_directed(embs_a, embs_b), _directed(embs_b, embs_a))


def _sentence_embeddings(texts: list[str], embedder) -> np.ndarray:
    """Split each text into sentences and embed them all."""
    import re
    all_sents = []
    for t in texts:
        sents = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        all_sents.extend(sents if sents else [t])
    return embedder.embed(all_sents) if all_sents else np.zeros((1, 768))


def compute_gene_embedding_diversity(
    gene_sets: list[dict[str, str]],
    embedder,
) -> dict:
    """Compute diversity in gene embedding space (768-dim).

    For each offspring, concatenate all gene texts and embed the result.
    Metrics:
      - mean pairwise cosine distance (genome-level)
      - sentence-level Hausdorff distance (captures fine-grained novelty)
      - per-category cosine distance
    """
    n = len(gene_sets)
    if n < 2:
        return {"mean_cosine_distance": 0.0,
                "mean_hausdorff_distance": 0.0,
                "per_category": {},
                "mean_per_category_distance": 0.0}

    from numpy.linalg import norm

    # Full-genome embeddings: concatenate all gene texts
    full_texts = []
    for genes in gene_sets:
        full_texts.append(" ".join(genes.get(cat, "") for cat in GENE_CATEGORIES))
    full_embs = embedder.embed(full_texts)

    # Pairwise cosine distance on full genome embeddings
    cosine_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(full_embs[i], full_embs[j]) /
                        (norm(full_embs[i]) * norm(full_embs[j]) + 1e-9))
            cosine_dists.append(1.0 - sim)
    mean_cosine = float(np.mean(cosine_dists))

    # Sentence-level Hausdorff distance between each pair of offspring
    import re
    offspring_sent_embs = []
    for genes in gene_sets:
        all_text = " ".join(genes.get(cat, "") for cat in GENE_CATEGORIES)
        sents = [s.strip() for s in re.split(r'[.!?]+', all_text) if s.strip()]
        if not sents:
            sents = [all_text]
        offspring_sent_embs.append(embedder.embed(sents))

    hausdorff_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            hausdorff_dists.append(
                _hausdorff_distance(offspring_sent_embs[i], offspring_sent_embs[j]))
    mean_hausdorff = float(np.mean(hausdorff_dists))

    # Per-category cosine diversity
    per_cat = {}
    for cat in GENE_CATEGORIES:
        cat_texts = [genes.get(cat, "") for genes in gene_sets]
        if all(t == "" for t in cat_texts):
            per_cat[cat] = 0.0
            continue
        cat_embs = embedder.embed(cat_texts)
        cat_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(cat_embs[i], cat_embs[j]) /
                            (norm(cat_embs[i]) * norm(cat_embs[j]) + 1e-9))
                cat_dists.append(1.0 - sim)
        per_cat[cat] = round(float(np.mean(cat_dists)), 6)

    return {
        "mean_cosine_distance": round(mean_cosine, 6),
        "mean_hausdorff_distance": round(mean_hausdorff, 6),
        "per_category": per_cat,
        "mean_per_category_distance": round(
            float(np.mean(list(per_cat.values()))), 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_llm_condition(
    parents: dict,
    llm: LLM,
    temperature: float,
    seed: int = 42,
) -> tuple[dict[int, list[list[float]]], dict[int, list[dict[str, str]]]]:
    """Run LLM breeding for all parent pairs at a given temperature.

    Returns (profiles_by_pair, genes_by_pair).
    """
    embedder = get_embedder()
    config = get_config()

    llm_offspring_profiles: dict[int, list[list[float]]] = {}
    llm_offspring_genes: dict[int, list[dict[str, str]]] = {}
    for pair_idx, (pa, pb) in parents.items():
        print(f"\n  Pair {pair_idx} (temp={temperature}, seed={seed}):")

        all_genes = asyncio.run(breed_llm_batch(
            pa.genes, pb.genes, llm,
            N_OFFSPRING, pair_idx, seed,
            MUTATION_RATE,
        ))

        # Build corpus + compute behavior profile for each offspring
        profiles = []
        for j, genes in enumerate(all_genes):
            child_name = f"LLMChild_{pair_idx}_{j}"
            corpus = build_corpus(child_name, genes)
            profile = compute_behavior_profile(
                corpus, config, shared_embedder=embedder)
            profiles.append(profile_to_vector(profile))

        llm_offspring_profiles[pair_idx] = profiles
        llm_offspring_genes[pair_idx] = all_genes
        print(f"    Bred {len(profiles)} LLM offspring for pair {pair_idx}")

    return llm_offspring_profiles, llm_offspring_genes


def _run_single_seed(
    seed: int,
    parents: dict,
    llm: LLM,
    embedder,
    config,
) -> dict:
    """Run all three conditions for a single seed. Returns results dict."""
    print(f"\n{'#'*70}")
    print(f"# SEED = {seed}")
    print(f"{'#'*70}")

    rng = random.Random(seed)

    # --- Condition 1: Numeric GA ---
    print(f"\n{'='*50}")
    print(f"Condition: Numeric GA (seed={seed})")
    print(f"{'='*50}")
    numeric_offspring_profiles: dict[int, list[list[float]]] = {}
    for pair_idx, (pa, pb) in parents.items():
        prof_a = profile_to_vector(pa.behavior_profile)
        prof_b = profile_to_vector(pb.behavior_profile)
        profiles = []
        for j in range(N_OFFSPRING):
            child_rng = random.Random(seed + pair_idx * 100 + j)
            child_prof = numeric_crossover_profile(
                prof_a, prof_b, child_rng, mutation_rate=MUTATION_RATE)
            profiles.append(child_prof)
        numeric_offspring_profiles[pair_idx] = profiles
        print(f"  Numeric GA: bred {len(profiles)} offspring for pair {pair_idx}")

    # --- Condition 2: BEAR deterministic ---
    print(f"\n{'='*50}")
    print(f"Condition: BEAR Deterministic (seed={seed})")
    print(f"{'='*50}")
    bear_det_offspring_profiles: dict[int, list[list[float]]] = {}
    bear_det_offspring_genes: dict[int, list[dict[str, str]]] = {}
    for pair_idx, (pa, pb) in parents.items():
        profiles = []
        gene_dicts = []
        for j in range(N_OFFSPRING):
            child = breed_deterministic(
                pa, pb, f"bear_{pair_idx}_{j}", f"BearChild_{j}",
                random.Random(seed + pair_idx * 100 + j),
                mutation_rate=MUTATION_RATE,
            )
            profiles.append(profile_to_vector(child.behavior_profile))
            gene_dicts.append(child.genes)
        bear_det_offspring_profiles[pair_idx] = profiles
        bear_det_offspring_genes[pair_idx] = gene_dicts
        print(f"  BEAR det: bred {len(profiles)} offspring for pair {pair_idx}")

    # --- Condition 3: BEAR LLM (both temperatures) ---
    llm_results_by_temp: dict[str, dict[int, list[list[float]]]] = {}
    llm_genes_by_temp: dict[str, dict[int, list[dict[str, str]]]] = {}
    for temp in [0.0, 0.7]:
        print(f"\n{'='*50}")
        print(f"Condition: BEAR LLM (temp={temp}, seed={seed})")
        print(f"{'='*50}")
        llm_profiles, llm_genes = run_llm_condition(
            parents, llm, temp, seed=seed)
        llm_results_by_temp[str(temp)] = llm_profiles
        llm_genes_by_temp[str(temp)] = llm_genes

    # --- Compute diversity metrics ---
    print(f"\n{'='*50}")
    print(f"Computing gene embedding diversity (seed={seed})...")
    print(f"{'='*50}")

    bear_det_gene_div_by_pair = {}
    for pair_idx in range(len(PARENT_PAIRS)):
        bear_det_gene_div_by_pair[pair_idx] = compute_gene_embedding_diversity(
            bear_det_offspring_genes[pair_idx], embedder)
        print(f"  BEAR det pair {pair_idx}: "
              f"cosine={bear_det_gene_div_by_pair[pair_idx]['mean_cosine_distance']:.4f}, "
              f"hausdorff={bear_det_gene_div_by_pair[pair_idx]['mean_hausdorff_distance']:.4f}")

    all_det_genes = []
    for pair_idx in range(len(PARENT_PAIRS)):
        all_det_genes.extend(bear_det_offspring_genes[pair_idx])
    global_det_gene_div = compute_gene_embedding_diversity(all_det_genes, embedder)

    all_results = {}
    for temp_key in ["0.0", "0.7"]:
        llm_profiles = llm_results_by_temp[temp_key]
        llm_genes = llm_genes_by_temp[temp_key]

        llm_gene_div_by_pair = {}
        for pair_idx in range(len(PARENT_PAIRS)):
            llm_gene_div_by_pair[pair_idx] = compute_gene_embedding_diversity(
                llm_genes[pair_idx], embedder)
            print(f"  BEAR LLM pair {pair_idx} (temp={temp_key}): "
                  f"cosine={llm_gene_div_by_pair[pair_idx]['mean_cosine_distance']:.4f}, "
                  f"hausdorff={llm_gene_div_by_pair[pair_idx]['mean_hausdorff_distance']:.4f}")

        all_llm_genes = []
        for pair_idx in range(len(PARENT_PAIRS)):
            all_llm_genes.extend(llm_genes[pair_idx])
        global_llm_gene_div = compute_gene_embedding_diversity(all_llm_genes, embedder)

        pair_results = []
        for pair_idx in range(len(PARENT_PAIRS)):
            pa, pb = parents[pair_idx]
            numeric_div = compute_diversity(numeric_offspring_profiles[pair_idx])
            bear_det_div = compute_diversity(bear_det_offspring_profiles[pair_idx])
            llm_div = compute_diversity(llm_profiles[pair_idx])

            pair_result = {
                "pair": pair_idx,
                "parents": (_NAMES[PARENT_PAIRS[pair_idx][0]],
                            _NAMES[PARENT_PAIRS[pair_idx][1]]),
                "parent_profiles": {
                    "a": [round(x, 4) for x in profile_to_vector(pa.behavior_profile)],
                    "b": [round(x, 4) for x in profile_to_vector(pb.behavior_profile)],
                },
                "profile_diversity": {
                    "numeric": numeric_div,
                    "bear_det": bear_det_div,
                    "bear_llm": llm_div,
                },
                "gene_embedding_diversity": {
                    "bear_det": bear_det_gene_div_by_pair[pair_idx],
                    "bear_llm": llm_gene_div_by_pair[pair_idx],
                },
                "llm_gene_wins_vs_det_cosine": (
                    llm_gene_div_by_pair[pair_idx]["mean_cosine_distance"] >
                    bear_det_gene_div_by_pair[pair_idx]["mean_cosine_distance"]),
                "llm_gene_wins_vs_det_hausdorff": (
                    llm_gene_div_by_pair[pair_idx]["mean_hausdorff_distance"] >
                    bear_det_gene_div_by_pair[pair_idx]["mean_hausdorff_distance"]),
            }
            pair_results.append(pair_result)

            ged_det = bear_det_gene_div_by_pair[pair_idx]
            ged_llm = llm_gene_div_by_pair[pair_idx]
            print(f"\nPair {pair_idx} ({pair_result['parents'][0]} x "
                  f"{pair_result['parents'][1]}) [temp={temp_key}, seed={seed}]:")
            print(f"  Profile diversity (7-dim retrieval scores):")
            print(f"    Numeric:  {numeric_div['mean_pairwise_distance']:.6f}")
            print(f"    BEAR det: {bear_det_div['mean_pairwise_distance']:.6f}")
            print(f"    BEAR LLM: {llm_div['mean_pairwise_distance']:.6f}")
            print(f"  Gene embedding diversity (768-dim):")
            print(f"    BEAR det cosine: {ged_det['mean_cosine_distance']:.4f}  "
                  f"hausdorff: {ged_det['mean_hausdorff_distance']:.4f}")
            print(f"    BEAR LLM cosine: {ged_llm['mean_cosine_distance']:.4f}  "
                  f"hausdorff: {ged_llm['mean_hausdorff_distance']:.4f}")

        # Global profile diversity
        all_numeric = []
        all_bear_det = []
        all_llm = []
        for pair_idx in range(len(PARENT_PAIRS)):
            all_numeric.extend(numeric_offspring_profiles[pair_idx])
            all_bear_det.extend(bear_det_offspring_profiles[pair_idx])
            all_llm.extend(llm_profiles[pair_idx])

        global_numeric = compute_diversity(all_numeric)
        global_bear_det = compute_diversity(all_bear_det)
        global_llm = compute_diversity(all_llm)

        llm_gene_wins_cos = sum(1 for pr in pair_results
                                if pr["llm_gene_wins_vs_det_cosine"])
        llm_gene_wins_haus = sum(1 for pr in pair_results
                                 if pr["llm_gene_wins_vs_det_hausdorff"])

        all_results[temp_key] = {
            "summary": {
                "profile_diversity": {
                    "numeric_global": global_numeric,
                    "bear_det_global": global_bear_det,
                    "bear_llm_global": global_llm,
                },
                "gene_embedding_diversity": {
                    "bear_det_global": global_det_gene_div,
                    "bear_llm_global": global_llm_gene_div,
                },
                "llm_gene_wins_vs_det_cosine": llm_gene_wins_cos,
                "llm_gene_wins_vs_det_hausdorff": llm_gene_wins_haus,
                "total_pairs": len(pair_results),
            },
            "pair_results": pair_results,
        }

    return {
        "seed": seed,
        "results_by_temperature": all_results,
        "numeric_offspring_profiles": numeric_offspring_profiles,
        "bear_det_offspring_profiles": bear_det_offspring_profiles,
        "bear_det_offspring_genes": bear_det_offspring_genes,
        "llm_results_by_temp": llm_results_by_temp,
        "llm_genes_by_temp": llm_genes_by_temp,
    }


def _aggregate_seeds(
    per_seed_results: list[dict],
) -> tuple[dict, dict]:
    """Aggregate per-seed results into final summary with stats.

    Returns (aggregated_results_by_temp, statistical_tests).
    """
    # Collect per-seed global metrics keyed by (temp, condition, metric)
    # For statistical tests we need per-seed scalar values.
    metric_values: dict[str, dict[str, list[float]]] = {}
    # key: (temp, metric_name) -> {condition: [val_per_seed]}

    for sr in per_seed_results:
        for temp_key in ["0.0", "0.7"]:
            s = sr["results_by_temperature"][temp_key]["summary"]
            pd = s["profile_diversity"]
            ged = s["gene_embedding_diversity"]

            prefix = f"temp_{temp_key}"

            # Profile diversity: mean_pairwise_distance
            for cond_label, cond_data in [
                ("numeric", pd["numeric_global"]),
                ("bear_det", pd["bear_det_global"]),
                ("bear_llm", pd["bear_llm_global"]),
            ]:
                key = f"{prefix}__profile__mean_pairwise_distance"
                metric_values.setdefault(key, {})
                metric_values[key].setdefault(cond_label, [])
                metric_values[key][cond_label].append(
                    cond_data["mean_pairwise_distance"])

            # Gene embedding diversity
            for metric_name in ["mean_cosine_distance", "mean_hausdorff_distance"]:
                for cond_label, cond_data in [
                    ("bear_det", ged["bear_det_global"]),
                    ("bear_llm", ged["bear_llm_global"]),
                ]:
                    key = f"{prefix}__gene_embedding__{metric_name}"
                    metric_values.setdefault(key, {})
                    metric_values[key].setdefault(cond_label, [])
                    metric_values[key][cond_label].append(
                        cond_data[metric_name])

    # Run pairwise statistical tests
    condition_pairs = [
        ("numeric", "bear_det"),
        ("numeric", "bear_llm"),
        ("bear_det", "bear_llm"),
    ]
    statistical_tests = []
    for metric_key, cond_vals in sorted(metric_values.items()):
        available_conds = set(cond_vals.keys())
        for ca, cb in condition_pairs:
            if ca not in available_conds or cb not in available_conds:
                continue
            test_result = _pairwise_stat_tests(
                cond_vals[ca], cond_vals[cb], ca, cb, metric_key)
            statistical_tests.append(test_result)

    # Build aggregated results_by_temperature using mean across seeds
    # Use first seed as base structure, then overlay aggregated means + CIs
    base = per_seed_results[0]["results_by_temperature"]
    aggregated = {}
    for temp_key in ["0.0", "0.7"]:
        # Deep copy base structure
        agg_temp = json.loads(json.dumps(base[temp_key], default=float))

        s = agg_temp["summary"]
        prefix = f"temp_{temp_key}"

        # Add CIs and cross-seed means to summary
        pd = s["profile_diversity"]
        for cond_label in ["numeric_global", "bear_det_global", "bear_llm_global"]:
            cond_short = cond_label.replace("_global", "")
            key = f"{prefix}__profile__mean_pairwise_distance"
            vals = metric_values[key][cond_short]
            pd[cond_label]["mean_pairwise_distance_across_seeds"] = round(
                float(np.mean(vals)), 6)
            pd[cond_label]["ci_95_mean_pairwise_distance"] = list(
                _ci_95(vals))

        ged = s["gene_embedding_diversity"]
        for cond_label in ["bear_det_global", "bear_llm_global"]:
            cond_short = cond_label.replace("_global", "")
            for metric_name in ["mean_cosine_distance",
                                "mean_hausdorff_distance"]:
                key = f"{prefix}__gene_embedding__{metric_name}"
                vals = metric_values[key][cond_short]
                ged[cond_label][f"{metric_name}_across_seeds"] = round(
                    float(np.mean(vals)), 6)
                ged[cond_label][f"ci_95_{metric_name}"] = list(
                    _ci_95(vals))

        aggregated[temp_key] = agg_temp

    return aggregated, statistical_tests


def main():
    parser = argparse.ArgumentParser(
        description="Eval 7b: LLM-mediated mutation diversity")
    parser.add_argument("--model", default="",
                        help="LLM model ID (auto-detected if omitted)")
    parser.add_argument("--backend", choices=["auto", "anthropic", "local"],
                        default="auto",
                        help="LLM backend: anthropic, local, or auto. Use --base-url to point at any OpenAI-compatible endpoint (Ollama, etc.)")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_URL,
                        help=f"Local LLM server URL (default: {DEFAULT_LOCAL_URL})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override retrieval threshold (default: use harness config)")
    args = parser.parse_args()

    backend, model, base_url = detect_backend(args)
    llm = make_llm(backend, model, base_url)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_eval_patched()

    embedder = get_embedder()
    config = get_config()
    if args.threshold is not None:
        config = config.model_copy(update={"default_threshold": args.threshold})

    run_timestamp = datetime.now().isoformat()

    print("=" * 70)
    print("EVAL 7b: LLM-Mediated Mutation Diversity (Multi-Seed)")
    print("=" * 70)
    print(f"LLM backend: {backend}")
    print(f"LLM model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Seeds: {SEEDS}")
    print(f"Parent pairs: {len(PARENT_PAIRS)}, Offspring per pair: {N_OFFSPRING}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print(f"Temperatures: 0.0, 0.7")
    print(f"Platform: {platform.platform()}")
    print(f"Timestamp: {run_timestamp}")

    # Create parent creatures (same across seeds — parents are deterministic)
    rng = random.Random(SEEDS[0])
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
        print(f"\nPair {i}: {name_a} x {name_b} (profile sim={parent_sim:.4f})")

    # --- Run all seeds ---
    per_seed_results: list[dict] = []
    for seed in SEEDS:
        seed_result = _run_single_seed(seed, parents, llm, embedder, config)
        per_seed_results.append(seed_result)

    # --- Aggregate across seeds ---
    all_results, statistical_tests = _aggregate_seeds(per_seed_results)

    # Keep backward-compatible references from first seed for plotting
    first = per_seed_results[0]
    numeric_offspring_profiles = first["numeric_offspring_profiles"]
    bear_det_offspring_profiles = first["bear_det_offspring_profiles"]
    bear_det_offspring_genes = first["bear_det_offspring_genes"]
    llm_results_by_temp = first["llm_results_by_temp"]
    llm_genes_by_temp = first["llm_genes_by_temp"]

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (aggregated over {len(SEEDS)} seeds: {SEEDS})")
    print(f"{'='*70}")
    for temp_key, res in all_results.items():
        s = res["summary"]
        pd = s["profile_diversity"]
        ged = s["gene_embedding_diversity"]
        print(f"\nTemperature = {temp_key}:")
        print(f"  --- Behavior Profile Diversity (7-dim retrieval scores) ---")
        for cond_label, cond_key in [("Numeric GA", "numeric_global"),
                                     ("BEAR det", "bear_det_global"),
                                     ("BEAR LLM", "bear_llm_global")]:
            d = pd[cond_key]
            mean_val = d.get("mean_pairwise_distance_across_seeds",
                             d["mean_pairwise_distance"])
            ci = d.get("ci_95_mean_pairwise_distance", [mean_val, mean_val])
            print(f"  {cond_label:12s}: {mean_val:.6f}  "
                  f"95% CI [{ci[0]:.6f}, {ci[1]:.6f}]")
        print(f"  --- Gene Embedding Diversity (768-dim semantic space) ---")
        for cond_label, cond_key in [("BEAR det", "bear_det_global"),
                                     ("BEAR LLM", "bear_llm_global")]:
            d = ged[cond_key]
            cos_mean = d.get("mean_cosine_distance_across_seeds",
                             d["mean_cosine_distance"])
            cos_ci = d.get("ci_95_mean_cosine_distance", [cos_mean, cos_mean])
            haus_mean = d.get("mean_hausdorff_distance_across_seeds",
                              d["mean_hausdorff_distance"])
            haus_ci = d.get("ci_95_mean_hausdorff_distance",
                            [haus_mean, haus_mean])
            print(f"  {cond_label:12s}: cosine={cos_mean:.4f} "
                  f"[{cos_ci[0]:.4f},{cos_ci[1]:.4f}]  "
                  f"hausdorff={haus_mean:.4f} "
                  f"[{haus_ci[0]:.4f},{haus_ci[1]:.4f}]")
        print(f"  LLM wins gene cosine: {s['llm_gene_wins_vs_det_cosine']}/{s['total_pairs']}")
        print(f"  LLM wins gene hausdorff: {s['llm_gene_wins_vs_det_hausdorff']}/{s['total_pairs']}")

    # Print statistical tests
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS (pairwise, across seeds)")
    print(f"{'='*70}")
    for t in statistical_tests:
        print(f"\n  {t['metric']}: {t['conditions'][0]} vs {t['conditions'][1]}")
        print(f"    means: {t['mean_a']:.6f} vs {t['mean_b']:.6f}")
        if "welch_t" in t:
            print(f"    Welch t: t={t['welch_t']['statistic']:.4f}, "
                  f"p={t['welch_t']['p']:.4f}")
        if "mann_whitney_u" in t:
            mw = t["mann_whitney_u"]
            if mw["statistic"] is not None:
                print(f"    Mann-Whitney U: U={mw['statistic']:.4f}, "
                      f"p={mw['p']:.4f}")
            else:
                print(f"    Mann-Whitney U: {mw.get('note', 'N/A')}")
        print(f"    Cohen's d: {t['cohens_d']:.4f}")
        if "shapiro_a" in t:
            print(f"    Shapiro-Wilk (a): W={t['shapiro_a']['statistic']:.4f}, "
                  f"p={t['shapiro_a']['p']:.4f}")
        if "shapiro_b" in t:
            print(f"    Shapiro-Wilk (b): W={t['shapiro_b']['statistic']:.4f}, "
                  f"p={t['shapiro_b']['p']:.4f}")

    # LaTeX table
    print("\n% === LaTeX Table ===")
    print("\\begin{table}[t]")
    print("\\caption{Offspring diversity: Behavior profile (retrieval scores)")
    print("and gene embedding space (768-dim semantic).}")
    print("\\label{tab:llm-mutation-diversity}")
    print("\\begin{tabular}{@{}llcccc@{}}")
    print("\\toprule")
    print("Temp & Condition & Profile Dist & Gene Cosine & Gene Hausdorff & Per-Cat \\\\")
    print("\\midrule")
    for temp_key, res in all_results.items():
        s = res["summary"]
        pd = s["profile_diversity"]
        ged = s["gene_embedding_diversity"]
        # Numeric GA (no gene embedding -- operates on floats, not text)
        print(f"{temp_key} & Numeric GA & "
              f"{pd['numeric_global']['mean_pairwise_distance']:.4f} & "
              f"--- & --- & --- \\\\")
        # BEAR det
        print(f" & BEAR det & "
              f"{pd['bear_det_global']['mean_pairwise_distance']:.4f} & "
              f"{ged['bear_det_global']['mean_cosine_distance']:.4f} & "
              f"{ged['bear_det_global']['mean_hausdorff_distance']:.4f} & "
              f"{ged['bear_det_global']['mean_per_category_distance']:.4f} \\\\")
        # BEAR LLM
        print(f" & BEAR LLM & "
              f"{pd['bear_llm_global']['mean_pairwise_distance']:.4f} & "
              f"{ged['bear_llm_global']['mean_cosine_distance']:.4f} & "
              f"{ged['bear_llm_global']['mean_hausdorff_distance']:.4f} & "
              f"{ged['bear_llm_global']['mean_per_category_distance']:.4f} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Build per_seed data for JSON (just results_by_temperature per seed)
    per_seed_json = []
    for sr in per_seed_results:
        per_seed_json.append({
            "seed": sr["seed"],
            "results_by_temperature": sr["results_by_temperature"],
        })

    # Save results
    output = {
        "metadata": {
            "eval": "7b",
            "description": "LLM-mediated mutation diversity (multi-seed)",
            "llm_backend": backend,
            "llm_model": model,
            "base_url": base_url,
            "temperatures": [0.0, 0.7],
            "n_parent_pairs": len(PARENT_PAIRS),
            "n_offspring_per_pair": N_OFFSPRING,
            "mutation_rate": MUTATION_RATE,
            "retrieval_threshold": config.default_threshold,
            "seeds": SEEDS,
            "seed": SEEDS[0],
            "n_seeds": len(SEEDS),
            "platform": platform.platform(),
            "timestamp": run_timestamp,
        },
        "results_by_temperature": all_results,
        "statistical_tests": statistical_tests,
        "per_seed": per_seed_json,
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    results_path = OUT_DIR / "eval7b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_mutation(output, numeric_offspring_profiles,
                       bear_det_offspring_profiles, llm_results_by_temp,
                       bear_det_offspring_genes, llm_genes_by_temp,
                       parents)
    except Exception as e:
        import traceback
        print(f"Chart generation failed: {e}")
        traceback.print_exc()


def _plot_mutation(output, numeric_profiles, bear_det_profiles,
                   llm_profiles_by_temp,
                   bear_det_genes, llm_genes_by_temp,
                   parents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    res_t0 = output["results_by_temperature"]["0.0"]
    res_t7 = output["results_by_temperature"]["0.7"]
    pair_results_t0 = res_t0["pair_results"]
    n_pairs = len(pair_results_t0)
    width = 0.35

    C_DET = "#4CAF50"
    C_LLM = "#2196F3"
    C_NUM = "#FF9800"

    # ===================================================================
    # (a) Gene Embedding Diversity: Cosine + Hausdorff, both temps
    # ===================================================================
    ax = fig.add_subplot(gs[0, :2])
    metrics = ["Cosine Distance", "Hausdorff Distance"]
    temps = ["0.0", "0.7"]
    x = np.arange(len(metrics) * len(temps))
    det_vals = []
    llm_vals = []
    labels = []
    for temp_key in temps:
        s = output["results_by_temperature"][temp_key]["summary"]
        ged = s["gene_embedding_diversity"]
        det_vals.append(ged["bear_det_global"]["mean_cosine_distance"])
        llm_vals.append(ged["bear_llm_global"]["mean_cosine_distance"])
        labels.append(f"Cosine\n(T={temp_key})")
        det_vals.append(ged["bear_det_global"]["mean_hausdorff_distance"])
        llm_vals.append(ged["bear_llm_global"]["mean_hausdorff_distance"])
        labels.append(f"Hausdorff\n(T={temp_key})")

    x = np.arange(len(det_vals))
    bars_det = ax.bar(x - width / 2, det_vals, width, label="BEAR Deterministic",
                      color=C_DET, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_llm = ax.bar(x + width / 2, llm_vals, width, label="BEAR LLM",
                      color=C_LLM, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bars in [bars_det, bars_llm]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Pairwise Distance", fontsize=10)
    ax.set_title("(a) Gene Embedding Diversity: BEAR Det vs LLM", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (b) Per-Category Gene Embedding Diversity (temp=0.0)
    # ===================================================================
    ax = fig.add_subplot(gs[0, 2])
    cats = list(GENE_CATEGORIES)
    # Get per-category from pair_results average
    det_per_cat = []
    llm_per_cat = []
    for cat in cats:
        det_cat_vals = [pr["gene_embedding_diversity"]["bear_det"]["per_category"].get(cat, 0)
                        for pr in pair_results_t0]
        llm_cat_vals = [pr["gene_embedding_diversity"]["bear_llm"]["per_category"].get(cat, 0)
                        for pr in pair_results_t0]
        det_per_cat.append(float(np.mean(det_cat_vals)))
        llm_per_cat.append(float(np.mean(llm_cat_vals)))

    x = np.arange(len(cats))
    ax.barh(x - 0.17, det_per_cat, 0.34, label="BEAR Det", color=C_DET, alpha=0.85)
    ax.barh(x + 0.17, llm_per_cat, 0.34, label="BEAR LLM", color=C_LLM, alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels([c.replace("_", "\n") for c in cats], fontsize=7)
    ax.set_xlabel("Mean Cosine Distance", fontsize=9)
    ax.set_title("(b) Per-Category Diversity\n(temp=0.0)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (c) Gene Embedding Diversity per Pair (cosine, temp=0.0)
    # ===================================================================
    ax = fig.add_subplot(gs[1, 0])
    det_pair_cos = [pr["gene_embedding_diversity"]["bear_det"]["mean_cosine_distance"]
                    for pr in pair_results_t0]
    llm_pair_cos = [pr["gene_embedding_diversity"]["bear_llm"]["mean_cosine_distance"]
                    for pr in pair_results_t0]
    x = np.arange(n_pairs)
    ax.bar(x - width / 2, det_pair_cos, width, label="BEAR Det", color=C_DET, alpha=0.85)
    ax.bar(x + width / 2, llm_pair_cos, width, label="BEAR LLM", color=C_LLM, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{pr['parents'][0]}x\n{pr['parents'][1]}" for pr in pair_results_t0],
        fontsize=8)
    ax.set_ylabel("Cosine Distance", fontsize=9)
    ax.set_title("(c) Gene Cosine Diversity per Pair\n(temp=0.0)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (d) Gene Embedding Diversity per Pair (Hausdorff, temp=0.0)
    # ===================================================================
    ax = fig.add_subplot(gs[1, 1])
    det_pair_haus = [pr["gene_embedding_diversity"]["bear_det"]["mean_hausdorff_distance"]
                     for pr in pair_results_t0]
    llm_pair_haus = [pr["gene_embedding_diversity"]["bear_llm"]["mean_hausdorff_distance"]
                     for pr in pair_results_t0]
    ax.bar(x - width / 2, det_pair_haus, width, label="BEAR Det", color=C_DET, alpha=0.85)
    ax.bar(x + width / 2, llm_pair_haus, width, label="BEAR LLM", color=C_LLM, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{pr['parents'][0]}x\n{pr['parents'][1]}" for pr in pair_results_t0],
        fontsize=8)
    ax.set_ylabel("Hausdorff Distance", fontsize=9)
    ax.set_title("(d) Gene Hausdorff Diversity per Pair\n(temp=0.0)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (e) Behavior Profile Diversity (retrieval scores) — all 3 conditions
    # ===================================================================
    ax = fig.add_subplot(gs[1, 2])
    conditions = ["Numeric GA", "BEAR Det", "BEAR LLM"]
    colors = [C_NUM, C_DET, C_LLM]
    bar_data = []
    for t in ["0.0", "0.7"]:
        s = output["results_by_temperature"][t]["summary"]
        pd = s["profile_diversity"]
        bar_data.append([
            pd["numeric_global"]["mean_pairwise_distance"],
            pd["bear_det_global"]["mean_pairwise_distance"],
            pd["bear_llm_global"]["mean_pairwise_distance"],
        ])
    x_t = np.arange(2)
    total_w = 0.7
    bw = total_w / 3
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        vals = [bd[i] for bd in bar_data]
        offset = (i - 1) * bw
        bars = ax.bar(x_t + offset, vals, bw, label=cond, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00003,
                    f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x_t)
    ax.set_xticklabels(["temp=0.0", "temp=0.7"])
    ax.set_ylabel("Mean Pairwise Cosine Dist", fontsize=9)
    ax.set_title("(e) Behavior Profile Diversity\n(7-dim retrieval scores)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (f) Delta chart: LLM advantage over Det per pair
    # ===================================================================
    ax = fig.add_subplot(gs[2, :2])
    pair_labels = [f"{pr['parents'][0]}x{pr['parents'][1]}" for pr in pair_results_t0]
    pair_results_t7 = res_t7["pair_results"]

    delta_cos_t0 = [
        pair_results_t0[i]["gene_embedding_diversity"]["bear_llm"]["mean_cosine_distance"] -
        pair_results_t0[i]["gene_embedding_diversity"]["bear_det"]["mean_cosine_distance"]
        for i in range(n_pairs)]
    delta_haus_t0 = [
        pair_results_t0[i]["gene_embedding_diversity"]["bear_llm"]["mean_hausdorff_distance"] -
        pair_results_t0[i]["gene_embedding_diversity"]["bear_det"]["mean_hausdorff_distance"]
        for i in range(n_pairs)]
    delta_cos_t7 = [
        pair_results_t7[i]["gene_embedding_diversity"]["bear_llm"]["mean_cosine_distance"] -
        pair_results_t7[i]["gene_embedding_diversity"]["bear_det"]["mean_cosine_distance"]
        for i in range(n_pairs)]
    delta_haus_t7 = [
        pair_results_t7[i]["gene_embedding_diversity"]["bear_llm"]["mean_hausdorff_distance"] -
        pair_results_t7[i]["gene_embedding_diversity"]["bear_det"]["mean_hausdorff_distance"]
        for i in range(n_pairs)]

    x = np.arange(n_pairs)
    ax.plot(x, delta_cos_t0, "o-", color=C_LLM, label="Cosine (T=0.0)", linewidth=2)
    ax.plot(x, delta_haus_t0, "s--", color=C_LLM, alpha=0.6,
            label="Hausdorff (T=0.0)", linewidth=2)
    ax.plot(x, delta_cos_t7, "o-", color="#E91E63", label="Cosine (T=0.7)", linewidth=2)
    ax.plot(x, delta_haus_t7, "s--", color="#E91E63", alpha=0.6,
            label="Hausdorff (T=0.7)", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_ylabel("Delta (LLM - Det)", fontsize=10)
    ax.set_title("(f) LLM Advantage per Pair (positive = LLM more diverse)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ===================================================================
    # (g) Summary: global gene embedding diversity both temps
    # ===================================================================
    ax = fig.add_subplot(gs[2, 2])
    summary_labels = ["Cosine\nT=0.0", "Cosine\nT=0.7",
                      "Hausdorff\nT=0.0", "Hausdorff\nT=0.7"]
    det_summary = []
    llm_summary = []
    for t in ["0.0", "0.7"]:
        ged = output["results_by_temperature"][t]["summary"]["gene_embedding_diversity"]
        det_summary.append(ged["bear_det_global"]["mean_cosine_distance"])
        llm_summary.append(ged["bear_llm_global"]["mean_cosine_distance"])
    for t in ["0.0", "0.7"]:
        ged = output["results_by_temperature"][t]["summary"]["gene_embedding_diversity"]
        det_summary.append(ged["bear_det_global"]["mean_hausdorff_distance"])
        llm_summary.append(ged["bear_llm_global"]["mean_hausdorff_distance"])

    x = np.arange(4)
    # Reorder to: cos0, cos7, haus0, haus7
    ax.bar(x - 0.17, det_summary, 0.34, label="BEAR Det", color=C_DET, alpha=0.85)
    ax.bar(x + 0.17, llm_summary, 0.34, label="BEAR LLM", color=C_LLM, alpha=0.85)
    for i, (d, l) in enumerate(zip(det_summary, llm_summary)):
        ax.text(i - 0.17, d + 0.002, f"{d:.3f}", ha="center", fontsize=7, fontweight="bold")
        ax.text(i + 0.17, l + 0.002, f"{l:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_labels, fontsize=8)
    ax.set_ylabel("Distance", fontsize=9)
    ax.set_title("(g) Global Gene Embedding\nDiversity Summary", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Eval 7b: LLM-Mediated Mutation Diversity\n"
        f"Model: {output['metadata']['llm_model']}  |  "
        f"4 pairs x {output['metadata']['n_offspring_per_pair']} offspring  |  "
        f"Mutation rate: {output['metadata']['mutation_rate']}",
        fontsize=13, fontweight="bold", y=0.98)
    chart_path = OUT_DIR / "eval7b_llm_mutation.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
