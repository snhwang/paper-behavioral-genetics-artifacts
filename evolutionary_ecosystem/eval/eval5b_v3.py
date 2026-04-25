#!/usr/bin/env python3
"""Eval 5b v3: Breeding method diversity comparison.

Compares population diversity across three breeding conditions:
  A) Locus-based haploid (deterministic) — baseline from existing checkpoints
  B) LLM blend haploid
  C) LLM blend diploid (co-dominant)

Diversity measured at:
  - Gene level: mean pairwise gene text similarity across population
  - Behavior level: mean pairwise bear_strength cosine distance
  - Per-gene diversity: per-category gene text diversity

Uses same seeds and tick count as eval_combined_v2 for comparability.

Usage:
    python -m examples.evolutionary_ecosystem.eval.eval5b_v3 \
        --base-url http://localhost:8355/v1 \
        --model gemma-4-e2b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.config import LLMBackend
from bear.llm import LLM

from examples.evolutionary_ecosystem.eval.harness import (
    GENE_CATEGORIES,
    SITUATION_NAMES,
    cosine_similarity,
    gene_diversity_mean,
    compute_per_gene_diversity,
    compute_hausdorff_diversity,
    get_embedder,
    make_world,
    patch_sim_for_eval,
    profile_to_vector,
    run_simulation,
)
from examples.evolutionary_ecosystem.eval.eval_combined_v2 import (
    batch_build_retrievers,
    make_locus_registry,
    make_breed_fn,
    BirthRecord,
    BEAR_CONFIG,
    N_TICKS,
    MAX_POP,
    SNAPSHOT_INT,
)

OUT_DIR = _HERE / "results"
SEEDS = [42, 1042, 2042, 3042, 4042]

# Use smaller population for LLM blend runs — matches original eval5b
N_CREATURES = 6
MAX_POP_5B  = 16

# bear_strength queries for behavior diversity
BEAR_QUERIES = [
    ("food_seeking",  "hungry foraging find food eat",            ["food"]),
    ("combat",        "fight aggression territorial combat",      ["combat"]),
    ("survival",      "survive starvation endurance resilience",  ["survival"]),
    ("stealth",       "hide conceal stealth avoid detection",     ["stealth"]),
    ("breeding",      "mate reproduce offspring eager breed",     []),
]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) +
                      (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def population_diversity(creatures, embedder) -> dict:
    """Compute diversity metrics for a population."""
    # Bear_strength behavior diversity
    vecs = []
    for c in creatures:
        if c.retriever is not None:
            vecs.append([c.bear_strength(q, t) for _, q, t in BEAR_QUERIES])

    behavior_div = 0.0
    if len(vecs) >= 2:
        dists = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                dists.append(1.0 - cosine_similarity(vecs[i], vecs[j]))
        behavior_div = float(np.mean(dists))

    # Gene-level diversity
    gene_div = gene_diversity_mean(creatures)
    per_gene = compute_per_gene_diversity(creatures)

    return {
        "behavior_diversity": round(behavior_div, 4),
        "gene_diversity":     round(gene_div, 4),
        "per_gene_diversity": {k: round(v, 4) for k, v in per_gene.items()},
        "n_creatures":        len(creatures),
        "n_with_retriever":   len(vecs),
    }


def run_locus_trial(seed: int) -> dict:
    """Extract diversity from existing locus checkpoint."""
    # Try to find existing checkpoint data
    for ckpt_path in OUT_DIR.glob("eval_combined_v2_checkpoint*.json"):
        ckpt = json.load(open(ckpt_path))
        births = ckpt.get("all_births", [])
        if not births:
            continue
        # Use births from this seed if available
        # For now just run a fresh locus trial
        break

    # Run fresh locus trial
    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng)
    embedder = get_embedder()

    pending = []
    name_to_c = {c.name: c for c in world.creatures.values()}
    breed_fn = make_breed_fn(make_locus_registry(), pending, name_to_c)

    def on_birth(child, world, tick):
        name_to_c[child.name] = child

    run_simulation(
        world, N_TICKS, rng,
        snapshot_interval=SNAPSHOT_INT,
        breed_enabled=True,
        max_population=MAX_POP_5B,
        verbose=False,
        on_birth=on_birth,
        breed_fn=breed_fn,
    )

    final = list(world.creatures.values())
    batch_build_retrievers(final, embedder)
    div = population_diversity(final, embedder)
    div["total_births"] = world.total_births
    div["max_generation"] = max((c.generation for c in final), default=0)
    print(f"  locus seed={seed}: births={world.total_births} "
          f"gen={div['max_generation']} "
          f"behav_div={div['behavior_diversity']:.4f} "
          f"gene_div={div['gene_diversity']:.4f}")
    return div


def run_locus_diploid_trial(seed: int) -> dict:
    """Run locus diploid trial — fast, no LLM."""
    from examples.evolutionary_ecosystem.eval.eval_combined_v2 import make_breed_fn
    from bear.models import Dominance, GeneLocus, LocusRegistry

    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng)
    embedder = get_embedder()

    # Diploid dominant locus registry
    registry = LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=Dominance.DOMINANT)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])

    pending = []
    name_to_c = {c.name: c for c in world.creatures.values()}

    from bear.evolution import BreedingConfig, CrossoverMethod, breed as bear_breed
    from examples.evolutionary_ecosystem.server.gene_engine import build_corpus
    from examples.evolutionary_ecosystem.eval.eval_combined_v2 import midparent_profile, BirthRecord, BEAR_CONFIG, SNAPSHOT_INT

    def breed_fn(pa, pb, child_id, child_name, rng_arg):
        config = BreedingConfig(
            crossover_rate=0.5,
            locus_key="gene_category",
            locus_registry=registry,
            crossover_method=CrossoverMethod.UNIFORM,
            scope_to_child=False,
            seed=rng_arg.randint(0, 2**31),
        )
        result = bear_breed(
            pa.corpus or build_corpus(pa.name, pa.genes),
            pb.corpus or build_corpus(pb.name, pb.genes),
            child_name, pa.name, pb.name, config=config,
        )
        from examples.evolutionary_ecosystem.eval.harness import make_creature
        bred_genes = {inst.metadata.get("gene_category"): inst.content
                      for inst in result.child.instructions
                      if inst.metadata.get("gene_category")}
        child = make_creature(child_id, bred_genes, child_name, rng_arg,
                              generation=max(pa.generation, pb.generation)+1,
                              parents=(pa.name, pb.name))
        child.corpus = result.child
        child.retriever = None
        child.behavior_profile = midparent_profile(pa, pb)
        return child

    def on_birth(child, world, tick):
        name_to_c[child.name] = child

    run_simulation(world, N_TICKS, rng,
                   snapshot_interval=SNAPSHOT_INT,
                   breed_enabled=True, max_population=MAX_POP_5B,
                   verbose=False, on_birth=on_birth, breed_fn=breed_fn)

    final = list(world.creatures.values())
    batch_build_retrievers(final, embedder)
    div = population_diversity(final, embedder)
    div["total_births"] = world.total_births
    div["max_generation"] = max((c.generation for c in final), default=0)
    print(f"  locus/diploid seed={seed}: births={world.total_births} "
          f"gen={div['max_generation']} "
          f"behav_div={div['behavior_diversity']:.4f} "
          f"gene_div={div['gene_diversity']:.4f}")
    return div


def run_blend_trial(seed: int, ploidy: str, llm: LLM) -> dict:
    """Run one blend trial using app.py headless via subprocess."""
    import asyncio
    from examples.evolutionary_ecosystem.server.gene_engine import (
        breed_offspring, BreedRequest, build_corpus
    )

    rng = random.Random(seed)
    patch_sim_for_eval()
    embedder = get_embedder()
    world = make_world(n_creatures=N_CREATURES, rng=rng)
    name_to_c = {c.name: c for c in world.creatures.values()}

    pending_births = []

    def sync_breed(pa, pb, child_id, child_name, rng_arg):
        """Synchronous wrapper for async LLM breeding."""
        request = BreedRequest(
            parent_a_genes=pa.genes,
            parent_b_genes=pb.genes,
            parent_a_name=pa.name,
            parent_b_name=pb.name,
            parent_a_corpus=pa.corpus or build_corpus(pa.name, pa.genes),
            parent_b_corpus=pb.corpus or build_corpus(pb.name, pb.genes),
            parent_a_appear=pa.appearance,
            parent_b_appear=pb.appearance,
            parent_a_fitness=pa.happiness,
            parent_b_fitness=pb.happiness,
            child_name=child_name,
            child_id=child_id,
            spawn_x=0.0,
            spawn_y=0.0,
            generation=max(pa.generation, pb.generation) + 1,
            mutation_rate=0.15,
            recombination="blend",
            ploidy=ploidy,
        )
        try:
            result = asyncio.run(
                breed_offspring(request, llm, embedder,
                                rng_arg, BEAR_CONFIG))
            from examples.evolutionary_ecosystem.eval.harness import make_creature
            child = make_creature(
                child_id, result.genes, child_name, rng_arg,
                generation=result.generation,
                parents=(pa.name, pb.name),
            )
            child.corpus = result.corpus
            child.behavior_profile = result.behavior
            name_to_c[child_name] = child
            pending_births.append(child_id)
            return child
        except Exception as e:
            print(f"    breed failed: {e}")
            # Fall back to locus
            from examples.evolutionary_ecosystem.eval.eval_combined_v2 import make_breed_fn
            locus_fn = make_breed_fn(make_locus_registry(), [], name_to_c)
            return locus_fn(pa, pb, child_id, child_name, rng_arg)

    def on_birth(child, world, tick):
        name_to_c[child.name] = child

    run_simulation(
        world, N_TICKS, rng,
        snapshot_interval=SNAPSHOT_INT,
        breed_enabled=True,
        max_population=MAX_POP_5B,
        verbose=False,
        on_birth=on_birth,
        breed_fn=sync_breed,
    )

    final = list(world.creatures.values())
    batch_build_retrievers(final, embedder)
    div = population_diversity(final, embedder)
    div["total_births"] = world.total_births
    div["max_generation"] = max((c.generation for c in final), default=0)
    print(f"  blend/{ploidy} seed={seed}: births={world.total_births} "
          f"gen={div['max_generation']} "
          f"behav_div={div['behavior_diversity']:.4f} "
          f"gene_div={div['gene_diversity']:.4f}")
    return div


def run_condition(label: str, trial_fn) -> dict:
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"{'='*60}")

    ckpt_path = OUT_DIR / f"eval5b_v3_checkpoint_{label.replace('/', '_')}.json"
    trials = []
    completed = set()

    if ckpt_path.exists():
        saved = json.load(open(ckpt_path))
        trials = saved.get("trials", [])
        completed = set(range(len(trials)))
        print(f"  Resumed: {len(trials)} trials done")

    for i, seed in enumerate(SEEDS):
        if i in completed:
            print(f"  Skipping seed={seed} (done)")
            continue
        print(f"\nTrial {i+1}/{len(SEEDS)} seed={seed}")
        result = trial_fn(seed)
        trials.append(result)
        with open(ckpt_path, "w") as f:
            json.dump({"label": label, "trials": trials}, f, indent=2)

    behavior_divs = [t["behavior_diversity"] for t in trials]
    gene_divs     = [t["gene_diversity"]     for t in trials]
    births        = [t["total_births"]       for t in trials]

    return {
        "label":    label,
        "n_trials": len(trials),
        "behavior_diversity": {
            "mean":   round(float(np.mean(behavior_divs)), 4),
            "std":    round(float(np.std(behavior_divs, ddof=1)), 4),
            "values": [round(v, 4) for v in behavior_divs],
        },
        "gene_diversity": {
            "mean":   round(float(np.mean(gene_divs)), 4),
            "std":    round(float(np.std(gene_divs, ddof=1)), 4),
            "values": [round(v, 4) for v in gene_divs],
        },
        "total_births": {
            "mean":   round(float(np.mean(births)), 1),
            "values": births,
        },
        "trials": trials,
    }


def compare_conditions(results: dict) -> dict:
    """Statistical comparison of all conditions."""
    conditions = list(results.keys())
    comparisons = {}

    for i, ca in enumerate(conditions):
        for cb in conditions[i+1:]:
            key = f"{ca}_vs_{cb}"
            a_vals = results[ca]["behavior_diversity"]["values"]
            b_vals = results[cb]["behavior_diversity"]["values"]
            t, p = scipy_stats.ttest_rel(b_vals, a_vals)
            d = cohens_d(b_vals, a_vals)
            ratio = (np.mean(b_vals) / np.mean(a_vals)
                     if np.mean(a_vals) > 0 else float("nan"))
            comparisons[key] = {
                "cohens_d":   round(float(d), 4),
                "t_statistic": round(float(t), 4),
                "p_value":    float(p),
                "ratio":      round(float(ratio), 4),
            }
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {cb} vs {ca}: d={d:.3f}  ratio={ratio:.2f}x  p={p:.4f} {sig}")

    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8355/v1")
    parser.add_argument("--model",    default="gemma-4-e2b")
    parser.add_argument("--skip-locus", action="store_true",
                        help="Skip locus conditions (use if already have data)")
    parser.add_argument("--skip-blend", action="store_true",
                        help="Skip blend conditions (run locus only)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("EVAL 5b v3: Breeding Method Diversity Comparison")
    print(f"Seeds: {SEEDS}  Ticks: {N_TICKS:,}  Pop: {N_CREATURES}")
    print(f"LLM: {args.model} @ {args.base_url}")
    print(f"Conditions: locus/haploid(from evals 3+4)  locus/diploid  blend/haploid  blend/diploid_dominant")

    llm = LLM(backend=LLMBackend.OPENAI, model=args.model,
              base_url=args.base_url)

    results = {}

    # Condition A: locus haploid (baseline)
    if not args.skip_locus:
        results["locus_haploid"] = run_condition(
            "locus_haploid",
            lambda seed: run_locus_trial(seed)
        )

    # Condition A2: locus diploid (fast, no LLM)
    results["locus_diploid"] = run_condition(
        "locus_diploid",
        lambda seed: run_locus_diploid_trial(seed)
    )

    # Condition B: blend haploid
    if not args.skip_blend:
        results["blend_haploid"] = run_condition(
            "blend_haploid",
            lambda seed: run_blend_trial(seed, "haploid", llm)
        )

        # Condition C: blend diploid dominant
        results["blend_diploid"] = run_condition(
            "blend_diploid",
            lambda seed: run_blend_trial(seed, "diploid_dominant", llm)
        )

    print(f"\n{'='*60}")
    print("COMPARISONS (behavior diversity)")
    print(f"{'='*60}")
    comparisons = compare_conditions(results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond, res in results.items():
        bd = res["behavior_diversity"]
        gd = res["gene_diversity"]
        tb = res["total_births"]
        print(f"\n{cond}:")
        print(f"  behavior_diversity: {bd['mean']:.4f} ± {bd['std']:.4f}")
        print(f"  gene_diversity:     {gd['mean']:.4f} ± {gd['std']:.4f}")
        print(f"  avg_births:         {tb['mean']:.1f}")

    out = {
        "metadata": {
            "model":    args.model,
            "base_url": args.base_url,
            "seeds":    SEEDS,
            "n_ticks":  N_TICKS,
        },
        "conditions":   results,
        "comparisons":  comparisons,
    }

    out_path = OUT_DIR / "eval5b_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
