#!/usr/bin/env python3
"""Eval 5b v2: LLM-mediated breeding diversity (redesigned).

Redesign rationale:
  The original measured diversity on the *final population* (~10 creatures
  at tick 10,000) — noisy because it depends on which creatures happened to
  survive. This version measures diversity *on offspring directly*: for each
  parent pair, breed once with locus-based (deterministic) and once with LLM
  blend, then compare the resulting children's behavior profiles.

  This cleanly answers: "does LLM blending produce more diverse offspring?"
  without simulation noise.

Two parts:
  Part A — Offspring diversity: N parent pairs × 2 methods × R seeds.
    Measures mean pairwise cosine distance across all offspring per method.
    No full sim needed — pure breeding mechanics.

  Part B — Population viability: short sim (10k ticks) confirming LLM
    breeding doesn't destabilize the population. Same metrics as original.

Usage:
    python -m examples.evolutionary_ecosystem.eval.eval5b_v2 \
        --base-url http://192.168.1.175:11434/v1 \
        --model gemma4:e2b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.config import LLMBackend
from bear.llm import LLM
from bear.evolution import BreedingConfig, CrossoverMethod, breed as bear_breed
from bear.models import Dominance, GeneLocus, LocusRegistry

from examples.evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    GENE_CATEGORIES,
    _NAMES,
    PopulationTracker,
    cosine_similarity,
    ensure_eval_patched,
    get_embedder,
    get_config,
    make_creature,
    make_world,
    patch_sim_for_eval,
    profile_to_vector,
    run_simulation,
    build_corpus,
)
from examples.evolutionary_ecosystem.server.gene_engine import (
    BehaviorProfile,
    SituationResult,
    BreedRequest,
    breed_offspring,
    compute_behavior_profile,
)

OUT_DIR = Path(__file__).resolve().parent / "results"

BEAR_CONFIG = Config(
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_backend=EmbeddingBackend.NUMPY,
    priority_weight=0.3,
    default_threshold=0.3,
    default_top_k=3,
)

# Part A parameters
N_PAIRS_PER_SEED = 10    # parent pairs per seed
N_OFFSPRING      = 3     # offspring per pair per method
SEEDS_A          = [42, 1042, 2042, 3042, 4042]

# Part B parameters
N_TICKS_B   = 10_000
N_CREATURES = 6
MAX_POP     = 16
N_TRIALS_B  = 7
SEEDS_B     = [42, 1042, 2042, 3042, 4042, 5042, 6042]
MUTATION_RATE = 0.15


def make_locus_registry() -> LocusRegistry:
    return LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=Dominance.HAPLOID)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) +
                      (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ── Part A: Offspring diversity ───────────────────────────────────────────────

def breed_locus(pa, pb, child_id, child_name, rng) -> Any:
    """Locus-based haploid breeding — no LLM."""
    registry = make_locus_registry()
    config = BreedingConfig(
        crossover_rate=0.5,
        locus_key="gene_category",
        locus_registry=registry,
        crossover_method=CrossoverMethod.UNIFORM,
        scope_to_child=False,
        seed=rng.randint(0, 2**31),
    )
    result = bear_breed(
        pa.corpus or build_corpus(pa.name, pa.genes),
        pb.corpus or build_corpus(pb.name, pb.genes),
        child_name, pa.name, pb.name,
        config=config,
    )
    # Use blended parent genes so make_creature has valid text for initial build
    blended_genes = {k: pa.genes.get(k, pb.genes.get(k, "")) for k in set(list(pa.genes)+list(pb.genes))}
    child = make_creature(child_id, blended_genes, child_name, rng,
                          generation=max(pa.generation, pb.generation) + 1,
                          parents=(pa.name, pb.name))
    child.corpus = result.child
    # Rebuild retriever from bred corpus
    from bear.retriever import Retriever as _R
    _r = _R(result.child, config=BEAR_CONFIG)
    _r._embedder = get_embedder()
    _r.build_index()
    child.retriever = _r
    return child


async def breed_llm_async(pa, pb, child_id, child_name, rng, llm) -> Any:
    """LLM-mediated blend breeding."""
    embedder = get_embedder()
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
        mutation_rate=MUTATION_RATE,
        recombination="blend",
        ploidy="haploid",
    )
    result = await breed_offspring(request, llm, embedder, rng, BEAR_CONFIG)
    child = make_creature(child_id, result.genes, child_name, rng,
                          generation=result.generation,
                          parents=(pa.name, pb.name))
    child.corpus = result.corpus
    child.behavior_profile = result.behavior
    return child


# bear_strength queries — consistent with sim fast path
_DIVERSITY_QUERIES = [
    ("fight aggression territorial combat",    ["combat"]),
    ("hungry foraging find food eat",          ["food"]),
    ("survive starvation endurance resilience",["survival"]),
    ("hide conceal stealth avoid detection",   ["stealth"]),
    ("mate reproduce offspring eager breed",   []),
]


def get_behavior_vec(child, embedder) -> list[float]:
    """Get bear_strength vector from child's retriever or corpus."""
    from bear import Context
    from bear.retriever import Retriever as _Retriever
    # Use attached retriever if available
    if child.retriever is not None:
        return [child.bear_strength(q, t) for q, t in _DIVERSITY_QUERIES]
    # Build retriever from corpus
    if child.corpus:
        r = _Retriever(child.corpus, config=BEAR_CONFIG)
        r._embedder = embedder
        r.build_index()
        child.retriever = r
        return [child.bear_strength(q, t) for q, t in _DIVERSITY_QUERIES]
    return [0.3] * len(_DIVERSITY_QUERIES)


def mean_pairwise_distance(vecs: list[list[float]]) -> float:
    """Mean cosine distance across all pairs."""
    dists = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            dists.append(1.0 - cosine_similarity(vecs[i], vecs[j]))
    return float(np.mean(dists)) if dists else 0.0


def run_part_a(llm: LLM) -> dict:
    print("\n" + "="*65)
    print("PART A: Offspring Diversity (direct comparison)")
    print(f"  {N_PAIRS_PER_SEED} pairs × {N_OFFSPRING} offspring × "
          f"{len(SEEDS_A)} seeds × 2 methods")
    print("="*65)

    patch_sim_for_eval()
    embedder = get_embedder()

    locus_diversities: list[float] = []
    llm_diversities:   list[float] = []
    pair_records = []

    # ── Resume from checkpoint ────────────────────────────────────────
    ckpt_a_path = OUT_DIR / "eval5b_v2_parta_checkpoint.json"
    completed_pairs: set[tuple] = set()
    if ckpt_a_path.exists():
        ckpt_a = json.load(open(ckpt_a_path))
        pair_records = ckpt_a.get("pair_records", [])
        locus_diversities = ckpt_a.get("locus_diversities", [])
        llm_diversities   = ckpt_a.get("llm_diversities", [])
        completed_pairs = {(r["seed"], r["pair"]) for r in pair_records}
        print(f"  Resumed Part A: {len(pair_records)} pairs already done")

    for seed in SEEDS_A:
        rng = random.Random(seed)
        parents = []
        for i in range(N_PAIRS_PER_SEED * 2):
            genes = GENE_BANK[i % len(GENE_BANK)]
            name  = _NAMES[i % len(_NAMES)]
            c = make_creature(f"s{seed}_p{i}", genes, name, rng, generation=0)
            parents.append(c)

        for pair_idx in range(N_PAIRS_PER_SEED):
            if (seed, pair_idx) in completed_pairs:
                print(f"  Skipping seed={seed} pair={pair_idx} (done)")
                continue

            pa = parents[pair_idx * 2]
            pb = parents[pair_idx * 2 + 1]

            locus_vecs = []
            llm_vecs   = []

            for j in range(N_OFFSPRING):
                cname = f"s{seed}_pair{pair_idx}_off{j}"

                lc = breed_locus(pa, pb, f"lc_{cname}", cname+"_L", rng)
                locus_vecs.append(get_behavior_vec(lc, embedder))

                print(f"  seed={seed} pair={pair_idx} offspring={j}: LLM breeding...",
                      end=" ", flush=True)
                t0 = time.time()
                try:
                    llm_child = asyncio.run(
                        breed_llm_async(pa, pb, f"llm_{cname}",
                                        cname+"_LLM", rng, llm))
                    llm_vecs.append(get_behavior_vec(llm_child, embedder))
                    print(f"OK ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    llm_vecs.append([0.3] * len(_DIVERSITY_QUERIES))

            l_div = mean_pairwise_distance(locus_vecs)
            m_div = mean_pairwise_distance(llm_vecs)
            locus_diversities.append(l_div)
            llm_diversities.append(m_div)
            pair_records.append({
                "seed": seed, "pair": pair_idx,
                "locus_diversity": round(l_div, 4),
                "llm_diversity":   round(m_div, 4),
            })
            print(f"  pair {pair_idx}: locus_div={l_div:.4f}  llm_div={m_div:.4f}")

            # Save after every pair
            with open(ckpt_a_path, "w") as f:
                json.dump({"pair_records": pair_records,
                           "locus_diversities": locus_diversities,
                           "llm_diversities": llm_diversities}, f, indent=2)

    # Stats
    t_stat, p_val = scipy_stats.ttest_rel(llm_diversities, locus_diversities)
    d = cohens_d(llm_diversities, locus_diversities)
    ratio = (np.mean(llm_diversities) / np.mean(locus_diversities)
             if np.mean(locus_diversities) > 0 else float("nan"))

    print(f"\nPart A Results:")
    print(f"  Locus: {np.mean(locus_diversities):.4f} ± {np.std(locus_diversities, ddof=1):.4f}")
    print(f"  LLM:   {np.mean(llm_diversities):.4f} ± {np.std(llm_diversities, ddof=1):.4f}")
    print(f"  Ratio: {ratio:.2f}x  d={d:.3f}  p={p_val:.4f}")

    return {
        "description": "mean pairwise cosine distance across offspring per pair",
        "n_pairs":     len(pair_records),
        "n_offspring_per_pair": N_OFFSPRING,
        "locus_based": {
            "mean": round(float(np.mean(locus_diversities)), 4),
            "std":  round(float(np.std(locus_diversities, ddof=1)), 4),
            "values": [round(v, 4) for v in locus_diversities],
        },
        "llm_blend": {
            "mean": round(float(np.mean(llm_diversities)), 4),
            "std":  round(float(np.std(llm_diversities, ddof=1)), 4),
            "values": [round(v, 4) for v in llm_diversities],
        },
        "ratio":       round(float(ratio), 4),
        "cohens_d":    round(float(d), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value":     float(p_val),
        "pair_records": pair_records,
    }


# ── Part B: Population viability ──────────────────────────────────────────────

def _collect_metrics(world, snapshots) -> dict:
    pops = [s.population for s in snapshots]
    final = list(world.creatures.values())
    profiles = [profile_to_vector(c.behavior_profile)
                for c in final if c.behavior_profile]
    dists = []
    if len(profiles) >= 2:
        for i in range(len(profiles)):
            for j in range(i+1, len(profiles)):
                dists.append(1.0 - cosine_similarity(profiles[i], profiles[j]))
    return {
        "avg_population":   round(float(np.mean(pops)) if pops else 0, 2),
        "pop_std":          round(float(np.std(pops)) if pops else 0, 2),
        "total_births":     world.total_births,
        "total_deaths":     world.total_deaths,
        "max_generation":   max((c.generation for c in final), default=0),
        "final_population": len(final),
        "behavior_diversity": round(float(np.mean(dists)) if dists else 0, 4),
    }


def run_part_b_locus(seed: int) -> dict:
    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng)
    tracker = PopulationTracker(history_length=N_TICKS_B // 100 + 100)

    registry = make_locus_registry()

    def breed_fn(pa, pb, child_id, child_name, rng_arg):
        return breed_locus(pa, pb, child_id, child_name, rng_arg)

    snapshots = run_simulation(
        world, N_TICKS_B, rng, tracker,
        snapshot_interval=100,
        breed_enabled=True,
        max_population=MAX_POP,
        verbose=False,
        breed_fn=breed_fn,
    )
    return _collect_metrics(world, tracker.history)


def run_part_b_llm(seed: int, llm: LLM) -> dict:
    """LLM breeding viability sim — same structure as original eval5b."""
    ensure_eval_patched()
    rng = random.Random(seed)
    world = make_world(n_creatures=N_CREATURES, rng=rng)

    from examples.evolutionary_ecosystem.server.sim import tick
    from examples.evolutionary_ecosystem.server.sim import WORLD_W, WORLD_H
    import math

    tracker = PopulationTracker(history_length=N_TICKS_B // 100 + 100)
    pending_breeds: list[tuple[str, str]] = []
    name_counter = [N_CREATURES]

    class FakeQueue:
        def put_nowait(self, item):
            if item[0] == "breed":
                pending_breeds.append((item[1], item[2]))

    world.breed_queue = FakeQueue()
    dt = 1.0 / 20.0

    for t in range(N_TICKS_B):
        tick(world, rng, dt)

        if pending_breeds:
            for aid, bid in pending_breeds:
                a = world.creatures.get(aid)
                b = world.creatures.get(bid)
                if not a or not b or len(world.creatures) >= MAX_POP:
                    continue
                name_counter[0] += 1
                child_id   = world.next_id()
                child_name = _NAMES[name_counter[0] % len(_NAMES)]
                sx = max(1.0, min(WORLD_W-1, (a.x+b.x)/2 + rng.uniform(-1, 1)))
                sy = max(1.0, min(WORLD_H-1, (a.y+b.y)/2 + rng.uniform(-1, 1)))

                request = BreedRequest(
                    parent_a_genes=a.genes,    parent_b_genes=b.genes,
                    parent_a_name=a.name,      parent_b_name=b.name,
                    parent_a_corpus=a.corpus or build_corpus(a.name, a.genes),
                    parent_b_corpus=b.corpus or build_corpus(b.name, b.genes),
                    parent_a_appear=a.appearance, parent_b_appear=b.appearance,
                    parent_a_fitness=a.happiness, parent_b_fitness=b.happiness,
                    child_name=child_name,     child_id=child_id,
                    spawn_x=sx,                spawn_y=sy,
                    generation=max(a.generation, b.generation) + 1,
                    mutation_rate=MUTATION_RATE,
                    recombination="blend",
                    ploidy="haploid",
                )
                try:
                    result = asyncio.run(
                        breed_offspring(request, llm, get_embedder(), rng, BEAR_CONFIG))
                    import examples.evolutionary_ecosystem.server.sim as sim_mod2
                    min_age = getattr(sim_mod2, 'MAX_AGE_MIN', 300.0)
                    max_age = getattr(sim_mod2, 'MAX_AGE_MAX', 500.0)
                    from examples.evolutionary_ecosystem.server.sim import Creature
                    child = Creature(
                        id=result.child_id, name=result.child_name,
                        x=result.spawn_x, y=result.spawn_y,
                        genes=result.genes, appearance=result.appearance,
                        skills=result.skills, stats=result.stats,
                        behavior_profile=result.behavior,
                        happiness=rng.uniform(68, 88),
                        heading=rng.uniform(0, 2*math.pi),
                        generation=result.generation,
                        parents=(result.parent_a_name, result.parent_b_name),
                        corpus=result.corpus,
                        max_age=rng.uniform(min_age, max_age),
                        hp=100.0, energy=rng.uniform(70, 100),
                    )
                    world.creatures[child_id] = child
                    world.total_births += 1
                except Exception as e:
                    print(f"  LLM breed failed tick {t}: {e}")
            pending_breeds.clear()

        if t % 100 == 0:
            tracker.update(world)

    tracker.update(world)
    return _collect_metrics(world, tracker.history)


def run_part_b(llm: LLM) -> dict:
    print("\n" + "="*65)
    print("PART B: Population Viability")
    print(f"  {N_TRIALS_B} trials × {N_TICKS_B:,} ticks, {N_CREATURES} starting creatures")
    print("="*65)

    ckpt_b_path = OUT_DIR / "eval5b_v2_partb_checkpoint.json"
    locus_trials = []
    llm_trials   = []
    completed_b: set[int] = set()

    if ckpt_b_path.exists():
        ckpt_b = json.load(open(ckpt_b_path))
        locus_trials = ckpt_b.get("locus_trials", [])
        llm_trials   = ckpt_b.get("llm_trials", [])
        completed_b  = set(range(min(len(locus_trials), len(llm_trials))))
        print(f"  Resumed Part B: {len(completed_b)} trials done")

    for i, seed in enumerate(SEEDS_B):
        print(f"\nTrial {i+1}/{N_TRIALS_B} (seed={seed})")
        if i in completed_b:
            print(f"  Skipping (already done)")
            continue

        print("  Running locus-based...", flush=True)
        locus_trials.append(run_part_b_locus(seed))
        print(f"  births={locus_trials[-1]['total_births']}  "
              f"div={locus_trials[-1]['behavior_diversity']:.4f}")

        print("  Running LLM blend...", flush=True)
        llm_trials.append(run_part_b_llm(seed, llm))
        print(f"  births={llm_trials[-1]['total_births']}  "
              f"div={llm_trials[-1]['behavior_diversity']:.4f}")

        # Save after every trial
        with open(ckpt_b_path, "w") as f:
            json.dump({"locus_trials": locus_trials,
                       "llm_trials": llm_trials}, f, indent=2)
        print(f"  Checkpoint saved ({i+1}/{N_TRIALS_B} trials done)")

    # Stats per metric
    metrics = ["avg_population", "total_births", "total_deaths",
               "max_generation", "behavior_diversity"]
    stat_tests = {}
    for m in metrics:
        lv = [t[m] for t in locus_trials]
        mv = [t[m] for t in llm_trials]
        t_s, p_v = scipy_stats.ttest_rel(mv, lv)
        d = cohens_d(mv, lv)
        stat_tests[m] = {
            "locus_mean": round(float(np.mean(lv)), 4),
            "locus_std":  round(float(np.std(lv, ddof=1)), 4),
            "llm_mean":   round(float(np.mean(mv)), 4),
            "llm_std":    round(float(np.std(mv, ddof=1)), 4),
            "t_statistic": round(float(t_s), 4),
            "p_value":    float(p_v),
            "cohens_d":   round(float(d), 4),
        }

    print("\nPart B Results:")
    for m, s in stat_tests.items():
        sig = "**" if s["p_value"] < 0.01 else "*" if s["p_value"] < 0.05 else ""
        print(f"  {m:22s}: locus={s['locus_mean']:.3f}  "
              f"llm={s['llm_mean']:.3f}  "
              f"d={s['cohens_d']:.2f}  p={s['p_value']:.3f}{sig}")

    return {
        "locus_trials": locus_trials,
        "llm_trials":   llm_trials,
        "statistical_tests": stat_tests,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://192.168.1.175:11434/v1")
    parser.add_argument("--model",    default="gemma4:e2b")
    parser.add_argument("--skip-b",   action="store_true",
                        help="Skip Part B (population viability)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EVAL 5b v2: LLM-Mediated Breeding Diversity")
    print(f"  Model: {args.model}  at  {args.base_url}")
    print("=" * 65)

    llm = LLM(backend=LLMBackend.OPENAI, model=args.model,
              base_url=args.base_url)

    results: dict[str, Any] = {
        "metadata": {
            "model":    args.model,
            "base_url": args.base_url,
            "n_pairs_per_seed":      N_PAIRS_PER_SEED,
            "n_offspring_per_pair":  N_OFFSPRING,
            "seeds_part_a":          SEEDS_A,
            "n_ticks_part_b":        N_TICKS_B,
            "n_trials_part_b":       N_TRIALS_B,
            "seeds_part_b":          SEEDS_B,
            "mutation_rate":         MUTATION_RATE,
        }
    }

    results["part_a"] = run_part_a(llm)

    if not args.skip_b:
        results["part_b"] = run_part_b(llm)

    out_path = OUT_DIR / "eval5b_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    # Clean up checkpoints
    for ckpt in [OUT_DIR / "eval5b_v2_parta_checkpoint.json",
                 OUT_DIR / "eval5b_v2_partb_checkpoint.json"]:
        if ckpt.exists():
            ckpt.unlink()
    print("Checkpoints removed (run complete)")

    # Final summary
    a = results["part_a"]
    print(f"\nFINAL SUMMARY")
    print(f"  Offspring diversity ratio: {a['ratio']:.2f}x "
          f"(LLM {a['llm_blend']['mean']:.4f} vs "
          f"locus {a['locus_based']['mean']:.4f})")
    print(f"  d={a['cohens_d']:.3f}  p={a['p_value']:.4f}")
    if "part_b" in results:
        b_div = results["part_b"]["statistical_tests"]["behavior_diversity"]
        print(f"  Population diversity: locus={b_div['locus_mean']:.4f}  "
              f"llm={b_div['llm_mean']:.4f}  p={b_div['p_value']:.3f}")


if __name__ == "__main__":
    main()
