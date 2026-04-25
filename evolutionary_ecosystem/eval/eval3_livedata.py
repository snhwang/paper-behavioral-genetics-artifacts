#!/usr/bin/env python3
"""Eval 3 (live sim): Inheritance fidelity from real population data.

Runs the main sim (same parameters as eval1) and captures every
parent-offspring trio at birth. Measures behavior profile cosine
similarity for parent-offspring pairs vs random-pair baseline.

Two conditions run back-to-back:
  A) locus-free   — breed_deterministic (text splicing)
  B) locus-based  — bear_breed with HAPLOID LocusRegistry (real BEAR pipeline)

For condition B the child's behavior_profile is set to the midparent
average during the sim (so tick() runs normally), then re-computed from
the BEAR corpus in a single batch-embedding pass after the sim finishes.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from bear import Config, Corpus, EmbeddingBackend
from bear.evolution import BreedingConfig, CrossoverMethod, breed as bear_breed
from bear.models import Dominance

from examples.evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    _NAMES,
    breed_deterministic,
    cosine_similarity,
    get_embedder,
    make_creature,
    make_world,
    patch_sim_for_eval,
    profile_to_vector,
    run_simulation,
)
from examples.evolutionary_ecosystem.server.gene_engine import (
    BehaviorProfile,
    GENE_CATEGORIES,
    SituationResult,
    compute_behavior_profile,
)
from bear.models import Dominance, GeneLocus, LocusRegistry

OUT_DIR = Path(__file__).resolve().parent / "results"

N_TICKS           = 200_000
N_CREATURES       = 30
MAX_POP           = 50
SNAPSHOT_INTERVAL = 100
SEEDS             = [42, 1042, 2042, 3042, 4042]


# ── helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def midparent_profile(pa, pb) -> BehaviorProfile:
    """Average of two parents' behavior profiles — cheap, no embedding."""
    situations = {}
    all_sits = set(pa.behavior_profile.situations) | set(pb.behavior_profile.situations)
    for sit in all_sits:
        ra = pa.behavior_profile.situations.get(sit)
        rb = pb.behavior_profile.situations.get(sit)
        sa = ra.strength if ra else 0.3
        sb = rb.strength if rb else 0.3
        situations[sit] = SituationResult(
            strength=(sa + sb) / 2,
            gene_category=(ra or rb).gene_category if (ra or rb) else "",
            gene_text="",
            similarity=(ra or rb).similarity if (ra or rb) else 0.3,
        )
    return BehaviorProfile(situations=situations)


def make_locus_registry() -> LocusRegistry:
    return LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=Dominance.HAPLOID)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])


def _get_bear_config() -> Config:
    return Config(
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_backend=EmbeddingBackend.NUMPY,
        priority_weight=0.3,
        default_threshold=0.3,
        default_top_k=3,
    )


# ── condition A: locus-free ───────────────────────────────────────────────────

def run_trial_locus_free(seed: int) -> list[dict]:
    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng)

    name_to_creature: dict[str, Any] = {c.name: c for c in world.creatures.values()}
    birth_records: list[dict] = []

    def on_birth(child, world, tick_num):
        name_to_creature[child.name] = child
        if not child.parents:
            return
        pa = name_to_creature.get(child.parents[0])
        pb = name_to_creature.get(child.parents[1])
        if pa is None or pb is None:
            return
        pa_vec  = profile_to_vector(pa.behavior_profile)
        pb_vec  = profile_to_vector(pb.behavior_profile)
        mid_vec = [(a + b) / 2 for a, b in zip(pa_vec, pb_vec)]
        ch_vec  = profile_to_vector(child.behavior_profile)
        birth_records.append({
            "mid_profile":   mid_vec,
            "child_profile": ch_vec,
            "generation":    child.generation,
        })

    run_simulation(
        world, N_TICKS, rng,
        snapshot_interval=SNAPSHOT_INTERVAL,
        breed_enabled=True,
        max_population=MAX_POP,
        verbose=False,
        on_birth=on_birth,
        breed_fn=breed_deterministic,
    )

    print(f"  seed={seed} [locus-free]: {len(birth_records)} births, "
          f"max_gen={max((r['generation'] for r in birth_records), default=0)}")
    return birth_records


# ── condition B: locus-based haploid ─────────────────────────────────────────

@dataclass
class BirthRecord:
    pa: Any
    pb: Any
    child_corpus: Corpus
    generation: int


def run_trial_locus_based(seed: int) -> list[dict]:
    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng)

    locus_registry = make_locus_registry()
    name_to_creature: dict[str, Any] = {c.name: c for c in world.creatures.values()}
    pending_records: list[BirthRecord] = []

    def breed_bear_fast(pa, pb, child_id, child_name, rng_arg):
        """Run bear_breed for corpus crossover; use midparent profile during sim."""
        config = BreedingConfig(
            crossover_rate=0.5,
            locus_key="gene_category",
            locus_registry=locus_registry,
            crossover_method=CrossoverMethod.UNIFORM,
            scope_to_child=False,
            seed=rng_arg.randint(0, 2**31),
        )
        result = bear_breed(
            pa.corpus, pb.corpus,
            child_name, pa.name, pb.name,
            config=config,
        )

        # Build child creature with midparent profile (no embedding during sim)
        child = make_creature(child_id, {}, child_name, rng_arg,
                              generation=max(pa.generation, pb.generation) + 1,
                              parents=(pa.name, pb.name))
        child.corpus = result.child
        child.behavior_profile = midparent_profile(pa, pb)

        pending_records.append(BirthRecord(
            pa=pa, pb=pb,
            child_corpus=result.child,
            generation=child.generation,
        ))
        return child

    def on_birth(child, world, tick_num):
        name_to_creature[child.name] = child

    run_simulation(
        world, N_TICKS, rng,
        snapshot_interval=SNAPSHOT_INTERVAL,
        breed_enabled=True,
        max_population=MAX_POP,
        verbose=False,
        on_birth=on_birth,
        breed_fn=breed_bear_fast,
    )

    print(f"  seed={seed} [locus-based]: {len(pending_records)} births captured, "
          f"max_gen={max((r.generation for r in pending_records), default=0)}")
    print(f"  Batch-embedding {len(pending_records)} child corpora...")

    # Batch embed: compute true behavior profiles post-sim
    bear_config = _get_bear_config()
    embedder = get_embedder()
    birth_records: list[dict] = []

    for rec in pending_records:
        child_profile = compute_behavior_profile(rec.child_corpus, bear_config,
                                                 shared_embedder=embedder)
        pa_vec  = profile_to_vector(rec.pa.behavior_profile)
        pb_vec  = profile_to_vector(rec.pb.behavior_profile)
        mid_vec = [(a + b) / 2 for a, b in zip(pa_vec, pb_vec)]
        ch_vec  = profile_to_vector(child_profile)
        birth_records.append({
            "mid_profile":   mid_vec,
            "child_profile": ch_vec,
            "generation":    rec.generation,
        })

    return birth_records


# ── stats + comparison ────────────────────────────────────────────────────────

def compute_stats(all_records: list[dict]) -> dict:
    po_sims = [cosine_similarity(r["child_profile"], r["mid_profile"])
               for r in all_records]

    rng_shuf = random.Random(99)
    mids     = [r["mid_profile"] for r in all_records]
    children = [r["child_profile"] for r in all_records]
    shuffled = mids[:]
    rng_shuf.shuffle(shuffled)
    rand_sims = [cosine_similarity(c, m) for c, m in zip(children, shuffled)]

    t, p = scipy_stats.ttest_ind(po_sims, rand_sims, equal_var=False)
    d    = cohens_d(po_sims, rand_sims)

    by_gen: dict[int, list[float]] = defaultdict(list)
    for r in all_records:
        by_gen[r["generation"]].append(
            cosine_similarity(r["child_profile"], r["mid_profile"]))
    decay = [
        {"generation": g,
         "mean_sim": round(float(np.mean(sims)), 4),
         "ci_95": [round(float(np.mean(sims) - 1.96 * np.std(sims, ddof=1) / np.sqrt(len(sims))), 4),
                   round(float(np.mean(sims) + 1.96 * np.std(sims, ddof=1) / np.sqrt(len(sims))), 4)]
                  if len(sims) > 1 else [float("nan"), float("nan")],
         "n": len(sims)}
        for g, sims in sorted(by_gen.items())
    ]

    return {
        "po_mean":   round(float(np.mean(po_sims)), 4),
        "po_std":    round(float(np.std(po_sims, ddof=1)), 4),
        "rand_mean": round(float(np.mean(rand_sims)), 4),
        "rand_std":  round(float(np.std(rand_sims, ddof=1)), 4),
        "cohens_d":  round(float(d), 4),
        "t":         round(float(t), 4),
        "p":         float(p),
        "n_births":  len(po_sims),
        "decay":     decay,
    }


def run_condition(label: str, trial_fn) -> dict:
    print(f"\n{'='*60}\nCONDITION: {label}\n{'='*60}")
    all_records = []
    for seed in SEEDS:
        all_records.extend(trial_fn(seed))

    stats = compute_stats(all_records)

    print(f"\n  P-O sim:  {stats['po_mean']:.4f} ± {stats['po_std']:.4f}")
    print(f"  Baseline: {stats['rand_mean']:.4f} ± {stats['rand_std']:.4f}")
    print(f"  d={stats['cohens_d']:.3f},  p={stats['p']:.2e},  n={stats['n_births']}")
    print(f"\n  Decay by generation (first 8):")
    for row in stats["decay"][:8]:
        print(f"    gen {row['generation']:>3}: {row['mean_sim']:.4f}  (n={row['n']})")

    return {"label": label, "stats": stats}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("EVAL 3 (live sim): Inheritance Fidelity from Real Population Data")
    print(f"Seeds: {SEEDS}  |  Ticks/trial: {N_TICKS:,}  |  Start pop: {N_CREATURES}")

    results = {
        "locus_free":         run_condition("locus-free (breed_deterministic)",
                                            run_trial_locus_free),
        "locus_based_haploid": run_condition("locus-based haploid (bear_breed UNIFORM)",
                                             run_trial_locus_based),
    }

    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    for key, res in results.items():
        s = res["stats"]
        print(f"\n{res['label']}")
        print(f"  P-O {s['po_mean']:.4f}±{s['po_std']:.4f}  "
              f"Rand {s['rand_mean']:.4f}±{s['rand_std']:.4f}  "
              f"d={s['cohens_d']:.3f}  p={s['p']:.2e}  n={s['n_births']}")

    out_path = OUT_DIR / "eval3_livedata_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
