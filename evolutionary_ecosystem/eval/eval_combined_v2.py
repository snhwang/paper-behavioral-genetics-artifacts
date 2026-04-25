#!/usr/bin/env python3
"""Combined eval v2: Eval 3 (inheritance) + Eval 4 (epoch shift).

Runs epoch-locked populations using the real BEAR locus-based pipeline.
Captures parent-offspring inheritance data from every birth as a side-effect.

Improvements over original:
  - Locus-based haploid breeding (bear_breed UNIFORM) throughout
  - 5 trials per epoch (vs 3) for tighter confidence intervals
  - p-values stored at full float precision
  - Behavior profiles measured from BEAR corpus post-sim (batch embedding)
  - Old result files untouched — outputs go to eval3_v2_results.json
    and eval4_v2_results.json

Usage:
    python -m examples.evolutionary_ecosystem.eval.eval_combined_v2
"""
from __future__ import annotations
import argparse

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
from bear.models import Dominance, GeneLocus, LocusRegistry

from examples.evolutionary_ecosystem.eval.harness import (
    SITUATION_NAMES,
    cosine_similarity,
    get_embedder,
    make_creature,
    make_world,
    patch_sim_for_eval,
    profile_to_vector,
    run_simulation,
    PopulationTracker,
    GENE_CATEGORIES,
)
from examples.evolutionary_ecosystem.server.gene_engine import (
    BehaviorProfile,
    SituationResult,
    compute_behavior_profile,
    build_corpus,
)
from examples.evolutionary_ecosystem.server.epochs import EPOCHS
from examples.evolutionary_ecosystem.server import sim as sim_mod

OUT_DIR = Path(__file__).resolve().parent / "results"


# ── Gene-space metric helpers (from eval3_inheritance_fidelity) ───────────────

def _per_gene_embeddings(genes: dict, embedder) -> dict:
    """Return {category: 768-dim embedding} for each gene category."""
    return {cat: embedder.embed_single(text)
            for cat, text in genes.items() if text}


def _mean_gene_embedding(genes: dict, embedder) -> np.ndarray:
    """Mean of all gene category embeddings — single 768-dim vector."""
    vecs = [embedder.embed_single(t) for t in genes.values() if t]
    return np.mean(vecs, axis=0) if vecs else np.zeros(768)


def _per_gene_cosine_sim(embs_a: dict, embs_b: dict) -> float:
    """Mean cosine similarity across shared gene categories."""
    shared = set(embs_a) & set(embs_b)
    if not shared:
        return 0.0
    sims = []
    for cat in shared:
        a, b = embs_a[cat], embs_b[cat]
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(float(np.dot(a, b)) / (norm + 1e-9))
    return float(np.mean(sims))


def _hausdorff_gene_distance(embs_a: dict, embs_b: dict) -> float:
    """Hausdorff distance over per-gene embeddings (cosine distance)."""
    va = list(embs_a.values())
    vb = list(embs_b.values())
    if not va or not vb:
        return 1.0
    def _cd(u, v):
        n = float(np.linalg.norm(u) * np.linalg.norm(v))
        return 1.0 - float(np.dot(u, v)) / (n + 1e-9)
    sup_a = max(min(_cd(a, b) for b in vb) for a in va)
    sup_b = max(min(_cd(b, a) for a in va) for b in vb)
    return max(sup_a, sup_b)


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / (n + 1e-9) if n > 0 else 0.0

# ── Parameters ────────────────────────────────────────────────────────────────
BASE_SEED    = 42
N_TICKS      = 30_000
N_CREATURES  = 30
MAX_POP      = 50
N_TRIALS     = 5          # increased from 3
SNAPSHOT_INT = 200

BEAR_CONFIG = Config(
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_backend=EmbeddingBackend.NUMPY,
    priority_weight=0.3,
    default_threshold=0.3,
    default_top_k=3,
)


def batch_build_retrievers(creatures: list, embedder) -> None:
    """Build retrievers for all creatures in one batched embedding call.
    Much faster than building individually — embeds all instructions at once."""
    from bear.retriever import Retriever

    # Collect all instruction texts across all creatures
    all_texts: list[str] = []
    creature_slices: list[tuple] = []  # (creature, start, end)

    for c in creatures:
        if c.corpus is None:
            continue
        instructions = c.corpus.instructions
        if not instructions:
            continue
        start = len(all_texts)
        all_texts.extend(inst.content for inst in instructions)
        end = len(all_texts)
        creature_slices.append((c, start, end))

    if not all_texts:
        return

    # Single batch embed call for all instructions
    embeddings = embedder.embed(all_texts, is_query=False)  # shape (N, 768)

    # Assign back to each creature's retriever using backend.build_index
    for c, start, end in creature_slices:
        r = Retriever(c.corpus, config=BEAR_CONFIG)
        r._embedder = embedder
        r._instruction_list = list(c.corpus.instructions)
        # Inject pre-computed embeddings directly into backend — skip embed step
        r._backend.build_index(embeddings[start:end])
        r._built = True
        c.retriever = r


def make_locus_registry(dominance=Dominance.HAPLOID) -> LocusRegistry:
    return LocusRegistry(loci=[
        GeneLocus(name=cat, position=i, dominance=dominance)
        for i, cat in enumerate(GENE_CATEGORIES)
    ])


def midparent_profile(pa, pb) -> BehaviorProfile:
    """Cheap midparent average — used during sim so tick() runs normally."""
    situations = {}
    for sit in set(pa.behavior_profile.situations) | set(pb.behavior_profile.situations):
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


@dataclass
class BirthRecord:
    pa_profile: list[float]
    pb_profile: list[float]
    child_corpus: Corpus
    generation: int
    pa_genes: dict = None   # for gene-space metrics
    pb_genes: dict = None
    child_genes: dict = None


# Module-level dominance setting (set from args in main)
_DOMINANCE = Dominance.HAPLOID


def make_breed_fn(locus_registry, pending_records, name_to_creature):
    def breed_bear_fast(pa, pb, child_id, child_name, rng_arg):
        config = BreedingConfig(
            crossover_rate=0.5,
            locus_key="gene_category",
            locus_registry=locus_registry,
            crossover_method=CrossoverMethod.UNIFORM,
            scope_to_child=False,
            seed=rng_arg.randint(0, 2**31),
        )
        result = bear_breed(
            pa.corpus or build_corpus(pa.name, pa.genes),
            pb.corpus or build_corpus(pb.name, pb.genes),
            child_name, pa.name, pb.name,
            config=config,
        )
        # Extract bred genes from result corpus for gene-space metrics
        bred_genes = {inst.metadata.get("gene_category", "unknown"): inst.content
                      for inst in result.child.instructions
                      if inst.metadata.get("gene_category")}

        child = make_creature(
            child_id, bred_genes, child_name, rng_arg,
            generation=max(pa.generation, pb.generation) + 1,
            parents=(pa.name, pb.name),
        )
        # Override corpus and rebuild retriever from bred corpus
        child.corpus = result.child
        # Retriever built in batch after sim — not per-birth
        child.retriever = None
        child.behavior_profile = midparent_profile(pa, pb)

        # Store bear_strength vectors using the rebuilt retriever
        def _bear_vec(c):
            from bear import Context
            _qs = [
                ("fight aggression territorial combat",    ["combat"]),
                ("hungry foraging find food eat",          ["food"]),
                ("survive starvation endurance resilience",["survival"]),
                ("hide conceal stealth avoid detection",   ["stealth"]),
                ("mate reproduce offspring eager breed",   []),
            ]
            return [c.bear_strength(q, t) for q, t in _qs]
        pending_records.append(BirthRecord(
            pa_profile=_bear_vec(pa),
            pb_profile=_bear_vec(pb),
            child_corpus=result.child,
            generation=child.generation,
            pa_genes=dict(pa.genes),
            pb_genes=dict(pb.genes),
            child_genes=bred_genes,
        ))
        return child
    return breed_bear_fast


def run_epoch_trial(epoch_index: int, seed: int) -> tuple[dict, list[BirthRecord]]:
    epoch = EPOCHS[epoch_index]
    rng = random.Random(seed)
    patch_sim_for_eval()
    world = make_world(n_creatures=N_CREATURES, rng=rng, epoch_index=epoch_index)

    original_epoch_check = sim_mod._epoch_check
    sim_mod._epoch_check = lambda w: None  # lock epoch

    pending_records: list[BirthRecord] = []
    name_to_creature: dict[str, Any] = {c.name: c for c in world.creatures.values()}
    breed_fn = make_breed_fn(make_locus_registry(_DOMINANCE), pending_records, name_to_creature)
    tracker = PopulationTracker(history_length=N_TICKS // SNAPSHOT_INT + 50)

    def on_birth(child, world, tick_num):
        name_to_creature[child.name] = child

    snapshots = run_simulation(
        world, N_TICKS, rng, tracker,
        snapshot_interval=SNAPSHOT_INT,
        breed_enabled=True,
        max_population=MAX_POP,
        verbose=False,
        on_birth=on_birth,
        breed_fn=breed_fn,
    )
    sim_mod._epoch_check = original_epoch_check

    # Batch-build retrievers for final population in one embedding call
    final_pop = list(world.creatures.values())
    print(f"    Batch-building retrievers for {len(final_pop)} final creatures...")
    batch_build_retrievers(final_pop, get_embedder())

    final_creatures = list(world.creatures.values())
    # Use bear_strength vectors — same queries as fast path
    # These reflect actual corpus-driven behavior, not precomputed scalars
    EVAL4_QUERIES = [
        ("food_seeking",   "hungry foraging find food eat",           ["food"]),
        ("combat",         "fight aggression territorial combat",     ["combat"]),
        ("survival",       "survive starvation endurance resilience", ["survival"]),
        ("stealth",        "hide conceal stealth avoid detection",    ["stealth"]),
        ("breeding",       "mate reproduce offspring eager breed",    []),
    ]
    # For ANOVA we need per-creature vectors — use bear_strength for measured dimensions
    # Fall back to behavior_profile for unmeasured SITUATION_NAMES
    def creature_vec(c):
        # Use only bear_strength dimensions — these are the only ones
        # with real genetic variation. The other 21 SITUATION_NAMES
        # use behavior_profile which is midparent average (no variation).
        if c.retriever is not None:
            return [c.bear_strength(q, t) for _, q, t in EVAL4_QUERIES]
        # Fallback for initial (gen-0) creatures that have real behavior_profile
        sit_idx = {s: i for i, s in enumerate(SITUATION_NAMES)}
        bp = profile_to_vector(c.behavior_profile) if c.behavior_profile else [0.3]*len(SITUATION_NAMES)
        return [bp[sit_idx[s]] if s in sit_idx else 0.3 for s, _, _ in EVAL4_QUERIES]

    final_profiles = [creature_vec(c) for c in final_creatures]

    avg_profile: dict[str, float] = {}
    per_situation_values: dict[str, list[float]] = defaultdict(list)
    if final_profiles:
        arr = np.array(final_profiles)
        # Only 5 dimensions — keyed by EVAL4_QUERIES situation names
        for i, (sit_name, _, _) in enumerate(EVAL4_QUERIES):
            avg_profile[sit_name] = round(float(arr[:, i].mean()), 4)
            per_situation_values[sit_name] = arr[:, i].tolist()
    else:
        avg_profile = {s: 0.3 for s, _, _ in EVAL4_QUERIES}

    max_gen = max((c.generation for c in final_creatures), default=0)

    eval4_result = {
        "epoch":            epoch.name,
        "epoch_index":      epoch_index,
        "final_population": len(final_creatures),
        "max_generation":   max_gen,
        "total_births":     world.total_births,
        "total_deaths":     world.total_deaths,
        "avg_behavior_profile":  avg_profile,
        "per_situation_values":  {k: [round(v, 4) for v in vs]
                                  for k, vs in per_situation_values.items()},
        "behavior_trajectory": [
            {"tick": s.tick, "population": s.population,
             "avg_behavior": {k: round(v, 4) for k, v in s.avg_behavior.items()}}
            for s in snapshots[::5]
        ],
    }

    print(f"    {epoch.name} seed={seed}: pop={len(final_creatures)} "
          f"gen={max_gen} births={world.total_births} "
          f"birth_records={len(pending_records)}")
    return eval4_result, pending_records


# ── Eval 3: inheritance fidelity ──────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def compute_eval3(all_births: list[BirthRecord]) -> dict:
    print(f"\nComputing bear_strength vectors for {len(all_births)} birth records...")
    embedder = get_embedder()

    child_vecs: list[list[float]] = []
    mid_vecs:   list[list[float]] = []
    generations: list[int] = []

    # Use bear_strength queries — same scoring used in sim fast path
    # This avoids the top-1 compression problem of compute_behavior_profile
    QUERIES = [
        ("fight aggression territorial combat",    ["combat"]),
        ("hungry foraging find food eat",          ["food"]),
        ("survive starvation endurance resilience",["survival"]),
        ("hide conceal stealth avoid detection",   ["stealth"]),
        ("mate reproduce offspring eager breed",   []),
    ]
    from bear.retriever import Retriever
    from bear import Context

    def corpus_to_vec(corpus):
        r = Retriever(corpus, config=BEAR_CONFIG)
        r._embedder = embedder
        r.build_index()
        vec = []
        for query, tags in QUERIES:
            ctx = Context(tags=tags) if tags else None
            res = r.retrieve(query=query, top_k=1, threshold=0.0, context=ctx)
            vec.append(res[0].similarity if res else 0.3)
        return vec

    for rec in all_births:
        if rec.child_corpus is None:
            continue
        child_vecs.append(corpus_to_vec(rec.child_corpus))
        mid_vecs.append([(a+b)/2 for a, b in zip(rec.pa_profile, rec.pb_profile)])
        generations.append(rec.generation)

    po_sims = [cosine_similarity(c, m) for c, m in zip(child_vecs, mid_vecs)]

    # Random baseline: shuffle midparents
    rng_shuf = random.Random(99)
    shuffled = mid_vecs[:]
    rng_shuf.shuffle(shuffled)
    rand_sims = [cosine_similarity(c, m) for c, m in zip(child_vecs, shuffled)]

    t, p = scipy_stats.ttest_ind(po_sims, rand_sims, equal_var=False)
    d = cohens_d(po_sims, rand_sims)

    # ── Gene-space metrics (768-dim) ──────────────────────────────────
    print("  Computing gene-space metrics (768-dim)...")
    po_gene_emb:     list[float] = []
    po_pergene:      list[float] = []
    po_hausdorff:    list[float] = []
    rand_gene_emb:   list[float] = []
    rand_pergene:    list[float] = []
    rand_hausdorff:  list[float] = []

    rng_gene = random.Random(99)
    all_pa_genes = [r.pa_genes for r in all_births if r.pa_genes]
    all_pb_genes = [r.pb_genes for r in all_births if r.pb_genes]
    all_ch_genes = [r.child_genes for r in all_births if r.child_genes]

    for pa_g, pb_g, ch_g in zip(all_pa_genes, all_pb_genes, all_ch_genes):
        # Midparent gene embedding
        pa_emb = _mean_gene_embedding(pa_g, embedder)
        pb_emb = _mean_gene_embedding(pb_g, embedder)
        ch_emb = _mean_gene_embedding(ch_g, embedder)
        mid_emb = (pa_emb + pb_emb) / 2
        po_gene_emb.append(_cosine_vec(ch_emb, mid_emb))

        # Per-gene cosine
        pa_per = _per_gene_embeddings(pa_g, embedder)
        pb_per = _per_gene_embeddings(pb_g, embedder)
        ch_per = _per_gene_embeddings(ch_g, embedder)
        mid_per = {cat: (pa_per[cat] + pb_per[cat]) / 2
                   for cat in set(pa_per) & set(pb_per)}
        po_pergene.append(_per_gene_cosine_sim(ch_per, mid_per))
        po_hausdorff.append(_hausdorff_gene_distance(ch_per, mid_per))

    # Random baselines — shuffle child genes against unrelated midparent
    ch_genes_shuf = list(all_ch_genes)
    rng_gene.shuffle(ch_genes_shuf)
    for pa_g, pb_g, ch_g in zip(all_pa_genes, all_pb_genes, ch_genes_shuf):
        pa_emb = _mean_gene_embedding(pa_g, embedder)
        pb_emb = _mean_gene_embedding(pb_g, embedder)
        ch_emb = _mean_gene_embedding(ch_g, embedder)
        mid_emb = (pa_emb + pb_emb) / 2
        rand_gene_emb.append(_cosine_vec(ch_emb, mid_emb))
        pa_per = _per_gene_embeddings(pa_g, embedder)
        pb_per = _per_gene_embeddings(pb_g, embedder)
        ch_per = _per_gene_embeddings(ch_g, embedder)
        mid_per = {cat: (pa_per[cat] + pb_per[cat]) / 2
                   for cat in set(pa_per) & set(pb_per)}
        rand_pergene.append(_per_gene_cosine_sim(ch_per, mid_per))
        rand_hausdorff.append(_hausdorff_gene_distance(ch_per, mid_per))

    def _stat(po_v, rand_v, label):
        t_, p_ = scipy_stats.ttest_ind(po_v, rand_v, equal_var=False)
        d_ = cohens_d(po_v, rand_v)
        print(f"  {label:20s}: P-O={np.mean(po_v):.4f}  Rand={np.mean(rand_v):.4f}  "
              f"d={d_:.3f}  p={p_:.2e}")
        return {"po_mean": round(float(np.mean(po_v)),4),
                "po_std":  round(float(np.std(po_v,ddof=1)),4),
                "rand_mean": round(float(np.mean(rand_v)),4),
                "cohens_d": round(float(d_),4),
                "t_statistic": round(float(t_),4),
                "p_value": float(p_)}

    gene_stats = {
        "behavior":      _stat(po_sims, rand_sims, "behavior (5-dim)"),
        "gene_embedding": _stat(po_gene_emb, rand_gene_emb, "gene_embedding"),
        "per_gene_cosine": _stat(po_pergene, rand_pergene, "per_gene_cosine"),
        "hausdorff":     _stat(po_hausdorff, rand_hausdorff, "hausdorff"),
    }

    # Lineage decay
    by_gen: dict[int, list[float]] = defaultdict(list)
    for sim, gen in zip(po_sims, generations):
        by_gen[gen].append(sim)

    decay = []
    for gen, sims in sorted(by_gen.items()):
        arr = np.array(sims)
        n = len(sims)
        ci_lo = ci_hi = float("nan")
        if n > 1:
            ci_lo = float(arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(n))
            ci_hi = float(arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(n))
        decay.append({
            "generation":        gen,
            "behavior_sim_mean": round(float(arr.mean()), 4),
            "behavior_sim_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n":                 n,
        })

    print(f"  P-O: {np.mean(po_sims):.4f}±{np.std(po_sims, ddof=1):.4f}  "
          f"Rand: {np.mean(rand_sims):.4f}±{np.std(rand_sims, ddof=1):.4f}  "
          f"d={d:.3f}  p={p:.2e}  n={len(po_sims)}")

    return {
        "breeding_mode": "locus-based haploid (bear_breed UNIFORM)",
        "source":        "epoch-locked live sim births",
        "n_births":      len(po_sims),
        "statistical_tests": gene_stats,
        "parent_offspring_similarity": {
            "behavior": {
                "mean":   round(float(np.mean(po_sims)), 4),
                "std":    round(float(np.std(po_sims, ddof=1)), 4),
                "values": [round(v, 4) for v in po_sims],
            },
            "gene_embedding": {"mean": gene_stats["gene_embedding"]["po_mean"],
                               "std":  gene_stats["gene_embedding"]["po_std"]},
            "per_gene_cosine": {"mean": gene_stats["per_gene_cosine"]["po_mean"],
                                "std":  gene_stats["per_gene_cosine"]["po_std"]},
        },
        "random_pair_baseline": {
            "behavior":      {"mean": gene_stats["behavior"]["rand_mean"]},
            "gene_embedding": {"mean": gene_stats["gene_embedding"]["rand_mean"]},
            "per_gene_cosine": {"mean": gene_stats["per_gene_cosine"]["rand_mean"]},
        },
        "lineage_decay_summary": decay,
    }


# ── Eval 4: epoch shift + ANOVA ───────────────────────────────────────────────

def compute_eval4(all_results: dict[str, list[dict]]) -> dict:
    summary: dict[str, dict] = {}
    ci_out: dict[str, dict] = {}

    for epoch_name, trials in all_results.items():
        agg: dict[str, Any] = {"epoch": epoch_name, "n_trials": len(trials)}
        epoch_cis: dict[str, Any] = {}

        for metric in ["final_population", "max_generation", "total_births", "total_deaths"]:
            vals = [t[metric] for t in trials]
            n, mean_v = len(vals), float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            agg[f"{metric}_mean"] = round(mean_v, 2)
            if n > 1 and std_v > 0:
                ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean_v,
                                             scale=std_v/np.sqrt(n))
                epoch_cis[metric] = [round(float(ci[0]), 4), round(float(ci[1]), 4)]
            else:
                epoch_cis[metric] = [round(mean_v, 4), round(mean_v, 4)]

        profile_means: dict[str, float] = {}
        profile_cis:   dict[str, list]  = {}
        for sit in SITUATION_NAMES:
            vals = [t["avg_behavior_profile"].get(sit, 0.3) for t in trials]
            n, mean_v = len(vals), float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            profile_means[sit] = round(mean_v, 4)
            if n > 1 and std_v > 0:
                ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean_v,
                                             scale=std_v/np.sqrt(n))
                profile_cis[sit] = [round(float(ci[0]), 4), round(float(ci[1]), 4)]
            else:
                profile_cis[sit] = [round(mean_v, 4), round(mean_v, 4)]

        epoch_cis["behavior_profile"] = profile_cis
        agg["avg_behavior_profile"] = profile_means
        ci_out[epoch_name] = epoch_cis
        summary[epoch_name] = agg

    # ANOVA — only on 5 bear_strength dimensions (others are 0.3 defaults)
    BEAR_SITS = ["food_seeking", "combat", "survival", "stealth", "breeding"]
    anova: dict[str, dict] = {}
    for sit in BEAR_SITS:
        groups = []
        for epoch_name, trials in all_results.items():
            vals = []
            for t in trials:
                vals.extend(t["per_situation_values"].get(sit, []))
            groups.append(vals)

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            f_stat, p_val = scipy_stats.f_oneway(*groups)
        else:
            f_stat, p_val = 0.0, 1.0

        anova[sit] = {
            "F_statistic": round(float(f_stat), 4),
            "p_value":     float(p_val),   # full precision — no rounding
            "per_epoch_mean": {
                en: round(float(np.mean(
                    [v for t in trials
                     for v in t["per_situation_values"].get(sit, [0.3])]
                )), 4)
                for en, trials in all_results.items()
            },
        }

    sig = sum(1 for v in anova.values() if v["p_value"] < 0.05)
    if anova:
        top = max(anova.items(), key=lambda x: x[1]["F_statistic"])
        print(f"\nANOVA: {sig}/{len(anova)} significant  "
              f"top F={top[1]['F_statistic']:.2f} ({top[0]})")
    else:
        print("\nANOVA: insufficient epochs for comparison (need 2+)")

    return {
        "parameters": {
            "base_seed":   BASE_SEED,
            "n_ticks":     N_TICKS,
            "n_creatures": N_CREATURES,
            "n_trials":    N_TRIALS,
            "breeding":    "locus-based haploid (bear_breed UNIFORM)",
        },
        "summary":               summary,
        "confidence_intervals_95": ci_out,
        "anova":                 anova,
        "trials":                {k: v for k, v in all_results.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_checkpoint(ckpt_path, all_eval4, all_births_serializable):
    """Save incremental checkpoint after every trial."""
    ckpt = {
        "all_eval4": dict(all_eval4),
        "all_births": all_births_serializable,
    }
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)


def _birth_to_dict(b: BirthRecord) -> dict:
    return {
        "pa_profile":    b.pa_profile,
        "pb_profile":    b.pb_profile,
        "child_corpus":  None,  # Corpus not JSON-serializable; recompute from genes
        "generation":    b.generation,
        "pa_genes":      b.pa_genes,
        "pb_genes":      b.pb_genes,
        "child_genes":   b.child_genes,
    }


def _birth_from_dict(d: dict) -> BirthRecord:
    child_genes = d.get("child_genes") or {}
    child_corpus = build_corpus("child", child_genes) if child_genes else None
    return BirthRecord(
        pa_profile=d["pa_profile"],
        pb_profile=d["pb_profile"],
        child_corpus=child_corpus,
        generation=d["generation"],
        pa_genes=d.get("pa_genes"),
        pb_genes=d.get("pb_genes"),
        child_genes=child_genes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="+", default=None,
                        help="Subset of epochs to run (e.g. abundance ice_age)")
    parser.add_argument("--ploidy", default="haploid",
                        choices=["haploid","diploid_codominant","diploid_dominant"],
                        help="Ploidy mode (default: haploid)")
    parser.add_argument("--eval", choices=["3", "4", "both"], default="both",
                        help="Which eval to compute at end: 3, 4, or both")
    args = parser.parse_args()

    # Set ploidy
    global _DOMINANCE
    _dom_map = {
        "haploid":            Dominance.HAPLOID,
        "diploid_codominant": Dominance.CODOMINANT,
        "diploid_dominant":   Dominance.DOMINANT,
    }
    _DOMINANCE = _dom_map.get(args.ploidy, Dominance.HAPLOID)
    print(f"Ploidy: {args.ploidy} ({_DOMINANCE})")

    # Filter epochs if --epochs specified
    epochs_to_run = EPOCHS
    if args.epochs:
        epochs_to_run = [e for e in EPOCHS if e.name in args.epochs]
        if not epochs_to_run:
            print(f"No matching epochs found. Available: {[e.name for e in EPOCHS]}")
            return

    # Use epoch-specific checkpoint and output names when running a subset
    ploidy_suffix = f"_{args.ploidy}" if args.ploidy != "haploid" else ""
    suffix = ("_" + "_".join(e.name for e in epochs_to_run) if args.epochs else "") + ploidy_suffix
    ckpt_path = OUT_DIR / f"eval_combined_v2_checkpoint{suffix}.json"
    eval3_out = OUT_DIR / f"eval3_v2_results{suffix}.json"
    eval4_out = OUT_DIR / f"eval4_v2_results{suffix}.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("COMBINED EVAL v2: Epoch Phenotype Shift + Inheritance Fidelity")
    print(f"Breeding: locus-based haploid (bear_breed UNIFORM)")
    print(f"Ticks: {N_TICKS:,}  |  Trials/epoch: {N_TRIALS}  |  Pop: {N_CREATURES}")
    print(f"Epochs: {[e.name for e in epochs_to_run]}")
    print("=" * 65)

    # ── Resume from checkpoint if available ──────────────────────────────────
    # ckpt_path already set above with epoch suffix — do NOT overwrite it here
    all_eval4: dict[str, list[dict]] = defaultdict(list)
    all_births_ser: list[dict] = []   # JSON-serializable birth records

    if ckpt_path.exists():
        print(f"\nResuming from checkpoint: {ckpt_path}")
        ckpt = json.load(open(ckpt_path))
        for epoch_name, trials in ckpt["all_eval4"].items():
            all_eval4[epoch_name] = trials
        all_births_ser = ckpt["all_births"]
        completed = {(t["epoch"], t["trial"])
                     for trials in all_eval4.values() for t in trials}
        print(f"  Resumed: {len(completed)} trials, {len(all_births_ser)} births")
    else:
        completed = set()

    # ── Run trials ───────────────────────────────────────────────────────────
    for epoch_idx, epoch in [(EPOCHS.index(e), e) for e in epochs_to_run]:
        print(f"\n{'='*50}\nEPOCH: {epoch.name}\n{'='*50}")
        for trial in range(N_TRIALS):
            if (epoch.name, trial) in completed:
                print(f"  Skipping {epoch.name} trial {trial} (already done)")
                continue
            seed = BASE_SEED + epoch_idx * 100 + trial * 1000
            result, births = run_epoch_trial(epoch_idx, seed)
            result["trial"] = trial
            all_eval4[epoch.name].append(result)
            all_births_ser.extend(_birth_to_dict(b) for b in births)

            # Save checkpoint immediately after each trial
            _save_checkpoint(ckpt_path, all_eval4, all_births_ser)
            print(f"  Checkpoint saved. Total births so far: {len(all_births_ser)}")

    # ── Compute and save eval 4 ───────────────────────────────────────────────
    if args.eval in ("4", "both"):
        print("\n" + "="*65)
        print("Aggregating eval 4...")
        eval4 = compute_eval4(all_eval4)
        eval4_path = eval4_out
        with open(eval4_path, "w") as f:
            json.dump(eval4, f, indent=2)
        print(f"Saved: {eval4_path}")

    # ── Compute and save eval 3 ───────────────────────────────────────────────
    if args.eval in ("3", "both"):
        all_births = [_birth_from_dict(d) for d in all_births_ser]
        print("\n" + "="*65)
        print("Computing eval 3 inheritance fidelity...")
        eval3 = compute_eval3(all_births)
        eval3_path = eval3_out
        with open(eval3_path, "w") as f:
            json.dump(eval3, f, indent=2)
        print(f"Saved: {eval3_path}")

    # Clean up checkpoint on successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint removed (run complete)")

    # Summary
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    st = eval3["statistical_tests"]
    print(f"\nEval 3 — {eval3['n_births']} births from live sim")
    for metric, s in st.items():
        po_m = eval3["parent_offspring_similarity"].get(metric, {}).get("mean", "?")
        rb_m = eval3["random_pair_baseline"].get(metric, {}).get("mean", "?")
        print(f"  {metric:20s}: P-O={po_m}  Rand={rb_m}  "
              f"d={s['cohens_d']:.3f}  p={s['p_value']:.2e}")

    anova = eval4["anova"]
    sig = sum(1 for v in anova.values() if v["p_value"] < 0.05)
    if anova and len(all_eval4) >= 2:
        top = max(anova.items(), key=lambda x: x[1]["F_statistic"])
        print(f"\nEval 4 — {sig}/{len(anova)} situations significant")
        print(f"  Top F={top[1]['F_statistic']:.2f} ({top[0]})  "
              f"p={top[1]['p_value']:.2e}")
    else:
        print("\nEval 4 — insufficient epochs for ANOVA (need 2+), skipping")

    print(f"\nOutput files:")
    print(f"  {eval3_path}")
    print(f"  {eval4_path}")


if __name__ == "__main__":
    main()
