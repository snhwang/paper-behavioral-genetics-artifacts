#!/usr/bin/env python3
"""Evaluation 5: Numeric GA baseline comparison.

Compares BEAR's retrieval-driven behavioral genetics against a classical
numeric genetic algorithm (GA) baseline.  Both systems use the same simulation
physics, breeding triggers, mortality mechanics, AND gene text inheritance
(text splicing + gene bank mutation).  The only difference is how the
behavior profile is determined:

  BEAR:       Gene text → BEAR retrieval → behavior profile
  Numeric GA: Gene text inherited but ignored for behavior;
              behavior = 7-float vector, arithmetic crossover + Gaussian mutation

Both conditions inherit gene text identically, so per-gene diversity is a
fair comparison.  The question: does coupling gene text to behavior (BEAR)
produce different evolutionary outcomes than decoupled numeric behavior?

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — no LLM calls)

Parameters:
- 30,000 ticks per trial, 6 starting creatures, max population 16
- 3 trials per condition (seeds: 42, 1042, 2042)
- Numeric GA: arithmetic crossover (alpha ∈ [0.3, 0.7]), Gaussian mutation
  (σ=0.08, rate=15%)
- BEAR: text splicing crossover, gene bank mutation (rate=15%)

Outputs:
- eval5_results.json  — Raw data per condition
- eval5_ga_baseline.png — Comparison chart
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    PopulationTracker,
    _NAMES,
    _blend_text,
    compute_hausdorff_diversity,
    compute_per_gene_diversity,
    cosine_similarity,
    ensure_eval_patched,
    get_config,
    get_embedder,
    make_creature,
    make_world,
    profile_to_vector,
    run_simulation,
)
from evolutionary_ecosystem.server.gene_engine import GENE_CATEGORIES
from evolutionary_ecosystem.server.gene_engine import (
    BehaviorProfile,
    SituationResult,
)
from evolutionary_ecosystem.server.sim import (
    WORLD_H,
    WORLD_W,
    MAX_POPULATION,
    Creature,
    World,
)

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


# ---------------------------------------------------------------------------
# Numeric GA behaviour profile — bypasses BEAR retrieval entirely
# ---------------------------------------------------------------------------


def random_profile(rng: random.Random) -> BehaviorProfile:
    """Generate a random 7-float behavior profile in [0.1, 1.0]."""
    results = {}
    for name in SITUATION_NAMES:
        val = rng.uniform(0.1, 1.0)
        results[name] = SituationResult(
            strength=val,
            gene_category="numeric_ga",
            gene_text="(numeric gene)",
            similarity=val,
        )
    return BehaviorProfile(situations=results)


def crossover_profile(
    a: BehaviorProfile,
    b: BehaviorProfile,
    rng: random.Random,
    mutation_rate: float = 0.15,
    mutation_sigma: float = 0.08,
) -> BehaviorProfile:
    """Arithmetic crossover with Gaussian mutation on 7-float profiles."""
    results = {}
    for name in SITUATION_NAMES:
        va = a.strength(name)
        vb = b.strength(name)
        # Uniform crossover: random weight per dimension
        alpha = rng.uniform(0.3, 0.7)
        child_val = alpha * va + (1 - alpha) * vb
        # Gaussian mutation
        if rng.random() < mutation_rate:
            child_val += rng.gauss(0, mutation_sigma)
        child_val = max(0.05, min(1.0, child_val))
        results[name] = SituationResult(
            strength=child_val,
            gene_category="numeric_ga",
            gene_text="(numeric gene)",
            similarity=child_val,
        )
    return BehaviorProfile(situations=results)


# ---------------------------------------------------------------------------
# GA-mode creature creation
# ---------------------------------------------------------------------------


def make_creature_ga(
    cid: str,
    name: str,
    rng: random.Random,
    generation: int = 0,
    parents: tuple[str, str] | None = None,
    profile: BehaviorProfile | None = None,
    genes: dict[str, str] | None = None,
    spawn_x: float | None = None,
    spawn_y: float | None = None,
) -> Creature:
    """Create a creature using a numeric GA profile (no BEAR retrieval).

    Gene text is inherited for diversity measurement but does NOT drive behavior.
    Behavior is driven entirely by the numeric profile vector.
    """
    ensure_eval_patched()
    embedder = get_embedder()

    if genes is None:
        genes = rng.choice(GENE_BANK)
    from evolutionary_ecosystem.server.gene_engine import (
        extract_appearance,
        extract_skills,
        extract_stats,
    )

    appearance = extract_appearance(genes, embedder)
    skills = extract_skills(genes, embedder)
    stats = extract_stats(genes, embedder)

    if profile is None:
        profile = random_profile(rng)

    x = spawn_x if spawn_x is not None else rng.uniform(1.0, WORLD_W - 1.0)
    y = spawn_y if spawn_y is not None else rng.uniform(1.0, WORLD_H - 1.0)

    from evolutionary_ecosystem.server import sim as sim_mod
    min_age = getattr(sim_mod, "MAX_AGE_MIN", 90.0)
    max_age_val = getattr(sim_mod, "MAX_AGE_MAX", 150.0)

    return Creature(
        id=cid,
        name=name,
        x=x,
        y=y,
        genes=genes,
        appearance=appearance,
        skills=skills,
        stats=stats,
        behavior_profile=profile,
        happiness=rng.uniform(68, 88),
        heading=rng.uniform(0, 2 * math.pi),
        generation=generation,
        parents=parents,
        corpus=None,
        max_age=rng.uniform(min_age, max_age_val),
        hp=100.0,
        energy=rng.uniform(70, 100),
    )


# ---------------------------------------------------------------------------
# GA-mode simulation loop (replaces breeding mechanism)
# ---------------------------------------------------------------------------


def run_ga_simulation(
    n_creatures: int,
    n_ticks: int,
    seed: int,
    max_population: int = MAX_POPULATION,
    mutation_rate: float = 0.15,
) -> dict:
    """Run simulation with numeric GA breeding instead of BEAR breeding."""
    ensure_eval_patched()
    rng = random.Random(seed)

    # Build world with GA creatures
    from evolutionary_ecosystem.server.sim import (
        Predator,
        PREDATOR_SPAWN_INTERVAL,
        tick,
    )
    from evolutionary_ecosystem.server.epochs import EPOCHS
    import time

    world = World(
        predator=Predator(x=0, y=0, spawn_at=time.time() + PREDATOR_SPAWN_INTERVAL),
        epoch=EPOCHS[0],
        epoch_index=0,
    )

    name_counter = [0]
    print(f"Creating {n_creatures} GA creatures...")
    for i in range(n_creatures):
        name = _NAMES[i % len(_NAMES)]
        cid = world.next_id()
        creature = make_creature_ga(cid, name, rng)
        world.creatures[cid] = creature
        name_counter[0] += 1

    print(f"World ready: {len(world.creatures)} GA creatures")

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)
    dt = 1.0 / 20.0

    pending_breeds: list[tuple[str, str]] = []

    class FakeQueue:
        def put_nowait(self, item):
            if item[0] == "breed":
                pending_breeds.append((item[1], item[2]))

    world.breed_queue = FakeQueue()
    world.autonomous_breeding = True  # proximity-based breeding trigger
    prev_ids = set(world.creatures.keys())

    for t in range(n_ticks):
        tick(world, rng, dt)

        curr_ids = set(world.creatures.keys())
        prev_ids = curr_ids

        # GA breeding
        if pending_breeds:
            for aid, bid in pending_breeds:
                a = world.creatures.get(aid)
                b = world.creatures.get(bid)
                if a is None or b is None:
                    continue
                if len(world.creatures) >= max_population:
                    break

                name_counter[0] += 1
                child_id = world.next_id()
                child_name = _NAMES[name_counter[0] % len(_NAMES)]

                child_profile = crossover_profile(
                    a.behavior_profile, b.behavior_profile, rng,
                    mutation_rate=mutation_rate,
                )
                # Inherit gene text via same text splicing as BEAR
                child_genes = {}
                for cat in GENE_CATEGORIES:
                    ga, gb = a.genes.get(cat, ""), b.genes.get(cat, "")
                    if ga and gb:
                        child_genes[cat] = _blend_text(ga, gb, rng)
                    else:
                        child_genes[cat] = ga or gb
                    # Mutation: blend with gene bank donor (same as BEAR)
                    if rng.random() < mutation_rate:
                        donor = rng.choice(GENE_BANK)
                        if cat in donor:
                            child_genes[cat] = _blend_text(child_genes[cat], donor[cat], rng)

                gen = max(a.generation, b.generation) + 1
                sx = (a.x + b.x) / 2 + rng.uniform(-1.0, 1.0)
                sy = (a.y + b.y) / 2 + rng.uniform(-1.0, 1.0)
                sx = max(1.0, min(WORLD_W - 1.0, sx))
                sy = max(1.0, min(WORLD_H - 1.0, sy))

                child = make_creature_ga(
                    child_id, child_name, rng,
                    generation=gen,
                    parents=(a.name, b.name),
                    profile=child_profile,
                    genes=child_genes,
                    spawn_x=sx,
                    spawn_y=sy,
                )
                world.creatures[child_id] = child
                world.total_births += 1

            pending_breeds.clear()

        if t % 100 == 0:
            tracker.update(world)
            if t % 1000 == 0:
                n = len(world.creatures)
                ep = world.epoch.name
                print(f"  tick {t:>6d}: pop={n:>2d} epoch={ep:<15s} "
                      f"births={world.total_births} deaths={world.total_deaths}")

        # Repopulate if extinction
        if not world.creatures:
            print(f"  tick {t:>6d}: EXTINCTION — repopulating with GA creatures...")
            for j in range(3):
                name_counter[0] += 1
                cid = world.next_id()
                name = _NAMES[name_counter[0] % len(_NAMES)]
                c = make_creature_ga(cid, name, rng)
                world.creatures[cid] = c

    tracker.update(world)

    # Collect metrics
    snapshots = tracker.history
    populations = [s.population for s in snapshots]
    avg_pop = float(np.mean(populations))
    pop_std = float(np.std(populations))
    extinctions = sum(1 for p in populations if p == 0)

    final_creatures = list(world.creatures.values())
    per_gene_div = compute_per_gene_diversity(final_creatures)
    gene_diversity = round(float(np.mean(list(per_gene_div.values()))), 4) if per_gene_div else 0.0
    hausdorff_div = compute_hausdorff_diversity(final_creatures)
    hausdorff_mean = round(float(np.mean(list(hausdorff_div.values()))), 4) if hausdorff_div else 0.0

    max_gen = max((c.generation for c in final_creatures), default=0)

    avg_profile = {}
    if final_creatures:
        for sit in SITUATION_NAMES:
            vals = [c.behavior_profile.strength(sit) if c.behavior_profile else 0.3
                    for c in final_creatures]
            avg_profile[sit] = round(float(np.mean(vals)), 4)

    return {
        "avg_population": round(avg_pop, 2),
        "pop_std": round(pop_std, 2),
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "max_generation": max_gen,
        "gene_diversity": gene_diversity,
        "per_gene_diversity": per_gene_div,
        "hausdorff_mean": hausdorff_mean,
        "hausdorff_diversity": hausdorff_div,
        "extinction_events": extinctions,
        "final_population": len(final_creatures),
        "avg_behavior_profile": avg_profile,
        "population_timeline": populations,
    }


def run_bear_condition(seed: int, n_ticks: int, n_creatures: int) -> dict:
    """Run the BEAR condition (reuses eval2 logic)."""
    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng)
    world.bear_disabled = False

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)
    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=100,
        breed_enabled=True,
        max_population=MAX_POPULATION,
        verbose=True,
    )

    populations = [s.population for s in snapshots]
    avg_pop = float(np.mean(populations))
    pop_std = float(np.std(populations))
    extinctions = sum(1 for p in populations if p == 0)

    final_creatures = list(world.creatures.values())
    per_gene_div = compute_per_gene_diversity(final_creatures)
    gene_diversity = round(float(np.mean(list(per_gene_div.values()))), 4) if per_gene_div else 0.0
    hausdorff_div = compute_hausdorff_diversity(final_creatures)
    hausdorff_mean = round(float(np.mean(list(hausdorff_div.values()))), 4) if hausdorff_div else 0.0

    max_gen = max((c.generation for c in final_creatures), default=0)

    avg_profile = {}
    if final_creatures:
        for sit in SITUATION_NAMES:
            vals = [c.behavior_profile.strength(sit) if c.behavior_profile else 0.3
                    for c in final_creatures]
            avg_profile[sit] = round(float(np.mean(vals)), 4)

    return {
        "avg_population": round(avg_pop, 2),
        "pop_std": round(pop_std, 2),
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "max_generation": max_gen,
        "gene_diversity": gene_diversity,
        "per_gene_diversity": per_gene_div,
        "hausdorff_mean": hausdorff_mean,
        "hausdorff_diversity": hausdorff_div,
        "extinction_events": extinctions,
        "final_population": len(final_creatures),
        "avg_behavior_profile": avg_profile,
        "population_timeline": populations,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_SEED = 42
    N_TICKS = 30000
    N_CREATURES = 30
    N_TRIALS = 5

    print("=" * 60)
    print("EVAL 5: Numeric GA Baseline Comparison")
    print("=" * 60)
    print(f"Ticks per trial: {N_TICKS}, Starting pop: {N_CREATURES}")
    print(f"Trials per condition: {N_TRIALS}")

    conditions = {
        "BEAR (semantic)": [],
        "Numeric GA": [],
    }

    for trial in range(N_TRIALS):
        seed = BASE_SEED + trial * 1000

        print(f"\n{'='*50}")
        print(f"BEAR trial {trial+1} (seed={seed})")
        print(f"{'='*50}")
        result = run_bear_condition(seed, N_TICKS, N_CREATURES)
        result["trial"] = trial
        conditions["BEAR (semantic)"].append(result)

        print(f"\n{'='*50}")
        print(f"Numeric GA trial {trial+1} (seed={seed})")
        print(f"{'='*50}")
        result = run_ga_simulation(N_CREATURES, N_TICKS, seed)
        result["trial"] = trial
        conditions["Numeric GA"].append(result)

    # Aggregate
    summary = {}
    confidence_intervals = {}
    for label, trials in conditions.items():
        agg = {"condition": label, "n_trials": len(trials)}
        label_cis = {}
        for metric in ["avg_population", "pop_std", "total_births", "total_deaths",
                       "max_generation", "gene_diversity", "hausdorff_mean", "extinction_events"]:
            vals = [t[metric] for t in trials]
            n = len(vals)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            agg[f"{metric}_mean"] = round(mean_val, 3)
            agg[f"{metric}_std"] = round(float(np.std(vals)), 3)
            if n > 1 and std_val > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
                label_cis[metric] = [round(float(ci_low), 4), round(float(ci_high), 4)]
            else:
                label_cis[metric] = [round(mean_val, 4), round(mean_val, 4)]
        confidence_intervals[label] = label_cis
        summary[label] = agg

    # Statistical tests
    stat_tests = {}
    for metric in ["gene_diversity", "hausdorff_mean", "max_generation", "avg_population", "total_births"]:
        bear_vals = [t[metric] for t in conditions["BEAR (semantic)"]]
        ga_vals = [t[metric] for t in conditions["Numeric GA"]]
        _, t_p = scipy_stats.ttest_ind(bear_vals, ga_vals)
        _, mw_p = scipy_stats.mannwhitneyu(bear_vals, ga_vals, alternative="two-sided")
        _, sw_bear_p = scipy_stats.shapiro(bear_vals)
        _, sw_ga_p = scipy_stats.shapiro(ga_vals)
        both_normal = bool(sw_bear_p > 0.05 and sw_ga_p > 0.05)
        stat_tests[metric] = {
            "bear_mean": round(float(np.mean(bear_vals)), 4),
            "ga_mean": round(float(np.mean(ga_vals)), 4),
            "t_p_value": round(float(t_p), 4),
            "mannwhitney_p_value": round(float(mw_p), 4),
            "both_normal": both_normal,
            "recommended_test": "t-test" if both_normal else "Mann-Whitney",
        }

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    for label, agg in summary.items():
        print(f"\n{label}:")
        for k, v in agg.items():
            if k not in ("condition", "n_trials"):
                print(f"  {k}: {v}")

    print("\nStatistical Tests (BEAR vs Numeric GA):")
    print(f"  {'Metric':<20s} {'BEAR':>8s} {'GA':>8s} {'t p':>8s} {'M-W p':>8s} {'Normal?':>8s}")
    print(f"  {'-'*60}")
    for metric, test in stat_tests.items():
        use_p = test['t_p_value'] if test['both_normal'] else test['mannwhitney_p_value']
        sig = "*" if use_p < 0.05 else ""
        print(f"  {metric:<20s} {test['bear_mean']:>8.4f} {test['ga_mean']:>8.4f} "
              f"{test['t_p_value']:>8.4f} {test['mannwhitney_p_value']:>8.4f} "
              f"{'yes' if test['both_normal'] else 'no':>8s} {sig}")

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
    for metric in ["avg_population", "gene_diversity", "hausdorff_mean",
                    "max_generation", "total_births"]:
        bear_vals = [t[metric] for t in conditions["BEAR (semantic)"]]
        ga_vals = [t[metric] for t in conditions["Numeric GA"]]
        _run_statistical_tests(bear_vals, ga_vals, "BEAR", "GA", metric)

    # Plot
    try:
        _plot_comparison(conditions, summary)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Save
    output = {
        "parameters": {
            "n_ticks": N_TICKS,
            "n_creatures": N_CREATURES,
            "n_trials": N_TRIALS,
            "base_seed": BASE_SEED,
        },
        "summary": summary,
        "confidence_intervals_95": confidence_intervals,
        "statistical_tests": stat_tests,
        "trials": {k: [
            {kk: vv for kk, vv in t.items() if kk != "population_timeline"}
            for t in v
        ] for k, v in conditions.items()},
    }

    results_path = OUT_DIR / "eval5_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


def _plot_comparison(all_results, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    conds = list(all_results.keys())
    colors = ["#2196F3", "#FF9800"]

    # Panel 1: Population over time (trial 1)
    ax = axes[0][0]
    for idx, label in enumerate(conds):
        trial = all_results[label][0]
        pops = trial["population_timeline"]
        ticks = list(range(0, len(pops) * 100, 100))
        ax.plot(ticks, pops, label=label, color=colors[idx], linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Population")
    ax.set_title("Population Over Time (Trial 1)")
    ax.legend(fontsize=8)

    # Panel 2: Avg population
    ax = axes[0][1]
    x = np.arange(len(conds))
    means = [summary[c]["avg_population_mean"] for c in conds]
    stds = [summary[c]["avg_population_std"] for c in conds]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Avg Population")
    ax.set_title("Average Population (mean ± std)")

    # Panel 3: Max generation
    ax = axes[0][2]
    means = [summary[c]["max_generation_mean"] for c in conds]
    stds = [summary[c]["max_generation_std"] for c in conds]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Max Generation")
    ax.set_title("Generational Depth Reached")

    # Panel 4: Births vs deaths
    ax = axes[1][0]
    width = 0.35
    births = [summary[c]["total_births_mean"] for c in conds]
    deaths = [summary[c]["total_deaths_mean"] for c in conds]
    b_std = [summary[c]["total_births_std"] for c in conds]
    d_std = [summary[c]["total_deaths_std"] for c in conds]
    ax.bar(x - width/2, births, width, label="Births", color="#4CAF50", alpha=0.8, yerr=b_std)
    ax.bar(x + width/2, deaths, width, label="Deaths", color="#F44336", alpha=0.8, yerr=d_std)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Total Births vs Deaths")
    ax.legend()

    # Panel 5: Gene diversity
    ax = axes[1][1]
    means = [summary[c]["gene_diversity_mean"] for c in conds]
    stds = [summary[c]["gene_diversity_std"] for c in conds]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Mean Per-Gene Cosine Distance")
    ax.set_title("Gene Diversity")

    # Panel 6: Population stability
    ax = axes[1][2]
    means = [summary[c]["pop_std_mean"] for c in conds]
    stds = [summary[c]["pop_std_std"] for c in conds]
    ax.bar(x, means, 0.5, color=colors, alpha=0.8, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("Population Std Dev")
    ax.set_title("Population Stability (lower = more stable)")

    plt.suptitle("BEAR Semantic Genetics vs Numeric GA Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = OUT_DIR / "eval5_ga_baseline.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
