#!/usr/bin/env python3
"""Evaluation 1: Multi-generational population dynamics.

Runs the Evolutionary Ecosystem for 200+ generations under natural epoch cycling.
Records per-generation behavior profile averages, gene diversity, and survival
rates to demonstrate that BEAR-driven selection pressure produces measurable
population-level behavioral adaptation.

Statistical testing: runs N_TRIALS independent simulations, fits a linear
regression slope per situation per trial, then reports one-sample t-test on
whether the mean slope differs from zero.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — no LLM calls)

Parameters:
- 200,000 ticks per trial, 6 starting creatures, max population 14
- 5 trials (seeds: 42, 1042, 2042, 3042, 4042)

Outputs:
- eval1_results.json   — Raw data + statistical tests
- eval1_dynamics.png   — Population dynamics chart
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Add repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    PopulationTracker,
    get_embedder,
    make_world,
    profile_to_vector,
    run_simulation,
    stats_to_vector,
)
from evolutionary_ecosystem.server.gene_engine import GENE_CATEGORIES

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


def compute_per_gene_diversity(creatures: list) -> dict[str, float]:
    """Mean pairwise cosine distance per gene category."""
    if len(creatures) < 2:
        return {cat: 0.0 for cat in GENE_CATEGORIES}
    embedder = get_embedder()
    result = {}
    for cat in GENE_CATEGORIES:
        vecs = []
        for c in creatures:
            text = c.genes.get(cat, "")
            if text:
                vecs.append(embedder.embed_single(text))
        if len(vecs) < 2:
            result[cat] = 0.0
            continue
        distances = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-9))
                distances.append(1.0 - sim)
        result[cat] = round(float(np.mean(distances)), 4)
    return result


def run_trial(seed: int, n_ticks: int, n_creatures: int, max_pop: int,
              snapshot_interval: int, trial_num: int) -> dict:
    """Run one trial and return per-generation profiles + metrics."""
    print(f"\n{'='*50}")
    print(f"Trial {trial_num + 1} (seed={seed})")
    print(f"{'='*50}")

    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng, max_population=max_pop)
    tracker = PopulationTracker(history_length=n_ticks // snapshot_interval + 100)

    generation_profiles: dict[int, list[list[float]]] = defaultdict(list)

    for c in world.creatures.values():
        generation_profiles[c.generation].append(profile_to_vector(c.behavior_profile))

    def on_birth(child, world, tick_num):
        generation_profiles[child.generation].append(profile_to_vector(child.behavior_profile))

    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=snapshot_interval,
        breed_enabled=True,
        max_population=max_pop,
        verbose=True,
        on_birth=on_birth,
    )

    # Per-generation averages
    gen_avg_behavior = {}
    for gen, profiles in sorted(generation_profiles.items()):
        arr = np.array(profiles)
        gen_avg_behavior[gen] = {
            SITUATION_NAMES[i]: round(float(arr[:, i].mean()), 4)
            for i in range(len(SITUATION_NAMES))
        }

    # Gene diversity
    final_creatures = list(world.creatures.values())
    per_gene_div = compute_per_gene_diversity(final_creatures)
    gene_diversity = round(float(np.mean(list(per_gene_div.values()))), 4)
    max_gen = max(c.generation for c in final_creatures) if final_creatures else 0

    # Compute slope per situation (linear regression: generation -> profile value)
    slopes = {}
    gens = sorted(gen_avg_behavior.keys())
    if len(gens) >= 10:
        x = np.array([int(g) for g in gens], dtype=float)
        for sit in SITUATION_NAMES:
            y = np.array([gen_avg_behavior[g].get(sit, 0.0) for g in gens])
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
            slopes[sit] = round(float(slope), 8)

    print(f"  Generations: {max_gen}, Births: {world.total_births}, "
          f"Deaths: {world.total_deaths}")
    print(f"  Gene diversity: {gene_diversity:.4f}")
    if slopes:
        print(f"  Slopes: { {s: f'{v:.6f}' for s, v in slopes.items()} }")

    return {
        "trial": trial_num,
        "seed": seed,
        "max_generation": max_gen,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "final_population": len(final_creatures),
        "gene_diversity": gene_diversity,
        "per_gene_diversity": per_gene_div,
        "gen0_profile": gen_avg_behavior.get(0, gen_avg_behavior.get(min(gens), {})),
        "final_gen_profile": gen_avg_behavior.get(max_gen, {}),
        "slopes": slopes,
        "per_generation_behavior": {str(k): v for k, v in gen_avg_behavior.items()},
        "population_timeline": [s.population for s in snapshots],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_SEED = 42
    N_TICKS = 200000
    N_CREATURES = 30
    MAX_POP = 50
    N_TRIALS = 5
    SNAPSHOT_INTERVAL = 100

    print("=" * 60)
    print("EVAL 1: Multi-generational Population Dynamics")
    print("=" * 60)
    print(f"Ticks: {N_TICKS}, Starting pop: {N_CREATURES}, Max pop: {MAX_POP}")
    print(f"Trials: {N_TRIALS}, Snapshot interval: {SNAPSHOT_INTERVAL}")
    print()

    trials = []
    for i in range(N_TRIALS):
        seed = BASE_SEED + i * 1000
        result = run_trial(seed, N_TICKS, N_CREATURES, MAX_POP, SNAPSHOT_INTERVAL, i)
        trials.append(result)

    # Aggregate metrics
    agg = {}
    confidence_intervals = {}
    for metric in ["max_generation", "total_births", "total_deaths",
                   "final_population", "gene_diversity"]:
        vals = [t[metric] for t in trials]
        n = len(vals)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        agg[f"{metric}_mean"] = round(mean_val, 3)
        agg[f"{metric}_std"] = round(float(np.std(vals)), 3)
        if n > 1 and std_val > 0:
            ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
            confidence_intervals[metric] = [round(float(ci_low), 4), round(float(ci_high), 4)]
        else:
            confidence_intervals[metric] = [round(mean_val, 4), round(mean_val, 4)]

    # Statistical test: one-sample t-test on slopes per situation
    # H0: mean slope = 0 (no directional evolution)
    stat_tests = {}
    for sit in SITUATION_NAMES:
        slope_vals = [t["slopes"].get(sit, 0.0) for t in trials if t["slopes"]]
        if len(slope_vals) >= 3:
            mean_slope = float(np.mean(slope_vals))
            std_slope = float(np.std(slope_vals, ddof=1))
            t_stat, p_value = scipy_stats.ttest_1samp(slope_vals, 0.0)
            n_slopes = len(slope_vals)
            if n_slopes > 1 and std_slope > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n_slopes - 1, loc=mean_slope, scale=std_slope / np.sqrt(n_slopes))
                slope_ci = [round(float(ci_low), 8), round(float(ci_high), 8)]
            else:
                slope_ci = [round(mean_slope, 8), round(mean_slope, 8)]
            stat_tests[sit] = {
                "mean_slope": round(mean_slope, 8),
                "std_slope": round(std_slope, 8),
                "ci_95": slope_ci,
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant_005": bool(p_value < 0.05),
                "n_trials": len(slope_vals),
            }

    # Print results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    for k, v in agg.items():
        print(f"  {k}: {v}")

    print("\nStatistical Tests (one-sample t-test, H0: slope = 0):")
    print(f"  {'Situation':<15s} {'Mean Slope':>12s} {'t-stat':>8s} {'p-value':>8s} {'Sig.':>5s}")
    print(f"  {'-'*50}")
    for sit, test in stat_tests.items():
        sig = "*" if test["significant_005"] else ""
        print(f"  {sit:<15s} {test['mean_slope']:>12.6f} {test['t_statistic']:>8.4f} "
              f"{test['p_value']:>8.4f} {sig:>5s}")

    # Save results
    output = {
        "parameters": {
            "n_ticks": N_TICKS,
            "n_creatures": N_CREATURES,
            "max_population": MAX_POP,
            "n_trials": N_TRIALS,
            "base_seed": BASE_SEED,
            "snapshot_interval": SNAPSHOT_INTERVAL,
        },
        "summary": agg,
        "confidence_intervals_95": confidence_intervals,
        "statistical_tests": stat_tests,
        "trials": trials,
    }

    # Remove large arrays before saving
    for t in output["trials"]:
        t.pop("population_timeline", None)
        t.pop("per_generation_behavior", None)

    results_path = OUT_DIR / "eval1_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate chart
    try:
        _plot_dynamics(trials, stat_tests)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_dynamics(trials, stat_tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Per-situation slopes across trials (box plot)
    ax1 = axes[0]
    situations = list(stat_tests.keys())
    slope_data = []
    for sit in situations:
        slopes = [t["slopes"].get(sit, 0.0) for t in trials if t["slopes"]]
        slope_data.append(slopes)

    bp = ax1.boxplot(slope_data, labels=[s[:8] for s in situations], patch_artist=True)
    colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
              "#FF9F40", "#C9CBCF", "#66BB6A", "#AB47BC"]
    for patch, color in zip(bp["boxes"], colors[:len(situations)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Slope (per generation)")
    ax1.set_title("Behavioral Profile Slopes Across Trials (positive = increasing over generations)")

    # Add significance markers
    for i, sit in enumerate(situations):
        test = stat_tests[sit]
        if test["significant_005"]:
            ax1.text(i + 1, max(slope_data[i]) * 1.1, "*", ha="center",
                    fontsize=14, fontweight="bold", color="red")

    # Panel 2: Gen 0 vs final generation profile comparison (mean across trials)
    ax2 = axes[1]
    x = np.arange(len(SITUATION_NAMES))
    width = 0.35
    gen0_vals = []
    final_vals = []
    for sit in SITUATION_NAMES:
        gen0_vals.append(float(np.mean([t["gen0_profile"].get(sit, 0) for t in trials])))
        final_vals.append(float(np.mean([t["final_gen_profile"].get(sit, 0) for t in trials])))

    ax2.bar(x - width/2, gen0_vals, width, label="Gen 0", color="#4CAF50", alpha=0.8)
    ax2.bar(x + width/2, final_vals, width, label="Final Gen", color="#2196F3", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[:8] for s in SITUATION_NAMES], fontsize=9)
    ax2.set_ylabel("Mean Behavior Strength")
    ax2.set_title("Gen 0 vs Final Generation Behavior Profiles (mean across trials)")
    ax2.legend()

    plt.tight_layout()
    chart_path = OUT_DIR / "eval1_dynamics.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
