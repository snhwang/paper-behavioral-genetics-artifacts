#!/usr/bin/env python3
"""Evaluation 2: Dual-pathway ablation study.

Compares two simulation conditions:
  A) BEAR On  (behavior profiles computed from gene text via embedding retrieval;
               tick() uses profile strengths for food-seeking, survival, combat, etc.)
  B) BEAR Off (all behavior profiles replaced with uniform 0.3 neutral scores;
               tick() has no BEAR-driven guidance — creatures behave randomly)

Both share identical breeding, mortality, and environmental mechanics.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — no LLM calls)

Parameters:
- 30,000 ticks per trial, 6 starting creatures
- 3 trials per condition (seeds: 42, 1042, 2042)

Measures: avg population, population stability, births/deaths,
max generation, behavioral diversity, extinction events.

Outputs:
- eval2_results.json   — Raw data per condition
- eval2_ablation.png   — Comparison chart
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evolutionary_ecosystem.eval.harness import (
    SITUATION_NAMES,
    PopulationTracker,
    compute_hausdorff_diversity,
    compute_per_gene_diversity,
    cosine_similarity,
    get_embedder,
    make_world,
    run_simulation,
)
from evolutionary_ecosystem.server.gene_engine import GENE_CATEGORIES

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


def run_condition(
    label: str,
    bear_disabled: bool,
    seed: int,
    n_ticks: int,
    n_creatures: int,
    max_population: int = 30,
) -> dict:
    """Run one experimental condition and collect metrics."""
    print(f"\n{'='*50}")
    print(f"Condition: {label}")
    print(f"{'='*50}")

    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng, max_population=max_population)
    world.bear_disabled = bear_disabled

    # When BEAR is off, replace all behavior profiles with uniform values
    # so movement decisions are effectively random (no gene-driven differentiation)
    if bear_disabled:
        from evolutionary_ecosystem.server.gene_engine import BehaviorProfile
        for c in world.creatures.values():
            c.behavior_profile = BehaviorProfile({})  # empty = all defaults (0.3)

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)

    # Tracking variables
    food_eaten = [0]
    lifespans: list[float] = []
    children_at_death: list[int] = []

    # Patch food pickup to count food eaten
    original_energy_vals: dict[str, float] = {}
    for c in world.creatures.values():
        original_energy_vals[c.id] = c.energy

    birth_count = [0]

    def on_birth(child, world, tick_num):
        birth_count[0] += 1
        original_energy_vals[child.id] = child.energy
        if bear_disabled:
            from evolutionary_ecosystem.server.gene_engine import BehaviorProfile
            child.behavior_profile = BehaviorProfile({})

    def on_death(cid, world, tick_num):
        # Approximate lifespan from tick
        lifespans.append(tick_num / 20.0)  # convert ticks to seconds (approx)

    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=100,
        breed_enabled=True,
        max_population=max_population,
        verbose=True,
        on_birth=on_birth,
        on_death=on_death,
    )

    # Compute metrics
    populations = [s.population for s in snapshots]
    avg_pop = float(np.mean(populations))
    pop_std = float(np.std(populations))
    min_pop = min(populations)
    max_pop = max(populations)
    extinctions = sum(1 for p in populations if p == 0)

    # Gene diversity: per-gene and aggregate (mean of per-gene)
    final_creatures = list(world.creatures.values())

    # Per-gene diversity
    per_gene_div = {}
    if len(final_creatures) >= 2:
        embedder = get_embedder()
        for cat in GENE_CATEGORIES:
            vecs = []
            for c in final_creatures:
                text = c.genes.get(cat, "")
                if text:
                    vecs.append(embedder.embed_single(text))
            if len(vecs) < 2:
                per_gene_div[cat] = 0.0
                continue
            dists = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    dists.append(1.0 - cosine_similarity(vecs[i], vecs[j]))
            per_gene_div[cat] = round(float(np.mean(dists)), 4)
    else:
        per_gene_div = {cat: 0.0 for cat in GENE_CATEGORIES}

    gene_diversity = round(float(np.mean(list(per_gene_div.values()))), 4)
    hausdorff_div = compute_hausdorff_diversity(final_creatures)
    hausdorff_mean = round(float(np.mean(list(hausdorff_div.values()))), 4)

    # Max generation reached
    max_gen = max((c.generation for c in final_creatures), default=0)

    # Average behavior profile
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
        "bear_disabled": bear_disabled,
        "avg_population": round(avg_pop, 2),
        "pop_std": round(pop_std, 2),
        "min_population": min_pop,
        "max_population": max_pop,
        "extinction_events": extinctions,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "max_generation": max_gen,
        "gene_diversity": gene_diversity,
        "per_gene_diversity": per_gene_div,
        "hausdorff_mean": hausdorff_mean,
        "hausdorff_diversity": hausdorff_div,
        "final_population": len(final_creatures),
        "avg_behavior_profile": avg_profile,
        "population_timeline": populations,
    }

    print(f"\n  Avg pop: {avg_pop:.1f} ± {pop_std:.1f}")
    print(f"  Births: {world.total_births}, Deaths: {world.total_deaths}")
    print(f"  Max generation: {max_gen}")
    print(f"  Gene diversity (mean per-gene): {gene_diversity:.4f}")
    print(f"  Per-gene diversity:")
    for cat, div in per_gene_div.items():
        print(f"    {cat}: {div:.4f}")
    print(f"  Extinctions: {extinctions}")

    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_SEED = 42
    N_TICKS = 30000  # ~25 min simulated time
    N_CREATURES = 30
    MAX_POP = 50
    N_TRIALS = 10

    print("=" * 60)
    print("EVAL 2: Dual-Pathway Ablation Study")
    print("=" * 60)
    print(f"Ticks per trial: {N_TICKS}, Starting pop: {N_CREATURES}, Max pop: {MAX_POP}")
    print(f"Trials per condition: {N_TRIALS}")

    conditions = [
        ("BEAR Fast Path (full)", False),
        ("BEAR Off (random drift)", True),
    ]

    all_results: dict[str, list[dict]] = defaultdict(list)

    for label, bear_off in conditions:
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial * 1000
            result = run_condition(
                f"{label} (trial {trial+1})",
                bear_disabled=bear_off,
                seed=seed,
                n_ticks=N_TICKS,
                n_creatures=N_CREATURES,
                max_population=MAX_POP,
            )
            result["trial"] = trial
            all_results[label].append(result)

    # Aggregate across all trials
    summary = {}
    confidence_intervals = {}
    for label, trials in all_results.items():
        agg = {
            "condition": label,
            "n_trials": len(trials),
        }
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

        # Per-gene diversity CIs
        per_gene_cis = {}
        for cat in GENE_CATEGORIES:
            vals = [t["per_gene_diversity"][cat] for t in trials]
            n = len(vals)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            if n > 1 and std_val > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
                per_gene_cis[cat] = [round(float(ci_low), 4), round(float(ci_high), 4)]
            else:
                per_gene_cis[cat] = [round(mean_val, 4), round(mean_val, 4)]
        label_cis["per_gene_diversity"] = per_gene_cis

        confidence_intervals[label] = label_cis
        summary[label] = agg

    # Statistical tests: t-test + Mann-Whitney + normality check
    stat_tests = {}
    for metric in ["gene_diversity", "hausdorff_mean", "max_generation", "avg_population", "total_births"]:
        bear_on = [t[metric] for t in all_results["BEAR Fast Path (full)"]]
        bear_off = [t[metric] for t in all_results["BEAR Off (random drift)"]]
        t_stat, t_p = scipy_stats.ttest_ind(bear_on, bear_off)
        mw_stat, mw_p = scipy_stats.mannwhitneyu(bear_on, bear_off, alternative="two-sided")
        sw_on_stat, sw_on_p = scipy_stats.shapiro(bear_on)
        sw_off_stat, sw_off_p = scipy_stats.shapiro(bear_off)
        both_normal = bool(sw_on_p > 0.05 and sw_off_p > 0.05)
        stat_tests[metric] = {
            "bear_on_mean": round(float(np.mean(bear_on)), 4),
            "bear_off_mean": round(float(np.mean(bear_off)), 4),
            "t_p_value": round(float(t_p), 4),
            "mannwhitney_p_value": round(float(mw_p), 4),
            "shapiro_on_p": round(float(sw_on_p), 4),
            "shapiro_off_p": round(float(sw_off_p), 4),
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

    # Per-gene diversity tests
    per_gene_tests = {}
    for cat in GENE_CATEGORIES:
        on_vals = [t["per_gene_diversity"][cat] for t in all_results["BEAR Fast Path (full)"]]
        off_vals = [t["per_gene_diversity"][cat] for t in all_results["BEAR Off (random drift)"]]
        _, t_p = scipy_stats.ttest_ind(on_vals, off_vals)
        _, mw_p = scipy_stats.mannwhitneyu(on_vals, off_vals, alternative="two-sided")
        _, sw_on_p = scipy_stats.shapiro(on_vals)
        _, sw_off_p = scipy_stats.shapiro(off_vals)
        both_normal = bool(sw_on_p > 0.05 and sw_off_p > 0.05)
        per_gene_tests[cat] = {
            "bear_on_mean": round(float(np.mean(on_vals)), 4),
            "bear_off_mean": round(float(np.mean(off_vals)), 4),
            "t_p_value": round(float(t_p), 4),
            "mannwhitney_p_value": round(float(mw_p), 4),
            "both_normal": both_normal,
            "recommended_test": "t-test" if both_normal else "Mann-Whitney",
        }
    stat_tests["per_gene_diversity"] = per_gene_tests

    print("\nStatistical Tests (BEAR On vs Off):")
    print(f"  {'Metric':<20s} {'On':>8s} {'Off':>8s} {'t p':>8s} {'M-W p':>8s} {'Normal?':>8s} {'Use':>10s}")
    print(f"  {'-'*72}")
    for metric, test in stat_tests.items():
        if metric == "per_gene_diversity":
            continue
        use_p = test['t_p_value'] if test['both_normal'] else test['mannwhitney_p_value']
        sig = "*" if use_p < 0.05 else ""
        print(f"  {metric:<20s} {test['bear_on_mean']:>8.3f} {test['bear_off_mean']:>8.3f} "
              f"{test['t_p_value']:>8.4f} {test['mannwhitney_p_value']:>8.4f} "
              f"{'yes' if test['both_normal'] else 'no':>8s} {test['recommended_test']:>10s} {sig}")

    print(f"\nPer-Gene Diversity (BEAR On vs Off):")
    print(f"  {'Gene':<20s} {'On':>8s} {'Off':>8s} {'t p':>8s} {'M-W p':>8s} {'Normal?':>8s} {'Use':>10s}")
    print(f"  {'-'*72}")
    for cat, test in per_gene_tests.items():
        use_p = test['t_p_value'] if test['both_normal'] else test['mannwhitney_p_value']
        sig = "*" if use_p < 0.05 else ""
        print(f"  {cat:<20s} {test['bear_on_mean']:>8.4f} {test['bear_off_mean']:>8.4f} "
              f"{test['t_p_value']:>8.4f} {test['mannwhitney_p_value']:>8.4f} "
              f"{'yes' if test['both_normal'] else 'no':>8s} {test['recommended_test']:>10s} {sig}")

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
        bear_on = [t[metric] for t in all_results["BEAR Fast Path (full)"]]
        bear_off = [t[metric] for t in all_results["BEAR Off (random drift)"]]
        _run_statistical_tests(bear_on, bear_off, "BEAR-on", "BEAR-off", metric)

    # Save
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
        "trials": {k: v for k, v in all_results.items()},
    }

    # Generate chart before removing timelines
    try:
        _plot_ablation(all_results, summary)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Remove timeline arrays from trials to keep file manageable
    for label, trials in output["trials"].items():
        for t in trials:
            t.pop("population_timeline", None)

    results_path = OUT_DIR / "eval2_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


def _plot_ablation(all_results, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    conditions = list(all_results.keys())
    colors = ["#4CAF50", "#F44336"]

    # Panel 1: Population over time (one trial each)
    ax1 = axes[0][0]
    for idx, label in enumerate(conditions):
        trial = all_results[label][0]
        pops = trial["population_timeline"]
        ticks = list(range(0, len(pops) * 100, 100))
        ax1.plot(ticks, pops, label=label, color=colors[idx], linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Population")
    ax1.set_title("Population Over Time (Trial 1)")
    ax1.legend(fontsize=8)

    # Panel 2: Births and deaths comparison
    ax2 = axes[0][1]
    x = np.arange(len(conditions))
    width = 0.35
    births = [summary[c]["total_births_mean"] for c in conditions]
    deaths = [summary[c]["total_deaths_mean"] for c in conditions]
    births_std = [summary[c]["total_births_std"] for c in conditions]
    deaths_std = [summary[c]["total_deaths_std"] for c in conditions]
    ax2.bar(x - width/2, births, width, label="Births", color="#4CAF50", alpha=0.8, yerr=births_std)
    ax2.bar(x + width/2, deaths, width, label="Deaths", color="#F44336", alpha=0.8, yerr=deaths_std)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split("(")[0].strip() for c in conditions], fontsize=9)
    ax2.set_ylabel("Count")
    ax2.set_title("Total Births vs Deaths (mean ± std)")
    ax2.legend()

    # Panel 3: Gene diversity (mean per-gene cosine distance)
    ax3 = axes[1][0]
    div_means = [summary[c]["gene_diversity_mean"] for c in conditions]
    div_stds = [summary[c]["gene_diversity_std"] for c in conditions]
    ax3.bar(x, div_means, 0.5, color=colors, alpha=0.8, yerr=div_stds)
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.split("(")[0].strip() for c in conditions], fontsize=9)
    ax3.set_ylabel("Mean Per-Gene Cosine Distance")
    ax3.set_title("Final Gene Diversity")

    # Panel 4: Max generation
    ax4 = axes[1][1]
    gen_means = [summary[c]["max_generation_mean"] for c in conditions]
    gen_stds = [summary[c]["max_generation_std"] for c in conditions]
    ax4.bar(x, gen_means, 0.5, color=colors, alpha=0.8, yerr=gen_stds)
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.split("(")[0].strip() for c in conditions], fontsize=9)
    ax4.set_ylabel("Max Generation")
    ax4.set_title("Generational Depth Reached")

    plt.tight_layout()
    chart_path = OUT_DIR / "eval2_ablation.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
