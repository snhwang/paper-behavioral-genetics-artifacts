#!/usr/bin/env python3
"""Evaluation 4: Epoch-driven phenotype shift.

Runs identical starting populations through each of the 5 epochs for a
fixed duration and compares final population behavior profiles. If the
framework works as claimed:
- Famine populations should show higher food_seeking
- Ice Age populations should show higher survival/climate
- Predator Bloom populations should show higher combat/territory
- Abundance populations should show balanced or breeding-biased profiles
- Expansion populations should favor exploration/breeding

Uses one-way ANOVA on behavior profile dimensions across epoch conditions
to test for statistically significant differences.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: None (headless mode — no LLM calls)

Parameters:
- 30,000 ticks per trial, 6 starting creatures
- 5 epochs × 3 trials each (15 total runs)
- Seeds: 42, 1042, 2042

Outputs:
- eval4_results.json      — Raw data per epoch
- eval4_epoch_shift.png   — Radar charts + comparison
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
    GENE_BANK,
    SITUATION_NAMES,
    PopulationTracker,
    make_world,
    profile_to_vector,
    run_simulation,
)
from evolutionary_ecosystem.server.epochs import EPOCHS
from evolutionary_ecosystem.server.sim import MAX_POPULATION

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


def run_epoch_condition(
    epoch_index: int,
    seed: int,
    n_ticks: int,
    n_creatures: int,
    lock_epoch: bool = True,
) -> dict:
    """Run simulation under a fixed epoch and collect final behavior profiles."""
    epoch = EPOCHS[epoch_index]
    label = epoch.name

    print(f"\n  Running epoch: {label} (food={epoch.food_multiplier}x, "
          f"weather={epoch.weather_severity}, aggression=+{epoch.aggression_bonus})")

    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng, epoch_index=epoch_index)

    # Lock epoch to prevent cycling
    if lock_epoch:
        from evolutionary_ecosystem.server import sim as sim_mod
        original_epoch_check = sim_mod._epoch_check
        sim_mod._epoch_check = lambda w: None  # no-op

    tracker = PopulationTracker(history_length=n_ticks // 200 + 50)

    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=200,
        breed_enabled=True,
        max_population=MAX_POPULATION,
        verbose=False,
    )

    # Restore epoch check
    if lock_epoch:
        sim_mod._epoch_check = original_epoch_check

    # Collect final population behavior profiles
    final_creatures = list(world.creatures.values())
    final_profiles = [profile_to_vector(c.behavior_profile) for c in final_creatures
                      if c.behavior_profile is not None]

    # Per-situation averages
    avg_profile = {}
    per_situation_values: dict[str, list[float]] = defaultdict(list)
    if final_profiles:
        arr = np.array(final_profiles)
        for i, sit in enumerate(SITUATION_NAMES):
            avg_profile[sit] = round(float(arr[:, i].mean()), 4)
            per_situation_values[sit] = arr[:, i].tolist()
    else:
        avg_profile = {s: 0.0 for s in SITUATION_NAMES}

    # Track behavior trajectory over time
    behavior_trajectory = []
    for snap in snapshots:
        behavior_trajectory.append({
            "tick": snap.tick,
            "avg_behavior": {k: round(v, 4) for k, v in snap.avg_behavior.items()},
            "population": snap.population,
        })

    max_gen = max((c.generation for c in final_creatures), default=0)

    result = {
        "epoch": label,
        "epoch_index": epoch_index,
        "food_multiplier": epoch.food_multiplier,
        "weather_severity": epoch.weather_severity,
        "aggression_bonus": epoch.aggression_bonus,
        "final_population": len(final_creatures),
        "max_generation": max_gen,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "avg_behavior_profile": avg_profile,
        "per_situation_values": {k: [round(v, 4) for v in vs]
                                 for k, vs in per_situation_values.items()},
        "behavior_trajectory": behavior_trajectory,
    }

    print(f"    Final pop: {len(final_creatures)}, Gen: {max_gen}, "
          f"Births: {world.total_births}, Deaths: {world.total_deaths}")
    for sit in ["food_seeking", "combat", "survival", "breeding"]:
        print(f"    {sit}: {avg_profile.get(sit, 0):.4f}")

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASE_SEED = 42
    N_TICKS = 30000  # ~25 min simulated (enough for selection)
    N_CREATURES = 30
    N_TRIALS = 3

    print("=" * 60)
    print("EVAL 4: Epoch-Driven Phenotype Shift")
    print("=" * 60)
    print(f"Ticks per trial: {N_TICKS}, Starting pop: {N_CREATURES}")
    print(f"Trials per epoch: {N_TRIALS}")
    print(f"Epochs: {[e.name for e in EPOCHS]}")

    all_results: dict[str, list[dict]] = defaultdict(list)

    for epoch_idx, epoch in enumerate(EPOCHS):
        print(f"\n{'='*50}")
        print(f"EPOCH: {epoch.name}")
        print(f"{'='*50}")

        for trial in range(N_TRIALS):
            seed = BASE_SEED + epoch_idx * 100 + trial * 1000
            result = run_epoch_condition(
                epoch_index=epoch_idx,
                seed=seed,
                n_ticks=N_TICKS,
                n_creatures=N_CREATURES,
                lock_epoch=True,
            )
            result["trial"] = trial
            all_results[epoch.name].append(result)

    # Aggregate per epoch
    summary: dict[str, dict] = {}
    confidence_intervals: dict[str, dict] = {}
    for epoch_name, trials in all_results.items():
        agg = {"epoch": epoch_name, "n_trials": len(trials)}
        epoch_cis: dict[str, object] = {}

        for metric in ["final_population", "max_generation", "total_births", "total_deaths"]:
            vals = [t[metric] for t in trials]
            n = len(vals)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            agg[f"{metric}_mean"] = round(mean_val, 2)
            if n > 1 and std_val > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
                epoch_cis[metric] = [round(float(ci_low), 4), round(float(ci_high), 4)]
            else:
                epoch_cis[metric] = [round(mean_val, 4), round(mean_val, 4)]

        # Average behavior profile across trials (with CIs)
        avg_profile: dict[str, float] = {}
        profile_cis: dict[str, list[float]] = {}
        for sit in SITUATION_NAMES:
            vals = [t["avg_behavior_profile"].get(sit, 0.3) for t in trials]
            n = len(vals)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            avg_profile[sit] = round(mean_val, 4)
            if n > 1 and std_val > 0:
                ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
                profile_cis[sit] = [round(float(ci_low), 4), round(float(ci_high), 4)]
            else:
                profile_cis[sit] = [round(mean_val, 4), round(mean_val, 4)]
        epoch_cis["behavior_profile"] = profile_cis

        agg["avg_behavior_profile"] = avg_profile
        confidence_intervals[epoch_name] = epoch_cis
        summary[epoch_name] = agg

    # ANOVA per situation across epochs
    print("\n" + "=" * 60)
    print("ONE-WAY ANOVA: Behavior Profile by Epoch")
    print("=" * 60)

    anova_results = {}
    for sit in SITUATION_NAMES:
        groups = []
        for epoch_name in [e.name for e in EPOCHS]:
            trials = all_results[epoch_name]
            # Pool all individual creature values across trials
            vals = []
            for t in trials:
                vals.extend(t["per_situation_values"].get(sit, []))
            groups.append(vals)

        # One-way ANOVA (scipy)
        if all(len(g) >= 2 for g in groups):
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            f_stat = float(f_stat)
            p_value = float(p_value)
        else:
            f_stat = 0.0
            p_value = 1.0

        anova_results[sit] = {
            "F_statistic": round(f_stat, 4),
            "p_value": round(p_value, 4),
            "per_epoch_mean": {e.name: round(float(np.mean(
                [v for t in all_results[e.name]
                 for v in t["per_situation_values"].get(sit, [0.3])]
            )), 4) for e in EPOCHS},
        }

        sig = "*" if p_value < 0.05 else ""
        print(f"  {sit:>15s}: F={f_stat:>8.4f}  p={p_value:.4f} {sig}")
        for e in EPOCHS:
            mean = anova_results[sit]["per_epoch_mean"][e.name]
            print(f"    {e.name:>17s}: {mean:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Average Behavior Profile by Epoch")
    print("=" * 60)
    header = f"{'Epoch':>17s}"
    for sit in SITUATION_NAMES:
        header += f" {sit[:8]:>8s}"
    print(header)
    print("-" * len(header))
    for epoch_name, agg in summary.items():
        row = f"{epoch_name:>17s}"
        for sit in SITUATION_NAMES:
            row += f" {agg['avg_behavior_profile'].get(sit, 0):>8.4f}"
        print(row)

    # Save results
    output = {
        "parameters": {
            "base_seed": BASE_SEED,
            "n_ticks": N_TICKS,
            "n_creatures": N_CREATURES,
            "n_trials": N_TRIALS,
        },
        "summary": summary,
        "confidence_intervals_95": confidence_intervals,
        "anova": anova_results,
        "trials": {k: v for k, v in all_results.items()},
    }

    # Trim trajectory data for file size
    for epoch_trials in output["trials"].values():
        for t in epoch_trials:
            t["behavior_trajectory"] = t["behavior_trajectory"][::5]  # keep every 5th

    results_path = OUT_DIR / "eval4_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_epoch_shift(summary, anova_results)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_epoch_shift(summary, anova_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    epoch_names = [e.name for e in EPOCHS]
    epoch_colors = {
        "abundance": "#4CAF50",
        "ice_age": "#2196F3",
        "predator_bloom": "#F44336",
        "expansion": "#FF9800",
        "famine": "#9C27B0",
    }

    # Panel 1: Radar chart
    ax1 = axes[0]
    situations = SITUATION_NAMES
    n_sits = len(situations)
    angles = np.linspace(0, 2 * np.pi, n_sits, endpoint=False).tolist()
    angles += angles[:1]

    ax1 = fig.add_subplot(121, polar=True)
    for epoch_name in epoch_names:
        profile = summary[epoch_name]["avg_behavior_profile"]
        values = [profile.get(s, 0.3) for s in situations]
        values += values[:1]
        ax1.plot(angles, values, "o-", label=epoch_name,
                color=epoch_colors[epoch_name], linewidth=2, markersize=4)
        ax1.fill(angles, values, alpha=0.1, color=epoch_colors[epoch_name])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([s.replace("_", "\n") for s in situations], fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_title("Behavior Profile by Epoch", y=1.08, fontsize=12)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    # Panel 2: Grouped bar chart with F-statistics
    ax2 = axes[1]
    x = np.arange(n_sits)
    width = 0.15
    for idx, epoch_name in enumerate(epoch_names):
        profile = summary[epoch_name]["avg_behavior_profile"]
        vals = [profile.get(s, 0.3) for s in situations]
        ax2.bar(x + idx * width, vals, width, label=epoch_name,
                color=epoch_colors[epoch_name], alpha=0.8)

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([s.replace("_", "\n") for s in situations], fontsize=8)
    ax2.set_ylabel("Mean Behavior Strength")
    ax2.set_title("Behavior Profile Comparison Across Epochs")
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1)

    # Add F-statistic annotations
    for i, sit in enumerate(situations):
        f_val = anova_results[sit]["F_statistic"]
        stars = "***" if f_val > 10 else "**" if f_val > 5 else "*" if f_val > 2 else ""
        if stars:
            ax2.annotate(f"F={f_val:.1f}{stars}",
                        xy=(i + width * 2, 0.95), fontsize=6,
                        ha="center", va="top", color="#333333")

    plt.tight_layout()
    chart_path = OUT_DIR / "eval4_epoch_shift.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
