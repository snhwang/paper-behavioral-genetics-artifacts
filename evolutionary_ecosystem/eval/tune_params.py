#!/usr/bin/env python3
"""Auto-tune eval parameters for population stability.

Runs short trials, adjusts parameters automatically until the population
is stable (no extinctions, avg pop > 8, births > 20).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import random

N_TICKS = 50000
N_CREATURES = 6
MAX_POP = 16
SEED = 42

# Starting parameters
params = {
    "BASE_METABOLISM": 0.008,
    "STARVATION_DMG": 0.3,
    "FOOD_SPAWN_BASE": 0.30,
    "MAX_FOOD": 50,
    "FOOD_ENERGY": 35.0,
    "FOOD_PICKUP_RANGE": 1.2,
    "BREED_COOLDOWN": 30.0,
    "BREED_HAPPINESS": 40.0,
    "BREED_DISTANCE": 3.5,
    "MAX_AGE_MIN": 400.0,
    "MAX_AGE_MAX": 600.0,
    "PREDATOR_SPAWN_INTERVAL": 300.0,
    "PREDATOR_ATTACK_DAMAGE": 8.0,
    "WEATHER_DAMAGE": 0.05,
    "MAX_WEATHER_SEVERITY": 0.3,
    "MIN_FOOD_MULTIPLIER": 0.8,
}


def run_trial(params, seed):
    """Run one trial, return stats dict."""
    # Must re-import to reset module state
    from examples.evolutionary_ecosystem.server import sim as sim_mod
    from examples.evolutionary_ecosystem.server import epochs as epochs_mod
    from examples.evolutionary_ecosystem.eval.harness import (
        make_world, run_simulation, PopulationTracker, GENE_BANK,
    )

    # Apply sim params
    for k in ["BASE_METABOLISM", "STARVATION_DMG", "FOOD_SPAWN_BASE", "MAX_FOOD",
              "FOOD_ENERGY", "FOOD_PICKUP_RANGE", "BREED_COOLDOWN", "BREED_HAPPINESS",
              "BREED_DISTANCE", "MAX_AGE_MIN", "MAX_AGE_MAX",
              "PREDATOR_SPAWN_INTERVAL", "PREDATOR_ATTACK_DAMAGE"]:
        setattr(sim_mod, k, params[k])
    sim_mod.MAX_POPULATION = MAX_POP

    # Apply epoch params
    epochs_mod.WEATHER_DAMAGE = params["WEATHER_DAMAGE"]
    for ep in epochs_mod.EPOCHS:
        ep.weather_severity = min(ep.weather_severity, params["MAX_WEATHER_SEVERITY"])
        ep.food_multiplier = max(ep.food_multiplier, params["MIN_FOOD_MULTIPLIER"])

    rng = random.Random(seed)
    world = make_world(n_creatures=N_CREATURES, rng=rng, max_population=MAX_POP)
    world.autonomous_breeding = True

    tracker = PopulationTracker(history_length=N_TICKS // 100 + 10)
    snapshots = run_simulation(
        world, N_TICKS, rng, tracker=tracker,
        snapshot_interval=100, max_population=MAX_POP,
        verbose=False,
    )

    populations = [s.population for s in snapshots]
    avg_pop = sum(populations) / len(populations) if populations else 0
    min_pop = min(populations) if populations else 0
    extinctions = sum(1 for i in range(1, len(populations))
                      if populations[i-1] == 0 and populations[i] > 0)

    return {
        "births": world.total_births,
        "deaths": world.total_deaths,
        "final_pop": len(world.creatures),
        "avg_pop": avg_pop,
        "min_pop": min_pop,
        "extinctions": extinctions,
        "max_gen": max((c.generation for c in world.creatures.values()), default=0),
    }


def is_stable(stats):
    return (stats["births"] > 20
            and stats["avg_pop"] >= 8
            and stats["extinctions"] == 0)


def main():
    # Adjustments to try, in order of priority
    adjustments = [
        # (param, direction, step, limit, reason)
        ("PREDATOR_ATTACK_DAMAGE", -0.5, 2.0, "reduce predator lethality"),
        ("PREDATOR_SPAWN_INTERVAL", +50.0, 600.0, "less frequent predators"),
        ("WEATHER_DAMAGE", -0.01, 0.01, "reduce weather damage"),
        ("MAX_WEATHER_SEVERITY", -0.05, 0.1, "fewer storms"),
        ("STARVATION_DMG", -0.05, 0.05, "slower starvation"),
        ("FOOD_SPAWN_BASE", +0.05, 0.60, "more food"),
        ("MAX_AGE_MIN", +50.0, 800.0, "longer lifespan"),
        ("MAX_AGE_MAX", +50.0, 1000.0, "longer max lifespan"),
        ("BASE_METABOLISM", -0.001, 0.003, "less energy drain"),
        ("BREED_COOLDOWN", -5.0, 10.0, "faster breeding"),
        ("BREED_DISTANCE", +0.5, 6.0, "wider breed range"),
        ("MIN_FOOD_MULTIPLIER", +0.05, 1.0, "more food in harsh epochs"),
    ]

    print("=" * 60)
    print("  Auto-tuning eval parameters")
    print("=" * 60)

    iteration = 0
    max_iterations = 30

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Run 1 seed for speed during tuning
        stats = run_trial(params, 42 + iteration)
        print(f"  Births: {stats['births']}  Avg pop: {stats['avg_pop']:.1f}  "
              f"Extinctions: {stats['extinctions']}  Max gen: {stats['max_gen']}")

        if is_stable(stats):
            # Validate with 3 seeds
            print("  Looks good — validating with 3 seeds...")
            all_stats = [stats]
            for seed in [1042, 2042]:
                s = run_trial(params, seed)
                all_stats.append(s)
                print(f"    Seed {seed}: births={s['births']} avg_pop={s['avg_pop']:.1f} ext={s['extinctions']}")

            avg_births = sum(s["births"] for s in all_stats) / 3
            avg_pop = sum(s["avg_pop"] for s in all_stats) / 3
            avg_ext = sum(s["extinctions"] for s in all_stats) / 3
            avg_gen = sum(s["max_gen"] for s in all_stats) / 3

            if all(is_stable(s) for s in all_stats):
                print("\n  STABLE across all seeds!")
                break
            else:
                print("  Failed validation — continuing tuning...")
        else:
            all_stats = [stats]
            avg_births = stats["births"]
            avg_pop = stats["avg_pop"]
            avg_ext = stats["extinctions"]
            avg_gen = stats["max_gen"]

        # Find worst problem and adjust
        adjusted = False
        for param, step, limit, reason in adjustments:
            current = params[param]
            if step > 0 and current >= limit:
                continue
            if step < 0 and current <= limit:
                continue

            new_val = current + step
            if step > 0:
                new_val = min(new_val, limit)
            else:
                new_val = max(new_val, limit)

            if new_val != current:
                params[param] = new_val
                print(f"  Adjusting {param}: {current} -> {new_val} ({reason})")
                adjusted = True
                break

        if not adjusted:
            print("\n  All parameters at limits — cannot improve further")
            break

    print("\n" + "=" * 60)
    print("  FINAL PARAMETERS")
    print("=" * 60)
    print()
    print("  # Paste into harness.py patch_sim_for_eval():")
    for k, v in params.items():
        if k in ("WEATHER_DAMAGE", "MAX_WEATHER_SEVERITY", "MIN_FOOD_MULTIPLIER"):
            continue
        fmt = f"{v}" if isinstance(v, int) else f"{v:.3f}" if v < 1 else f"{v:.1f}"
        print(f"  sim_mod.{k} = {fmt}")
    print(f"  epochs_mod.WEATHER_DAMAGE = {params['WEATHER_DAMAGE']}")
    print(f"  # weather_severity cap: {params['MAX_WEATHER_SEVERITY']}")
    print(f"  # food_multiplier floor: {params['MIN_FOOD_MULTIPLIER']}")

    print("\n  Final stats (3-seed average):")
    print(f"    Births: {avg_births:.0f}")
    print(f"    Avg pop: {avg_pop:.1f}")
    print(f"    Extinctions: {avg_ext:.1f}")
    print(f"    Max generation: {avg_gen:.0f}")


if __name__ == "__main__":
    main()
