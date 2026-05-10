#!/usr/bin/env python3
"""Evaluation 5b: LLM-mediated breeding in full simulation.

Extends eval5 (BEAR vs numeric GA baseline) by comparing BEAR's deterministic
text splicing breeding against LLM-mediated breeding (blend_gene + mutate_gene)
within a full simulation run.

Design:
  - Two conditions:
    1. BEAR deterministic: text splicing breeding (same as eval5)
    2. BEAR LLM: LLM-mediated breeding using breed_offspring() from gene_engine
  - Multiple trials (N_TRIALS=3) with fixed seeds for reproducibility
  - 10,000 ticks per trial (each breed event costs ~8 LLM calls)
  - Metrics: avg population, births/deaths, behavioral diversity, max generation
  - Statistical tests: paired t-test, Wilcoxon signed-rank, Cohen's d,
    Shapiro-Wilk on differences, 95% confidence intervals

The claim being validated: LLM-mediated breeding produces qualitatively
different population dynamics and behavioral diversity compared to
deterministic text splicing.

Models:
- Embedding: BAAI/bge-base-en-v1.5 (768-dim sentence-transformer)
- LLM: mlx-community/mistral-nemo-instruct-2407:3 (default, configurable)

Parameters:
- 10,000 ticks, 6 starting creatures, max population 16
- Seeds: [42, 1042, 2042]
- Mutation rate: 15%

Outputs:
- eval5b_results.json  — Per-condition simulation data, metrics, and stats
- eval5b_llm_breeding.png — Comparison chart

Usage:
    python eval5b_llm_breeding.py                          # auto-detect backend
    python eval5b_llm_breeding.py --backend local          # LM Studio
    python eval5b_llm_breeding.py --model MODEL_ID         # specific model
    python eval5b_llm_breeding.py --ticks 5000             # shorter run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from dotenv import load_dotenv; load_dotenv()

# ---------------------------------------------------------------------------
# Trial configuration
# ---------------------------------------------------------------------------

N_TRIALS = 7
SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bear import Config, EmbeddingBackend
from bear.config import LLMBackend
from bear.llm import LLM

from evolutionary_ecosystem.eval.harness import (
    GENE_BANK,
    SITUATION_NAMES,
    _NAMES,
    PopulationTracker,
    cosine_similarity,
    ensure_eval_patched,
    get_config,
    get_embedder,
    make_creature,
    make_world,
    profile_to_vector,
    run_simulation,
)
from evolutionary_ecosystem.server.gene_engine import (
    BreedRequest,
    BreedResult,
    breed_offspring,
    build_corpus,
    extract_appearance,
)
from evolutionary_ecosystem.server.sim import (
    Creature,
    Predator,
    World,
    tick,
    WORLD_W,
    WORLD_H,
    MAX_POPULATION,
    PREDATOR_SPAWN_INTERVAL,
)
from evolutionary_ecosystem.server.epochs import EPOCHS

OUT_DIR = Path(__file__).resolve().parents[2] / "results"

# ---------------------------------------------------------------------------
# LLM backend detection
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "mistral-nemo-instruct-2407"
try:
    from bear.utils import detect_local_llm_url
    DEFAULT_LOCAL_URL = detect_local_llm_url()
except ImportError:
    DEFAULT_LOCAL_URL = "http://127.0.0.1:1234/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def detect_backend(args):
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
    if backend == "anthropic":
        return LLM(backend=LLMBackend.ANTHROPIC, model=model)
    else:
        return LLM(backend=LLMBackend.OPENAI, model=model, base_url=base_url)


# ---------------------------------------------------------------------------
# BEAR deterministic condition (reuses harness)
# ---------------------------------------------------------------------------

def run_bear_deterministic(seed: int, n_ticks: int, n_creatures: int) -> dict:
    """Run BEAR condition with deterministic text splicing breeding."""
    rng = random.Random(seed)
    world = make_world(n_creatures=n_creatures, rng=rng)

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)
    snapshots = run_simulation(
        world, n_ticks, rng, tracker,
        snapshot_interval=100,
        breed_enabled=True,
        max_population=MAX_POPULATION,
        verbose=True,
    )

    return _collect_metrics(world, snapshots)


# ---------------------------------------------------------------------------
# BEAR LLM condition
# ---------------------------------------------------------------------------

def run_bear_llm(
    seed: int, n_ticks: int, n_creatures: int,
    llm: LLM,
) -> dict:
    """Run BEAR condition with LLM-mediated breeding."""
    ensure_eval_patched()
    rng = random.Random(seed)
    embedder = get_embedder()
    config = get_config()

    # Build world
    world = World(
        predator=Predator(x=0, y=0,
                          spawn_at=time.time() + PREDATOR_SPAWN_INTERVAL),
        epoch=EPOCHS[0],
        epoch_index=0,
    )
    # _check_breeding gates on world.autonomous_breeding (defaults False on
    # the World dataclass) -- without this enabled, no breed tuples are
    # ever pushed to world.breed_queue and the LLM breeding path stays
    # idle for the entire run, producing zero births. The deterministic
    # path works because harness.make_world / run_simulation set this
    # flag explicitly.
    world.autonomous_breeding = True

    name_counter = [0]
    print(f"Creating {n_creatures} creatures...")
    for i in range(n_creatures):
        genes = GENE_BANK[i % len(GENE_BANK)]
        name = _NAMES[i % len(_NAMES)]
        cid = world.next_id()
        creature = make_creature(cid, genes, name, rng)
        world.creatures[cid] = creature
        name_counter[0] += 1

    print(f"World ready: {len(world.creatures)} creatures")

    tracker = PopulationTracker(history_length=n_ticks // 100 + 100)
    dt = 1.0 / 20.0

    pending_breeds: list[tuple[str, str]] = []

    class FakeQueue:
        def put_nowait(self, item):
            if item[0] == "breed":
                pending_breeds.append((item[1], item[2]))

    world.breed_queue = FakeQueue()

    breed_count = 0
    for t in range(n_ticks):
        tick(world, rng, dt)

        # Process LLM breeding
        if pending_breeds:
            for aid, bid in pending_breeds:
                a = world.creatures.get(aid)
                b = world.creatures.get(bid)
                if a is None or b is None:
                    continue
                if len(world.creatures) >= MAX_POPULATION:
                    break

                name_counter[0] += 1
                child_id = world.next_id()
                child_name = _NAMES[name_counter[0] % len(_NAMES)]
                gen = max(a.generation, b.generation) + 1
                sx = (a.x + b.x) / 2 + rng.uniform(-1.0, 1.0)
                sy = (a.y + b.y) / 2 + rng.uniform(-1.0, 1.0)
                sx = max(1.0, min(WORLD_W - 1.0, sx))
                sy = max(1.0, min(WORLD_H - 1.0, sy))

                # Build BreedRequest
                request = BreedRequest(
                    parent_a_genes=a.genes,
                    parent_b_genes=b.genes,
                    parent_a_name=a.name,
                    parent_b_name=b.name,
                    parent_a_corpus=a.corpus or build_corpus(a.name, a.genes),
                    parent_b_corpus=b.corpus or build_corpus(b.name, b.genes),
                    parent_a_appear=a.appearance,
                    parent_b_appear=b.appearance,
                    parent_a_fitness=a.happiness,
                    parent_b_fitness=b.happiness,
                    child_name=child_name,
                    child_id=child_id,
                    spawn_x=sx,
                    spawn_y=sy,
                    generation=gen,
                )

                # Run async breeding synchronously
                breed_count += 1
                print(f"  tick {t}: LLM breeding #{breed_count} "
                      f"({a.name} × {b.name})...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result: BreedResult = asyncio.run(
                        breed_offspring(request, llm, embedder, rng, config))

                    from evolutionary_ecosystem.server import sim as sim_mod
                    min_age = getattr(sim_mod, 'MAX_AGE_MIN', 90.0)
                    max_age_val = getattr(sim_mod, 'MAX_AGE_MAX', 150.0)

                    child = Creature(
                        id=result.child_id,
                        name=result.child_name,
                        x=result.spawn_x,
                        y=result.spawn_y,
                        genes=result.genes,
                        appearance=result.appearance,
                        skills=result.skills,
                        stats=result.stats,
                        behavior_profile=result.behavior,
                        happiness=rng.uniform(68, 88),
                        heading=rng.uniform(0, 2 * math.pi),
                        generation=result.generation,
                        parents=(result.parent_a_name, result.parent_b_name),
                        corpus=result.corpus,
                        max_age=rng.uniform(min_age, max_age_val),
                        hp=100.0,
                        energy=rng.uniform(70, 100),
                    )
                    world.creatures[child_id] = child
                    world.total_births += 1
                    elapsed = time.time() - t0
                    print(f"OK ({elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"FAILED ({elapsed:.1f}s): {e}")

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
            print(f"  tick {t:>6d}: EXTINCTION — repopulating...")
            genes = GENE_BANK[rng.randint(0, len(GENE_BANK) - 1)]
            for j in range(3):
                name_counter[0] += 1
                cid = world.next_id()
                name = _NAMES[name_counter[0] % len(_NAMES)]
                c = make_creature(cid, genes, name, rng)
                world.creatures[cid] = c

    tracker.update(world)
    print(f"\nTotal LLM breed events: {breed_count}")
    return _collect_metrics(world, tracker.history)


def _collect_metrics(world: World, snapshots: list) -> dict:
    populations = [s.population for s in snapshots]
    avg_pop = float(np.mean(populations)) if populations else 0
    pop_std = float(np.std(populations)) if populations else 0
    extinctions = sum(1 for p in populations if p == 0)

    final_creatures = list(world.creatures.values())
    if len(final_creatures) >= 2:
        profiles = [profile_to_vector(c.behavior_profile)
                    for c in final_creatures]
        dists = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                dists.append(1.0 - cosine_similarity(profiles[i], profiles[j]))
        behavior_diversity = float(np.mean(dists))
    else:
        behavior_diversity = 0.0

    max_gen = max((c.generation for c in final_creatures), default=0)

    avg_profile = {}
    if final_creatures:
        for sit in SITUATION_NAMES:
            vals = [c.behavior_profile.strength(sit)
                    if c.behavior_profile else 0.3
                    for c in final_creatures]
            avg_profile[sit] = round(float(np.mean(vals)), 4)

    return {
        "avg_population": round(avg_pop, 2),
        "pop_std": round(pop_std, 2),
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "max_generation": max_gen,
        "behavior_diversity": round(behavior_diversity, 4),
        "extinction_events": extinctions,
        "final_population": len(final_creatures),
        "avg_behavior_profile": avg_profile,
        "population_timeline": populations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _compute_cohen_d(x: list[float], y: list[float]) -> float:
    """Cohen's d for paired samples."""
    diffs = [a - b for a, b in zip(x, y)]
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    if std_d == 0:
        return 0.0
    return mean_d / std_d


def _ci95(values: list[float]) -> tuple[float, float]:
    """95% confidence interval for the mean (t-based)."""
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if values else 0.0
        return (m, m)
    m = float(np.mean(values))
    se = float(scipy_stats.sem(values))
    h = se * scipy_stats.t.ppf(0.975, n - 1)
    return (round(m - h, 6), round(m + h, 6))


def _run_statistical_tests(
    det_trials: list[dict], llm_trials: list[dict],
) -> dict:
    """Run paired statistical tests across trial results."""
    metrics = [
        "avg_population", "pop_std", "total_births",
        "total_deaths", "max_generation", "behavior_diversity",
    ]
    tests = {}
    for metric in metrics:
        det_vals = [t[metric] for t in det_trials]
        llm_vals = [t[metric] for t in llm_trials]
        diffs = [d - l for d, l in zip(det_vals, llm_vals)]

        entry: dict = {
            "det_values": det_vals,
            "llm_values": llm_vals,
            "det_mean": round(float(np.mean(det_vals)), 4),
            "llm_mean": round(float(np.mean(llm_vals)), 4),
            "det_ci95": list(_ci95(det_vals)),
            "llm_ci95": list(_ci95(llm_vals)),
        }

        # Paired t-test
        if len(det_vals) >= 2:
            t_stat, t_p = scipy_stats.ttest_rel(det_vals, llm_vals)
            entry["paired_ttest"] = {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(t_p), 6),
            }
        else:
            entry["paired_ttest"] = {"t_statistic": None, "p_value": None,
                                     "note": "insufficient trials"}

        # Wilcoxon signed-rank test
        if len(det_vals) >= 3 and any(d != 0 for d in diffs):
            try:
                w_stat, w_p = scipy_stats.wilcoxon(det_vals, llm_vals)
                entry["wilcoxon"] = {
                    "statistic": round(float(w_stat), 4),
                    "p_value": round(float(w_p), 6),
                }
            except ValueError as e:
                entry["wilcoxon"] = {"statistic": None, "p_value": None,
                                     "note": str(e)}
        else:
            entry["wilcoxon"] = {"statistic": None, "p_value": None,
                                 "note": "insufficient non-zero differences"}

        # Cohen's d
        entry["cohens_d"] = round(_compute_cohen_d(det_vals, llm_vals), 4)

        # Shapiro-Wilk on differences
        if len(diffs) >= 3:
            sw_stat, sw_p = scipy_stats.shapiro(diffs)
            entry["shapiro_wilk_diffs"] = {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 6),
            }
        else:
            entry["shapiro_wilk_diffs"] = {
                "statistic": None, "p_value": None,
                "note": "insufficient samples",
            }

        tests[metric] = entry

    return tests


def main():
    parser = argparse.ArgumentParser(
        description="Eval 5b: LLM-mediated breeding in full simulation")
    parser.add_argument("--model", default="",
                        help="LLM model ID (auto-detected if omitted)")
    parser.add_argument("--backend", choices=["auto", "anthropic", "local"],
                        default="auto",
                        help="LLM backend: anthropic, local, or auto. Use --base-url to point at any OpenAI-compatible endpoint (Ollama, etc.)")
    parser.add_argument("--base-url", default=DEFAULT_LOCAL_URL,
                        help=f"Local LLM server URL (default: {DEFAULT_LOCAL_URL})")
    parser.add_argument("--ticks", type=int, default=10000,
                        help="Number of simulation ticks (default: 10000)")
    args = parser.parse_args()

    backend, model, base_url = detect_backend(args)
    llm = make_llm(backend, model, base_url)

    N_TICKS = args.ticks
    N_CREATURES = 6

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    print("=" * 70)
    print("EVAL 5b: LLM-Mediated Breeding in Full Simulation")
    print("=" * 70)
    print(f"LLM backend: {backend}")
    print(f"LLM model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Ticks: {N_TICKS}, Starting pop: {N_CREATURES}")
    print(f"N_TRIALS: {N_TRIALS}, Seeds: {SEEDS}")
    print(f"Platform: {platform.platform()}")
    print(f"Timestamp: {timestamp}")

    # -----------------------------------------------------------------------
    # Multi-trial loop
    # -----------------------------------------------------------------------
    per_trial: list[dict] = []
    all_det_results: list[dict] = []
    all_llm_results: list[dict] = []

    for trial_idx, seed in enumerate(SEEDS[:N_TRIALS]):
        print(f"\n{'#'*70}")
        print(f"TRIAL {trial_idx + 1}/{N_TRIALS}  (seed={seed})")
        print(f"{'#'*70}")

        # Condition 1: BEAR deterministic
        print(f"\n{'='*50}")
        print(f"Condition: BEAR Deterministic  (seed={seed})")
        print(f"{'='*50}")
        t0 = time.time()
        det_result = run_bear_deterministic(seed, N_TICKS, N_CREATURES)
        det_time = time.time() - t0
        print(f"Deterministic condition completed in {det_time:.1f}s")

        # Condition 2: BEAR LLM
        print(f"\n{'='*50}")
        print(f"Condition: BEAR LLM  (seed={seed})")
        print(f"{'='*50}")
        t0 = time.time()
        llm_result = run_bear_llm(seed, N_TICKS, N_CREATURES, llm)
        llm_time = time.time() - t0
        print(f"LLM condition completed in {llm_time:.1f}s")

        trial_record = {
            "trial": trial_idx + 1,
            "seed": seed,
            "det_runtime_seconds": round(det_time, 1),
            "llm_runtime_seconds": round(llm_time, 1),
            "bear_deterministic": {
                k: v for k, v in det_result.items()
                if k != "population_timeline"
            },
            "bear_llm": {
                k: v for k, v in llm_result.items()
                if k != "population_timeline"
            },
            "population_timelines": {
                "bear_deterministic": det_result["population_timeline"],
                "bear_llm": llm_result["population_timeline"],
            },
        }
        per_trial.append(trial_record)
        all_det_results.append(det_result)
        all_llm_results.append(llm_result)

    # -----------------------------------------------------------------------
    # Statistical tests across trials
    # -----------------------------------------------------------------------
    statistical_tests = _run_statistical_tests(
        [{k: v for k, v in r.items() if k != "population_timeline"}
         for r in all_det_results],
        [{k: v for k, v in r.items() if k != "population_timeline"}
         for r in all_llm_results],
    )

    # -----------------------------------------------------------------------
    # Summary (use first trial for backward-compatible top-level keys)
    # -----------------------------------------------------------------------
    first_det = all_det_results[0]
    first_llm = all_llm_results[0]

    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY (mean +/- 95% CI across trials)")
    print(f"{'='*70}")
    print(f"{'':30s} {'Deterministic':>20s} {'LLM':>20s}")
    for metric in ["avg_population", "pop_std", "total_births",
                    "total_deaths", "max_generation", "behavior_diversity",
                    "extinction_events", "final_population"]:
        det_vals = [r[metric] for r in all_det_results]
        llm_vals = [r[metric] for r in all_llm_results]
        det_m = float(np.mean(det_vals))
        llm_m = float(np.mean(llm_vals))
        det_ci = _ci95(det_vals)
        llm_ci = _ci95(llm_vals)
        print(f"{metric:30s} {det_m:>8.2f} [{det_ci[0]:.2f},{det_ci[1]:.2f}]"
              f" {llm_m:>8.2f} [{llm_ci[0]:.2f},{llm_ci[1]:.2f}]")

    # Print statistical test results
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS (deterministic vs LLM)")
    print(f"{'='*70}")
    for metric, entry in statistical_tests.items():
        print(f"\n  {metric}:")
        print(f"    Det mean: {entry['det_mean']}, LLM mean: {entry['llm_mean']}")
        pt = entry["paired_ttest"]
        print(f"    Paired t-test: t={pt['t_statistic']}, p={pt['p_value']}")
        wt = entry["wilcoxon"]
        print(f"    Wilcoxon: W={wt['statistic']}, p={wt['p_value']}")
        print(f"    Cohen's d: {entry['cohens_d']}")
        sw = entry["shapiro_wilk_diffs"]
        print(f"    Shapiro-Wilk (diffs): W={sw['statistic']}, p={sw['p_value']}")

    # LaTeX table (uses aggregate means)
    print("\n% === LaTeX Table ===")
    print("\\begin{table}[t]")
    print("\\caption{Simulation metrics: BEAR deterministic breeding vs")
    print(f"LLM-mediated breeding (mean over {N_TRIALS} trials).}}")
    print("\\label{tab:llm-breeding-simulation}")
    print("\\begin{tabular}{@{}lcc@{}}")
    print("\\toprule")
    print("Metric & Deterministic & LLM \\\\")
    print("\\midrule")
    for metric, label in [
        ("avg_population", "Avg Population"),
        ("pop_std", "Pop Std Dev"),
        ("total_births", "Total Births"),
        ("total_deaths", "Total Deaths"),
        ("max_generation", "Max Generation"),
        ("behavior_diversity", "Behavioral Diversity"),
    ]:
        det_m = statistical_tests[metric]["det_mean"]
        llm_m = statistical_tests[metric]["llm_mean"]
        print(f"{label} & {det_m} & {llm_m} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    total_det_runtime = sum(t["det_runtime_seconds"] for t in per_trial)
    total_llm_runtime = sum(t["llm_runtime_seconds"] for t in per_trial)

    output = {
        "metadata": {
            "eval": "5b",
            "description": "LLM-mediated breeding in full simulation",
            "llm_backend": backend,
            "llm_model": model,
            "base_url": base_url,
            "n_ticks": N_TICKS,
            "n_creatures": N_CREATURES,
            "n_trials": N_TRIALS,
            "seeds": SEEDS[:N_TRIALS],
            "seed": SEEDS[0],  # backward compat
            "det_runtime_seconds": round(total_det_runtime, 1),
            "llm_runtime_seconds": round(total_llm_runtime, 1),
            "platform": platform.platform(),
            "timestamp": timestamp,
        },
        # Backward compatible: first trial at top level
        "conditions": {
            "bear_deterministic": {
                k: v for k, v in first_det.items()
                if k != "population_timeline"
            },
            "bear_llm": {
                k: v for k, v in first_llm.items()
                if k != "population_timeline"
            },
        },
        "population_timelines": {
            "bear_deterministic": first_det["population_timeline"],
            "bear_llm": first_llm["population_timeline"],
        },
        # New multi-trial data
        "per_trial": per_trial,
        "statistical_tests": statistical_tests,
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

    results_path = OUT_DIR / "eval5b_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    try:
        _plot_comparison(output, first_det, first_llm)
    except Exception as e:
        print(f"Chart generation failed: {e}")


def _plot_comparison(output, det_result, llm_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    conditions = ["Deterministic", "LLM"]
    colors = ["#4CAF50", "#2196F3"]

    # Panel 1: Population over time
    ax = axes[0][0]
    for idx, (label, result) in enumerate(
            [("Deterministic", det_result), ("LLM", llm_result)]):
        pops = result["population_timeline"]
        ticks = list(range(0, len(pops) * 100, 100))
        ax.plot(ticks, pops, label=label, color=colors[idx],
                linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Population")
    ax.set_title("Population Over Time")
    ax.legend(fontsize=8)

    # Panel 2: Avg population
    ax = axes[0][1]
    x = np.arange(2)
    vals = [det_result["avg_population"], llm_result["avg_population"]]
    ax.bar(x, vals, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Avg Population")
    ax.set_title("Average Population")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontweight="bold")

    # Panel 3: Max generation
    ax = axes[0][2]
    vals = [det_result["max_generation"], llm_result["max_generation"]]
    ax.bar(x, vals, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Max Generation")
    ax.set_title("Generational Depth")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    # Panel 4: Births vs deaths
    ax = axes[1][0]
    width = 0.35
    births = [det_result["total_births"], llm_result["total_births"]]
    deaths = [det_result["total_deaths"], llm_result["total_deaths"]]
    ax.bar(x - width / 2, births, width, label="Births",
           color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, deaths, width, label="Deaths",
           color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Total Births vs Deaths")
    ax.legend()

    # Panel 5: Behavioral diversity
    ax = axes[1][1]
    vals = [det_result["behavior_diversity"], llm_result["behavior_diversity"]]
    ax.bar(x, vals, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.set_title("Behavioral Diversity")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")

    # Panel 6: Runtime comparison
    ax = axes[1][2]
    runtimes = [output["metadata"]["det_runtime_seconds"],
                output["metadata"]["llm_runtime_seconds"]]
    ax.bar(x, runtimes, 0.5, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime")
    for i, v in enumerate(runtimes):
        ax.text(i, v + 0.5, f"{v:.0f}s", ha="center", fontweight="bold")

    plt.suptitle(
        "Eval 5b: BEAR Deterministic vs LLM-Mediated Breeding\n"
        f"Model: {output['metadata']['llm_model']} | "
        f"{output['metadata']['n_ticks']} ticks",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = OUT_DIR / "eval5b_llm_breeding.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
