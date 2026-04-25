#!/usr/bin/env python3
"""Evaluation 2b: BEAR On vs BEAR Off — full LLM pipeline.

Runs the complete live simulation (brain loop + BEAR retrieval + action tags)
in headless mode via app.py --headless, comparing:

  A) BEAR On  — full pipeline: LLM decisions, retrieval drives behavior,
                [!breed(nearest)] action tags trigger breeding.
  B) BEAR Off — NullBehaviorProfile, no LLM, autonomous proximity breeding.

Multiple seeded trials give statistical validity (mean, std, p-values, Cohen's d).

Usage:
  python eval2b_bear_on_off.py \\
    --base-url http://192.168.1.175:11434/v1 \\
    --model gemma4:e2b [--trials 5] [--ticks 30000]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE  = Path(__file__).resolve().parent
_ECO   = _HERE.parent                  # examples/evolutionary_ecosystem
_ROOT  = _ECO.parent.parent            # bear repo root
_RUNPY = _ECO / "run.py"

RESULTS_PATH = _HERE / "results" / "eval2b_results.json"

N_TRIALS  = 5
N_TICKS   = 30_000
BASE_SEED = 42


def run_single(seed: int, bear_on: bool, base_url: str, model: str,
               n_ticks: int, n_creatures: int) -> dict:
    """Run one headless trial via run.py and return the results dict."""
    out_file = _HERE / "results" / f"_eval2b_tmp_{seed}_{'on' if bear_on else 'off'}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_RUNPY.resolve()),
        "--seed",      str(seed),
        "--ticks",     str(n_ticks),
        "--creatures", str(n_creatures),
        "--output",    str(out_file),
        "--base-url",  base_url,
        "--model",     model,
    ]
    if not bear_on:
        cmd.append("--bear-disabled")

    label = "BEAR On " if bear_on else "BEAR Off"
    print(f"  [{label}] seed={seed} ...", flush=True)

    result = subprocess.run(cmd, capture_output=False, text=True)
<<<<<<< Updated upstream
    # Check output file regardless of exit code \u2014 uvicorn exit is noisy
    if out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
            out_file.unlink(missing_ok=True)
            return data
        except Exception as e:
            print(f"  ERROR reading output: {e}")
    elif result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}, no output file")
=======
    if result.returncode != 0:
        print(f"  WARNING: exit code {result.returncode}")

    if out_file.exists():
        with open(out_file) as f:
            data = json.load(f)
        return data
    print(f"  ERROR: no results file produced")
>>>>>>> Stashed changes
    return {}


def summarise(on_vals, off_vals):
    on  = np.array(on_vals,  dtype=float)
    off = np.array(off_vals, dtype=float)
    if on.std(ddof=1) == 0 and off.std(ddof=1) == 0:
        return {"bear_on_mean": float(on.mean()), "bear_on_std": 0.0,
                "bear_off_mean": float(off.mean()), "bear_off_std": 0.0,
                "cohens_d": 0.0, "p_value": 1.0, "significant_005": False}
    pooled = np.sqrt(((on.std(ddof=1)**2 + off.std(ddof=1)**2) / 2))
    d = float((on.mean() - off.mean()) / pooled) if pooled > 0 else 0.0
    _, p = stats.mannwhitneyu(on, off, alternative="two-sided")
    return {
        "bear_on_mean":    round(float(on.mean()),        4),
        "bear_on_std":     round(float(on.std(ddof=1)),   4),
        "bear_off_mean":   round(float(off.mean()),       4),
        "bear_off_std":    round(float(off.std(ddof=1)),  4),
        "cohens_d":        round(d, 3),
        "p_value":         round(float(p), 4),
        "significant_005": bool(p < 0.05),
    }


def _save_intermediate(on_trials, off_trials, args, seeds, out_path):
    """Save partial results after each trial so progress is preserved."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = {
        "parameters": {
            "n_ticks": args.ticks, "n_creatures": args.creatures,
            "n_trials": args.trials, "seeds": seeds,
            "llm_model": args.model, "llm_base_url": args.base_url,
            "status": f"{len(on_trials)}/{args.trials} trials complete",
        },
        "bear_on_trials":  on_trials,
        "bear_off_trials": off_trials,
    }
    out_path.write_text(json.dumps(partial, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Eval 2b: BEAR On vs Off — full LLM pipeline via headless sim")
    parser.add_argument("--base-url",   required=True, dest="base_url",
                        help="OpenAI-compatible endpoint e.g. http://192.168.1.175:11434/v1")
    parser.add_argument("--model",      required=True,
                        help="LLM model name e.g. gemma4:e2b")
    parser.add_argument("--trials",     type=int, default=N_TRIALS)
    parser.add_argument("--ticks",      type=int, default=N_TICKS)
    parser.add_argument("--creatures",  type=int, default=12)
    parser.add_argument("--seeds",      type=int, nargs="+", default=None,
                        help="Explicit seeds for each trial (overrides --trials and BASE_SEED)")
    parser.add_argument("--output",     default=None,
                        help="Output JSON path (default: results/eval2b_results.json)")
    args = parser.parse_args()

    # Build seed list
    if args.seeds:
        seeds = args.seeds
        args.trials = len(seeds)
    else:
        seeds = [BASE_SEED + i * 100 for i in range(args.trials)]

    print("=" * 60)
    print("EVAL 2b: BEAR On vs BEAR Off (full LLM pipeline)")
    print("=" * 60)
    print(f"Ticks: {args.ticks}  Creatures: {args.creatures}  Trials: {args.trials}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {args.base_url} / {args.model}")
    print()

    on_trials:  list[dict] = []
    off_trials: list[dict] = []

    for i, seed in enumerate(seeds):
        print(f"--- Trial {i+1}/{args.trials} (seed={seed}) ---")

        r_on = run_single(seed, True,  args.base_url, args.model, args.ticks, args.creatures)
        if r_on:
            on_trials.append(r_on)
            print(f"  BEAR On:  pop={r_on.get('final_population',0)} "
                  f"births={r_on.get('total_births',0)} "
                  f"gen={r_on.get('max_generation',0)} "
                  f"div={r_on.get('gene_diversity',0):.3f} "
                  f"t={r_on.get('elapsed_seconds',0)}s")

        r_off = run_single(seed, False, args.base_url, args.model, args.ticks, args.creatures)
        if r_off:
            off_trials.append(r_off)
            print(f"  BEAR Off: pop={r_off.get('final_population',0)} "
                  f"births={r_off.get('total_births',0)} "
                  f"gen={r_off.get('max_generation',0)} "
                  f"div={r_off.get('gene_diversity',0):.3f} "
                  f"t={r_off.get('elapsed_seconds',0)}s")
        print()

        # Save intermediate results after each trial pair
        _interim_path = Path(args.output) if args.output else RESULTS_PATH
        _save_intermediate(on_trials, off_trials, args, seeds, _interim_path)

    if not on_trials or not off_trials:
        print("ERROR: no results collected")
        sys.exit(1)

    metrics = ["final_population", "total_births", "max_generation",
               "gene_diversity", "hausdorff"]

    summary = {
        "bear_on":  {m: {"mean": round(float(np.mean([r.get(m,0) for r in on_trials])),  4),
                         "std":  round(float(np.std( [r.get(m,0) for r in on_trials], ddof=1)), 4)}
                     for m in metrics},
        "bear_off": {m: {"mean": round(float(np.mean([r.get(m,0) for r in off_trials])), 4),
                         "std":  round(float(np.std( [r.get(m,0) for r in off_trials], ddof=1)), 4)}
                     for m in metrics},
    }

    statistical_tests = {
        m: summarise([r.get(m,0) for r in on_trials],
                     [r.get(m,0) for r in off_trials])
        for m in metrics
    }

    results = {
        "parameters": {
            "n_ticks":     args.ticks,
            "n_creatures": args.creatures,
            "n_trials":    args.trials,
            "seeds":       seeds,
            "llm_model":   args.model,
            "llm_base_url": args.base_url,
        },
        "summary":           summary,
        "statistical_tests": statistical_tests,
        "bear_on_trials":    on_trials,
        "bear_off_trials":   off_trials,
    }

    out_path = Path(args.output) if args.output else RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")

    print("\n=== RESULTS ===")
    print(f"{'Metric':<22} {'BEAR On':>12} {'BEAR Off':>12} {'p':>8} {'d':>6} {'sig':>4}")
    print("-" * 68)
    for m, v in statistical_tests.items():
        sig = "*" if v["significant_005"] else ""
        print(f"{m:<22} {v['bear_on_mean']:>12.3f} {v['bear_off_mean']:>12.3f} "
              f"{v['p_value']:>8.4f} {v['cohens_d']:>6.3f} {sig:>4}")


if __name__ == "__main__":
    main()
