#!/usr/bin/env python3
"""Evaluation 9b: Haploid vs Diploid Inheritance Under Selection Pressure.

Runs the complete live simulation (brain loop + BEAR retrieval + action tags)
comparing inheritance ploidy modes:

  A) Haploid         — one allele per locus (default)
  B) Diploid Dominant — two alleles per locus; dominant expressed
  C) Diploid Codominant — two alleles; both expressed (blended)

Hypothesis: Diploid inheritance preserves latent diversity via recessive
alleles that survive silently in heterozygotes, re-emerging when
environmental shifts make them advantageous.

Uses the same live sim infrastructure as eval2b (full LLM pipeline).
Multiple seeded trials give statistical validity.

Usage:
  python eval9b_diploid_selection.py \\
    --base-url http://192.168.1.175:11434/v1 \\
    --model gemma4:e2b [--seeds 42 142 242 342 442] [--ticks 10000]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE  = Path(__file__).resolve().parent
_ECO   = _HERE.parent
_ROOT  = _ECO.parent.parent
_RUNPY = _ECO / "run.py"

RESULTS_PATH = _HERE / "results" / "eval9b_results.json"

N_TRIALS  = 5
N_TICKS   = 10_000
BASE_SEED = 42

CONDITIONS = ["diploid_dominant"]
LABELS     = {"diploid_dominant": "Diploid Dom.",
              "diploid_codominant": "Diploid Codom."}
EVAL2B_RESULTS = Path(__file__).resolve().parent / "results" / "eval2b_results.json"


def run_single(seed: int, ploidy: str, base_url: str, model: str,
               n_ticks: int, n_creatures: int) -> dict:
    out_file = _HERE / "results" / f"_eval9b_tmp_{seed}_{ploidy}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_RUNPY.resolve()),
        "--seed",         str(seed),
        "--ticks",        str(n_ticks),
        "--creatures",    str(n_creatures),
        "--output",       str(out_file),
        "--base-url",     base_url,
        "--model",        model,
        "--ploidy",       ploidy,
    ]

    print(f"  [{LABELS[ploidy]:<18}] seed={seed} ...", flush=True)
    result = subprocess.run(cmd, capture_output=False, text=True)

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
    return {}


def summarise_pair(a_vals, b_vals):
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
        return {"a_mean": float(a.mean()), "b_mean": float(b.mean()),
                "cohens_d": 0.0, "p_value": 1.0, "significant_005": False}
    pooled = np.sqrt(((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2))
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "a_mean":          round(float(a.mean()),       4),
        "a_std":           round(float(a.std(ddof=1)),  4),
        "b_mean":          round(float(b.mean()),       4),
        "b_std":           round(float(b.std(ddof=1)),  4),
        "cohens_d":        round(d, 3),
        "p_value":         round(float(p), 4),
        "significant_005": bool(p < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eval 9b: Haploid vs Diploid under selection pressure (live sim)")
    parser.add_argument("--base-url",  required=True, dest="base_url")
    parser.add_argument("--model",     required=True)
    parser.add_argument("--trials",    type=int, default=N_TRIALS)
    parser.add_argument("--ticks",     type=int, default=N_TICKS)
    parser.add_argument("--creatures", type=int, default=12)
    parser.add_argument("--seeds",     type=int, nargs="+", default=None)
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [BASE_SEED + i * 100 for i in range(args.trials)]
    args.trials = len(seeds)

    print("=" * 60)
    print("EVAL 9b: Haploid vs Diploid Under Selection Pressure")
    print("=" * 60)
    print(f"Ticks: {args.ticks}  Creatures: {args.creatures}  Trials: {args.trials}")
    print(f"Seeds: {seeds}")
    print(f"LLM: {args.base_url} / {args.model}")
    print()

    # Load haploid baseline from eval2b BEAR On results
    haploid_trials = []
    if EVAL2B_RESULTS.exists():
        with open(EVAL2B_RESULTS) as f:
            eval2b = json.load(f)
        haploid_trials = eval2b.get("bear_on_trials", [])
        print(f"Loaded {len(haploid_trials)} haploid baseline trials from eval2b")
    else:
        print(f"WARNING: eval2b results not found at {EVAL2B_RESULTS}")
        print("Run eval2b first, or haploid comparison will be skipped.")
    print()

    all_trials: dict[str, list[dict]] = {c: [] for c in CONDITIONS}

    for i, seed in enumerate(seeds):
        print(f"--- Trial {i+1}/{args.trials} (seed={seed}) ---")
        for cond in CONDITIONS:
            r = run_single(seed, cond, args.base_url, args.model, args.ticks, args.creatures)
            if r:
                r["ploidy"] = cond
                all_trials[cond].append(r)
                print(f"  {LABELS[cond]:<20} pop={r.get('final_population',0)} "
                      f"births={r.get('total_births',0)} "
                      f"gen={r.get('max_generation',0)} "
                      f"div={r.get('gene_diversity',0):.3f}")
        print()

    metrics = ["final_population", "total_births", "max_generation",
               "gene_diversity", "hausdorff"]

    # Summary stats per condition (including haploid baseline)
    summary = {}
    if haploid_trials:
        summary["haploid"] = {
            m: {"mean": round(float(np.mean([r.get(m, 0) for r in haploid_trials])), 4),
                "std":  round(float(np.std( [r.get(m, 0) for r in haploid_trials], ddof=1)), 4),
                "source": "eval2b_bear_on"}
            for m in metrics
        }
    for cond in CONDITIONS:
        trials = all_trials[cond]
        summary[cond] = {
            m: {"mean": round(float(np.mean([r.get(m, 0) for r in trials])), 4),
                "std":  round(float(np.std( [r.get(m, 0) for r in trials], ddof=1)), 4)}
            for m in metrics
        }

    # Pairwise comparisons: haploid vs each diploid mode
    comparisons = {}
    if haploid_trials:
        for cond in CONDITIONS:
            comparisons[f"haploid_vs_{cond}"] = {
                m: summarise_pair(
                    [r.get(m, 0) for r in haploid_trials],
                    [r.get(m, 0) for r in all_trials[cond]]
                )
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
        "summary":     summary,
        "comparisons": comparisons,
        "trials":      all_trials,
    }

    out_path = Path(args.output) if args.output else RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Metric':<22} {'Haploid':>12} {'Dip.Dom.':>12} {'Dip.Codom.':>12}")
    print("-" * 62)
    for m in metrics:
        row = f"{m:<22}"
        for cond in CONDITIONS:
            row += f" {summary[cond][m]['mean']:>12.3f}"
        print(row)

    print("\n=== HAPLOID vs DIPLOID DOMINANT ===")
    print(f"{'Metric':<22} {'Haploid':>10} {'Dip.Dom.':>10} {'p':>8} {'d':>6} {'sig':>4}")
    print("-" * 64)
    comp = comparisons["haploid_vs_diploid_dominant"]
    for m, v in comp.items():
        sig = "*" if v["significant_005"] else ""
        print(f"{m:<22} {v['a_mean']:>10.3f} {v['b_mean']:>10.3f} "
              f"{v['p_value']:>8.4f} {v['cohens_d']:>6.3f} {sig:>4}")

    print("\n=== HAPLOID vs DIPLOID CODOMINANT ===")
    comp2 = comparisons["haploid_vs_diploid_codominant"]
    for m, v in comp2.items():
        sig = "*" if v["significant_005"] else ""
        print(f"{m:<22} {v['a_mean']:>10.3f} {v['b_mean']:>10.3f} "
              f"{v['p_value']:>8.4f} {v['cohens_d']:>6.3f} {sig:>4}")


if __name__ == "__main__":
    main()
