#!/usr/bin/env python3
"""Post-process eval2b_results.json after an in-flight run that used the
old (pre-flattening) run_single().

The old code path saved trials with the nested {metadata: ..., final: ...}
schema from app.py instead of the flat keys the rest of the eval expects.
This script:

  1. Flattens each trial so final_population / total_births / etc. live
     at the trial top level, like new runs produce natively.
  2. Recomputes summary (mean/std per metric) from the flattened trials.
  3. Recomputes statistical_tests (Cohen's d + Mann-Whitney p) for the
     bear_on vs bear_off comparison.

Idempotent: running it on an already-flat results file is a no-op.

Usage:
    python3 evolutionary_ecosystem/eval/postprocess_eval2b.py
    python3 evolutionary_ecosystem/eval/postprocess_eval2b.py path/to/eval2b_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_PATH = Path(__file__).resolve().parent / "results" / "eval2b_results.json"

METRICS = ("final_population", "total_births", "max_generation",
           "gene_diversity", "hausdorff")


def _flatten_trial(t: dict) -> dict:
    """Return a flat copy of a trial dict.

    If the trial is already flat (has final_population at top level),
    return it unchanged. Otherwise pull from t["final"] and t["metadata"].
    """
    if "final_population" in t and "total_births" in t:
        return t
    final    = t.get("final",    {}) or {}
    metadata = t.get("metadata", {}) or {}
    return {
        "final_population": final.get("population", 0),
        "total_births":     final.get("total_births", 0),
        "total_deaths":     final.get("total_deaths", 0),
        "max_generation":   final.get("max_generation", 0),
        "gene_diversity":   final.get("gene_diversity", 0.0),
        "hausdorff":        final.get("hausdorff", 0.0),
        "elapsed_seconds":  metadata.get("elapsed_seconds", 0.0),
        "n_ticks":          metadata.get("n_ticks"),
        "seed":             metadata.get("seed"),
        "birth_log":        t.get("birth_log", []),
        "snapshots":        t.get("snapshots", []),
    }


def _summarise_one(vals: list) -> dict:
    arr = np.array(vals, dtype=float)
    return {"mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0}


def _compare(on_vals: list, off_vals: list) -> dict:
    on  = np.array(on_vals,  dtype=float)
    off = np.array(off_vals, dtype=float)
    if on.std(ddof=1) == 0 and off.std(ddof=1) == 0:
        return {"bear_on_mean": float(on.mean()), "bear_on_std": 0.0,
                "bear_off_mean": float(off.mean()), "bear_off_std": 0.0,
                "cohens_d": 0.0, "p_value": 1.0, "significant_005": False}
    pooled = np.sqrt(((on.std(ddof=1) ** 2 + off.std(ddof=1) ** 2) / 2))
    d = float((on.mean() - off.mean()) / pooled) if pooled > 0 else 0.0
    _, p = stats.mannwhitneyu(on, off, alternative="two-sided")
    return {
        "bear_on_mean":    round(float(on.mean()),       4),
        "bear_on_std":     round(float(on.std(ddof=1)),  4),
        "bear_off_mean":   round(float(off.mean()),      4),
        "bear_off_std":    round(float(off.std(ddof=1)), 4),
        "cohens_d":        round(d, 3),
        "p_value":         round(float(p), 4),
        "significant_005": bool(p < 0.05),
    }


def main(path: Path) -> None:
    data = json.loads(path.read_text())

    on_trials  = [_flatten_trial(t) for t in data.get("bear_on_trials",  [])]
    off_trials = [_flatten_trial(t) for t in data.get("bear_off_trials", [])]

    data["bear_on_trials"]  = on_trials
    data["bear_off_trials"] = off_trials

    if on_trials and off_trials:
        data["summary"] = {
            "bear_on":  {m: _summarise_one([t[m] for t in on_trials])  for m in METRICS},
            "bear_off": {m: _summarise_one([t[m] for t in off_trials]) for m in METRICS},
        }
        data["statistical_tests"] = {
            m: _compare([t[m] for t in on_trials],
                        [t[m] for t in off_trials])
            for m in METRICS
        }

    path.write_text(json.dumps(data, indent=2))

    print(f"Post-processed {path}")
    print(f"  bear_on  trials: {len(on_trials)}")
    print(f"  bear_off trials: {len(off_trials)}")
    if on_trials and off_trials:
        print()
        print(f"{'Metric':<22} {'BEAR On':>14} {'BEAR Off':>14} {'p':>8} {'d':>7} sig")
        print("-" * 70)
        for m in METRICS:
            s = data["summary"]
            t = data["statistical_tests"][m]
            on_s  = f"{s['bear_on'][m]['mean']:.3f}±{s['bear_on'][m]['std']:.3f}"
            off_s = f"{s['bear_off'][m]['mean']:.3f}±{s['bear_off'][m]['std']:.3f}"
            sig = "*" if t["significant_005"] else ""
            print(f"{m:<22} {on_s:>14} {off_s:>14} {t['p_value']:>8.4f} {t['cohens_d']:>7.3f} {sig}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    main(path)
