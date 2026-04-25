#!/usr/bin/env python3
"""Evaluation 4b: Epoch-driven phenotype shift — BEAR On vs Off, full LLM pipeline.

Mirrors eval4 but uses the full live sim with LLM brain loop, comparing:

  A) BEAR On  — full pipeline: LLM decisions, retrieval drives behavior
  B) BEAR Off — NullBehaviorProfile, no LLM, autonomous breeding

Both conditions run through all 5 epochs sequentially on the same population
(15,000 ticks = 5 epochs × 3,000 ticks each). Behavior profiles are captured
at each epoch boundary and compared via ANOVA.

Hypothesis: LLM-guided creatures show larger epoch-driven phenotype shifts
than null-behavior creatures, because the LLM actively responds to
situational context (predator bloom → evasion, famine → food-seeking).

Usage:
  python eval4b_epoch_phenotype_llm.py \\
    --base-url http://192.168.1.175:11434/v1 \\
    --model gemma4:e2b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

_HERE  = Path(__file__).resolve().parent
_ECO   = _HERE.parent
_RUNPY = _ECO / "run.py"

RESULTS_PATH = _HERE / "results" / "eval4b_results.json"
EVAL4_PATH   = _HERE / "results" / "eval4_results.json"

EPOCHS      = ["abundance", "ice_age", "predator_bloom", "expansion", "famine"]
SEEDS       = [42, 1042, 2042]
N_TICKS     = 15_000   # 5 epochs × 3000 ticks each
N_CREATURES = 6


def run_single(seed: int, bear_on: bool, base_url: str, model: str,
               n_ticks: int, n_creatures: int) -> dict:
    label = "BEAR On " if bear_on else "BEAR Off"
    out_file = _HERE / "results" / f"_eval4b_tmp_{seed}_{'on' if bear_on else 'off'}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_RUNPY.resolve()),
        "--seed",         str(seed),
        "--ticks",        str(n_ticks),
        "--creatures",    str(n_creatures),
        "--output",       str(out_file),
        "--base-url",     base_url,
        "--model",        model,
    ]
    if not bear_on:
        cmd.append("--bear-disabled")

    print(f"  [{label}] seed={seed} ticks={n_ticks} ...", flush=True)
    result = subprocess.run(cmd, capture_output=False, text=True)
<<<<<<< Updated upstream
=======
    if result.returncode != 0:
        print(f"  WARNING: exit code {result.returncode}")
>>>>>>> Stashed changes

    # Check output file regardless of exit code — uvicorn exit is noisy
    if out_file.exists():
<<<<<<< Updated upstream
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
        with open(out_file) as f:
            data = json.load(f)
        return data
    print(f"  ERROR: no results file produced")
>>>>>>> Stashed changes
    return {}


def compute_anova(trials_by_epoch: dict) -> dict:
    """Run one-way ANOVA across epochs per behavioral situation."""
    situations = set()
    for epoch_trials in trials_by_epoch.values():
        for t in epoch_trials:
            situations.update(t.get('avg_behavior_profile', {}).keys())

    anova = {}
    for sit in sorted(situations):
        groups = []
        for epoch in EPOCHS:
            vals = [t['avg_behavior_profile'].get(sit, 0)
                    for t in trials_by_epoch.get(epoch, [])
                    if t.get('avg_behavior_profile')]
            if vals:
                groups.append(vals)
        if len(groups) >= 2:
            f, p = scipy_stats.f_oneway(*groups)
            anova[sit] = {
                "F_statistic":     round(float(f), 4),
                "p_value":         round(float(p), 4),
                "significant_005": bool(p < 0.05),
            }
    return anova


def extract_epoch_profiles(trial: dict) -> dict[str, dict]:
    """Extract per-epoch profiles from epoch_snapshots in trial result."""
    profiles = {}
    snapshots = trial.get("epoch_snapshots", [])
    for snap in snapshots:
        epoch = snap.get("epoch")
        if epoch and epoch not in profiles:
            profiles[epoch] = snap.get("avg_behavior_profile", {})
    # Also add final state under the last epoch
    if trial.get("epoch") and trial.get("avg_behavior_profile"):
        profiles[trial["epoch"]] = trial["avg_behavior_profile"]
    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Eval 4b: Epoch phenotype shifts — BEAR On vs Off, full LLM")
    parser.add_argument("--base-url",   required=True, dest="base_url")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--seeds",      type=int, nargs="+", default=SEEDS)
    parser.add_argument("--ticks",      type=int, default=N_TICKS)
    parser.add_argument("--creatures",  type=int, default=N_CREATURES)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("EVAL 4b: Epoch Phenotype Shifts — BEAR On vs Off")
    print("=" * 60)
    print(f"Ticks: {args.ticks} ({args.ticks//3000} epochs)  "
          f"Creatures: {args.creatures}  Seeds: {args.seeds}")
    print(f"LLM: {args.base_url} / {args.model}")
    print()

    on_trials:  list[dict] = []
    off_trials: list[dict] = []

    for i, seed in enumerate(args.seeds):
        print(f"--- Trial {i+1}/{len(args.seeds)} (seed={seed}) ---")

        r_on = run_single(seed, True,  args.base_url, args.model,
                          args.ticks, args.creatures)
        if r_on:
            on_trials.append(r_on)
            print(f"  BEAR On:  pop={r_on.get('final_population',0)} "
                  f"births={r_on.get('total_births',0)} "
                  f"snapshots={len(r_on.get('epoch_snapshots',[]))}")

        r_off = run_single(seed, False, args.base_url, args.model,
                           args.ticks, args.creatures)
        if r_off:
            off_trials.append(r_off)
            print(f"  BEAR Off: pop={r_off.get('final_population',0)} "
                  f"births={r_off.get('total_births',0)} "
                  f"snapshots={len(r_off.get('epoch_snapshots',[]))}")
        print()
        _save(on_trials, off_trials, {}, {}, args)

    # Build per-epoch profile dicts from snapshots
    def build_epoch_profiles(trials):
        by_epoch = {e: [] for e in EPOCHS}
        for trial in trials:
            profiles = extract_epoch_profiles(trial)
            for epoch, profile in profiles.items():
                if epoch in by_epoch and profile:
                    by_epoch[epoch].append({
                        "seed": trial.get("seed"),
                        "avg_behavior_profile": profile,
                    })
        return by_epoch

    on_by_epoch  = build_epoch_profiles(on_trials)
    off_by_epoch = build_epoch_profiles(off_trials)

    on_anova  = compute_anova(on_by_epoch)
    off_anova = compute_anova(off_by_epoch)

    on_sig  = sum(1 for v in on_anova.values()  if v['significant_005'])
    off_sig = sum(1 for v in off_anova.values() if v['significant_005'])

    print(f"\n=== ANOVA: Significant epoch shifts ===")
    print(f"  BEAR On:  {on_sig}/{len(on_anova)}  significant")
    print(f"  BEAR Off: {off_sig}/{len(off_anova)} significant")

    # Compare with headless eval4
    if EVAL4_PATH.exists():
        with open(EVAL4_PATH) as f:
            eval4 = json.load(f)
        eval4_sig = sum(1 for v in eval4.get('anova', {}).values()
                        if v.get('significant_005'))
        print(f"  Eval 4 (headless): {eval4_sig}/{len(eval4.get('anova',{}))} significant")

    _save(on_trials, off_trials, on_anova, off_anova, args)
    print(f"\nResults saved.")


def _save(on_trials, off_trials, on_anova, off_anova, args):
    results = {
        "parameters": {
            "n_ticks":      args.ticks,
            "n_creatures":  args.creatures,
            "seeds":        args.seeds,
            "llm_model":    args.model,
            "llm_base_url": args.base_url,
        },
        "bear_on_trials":   on_trials,
        "bear_off_trials":  off_trials,
        "bear_on_anova":    on_anova,
        "bear_off_anova":   off_anova,
    }
    out = Path(args.output) if args.output else RESULTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
