#!/usr/bin/env python3
"""Spark version of eval_combined_v2.

Maps run_epoch_trial across all 25 (epoch_idx, seed) combinations in
parallel. Each worker runs one trial independently and returns serializable
results. Driver collects and runs compute_eval3 + compute_eval4.

Usage:
    spark-submit \\
        --master <your-master-url> \\
        examples/evolutionary_ecosystem/eval/eval_combined_v2_spark.py

    # Or from within a Spark session:
    python -m examples.evolutionary_ecosystem.eval.eval_combined_v2_spark
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

from examples.evolutionary_ecosystem.eval.eval_combined_v2 import (
    BASE_SEED,
    N_TRIALS,
    OUT_DIR,
    BirthRecord,
    _birth_from_dict,
    _birth_to_dict,
    compute_eval3,
    compute_eval4,
    run_epoch_trial,
)
from examples.evolutionary_ecosystem.server.epochs import EPOCHS


def run_trial_task(args: tuple) -> tuple[dict, list[dict]]:
    """Worker function: runs one trial, returns JSON-serializable results.
    Called on Spark workers — no shared state.
    """
    epoch_idx, trial, seed = args
    result, births = run_epoch_trial(epoch_idx, seed)
    result["trial"] = trial
    births_ser = [_birth_to_dict(b) for b in births]
    return result, births_ser


def main():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("eval_combined_v2") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build task list: (epoch_idx, trial, seed)
    tasks = []
    for epoch_idx, epoch in enumerate(EPOCHS):
        for trial in range(N_TRIALS):
            seed = BASE_SEED + epoch_idx * 100 + trial * 1000
            tasks.append((epoch_idx, trial, seed))

    n_tasks = len(tasks)
    print(f"Submitting {n_tasks} tasks across {len(EPOCHS)} epochs x {N_TRIALS} trials")

    # Distribute — one partition per task so all run in parallel
    rdd = sc.parallelize(tasks, numSlices=n_tasks)
    results_rdd = rdd.map(run_trial_task)

    # Collect all results to driver
    print("Waiting for all trials to complete...")
    all_results = results_rdd.collect()
    print(f"All {len(all_results)} trials complete")

    spark.stop()

    # Reassemble
    all_eval4: dict[str, list[dict]] = defaultdict(list)
    all_births_ser: list[dict] = []

    for eval4_result, births_ser in all_results:
        epoch_name = eval4_result["epoch"]
        all_eval4[epoch_name].append(eval4_result)
        all_births_ser.extend(births_ser)

    print(f"Total births captured: {len(all_births_ser)}")

    # Save raw collected data as backup before computing stats
    raw_path = OUT_DIR / "eval_combined_v2_raw.json"
    with open(raw_path, "w") as f:
        json.dump({"all_eval4": dict(all_eval4),
                   "all_births": all_births_ser}, f, indent=2)
    print(f"Raw data saved: {raw_path}")

    # Eval 4
    print("\nAggregating eval 4...")
    eval4 = compute_eval4(all_eval4)
    eval4_path = OUT_DIR / "eval4_v2_results.json"
    with open(eval4_path, "w") as f:
        json.dump(eval4, f, indent=2)
    print(f"Saved: {eval4_path}")

    # Eval 3
    print("\nComputing eval 3 inheritance fidelity...")
    all_births = [_birth_from_dict(d) for d in all_births_ser]
    eval3 = compute_eval3(all_births)
    eval3_path = OUT_DIR / "eval3_v2_results.json"
    with open(eval3_path, "w") as f:
        json.dump(eval3, f, indent=2)
    print(f"Saved: {eval3_path}")

    # Summary
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    st = eval3["statistical_tests"]
    po = eval3["parent_offspring_similarity"]
    rb = eval3["random_pair_baseline"]
    print(f"\nEval 3 — {eval3['n_births']} births")
    for metric in po:
        s = st.get(metric, {})
        print(f"  {metric:20s}: P-O={po[metric]['mean']:.4f}  "
              f"Rand={rb[metric]['mean']:.4f}  "
              f"d={s.get('cohens_d','?')}  p={s.get('p_value','?'):.2e}"
              if isinstance(s.get('p_value'), float) else
              f"  {metric:20s}: P-O={po[metric]['mean']:.4f}")

    anova = eval4["anova"]
    sig = sum(1 for v in anova.values() if v["p_value"] < 0.05)
    top = max(anova.items(), key=lambda x: x[1]["F_statistic"])
    print(f"\nEval 4 — {sig}/{len(anova)} situations significant")
    print(f"  Top F={top[1]['F_statistic']:.2f} ({top[0]})  "
          f"p={top[1]['p_value']:.2e}")


if __name__ == "__main__":
    main()
