#!/usr/bin/env python3
"""Compute eval 3 stats from an existing checkpoint file.

Usage:
    python -m examples.evolutionary_ecosystem.eval.compute_eval3_from_checkpoint \
        --checkpoint results/eval_combined_v2_checkpoint_famine.json \
        --output results/eval3_v2_results_famine.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.evolutionary_ecosystem.eval.eval_combined_v2 import (
    compute_eval3, _birth_from_dict
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = json.load(open(args.checkpoint))
    births_ser = ckpt.get("all_births", [])
    print(f"Loaded {len(births_ser)} births from checkpoint")

    # Check child_genes populated
    sample = births_ser[0] if births_ser else {}
    if not sample.get("child_genes"):
        print("ERROR: child_genes not populated — checkpoint is from old bad run")
        return 1

    all_births = [_birth_from_dict(d) for d in births_ser]
    print("Computing eval 3 gene-space metrics...")
    eval3 = compute_eval3(all_births)

    with open(args.output, "w") as f:
        json.dump(eval3, f, indent=2)
    print(f"Saved: {args.output}")

    st = eval3["statistical_tests"]
    for metric, s in st.items():
        print(f"  {metric:20s}: d={s['cohens_d']:.3f}  p={s['p_value']:.2e}")

if __name__ == "__main__":
    sys.exit(main())
