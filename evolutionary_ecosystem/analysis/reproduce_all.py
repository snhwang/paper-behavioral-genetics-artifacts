#!/usr/bin/env python3
"""Reproduce the paper's data-backed figures and tables in one run.

Run from the repo root:  python evolutionary_ecosystem/analysis/reproduce_all.py

Regenerates every figure/table that has a data pipeline, from committed results
(no simulation and no LLM required). See REPRODUCE.md for per-item detail,
inputs, and the optional from-scratch reruns.

Figures land in figures/ (and evolutionary_ecosystem/analysis/ for the memory
figure); table statistics are printed to stdout.
"""

import subprocess
import sys

# Scripts are invoked with paths relative to the repo root; each globs its
# inputs relative to the current directory, so run this from the repo root.
STEPS = [
    ("evolutionary_ecosystem/analysis/regen_paper_figures.py",
     "fig:inheritance + fig:epoch-heatmap (eval3 + eval4, from committed results)"),
    ("evolutionary_ecosystem/analysis/compute_inheritance_stats.py",
     "tab:inheritance-comparison (cross-mode inheritance stats)"),
    ("evolutionary_ecosystem/analysis/aggregate_selection.py",
     "tab:selection-pressure (flee vs. rally under mutation rate 0)"),
    ("evolutionary_ecosystem/analysis/plot_marker_decay.py",
     "fig:marker-decay (action-marker prevalence across generations)"),
    ("evolutionary_ecosystem/eval/eval10_memory_inheritance.py",
     "fig:memory-inheritance, phase 2 (headless, from cached extraction)"),
    ("evolutionary_ecosystem/analysis/plot_memory_inheritance.py",
     "fig:memory-inheritance (plot)"),
]


def run(script, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")
    return result.returncode


def main():
    failures = [s for s, d in STEPS if run(s, d) != 0]

    print(f"\n{'='*60}")
    print("  Done. Figures in figures/ (memory figure in "
          "evolutionary_ecosystem/analysis/); table numbers above.")
    print("  tab:population-dynamics needs no script — its values are the "
          "committed results/eval1_results.json.")
    print("  From-scratch reruns of eval1/eval3/eval4/eval10 extraction: see "
          "REPRODUCE.md.")
    if failures:
        print(f"  {len(failures)} step(s) failed: {failures}")
    print(f"{'='*60}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
