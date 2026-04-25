#!/usr/bin/env python3
"""Reproduce all paper figures and tables in one run.

Run from repo root: python examples/evolutionary_ecosystem/analysis/reproduce_all.py

Outputs all figures to paper_figures/ directory.
Prints table numbers to stdout.
"""

import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path("examples/evolutionary_ecosystem/analysis")
OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

def run(script, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(ANALYSIS_DIR / script)],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")

# ── Eval figures (headless data) ──────────────────────────────
run("regen_paper_figures.py", "Eval 1: Population dynamics figure")
run("regen_paper_figures.py", "Eval 3: Inheritance fidelity figure + table numbers")
run("regen_paper_figures.py", "Eval 4: Epoch shift figure + heatmap + table numbers")

# ── Cross-mode inheritance table ─────────────────────────────
run("compute_inheritance_stats.py", "Cross-mode inheritance stats (tab:inheritance-comparison)")

# ── Action log figures ────────────────────────────────────────
run("plot_action_results.py",
    "Action log: flee/rally epoch, mood(happy) fitness, children dist")

# ── Supporting figures ────────────────────────────────────────
run("plot_mood_happy.py",    "Mood(happy) tag frequency over births")
run("plot_mating_drift.py",  "LLM blend mating drift")
run("plot_action_tags.py",   "Action tag frequency and count")

print(f"\n{'='*60}")
print("  Done. Check paper_figures/ and figures/ for outputs.")
print(f"{'='*60}\n")
