"""Plot eval10 memory-inheritance results.

Reads eval10_results.json (written by eval10_memory_inheritance.py) and renders a
two-panel figure:
  (left)  transmission rate of the acquired memory in the F1 and F2 generations,
          with 95% Wilson CIs and the expected crossover (0.5) reference line;
  (right) expression recall of the inherited memory in carriers (F1, F2) versus
          the control false-carrier rate.

Usage:
    python evolutionary_ecosystem/analysis/plot_memory_inheritance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE.parent / "eval" / "eval10_results.json"
OUT = _HERE / "eval10_memory_inheritance.png"


def main() -> None:
    d = json.loads(RESULTS.read_text())

    tr, trci = d["transmission_rate"], d["transmission_ci95"]
    f2, f2ci = d["f2_transmission_rate"], d["f2_transmission_ci95"]
    rec = d["expression_recall_in_carriers"]
    f2rec = d["f2_expression_recall"]
    ctrl_rate = d["total_control_false_carriers"] / d["total_offspring"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # --- Panel A: transmission ---
    vals = [tr, f2]
    cis = [trci, f2ci]
    lo = [v - c[0] for v, c in zip(vals, cis)]
    hi = [c[1] - v for v, c in zip(vals, cis)]
    bars = ax1.bar(["F1", "F2"], vals, yerr=[lo, hi], capsize=8,
                   color=["#4C72B0", "#55A868"], width=0.55, edgecolor="black", linewidth=0.6)
    ax1.axhline(0.5, ls="--", color="gray", lw=1.2)
    ax1.text(1.45, 0.515, "expected 0.5\n(crossover rate)", ha="right", va="bottom",
             fontsize=9, color="gray")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("transmission rate")
    ax1.set_title(f"Inheritance of acquired memory\n(n={d['total_offspring']} offspring)")
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", fontsize=9)

    # --- Panel B: expression ---
    labels = ["carriers\n(F1)", "carriers\n(F2)", "control"]
    evals = [rec, f2rec, ctrl_rate]
    ebars = ax2.bar(labels, evals, color=["#4C72B0", "#55A868", "#C44E52"],
                    width=0.6, edgecolor="black", linewidth=0.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("memory expression recall")
    ax2.set_title("Expression of inherited memory")
    for b, v in zip(ebars, evals):
        ax2.text(b.get_x() + b.get_width() / 2, min(v + 0.03, 1.0), f"{v:.3f}",
                 ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
