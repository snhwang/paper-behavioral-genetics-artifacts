#!/usr/bin/env python3
"""Plot action marker prevalence across generations.

Produces a 2x2 figure comparing marker dynamics under default mutation
rate (0.15, top row) versus mutation rate zero (bottom row), in both
haploid and diploid co-dominant populations.

Top row uses the 100k figure-data sims to show LLM-mediated mutation
purging literal action tokens from the gene pool over generations.

Bottom row uses the 30k mutation-rate-zero sims aggregated over 5
seeds per ploidy. With mutation disabled, founder marker frequencies
stay roughly stable across generations and selection visibly shifts
the favored/disfavored markers in opposite directions.

Output: figures/fig_marker_decay.png at the repo root.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))


MARKERS = [
    ("predator_defense", "!flee",           "[!flee]"),
    ("predator_defense", "!rally",          "[!rally]"),
    ("mating",           "!mood(happy)",    "[!mood(happy)]"),
    ("mating",           "!breed(nearest)", "[!breed(nearest)]"),
]
COLORS = {
    "[!flee]":            "#1f77b4",
    "[!rally]":           "#ff7f0e",
    "[!mood(happy)]":     "#2ca02c",
    "[!breed(nearest)]":  "#d62728",
}
GEN_BIN_DEFAULT = 10
GEN_BIN_MUTZERO = 5


def load_births(base_prefix: str) -> list[dict]:
    """Concatenate birth_log across chunks of one simulation."""
    blog = []
    for f in sorted(glob.glob(f"{base_prefix}_0[1-9].json")):
        d = json.load(open(f))
        blog.extend(d.get("birth_log", []))
    return blog


def load_births_seeds(prefix_template: str, seeds: list[int]) -> list[dict]:
    """Aggregate births across all seeds for one condition."""
    blog = []
    for seed in seeds:
        blog.extend(load_births(prefix_template.format(seed=seed)))
    return blog


def marker_freq_by_gen(blog: list[dict], gen_bin: int) -> dict:
    """Return {marker_label: {gen_bin_start: [carriers, total]}}."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for b in blog:
        gen = b.get("generation", 0)
        gbin = (gen // gen_bin) * gen_bin
        cg = b.get("child_genes", {})
        for cat, token, label in MARKERS:
            counts[label][gbin][1] += 1
            if token in cg.get(cat, ""):
                counts[label][gbin][0] += 1
    return counts


def plot_panel(ax, counts: dict, title: str, max_gen: int, gen_bin: int,
               show_legend: bool, show_ylabel: bool) -> None:
    for cat, token, label in MARKERS:
        bins = sorted(counts[label])
        xs, ys = [], []
        for b in bins:
            carriers, total = counts[label][b]
            if total < 5:
                continue
            if b > max_gen:
                continue
            xs.append(b + gen_bin / 2)
            ys.append(100 * carriers / total)
        ax.plot(xs, ys, marker="o", label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("Generation")
    if show_ylabel:
        ax.set_ylabel("Carrier prevalence (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)


def main() -> None:
    # Top row: 100k figure-data sims with default mutation rate
    hap_default = load_births(str(_ROOT / "sim_log_mendelian_haploid_action"))
    dip_default = load_births(str(_ROOT / "sim_log_diploid_codominant_action"))

    # Bottom row: 30k mutation-rate-zero sims, aggregated over 5 seeds
    SEEDS = [42, 142, 242, 342, 442]
    hap_mutzero = load_births_seeds(
        str(_ROOT / "sim_log_selection_haploid_seed{seed}"), SEEDS)
    dip_mutzero = load_births_seeds(
        str(_ROOT / "sim_log_selection_diploid_codominant_seed{seed}"), SEEDS)

    print(f"haploid default-rate births: {len(hap_default)}")
    print(f"diploid default-rate births: {len(dip_default)}")
    print(f"haploid mut=0 births (5 seeds): {len(hap_mutzero)}")
    print(f"diploid mut=0 births (5 seeds): {len(dip_mutzero)}")

    hap_default_counts = marker_freq_by_gen(hap_default, GEN_BIN_DEFAULT)
    dip_default_counts = marker_freq_by_gen(dip_default, GEN_BIN_DEFAULT)
    hap_mutzero_counts = marker_freq_by_gen(hap_mutzero, GEN_BIN_MUTZERO)
    dip_mutzero_counts = marker_freq_by_gen(dip_mutzero, GEN_BIN_MUTZERO)

    # Determine x-axis limits per row
    max_gen_default = max(
        max((b.get("generation", 0) for b in hap_default), default=0),
        max((b.get("generation", 0) for b in dip_default), default=0),
    )
    max_gen_mutzero = max(
        max((b.get("generation", 0) for b in hap_mutzero), default=0),
        max((b.get("generation", 0) for b in dip_mutzero), default=0),
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

    plot_panel(axes[0, 0], hap_default_counts,
               "Mendelian haploid, default mutation rate (0.15)",
               max_gen_default, GEN_BIN_DEFAULT,
               show_legend=True, show_ylabel=True)
    plot_panel(axes[0, 1], dip_default_counts,
               "Diploid co-dominant, default mutation rate (0.15)",
               max_gen_default, GEN_BIN_DEFAULT,
               show_legend=False, show_ylabel=False)
    plot_panel(axes[1, 0], hap_mutzero_counts,
               "Mendelian haploid, mutation rate 0 (5 seeds aggregated)",
               max_gen_mutzero, GEN_BIN_MUTZERO,
               show_legend=False, show_ylabel=True)
    plot_panel(axes[1, 1], dip_mutzero_counts,
               "Diploid co-dominant, mutation rate 0 (5 seeds aggregated)",
               max_gen_mutzero, GEN_BIN_MUTZERO,
               show_legend=False, show_ylabel=False)

    fig.suptitle(
        "Action marker prevalence across generations\n"
        "Top: 100k-tick sims with default mutation (substrate erosion). "
        "Bottom: 30k-tick sims with mutation disabled (substrate preserved).",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = _ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_marker_decay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
