#!/usr/bin/env python3
"""Plot action marker prevalence across generations.

Reads the 100k haploid + diploid sim_logs (action variants) and produces
a two-panel figure showing the fraction of births at each generation
that contain four action markers: [!flee], [!rally], [!mood(happy)],
[!breed(nearest)].

Demonstrates the LLM-mediated mutation operator's gradual purging of
literal action tokens from the gene pool.

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

from evolutionary_ecosystem.analysis.sim_log_loader import load_sim_log  # noqa: E402


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
GEN_BIN = 10


def load_births(base_prefix: str) -> list[dict]:
    """Concatenate birth_log across the 6 chunks of a 100k sim."""
    blog = []
    for f in sorted(glob.glob(f"{base_prefix}_0[1-6].json")):
        d = json.load(open(f))
        blog.extend(d.get("birth_log", []))
    return blog


def marker_freq_by_gen(blog: list[dict]) -> dict:
    """For each marker, return {gen_bin: (carriers, total)}."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # marker -> bin -> [carriers, total]
    for b in blog:
        gen = b.get("generation", 0)
        gbin = (gen // GEN_BIN) * GEN_BIN
        cg = b.get("child_genes", {})
        for cat, token, label in MARKERS:
            counts[label][gbin][1] += 1
            if token in cg.get(cat, ""):
                counts[label][gbin][0] += 1
    return counts


def plot_panel(ax, counts: dict, title: str, max_gen: int) -> None:
    for cat, token, label in MARKERS:
        bins = sorted(counts[label])
        xs, ys = [], []
        for b in bins:
            carriers, total = counts[label][b]
            if total < 5:
                continue
            if b > max_gen:
                continue
            xs.append(b + GEN_BIN / 2)
            ys.append(100 * carriers / total)
        ax.plot(xs, ys, marker="o", label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Carrier prevalence (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def main() -> None:
    hap = load_births(str(_ROOT / "sim_log_mendelian_haploid_action"))
    dip = load_births(str(_ROOT / "sim_log_diploid_codominant_action"))

    print(f"haploid births: {len(hap)}")
    print(f"diploid births: {len(dip)}")

    hap_counts = marker_freq_by_gen(hap)
    dip_counts = marker_freq_by_gen(dip)

    max_gen_h = max((b.get("generation", 0) for b in hap), default=0)
    max_gen_d = max((b.get("generation", 0) for b in dip), default=0)
    max_gen = max(max_gen_h, max_gen_d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_panel(axes[0], hap_counts, "Mendelian haploid", max_gen)
    plot_panel(axes[1], dip_counts, "Diploid co-dominant", max_gen)
    fig.suptitle("Action marker prevalence across generations (100k tick sim)",
                 fontsize=13)
    fig.tight_layout()

    out_dir = _ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_marker_decay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
