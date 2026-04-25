#!/usr/bin/env python3
"""Track action tag frequency in mating genes over births.
For each birth, extract which action tags are present in the child's mating gene
and plot their frequency over time."""

import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path("/home/user/workspace/bear_repo")))

SIMS = [
    ("sim_log_mendelian_haploid.json",       "Mendelian haploid",    "#20808D"),
    ("sim_log_diploid_codominant.json",      "Diploid co-dominant",  "#A84B2F"),
    ("sim_log_llm_synthesis_free_epoch.json","LLM blend (free epoch)","#7A39BB"),
    ("sim_log_llm_synthesis_abundance_locked.json", "LLM blend (abundance)", "#D19900"),
]

# Action tags to track
TAGS = ["!breed(nearest)", "!approach(nearest)", "!mood(happy)", "!roll"]
TAG_LABELS = {
    "!breed(nearest)":    "breed(nearest)",
    "!approach(nearest)": "approach(nearest)",
    "!mood(happy)":       "mood(happy)",
    "!roll":              "roll",
}
TAG_COLORS = {
    "!breed(nearest)":    "#20808D",
    "!approach(nearest)": "#A84B2F",
    "!mood(happy)":       "#7A39BB",
    "!roll":              "#D19900",
}

WINDOW = 40
OUT_DIR = Path("/home/user/workspace/gene_drive_plots")
OUT_DIR.mkdir(exist_ok=True)

def extract_tags(text):
    return set(re.findall(r'!\w+\([^)]*\)|!\w+', text))

def rolling(x, w):
    if len(x) < w:
        return np.array(x)
    return uniform_filter1d(np.array(x, dtype=float), size=w, mode="nearest")

def main():
    # --- Plot 1: per-tag frequency over births, one panel per sim ---
    fig, axes = plt.subplots(1, len(SIMS), figsize=(16, 4.5), sharey=True)

    for ax, (fname, label, _) in zip(axes, SIMS):
        path = Path("/home/user/workspace/bear_repo") / fname
        if not path.exists():
            ax.set_visible(False)
            continue
        with open(path) as f:
            data = json.load(f)
        blog = [e for e in data.get("birth_log", [])
                if e.get("child_genes", {}).get("mating")]
        n = len(blog)
        print(f"{label}: {n} births")

        birth_idx = np.arange(n)
        for tag in TAGS:
            presence = np.array([1.0 if tag in extract_tags(
                e["child_genes"]["mating"]) else 0.0 for e in blog])
            rm = rolling(presence, WINDOW)
            ax.plot(birth_idx, rm, linewidth=2.0,
                    color=TAG_COLORS[tag], label=TAG_LABELS[tag])
            ax.scatter(birth_idx, presence, color=TAG_COLORS[tag],
                       alpha=0.12, s=8, zorder=1)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Birth number", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Fraction carrying tag", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Mating Gene: Action Tag Frequency Over Births", fontsize=13, y=1.01)
    fig.tight_layout()
    out1 = OUT_DIR / "action_tag_frequency.png"
    fig.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1.name}")

    # --- Plot 2: distribution of tag count (0,1,2,3,4) per birth over time ---
    fig2, axes2 = plt.subplots(1, len(SIMS), figsize=(16, 4.5), sharey=True)
    count_colors = ["#BCE2E7", "#20808D", "#1B474D", "#A84B2F", "#944454"]
    count_labels = ["0 tags", "1 tag", "2 tags", "3 tags", "4 tags"]

    for ax, (fname, label, _) in zip(axes2, SIMS):
        path = Path("/home/user/workspace/bear_repo") / fname
        if not path.exists():
            ax.set_visible(False)
            continue
        with open(path) as f:
            data = json.load(f)
        blog = [e for e in data.get("birth_log", [])
                if e.get("child_genes", {}).get("mating")]
        n = len(blog)

        # Tag count per birth
        counts = np.array([len(extract_tags(e["child_genes"]["mating"]) & set(TAGS))
                           for e in blog])

        # If no tags at all (e.g. LLM blend erases tags), show as annotation
        if counts.max() == 0:
            ax.text(0.5, 0.5, 'No action tags\n(LLM blending\nerases tags)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='#7A7974')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        # Window-based frequency of each count
        n_windows = max(1, n // WINDOW)
        window_centers = []
        freq_by_count = {c: [] for c in range(5)}

        for w in range(n_windows):
            chunk = counts[w*WINDOW:(w+1)*WINDOW]
            window_centers.append(w * WINDOW + WINDOW // 2)
            for c in range(5):
                freq_by_count[c].append(np.mean(chunk == c))

        wc = np.array(window_centers)
        freq_matrix = np.array([freq_by_count[c] for c in range(5)])

        ax.stackplot(wc, freq_matrix, labels=count_labels,
                     colors=count_colors, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Birth number", fontsize=10)
        if ax == axes2[0]:
            ax.set_ylabel("Fraction of births", fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="lower center", ncol=5,
                fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, -0.08))
    fig2.suptitle("Mating Gene: Number of Action Tags per Birth Over Time",
                  fontsize=13, y=1.01)
    fig2.tight_layout()
    out2 = OUT_DIR / "action_tag_count.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2.name}")

if __name__ == "__main__":
    main()
