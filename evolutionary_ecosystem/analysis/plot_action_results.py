#!/usr/bin/env python3
"""Generate figures for action log analysis results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

OUT_DIR = Path(".")

def ci95(data):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    return m, se * stats.t.ppf(0.975, n-1)

def prop_ci95(k, n):
    p = k/n
    return p, 1.96 * np.sqrt(p*(1-p)/n)

print("Loading data...")
with open("sim_log_mendelian_haploid_action.json") as f:
    hap = json.load(f)
with open("sim_log_diploid_codominant_action.json") as f:
    dip = json.load(f)

dlog_h = hap.get("death_log", [])
dlog_d = dip.get("death_log", [])
alog_h = hap.get("action_log", [])
alog_d = dip.get("action_log", [])

EPOCHS = ["abundance", "expansion", "predator_bloom", "ice_age", "famine"]
EPOCH_LABELS = ["Abundance", "Expansion", "Predator\nBloom", "Ice Age", "Famine"]
HAP_COLOR = "#20808D"
DIP_COLOR = "#A84B2F"
FONT = 13

# ─────────────────────────────────────────────────────────────
# Figure 1: mood(happy) fitness effect — children count
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

for ax, (label, dlog, color) in zip(axes, [
    ("Mendelian haploid", dlog_h, HAP_COLOR),
    ("Diploid co-dominant", dlog_d, DIP_COLOR),
]):
    with_tag    = [e["children"] for e in dlog if "!mood(happy)" in e.get("genes",{}).get("mating","")]
    without_tag = [e["children"] for e in dlog if "!mood(happy)" not in e.get("genes",{}).get("mating","")]

    means = [np.mean(with_tag), np.mean(without_tag)]
    cis   = [ci95(with_tag)[1], ci95(without_tag)[1]]
    ns    = [len(with_tag), len(without_tag)]

    bars = ax.bar(["With\n[!mood(happy)]", "Without\n[!mood(happy)]"],
                  means, yerr=cis, capsize=6,
                  color=[color, "#BCE2E7"], edgecolor="white",
                  linewidth=0.5, error_kw={"linewidth": 1.5})

    for bar, m, ci, n in zip(bars, means, cis, ns):
        ax.text(bar.get_x() + bar.get_width()/2, m + ci + 0.1,
                f"{m:.2f}\n(N={n})", ha="center", va="bottom", fontsize=10)

    t, p = stats.ttest_ind(with_tag, without_tag)
    d = (np.mean(with_tag)-np.mean(without_tag)) / np.sqrt((np.std(with_tag)**2+np.std(without_tag)**2)/2)
    ax.set_title(f"{label}\nd={d:.2f}, p{'≈0' if p < 1e-10 else f'={p:.2e}'}", fontsize=FONT)
    ax.set_ylabel("Mean offspring count", fontsize=FONT)
    ax.set_ylim(0, max(means) * 1.3)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.suptitle("Fitness Effect of [!mood(happy)] Allele: Offspring Count at Death",
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_mood_fitness.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved fig_mood_fitness.png")

# ─────────────────────────────────────────────────────────────
# Figure 2: flee vs rally by epoch — both modes side by side
# ─────────────────────────────────────────────────────────────
x = np.arange(len(EPOCHS))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 5))

offsets = [-1.5, -0.5, 0.5, 1.5]
series = [
    ("Haploid flee",  alog_h, "flee",  HAP_COLOR,    "--"),
    ("Haploid rally", alog_h, "rally", HAP_COLOR,    "-"),
    ("Diploid flee",  alog_d, "flee",  DIP_COLOR,    "--"),
    ("Diploid rally", alog_d, "rally", DIP_COLOR,    "-"),
]

for i, (label, alog, action, color, ls) in enumerate(series):
    vals, errs = [], []
    for ep in EPOCHS:
        pred = [e for e in alog if e["trigger"]=="predator" and e["epoch"]==ep]
        if pred:
            k = sum(1 for e in pred if e[action])
            p, ci = prop_ci95(k, len(pred))
        else:
            p, ci = 0, 0
        vals.append(p * 100)
        errs.append(ci * 100)

    alpha = 0.85 if "flee" in label else 0.55
    hatch = "" if "flee" in label else "///"
    bars = ax.bar(x + offsets[i]*width, vals, width,
                  yerr=errs, capsize=4,
                  color=color, alpha=alpha, hatch=hatch,
                  edgecolor="white", linewidth=0.4,
                  error_kw={"linewidth": 1.2},
                  label=label)

ax.set_xticks(x)
ax.set_xticklabels(EPOCH_LABELS, fontsize=11)
ax.set_ylabel("Response rate (%)", fontsize=FONT)
ax.set_title("Predator Response by Epoch: Flee vs Rally\n(Mendelian haploid vs Diploid co-dominant)",
             fontsize=FONT)
ax.legend(fontsize=10, framealpha=0.85, loc="upper right")
ax.set_ylim(0, 105)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_flee_rally_epoch.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved fig_flee_rally_epoch.png")

# ─────────────────────────────────────────────────────────────
# Figure 3: children count distribution — haploid vs diploid
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ch_h = [e["children"] for e in dlog_h]
ch_d = [e["children"] for e in dlog_d]
max_c = max(max(ch_h), max(ch_d))
bins = np.arange(0, max_c + 2) - 0.5

mh, cih = ci95(ch_h)
md, cid = ci95(ch_d)
t, p = stats.ttest_ind(ch_h, ch_d)
d = (md - mh) / np.sqrt((np.std(ch_h)**2 + np.std(ch_d)**2)/2)

ax.hist(ch_h, bins=bins, density=True, alpha=0.65, color=HAP_COLOR,
        label=f"Mendelian haploid  μ={mh:.2f}±{cih:.2f} (N={len(ch_h)})")
ax.hist(ch_d, bins=bins, density=True, alpha=0.65, color=DIP_COLOR,
        label=f"Diploid co-dominant  μ={md:.2f}±{cid:.2f} (N={len(ch_d)})")

ax.axvline(mh, color=HAP_COLOR, linewidth=2, linestyle="--")
ax.axvline(md, color=DIP_COLOR, linewidth=2, linestyle="--")

ax.set_xlabel("Offspring count at death", fontsize=FONT)
ax.set_ylabel("Density", fontsize=FONT)
ax.set_title(f"Offspring Count Distribution\nd={d:.2f}, p≈0 (diploid produces more offspring)",
             fontsize=FONT)
ax.legend(fontsize=10, framealpha=0.85)
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_children_dist.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved fig_children_dist.png")

print("\nAll figures saved.")
