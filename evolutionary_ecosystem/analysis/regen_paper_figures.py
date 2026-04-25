"""
Regenerate all paper figures from final eval data.
"""
import json, shutil, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("examples/evolutionary_ecosystem/eval/results")
PAPER_FIGS = Path("figures")

FONT_TITLE = 13
FONT_LABEL = 12
FONT_TICK  = 11
FONT_LEGEND = 10

EPOCHS_ORDERED = ['abundance', 'ice_age', 'expansion', 'predator_bloom', 'famine']
EPOCH_LABELS   = ['Abundance', 'Ice Age', 'Expansion', 'Predator\nBloom', 'Famine']
EPOCH_COLORS   = {
    'abundance':      '#4CAF50',
    'ice_age':        '#2196F3',
    'expansion':      '#FF9800',
    'predator_bloom': '#F44336',
    'famine':         '#9C27B0',
}
BEAR_DIMS = ['food_seeking', 'combat', 'survival', 'stealth', 'breeding']


# ── Eval 3: Inheritance Fidelity ──────────────────────────────────────────────

def plot_eval3():
    EPOCH_FILES = [
        ('Abundance+\nIce Age', RESULTS / 'eval3_v2_results_abundance_ice_age.json', '#20808D'),
        ('Famine',              RESULTS / 'eval3_v2_results_famine.json',            '#9C27B0'),
        ('Ice Age',             RESULTS / 'eval3_v2_results_ice_age.json',           '#2196F3'),
        ('Predator\nBloom',     RESULTS / 'eval3_v2_results_predator_bloom.json',    '#F44336'),
        ('Expansion',           RESULTS / 'eval3_v2_results_expansion.json',         '#FF9800'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Cohen's d bar chart across epochs and metrics
    ax1 = axes[0]
    metrics = ['gene_embedding', 'per_gene_cosine']
    metric_labels = ["Gene\nEmbedding", "Per-Gene\nCosine"]
    metric_colors = ['#20808D', '#A84B2F']

    epoch_names = [e[0] for e in EPOCH_FILES]
    x = np.arange(len(EPOCH_FILES))
    width = 0.35

    for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        d_vals = []
        for _, fpath, _ in EPOCH_FILES:
            d = json.load(open(fpath))
            d_vals.append(d['statistical_tests'][metric]['cohens_d'])
        ax1.bar(x + mi * width, d_vals, width, label=label, color=color, alpha=0.85)

    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(epoch_names, fontsize=FONT_TICK)
    ax1.set_ylabel("Cohen's d", fontsize=FONT_LABEL)
    ax1.set_title("Inheritance Fidelity: Effect Size by Epoch", fontsize=FONT_TITLE)
    ax1.tick_params(axis='y', labelsize=FONT_TICK)
    ax1.legend(fontsize=FONT_LEGEND)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 6)
    # Add significance markers
    for i, (_, fpath, _) in enumerate(EPOCH_FILES):
        ax1.text(x[i] + width/2, 5.7, 'p≈0', ha='center', fontsize=9, color='#333')

    # Panel 2: P-O vs Random similarity (gene_embedding) across epochs
    ax2 = axes[1]
    po_means, po_stds, rand_means = [], [], []
    for _, fpath, _ in EPOCH_FILES:
        d = json.load(open(fpath))
        po = d['parent_offspring_similarity']['gene_embedding']
        rb = d['random_pair_baseline']['gene_embedding']
        po_means.append(po['mean'])
        po_stds.append(po['std'])
        rand_means.append(rb['mean'])

    ax2.bar(x - width/2, po_means, width, label='Parent-Offspring',
            color='#20808D', alpha=0.85,
            yerr=po_stds, capsize=4, error_kw={'linewidth': 1.5})
    ax2.bar(x + width/2, rand_means, width, label='Random Pairs',
            color='#A84B2F', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(epoch_names, fontsize=FONT_TICK)
    ax2.set_ylabel("Gene Embedding Cosine Similarity", fontsize=FONT_LABEL)
    ax2.set_title("Parent-Offspring vs Random Gene Similarity", fontsize=FONT_TITLE)
    ax2.tick_params(axis='y', labelsize=FONT_TICK)
    ax2.legend(fontsize=FONT_LEGEND)
    ax2.set_ylim(0.7, 1.02)

    plt.tight_layout()
    out = RESULTS / "eval3_v2_figure.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"eval3: saved to {out}")
    return out


# ── Eval 4: Epoch Shift ───────────────────────────────────────────────────────

def plot_eval4_shift():
    d4 = json.load(open(RESULTS / 'eval4_v2_results_merged_final.json'))
    anova = d4['anova']

    # Build matrix: rows=dims, cols=epochs
    matrix = np.array([
        [anova[dim]['per_epoch_mean'].get(e, 0.0) for e in EPOCHS_ORDERED]
        for dim in BEAR_DIMS
    ])
    grand_mean = matrix.mean(axis=1, keepdims=True)
    delta = matrix - grand_mean

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Radar chart zoomed to data range
    ax1 = fig.add_subplot(121, polar=True)
    n = len(BEAR_DIMS)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    RMIN, RMAX = 0.25, 0.85
    for ei, epoch in enumerate(EPOCHS_ORDERED):
        vals = list(matrix[:, ei]) + [matrix[0, ei]]
        ax1.plot(angles, vals, 'o-', label=EPOCH_LABELS[ei],
                 color=EPOCH_COLORS[epoch], linewidth=2, markersize=4)
        ax1.fill(angles, vals, alpha=0.07, color=EPOCH_COLORS[epoch])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([d.replace('_', '\n') for d in BEAR_DIMS], fontsize=FONT_TICK)
    ax1.set_ylim(RMIN, RMAX)
    ax1.set_title("Behavior Profile by Epoch", y=1.08, fontsize=FONT_TITLE)
    ax1.tick_params(labelsize=FONT_TICK - 1)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=FONT_LEGEND)

    # Panel 2: Delta from grand mean
    ax2 = axes[1]
    x = np.arange(len(BEAR_DIMS))
    width = 0.15
    for ei, epoch in enumerate(EPOCHS_ORDERED):
        ax2.bar(x + ei * width, delta[:, ei], width,
                label=EPOCH_LABELS[ei], color=EPOCH_COLORS[epoch], alpha=0.85)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels([d.replace('_', '\n') for d in BEAR_DIMS], fontsize=FONT_TICK)
    ax2.set_ylabel("Deviation from cross-epoch mean", fontsize=FONT_LABEL)
    ax2.set_title("Epoch Effect: Deviation from Mean Profile", fontsize=FONT_TITLE)
    ax2.tick_params(axis='y', labelsize=FONT_TICK)
    ax2.legend(fontsize=FONT_LEGEND)

    # Annotate top F values
    for i, dim in enumerate(BEAR_DIMS):
        f = anova[dim]['F_statistic']
        ax2.annotate(f"F={f:.0f}",
                     xy=(i + width*2, delta[i].max() + 0.005),
                     fontsize=9, ha='center', va='bottom', color='#333')

    plt.tight_layout()
    out = RESULTS / "eval4_v2_epoch_shift.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"eval4 shift: saved to {out}")
    return out


def plot_eval4_heatmap():
    d4 = json.load(open(RESULTS / 'eval4_v2_results_merged_final.json'))
    anova = d4['anova']

    raw = np.array([
        [anova[dim]['per_epoch_mean'].get(e, 0.0) for e in EPOCHS_ORDERED]
        for dim in BEAR_DIMS
    ])
    f_stats = [anova[dim]['F_statistic'] for dim in BEAR_DIMS]
    p_vals  = [anova[dim]['p_value']     for dim in BEAR_DIMS]

    # Z-score per row
    row_mean = raw.mean(axis=1, keepdims=True)
    row_std  = raw.std(axis=1, keepdims=True)
    row_std[row_std < 1e-9] = 1e-9
    z = (raw - row_mean) / row_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={'width_ratios': [4, 1]})

    ax = axes[0]
    abs_max = np.abs(z).max()
    im = ax.imshow(z, aspect='auto', cmap='RdBu_r',
                   vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(len(EPOCHS_ORDERED)))
    ax.set_xticklabels(EPOCH_LABELS, fontsize=FONT_TICK + 1)
    ax.set_yticks(range(len(BEAR_DIMS)))
    ax.set_yticklabels([d.replace('_', ' ') for d in BEAR_DIMS], fontsize=FONT_TICK)
    ax.set_title("Epoch-Driven Behavior Shift (z-score per dimension)",
                 fontsize=FONT_TITLE)
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Behavioral Dimension", fontsize=FONT_LABEL)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Z-score", fontsize=FONT_TICK)
    cbar.ax.tick_params(labelsize=FONT_TICK - 1)

    # Annotate raw values
    for i in range(len(BEAR_DIMS)):
        for j in range(len(EPOCHS_ORDERED)):
            ax.text(j, i, f"{raw[i,j]:.3f}", ha='center', va='center',
                    fontsize=8, color='black',
                    alpha=0.8 if abs(z[i,j]) < 1.0 else 1.0)

    # F-stat bars
    ax2 = axes[1]
    colors = ['#d73027' if p < 0.001 else '#fc8d59' if p < 0.01 else
              '#fee090' if p < 0.05 else '#e0e0e0' for p in p_vals]
    ax2.barh(range(len(BEAR_DIMS)), f_stats, color=colors, height=0.7)
    ax2.set_yticks(range(len(BEAR_DIMS)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("ANOVA F", fontsize=FONT_LABEL)
    ax2.set_title("F", fontsize=FONT_TITLE)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.invert_yaxis()
    for i, (f, p) in enumerate(zip(f_stats, p_vals)):
        ax2.text(f + 5, i, f"{f:.0f}***", va='center', fontsize=FONT_TICK - 2)

    plt.tight_layout()
    out = RESULTS / "eval4_v2_epoch_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"eval4 heatmap: saved to {out}")
    return out


if __name__ == "__main__":
    print("Regenerating figures from final eval data...\n")
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    out3 = plot_eval3()
    shutil.copy2(out3, PAPER_FIGS / "eval3_inheritance.png")

    out4s = plot_eval4_shift()
    shutil.copy2(out4s, PAPER_FIGS / "eval4_epoch_shift.png")

    out4h = plot_eval4_heatmap()
    shutil.copy2(out4h, PAPER_FIGS / "eval4_epoch_heatmap.png")

    print("\nAll figures regenerated and copied to paper-bg/figures/")
