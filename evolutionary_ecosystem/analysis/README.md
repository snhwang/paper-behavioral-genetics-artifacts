# Analysis Scripts

Figure generation scripts for the behavioral genetics paper.
Run from the repo root: `python evolutionary_ecosystem/analysis/<script>.py`

## Paper figures (eval data)

**`regen_paper_figures.py`** — Regenerates all main paper figures (eval3 inheritance, eval4 epoch shift, heatmap) from headless eval result files.
- Inputs: `eval3_v2_results_*.json`, `eval4_v2_results_merged_final.json`
- Outputs: `figures/eval3_inheritance.png`, `figures/eval4_epoch_shift.png`, `figures/eval4_epoch_heatmap.png`

## Live sim analysis scripts

**`plot_action_results.py`** — Main action log analysis figures (3 figures):
- `fig_mood_fitness.png` — [!mood(happy)] allele fitness effect on offspring count
- `fig_flee_rally_epoch.png` — Flee vs rally rates by epoch for Mendelian and diploid
- `fig_children_dist.png` — Offspring count distribution comparison
- Inputs: `sim_log_mendelian_haploid_action.json`, `sim_log_diploid_codominant_action.json`

**`plot_mood_happy.py`** — [!mood(happy)] tag frequency over births for Mendelian vs diploid.
- Inputs: `sim_log_mendelian_haploid_action.json`, `sim_log_diploid_codominant_action.json`

**`plot_action_tags.py`** — Action tag frequency and count over births across all modes.
- Inputs: `sim_log_mendelian_haploid.json`, `sim_log_diploid_codominant.json`, blend logs

**`plot_mating_drift.py`** — Mating gene semantic drift over births (LLM blend).
- Input: `sim_log_llm_synthesis_abundance_locked.json`

**`plot_gene_drive.py`** — BEAR retrieval strength per behavioral dimension over births (all modes).
- Inputs: `sim_log_mendelian_haploid.json`, `sim_log_diploid_codominant.json`, blend logs

**`plot_gene_diversity.py`** — Pairwise cosine distance distributions over time windows.
- Inputs: `sim_log_mendelian_haploid.json`, `sim_log_diploid_codominant.json`

**`plot_allele_frequency.py`** — Founding archetype allele frequency over births (stacked area).
- Inputs: `sim_log_mendelian_haploid.json`, `sim_log_diploid_codominant.json`

## Statistics and quantitative analysis (no plotting)

**`mutation_novelty.py`** — Mutation-driven novelty (paper `sec:mutation-novelty`). Reconstructs, from `birth_log` records, the per-locus rate at which an offspring gene matches *neither* parent, i.e. de novo content introduced by the mutation/drift operators. Reports haploid 13.9% (mutation on) vs 0% (off), and the diploid figures (26.7% on / 13.5% off) with the hidden-allele confound explained (logs store only the expressed allele, so recessive variation surfacing through diploid inheritance reads as "novel"). No new sims.
- Inputs: `sim_log_mendelian_haploid_action_*`, `sim_log_diploid_codominant_action_*` (mutation on, μ=0.15); `sim_log_selection_haploid_seed*_*`, `sim_log_selection_diploid_codominant_seed*_*` (mutation off, μ=0)
- Output: `results/mutation_novelty_results.json`

**`aggregate_selection.py`** — Selection-pressure statistics (paper `sec:selection-pressure` / selection table). Welch t-test and Cohen's d for flee-vs-rally and other action markers, aggregated across 5 seeds per ploidy from the `death_log` offspring counts.
- Inputs: `sim_log_selection_{haploid,diploid_codominant}_seed*_*.json`
- Output: `results/selection_pressure_results.json`

**`compute_inheritance_stats.py`** — Cross-mode inheritance statistics for `tab:inheritance-comparison` (Mendelian vs diploid vs LLM-blend fidelity: per-gene cosine, gene-embedding, behavior-profile).

**`plot_marker_decay.py`** — Action-marker prevalence across generations (paper `fig:marker-decay`): default-rate decay (top row, μ=0.15) vs mutation-rate-zero (bottom row).

## Reproduce-all and utilities

**`reproduce_all.py`** — Regenerate all paper figures and tables in one run.

**`sim_log_loader.py`** — Helper to load a `sim_log_*.json` written by `server/app.py`.

## Dependencies

```
pip install matplotlib numpy scipy sentence-transformers
```

The scripts import from `evolutionary_ecosystem/eval/harness.py` for the embedder.
Run from the repo root so imports resolve correctly.
