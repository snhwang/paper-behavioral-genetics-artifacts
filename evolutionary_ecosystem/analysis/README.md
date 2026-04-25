# Analysis Scripts

Figure generation scripts for the behavioral genetics paper.
Run from the repo root: `python examples/evolutionary_ecosystem/analysis/<script>.py`

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

## Dependencies

```
pip install matplotlib numpy scipy sentence-transformers
```

The scripts import from `examples/evolutionary_ecosystem/eval/harness.py` for the embedder.
Run from the repo root so imports resolve correctly.
