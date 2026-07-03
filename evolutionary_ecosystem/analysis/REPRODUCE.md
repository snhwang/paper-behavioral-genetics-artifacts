# Reproducing the Figures and Tables in the Paper

This document reproduces every **data-backed** figure and table in the paper
*"Behavioral inheritance and evolution in LLM-controlled agent populations"*
(Artificial Life submission).

**Run all commands from the repository root.**

---

## Dependencies

```bash
pip install -r requirements.txt
```

This installs `bear` (pinned to `v0.1.8` from the public repo) plus `numpy`,
`scipy`, `matplotlib`, `sentence-transformers`, and `torch`. The embedding model
(`BAAI/bge-base-en-v1.5`) downloads automatically on first use.

No GPU or LLM endpoint is required to reproduce any figure or table. The only
step that can use an LLM is the *optional* re-extraction phase of the memory
experiment (§6); a cached extraction is committed so the result reproduces
deterministically without one.

---

## Data: committed vs. Zenodo

**Every input needed by the scripts below is committed in this repository** —
headless eval results under `evolutionary_ecosystem/eval/results/` and `results/`,
and chunked simulation logs (`sim_log_*_NN.json`) at the repo root.

The **complete raw simulation logs** are additionally archived on Zenodo as a
citable dataset:

> **Simulation logs** — doi.org/[10.5281/zenodo.21151214](https://doi.org/10.5281/zenodo.21151214)

The Zenodo logs are the full-resolution raw archive; they are **not required** to
reproduce any number or figure here (the committed chunked logs suffice). See
[`SIM_LOGS.md`](../../SIM_LOGS.md).

---

## Quick reproduction from committed results (no simulation, no LLM)

These commands regenerate the paper's figures and print its table statistics
directly from the committed result files:

```bash
# fig:inheritance, fig:epoch-heatmap (+ eval4 shift)  ->  figures/*.png
python evolutionary_ecosystem/analysis/regen_paper_figures.py

# tab:inheritance-comparison  (prints d, p per breeding mode)
python evolutionary_ecosystem/analysis/compute_inheritance_stats.py

# tab:selection-pressure  (prints flee-vs-rally d, t, p under mutation rate 0)
python evolutionary_ecosystem/analysis/aggregate_selection.py

# fig:marker-decay  ->  figures/fig_marker_decay.png
python evolutionary_ecosystem/analysis/plot_marker_decay.py

# fig:memory-inheritance  (headless, from cached extraction)  ->  eval10 figure
python evolutionary_ecosystem/eval/eval10_memory_inheritance.py
python evolutionary_ecosystem/analysis/plot_memory_inheritance.py
```

`tab:population-dynamics` needs no command — its values are the committed
`results/eval1_results.json` (see §3).

---

## Per-item reproduction

### 1. fig:inheritance + tab:inheritance — inheritance fidelity

```bash
python evolutionary_ecosystem/analysis/regen_paper_figures.py
```

- **Inputs:** `evolutionary_ecosystem/eval/results/eval3_v2_results_{abundance_ice_age,famine,ice_age,predator_bloom,expansion}.json` *[committed]*
- **Output:** `figures/eval3_inheritance.png`; per-epoch Cohen's *d* and *p*-values are in the input files.
- **LLM required:** no.

### 2. fig:epoch-heatmap + tab:epoch-shift — epoch-driven phenotype shift

```bash
python evolutionary_ecosystem/analysis/regen_paper_figures.py   # same run as item 1
```

- **Inputs:** `evolutionary_ecosystem/eval/results/eval4_v2_results_merged_final.json` *[committed]*
- **Outputs:** `figures/eval4_epoch_heatmap.png` (and `figures/eval4_epoch_shift.png`); ANOVA *F*-statistics per behavioral dimension printed to stdout.
- **LLM required:** no.

### 3. tab:population-dynamics — multi-generational dynamics (~340 generations)

- **Committed values:** `results/eval1_results.json` *[committed]* —
  max generation `340.6 ± 24.9`, total births `1,284.4 ± 8.2`, gene diversity `0.204 ± 0.021` (5 trials).
- **To regenerate from scratch** (headless simulation, ~200k ticks × 5 seeds; no LLM):

  ```bash
  python evolutionary_ecosystem/eval/eval1_population_dynamics.py
  ```

  Writes `results/eval1_results.json` and `results/eval1_dynamics.png`.
- **LLM required:** no.

### 4. tab:inheritance-comparison — inheritance fidelity across breeding modes

```bash
python evolutionary_ecosystem/analysis/compute_inheritance_stats.py
```

- **Inputs (auto-merged from chunks by `sim_log_loader`):**
  `sim_log_mendelian_haploid_01..06.json`, `sim_log_diploid_codominant_01..06.json`,
  `sim_log_llm_synthesis_free_epoch.json` *[all committed]*
- **Output:** prints N, per-gene-cosine *d*/*p*, gene-embedding *d*/*p*, and mean behavior per mode.
- **LLM required:** no (uses `bge-base-en-v1.5` embeddings).

### 5. tab:selection-pressure + fig:marker-decay — selection on action markers

```bash
python evolutionary_ecosystem/analysis/aggregate_selection.py   # tab:selection-pressure
python evolutionary_ecosystem/analysis/plot_marker_decay.py     # fig:marker-decay
```

- **Inputs:**
  `sim_log_selection_haploid_seed{42,142,242,342,442}_0N.json`,
  `sim_log_selection_diploid_codominant_seed{...}_0N.json` (mutation rate 0),
  `sim_log_mendelian_haploid_action_01..06.json`,
  `sim_log_diploid_codominant_action_01..06.json` (default mutation rate) *[all committed]*
- **Outputs:** `evolutionary_ecosystem/eval/results/selection_pressure_results.json` (+ stdout);
  `figures/fig_marker_decay.png`.
- **LLM required:** no.

### 6. fig:memory-inheritance — inheritance of an acquired (Lamarckian) memory

```bash
# Phase 2 — deterministic headless measurement from the committed extraction:
python evolutionary_ecosystem/eval/eval10_memory_inheritance.py
python evolutionary_ecosystem/analysis/plot_memory_inheritance.py
```

- **Inputs:** `evolutionary_ecosystem/eval/eval10_memories.json` (cached LLM extraction),
  `evolutionary_ecosystem/eval/eval10_results.json` *[both committed]*
- **Output:** `evolutionary_ecosystem/analysis/eval10_memory_inheritance.png`
  (the paper's `figures/eval10_memory_inheritance.png`).
- **Optional Phase 1** — re-extract memories with an LLM (regenerates the committed
  `eval10_memories.json`; **requires** an OpenAI-compatible endpoint):

  ```bash
  python evolutionary_ecosystem/eval/eval10_memory_inheritance.py \
      --extract --base-url http://localhost:11434/v1 --model gemma4:e2b
  ```
- **LLM required:** only for the optional Phase 1.

---

## Figures and tables with no reproduction script

These are hand-authored in the manuscript and have no data pipeline:

- **fig:governance-pipeline**, **fig:lifecycle** — TikZ diagrams in the manuscript source (`body.tex`).
- **fig:sim-screenshot** — screenshot of the running simulation.
- **tab:blending-example**, **tab:gene-categories**, **tab:phenotype**, **tab:epochs**,
  **tab:related-comparison** — hand-constructed in `body.tex`.
- **alg:breeding**, **alg:gap-detection** — pseudocode listings.

---

## Regenerating the underlying data from scratch (optional)

Items 3 and 6 above already document their from-scratch reruns. For the eval3 /
eval4 inputs consumed by `regen_paper_figures.py`:

```bash
# Requires an OpenAI-compatible LLM endpoint for the breeding/expression calls.
python evolutionary_ecosystem/eval/eval3_inheritance_fidelity.py
python evolutionary_ecosystem/eval/eval4_epoch_phenotype_shift.py
```

Note: these eval scripts write their result JSON to `results/` (repo root), whereas
`regen_paper_figures.py` reads the committed copies under
`evolutionary_ecosystem/eval/results/`. To plot a fresh run, copy the regenerated
`eval3_v2_results_*.json` / `eval4_v2_results_*.json` into
`evolutionary_ecosystem/eval/results/` before running the figure script.
