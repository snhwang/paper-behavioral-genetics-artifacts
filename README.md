# Behavioral Genetics — Paper Artifacts

**Provisional Patent Pending (filed April 15, 2026)** | Copyright (c) 2026 The Pennsylvania State University. All rights reserved.
Inventor: Scott N. Hwang

Licensed under the Open Core Ventures Source Available License (OCVSAL) v1.0. See [LICENSE](LICENSE). Production use requires a commercial agreement. For commercial licensing, contact the Penn State Office of Technology Transfer at ottinfo@psu.edu.

Evaluation scripts, simulation module, and result files for the behavioral-genetics
paper (preprint link TBD; will point to arXiv or equivalent once posted, and to
the journal DOI if/when accepted). Uses the BEAR library at
[snhwang/bear](https://github.com/snhwang/bear), pinned to `v0.1.8`.

## What's in the paper vs. supplementary

The paper's main text directly cites a small subset of the evals here. The
remaining scripts are supplementary diagnostics that probe other facets of
the BEAR + genetics pipeline; they are included for completeness but are
not load-bearing for any quantitative claim in the manuscript.

| Eval | LLM? | In paper? | Section / figure |
| --- | --- | --- | --- |
| eval1_population_dynamics      | no  | **yes** | `sec:eval-population-dynamics`, `figures/eval1_dynamics.png` |
| eval3_inheritance_fidelity     | no  | **yes** | `sec:eval-evoeco-inherit`, `figures/eval3_inheritance.png` |
| eval4_epoch_phenotype_shift    | no  | **yes** | `sec:eval-evoeco-epoch`, `figures/eval4_epoch_shift.png`, `figures/eval4_epoch_heatmap.png` |
| eval5b_llm_breeding            | yes | **yes** | `sec:eval-breeding-sim` |
| 100k live sims (haploid + diploid co-dominant) | yes | **yes** | `sec:action-log` — action-log + predator-response analysis |
| eval2_dual_pathway_ablation    | no  | no      | supplementary |
| eval2b_bear_on_off             | yes | no      | supplementary (live-sim ablation) |
| eval3b_locus_breeding          | no  | no      | supplementary |
| eval4b_epoch_phenotype_llm     | yes | no      | supplementary (live-sim epoch shift) |
| eval5_ga_baseline              | no  | no      | supplementary |
| eval6_dialogue_quality         | no  | no      | supplementary |
| eval6b_llm_dialogue            | yes | no      | supplementary |
| eval7_mutation_diversity       | no  | no      | supplementary |
| eval7b_llm_mutation            | yes | no      | supplementary |
| eval8_evolution_dynamics       | no  | no      | supplementary |
| eval8_diploid_diversity        | no  | no      | supplementary |
| eval9_diploid_selection        | no  | no      | supplementary |
| eval9b_diploid_selection       | yes | no      | supplementary (live-sim diploid vs haploid) |

If you only want to reproduce the published numbers, you can run the four
paper-cited evals plus regenerate the two 100k live-sim logs — everything
else is optional.

## Layout

```
evolutionary_ecosystem/
├── eval/                   # 14 eval scripts + harness + __init__
│   ├── eval1_population_dynamics.py         # paper §eval-population-dynamics
│   ├── eval2_dual_pathway_ablation.py       # supplementary
│   ├── eval3_inheritance_fidelity.py        # paper §eval-evoeco-inherit
│   ├── eval3b_locus_breeding.py             # supplementary
│   ├── eval4_epoch_phenotype_shift.py       # paper §eval-evoeco-epoch
│   ├── eval5_ga_baseline.py                 # supplementary
│   ├── eval5b_llm_breeding.py               # paper §eval-breeding-sim (LLM)
│   ├── eval6_dialogue_quality.py            # supplementary
│   ├── eval6b_llm_dialogue.py               # supplementary (LLM)
│   ├── eval7_mutation_diversity.py          # supplementary
│   ├── eval7b_llm_mutation.py               # supplementary (LLM)
│   ├── eval8_evolution_dynamics.py          # supplementary
│   ├── eval8_diploid_diversity.py           # supplementary
│   ├── eval9_diploid_selection.py           # supplementary
│   └── harness.py
└── server/                 # simulation module (creatures, genes, epochs, stats)
    ├── epochs.py
    ├── gene_engine.py
    ├── sim.py
    └── stats.py

results/                    # paper-canonical result files
SIM_LOGS.md                 # Zenodo pointer for the live-sim logs
run_evals.sh                # runner for all evals (Part A deterministic, Part B LLM)
run_sim_100k_haploid.sh     # 100k-tick haploid live sim (paper figure data)
run_sim_100k_diploid.sh     # 100k-tick diploid co-dominant live sim (paper figure data)
requirements.txt            # bear@v0.1.8 + numerics + FastAPI for live-sim runs
```

The artifacts repo is **self-contained**: `run.py`, `server/app.py`, and
`server/brain.py` are vendored from `bear/examples/evolutionary_ecosystem/`
(at the same v0.1.8 tag pinned in `requirements.txt`) so the live-sim
evals and direct invocations of `run.py` work without needing a separate
bear checkout. The FastAPI / uvicorn / websockets deps required by the
live sim are listed in `requirements.txt`. If you patch bear and want
the artifacts to follow, re-copy the three files from your local bear
checkout and bump the pin.

## Reproduce just the paper numbers

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Part A: the three deterministic paper-cited evals (no LLM needed)
python3 evolutionary_ecosystem/eval/eval1_population_dynamics.py
python3 evolutionary_ecosystem/eval/eval3_inheritance_fidelity.py
python3 evolutionary_ecosystem/eval/eval4_epoch_phenotype_shift.py

# Part B: the one LLM-mediated paper-cited eval (needs an OpenAI-compatible LLM)
python3 evolutionary_ecosystem/eval/eval5b_llm_breeding.py \
    --backend local --base-url http://127.0.0.1:8355/v1 --model gemma-4-e2b

# Part C: the two 100k-tick live sims that anchor sec:action-log
./run_sim_100k_haploid.sh
./run_sim_100k_diploid.sh
```

## Reproduce everything (paper + supplementary)

```bash
./run_evals.sh                  # Part A: headless ecosystem evals (no LLM)
./run_evals.sh --all            # Part B: add LLM-mediated evals (needs Anthropic or local LLM)
```

Part A is deterministic against the frozen bear `v0.1.8` and should reproduce
the paper's numbers closely. Part B involves LLM sampling and will diverge
on each run; the committed `results/` files are the paper's reported runs.

## Simulation logs (Zenodo)

The `sim_log_*.json` files the paper analyzes (large; some chunks >12 MB)
are archived on Zenodo rather than in git — see `SIM_LOGS.md` for the DOI
and file-by-file index. Paper numbers in `results/` do not require these
logs to regenerate; they're included as the raw simulation traces for
independent analysis.

## Bear version

Pinned to bear `v0.1.8`. Numeric results depend on the bear version;
bumping bear may shift point estimates (qualitative conclusions
typically hold). Update the pin in `requirements.txt` and re-run the
relevant evals before comparing to older results.

Notable bear changes since the initial submission:
- `v0.1.7`: per-allele dominance scores; unified `DOMINANT`/`CODOMINANT`
  expression rule (max-score wins, ties produce codominance);
  `_meiotic_gamete()` for proper Mendelian segregation across generations.
- `v0.1.8`: `breed_offspring` passes `custom_persona=child_name` to bear's
  `breed()` so the PERSONA instruction stays bounded — bear's default
  recursive template otherwise doubles persona content every generation
  and eventually crashes long evolutionary loops.
