# Behavioral Genetics — Paper Artifacts

**Provisional Patent Pending (filed April 15, 2026)** | Copyright (c) 2026 The Pennsylvania State University. All rights reserved.
Inventor: Scott N. Hwang

Licensed under the Open Core Ventures Source Available License (OCVSAL) v1.0. See [LICENSE](LICENSE). Production use requires a commercial agreement. For commercial licensing, contact the Penn State Office of Technology Transfer at ottinfo@psu.edu.

Evaluation scripts, simulation module, and result files for the behavioral-genetics
paper (preprint link TBD; will point to arXiv or equivalent once posted, and to
the journal DOI if/when accepted). Uses the BEAR library at
[snhwang/bear](https://github.com/snhwang/bear), pinned to `v0.1.0`.

## Layout

```
evolutionary_ecosystem/
├── eval/                   # 14 eval scripts + harness + __init__
│   ├── eval1_population_dynamics.py         # §11.10
│   ├── eval2_dual_pathway_ablation.py       # §11.11
│   ├── eval3_inheritance_fidelity.py        # §11.12
│   ├── eval3b_locus_breeding.py             # §11.12b
│   ├── eval4_epoch_phenotype_shift.py       # §11.13
│   ├── eval5_ga_baseline.py                 # §11.14
│   ├── eval5b_llm_breeding.py               # §11.14b (LLM)
│   ├── eval6_dialogue_quality.py            # §11.15
│   ├── eval6b_llm_dialogue.py               # §11.15b (LLM)
│   ├── eval7_mutation_diversity.py          # §11.16
│   ├── eval7b_llm_mutation.py               # §11.16b (LLM)
│   ├── eval8_evolution_dynamics.py          # §11.17
│   ├── eval8_diploid_diversity.py           # §11.18
│   ├── eval9_diploid_selection.py           # §11.19
│   └── harness.py
└── server/                 # simulation module (creatures, genes, epochs, stats)
    ├── epochs.py
    ├── gene_engine.py
    ├── sim.py
    └── stats.py

results/                    # 63 paper-canonical result files
SIM_LOGS.md                 # Zenodo pointer for 13 simulation logs (~266 MB)
run_evals.sh                # runner for §11.10–§11.19
requirements.txt            # bear@v0.1.0 + scipy/numpy/python-dotenv/openai/PyYAML
```

The live-demo webserver pieces (`server/app.py`, `server/brain.py`) and their
FastAPI dependencies are **not** included here — they live in the full
`bear/examples/evolutionary_ecosystem/` directory. This artifacts repo contains
only what the 14 eval scripts need.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
./run_evals.sh                  # Part A: headless ecosystem evals (~60 min, no LLM)
./run_evals.sh --all            # Part B: add LLM-mediated evals (needs Anthropic or local LLM)
```

Part A is deterministic against the frozen bear `v0.1.0` and should reproduce
the paper's numbers closely. Part B involves LLM sampling and will diverge
on each run; the committed `results/` files are the paper's reported runs.

## Simulation logs (Zenodo)

The 13 `sim_log_*.json` files the paper analyzes (~266 MB total, four files
>42 MB) are archived on Zenodo rather than in git — see `SIM_LOGS.md` for the
DOI and file-by-file index. Paper numbers in `results/` do not require these
logs to regenerate; they're included as the raw simulation traces for
independent analysis.

## Bear version

Pinned to bear `v0.1.0` (commit `515366e`). Bumping bear will likely change
numeric results; update the pin in `requirements.txt` and re-run the full
suite before comparing to older results.
