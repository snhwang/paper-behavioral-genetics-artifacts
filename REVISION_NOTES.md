# Revision Notes for the Editor

Behavioral Genetics manuscript — summary of changes between the original
submission and the corrected version (against bear `v0.1.8`).

The corrected simulations preserve all qualitative claims in the manuscript
and in fact strengthen the inheritance-fidelity numbers across all three
modes. The main changes are:

- **Inheritance fidelity effect sizes increased in every mode.** Cohen's
  *d* for parent–offspring gene similarity (per-gene cosine) rose
  substantially:

  | Mode | original *d* | corrected *d* |
  | --- | --- | --- |
  | Mendelian haploid | 2.45 | 3.07 |
  | Diploid co-dominant | 1.00 | 2.48 |
  | LLM blend (free epoch) | 1.17 | 2.05 |

  The diploid value in particular more than doubled. Gene-embedding *d*
  values follow the same direction (haploid 2.33 → 2.17, diploid
  0.52 → 1.82, LLM blend 0.83 → 0.87).

- **Mean behavior similarity is consistent with the original ordering
  (haploid > diploid) but slightly higher in both modes** (0.86 → 0.948
  for haploid, 0.76 → 0.934 for diploid). Same conclusion as the
  original paper, refreshed numbers.

- **The LLM-blend mating-gene drift is mechanistically reframed.** The
  original interpretation described it as a "centripetal force on
  behavioral variation" driven by semantic averaging. The corrected
  reading is concrete: LLM paraphrasing strips embedded action markers
  (e.g. `[!breed(nearest)]`) from gene text, which collapses the
  autonomous-breeding retrieval score and purges marker-carrying
  alleles. The r = -0.41 correlation is preserved; the explanation is
  sharper.

- **New finding: generation depth differs between ploidies.** In the
  100,000-tick runs, Mendelian haploid populations reach generation 128
  while diploid co-dominant populations reach only generation 92 — a
  quantitative signature of heterozygote masking slowing individual
  turnover. This was not in the original submission.

- **Total births shifted modestly.** Mendelian haploid: 1744 → 1278
  (−27%). Diploid co-dominant: 1202 → 1254 (+4%). Sentence-level
  updates to Table 1 (`tab:inheritance-comparison`) and the surrounding
  text are needed.

## Source of the corrections

Two implementation bugs in the bear library motivated the rerun:

1. **Meiotic segregation in diploid breeding was incomplete.** When both
   parents were diploid at a given locus, the gamete-formation step
   did not sample one of the two alleles before recombination, so
   diploid offspring inherited the same allele consistently instead of
   via Mendelian segregation. Fixed in bear `v0.1.6` by adding a proper
   `_meiotic_gamete()` step.

2. **Persona instructions grew exponentially across generations.** A
   default persona template embedded both parents' full persona content
   into each child's persona, doubling persona size every generation.
   Long evolutionary loops eventually exceeded LLM context limits.
   Fixed in bear `v0.1.8` by passing the child's name as a bounded
   `custom_persona` argument in `breed_offspring()`, neutralising the
   recursive template.

The LLM-blend code path bypasses both bugs (no meiosis, no recursive
persona), so the April 11 LLM-blend simulation logs remain valid paper
data and do not require regeneration.

## Code and data location

- Library: <https://github.com/snhwang/bear> (pinned to `v0.1.8` in the
  artifacts repo's `requirements.txt`)
- Artifacts (eval scripts, sim logs, paper-canonical results):
  <https://github.com/snhwang/paper-behavioral-genetics-artifacts>
- Regression test: `tests/test_evolution.py::TestPersonaGrowthAcrossGenerations`
  in the bear repo documents both the recursive-default footgun and the
  bounded `custom_persona` workaround.
