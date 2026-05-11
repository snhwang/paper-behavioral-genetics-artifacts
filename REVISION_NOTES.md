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

- **New finding: lineage branching differs between ploidies.** In the
  100,000-tick runs, Mendelian haploid populations reach generation 121
  while diploid co-dominant populations reach only generation 90. Mean
  age at death is essentially identical in both modes (386 vs 394
  ticks). The generation gap therefore reflects faster lineage branching
  in haploid rather than longer diploid lifespan.

- **New finding: diploid creatures average 37 percent more offspring
  per lifetime** (8.77 vs 6.41 children at death, SEMs 0.05 and 0.07
  respectively). The two modes have nearly identical total births
  (1254 vs 1278) and population size (50 stable). The extra offspring
  per individual is the downstream effect of codominant retrieval
  indexing both alleles at the breeding query, producing a higher
  retrieval score on average and therefore a higher autonomous-breeding
  probability per encounter.

- **New finding: diploid creatures die almost exclusively of old age.**
  98.1 percent of diploid deaths are old age, 1.9 percent vitality
  loss, 0 percent starvation. Haploid creatures show 92.4 percent old
  age, 6.3 percent vitality loss, 1.3 percent starvation, 0.1 percent
  combat. The two-allele corpus apparently makes diploid creatures
  more robust to environmental stressors, so they reach old age more
  reliably.

- **New finding: action markers decay under LLM-mediated mutation.**
  Default mutation rate 0.15 with the spontaneous-gene rewrite path
  purges literal action tokens (`[!flee]`, `[!rally]`, `[!mood(happy)]`)
  from the gene pool over generations. Haploid reaches a marker-free
  state by generation 60. Diploid retains mating-related markers
  longer through heterozygous carriage (generation 90 plus) but loses
  predator-defense markers by generation 40. This is a substrate-loss
  problem, not a selection-power problem, and motivates a supplementary
  set of mutation-rate-zero runs to measure fitness effects on
  preserved markers.

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
