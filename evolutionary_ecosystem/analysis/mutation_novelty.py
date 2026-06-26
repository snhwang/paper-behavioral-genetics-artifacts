#!/usr/bin/env python3
"""Mutation-driven novelty analysis (paper section sec:mutation-novelty).

Quantifies how much *de novo* heritable variation the mutation and drift
operators introduce, reconstructed entirely from the saved birth records of
runs already reported in the paper. No new simulations are required.

Method
------
Each entry in a sim_log's ``birth_log`` stores the child's expressed genes
(``child_genes``) together with both parents' expressed genes (``pa_genes``,
``pb_genes``), keyed by gene category. For every inherited gene we ask whether
the child's text matches at least one parent (inherited unchanged) or matches
neither (novel content produced by the mutation/drift operators). The
per-locus novelty rate is the fraction of inherited gene loci that match
neither parent.

We run this over two configurations that already exist in the data:
  * mutation ON  : the 100k-tick action-log sims at the default rate 0.15
                   (sim_log_mendelian_haploid_action_*, sim_log_diploid_codominant_action_*)
  * mutation OFF : the 30k-tick selection sims at rate 0
                   (sim_log_selection_haploid_seed*_*, sim_log_selection_diploid_codominant_seed*_*)

Interpretation and the diploid confound
---------------------------------------
Haploid is clean: a haploid child can only inherit a parent's single visible
allele, so with mutation OFF the novelty rate is exactly 0, and with mutation
ON it equals the realized per-locus mutation rate.

Diploid is confounded because the logs record only each individual's
*expressed* allele, not its full genotype. A diploid child can express an
allele inherited from a parent's *hidden* (unexpressed) allele; that allele
never appeared in the parent's log, so it reads as "novel" even though it was
inherited. The mutation-OFF diploid run isolates this effect: any novelty
there is recessive variation surfacing through diploid inheritance, not
mutation. Subtracting that baseline from the mutation-ON diploid rate
recovers an approximate mutation contribution (the ON and OFF diploid runs
differ in length and selection pressure, so it is only approximate).

Reported figures (bear v0.1.8 data committed in this repo)
----------------------------------------------------------
  haploid  ON  : 13.94% novel loci (1,959 / 14,058), 82.9% of births mutated
  haploid  OFF : 0.00%  novel loci (0 / 21,120)
  diploid  ON  : 26.72% novel loci  (mutation + hidden-allele surfacing)
  diploid  OFF : 13.45% novel loci  (hidden-allele surfacing only)
  => diploid mutation contribution ~= 26.72 - 13.45 ~= 13.3%, consistent with haploid.

Usage
-----
Run from the repo root (where the sim_log_*.json files live)::

    python evolutionary_ecosystem/analysis/mutation_novelty.py

Writes results/mutation_novelty_results.json and prints a summary table.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

OUT = Path("results/mutation_novelty_results.json")

# (glob pattern, label, mutation rate) for each configuration present in the repo.
CONFIGS = [
    ("sim_log_mendelian_haploid_action_0[1-9].json",            "haploid_action",   0.15),
    ("sim_log_diploid_codominant_action_0[1-9].json",           "diploid_action",   0.15),
    ("sim_log_selection_haploid_seed*_0[1-9].json",             "haploid_selection", 0.0),
    ("sim_log_selection_diploid_codominant_seed*_0[1-9].json",  "diploid_selection", 0.0),
]


def _alleles(value) -> list[str]:
    """Flatten a gene value to a list of allele strings.

    Logs store a single expressed string per category; this also tolerates
    dict/list shapes in case future logs record full genotypes.
    """
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, dict):
        return [a for v in value.values() for a in _alleles(v)]
    if isinstance(value, list):
        return [a for v in value for a in _alleles(v)]
    return [str(value)]


def assess(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    loci = novel = births = births_with_novel = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for e in d.get("birth_log", []):
            cg = e.get("child_genes") or {}
            pa = e.get("pa_genes") or {}
            pb = e.get("pb_genes") or {}
            if not cg:
                continue
            births += 1
            had_novel = False
            for cat, cval in cg.items():
                parent = set(_alleles(pa.get(cat, ""))) | set(_alleles(pb.get(cat, "")))
                for allele in set(_alleles(cval)):
                    if not allele:
                        continue
                    loci += 1
                    if allele not in parent:
                        novel += 1
                        had_novel = True
            if had_novel:
                births_with_novel += 1
    return {
        "files": len(files),
        "births": births,
        "loci_examined": loci,
        "novel_loci": novel,
        "novel_pct": round(100 * novel / loci, 2) if loci else None,
        "births_with_novel_pct": round(100 * births_with_novel / births, 1) if births else None,
    }


def main() -> None:
    out = {}
    print(f"{'config':22s} {'mu':>4s} {'births':>7s} {'loci':>7s} {'novel%':>7s} {'births_mut%':>11s}")
    print("-" * 64)
    for pattern, label, mu in CONFIGS:
        r = assess(pattern)
        r["mutation_rate"] = mu
        out[label] = r
        print(f"{label:22s} {mu:>4.2f} {r['births']:>7d} {r['loci_examined']:>7d} "
              f"{str(r['novel_pct']):>7s} {str(r['births_with_novel_pct']):>11s}")

    # Approximate diploid mutation contribution, baseline-subtracted.
    don = out.get("diploid_action", {}).get("novel_pct")
    doff = out.get("diploid_selection", {}).get("novel_pct")
    if don is not None and doff is not None:
        out["diploid_mutation_contribution_approx_pct"] = round(don - doff, 2)
        print(f"\nDiploid mutation contribution (approx) = {don} - {doff} = {round(don - doff, 2)}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
