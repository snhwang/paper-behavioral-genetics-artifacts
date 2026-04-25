#!/usr/bin/env python3
"""Compute cross-mode inheritance statistics for Table: tab:inheritance-comparison.

Outputs PGC d, GE d, p-values, and N for each breeding mode.
Run from repo root: python examples/evolutionary_ecosystem/analysis/compute_inheritance_stats.py
"""

import json
import sys
import numpy as np
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(".")))
from examples.evolutionary_ecosystem.eval.harness import get_embedder

FILES = [
    ("sim_log_mendelian_haploid.json",      "Mendelian haploid"),
    ("sim_log_diploid_codominant.json",     "Diploid co-dominant"),
    ("sim_log_llm_synthesis_free_epoch.json", "LLM blend (free epoch)"),
]

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("Loading embedder...")
    embedder = get_embedder()
    rng = np.random.default_rng(42)

    print(f"\n{'Mode':<25} {'N':>6} {'PGC d':>8} {'PGC p':>12} {'GE d':>8} {'GE p':>12} {'Mean behav':>12}")
    print("-" * 90)

    for fname, label in FILES:
        path = Path(fname)
        if not path.exists():
            print(f"{label:<25} FILE NOT FOUND: {fname}")
            continue

        with open(path) as f:
            data = json.load(f)

        blog = [e for e in data.get("birth_log", [])
                if e.get("pa_genes") and e.get("pb_genes") and e.get("child_genes")]
        n = len(blog)
        if n == 0:
            print(f"{label:<25} no valid births")
            continue

        gene_cats = list(blog[0]["child_genes"].keys())

        # PGC — per-gene cosine
        all_texts, index = [], []
        for entry in blog:
            for cat in gene_cats:
                c = entry["child_genes"].get(cat, "")
                pa = entry["pa_genes"].get(cat, "")
                pb = entry["pb_genes"].get(cat, "")
                if c and pa and pb:
                    all_texts.extend([c, pa, pb])
                    index.append(len(all_texts) - 3)

        embs = embedder.embed(all_texts, is_query=False)
        po_sims = [cosine(embs[i], (embs[i+1]+embs[i+2])/2) for i in index]
        rand_order = rng.permutation(len(po_sims))
        while np.any(rand_order == np.arange(len(po_sims))):
            rand_order = rng.permutation(len(po_sims))
        rand_sims = [cosine(embs[index[i]], (embs[index[rand_order[i]]+1]+embs[index[rand_order[i]]+2])/2)
                     for i in range(len(po_sims))]
        po = np.array(po_sims); rand = np.array(rand_sims)
        d_pgc = (po.mean()-rand.mean()) / np.sqrt((po.std()**2+rand.std()**2)/2)
        _, p_pgc = stats.ttest_ind(po, rand)

        # GE — gene embedding (full corpus)
        child_t = [" ".join(e["child_genes"].values()) for e in blog]
        pa_t    = [" ".join(e["pa_genes"].values()) for e in blog]
        pb_t    = [" ".join(e["pb_genes"].values()) for e in blog]
        all_ge  = embedder.embed(child_t + pa_t + pb_t, is_query=False)
        c_e = all_ge[:n]; pa_e = all_ge[n:2*n]; pb_e = all_ge[2*n:]
        mid_e = (pa_e + pb_e) / 2
        ge_po   = np.array([cosine(c_e[i], mid_e[i]) for i in range(n)])
        ro2 = rng.permutation(n)
        while np.any(ro2 == np.arange(n)): ro2 = rng.permutation(n)
        ge_rand = np.array([cosine(c_e[i], mid_e[ro2[i]]) for i in range(n)])
        d_ge = (ge_po.mean()-ge_rand.mean()) / np.sqrt((ge_po.std()**2+ge_rand.std()**2)/2)
        _, p_ge = stats.ttest_ind(ge_po, ge_rand)

        mean_beh = ge_po.mean()

        def fmt_p(p):
            return "≈0" if p < 1e-10 else f"{p:.2e}"

        print(f"{label:<25} {n:>6} {d_pgc:>8.2f} {fmt_p(p_pgc):>12} {d_ge:>8.2f} {fmt_p(p_ge):>12} {mean_beh:>12.3f}")

    print("\nValues go into Table: tab:inheritance-comparison in main.tex")

if __name__ == "__main__":
    main()
