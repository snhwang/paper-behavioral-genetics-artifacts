#!/usr/bin/env python3
"""Aggregate selection-pressure analysis across 5 seeds per ploidy."""
import json, glob
import numpy as np
from scipy import stats

def load(prefix):
    dlog = []
    for f in sorted(glob.glob(f'{prefix}_0[1-9].json')):
        d = json.load(open(f))
        dlog.extend(d.get('death_log', []))
    return dlog

SEEDS = [42, 142, 242, 342, 442]
MARKERS = [
    ('predator_defense','!flee'),
    ('predator_defense','!rally'),
    ('predator_defense','!challenge'),
    ('mating','!mood(happy)'),
    ('mating','!breed(nearest)'),
]

out = {'haploid': {}, 'diploid_codominant': {}}
for ploidy in ['haploid', 'diploid_codominant']:
    prefix = f'sim_log_selection_{ploidy}'
    per_seed_d = {f'{cat}/{m}': [] for cat,m in MARKERS}
    per_seed_d['flee_vs_rally'] = []
    all_flee, all_rally = [], []
    for seed in SEEDS:
        dlog = load(f'{prefix}_seed{seed}')
        for cat, m in MARKERS:
            c = [e['children'] for e in dlog if m in e.get('genes',{}).get(cat,'')]
            n = [e['children'] for e in dlog if m not in e.get('genes',{}).get(cat,'')]
            if c and n:
                pool = np.sqrt((np.std(c,ddof=1)**2+np.std(n,ddof=1)**2)/2)
                d = (np.mean(c)-np.mean(n))/pool if pool>0 else 0
                per_seed_d[f'{cat}/{m}'].append(d)
        flee = [e['children'] for e in dlog if '!flee' in e.get('genes',{}).get('predator_defense','')]
        rally = [e['children'] for e in dlog if '!rally' in e.get('genes',{}).get('predator_defense','')]
        all_flee.extend(flee); all_rally.extend(rally)
        if flee and rally:
            pool = np.sqrt((np.std(flee,ddof=1)**2+np.std(rally,ddof=1)**2)/2)
            per_seed_d['flee_vs_rally'].append((np.mean(flee)-np.mean(rally))/pool)

    summary = {}
    for k, vals in per_seed_d.items():
        if vals:
            summary[k] = {'mean_d': float(np.mean(vals)),
                          'sem_d':  float(np.std(vals,ddof=1)/np.sqrt(len(vals))),
                          'min':    float(min(vals)),
                          'max':    float(max(vals)),
                          'n_seeds': len(vals)}
    pool = np.sqrt((np.std(all_flee,ddof=1)**2+np.std(all_rally,ddof=1)**2)/2)
    d_agg = (np.mean(all_flee)-np.mean(all_rally))/pool
    t, p = stats.ttest_ind(all_flee, all_rally, equal_var=False)
    summary['flee_vs_rally_aggregate'] = {
        'N_flee': len(all_flee), 'mean_flee': float(np.mean(all_flee)),
        'sem_flee': float(np.std(all_flee,ddof=1)/np.sqrt(len(all_flee))),
        'N_rally': len(all_rally), 'mean_rally': float(np.mean(all_rally)),
        'sem_rally': float(np.std(all_rally,ddof=1)/np.sqrt(len(all_rally))),
        'cohens_d': float(d_agg), 't': float(t), 'p_value': float(p),
    }
    out[ploidy] = summary

out['metadata'] = {
    'description': 'Selection-pressure analysis on 30k-tick sims with mutation_rate=0',
    'seeds': SEEDS,
    'n_ticks_per_sim': 30000,
    'n_seeds': len(SEEDS),
    'bear_version': '0.1.8',
}

import json
with open('evolutionary_ecosystem/eval/results/selection_pressure_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved results/selection_pressure_results.json')
print(json.dumps(out, indent=2)[:2000])
