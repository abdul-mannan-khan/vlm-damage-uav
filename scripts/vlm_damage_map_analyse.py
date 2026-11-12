#!/usr/bin/env python3
"""Analysis of step 3 (results/vlm_damage/step3_map.jsonl) against the rule pre-registered in the plan file."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
R = [json.loads(l) for l in open(ROOT / 'results' / 'vlm_damage' / 'step3_map.jsonl')]
R = [r for r in R if 'error' not in r]
cp_lo = lambda k, n: float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
cp_hi = lambda k, n: float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
cell = defaultdict(dict)
for r in R:
    cell[(r['W'], r['n'], r['arm'])][r['seed']] = r


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj


out = dict(n_missions=len(R), cells={}, survey={})
for lay in ('indep', 'colo'):
    for (W, n, arm), d in sorted(cell.items()):
        rs = list(d.values()); L = [r[lay] for r in rs]
        ks = sum(x['severe_seen_t']['600'] for x in L); ns = sum(x['n_severe'] for x in L)
        fr = np.array([x['severe_seen_t']['600'] / x['n_severe'] for x in L])
        ck, cn = sum(x['cov_k'] for x in L), sum(x['cov_n'] for x in L)
        mk, mn = sum(x['severe_missed'] for x in L), sum(x['severe_assessed'] for x in L)
        tf = [x['t_first_severe_median'] for x in L if x['t_first_severe_median'] is not None]
        out['cells'][f'{lay}|{int(W)}|{n}|{arm}'] = dict(
            seeds=len(rs), J=float(np.mean([r['J_session'] for r in rs])),
            severe10_pooled=ks / ns, severe10_cp=[cp_lo(ks, ns), cp_hi(ks, ns)],
            severe10_seed_mean=float(fr.mean()), severe10_seed_sd=float(fr.std(ddof=1)),
            severe15=sum(x['severe_seen_t']['900'] for x in L) / ns,
            t_first_severe_median_s=float(np.median(tf)) if tf else None,
            map_cov=ck / cn, map_cov_cp_lo=cp_lo(ck, cn), set_size=float(np.mean([x['set_size_mean'] for x in L])),
            acc_single=float(np.mean([x['acc1'] for x in L])), acc_product=float(np.mean([x['accB'] for x in L])),
            severe_missed=mk / mn, severe_missed_cp_hi=cp_hi(mk, mn),
            area_seen15=float(np.mean([r['seen_t']['900'] / r['n_build'] for r in rs])))
    # survey arm vs MAQAC-S, paired by seed, Holm over the 6 cells
    keys, pJ, pS, rows = [], [], [], []
    for W in (1000.0, 3000.0):
        for n in (4, 6, 8):
            a, b = cell[(W, n, 'maqac_seq')], cell[(W, n, 'survey')]
            s = sorted(set(a) & set(b))
            Ja, Jb = np.array([a[k]['J_session'] for k in s]), np.array([b[k]['J_session'] for k in s])
            fa = np.array([a[k][lay]['severe_seen_t']['600'] / a[k][lay]['n_severe'] for k in s])
            fb = np.array([b[k][lay]['severe_seen_t']['600'] / b[k][lay]['n_severe'] for k in s])
            dJ = 100 * (Jb / Ja - 1)
            rows.append(dict(cell=f'{int(W)}|{n}', seeds=len(s), dJ_pct=float(dJ.mean()),
                             dJ_ci=[float(np.percentile(dJ, 2.5)), float(np.percentile(dJ, 97.5))],
                             d_severe10=float((fb - fa).mean())))
            pJ.append(ttest_rel(Jb, Ja).pvalue); pS.append(ttest_rel(fb, fa).pvalue)
    for r, a, b in zip(rows, holm(np.array(pJ)), holm(np.array(pS))):
        r['p_J_holm'], r['p_severe_holm'] = float(a), float(b)
    out['survey'][lay] = rows

# pre-registered rule (primary layout 'indep'; 'colo' reported alongside)
crit = {}
for lay in ('indep', 'colo'):
    c = out['cells'][f'{lay}|3000|6|maqac_seq']
    allk = [v for k, v in out['cells'].items() if k.startswith(lay)]
    sv = [r for r in out['survey'][lay] if r['cell'] == '3000|6'][0]
    cost_ok = sv['dJ_pct'] <= -3 and sv['p_J_holm'] < 0.05
    not_more = not (sv['d_severe10'] > 0 and sv['p_severe_holm'] < 0.05)
    crit[lay] = dict(severe10=c['severe10_pooled'], severe10_cp_lo=c['severe10_cp'][0], severe10_pass=c['severe10_pooled'] >= 0.5,
                     map_cov_min=min(v['map_cov'] for v in allk), map_cov_cp_lo_min=min(v['map_cov_cp_lo'] for v in allk),
                     cov_pass=all(v['map_cov'] >= 0.9 and v['map_cov_cp_lo'] >= 0.87 for v in allk),
                     survey_dJ=sv['dJ_pct'], survey_d_severe10=sv['d_severe10'], survey_pass=bool(cost_ok or not_more))
    crit[lay]['PASS'] = bool(crit[lay]['severe10_pass'] and crit[lay]['cov_pass'] and crit[lay]['survey_pass'])
out['criteria'] = crit
json.dump(out, open(ROOT / 'results' / 'vlm_damage' / 'step3_summary.json', 'w'), indent=1, default=float)
print('missions', len(R))
for k, v in out['cells'].items():
    print(f"{k:24s} J {v['J']:7.0f} severe@10min {v['severe10_pooled']:.2f} [{v['severe10_cp'][0]:.2f}] @15 {v['severe15']:.2f} "
          f"area15 {v['area_seen15']:.2f} cov {v['map_cov']:.3f} [{v['map_cov_cp_lo']:.3f}] set {v['set_size']:.2f} "
          f"acc {v['acc_single']:.2f}/{v['acc_product']:.2f} sevmiss {v['severe_missed']:.3f}")
for lay, rows in out['survey'].items():
    for r in rows:
        print(lay, 'survey vs MAQAC-S', r)
print(json.dumps(crit, indent=1, default=float))
