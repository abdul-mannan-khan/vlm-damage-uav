#!/usr/bin/env python3
"""Step 12 analysis (pre-registered: results/vlm_damage/step12_prereg.txt). Input results/vlm_damage/step12/step12.jsonl.
Writes results/vlm_damage/step12_summary.json."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, spearmanr

R = Path(__file__).resolve().parents[1] / 'results' / 'vlm_damage'
NS, LINKS = (6, 8, 10, 12, 16), ('full', 'limited', 'sparse')
CH, BF, RAG = 'maqac_seq_chain', 'maqac_seq_chain_bf', 'rag'


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj.tolist()


def main():
    rows = [json.loads(l) for l in open(R / 'step12' / 'step12.jsonl')]
    D = {}
    for r in rows:
        D.setdefault((r['n'], r['link'], r['method']), {})[r['seed']] = r
    cells, out = {}, {}
    for n in NS:
        for L in LINKS:
            s = sorted(D[(n, L, CH)])
            g = {m: np.array([D[(n, L, m)][k]['J_session'] for k in s]) for m in (CH, BF, RAG)}
            msg = {m: float(np.mean([D[(n, L, m)][k]['coord_msgs'] for k in s])) for m in (CH, BF, RAG)}
            disc = {m: float(np.mean([D[(n, L, m)][k]['disconnected_uav_s'] for k in s])) for m in (CH, BF, RAG)}
            col = {m: int(sum(D[(n, L, m)][k]['collisions'] or 0 for k in s)) for m in (CH, BF, RAG)}
            dms = {m: float(np.mean([D[(n, L, m)][k]['decide_ms'] for k in s])) for m in (CH, BF, RAG)}
            c = dict(pairs=len(s), J={m: float(g[m].mean()) for m in g},
                     bf_vs_rag_pct=float(np.mean(100 * (g[BF] - g[RAG]) / g[RAG])),
                     ch_vs_rag_pct=float(np.mean(100 * (g[CH] - g[RAG]) / g[RAG])),
                     bf_vs_ch_pct=float(np.mean(100 * (g[BF] - g[CH]) / g[CH])),
                     p_bf_gt_rag=float(wilcoxon(g[BF], g[RAG], alternative='greater').pvalue),
                     wins_bf_vs_rag=int((g[BF] > g[RAG]).sum()), msgs=msg, disc=disc, collisions=col, decide_ms=dms)
            cells[f'{n}|{L}'] = c
    out['cells'] = cells
    # H12a: BF > fixed order, pooled over n, per limited/sparse link
    h = {}
    for L in ('limited', 'sparse'):
        a = np.concatenate([[D[(n, L, BF)][k]['J_session'] for k in sorted(D[(n, L, CH)])] for n in NS])
        b = np.concatenate([[D[(n, L, CH)][k]['J_session'] for k in sorted(D[(n, L, CH)])] for n in NS])
        h[L] = float(wilcoxon(a, b, alternative='greater').pvalue)
    adj = holm([h['limited'], h['sparse']])
    out['H12a'] = dict(p=h, p_holm=dict(zip(('limited', 'sparse'), adj)), pass_=all(p < 0.01 for p in adj))
    # H12b: BF > RAG every n >= 8 cell on limited and sparse links, Holm over 8 cells
    keys = [f'{n}|{L}' for L in ('limited', 'sparse') for n in NS if n >= 8]
    adj = holm([cells[k]['p_bf_gt_rag'] for k in keys])
    out['H12b'] = dict(p_holm=dict(zip(keys, adj)), pass_=all(p < 0.05 for p in adj))
    # H12c: growth on sparse links
    m = [cells[f'{n}|sparse']['bf_vs_rag_pct'] for n in NS]
    rho = float(spearmanr(NS, m).correlation)
    out['H12c'] = dict(margins=m, rho=rho, pass_=rho > 0 and m[-1] > m[0])
    # H12d: fewer messages than RAG every cell
    out['H12d'] = dict(pass_=all(c['msgs'][BF] < c['msgs'][RAG] for c in cells.values()),
                       ratio={k: c['msgs'][BF] / c['msgs'][RAG] for k, c in cells.items()})
    # H12e: disconnection BF <= 1.1 x fixed-order chain, limited/sparse; 0 collisions
    lk = [k for k in cells if not k.endswith('full')]
    out['H12e'] = dict(pass_=all(cells[k]['disc'][BF] <= 1.1 * cells[k]['disc'][CH] + 1e-9 for k in lk)
                       and all(c['collisions'][BF] == 0 for c in cells.values()))
    # H12f: within 1 % of RAG on full links
    out['H12f'] = dict(margins=[cells[f'{n}|full']['bf_vs_rag_pct'] for n in NS],
                       pass_=all(abs(cells[f'{n}|full']['bf_vs_rag_pct']) <= 1 or cells[f'{n}|full']['bf_vs_rag_pct'] > 0 for n in NS))
    json.dump(out, open(R / 'step12_summary.json', 'w'), indent=1)
    print(f"{'cell':12s} {'BF-RAG%':>8s} {'CH-RAG%':>8s} {'BF-CH%':>7s} {'p':>8s} wins  msgs BF/CH/RAG       disc BF/CH/RAG     ms BF/CH/RAG")
    for k, c in cells.items():
        print(f"{k:12s} {c['bf_vs_rag_pct']:8.2f} {c['ch_vs_rag_pct']:8.2f} {c['bf_vs_ch_pct']:7.2f} {c['p_bf_gt_rag']:8.1e} {c['wins_bf_vs_rag']:3d}  "
              f"{c['msgs'][BF]:5.0f}/{c['msgs'][CH]:4.0f}/{c['msgs'][RAG]:5.0f}  {c['disc'][BF]:6.1f}/{c['disc'][CH]:6.1f}/{c['disc'][RAG]:6.1f}  "
              f"{c['decide_ms'][BF]:5.1f}/{c['decide_ms'][CH]:4.1f}/{c['decide_ms'][RAG]:4.1f}")
    for k in ('H12a', 'H12b', 'H12c', 'H12d', 'H12e', 'H12f'):
        print(k, 'PASS' if out[k]['pass_'] else 'FAIL', {kk: v for kk, v in out[k].items() if kk not in ('pass_', 'ratio')})


if __name__ == '__main__':
    main()
