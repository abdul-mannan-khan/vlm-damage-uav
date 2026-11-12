#!/usr/bin/env python3
"""Step 8 analysis (pre-registered 10 Nov 2026, plan file "Step 8 plan").
P2: results/vlm_damage/step8_netsim.jsonl -> H5a (limited links, MAQAC-S > RAG on J, Wilcoxon p < 0.01 at n = 6; Holm
    over n = 6, 8, 10) and H5b (full links, |mean relative difference| <= 2 % at every n), plus cost/safety descriptives.
P3: results/vlm_damage/jetson/<run>/ -> onboard model vs the bf16 s0 model (same images), single-seed re-calibration
    (V1 temperature / V2 vector scaling on val half A, conformal on half B), latency bar (city frames p90 <= 5 s).
Writes results/vlm_damage/step8_summary.json.

    python3 scripts/step8_analyse.py [--jetson results/vlm_damage/jetson/<run>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_damage_analyse import ece, f1_score, temp_scale, softmax, conformal, R   # noqa: E402
from vlm_damage_posthoc import split, vec_fit, load as load_bf16                    # noqa: E402


def holm(ps):
    o = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out.tolist()


def p2(path):
    rows = [json.loads(l) for l in open(path)]
    S, ps = {}, []
    for n in (6, 8, 10):
        for link in ('full', 'limited'):
            A = {r['seed']: r for r in rows if r['n'] == n and r['link'] == link and r['method'] == 'maqac_seq'}
            B = {r['seed']: r for r in rows if r['n'] == n and r['link'] == link and r['method'] == 'rag'}
            sd = sorted(set(A) & set(B))
            if not sd:
                continue
            d = np.array([A[s]['J_session'] - B[s]['J_session'] for s in sd])
            rel = np.array([A[s]['J_session'] / B[s]['J_session'] - 1 for s in sd])
            c = dict(pairs=len(sd), J_maqac_s=float(np.mean([A[s]['J_session'] for s in sd])),
                     J_rag=float(np.mean([B[s]['J_session'] for s in sd])), diff_mean=float(d.mean()),
                     rel_diff_mean_pct=100 * float(rel.mean()), positive=int((d > 0).sum()), negative=int((d < 0).sum()),
                     wilcoxon_p=float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0)
            for lab, X in (('maqac_s', A), ('rag', B)):
                for q in ('coord_msgs', 'coord_iters', 'decide_ms', 'decide_ms_p95', 'disconnected_uav_s', 'near_misses',
                          'collisions', 'mean_hops', 'wall_s'):
                    v = [X[s][q] for s in sd if X[s].get(q) is not None]
                    c[f'{lab}_{q}'] = float(np.mean(v)) if v else None
            c['host'] = sorted({X[s]['host'] for X in (A, B) for s in sd})
            S[f'{n}|{link}'] = c
            if link == 'limited':
                ps.append((f'{n}|{link}', c['wilcoxon_p']))
    for (k, _), h in zip(ps, holm([p for _, p in ps])):
        S[k]['holm_p'] = h
    crit = dict(H5a_PASS=bool('6|limited' in S and S['6|limited']['wilcoxon_p'] < 0.01),
                H5b_PASS=bool(all(abs(S[f'{n}|full']['rel_diff_mean_pct']) <= 2 for n in (6, 8, 10) if f'{n}|full' in S)),
                episodes=len(rows))
    return dict(cells=S, criteria=crit)


def onboard(run, split_name):
    r = [json.loads(l) for l in open(run / f'{split_name}.jsonl')]
    return {x['id']: x for x in r}


def p3(run, batch_start=None):
    run = Path(run)
    summ = json.load(open(run / 'summary.json'))
    out = dict(info=summ['info'], tegrastats=summ.get('tegrastats'), latency={})
    for ph, v in summ['phases'].items():
        out['latency'][ph] = v
    city = out['latency'].get('city', {}).get('total_ms')
    if batch_start is not None and (run / 'load.jsonl').exists():     # load rows split by the batch start (run log 10 Nov)
        ld = [json.loads(l) for l in open(run / 'load.jsonl')]
        for lab, rows in (('load_idle', [r for r in ld if r['t'] < batch_start]), ('load_with_batch', [r for r in ld if r['t'] >= batch_start])):
            v = np.array([r['total_ms'] for r in rows])
            out['latency'][lab] = dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)),
                                       max=float(v.max())) if len(v) else dict(n=0)
    out['latency_bar_PASS'] = bool(city and city['p90'] <= 5000)
    ids = {sp: [json.loads(l)['id'] for l in open(R / 'tune' / f'eval_sft_s0_{sp}.jsonl') if json.loads(l)['q'] == 'damage']
           for sp in ('val', 'test')}
    L, Y, Lb = {}, {}, {}
    for sp in ('val', 'test'):
        ob = onboard(run, sp)
        Lb[sp], Y[sp] = load_bf16(0, sp, 'damage')
        L[sp] = np.array([ob[k]['logits'] for k in ids[sp]], float)
        assert all(ob[k]['y'] == y for k, y in zip(ids[sp], Y[sp])), 'label mismatch with the training cache'
    agree = {sp: dict(argmax_agree=float((L[sp].argmax(1) == Lb[sp].argmax(1)).mean()),
                      mean_abs_dp=float(np.abs(softmax(L[sp], 1) - softmax(Lb[sp], 1)).sum(1).mean() / 2))
             for sp in ('val', 'test')}
    out['vs_bf16_s0'] = agree
    A = split(Y['val']); B = ~A
    res = {}
    for name, LL in (('onboard_int8', L), ('bf16_s0', Lb)):
        t1 = temp_scale(LL['val'][A], Y['val'][A])
        P1c, P1t = softmax(LL['val'][B], t1), softmax(LL['test'], t1)
        g = vec_fit(LL['val'][A], Y['val'][A])
        P2c, P2t = g(LL['val'][B]), g(LL['test'])
        for v, (Pc, Pt) in (('V1', (P1c, P1t)), ('V2', (P2c, P2t))):
            cv = conformal(Pc, Y['val'][B], Pt, Y['test'])
            res[f'{name}|{v}'] = dict(f1=f1_score(Y['test'], Pt.argmax(1)), ece=ece(Pt, Y['test']), cov=cv[0], cp_lo=cv[1], size=cv[2])
    out['calibration'] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--netsim', default=str(R / 'step8_netsim.jsonl'))
    ap.add_argument('--jetson', default=None)
    ap.add_argument('--batch-start', type=float, default=None, help='epoch s when the planner batch started on the Jetson')
    a = ap.parse_args()
    S = {}
    if Path(a.netsim).exists():
        S['P2'] = p2(a.netsim)
        for k, c in S['P2']['cells'].items():
            print(k, 'J %.2f vs %.2f (%+.2f %%) +%d/-%d p %.2g' % (c['J_maqac_s'], c['J_rag'], c['rel_diff_mean_pct'], c['positive'],
                  c['negative'], c['wilcoxon_p']), 'msgs %.1f vs %.1f' % (c['maqac_s_coord_msgs'], c['rag_coord_msgs']),
                  'decide ms %.1f vs %.1f' % (c['maqac_s_decide_ms'], c['rag_decide_ms']),
                  'disc s %.0f vs %.0f' % (c['maqac_s_disconnected_uav_s'], c['rag_disconnected_uav_s']))
        print(S['P2']['criteria'])
    if a.jetson:
        S['P3'] = p3(a.jetson, a.batch_start)
        print(json.dumps({k: S['P3'][k] for k in ('latency_bar_PASS', 'vs_bf16_s0', 'calibration', 'tegrastats')}, indent=1))
    (R / 'step8_summary.json').write_text(json.dumps(S, indent=1, default=float))


if __name__ == '__main__':
    main()
