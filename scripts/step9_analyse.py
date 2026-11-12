#!/usr/bin/env python3
"""Step 9 analysis (pre-registered 10 Nov 2026, plan file "Step 9 plan"): chain-connectivity rule for MAQAC-S.
Inputs: results/vlm_damage/step9/step9_chain.jsonl (Vast), step9_plain_check.jsonl (Vast, plain MAQAC-S, cross-machine
check), results/vlm_damage/step8_netsim.jsonl (Jetson; plain MAQAC-S and RAG). Writes results/vlm_damage/step9_summary.json.
H6a  250 m: chain has fewer disconnected UAV-s than plain MAQAC-S at every n (one-sided Wilcoxon, Holm over n, p < 0.01).
H6b  250 m: mean relative J difference chain vs plain >= -1 % at every n.
H6c  250 m: chain beats RAG on J at n = 8 and 10 (one-sided Wilcoxon, Holm over the two, p < 0.01); n = 6 reported.
H6d  default links: |mean relative J difference chain vs plain| <= 1 % at every n.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parents[1] / 'results' / 'vlm_damage'


def rows(p):
    return {(r['n'], r['link'], r['seed']): r for r in map(json.loads, open(p))}


def holm(ps):
    o = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); out[i] = run
    return out.tolist()


def wx(d):
    return float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0


def main():
    s8 = [json.loads(l) for l in open(R / 'step8_netsim.jsonl')]
    plain = {(r['n'], r['link'], r['seed']): r for r in s8 if r['method'] == 'maqac_seq'}
    rag = {(r['n'], r['link'], r['seed']): r for r in s8 if r['method'] == 'rag'}
    ch = rows(R / 'step9' / 'step9_chain.jsonl')
    chk = rows(R / 'step9' / 'step9_plain_check.jsonl')
    # cross-machine check: Vast plain MAQAC-S vs Jetson plain MAQAC-S
    k_ = sorted(set(chk) & set(plain))
    dj = np.array([chk[k]['J_session'] - plain[k]['J_session'] for k in k_])
    xm = dict(pairs=len(k_), identical=int((np.abs(dj) < 1e-6).sum()), max_abs_diff=float(np.abs(dj).max()),
              mean_abs_rel_pct=float(100 * np.mean([abs(chk[k]['J_session'] / plain[k]['J_session'] - 1) for k in k_])))
    C, pa, pc = {}, [], []
    for n in (6, 8, 10):
        for link in ('full', 'limited'):
            ks = sorted(k for k in ch if k[0] == n and k[1] == link and k in plain and k in rag)
            J = lambda X: np.array([X[k]['J_session'] for k in ks])
            D = lambda X: np.array([X[k]['disconnected_uav_s'] for k in ks])
            c = dict(pairs=len(ks), J_chain=float(J(ch).mean()), J_plain=float(J(plain).mean()), J_rag=float(J(rag).mean()),
                     rel_vs_plain_pct=100 * float(np.mean(J(ch) / J(plain) - 1)), rel_vs_rag_pct=100 * float(np.mean(J(ch) / J(rag) - 1)),
                     disc_chain=float(D(ch).mean()), disc_plain=float(D(plain).mean()), disc_rag=float(D(rag).mean()),
                     p_disc=wx(D(plain) - D(ch)), p_vs_rag=wx(J(ch) - J(rag)),
                     pos_vs_rag=int((J(ch) > J(rag)).sum()), neg_vs_rag=int((J(ch) < J(rag)).sum()),
                     msgs_chain=float(np.mean([ch[k]['coord_msgs'] for k in ks])),
                     decide_ms_chain=float(np.mean([ch[k]['decide_ms'] for k in ks])),
                     near_misses_chain=float(np.mean([ch[k]['near_misses'] for k in ks])),
                     collisions_chain=int(sum(ch[k]['collisions'] for k in ks)))
            C[f'{n}|{link}'] = c
            if link == 'limited':
                pa.append((f'{n}|{link}', c['p_disc']))
                if n in (8, 10):
                    pc.append((f'{n}|{link}', c['p_vs_rag']))
    for lst, key in ((pa, 'holm_disc'), (pc, 'holm_vs_rag')):
        for (k, _), h in zip(lst, holm([p for _, p in lst])):
            C[k][key] = h
    crit = dict(H6a_PASS=all(C[f'{n}|limited']['holm_disc'] < 0.01 for n in (6, 8, 10)),
                H6b_PASS=all(C[f'{n}|limited']['rel_vs_plain_pct'] >= -1 for n in (6, 8, 10)),
                H6c_PASS=all(C[f'{n}|limited']['holm_vs_rag'] < 0.01 for n in (8, 10)),
                H6d_PASS=all(abs(C[f'{n}|full']['rel_vs_plain_pct']) <= 1 for n in (6, 8, 10)))
    S = dict(cross_machine=xm, cells=C, criteria=crit)
    (R / 'step9_summary.json').write_text(json.dumps(S, indent=1))
    print('cross-machine', xm)
    for k, c in C.items():
        print(k, 'J chain %.2f plain %.2f rag %.2f | vs plain %+.2f %% vs rag %+.2f %% (+%d/-%d, p %.2g) | disc %.0f / %.0f / %.0f (p %.2g)' % (
            c['J_chain'], c['J_plain'], c['J_rag'], c['rel_vs_plain_pct'], c['rel_vs_rag_pct'], c['pos_vs_rag'], c['neg_vs_rag'],
            c['p_vs_rag'], c['disc_chain'], c['disc_plain'], c['disc_rag'], c['p_disc']))
    print(crit)


if __name__ == '__main__':
    main()
