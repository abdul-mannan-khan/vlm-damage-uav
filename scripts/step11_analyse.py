#!/usr/bin/env python3
"""Step 11 analysis (prereg results/vlm_damage/step11_jetson_ensemble_prereg.txt): the four-seed ensemble ON the Jetson.
Re-calibration exactly as the desktop ensemble (vlm_damage_posthoc.py V3), using the Jetson's own validation logits.
Inputs: results/vlm_damage/jetson/ens_20261112/ (phase A: city, test[:40]) and ens_20261112B/ (phase B: val, test).
Writes results/vlm_damage/step11_summary.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_damage_analyse import ece, f1_score, temp_scale, softmax, conformal, R
from vlm_damage_posthoc import split, vec_fit

A_DIR, B_DIR = R / 'jetson' / 'ens_20261112', R / 'jetson' / 'ens_20261112B'


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    out = {}
    sa, sb = json.load(open(A_DIR / 'summary.json')), json.load(open(B_DIR / 'summary.json'))
    city = sa['phases']['city']['total_ms']
    out['H11a'] = dict(city_p90_ms=city['p90'], city_median_ms=city['median'], n=city['n'], pass_=city['p90'] <= 5000)
    ram = max(sa['tegrastats']['ram_peak_mb'] or 0, sb['tegrastats']['ram_peak_mb'] or 0)
    out['H11d'] = dict(ram_peak_mb=ram, pass_=ram < 7400)
    # H11b (phase A, 40 test images)
    E = {r['id']: np.array(r['logits']) for r in rows(A_DIR / 'test.jsonl')}
    J = {r['id']: np.array(r['logits']) for r in rows(R / 'jetson' / '20261110_173516' / 'test.jsonl')}
    dk = [{r['id']: np.array(r['logits']) for r in rows(R / 'tune' / f'eval_sft_s{s}_test.jsonl') if r['q'] == 'damage'} for s in range(4)]
    ag0 = sum(E[i][0].argmax() == J[i].argmax() for i in E)
    ags = [sum(E[i][s].argmax() == dk[s][i].argmax() for i in E) for s in range(4)]
    out['H11b'] = dict(seed0_vs_onboard=int(ag0), seeds_vs_desktop=[int(a) for a in ags], n=len(E),
                       pass_=ag0 >= 39 and all(a >= 0.9 * len(E) for a in ags[1:]))
    # H11c (phase B): re-calibrate on the Jetson's validation logits, test once
    val = rows(B_DIR / 'val.jsonl'); tst = rows(B_DIR / 'test.jsonl')
    ids_v = [r['id'] for r in rows(R / 'tune' / 'eval_sft_s0_val.jsonl') if r['q'] == 'damage']
    ids_t = [r['id'] for r in rows(R / 'tune' / 'eval_sft_s0_test.jsonl') if r['q'] == 'damage']
    V = {r['id']: r for r in val}; T = {r['id']: r for r in tst}
    ids_v = [i for i in ids_v if i in V]; ids_t = [i for i in ids_t if i in T]
    Lv = np.array([V[i]['logits'] for i in ids_v], float); yv = np.array([V[i]['y'] for i in ids_v])
    Lt = np.array([T[i]['logits'] for i in ids_t], float); yt = np.array([T[i]['y'] for i in ids_t])
    A = split(yv); B = ~A
    PvA, PvB, PtE, per = [], [], [], []
    for s in range(4):
        g = vec_fit(Lv[A, s], yv[A]); PvA.append(g(Lv[A, s])); PvB.append(g(Lv[B, s])); P = g(Lt[:, s]); PtE.append(P)
        c = conformal(g(Lv[B, s]), yv[B], P, yt)
        per.append(dict(f1=f1_score(yt, P.argmax(1)), ece=ece(P, yt), cov=c[0], cp_lo=c[1], size=c[2]))
    te = temp_scale(np.log(np.mean(PvA, 0)), yv[A])
    PB = softmax(np.log(np.mean(PvB, 0)), te); PT = softmax(np.log(np.mean(PtE, 0)), te)
    c = conformal(PB, yv[B], PT, yt)
    ens = dict(f1=f1_score(yt, PT.argmax(1)), acc=float((PT.argmax(1) == yt).mean()), ece=ece(PT, yt), cov=c[0], cp_lo=c[1], size=c[2])
    out['onboard_seeds'] = per; out['onboard_ensemble'] = ens; out['n_val'] = len(yv); out['n_test'] = len(yt)
    out['H11c'] = dict(pass_=ens['ece'] <= 0.05 and ens['cov'] >= 0.90)
    out['test_ms'] = sb['phases']['test']['total_ms']
    json.dump(out, open(R / 'step11_summary.json', 'w'), indent=1, default=float)
    for k in ('H11a', 'H11b', 'H11c', 'H11d'):
        print(k, 'PASS' if out[k]['pass_'] else 'FAIL', {kk: v for kk, v in out[k].items() if kk != 'pass_'})
    print('ensemble', {k: round(v, 3) for k, v in ens.items()})
    for s, p in enumerate(per):
        print('seed', s, {k: round(v, 3) for k, v in p.items()})


if __name__ == '__main__':
    main()
