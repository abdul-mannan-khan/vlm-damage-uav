#!/usr/bin/env python3
"""Save the onboard (int8, seed 0) belief model as an .npz in the V3 format (W, B, te, qhat), so the HIL runner applies it
with the same Belief class as the step-6 live VLM (scripts/vlm_damage_live.py).

Step 8 / P3 (scripts/step8_analyse.py) fitted it but kept only the scores: V2 = vector scaling on validation half A
(vlm_damage_posthoc.vec_fit, split seed 0), split conformal at alpha = 0.1 on half B (vlm_damage_analyse.conformal).
This script repeats that fit on the Jetson logits of results/vlm_damage/jetson/20261110_173516/val.jsonl, saves the
parameters, and checks that test F1 / ECE / coverage / set size reproduce Table "Onboard, int8, one seed"
(step8_summary.json P3 calibration onboard_int8|V2: F1 0.663, ECE 0.060, coverage 0.902, 1.59 labels).

    python3 export_int8_calibration.py [--run results/vlm_damage/jetson/20261110_173516] [--out .../int8_v2_calibration.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from vlm_damage_analyse import ece, f1_score, temp_scale, softmax, conformal   # noqa: E402
from vlm_damage_posthoc import split                                           # noqa: E402


def vec_fit_params(L, y, lam=1e-3):
    """vlm_damage_posthoc.vec_fit, returning the parameters instead of a closure (same objective, start and solver)."""
    K = L.shape[1]; Z = L - L.max(1, keepdims=True); T = temp_scale(L, y)

    def f(t):
        w, b = t[:K], t[K:]; z = Z * w + b; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + lam * ((w - 1 / T) ** 2).sum() + lam * (b ** 2).sum()
    t = minimize(f, np.r_[np.full(K, 1 / T), np.zeros(K)], method='L-BFGS-B').x
    return t[:K], t[K:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=str(ROOT / 'results/vlm_damage/jetson/20261110_173516'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / 'int8_v2_calibration.npz'
    L, Y = {}, {}
    for sp in ('val', 'test'):                      # same image order as step8_analyse (training-cache order)
        ids = [json.loads(l)['id'] for l in open(ROOT / 'results/vlm_damage/tune' / f'eval_sft_s0_{sp}.jsonl')
               if json.loads(l)['q'] == 'damage']
        ob = {x['id']: x for x in map(json.loads, open(run / f'{sp}.jsonl'))}
        L[sp] = np.array([ob[k]['logits'] for k in ids], float)
        Y[sp] = np.array([ob[k]['y'] for k in ids], int)
    A = split(Y['val']); B = ~A
    w, b = vec_fit_params(L['val'][A], Y['val'][A])

    def post(X):
        return softmax((X - X.max(1, keepdims=True)) * w + b)
    cov, lo, size, qhat = conformal(post(L['val'][B]), Y['val'][B], post(L['test']), Y['test'])
    Pt = post(L['test'])
    chk = dict(f1=f1_score(Y['test'], Pt.argmax(1)), ece=ece(Pt, Y['test']), cov=cov, cp_lo=lo, size=size, qhat=qhat)
    ref = json.load(open(ROOT / 'results/vlm_damage/step8_summary.json'))['P3']['calibration']['onboard_int8|V2']
    ok = all(abs(chk[k] - ref[k]) < 1e-6 for k in ('f1', 'ece', 'cov', 'size'))
    np.savez(out, W=w[None, :], B=b[None, :], te=1.0, qhat=qhat, seeds=np.array([0]), letters=np.array(list('ABCDE')),
             source=str(run), rule='V2 vector scaling on val half A (split seed 0), conformal alpha 0.1 on half B')
    print(json.dumps(dict(out=str(out), check=chk, reference=ref, reproduces_step8=ok), indent=1, default=float))
    if not ok:
        raise SystemExit('does not reproduce step 8 P3: do not use')


if __name__ == '__main__':
    main()
