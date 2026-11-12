#!/usr/bin/env python3
"""Export the V3 calibration (per-seed vector scaling on validation half A, ensemble temperature, alpha = 0.1 conformal
threshold on half B) as plain numbers, so the live VLM on the flight box applies exactly the saved belief model.
Check: re-applying the numbers to the saved test logits reproduces step3_obs_bundle.npz."""
import math
import numpy as np
from scipy.optimize import minimize
from vlm_damage_posthoc import load, split, S, R
from vlm_damage_analyse import temp_scale, softmax


def vec_params(L, y, lam=1e-3):                      # identical to vlm_damage_posthoc.vec_fit, returning the parameters
    K = L.shape[1]; Z = L - L.max(1, keepdims=True)
    def f(t):
        w, b = t[:K], t[K:]; z = Z * w + b; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + lam * ((w - 1 / temp_scale(L, y)) ** 2).sum() + lam * (b ** 2).sum()
    t0 = np.r_[np.full(K, 1 / temp_scale(L, y)), np.zeros(K)]
    t = minimize(f, t0, method='L-BFGS-B').x
    return t[:K], t[K:]


def apply(L, w, b):
    return softmax((L - L.max(1, keepdims=True)) * w + b)


Wv, Bv, PvA, PvB, PtE = [], [], [], [], []
for s in S:
    Lv, yv = load(s, 'val', 'damage'); Lt, yt = load(s, 'test', 'damage'); A = split(yv)
    w, b = vec_params(Lv[A], yv[A]); Wv.append(w); Bv.append(b)
    PvA.append(apply(Lv[A], w, b)); PvB.append(apply(Lv[~A], w, b)); PtE.append(apply(Lt, w, b))
te = temp_scale(np.log(np.mean(PvA, 0)), yv[A])
PB, PT = softmax(np.log(np.mean(PvB, 0)), te), softmax(np.log(np.mean(PtE, 0)), te)
yB = yv[~A]; sc = 1 - PB[np.arange(len(yB)), yB]; n = len(sc)
qhat = float(np.quantile(sc, min(1, math.ceil((n + 1) * 0.9) / n), method='higher'))
ref = np.load(R / 'step3_obs_bundle.npz')
print('seeds', list(S), 'te', round(float(te), 4), 'qhat', round(qhat, 4), 'max |P - bundle|', float(np.abs(PT - ref['P']).max()),
      'qhat match', abs(qhat - float(ref['qhat'])) < 1e-12)
np.savez(R / 'v3_calibration.npz', W=np.array(Wv), B=np.array(Bv), te=te, qhat=qhat, seeds=np.array(list(S)),
         letters=np.array(list('ABCDE')))
