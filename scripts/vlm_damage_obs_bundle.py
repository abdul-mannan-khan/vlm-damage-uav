#!/usr/bin/env python3
"""Step 3 input: the V3-calibrated seed-ensemble posteriors of the damage question on RescueNet test (450 images) and
the conformal threshold from validation half B (alpha 0.1). Saved model outputs only; no images, no city data."""
import math
import numpy as np
from vlm_damage_posthoc import load, split, vec_fit, S, R
from vlm_damage_analyse import temp_scale, softmax

PvA, PvB, PtE = [], [], []
for s in S:
    Lv, yv = load(s, 'val', 'damage'); Lt, yt = load(s, 'test', 'damage'); A = split(yv)
    g = vec_fit(Lv[A], yv[A]); PvA.append(g(Lv[A])); PvB.append(g(Lv[~A])); PtE.append(g(Lt))
te = temp_scale(np.log(np.mean(PvA, 0)), yv[A])
PB, PT = softmax(np.log(np.mean(PvB, 0)), te), softmax(np.log(np.mean(PtE, 0)), te)
yB = yv[~A]; sc = 1 - PB[np.arange(len(yB)), yB]; n = len(sc)
qhat = float(np.quantile(sc, min(1, math.ceil((n + 1) * 0.9) / n), method='higher'))
np.savez(R / 'step3_obs_bundle.npz', P=PT, y=yt, qhat=qhat)
Sset = (1 - PT) <= qhat
print('test images', len(yt), 'qhat', round(qhat, 4), 'coverage', Sset[np.arange(len(yt)), yt].mean().round(3), 'size', Sset.sum(1).mean().round(2))
