#!/usr/bin/env python3
"""Post-hoc calibration fix (declared in the plan file before running): V0-V4. Saved logits only; test used once."""
import json, math, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
sys.path.insert(0, 'scripts')
from vlm_damage_analyse import ece, f1_score, temp_scale, softmax, conformal, R

T = R / 'tune'; S = (0, 1, 2, 3)

def load(s, sp, q):
    r = [json.loads(l) for l in open(T / f'eval_sft_s{s}_{sp}.jsonl')]; r = [x for x in r if x['q'] == q]
    return np.array([x['logits'] for x in r], float), np.array([x['y'] for x in r])

def split(y, seed=0):
    rng = np.random.default_rng(seed); A = []
    for k in np.unique(y):
        i = rng.permutation(np.where(y == k)[0]); A += list(i[: len(i) // 2])
    m = np.zeros(len(y), bool); m[A] = True; return m

def vec_fit(L, y, lam=1e-3):
    K = L.shape[1]; Z = L - L.max(1, keepdims=True)
    def f(t):
        w, b = t[:K], t[K:]; z = Z * w + b; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + lam * ((w - 1 / temp_scale(L, y)) ** 2).sum() + lam * (b ** 2).sum()
    t0 = np.r_[np.full(K, 1 / temp_scale(L, y)), np.zeros(K)]
    t = minimize(f, t0, method='L-BFGS-B').x
    return lambda X: softmax((X - X.max(1, keepdims=True)) * t[:K] + t[K:])

def mondrian(Pc, yc, Pt, yt, alpha=0.1):
    K = Pc.shape[1]; S = np.zeros_like(Pt, bool)
    for k in range(K):
        s = 1 - Pc[yc == k, k]; n = len(s)
        q = np.quantile(s, min(1, math.ceil((n + 1) * (1 - alpha)) / n), method='higher') if n else 1
        S[:, k] = (1 - Pt[:, k]) <= q
    cov = S[np.arange(len(yt)), yt]; k = int(cov.sum()); N = len(cov)
    per = [float(cov[yt == c].mean()) for c in range(K)]
    return float(cov.mean()), float(beta.ppf(0.025, k, N - k + 1)), float(S.sum(1).mean()), per

def row(P, y, cv):
    return dict(f1=f1_score(y, P.argmax(1)), ece=ece(P, y), cov=cv[0], cp_lo=cv[1], size=cv[2])

def main():
    out = {}
    for q in ('damage', 'flood', 'road'):
        res = {f'V{i}': [] for i in range(4)}; PvB, PtE, PvA = [], [], []
        for s in S:
            Lv, yv = load(s, 'val', q); Lt, yt = load(s, 'test', q); A = split(yv); B = ~A
            t0 = temp_scale(Lv, yv); res['V0'].append(row(softmax(Lt, t0), yt, conformal(softmax(Lv, t0), yv, softmax(Lt, t0), yt)))
            t1 = temp_scale(Lv[A], yv[A]); res['V1'].append(row(softmax(Lt, t1), yt, conformal(softmax(Lv[B], t1), yv[B], softmax(Lt, t1), yt)))
            g = vec_fit(Lv[A], yv[A]); res['V2'].append(row(g(Lt), yt, conformal(g(Lv[B]), yv[B], g(Lt), yt)))
            PvA.append(g(Lv[A])); PvB.append(g(Lv[B])); PtE.append(g(Lt))
        yA, yB = yv[A], yv[B]
        te = temp_scale(np.log(np.mean(PvA, 0)), yA)
        PB, PT = softmax(np.log(np.mean(PvB, 0)), te), softmax(np.log(np.mean(PtE, 0)), te)
        res['V3'] = row(PT, yt, conformal(PB, yB, PT, yt))
        m = mondrian(PB, yB, PT, yt); res['V4'] = dict(row(PT, yt, m[:3]), per_class_cov=m[3])
        for k in ('V0', 'V1', 'V2'):
            res[k] = {kk: float(np.mean([r[kk] for r in res[k]])) for kk in res[k][0]} | dict(per_seed=res[k])
        out[q] = res
        for k in ('V0', 'V1', 'V2', 'V3', 'V4'):
            r = res[k]; ok = r['ece'] <= .05 and r['cov'] >= .9 and r['cp_lo'] >= .87 and r['size'] <= 1.5
            print(f"{q:6s} {k} F1 {r['f1']:.3f} ECE {r['ece']:.3f} cov {r['cov']:.3f} CPlo {r['cp_lo']:.3f} set {r['size']:.2f} {'PASS' if ok else 'fail'}"
                  + (f" per-class cov {np.round(r['per_class_cov'],2).tolist()}" if k == 'V4' else ''))
    json.dump(out, open(R / 'posthoc_calibration.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
