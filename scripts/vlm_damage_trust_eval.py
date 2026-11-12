#!/usr/bin/env python3
"""Step 10: trust and belief evaluation of the fine-tuned damage VLM (pre-registered in
results/vlm_damage/step10_trust_prereg.txt). Saved logits only; no retraining, no new inference.
Writes results/vlm_damage/step10_trust.json and paper/vlm_tvt/figures/fig_trust.pdf."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_damage_analyse import ece, ece_mass, f1_score, temp_scale, softmax, conformal, R
from vlm_damage_posthoc import load, split, vec_fit, mondrian

ROOT = Path(__file__).resolve().parents[1]
J = R / 'jetson' / '20261110_173516'
K = 5; S = (0, 1, 2, 3); ALPHA = 0.1; B_BOOT = 10000
CLS = ['no bldg', 'intact', 'minor', 'major', 'destroyed']


def H(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1)


def brier(P, y):
    Y = np.eye(K)[y]; return float(((P - Y) ** 2).sum(1).mean())


def murphy(P, y, bins=10):
    """Murphy decomposition of the multiclass Brier score, binned per class on predicted probability."""
    Y = np.eye(K)[y]; rel = res = 0.0
    for k in range(K):
        p, o = P[:, k], Y[:, k]; ob = o.mean()
        idx = np.minimum((p * bins).astype(int), bins - 1)
        for b in range(bins):
            m = idx == b
            if m.any():
                rel += m.mean() * (p[m].mean() - o[m].mean()) ** 2
                res += m.mean() * (o[m].mean() - ob) ** 2
    unc = float((Y.mean(0) * (1 - Y.mean(0))).sum())
    return dict(reliability=float(rel), resolution=float(res), uncertainty=unc)


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def risk_cov(conf, err):
    o = np.argsort(-conf); e = err[o].astype(float)
    cov = np.arange(1, len(e) + 1) / len(e); risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return cov, risk, float(risk.mean())


def acc_at(conf, err, c=0.9):
    o = np.argsort(-conf); n = int(np.ceil(c * len(o))); return float(1 - err[o[:n]].mean())


def auroc(score, pos):
    """P(score of an error > score of a correct prediction), ties count 1/2."""
    a, b = score[pos], score[~pos]
    if not len(a) or not len(b):
        return float('nan')
    r = np.argsort(np.argsort(np.r_[a, b], kind='mergesort'), kind='mergesort').astype(float)
    # average ranks for ties
    v = np.r_[a, b]; u, inv = np.unique(v, return_inverse=True)
    sums = np.bincount(inv, r); cnt = np.bincount(inv); r = (sums / cnt)[inv]
    return float((r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b)))


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def false_alarm(P, y):
    neg = y <= 2                                   # no building, intact, minor
    fa = (P.argmax(1) >= 3) & neg; k, n = int(fa.sum()), int(neg.sum())
    return dict(k=k, n=n, rate=k / n, cp95=cp(k, n))


def jload(name):
    r = [json.loads(l) for l in open(J / f'{name}.jsonl')]
    return {x['id']: x for x in r}


def main():
    # ---- calibrated seeds and ensemble, exactly as V3 in vlm_damage_posthoc.py
    Pv_A, Pv_B, Pt, F = [], [], [], []
    for s in S:
        Lv, yv = load(s, 'val', 'damage'); Lt, yt = load(s, 'test', 'damage'); A = split(yv); Bm = ~A
        g = vec_fit(Lv[A], yv[A]); Pv_A.append(g(Lv[A])); Pv_B.append(g(Lv[Bm])); Pt.append(g(Lt))
    yA, yB = yv[A], yv[Bm]
    te = temp_scale(np.log(np.mean(Pv_A, 0)), yA)
    PB = softmax(np.log(np.mean(Pv_B, 0)), te); PT = softmax(np.log(np.mean(Pt, 0)), te)
    Pt = np.array(Pt)                                                     # seeds x n x K
    out = dict(n_test=int(len(yt)), class_counts=np.bincount(yt, minlength=K).tolist())

    # ---- M1 calibration and proper scores
    def scores(P):
        return dict(f1=f1_score(yt, P.argmax(1)), acc=float((P.argmax(1) == yt).mean()), brier=brier(P, yt),
                    nll=nll(P, yt), ece=ece(P, yt), ece_mass=ece_mass(P, yt), **murphy(P, yt))
    out['M1'] = dict(seeds=[scores(P) for P in Pt], ensemble=scores(PT))

    # ---- M2/M3 risk-coverage and uncertainty decomposition
    err = PT.argmax(1) != yt
    tot = H(Pt.mean(0)); ale = H(Pt).mean(0); mi = np.clip(tot - ale, 0, None)   # decomposition on the raw seed average (no ensemble temperature)
    rc = {}
    for s in S:
        e_s = Pt[s].argmax(1) != yt; c, r, a = risk_cov(Pt[s].max(1), e_s)
        rc[f'seed{s}'] = dict(aurc=a, acc90=acc_at(Pt[s].max(1), e_s))
    for nm, conf in (('ens_maxp', PT.max(1)), ('ens_entropy', -tot), ('ens_mi', -mi)):
        c, r, a = risk_cov(conf, err); rc[nm] = dict(aurc=a, acc90=acc_at(conf, err))
    rc['oracle_aurc'] = float(np.mean(np.cumsum(np.sort(err)[::1].astype(float)) / np.arange(1, len(err) + 1)))
    out['M2'] = rc
    out['M3'] = dict(auroc_maxp=auroc(-PT.max(1), err), auroc_total=auroc(tot, err), auroc_aleatoric=auroc(ale, err),
                     auroc_mi=auroc(mi, err), mean_total=float(tot.mean()), mean_aleatoric=float(ale.mean()),
                     mean_mi=float(mi.mean()), mi_share=float(mi.mean() / tot.mean()))

    # ---- M4 conformal: marginal and Mondrian, per class
    def per_class(Sset):
        cov = Sset[np.arange(len(yt)), yt]; rows = []
        for k in range(K):
            m = yt == k; rows.append(dict(cls=CLS[k], n=int(m.sum()), cov=float(cov[m].mean()),
                                          cp_lo=cp(int(cov[m].sum()), int(m.sum()))[0], size=float(Sset[m].sum(1).mean())))
        return dict(cov=float(cov.mean()), cp_lo=cp(int(cov.sum()), len(cov))[0], size=float(Sset.sum(1).mean()), per_class=rows)
    q = conformal(PB, yB, PT, yt, ALPHA)[3]; S_marg = (1 - PT) <= q
    S_mon = np.zeros_like(PT, bool)
    for k in range(K):
        sc = 1 - PB[yB == k, k]; n = len(sc)
        qk = np.quantile(sc, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n), method='higher')
        S_mon[:, k] = (1 - PT[:, k]) <= qk
    out['M4'] = dict(marginal=per_class(S_marg), mondrian=per_class(S_mon))

    # ---- M5 false alarms (ensemble, single seeds, int8 Jetson seed, Jetson city frames)
    jt, jv = jload('test'), jload('val')
    ids_t = [json.loads(l)['id'] for l in open(R / 'tune' / 'eval_sft_s0_test.jsonl') if json.loads(l)['q'] == 'damage']
    ids_v = [json.loads(l)['id'] for l in open(R / 'tune' / 'eval_sft_s0_val.jsonl') if json.loads(l)['q'] == 'damage']
    LJt = np.array([jt[i]['logits'] for i in ids_t], float); LJv = np.array([jv[i]['logits'] for i in ids_v if i in jv], float)
    yJv = np.array([jv[i]['y'] for i in ids_v if i in jv]); mv = np.array([i in jv for i in ids_v])
    Av, Bv = split(yJv), ~split(yJv)
    gJ = vec_fit(LJv[Av], yJv[Av]); PJ = gJ(LJt)
    Lc = np.array([r['logits'] for r in jload('city').values()], float); PC = gJ(Lc)
    out['M5'] = dict(ensemble=false_alarm(PT, yt), seeds=[false_alarm(P, yt) for P in Pt], jetson_int8=false_alarm(PJ, yt),
                     jetson_city=dict(n=len(PC), k=int((PC.argmax(1) >= 3).sum()), cp95=cp(int((PC.argmax(1) >= 3).sum()), len(PC))))

    # ---- M6 shift: desktop seed-0 sets on int8 logits, then re-calibrated on Jetson
    Lv0, yv0 = load(0, 'val', 'damage'); A0 = split(yv0); g0 = vec_fit(Lv0[A0], yv0[A0])
    q0 = conformal(g0(Lv0[~A0]), yv0[~A0], g0(Lv0[~A0]), yv0[~A0], ALPHA)[3]
    cov_before = ((1 - g0(LJt)) <= q0)[np.arange(len(yt)), yt]
    cJ = conformal(gJ(LJv[Bv]), yJv[Bv], PJ, yt, ALPHA)
    out['M6'] = dict(desktop_sets_on_int8=dict(cov=float(cov_before.mean()), cp_lo=cp(int(cov_before.sum()), len(yt))[0],
                                               ece_desktop_scaling=ece(g0(LJt), yt)),
                     recalibrated_on_jetson=dict(cov=cJ[0], cp_lo=cJ[1], size=cJ[2], ece=ece(PJ, yt), brier=brier(PJ, yt),
                                                 f1=f1_score(yt, PJ.argmax(1))))

    # ---- M7 paired bootstrap: ensemble minus mean single seed
    rng = np.random.default_rng(0); n = len(yt); d = {'aurc': [], 'brier': [], 'f1': []}; mi_auc = []
    for _ in range(B_BOOT):
        i = rng.integers(0, n, n); e = err[i]
        d['aurc'].append(risk_cov(PT[i].max(1), e)[2] - np.mean([risk_cov(Pt[s][i].max(1), Pt[s][i].argmax(1) != yt[i])[2] for s in S]))
        d['brier'].append(brier(PT[i], yt[i]) - np.mean([brier(Pt[s][i], yt[i]) for s in S]))
        d['f1'].append(f1_score(yt[i], PT[i].argmax(1)) - np.mean([f1_score(yt[i], Pt[s][i].argmax(1)) for s in S]))
        mi_auc.append(auroc(mi[i], e))
    out['M7'] = {k: dict(mean=float(np.mean(v)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in d.items()}
    out['M7']['auroc_mi_ci'] = [float(np.nanpercentile(mi_auc, 2.5)), float(np.nanpercentile(mi_auc, 97.5))]

    # ---- hypotheses
    out['H10a'] = out['M7']['aurc']['ci'][1] < 0
    out['H10b'] = out['M7']['auroc_mi_ci'][0] > 0.5
    out['H10c'] = all(r['cov'] >= 0.85 for r in out['M4']['mondrian']['per_class'])
    out['H10d'] = out['M5']['ensemble']['cp95'][1] <= 0.05
    json.dump(out, open(R / 'step10_trust.json', 'w'), indent=1)
    np.savez(R / 'step10_trust_arrays.npz', PT=PT, Pt=Pt, yt=yt, tot=tot, ale=ale, mi=mi, S_marg=S_marg, S_mon=S_mon, PJ=PJ)
    for k in ('H10a', 'H10b', 'H10c', 'H10d'):
        print(k, 'PASS' if out[k] else 'FAIL')
    print(json.dumps({k: out[k] for k in ('M1', 'M2', 'M3', 'M6', 'M7')}, indent=0, default=float)[:4000])
    print('false alarms', json.dumps(out['M5'], default=float))
    print('conformal', json.dumps(out['M4'], default=float))


if __name__ == '__main__':
    main()
