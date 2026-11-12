#!/usr/bin/env python3
"""Analysis helpers for the VLM damage study: macro-F1, ECE, Brier, split conformal, bootstrap CIs (pre-registered metrics)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta


def f1_score(y, p, average='macro'):
    ks = np.union1d(np.unique(y), np.unique(p)); f = []
    for k in ks:
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'vlm_damage'


def ece(P, y, bins=15):
    c = P.max(1); a = (P.argmax(1) == y).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(a[m].mean() - c[m].mean())
    return float(e)


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(1).mean())


def conformal(Pc, yc, Pt, yt, alpha=0.1):
    s = 1 - Pc[np.arange(len(yc)), yc]
    n = len(s); q = np.quantile(s, min(1, math.ceil((n + 1) * (1 - alpha)) / n), method='higher')
    S = (1 - Pt) <= q
    cov = S[np.arange(len(yt)), yt]
    k = int(cov.sum()); N = len(cov)
    lo = beta.ppf(0.025, k, N - k + 1) if k > 0 else 0.0
    return float(cov.mean()), float(lo), float(S.sum(1).mean()), float(q)


def temp_scale(Lc, yc):
    best, bt = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 400)):
        z = Lc / T; z = z - z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        nll = -np.log(P[np.arange(len(yc)), yc] + 1e-12).mean()
        if nll < best:
            best, bt = nll, T
    return bt


def softmax(L, T=1.0):
    z = L / T; z = z - z.max(1, keepdims=True); P = np.exp(z); return P / P.sum(1, keepdims=True)


def boot_f1(y, pred, B=4000, seed=0):
    r = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = r.integers(0, n, n); v.append(f1_score(y[i], pred[i], average='macro'))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def step2():
    lab = json.load(open(R / 'labels.json'))
    ids = sorted(lab); perm = np.random.default_rng(0).permutation(len(ids))
    cal = set(ids[i] for i in perm[:len(ids) // 2])
    out = {}
    for name in ('qwen3b', 'qwen7b', 'clip'):
        rows = {json.loads(l)['id']: json.loads(l) for l in open(R / f'preds_{name}.jsonl')}
        res = {}
        for q in ('damage', 'flood', 'road'):
            P = {s: np.array([rows[k][q] for k in ids if (k in cal) == (s == 'cal')]) for s in ('cal', 'test')}
            y = {s: np.array([lab[k][q] for k in ids if (k in cal) == (s == 'cal')]) for s in ('cal', 'test')}
            Pt, yt = P['test'], y['test']
            pred = Pt.argmax(1)
            cov, lo, size, qh = conformal(P['cal'], y['cal'], Pt, yt)
            res[q] = dict(acc=float((pred == yt).mean()), macro_f1=float(f1_score(yt, pred, average='macro')),
                          f1_ci=boot_f1(yt, pred), ece=ece(Pt, yt), brier=brier(Pt, yt),
                          conf_cov=cov, conf_cov_cp_lo=lo, conf_size=size, pred_hist=np.bincount(pred, minlength=Pt.shape[1]).tolist())
        res['pass'] = bool(res['damage']['macro_f1'] >= 0.60 and res['damage']['conf_cov'] >= 0.90 and res['damage']['conf_size'] <= 2)
        out[name] = res
        d = res['damage']
        print(f"{name:7s} damage F1 {d['macro_f1']:.3f} [{d['f1_ci'][0]:.3f},{d['f1_ci'][1]:.3f}] acc {d['acc']:.3f} ECE {d['ece']:.3f} "
              f"Brier {d['brier']:.3f} conformal cov {d['conf_cov']:.3f} (CP lo {d['conf_cov_cp_lo']:.3f}) size {d['conf_size']:.2f} "
              f"pred_hist {d['pred_hist']} | flood F1 {res['flood']['macro_f1']:.3f} road F1 {res['road']['macro_f1']:.3f} | PASS {res['pass']}")
    json.dump(out, open(R / 'step2_summary.json', 'w'), indent=1)


def ece_mass(P, y, bins=15):
    """ECE with equal-mass bins (secondary; TrustTAI protocol)."""
    c = P.max(1); a = (P.argmax(1) == y).astype(float); o = np.argsort(c); e = 0.0
    for b in np.array_split(o, bins):
        if len(b):
            e += len(b) / len(c) * abs(a[b].mean() - c[b].mean())
    return float(e)


def cp_upper(k, n, a=0.05):
    return float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)


def tune(seeds=(0, 1, 2, 3)):
    """Step 4 (pre-registered rule, plan file Step 4 + run amendments): SFT-LoRA seeds vs the untuned 3B on RescueNet test.
    Temperature and conformal threshold are fitted on the validation split only; the test split is used once."""
    T = R / 'tune'
    load = lambda st, s, sp: [json.loads(l) for l in open(T / f'eval_{st}_s{s}_{sp}.jsonl')]
    out = dict(seeds=list(seeds), questions={})
    s2 = json.load(open(R / 'step2_summary.json'))
    untuned_7b_val_half = s2['qwen7b']['damage']['macro_f1']
    for q in ('damage', 'flood', 'road'):
        res = dict(per_seed={})
        def arr(rows):
            r = [x for x in rows if x['q'] == q]
            return [x['id'] for x in r], np.array([x['logits'] for x in r]), np.array([x['y'] for x in r])
        _, Lbv, ybv = arr(load('base', 0, 'val')); ids_t, Lbt, ybt = arr(load('base', 0, 'test'))
        Tb = temp_scale(Lbv, ybv); Pbv, Pbt = softmax(Lbv, Tb), softmax(Lbt, Tb)
        base_pred = Pbt.argmax(1)
        cov, lo, size, _ = conformal(Pbv, ybv, Pbt, ybt)
        res['base_3b'] = dict(macro_f1=f1_score(ybt, base_pred), f1_ci=boot_f1(ybt, base_pred), acc=float((base_pred == ybt).mean()),
                              ece_raw=ece(softmax(Lbt), ybt), ece_ts=ece(Pbt, ybt), brier_ts=brier(Pbt, ybt), T=Tb,
                              conf_cov=cov, conf_cov_cp_lo=lo, conf_size=size)
        Pv_all, Pt_all = [], []
        for s in seeds:
            _, Lv, yv = arr(load('sft', s, 'val')); idt, Lt, yt = arr(load('sft', s, 'test'))
            assert idt == ids_t and (yt == ybt).all()
            Ts = temp_scale(Lv, yv); Pv, Pt = softmax(Lv, Ts), softmax(Lt, Ts)
            Pv_all.append(Pv); Pt_all.append(Pt)
            pred = Pt.argmax(1)
            cov, lo, size, qh = conformal(Pv, yv, Pt, yt)
            r = np.random.default_rng(s); dif = []
            for _ in range(4000):
                i = r.integers(0, len(yt), len(yt)); dif.append(f1_score(yt[i], pred[i]) - f1_score(yt[i], base_pred[i]))
            res['per_seed'][s] = dict(macro_f1=f1_score(yt, pred), f1_ci=boot_f1(yt, pred, seed=s), acc=float((pred == yt).mean()),
                                      gain_vs_base=f1_score(yt, pred) - res['base_3b']['macro_f1'],
                                      gain_ci=[float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))],
                                      ece_raw=ece(softmax(Lt), yt), ece_ts=ece(Pt, yt), brier_raw=brier(softmax(Lt), yt), brier_ts=brier(Pt, yt),
                                      T=Ts, conf_cov=cov, conf_cov_cp_lo=lo, conf_size=size, qhat=qh, ece_mass_ts=ece_mass(Pt, yt),
                                      conf05=conformal(Pv, yv, Pt, yt, alpha=0.05),
                                      pred_hist=np.bincount(pred, minlength=Pt.shape[1]).tolist())
        # seed ensemble: belief = mean of temperature-scaled seed posteriors; epistemic = mutual information
        Pv_e, Pt_e = np.mean(Pv_all, 0), np.mean(Pt_all, 0)
        H_tot = entropy(Pt_e); H_ale = np.mean([entropy(P) for P in Pt_all], 0); MI = H_tot - H_ale
        pe = Pt_e.argmax(1); err = (pe != ybt)
        cov, lo, size, _ = conformal(Pv_e, ybv, Pt_e, ybt)
        rc = []
        for score_name, sc in (('max_prob', -Pt_e.max(1)), ('mutual_info', MI), ('total_entropy', H_tot)):
            o = np.argsort(sc); curve = []
            for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
                k = max(1, int(round(c * len(o)))); keep = o[:k]
                curve.append(dict(coverage=c, risk=float(err[keep].mean()),
                                  risk_cp_hi=cp_upper(int(err[keep].sum()), k), n=k))
            aurc = float(np.mean([err[o[:k]].mean() for k in range(1, len(o) + 1)]))
            # AUROC of the uncertainty score for detecting errors (Mann-Whitney; ties count 1/2)
            a, b = sc[err], sc[~err]
            auroc = float(((a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()) / max(len(a) * len(b), 1))
            rc.append(dict(score=score_name, aurc=aurc, err_auroc=auroc, curve=curve))
        bins = np.linspace(0, 1, 11); conf = Pt_e.max(1); acc = (pe == ybt)
        rel = [dict(lo=float(a), hi=float(b), n=int(((conf > a) & (conf <= b)).sum()),
                    acc=float(acc[(conf > a) & (conf <= b)].mean()) if ((conf > a) & (conf <= b)).any() else None,
                    conf=float(conf[(conf > a) & (conf <= b)].mean()) if ((conf > a) & (conf <= b)).any() else None)
               for a, b in zip(bins[:-1], bins[1:])]
        res['ensemble'] = dict(macro_f1=f1_score(ybt, pe), f1_ci=boot_f1(ybt, pe), ece=ece(Pt_e, ybt), brier=brier(Pt_e, ybt),
                               conf_cov=cov, conf_cov_cp_lo=lo, conf_size=size,
                               mean_total_entropy=float(H_tot.mean()), mean_aleatoric=float(H_ale.mean()), mean_epistemic_MI=float(MI.mean()),
                               MI_wrong=float(MI[err].mean()), MI_right=float(MI[~err].mean()),
                               risk_coverage=rc, reliability=rel)
        if q == 'damage':   # safety-relevant miss: true major/total (3, 4) predicted as no building / no damage (0, 1)
            sev = ybt >= 3; miss = sev & (pe <= 1); k, n = int(miss.sum()), int(sev.sum())
            res['ensemble']['severe_missed'] = dict(k=k, n=n, rate=k / n, cp_hi=cp_upper(k, n))
            from collections import Counter
            res['ensemble']['confusion'] = [[int(((ybt == i) & (pe == j)).sum()) for j in range(5)] for i in range(5)]
        ps = res['per_seed']; f1s = np.array([ps[s]['macro_f1'] for s in seeds])
        res['mean_f1'], res['sd_f1'] = float(f1s.mean()), float(f1s.std(ddof=1))
        out['questions'][q] = res
    d = out['questions']['damage']; ps = d['per_seed']
    bar = max(d['base_3b']['macro_f1'], untuned_7b_val_half)
    n_gain = sum(ps[s]['macro_f1'] - bar >= 0.10 and ps[s]['gain_ci'][0] > 0 for s in seeds)
    crit = dict(best_untuned=bar, best_untuned_note='max(untuned 3B on test, untuned 7B on the step-2 test half)',
                gain_mean=d['mean_f1'] - bar, gain_seeds_ok=n_gain,
                f1_pass=bool(d['mean_f1'] - bar >= 0.10 and n_gain >= 3),
                ece_ts_max=max(ps[s]['ece_ts'] for s in seeds),
                ece_pass=bool(np.mean([ps[s]['ece_ts'] for s in seeds]) <= 0.05),
                conformal_pass=bool(np.mean([ps[s]['conf_cov'] for s in seeds]) >= 0.90 and
                                    np.mean([ps[s]['conf_cov_cp_lo'] for s in seeds]) >= 0.87 and
                                    np.mean([ps[s]['conf_size'] for s in seeds]) <= 1.5))
    crit['PASS'] = bool(crit['f1_pass'] and crit['ece_pass'] and crit['conformal_pass'])
    out['criteria'] = crit
    json.dump(out, open(R / 'tune_summary.json', 'w'), indent=1, default=float)
    for q, r in out['questions'].items():
        b = r['base_3b']
        print(f"{q:6s} base3B F1 {b['macro_f1']:.3f} ECE {b['ece_ts']:.3f} set {b['conf_size']:.2f} | SFT F1 mean {r['mean_f1']:.3f} sd {r['sd_f1']:.3f}")
        for s, v in r['per_seed'].items():
            print(f"   s{s} F1 {v['macro_f1']:.3f} [{v['f1_ci'][0]:.3f},{v['f1_ci'][1]:.3f}] gain {v['gain_vs_base']:+.3f} "
                  f"[{v['gain_ci'][0]:+.3f},{v['gain_ci'][1]:+.3f}] ECE raw {v['ece_raw']:.3f} ts {v['ece_ts']:.3f} T {v['T']:.2f} "
                  f"cov {v['conf_cov']:.3f} (CP lo {v['conf_cov_cp_lo']:.3f}) set {v['conf_size']:.2f}")
        e = r['ensemble']
        print(f"   ensemble F1 {e['macro_f1']:.3f} ECE {e['ece']:.3f} cov {e['conf_cov']:.3f} set {e['conf_size']:.2f} "
              f"H {e['mean_total_entropy']:.3f} = ale {e['mean_aleatoric']:.3f} + MI {e['mean_epistemic_MI']:.4f} "
              f"(MI wrong {e['MI_wrong']:.4f} right {e['MI_right']:.4f}) AURC " + ' '.join(f"{x['score']} {x['aurc']:.3f}/AUROC {x['err_auroc']:.2f}" for x in e['risk_coverage']))
    print('severe missed', out['questions']['damage']['ensemble']['severe_missed'])
    print('criteria', json.dumps(crit))


if __name__ == '__main__':
    tune() if sys.argv[1:] == ['tune'] else step2()
