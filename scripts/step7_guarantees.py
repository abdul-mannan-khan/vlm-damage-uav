#!/usr/bin/env python3
"""Step 7 (pre-registered 10 Nov 2026, plan file "Step 7 plan"): guarantees under sparse backhaul links (idea 1, H3)
and under VLM errors (idea 2, H4). One decision round, weighted facility location on a partition matroid:
    f(S) = sum_u w_u max_{e in S covering u} exp(-mu t_e)
n UAVs, K candidate cells each (drawn over the whole area, so UAVs compete), m buildings with RescueNet test images.

Planners
  maqac_s : sequential greedy in index order; the decision list is relayed along shortest backhaul paths
            (messages = latency = sum of hops i -> i+1)
  rag     : Alg. 1 of Xu, Garimella & Tzoumas (T-RO 2025), as in sim/airsim_vlm/netsim/episode.py rag_round
  indep   : every UAV picks its best cell alone (no coordination)
  opt     : exact optimum by enumeration (n <= 6); for larger n the upper bound
            UB = min over plans S of f(S) + sum_i max_k Delta(e_ik | S)

    python3 scripts/step7_guarantees.py            # writes results/vlm_damage/step7_summary.json
    python3 scripts/step7_guarantees.py --quick    # 20 instances per cell, no file written (smoke test)
"""
import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
AREA, M_B, K, R, V, MU = 600.0, 40, 5, 60.0, 14.0, 0.01
G = {1: 1.0, 2: 1.5, 3: 2.5, 4: 3.0}
GMAX = max(G.values()) - min(G.values())
REGIMES = {'complete': np.inf, 'medium': 350.0, 'sparse': 200.0}


# ----------------------------------------------------------------------------- instance
def instance(seed, n, pool):
    r = np.random.default_rng(seed)
    bxy = r.uniform(0, AREA, (M_B, 2))
    img = r.choice(pool, M_B)
    cells = r.uniform(0, AREA, (n, K, 2))
    for _ in range(100000):                       # starts: connected under the sparsest range (so under all)
        st = r.uniform(0, AREA, (n, 2))
        if connected(adj(st, REGIMES['sparse'])):
            break
    t = np.linalg.norm(cells - st[:, None], axis=-1) / V                      # n x K
    cov = np.linalg.norm(cells[:, :, None] - bxy[None, None], axis=-1) <= R   # n x K x m
    A = np.where(cov, np.exp(-MU * t)[..., None], 0.0)
    return dict(A=A, st=st, img=img, u_err=r.random(M_B), c_err=r.integers(1, 4, M_B))


def adj(st, rc):
    D = np.linalg.norm(st[:, None] - st[None], axis=-1)
    a = D <= rc
    np.fill_diagonal(a, False)
    return a


def connected(a):
    seen, q = {0}, deque([0])
    while q:
        u = q.popleft()
        for v in np.where(a[u])[0]:
            if v not in seen:
                seen.add(v); q.append(v)
    return len(seen) == len(a)


def hops(a, s, d):
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        if u == d:
            return dist[u]
        for v in np.where(a[u])[0]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return np.inf


# ----------------------------------------------------------------------------- objective and planners
def f(A, w, plan):
    if not plan:
        return 0.0
    return float(w @ np.max(np.stack([A[i, k] for i, k in plan.items()]), 0))


def best(A, w, i, cur):
    g = (np.maximum(A[i], cur[None]) - cur[None]) @ w
    k = int(np.argmax(g))
    return k, float(g[k])


def cur_of(A, plan, m):
    c = np.zeros(m)
    for i, k in plan.items():
        c = np.maximum(c, A[i, k])
    return c


def maqac_s(A, w, a):
    n, m = A.shape[0], A.shape[2]
    plan, cur = {}, np.zeros(m)
    for i in range(n):
        k, _ = best(A, w, i, cur)
        plan[i] = k
        cur = np.maximum(cur, A[i, k])
    h = int(sum(hops(a, i, i + 1) for i in range(n - 1)))
    return plan, dict(msgs=h, latency=h)


def rag(A, w, a):
    """Same rule and message count as netsim/episode.py rag_round (no loss: ARQ)."""
    n, m = A.shape[0], A.shape[2]
    nb = {i: set(np.where(a[:, i])[0]) for i in range(n)}       # in-neighbours (symmetric graph)
    out = {i: [v for v in range(n) if i in nb[v]] for i in range(n)}
    sel, todo, it, msgs = {}, list(range(n)), 0, 0
    while todo:
        it += 1
        cand = {}
        for i in todo:
            cur = cur_of(A, {j: sel[j] for j in nb[i] if j in sel}, m)
            k, g = best(A, w, i, cur)
            cand[i] = (g, k)
            msgs += sum(1 for v in out[i] if v not in sel)
        commit = [i for i in todo if all((cand[i][0], -i) > (cand[j][0], -j) for j in nb[i] if j in cand)]
        for i in commit:
            sel[i] = cand[i][1]
            msgs += len(out[i])
        todo = [i for i in todo if i not in sel]
    return sel, dict(msgs=msgs, latency=it)


def indep(A, w, a):
    m = A.shape[2]
    return {i: best(A, w, i, np.zeros(m))[0] for i in range(A.shape[0])}, dict(msgs=0, latency=0)


def opt(A, w, plan=False):
    n = A.shape[0]
    idx = np.array(list(itertools.product(range(A.shape[1]), repeat=n)))
    cur = np.zeros((len(idx), A.shape[2]))
    for i in range(n):
        cur = np.maximum(cur, A[i][idx[:, i]])
    v = cur @ w
    j = int(v.argmax())
    return (float(v[j]), dict(enumerate(idx[j].tolist()))) if plan else float(v[j])


def delta_on(A, err, plan):
    """delta(S) = sum_u |w_hat_u - w_u| phi_u(S): weight errors only where the plan S serves (tight Lemma 1)."""
    return float(err @ cur_of(A, plan, A.shape[2]))


def ub(A, w, plan):
    cur = cur_of(A, plan, A.shape[2])
    return f(A, w, plan) + sum(best(A, w, i, cur)[1] for i in range(A.shape[0]))


PLANNERS = dict(maqac_s=maqac_s, rag=rag, indep=indep)


# ----------------------------------------------------------------------------- perception
def perceive(P, y, qhat, level, u_err, c_err):
    """Return perceived weights and the Lemma-2 delta bound (realised misses) for one perception level."""
    w_hat, db = np.zeros(len(y)), 0.0
    for b in range(len(y)):
        if level == 'oracle':
            w_hat[b] = G[y[b]]; continue
        e = float(level.split('_')[1]) if level.startswith('inj_') else 0.0
        if u_err[b] < e:                          # injected error: one-hot on a uniformly random wrong class
            wrong = [c for c in (1, 2, 3, 4) if c != y[b]][c_err[b] - 1]
            w_hat[b] = G[wrong]; db += GMAX; continue
        p = P[b, 1:] / P[b, 1:].sum()
        if level == 'argmax':
            w_hat[b] = G[int(p.argmax()) + 1]; continue
        C = [c for c in (1, 2, 3, 4) if 1 - P[b, c] <= qhat] or [int(p.argmax()) + 1]
        lo, hi = min(G[c] for c in C), max(G[c] for c in C)
        w_hat[b] = 0.5 * (lo + hi)
        db += (hi - lo) / 2 if y[b] in C else GMAX
    return w_hat, db


# ----------------------------------------------------------------------------- experiments
def idea1(seeds, pool, y_all):
    rows = []
    for n in (4, 6, 8, 10):
        for s in seeds:
            I = instance([s, n], n, pool)
            w = np.array([G[int(y_all[i])] for i in I['img']])
            o = opt(I['A'], w) if n <= 6 else None
            for reg, rc in REGIMES.items():
                a = adj(I['st'], rc)
                res = {p: fn(I['A'], w, a) for p, fn in PLANNERS.items()}
                ref = o if o is not None else min(ub(I['A'], w, pl) for pl, _ in res.values())
                rows.append(dict(n=n, seed=s, regime=reg, exact=o is not None, ref=ref,
                                 degree=float(a.sum(1).mean()),
                                 **{p: dict(ratio=f(I['A'], w, pl) / ref, **c) for p, (pl, c) in res.items()}))
        print('idea 1: n =', n, 'done', flush=True)
    return rows


def adversarial(kmax=8, eps=0.01):
    """k leaf UAVs pairwise out of range, all linked to a hub UAV whose cells cover nothing. Building 0 (weight
    1 + eps) is reachable by every leaf; building j (weight 1) only by leaf j. Times set to 0 (no decay)."""
    out = []
    for k in range(2, kmax + 1):
        n, m = k + 1, k + 1
        A = np.zeros((n, 2, m))
        for j in range(k):
            A[j, 0, 0] = 1.0          # shared site
            A[j, 1, j + 1] = 1.0      # private site
        w = np.r_[1 + eps, np.ones(k)]
        a = np.zeros((n, n), bool); a[:k, k] = a[k, :k] = True
        o = 1 + eps + (k - 1)
        out.append(dict(k=k, rag=f(A, w, rag(A, w, a)[0]) / o, maqac_s=f(A, w, maqac_s(A, w, a)[0]) / o,
                        predicted_rag=(1 + eps) / o))
    return out


LEVELS = ['oracle', 'argmax', 'conformal', 'inj_0.05', 'inj_0.10', 'inj_0.20']


def idea2(seeds, pool, P, y_all, qhat, n=6):
    rows = []
    for s in seeds:
        I = instance([s, n], n, pool)
        y = np.array([int(y_all[i]) for i in I['img']]); Pb = P[I['img']]
        w = np.array([G[c] for c in y])
        o, oplan = opt(I['A'], w, plan=True)
        for reg in ('complete', 'sparse'):
            a = adj(I['st'], REGIMES[reg])
            row = dict(seed=s, regime=reg, opt=o)
            for lev in LEVELS:
                w_hat, db = perceive(Pb, y, qhat, lev, I['u_err'], I['c_err'])
                delta = float(np.abs(w_hat - w).sum())
                err = np.abs(w_hat - w)
                plans = {p: PLANNERS[p](I['A'], w_hat, a)[0] for p in ('maqac_s', 'rag')}
                row[lev] = dict(delta=delta, delta_bound=db, delta_O=delta_on(I['A'], err, oplan),
                                **{p: f(I['A'], w, pl) for p, pl in plans.items()},
                                **{p + '_deltaS': delta_on(I['A'], err, pl) for p, pl in plans.items()})
            rows.append(row)
    print('idea 2 done', flush=True)
    return rows


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = run
    return out.tolist()


def summarise(r1, adv, r2):
    S = dict(idea1=dict(cells={}), adversarial=adv, idea2=dict(cells={}))
    keys, ps = [], []
    for n in (4, 6, 8, 10):
        for reg in REGIMES:
            R_ = [x for x in r1 if x['n'] == n and x['regime'] == reg]
            q = lambda p, qq: float(np.quantile([x[p]['ratio'] for x in R_], qq))
            d = np.array([x['maqac_s']['ratio'] - x['rag']['ratio'] for x in R_])
            p = float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0
            S['idea1']['cells'][f'{n}|{reg}'] = dict(
                instances=len(R_), exact=R_[0]['exact'], mean_degree=float(np.mean([x['degree'] for x in R_])),
                **{pl: dict(mean=float(np.mean([x[pl]['ratio'] for x in R_])), min=q(pl, 0), p01=q(pl, .01),
                            p05=q(pl, .05), msgs=float(np.mean([x[pl]['msgs'] for x in R_])),
                            latency=float(np.mean([x[pl]['latency'] for x in R_]))) for pl in PLANNERS},
                diff_mean_pts=100 * float(d.mean()), diff_positive=int((d > 0).sum()), diff_negative=int((d < 0).sum()),
                wilcoxon_p=p)
            keys.append(f'{n}|{reg}'); ps.append(p)
    for k_, ph in zip(keys, holm(ps)):
        S['idea1']['cells'][k_]['holm_p'] = ph
    c6 = S['idea1']['cells']
    S['idea1']['H3a_floor_holds'] = bool(all(x['maqac_s']['ratio'] >= 0.5 - 1e-9 for x in r1 if x['exact']))
    S['idea1']['H3a_min_ratio_exact'] = float(min(x['maqac_s']['ratio'] for x in r1 if x['exact']))
    S['idea1']['H3b_PASS'] = bool(c6['6|sparse']['wilcoxon_p'] < 0.01 and c6['6|sparse']['rag']['p01'] < c6['6|sparse']['maqac_s']['p01'])
    S['idea1']['H3c_PASS'] = bool(abs(c6['6|complete']['diff_mean_pts']) <= 1.0)
    S['adversarial_H3d_PASS'] = bool(all(abs(x['rag'] - x['predicted_rag']) < 1e-9 and x['maqac_s'] > 1 - 1e-9 for x in adv))
    for reg in ('complete', 'sparse'):
        R_ = [x for x in r2 if x['regime'] == reg]
        cell = {}
        for pl in ('maqac_s', 'rag'):
            orc = np.array([x['oracle'][pl] / x['opt'] for x in R_])
            for lev in LEVELS:
                rr = np.array([x[lev][pl] / x['opt'] for x in R_]); dl = np.array([x[lev]['delta'] / x['opt'] for x in R_])
                floor = all(x[lev][pl] >= x['opt'] / 2 - 1.5 * x[lev]['delta'] - 1e-9 for x in R_)
                loss, allow = float((orc - rr).mean()), float(1.5 * dl.mean())
                tight = [x[lev][pl] >= x['opt'] / 2 - 0.5 * x[lev]['delta_O'] - x[lev][pl + '_deltaS'] - 1e-9 for x in R_]
                allow_t = float(np.mean([(0.5 * x[lev]['delta_O'] + x[lev][pl + '_deltaS']) / x['opt'] for x in R_]))
                floor_t = np.array([(x['opt'] / 2 - 0.5 * x[lev]['delta_O'] - x[lev][pl + '_deltaS']) / x['opt'] for x in R_])
                cell[f'{pl}|{lev}'] = dict(mean=float(rr.mean()), min=float(rr.min()), p05=float(np.quantile(rr, .05)),
                                           delta_over_opt=float(dl.mean()), floor_holds_all=bool(floor),
                                           loss_vs_oracle=loss, allowance=allow, loss_over_allowance=loss / allow if allow > 0 else None,
                                           H4b_PASS=bool(loss <= allow),
                                           tight_floor_holds_all=bool(all(tight)), tight_allowance=allow_t,
                                           tight_floor_mean=float(floor_t.mean()), tight_floor_nonvacuous=float((floor_t > 0).mean()),
                                           loss_over_tight_allowance=loss / allow_t if allow_t > 0 else None)
            d = np.array([x['conformal'][pl] - x['argmax'][pl] for x in R_]) / np.array([x['opt'] for x in R_])
            cell[f'{pl}|H4c'] = dict(p05_conformal=cell[f'{pl}|conformal']['p05'], p05_argmax=cell[f'{pl}|argmax']['p05'],
                                     diff_mean_pts=100 * float(d.mean()),
                                     wilcoxon_p=float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0)
            cell[f'{pl}|H4c']['PASS'] = bool(cell[f'{pl}|H4c']['p05_conformal'] > cell[f'{pl}|H4c']['p05_argmax'] and cell[f'{pl}|H4c']['wilcoxon_p'] < 0.05)
        cell['delta_bound_ge_realised_all'] = bool(all(x[lev]['delta_bound'] >= x[lev]['delta'] - 1e-9 for x in R_ for lev in LEVELS if lev != 'oracle' and lev != 'argmax'))
        S['idea2']['cells'][reg] = cell
    S['idea2']['H4a_PASS'] = bool(all(S['idea2']['cells'][r][f'maqac_s|{l}']['floor_holds_all'] for r in ('complete', 'sparse') for l in LEVELS))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    B = np.load(ROOT / 'results' / 'vlm_damage' / 'step3_obs_bundle.npz')
    P, y_all, qhat = B['P'], B['y'], float(B['qhat'])
    pool = np.where(y_all >= 1)[0]
    N = 20 if a.quick else 300
    r1 = idea1(range(7000, 7000 + N), pool, y_all)
    adv = adversarial()
    r2 = idea2(range(7500, 7500 + N), pool, P, y_all, qhat)
    S = summarise(r1, adv, r2)
    S['instances_per_cell'] = N
    print(json.dumps({k: S['idea1'][k] for k in S['idea1'] if k != 'cells'}, indent=1))
    for k_, c in S['idea1']['cells'].items():
        print(k_, 'deg %.1f' % c['mean_degree'], {p: (round(c[p]['mean'], 3), round(c[p]['p01'], 3), round(c[p]['min'], 3), c[p]['msgs'], c[p]['latency']) for p in PLANNERS},
              'diff %.2f pts' % c['diff_mean_pts'], '+%d/-%d' % (c['diff_positive'], c['diff_negative']), 'p %.2g' % c['wilcoxon_p'])
    print('H3d', S['adversarial_H3d_PASS'], [(x['k'], round(x['rag'], 3)) for x in adv])
    print(json.dumps(S['idea2'], indent=1))
    if not a.quick:
        (ROOT / 'results' / 'vlm_damage' / 'step7_summary.json').write_text(json.dumps(S, indent=1))
        (ROOT / 'results' / 'vlm_damage' / 'step7_rows.json').write_text(json.dumps(dict(idea1=r1, idea2=r2), default=float))


if __name__ == '__main__':
    main()
