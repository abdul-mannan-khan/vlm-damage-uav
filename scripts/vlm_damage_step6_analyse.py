#!/usr/bin/env python3
"""Step 6 analysis (pre-registered 9 Nov 2026, plan file "Step 6 plan"): 2-UAV CityEnviron flights, calibrated
visibility model vs flown camera, MAQAC-S vs RAG.

Buildings = merged footprints (stacked pieces with the same rounded N, E are one building; pixels summed per frame).
Seen = >= 2 % of a 640 x 480 frame. Calibrated twin: per frame (every 5 s, each UAV of the mission's kinematic twin track)
P(seen | horizontal distance d) = v * p(d), p(d) from results/vlm_damage/visibility_model_step5.json (10 m bins),
v ~ Gamma(k, 1/k) per building (Monte Carlo, fixed seed). The 58 m disc prediction is reported next to it.

    python3 scripts/vlm_damage_step6_analyse.py [--runs results/vlm_damage/city_step6/runs] [--glob 's6_*']
Output: results/vlm_damage/step6_summary.json (only for the default runs folder).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
PX_MIN, TS, FOOT = int(0.02 * 640 * 480), (60, 120, 180, 300), 58.0
cp_hi = lambda k, n: float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
cp_lo = lambda k, n: float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(ROOT / 'results/vlm_damage/city_step6/runs'))
    ap.add_argument('--glob', default='s6_*')
    a = ap.parse_args()
    vm = json.load(open(ROOT / 'results/vlm_damage/visibility_model_step5.json'))
    pc, k = np.array(vm['pc']), vm['k']
    blds = json.load(open(ROOT / 'results/vlm_damage/city/buildings.json'))
    fp = {b['name']: (round(b['N']), round(b['E'])) for b in blds}
    v = np.random.default_rng(1).gamma(k, 1 / k, size=(400, 1))
    pd = lambda d: np.where(d < 10 * len(pc), pc[np.minimum((d // 10).astype(int), len(pc) - 1)], 0.0)

    def predict(track, T):
        lm = np.zeros((len(v), len(K)))
        for t in range(0, min(T, len(track) - 1) + 1, 5):
            for p in track[t]:
                lm += np.log(np.clip(1 - np.minimum(1, v * pd(np.hypot(*(K - np.array(p[:2])).T))[None]), 1e-12, 1))
        return float((1 - np.exp(lm)).mean())

    M = []
    for run in sorted(Path(a.runs).glob(a.glob)):
        if not (run / 'result.json').exists():
            continue
        r = json.load(open(run / 'result.json'))
        o, L = r['origin_ne'], r['L']
        K = np.array(sorted({fp[b['name']] for b in blds if abs(b['N'] - o[0]) <= L / 2 and abs(b['E'] - o[1]) <= L / 2}), float)
        kid = {tuple(int(x) for x in key): i for i, key in enumerate(K)}
        first = np.full(len(K), np.inf)
        fa_k = fa_n = 0
        vlm = {}
        if (run / 'frames/vlm.jsonl').exists():
            for l in open(run / 'frames/vlm.jsonl'):
                x = json.loads(l); vlm[x['file']] = x
        for l in open(run / 'frames/labels.jsonl'):
            x = json.loads(l); acc = defaultdict(int)
            for n, px in x.get('bld') or []:
                if n in fp and fp[n] in kid:
                    acc[kid[fp[n]]] += px
            seen = [i for i, px in acc.items() if px >= PX_MIN]
            for i in seen:
                first[i] = min(first[i], x['t'])
            if seen and x['file'] in vlm:
                fa_n += 1; fa_k += int(vlm[x['file']]['top'] in (3, 4))
        fx = run / 'tracks_kin_fixed.json'
        kin = json.load(open(fx))['kinematic'] if fx.exists() else json.load(open(run / 'tracks.json'))['kinematic']
        kin = np.array(kin)
        disc = np.full(len(K), np.inf)
        for t, pos in enumerate(kin):
            d = np.hypot(K[:, None, 0] - pos[None, :, 0], K[:, None, 1] - pos[None, :, 1]).min(1)
            disc[(d <= FOOT) & np.isinf(disc)] = t
        M.append(dict(name=run.name, method=r['method'], n=r['n_uav'], seed=r['seed'], n_build=len(K),
                      flown={T: float((first <= T).mean()) for T in TS},
                      calib={T: predict(kin, T) for T in TS},
                      disc={T: float((disc <= T).mean()) for T in TS},
                      coll=len(r.get('collisions_in_flight') or {}),
                      min_sep=(r['safety'].get('mission') or {}).get('min_sep_m'), fa_k=fa_k, fa_n=fa_n,
                      J=r['px4'].get('J_session'), J_kin=r['kinematic'].get('J_session')))
    out = dict(missions=len(M), rows=M, cells={})
    by = defaultdict(list)
    for m in M:
        by[(m['n'], m['method'])].append(m)
    h1 = True
    for (n, meth), ms in sorted(by.items()):
        c = {f: {T: float(np.mean([m[f][T] for m in ms])) for T in TS} for f in ('flown', 'calib', 'disc')}
        c['err_calib_pts'] = {T: 100 * (c['calib'][T] - c['flown'][T]) for T in TS}
        c['err_disc_pts'] = {T: 100 * (c['disc'][T] - c['flown'][T]) for T in TS}
        c['missions'] = len(ms)
        h1 &= all(abs(e) <= 10 for e in c['err_calib_pts'].values())
        out['cells'][f'{n}|{meth}'] = c
    pairs = []
    for n in sorted({m['n'] for m in M}):
        A = {m['seed']: m for m in by[(n, 'maqac_seq')]}; B = {m['seed']: m for m in by[(n, 'rag')]}
        pairs += [(s, round((A[s]['flown'][300] - B[s]['flown'][300]) * A[s]['n_build'], 1)) for s in sorted(set(A) & set(B))]
    d = [x for _, x in pairs]; pos = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    coll = sum(m['coll'] for m in M); fk = sum(m['fa_k'] for m in M); fn = sum(m['fa_n'] for m in M)
    out['criteria'] = dict(
        H1_twin_validity=dict(PASS=bool(h1) and len(M) > 0),
        H2_maqacs_vs_rag=dict(per_seed_buildings=pairs, mean=float(np.mean(d)) if d else None, positive=pos, pairs=len(d),
                              sign_p_one_sided=float(binomtest(pos, nz, alternative='greater').pvalue) if nz else None,
                              CLAIM=bool(pos >= 7 and len(d) == 8)),
        safety=dict(collisions_in_flight=coll, missions=len(M), cp_hi=cp_hi(coll, len(M)), PASS=coll == 0,
                    min_sep_m=min((m['min_sep'] for m in M if m['min_sep'] is not None), default=None)),
        live_vlm=dict(false_alarm=fk / max(1, fn), frames=fn, cp=[cp_lo(fk, fn), cp_hi(fk, fn)], PASS=fn > 0 and fk / fn <= 0.10),
        J_flown_vs_kin_pct=[100 * (m['J'] / m['J_kin'] - 1) for m in M if m['J'] and m['J_kin']])
    for key, c in out['cells'].items():
        print(key, 'flown', [round(c['flown'][T], 2) for T in TS], 'calib', [round(c['calib'][T], 2) for T in TS],
              'disc', [round(c['disc'][T], 2) for T in TS])
    print(json.dumps({k: v for k, v in out['criteria'].items() if k != 'J_flown_vs_kin_pct'}, indent=1))
    if Path(a.runs) == ROOT / 'results/vlm_damage/city_step6/runs':
        (ROOT / 'results/vlm_damage/step6_summary.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
