#!/usr/bin/env python3
"""Step 5 analysis: CityEnviron flights with the VLM in the loop (hybrid design), against the rules pre-registered in
paper/adhocnet/LLM_INTENT_PLAN_2026-11-05_damage.md ("Step 5 plan" + clarifications + correction).

Input: results/vlm_damage/city/runs/<mission>/{result.json, tracks.json, frames/labels.jsonl, frames/vlm.jsonl},
results/vlm_damage/city/buildings.json, results/vlm_damage/step3_obs_bundle.npz.
Output: results/vlm_damage/step5_summary.json (+ printed table).

Definitions
- Buildings: inventory entries inside the mission square (L = 600 m around the origin).
- Damage label per building: step-3 label model, layout 'indep', seed = mission seed (positions in the simulator frame).
- Assessed (flown): first frame time with >= 2 % of the frame's pixels (6,144 of 640 x 480) on that building.
- Assessed (kinematic twin): first round with a twin UAV within 58 m horizontally (step-3 footprint model).
- Verdict: the FIRST view only; it replays a RescueNet test image of the building's label through the saved V3 posterior
  and conformal set (draw without replacement per class, seeded by mission seed and building index).
- Live VLM false alarm (criterion v): frames showing >= 1 building at >= 2 % whose top class is D or E (major/total).

Correction (9 Nov 2026, after the run): the scene inventory lists some buildings as several stacked floor pieces with the
same (N, E) and different heights (107 entries = 87 buildings). Upper pieces hide lower ones from a downward camera, so 10
pieces could never reach 2 % and every piece also got its own damage label. Default now: one building per footprint
(pieces with the same rounded N, E are merged; their pixels in a frame are summed). --pieces reproduces the
pre-registered per-piece scoring (results/vlm_damage/step5_summary_pieces.json).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from vlm_damage_map import labels                     # noqa: E402  (step-3 label model, unchanged)

D = ROOT / 'results' / 'vlm_damage' / 'city'
PX_MIN = int(0.02 * 640 * 480)
FOOT, T5 = 58.0, 300
cp_lo = lambda k, n: float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
cp_hi = lambda k, n: float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0


def to_area(ne, origin, L):
    ne = np.asarray(ne, float)
    return np.stack([ne[..., 1] - origin[1] + L / 2, ne[..., 0] - origin[0] + L / 2], -1)


MERGE = '--pieces' not in sys.argv


def footprints(inside):
    """Merge stacked pieces (same rounded N, E) into one building. Returns building list and piece-name -> index."""
    if not MERGE:
        return inside, {b['name']: i for i, b in enumerate(inside)}
    keys = sorted({(round(b['N']), round(b['E'])) for b in inside})
    kid = {k: i for i, k in enumerate(keys)}
    out = [dict(name='fp_%d_%d' % k, N=float(k[0]), E=float(k[1])) for k in keys]
    return out, {b['name']: kid[(round(b['N']), round(b['E']))] for b in inside}


def mission(run, blds, bundle):
    r = json.load(open(run / 'result.json')); tr = json.load(open(run / 'tracks.json'))
    L, origin, seed = r['L'], r['origin_ne'], r['seed']
    inside = [b for b in blds if abs(b['N'] - origin[0]) <= L / 2 and abs(b['E'] - origin[1]) <= L / 2]
    inside, idx = footprints(inside)
    names = [b['name'] for b in inside]
    B = to_area(np.array([[b['N'], b['E']] for b in inside]), origin, L)
    lab = labels(B, L, seed, 'indep', None)
    first = np.full(len(names), np.nan)                  # flown: first assessment time
    frames, fa_k, fa_n, nob = [], 0, 0, 0
    vlm = {}
    if (run / 'frames' / 'vlm.jsonl').exists():
        for l in open(run / 'frames' / 'vlm.jsonl'):
            x = json.loads(l); vlm[x['file']] = x
    for l in open(run / 'frames' / 'labels.jsonl'):
        x = json.loads(l)
        acc = defaultdict(int)
        for n, px in (x.get('bld') or []):
            if n in idx:
                acc[idx[n]] += px
        seen = [i for i, px in acc.items() if px >= PX_MIN]
        for i in seen:
            if np.isnan(first[i]) or x['t'] < first[i]:
                first[i] = x['t']
        v = vlm.get(x['file'])
        if v is not None and seen:
            fa_n += 1; fa_k += int(v['top'] in (3, 4)); nob += int(v['top'] == 0)
        if v is not None:
            frames.append(dict(compute=v['compute_s'], lag=v['lag_s']))
    fx = run / 'tracks_kin_fixed.json'                   # the driver logged users (W.pos), not UAVs; fixed tracks (vlm_damage_city_kin.py)
    kin = np.array(json.load(open(fx))['kinematic'] if fx.exists() else tr['kinematic'])   # rounds x n x 2 (world N, E)
    kfirst = np.full(len(names), np.nan)
    for t, pos in enumerate(kin):
        d = np.hypot(np.array([b['N'] for b in inside])[:, None] - pos[None, :, 0],
                     np.array([b['E'] for b in inside])[:, None] - pos[None, :, 1]).min(1)
        hit = (d <= FOOT) & np.isnan(kfirst)
        kfirst[hit] = t
    # verdict replay on the first view
    P, y, qhat = bundle['P'], bundle['y'], float(bundle['qhat'])
    rng = np.random.default_rng([seed, 5])
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(1, 5)}
    cov_k = cov_n = 0; acc = []; sizes = []; sev_missed = sev_ass = 0
    for i in np.where(~np.isnan(first))[0]:
        c = int(lab[i])
        if not pools[c]:
            pools[c] = list(rng.permutation(np.where(y == c)[0]))
        j = pools[c].pop()
        S = np.where(1 - P[j] <= qhat)[0]
        cov_n += 1; cov_k += int(c in S); sizes.append(len(S)); acc.append(int(P[j].argmax() == c))
        if c >= 3:
            sev_ass += 1; sev_missed += int(P[j].argmax() <= 1)
    sev = lab >= 3
    f5 = lambda F: (F <= T5) & ~np.isnan(F)
    coll = r.get('collisions_in_flight') or {}
    return dict(method=r['method'], n=r['n_uav'], seed=seed, n_build=len(names), n_severe=int(sev.sum()),
                severe5=int((f5(first) & sev).sum()), severe5_kin=int((f5(kfirst) & sev).sum()),
                seen5=int(f5(first).sum()), seen5_kin=int(f5(kfirst).sum()),
                cov_k=cov_k, cov_n=cov_n, set_size=float(np.mean(sizes)) if sizes else None,
                acc=float(np.mean(acc)) if acc else None, sev_missed=sev_missed, sev_assessed=sev_ass,
                fa_k=fa_k, fa_n=fa_n, no_building_answers=nob, frames=frames,
                collisions_in_flight=len(coll), collision_events=len(r.get('collision_events') or []),
                J=r['px4'].get('J_session'), J_kin=r['kinematic'].get('J_session'),
                min_sep=(r['safety'].get('mission') or {}).get('min_sep_m'))


def main():
    blds = json.load(open(D / 'buildings.json'))
    bundle = np.load(ROOT / 'results' / 'vlm_damage' / 'step3_obs_bundle.npz')
    M = [mission(p.parent, blds, bundle) for p in sorted((D / 'runs').glob('vlm_*/result.json'))]
    out = dict(missions=len(M), cells={}, rows=[{k: v for k, v in m.items() if k != 'frames'} for m in M])
    cell = defaultdict(list)
    for m in M:
        cell[(m['n'], m['method'])].append(m)
    for (n, meth), ms in sorted(cell.items()):
        ns = sum(m['n_severe'] for m in ms)
        out['cells'][f'{n}|{meth}'] = dict(
            missions=len(ms), severe5=sum(m['severe5'] for m in ms) / max(1, ns),
            severe5_kin=sum(m['severe5_kin'] for m in ms) / max(1, ns),
            seen5=float(np.mean([m['seen5'] / m['n_build'] for m in ms])),
            seen5_kin=float(np.mean([m['seen5_kin'] / m['n_build'] for m in ms])))
    # criteria
    coll = sum(m['collisions_in_flight'] for m in M)
    ck, cn = sum(m['cov_k'] for m in M), sum(m['cov_n'] for m in M)
    gaps = {k: 100 * (v['severe5'] - v['severe5_kin']) for k, v in out['cells'].items()}
    pairs = []
    for n in (4, 6):
        a = {m['seed']: m for m in cell[(n, 'maqac_seq')]}; b = {m['seed']: m for m in cell[(n, 'rag')]}
        pairs += [a[s]['seen5'] - b[s]['seen5'] for s in sorted(set(a) & set(b))]
    pos = sum(d > 0 for d in pairs); nz = sum(d != 0 for d in pairs)
    fk, fn = sum(m['fa_k'] for m in M), sum(m['fa_n'] for m in M)
    fr = [f for m in M for f in m['frames']]
    out['criteria'] = {
        'i_safety': dict(collisions_in_flight=coll, missions=len(M), cp_hi_per_mission=cp_hi(0, len(M)) if coll == 0 else None,
                         PASS=coll == 0),
        'ii_footprint': dict(gap_points=gaps, PASS=all(abs(g) <= 10 for g in gaps.values())),
        'iii_coverage': dict(coverage=ck / max(1, cn), cp_lo=cp_lo(ck, cn), n=cn, PASS=cn > 0 and ck / cn >= 0.90 and cp_lo(ck, cn) >= 0.85),
        'iv_maqacs_vs_rag': dict(mean_diff_buildings_seen5=float(np.mean(pairs)) if pairs else None, pairs=len(pairs),
                                 positive=pos, sign_p=float(binomtest(pos, nz).pvalue) if nz else None),
        'v_live_vlm': dict(false_alarm=fk / max(1, fn), cp=[cp_lo(fk, fn), cp_hi(fk, fn)], frames=fn,
                           PASS=fn > 0 and fk / fn <= 0.10,
                           no_building_answer_share=sum(m['no_building_answers'] for m in M) / max(1, fn),
                           compute_s_p50=float(np.median([f['compute'] for f in fr])) if fr else None,
                           compute_s_p95=float(np.percentile([f['compute'] for f in fr], 95)) if fr else None,
                           lag_s_p95=float(np.percentile([f['lag'] for f in fr], 95)) if fr else None)}
    (ROOT / 'results' / 'vlm_damage' / ('step5_summary.json' if MERGE else 'step5_summary_pieces.json')).write_text(json.dumps(out, indent=1, default=float))
    for k, v in out['cells'].items():
        print(f"{k:16s} severe@5min flown {v['severe5']:.2f} kin {v['severe5_kin']:.2f} | seen@5min {v['seen5']:.2f} kin {v['seen5_kin']:.2f}")
    print(json.dumps(out['criteria'], indent=1, default=float))


if __name__ == '__main__':
    main()
