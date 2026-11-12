#!/usr/bin/env python3
"""Step 3 of paper/adhocnet/LLM_INTENT_PLAN_2026-11-05_damage.md: damage mapping as a by-product of MAQAC's flight.

World: frozen MAQAC netsim v3 copy (sim/netsim_v3_snap_20261103), realistic(), scenario 'disaster', no city files.
Buildings: centres of the procedural Manhattan grid (side 24.5 m, street 20.2 m, pitch 44.7 m) over the area; the radio
model is unchanged (same as the v3 area study), so the service score is comparable with earlier rows.
Damage labels (1 no damage, 2 minor, 3 major, 4 total; fixed before any run): with s = exp(-d^2 / (2 R^2)), R = 0.08 L,
d = distance to the nearest damage epicentre, p(label) = (1 - s) * P_FAR + s * P_NEAR.
Two layouts: 'indep' (primary, 3 epicentres from an independent stream) and 'colo' (epicentres = the 5 user hotspots of
the disaster scenario, where the users needing service are).
Camera: nadir, footprint radius 58 m (100 m altitude, 60 deg FOV); a building is viewed when an airborne UAV at >= 90 m
is within 58 m. Each view of a building draws a real RescueNet test image with the same label (without replacement per
building) and uses its saved V3-calibrated ensemble posterior and the conformal threshold (step3_obs_bundle.npz).
Arms: maqac_seq (MAQAC-S), rag, survey (MAQAC-S with UAV 0 flying a lawnmower over the whole area, lane 2 x 58 m).

    python3 scripts/vlm_damage_map.py check          # 6 v3 rows must reproduce exactly
    python3 scripts/vlm_damage_map.py smoke          # 2 missions, wall time
    python3 scripts/vlm_damage_map.py run [workers]  # resumable
"""
import copy
import itertools
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim' / 'netsim_v3_snap_20261103'))
import episode                                   # noqa: E402
import sim                                       # noqa: E402
from sim import realistic                        # noqa: E402
from agents import MAQACAgent, METHODS           # noqa: E402

OUT = ROOT / 'results' / 'vlm_damage' / 'step3_map.jsonl'
OBS = np.load(ROOT / 'results' / 'vlm_damage' / 'step3_obs_bundle.npz')
P_IMG, Y_IMG, QHAT = OBS['P'], OBS['y'], float(OBS['qhat'])
POOL = {k: np.where(Y_IMG == k)[0] for k in (1, 2, 3, 4)}
P_FAR = np.array([0.60, 0.30, 0.07, 0.03])
P_NEAR = np.array([0.05, 0.15, 0.35, 0.45])
FOOT, ALT_MIN, BLOCK, PITCH = 58.0, 90.0, 24.5, 24.5 + 20.2
AREAS, FLEETS, SEEDS = (1000.0, 3000.0), (4, 6, 8), range(20000, 20020)
ARMS, LAYOUTS = ('maqac_seq', 'rag', 'survey'), ('indep', 'colo')
CHECKPOINTS = (300, 600, 900)


class MapWorld(sim.World):
    def _positions(self, r, N):
        if self.cfg.scenario == 'disaster':
            r2 = copy.deepcopy(r)
            W = self.cfg.world
            self.hot_cen = r2.uniform(0.1 * W, 0.9 * W, (5, 2))   # the same draw as sim.World._positions
        return super()._positions(r, N)


class SurveyAgent(MAQACAgent):
    """MAQAC-S, except UAV 0 flies a boustrophedon survey of the whole area (lanes 2 x footprint apart)."""
    def __init__(self, i, cfg, world_info, rng, **flags):
        super().__init__(i, cfg, world_info, rng, **flags)
        if i == 0:
            L, lane = cfg.world, 2 * FOOT
            xs = np.arange(lane / 2, L, lane); wp = []
            for q, x in enumerate(xs):
                wp += [(x, lane / 2), (x, L - lane / 2)] if q % 2 == 0 else [(x, L - lane / 2), (x, lane / 2)]
            self.wp, self.k = np.array(wp), 0

    def decide(self, prev_events=()):
        if self.i != 0:
            return super().decide(prev_events)
        if np.hypot(*(self.wp[self.k] - self.pos)) < 5.0:
            self.k = (self.k + 1) % len(self.wp)
        tgt = self.wp[self.k].copy()
        self.last = None
        return tgt, (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))


METHODS['survey'] = (SurveyAgent,) + tuple(METHODS['maqac_seq'][1:])


def grid(L):
    g = np.arange(PITCH - BLOCK / 2, L, PITCH)            # block centres (street at the start of each pitch)
    X, Y = np.meshgrid(g, g, indexing='ij'); return np.c_[X.ravel(), Y.ravel()], g


def labels(B, L, seed, layout, hot_cen):
    r = np.random.default_rng([seed, 91, LAYOUTS.index(layout)])
    epi = hot_cen if layout == 'colo' else r.uniform(0.1 * L, 0.9 * L, (3, 2))
    d = np.hypot(B[:, None, 0] - epi[None, :, 0], B[:, None, 1] - epi[None, :, 1]).min(1)
    s = np.exp(-d ** 2 / (2 * (0.08 * L) ** 2))
    p = (1 - s)[:, None] * P_FAR + s[:, None] * P_NEAR
    u = r.random(len(B))[:, None]
    lab = 1 + (u > np.cumsum(p, 1)).sum(1)
    return np.minimum(lab, 4)


def run_one(W, n, arm, seed):
    f = (W / 1000.0) ** 2
    cfg = realistic(scenario='disaster', n_uav=n, world=W, n_users=int(200 * f), arrival_rate=0.1 * f,
                    gateway=(W / 2, W / 2), name=f'v3_{int(W)}_disaster_{n}')
    episode.World = MapWorld
    st = {}

    def hook(Wd, agents, step):
        if 'B' not in st:
            st['B'], st['g'] = grid(W); st['hot'] = Wd.hot_cen
            st['first'] = np.full(len(st['B']), np.nan); st['nv'] = np.zeros(len(st['B']), int)
            st['ng'] = len(st['g'])
        B, g, ng = st['B'], st['g'], st['ng']
        for i in range(n):
            if Wd.ualt[i] < ALT_MIN:
                continue
            x, y = Wd.upos[i]
            ix = np.arange(max(0, int((x - FOOT) // PITCH) - 1), min(ng, int((x + FOOT) // PITCH) + 2))
            iy = np.arange(max(0, int((y - FOOT) // PITCH) - 1), min(ng, int((y + FOOT) // PITCH) + 2))
            idx = (ix[:, None] * ng + iy[None, :]).ravel()
            hit = idx[np.hypot(B[idx, 0] - x, B[idx, 1] - y) <= FOOT]
            new = hit[np.isnan(st['first'][hit])]
            st['first'][new] = step
            st['nv'][hit] += 1                             # one view per UAV-second over the building

    t0 = time.time()
    r = episode.run_episode(cfg, arm, seed, log_rounds=False, step_hook=hook)
    first, nv = st['first'], st['nv']
    res = dict(J_session=r['J_session'], users_served=r.get('users_served'), users_total=r.get('users_total'),
               collisions=r.get('collisions'), n_build=int(len(first)), n_seen=int((~np.isnan(first)).sum()),
               seen_t={str(c): int((first < c).sum()) for c in CHECKPOINTS}, wall_s=time.time() - t0)
    for layout in LAYOUTS:
        res[layout] = score(labels(st['B'], W, seed, layout, st['hot']), first, nv, seed, arm, layout)
    return res


def score(lab, first, nv, seed, arm, layout):
    # observations: distinct images per building, one per 10 s of dwell (at most 5), at least 1 if viewed
    ro = np.random.default_rng([seed, 92, ARMS.index(arm), LAYOUTS.index(layout)])
    seen = np.where(~np.isnan(first))[0]
    pred1, predB, cov, size = [], [], [], []
    for b in seen:
        k = int(min(5, max(1, nv[b] // 10)))
        imgs = ro.choice(POOL[lab[b]], min(k, len(POOL[lab[b]])), replace=False)
        P = P_IMG[imgs]
        j = int(np.argmax(P.max(1)))                       # primary: the most confident single view
        pred1.append(int(P[j].argmax())); Sset = (1 - P[j]) <= QHAT
        cov.append(bool(Sset[lab[b]])); size.append(int(Sset.sum()))
        lp = np.log(np.clip(P, 1e-12, 1)).sum(0); predB.append(int(lp.argmax()))   # secondary: product (optimistic)
    pred1, predB, cov, size = map(np.array, (pred1, predB, cov, size))
    sev = lab >= 3; ls = lab[seen]
    out = dict(n_severe=int(sev.sum()), severe_seen=int(sev[seen].sum()),
               severe_seen_t={str(c): int((sev & (first < c)).sum()) for c in CHECKPOINTS},
               t_first_severe_median=float(np.nanmedian(first[sev])) if (sev & ~np.isnan(first)).any() else None,
               cov_k=int(cov.sum()), cov_n=int(len(cov)), set_size_mean=float(size.mean()) if len(size) else None,
               acc1=float((pred1 == ls).mean()) if len(ls) else None, accB=float((predB == ls).mean()) if len(ls) else None,
               severe_missed=int(((ls >= 3) & (pred1 <= 1)).sum()), severe_assessed=int((ls >= 3).sum()),
               conf_counts=np.bincount(ls * 5 + pred1, minlength=25).tolist() if len(ls) else None)
    return out


def job(a):
    W, n, arm, seed = a
    try:
        out = run_one(*a)
    except Exception as e:
        import traceback
        out = dict(error=repr(e), tb=traceback.format_exc()[-800:])
    return dict(W=W, n=n, arm=arm, seed=seed, **out)


def check_job(a):
    W, sc, n, m, s = a
    episode.World = sim.World
    f = (W / 1000.0) ** 2
    cfg = realistic(scenario=sc, n_uav=n, world=W, n_users=int(200 * f), arrival_rate=0.1 * f, gateway=(W / 2, W / 2),
                    name=f'v3_{int(W)}_{sc}_{n}')
    return episode.run_episode(cfg, m, s, log_rounds=False)['J_session']


if __name__ == '__main__':
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if mode == 'check':
        CHECK = [(1000.0, 'urban', 4, 'maqac', 20000), (1000.0, 'disaster', 6, 'maqac_seq', 20001),
                 (1000.0, 'suburban', 8, 'rag', 20002), (1000.0, 'disaster', 4, 'rag', 20004),
                 (3000.0, 'disaster', 6, 'maqac_seq', 20000), (3000.0, 'disaster', 6, 'rag', 20001)]
        ref = {}
        for l in open(ROOT / 'results' / 'v3_area_rest' / 'netsim_v3_area_rest.jsonl'):
            d = json.loads(l); ref[(d['W'], d['scenario'], d['n'], d['method'], d['seed'])] = d['J_session']
        CHECK = [a for a in CHECK if a in ref]
        with Pool(len(CHECK)) as p:
            res = p.map(check_job, CHECK)
        ok = all(x == ref[a] for x, a in zip(res, CHECK))
        for x, a in zip(res, CHECK):
            print(a, x, ref[a], 'MATCH' if x == ref[a] else 'DIFF')
        print(f'CHECK {"PASS" if ok else "FAIL"} ({len(CHECK)} rows)', flush=True)
        sys.exit(0 if ok else 1)
    if mode == 'smoke':
        for a in [(1000.0, 4, 'survey', 20000), (3000.0, 6, 'maqac_seq', 20000)]:
            r = job(a)
            for l in LAYOUTS:
                if l in r: r[l].pop('conf_counts', None)
            print(a, json.dumps(r, default=float), flush=True)
        sys.exit(0)
    if mode == 'run':
        done = set()
        if OUT.exists():
            done = {(d['W'], d['n'], d['arm'], d['seed']) for d in map(json.loads, open(OUT)) if 'error' not in d}
        todo = [a for a in itertools.product(AREAS, FLEETS, ARMS, SEEDS) if a not in done]
        todo.sort(key=lambda a: (-a[0], -a[1]))
        print(f'{len(todo)} missions to run', flush=True)
        with Pool(workers) as p, open(OUT, 'a') as f:
            for i, r in enumerate(p.imap_unordered(job, todo)):
                f.write(json.dumps(r, default=float) + '\n'); f.flush()
                if (i + 1) % 20 == 0:
                    print(f'{i + 1}/{len(todo)}', time.strftime('%H:%M:%S'), flush=True)
        print('finished', flush=True)
