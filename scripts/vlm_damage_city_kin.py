#!/usr/bin/env python3
"""Step 5: recompute the kinematic twin's UAV track for every flown mission (the driver logged W.pos, which holds the
USERS' positions, instead of W.upos; found 9 Nov 2026). The twin is deterministic: same cfg, method and seed as the
driver's v3 branch, no AirSim. Check: the recomputed twin's J_session must equal the twin J logged in result.json.
Writes <mission>/tracks_kin_fixed.json (rounds x n x 2, world N, E)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim' / 'airsim_vlm' / 'netsim')); sys.path.insert(0, str(ROOT / 'sim' / 'airsim_vlm' / 'airsim' / 'maqac_px4'))
from sim import realistic                 # noqa: E402
from episode import run_episode           # noqa: E402
from heightmap import AreaFrame           # noqa: E402

D = ROOT / 'results' / 'vlm_damage' / 'city' / 'runs'
bad = 0
for p in sorted(D.glob('vlm_*/result.json')):
    r = json.load(open(p)); s = (r['L'] / 1000.0) ** 2
    cfg = realistic(name=f"{r['scenario']}_{r['n_uav']}uav", scenario=r['scenario'], n_uav=r['n_uav'], T=r['T'],
                    perception='onboard', altitude=r['alt'], world=r['L'], gateway=(r['L'] / 2, r['L'] / 2),
                    n_users=int(round(200 * s)), arrival_rate=0.1 * s, takeoff=False)
    fr = AreaFrame(cfg.world, r['origin_ne']); tr = []
    kin = run_episode(cfg, r['method'], r['seed'], step_hook=lambda W, a, st: tr.append(np.round(fr.to_world(W.upos[:, :2]), 2).tolist()))
    ok = abs(kin['J_session'] - r['kinematic']['J_session']) < 1e-6
    bad += not ok
    json.dump(dict(kinematic=tr), open(p.parent / 'tracks_kin_fixed.json', 'w'))
    print(p.parent.name, 'J twin', round(kin['J_session'], 3), 'logged', round(r['kinematic']['J_session'], 3), 'OK' if ok else 'MISMATCH', len(tr), 'rounds')
print('mismatches', bad)
