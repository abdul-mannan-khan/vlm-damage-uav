#!/usr/bin/env python3
"""Calibration missions for the certificate transfer test (user, 5 Nov 2026: test Theorem 9 on PX4-flown missions).
Kinematic missions with exactly the configuration of the flown test missions (run_maqac_airsim.py --model old, urban,
4 UAVs, MAQAC, on-board perception, T = 300 s, 600 m area; plan_city_cert_n4.txt), on calibration seeds 31000-31079,
disjoint from the test seeds 20000-20019. Output: one JSON line per mission with the mission-level certificate terms.
    python3 scripts/airsim_cert_calibrate.py WORKERS OUT.jsonl
"""
import json, sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]       # sim/airsim/maqac_px4 -> repo root (also /work/maqac on the VM)
sys.path.insert(0, str(ROOT / 'src' / 'netsim'))
from sim import NetConfig          # noqa: E402
from episode import run_episode    # noqa: E402
L, N, T = 600.0, 4, 300


def cfg():
    base = NetConfig(); s = (L / 1000.0) ** 2
    return replace(base, name=f'urban_{N}uav', scenario='urban', n_uav=N, T=T, perception='onboard', altitude=100.0,
                   battery_wh=40.0, world=L, gateway=(L / 2, L / 2), n_users=int(round(base.n_users * s)),
                   arrival_rate=base.arrival_rate * s)


def job(seed):
    r = run_episode(cfg(), 'maqac', seed, log_rounds=True)
    rr = [x for x in r['rounds'] if 'g_hat_full' in x]
    return dict(seed=seed, qos_score=r['qos_score'], gh=sum(x['g_hat_full'] for x in rr),
                ubh=sum(x['ub_hat_full'] for x in rr), g=sum(x['g_full'] for x in rr), rounds=len(rr))


if __name__ == '__main__':
    w, out = int(sys.argv[1]), sys.argv[2]
    with Pool(w) as p, open(out, 'w') as f:
        for r in p.imap_unordered(job, range(31000, 31080)):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('calibration done')
