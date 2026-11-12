#!/usr/bin/env python3
"""Step 8 / P2 (pre-registered 10 Nov 2026, plan file "Step 8 plan"): netsim kinematic twin with a range-limited backhaul
and multi-hop relay of the sequential decision list (NetConfig.seq_relay). 360 episodes:
n in {6, 8, 10} x links {full (default range), limited (bh_range 250 m)} x {maqac_seq, rag} x seeds 30000..30029.
Disaster scenario, 600 m square, gateway at the centre, T = 300 s, onboard perception, no take-off phase.
One JSON line per episode (resumable: finished keys are skipped).

    python3 step8_netsim_batch.py --out results/vlm_damage/step8_netsim.jsonl [--workers 5] [--netsim DIR] [--quick]
"""
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS, LINKS, METHODS, SEEDS = (6, 8, 10), {'full': None, 'limited': 250.0}, ('maqac_seq', 'rag'), range(30000, 30030)
KEEP = ('J_session', 'coord_iters', 'coord_iters_max', 'coord_msgs', 'decide_ms', 'decide_ms_p95', 'disconnected_uav_s',
        'near_misses', 'collisions', 'mean_hops', 'users_served', 'tx_msgs')


def job(k):
    n, link, method, seed = k
    chain = method == 'maqac_seq_chain'                       # step 9 (10 Nov 2026): seq planner with the chain rule
    sys.path.insert(0, os.environ['NETSIM'])
    from sim import realistic
    from episode import run_episode
    L = 600.0; s = (L / 1000.0) ** 2
    kw = dict(name=f'disaster_{n}uav', scenario='disaster', n_uav=n, T=300, perception='onboard', altitude=100.0, world=L,
              gateway=(L / 2, L / 2), n_users=int(round(200 * s)), arrival_rate=0.1 * s, takeoff=False)
    if LINKS[link] is not None:
        kw['bh_range'] = LINKS[link]
    cfg = replace(realistic(**kw), seq_relay=True, seq_chain=chain)
    t0 = time.time()
    r = run_episode(cfg, 'maqac_seq' if chain else method, seed)
    return dict(n=n, link=link, method=method, seed=seed, bh_range=cfg.bh_range, wall_s=time.time() - t0,
                host=platform.node(), **{q: r.get(q) for q in KEEP})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--netsim', default=str(ROOT / 'sim' / 'airsim_vlm' / 'netsim'))
    ap.add_argument('--methods', nargs='+', default=list(METHODS), help="step 9: --methods maqac_seq_chain")
    ap.add_argument('--quick', action='store_true', help='one seed per cell, no write to --out (prints)')
    a = ap.parse_args()
    os.environ['NETSIM'] = a.netsim
    keys = [(n, l, m, s) for n in NS for l in LINKS for m in a.methods for s in SEEDS]
    if a.quick:
        keys = [k for k in keys if k[3] == 30000 and k[0] == 6]
        for k in keys:
            print(json.dumps(job(k)))
        return
    done = set()
    if os.path.exists(a.out):
        done = {(r['n'], r['link'], r['method'], r['seed']) for r in map(json.loads, open(a.out))}
    todo = [k for k in keys if k not in done]
    print(f'{len(todo)} of {len(keys)} episodes to run, {a.workers} workers', flush=True)
    with Pool(a.workers) as p, open(a.out, 'a') as fo:
        for i, r in enumerate(p.imap_unordered(job, todo), 1):
            fo.write(json.dumps(r) + '\n'); fo.flush()
            if i % 20 == 0:
                print(f'{i}/{len(todo)}', time.strftime('%H:%M:%S'), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
