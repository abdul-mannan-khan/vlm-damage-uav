#!/usr/bin/env python3
"""Why does MAQAC-S lose the gateway path on a 250 m backhaul (step 8)? Diagnostic, not a hypothesis test (10 Nov 2026).
Re-runs a few step-8 episodes (same config as step8_netsim_batch.py) with a step hook that records, every second:
airborne UAVs' distance from the gateway, the nearest other airborne UAV, gateway-connected count, and for each
disconnected UAV its distance to the nearest connected node. Checks first that J matches the batch rows.

    python3 scripts/step8_disconnect_diag.py --n 8 --seeds 30000 30001 30002 30003
"""
import argparse, json, os, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim' / 'airsim_vlm' / 'netsim'))
from sim import realistic          # noqa: E402
from episode import run_episode    # noqa: E402


def cfg_for(n, chain=False):
    L = 600.0; s = (L / 1000.0) ** 2
    return replace(realistic(name=f'disaster_{n}uav', scenario='disaster', n_uav=n, T=300, perception='onboard', altitude=100.0,
                             world=L, gateway=(L / 2, L / 2), n_users=int(round(200 * s)), arrival_rate=0.1 * s, takeoff=False,
                             bh_range=250.0), seq_relay=True, seq_chain=chain)


def run(n, method, seed):
    chain = method.endswith('+chain')
    cfg = cfg_for(n, chain); rec = []
    method = method.replace('+chain', '')

    def hook(W, agents, step):
        bh = W.bh; air = W.airborne(); P = W.upos
        gw = np.asarray(W.gw, float)
        dg = np.linalg.norm(P - gw, axis=1)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(D, np.inf)
        con = bh['connected'] & air
        far = []
        for i in np.where(air & ~bh['connected'])[0]:          # gap to the nearest connected node (UAV or gateway)
            cand = [np.linalg.norm(P[i] - gw)] + [D[i, j] for j in np.where(con)[0]]
            far.append(float(min(cand)))
        rec.append(dict(dg=dg[air].tolist(), nn=D[air][:, air].min(1).tolist() if air.sum() > 1 else [],
                        n_air=int(air.sum()), n_con=int(con.sum()), gap=far))
    r = run_episode(cfg, method, seed, step_hook=hook)
    method = method + ('+chain' if chain else '')
    dg = np.concatenate([x['dg'] for x in rec]); nn = np.concatenate([x['nn'] for x in rec if x['nn']])
    gaps = [g for x in rec for g in x['gap']]
    return dict(n=n, method=method, seed=seed, J=r['J_session'], disc=r['disconnected_uav_s'],
                dist_gw_mean=float(dg.mean()), dist_gw_p90=float(np.quantile(dg, 0.9)),
                frac_beyond_250_of_gw=float((dg > 250).mean()), nn_mean=float(nn.mean()),
                gap_median=float(np.median(gaps)) if gaps else None, gap_p90=float(np.quantile(gaps, 0.9)) if gaps else None,
                first_disc_s=next((k for k, x in enumerate(rec) if x['n_con'] < x['n_air']), None))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=[30000, 30001, 30002, 30003])
    ap.add_argument('--methods', nargs='+', default=['maqac_seq', 'rag'])
    ap.add_argument('--out', default='results/vlm_damage/step8_disconnect_diag.json')
    a = ap.parse_args()
    batch = {(r['n'], r['method'], r['seed']): r for r in map(json.loads, open(ROOT / 'results/vlm_damage/step8_netsim.jsonl'))
             if r['link'] == 'limited'}
    out = []
    for seed in a.seeds:
        for m in a.methods:
            d = run(a.n, m, seed)
            b = batch.get((a.n, m, seed))
            d['J_batch'] = b['J_session'] if b else None
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
            out.append(d)
    p = ROOT / a.out
    p.write_text(json.dumps(out, indent=1))
