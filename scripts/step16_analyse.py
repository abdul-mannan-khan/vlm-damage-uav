#!/usr/bin/env python3
"""Step 16 analysis (prereg results/vlm_damage/step16_hil_planners_prereg.txt): RSG, RSG-BF and RAG on flight hardware
with the int8 VLM in the loop. Input: results/vlm_damage/hil_step16/<mission>/result.json (copied from the Jetson).
Writes results/vlm_damage/step16_summary.json. Memory in MB/1000 (as step 13)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

R = Path(__file__).resolve().parents[1] / 'results' / 'vlm_damage'
D = R / 'hil_step16'
PLAN = {'maqac_seq': 'RSG', 'maqac_seq_bf': 'RSG-BF', 'rag': 'RAG'}
SEEDS = [21000, 21001, 21002, 21003, 21004]


def cp_upper(k, n, a=0.05):
    return float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0


def load():
    M = {}
    for p in PLAN:
        for s in SEEDS:
            f = D / f'{p}_chain_vlmprior_s{s}' / 'result.json'
            if f.exists():
                M[(p, s)] = json.load(open(f))
    return M


def main():
    M = load()
    n = len(M)
    out = dict(n_missions=n, expected=len(PLAN) * len(SEEDS), missing=[f'{p}/{s}' for p in PLAN for s in SEEDS if (p, s) not in M])
    bad = [k for k, r in M.items() if r['aborted'] or r['link']['mode_drops'] or r['link']['disarms_in_air']
           or r['link']['failsafe_msgs']]
    out['Q1'] = dict(k=len(bad), n=n, cp_upper=cp_upper(len(bad), n), pass_=not bad and n == out['expected'])
    ov = sum(r['flight']['round_overruns'] for r in M.values())
    out['Q2'] = dict(overruns=ov, rounds=sum(r['flight']['rounds'] for r in M.values()), pass_=ov == 0)
    v = [r['vlm'] for r in M.values()]
    req, res = sum(x['requests'] for x in v), sum(x['results'] for x in v)
    out['Q3'] = dict(vlm_p90_ms=[min(x['vlm_ms']['p90'] for x in v), max(x['vlm_ms']['p90'] for x in v)],
                     answered=res, requested=req, errors=sum(x['errors'] for x in v),
                     age_max_s=max(x['belief_age_at_use_s']['max'] for x in v))
    out['Q3']['pass_'] = (out['Q3']['vlm_p90_ms'][1] <= 5000 and res >= 0.95 * req and out['Q3']['errors'] == 0
                          and out['Q3']['age_max_s'] <= 10)
    per = {}
    for p, lab in PLAN.items():
        rs = [M[(p, s)] for s in SEEDS if (p, s) in M]
        if not rs:
            continue
        per[lab] = dict(
            fleet_ms_p95=[min(r['flight']['fleet_decide_ms']['p95'] for r in rs), max(r['flight']['fleet_decide_ms']['p95'] for r in rs)],
            fleet_ms_max=max(r['flight']['fleet_decide_ms']['max'] for r in rs),
            round_ms_max=max(r['flight']['round_compute_ms']['max'] for r in rs),
            coord_msgs=float(np.mean([r['hil']['coord_msgs'] for r in rs])),
            coord_iters=float(np.mean([r['hil']['coord_iters'] for r in rs])),
            dJ_replay_pct=[min(r['delta_J_vs_replay_pct'] for r in rs), max(r['delta_J_vs_replay_pct'] for r in rs)],
            dJ_kin_pct=[min(r['delta_J_pct'] for r in rs), max(r['delta_J_pct'] for r in rs)],
            track_mean_m=[min(r['flight']['track_err_h_m']['mean'] for r in rs), max(r['flight']['track_err_h_m']['mean'] for r in rs)],
            mem_peak_gb=[min(r['vlm']['mem_used_peak_mb'] for r in rs) / 1000, max(r['vlm']['mem_used_peak_mb'] for r in rs) / 1000],
            weights_applied=int(sum(r['vlm']['weights_applied'] for r in rs)),
            n=len(rs))
    out['per_planner'] = per
    out['Q4'] = dict(pass_=all(x['fleet_ms_p95'][1] <= 100 for x in per.values()))
    mem = max(r['vlm']['mem_used_peak_mb'] for r in M.values())
    out['Q5'] = dict(mem_peak_gb=mem / 1000, pass_=mem <= 7300)   # OOM: checked in the campaign logs (no kill)
    dj = [abs(r['delta_J_vs_replay_pct']) for r in M.values()]
    out['Q6'] = dict(max_abs_pct=max(dj), pass_=max(dj) <= 5)
    tr = [r['flight']['track_err_h_m']['mean'] for r in M.values()]
    out['Q7'] = dict(range_m=[min(tr), max(tr)], pass_=max(tr) <= 5)
    # descriptive (no test): paired J differences, hardware runs and their kinematic twins
    pairs = {}
    for a, b in (('maqac_seq_bf', 'maqac_seq'), ('maqac_seq', 'rag'), ('maqac_seq_bf', 'rag')):
        ss = [s for s in SEEDS if (a, s) in M and (b, s) in M]
        for src in ('hil', 'kinematic'):
            d = [100 * (M[(a, s)][src]['J_session'] / M[(b, s)][src]['J_session'] - 1) for s in ss]
            if d:
                pairs[f'{PLAN[a]} - {PLAN[b]} ({src})'] = dict(mean_pct=float(np.mean(d)), min_pct=min(d), max_pct=max(d),
                                                              pos=sum(x > 0 for x in d), neg=sum(x < 0 for x in d), n=len(d))
    out['paired'] = pairs
    json.dump(out, open(R / 'step16_summary.json', 'w'), indent=1, default=float)
    for k in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'):
        print(k, 'PASS' if out[k]['pass_'] else 'FAIL', {kk: v for kk, v in out[k].items() if kk != 'pass_'})
    for lab, x in per.items():
        print(lab, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 2)) for k, v in x.items()})
    for k, x in pairs.items():
        print(k, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in x.items()})
    print('missions', n, 'missing', out['missing'])


if __name__ == '__main__':
    main()
