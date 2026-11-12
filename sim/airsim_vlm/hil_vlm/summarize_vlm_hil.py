#!/usr/bin/env python3
"""Pass/fail table of the closed-loop VLM HIL against the pre-registration in PLAN.md (bars P1-P8), plus the score
against the kinematic twin and the step-6 SITL twin (results/vlm_damage/step6_summary.json, same method and seed).

    python3 summarize_vlm_hil.py RESULTS_DIR [--json OUT.json]
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
STEP6 = os.path.join(HERE, '..', '..', '..', 'results', 'vlm_damage', 'step6_summary.json')


def cp_upper(k, n, a=0.05):
    return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, '*', 'result.json')))
         if '/shakedown_' not in f]
    s6 = {(r['method'], r['seed']): r for r in json.load(open(STEP6))['rows']} if os.path.exists(STEP6) else {}
    rows, fails = [], 0
    for r in R:
        v, f, l = r['vlm'], r['flight'], r['link'] or {}
        fs = len(l.get('failsafe_msgs') or []) + int(l.get('mode_drops') or 0) + int(r['aborted'])
        fails += fs > 0
        sitl = s6.get(('maqac_seq', r['seed']))
        rows.append(dict(name=f"{r['vlm_planner']}_s{r['seed']}", T=r['T'],
                         failsafe_or_abort=fs, overruns=f['round_overruns'], round_ms_max=(f['round_compute_ms'] or {}).get('max'),
                         vlm_p90_ms=(v['vlm_ms'] or {}).get('p90'), age_max_s=(v['belief_age_at_use_s'] or {}).get('max'),
                         answered=f"{v['results']}/{v['requests']}", dropped=v['dropped'], errors=v['errors'],
                         mem_peak_mb=v['mem_used_peak_mb'], err_h_mean=(f['track_err_h_m'] or {}).get('mean'),
                         fleet_ms_p95=(f['fleet_decide_ms'] or {}).get('p95'), min_clear_m=(f.get('city') or {}).get('min_clearance_m'),
                         J_hil=(r['hil'] or {}).get('J_session'), J_kin=r['kinematic']['J_session'],
                         J_replay=(r['replay_twin'] or {}).get('J_session'), J_sitl_step6=sitl['J'] if sitl else None))
    n = len(rows)
    out = dict(missions=n, rows=rows,
               P1_no_failsafe=dict(k=fails, n=n, cp95_upper=cp_upper(fails, n) if n else None, PASS=fails == 0),
               P2_no_overrun=all(x['overruns'] == 0 for x in rows),
               P3_vlm_p90_le_5s=all((x['vlm_p90_ms'] or 1e9) <= 5000 for x in rows),
               P4_belief_age_le_10s=all((x['age_max_s'] or 1e9) <= 10 for x in rows),
               P4_answered_ge_95pct=all(int(x['answered'].split('/')[0]) >= 0.95 * int(x['answered'].split('/')[1]) for x in rows),
               P5_fleet_p95_le_100ms=all((x['fleet_ms_p95'] or 1e9) <= 100 for x in rows),
               P6_mem_peak_le_7300mb=all((x['mem_peak_mb'] or 1e9) <= 7300 for x in rows),
               P9_track_le_5m_and_clear=all((x['err_h_mean'] or 1e9) <= 5 and (x['min_clear_m'] is None or x['min_clear_m'] > 0) for x in rows))
    off = [x for x in rows if x['name'].startswith('off') and x['J_hil'] and x['J_sitl_step6'] and x['T'] == 300]   # step 6 flew 300 s
    if off:
        d = np.array([100 * (x['J_hil'] / x['J_sitl_step6'] - 1) for x in off])
        out['P7_J_vs_sitl_pct'] = dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()), PASS=bool(np.all(np.abs(d) <= 5)))
    pr = [x for x in rows if x['name'].startswith('prior') and x['J_hil'] and x['J_replay']]
    if pr:
        d = np.array([100 * (x['J_hil'] / x['J_replay'] - 1) for x in pr])
        out['P8_J_vs_replay_pct'] = dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()), PASS=bool(np.all(np.abs(d) <= 5)))
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
