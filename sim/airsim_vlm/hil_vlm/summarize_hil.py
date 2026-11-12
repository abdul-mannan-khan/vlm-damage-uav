#!/usr/bin/env python3
"""Per-method summary of the MAQAC HIL missions (mean over seeds, min-max): results/hil/<method>_s<seed>/result.json.
  python3 sim/hil/summarize_hil.py results/hil [--json results/hil/hil_summary.json]"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--json')
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.root, '*', 'result.json'))):
    r = json.load(open(f))
    if os.path.basename(os.path.dirname(f)).startswith('shakedown') or r.get('T') != 300:
        continue
    rows.append(r)
out = {}
for m in ('maqac_seq', 'maqac', 'rag'):
    R = [r for r in rows if r['method'] == m]
    ok = [r for r in R if not r['aborted']]
    if not R:
        continue

    def col(fn):
        v = np.array([fn(r) for r in ok], float)
        return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=int(v.size)) if v.size else None
    out[m] = dict(
        missions=len(R), aborted=len(R) - len(ok), seeds=sorted(r['seed'] for r in ok),
        J_hil=col(lambda r: r['hil']['J_session']), J_kin=col(lambda r: r['kinematic']['J_session']),
        delta_J_pct=col(lambda r: r['delta_J_pct']),
        err_h_mean=col(lambda r: r['flight']['track_err_h_m']['mean']),
        err_h_p95=col(lambda r: r['flight']['track_err_h_m']['p95']),
        err_h_max=col(lambda r: r['flight']['track_err_h_m']['max']),
        err_3d_p95=col(lambda r: r['flight']['track_err_3d_m']['p95']),
        sp_err_mean=col(lambda r: r['link']['sp_err_mean_m']), sp_err_p95=col(lambda r: r['link']['sp_err_p95_m']),
        min_sep=col(lambda r: r['flight']['min_sep_3d_m']),
        fleet_ms_mean=col(lambda r: r['flight']['fleet_decide_ms']['mean']),
        fleet_ms_p95=col(lambda r: r['flight']['fleet_decide_ms']['p95']),
        fleet_ms_max=col(lambda r: r['flight']['fleet_decide_ms']['max']),
        uav_ms_mean=col(lambda r: r['flight']['uav_decide_ms']['mean']),
        uav_ms_p95=col(lambda r: r['flight']['uav_decide_ms']['p95']),
        round_ms_p95=col(lambda r: r['flight']['round_compute_ms']['p95']),
        overruns=sum(r['flight']['round_overruns'] for r in ok),
        coord_msgs=col(lambda r: r['hil']['coord_msgs']),
        setpoint_hz=col(lambda r: r['link']['setpoint_hz']), actuator_hz=col(lambda r: r['link'].get('actuator_hz') or np.nan),
        rtt_ms_mean=col(lambda r: r['link'].get('rtt_ms_mean') or np.nan),
        rtt_ms_p95=col(lambda r: r['link'].get('rtt_ms_p95') or np.nan),
        mode_drops=sum(r['link'].get('mode_drops', 0) for r in ok),
        disarms=sum(r['link'].get('disarms_in_air', 0) for r in ok),
        failsafe_msgs=sum(len(r['link'].get('failsafe_msgs') or []) for r in ok))


def f(d, k=1):
    return '-' if d is None else f"{d['mean']:.{k}f} [{d['min']:.{k}f}, {d['max']:.{k}f}]"


print('| method | n | J_session HIL | J_session kinematic | delta J % | track err mean (m) | track err p95 (m) | '
      'min 3-D sep (m) | fleet decide mean / p95 (ms) | coord msgs/round |')
print('|---|---|---|---|---|---|---|---|---|---|')
for m, d in out.items():
    print(f"| {m} | {d['missions'] - d['aborted']}/{d['missions']} | {f(d['J_hil'])} | {f(d['J_kin'])} | "
          f"{f(d['delta_J_pct'], 2)} | {f(d['err_h_mean'], 2)} | {f(d['err_h_p95'], 2)} | {f(d['min_sep'])} | "
          f"{f(d['fleet_ms_mean'])} / {f(d['fleet_ms_p95'])} | {f(d['coord_msgs'])} |")
for m, d in out.items():
    print(f"{m}: per-UAV decide mean {f(d['uav_ms_mean'])} p95 {f(d['uav_ms_p95'])} ms; round compute p95 "
          f"{f(d['round_ms_p95'])} ms; overruns {d['overruns']}; setpoint {f(d['setpoint_hz'])} Hz; actuator "
          f"{f(d['actuator_hz'], 0)} Hz; RTT {f(d['rtt_ms_mean'])} / p95 {f(d['rtt_ms_p95'])} ms; 50 Hz setpoint err "
          f"mean {f(d['sp_err_mean'], 2)} p95 {f(d['sp_err_p95'], 2)} m; mode drops {d['mode_drops']}, "
          f"disarms {d['disarms']}, failsafe msgs {d['failsafe_msgs']}; aborted {d['aborted']}")
if a.json:
    json.dump(out, open(a.json, 'w'), indent=1)
