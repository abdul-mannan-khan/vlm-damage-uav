#!/usr/bin/env python3
# Copied unchanged (apart from this note) from Paper A's HIL (wind-aware-delivery hil/param_dump.py, Jetson ~/paperA_hil_v4h/hil/, 2 Nov 2026).
"""Read ALL PX4 parameters (read-only) and write them in QGroundControl .params format.

  param_dump.py OUT.params [--dev /dev/ttyACM0]
"""
import argparse, struct, time
from pymavlink import mavutil

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--dev', default='/dev/ttyACM0')
a = ap.parse_args()
m = mavutil.mavlink_connection(a.dev, baud=921600, source_system=254)
hb = m.wait_heartbeat(timeout=15)
if hb is None:
    raise SystemExit('no heartbeat')
sysid, comp = m.target_system, m.target_component
params, total = {}, None
m.mav.param_request_list_send(sysid, comp)
t_last = time.time()
while time.time() - t_last < 5:
    r = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
    if r is None:
        continue
    t_last = time.time()
    total = r.param_count
    params[r.param_id] = (r.param_value, r.param_type, r.param_index)
    if len(params) >= total:
        break
for attempt in range(5):                       # fetch any missing indices one by one
    missing = [i for i in range(total or 0) if i not in {v[2] for v in params.values()}]
    if not missing:
        break
    for i in missing:
        m.mav.param_request_read_send(sysid, comp, b'', i)
        r = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
        if r:
            params[r.param_id] = (r.param_value, r.param_type, r.param_index)
with open(a.out, 'w') as f:
    f.write(f'# PX4 parameter dump {time.strftime("%Y-%m-%d %H:%M:%S")} ({len(params)}/{total})\n')
    f.write('# Vehicle-Id Component-Id Name Value Type\n')
    for name in sorted(params):
        v, t, _ = params[name]
        if t in (mavutil.mavlink.MAV_PARAM_TYPE_INT32, mavutil.mavlink.MAV_PARAM_TYPE_UINT32,
                 mavutil.mavlink.MAV_PARAM_TYPE_INT8, mavutil.mavlink.MAV_PARAM_TYPE_UINT8,
                 mavutil.mavlink.MAV_PARAM_TYPE_INT16, mavutil.mavlink.MAV_PARAM_TYPE_UINT16):
            val = str(struct.unpack('<i', struct.pack('<f', v))[0])
        else:
            val = repr(float(v))
        f.write(f'{sysid}\t{comp}\t{name}\t{val}\t{t}\n')
print(f'wrote {a.out}: {len(params)}/{total} parameters')
