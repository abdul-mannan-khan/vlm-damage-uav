#!/usr/bin/env python3
# Copied unchanged (apart from this note) from Paper A's HIL (wind-aware-delivery hil/backup_by_name.py, Jetson ~/paperA_hil_v4h/hil/, 2 Nov 2026).
"""Back up PX4 parameters one by one (a full PARAM_REQUEST_LIST is not answered while the board runs SIH/HIL).
The parameter names and types come from an earlier full backup; the values are read now from the board.

  backup_by_name.py REFERENCE.params OUT.params
"""
import os, sys, time
from param_set import INT_TYPES, connect, read

ref, out = sys.argv[1], sys.argv[2]
m = connect(os.environ.get('PX4_DEV', '/dev/ttyACM0'))   # MAQAC 4 Nov: by-id port in the AirSim HITL
rows, missing = [], []
for line in open(ref):
    if line.startswith('#') or not line.strip():
        continue
    sysid, comp, name, _, ptype = line.rstrip('\n').split('\t')
    val, pt = read(m, name)
    if val is None:
        missing.append(name)
        continue
    rows.append(f"{sysid}\t{comp}\t{name}\t{val if pt in INT_TYPES else repr(float(val))}\t{pt}\n")
with open(out, 'w') as f:
    f.write(f"# Onboard parameters read one by one on {time.strftime('%Y-%m-%d %H:%M:%S')} (names from {ref})\n")
    f.write("# Vehicle-Id Component-Id Name Value Type\n")
    f.writelines(rows)
print(f"{out}: {len(rows)} parameters, {len(missing)} not present: {missing[:10]}")
