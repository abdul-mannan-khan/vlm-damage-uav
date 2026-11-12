#!/usr/bin/env python3
# Copied unchanged (apart from this note) from Paper A's HIL (wind-aware-delivery hil/param_set.py, Jetson ~/paperA_hil_v4h/hil/, 2 Nov 2026).
"""Set PX4 parameters and verify them, or restore a full .params backup.

  param_set.py NAME=VALUE [NAME=VALUE ...]        # set and read back
  param_set.py --restore BACKUP.params             # write every value that differs from the backup
  param_set.py --reboot                            # reboot the flight controller
Integer parameters are sent as their bit pattern in the float field, as PX4 expects.
"""
import argparse, struct, sys, time
from pymavlink import mavutil

INT_TYPES = {1, 2, 3, 4, 5, 6}   # MAV_PARAM_TYPE_UINT8 ... INT32


def connect(dev):
    m = mavutil.mavlink_connection(dev, baud=921600, source_system=254)
    if m.wait_heartbeat(timeout=15) is None:
        sys.exit('no heartbeat')
    return m


def decode(r):
    if r.param_type in INT_TYPES:
        return struct.unpack('<i', struct.pack('<f', r.param_value))[0]
    return float(r.param_value)


def read(m, name, tries=3):
    for _ in range(tries):
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        t = time.time()
        while time.time() - t < 2:
            r = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
            if r and r.param_id == name:
                return decode(r), r.param_type
    return None, None


def write(m, name, value):
    cur, ptype = read(m, name)
    if ptype is None:
        return f'{name}: not found'
    if ptype in INT_TYPES:
        raw = struct.unpack('<f', struct.pack('<i', int(value)))[0]
    else:
        raw = float(value)
    for _ in range(3):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(), raw, ptype)
        t = time.time()
        while time.time() - t < 2:
            r = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
            if r and r.param_id == name:
                new = decode(r)
                ok = (new == int(value)) if ptype in INT_TYPES else abs(new - float(value)) < 1e-5 * max(1, abs(float(value)))
                if ok:
                    return f'{name}: {cur} -> {new}'
    return f'{name}: FAILED (still {read(m, name)[0]})'


def load_backup(path):
    vals = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        _, _, name, val, ptype = line.split('\t')
        vals[name] = int(val) if int(ptype) in INT_TYPES else float(val)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('assign', nargs='*')
    ap.add_argument('--dev', default='/dev/ttyACM0')
    ap.add_argument('--restore')
    ap.add_argument('--reboot', action='store_true')
    a = ap.parse_args()
    m = connect(a.dev)
    if a.restore:
        want = load_backup(a.restore)
        n_diff = 0
        for name, val in sorted(want.items()):
            cur, ptype = read(m, name)
            if cur is None:
                print(f'{name}: not present on the board')
                continue
            same = (cur == val) if ptype in INT_TYPES else abs(cur - val) < 1e-5 * max(1, abs(val))
            if not same:
                n_diff += 1
                print(write(m, name, val))
        print(f'restore: {n_diff} parameters differed from the backup')
    for s in a.assign:
        name, val = s.split('=', 1)
        print(write(m, name, val))
    if a.reboot:
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 0, 1, 0, 0, 0, 0, 0, 0)
        print('reboot sent')


if __name__ == '__main__':
    main()
