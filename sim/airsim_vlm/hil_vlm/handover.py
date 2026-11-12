#!/usr/bin/env python3
"""Rig hand-over checks for the shared X500 / Pixhawk 6C (user rule 5 Nov 2026; checklist in
~/lit_results/x500_airsim_hitl_bridge/README.md). Used by hil_campaign_city.sh.

  handover.py --dev DEV secure    # if armed: force-disarm (MAV_CMD_COMPONENT_ARM_DISARM 0, 21196); then Hold mode
  handover.py --dev DEV reboot    # secure, note time_boot_ms, reboot, reconnect, require time_boot_ms to have reset
  handover.py --dev DEV final     # disarmed and heartbeat system_status != 8 (MAV_STATE_FLIGHT_TERMINATION)
Prints one 'HANDOVER {json}' line; exit 0 = pass, 1 = fail, 2 = no link.
"""
import argparse, json, os, sys, time
from pymavlink import mavutil

FLIGHT_TERMINATION = 8
PX4_AUTO, PX4_LOITER = 4, 3          # PX4 custom main mode AUTO, sub mode LOITER = "Hold"


def connect(dev, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        if os.path.exists(dev):
            try:
                m = mavutil.mavlink_connection(dev, baud=921600, source_system=254, source_component=190)
                if m.wait_heartbeat(timeout=10):
                    return m
                m.close()
            except Exception:
                pass
        time.sleep(2)
    return None


def heartbeat(m, wait=4.0):
    t = time.time()
    while time.time() - t < wait:
        h = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if h and h.get_srcSystem() == 1 and h.get_srcComponent() == 1:
            return dict(armed=bool(h.base_mode & 128), main_mode=(h.custom_mode >> 16) & 255,
                        sub_mode=(h.custom_mode >> 24) & 255, status=int(h.system_status))
    return None


def boot_ms(m, wait=6.0):
    m.mav.command_long_send(1, 1, 512, 0, 2, 0, 0, 0, 0, 0, 0)        # MAV_CMD_REQUEST_MESSAGE SYSTEM_TIME
    t = time.time()
    while time.time() - t < wait:
        s = m.recv_match(type='SYSTEM_TIME', blocking=True, timeout=1)
        if s and s.get_srcSystem() == 1:
            return int(s.time_boot_ms)
    return None


def secure(m):
    st = heartbeat(m)
    out = dict(before=st, force_disarm=False)
    if st and st['armed']:
        out['force_disarm'] = True
        for _ in range(5):
            m.mav.command_long_send(1, 1, 400, 0, 0, 21196, 0, 0, 0, 0, 0)
            time.sleep(0.3)
            st = heartbeat(m, 2)
            if st and not st['armed']:
                break
    for _ in range(3):                                               # Hold (AUTO.LOITER)
        m.mav.command_long_send(1, 1, 176, 0, 1, PX4_AUTO, PX4_LOITER, 0, 0, 0, 0)
        time.sleep(0.5)
        st = heartbeat(m, 2)
        if st and st['main_mode'] == PX4_AUTO and st['sub_mode'] == PX4_LOITER:
            break
    out['after'] = st
    out['ok'] = bool(st) and not st['armed']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', default='/dev/serial/by-id/usb-Auterion_PX4_FMU_v6C.x_0-if00')
    ap.add_argument('--wait', type=float, default=45.0, help='seconds to wait after the reboot command')
    ap.add_argument('cmd', choices=['secure', 'reboot', 'final'])
    a = ap.parse_args()
    m = connect(a.dev)
    if m is None:
        print('HANDOVER ' + json.dumps(dict(cmd=a.cmd, ok=False, error='no heartbeat')))
        return 2
    if a.cmd == 'secure':
        r = secure(m)
    elif a.cmd == 'reboot':
        r = dict(secure=secure(m), boot_ms_before=boot_ms(m))
        m.mav.command_long_send(1, 1, 246, 0, 1, 0, 0, 0, 0, 0, 0)    # MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
        ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        r['reboot_ack'] = int(ack.result) if ack else None
        m.close()
        t_sent = time.time()
        time.sleep(a.wait)
        m = connect(a.dev, timeout=90)
        r['boot_ms_after'] = boot_ms(m) if m else None
        since = (time.time() - t_sent) * 1000.0
        b0, b1 = r['boot_ms_before'], r['boot_ms_after']
        # reset = uptime now is shorter than the time since the reboot command (plus 10 s slack), and below the old uptime
        r['rebooted'] = b1 is not None and b1 < since + 10000 and (b0 is None or b1 < b0)
        r['ok'] = r['rebooted']
    else:
        st = heartbeat(m)
        r = dict(state=st, ok=bool(st) and not st['armed'] and st['status'] != FLIGHT_TERMINATION)
    r['cmd'] = a.cmd
    print('HANDOVER ' + json.dumps(r))
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
