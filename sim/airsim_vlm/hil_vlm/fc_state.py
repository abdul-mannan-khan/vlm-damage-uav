#!/usr/bin/env python3
# Copied unchanged (apart from this note) from Paper A's HIL (wind-aware-delivery hil/fc_state.py, Jetson ~/paperA_hil_v4h/hil/, 2 Nov 2026).
"""Show armed state / main mode of the Pixhawk; optionally force-disarm (simulation only) and reboot.
  fc_state.py [disarm] [reboot]"""
import sys, time
from pymavlink import mavutil
m = mavutil.mavlink_connection("/dev/ttyACM0", baud=921600, source_system=255, source_component=190)
m.wait_heartbeat(timeout=15)


def state():
    t = time.time()
    while time.time() - t < 3:
        h = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if h and h.get_srcComponent() == 1:
            return dict(armed=bool(h.base_mode & 128), main_mode=(h.custom_mode >> 16) & 255, status=h.system_status)


print("state", state())
if "disarm" in sys.argv:
    for _ in range(5):
        m.mav.command_long_send(1, 1, 400, 0, 0, 21196, 0, 0, 0, 0, 0)
        time.sleep(0.2)
    print("after disarm", state())
if "reboot" in sys.argv:
    m.mav.command_long_send(1, 1, 246, 0, 1, 0, 0, 0, 0, 0, 0)
    a = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
    print("reboot ack", a.result if a else None)
