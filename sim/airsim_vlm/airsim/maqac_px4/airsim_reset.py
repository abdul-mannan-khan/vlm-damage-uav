#!/usr/bin/env python3
"""Reset AirSim between missions: remove spawned users (user_*), reset the vehicles to their spawn points.
Run with PX4 stopped (px4_stack.sh stop), then start PX4 again: PX4 re-reads its offsets at rest.
    python3 airsim_reset.py --ip 10.142.0.5"""
import argparse
import time
import airsim

ap = argparse.ArgumentParser()
ap.add_argument('--ip', default='10.142.0.5')
a = ap.parse_args()
c = airsim.MultirotorClient(ip=a.ip)
c.confirmConnection()
c.simPause(False)
gone = 0
for nm in c.simListSceneObjects('user_.*'):
    gone += bool(c.simDestroyObject(nm))
c.reset()
time.sleep(3)
left = len(c.simListSceneObjects('user_.*'))
print(f'reset done: removed {gone} users, {left} left')
