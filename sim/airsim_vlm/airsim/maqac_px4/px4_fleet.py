"""PX4 SITL fleet in AirSim, flown in OFFBOARD mode over MAVLink.

Links: PX4 instance i runs an extra MAVLink instance started by px4_stack.sh (UDP 14700+i on the PX4 side, sending to
the bridge on 14800+i, custom mode with the streams listed there). Commands and setpoints use LOCAL_NED.

Time: AirSim is paused between bridge steps and advanced with simContinueForTime(dt) (PX4 runs in lockstep, so its
clock stops with AirSim's). Setpoints are re-sent before each sub-step, i.e. at 1/dt Hz of simulated time.
With --free-run the simulator is not paused; the bridge then paces itself on the AirSim SIMULATION clock (vehicle
state timestamp), so a 1 s round is 1 s of simulated physics whatever the bridge's own overhead. If the bridge falls
behind (a sub-step starts after its scheduled simulated time), the overrun is counted in pace_stats(); lower the
AirSim ClockSpeed if overruns are frequent.
on_step(fleet, phase), if set, is called after every sub-step (safety logging at the setpoint rate).

Frames: world = AirSim NED relative to PlayerStart. PX4 local NED of vehicle i has its origin where the vehicle booted;
offset_i = world position at rest (settings spawn + AirSim pose) - PX4 local position at rest, so
world = local + offset_i.
"""
import socket
import threading
import time

import numpy as np
from pymavlink import mavutil

BRIDGE_PORT = 14800
TYPE_MASK_POS_VEL = 448 | 1024 | 2048          # ignore acceleration, yaw, yaw rate
PX4_OFFBOARD = 6
MODE_NAMES = {(6, 0): 'OFFBOARD', (4, 2): 'TAKEOFF', (4, 3): 'LOITER', (4, 6): 'LAND', (3, 0): 'POSCTL',
              (2, 0): 'ALTCTL', (1, 0): 'MANUAL', (4, 5): 'RTL', (4, 4): 'MISSION'}


class Link:
    def __init__(self, i):
        self.i = i
        self.m = mavutil.mavlink_connection(f'udpin:0.0.0.0:{BRIDGE_PORT + i}', source_system=255,
                                            source_component=190, dialect='common')
        self.sys = None
        self.hb = None              # (armed, main, sub, wall)
        self.lpos = None            # (t_boot_ms, x, y, z, vx, vy, vz)
        self.bat = None
        self.landed = None
        self.acks = {}
        self.texts = []
        self.alive = True
        threading.Thread(target=self._rx, daemon=True).start()

    def _rx(self):
        while self.alive:
            try:
                msg = self.m.recv_match(blocking=True, timeout=0.5)
            except (OSError, socket.error):
                continue
            if msg is None or msg.get_type() == 'BAD_DATA':
                continue
            t = msg.get_type()
            if self.sys is None and msg.get_srcComponent() == 1:
                self.sys = msg.get_srcSystem()
            if t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                self.hb = (bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
                           (msg.custom_mode >> 16) & 0xFF, (msg.custom_mode >> 24) & 0xFF, time.time())
            elif t == 'LOCAL_POSITION_NED':
                self.lpos = (msg.time_boot_ms, msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz)
            elif t == 'BATTERY_STATUS':
                self.bat = msg.battery_remaining
            elif t == 'EXTENDED_SYS_STATE':
                self.landed = msg.landed_state
            elif t == 'COMMAND_ACK':
                self.acks[msg.command] = msg.result
            elif t == 'STATUSTEXT':
                self.texts.append(msg.text)

    def heartbeat(self):
        self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def send_command(self, cmd, *p):
        p = list(p) + [0] * (7 - len(p))
        self.acks.pop(cmd, None)
        self.m.mav.command_long_send(self.sys or 1, 1, cmd, 0, *p)

    def setpoint(self, pos, vel):
        self.m.mav.set_position_target_local_ned_send(
            0, self.sys or 1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_POS_VEL,
            float(pos[0]), float(pos[1]), float(pos[2]), float(vel[0]), float(vel[1]), float(vel[2]), 0, 0, 0, 0, 0)

    def mode(self):
        return None if not self.hb else MODE_NAMES.get((self.hb[1], self.hb[2]), f'{self.hb[1]}.{self.hb[2]}')

    def armed(self):
        return bool(self.hb and self.hb[0])

    def local(self):
        return None if self.lpos is None else np.array(self.lpos[1:4])

    def vel(self):
        return None if self.lpos is None else np.array(self.lpos[4:7])


class Fleet:
    def __init__(self, n, names, spawn_ne, client=None, dt=0.2, free_run=False, log=print):
        self.n, self.names = n, names
        self.spawn = np.array([[x, y, 0.0] for x, y in spawn_ne], float)
        self.c, self.dt, self.free_run, self.log = client, dt, free_run, log
        self.links = [Link(i) for i in range(n)]
        self.offset = np.zeros((n, 3))
        self.sp_local = [None] * n            # last setpoint (local position, velocity)
        self.t_sim = 0.0
        self.on_step, self.phase = None, 'startup'
        self._ts0 = None                      # AirSim sim-clock origin (ns) for free-run pacing
        self._over, self._over_max, self._nsteps = 0, 0.0, 0

    def sim_ns(self):
        return int(self.c.getMultirotorState(vehicle_name=self.names[0]).timestamp)

    def pace_stats(self):
        return dict(free_run=bool(self.free_run), steps=self._nsteps, overruns=self._over,
                    overrun_frac=self._over / max(1, self._nsteps), overrun_max_s=round(self._over_max, 3))

    # ---- time
    def step(self, dt=None):
        """Advance simulated time by dt, re-sending every vehicle's setpoint first."""
        dt = self.dt if dt is None else dt
        for i, L in enumerate(self.links):
            L.heartbeat()
            if self.sp_local[i] is not None:
                L.setpoint(*self.sp_local[i])
        if self.c is None:
            time.sleep(dt)
        elif self.free_run:                   # pace on the simulation clock
            if self._ts0 is None:
                self._ts0 = self.sim_ns() - int(self.t_sim * 1e9)
            target = self._ts0 + int((self.t_sim + dt) * 1e9)
            now = self.sim_ns()
            late = (now - (target - int(dt * 1e9))) / 1e9      # how late this sub-step started
            if late > 0.25 * dt:
                self._over += 1
                self._over_max = max(self._over_max, late)
            t0 = time.time()
            while now < target:
                time.sleep(min(0.005, max(0.0005, (target - now) / 2e9)))
                now = self.sim_ns()
                if time.time() - t0 > 30 + 20 * dt:
                    raise RuntimeError('AirSim simulation clock did not advance (paused?)')
        else:
            self.c.simContinueForTime(dt)
            t0 = time.time()
            while not self.c.simIsPause():
                if time.time() - t0 > 30 + 20 * dt:
                    raise RuntimeError('AirSim did not pause after simContinueForTime')
                time.sleep(0.002)
        time.sleep(0.01)                      # let the MAVLink threads take the last messages
        self.t_sim += dt
        self._nsteps += 1
        if self.on_step is not None:
            self.on_step(self, self.phase)

    def run_for(self, seconds, until=None):
        k = 0
        while k * self.dt < seconds:
            self.step()
            k += 1
            if until is not None and until():
                return True
        return until() if until is not None else True

    # ---- start-up
    def wait_ready(self, timeout_sim=120.0):
        """Heartbeat and a local position from every vehicle (EKF initialised)."""
        ok = self.run_for(timeout_sim, until=lambda: all(L.hb and L.lpos for L in self.links))
        if not ok:
            raise RuntimeError('PX4 not ready: ' + str([(L.i, bool(L.hb), bool(L.lpos)) for L in self.links]))
        self.run_for(10.0)                    # let the estimators settle at rest
        for i, L in enumerate(self.links):
            gt = self.gt_local(i)
            w = self.spawn[i] + (gt if gt is not None else 0.0)
            self.offset[i] = w - L.local()
        self.log(f'PX4 ready; offsets (m) {np.round(self.offset, 2).tolist()}')

    def gt_local(self, i):
        """AirSim ground truth position relative to the vehicle's spawn point (NED)."""
        if self.c is None:
            return None
        p = self.c.simGetGroundTruthKinematics(self.names[i]).position
        return np.array([p.x_val, p.y_val, p.z_val])

    def gt_world(self):
        if self.c is None:
            return self.world()
        return np.array([self.spawn[i] + self.gt_local(i) for i in range(self.n)])

    def command_all(self, cmd, *p, timeout_sim=10.0):
        for L in self.links:
            L.send_command(cmd, *p)
        self.run_for(timeout_sim, until=lambda: all(cmd in L.acks for L in self.links))
        return [L.acks.get(cmd) for L in self.links]

    def takeoff(self, alt_world_d, start_ne, v_climb=3.0):
        """Arm in OFFBOARD at the pad, climb vertically to world D alt_world_d[i], then fly to start_ne[i] (world)."""
        for i, L in enumerate(self.links):
            p = L.local()
            self.sp_local[i] = (p.copy(), np.zeros(3))
        self.run_for(1.0)                     # setpoints must stream before OFFBOARD is accepted
        acks = self.command_all(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, PX4_OFFBOARD, 0)
        self.log(f'OFFBOARD acks {acks}')
        acks = self.command_all(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        self.log(f'ARM acks {acks}')
        ready = lambda: all(L.armed() and L.mode() == 'OFFBOARD' for L in self.links)
        if not self.run_for(5.0, until=ready):          # heartbeats arrive at 2 Hz: wait for fresh ones
            self.log('not all armed in OFFBOARD yet, re-sending mode and arm')
            self.command_all(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                             PX4_OFFBOARD, 0)
            self.command_all(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.run_for(5.0, until=ready)
        if not ready():
            raise RuntimeError('arming failed: ' + str([(L.i, L.mode(), L.armed(), L.texts[-3:]) for L in self.links]))
        pads = self.world()
        top = pads.copy()
        top[:, 2] = alt_world_d
        self.fly_lines(pads, top, v=v_climb)
        dest = top.copy()
        dest[:, :2] = start_ne
        self.fly_lines(top, dest, v=10.0)

    def fly_lines(self, a, b, v, tol=2.0, extra_s=60.0):
        """Straight world lines a[i] -> b[i] at speed v (moving setpoint), then wait until within tol of b."""
        span = np.linalg.norm(b - a, axis=1)
        T = float(span.max() / v) if span.max() > 0 else 0.0
        k = 0
        while k * self.dt < T:
            f = np.minimum(1.0, (k + 1) * self.dt * v / np.maximum(span, 1e-6))
            for i in range(self.n):
                p = a[i] + f[i] * (b[i] - a[i])
                u = (b[i] - a[i]) / max(span[i], 1e-6) * (v if f[i] < 1 else 0.0)
                self.set_world(i, p, u)
            self.step()
            k += 1
        for i in range(self.n):
            self.set_world(i, b[i], np.zeros(3))
        ok = self.run_for(extra_s, until=lambda: np.all(np.linalg.norm(self.world() - b, axis=1) < tol))
        if not ok:
            self.log(f'fly_lines: not all within {tol} m: {np.round(np.linalg.norm(self.world() - b, axis=1), 1)}')

    # ---- mission
    def set_world(self, i, pos_w, vel_w):
        self.sp_local[i] = (np.asarray(pos_w, float) - self.offset[i], np.asarray(vel_w, float))

    def world(self):
        return np.array([L.local() + self.offset[i] for i, L in enumerate(self.links)])

    def status(self):
        return [dict(mode=L.mode(), armed=L.armed(), bat=L.bat, landed=L.landed) for L in self.links]

    def land(self, timeout_sim=120.0):
        for L in self.links:
            L.send_command(mavutil.mavlink.MAV_CMD_NAV_LAND)
        self.sp_local = [None] * self.n
        self.run_for(timeout_sim, until=lambda: all(L.landed == 1 for L in self.links))

    def close(self):
        for L in self.links:
            L.alive = False

