"""[VLM paper copy, 12 Nov 2026: only the netsim search path changed; source an earlier MAQAC codebase hil_link.py]
PX4 HITL link for one UAV of the MAQAC fleet: 6-DOF physics + sensors on the companion computer, setpoint server.

Source: the classes Sim and Link and the 250 Hz loop are copied from Paper A's HIL
(wind-aware-delivery, hil/hil_run.py, as deployed on the Jetson in ~/paperA_hil_v4h/hil/, 2 Nov 2026) with these
changes: no wind field and no gusts, no payload drop, no city/clearance logic; the mission logic is replaced by a
setpoint server (hold/ramp, tracked segments, landing) driven by the network simulator through a pipe. The quadrotor
model is src/netsim/third_party/wad_quadrotor.py (vendored unchanged from the same repository, sim/quadrotor.py).

The server runs in its own process, so the network simulator's decisions never block the 250 Hz sensor loop.
  backend 'px4' : Pixhawk on a serial port, PX4 in HITL (SYS_HITL=1, SYS_AUTOSTART=1001); HIL_SENSOR 250 Hz,
                  HIL_GPS 10 Hz, HIL_ACTUATOR_CONTROLS -> motors, OFFBOARD position+velocity(+acc) setpoints at 50 Hz.
  backend 'router': physics and sensors from AirSim on the laptop (CityEnviron, PX4Multirotor over TCP 4560) through
                  W2's p27_router.py on the Jetson (~/lit_results/x500_airsim_hitl_bridge/); this process only sends
                  OFFBOARD setpoints to the router's UDP port and reads PX4's own estimate (LOCAL_POSITION_NED), so the
                  position used by the network model is what the autopilot believes, not a simulator truth.
                  p0 is then the area point of AirSim's spawn pad (PX4's local origin).
  backend 'fake': no hardware; first-order autopilot model (dry run of the same code path).
Frames: simulator ENU (x = network-model area x, y = area y, z up), body FLU; PX4 NED/FRD.
"""
import math, os, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', 'netsim'), os.path.join(HERE, 'netsim')):     # VLM paper: sim/airsim_vlm/netsim snapshot (or ./netsim on the Jetson)
    if os.path.isdir(p):
        sys.path.insert(0, p)
from third_party.wad_quadrotor import G, Params, Quadrotor   # noqa: E402

C = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])     # ENU <-> NED
D = np.diag([1.0, -1, -1])                             # FLU <-> FRD
LAT0, LON0, ALT0 = 51.5074, -0.1278, 20.0              # geodetic origin of area (0, 0)
R_EARTH = 6378137.0
MAG_B, MAG_I, MAG_D = 0.489, np.radians(66.3), np.radians(1.0)
M_NED = MAG_B * np.array([np.cos(MAG_I) * np.cos(MAG_D), np.cos(MAG_I) * np.sin(MAG_D), np.sin(MAG_I)])
M_ENU = C @ M_NED
YAW_NED = np.pi / 2
DT_PHYS = 0.004                                        # 250 Hz physics and HIL_SENSOR
RAMP_H, RAMP_UP, RAMP_DN = 8.0, 2.5, 1.5               # m/s, hold/transit setpoint ramp (not used in the mission)
A_FF_MAX = 4.0                                         # m/s^2, clip on the acceleration feed-forward

# shared state layout (doubles)
S_T, S_P, S_V, S_SP, S_ARMED, S_MODE, S_SUB, S_READY, S_PHASE, S_NSP = 0, 1, 4, 7, 10, 11, 12, 13, 14, 15
S_LEN = 16
PHASES = ['boot', 'ready', 'hold', 'track', 'land', 'landed', 'error']


def hermite(seg, t):
    """Cubic Hermite segment seg = (ta, tb, pa, va, pb, vb) evaluated at time t: position, velocity, acceleration.
    After tb: constant-velocity extrapolation for up to 1 s, then hold."""
    ta, tb, pa, va, pb, vb = seg
    h = tb - ta
    if t >= tb:
        dt = min(t - tb, 1.0)
        return pb + vb * dt, (vb.copy() if t - tb < 1.0 else np.zeros(3)), np.zeros(3)
    s = max(0.0, (t - ta) / h)
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    p = h00 * pa + h10 * h * va + h01 * pb + h11 * h * vb
    d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    v = (d00 * pa + d10 * h * va + d01 * pb + d11 * h * vb) / h
    e00, e10, e01, e11 = 12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2
    a = (e00 * pa + e10 * h * va + e01 * pb + e11 * h * vb) / h**2
    return p, v, a


class Sim:
    """Physics + sensors (copied from Paper A hil_run.py, wind removed). World ENU, body FLU."""

    def __init__(self, P, p0, rng):
        self.P, self.rng = P, rng
        self.quad = Quadrotor(P, p0)
        km, L = P.k_m, P.arm / np.sqrt(2)
        self.quad.Mix = np.array([[1, 1, 1, 1], [-L, L, L, -L], [-L, L, -L, L], [-km, -km, km, km]])
        self.quad.T = np.zeros(4)
        self.u = np.zeros(4)
        self.acc = np.zeros(3)
        self.gyro_bias = rng.normal(0, 0.002, 3)
        self.gps_err = np.zeros(3)
        self.w0 = np.zeros(3)

    def step(self, dt):
        q = self.quad
        airborne = q.p[2] > 0.05 or self.u.sum() * q.P.T_max > 0.9 * q.mass * G
        v0 = q.v.copy()
        T_cmd = np.clip(self.u, 0, 1) * q.P.T_max
        if airborne:
            q.step(dt, T_cmd, self.w0)
        else:
            q.T = q.T + (T_cmd - q.T) * min(dt / q.P.tau_m, 1.0)
        if q.p[2] < 0.0:
            yaw = np.arctan2(q.R[1, 0], q.R[0, 0])
            q.p[2], q.v[:], q.w[:] = 0.0, 0.0, 0.0
            q.q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.acc = (q.v - v0) / dt

    def hil_sensor(self, mav, t_us):
        q, rng = self.quad, self.rng
        R = q.R
        f_flu = R.T @ (self.acc + G * np.array([0, 0, 1.0]))
        acc = D @ f_flu + rng.normal(0, 0.03, 3)
        gyro = D @ q.w + self.gyro_bias + rng.normal(0, 0.003, 3)
        mag = D @ (R.T @ M_ENU) + rng.normal(0, 0.003, 3)
        alt = ALT0 + q.p[2] + rng.normal(0, 0.05)
        p_hpa = 1013.25 * (1 - 2.25577e-5 * alt) ** 5.25588
        mav.hil_sensor_send(t_us, *acc, *gyro, *mag, p_hpa, 0.0, alt, 15.0, 0x1FFF)

    def hil_gps(self, mav, t_us):
        q = self.quad
        self.gps_err = 0.98 * self.gps_err + self.rng.normal(0, 0.03, 3)
        n, e, d = C @ q.p + self.gps_err
        lat = LAT0 + np.degrees(n / R_EARTH)
        lon = LON0 + np.degrees(e / (R_EARTH * np.cos(np.radians(LAT0))))
        vn, ve, vd = C @ q.v
        cog = (np.degrees(np.arctan2(ve, vn)) + 360) % 360
        mav.hil_gps_send(t_us, 3, int(lat * 1e7), int(lon * 1e7), int((ALT0 - d) * 1000), 30, 40,
                         int(np.hypot(vn, ve) * 100), int(vn * 100), int(ve * 100), int(vd * 100),
                         int(cog * 100), 14)


class Link:
    """MAVLink link to PX4 (copied from Paper A hil_run.py; TIMESYNC round-trip, mode-drop and failsafe counters added)."""

    def __init__(self, dev, origin):
        from pymavlink import mavutil
        self.mu = mavutil
        self.origin = np.asarray(origin, float)
        self.m = mavutil.mavlink_connection(dev, baud=921600, source_system=255, source_component=190)
        self.mode = self.armed = None
        self.local = None
        self.local_v = np.zeros(3)
        self.status = []
        self.n_act = 0
        self.rtt = []
        self.ts_sent = {}

    def poll(self, sim, t_now):
        mavutil = self.mu
        while True:
            msg = self.m.recv_msg()
            if msg is None:
                return
            t = msg.get_type()
            if t == 'HIL_ACTUATOR_CONTROLS':
                if sim is not None:
                    sim.u = np.clip(np.array(msg.controls[:4]), 0.0, 1.0)
                self.n_act += 1
            elif t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = (msg.custom_mode >> 16) & 0xFF, (msg.custom_mode >> 24) & 0xFF
            elif t == 'LOCAL_POSITION_NED':
                self.local = np.array([msg.x, msg.y, msg.z])
                self.local_v = np.array([msg.vx, msg.vy, msg.vz])
            elif t == 'STATUSTEXT':
                self.status.append((round(t_now, 2), msg.text))
                print(f'  PX4: {msg.text}', flush=True)
            elif t == 'TIMESYNC' and msg.tc1 != 0:
                t0 = self.ts_sent.pop(msg.ts1, None)
                if t0 is not None:
                    self.rtt.append(time.monotonic() - t0)

    def heartbeat(self):
        self.m.mav.heartbeat_send(self.mu.mavlink.MAV_TYPE_GCS, self.mu.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def timesync(self):
        ts = time.monotonic_ns()
        self.ts_sent[ts] = time.monotonic()
        if len(self.ts_sent) > 20:
            self.ts_sent.pop(next(iter(self.ts_sent)))
        self.m.mav.timesync_send(0, ts)

    def cmd(self, c, *p):
        p = list(p) + [0] * (7 - len(p))
        self.m.mav.command_long_send(1, 1, c, 0, *p)

    def setpoint(self, p_enu, v_enu, a_enu):
        p, v, a = C @ (p_enu - self.origin), C @ v_enu, C @ a_enu
        mask = 0b0000100000000000                    # position, velocity, acceleration, yaw (ignore yaw rate)
        self.m.mav.set_position_target_local_ned_send(0, 1, 1, self.mu.mavlink.MAV_FRAME_LOCAL_NED, mask,
                                                      *p, *v, *a, YAW_NED, 0)


class FakeAutopilot:
    """Dry run: first-order position autopilot with velocity and acceleration limits (no PX4, no 6-DOF)."""

    def __init__(self, p0, tau=1.0, vmax=15.0, amax=4.0):
        self.p, self.v = np.array(p0, float), np.zeros(3)
        self.tau, self.vmax, self.amax = tau, vmax, amax
        self.armed, self.mode = False, (4, 3)

    def step(self, dt, sp):
        if sp is None or not self.armed:
            return
        p_sp, v_sp, _ = sp
        vc = v_sp + (p_sp - self.p) / self.tau
        s = np.linalg.norm(vc)
        if s > self.vmax:
            vc *= self.vmax / s
        dv = vc - self.v
        a = np.linalg.norm(dv) / dt
        if a > self.amax:
            dv *= self.amax / a
        self.v += dv
        self.p += self.v * dt
        if self.p[2] < 0:
            self.p[2], self.v[2] = 0.0, max(0.0, self.v[2])


def serve(conn, shared, backend, dev, p0, warmup, verbose=False):
    """HIL server process. Commands on conn:
       ('hold', p)                    ramp the setpoint to p (arming and OFFBOARD if needed), then hold
       ('seg', ta, tb, pa, va, pb, vb) track a Hermite segment (monotonic times)
       ('mark',)                      start a statistics window      ('stats',) -> dict for the window
       ('land',)                      descend, land, disarm          ('stop',)"""
    p0 = np.asarray(p0, float)
    rng = np.random.default_rng(1234)
    px4 = backend in ('px4', 'router')         # a real PX4 on the other end of the link
    own_phys = backend == 'px4'                # physics + HIL_SENSOR/HIL_GPS computed here
    if px4:
        sim = Sim(Params(), p0, rng) if own_phys else None
        link = Link(dev, p0)
        mv = link.mu.mavlink
        for msg_id, hz in ((32, 20), (0, 2)):            # LOCAL_POSITION_NED, HEARTBEAT
            link.cmd(mv.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
    else:
        sim, link = None, None
        fake = FakeAutopilot(p0)

    def state():
        if own_phys:
            return sim.quad.p, sim.quad.v, link.armed, link.mode
        if px4:
            p = C @ link.local + p0 if link.local is not None else p0.copy()
            return p, C @ link.local_v, link.armed, link.mode
        return fake.p, fake.v, fake.armed, fake.mode

    phase, t_phase = 'boot', time.monotonic()
    target = None                    # hold target
    carrot = None                    # (p, v) of the ramping setpoint
    seg = None
    sp = None                        # last setpoint (p, v, a)
    st = None                        # statistics window

    def new_stats():
        return dict(t0=time.monotonic(), n_sp=0, n_act0=link.n_act if px4 else 0, n_sensor=0, n_gps=0, err=[],
                    mode_drops=0, disarms=0, status0=len(link.status) if px4 else 0, rtt0=len(link.rtt) if px4 else 0,
                    loop_late=0, max_late=0.0, track_s=0.0)

    def enter(ph):
        nonlocal phase, t_phase
        if ph != phase:
            print(f'{time.strftime("%T")} HIL phase {phase} -> {ph}', flush=True)
        phase, t_phase = ph, time.monotonic()

    t_start = time.monotonic()
    t_next = t_start
    k = 0
    prev_mode = None
    try:
        while True:
            now = time.monotonic()
            if now < t_next:
                time.sleep(t_next - now)
            late = time.monotonic() - t_next
            t_next += DT_PHYS
            if late > 0.1:
                t_next = time.monotonic()
            if st is not None and late > 0.008:
                st['loop_late'] += 1
                st['max_late'] = max(st['max_late'], late)
            tm = time.monotonic()
            t_us = int((tm - t_start) * 1e6)
            if px4:
                link.poll(sim, tm - t_start)
                if own_phys:
                    sim.step(DT_PHYS)
                    sim.hil_sensor(link.m.mav, t_us)
                    if st is not None:
                        st['n_sensor'] += 1
                    if k % 25 == 0:
                        sim.hil_gps(link.m.mav, t_us)
                        if st is not None:
                            st['n_gps'] += 1
                if k % 125 == 0:
                    link.heartbeat()
                if k % 250 == 0:
                    link.timesync()
            k += 1
            if k % 5:
                continue
            # ---------------- 50 Hz: commands, setpoints, state -------------------------------------------------
            dt = 5 * DT_PHYS
            if not px4:
                fake.step(dt, sp if phase in ('hold', 'track', 'land') else None)
            while conn.poll():
                c = conn.recv()
                if c[0] == 'hold':
                    target = np.asarray(c[1], float)
                    if carrot is None:
                        carrot = (sp[0].copy(), np.zeros(3)) if sp is not None else (state()[0].copy(), np.zeros(3))
                    if phase in ('ready', 'landed', 'track', 'hold'):
                        enter('hold')
                elif c[0] == 'seg':
                    seg = (c[1], c[2], *[np.asarray(x, float) for x in c[3:7]])
                    carrot = None
                    enter('track')
                elif c[0] == 'mark':
                    st = new_stats()
                elif c[0] == 'stats':
                    conn.send(window_stats(st, link, px4))
                elif c[0] == 'land':
                    carrot = (sp[0].copy(), np.zeros(3)) if sp is not None else (state()[0].copy(), np.zeros(3))
                    p = state()[0]
                    target = np.array([p[0], p[1], 0.0])
                    enter('land')
                elif c[0] == 'stop':
                    return
            p, v, armed, mode = state()
            el = time.monotonic() - t_phase
            if phase == 'boot':
                if px4:
                    if el > warmup and link.local is not None and not own_phys:
                        enter('ready')       # AirSim mode: no truth here to compare with; the estimate is the state
                    elif el > warmup and link.local is not None:
                        est = C @ link.local + p0
                        if np.linalg.norm(est[:2] - p[:2]) < 1.0:
                            enter('ready')
                        elif el > warmup + 60:
                            print(f'PX4 estimate {np.round(est, 1)} not at the start {np.round(p, 1)}: reboot the FC', flush=True)
                            enter('error')
                else:
                    enter('ready')
            if phase in ('hold', 'land') and carrot is not None:
                cp, _ = carrot
                d = target - cp
                dh, dz = d[:2], d[2]
                nh = np.linalg.norm(dh)
                vh = dh / nh * min(RAMP_H, nh / dt) if nh > 1e-9 else np.zeros(2)
                vz = float(np.clip(dz / dt, -RAMP_DN, RAMP_UP))
                vc = np.array([vh[0], vh[1], vz])
                carrot = (cp + vc * dt, vc)
                sp = (carrot[0].copy(), vc, np.zeros(3))
            elif phase == 'track' and seg is not None:
                pp, vv, aa = hermite(seg, time.monotonic())
                n_a = np.linalg.norm(aa)
                if n_a > A_FF_MAX:
                    aa = aa * A_FF_MAX / n_a
                sp = (pp, vv, aa)
                if st is not None:
                    st['err'].append(float(np.linalg.norm(p - pp)))
                    st['track_s'] += dt
            if sp is not None and phase in ('hold', 'track', 'land'):
                if px4:
                    link.setpoint(*sp)
                    if phase in ('hold', 'track'):
                        if mode != (6, 0) and int(el * 50) % 25 == 0 and el > 1.0:
                            link.cmd(mv.MAV_CMD_DO_SET_MODE, 1, 6)
                        if not armed and el > 1.5 and int(el * 50) % 50 == 0:
                            link.cmd(mv.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                    elif phase == 'land' and carrot is not None and p[2] < 1.5:
                        carrot = None
                else:
                    fake.armed, fake.mode = phase != 'landed', (6, 0)
                if st is not None:
                    st['n_sp'] += 1
            if phase == 'land' and carrot is None and px4:
                if armed:
                    if int(el * 50) % 100 == 0:
                        link.cmd(mv.MAV_CMD_NAV_LAND, 0, 0, 0, np.nan, np.nan, np.nan, np.nan)
                    if el > 90:
                        link.cmd(mv.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)
                else:
                    enter('landed')
            if phase == 'land' and not px4 and p[2] < 0.05:
                fake.armed = False
                enter('landed')
            # mode drops / disarms while flying a mission or holding in the air
            if st is not None and px4 and prev_mode is not None and phase in ('hold', 'track'):
                if prev_mode[0] and not armed and p[2] > 0.5:
                    st['disarms'] += 1
                if prev_mode[1] == (6, 0) and mode != (6, 0):
                    st['mode_drops'] += 1
                    print(f'  mode drop: {prev_mode[1]} -> {mode} at {np.round(p, 1)}', flush=True)
            prev_mode = (armed, mode)
            if verbose and k % 500 == 0:
                print(f'    {phase:6s} p={np.round(p, 2)} v={np.round(v, 2)} armed={armed} mode={mode} '
                      f'act/s={(link.n_act if px4 else 0)}', flush=True)
            with shared.get_lock():
                shared[S_T] = time.monotonic()
                shared[S_P:S_P + 3] = list(p)
                shared[S_V:S_V + 3] = list(v)
                if sp is not None:
                    shared[S_SP:S_SP + 3] = list(sp[0])
                shared[S_ARMED] = 1.0 if armed else 0.0
                shared[S_MODE] = mode[0] if mode else -1
                shared[S_SUB] = mode[1] if mode else -1
                shared[S_READY] = 1.0 if phase not in ('boot', 'error') else (-1.0 if phase == 'error' else 0.0)
                shared[S_PHASE] = PHASES.index(phase)
    finally:
        if px4:                                         # never leave PX4 armed
            for _ in range(5):
                link.cmd(mv.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)
                time.sleep(0.1)


def window_stats(st, link, px4):
    if st is None:
        return {}
    dur = time.monotonic() - st['t0']
    e = np.array(st['err'], float)
    out = dict(window_s=round(dur, 2), setpoint_hz=st['n_sp'] / dur, hil_sensor_hz=st['n_sensor'] / dur,
               hil_gps_hz=st['n_gps'] / dur, mode_drops=st['mode_drops'], disarms_in_air=st['disarms'],
               loop_late_4ms=st['loop_late'], loop_max_late_ms=1e3 * st['max_late'],
               sp_err_mean_m=float(e.mean()) if e.size else None,
               sp_err_p95_m=float(np.percentile(e, 95)) if e.size else None,
               sp_err_max_m=float(e.max()) if e.size else None)
    if px4:
        r = np.array(link.rtt[st['rtt0']:], float) * 1e3
        txt = [s for s in link.status[st['status0']:]]
        out.update(actuator_hz=(link.n_act - st['n_act0']) / dur,
                   rtt_ms_mean=float(r.mean()) if r.size else None,
                   rtt_ms_p95=float(np.percentile(r, 95)) if r.size else None,
                   rtt_ms_max=float(r.max()) if r.size else None, rtt_n=int(r.size),
                   statustext=txt, failsafe_msgs=[s for s in txt if 'ailsafe' in s[1] or 'FAILSAFE' in s[1]],
                   geofence_msgs=[s for s in txt if 'eofence' in s[1] or 'GEOFENCE' in s[1]])
    return out


class HilClient:
    """Main-process handle of the server."""

    def __init__(self, backend, dev, p0, warmup=40.0, verbose=False):
        ctx = mp.get_context('fork')
        self.shared = ctx.Array('d', S_LEN)
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=serve, args=(child, self.shared, backend, dev, p0, warmup, verbose), daemon=True)
        self.proc.start()
        self.backend = backend

    def state(self):
        with self.shared.get_lock():
            s = list(self.shared)
        return dict(t=s[S_T], p=np.array(s[S_P:S_P + 3]), v=np.array(s[S_V:S_V + 3]), sp=np.array(s[S_SP:S_SP + 3]),
                    armed=bool(s[S_ARMED]), mode=(int(s[S_MODE]), int(s[S_SUB])), ready=s[S_READY],
                    phase=PHASES[int(s[S_PHASE])])

    def wait_ready(self, timeout=300):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            s = self.state()
            if s['ready'] > 0:
                return True
            if s['ready'] < 0 or not self.proc.is_alive():
                return False
            time.sleep(0.5)
        return False

    def hold(self, p):
        self.conn.send(('hold', list(map(float, p))))

    def wait_at(self, p, tol=1.0, vtol=0.3, settle=3.0, timeout=400):
        """Wait until the vehicle is within tol of p, slower than vtol, armed and in OFFBOARD for `settle` s."""
        t, ok_since = time.monotonic(), None
        p = np.asarray(p, float)
        while time.monotonic() - t < timeout:
            s = self.state()
            ok = (np.linalg.norm(s['p'] - p) < tol and np.linalg.norm(s['v']) < vtol and s['armed']
                  and (self.backend == 'fake' or s['mode'] == (6, 0)))
            ok_since = (ok_since or time.monotonic()) if ok else None
            if ok_since and time.monotonic() - ok_since > settle:
                return True
            if not self.proc.is_alive():
                return False
            time.sleep(0.1)
        return False

    def seg(self, ta, tb, pa, va, pb, vb):
        self.conn.send(('seg', ta, tb, *[list(map(float, x)) for x in (pa, va, pb, vb)]))

    def mark(self):
        self.conn.send(('mark',))

    def stats(self):
        self.conn.send(('stats',))
        return self.conn.recv() if self.conn.poll(5) else {}

    def land(self, timeout=200):
        self.conn.send(('land',))
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if self.state()['phase'] == 'landed' or not self.proc.is_alive():
                return True
            time.sleep(0.5)
        return False

    def close(self):
        try:
            self.conn.send(('stop',))
        except Exception:
            pass
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.terminate()
