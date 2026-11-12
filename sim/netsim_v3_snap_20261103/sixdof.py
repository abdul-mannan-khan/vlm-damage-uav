"""6-DOF flight for the network simulator (realistic model v3, NetConfig.flight = 'sixdof').

Each UAV is the 6-DOF quadrotor of wind-aware-delivery v3.0 (third_party/wad_quadrotor.py: first-order motors, rotor and
body drag, momentum-theory power with blade profile power, geometric SE(3) controller), re-parameterised to a 6 kg
UAV base station. The fleet logic of World.move_uavs (targets, separation on GNSS fixes, acceleration limit,
take-off, tower-climb, battery and swaps) produces the reference path of each 1 s round; the quadrotor tracks that
path at `hz` with a smooth reference (position, velocity, acceleration interpolated over the round). The UAV's true
position and height (World.upos, World.ualt) become the quadrotor's, and the round's energy becomes its electrical
energy plus the communication power. No wind (the network study has none).

    run_episode(cfg, method, seed, move=SixDofFleet(cfg))       # or cfg.flight = 'sixdof' (episode.py does this)
"""
import numpy as np
from third_party.wad_quadrotor import Params, Quadrotor, GeometricController, rotor_power, vee, G


class Params6kg(Params):
    """6 kg quadrotor base station (Matrice-300 class): 21-inch propellers, 0.45 m arms, thrust-to-weight 2."""
    m_frame = 6.0
    m_payload = 0.0
    J = np.diag([0.15, 0.15, 0.24])          # 4 x 0.25 kg motors 0.32 m from the roll/pitch axes, 0.45 m from yaw + body
    k_m = 0.032                               # yaw torque / thrust (m), about twice the 10-inch value for 21-inch propellers
    arm = 0.45
    T_max = 29.5                              # N per rotor (4 x 29.5 = 2 x weight)
    tau_m = 0.05
    r_prop = 0.267
    d_rotor = np.array([0.50, 0.50, 0.09])   # kg/s, linear rotor drag (1.7 x the 10-inch value)
    CdA = 0.12                                # m^2, frontal area ~0.25 m^2 x Cd ~0.5
    CdA_empty = 0.12
    max_tilt_deg = 35.0                       # payload-carrying UAV-BS tilt limit
    P_avionics = 15.0


class Controller6kg(GeometricController):
    """Geometric controller retuned for the 6 kg UAV (fix of 2 Nov 2026).
    The vendored controller scales its attitude gains as J / 0.03 (a 0.03 kg m^2 drone); at J = 0.15 it asks for about
    30 N m per rad while the rotors give at most about 9 N m, so in sharp turns at speed single motors clipped at 0 or
    T_max, the body rates oscillated and 1.9 % of the city missions diverged. Here the attitude loop has its own
    bandwidth (omega 8 rad/s, damping 0.8, well below the 20 rad/s motor pole) and the mixer keeps the collective
    thrust and scales the moments down to fit the motor limits (yaw first, then roll/pitch) instead of clipping
    motors one by one."""
    w_att, z_att = 8.0, 0.8

    def __call__(self, dt, p_d, v_d, a_d):
        qd = self.quad
        m = qd.mass
        ep = qd.p - p_d
        ev = qd.v - v_d
        self.ei = np.clip(self.ei + ep * dt, -3.0, 3.0)
        F = m * (-self.kp * ep - self.kv * ev - self.ki * self.ei + G * np.array([0.0, 0.0, 1.0]) + a_d)
        Fh = np.linalg.norm(F[:2])
        Fz = max(F[2], 0.3 * m * G)
        if Fh > Fz * np.tan(self.max_tilt):
            F[:2] *= Fz * np.tan(self.max_tilt) / Fh
        F[2] = Fz
        R = qd.R
        b3 = F / np.linalg.norm(F)
        c1 = np.array([np.cos(self.yaw_d), np.sin(self.yaw_d), 0.0])
        b2 = np.cross(b3, c1)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        Jd = np.diag(qd.P.J)
        M = Jd * (-self.w_att ** 2 * eR - 2 * self.z_att * self.w_att * qd.w) + np.cross(qd.w, qd.P.J @ qd.w)
        f = float(np.clip(F @ R @ np.array([0.0, 0.0, 1.0]), 0.05 * 4 * qd.P.T_max, 0.95 * 4 * qd.P.T_max))
        T0 = qd.Mix_inv @ np.array([f, 0.0, 0.0, 0.0])          # collective only
        Ty = qd.Mix_inv @ np.array([0.0, 0.0, 0.0, M[2]])
        Trp = qd.Mix_inv @ np.array([0.0, M[0], M[1], 0.0])
        a_rp = _fit(T0, Trp, qd.P.T_max)                          # roll/pitch first
        a_y = _fit(T0 + a_rp * Trp, Ty, qd.P.T_max)               # yaw with what is left
        T_cmd = np.clip(T0 + a_rp * Trp + a_y * Ty, 0.0, qd.P.T_max)
        sat = bool(a_rp < 1.0 - 1e-9)
        return T_cmd, sat


def _fit(base, d, tmax):
    """Largest a in [0, 1] with 0 <= base + a d <= tmax (base assumed inside)."""
    a = 1.0
    for b, x in zip(base, d):
        if x > 1e-12:
            a = min(a, (tmax - b) / x)
        elif x < -1e-12:
            a = min(a, (0.0 - b) / x)
    return max(0.0, a)


class SixDofFleet:
    def __init__(self, cfg, hz=50):
        self.cfg, self.dt, self.k = cfg, 1.0 / hz, hz
        self.P = Params6kg()
        self.q = None
        self.stats = dict(track_err=[], ref_lag=[], sat_steps=0, steps=0)

    def _init(self, W):
        self.q = [Quadrotor(self.P, (W.upos[i, 0], W.upos[i, 1], W.ualt[i])) for i in range(W.cfg.n_uav)]
        self.c = [Controller6kg(q) for q in self.q]
        for c in self.c:
            c.max_tilt = np.deg2rad(self.P.max_tilt_deg)
        self.v_ref = np.zeros((W.cfg.n_uav, 3))
        self.pr = np.c_[W.upos, W.ualt].astype(float).copy()   # filtered reference position / velocity
        self.vr = np.zeros((W.cfg.n_uav, 3))
        self.z_cmd = W.ualt.copy()                   # commanded height (the fleet logic's), kept apart from the true one

    def __call__(self, W, targets, sep, t):
        if self.q is None:
            self._init(W)
        n = W.cfg.n_uav
        W.ualt[:] = self.z_cmd                        # the fleet logic plans from the commanded height
        p0 = np.c_[W.upos, W.ualt].copy()
        e0 = W.energy.copy()
        W.move_uavs(targets, sep, t)                  # reference path, modes, battery logic
        p1 = np.c_[W.upos, W.ualt].copy()
        self.z_cmd = W.ualt.copy()
        dE_kin = e0 - W.energy                        # kinematic energy of the round, replaced below
        v1 = p1 - p0                                  # m/s over the 1 s round
        E = np.zeros(n)
        for i in range(n):
            q, c = self.q[i], self.c[i]
            if W.mode[i] == 2 or not W.alive[i]:      # landed for a swap / failed: on the pad, motors idle
                q.p, q.v, q.w = p1[i].copy(), np.zeros(3), np.zeros(3)
                q.T = np.full(4, q.mass * G / 4)
                self.v_ref[i] = 0.0
                self.pr[i], self.vr[i] = p1[i].copy(), np.zeros(3)
                continue
            # reference filter: second order (omega 1.5 rad/s, critically damped) towards the fleet logic's path,
            # acceleration capped (4 m/s^2 horizontal, 2 m/s^2 vertical), so the reference is always flyable
            pr, vr = self.pr[i], self.vr[i]
            for k in range(self.k):
                s = (k + 1) / self.k
                pk = p0[i] + v1[i] * s
                a = 2.25 * (pk - pr) + 3.0 * (v1[i] - vr)
                ah = np.hypot(a[0], a[1])
                if ah > 4.0:
                    a[:2] *= 4.0 / ah
                a[2] = np.clip(a[2], -2.0, 2.0)
                vr = vr + a * self.dt
                sp = np.hypot(vr[0], vr[1])
                if sp > W.cfg.speed:
                    vr[:2] *= W.cfg.speed / sp
                vr[2] = np.clip(vr[2], -W.cfg.v_climb, W.cfg.v_climb)
                pr = pr + vr * self.dt
                T, sat = c(self.dt, pr, vr, a)
                q.step(self.dt, T, np.zeros(3))
                E[i] += rotor_power(q.T, q.v, q.R @ np.array([0.0, 0.0, 1.0]), self.P) * self.dt
                self.stats['sat_steps'] += int(sat)
                self.stats['steps'] += 1
            self.pr[i], self.vr[i] = pr, vr
            self.stats['ref_lag'].append(float(np.linalg.norm(pr - p1[i])))
            self.v_ref[i] = v1[i]
            self.stats['track_err'].append(float(np.linalg.norm(q.p - self.pr[i])))
            W.upos[i] = np.clip(q.p[:2], 0, W.cfg.world)
            W.ualt[i] = max(0.0, float(q.p[2]))
            E[i] += W.cfg.p_comm_w * 1.0
            W.energy[i] += dE_kin[i] - E[i]           # electrical energy of the 6-DOF flight instead of the model's
            W.m['energy_j'] += E[i] - dE_kin[i]

    def summary(self):
        e = np.array(self.stats['track_err']) if self.stats['track_err'] else np.zeros(1)
        g = np.array(self.stats['ref_lag']) if self.stats['ref_lag'] else np.zeros(1)
        return dict(track_err_mean=float(e.mean()), track_err_p95=float(np.percentile(e, 95)),
                    ref_lag_mean=float(g.mean()), ref_lag_p95=float(np.percentile(g, 95)),
                    tilt_sat_share=self.stats['sat_steps'] / max(1, self.stats['steps']))
