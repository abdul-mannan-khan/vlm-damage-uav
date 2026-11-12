#!/usr/bin/env python3
"""One MAQAC (or baseline) mission of the network-level simulator flown by PX4 SITL autopilots in AirSim CityEnviron.

The network model (users, radio access, backhaul, record sharing, decisions, metrics) is src/netsim unchanged; only the
UAV motion is replaced (run_episode(move=...)): each 1 s round, the kinematic step of World.move_uavs (speed limit,
separation rule and battery logic of every method) gives a moving reference that is sent to PX4 as an OFFBOARD
position + velocity setpoint at altitude ALT (+ LAYER m per vehicle); after 1 s of simulated time the PX4 positions
replace the kinematic ones, and the propulsion energy is recomputed from the distance actually flown.
The UAVs climb over towers taller than the flight altitude (Mover, surveyed ceiling map); the radio model uses each
UAV's true height above the local ground (World.ualt), while the planners keep the nominal altitude.
Users are spawned in AirSim as BicycleMan objects at their network-model positions, snapped to open ground with the
surveyed height map (survey_heightmap.py), and moved with the mobility model.

Outputs in OUT/: result.json (network metrics of the PX4 mission and of the kinematic twin with the same seed, flight
statistics), rounds.jsonl (per-round flight log), frames/ (optional per-UAV camera frames + labels), video/ (optional).

    # on the PX4 VM, after px4_stack.sh start N:
    python3 run_maqac_airsim.py --n 4 --method maqac --scenario urban --seed 20000 --heightmap survey/heightmap.npz \
        --out runs/urban4_maqac_20000 [--T 900] [--frames-every 10] [--video]
    # local test without simulators (first-order autopilot model):
    python3 run_maqac_airsim.py --dry-run --n 4 --T 120 --out /tmp/dry
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'netsim'))   # VLM-paper snapshot layout
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from sim import NetConfig, World, propulsion_power, realistic   # noqa: E402
from episode import run_episode                       # noqa: E402
from heightmap import AreaFrame, HeightMap             # noqa: E402
from make_settings_city import SPOTS                   # noqa: E402


# ----------------------------------------------------------------------------- dry-run autopilot
class FakeFleet:
    """First-order position tracker with velocity and acceleration limits (no simulator); same interface as Fleet."""
    def __init__(self, n, spawn_ne, dt=0.2, vmax=15.0, amax=4.0, tau=1.0, log=print):
        self.n, self.dt, self.vmax, self.amax, self.tau, self.log = n, dt, vmax, amax, tau, log
        self.p = np.array([[x, y, 1.6] for x, y in spawn_ne], float)
        self.v = np.zeros((n, 3))
        self.sp = [None] * n
        self.t_sim = 0.0

    def step(self, dt=None):
        dt = self.dt if dt is None else dt
        for i in range(self.n):
            if self.sp[i] is None:
                continue
            pos, vel = self.sp[i]
            vc = vel + (pos - self.p[i]) / self.tau
            s = np.linalg.norm(vc)
            if s > self.vmax:
                vc *= self.vmax / s
            dv = vc - self.v[i]
            a = np.linalg.norm(dv) / dt
            if a > self.amax:
                dv *= self.amax / a
            self.v[i] += dv
            self.p[i] += self.v[i] * dt
        self.t_sim += dt

    def run_for(self, s, until=None):
        for _ in range(int(round(s / self.dt))):
            self.step()
            if until is not None and until():
                return True
        return True

    def wait_ready(self):
        self.log('dry run: fake PX4 ready')

    def set_world(self, i, p, v):
        self.sp[i] = (np.asarray(p, float), np.asarray(v, float))

    def world(self):
        return self.p.copy()

    def gt_world(self):
        return self.p.copy()

    def takeoff(self, alt_d, start_ne, v_climb=3.0):
        for i in range(self.n):
            self.set_world(i, [start_ne[i][0], start_ne[i][1], alt_d[i]], np.zeros(3))
        self.run_for(120.0, until=lambda: np.all(np.linalg.norm(self.p[:, :2] - start_ne, axis=1) < 2.0))

    def status(self):
        return [dict(mode='FAKE')] * self.n

    def land(self):
        pass

    def close(self):
        pass


# ----------------------------------------------------------------------------- AirSim users
class Users:
    def __init__(self, c, hm, frame, every=1, log=print):
        import airsim
        self.A, self.c, self.hm, self.fr, self.every, self.log = airsim, c, hm, frame, every, log
        self.spawned = {}            # user id -> (object name, last world N, E)
        self.snap_m = []

    def _pose(self, N, E, D, yaw):
        A = self.A
        return A.Pose(A.Vector3r(float(N), float(E), float(D)), A.to_quaternion(0, 0, float(yaw)))

    def update(self, W, t):
        if int(t) % self.every:
            return
        idx = np.where(W.present(t))[0]
        ne = self.fr.to_world(W.pos[idx])
        N2, E2, g, moved = self.hm.snap(ne[:, 0], ne[:, 1])
        for k, j in enumerate(idx):
            j = int(j)
            if j not in self.spawned:
                nm = self.c.simSpawnObject(f'user_{j}', 'BicycleMan', self._pose(N2[k], E2[k], -g[k], 0.0),
                                           self.A.Vector3r(1, 1, 1), False)
                self.c.simSetSegmentationObjectID(nm, 100 + int(W.cls[j]), False)
                self.spawned[j] = (nm, N2[k], E2[k])
                self.snap_m.append(float(moved[k]))
            elif W.mobile[j]:
                nm, n0, e0 = self.spawned[j]
                if math.hypot(N2[k] - n0, E2[k] - e0) > 0.3:
                    yaw = math.atan2(E2[k] - e0, N2[k] - n0)
                    self.c.simSetObjectPose(nm, self._pose(N2[k], E2[k], -g[k], yaw), True)
                    self.spawned[j] = (nm, N2[k], E2[k])

    def cleanup(self):
        """Remove every spawned user so the next mission starts from an empty scene."""
        n = 0
        for nm, _, _ in self.spawned.values():
            try:
                n += bool(self.c.simDestroyObject(nm))
            except Exception:
                pass
        self.spawned = {}
        return n

    def visible(self, W, t, ne_uav, r):
        """Users (id, class, world N, E) within r metres (horizontal) of a UAV."""
        out = []
        for j, (nm, n0, e0) in self.spawned.items():
            if math.hypot(n0 - ne_uav[0], e0 - ne_uav[1]) <= r:
                out.append([j, int(W.cls[j]), round(float(n0), 2), round(float(e0), 2)])
        return out

    def stats(self):
        s = np.array(self.snap_m) if self.snap_m else np.zeros(1)
        return dict(spawned=len(self.spawned), snap_mean_m=float(s.mean()), snap_p95_m=float(np.percentile(s, 95)),
                    snap_moved_frac=float((s > 0.5).mean()))


# ----------------------------------------------------------------------------- tower clearance
W_UAV_N = 20.0          # vehicle weight (N) behind the propulsion-power model's parameters; climb energy = W * dh


class Clearance:
    """Ceiling map: highest surface within r metres of each cell (surveyed height map, world 'up' heights).
    need(p0, p1) is the highest ceiling on the straight segment p0 -> p1 (world N, E), sampled every 5 m."""
    def __init__(self, hm, r=15.0, brake=50.0):
        from scipy import ndimage
        self.hm = hm

        def disk_max(h, r):
            k = int(round(r / hm.res))
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            return ndimage.maximum_filter(h, footprint=(xx ** 2 + yy ** 2) <= k * k, mode='nearest')
        self.C = disk_max(hm.h, r)
        self.Cb = disk_max(self.C, brake)        # ceiling within braking distance in any direction (turns)

    def at(self, N, E, brake=False):
        i, j = self.hm._clip(*self.hm.cell(N, E))
        return (self.Cb if brake else self.C)[i, j]

    def need(self, p0, p1):
        d = float(np.hypot(*(np.asarray(p1) - p0)))
        s = np.linspace(0.0, 1.0, max(2, int(d / 5.0) + 2))
        q = np.asarray(p0)[None, :] + s[:, None] * (np.asarray(p1) - p0)[None, :]
        return float(self.at(q[:, 0], q[:, 1]).max())


def toward(p0, p1, dmax):
    """Point on p0 -> p1 at most dmax from p0."""
    v = np.asarray(p1, float) - p0
    d = float(np.hypot(*v))
    return np.asarray(p1, float) if d <= dmax else p0 + v / d * dmax


# ----------------------------------------------------------------------------- motion hook
class Mover:
    """Climb rule (towers above the flight altitude): each UAV flies at its nominal altitude (alt + layer*i) unless the
    ceiling map on its path to the target, up to LOOKAHEAD m ahead, is higher; it then climbs to that ceiling + MARGIN
    (+ layer*i); the ceiling within the braking radius in every direction counts too, so a sudden turn cannot lead
    into a tower. If the ceiling within HOLD m ahead (or the braking radius) is still above it, it holds its horizontal position while climbing,
    so it never flies into a tower; the network model sees the UAV hold (actual positions replace the reference)."""
    def __init__(self, fleet, frame, alt_d, nsub, hm=None, log_f=None, clear=None, margin=20.0, layer=3.0,
                 lookahead=150.0, hold=80.0):
        self.f, self.fr, self.alt_d, self.nsub, self.hm, self.log_f = fleet, frame, alt_d, nsub, hm, log_f
        self.cl, self.margin, self.layer, self.look, self.hold = clear, margin, layer, lookahead, hold
        self.err, self.clear, self.clear_r, self.minsep = [], [], [], []
        self.hold_s = np.zeros(len(alt_d))
        self.climb_s = np.zeros(len(alt_d))
        self.climb_m = np.zeros(len(alt_d))
        self.max_up = np.full(len(alt_d), -np.inf)
        self.prev_up = None
        self.alt_log = []

    def _cmd(self, i, cur, ref_w, tgt_w):
        """World D command and whether to hold horizontally, for UAV i at world NE cur."""
        up_nom = -self.alt_d[i]
        if self.cl is None:
            return self.alt_d[i], False
        disk = float(self.cl.at(cur[0], cur[1], brake=True))
        far = max(disk, self.cl.need(cur, toward(cur, tgt_w, self.look)), self.cl.need(cur, toward(cur, ref_w, self.look)))
        near = max(disk, self.cl.need(cur, toward(cur, tgt_w, self.hold)), self.cl.need(cur, toward(cur, ref_w, self.hold)))
        up_req = max(up_nom, far + self.margin + self.layer * i)
        return -up_req, near + self.margin - 3.0 > self._up[i]

    def __call__(self, W, targets, sep, t):
        n = W.cfg.n_uav
        air0 = W.airborne().copy()
        mode0 = W.mode.copy()
        old = W.upos.copy()
        W.move_uavs(targets, sep, t)                 # kinematic reference, separation rule, battery logic
        ref = W.upos.copy()
        W.upos[:] = old
        rw, ow = self.fr.to_world(ref), self.fr.to_world(old)
        tw = self.fr.to_world(np.asarray(targets, float))
        ring = self.fr.to_world(W.gw + 30.0 * np.stack([np.cos(2 * np.pi * np.arange(n) / n),
                                                         np.sin(2 * np.pi * np.arange(n) / n)], -1))
        cur = self.f.gt_world()
        self._up = -cur[:, 2]
        for i in range(n):
            if W.mode[i] == 2 or not W.alive[i]:      # battery swap (landed in the model) / failed: wait on own ring point
                d, hold = self._cmd(i, cur[i, :2], ring[i], ring[i])
                p, v = ring[i], np.zeros(2)
            else:
                d, hold = self._cmd(i, cur[i, :2], rw[i], tw[i])
                p, v = rw[i], rw[i] - ow[i]
            if hold:                                 # tower ahead and not yet above it: climb in place
                p, v = cur[i, :2], np.zeros(2)
                self.hold_s[i] += 1.0
            if d < self.alt_d[i] - 0.5:
                self.climb_s[i] += 1.0
            self.f.set_world(i, [p[0], p[1], d], [v[0], v[1], 0.0])
        for _ in range(self.nsub):
            self.f.step(1.0 / self.nsub)
        aw = self.f.world()
        act = np.clip(self.fr.to_area(aw[:, :2]), 0, W.cfg.world)
        e = np.full(n, np.nan)
        for i in range(n):
            # the model starts the swap when its reference reached the gateway; require the real vehicle to be there
            if mode0[i] == 1 and W.mode[i] == 2 and np.hypot(*(act[i] - W.gw)) >= 5.0:
                W.mode[i] = 1
            if W.alive[i] and W.mode[i] != 2:
                W.upos[i] = act[i]
                e[i] = float(np.hypot(*(act[i] - ref[i])))
        gt = self.f.gt_world()
        up = -gt[:, 2]
        dup = np.zeros(n) if self.prev_up is None else np.maximum(0.0, up - self.prev_up)
        self.prev_up = up
        self.max_up = np.maximum(self.max_up, up)
        if self.hm is not None:                      # radio links use the true height above the local ground
            i_, j_ = self.hm._clip(*self.hm.cell(gt[:, 0], gt[:, 1]))
            W.ualt[:] = np.maximum(10.0, up - self.hm.ground[i_, j_])
            self.alt_log.append(W.ualt.copy())
        for i in range(n):
            if air0[i]:                              # energy and distance from the distance actually flown, + climbs
                mr, ma = float(np.hypot(*(ref[i] - old[i]))), float(np.hypot(*(W.upos[i] - old[i])))
                d = propulsion_power(ma) - propulsion_power(mr) + W_UAV_N * dup[i]
                W.energy[i] -= d
                W.m['energy_j'] += d
                W.m['dist'] += ma - mr
                self.climb_m[i] += dup[i]
        dmin = min((float(np.linalg.norm(gt[i] - gt[j])) for i in range(n) for j in range(i + 1, n)), default=np.inf)
        clr = None
        if self.hm is not None:
            clr = float(np.min(up - self.hm.height(gt[:, 0], gt[:, 1])))
            self.clear.append(clr)
            if self.cl is not None:
                self.clear_r.append(float(np.min(up - self.cl.at(gt[:, 0], gt[:, 1]))))
        self.err.append(e)
        self.minsep.append(dmin)
        if self.log_f is not None:
            self.log_f.write(json.dumps(dict(
                t=int(t), ref=np.round(ref, 2).tolist(), act=np.round(W.upos, 2).tolist(),
                gt_world=np.round(gt, 2).tolist(), mode=W.mode.tolist(), err=[None if np.isnan(x) else round(x, 2) for x in e],
                min_sep=round(dmin, 2), clearance=None if clr is None else round(clr, 2),
                clearance_r=round(self.clear_r[-1], 2) if self.clear_r else None, up=np.round(up, 1).tolist(),
                served=int(np.sum(~np.isnan(W.first_t))))) + '\n')

    def stats(self):
        E = np.array(self.err, float)
        v = E[np.isfinite(E)]
        out = dict(track_err_mean_m=float(v.mean()) if v.size else None,
                   track_err_p95_m=float(np.percentile(v, 95)) if v.size else None,
                   track_err_max_m=float(v.max()) if v.size else None,
                   min_separation_m=float(np.min(self.minsep)) if self.minsep else None)
        if self.clear:
            out['min_clearance_m'] = float(np.min(self.clear))
        if self.clear_r:
            out['min_clearance_15m_m'] = float(np.min(self.clear_r))
        out.update(hold_s=self.hold_s.tolist(), climb_s=self.climb_s.tolist(), climb_m=np.round(self.climb_m, 1).tolist(),
                   max_up_m=np.round(self.max_up, 1).tolist(),
                   radio_alt_mean_m=float(np.mean(self.alt_log)) if self.alt_log else None,
                   radio_alt_p95_m=float(np.percentile(self.alt_log, 95)) if self.alt_log else None, climb_energy_wh=float(W_UAV_N * self.climb_m.sum() / 3600.0))
        return out


# ----------------------------------------------------------------------------- safety log (5 Hz)
class SafetyLog:
    """Ground-truth separation between UAVs and clearance to buildings at every fleet sub-step (setpoint rate),
    per phase ('takeoff', 'mission'). A near-contact EVENT starts when a pair comes closer than NEAR m and ends when it
    is back beyond NEAR + 1 m (hysteresis); exposure is counted in pair-seconds (pairs airborne x time)."""
    NEAR, CLOSE = 5.0, 15.0

    def __init__(self, n, dt, hm=None, f=None):
        self.n, self.dt, self.hm, self.f = n, dt, hm, f
        self.ph = {}
        self.inside = np.zeros((n, n), bool)

    def _p(self, phase):
        return self.ph.setdefault(phase, dict(samples=0, pair_s=0.0, min_sep_m=np.inf, near_events=0, near_s=0.0,
                                              close_s=0.0, min_clear_m=np.inf, below_10m_clear_s=0.0, events=[]))

    def __call__(self, fleet, phase, t=None):
        gt = fleet.gt_world()
        P = self._p(phase)
        P['samples'] += 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(gt[i] - gt[j]))
                P['pair_s'] += self.dt
                P['min_sep_m'] = min(P['min_sep_m'], d)
                if d < self.NEAR:
                    P['near_s'] += self.dt
                    if not self.inside[i, j]:
                        P['near_events'] += 1
                        P['events'].append(dict(t_sim=round(fleet.t_sim, 2), pair=[i, j], d_m=round(d, 2)))
                    self.inside[i, j] = True
                elif d > self.NEAR + 1.0:
                    self.inside[i, j] = False
                if d < self.CLOSE:
                    P['close_s'] += self.dt
        if self.hm is not None:
            up = -gt[:, 2]
            clr = up - self.hm.height(gt[:, 0], gt[:, 1])
            P['min_clear_m'] = min(P['min_clear_m'], float(np.min(clr)))
            P['below_10m_clear_s'] += self.dt * float(np.sum(clr < 10.0))

    def stats(self):
        out = {}
        for k, P in self.ph.items():
            q = dict(P)
            q['events'] = P['events'][:50]
            for x in ('min_sep_m', 'min_clear_m'):
                q[x] = None if not np.isfinite(P[x]) else round(P[x], 2)
            q['near_contact'] = bool(P['near_events'] > 0)
            out[k] = q
        return out


def collision_details(c, names):
    """AirSim's sticky collision flag with the object hit and its time (None if never collided)."""
    out = {}
    for nm in names:
        ci = c.simGetCollisionInfo(nm)
        out[nm] = None if not ci.has_collided else dict(object=ci.object_name, object_id=int(ci.object_id),
                                                        t_ns=int(ci.time_stamp),
                                                        pos_ned=[round(ci.position.x_val, 2), round(ci.position.y_val, 2),
                                                                 round(ci.position.z_val, 2)])
    return out


# ----------------------------------------------------------------------------- frames / video
class Recorder:
    def __init__(self, c, names, out, users, frame, frames_every, video, r_view, resolver=None):
        self.resolver = resolver          # VLM paper: per-frame building pixel counts (buildings.Resolver)
        import airsim
        from PIL import Image
        self.A, self.Image, self.c, self.names, self.out = airsim, Image, c, names, out
        self.users, self.fr, self.fe, self.video, self.r_view = users, frame, frames_every, video, r_view
        if frames_every:
            os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
            self.meta = open(os.path.join(out, 'frames', 'labels.jsonl'), 'a')
            for nm in names:                    # VLM paper: the first segmentation capture per camera is stale (pilot 9 Nov); discard it
                self.c.simGetImages([airsim.ImageRequest('down', airsim.ImageType.Segmentation, False, False)], vehicle_name=nm)
        if video:
            os.makedirs(os.path.join(out, 'video'), exist_ok=True)

    def _rgb(self, r):
        return np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1]

    def __call__(self, W, gt, t):
        A = self.A
        if self.fe and int(t) % self.fe == 0:
            for i, nm in enumerate(self.names):
                rs = self.c.simGetImages([A.ImageRequest('down', A.ImageType.Scene, False, False),
                                          A.ImageRequest('down', A.ImageType.Segmentation, False, False)], vehicle_name=nm)
                base = f'{int(t):04d}_uav{i}'
                self.Image.fromarray(self._rgb(rs[0])).save(os.path.join(self.out, 'frames', base + '_rgb.jpg'), quality=92)
                seg = self._rgb(rs[1])
                self.Image.fromarray(seg).save(os.path.join(self.out, 'frames', base + '_seg.png'))
                bld, unk = self.resolver.counts(seg, gt[i][:2]) if self.resolver is not None else (None, None)
                self.meta.write(json.dumps(dict(t=int(t), uav=i, file=base, pose_ned=np.round(gt[i], 2).tolist(),
                                                bld=bld, seg_unknown_px=unk,
                                                mode=int(W.mode[i]),
                                                users_near=self.users.visible(W, t, gt[i], self.r_view)
                                                if self.users else [])) + '\n')
        if self.video:
            k = 0                                   # chase camera behind and above UAV 0, looking at it
            p = gt[k]
            cam = A.Pose(A.Vector3r(float(p[0] - 30.0), float(p[1]), float(p[2] - 15.0)), A.to_quaternion(-0.45, 0, 0))
            self.c.simSetCameraPose('overview', cam, external=True)
            r = self.c.simGetImages([A.ImageRequest('overview', A.ImageType.Scene, False, False)], external=True)[0]
            self.Image.fromarray(self._rgb(r)).save(os.path.join(self.out, 'video', f'chase_{int(t):04d}.jpg'), quality=90)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default='10.142.0.5', help='AirSim API host (Windows VM)')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--method', default='maqac')
    ap.add_argument('--scenario', default='urban')
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--T', type=int, default=900)
    ap.add_argument('--perception', default='onboard')
    ap.add_argument('--alt', type=float, default=100.0)
    ap.add_argument('--battery-wh', type=float, default=None, help='default: 40 (old model), 500 (v3)')
    ap.add_argument('--model', choices=('v3', 'old'), default='v3',
                    help='v3 = sim.realistic() as in the paper (take-off flown by PX4, not by the network model); old = NetConfig()')
    ap.add_argument('--layer', type=float, default=3.0, help='extra altitude per vehicle index (m)')
    ap.add_argument('--L', type=float, default=1000.0, help='side of the mission area (m); users scale with area')
    ap.add_argument('--margin-up', type=float, default=20.0, help='clearance above towers on the path (m)')
    ap.add_argument('--clear-r', type=float, default=15.0, help='horizontal radius of the ceiling map (m)')
    ap.add_argument('--lookahead', type=float, default=150.0)
    ap.add_argument('--brake', type=float, default=50.0, help='radius (m) beyond --clear-r checked in every direction')
    ap.add_argument('--hold-ahead', type=float, default=80.0, help='> braking distance at 14 m/s plus one round')
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(0.0, 0.0), help='world N, E of the gateway')
    ap.add_argument('--heightmap', default=None)
    ap.add_argument('--keepout', type=float, nargs=5, action='append', default=[], metavar=('N_LO', 'N_HI', 'E_LO', 'E_HI', 'TOP'),
                    help='no-fly volume for the planner only: world N/E box (m) treated as TOP m high (world up). '
                         'For collision shapes that extend above the visible model (CityEnviron Building_C_Mid_CornerIn_C264). '
                         'Radio and the logged clearance still use the surveyed (visible) map.')
    ap.add_argument('--geom-los', action='store_true', help='radio LOS by ray march through the height map; users kept on open ground')
    ap.add_argument('--nsub', type=int, default=5, help='sub-steps per 1 s round (setpoint rate, Hz)')
    ap.add_argument('--free-run', action='store_true')
    ap.add_argument('--no-users', action='store_true')
    ap.add_argument('--user-every', type=int, default=1)
    ap.add_argument('--frames-every', type=int, default=0)
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--buildings', default=None, help='VLM paper: building inventory with segmentation IDs (buildings.py build)')
    ap.add_argument('--land', action='store_true', help='land at the gateway ring at the end')
    ap.add_argument('--dry-run', action='store_true', help='no simulators: first-order autopilot model')
    ap.add_argument('--cert', action='store_true', help='log the certificate terms (Theorem 9) for the flown and kinematic runs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'bridge.log'), 'a')

    def log(*x):
        s = f'[{time.strftime("%H:%M:%S")}] ' + ' '.join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    assert a.n <= len(SPOTS)
    s = (a.L / 1000.0) ** 2                     # keep users per km^2 of the 1 km x 1 km network-model scenarios
    if a.model == 'v3':
        kw = dict(battery_wh=a.battery_wh) if a.battery_wh else {}
        cfg = realistic(name=f'{a.scenario}_{a.n}uav', scenario=a.scenario, n_uav=a.n, T=a.T, perception=a.perception,
                        altitude=a.alt, world=a.L, gateway=(a.L / 2, a.L / 2), n_users=int(round(200 * s)),
                        arrival_rate=0.1 * s, takeoff=False, **kw)
    else:
        base = NetConfig()
        cfg = replace(base, name=f'{a.scenario}_{a.n}uav', scenario=a.scenario, n_uav=a.n, T=a.T,
                      perception=a.perception, altitude=a.alt, battery_wh=a.battery_wh or 40.0, world=a.L,
                      gateway=(a.L / 2, a.L / 2), n_users=int(round(base.n_users * s)), arrival_rate=base.arrival_rate * s)
    fr = AreaFrame(cfg.world, a.origin_ne)
    hm = HeightMap.load(a.heightmap) if a.heightmap else None
    hm_plan = hm
    if hm is not None and a.keepout:
        import copy
        hm_plan = copy.copy(hm)
        hm_plan.h = hm.h.copy()
        for n_lo, n_hi, e_lo, e_hi, top in a.keepout:
            i0, j0 = hm._clip(*hm.cell(n_lo, e_lo))
            i1, j1 = hm._clip(*hm.cell(n_hi, e_hi))
            hm_plan.h[i0:i1 + 1, j0:j1 + 1] = np.maximum(hm_plan.h[i0:i1 + 1, j0:j1 + 1], top)
    clear = Clearance(hm_plan, a.clear_r, a.brake) if hm is not None else None
    names = [f'UAV{i}' for i in range(a.n)]
    spawn = [SPOTS[i] for i in range(a.n)]
    c = None
    t_wall = time.time()
    if a.dry_run:
        fleet = FakeFleet(a.n, spawn, dt=1.0 / a.nsub, log=log)
    else:
        import airsim
        from px4_fleet import Fleet
        c = airsim.MultirotorClient(ip=a.ip, timeout_value=120)   # default 3600 s; a stuck call blocked a mission for 29 min (3 Nov)
        c.confirmConnection()
        if not a.free_run:
            c.simPause(True)
        fleet = Fleet(a.n, names, spawn, client=c, dt=1.0 / a.nsub, free_run=a.free_run, log=log)
        if a.buildings:
            assert a.no_users, '--buildings needs --no-users (user segmentation IDs 100+ would clash)'
            from buildings import apply_ids
            log(f'segmentation IDs set for {apply_ids(c, json.load(open(a.buildings)))} buildings')
    safety = SafetyLog(a.n, 1.0 / a.nsub, hm)
    if hasattr(fleet, 'phase'):
        fleet.on_step = lambda fl, ph: safety(fl, ph)
    else:                                   # dry run: FakeFleet has no hook; wrap its step
        _st = fleet.step
        def _step(dt=None):
            _st(dt)
            safety(fleet, getattr(fleet, 'phase', 'startup'))
        fleet.step = _step
    fleet.phase = 'startup'
    users = None
    try:
        fleet.wait_ready()
        rest = fleet.world()
        d_ground = float(np.median(rest[:, 2]))
        alt_d = np.array([d_ground - (a.alt + a.layer * i) for i in range(a.n)])
        start = fr.to_world(World(cfg, a.seed).upos)
        alt_to = alt_d.copy()                   # transfer altitude: above every tower between pad and start point
        if clear is not None:
            for i in range(a.n):
                alt_to[i] = min(alt_d[i], -(clear.need(rest[i, :2], start[i]) + a.margin_up + a.layer * i))
        log(f'takeoff to D {np.round(alt_to, 1).tolist()} (nominal {np.round(alt_d, 1).tolist()}), '
            f'start points {np.round(start, 1).tolist()}, users {cfg.n_users} + {cfg.arrival_rate:.3f}/s, L {cfg.world}')
        coll_rest = collision_details(c, names) if c is not None else None   # ground contact on the pads (sticky flag)
        fleet.phase = 'takeoff'
        fleet.takeoff(alt_to, start)
        fleet.phase = 'mission'
        log(f'at start points after {fleet.t_sim:.1f} s sim; status {fleet.status()}')
        users = None if (a.dry_run or a.no_users or hm is None) else Users(c, hm, fr, a.user_every, log)
        resolver = None
        if a.buildings:
            from buildings import Resolver
            resolver = Resolver(json.load(open(a.buildings)))
        rec = None if (a.dry_run or not (a.frames_every or a.video)) else \
            Recorder(c, names, a.out, users, fr, a.frames_every, a.video, cfg.detect_range, resolver)
        track_px4, coll_events, coll_seen = [], [], {nm: (v['t_ns'] if v else None) for nm, v in (coll_rest or {}).items()}
        rounds = open(os.path.join(a.out, 'rounds.jsonl'), 'w')
        mover = Mover(fleet, fr, alt_d, a.nsub, hm, rounds, clear=clear, margin=a.margin_up, layer=a.layer,
                      lookahead=a.lookahead, hold=a.hold_ahead)

        vis = dict(users_disabled_at=None)

        def hook(W, agents, step):
            nonlocal_users = vis.get('users', users)
            if nonlocal_users is not None:
                try:            # pedestrian models are visual only (no metric uses them); a stuck spawn must not end the mission
                    nonlocal_users.update(W, float(step))
                except Exception as e:
                    log(f'user visuals disabled at t={step}: {e!r}')
                    vis['users'] = None
                    vis['users_disabled_at'] = int(step)
            if rec is not None:
                rec(W, fleet.gt_world(), float(step))
            track_px4.append(np.round(fleet.gt_world(), 2).tolist())      # VLM paper: flown track per round (world NED)
            if c is not None:                                               # per-round poll (AirSim keeps only the last collision)
                for nm in names:
                    ci = c.simGetCollisionInfo(nm)
                    if ci.has_collided and coll_seen.get(nm) != int(ci.time_stamp):
                        coll_seen[nm] = int(ci.time_stamp)
                        coll_events.append(dict(t=int(step), uav=nm, object=ci.object_name,
                                                pos_ned=[round(ci.position.x_val, 2), round(ci.position.y_val, 2), round(ci.position.z_val, 2)]))
            if step % 60 == 0:
                log(f't={step} served={int(np.sum(~np.isnan(W.first_t)))} modes={W.mode.tolist()} '
                    f'track_err_last={np.round(mover.err[-1], 1).tolist() if mover.err else None}')

        t_mis = fleet.t_sim
        setup = None
        if a.geom_los:
            assert hm is not None, '--geom-los needs --heightmap'
            from heightmap import geom_setup
            setup = geom_setup(hm, fr)
        res = run_episode(cfg, a.method, a.seed, log_rounds=a.cert, step_hook=hook, move=mover, setup=setup)
        rounds.close()
        track_kin = []
        def kin_hook(W, agents, step):                                      # VLM paper: kinematic twin track (world N, E)
            track_kin.append(np.round(fr.to_world(W.upos[:, :2]), 2).tolist())   # UAV positions (W.pos = users; bug found 9 Nov)
        kin = run_episode(cfg, a.method, a.seed, log_rounds=a.cert, setup=setup, step_hook=kin_hook)   # kinematic twin, same seed (and radio geometry)
        json.dump(dict(px4=track_px4, kinematic=track_kin, alt=a.alt, origin_ne=list(a.origin_ne), L=a.L),
                  open(os.path.join(a.out, 'tracks.json'), 'w'))
        def cert_sums(r):               # mission-level certificate terms (scripts/netsim_mission_certificate.py)
            rr = [x for x in r.pop('rounds', []) or [] if 'g_hat_full' in x]
            return dict(gh=sum(x['g_hat_full'] for x in rr), ubh=sum(x['ub_hat_full'] for x in rr),
                        g=sum(x['g_full'] for x in rr), rounds=len(rr)) if rr else None
        cert = dict(px4=cert_sums(res), kinematic=cert_sums(kin)) if a.cert else None
        keys = ('qos_score', 'users_served', 'mean_wait_s', 'energy_wh', 'distance_m', 'continuity', 'rtb', 'J_session',
                'session_frac_mean', 'users_total', 'collisions', 'coord_msgs')
        out = dict(method=a.method, keepout=a.keepout, scenario=a.scenario, n_uav=a.n, seed=a.seed, T=a.T, alt=a.alt, layer=a.layer,
                   L=a.L, origin_ne=list(a.origin_ne), n_users=cfg.n_users, margin_up=a.margin_up,
                   dry_run=a.dry_run, px4=res, kinematic={k: kin.get(k) for k in keys},
                   delta_qos_pct=100.0 * (res['qos_score'] - kin['qos_score']) / max(1e-9, kin['qos_score']),
                   delta_J_pct=(100.0 * (res['J_session'] - kin['J_session']) / max(1e-9, kin['J_session'])) if res.get('J_session') is not None else None,
                   flight=mover.stats(), safety=safety.stats(), users=users.stats() if users else None, users_disabled_at=vis['users_disabled_at'],
                   pacing=fleet.pace_stats() if hasattr(fleet, 'pace_stats') else None,
                   sim_s_mission=fleet.t_sim - t_mis, wall_s=round(time.time() - t_wall, 1), cert=cert)
        if c is not None:
            out['collisions_airsim'] = collision_details(c, names)
            out['collisions_at_rest'] = coll_rest
            out['collisions_in_flight'] = {nm: v for nm, v in out['collisions_airsim'].items()
                                           if v is not None and (coll_rest.get(nm) is None or v['t_ns'] != coll_rest[nm]['t_ns'])}
        out['collision_events'] = coll_events
        json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
        log(f"done: J px4 {res['qos_score']:.1f} vs kinematic {kin['qos_score']:.1f} "
            f"({out['delta_qos_pct']:+.1f} %); flight {out['flight']}")
        if a.land:
            fleet.phase = 'landing'
            fleet.land()
    finally:
        if users is not None:
            log(f'removed {users.cleanup()} spawned users')
        fleet.close()
        if c is not None and not a.free_run:
            c.simPause(False)


if __name__ == '__main__':
    main()
