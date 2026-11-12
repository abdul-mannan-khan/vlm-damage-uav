"""[VLM paper copy, 12 Nov 2026: keep-out boxes added (planner ceiling only, as run_maqac_airsim.py --keepout in step 6);
 heightmap.py also looked up next to this file (Jetson bundle). Source: an earlier MAQAC codebase city.py]
City geometry for the MAQAC HIL runner: the CityEnviron scenario of the AirSim campaign without AirSim.

Same 600 m area, frame (origin N 40, E -10), surveyed height map and tower-climb rule as
sim/airsim/maqac_px4/run_maqac_airsim.py (classes Clearance and Mover._cmd, copied here so the Jetson needs no AirSim
code). Heights are world 'up' metres of the AirSim map; the HIL physics flies in the network-model area frame (x, y)
with z = world up, so the Pixhawk vehicle climbs over the same towers as the PX4 SITL vehicles did.

Per round and UAV i: nominal height alt + layer*i; if the ceiling map (highest surface within 15 m, and within the
50 m braking radius in any direction) on the path ahead (150 m) is higher, the commanded height rises to that ceiling
+ 20 m (+ layer*i); if the ceiling within 80 m ahead is still above the UAV, it holds its horizontal position while
climbing. The commanded height moves at most 3 m/s up and 2 m/s down per round (PX4's MPC_Z_VEL_MAX_UP/DN in the HIL
set-up). UAV 0 (Pixhawk) gets the command as its setpoint; the kinematic UAVs 1..n-1 follow it exactly.
Radio links use each UAV's height above the ground beneath it; LOS stays the statistical model, as in the campaign.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, '..', 'airsim', 'maqac_px4')):
    if os.path.isfile(os.path.join(p, 'heightmap.py')):
        sys.path.insert(0, p)
        break
from heightmap import AreaFrame, HeightMap      # noqa: E402

W_UAV_N = 20.0          # as the AirSim bridge: climb energy = W * dh


class Clearance:
    """Ceiling map (copied from run_maqac_airsim.py)."""
    def __init__(self, hm, r=15.0, brake=50.0):
        from scipy import ndimage
        self.hm = hm

        def disk_max(h, r):
            k = int(round(r / hm.res))
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            return ndimage.maximum_filter(h, footprint=(xx ** 2 + yy ** 2) <= k * k, mode='nearest')
        self.C = disk_max(hm.h, r)
        self.Cb = disk_max(self.C, brake)

    def at(self, N, E, brake=False):
        i, j = self.hm._clip(*self.hm.cell(N, E))
        return (self.Cb if brake else self.C)[i, j]

    def need(self, p0, p1):
        d = float(np.hypot(*(np.asarray(p1) - p0)))
        s = np.linspace(0.0, 1.0, max(2, int(d / 5.0) + 2))
        q = np.asarray(p0)[None, :] + s[:, None] * (np.asarray(p1) - p0)[None, :]
        return float(self.at(q[:, 0], q[:, 1]).max())


def toward(p0, p1, dmax):
    v = np.asarray(p1, float) - p0
    d = float(np.hypot(*v))
    return np.asarray(p1, float) if d <= dmax else p0 + v / d * dmax


class City:
    def __init__(self, heightmap, L=600.0, origin_ne=(40.0, -10.0), alt=100.0, layer=3.0, n=3, margin=20.0,
                 clear_r=15.0, brake=50.0, lookahead=150.0, hold=80.0, v_up=3.0, v_dn=2.0, keepout=()):
        self.hm = HeightMap.load(heightmap)
        hm_plan = self.hm
        if keepout:                        # step-6 keep-out volumes (world N/E box, TOP m up): planner/climb rule only
            import copy
            hm_plan = copy.copy(self.hm)
            hm_plan.h = self.hm.h.copy()
            for n_lo, n_hi, e_lo, e_hi, top in keepout:
                i0, j0 = self.hm._clip(*self.hm.cell(n_lo, e_lo))
                i1, j1 = self.hm._clip(*self.hm.cell(n_hi, e_hi))
                hm_plan.h[i0:i1 + 1, j0:j1 + 1] = np.maximum(hm_plan.h[i0:i1 + 1, j0:j1 + 1], top)
        self.fr = AreaFrame(L, origin_ne)
        self.cl = Clearance(hm_plan, clear_r, brake)
        self.nom = np.array([alt + layer * i for i in range(n)], float)
        self.layer, self.margin, self.look, self.hold, self.v_up, self.v_dn = layer, margin, lookahead, hold, v_up, v_dn
        self.z = None                      # commanded height per UAV (= true height of the kinematic UAVs)
        self.hold_s = np.zeros(n)
        self.climb_m = np.zeros(n)
        self.clear, self.clear_r = [], []

    def w(self, xy):
        return self.fr.to_world(np.asarray(xy, float))

    def required(self, i, cur_xy, ref_xy, tgt_xy, up_now):
        """(required height, hold) for UAV i at area point cur_xy (Mover._cmd of the AirSim bridge)."""
        c, r, t = self.w(cur_xy), self.w(ref_xy), self.w(tgt_xy)
        disk = float(self.cl.at(c[0], c[1], brake=True))
        far = max(disk, self.cl.need(c, toward(c, t, self.look)), self.cl.need(c, toward(c, r, self.look)))
        near = max(disk, self.cl.need(c, toward(c, t, self.hold)), self.cl.need(c, toward(c, r, self.hold)))
        up_req = max(self.nom[i], far + self.margin + self.layer * i)
        return up_req, near + self.margin - 3.0 > up_now

    def start_height(self, i, xy):
        c = self.w(xy)
        return max(self.nom[i], float(self.cl.at(c[0], c[1], brake=True)) + self.margin + self.layer * i)

    def step_z(self, i, up_req):
        dz = float(np.clip(up_req - self.z[i], -self.v_dn, self.v_up))
        self.z[i] += dz
        return dz

    def ground(self, xy):
        c = self.w(np.atleast_2d(xy))
        i, j = self.hm._clip(*self.hm.cell(c[:, 0], c[:, 1]))
        return self.hm.ground[i, j]

    def log_clearance(self, xy, up):
        c = self.w(np.atleast_2d(xy))
        clr = float(np.min(np.asarray(up) - self.hm.height(c[:, 0], c[:, 1])))
        clr_r = float(np.min(np.asarray(up) - self.cl.at(c[:, 0], c[:, 1])))
        self.clear.append(clr)
        self.clear_r.append(clr_r)
        return clr, clr_r

    def stats(self):
        return dict(min_clearance_m=float(np.min(self.clear)) if self.clear else None,
                    min_clearance_15m_m=float(np.min(self.clear_r)) if self.clear_r else None,
                    hold_s=self.hold_s.tolist(), climb_m=np.round(self.climb_m, 1).tolist())
