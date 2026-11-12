"""Height map of the AirSim city from downward depth images, and ground snapping of users.

Frames
  world  AirSim NED relative to PlayerStart (N, E, D), metres; height h = -D (up-positive).
  area   network-simulator frame: x east, y north in [0, L]; the gateway (L/2, L/2) maps to world (o_N, o_E).

A downward DepthPlanar image (camera at (Nc, Ec, Dc), yaw 0, horizontal FOV fov, W x H pixels) gives one 3-D point per
pixel: with f = W / (2 tan(fov/2)), a = (v + 0.5 - H/2) d / f and b = (u + 0.5 - W/2) d / f,
    N = Nc + sN * a', E = Ec + sE * b', D = Dc + d,   (a', b') = (b, a) if swap else (a, b).
The default orientation (swap False, sN -1, sE +1: image top = north, image right = east) is checked by
survey_heightmap.py with markers at known positions, which stores the orientation that fits.
Cells of a 1 m grid keep the highest point that falls in them (roof, canopy or ground). Ground level is a local minimum
of the height map; cells more than OBST_M above it are obstacles (buildings, trees, vehicles), and a user placed on an
obstacle is moved to the nearest open cell.
"""
import numpy as np
from scipy import ndimage

OBST_M = 2.5          # height above local ground that counts as an obstacle (roof, canopy, parked vehicle)
GROUND_WIN_M = 100.0  # window of the local-ground minimum


def project(depth, cam_ned, fov_deg, orient=(False, -1, 1), max_depth=None):
    """Depth-planar image (H x W, metres) -> (M, 3) world NED points of valid pixels."""
    H, W = depth.shape
    f = W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    v, u = np.mgrid[0:H, 0:W]
    d = depth.astype(np.float64)
    ok = np.isfinite(d) & (d > 0.1)
    if max_depth is not None:
        ok &= d < max_depth
    a = (v + 0.5 - H / 2.0) * d / f
    b = (u + 0.5 - W / 2.0) * d / f
    swap, sN, sE = orient
    if swap:
        a, b = b, a
    N = cam_ned[0] + sN * a
    E = cam_ned[1] + sE * b
    D = cam_ned[2] + d
    return np.stack([N[ok], E[ok], D[ok]], -1)


class HeightMap:
    def __init__(self, n0, e0, size_n, size_e, res=1.0):
        self.n0, self.e0, self.res = float(n0), float(e0), float(res)
        self.shape = (int(np.ceil(size_n / res)), int(np.ceil(size_e / res)))
        self.h = np.full(self.shape, -np.inf)
        self.orient = (False, -1, 1)
        self.meta = {}

    # ---- building
    def cell(self, N, E):
        i = np.floor((np.asarray(N) - self.n0) / self.res).astype(int)
        j = np.floor((np.asarray(E) - self.e0) / self.res).astype(int)
        return i, j

    def add_points(self, P):
        i, j = self.cell(P[:, 0], P[:, 1])
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        np.maximum.at(self.h, (i[ok], j[ok]), -P[ok, 2])

    def finish(self):
        """Fill unseen cells from the nearest seen cell, estimate local ground, mark obstacles, index nearest open cells."""
        self.valid = np.isfinite(self.h)
        if not self.valid.any():
            raise ValueError('height map is empty')
        _, (ii, jj) = ndimage.distance_transform_edt(~self.valid, return_indices=True)
        self.h = self.h[ii, jj]
        k = max(1, int(round(10.0 / self.res)))          # 10 m blocks -> block minimum -> window minimum
        Hb = self.shape[0] // k * k, self.shape[1] // k * k
        blk = self.h[:Hb[0], :Hb[1]].reshape(Hb[0] // k, k, Hb[1] // k, k).min((1, 3))
        w = max(1, int(round(GROUND_WIN_M / (k * self.res)))) | 1
        g = ndimage.minimum_filter(blk, size=w, mode='nearest')
        g = ndimage.uniform_filter(g, size=3, mode='nearest')
        g = np.repeat(np.repeat(g, k, 0), k, 1)
        G = np.empty(self.shape)
        G[:g.shape[0], :g.shape[1]] = g
        G[g.shape[0]:, :] = G[g.shape[0] - 1:g.shape[0], :] if g.shape[0] < self.shape[0] else 0
        G[:, g.shape[1]:] = G[:, g.shape[1] - 1:g.shape[1]] if g.shape[1] < self.shape[1] else 0
        self.ground = np.minimum(G, self.h)
        self.obst = (self.h - self.ground) > OBST_M
        _, idx = ndimage.distance_transform_edt(self.obst, return_indices=True)
        self.free_i, self.free_j = idx[0].astype(np.int32), idx[1].astype(np.int32)
        return self

    # ---- queries
    def _clip(self, i, j):
        return np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1)

    def height(self, N, E):
        i, j = self._clip(*self.cell(N, E))
        return self.h[i, j]

    def snap(self, N, E):
        """Nearest open ground cell: returns (N', E', h_ground, moved_m)."""
        N, E = np.asarray(N, float), np.asarray(E, float)
        i, j = self._clip(*self.cell(N, E))
        fi, fj = self.free_i[i, j], self.free_j[i, j]
        moved = (fi != i) | (fj != j)
        N2 = np.where(moved, self.n0 + (fi + 0.5) * self.res, N)
        E2 = np.where(moved, self.e0 + (fj + 0.5) * self.res, E)
        return N2, E2, self.ground[fi, fj], np.hypot(N2 - N, E2 - E)

    def stats(self, n_lo, n_hi, e_lo, e_hi, alt):
        """Obstacle heights above ground inside a world window (the mission area) against the flight altitude."""
        i0, j0 = self._clip(*self.cell(n_lo, e_lo))
        i1, j1 = self._clip(*self.cell(n_hi, e_hi))
        h = self.h[i0:i1 + 1, j0:j1 + 1]
        g = self.ground[i0:i1 + 1, j0:j1 + 1]
        rel = h - np.median(g)
        v = self.valid[i0:i1 + 1, j0:j1 + 1]
        ob = self.obst[i0:i1 + 1, j0:j1 + 1]
        return dict(seen_frac=float(v.mean()), obstacle_frac=float(ob.mean()),
                    ground_median=float(np.median(g)), ground_p5_p95=[float(np.percentile(g, 5)), float(np.percentile(g, 95))],
                    top_m=float(rel.max()), p99_m=float(np.percentile(rel, 99)), p999_m=float(np.percentile(rel, 99.9)),
                    frac_above_alt_minus_20=float((rel > alt - 20).mean()), frac_above_alt=float((rel > alt).mean()))

    # ---- io
    def save(self, path):
        np.savez_compressed(path, n0=self.n0, e0=self.e0, res=self.res, h=self.h.astype(np.float32),
                            ground=self.ground.astype(np.float32), obst=self.obst, valid=self.valid,
                            free_i=self.free_i, free_j=self.free_j, orient=np.array(self.orient, float),
                            meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls.__new__(cls)
        m.n0, m.e0, m.res = float(z['n0']), float(z['e0']), float(z['res'])
        m.h, m.ground = z['h'].astype(np.float64), z['ground'].astype(np.float64)
        m.obst, m.valid, m.free_i, m.free_j = z['obst'], z['valid'], z['free_i'], z['free_j']
        m.shape = m.h.shape
        o = z['orient']
        m.orient = (bool(o[0]), int(o[1]), int(o[2]))
        m.meta = str(z['meta'])
        return m


ORIENTS = [(s, a, b) for s in (False, True) for a in (-1, 1) for b in (-1, 1)]


def fit_orientation(blobs, cam_ned, fov_deg, shape, known):
    """blobs: list of (v, u, depth) pixel centroids of markers placed at known world (N, E); returns the orientation with
    the smallest mean horizontal error and the errors of every orientation."""
    H, W = shape
    f = W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    errs = {}
    for o in ORIENTS:
        e = []
        for (v, u, d), (Nk, Ek) in zip(blobs, known):
            a, b = (v + 0.5 - H / 2) * d / f, (u + 0.5 - W / 2) * d / f
            if o[0]:
                a, b = b, a
            e.append(np.hypot(cam_ned[0] + o[1] * a - Nk, cam_ned[1] + o[2] * b - Ek))
        errs[o] = float(np.mean(e))
    best = min(errs, key=errs.get)
    return best, errs


class AreaFrame:
    """Network-simulator area <-> AirSim world."""
    def __init__(self, L=1000.0, origin_ne=(0.0, 0.0)):
        self.L, self.oN, self.oE = float(L), float(origin_ne[0]), float(origin_ne[1])

    def to_world(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([self.oN + xy[..., 1] - self.L / 2, self.oE + xy[..., 0] - self.L / 2], -1)

    def to_area(self, ne):
        ne = np.asarray(ne, float)
        return np.stack([ne[..., 1] - self.oE + self.L / 2, ne[..., 0] - self.oN + self.L / 2], -1)

    def window(self, margin=0.0):
        return (self.oN - self.L / 2 - margin, self.oN + self.L / 2 + margin,
                self.oE - self.L / 2 - margin, self.oE + self.L / 2 + margin)


# ---- radio geometry for the network simulator (World.los_override) ---------------------------------------------
def ground_at(hm, N, E):
    i, j = hm._clip(*hm.cell(N, E))
    return hm.ground[i, j]


def make_los_fn(hm, frame, samples=64, ue_h=1.5):
    """LOS between users and UAVs by a ray march through the surveyed height map. Inputs are in the network-simulator
    area frame: users P (M,2) at ue_h above local ground, UAVs Q (n,2) at hq (n,) above the ground under them.
    Returns f(P, Q, hq) -> (M, n) bool, to be set as World.los_override."""
    s = (np.arange(1, samples) / samples)[None, None, :]

    def f(P, Q, hq):
        Pw, Qw = frame.to_world(P), frame.to_world(Q)
        zu = ground_at(hm, Pw[:, 0], Pw[:, 1]) + ue_h
        zq = ground_at(hm, Qw[:, 0], Qw[:, 1]) + np.asarray(hq, float)
        N = Pw[:, None, 0, None] + (Qw[None, :, 0, None] - Pw[:, None, 0, None]) * s
        E = Pw[:, None, 1, None] + (Qw[None, :, 1, None] - Pw[:, None, 1, None]) * s
        Z = zu[:, None, None] + (zq[None, :, None] - zu[:, None, None]) * s
        return (hm.height(N, E) < Z).all(-1)
    return f


def snap_area(hm, frame, xy):
    """Move area-frame points that fall on an obstacle (building, tree, vehicle) to the nearest open ground cell."""
    w = frame.to_world(xy)
    N2, E2, _, _ = hm.snap(w[:, 0], w[:, 1])
    return frame.to_area(np.stack([N2, E2], -1))


def geom_setup(hm, frame, samples=64):
    """run_episode(setup=...) for the AirSim bridge: height-map LOS on every user-UAV link, and users (home, position,
    next waypoint, and every new waypoint) kept on open ground, so the radio model and the rendered city agree."""
    los = make_los_fn(hm, frame, samples)

    def setup(W):
        W.los_override = los
        W.home = snap_area(hm, frame, W.home)
        W.pos = W.home.copy()
        W.wp = snap_area(hm, frame, W.wp)
        W.hist[:] = W.pos
        raw = W._new_wp
        W._new_wp = lambda idx: snap_area(hm, frame, raw(idx))
    return setup
