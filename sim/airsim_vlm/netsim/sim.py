"""
Network-level simulator for the MAQAC v2 experiments (Ad Hoc Networks resubmission).

What it models, beyond the old kinematic runner (src/experiments/run_adhocnet_experiments.py):
  * access links: 3GPP UMa-AV LOS probability and path loss (the paper's Eqs. for P_LOS, PL), fixed
    per-link shadowing, co-channel interference with K-channel reuse, SINR threshold, per-UAV bandwidth
    shared by the admitted users, per-class minimum rate, priority admission (capacity);
  * backhaul: multi-hop UAV-UAV/UAV-gateway graph (range-limited, free-space 5.8 GHz Shannon capacity),
    a UAV serves only while it has a path to the gateway, link capacity shared by the UAVs routed through
    it, offloaded perception frames consume that capacity and see a queueing delay;
  * consensus: service records travel one hop per round over that graph, with per-message loss;
  * perception: detection probability, position noise, class confusion, false positives and latency
    (presets 'oracle', 'onboard', 'offload'; to be replaced by the measured VLA in the AirSim study);
  * users: initial population + Poisson arrivals, random-waypoint mobility around a home position,
    survival clock from arrival, session decided at first admission, kept sessions keep using capacity;
  * UAVs: rotary-wing propulsion power, battery, return-to-base and battery swap, random failures.

Information boundary: agents receive only an observation built by World.observe() -- their own state,
camera detections, the users attached to their own base station, and backhaul messages. They never read
World.first_t / World.kept (the ground truth that leaked into every method in the old runner).

The certificate of Theorem "calibrated online certificate" is computed every `cert_every` rounds:
rho_hat from each UAV's own beliefs, rho_true from the ground truth (uncapacitated disc footprint).
"""
import math
from dataclasses import dataclass, replace, field
import numpy as np
from scipy.ndimage import correlate
from scipy.optimize import linear_sum_assignment

CLASSES = ('emergency', 'realtime', 'besteffort')
SCEN_MIX = {'urban': (0.20, 0.30, 0.50), 'disaster': (0.35, 0.35, 0.30), 'suburban': (0.15, 0.25, 0.60)}
PRIOR_MIX = np.array([0.20, 0.30, 0.50])      # class mix assumed by the agents (operator prior)

# Perception presets. PLACEHOLDERS until the LoRA VLA is measured (work package E); 'lat' in seconds,
# None = computed from the backhaul (offload).
PERCEPTION = {
    'oracle':  dict(pd=1.00, sigma=0.0, pcls=1.00, fp=0.0, lat=0.0),
    'onboard': dict(pd=0.80, sigma=5.0, pcls=0.75, fp=0.5, lat=0.5),
    'offload': dict(pd=0.95, sigma=2.0, pcls=0.92, fp=0.1, lat=None),
}


@dataclass
class NetConfig:
    name: str = 'urban_4uav'
    scenario: str = 'urban'
    n_uav: int = 4
    n_users: int = 200
    arrival_rate: float = 0.1          # new users per second (Poisson)
    T: int = 900                       # mission length, s (1 round = 1 s)
    world: float = 1000.0
    cell: float = 20.0
    altitude: float = 100.0
    speed: float = 14.0
    r_s: float = 50.0                  # service radius
    gamma_db: float = 5.0              # SINR threshold
    # access network
    f_ghz: float = 3.5
    tx_dbm: float = 30.0
    bw_mhz: float = 20.0
    nf_db: float = 7.0
    n_channels: int = 3                # frequency reuse factor (UAV i uses channel i mod K)
    n_max: int = 64                    # max attached users per UAV
    rmin_mbps: tuple = (0.1, 1.0, 0.25)
    # survival / priority
    survival: str = 'exp'              # 'exp' | 'weibull' | 'deadline'
    mu: tuple = (0.028, 0.010, 0.003)
    weights: tuple = (3.0, 2.0, 1.0)
    mu_scale: float = 1.0
    # mobility
    mobile_frac: float = 0.5
    v_user: tuple = (0.5, 1.5)
    wander: float = 150.0
    # vehicular users (off = the main grid): a share drives on a Manhattan road grid
    veh_frac: float = 0.0
    road_spacing: float = 200.0
    v_veh: tuple = (8.0, 15.0)
    # backhaul
    gateway: tuple = (500.0, 500.0)
    gw_height: float = 30.0
    bh_range: float = 500.0
    bh_f_ghz: float = 5.8
    bh_tx_dbm: float = 20.0
    bh_bw_mhz: float = 40.0
    p_loss: float = 0.0                # per-message loss probability on each backhaul link
    seq_relay: bool = False            # step 7 (10 Nov 2026): seq planners relay the decision list over the backhaul graph
    seq_chain: bool = False            # 10 Nov 2026: seq planners keep the PLANNED targets chained to the gateway (see agents._allowed)
                                       # (multi-hop, incl. the gateway); a UAV knows only decisions that can reach it
    bg_load: float = 0.0               # fraction of backhaul capacity used by other traffic
    require_gateway: bool = True
    # energy
    battery_wh: float = 40.0
    p_comm_w: float = 20.0
    swap_s: float = 120.0
    reserve: float = 0.10
    # failures
    p_fail: float = 0.0                # probability that a UAV fails during the mission
    # perception
    perception: str = 'onboard'
    detect_range: float = 150.0
    frame_mbit: float = 4.0            # offloaded frame size (1 frame/s per UAV)
    lat_server: float = 0.15
    # method parameters (MAQAC)
    beta: float = 0.5
    d_min: float = 20.0
    # separation rule: 'first' = target pushed 2 d_min away from the first close neighbour (old model);
    # 'sum' (v3) = repulsion summed over every neighbour closer than 2 d_min plus the braking distance at the
    # closing speed, pads spaced >= 2 d_min + 5 m (fix of 2 Nov 2026: with a_max and GNSS noise the old rule
    # pushed UAVs across each other near the pad ring)
    sep_model: str = 'first'
    conn_filter: bool = True
    cert_every: int = 5
    seed_base: int = 1000
    # channel sensitivity (all off = the channel of the main grid)
    shadow_corr_m: float = 0.0         # > 0: shadowing evolves with the link's displacement, correlation exp(-dx/d)
    fading: bool = False               # per-round block fading: Rician (LOS, K = k_rice_db) / Rayleigh (NLOS)
    k_rice_db: float = 10.0
    interf_los: bool = False           # worst case: every interfering link is LOS
    # ---- v3 physical model (realism audit, 1 Nov 2026). Defaults = the old model; sim.realistic() sets v3.
    chan: str = 'v2'                   # 'v2' | 'uma_av' (TR 36.777 UMa-AV, aerial UE <-> 25 m mast) | 'a2g' (Al-Hourani elevation model, UAV-BS <-> street-level user)
    a2g_env: str = ''                  # '' = from the scenario (urban/disaster -> urban, suburban -> suburban)
    admit_disc: bool = True            # True: a user can attach only within r_s; False: link-limited (SINR, rate, capacity)
    se_alpha: float = 1.0              # SE = min(se_max, se_alpha log2(1 + SINR))
    se_max: float = float('inf')
    session_mean: tuple = (0.0, 0.0, 0.0)   # mean session length (s) per class; 0 = first contact settles the session
    lat_interp: bool = True            # perception uses positions interpolated at t - lat (fix of round(0.5) = 0, 1 Nov 2026; False = old grid)
    cls_visible: bool = True           # False: images give no traffic class (known after attach)
    a_max: float = float('inf')        # m/s^2, horizontal acceleration limit
    takeoff: bool = False              # start on ground pads, climb at v_climb, staggered by takeoff_gap s
    v_climb: float = 3.0
    takeoff_gap: float = 5.0
    weight_n: float = 20.0             # aircraft weight (N) for the propulsion model (Zeng et al. 2019: 20 N)
    rotor_area: float = 0.503          # total rotor disc area (m^2)
    prop_eff: float = 1.0              # electrical efficiency of propulsion (1 = mechanical power, the old model)
    init_charge: tuple = (1.0, 1.0)    # initial state of charge, uniform range
    swap_slots: int = 0                # simultaneous battery swaps at the pad (0 = unlimited)
    gnss_sigma: float = 0.0            # m, horizontal position error of the positions used for separation
    bh_model: str = 'v2'               # 'v2' (hard range, Shannon) | 'v3' (SNR >= bh_snr_db, MAC efficiency, SE cap, relays share air time)
    bh_snr_db: float = 5.0
    bh_mac_eff: float = 0.5
    bh_ant_db: float = 0.0             # antenna gain at each end of a backhaul link (dBi)
    lambda_hold: float = 0.0           # agents: value weight of a known in-session user in the footprint
    # geometric city (v3 option): Manhattan blocks with random building heights; chan='geom' ray-marches LOS through it
    city: str = ''                     # '' = none | 'manhattan'
    # defaults = ITU-R P.1410 'urban': built-up ratio 0.3, 500 buildings/km^2, Rayleigh heights with scale 15 m
    city_block: float = 24.5           # building side (m): block / pitch = sqrt(0.3), pitch = sqrt(1e6 / 500)
    city_street: float = 20.2          # street width (m)
    city_h_dist: str = 'rayleigh'      # 'rayleigh' (scale city_h_median) | 'lognormal' (median city_h_median, shape city_h_sigma)
    city_h_median: float = 15.0
    city_h_sigma: float = 0.6
    city_open: float = 0.0             # share of blocks left open (squares, parks)
    city_file: str = ''                # real city: building-height raster (data/cities/london.npz ...); sets the geometry for chan='geom'
    flight: str = 'kinematic'          # 'kinematic' | 'sixdof' (6-DOF quadrotor + geometric controller tracking the kinematic path, sixdof.py)
    los_samples: int = 32              # points per link for the ray march
    # message loss model
    loss_model: str = 'iid'            # 'iid' (Bernoulli per message) | 'ge' (Gilbert-Elliott bursts, same mean loss)
                                       # | 'node' (correlated across links: per-UAV Gilbert-Elliott state, a link is down if either end is)
    ge_burst_s: float = 5.0            # mean length of a loss burst (s)
    arq_retx: int = 0                  # link-layer retransmissions per message (0 = none); each attempt is counted


# ----------------------------------------------------------------------------- helpers
def survival(kind, mu, t):
    t = np.maximum(t, 0.0)
    if kind == 'exp':
        return np.exp(-mu * t)
    if kind == 'weibull':
        return np.exp(-(mu * t) ** 1.5)
    if kind == 'deadline':
        return (mu * t <= 1.0).astype(float)
    raise ValueError(kind)


def plos(d2, h):
    """LOS probability, as in src/simulation/channel_model_3gpp.py (paper Eq. P_LOS), vectorised."""
    d2 = np.maximum(d2, 1e-3)
    th = np.degrees(np.arctan2(h - 1.5, d2))
    d1 = np.where(th < 15, 18.0, np.where(th < 30, 18 + (th - 15) * 36 / 15, np.where(th < 60, 54 + (th - 30) * 18 / 30, 72.0)))
    p1 = np.where(th < 15, 36.0, np.where(th < 30, 36 + (th - 15) * 114 / 15, np.where(th < 60, 150 + (th - 30) * 350 / 30, 500.0)))
    return np.where(d2 <= d1, 1.0, d1 / d2 + (1 - d1 / d2) * np.exp(-d2 / p1))


def plos_uma_av(d2, h):
    """TR 36.777 Table B-1, UMa-AV: LOS probability for UAV height h (22.5 m < h <= 100 m; 1 above 100 m)."""
    d2 = np.maximum(d2, 1e-3)
    h = np.minimum(np.maximum(h, 22.5), 100.0)
    d1 = np.maximum(460 * np.log10(h) - 700, 18.0)
    p1 = 4300 * np.log10(h) - 3800
    return np.where(d2 <= d1, 1.0, d1 / d2 + np.exp(-d2 / p1) * (1 - d1 / d2))


def pathloss_uma_av(d3, los, fc, h):
    """TR 36.777 Table B-2, UMa-AV path loss (dB): LOS 22.5-300 m, NLOS 10-100 m (h clipped to the valid range)."""
    d3 = np.maximum(d3, 10.0)
    hn = np.minimum(np.maximum(h, 10.0), 100.0)
    pl_los = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(fc)
    pl_nlos = -17.5 + (46 - 7 * np.log10(hn)) * np.log10(d3) + 20 * np.log10(40 * np.pi * fc / 3)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


# Elevation-angle air-to-ground model for street-level users (Al-Hourani, Kandeepan & Lardner, IEEE WCL 3(6), 2014,
# ITU-R P.1410 geometry): P_LOS = 1 / (1 + a exp(-b (theta - a))), PL = FSPL + eta_LOS / eta_NLOS (dB).
A2G_ENV = {'suburban': (4.88, 0.43, 0.1, 21.0), 'urban': (9.61, 0.16, 1.0, 20.0),
           'dense': (12.08, 0.11, 1.6, 23.0), 'highrise': (27.23, 0.08, 2.3, 34.0)}


def plos_a2g(d2, h, env):
    a, b, _, _ = A2G_ENV[env]
    th = np.degrees(np.arctan2(np.maximum(h - 1.5, 0.0), np.maximum(d2, 1e-3)))
    return 1.0 / (1.0 + a * np.exp(-b * (th - a)))


def pathloss_a2g(d3, los, fc, env):
    _, _, el, en = A2G_ENV[env]
    fspl = 20 * np.log10(np.maximum(d3, 1.0)) + 20 * np.log10(fc * 1e9) - 147.55
    return fspl + np.where(los, el, en)


class City:
    """Procedural Manhattan city: square buildings of side `block` separated by streets of width `street`, heights
    Rayleigh (scale h_median) or lognormal; a block is left open with probability p_open. 1 m height grid.
    los(): a link is LOS when the straight segment clears every roof it passes over (ray march, `samples` points)."""
    RES = 1.0

    def __init__(self, world, block, street, h_median, h_sigma, p_open, rng, h_dist='rayleigh'):
        self.world, self.pitch, self.street = world, block + street, street
        nb = int(math.ceil(world / self.pitch)) + 1
        if h_dist == 'rayleigh':
            hb = rng.rayleigh(h_median, (nb, nb))
        else:
            hb = np.exp(np.log(h_median) + h_sigma * rng.standard_normal((nb, nb)))
        hb[rng.random((nb, nb)) < p_open] = 0.0
        g = np.arange(0, world + self.RES, self.RES)
        X, Y = np.meshgrid(g, g, indexing='ij')
        on_st = self.on_street(X, Y)
        self.H = np.where(on_st, 0.0, hb[(X // self.pitch).astype(int), (Y // self.pitch).astype(int)])
        self.n = len(g)

    def on_street(self, x, y):
        return (np.mod(x, self.pitch) < self.street) | (np.mod(y, self.pitch) < self.street)

    def height(self, x, y):
        i = np.clip(np.round(np.asarray(x) / self.RES).astype(int), 0, self.n - 1)
        j = np.clip(np.round(np.asarray(y) / self.RES).astype(int), 0, self.n - 1)
        return self.H[i, j]

    def snap(self, P):
        """Move points inside a building to the nearest street (pedestrians and vehicles are outdoors)."""
        P = np.asarray(P, float).copy()
        bad = ~self.on_street(P[:, 0], P[:, 1])
        if bad.any():
            q = P[bad]
            r = np.mod(q, self.pitch)
            base = q - r
            # nearest street line in x or y (streets occupy [k*pitch, k*pitch + street))
            dx_lo, dx_hi = r[:, 0] - (self.street - 0.5), self.pitch - r[:, 0]
            dy_lo, dy_hi = r[:, 1] - (self.street - 0.5), self.pitch - r[:, 1]
            cand = np.stack([dx_lo, dx_hi, dy_lo, dy_hi], 1)
            k = cand.argmin(1)
            q[k == 0, 0] = base[k == 0, 0] + self.street - 0.5
            q[k == 1, 0] = base[k == 1, 0] + self.pitch + 0.5
            q[k == 2, 1] = base[k == 2, 1] + self.street - 0.5
            q[k == 3, 1] = base[k == 3, 1] + self.pitch + 0.5
            P[bad] = np.clip(q, 0, self.world)
        return P

    def ceiling(self, x, y, r=15.0):
        """Highest roof within r metres (disc max filter, computed once per radius)."""
        if not hasattr(self, '_ceil') or self._ceil[0] != r:
            from scipy.ndimage import maximum_filter
            k = int(math.ceil(r / self.RES))
            ax = np.arange(-k, k + 1)
            fp = np.hypot(*np.meshgrid(ax, ax, indexing='ij')) * self.RES <= r
            self._ceil = (r, maximum_filter(self.H, footprint=fp, mode='nearest'))
        i = np.clip(np.floor(np.asarray(x) / self.RES).astype(int), 0, self.n - 1)
        j = np.clip(np.floor(np.asarray(y) / self.RES).astype(int), 0, self.n - 1)
        return self._ceil[1][i, j]

    def los(self, P, hp, Q, hq, samples=32):
        """LOS matrix (M x n) between ground points P (M,2) at height hp and UAVs Q (n,2) at heights hq (n,)."""
        s = (np.arange(1, samples) / samples)[None, None, :]
        X = P[:, None, 0, None] + (Q[None, :, 0, None] - P[:, None, 0, None]) * s
        Y = P[:, None, 1, None] + (Q[None, :, 1, None] - P[:, None, 1, None]) * s
        Z = hp + (np.asarray(hq)[None, :, None] - hp) * s
        return (self.height(X, Y) < Z).all(-1)


class RealCity(City):
    """Real city from a building-height raster (wind-aware-delivery format: npz with H[ix, iy] at 1 m and `size`,
    e.g. data/cities/london.npz). Open ground = height <= 2 m. Same interface as City (height, on_street, snap, los).

    The measured building heights of the two city blocks (london_lidar.npz, manhattan_nyc.npz) come from third-party
    sources and are not included; their OpenStreetMap height maps (london.npz, manhattan.npz) are."""

    def __init__(self, path):
        from scipy.ndimage import distance_transform_edt
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found. The measured building heights of the two city blocks come from '
                                    'third-party sources and are not included; their OpenStreetMap height maps are '
                                    '(data/cities/london.npz, data/cities/manhattan.npz).')
        d = np.load(path, allow_pickle=False)
        self.H = d['H'].astype(float)
        self.world = float(d['size'])
        self.RES = self.world / self.H.shape[0]
        self.n = self.H.shape[0]
        self.name = str(d['name']) if 'name' in d.files else path
        free = self.H <= 2.0
        _, (self.fi, self.fj) = distance_transform_edt(~free, return_indices=True)

    def on_street(self, x, y):
        return self.height(x, y) <= 2.0

    def height(self, x, y):
        i = np.clip(np.floor(np.asarray(x) / self.RES).astype(int), 0, self.n - 1)
        j = np.clip(np.floor(np.asarray(y) / self.RES).astype(int), 0, self.n - 1)
        return self.H[i, j]

    def snap(self, P):
        P = np.asarray(P, float).copy()
        i = np.clip(np.floor(P[:, 0] / self.RES).astype(int), 0, self.n - 1)
        j = np.clip(np.floor(P[:, 1] / self.RES).astype(int), 0, self.n - 1)
        fi, fj = self.fi[i, j], self.fj[i, j]
        moved = (fi != i) | (fj != j)
        P[moved, 0] = (fi[moved] + 0.5) * self.RES
        P[moved, 1] = (fj[moved] + 0.5) * self.RES
        return P


def shadow_sigma(los, h, chan):
    if chan == 'uma_av':
        return np.where(los, 4.64 * np.exp(-0.0066 * np.minimum(np.maximum(h, 22.5), 300.0)), 6.0)
    return np.where(los, 4.0, 6.0)


def realistic(**kw):
    """The v3 physical model of docs/REALISM_REBUILD_PLAN.md; keyword overrides on top."""
    v3 = dict(chan='a2g', admit_disc=False, r_s=150.0, nf_db=9.0, se_alpha=0.6, se_max=5.5,
              rmin_mbps=(1.0, 2.0, 0.5), session_mean=(180.0, 300.0, 120.0), lat_interp=True, cls_visible=False,
              detect_range=100.0 * math.tan(math.radians(42.0)), a_max=4.0, takeoff=True, weight_n=60.0,
              rotor_area=1.0, prop_eff=0.638, p_comm_w=30.0, battery_wh=500.0,   # hover 637 W electrical = 6-DOF model (sixdof.py); 500 Wh (Matrice-300 class)
              init_charge=(0.6, 1.0), swap_slots=2, gnss_sigma=3.0,
              bh_model='v3', bh_f_ghz=5.6, bh_tx_dbm=27.0, bh_ant_db=3.0,   # 1 W EIRP, 5.47-5.725 GHz RLAN band
              shadow_corr_m=50.0, fading=True, lambda_hold=0.2, sep_model='sum')
    if kw.get('city_file'):                       # a real city fixes the area: world = raster size, gateway at the centre
        size = float(np.load(kw['city_file'])['size'])
        f = (size / 1000.0) ** 2
        v3.update(world=size, gateway=(size / 2, size / 2), n_users=int(200 * f), arrival_rate=0.1 * f, chan='geom')
    v3.update(kw)
    cfg = NetConfig(**v3)
    # backhaul range = distance at which the free-space SNR falls to bh_snr_db (agents' connectivity filter uses it)
    noise_dbm = -174 + 10 * math.log10(cfg.bh_bw_mhz * 1e6) + cfg.nf_db
    fspl_max = cfg.bh_tx_dbm + 2 * cfg.bh_ant_db - noise_dbm - cfg.bh_snr_db
    if 'bh_range' not in kw:
        cfg = replace(cfg, bh_range=float(10 ** ((fspl_max - 20 * math.log10(cfg.bh_f_ghz * 1e9) + 147.55) / 20)))
    return cfg


def pathloss(d3, los, fc):
    d3 = np.maximum(d3, 10.0)
    pl_los = 28.0 + 22.0 * np.log10(d3) + 20.0 * np.log10(fc)       # below the breakpoint (> 2 km here)
    pl_nlos = np.maximum(pl_los, 13.54 + 39.08 * np.log10(d3) + 20.0 * np.log10(fc))
    return np.where(los, pl_los, pl_nlos)


def propulsion_power(v, W=20.0, A=0.503):
    """Rotary-wing propulsion power (W) at forward speed v (m/s); standard blade-profile/induced/parasite model
    (Zeng et al., IEEE TWC 2019). Induced power and v0 follow momentum theory for weight W (N) and disc area A (m^2);
    W = 20 N, A = 0.503 m^2 give Zeng's values (Pi = 88.63 W, v0 = 4.03 m/s)."""
    P0, Utip, d0, rho, s = 79.86, 120.0, 0.6, 1.225, 0.05
    Pi = 1.1 * W ** 1.5 / math.sqrt(2 * rho * A)                      # incremental correction k = 0.1
    v0 = math.sqrt(W / (2 * rho * A))
    return (P0 * (1 + 3 * v ** 2 / Utip ** 2)
            + Pi * np.sqrt(np.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2))
            + 0.5 * d0 * rho * s * A * v ** 3)


_OFFS = {}


def _offsets(k):
    if k not in _OFFS:
        _OFFS[k] = np.array([(dx, dy) for dx in range(-k, k + 1) for dy in range(-k, k + 1)])
    return _OFFS[k]


def footprint_cells(xy, G, cell, r_s):
    """For points xy (M,2): flat indices of candidate cells whose centre is within r_s, and mask.
    The offset window covers r_s (at least +-3 cells, the old fixed window, so r_s <= 70 m is unchanged)."""
    off = _offsets(3 if r_s <= 70.0 else int(math.ceil(r_s / cell)) + 1)
    cx = np.floor(xy[:, 0] / cell).astype(int)
    cy = np.floor(xy[:, 1] / cell).astype(int)
    gx = cx[:, None] + off[None, :, 0]
    gy = cy[:, None] + off[None, :, 1]
    inside = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)
    d = np.hypot((gx + 0.5) * cell - xy[:, :1], (gy + 0.5) * cell - xy[:, 1:2])
    mask = inside & (d <= r_s)
    flat = np.clip(gx, 0, G - 1) * G + np.clip(gy, 0, G - 1)
    return flat, mask


def disc_kernel(cell, r_s):
    k = int(math.ceil(r_s / cell))
    ax = np.arange(-k, k + 1)
    return (np.hypot(*np.meshgrid(ax, ax, indexing='ij')) * cell <= r_s).astype(float)


# ----------------------------------------------------------------------------- world (ground truth)
class World:
    def __init__(self, cfg: NetConfig, seed: int):
        self.cfg = c = cfg
        self.seed = seed
        self.G = int(round(c.world / c.cell))
        g = np.arange(self.G)
        gx, gy = np.meshgrid(g, g, indexing='ij')
        self.centres = np.stack([(gx + 0.5) * c.cell, (gy + 0.5) * c.cell], -1).reshape(-1, 2)
        rw = np.random.default_rng([seed, 1])     # world draws: identical for every method (paired seeds)
        self.rm = np.random.default_rng([seed, 2])  # mobility
        rf = np.random.default_rng([seed, 3])     # failures
        self.rl = np.random.default_rng([seed, 4])  # message loss
        self.rp = [np.random.default_rng([seed, 10 + i]) for i in range(c.n_uav)]  # perception per UAV
        self.rc = np.random.default_rng([seed, 5])  # channel sensitivity draws (shadowing evolution, fading)
        n0 = c.n_users
        na = int(rw.poisson(c.arrival_rate * c.T)) if c.arrival_rate > 0 else 0
        self.N = N = n0 + na
        self.arr = np.concatenate([np.zeros(n0), np.sort(rw.uniform(0, c.T, na))])
        self.home = self._positions(rw, N)
        self.cls = rw.choice(3, N, p=SCEN_MIX[c.scenario])
        self.w = np.array(c.weights)[self.cls]
        self.mu = np.array(c.mu)[self.cls] * c.mu_scale
        self.rmin = np.array(c.rmin_mbps)[self.cls]
        self.mobile = rw.random(N) < c.mobile_frac
        self.vel = rw.uniform(*c.v_user, N)
        self.shadow = rw.standard_normal((N, c.n_uav))
        self.ulos = rw.random((N, c.n_uav))
        self.usess = rw.random(N)
        self.pos = self.home.copy()
        self.veh = np.zeros(N, bool)
        if c.veh_frac > 0:                                # own stream: the main grid's draws are unchanged
            rv = np.random.default_rng([seed, 6])
            self.veh = rv.random(N) < c.veh_frac
            iv = np.where(self.veh)[0]
            sp = c.road_spacing
            horiz = rv.random(len(iv)) < 0.5
            h = self.home[iv].copy()
            h[horiz, 1] = np.clip(np.round(h[horiz, 1] / sp) * sp, 0, c.world)
            h[~horiz, 0] = np.clip(np.round(h[~horiz, 0] / sp) * sp, 0, c.world)
            self.home[iv] = h
            self.pos[iv] = h
            self.mobile[iv] = True
            self.vel[iv] = rv.uniform(*c.v_veh, len(iv))
            self.rv = rv
        self.wp = self._new_wp(np.arange(N))
        self.pause = np.zeros(N)
        self.H = 32
        self.hist = np.repeat(self.pos[None], self.H, 0)
        # UAVs
        n = c.n_uav
        ang = 2 * np.pi * np.arange(n) / n
        self.gw = np.array(c.gateway)
        self.upos = self.gw + 30.0 * np.stack([np.cos(ang), np.sin(ang)], -1)
        self.pad = np.repeat(self.gw[None], n, 0)            # where a UAV lands for a battery swap (old model: the gateway)
        if c.takeoff:                                        # v3: own pads on a ring (spacing: see sep_model)
            if c.sep_model == 'sum':                         # neighbouring pads >= 2 d_min + 5 m apart (chord)
                rr = max(30.0, (2 * c.d_min + 5.0) / (2 * np.sin(np.pi / n))) if n > 1 else 30.0
            else:                                            # first v3 study (1 Nov): arc spacing 25 m
                rr = max(30.0, 25.0 * n / (2 * np.pi))
            self.upos = self.gw + rr * np.stack([np.cos(ang), np.sin(ang)], -1)
            self.pad = self.upos.copy()
        self.ualt = np.full(n, float(c.altitude))  # true altitude above the users' ground; the AirSim bridge sets it per round
        self.alive = np.ones(n, bool)
        self.mode = np.zeros(n, int)          # 0 mission, 1 return-to-base, 2 battery swap (landed), 3 climbing after take-off/swap
        self.E_full = c.battery_wh * 3600.0
        self.energy = np.full(n, self.E_full)
        self.uvel = np.zeros((n, 2))          # horizontal velocity (used when a_max is finite)
        r7 = np.random.default_rng([seed, 7])  # v3 draws (own stream: the old draws are unchanged)
        self.sess_len = np.zeros(N)
        sm = np.array(c.session_mean, float)[self.cls]
        if (sm > 0).any():
            self.sess_len = r7.exponential(1.0, N) * sm
        self.sess_end = np.full(N, np.inf)
        self.svc_s = np.zeros(N)              # seconds in service (attached, rate met) during the session window
        self.depart_t = np.full(N, np.inf)    # never-served users leave when their patience runs out (v3)
        if c.init_charge != (1.0, 1.0):
            self.energy = r7.uniform(*c.init_charge, n) * self.E_full
        self.r7 = r7
        self.sessions = bool((np.array(c.session_mean) > 0).any())
        self.a2g_env = c.a2g_env or {'urban': 'urban', 'disaster': 'urban', 'suburban': 'suburban'}[c.scenario]
        self.gw_h = c.gw_height                # gateway antenna height (above its roof in a real city)
        self.los_override = None               # callable(P, Q, hq) -> LOS matrix; the AirSim bridge sets a height-map ray march
        self.city = None
        if c.city_file:
            self.city = RealCity(c.city_file)
            self.pad = self.city.snap(self.pad)
            self.upos = self.pad.copy()
            self.gw_h = max(c.gw_height, float(self.city.height(*self.gw)) + 5.0)
            self.home = self.city.snap(self.home)
            self.pos = self.home.copy()
            self.wp = self.city.snap(self.wp)
            self.hist[:] = self.pos
        elif c.city == 'manhattan':
            self.city = City(c.world, c.city_block, c.city_street, c.city_h_median, c.city_h_sigma, c.city_open,
                             np.random.default_rng([seed, 8]), c.city_h_dist)
            self.home = self.city.snap(self.home)
            self.pos = self.home.copy()
            self.wp = self.city.snap(self.wp)
            self.hist[:] = self.pos
        self.ge_bad = {}                       # Gilbert-Elliott state per backhaul edge
        self.node_bad = {}                     # per-node Gilbert-Elliott state ('node' loss model), updated once per round
        self.node_round = -1
        self.round = 0
        self.heard_by = [np.zeros(0, int) for _ in range(n)]
        self.climb_from = np.full(n, -np.inf)
        if c.takeoff:                          # on the ground pads; UAV i lifts off at i * takeoff_gap
            self.ualt[:] = 0.0
            self.mode[:] = 3
            self.climb_from = np.arange(n) * c.takeoff_gap
        self.swap_until = np.zeros(n)
        self.fail_t = np.where(rf.random(n) < c.p_fail, rf.uniform(0.2 * c.T, 0.8 * c.T, n), np.inf)
        # outcome
        self.first_t = np.full(N, np.nan)
        self.kept = np.zeros(N, bool)
        self.server = np.full(N, -1)
        self.seen_t = np.full(N, -np.inf)     # last time a user was inside some UAV's camera view
        # metrics
        self.m = dict(cont_num=0, cont_den=0, blocked=0, interf_outage=0, disconnected_s=0, rtb=0, collisions=0,
                      near=0, energy_j=0.0, dist=0.0, hops_sum=0.0, hops_n=0, offload_lat_sum=0.0, offload_n=0,
                      admitted_sum=0, admitted_n=0)
        self.noise = 10 ** ((-174 + 10 * math.log10(c.bw_mhz * 1e6) + c.nf_db) / 10)
        self.bh_noise = 10 ** ((-174 + 10 * math.log10(c.bh_bw_mhz * 1e6) + c.nf_db) / 10)
        self.attached = [np.zeros(0, int) for _ in range(n)]
        self.bh = None

    # --- users
    def _positions(self, r, N):
        c, W = self.cfg, self.cfg.world
        if c.scenario == 'suburban':
            p = r.uniform(0, W, (N, 2))
        elif c.scenario == 'disaster':
            cen = r.uniform(0.1 * W, 0.9 * W, (5, 2))
            sd = r.uniform(20, 50, 5)
            k = r.integers(0, 5, N)
            p = cen[k] + r.standard_normal((N, 2)) * sd[k, None]
        else:  # urban: 40 % in five hotspots, the rest uniform
            cen = r.uniform(0.15 * W, 0.85 * W, (5, 2))
            k = r.integers(0, 5, N)
            hot = r.random(N) < 0.4
            p = np.where(hot[:, None], cen[k] + r.standard_normal((N, 2)) * 40.0, r.uniform(0, W, (N, 2)))
        return np.clip(p, 0, W)

    def _new_wp(self, idx):
        out = self._new_wp_raw(idx)
        city = getattr(self, 'city', None)
        return city.snap(out) if city is not None else out

    def _new_wp_raw(self, idx):
        c = self.cfg
        idx = np.asarray(idx)
        if self.veh.any():
            v = self.veh[idx]
            out = np.zeros((len(idx), 2))
            if (~v).any():
                out[~v] = self._new_wp_ped(idx[~v])
            if v.any():
                out[v] = self._new_wp_veh(idx[v])
            return out
        return self._new_wp_ped(idx)

    def _new_wp_veh(self, idx):
        """Next intersection along a road: from an intersection turn to a random feasible direction."""
        c, sp = self.cfg, self.cfg.road_spacing
        out = np.zeros((len(idx), 2))
        dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float)
        for k, j in enumerate(idx):
            p = self.pos[j]
            on_x = abs(p[1] / sp - round(p[1] / sp)) < 1e-6      # on a horizontal road (y fixed)
            on_y = abs(p[0] / sp - round(p[0] / sp)) < 1e-6
            if on_x and on_y:
                cand = [d for d in dirs if 0 <= p[0] + d[0] * sp <= c.world and 0 <= p[1] + d[1] * sp <= c.world]
                d = cand[self.rv.integers(len(cand))]
                out[k] = p + d * sp
            elif on_x:                                             # drive to the next intersection along x
                lo, hi = np.floor(p[0] / sp) * sp, np.ceil(p[0] / sp) * sp
                out[k] = [hi if self.rv.random() < 0.5 else lo, p[1]]
            else:
                lo, hi = np.floor(p[1] / sp) * sp, np.ceil(p[1] / sp) * sp
                out[k] = [p[0], hi if self.rv.random() < 0.5 else lo]
        return np.clip(out, 0, c.world)

    def _new_wp_ped(self, idx):
        c = self.cfg
        a = self.rm.uniform(0, 2 * np.pi, len(idx))
        rr = c.wander * np.sqrt(self.rm.random(len(idx)))
        return np.clip(self.home[idx] + rr[:, None] * np.stack([np.cos(a), np.sin(a)], -1), 0, c.world)

    def present(self, t):
        return self.arr <= t

    def move_users(self, t, dt=1.0):
        self.round = int(t)
        mv = self.mobile & (self.pause <= t)
        d = self.wp - self.pos
        dist = np.hypot(d[:, 0], d[:, 1])
        step = np.minimum(dist, self.vel * dt)
        ok = mv & (dist > 1e-9)
        self.pos[ok] += d[ok] / dist[ok, None] * step[ok, None]
        arrived = np.where(mv & (dist <= self.vel * dt))[0]
        if len(arrived):
            self.pause[arrived] = t + self.rm.uniform(0, 30, len(arrived))
            if self.veh.any():
                self.pause[arrived[self.veh[arrived]]] = t           # vehicles do not stop at intersections
            self.wp[arrived] = self._new_wp(arrived)
        self.hist[int(t) % self.H] = self.pos

    def _lost_once(self, u, v):
        """One transmission attempt on backhaul edge (u, v) is lost? 'iid': Bernoulli(p_loss); 'ge': two-state
        Gilbert-Elliott chain per edge, advanced once per round (stationary loss p_loss, mean burst ge_burst_s); 'node': per-UAV Gilbert-Elliott
        state updated once per round, a link is down when either end is bad, with P(bad) = 1 - sqrt(1 - p_loss) so that
        each link still loses p_loss on average but losses are correlated across all links of a UAV."""
        c = self.cfg
        if c.p_loss <= 0:
            return False
        if c.loss_model == 'node':
            if self.node_round != self.round:
                b = 1.0 - (1.0 - c.p_loss) ** 0.5
                q = 1.0 / max(c.ge_burst_s, 1.0)
                pg = min(1.0, b * q / max(1e-9, 1 - b))
                for k in range(c.n_uav + 1):
                    bad = self.node_bad.get(k, self.rl.random() < b)
                    self.node_bad[k] = (self.rl.random() >= q) if bad else (self.rl.random() < pg)
                self.node_round = self.round
            return self.node_bad.get(u, False) or self.node_bad.get(v, False)
        if c.loss_model != 'ge':
            return self.rl.random() < c.p_loss
        q = 1.0 / max(c.ge_burst_s, 1.0)                     # bad -> good
        p = min(1.0, c.p_loss * q / max(1e-9, 1 - c.p_loss))  # good -> bad, so that P(bad) = p_loss
        k = (u, v)                                            # state advances once per 1 s round, not per message
        st = self.ge_bad.get(k)                              # (fixed 6 Nov 2026: per-call advance gave methods that
        if st is None:                                       #  send more messages per round shorter bursts in seconds)
            bad = self.rl.random() < c.p_loss
        else:
            bad, last = st
            gap = self.round - last
            if gap > 50:                                     # chain has mixed: draw from the stationary law
                bad = self.rl.random() < c.p_loss
            else:
                for _ in range(gap):
                    bad = (self.rl.random() >= q) if bad else (self.rl.random() < p)
        self.ge_bad[k] = (bad, self.round)
        return bad

    def msg_lost(self, u, v):
        """One message on backhaul edge (u, v) in this round is lost after up to arq_retx retransmissions? With
        arq_retx = 0 this is exactly the earlier single draw. Retransmissions are independent draws under 'iid'; under
        the burst models the state does not change within the round, so a message in a bad state stays lost.
        Transmission attempts are counted in m['tx_attempts'] (messages in m['tx_msgs'])."""
        c = self.cfg
        self.m['tx_msgs'] = self.m.get('tx_msgs', 0) + 1
        if c.arq_retx <= 0 or c.loss_model != 'iid':
            lost = self._lost_once(u, v)
            tries = 1 if (c.arq_retx <= 0 or not lost) else 1 + c.arq_retx
            self.m['tx_attempts'] = self.m.get('tx_attempts', 0) + tries
            return lost
        for a in range(1 + c.arq_retx):
            lost = self._lost_once(u, v)
            if not lost:
                self.m['tx_attempts'] = self.m.get('tx_attempts', 0) + a + 1
                return False
        self.m['tx_attempts'] = self.m.get('tx_attempts', 0) + 1 + c.arq_retx
        return True

    # --- UAV state
    def airborne(self):
        return self.alive & (self.mode != 2)

    def update_failures(self, t):
        self.alive &= ~(self.fail_t <= t)

    # --- backhaul graph
    def backhaul(self, offloading):
        c, n = self.cfg, self.cfg.n_uav
        gwh = getattr(self, 'gw_h', c.gw_height)
        nodes = np.vstack([np.c_[self.upos, np.where(self.mode == 2, c.gw_height, self.ualt)],
                           [*self.gw, gwh]])
        on = np.r_[self.alive, True]
        D = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
        A = (D <= c.bh_range) & on[:, None] & on[None, :]
        np.fill_diagonal(A, False)
        hops = np.full(n + 1, np.inf)
        parent = np.full(n + 1, -1)
        hops[n] = 0
        frontier = [n]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.where(A[u] & np.isinf(hops))[0]:
                    hops[v] = hops[u] + 1
                    parent[v] = u
                    nxt.append(v)
            frontier = nxt
        fspl = 20 * np.log10(np.maximum(D, 1.0)) + 20 * np.log10(c.bh_f_ghz * 1e9) - 147.55
        snr = 10 ** ((c.bh_tx_dbm + 2 * c.bh_ant_db - fspl) / 10) / self.bh_noise
        if c.bh_model == 'v3':                         # MAC efficiency, SE cap (802.11ac-class), relays share air time
            cap = c.bh_mac_eff * c.bh_bw_mhz * np.minimum(6.5, 0.75 * np.log2(1 + snr)) * (1 - c.bg_load)
        else:
            cap = c.bh_bw_mhz * np.log2(1 + snr) * (1 - c.bg_load)            # Mbps
        paths = []
        for i in range(n):
            p, u = [], i
            if np.isfinite(hops[i]):
                while u != n:
                    p.append(u)
                    u = parent[u]
            paths.append(p)
        air = self.airborne()
        n_sub = np.zeros(n + 1)
        n_off = np.zeros(n + 1)
        for i in range(n):
            for u in paths[i]:
                n_sub[u] += air[i]
                n_off[u] += air[i] and offloading
        budget = np.zeros(n)
        lat = np.full(n, np.inf)
        for i in range(n):
            if not paths[i]:
                continue
            caps = np.array([cap[u, parent[u]] for u in paths[i]])
            off = np.array([n_off[u] for u in paths[i]])
            sub = np.array([max(1.0, n_sub[u]) for u in paths[i]])
            budget[i] = max(0.0, np.min((caps - c.frame_mbit * off) / sub))
            if c.bh_model == 'v3':
                budget[i] /= len(paths[i])             # half-duplex relaying on one channel: throughput ~ 1/hops
            util = np.minimum(0.95, c.frame_mbit * off / np.maximum(caps, 1e-9))
            share = caps / np.maximum(off, 1.0)
            lat[i] = c.lat_server + np.max(c.frame_mbit / share / (1 - util)) + 0.005 * len(paths[i])
        edges = [(int(a), int(b)) for a, b in zip(*np.where(A))]
        self.bh = dict(hops=hops, connected=np.isfinite(hops[:n]), budget=budget, off_lat=lat, edges=edges)
        return self.bh

    # --- access network: association, SINR, capacity, admission
    def access(self, t):
        c, n = self.cfg, self.cfg.n_uav
        self.attached = [np.zeros(0, int) for _ in range(n)]
        air = self.airborne()
        never = np.isnan(self.first_t)
        if self.sessions:                                  # v3: a never-served user leaves when its patience runs out,
            gone = never & self.present(t) & np.isinf(self.depart_t) & \
                (self.usess >= survival(c.survival, self.mu, t - self.arr))
            self.depart_t[gone] = t                        # a served user leaves when its session ends
            demand = self.present(t) & ((never & np.isinf(self.depart_t)) | (self.kept & (t < self.sess_end)))
        else:
            demand = self.present(t) & (never | self.kept)
        self.heard_by = [np.zeros(0, int) for _ in range(n)]
        uid = np.where(demand)[0]
        if not len(uid) or not air.any():
            return
        P = self.pos[uid]
        d2 = np.hypot(P[:, None, 0] - self.upos[None, :, 0], P[:, None, 1] - self.upos[None, :, 1])
        h = np.where(self.mode == 2, 0.0, self.ualt)[None, :]
        d3 = np.sqrt(d2 ** 2 + (h - 1.5) ** 2)
        if self.los_override is not None:
            los = self.los_override(P, self.upos, self.ualt)
        elif c.chan == 'geom':
            if self.city is None:
                raise ValueError("chan='geom' needs city='manhattan' or city_file (or a los_override from the AirSim bridge)")
            los = self.city.los(P, 1.5, self.upos, self.ualt, c.los_samples)
        elif c.chan == 'uma_av':
            los = self.ulos[uid] < plos_uma_av(d2, self.ualt[None, :])
        elif c.chan == 'a2g':
            los = self.ulos[uid] < plos_a2g(d2, self.ualt[None, :], self.a2g_env)
        else:
            los = self.ulos[uid] < plos(d2, self.ualt[None, :])
        if c.shadow_corr_m > 0:                          # Gudmundson-type evolution along the link's displacement
            if not hasattr(self, '_sh_pos'):
                self._sh_pos, self._sh_upos = self.pos.copy(), self.upos.copy()
            dx = (np.hypot(*(P - self._sh_pos[uid]).T)[:, None]
                  + np.hypot(*(self.upos - self._sh_upos).T)[None, :])
            rho = np.exp(-dx / c.shadow_corr_m)
            self.shadow[uid] = rho * self.shadow[uid] + np.sqrt(1 - rho ** 2) * self.rc.standard_normal(rho.shape)
            self._sh_pos[uid] = P
            self._sh_upos = self.upos.copy()
        hU = self.ualt[None, :]
        PL = (pathloss_uma_av(d3, los, c.f_ghz, hU) if c.chan == 'uma_av' else
              pathloss_a2g(d3, los, c.f_ghz, self.a2g_env) if c.chan in ('a2g', 'geom') else pathloss(d3, los, c.f_ghz))
        rx = c.tx_dbm - PL - self.shadow[uid] * shadow_sigma(los, hU, c.chan)
        rxl = 10 ** (rx / 10) * air[None, :]            # mW, same unit as self.noise
        if c.fading:
            K = 10 ** (c.k_rice_db / 10)
            h = np.sqrt(K / (K + 1)) + np.sqrt(1 / (2 * (K + 1))) * (self.rc.standard_normal(rxl.shape)
                                                                   + 1j * self.rc.standard_normal(rxl.shape))
            ray = self.rc.exponential(1.0, rxl.shape)
            rxl = rxl * np.where(los, np.abs(h) ** 2, ray)
        rxI = rxl
        if c.interf_los:                                 # interference with LOS path loss on every link
            PLl = (pathloss_uma_av(d3, True, c.f_ghz, hU) if c.chan == 'uma_av' else
                   pathloss_a2g(d3, True, c.f_ghz, self.a2g_env) if c.chan in ('a2g', 'geom') else pathloss(d3, True, c.f_ghz))
            rxI = 10 ** ((c.tx_dbm - PLl - self.shadow[uid] * shadow_sigma(True, hU, c.chan)) / 10) * air[None, :]
        ch = np.arange(n) % max(1, c.n_channels)
        I = np.zeros_like(rxl)
        for k in np.unique(ch):
            cols = ch == k
            I[:, cols] = rxI[:, cols].sum(1, keepdims=True) - rxI[:, cols]
        sinr = rxl / (I + self.noise)
        conn = self.bh['connected'] if c.require_gateway else np.ones(n, bool)
        inr = (d2 <= c.r_s) if c.admit_disc else np.ones(d2.shape, bool)
        elig = inr & air[None, :] & conn[None, :]
        s_e = np.where(elig, sinr, 0.0)
        best = s_e.argmax(1)
        bs = s_e.max(1)
        ok = bs >= 10 ** (c.gamma_db / 10)
        in_rad = (inr & air[None, :]).any(1)
        if c.admit_disc:
            self.m['interf_outage'] += int((in_rad & ~ok).sum())
        else:                                              # link-limited: outage = SNR would pass, SINR does not
            snr_ok = (np.where(elig, rxl, 0.0) / self.noise).max(1) >= 10 ** (c.gamma_db / 10)
            self.m['interf_outage'] += int((snr_ok & ~ok).sum())
            reach = (rxl / self.noise >= 10 ** (c.gamma_db / 10)) & air[None, :]      # control channel reachable
            insess = self.kept[uid] & (t < self.sess_end[uid])
            for i in range(n):
                self.heard_by[i] = uid[reach[:, i] & insess]
        se = np.minimum(c.se_max, c.se_alpha * np.log2(1 + bs))
        ongoing = self.kept[uid]
        for i in range(n):
            cand = np.where(ok & (best == i))[0]
            if not len(cand):
                continue
            order = np.lexsort((-se[cand], ~ongoing[cand], -self.w[uid[cand]]))
            need_max, n_adm, rsum, adm = 0.0, 0, 0.0, []
            for q in cand[order]:
                need = self.rmin[uid[q]] / max(se[q], 1e-9)       # MHz needed at this SINR
                nm = max(need_max, need)
                if n_adm + 1 <= c.n_max and (n_adm + 1) * nm <= c.bw_mhz and rsum + self.rmin[uid[q]] <= self.bh['budget'][i] + 1e-9:
                    need_max, n_adm, rsum = nm, n_adm + 1, rsum + self.rmin[uid[q]]
                    adm.append(q)
            self.m['blocked'] += len(cand) - len(adm)
            adm = uid[np.array(adm, int)]
            self.attached[i] = adm
            self.m['admitted_sum'] += len(adm); self.m['admitted_n'] += 1
            new = adm[np.isnan(self.first_t[adm])]
            if len(new):
                self.first_t[new] = t
                self.server[new] = i
                self.kept[new] = self.usess[new] < survival(c.survival, self.mu[new], t - self.arr[new])
                if self.sessions:
                    self.sess_end[new] = t + self.sess_len[new]
        att = np.concatenate(self.attached) if self.attached else np.zeros(0, int)
        if self.sessions and len(att):
            ins = att[self.kept[att] & (t < self.sess_end[att])]
            self.svc_s[ins] += 1.0
        kept_present = self.kept & self.present(t)
        self.m['cont_den'] += int(kept_present.sum())
        self.m['cont_num'] += int(kept_present[att].sum()) if len(att) else 0

    # --- perception (what UAV i's camera reports this round)
    def perceive(self, i, t, preset, lat):
        c, r = self.cfg, self.rp[i]
        if c.lat_interp:                                   # positions at t - lat, linear between rounds
            tq = max(0.0, t - min(lat, self.H - 2))
            t0 = int(math.floor(tq)); f = tq - t0
            h0 = self.hist[t0 % self.H]; h1 = self.hist[min(t0 + 1, int(t)) % self.H]
            hp = (1 - f) * h0 + f * h1
            pres = self.arr <= tq
        else:
            L = int(min(self.H - 1, round(lat)))
            tp = max(0, int(t) - L)
            hp = self.hist[tp % self.H] if t >= L else self.hist[0]
            pres = self.arr <= tp
        if self.sessions:                                  # departed users are no longer there to be seen
            pres = pres & ~(self.depart_t <= t) & ~(self.sess_end <= t)
        d = np.hypot(hp[:, 0] - self.upos[i, 0], hp[:, 1] - self.upos[i, 1])
        inview = pres & (d <= c.detect_range)
        self.seen_t[inview] = t
        det = inview & (r.random(self.N) < preset['pd'])
        xy = hp[det] + r.standard_normal((det.sum(), 2)) * preset['sigma']
        k = self.cls[det].copy()
        wrong = r.random(len(k)) > preset['pcls']
        k[wrong] = (k[wrong] + r.integers(1, 3, wrong.sum())) % 3
        if not c.cls_visible:                              # an image does not show the traffic class
            k[:] = -1
        nf = r.poisson(preset['fp']) if preset['fp'] > 0 else 0
        if nf:
            a = r.uniform(0, 2 * np.pi, nf)
            rr = c.detect_range * np.sqrt(r.random(nf))
            xy = np.vstack([xy, self.upos[i] + rr[:, None] * np.stack([np.cos(a), np.sin(a)], -1)])
            k = np.r_[k, r.choice(3, nf, p=PRIOR_MIX) if c.cls_visible else np.full(nf, -1)]
        return np.clip(xy, 0, c.world), k

    def observe(self, i, t, preset, lat):
        c = self.cfg
        xy, k = self.perceive(i, t, preset, lat)
        att = self.attached[i]
        # UEs of earlier-served users inside the footprint report their position and first-service time
        # (registered UE, location update) even when their session ended
        heard = np.zeros(0, int)
        if self.airborne()[i]:
            if c.admit_disc:
                dd = np.hypot(self.pos[:, 0] - self.upos[i, 0], self.pos[:, 1] - self.upos[i, 1])
                heard = np.where(self.present(t) & ~np.isnan(self.first_t) & (dd <= c.r_s))[0]
            else:                                          # v3: in-session UEs reachable on the control channel
                heard = self.heard_by[i]
        o = dict(t=t, pos=self.upos[i].copy(), det_xy=xy, det_cls=k, att=att, att_pos=self.pos[att].copy(),
                 heard=heard, heard_pos=self.pos[heard].copy(), heard_t=self.first_t[heard].copy(),
                 hops=self.bh['hops'][i], mode=self.mode[i])
        if self.sessions:                                  # v3: the UE's class is known once it has attached
            o.update(att_cls=self.cls[att], heard_cls=self.cls[heard])
        return o

    # --- motion and energy
    def move_uavs(self, targets, sep, t, dt=1.0):
        c = self.cfg
        PW = dict(W=c.weight_n, A=c.rotor_area)
        air = self.airborne()
        old = self.upos.copy()
        # positions used for separation: own GNSS fix and the neighbours' broadcast fixes (error gnss_sigma)
        seen = old + (self.r7.standard_normal(old.shape) * c.gnss_sigma if c.gnss_sigma > 0 else 0.0)
        for i in range(c.n_uav):
            if not air[i]:
                continue
            climbing = c.takeoff and (self.mode[i] == 3 or (self.mode[i] == 1 and np.hypot(*(old[i] - self.pad[i])) < 5.0))
            if climbing:                                  # vertical phase: take-off/after a swap (up) or landing (down)
                up = self.mode[i] == 3
                if up and t < self.climb_from[i]:
                    continue                              # still waiting on the pad
                dz = c.v_climb * dt
                self.ualt[i] = min(c.altitude, self.ualt[i] + dz) if up else max(0.0, self.ualt[i] - dz)
                e = ((propulsion_power(0.0, **PW) / c.prop_eff) + c.p_comm_w + (c.weight_n * c.v_climb / 0.7 if up else 0.0)) * dt
                self.energy[i] -= e
                self.m['energy_j'] += e
                self.uvel[i] = 0.0
                continue
            tgt = np.asarray(targets[i], float)
            if sep[i] and c.sep_model == 'sum':
                rep = np.zeros(2)
                for j in range(c.n_uav):
                    if j != i and air[j]:
                        v = seen[i] - seen[j]
                        dd = max(float(np.hypot(*v)), 1e-6)
                        u = v / dd
                        close = max(0.0, -float((self.uvel[i] - self.uvel[j]) @ u))      # closing speed
                        R = 2 * c.d_min + (close ** 2 / (2 * c.a_max) if np.isfinite(c.a_max) else 0.0)
                        if dd < R:
                            rep += u * (R - dd)
                if np.hypot(*rep) > 1e-9:
                    tgt = old[i] + rep / np.hypot(*rep) * max(2 * c.d_min, float(np.hypot(*rep)))
            elif sep[i]:
                for j in range(c.n_uav):
                    if j != i and air[j]:
                        v = seen[i] - seen[j]
                        dd = np.hypot(*v)
                        if dd < 2 * c.d_min:
                            tgt = old[i] + v / max(dd, 1e-6) * 2 * c.d_min
                            break
            d = tgt - old[i]
            dist = np.hypot(*d)
            city = getattr(self, 'city', None)
            if city is not None and c.takeoff:             # tower-climb rule: clear the roofs ahead by 20 m
                u = d / dist if dist > 1e-9 else np.zeros(2)
                ahead = old[i][None, :] + u[None, :] * np.linspace(0.0, min(dist, 80.0), 9)[:, None]
                req = float(city.ceiling(ahead[:, 0], ahead[:, 1]).max()) + 20.0
                want = max(c.altitude, req)
                dz = c.v_climb * dt
                if self.ualt[i] < req - 0.5:               # not high enough yet: brake (a_max) and climb
                    self.ualt[i] = min(want, self.ualt[i] + dz)
                    sp = float(np.hypot(*self.uvel[i]))
                    if sp > 1e-9:
                        nv = max(0.0, sp - (c.a_max if np.isfinite(c.a_max) else sp) * dt)
                        self.upos[i] = np.clip(old[i] + self.uvel[i] / sp * 0.5 * (sp + nv) * dt, 0, c.world)
                        self.uvel[i] *= nv / sp
                    e = ((propulsion_power(0.0, **PW) / c.prop_eff) + c.p_comm_w + c.weight_n * c.v_climb / 0.7) * dt
                    self.energy[i] -= e
                    self.m['energy_j'] += e
                    self.m['tower_climb_s'] = self.m.get('tower_climb_s', 0) + 1
                    continue
                if self.ualt[i] > want + 0.5:
                    self.ualt[i] = max(want, self.ualt[i] - dz)
            if np.isfinite(c.a_max):                      # accelerate/brake towards the target, |a| <= a_max
                vmax = min(c.speed, math.sqrt(2 * c.a_max * dist))          # can still stop at the target
                vdes = d / dist * vmax if dist > 1e-9 else np.zeros(2)
                dv = vdes - self.uvel[i]
                ndv = np.hypot(*dv)
                if ndv > c.a_max * dt:
                    dv *= c.a_max * dt / ndv
                vnew = self.uvel[i] + dv
                step = 0.5 * (self.uvel[i] + vnew) * dt
                if np.hypot(*step) > dist:
                    step = d.copy()
                self.uvel[i] = vnew
                self.upos[i] = np.clip(old[i] + step, 0, c.world)
                mv = float(np.hypot(*step))
            else:
                mv = min(dist, c.speed * dt)
                if dist > 1e-9:
                    self.upos[i] = np.clip(old[i] + d / dist * mv, 0, c.world)
            self.m['dist'] += mv
            e = ((propulsion_power(mv / dt, **PW) / c.prop_eff) + c.p_comm_w) * dt
            self.energy[i] -= e
            self.m['energy_j'] += e
        for i in range(c.n_uav):
            for j in range(i + 1, c.n_uav):
                if air[i] and air[j] and self.mode[i] == 0 and self.mode[j] == 0:
                    dd = np.hypot(*(self.upos[i] - self.upos[j]))
                    if c.takeoff:                                    # 3-D distance in the v3 model
                        dd = math.hypot(dd, self.ualt[i] - self.ualt[j])
                    self.m['collisions'] += dd < 5.0
                    self.m['near'] += 5.0 <= dd < 15.0
        # battery logic shared by every method
        PH = (propulsion_power(0.0, **PW) / c.prop_eff)
        for i in range(c.n_uav):
            if not self.alive[i]:
                continue
            dg = np.hypot(*(self.upos[i] - self.pad[i]))
            if self.mode[i] == 3 and self.ualt[i] >= c.altitude:
                self.mode[i] = 0                         # climb finished: mission mode
            if self.mode[i] in (0, 3) and not (self.mode[i] == 3 and t < self.climb_from[i]):
                need = ((propulsion_power(c.speed, **PW) / c.prop_eff) + c.p_comm_w) * dg / c.speed + 30 * PH
                if c.takeoff:
                    need += (PH + c.p_comm_w) * self.ualt[i] / c.v_climb          # descent
                if self.energy[i] < need * (1 + c.reserve) + c.reserve * self.E_full:
                    self.mode[i] = 1
                    self.m['rtb'] += 1
            elif self.mode[i] == 1 and dg < 5.0 and (not c.takeoff or self.ualt[i] <= 0.0):
                busy = int((self.mode == 2).sum())
                if c.swap_slots <= 0 or busy < c.swap_slots:
                    self.mode[i] = 2
                    self.swap_until[i] = t + c.swap_s
                else:
                    self.m['swap_wait_s'] = self.m.get('swap_wait_s', 0) + 1           # on the pad, queued
            elif self.mode[i] == 2 and t >= self.swap_until[i]:
                self.energy[i] = self.E_full
                if c.takeoff:
                    self.mode[i] = 3
                    self.climb_from[i] = t
                else:
                    self.mode[i] = 0
            if self.energy[i] <= 0 and self.mode[i] != 2:
                self.alive[i] = False                    # battery exhausted in flight

    # --- ground-truth certificate (uncapacitated disc footprint, the theory's objective)
    def true_maps(self, t, idx, events, cfgS, scope='full', t_mem=90.0):
        """Return g(G), per-UAV block maxima of Delta(e | X u G) and multiplicity m(G).
        idx: deciding UAVs; events: list of (x, y, tau) chosen by them.
        scope 'full': all unserved users; 'obs': only users inside some camera view during the last t_mem s."""
        c = self.cfg
        mask_u = self.present(t) & np.isnan(self.first_t)
        if scope == 'obs':
            mask_u &= self.seen_t >= t - t_mem
        elif scope == 'view':
            mask_u &= self.seen_t >= t - 1e-9          # inside some camera view in this round
        un = np.where(mask_u)[0]
        if not len(un) or not len(idx):
            return 0.0, np.zeros(len(idx)), 0
        xy, w, mu, a = self.pos[un], self.w[un], self.mu[un], self.arr[un]
        ev = np.array(events, float).reshape(-1, 3)
        cov = np.hypot(xy[:, None, 0] - ev[None, :, 0], xy[:, None, 1] - ev[None, :, 1]) <= c.r_s
        sv = w[:, None] * survival(cfgS, mu[:, None], ev[None, :, 2] - a[:, None]) * cov
        b = sv.max(1) if len(ev) else np.zeros(len(un))
        g = float(b.sum())
        m = int((sv > 0).sum(1).max()) if len(ev) else 0
        flat, mask = footprint_cells(xy, self.G, c.cell, c.r_s)
        blocks = np.zeros(len(idx))
        for q, i in enumerate(idx):
            tau = t + np.hypot(*(self.centres - self.upos[i]).T) / c.speed
            val = w[:, None] * survival(cfgS, mu[:, None], tau[flat] - a[:, None]) - b[:, None]
            val = np.where(mask, np.maximum(val, 0.0), 0.0)
            blocks[q] = np.bincount(flat.ravel(), val.ravel(), self.G ** 2).max()
        return g, blocks, m

    def true_map(self, t, i, events, cfgS):
        """True marginal value map of UAV i given already-chosen events (centralized reference)."""
        c = self.cfg
        un = np.where(self.present(t) & np.isnan(self.first_t))[0]
        if not len(un):
            return np.zeros(self.G ** 2)
        xy, w, mu, a = self.pos[un], self.w[un], self.mu[un], self.arr[un]
        b = np.zeros(len(un))
        if events:
            ev = np.array(events, float).reshape(-1, 3)
            cov = np.hypot(xy[:, None, 0] - ev[None, :, 0], xy[:, None, 1] - ev[None, :, 1]) <= c.r_s
            b = (w[:, None] * survival(cfgS, mu[:, None], ev[None, :, 2] - a[:, None]) * cov).max(1)
        flat, mask = footprint_cells(xy, self.G, c.cell, c.r_s)
        tau = t + np.hypot(*(self.centres - self.upos[i]).T) / c.speed
        val = w[:, None] * survival(cfgS, mu[:, None], tau[flat] - a[:, None]) - b[:, None]
        val = np.where(mask, np.maximum(val, 0.0), 0.0)
        V = np.bincount(flat.ravel(), val.ravel(), self.G ** 2)
        if c.lambda_hold > 0 and self.sessions:            # v3: same hold term as the agents, on the true in-session users
            hs = np.where(self.present(t) & self.kept & (t < self.sess_end))[0]
            if len(hs):
                hx = self.pos[hs]
                if events:
                    ev = np.array(events, float).reshape(-1, 3)
                    keep = ~(np.hypot(hx[:, None, 0] - ev[None, :, 0], hx[:, None, 1] - ev[None, :, 1]) <= c.r_s).any(1)
                    hs, hx = hs[keep], hx[keep]
                if len(hs):
                    f2, m2 = footprint_cells(hx, self.G, c.cell, c.r_s)
                    V += np.bincount(f2.ravel(), (m2 * (c.lambda_hold * self.w[hs])[:, None]).ravel(), self.G ** 2)
        return V

    def results(self):
        c = self.cfg
        served = ~np.isnan(self.first_t)
        wait = np.where(served, self.first_t - self.arr, 0.0)
        surv = survival(c.survival, self.mu, wait) * served
        if self.sessions:                                  # v3 score: first contact before abandonment x share of the session in service
            win = np.maximum(1.0, np.minimum(self.sess_end, c.T) - np.where(served, self.first_t, 0.0))
            frac = np.where(served & self.kept, np.minimum(1.0, self.svc_s / win), 0.0)
            ex = dict(J_session=float((self.w * frac).sum()), session_frac_mean=float(frac[served].mean()) if served.any() else 0.0,
                      users_abandoned=int((np.isnan(self.first_t) & np.isfinite(self.depart_t)).sum()),
                      swap_wait_s=int(self.m.get('swap_wait_s', 0)))
            for k, nm in enumerate(CLASSES):
                mk = self.cls == k
                ex[f'{nm}_J_session'] = float((self.w * frac)[mk].sum())
        else:
            ex = {}
        out = dict(qos_score=float((self.w * surv).sum()), qos_score_realised=float((self.w * self.kept).sum()),
                   qos_max=float(self.w.sum()), users_total=int(self.N), users_served=int(served.sum()),
                   sessions_expected=float(surv.sum()),
                   mean_wait_s=float(wait[served].mean()) if served.any() else float('nan'))
        for k, nm in enumerate(CLASSES):
            mk = self.cls == k
            out[f'{nm}_total'] = int(mk.sum())
            out[f'{nm}_served'] = int((served & mk).sum())
            out[f'{nm}_score'] = float((self.w * surv)[mk].sum())
            out[f'{nm}_wait_s'] = float(wait[served & mk].mean()) if (served & mk).any() else float('nan')
        m = self.m
        out.update(continuity=m['cont_num'] / max(1, m['cont_den']), blocked=m['blocked'],
                   interf_outage=m['interf_outage'], disconnected_uav_s=m['disconnected_s'], rtb=m['rtb'],
                   collisions=int(m['collisions']), near_misses=int(m['near']), energy_wh=m['energy_j'] / 3600,
                   score_per_wh=out['qos_score'] / max(1e-9, m['energy_j'] / 3600), distance_m=m['dist'],
                   mean_hops=m['hops_sum'] / max(1, m['hops_n']),
                   mean_offload_lat_s=m['offload_lat_sum'] / max(1, m['offload_n']) if m['offload_n'] else None,
                   mean_attached=m['admitted_sum'] / max(1, m['admitted_n']),
                   failed_uavs=int((~self.alive).sum()))
        out.update(ex)
        return out
