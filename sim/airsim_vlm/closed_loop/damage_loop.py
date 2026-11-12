#!/usr/bin/env python3
"""Step 14 (VLM paper, IEEE TVT; drafted 12 Nov 2026, NOT pre-registered yet): closed-loop damage weighting.

Shared pieces of the closed-loop study, used by the kinematic twin (run_closed_loop_kin.py) and the AirSim/PX4 flight
driver (run_closed_loop_airsim.py). Nothing in the step-5/6 code is changed; the netsim is imported read-only.

TRUE objective (the quantity every arm is scored on; theory: w_u = g(y_u), paper eq. (f)):
    J_true = sum_users  w_cls(u) * g_u * frac_u,      g_u = max(1, max_{b: |home_u - c_b| <= R_B} g(y_b))
w_cls and frac_u are exactly those of World.results()['J_session'] (class weight 3/2/1 and the share of the user's session
in service); g = (1, 1.5, 2.5, 3) for no/minor/major/total damage (paper sec:model); y_b is the building's TRUE grade from
the step-3 label model; c_b the merged footprint centre; R_B = 25 m (as the HIL prior arm's --bld-radius). Users far from
every building keep g_u = 1. See true_score() below.

PLANNER weights (what the arm feeds to MAQAC-S as soon as building b is first seen, i.e. first frame with >= 2 % of its
pixels; used from the next 1 s round on, by every UAV of the fleet):
    A blind   : w_hat_b = 1                      (= plain MAQAC-S; checked to reproduce it exactly)
    B top     : w_hat_b = g(argmax_c p(c|x_b))
    C set     : w_hat_b = (min_{c in C(x_b)} g(c) + max_{c in C(x_b)} g(c)) / 2   (Lemma lem:tight / lem:conf midpoint)
    D oracle  : w_hat_b = g(y_b)
x_b is ONE RescueNet test image of the building's true grade, fixed per (seed, building) before any mission (balanced over
the pre-registered seed list so images are reused as evenly as possible); p(.|x) and C(x) are the saved V3-calibrated
ensemble posterior and alpha = 0.1 split-conformal set (results/vlm_damage/step3_obs_bundle.npz, the same bundle as steps
3, 5, 6 and the step-7 bench). Replaying stored outputs is equivalent to running the model on that image; the VLM does not
run live, so the flight box needs no GPU memory for it.
The planner multiplies the value of every believed user (detected tracks, in-session hold term) and of the prior density
of unseen users in the cells within R_B of a seen building by max(1, max_b w_hat_b) - the same rule as the true g_u, with
w_hat in place of g(y) and only for buildings seen so far.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys
sys.path.insert(0, str(HERE.parent / 'netsim'))
from agents import MAQACAgent, METHODS                 # noqa: E402

G = {1: 1.0, 2: 1.5, 3: 2.5, 4: 3.0}
ARMS = ('A_blind', 'B_top', 'C_set', 'D_oracle')
R_B = 25.0
# Step-14 gate candidates (prereg sec. 9, fixed 12 Nov before any gate run). 'base' = the step-6 scenario.
SCENARIOS = {
    'base':    dict(R_B=25.0, users_mult=1, demand=False),
    'rb50':    dict(R_B=50.0, users_mult=1, demand=False),     # S1
    'demand':  dict(R_B=25.0, users_mult=3, demand=True),      # S2: 3x generated, kept with prob g_u / 3, prior at 1x
    'dense3':  dict(R_B=25.0, users_mult=3, demand=False),     # S3: 3x users and arrivals, prior 3x
    'urgency': dict(R_B=25.0, users_mult=1, demand=False, urgency=True),   # step 15: emergency share rises with g_u
}
PX_MIN = int(0.02 * 640 * 480)                          # 6,144 px: the step-5/6 'seen' rule
BUNDLE = ROOT / 'results/vlm_damage/step3_obs_bundle.npz'
VIS = ROOT / 'results/vlm_damage/visibility_model_step5.json'
BLDS = ROOT / 'results/vlm_damage/city/buildings.json'

# step-3 label model (scripts/vlm_damage_map.py labels(), copied verbatim so the netsim snapshot it imports is not needed)
P_FAR = np.array([0.60, 0.30, 0.07, 0.03])
P_NEAR = np.array([0.05, 0.15, 0.35, 0.45])
LAYOUTS = ('indep', 'colo')


def labels(B, L, seed, layout, hot_cen):
    r = np.random.default_rng([seed, 91, LAYOUTS.index(layout)])
    epi = hot_cen if layout == 'colo' else r.uniform(0.1 * L, 0.9 * L, (3, 2))
    d = np.hypot(B[:, None, 0] - epi[None, :, 0], B[:, None, 1] - epi[None, :, 1]).min(1)
    s = np.exp(-d ** 2 / (2 * (0.08 * L) ** 2))
    p = (1 - s)[:, None] * P_FAR + s[:, None] * P_NEAR
    u = r.random(len(B))[:, None]
    lab = 1 + (u > np.cumsum(p, 1)).sum(1)
    return np.minimum(lab, 4)


def hotspots(cfg, seed):
    """The 5 disaster user-hotspot centres World._positions draws (replicates World.__init__'s draw order)."""
    rw = np.random.default_rng([seed, 1])
    na = int(rw.poisson(cfg.arrival_rate * cfg.T)) if cfg.arrival_rate > 0 else 0
    rw.uniform(0, cfg.T, na)
    W = cfg.world
    return rw.uniform(0.1 * W, 0.9 * W, (5, 2))


def make_cfg(n=2, T=300, L=600.0, scenario='disaster', perception='onboard', alt=100.0, chain=False, users_mult=1):
    """Step-6 network configuration (box/plan_step6.txt via run_maqac_airsim.py --model v3; = hil_vlm/run_vlm_hil.make_cfg)."""
    from dataclasses import replace
    from sim import realistic
    s = (L / 1000.0) ** 2
    cfg = realistic(name=f'{scenario}_{n}uav', scenario=scenario, n_uav=n, T=T, perception=perception, altitude=alt,
                    world=L, gateway=(L / 2, L / 2), n_users=int(round(200 * s)) * users_mult, arrival_rate=0.1 * s * users_mult,
                    takeoff=False)
    return replace(cfg, seq_relay=True, seq_chain=True) if chain else cfg


# ----------------------------------------------------------------------------- buildings (merged footprints)
def footprints(L=600.0, origin_ne=(40.0, -10.0), path=BLDS):
    """Buildings inside the mission square, stacked pieces merged per rounded (N, E) as in the step-5/6 analysis.
    Returns names, area xy (m x 2), piece name -> footprint index."""
    blds = json.load(open(path))
    inside = [b for b in blds if abs(b['N'] - origin_ne[0]) <= L / 2 and abs(b['E'] - origin_ne[1]) <= L / 2]
    keys = sorted({(round(b['N']), round(b['E'])) for b in inside})
    kid = {k: i for i, k in enumerate(keys)}
    ne = np.array(keys, float)
    xy = np.stack([ne[:, 1] - origin_ne[1] + L / 2, ne[:, 0] - origin_ne[0] + L / 2], -1)   # AreaFrame.to_area
    return ['fp_%d_%d' % k for k in keys], xy, {b['name']: kid[(round(b['N']), round(b['E']))] for b in inside}


# ----------------------------------------------------------------------------- verdict table (built on the laptop)
def conformal_set(p, qhat):
    C = [c for c in (1, 2, 3, 4) if 1 - p[c] <= qhat]
    return C or [int(np.argmax(p[1:] / p[1:].sum())) + 1]


def arm_weights(p, y, qhat):
    pp = p[1:] / p[1:].sum()
    C = conformal_set(p, qhat)
    return dict(A_blind=1.0, B_top=G[int(pp.argmax()) + 1], C_set=0.5 * (min(G[c] for c in C) + max(G[c] for c in C)),
                D_oracle=G[int(y)]), C, int(pp.argmax()) + 1


def verdict_table(seeds, cfg, layout='indep', L=600.0, origin_ne=(40.0, -10.0), bundle=BUNDLE, img_seed=1414):
    """Per seed: true grade, assigned test image, conformal set, top label and the four arms' weights for every footprint.
    Images: per class one fixed permutation of the test images of that grade (img_seed); seeds are processed in the given
    (pre-registered) order and take the next unused images, wrapping only when a class pool is exhausted, so reuse is as
    even as possible and independent of which arm or flight path later sees the building."""
    B = np.load(bundle)
    P, y, qhat = B['P'], B['y'], float(B['qhat'])
    names, xy, _ = footprints(L, origin_ne)
    rng = np.random.default_rng(img_seed)
    perm = {c: rng.permutation(np.where(y == c)[0]) for c in (1, 2, 3, 4)}
    ptr = {c: 0 for c in (1, 2, 3, 4)}
    out = dict(meta=dict(layout=layout, L=L, origin_ne=list(origin_ne), qhat=qhat, R_B=R_B, g=G, img_seed=img_seed,
                         bundle=str(bundle.relative_to(ROOT)), seeds=list(seeds), n_buildings=len(names)), seeds={})
    for s in seeds:
        hot = hotspots(cfg, s) if layout == 'colo' else None
        lab = labels(xy, L, s, layout, hot)
        rows = []
        for b in range(len(names)):
            c = int(lab[b])
            j = int(perm[c][ptr[c] % len(perm[c])]); ptr[c] += 1
            w, C, top = arm_weights(P[j], c, qhat)
            rows.append(dict(name=names[b], x=round(float(xy[b, 0]), 2), y=round(float(xy[b, 1]), 2), grade=c, img=j,
                             top=top, set=C, covered=c in C, w=w))
        out['seeds'][str(s)] = rows
    out['meta']['image_uses'] = {str(c): ptr[c] for c in ptr}
    out['meta']['pool_sizes'] = {str(c): int(len(perm[c])) for c in perm}
    return out


# ----------------------------------------------------------------------------- planner side
class DamageState:
    """Building beliefs shared by the fleet (they travel with the relayed decision list; 87 x one float, not counted as
    messages). update() marks buildings first seen in a frame; the planner reads mult() from the next round on."""

    def __init__(self, rows, arm, centres, R=R_B):
        self.arm, self.R = arm, R
        self.xy = np.array([[r['x'], r['y']] for r in rows], float)
        self.what = np.array([r['w'][arm] for r in rows], float)
        self.grade = np.array([r['grade'] for r in rows], int)
        self.covered = np.array([r['covered'] for r in rows], bool)
        self.setsz = np.array([len(r['set']) for r in rows], int)
        self.seen_t = np.full(len(rows), np.inf)
        self.seen_by = np.full(len(rows), -1)
        self.centres = centres
        self.cells_near = np.hypot(centres[:, None, 0] - self.xy[None, :, 0], centres[:, None, 1] - self.xy[None, :, 1]) <= R
        self.version = 0
        self._cell = np.ones(len(centres))
        self.active = arm != 'A_blind'
        self.log = []

    def update(self, seen_idx, t, uav=-1):
        new = [int(b) for b in seen_idx if np.isinf(self.seen_t[b])]
        for b in new:
            self.seen_t[b] = t; self.seen_by[b] = uav
            self.log.append(dict(t=int(t), uav=int(uav), b=b, grade=int(self.grade[b]), w_hat=float(self.what[b])))
        if new and self.active:
            seen = np.isfinite(self.seen_t)
            m = np.where(self.cells_near[:, seen], self.what[seen][None, :], 1.0)
            self._cell = np.maximum(1.0, m.max(1)) if seen.any() else np.ones(len(self.centres))
            self.version += 1
        return new

    def _recompute(self):
        seen = np.isfinite(self.seen_t)
        m = np.where(self.cells_near[:, seen], self.what[seen][None, :], 1.0)
        self._cell = np.maximum(1.0, m.max(1)) if seen.any() else np.ones(len(self.centres))

    def cell_mult(self):
        return self._cell

    def point_mult(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        seen = np.isfinite(self.seen_t)
        if not self.active or not seen.any() or not len(xy):
            return np.ones(len(xy))
        d = np.hypot(xy[:, None, 0] - self.xy[None, seen, 0], xy[:, None, 1] - self.xy[None, seen, 1])
        return np.maximum(1.0, np.where(d <= self.R, self.what[seen][None, :], 1.0).max(1))


class DamageAgent(MAQACAgent):
    """MAQAC-S whose user values are scaled by the fleet's building weights (DamageState); dmg=None -> plain MAQAC-S."""
    dmg = None
    prior_scale = 1.0    # S2: the planner's prior density stays at the step-6 value (cfg holds 3x generated users)

    def __init__(self, i, cfg, world_info, rng, **flags):
        super().__init__(i, cfg, world_info, rng, **flags)
        self.rho0, self.rho_arr = self.rho0 * self.prior_scale, self.rho_arr * self.prior_scale
        self._pm = None
        self._rho_base = (self.rho0, self.rho_arr)

    def belief(self, fresh=False):
        xy, k, a = super().belief(fresh)
        self._pm = self.dmg.point_mult(xy) if (self.dmg is not None and self.dmg.version) else None
        return xy, k, a

    def _wS(self, k, dt):
        out = super()._wS(k, dt)
        if self._pm is not None and len(self._pm) == out.shape[0]:
            out = out * self._pm[:, None]
        return out

    def in_session(self):
        hx, hw = super().in_session()
        if self.dmg is not None and self.dmg.version and len(hx):
            hw = hw * self.dmg.point_mult(hx)
        return hx, hw

    def value_map(self, events=(), use_prior=None, fresh=False):
        if self.dmg is None or not self.dmg.version:
            return super().value_map(events, use_prior, fresh)
        m = self.dmg.cell_mult()
        self.rho0, self.rho_arr = self._rho_base[0] * m, self._rho_base[1] * m
        try:
            return super().value_map(events, use_prior, fresh)
        finally:
            self.rho0, self.rho_arr = self._rho_base


for _base in ('maqac_seq', 'rag'):
    _c, _f, _u, _s = METHODS[_base]
    METHODS[_base + '_dmg'] = (DamageAgent, dict(_f), _u, _s)


# ----------------------------------------------------------------------------- secondary endpoint (prereg sec. 10)
def f_true_round(W, events, rows, R=R_B):
    """One-round true value of a plan: sum over present unserved users of w_cls g_u S(tau_e - a_u), each user counted once
    at its best covering event (cells cover users within r_s, as World.true_maps)."""
    from sim import survival
    c = W.cfg
    t = float(W.round)
    ev = np.array(events, float).reshape(-1, 3)
    if not len(ev):
        return 0.0
    g = user_g(W, rows, R)
    pres = W.present(t)
    un = np.where(pres & np.isnan(W.first_t))[0]                       # first contact still to come
    ins = np.where(pres & ~np.isnan(W.first_t) & W.kept & (t < W.sess_end))[0]   # amendment 12 Nov: in session
    tot = 0.0
    if len(un):
        xy = W.pos[un]
        cov = np.hypot(xy[:, None, 0] - ev[None, :, 0], xy[:, None, 1] - ev[None, :, 1]) <= c.r_s
        sv = (W.w[un] * g[un])[:, None] * survival(c.survival, W.mu[un][:, None], ev[None, :, 2] - W.arr[un][:, None]) * cov
        tot += float(sv.max(1).sum())
    if len(ins):                                                        # one round of service for covered in-session users
        xy = W.pos[ins]
        cov = (np.hypot(xy[:, None, 0] - ev[None, :, 0], xy[:, None, 1] - ev[None, :, 1]) <= c.r_s).any(1)
        tot += float((W.w[ins] * g[ins] * cov).sum())
    return tot


def counterfactual_round(W, agents, D, rows, arms=('B_top', 'C_set', 'D_oracle'), R=R_B):
    """Re-plan the current round with each arm's weights on the SAME agent state (sequential, index order, full list as
    maqac_seq with p_loss = 0) and score each plan with the true weights. Agent state touched by decide() ('last') is
    restored, and the arm's own weights are put back, so the flown mission is unchanged. Call from the step hook (agents
    still hold round t's beliefs and positions)."""
    if D.version == 0 or agents is None:
        return None
    deciding = [i for i in range(W.cfg.n_uav) if W.alive[i] and W.mode[i] == 0]
    if not deciding:
        return None
    own, saved = D.what.copy(), [getattr(a, 'last', None) for a in agents]
    out = {}
    try:
        for arm in arms:
            D.what = np.array([r['w'][arm] for r in rows], float); D._recompute()
            ev = []
            for i in deciding:
                ev.append(agents[i].decide(list(ev))[1])
            out[arm] = dict(f=f_true_round(W, ev, rows, R), cells=[(round(e[0], 1), round(e[1], 1)) for e in ev])
    finally:
        D.what = own; D._recompute()
        for a, l in zip(agents, saved):
            a.last = l
    return out


# ----------------------------------------------------------------------------- true score
def user_g(W, rows, R=R_B):
    """True damage multiplier g_u of every user (home position), the rule of the module docstring."""
    bxy = np.array([[r['x'], r['y']] for r in rows], float)
    gb = np.array([G[r['grade']] for r in rows], float)
    d = np.hypot(W.home[:, None, 0] - bxy[None, :, 0], W.home[:, None, 1] - bxy[None, :, 1])
    return np.maximum(1.0, np.where(d <= R, gb[None, :], 1.0).max(1))


def demand_setup(rows, R=R_B):
    """S2 (prereg sec. 9): keep each generated user with probability g_u / 3; dropped users never arrive (T + 1)."""
    def setup(W):
        g = user_g(W, rows, R)
        keep = np.random.default_rng([W.seed, 1416]).random(W.N) < g / 3.0
        W.arr[~keep] = W.cfg.T + 1.0
    return setup


def urgency_setup(rows, R=R_B, base=(0.35, 0.35, 0.30), cap=0.95):
    """Step 15 (prereg step15 sec. 1): redraw each user's class with P(emergency) = min(cap, base_E g_u / mean g),
    the rest split realtime : best effort in the default ratio; class-dependent w, mu, rmin and session length follow."""
    def setup(W):
        c = W.cfg
        g = user_g(W, rows, R)
        pE = np.minimum(cap, base[0] * g / g.mean())
        rest = 1.0 - pE
        P = np.stack([pE, rest * base[1] / (base[1] + base[2]), rest * base[2] / (base[1] + base[2])], 1)
        u = np.random.default_rng([W.seed, 1417]).random(W.N)
        new = (u[:, None] > np.cumsum(P, 1)).sum(1).clip(0, 2)
        sm = np.array(c.session_mean, float)
        old_sm = sm[W.cls]
        W.cls = new
        W.w = np.array(c.weights)[new]
        W.mu = np.array(c.mu)[new] * c.mu_scale
        W.rmin = np.array(c.rmin_mbps)[new]
        if (sm > 0).any():
            W.sess_len = np.where(old_sm > 0, W.sess_len / np.where(old_sm > 0, old_sm, 1.0), 0.0) * sm[new]
        W.urgency_info = dict(emergency_share=float((new == 0).mean()), cap_bound=int((pE >= cap).sum()))
    return setup


def true_score(W, rows, R=R_B):
    """J_true = sum w_cls g_u frac_u, frac_u computed exactly as World.results() (v3 session score)."""
    c = W.cfg
    served = ~np.isnan(W.first_t)
    win = np.maximum(1.0, np.minimum(W.sess_end, c.T) - np.where(served, W.first_t, 0.0))
    frac = np.where(served & W.kept, np.minimum(1.0, W.svc_s / win), 0.0)
    g = user_g(W, rows, R)
    act = W.arr <= c.T                                   # S2: dropped users (arrival T + 1) do not exist
    return dict(J_true=float((W.w * g * frac).sum()), J_session_check=float((W.w * frac).sum()),
                J_true_max=float((W.w * g)[act].sum()), users=int(act.sum()), users_g_gt1=int((g[act] > 1).sum()),
                weight_share_g_gt1=float((W.w * g)[act & (g > 1)].sum() / max(1e-9, (W.w * g)[act].sum())))


# ----------------------------------------------------------------------------- kinematic 'seen' model
class VisibilityModel:
    """Calibrated step-6 camera model: per frame (every 5 s per UAV) P(seen | d) = min(1, v_b p(d)), p(d) in 10 m bins
    (visibility_model_step5.json), v_b ~ Gamma(k, 1/k) per (seed, building). Common random numbers: the uniforms are
    fixed per (seed, frame, UAV, building), so all arms of a seed share them."""

    def __init__(self, seed, n_uav, m, T, every=5, path=VIS):
        vm = json.load(open(path))
        self.pc, k = np.array(vm['pc']), vm['k']
        r = np.random.default_rng([seed, 1415])
        self.v = r.gamma(k, 1.0 / k, m)
        self.U = r.random((T // every + 2, n_uav, m))
        self.every = every

    def p(self, d):
        return np.where(d < 10 * len(self.pc), self.pc[np.minimum((d // 10).astype(int), len(self.pc) - 1)], 0.0)

    def seen(self, t, i, pos, bxy):
        d = np.hypot(bxy[:, 0] - pos[0], bxy[:, 1] - pos[1])
        return np.where(self.U[t // self.every, i] < np.minimum(1.0, self.v * self.p(d)))[0]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Write the verdict table (ship this JSON to the box; no images, no npz).')
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--layout', default='indep', choices=LAYOUTS)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = make_cfg()
    vt = verdict_table(a.seeds, cfg, a.layout)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(vt, open(a.out, 'w'), indent=0)
    print(a.out, vt['meta'])
