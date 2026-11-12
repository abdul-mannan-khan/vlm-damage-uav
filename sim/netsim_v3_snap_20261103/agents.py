"""
Agents for the network-level simulator. Every agent sees only:
  obs = World.observe(i, ...)  (own position/mode/hop count, camera detections, own attached users)
  messages merged by Store.merge() over the backhaul graph (one hop per round).

MAQAC follows Algorithm 1 of maqac_adhocnet_v2.tex: record-merging consensus, soft Voronoi weight beta,
footprint value of Eq. (value) with the detection memory and the density prior, simultaneous choice,
optional certificate (Phase 5). Flags give the ablations and the information-matched greedy baseline.
"""
import math
import numpy as np
from scipy.ndimage import correlate
from scipy.optimize import linear_sum_assignment

from sim import survival, footprint_cells, disc_kernel, PRIOR_MIX

T_MEM = 90.0       # s, a remembered detection expires if not seen again
MISS_DROP = 3      # consecutive misses (while in view) before a track is dropped
GATE = 13.0        # m, track-detection association gate
T_POS = 30.0       # s, neighbour position considered valid
T_ACT = 10.0       # s, a user last seen in session this recently is believed still in session (v3)


def cls_of(cnt):
    """Believed class per track: majority of the class reports, -1 if the images gave no class (v3)."""
    return np.where(cnt.sum(1) > 0, cnt.argmax(1), -1) if len(cnt) else np.zeros(0, int)
CONFIRM = 2        # detections needed before a track counts (M-of-N confirmation)
DEDUP = 6.0        # m, a shared track this close to a known track is the same person


class Store:
    """Information a node holds and floods: service records, observed-cell times, neighbour positions."""

    def __init__(self, N, G, n_nodes):
        self.rec_t = np.full(N, np.nan)        # earliest known first-service time per user id
        self.rec_pos = np.zeros((N, 2))        # last reported position of the served user
        self.rec_tp = np.full(N, -np.inf)      # time of that report
        self.learn_t = np.full(N, np.nan)      # when this node learned the record (information age metric)
        self.tobs = np.full(G * G, -np.inf)    # last time any known UAV observed each cell
        self.kpos = np.zeros((n_nodes, 2))
        self.kt = np.full(n_nodes, -np.inf)
        self.khops = np.full(n_nodes, np.inf)
        self.tracks = {}                        # origin node -> (time, xy, class, a_hat): its confirmed unserved tracks
        self.rec_act = np.full(N, -np.inf)     # last time the user was seen in session (attached or heard), v3
        self.rec_cls = np.full(N, -1)          # class, known once the UE has attached

    def snapshot(self):
        s = Store.__new__(Store)
        for k in ('rec_t', 'rec_pos', 'rec_tp', 'tobs', 'kpos', 'kt', 'khops', 'rec_act', 'rec_cls'):
            setattr(s, k, getattr(self, k).copy())
        s.tracks = dict(self.tracks)            # entries are replaced, never mutated
        return s

    def merge(self, o, t):
        new = np.isnan(self.rec_t) & ~np.isnan(o.rec_t)
        self.learn_t[new] = t
        self.rec_t = np.fmin(self.rec_t, o.rec_t)
        nw = o.rec_tp > self.rec_tp
        self.rec_pos[nw] = o.rec_pos[nw]
        self.rec_tp[nw] = o.rec_tp[nw]
        np.maximum(self.tobs, o.tobs, out=self.tobs)
        np.maximum(self.rec_act, o.rec_act, out=self.rec_act)
        np.maximum(self.rec_cls, o.rec_cls, out=self.rec_cls)
        nk = o.kt > self.kt
        self.kpos[nk] = o.kpos[nk]
        self.kt[nk] = o.kt[nk]
        self.khops[nk] = o.khops[nk]
        for j, v in o.tracks.items():
            if j not in self.tracks or v[0] > self.tracks[j][0]:
                self.tracks[j] = v


class BaseAgent:
    consensus = True
    sep = True

    def __init__(self, i, cfg, world_info, rng):
        self.i, self.cfg, self.rng = i, cfg, rng
        self.G, self.N = world_info['G'], world_info['N']
        self.centres = world_info['centres']
        self.gw = np.asarray(cfg.gateway, float)
        self.store = Store(self.N, self.G, cfg.n_uav + 1)
        self.store.kpos[cfg.n_uav] = self.gw
        self.store.kt[cfg.n_uav] = np.inf
        self.store.khops[cfg.n_uav] = 0
        self.mu = np.array(cfg.mu) * cfg.mu_scale
        self.wk = np.array(cfg.weights)
        self.pts = dict(xy=np.zeros((0, 2)), cnt=np.zeros((0, 3)), a=np.zeros(0), seen=np.zeros(0), miss=np.zeros(0, int),
                        hits=np.zeros(0, int), served=np.zeros(0, bool))
        self.share = True
        self.pos = None
        self.t = 0.0

    # -- own information
    def update(self, obs, memory=True):
        t, s = obs['t'], self.store
        self.t, self.pos = t, obs['pos']
        att = obs['att']
        if len(att):
            new = np.isnan(s.rec_t[att])
            s.learn_t[att[new]] = t
            s.rec_t[att] = np.fmin(s.rec_t[att], t)
            s.rec_pos[att] = obs['att_pos']
            s.rec_tp[att] = t
            if 'att_cls' in obs:
                s.rec_act[att] = t
                s.rec_cls[att] = obs['att_cls']
        hd = obs.get('heard', np.zeros(0, int))
        if len(hd):
            new = np.isnan(s.rec_t[hd])
            s.learn_t[hd[new]] = t
            s.rec_t[hd] = np.fmin(s.rec_t[hd], obs['heard_t'])
            s.rec_pos[hd] = obs['heard_pos']
            s.rec_tp[hd] = t
            if 'heard_cls' in obs:
                s.rec_act[hd] = t
                s.rec_cls[hd] = obs['heard_cls']
        s.kpos[self.i] = self.pos
        s.kt[self.i] = t
        s.khops[self.i] = obs['hops']
        tobs_before = s.tobs.copy()
        inview = np.hypot(*(self.centres - self.pos).T) <= self.cfg.detect_range
        self._track(t, obs['det_xy'], obs['det_cls'], tobs_before, memory)
        s.tobs[inview] = t
        P = self.pts
        conf = np.where((P['hits'] >= (CONFIRM if memory else 1)) & ~P['served'])[0]
        if len(conf):
            srv = self._match_records(P['xy'][conf], max_d=12.0)
            P['served'][conf[srv]] = True                    # confident match: identity persists while tracked
        own = np.where((P['hits'] >= (CONFIRM if memory else 1)) & ~P['served'])[0]
        s.tracks[self.i] = (t, P['xy'][own].copy(), cls_of(P['cnt'][own]), P['a'][own].copy())

    def _track(self, t, dxy, dk, tobs_before, memory):
        P, r_d = self.pts, self.cfg.detect_range
        if not memory:
            P['xy'], P['cnt'] = np.zeros((0, 2)), np.zeros((0, 3))
            P['a'], P['seen'], P['miss'] = np.zeros(0), np.zeros(0), np.zeros(0, int)
            P['hits'], P['served'] = np.zeros(0, int), np.zeros(0, bool)
        M = len(P['xy'])
        inr = np.where(np.hypot(*(P['xy'] - self.pos).T) <= r_d)[0] if M else np.zeros(0, int)
        matched_p, matched_d = np.zeros(0, int), np.zeros(0, int)
        if len(inr) and len(dxy):
            C = np.hypot(P['xy'][inr, None, 0] - dxy[None, :, 0], P['xy'][inr, None, 1] - dxy[None, :, 1])
            C = np.where(C <= GATE, C, 1e6)
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < 1e6
            matched_p, matched_d = inr[r[ok]], c[ok]
        keep = np.ones(M, bool)
        if len(matched_p):
            P['xy'][matched_p] = dxy[matched_d]
            kv = dk[matched_d] >= 0
            P['cnt'][matched_p[kv], dk[matched_d][kv]] += 1
            P['seen'][matched_p] = t
            P['miss'][matched_p] = 0
            P['hits'][matched_p] += 1
        unm = np.setdiff1d(inr, matched_p)
        P['miss'][unm] += 1
        keep[unm[P['miss'][unm] >= MISS_DROP]] = False
        if M:
            keep &= P['seen'] >= t - T_MEM
        for k in P:
            P[k] = P[k][keep]
        newd = np.setdiff1d(np.arange(len(dxy)), matched_d)
        if len(newd):
            xy = dxy[newd]
            cidx = np.clip((xy[:, 0] // self.cfg.cell).astype(int), 0, self.G - 1) * self.G + \
                np.clip((xy[:, 1] // self.cfg.cell).astype(int), 0, self.G - 1)
            tb = tobs_before[cidx]
            a = np.where(np.isinf(tb), 0.0, 0.5 * (tb + t))   # never-observed cell: present since the start
            cnt = np.zeros((len(newd), 3))
            kv = dk[newd] >= 0
            cnt[np.arange(len(newd))[kv], dk[newd][kv]] = 1
            P['xy'] = np.vstack([P['xy'], xy]); P['cnt'] = np.vstack([P['cnt'], cnt])
            P['a'] = np.r_[P['a'], a]; P['seen'] = np.r_[P['seen'], np.full(len(newd), t)]
            P['miss'] = np.r_[P['miss'], np.zeros(len(newd), int)]
            P['hits'] = np.r_[P['hits'], np.ones(len(newd), int)]
            P['served'] = np.r_[P['served'], np.zeros(len(newd), bool)]

    def _match_records(self, xy, max_d=None):
        """Boolean mask: points associated (one-to-one) with a known service record.
        Gate: 10 m + 1.5 m/s since the record's last position report, capped at 60 m; max_d tightens it."""
        s, t = self.store, self.t
        out = np.zeros(len(xy), bool)
        R = np.where(~np.isnan(s.rec_t))[0]
        if not len(xy) or not len(R):
            return out
        rad = np.minimum(60.0, 10.0 + 1.5 * (t - s.rec_tp[R]))
        if max_d is not None:
            rad = np.minimum(rad, max_d)
        D = np.hypot(xy[:, None, 0] - s.rec_pos[None, R, 0], xy[:, None, 1] - s.rec_pos[None, R, 1])
        F = D <= rad[None, :]
        rows, cols = np.where(F.any(1))[0], np.where(F.any(0))[0]
        if not len(rows):
            return out
        C = np.where(F[np.ix_(rows, cols)], D[np.ix_(rows, cols)], 1e6)
        r, c = linear_sum_assignment(C)
        out[rows[r[C[r, c] < 1e6]]] = True
        return out

    def belief(self, fresh=False):
        """Believed unserved users (xy, class, a_hat): own confirmed tracks plus tracks shared by other UAVs.
        fresh=True: only tracks detected in this round (certificate scope 'view')."""
        s = self.store
        own = s.tracks.get(self.i, (0, np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)))
        if fresh:
            P = self.pts
            idx = np.where((P['hits'] >= 1) & ~P['served'] & (P['seen'] >= self.t))[0]
            xy, k, a = P['xy'][idx], cls_of(P['cnt'][idx]), P['a'][idx]
            if len(xy):
                srv = self._match_records(xy)
                xy, k, a = xy[~srv], k[~srv], a[~srv]
            return xy, k, a
        xy, k, a = [own[1]], [own[2]], [own[3]]
        if self.share:
            for j, (tj, xj, kj, aj) in s.tracks.items():
                if j == self.i or tj < self.t - T_POS or not len(xj):
                    continue
                known = np.vstack(xy)
                if len(known):
                    d = np.hypot(xj[:, None, 0] - known[None, :, 0], xj[:, None, 1] - known[None, :, 1]).min(1)
                    keep = d > DEDUP
                    xj, kj, aj = xj[keep], kj[keep], aj[keep]
                xy.append(xj); k.append(kj); a.append(aj)
        xy, k, a = np.vstack(xy), np.concatenate(k).astype(int), np.concatenate(a)
        if len(xy):
            srv = self._match_records(xy)                      # records may have arrived since the track was shared
            xy, k, a = xy[~srv], k[~srv], a[~srv]
        return xy, k, a

    def in_session(self):
        """Known users believed in session (seen attached/heard within T_ACT): positions and weights (v3)."""
        s = self.store
        idx = np.where(s.rec_act >= self.t - T_ACT)[0]
        return s.rec_pos[idx], self._w_cls(s.rec_cls[idx])

    def _w_cls(self, k):
        k = np.asarray(k, int)
        return np.where(k >= 0, self.wk[np.maximum(k, 0)], float(PRIOR_MIX @ self.wk)) if len(k) else np.zeros(0)

    def tau_map(self):
        return self.t + np.hypot(*(self.centres - self.pos).T) / self.cfg.speed

    def explore_target(self):
        tobs = self.store.tobs
        d = np.hypot(*(self.centres - self.pos).T)
        never = np.isinf(tobs)
        if never.any():
            c = np.argmin(np.where(never, d, np.inf))
        else:
            c = np.argmin(tobs + d / self.cfg.speed)      # least recently observed, nearer first
        return self.centres[c].copy()


def cochannel_distance(cfg):
    """Horizontal UAV-UAV distance beyond which a co-channel UAV cannot push a cell-edge user (distance r_s from its
    server) below the SINR threshold, with LOS path loss for both links (conservative; the user's shadowing is common to
    both links and cancels; noise neglected)."""
    h = cfg.altitude - 1.5
    d_s = math.hypot(cfg.r_s, h)
    d_i = d_s * 10 ** (cfg.gamma_db / 22.0)          # LOS exponent 2.2 (sim.pathloss)
    return cfg.r_s + math.sqrt(max(d_i ** 2 - h ** 2, 0.0))


class MAQACAgent(BaseAgent):
    """value: 'maqac' (w * S(tau - a), footprint), 'weighted' (w, footprint, no decay),
    'count' (1, footprint, no decay), 'cell' (count in the target cell only)."""
    value = 'maqac'
    prior = True
    voronoi = True
    memory = True
    seq = False
    fallback = 'explore'
    conn = True
    stay = 0.0          # stay-with-target bonus (2 Nov 2026): previous target cell's score x (1 + stay); fixed a priori at 0.1
    swap = False        # pairwise target swaps among backhaul neighbours after the sequential choice (episode.swap_pass)
    cochan = False      # co-channel exclusion (MAQAC): avoid targets within the reuse distance of higher-priority co-channel UAVs

    def __init__(self, i, cfg, world_info, rng, **flags):
        super().__init__(i, cfg, world_info, rng)
        for k, v in flags.items():
            setattr(self, k, v)
        self.kernel = disc_kernel(cfg.cell, cfg.r_s)
        self.rho0 = cfg.n_users / self.G ** 2
        self.rho_arr = cfg.arrival_rate / self.G ** 2
        self.rw_tgt = None
        self.d_cc = cochannel_distance(cfg)

    def observe(self, obs):
        self.update(obs, memory=self.memory)

    def _wS(self, k, dt):
        """w_k S_k(dt), rows = points; class -1 (not visible in images): expectation under the prior class mix."""
        if (k >= 0).all():
            return self.wk[k][:, None] * survival(self.cfg.survival, self.mu[k][:, None], dt)
        dt = np.broadcast_to(dt, (len(k), np.shape(dt)[-1]))
        kk = np.maximum(k, 0)
        out = self.wk[kk][:, None] * survival(self.cfg.survival, self.mu[kk][:, None], dt)
        u = k < 0
        out[u] = sum(PRIOR_MIX[q] * self.wk[q] * survival(self.cfg.survival, self.mu[q], dt[u]) for q in range(3))
        return out

    def _point_vals(self, xy, k, a, tau, b):
        flat, mask = footprint_cells(xy, self.G, self.cfg.cell, self.cfg.r_s)
        if self.value == 'maqac':
            v = self._wS(k, tau[flat] - a[:, None])
        elif self.value == 'weighted':
            v = np.broadcast_to(self._w_cls(k)[:, None], flat.shape).copy()
        else:
            v = np.ones(flat.shape)
        v = np.where(mask, np.maximum(v - b[:, None], 0.0), 0.0)
        return flat, v

    def _ev_class(self, tau, a):
        """Expected w S(tau - a) under the prior class mix (per cell)."""
        if self.value != 'maqac':
            return np.ones_like(tau) if self.value == 'count' else np.full_like(tau, float(PRIOR_MIX @ self.wk))
        return sum(PRIOR_MIX[k] * self.wk[k] * survival(self.cfg.survival, self.mu[k], tau - a) for k in range(3))

    def value_map(self, events=(), use_prior=None, fresh=False):
        """Estimated marginal value Delta_hat_i(e | events) for every candidate cell."""
        use_prior = self.prior if use_prior is None else use_prior
        G2 = self.G ** 2
        tau = self.tau_map()
        V = np.zeros(G2)
        xy, k, a = self.belief(fresh)
        ev = np.array(events, float).reshape(-1, 3)
        if len(xy):
            b = np.zeros(len(xy))
            if len(ev):
                cov = np.hypot(xy[:, None, 0] - ev[None, :, 0], xy[:, None, 1] - ev[None, :, 1]) <= self.cfg.r_s
                if self.value == 'maqac':
                    sv = self._wS(k, ev[None, :, 2] - a[:, None])
                else:
                    sv = np.broadcast_to((self._w_cls(k) if self.value == 'weighted' else np.ones(len(k)))[:, None], cov.shape)
                b = (sv * cov).max(1)
            if self.value == 'cell':
                cidx = np.clip((xy[:, 0] // self.cfg.cell).astype(int), 0, self.G - 1) * self.G + \
                    np.clip((xy[:, 1] // self.cfg.cell).astype(int), 0, self.G - 1)
                V += np.bincount(cidx, np.maximum(1.0 - b, 0.0), G2)
            else:
                flat, v = self._point_vals(xy, k, a, tau, b)
                V += np.bincount(flat.ravel(), v.ravel(), G2)
        if self.cfg.lambda_hold > 0 and not fresh:          # v3: keep serving known in-session users
            hx, hw = self.in_session()
            if len(hx):
                keep = np.ones(len(hx), bool)
                if len(ev):
                    keep = ~(np.hypot(hx[:, None, 0] - ev[None, :, 0], hx[:, None, 1] - ev[None, :, 1]) <= self.cfg.r_s).any(1)
                if keep.any():
                    flat, mask = footprint_cells(hx[keep], self.G, self.cfg.cell, self.cfg.r_s)
                    V += np.bincount(flat.ravel(), (mask * (self.cfg.lambda_hold * hw[keep])[:, None]).ravel(), G2)
        if use_prior:
            tobs = self.store.tobs
            never = np.isinf(tobs)
            rho = np.where(never, self.rho0, self.rho_arr * (self.t - np.where(never, 0, tobs)))
            a_c = np.where(never, 0.0, 0.5 * (self.t + np.where(never, 0, tobs)))
            pv = rho * self._ev_class(tau, a_c)
            if len(ev):
                cov = np.hypot(self.centres[:, None, 0] - ev[None, :, 0], self.centres[:, None, 1] - ev[None, :, 1]) <= self.cfg.r_s
                pb = np.zeros(G2)
                for q in range(len(ev)):
                    pb = np.where(cov[:, q], np.maximum(pb, rho * self._ev_class(np.full(G2, ev[q, 2]), a_c)), pb)
                pv = np.maximum(pv - pb, 0.0)
            V += correlate(pv.reshape(self.G, self.G), self.kernel, mode='constant').ravel()
        return V

    def _alpha(self):
        al = np.ones(self.G ** 2)
        if not self.voronoi:
            return al
        s, n = self.store, self.cfg.n_uav
        oth = [j for j in range(n) if j != self.i and s.kt[j] >= self.t - T_POS]
        if not oth:
            return al
        ds = np.hypot(*(self.centres - self.pos).T)
        do = np.min([np.hypot(*(self.centres - s.kpos[j]).T) for j in oth], axis=0)
        al[do < ds] = self.cfg.beta
        return al

    def _allowed(self):
        s, n = self.store, self.cfg.n_uav
        ok = np.ones(self.G ** 2, bool)
        if self.conn and self.cfg.conn_filter:
            R = 0.9 * self.cfg.bh_range
            ok = np.hypot(*(self.centres - self.gw).T) <= R
            for j in range(n):
                if j != self.i and s.kt[j] >= self.t - 10.0 and np.isfinite(s.khops[j]):
                    ok |= np.hypot(*(self.centres - s.kpos[j]).T) <= R
        if self.cochan:
            K = max(1, self.cfg.n_channels)
            for j in range(self.i):                 # priority by index: lower index keeps its area
                if j % K == self.i % K and s.kt[j] >= self.t - T_POS:
                    ok &= np.hypot(*(self.centres - s.kpos[j]).T) > self.d_cc
        return ok

    def decide(self, prev_events=()):
        V = self.value_map(prev_events if self.seq else ())
        score = self._alpha() * V
        allowed = self._allowed()
        score[~allowed] = -1.0
        prev = getattr(self, 'last', None)
        if self.stay > 0 and prev is not None and allowed[prev['cell']] and score[prev['cell']] > 0:
            score[prev['cell']] *= 1.0 + self.stay
        c = int(np.argmax(score))
        if score[c] > 1e-9:
            tgt = self.centres[c].copy()
        elif self.fallback == 'random':
            if self.rw_tgt is None or np.hypot(*(self.rw_tgt - self.pos)) < 10:
                self.rw_tgt = self.rng.uniform(0, self.cfg.world, 2)
            tgt, c = self.rw_tgt.copy(), None
        else:
            tgt, c = self.explore_target(), None
        if c is None:
            cc = np.clip((tgt // self.cfg.cell).astype(int), 0, self.G - 1)
            c = int(cc[0] * self.G + cc[1])
        tau = self.t + np.hypot(*(self.centres[c] - self.pos)) / self.cfg.speed
        self.last = dict(cell=c, V=V, gain=max(0.0, float(score[c])))
        return tgt, (float(self.centres[c, 0]), float(self.centres[c, 1]), float(tau))

    def cert_terms(self, own_event, events_before, all_events, use_prior=False, fresh=False):
        """Own gain Delta_hat(e_i | e_<i) and block maximum Delta_hat(. | G) for Proposition 'online certificate'.
        use_prior=False: certificate over the detected users only (scope 'obs')."""
        c = self.last['cell']
        gain = self.value_map(events_before, use_prior, fresh)[c]
        block = self.value_map(all_events, use_prior, fresh).max()
        return float(gain), float(block)


class SystematicAgent(BaseAgent):
    consensus = False
    sep = True

    def __init__(self, i, cfg, world_info, rng, **_):
        super().__init__(i, cfg, world_info, rng)
        W, n = cfg.world, cfg.n_uav
        x0, x1 = i * W / n, (i + 1) * W / n
        lane = 1.8 * cfg.r_s
        ys = np.arange(lane / 2, W, lane)
        xs = np.arange(x0 + lane / 2, x1, lane) if x1 - x0 > lane else np.array([(x0 + x1) / 2])
        wp = []
        for q, x in enumerate(xs):
            col = ys if q % 2 == 0 else ys[::-1]
            wp += [(x, y) for y in col]
        self.wp, self.k = np.array(wp), 0

    def observe(self, obs):
        self.t, self.pos = obs['t'], obs['pos']

    def decide(self, prev_events=()):
        if np.hypot(*(self.wp[self.k] - self.pos)) < 5.0:
            self.k = (self.k + 1) % len(self.wp)
        tgt = self.wp[self.k].copy()
        return tgt, (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))


class StaticAgent(BaseAgent):
    """Reference without coordination or sensing: 'pad' = hover above the take-off pad; 'grid' = fly to a fixed
    lattice point (the n-point square-ish lattice over the area, classic static UAV-BS layout) and hover there."""
    consensus = False
    sep = True
    layout = 'grid'

    def __init__(self, i, cfg, world_info, rng, **flags):
        super().__init__(i, cfg, world_info, rng)
        for k, v in flags.items():
            setattr(self, k, v)
        n, W = cfg.n_uav, cfg.world
        nx = int(math.ceil(math.sqrt(n))); ny = int(math.ceil(n / nx))
        pts = [((qx + 0.5) * W / nx, (qy + 0.5) * W / ny) for qy in range(ny) for qx in range(nx)][:n]
        self.home = np.array(pts[i], float)

    def observe(self, obs):
        self.t, self.pos = obs['t'], obs['pos']
        if self.layout == 'pad' and not hasattr(self, 'pad0'):
            self.pad0 = obs['pos'].copy()

    def decide(self, prev_events=()):
        tgt = self.pad0.copy() if self.layout == 'pad' else self.home.copy()
        return tgt, (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))


METHODS = {
    'static_pad':     (StaticAgent, dict(layout='pad'), False, True),
    'static_grid':    (StaticAgent, dict(layout='grid'), False, True),
    # name: (class, flags, uses consensus, separation)
    # MAQAC shares service records and positions but NOT raw detections (sharing detections causes herding,
    # pilot 2026-10-29: -13 to -20 %); 'maqac_share' keeps detection sharing as an ablation.
    'maqac':          (MAQACAgent, dict(share=False, cochan=True), True, True),
    'maqac_seq':      (MAQACAgent, dict(seq=True, share=False, cochan=True), True, True),
    'maqac_nocc':     (MAQACAgent, dict(share=False), True, True),             # without co-channel exclusion (v2 grid rows)
    'maqac_share':    (MAQACAgent, dict(cochan=True), True, True),
    'maqac_nocons':   (MAQACAgent, dict(share=False, cochan=True), False, True),
    'maqac_hard':     (MAQACAgent, dict(share=False, cochan=True), True, True),            # beta = 0 via config
    'maqac_noconn':   (MAQACAgent, dict(conn=False, share=False, cochan=True), True, True),
    'maqac_noqos':    (MAQACAgent, dict(value='count', share=False, cochan=True), True, True),
    'maqac_noprior':  (MAQACAgent, dict(prior=False, share=False, cochan=True), True, True),
    'greedy':         (MAQACAgent, dict(value='cell', prior=False, voronoi=False, memory=False, conn=False, fallback='random'), False, False),
    'greedy_info':    (MAQACAgent, dict(value='weighted', prior=False, voronoi=False, conn=False, share=False), True, True),  # same information as MAQAC
    'systematic':     (SystematicAgent, dict(), False, True),
    # RAG (Xu, Garimella & Tzoumas, IEEE T-RO 41, 2025): same objective and information as MAQAC, no Voronoi weight;
    # coordination by RAG's neighbourhood greedy rule over the live backhaul links (run in episode.py)
    'rag':            (MAQACAgent, dict(seq=True, voronoi=False, share=False), True, True),
    # RAG-rule variants (test of 2 Nov 2026, scripts/maqac_rag_test.py): coordination by rag_round in episode.py
    # maqac_rag: MAQAC's value map and co-channel exclusion under RAG's neighbourhood rounds
    # rag_count: RAG's rounds with a plain objective (detected users in the footprint; no QoS decay, no prior)
    'maqac_rag':      (MAQACAgent, dict(seq=True, voronoi=False, share=False, cochan=True), True, True),
    'rag_count':      (MAQACAgent, dict(seq=True, voronoi=False, share=False, value='count', prior=False), True, True),
}
RAG_RULE = {'rag', 'maqac_rag', 'rag_count'}      # methods coordinated by rag_round
# MAQAC-hyb (2 Nov 2026): MAQAC's simultaneous choice with its soft partition; UAVs whose tentative choices conflict
# re-decide by RAG's rule among themselves (episode.hybrid_round)
METHODS['maqac_hyb'] = (MAQACAgent, dict(seq=True, share=False, cochan=True), True, True)
HYB_RULE = {'maqac_hyb'}
# 2 Nov 2026 (user's choice "try both"): settings fixed before any run, screened on tuning seeds 900-904 only.
# maqac_swap: sequential MAQAC + one pass of pairwise target swaps among backhaul neighbours (accept if the pair's
# estimated value rises by > 1 %); never below the sequential choice in estimated value. *_stay: stay bonus 0.1.
# rag_stay gives RAG the same stay bonus, so the comparison stays like for like.
METHODS['maqac_swap'] = (MAQACAgent, dict(seq=True, share=False, cochan=True, swap=True), True, True)
METHODS['maqac_seq_stay'] = (MAQACAgent, dict(seq=True, share=False, cochan=True, stay=0.1), True, True)
METHODS['maqac_swap_stay'] = (MAQACAgent, dict(seq=True, share=False, cochan=True, swap=True, stay=0.1), True, True)
METHODS['rag_stay'] = (MAQACAgent, dict(seq=True, voronoi=False, share=False, stay=0.1), True, True)
RAG_RULE.add('rag_stay')
