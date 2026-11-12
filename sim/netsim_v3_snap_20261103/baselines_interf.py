"""Interference-aware deployment baselines for the network-level simulator.

Both use the same information as MAQAC (own detections, shared service records, neighbour positions received within
T_POS) and the same backhaul-connectivity filter, and score users by priority weight w (no survival term, no prior).
SINR is estimated with the line-of-sight path loss of sim.pathloss, the same assumption as MAQAC's reuse distance.

* IADAgent  - interference-aware deployment after Lai, Tsai & Ting, IEEE WCL 12(7), 2023 (doi 10.1109/LWC.2023.3266011):
  each UAV moves to the cell that maximises the weighted number of users it would serve with SINR >= gamma, counting
  only users not already within the footprint of another UAV (reduced coverage overlap). Simultaneous decisions.
* PotentialGameAgent - distributed deployment as a potential game after Furuta, Kimura & Takine, IEEE PIMRC 2024
  (doi 10.1109/PIMRC59610.2024.10817333): utility = marginal contribution of the UAV to the fleet's weighted number of
  satisfied users (wonderful-life utility, so the game is an exact potential game with the fleet objective as
  potential), including the interference it causes to other UAVs' users; log-linear learning over a local action set
  (cells within R_LOCAL of its current target) with a decreasing temperature, which converges in probability to a
  potential maximiser.
"""
import math
import numpy as np
from sim import pathloss, plos_a2g, pathloss_a2g
from agents import MAQACAgent, METHODS, T_POS

R_LOCAL = 200.0        # m, local action set of the potential game
# P_UPDATE and TAU0 were tuned on seeds 900-903 (disjoint from all evaluation seeds): P_UPDATE 0.2/0.5/1.0 x TAU0
# 2/0.5/0.1/0.02 -> best 1.0 / 0.1 (J 223 vs 189 for 0.2 / 2.0, urban 6 UAVs, 300 s).
P_UPDATE = 1.0         # probability that a UAV revises its action in a round (asynchronous log-linear learning)
TAU0 = 0.1             # initial temperature; tau = max(0.05, TAU0 / log(2 + t/10))


class _InterfMixin:
    def _setup(self):
        c = self.cfg
        self.noise = 10 ** ((-174 + 10 * math.log10(c.bw_mhz * 1e6) + c.nf_db) / 10)
        self.gam = 10 ** (c.gamma_db / 10)
        self.K = max(1, c.n_channels)

    def _rx(self, uxy, pxy):
        """Received power (mW) at users uxy from transmitters pxy, LOS path loss."""
        d2 = np.hypot(uxy[:, None, 0] - pxy[None, :, 0], uxy[:, None, 1] - pxy[None, :, 1])
        d3 = np.sqrt(d2 ** 2 + (self.cfg.altitude - 1.5) ** 2)
        if self.cfg.chan == 'a2g':                      # v3: mean A2G path loss (P_LOS-weighted excess loss)
            env = self.cfg.a2g_env or {'urban': 'urban', 'disaster': 'urban', 'suburban': 'suburban'}[self.cfg.scenario]
            p = plos_a2g(d2, self.cfg.altitude, env)
            pl = p * pathloss_a2g(d3, True, self.cfg.f_ghz, env) + (1 - p) * pathloss_a2g(d3, False, self.cfg.f_ghz, env)
            return 10 ** ((self.cfg.tx_dbm - pl) / 10), d2
        return 10 ** ((self.cfg.tx_dbm - pathloss(d3, True, self.cfg.f_ghz)) / 10), d2

    def _others(self):
        s, n = self.store, self.cfg.n_uav
        J = [j for j in range(n) if j != self.i and s.kt[j] >= self.t - T_POS]
        return np.array([s.kpos[j] for j in J]).reshape(-1, 2), np.array([j % self.K for j in J], int)

    def _users(self):
        """Known users to serve: believed unserved users, plus (v3) known in-session users, whom IAD and the
        potential game keep serving as ordinary users (their objective counts every served user)."""
        xy, k, _ = self.belief()
        w = self._w_cls(k) if len(k) else np.zeros(0)
        if any(m > 0 for m in self.cfg.session_mean):
            hx, hw = self.in_session()
            if len(hx):
                xy, w = np.vstack([xy.reshape(-1, 2), hx]), np.r_[w, hw]
        return xy, w

    def _terms(self, uxy, cand):
        """Per candidate (C) x user (M): eligibility and SINR when served by self at the candidate (A), and the best
        other server's SINR with (B) and without (B0) this UAV's interference."""
        P, ch = self._others()
        r_s, mych = self.cfg.r_s, self.i % self.K
        rxc, dc = self._rx(uxy, cand)                          # M x C
        rxc, dc = rxc.T, dc.T                                  # C x M
        M = len(uxy)
        if len(P):
            rxo, do = self._rx(uxy, P)                         # M x J
            Ico = np.stack([(rxo * (ch == k)).sum(1) for k in range(self.K)], 1)     # M x K, co-channel sums
            elig_o = do <= r_s
            sinr_o = np.where(elig_o, rxo / (Ico[:, ch] - rxo + self.noise), 0.0)  # M x J (without this UAV)
            best = sinr_o.argmax(1) if P.shape[0] else np.zeros(M, int)
            B_elig = elig_o.any(1)
            rx_s = rxo[np.arange(M), best]
            I_s = Ico[np.arange(M), ch[best]] - rx_s
            B0 = np.where(B_elig, rx_s / (I_s + self.noise), 0.0)
            same = (ch[best] == mych)[None, :]
            B = np.where(B_elig[None, :], rx_s[None, :] / (I_s[None, :] + same * rxc + self.noise), 0.0)
            I_mine = Ico[:, mych][None, :]
        else:
            B_elig = np.zeros(M, bool); B0 = np.zeros(M); B = np.zeros((len(cand), M)); I_mine = np.zeros((1, M))
        A_elig = dc <= r_s
        A = np.where(A_elig, rxc / (I_mine + self.noise), 0.0)
        return A_elig, A, B_elig, B, B0

    def _fallback(self):
        tgt = self.explore_target()
        return tgt, (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))

    def _event(self, c):
        tgt = self.centres[c].copy()
        return tgt, (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))


class IADAgent(_InterfMixin, MAQACAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._setup()

    def decide(self, prev_events=()):
        uxy, w = self._users()
        ok = self._allowed()
        if not len(uxy):
            return self._fallback()
        near = np.zeros(self.G ** 2, bool)                     # candidates: cells whose footprint holds a known user
        d = np.hypot(self.centres[:, None, 0] - uxy[None, :, 0], self.centres[:, None, 1] - uxy[None, :, 1])
        near = (d <= self.cfg.r_s).any(1) & ok
        cand = np.where(near)[0]
        if not len(cand):
            return self._fallback()
        A_elig, A, B_elig, _, _ = self._terms(uxy, self.centres[cand])
        score = ((A_elig & (A >= self.gam) & ~B_elig[None, :]) * w[None, :]).sum(1)
        if score.max() <= 0:
            return self._fallback()
        return self._event(int(cand[int(np.argmax(score))]))


class PotentialGameAgent(_InterfMixin, MAQACAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._setup()
        self.cur = None

    def decide(self, prev_events=()):
        uxy, w = self._users()
        if not len(uxy):
            self.cur = None
            return self._fallback()
        if self.cur is not None and self.rng.random() > P_UPDATE:
            return self._event(self.cur)
        ok = self._allowed()
        ref = self.centres[self.cur] if self.cur is not None else self.pos
        local = (np.hypot(*(self.centres - ref).T) <= R_LOCAL) & ok
        if self.cur is not None:
            local[self.cur] = True
        cand = np.where(local)[0]
        if not len(cand):
            return self._fallback()
        A_elig, A, B_elig, B, B0 = self._terms(uxy, self.centres[cand])
        with_i = np.maximum(np.where(A_elig, A, 0.0), B) >= self.gam      # C x M
        without = (B0 >= self.gam)[None, :]
        U = ((with_i.astype(float) - without) * w[None, :]).sum(1)         # marginal contribution (wonderful life)
        tau = max(0.05, TAU0 / math.log(2.0 + self.t / 10.0))
        z = (U - U.max()) / tau
        p = np.exp(z); p /= p.sum()
        if U.max() <= 0 and self.cur is None:
            return self._fallback()
        self.cur = int(cand[self.rng.choice(len(cand), p=p)])
        return self._event(self.cur)


METHODS['iad'] = (IADAgent, dict(value='weighted', prior=False, voronoi=False, share=False), True, True)
METHODS['potgame'] = (PotentialGameAgent, dict(value='weighted', prior=False, voronoi=False, share=False), True, True)
