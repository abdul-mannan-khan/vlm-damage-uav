"""Distributed QoE-aware user association + UAV deployment after He, Liu & Xie, IEEE TNSM 22(5):4394-4406, 2025
(doi 10.1109/TNSM.2025.3572890). Implementation notes: ~/lit_results/he2025_tnsm_method.txt (from the full text).

Per 1 s round (their method re-runs on events; here every round, warm-started from the nearest-UAV association):
 1. Users: the users known to the deciding UAVs (own confirmed tracks + shared tracks + known in-session users, the
    same information as MAQAC, IAD and the potential game), pooled and merged within 12 m (one user per report).
 2. Association (their Alg. 1, coalition formation with switch and swap, p_switch = 0.5): each UAV in turn picks a
    random one-hop backhaul neighbour; a switch hands one random user over, a swap exchanges one each; kept only if
    the two UAVs' summed QoE strictly rises. One attempt = 4 link messages (random number, hand-over, QoE sums,
    confirm/undo); if any of them is lost (World.msg_lost, the same loss process as RAG's) the attempt is void.
    Stop after X1 = 300 passes over the UAVs or 2M consecutive attempts without a change.
 3. Deployment (their Alg. 2, per-UAV PSO, w = 0.729, c = 1.4962): each UAV maximises the QoE of its own users,
    20 particles x 50 iterations, no messages; the result is moved to the nearest cell the UAV may use (backhaul
    connectivity filter, as for every distributed method here).
QoE (their eq. 6) f = 1 / (1 + exp(-lambda (R - R'))) with R = (B / N_m) log2(1 + SNR) in Mbps (Shannon; the printed
eq. 5 has no log2), lambda = the class weight (3/2/1, their urgency set {1,2,3}), R' = the class's minimum rate.
Choices stated in the paper: A = 0 instead of 7 (with A = 7 every user has f >= 0.73 even at zero rate, which makes
the association meaningless; A = 0 puts f = 0.5 at the demand); f = 0 beyond the service radius r_s (our admission
rule); SNR from the same mean A2G path loss as IAD (no interference: their FDMA model).
"""
import math
import time
import numpy as np
from agents import MAQACAgent, METHODS, PRIOR_MIX, T_ACT
from baselines_interf import _InterfMixin

P_SWITCH, X1, NP, X2 = 0.5, 300, 20, 50
W_PSO, C_PSO = 0.729, 1.4962
MERGE = 12.0


class He2025Agent(_InterfMixin, MAQACAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._setup()

    def users_cls(self):
        """Known users with class weight (lambda) and demand (Mbps)."""
        xy, k, _ = self.belief()
        s = self.store
        idx = np.where(s.rec_act >= self.t - T_ACT)[0]
        xy = np.vstack([xy.reshape(-1, 2), s.rec_pos[idx].reshape(-1, 2)])
        k = np.r_[np.asarray(k, int), np.asarray(s.rec_cls[idx], int)]
        lam = self._w_cls(k) if len(k) else np.zeros(0)
        rmin = np.array(self.cfg.rmin_mbps, float)
        dem = np.where(k >= 0, rmin[np.maximum(k, 0)], float(PRIOR_MIX @ rmin)) if len(k) else np.zeros(0)
        return xy, lam, dem


def _qoe_mat(ag, uxy, lam, dem, pos, N):
    """f (users x UAVs) when each user is served by UAV m with N[m] users sharing its band."""
    rx, d2 = ag._rx(uxy, pos)                                   # mW, horizontal distance
    snr = rx / ag.noise
    R = (ag.cfg.bw_mhz / np.maximum(N, 1))[None, :] * np.log2(1.0 + snr)
    f = 1.0 / (1.0 + np.exp(-lam[:, None] * (R - dem[:, None])))
    return np.where(d2 <= ag.cfg.r_s, f, 0.0)


def _F(ag, uxy, lam, dem, p, members):
    """Their F_m: summed QoE of UAV m's users if m is at p (vectorised over candidate positions p: P x 2)."""
    if not len(members):
        return np.zeros(len(p))
    rx, d2 = ag._rx(uxy[members], p)                            # |G| x P
    R = (ag.cfg.bw_mhz / len(members)) * np.log2(1.0 + rx / ag.noise)
    f = 1.0 / (1.0 + np.exp(-lam[members, None] * (R - dem[members, None])))
    return np.where(d2 <= ag.cfg.r_s, f, 0.0).sum(0)


def he_round(agents, deciding, edges, W, rng, targets):
    stats = dict(iters=0, msgs=0, cpu={i: 0.0 for i in deciding})
    if not deciding:
        return targets, [], stats
    t0 = time.perf_counter()
    ag0 = agents[deciding[0]]
    pools = [agents[i].users_cls() for i in deciding]
    xy = np.vstack([p[0] for p in pools]); lam = np.concatenate([p[1] for p in pools]); dem = np.concatenate([p[2] for p in pools])
    keep = []
    for u in range(len(xy)):                                    # merge reports of the same user
        if not keep or np.hypot(*(xy[keep] - xy[u]).T).min() > MERGE:
            keep.append(u)
    xy, lam, dem = xy[keep], lam[keep], dem[keep]
    M, U = len(deciding), len(xy)
    pos = np.array([agents[i].pos for i in deciding], float)
    nb = {q: [] for q in range(M)}
    ix = {i: q for q, i in enumerate(deciding)}
    for (u, v) in edges:
        if u in ix and v in ix and u != v and ix[u] not in nb[ix[v]]:
            nb[ix[v]].append(ix[u]); nb[ix[u]].append(ix[v])
    if U:
        d = np.hypot(xy[:, None, 0] - pos[None, :, 0], xy[:, None, 1] - pos[None, :, 1])
        a = d.argmin(1)                                         # warm start: nearest UAV
        G = [list(np.where(a == q)[0]) for q in range(M)]
        rx, d2 = ag0._rx(xy, pos)
        lse = np.log2(1.0 + rx / ag0.noise); inr = d2 <= ag0.cfg.r_s

        def Fq(q, mem):
            if not mem:
                return 0.0
            R = (ag0.cfg.bw_mhz / len(mem)) * lse[mem, q]
            return float((inr[mem, q] / (1.0 + np.exp(-lam[mem] * (R - dem[mem])))).sum())

        ge = {}

        def lost(u, v):             # i.i.d.: per message; Gilbert-Elliott: one state per link and round (bursts ~5 s)
            if W.cfg.loss_model != 'ge':
                return W.msg_lost(u, v)
            if (u, v) not in ge:
                ge[(u, v)] = W.msg_lost(u, v)
            return ge[(u, v)]

        Fv = [Fq(q, G[q]) for q in range(M)]
        idle = 0
        for it in range(X1):
            stats['iters'] += 1
            for q in rng.permutation(M):
                if not nb[q]:
                    idle += 1
                    continue
                q2 = nb[q][rng.integers(len(nb[q]))]
                stats['msgs'] += 4
                if any(lost(deciding[x], deciding[y]) for x, y in ((q, q2), (q, q2), (q2, q), (q, q2))):
                    idle += 1
                    continue
                before = Fv[q] + Fv[q2]
                if rng.random() < P_SWITCH:
                    if not G[q]:
                        idle += 1; continue
                    n_ = G[q][rng.integers(len(G[q]))]
                    g1 = [x for x in G[q] if x != n_]; g2 = G[q2] + [n_]
                else:
                    if not G[q] or not G[q2]:
                        idle += 1; continue
                    n_ = G[q][rng.integers(len(G[q]))]; n2 = G[q2][rng.integers(len(G[q2]))]
                    g1 = [x for x in G[q] if x != n_] + [n2]; g2 = [x for x in G[q2] if x != n2] + [n_]
                f1, f2 = Fq(q, g1), Fq(q2, g2)
                if f1 + f2 > before + 1e-12:
                    G[q], G[q2], Fv[q], Fv[q2] = g1, g2, f1, f2
                    idle = 0
                else:
                    idle += 1
            if idle >= 2 * M:
                break
    else:
        G = [[] for _ in range(M)]
    t_assoc = time.perf_counter() - t0
    events = []
    L = float(ag0.cfg.world)
    for q, i in enumerate(deciding):
        tq = time.perf_counter()
        ag = agents[i]
        mem = np.array(G[q], int)
        if not len(mem):
            tgt, ev = ag._fallback()
        else:
            vmax = 0.15 * L
            P = rng.uniform(0, L, (NP, 2)); P[0] = ag.pos            # one particle at the current position
            V = rng.uniform(-vmax, vmax, (NP, 2))
            PB, FB = P.copy(), _F(ag, xy, lam, dem, P, mem)
            gb = PB[FB.argmax()].copy()
            for _ in range(X2):
                V = W_PSO * V + C_PSO * rng.random((NP, 2)) * (PB - P) + C_PSO * rng.random((NP, 2)) * (gb - P)
                V = np.clip(V, -vmax, vmax)
                P = np.clip(P + V, 0, L)
                Fp = _F(ag, xy, lam, dem, P, mem)
                up = Fp > FB
                PB[up], FB[up] = P[up], Fp[up]
                gb = PB[FB.argmax()].copy()
            ok = ag._allowed()
            cells = np.where(ok)[0] if ok.any() else np.arange(len(ag.centres))
            c = int(cells[np.hypot(*(ag.centres[cells] - gb).T).argmin()])
            tgt, ev = ag._event(c)
        targets[i] = tgt
        events.append(ev)
        stats['cpu'][i] = time.perf_counter() - tq + t_assoc / M
    return targets, events, stats


METHODS['he2025'] = (He2025Agent, dict(value='weighted', prior=False, voronoi=False, share=False), True, True)
HE_RULE = {'he2025'}
