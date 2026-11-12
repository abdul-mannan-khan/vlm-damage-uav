"""One episode of the network-level simulator: world + agents + backhaul consensus + certificate logging."""
import time
from dataclasses import asdict
import numpy as np

from sim import World, PERCEPTION, survival
from agents import METHODS, RAG_RULE, HYB_RULE, Store
import baselines_interf  # noqa: F401  (registers the interference-aware baselines)
from baseline_he2025 import HE_RULE, he_round   # registers he2025 (He, Liu & Xie, TNSM 2025)

CENTRAL = 'central'   # full-information sequential greedy (reference, not deployable)


def relay_hops(edges, N):
    """All-pairs hop counts on the backhaul graph (nodes 0..n-1 UAVs, n = gateway); inf when unreachable."""
    from collections import deque
    nb = [[] for _ in range(N)]
    for u, v in edges:
        nb[u].append(v)
    D = np.full((N, N), np.inf)
    for s in range(N):
        D[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if np.isinf(D[s, v]):
                    D[s, v] = D[s, u] + 1; q.append(v)
    return D


def rag_round(agents, deciding, edges, n, p_loss, rng, targets, lost=None):
    """Resource-Aware distributed Greedy (RAG), Alg. 1 of Xu, Garimella & Tzoumas, IEEE T-RO 41 (2025).
    Each undecided UAV computes its best action given the actions already selected by its in-neighbours; it commits
    when its marginal gain is the largest among its undecided in-neighbours (ties: lower index). In-neighbours are the
    one-hop backhaul links of this round; each link is lost for the round with probability p_loss."""
    D = set(deciding)
    nb = {i: set() for i in deciding}
    for (u, v) in edges:                       # v receives from u
        if u in D and v in D and u != v and not (lost(u, v) if lost is not None else (p_loss > 0 and rng.random() < p_loss)):
            nb[v].add(u)
    sel = {}                                   # i -> event
    todo = list(deciding)
    out = {i: [v for v in deciding if i in nb[v]] for i in deciding}
    stats = dict(iters=0, msgs=0, cpu={i: 0.0 for i in deciding})
    while todo:
        cand = {}
        stats['iters'] += 1
        for i in todo:
            known = [sel[j] for j in nb[i] if j in sel]
            t0 = time.perf_counter()
            tgt, ev = agents[i].decide(known)
            stats['cpu'][i] += time.perf_counter() - t0
            cand[i] = (agents[i].last['gain'], tgt, ev)
            stats['msgs'] += sum(1 for v in out[i] if v not in sel)          # gain to undecided out-neighbours
        commit = [i for i in todo
                  if all((cand[i][0], -i) > (cand[j][0], -j) for j in nb[i] if j in cand)]
        for i in commit:
            targets[i], sel[i] = cand[i][1], cand[i][2]      # agent state already holds this choice (last decide call)
            stats['msgs'] += len(out[i])                                       # broadcast of the committed action
        todo = [i for i in todo if i not in sel]
    return targets, [sel[i] for i in deciding], stats


def hybrid_round(agents, deciding, edges, n, p_loss, rng, targets, lost=None, eps=0.01):
    """Conflict-triggered turn-taking (MAQAC-hyb, 2 Nov 2026). (1) Every deciding UAV chooses simultaneously, as in
    MAQAC, and sends its tentative choice to its one-hop backhaul neighbours (counted; each link lost with p_loss).
    (2) UAV i is in conflict if a received tentative choice e_j lowers the estimated value of its own choice by more
    than eps: Delta_hat_i(e_i | e_j) < (1 - eps) Delta_hat_i(e_i), the observable counterpart of the overlap
    multiplicity m > 1 of Theorem 4. (3) UAVs without conflict keep their choice; the conflicted ones re-decide by
    RAG's rule among themselves, each knowing the committed choices of its neighbours. No conflict: one broadcast."""
    D = set(deciding)
    nb = {i: set() for i in deciding}
    for (u, v) in edges:                       # v receives from u
        if u in D and v in D and u != v and not (lost(u, v) if lost is not None else (p_loss > 0 and rng.random() < p_loss)):
            nb[v].add(u)
    out = {i: [v for v in deciding if i in nb[v]] for i in deciding}
    stats = dict(iters=1, msgs=0, cpu={i: 0.0 for i in deciding}, conflicted=0)
    tent = {}
    for i in deciding:
        t0 = time.perf_counter()
        tgt, ev = agents[i].decide(())
        stats['cpu'][i] += time.perf_counter() - t0
        tent[i] = (tgt, ev, agents[i].last['cell'], float(agents[i].last['V'][agents[i].last['cell']]))
        stats['msgs'] += len(out[i])                                  # tentative choice to the out-neighbours
    conf = set()
    for i in deciding:
        c, v0 = tent[i][2], tent[i][3]
        if v0 <= 1e-9:
            continue
        t0 = time.perf_counter()
        for j in nb[i]:
            if agents[i].value_map([tent[j][1]])[c] < (1 - eps) * v0:
                conf.add(i); break
        stats['cpu'][i] += time.perf_counter() - t0
    sel = {}
    for i in deciding:
        if i not in conf:
            targets[i], sel[i] = tent[i][0], tent[i][1]
    stats['conflicted'] = len(conf)
    todo = sorted(conf)
    while todo:                                                       # RAG's rule among the conflicted UAVs
        cand = {}
        stats['iters'] += 1
        for i in todo:
            known = [sel[j] for j in nb[i] if j in sel]
            t0 = time.perf_counter()
            tgt, ev = agents[i].decide(known)
            stats['cpu'][i] += time.perf_counter() - t0
            cand[i] = (agents[i].last['gain'], tgt, ev)
            stats['msgs'] += sum(1 for v in out[i] if v in conf and v not in sel)
        commit = [i for i in todo if all((cand[i][0], -i) > (cand[j][0], -j) for j in nb[i] if j in cand)]
        for i in commit:
            targets[i], sel[i] = cand[i][1], cand[i][2]
            stats['msgs'] += len(out[i])
        todo = [i for i in todo if i not in sel]
    return targets, [sel[i] for i in deciding], stats


def swap_pass(agents, deciding, events, targets, edges, lost, eps=0.01):
    """One pass of pairwise target swaps after the sequential choice (MAQAC-swap, 2 Nov 2026). For each backhaul
    neighbour pair (i, j), i < j, both UAVs know all chosen events (one extra relay down the chain). With E' the events
    of the other UAVs, the pair's estimated value is Delta_i(e_i | E') + Delta_j(e_j | E' + e_i); the swap gives i the
    cell of j and j the cell of i (each with its own arrival time). The swap is kept if the pair value rises by more than
    eps and both cells are allowed; total estimated value then never falls (f(E) = f(E') + pair value)."""
    pos = {i: q for q, i in enumerate(deciding)}
    D = set(deciding)
    pairs = sorted({(min(u, v), max(u, v)) for (u, v) in edges if u in D and v in D and u != v
                    and not (lost is not None and (lost(u, v) or lost(v, u)))})
    st = dict(msgs=max(len(deciding) - 1, 0), swaps=0)      # relay of the complete choice list
    for (i, j) in pairs:
        ai, aj = agents[i], agents[j]
        rest = [events[pos[k]] for k in deciding if k not in (i, j)]
        ci, cj = ai.last['cell'], aj.last['cell']
        if ci == cj:
            continue
        st['msgs'] += 2                                       # each tells the other its values
        ei, ej = events[pos[i]], events[pos[j]]
        cur = ai.value_map(rest)[ci] + aj.value_map(rest + [ei])[cj]
        if not (ai._allowed()[cj] and aj._allowed()[ci]):
            continue
        ti = ai.t + np.hypot(*(ai.centres[cj] - ai.pos)) / ai.cfg.speed
        tj = aj.t + np.hypot(*(aj.centres[ci] - aj.pos)) / aj.cfg.speed
        ni = (float(ai.centres[cj, 0]), float(ai.centres[cj, 1]), float(ti))
        nj = (float(aj.centres[ci, 0]), float(aj.centres[ci, 1]), float(tj))
        new = ai.value_map(rest)[cj] + aj.value_map(rest + [ni])[ci]
        if new > (1 + eps) * cur and new > 1e-9:
            events[pos[i]], events[pos[j]] = ni, nj
            targets[i], targets[j] = ai.centres[cj].copy(), aj.centres[ci].copy()
            ai.last['cell'], aj.last['cell'] = cj, ci
            st['swaps'] += 1
            st['msgs'] += 2                                   # both announce the new choice
    return st


def run_episode(cfg, method, seed, log_rounds=True, step_hook=None, agent_flags=None, move=None, setup=None):
    """move(W, targets, sep, t), if given, replaces the kinematic UAV step W.move_uavs (e.g. PX4 SITL in AirSim).
    setup(W), if given, runs once after the world is built (e.g. the AirSim bridge's height-map LOS and user snapping)."""
    t0 = time.time()
    W = World(cfg, seed)
    if setup is not None:
        setup(W)
    sixdof = None
    if move is None and getattr(cfg, 'flight', 'kinematic') == 'sixdof':
        from sixdof import SixDofFleet
        move = sixdof = SixDofFleet(cfg)
    n = cfg.n_uav
    info = dict(G=W.G, N=W.N, centres=W.centres)
    central = method == CENTRAL
    if central:
        agents, uses_cons, sep = None, False, [True] * n
    else:
        cls, flags, uses_cons, s = METHODS[method]
        if method == 'maqac_hard':
            from dataclasses import replace
            cfg = replace(cfg, beta=0.0)
        if agent_flags:                        # e.g. the learned policy for 'mappo'
            flags = {**flags, **agent_flags}
        agents = [cls(i, cfg, info, np.random.default_rng([seed, 50 + i]), **flags) for i in range(n)]
        sep = [s] * n
    gw_store = Store(W.N, W.G, n + 1)
    offloading = cfg.perception == 'offload'
    rounds = []
    coord = []                                    # per decision round: (sequential iterations, coordination messages, max per-UAV decide time s)
    for step in range(cfg.T):
        t = float(step)
        W.update_failures(t)
        W.move_users(t)
        bh = W.backhaul(offloading)
        air = W.airborne()
        W.m['disconnected_s'] += int((air & ~bh['connected']).sum())
        hf = bh['hops'][:n][air & bh['connected']]
        W.m['hops_sum'] += float(hf.sum()); W.m['hops_n'] += len(hf)
        # Phase 1: flooding over the backhaul graph, one hop per round, with message loss
        if agents is not None and uses_cons:
            nodes = [a.store for a in agents] + [gw_store]
            snaps = [s.snapshot() for s in nodes]
            for (u, v) in bh['edges']:                  # v receives u's snapshot
                if W.msg_lost(u, v):
                    continue
                nodes[v].merge(snaps[u], t)
        W.access(t)
        # perception + own information
        if agents is not None:
            for i, a in enumerate(agents):
                if not W.alive[i]:
                    continue
                if cfg.perception == 'offload':
                    if bh['connected'][i]:
                        pre, lat = PERCEPTION['offload'], bh['off_lat'][i]
                        W.m['offload_lat_sum'] += lat; W.m['offload_n'] += 1
                    else:
                        pre, lat = PERCEPTION['onboard'], PERCEPTION['onboard']['lat']   # fallback on board
                else:
                    pre = PERCEPTION[cfg.perception]
                    lat = pre['lat']
                a.observe(W.observe(i, t, pre, lat))
        # decisions of UAVs in mission mode
        deciding = [i for i in range(n) if W.alive[i] and W.mode[i] == 0]
        targets = [W.pad[i].copy() if W.alive[i] and W.mode[i] in (1, 2) else W.upos[i].copy() for i in range(n)]
        events = []
        if central:
            for i in deciding:
                V = W.true_map(t, i, events, cfg.survival)
                c = int(np.argmax(V))
                tgt = W.centres[c].copy() if V[c] > 0 else W.upos[i].copy()
                targets[i] = tgt
                events.append((tgt[0], tgt[1], t + np.hypot(*(tgt - W.upos[i])) / cfg.speed))
        elif method in RAG_RULE:
            targets, events, st = rag_round(agents, deciding, bh['edges'], n, cfg.p_loss, W.rl, targets, W.msg_lost)
            if deciding:
                coord.append((st['iters'], st['msgs'], max(st['cpu'].values())))
        elif method in HE_RULE:
            targets, events, st = he_round(agents, deciding, bh['edges'], W, W.rl, targets)
            if deciding:
                coord.append((st['iters'], st['msgs'], max(st['cpu'].values())))
        elif method in HYB_RULE:
            targets, events, st = hybrid_round(agents, deciding, bh['edges'], n, cfg.p_loss, W.rl, targets, W.msg_lost)
            if deciding:
                coord.append((st['iters'], st['msgs'], max(st['cpu'].values())))
                W.m['conflicted_sum'] = W.m.get('conflicted_sum', 0) + st['conflicted']
                W.m['conflicted_rounds'] = W.m.get('conflicted_rounds', 0) + int(st['conflicted'] > 0)
        else:
            cpu = []
            seq0 = getattr(agents[0], 'seq', False)
            lossy = seq0 and (cfg.p_loss > 0)          # 2 Nov 2026: in-turn hand-overs face the same message loss as RAG's
            relay = seq0 and getattr(cfg, 'seq_relay', False)
            if relay:
                hd = relay_hops(bh['edges'], n + 1)
                relay_msgs = sum((hd[a][b] if np.isfinite(hd[a][b]) else 0) for a, b in zip(deciding[:-1], deciding[1:]))
            for q, i in enumerate(deciding):
                if relay:    # the list reaches i from every earlier decider in its backhaul component
                    known = [ev for j, ev in zip(deciding[:q], events) if np.isfinite(hd[j][i])]
                else:
                    known = [ev for j, ev in zip(deciding[:q], events) if not W.msg_lost(j, i)] if lossy else events
                tc = time.perf_counter()
                tgt, ev = agents[i].decide(known)
                cpu.append(time.perf_counter() - tc)
                targets[i] = tgt
                events.append(ev)
            if deciding:     # simultaneous rule: one decision per UAV, no in-round messages; seq variants need n-1 hand-overs
                seq = getattr(agents[0], 'seq', False)
                extra = 0
                if getattr(agents[0], 'swap', False) and len(deciding) > 1:
                    sw = swap_pass(agents, deciding, events, targets, bh['edges'], W.msg_lost)
                    extra = sw['msgs']
                    W.m['swaps_sum'] = W.m.get('swaps_sum', 0) + sw['swaps']
                if relay:    # messages = relay hops between consecutive deciders in the same component; latency = same
                    coord.append((len(deciding), int(relay_msgs) + extra, max(cpu)))
                else:
                    coord.append((len(deciding) if seq else 1, (len(deciding) - 1 if seq else 0) + extra, max(cpu)))
        # certificate (Prop. online certificate, Thm. calibrated certificate)
        if log_rounds and step % cfg.cert_every == 0 and deciding:
            rec = dict(t=int(t), n=len(deciding))
            for scope in ('full', 'obs', 'view'):
                g, blocks, m = W.true_maps(t, deciding, events, cfg.survival, scope=scope)
                ub = g + float(blocks.sum())
                rec[f'g_{scope}'], rec[f'ub_{scope}'] = round(g, 4), round(ub, 4)
                rec[f'rho_{scope}'] = round(g / ub, 4) if ub > 0 else 1.0
                rec[f'm_{scope}'] = m
            if agents is not None and getattr(agents[0], 'value', None) == 'maqac':
                for scope, up, fr in (('obs', False, False), ('full', True, False), ('view', False, True)):
                    gains, blks = [], []
                    for q, i in enumerate(deciding):
                        gi, bi = agents[i].cert_terms(events[q], events[:q], events, use_prior=up, fresh=fr)
                        gains.append(gi); blks.append(bi)
                    gh, bh_ = sum(gains), sum(blks)
                    rec[f'rho_hat_{scope}'] = round(gh / (gh + bh_), 4) if gh + bh_ > 0 else 1.0
                    rec[f'g_hat_{scope}'] = round(gh, 4)
                    rec[f'ub_hat_{scope}'] = round(gh + bh_, 4)
            rounds.append(rec)
        if move is None:
            W.move_uavs(targets, sep, t)
        else:
            move(W, targets, sep, t)
        if step_hook is not None:
            step_hook(W, agents, step)
    res = W.results()
    res['tx_msgs'] = int(W.m.get('tx_msgs', 0)); res['tx_attempts'] = int(W.m.get('tx_attempts', 0))
    if sixdof is not None:
        res.update(sixdof.summary())
    res['tower_climb_s'] = int(W.m.get('tower_climb_s', 0))
    if coord:
        C = np.array(coord)
        res.update(coord_iters=float(C[:, 0].mean()), coord_iters_max=int(C[:, 0].max()), coord_msgs=float(C[:, 1].mean()),
                   decide_ms=float(1e3 * C[:, 2].mean()), decide_ms_p95=float(1e3 * np.percentile(C[:, 2], 95)))
        if 'swaps_sum' in W.m:
            res['swaps_per_round'] = W.m['swaps_sum'] / len(C)
        if method in HYB_RULE:
            res.update(conflict_round_share=W.m.get('conflicted_rounds', 0) / len(C),
                       conflicted_per_round=W.m.get('conflicted_sum', 0) / len(C))
    # information age of service records (mean delay until an alive UAV learned a record)
    if agents is not None and uses_cons:
        served = ~np.isnan(W.first_t)
        ages = []
        for i, a in enumerate(agents):
            if W.alive[i]:
                lt = a.store.learn_t[served]
                ok = ~np.isnan(lt)
                ages.append(np.mean(lt[ok] - W.first_t[served][ok]) if ok.any() else np.nan)
                res.setdefault('record_coverage', []).append(float(ok.mean()) if served.any() else 1.0)
        res['record_age_s'] = float(np.nanmean(ages)) if ages else None
        res['record_coverage'] = float(np.mean(res.get('record_coverage', [1.0])))
    res.update(method=method, config=cfg.name, seed=seed, n_uav=n, scenario=cfg.scenario,
               perception=cfg.perception, episode_time_s=round(time.time() - t0, 2))
    if log_rounds:
        res['rounds'] = rounds
    return res
