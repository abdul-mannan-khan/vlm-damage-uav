"""MAPPO baseline (Yu et al., NeurIPS 2022) for the network-level simulator.

Fair-information design: each UAV observes what a MAQAC UAV knows (its own detections, the shared service records and
the density prior, summarised by its own value map) and the neighbour positions it has received. The policy replaces
MAQAC's argmax: it picks one of 10x10 coarse cells (100 m), and the UAV flies to the best 20 m cell of its value map
inside that block. The same backhaul-connectivity constraint as MAQAC and RAG masks the actions. Co-channel
exclusion is NOT imposed; the co-channel neighbour map is an input, so the policy can learn it.
Training: shared actor, centralised critic with an agent-specific global state (true unserved value map and all UAV
positions), shared team reward = increase of the QoS-weighted score J (scaled), decision every HOLD seconds.

    python3 src/netsim/mappo.py train --n 6 --iters 150 --workers 8 --out results/mappo_netsim_6.pt
    python3 src/netsim/mappo.py eval  --n 6 --ckpt results/mappo_netsim_6.pt      # appends to results/netsim_grid.jsonl
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import numpy as np
from sim import NetConfig, survival, realistic
from agents import MAQACAgent, METHODS
from episode import run_episode

C = 10                 # coarse grid (C x C actions)
HOLD = 5               # seconds between decisions
OBS = 4 * C * C + 4
GLOB = 2 * C * C
T_POS = 30.0
REW_SCALE = 0.1


def coarse(xy, world):
    k = np.clip((np.asarray(xy) / world * C).astype(int), 0, C - 1)
    return k[..., 1] * C + k[..., 0]


class MAPPOAgent(MAQACAgent):
    policy = None          # callable(obs, mask) -> (action, logp); set through agent_flags
    greedy = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.traj = []
        self.cur = None

    def features(self):
        V = self.value_map(())
        g = self.G; b = g // C
        Vc = V.reshape(g, g)[:C * b, :C * b].reshape(C, b, C, b).max(axis=(1, 3)).ravel()
        vmax = Vc.max()
        s, n, K, W = self.store, self.cfg.n_uav, max(1, self.cfg.n_channels), self.cfg.world
        oth, coc = np.zeros(C * C), np.zeros(C * C)
        for j in range(n):
            if j != self.i and s.kt[j] >= self.t - T_POS:
                c = coarse(s.kpos[j], W)
                oth[c] += 1
                if j % K == self.i % K:
                    coc[c] += 1
        own = np.zeros(C * C); own[coarse(self.pos, W)] = 1
        obs = np.concatenate([Vc / (vmax + 1e-6), oth, coc, own,
                              [self.pos[0] / W, self.pos[1] / W, self.t / self.cfg.T, np.log1p(vmax) / 5]]).astype(np.float32)
        ok = self._allowed()                                    # connectivity filter (cochan is off for MAPPO)
        okc = ok.reshape(g, g)[:C * b, :C * b].reshape(C, b, C, b).any(axis=(1, 3)).ravel()
        if not okc.any():
            okc[:] = True
        return obs, okc, V, ok

    def decide(self, prev_events=()):
        t = int(round(self.t))
        if self.cur is not None and t % HOLD != 0:
            tgt = self.cur
        else:
            obs, mask, V, ok = self.features()
            a, logp = self.policy(obs, mask)
            self.traj.append((t, obs, mask, a, logp))
            g = self.G; b = g // C
            cy, cx = divmod(a, C)
            blk = np.zeros((g, g), bool); blk[cy * b:(cy + 1) * b, cx * b:(cx + 1) * b] = True
            cand = np.where(blk.ravel() & ok, V, -1.0)
            c = int(np.argmax(cand))
            if cand[c] <= 1e-9:
                tgt = np.array([(cx + 0.5) * self.cfg.world / C, (cy + 0.5) * self.cfg.world / C])
            else:
                tgt = self.centres[c].copy()
            self.cur = tgt
        return tgt.copy(), (float(tgt[0]), float(tgt[1]), float(self.t + np.hypot(*(tgt - self.pos)) / self.cfg.speed))


METHODS['mappo'] = (MAPPOAgent, dict(share=False, voronoi=False), True, True)


# ----------------------------------------------------------------------------- numpy actor for rollout workers
def np_actor(Wt, greedy=False, rng=None):
    def f(obs, mask):
        h = np.tanh(obs @ Wt[0] + Wt[1]); h = np.tanh(h @ Wt[2] + Wt[3]); z = h @ Wt[4] + Wt[5]
        z = np.where(mask, z, -1e9); z = z - z.max(); p = np.exp(z); p /= p.sum()
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        return a, float(np.log(p[a] + 1e-12))
    return f


def make_cfg(cfgkw):
    """cfgkw with v3=True: the realistic model (sim.realistic) with the remaining keys as overrides."""
    kw = dict(cfgkw)
    if kw.pop('v3', False):
        return realistic(**kw)
    return replace(NetConfig(), **kw)


def v3_kw(a, sc, n, name):
    """Scenario keys of the v3 grid: area a.world (m), density fixed at 200 users + 0.1/s per km^2, gateway at the centre."""
    f = (a.world / 1000.0) ** 2
    kw = dict(v3=True, scenario=sc, n_uav=n, name=name, world=a.world, n_users=int(200 * f), arrival_rate=0.1 * f,
              gateway=(a.world / 2, a.world / 2))
    if a.r_s:
        kw['r_s'] = a.r_s
    if a.lambda_hold is not None:
        kw['lambda_hold'] = a.lambda_hold
    return kw


def SC(r):
    return r.get('J_session', r['qos_score'])


def score_now(W, t):
    if getattr(W, 'sessions', False):                   # v3: running session score (seconds in service so far)
        served = ~np.isnan(W.first_t) & W.kept
        win = np.maximum(1.0, np.minimum(W.sess_end, W.cfg.T) - np.where(served, W.first_t, 0.0))
        return float((W.w * np.where(served, np.minimum(1.0, W.svc_s / win), 0.0)).sum())
    served = ~np.isnan(W.first_t)
    return float((W.w * survival(W.cfg.survival, W.mu, np.where(served, W.first_t - W.arr, 0.0)) * served).sum())


def global_state(W, t):
    pres = (W.arr <= t) & np.isnan(W.first_t)
    v = W.w * survival(W.cfg.survival, W.mu, t - W.arr) * pres
    if getattr(W, 'sessions', False):                   # v3: in-session users count too (they earn the reward)
        v = v + W.w * ((W.arr <= t) & W.kept & (t < W.sess_end))
    um = np.bincount(coarse(W.pos, W.cfg.world), v, C * C) / 20.0
    air = W.alive & (W.mode == 0)
    pm = np.bincount(coarse(W.upos[air], W.cfg.world), minlength=C * C).astype(float) if air.any() else np.zeros(C * C)
    return np.concatenate([um, pm]).astype(np.float32)


def rollout(args):
    Wt, cfgkw, seed, greedy = args
    cfg = make_cfg(cfgkw)
    rng = np.random.default_rng([seed, 7])
    scores, globs = {}, {}

    def hook(W, agents, step):
        scores[step] = score_now(W, step)
        if step % HOLD == 0:
            globs[step] = global_state(W, step)
    keep = {}

    def capture(W, agents, step):
        hook(W, agents, step)
        keep['agents'] = agents
    res = run_episode(cfg, 'mappo', seed, log_rounds=False, step_hook=capture,
                      agent_flags=dict(policy=np_actor(Wt, greedy, rng)))
    if greedy:
        return res, None
    out = []
    T = cfg.T - 1
    for a in keep['agents']:
        if not a.traj:
            continue
        ts = [x[0] for x in a.traj]
        nxt = ts[1:] + [T]
        rew = [REW_SCALE * (scores.get(min(t2, T), scores[T]) - scores.get(t1, 0.0)) for t1, t2 in zip(ts, nxt)]
        gap = [(t2 - t1) / HOLD for t1, t2 in zip(ts, nxt)]
        out.append(dict(obs=np.stack([x[1] for x in a.traj]), mask=np.stack([x[2] for x in a.traj]),
                        act=np.array([x[3] for x in a.traj]), logp=np.array([x[4] for x in a.traj]),
                        glob=np.stack([globs.get(t, globs[max(k for k in globs if k <= t)]) for t in ts]),
                        rew=np.array(rew, np.float32), gap=np.array(gap, np.float32)))
    return res, out


# ----------------------------------------------------------------------------- learner
def train(a):
    import torch, torch.nn as nn
    from multiprocessing import Pool
    torch.set_num_threads(4)
    actor = nn.Sequential(nn.Linear(OBS, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, C * C))
    critic = nn.Sequential(nn.Linear(OBS + GLOB, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    gamma, lam, clip, ent_c = 0.99, 0.95, 0.2, 0.01
    cfgkw = dict(scenario=a.scenario, n_uav=a.n, name=f'mappo_{a.scenario}_{a.n}')
    if a.v3:
        cfgkw = v3_kw(a, a.scenario, a.n, f'mappo_v3_{a.scenario}_{a.n}')
    if a.T:
        cfgkw['T'] = a.T
    log = open(a.out.replace('.pt', '.log'), 'a')
    best, hist, t0 = -1e9, [], time.time()
    with Pool(a.workers) as pool:
        for it in range(a.iters):
            Wt = [p.detach().numpy().T if p.dim() == 2 else p.detach().numpy() for p in actor.parameters()]
            seeds = [100000 + it * a.episodes + k for k in range(a.episodes)]
            R = pool.map(rollout, [(Wt, cfgkw, s, False) for s in seeds])
            J = float(np.mean([SC(r[0]) for r in R]))
            hist.append(J)
            trajs = [tr for r in R for tr in r[1]]
            obs, glob, mask, act, logp_old, adv, ret = [], [], [], [], [], [], []
            for tr in trajs:
                o = torch.tensor(tr['obs']); g = torch.tensor(tr['glob'])
                with torch.no_grad():
                    v = critic(torch.cat([o, g], 1)).squeeze(1).numpy()
                Tn = len(v); A = np.zeros(Tn, np.float32); last = 0.0
                for k in reversed(range(Tn)):
                    disc = gamma ** tr['gap'][k]
                    vn = v[k + 1] if k + 1 < Tn else 0.0
                    delta = tr['rew'][k] + disc * vn - v[k]
                    last = delta + disc * lam * last
                    A[k] = last
                obs.append(tr['obs']); glob.append(tr['glob']); mask.append(tr['mask']); act.append(tr['act'])
                logp_old.append(tr['logp']); adv.append(A); ret.append(A + v)
            obs, glob, mask = map(lambda x: torch.tensor(np.concatenate(x)), (obs, glob, mask))
            act = torch.tensor(np.concatenate(act)); lpo = torch.tensor(np.concatenate(logp_old), dtype=torch.float32)
            adv = torch.tensor(np.concatenate(adv)); ret = torch.tensor(np.concatenate(ret))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            N = len(act)
            for ep in range(4):
                perm = torch.randperm(N)
                for s in range(0, N, 2048):
                    ix = perm[s:s + 2048]
                    z = actor(obs[ix]).masked_fill(~mask[ix], -1e9)
                    dist = torch.distributions.Categorical(logits=z)
                    lp = dist.log_prob(act[ix])
                    r = torch.exp(lp - lpo[ix])
                    pl = -torch.min(r * adv[ix], torch.clamp(r, 1 - clip, 1 + clip) * adv[ix]).mean()
                    vl = ((critic(torch.cat([obs[ix], glob[ix]], 1)).squeeze(1) - ret[ix]) ** 2).mean()
                    loss = pl + 0.5 * vl - ent_c * dist.entropy().mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                    opt.step()
            run = float(np.mean(hist[-5:]))
            if run > best and it >= min(4, a.iters - 1):
                best = run
                torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), it=it, J=run, cfg=cfgkw), a.out)
            msg = f'it {it} J {J:.1f} run5 {run:.1f} best {best:.1f} samples {N} t {time.time() - t0:.0f}s'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), it=a.iters - 1, J=float(np.mean(hist[-5:])), cfg=cfgkw),
               a.out.replace('.pt', '_last.pt'))


def evaluate(a):
    import torch
    from multiprocessing import Pool
    ck = torch.load(a.ckpt)
    sd = ck['actor']
    Wt = [sd[k].numpy().T if sd[k].dim() == 2 else sd[k].numpy() for k in sd]
    jobs, meta = [], []
    for sc in a.scenarios.split(','):
        for s in range(30):
            kw = v3_kw(a, sc, a.n, f'v3_{sc}_{a.n}') if a.v3 else dict(scenario=sc, n_uav=a.n, name=f'main_{sc}_{a.n}')
            jobs.append((Wt, kw, 20000 + s, True))
            meta.append(dict(block='main', scenario=sc, n_uav=a.n, method=a.method_name, seed=20000 + s, param=None,
                             value=None))
    with Pool(a.workers) as pool:
        R = pool.map(rollout, jobs)
    with open(a.results, 'a') as f:
        for (res, _), m in zip(R, meta):
            res.update(m); res['ckpt_iter'] = ck['it']; res['train_seed'] = ck.get('train_seed')
            f.write(json.dumps(res, default=float) + '\n')
    for sc in a.scenarios.split(','):
        J = [SC(r[0]) for r, m in zip(R, meta) if m['scenario'] == sc]
        print(sc, a.n, f'J = {np.mean(J):.1f} +- {np.std(J):.1f}')


# ----------------------------------------------------------------------------- learner v2 (protocol of 2026-10-30)
VAL_SEEDS = list(range(90000, 90016))   # fixed validation missions (disjoint from training 1e5+ and test 2e4+ seeds)


class ValueNorm:
    """Running mean / variance of the value targets (MAPPO's ValueNorm, debiased EMA)."""
    def __init__(self, beta=0.99):
        self.beta, self.m, self.m2, self.w = beta, 0.0, 0.0, 0.0

    def update(self, x):
        b = self.beta
        self.m = b * self.m + (1 - b) * float(x.mean())
        self.m2 = b * self.m2 + (1 - b) * float((x ** 2).mean())
        self.w = b * self.w + (1 - b)

    def stats(self):
        if self.w <= 0:
            return 0.0, 1.0
        m = self.m / self.w
        return m, max(1e-4, self.m2 / self.w - m * m) ** 0.5


def plateau(V, window):
    """Stopping rule: the mean validation score of the last `window` checks is not above that of the `window` checks
    before it by more than two standard errors (Welch). Returns (stop, gain, 2*SE)."""
    if len(V) < 2 * window:
        return False, None, None
    a, b = np.array(V[-window:]), np.array(V[-2 * window:-window])
    se = np.sqrt(a.var(ddof=1) / window + b.var(ddof=1) / window)
    g = float(a.mean() - b.mean())
    return g <= 2 * se, g, float(2 * se)


def train2(a):
    import torch, torch.nn as nn
    from multiprocessing import Pool
    torch.set_num_threads(2)
    torch.manual_seed(a.train_seed)
    actor = nn.Sequential(nn.Linear(OBS, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, C * C))
    critic = nn.Sequential(nn.Linear(OBS + GLOB, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    gamma, lam, clip, ent_c = 0.99, 0.95, 0.2, 0.01
    vn = ValueNorm()
    cfgkw = dict(scenario=a.scenario, n_uav=a.n, name=f'mappo_{a.scenario}_{a.n}')
    if a.v3:
        cfgkw = v3_kw(a, a.scenario, a.n, f'mappo_v3_{a.scenario}_{a.n}')
    if a.T:
        cfgkw['T'] = a.T
    best, V, t0, stop_info, it0 = -1e9, [], time.time(), None, 0
    state_f = a.out.replace('.pt', '_state.pt')
    if a.resume and os.path.exists(state_f):             # spot preemption: continue from the last full state
        st = torch.load(state_f, weights_only=False)
        actor.load_state_dict(st['actor']); critic.load_state_dict(st['critic']); opt.load_state_dict(st['opt'])
        vn.m, vn.m2, vn.w = st['vn']
        best, V, it0 = st['best'], st['V'], st['it'] + 1
        torch.set_rng_state(st['torch_rng'])
        t0 -= st['elapsed']
        print(json.dumps(dict(resumed_at=it0)), flush=True)
    curve = open(a.out.replace('.pt', '_curve.jsonl'), 'a' if it0 else 'w')
    if it0:                                              # drop curve lines written after the saved state
        L = [l for l in open(a.out.replace('.pt', '_curve.jsonl')) if json.loads(l)['it'] < it0]
        curve.close(); curve = open(a.out.replace('.pt', '_curve.jsonl'), 'w'); curve.writelines(L)
    base = 100000 + 1000000 * a.train_seed
    val_seeds = VAL_SEEDS if a.n_val == len(VAL_SEEDS) else list(range(90000, 90000 + a.n_val))
    with Pool(a.workers) as pool:
        for it in range(it0, a.max_iters):
            for g_ in opt.param_groups:                      # linear learning-rate decay over the iteration cap
                g_['lr'] = a.lr * (1 - it / a.max_iters)
            Wt = [p.detach().numpy().T if p.dim() == 2 else p.detach().numpy() for p in actor.parameters()]
            seeds = [base + it * a.episodes + k for k in range(a.episodes)]
            R = pool.map(rollout, [(Wt, cfgkw, s, False) for s in seeds])
            J = float(np.mean([SC(r[0]) for r in R]))
            trajs = [tr for r in R for tr in r[1]]
            m, sd = vn.stats()
            obs, glob, mask, act, lpo, adv, ret = [], [], [], [], [], [], []
            for tr in trajs:
                o = torch.tensor(tr['obs']); g = torch.tensor(tr['glob'])
                with torch.no_grad():
                    v = critic(torch.cat([o, g], 1)).squeeze(1).numpy() * sd + m      # de-normalised values
                Tn = len(v); A = np.zeros(Tn, np.float32); last = 0.0
                for k in reversed(range(Tn)):
                    disc = gamma ** tr['gap'][k]
                    vnx = v[k + 1] if k + 1 < Tn else 0.0
                    delta = tr['rew'][k] + disc * vnx - v[k]
                    last = delta + disc * lam * last
                    A[k] = last
                obs.append(tr['obs']); glob.append(tr['glob']); mask.append(tr['mask']); act.append(tr['act'])
                lpo.append(tr['logp']); adv.append(A); ret.append(A + v)
            obs, glob, mask = map(lambda x: torch.tensor(np.concatenate(x)), (obs, glob, mask))
            act = torch.tensor(np.concatenate(act)); lpo = torch.tensor(np.concatenate(lpo), dtype=torch.float32)
            adv = torch.tensor(np.concatenate(adv)); ret = np.concatenate(ret).astype(np.float32)
            vn.update(ret)
            m, sd = vn.stats()
            retn = torch.tensor((ret - m) / sd)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            N = len(act)
            for ep in range(a.ppo_epochs):
                perm = torch.randperm(N)
                for s0 in range(0, N, 2048):
                    ix = perm[s0:s0 + 2048]
                    z = actor(obs[ix]).masked_fill(~mask[ix], -1e9)
                    dist = torch.distributions.Categorical(logits=z)
                    r = torch.exp(dist.log_prob(act[ix]) - lpo[ix])
                    pl = -torch.min(r * adv[ix], torch.clamp(r, 1 - clip, 1 + clip) * adv[ix]).mean()
                    vl = ((critic(torch.cat([obs[ix], glob[ix]], 1)).squeeze(1) - retn[ix]) ** 2).mean()
                    loss = pl + 0.5 * vl - ent_c * dist.entropy().mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(params, 0.5)
                    opt.step()
            rec = dict(it=it, J_train=round(J, 2), samples=N, t=round(time.time() - t0))
            if (it + 1) % a.val_every == 0:
                Wt = [p.detach().numpy().T if p.dim() == 2 else p.detach().numpy() for p in actor.parameters()]
                RV = pool.map(rollout, [(Wt, cfgkw, s, True) for s in val_seeds])
                Jv = float(np.mean([SC(r[0]) for r in RV]))
                V.append(Jv)
                rec['J_val'] = round(Jv, 2)
                if Jv > best:
                    best = Jv
                    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), it=it, J_val=Jv, cfg=cfgkw,
                                    train_seed=a.train_seed), a.out)
                stop, gain, se2 = plateau(V, a.window)
                rec.update(best_val=round(best, 2), plateau_gain=gain, plateau_2se=se2)
                if stop and it + 1 >= a.min_iters and not a.no_stop:
                    stop_info = dict(it=it, gain=gain, two_se=se2)
            curve.write(json.dumps(rec) + '\n'); curve.flush()
            print(json.dumps(rec), flush=True)
            if (it + 1) % a.val_every == 0:              # full state for --resume (after the curve line is written)
                torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(),
                                vn=(vn.m, vn.m2, vn.w), best=best, V=V, it=it, torch_rng=torch.get_rng_state(),
                                elapsed=time.time() - t0), state_f + '.tmp')
                os.replace(state_f + '.tmp', state_f)
            if stop_info:
                break
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), it=it, cfg=cfgkw, train_seed=a.train_seed),
               a.out.replace('.pt', '_last.pt'))
    json.dump(dict(stopped=stop_info, iters=it + 1, best_val=best, max_iters=a.max_iters, min_iters=a.min_iters,
                   window_checks=a.window, val_every=a.val_every, val_seeds=val_seeds, no_stop=a.no_stop, train_seed=a.train_seed,
                   lr=a.lr, ppo_epochs=a.ppo_epochs, episodes=a.episodes, wall_s=round(time.time() - t0)),
              open(a.out.replace('.pt', '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'train2', 'eval'])
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--T', type=int, default=0)        # shorter missions for smoke tests only
    ap.add_argument('--scenario', default='urban')
    ap.add_argument('--scenarios', default='urban')
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='results/mappo_netsim.pt')
    ap.add_argument('--ckpt', default='results/mappo_netsim.pt')
    ap.add_argument('--results', default='results/netsim_grid.jsonl')
    ap.add_argument('--train-seed', type=int, default=0)
    ap.add_argument('--max-iters', type=int, default=1500)
    ap.add_argument('--min-iters', type=int, default=300)
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--window', type=int, default=10, help='validation checks per plateau window (10 x 10 = 100 it)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ppo-epochs', type=int, default=4)
    ap.add_argument('--resume', action='store_true', help='continue from <out>_state.pt if present')
    ap.add_argument('--no-stop', action='store_true', help='train to --max-iters; saturation is judged afterwards on all seeds')
    ap.add_argument('--n-val', type=int, default=16, help='validation missions (seeds 90000...)')
    ap.add_argument('--method-name', default='mappo')
    ap.add_argument('--v3', action='store_true', help='realistic physical model (sim.realistic)')
    ap.add_argument('--world', type=float, default=1000.0)
    ap.add_argument('--r-s', type=float, default=0.0)
    ap.add_argument('--lambda-hold', type=float, default=None)
    a = ap.parse_args()
    {'train': train, 'train2': train2, 'eval': evaluate}[a.cmd](a)
