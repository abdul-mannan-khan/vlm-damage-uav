#!/usr/bin/env python3
"""Step 14 kinematic twin (PILOT / GATE, no AirSim): closed-loop arms of damage_loop.py on the netsim twin of the step-6
flights (disaster, 2 UAVs, 600 m, T = 300 s, onboard user perception, MAQAC-S). 'Seen' = the calibrated step-6 camera
model (VisibilityModel, frames every 5 s per UAV, common random numbers per seed). Every arm of a seed shares the world,
the verdict table and the visibility draws; only the planner's building weights differ.
Scenarios (prereg sec. 9): base (step 6), rb50, demand, dense3 (damage_loop.SCENARIOS).

    python3 run_closed_loop_kin.py --seed-range 22000 22049 --scenario rb50 --arms A_blind D_oracle --out gate_rb50.jsonl
    python3 run_closed_loop_kin.py --seed-range 21000 21007 --scenario rb50 --cf --out pilot_rb50.jsonl   # 4 arms + sec. 10
    python3 run_closed_loop_kin.py --summary <file.jsonl>
Sanity checks built in: arm A must give exactly the J_session of plain 'maqac_seq' (same seed, same scenario); with --cf,
the counterfactual C plan must equal the flown C plan in every round.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import damage_loop as DL                                # noqa: E402  (registers maqac_seq_dmg)
import episode as EP                                    # noqa: E402
from sim import World                                   # noqa: E402


def scen_setup(sc, rows):
    if sc.get('urgency'):
        return DL.urgency_setup(rows, sc['R_B'])
    return DL.demand_setup(rows, sc['R_B']) if sc['demand'] else None


def cell_cfg(sc, chain, n, r_s):
    from dataclasses import replace
    cfg = DL.make_cfg(n=n, chain=chain, users_mult=sc['users_mult'])
    return replace(cfg, r_s=float(r_s)) if r_s else cfg


def one(job):
    seed, arm, rows, layout, method, chain, scen, cf, n, r_s = job
    sc = DL.SCENARIOS[scen]
    cfg = cell_cfg(sc, chain, n, r_s)
    flags = dict(prior_scale=1.0 / sc['users_mult']) if sc['demand'] else None
    setup = scen_setup(sc, rows)
    st = dict(cf=[])

    def hook(W, agents, step):
        if 'W' not in st:
            st['W'] = W
            st['vis'] = DL.VisibilityModel(seed, cfg.n_uav, len(rows), cfg.T)
            st['bxy'] = np.array([[r['x'], r['y']] for r in rows], float)
        if agents is not None and getattr(agents[0], 'dmg', None) is None:
            for ag in agents:
                ag.dmg = st['dmg']
        t = int(step)
        if cf:                                          # before this round's new sightings: same state as the decision
            flown = [getattr(a, 'last', None) and a.last['cell'] for a in agents]
            r = DL.counterfactual_round(W, agents, st['dmg'], rows, R=sc['R_B'])
            if r is not None:
                cells_c = [int(np.argmin(np.hypot(*(agents[0].centres - np.array(c)).T))) for c in r['C_set']['cells']]
                dec = [i for i in range(cfg.n_uav) if W.alive[i] and W.mode[i] == 0]
                r['C_matches_flown'] = cells_c == [flown[i] for i in dec]
                st['cf'].append(dict(t=t, **{k: (v['f'] if isinstance(v, dict) else v) for k, v in r.items()},
                                     differ_B=r['B_top']['cells'] != r['C_set']['cells'],
                                     differ_D=r['D_oracle']['cells'] != r['C_set']['cells']))
        if t % st['vis'].every == 0:
            for i in range(cfg.n_uav):
                if W.airborne()[i]:
                    st['dmg'].update(st['vis'].seen(t, i, W.upos[i], st['bxy']), t, i)

    st['dmg'] = DL.DamageState(rows, arm, World(cfg, seed).centres, R=sc['R_B'])
    t0 = time.time()
    res = EP.run_episode(cfg, method + '_dmg', seed, log_rounds=False, step_hook=hook, agent_flags=flags, setup=setup)
    ts = DL.true_score(st['W'], rows, sc['R_B'])
    D = st['dmg']
    seen = np.isfinite(D.seen_t)
    out = dict(seed=seed, arm=arm, layout=layout, method=method, chain=chain, scenario=scen, n=n, r_s=cfg.r_s,
               J_session=res['J_session'], **ts, urgency=getattr(st['W'], 'urgency_info', None),
               seen=int(seen.sum()), severe=int((D.grade >= 3).sum()), severe_seen=int((seen & (D.grade >= 3)).sum()),
               coverage_seen=float(D.covered[seen].mean()) if seen.any() else None,
               set_size_seen=float(D.setsz[seen].mean()) if seen.any() else None,
               abs_err_seen=float(np.abs(D.what[seen] - np.array([DL.G[g] for g in D.grade[seen]])).mean()) if seen.any() else None,
               collisions=res.get('collisions'), wall_s=round(time.time() - t0, 2))
    if cf and st['cf']:
        C = st['cf']
        ok = [x for x in C if x['D_oracle'] > 0]
        out['cf'] = dict(rounds=len(C), rounds_scored=len(ok), C_matches_flown_all=all(x['C_matches_flown'] for x in C),
                         loss_B_pct=float(np.mean([100 * (x['D_oracle'] - x['B_top']) / x['D_oracle'] for x in ok])) if ok else None,
                         loss_C_pct=float(np.mean([100 * (x['D_oracle'] - x['C_set']) / x['D_oracle'] for x in ok])) if ok else None,
                         share_differ_B=float(np.mean([x['differ_B'] for x in C])),
                         share_differ_D=float(np.mean([x['differ_D'] for x in C])))
    if arm == 'A_blind':                                # sanity: arm A = plain MAQAC-S in the same scenario
        if sc['demand']:   # plain maqac_seq cannot hold the 1x prior: compare with the damage agent without any building state
            out['J_plain'] = EP.run_episode(cfg, method + '_dmg', seed, log_rounds=False, agent_flags=flags,
                                            setup=scen_setup(sc, rows))['J_session']
        else:
            out['J_plain'] = EP.run_episode(cfg, method, seed, log_rounds=False, setup=scen_setup(sc, rows))['J_session']
    return out


def paired(J, x, y, ref):
    from scipy.stats import wilcoxon
    d = 100 * (J[x] - J[y]) / J[ref]
    nz = d[np.abs(d) > 1e-12]
    sd = float(d.std(ddof=1)) if len(d) > 1 else None
    return dict(mean_pts=float(d.mean()), sd=sd, dz=float(d.mean() / sd) if sd else None,
                positive=int((d > 1e-12).sum()), negative=int((d < -1e-12).sum()),
                wilcoxon_greater_p=float(wilcoxon(nz, alternative='greater').pvalue) if len(nz) else None)


def summary(path):
    from scipy.stats import wilcoxon
    R = [json.loads(l) for l in open(path)]
    by = {}
    for r in R:
        by.setdefault((r.get('scenario', 'base'), r['layout'], r['method'], 'n%s' % r.get('n', 2), 'rs%g' % r.get('r_s', 150.0)), {}).setdefault(r['seed'], {})[r['arm']] = r
    S = {}
    for key, d in by.items():
        arms = sorted({a for s in d for a in d[s]})
        seeds = sorted(s for s in d if len(d[s]) == len(arms))
        J = {a: np.array([d[s][a]['J_true'] for s in seeds]) for a in arms}
        A = [d[s]['A_blind'] for s in seeds] if 'A_blind' in arms else []
        cell = dict(arms=arms, seeds=len(seeds), J_true_mean={a: float(J[a].mean()) for a in arms},
                    sanity_A_equals_plain=all(abs(x['J_session'] - x['J_plain']) < 1e-9 for x in A),
                    headroom_A=float(np.mean([x['J_true'] / x['J_true_max'] for x in A])) if A else None,
                    users_mean=float(np.mean([x['users'] for x in A])) if A else None,
                    users_g_gt1_mean=float(np.mean([x['users_g_gt1'] for x in A])) if A else None,
                    seen_mean={a: float(np.mean([d[s][a]['seen'] for s in seeds])) for a in arms},
                    abs_err_seen={a: float(np.mean([d[s][a]['abs_err_seen'] for s in seeds])) for a in arms})
        if 'D_oracle' in arms:
            for x, y in (('D_oracle', 'A_blind'), ('C_set', 'B_top'), ('C_set', 'A_blind'), ('B_top', 'A_blind'), ('D_oracle', 'C_set')):
                if x in arms and y in arms:
                    cell[f'{x}-{y}'] = paired(J, x, y, 'D_oracle')
            if 'A_blind' in arms:
                g = cell['D_oracle-A_blind']
                cell['GATE_PASS'] = bool(g['mean_pts'] >= 1.5 and (g['dz'] or 0) >= 0.8)
        if 'C_set' in arms:
            cf = [d[s]['C_set'].get('cf') for s in seeds]
            if all(cf):
                lb, lc = np.array([c['loss_B_pct'] for c in cf]), np.array([c['loss_C_pct'] for c in cf])
                dd = lb - lc
                nz = dd[np.abs(dd) > 1e-12]
                cell['cf_secondary'] = dict(loss_B_pct=float(lb.mean()), loss_C_pct=float(lc.mean()),
                                            B_minus_C=float(dd.mean()), positive=int((dd > 0).sum()),
                                            wilcoxon_C_smaller_p=float(wilcoxon(nz, alternative='greater').pvalue) if len(nz) else None,
                                            share_rounds_B_differs=float(np.mean([c['share_differ_B'] for c in cf])),
                                            C_matches_flown_all=all(c['C_matches_flown_all'] for c in cf))
        S['|'.join(key)] = cell
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--seed-range', nargs=2, type=int, help='first last (inclusive)')
    ap.add_argument('--layouts', nargs='+', default=['indep'])
    ap.add_argument('--scenario', default='base', choices=sorted(DL.SCENARIOS))
    ap.add_argument('--arms', nargs='+', default=list(DL.ARMS), choices=DL.ARMS)
    ap.add_argument('--cf', action='store_true', help='secondary endpoint (prereg sec. 10) in the C missions')
    ap.add_argument('--method', default='maqac_seq', choices=('maqac_seq', 'rag'))
    ap.add_argument('--chain', action='store_true')
    ap.add_argument('--n', type=int, nargs='+', default=[2], help='fleet sizes (grid)')
    ap.add_argument('--rs', type=float, nargs='+', default=[0.0], help='service radii (grid); 0 = default 150 m')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out')
    ap.add_argument('--summary')
    a = ap.parse_args()
    if a.summary:
        S = summary(a.summary)
        print(json.dumps(S, indent=1))
        Path(a.summary).with_name(Path(a.summary).stem + '_summary.json').write_text(json.dumps(S, indent=1))
        return
    seeds = a.seeds or list(range(a.seed_range[0], a.seed_range[1] + 1))
    cfg = DL.make_cfg(chain=a.chain, users_mult=DL.SCENARIOS[a.scenario]['users_mult'])
    jobs = []
    for lay in a.layouts:
        vt = DL.verdict_table(seeds, cfg, lay)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(vt, open(Path(a.out).with_name(Path(a.out).stem + f'_verdicts_{lay}.json'), 'w'))
        jobs += [(s, arm, vt['seeds'][str(s)], lay, a.method, a.chain, a.scenario, a.cf and arm == 'C_set', n, r)
                 for n in a.n for r in a.rs for s in seeds for arm in a.arms]
    t0 = time.time()
    with open(a.out, 'w') as f, Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs)):
            f.write(json.dumps(r) + '\n'); f.flush()
            if k % 40 == 0:
                print(f'{k + 1}/{len(jobs)} {time.time() - t0:.0f} s', flush=True)
    S = summary(a.out)
    print(json.dumps(S, indent=1))
    Path(a.out).with_name(Path(a.out).stem + '_summary.json').write_text(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
