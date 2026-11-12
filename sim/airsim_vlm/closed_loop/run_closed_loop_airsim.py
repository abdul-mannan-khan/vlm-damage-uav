#!/usr/bin/env python3
"""Step 14 closed-loop mission: MAQAC-S flown by PX4 SITL in AirSim CityEnviron with the damage weights of one arm fed
to the planner IN FLIGHT (damage_loop.py: A blind, B top label, C conformal-set midpoint, D oracle).

Closed-loop variant of the step-5/6 driver (sim/airsim_vlm/airsim/maqac_px4/run_maqac_airsim.py, NOT modified: its
Mover, SafetyLog, Clearance, FakeFleet, Recorder, collision_details are imported unchanged). Differences:
  * every 5 s each UAV's down camera gives scene + segmentation frames (as step 6); per merged footprint the pixels are
    summed; a building is SEEN at its first frame with >= 6,144 px (2 %). Its verdict is looked up in the verdict table
    (--verdicts, built on the laptop by damage_loop.py: true grade from the step-3 label model, one assigned RescueNet test
    image, the saved calibrated posterior and conformal set; no images or model on the box) and the arm's weight w_hat is
    applied to every UAV's planner from the next round on (DamageState / DamageAgent).
  * the TRUE damage-weighted score J_true (damage_loop.true_score) is computed for the flown mission and for two twins on
    the kinematic model: 'kin' (same arm, seen from the calibrated camera model) and 'replay' (the flown first-seen events
    applied at the same rounds), so flight effects and weighting effects can be separated.
  * no live VLM, no pedestrians (--no-users as step 6); RGB frames are kept for the record.

    # on the box, after px4_stack.sh start 2 (campaign_cl.sh does this):
    python3 run_closed_loop_airsim.py --ip 127.0.0.1 --n 2 --free-run --arm C_set --seed 21000 --T 300 --L 600 \
        --origin-ne 40 -10 --heightmap /work/vlm/survey/heightmap.npz --layer 3 --keepout -80 36 -339 -235 240 \
        --frames-every 5 --buildings /work/vlm/buildings.json --verdicts /work/vlm/verdicts.json --out runs/x
    # laptop dry run (no simulators; seen from the calibrated camera model on the fake-autopilot track):
    python3 run_closed_loop_airsim.py --dry-run --n 2 --arm C_set --seed 21000 --T 60 --L 600 --origin-ne 40 -10 \
        --heightmap ../survey/heightmap.npz --verdicts <table.json> --out /tmp/cl_dry
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.join(HERE, '..', 'airsim', 'maqac_px4')
sys.path.insert(0, os.path.join(HERE, '..', 'netsim'))
sys.path.insert(0, os.path.join(HERE, '..', 'airsim'))
sys.path.insert(0, DRV)
sys.path.insert(0, HERE)

import damage_loop as DL                                 # noqa: E402  (registers maqac_seq_dmg)
from sim import World                                    # noqa: E402
from episode import run_episode                          # noqa: E402
from heightmap import AreaFrame, HeightMap               # noqa: E402
from make_settings_city import SPOTS                     # noqa: E402
import run_maqac_airsim as R6                            # noqa: E402  (step-5/6 driver, unchanged)


class CLRecorder(R6.Recorder):
    """Step-6 Recorder (same frames, same labels.jsonl) that also returns the per-UAV building pixel counts."""

    def __call__(self, W, gt, t):
        A, out = self.A, {}
        if self.fe and int(t) % self.fe == 0:
            for i, nm in enumerate(self.names):
                rs = self.c.simGetImages([A.ImageRequest('down', A.ImageType.Scene, False, False),
                                          A.ImageRequest('down', A.ImageType.Segmentation, False, False)], vehicle_name=nm)
                base = f'{int(t):04d}_uav{i}'
                self.Image.fromarray(self._rgb(rs[0])).save(os.path.join(self.out, 'frames', base + '_rgb.jpg'), quality=92)
                seg = self._rgb(rs[1])
                self.Image.fromarray(seg).save(os.path.join(self.out, 'frames', base + '_seg.png'))
                bld, unk = self.resolver.counts(seg, gt[i][:2])
                out[i] = bld
                self.meta.write(json.dumps(dict(t=int(t), uav=i, file=base, pose_ned=np.round(gt[i], 2).tolist(), bld=bld,
                                                seg_unknown_px=unk, mode=int(W.mode[i]), users_near=[])) + '\n')
            self.meta.flush()
        return out


def seen_from_counts(bld, piece2fp):
    acc = defaultdict(int)
    for n, px in bld or []:
        if n in piece2fp:
            acc[piece2fp[n]] += px
    return [b for b, px in acc.items() if px >= DL.PX_MIN]


def kin_twin(cfg, seed, rows, arm, vis_path, replay=None):
    """Kinematic twin of the same arm: seen from the calibrated camera model, or (replay) from the flown seen events."""
    st = {}
    D = DL.DamageState(rows, arm, World(cfg, seed).centres)
    bxy = np.array([[r['x'], r['y']] for r in rows], float)
    vis = DL.VisibilityModel(seed, cfg.n_uav, len(rows), cfg.T, path=vis_path) if replay is None else None
    ev = defaultdict(list)
    for e in replay or []:
        ev[e['t']].append((e['b'], e['uav']))

    def hook(W, agents, step):
        st['W'] = W
        if agents is not None and agents[0].dmg is None:
            for ag in agents:
                ag.dmg = D
        t = int(step)
        if replay is not None:
            for b, u in ev.get(t, []):
                D.update([b], t, u)
        elif t % vis.every == 0:
            for i in range(cfg.n_uav):
                if W.airborne()[i]:
                    D.update(vis.seen(t, i, W.upos[i], bxy), t, i)
    res = run_episode(cfg, cfg_method(), seed, log_rounds=False, step_hook=hook)
    return res, DL.true_score(st['W'], rows), D


def cfg_method():
    return 'maqac_seq_dmg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default='127.0.0.1')
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--arm', required=True, choices=DL.ARMS)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--scenario', default='disaster')
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--perception', default='onboard')
    ap.add_argument('--alt', type=float, default=100.0)
    ap.add_argument('--layer', type=float, default=3.0)
    ap.add_argument('--L', type=float, default=600.0)
    ap.add_argument('--margin-up', type=float, default=20.0)
    ap.add_argument('--clear-r', type=float, default=15.0)
    ap.add_argument('--lookahead', type=float, default=150.0)
    ap.add_argument('--brake', type=float, default=50.0)
    ap.add_argument('--hold-ahead', type=float, default=80.0)
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(40.0, -10.0))
    ap.add_argument('--heightmap', required=True)
    ap.add_argument('--keepout', type=float, nargs=5, action='append', default=[])
    ap.add_argument('--nsub', type=int, default=5)
    ap.add_argument('--free-run', action='store_true')
    ap.add_argument('--no-users', action='store_true', help='accepted for plan compatibility; users are never spawned')
    ap.add_argument('--frames-every', type=int, default=5)
    ap.add_argument('--buildings', default=str(DL.BLDS))
    ap.add_argument('--verdicts', required=True, help='verdict table JSON (damage_loop.py)')
    ap.add_argument('--vis', default=str(DL.VIS), help='calibrated camera model (kinematic twin, dry run)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'bridge.log'), 'a')

    def log(*x):
        s = f'[{time.strftime("%H:%M:%S")}] ' + ' '.join(str(v) for v in x)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    vt = json.load(open(a.verdicts))
    assert vt['meta']['L'] == a.L and list(vt['meta']['origin_ne']) == list(a.origin_ne), 'verdict table frame mismatch'
    rows = vt['seeds'][str(a.seed)]
    names, fxy, piece2fp = DL.footprints(a.L, a.origin_ne, a.buildings)
    assert names == [r['name'] for r in rows], 'verdict table and building inventory disagree'
    cfg = DL.make_cfg(n=a.n, T=a.T, L=a.L, scenario=a.scenario, perception=a.perception, alt=a.alt)
    fr = AreaFrame(cfg.world, a.origin_ne)
    hm = HeightMap.load(a.heightmap)
    import copy
    hm_plan = copy.copy(hm); hm_plan.h = hm.h.copy()
    for n_lo, n_hi, e_lo, e_hi, top in a.keepout:
        i0, j0 = hm._clip(*hm.cell(n_lo, e_lo)); i1, j1 = hm._clip(*hm.cell(n_hi, e_hi))
        hm_plan.h[i0:i1 + 1, j0:j1 + 1] = np.maximum(hm_plan.h[i0:i1 + 1, j0:j1 + 1], top)
    clear = R6.Clearance(hm_plan, a.clear_r, a.brake)
    names_v = [f'UAV{i}' for i in range(a.n)]
    spawn = [SPOTS[i] for i in range(a.n)]
    c = None
    t_wall = time.time()
    if a.dry_run:
        fleet = R6.FakeFleet(a.n, spawn, dt=1.0 / a.nsub, log=log)
    else:
        import airsim
        from px4_fleet import Fleet
        c = airsim.MultirotorClient(ip=a.ip, timeout_value=120)
        c.confirmConnection()
        if not a.free_run:
            c.simPause(True)
        fleet = Fleet(a.n, names_v, spawn, client=c, dt=1.0 / a.nsub, free_run=a.free_run, log=log)
        from buildings import apply_ids
        log(f'segmentation IDs set for {apply_ids(c, json.load(open(a.buildings)))} buildings')
    safety = R6.SafetyLog(a.n, 1.0 / a.nsub, hm)
    if hasattr(fleet, 'phase'):
        fleet.on_step = lambda fl, ph: safety(fl, ph)
    else:
        _st = fleet.step

        def _step(dt=None):
            _st(dt); safety(fleet, getattr(fleet, 'phase', 'startup'))
        fleet.step = _step
    fleet.phase = 'startup'
    D = DL.DamageState(rows, a.arm, World(cfg, a.seed).centres)
    vis = DL.VisibilityModel(a.seed, a.n, len(rows), a.T, path=a.vis) if a.dry_run else None
    try:
        fleet.wait_ready()
        rest = fleet.world()
        d_ground = float(np.median(rest[:, 2]))
        alt_d = np.array([d_ground - (a.alt + a.layer * i) for i in range(a.n)])
        start = fr.to_world(World(cfg, a.seed).upos)
        alt_to = alt_d.copy()
        for i in range(a.n):
            alt_to[i] = min(alt_d[i], -(clear.need(rest[i, :2], start[i]) + a.margin_up + a.layer * i))
        log(f'arm {a.arm} seed {a.seed}: takeoff to D {np.round(alt_to, 1).tolist()}, start {np.round(start, 1).tolist()}')
        coll_rest = R6.collision_details(c, names_v) if c is not None else None
        fleet.phase = 'takeoff'
        fleet.takeoff(alt_to, start)
        fleet.phase = 'mission'
        log(f'at start points after {fleet.t_sim:.1f} s sim')
        rec = None
        if not a.dry_run:
            from buildings import Resolver
            rec = CLRecorder(c, names_v, a.out, None, fr, a.frames_every, False, cfg.detect_range,
                             Resolver(json.load(open(a.buildings))))
        track_px4, coll_events = [], []
        coll_seen = {nm: (v['t_ns'] if v else None) for nm, v in (coll_rest or {}).items()}
        rounds = open(os.path.join(a.out, 'rounds.jsonl'), 'w')
        mover = R6.Mover(fleet, fr, alt_d, a.nsub, hm, rounds, clear=clear, margin=a.margin_up, layer=a.layer,
                         lookahead=a.lookahead, hold=a.hold_ahead)
        seen_f = open(os.path.join(a.out, 'seen.jsonl'), 'w')
        st = {}

        def hook(W, agents, step):
            st['W'] = W
            if agents is not None and agents[0].dmg is None:
                for ag in agents:
                    ag.dmg = D
            gt = fleet.gt_world()
            t = int(step)
            if rec is not None:                                    # flight: segmentation decides what was seen
                for i, bld in rec(W, gt, float(step)).items():
                    new = D.update(seen_from_counts(bld, piece2fp), t, i)
                    for b in new:
                        seen_f.write(json.dumps(dict(t=t, uav=i, b=b, name=rows[b]['name'], grade=rows[b]['grade'],
                                                     img=rows[b]['img'], set=rows[b]['set'], w_hat=float(D.what[b]))) + '\n')
            elif vis is not None and t % a.frames_every == 0:     # dry run: calibrated camera model on the fake track
                for i in range(a.n):
                    for b in D.update(vis.seen(t, i, W.upos[i], fxy), t, i):
                        seen_f.write(json.dumps(dict(t=t, uav=i, b=b, name=rows[b]['name'], grade=rows[b]['grade'],
                                                     w_hat=float(D.what[b]), source='visibility_model_dry')) + '\n')
            track_px4.append(np.round(gt, 2).tolist())
            if c is not None:
                for nm in names_v:
                    ci = c.simGetCollisionInfo(nm)
                    if ci.has_collided and coll_seen.get(nm) != int(ci.time_stamp):
                        coll_seen[nm] = int(ci.time_stamp)
                        coll_events.append(dict(t=t, uav=nm, object=ci.object_name,
                                                pos_ned=[round(ci.position.x_val, 2), round(ci.position.y_val, 2), round(ci.position.z_val, 2)]))
            if step % 60 == 0:
                log(f't={step} seen={int(np.isfinite(D.seen_t).sum())} planner-version={D.version} modes={W.mode.tolist()}')

        t_mis = fleet.t_sim
        res = run_episode(cfg, 'maqac_seq_dmg', a.seed, log_rounds=False, step_hook=hook, move=mover)
        rounds.close(); seen_f.close()
        ts = DL.true_score(st['W'], rows)
        kin, kin_ts, Dk = kin_twin(cfg, a.seed, rows, a.arm, a.vis)
        rep, rep_ts, _ = kin_twin(cfg, a.seed, rows, a.arm, a.vis, replay=D.log)
        seen = np.isfinite(D.seen_t)
        keys = ('J_session', 'qos_score', 'users_served', 'energy_wh', 'distance_m', 'collisions', 'coord_msgs')
        out = dict(step=14, arm=a.arm, method='maqac_seq', seed=a.seed, n_uav=a.n, T=a.T, L=a.L, origin_ne=list(a.origin_ne),
                   keepout=a.keepout, dry_run=a.dry_run, verdict_meta=vt['meta'],
                   true=ts, px4=res, kinematic=dict({k: kin.get(k) for k in keys}, **kin_ts),
                   replay_twin=dict({k: rep.get(k) for k in keys}, **rep_ts),
                   seen=dict(n=int(seen.sum()), by_5min=int((D.seen_t <= 300).sum()), severe=int((D.grade >= 3).sum()),
                             severe_seen=int((seen & (D.grade >= 3)).sum()),
                             coverage=float(D.covered[seen].mean()) if seen.any() else None, covered_k=int(D.covered[seen].sum()),
                             set_size=float(D.setsz[seen].mean()) if seen.any() else None,
                             kin_n=int(np.isfinite(Dk.seen_t).sum())),
                   flight=mover.stats(), safety=safety.stats(),
                   pacing=fleet.pace_stats() if hasattr(fleet, 'pace_stats') else None,
                   sim_s_mission=fleet.t_sim - t_mis, wall_s=round(time.time() - t_wall, 1))
        out['delta_Jtrue_vs_kin_pct'] = 100.0 * (ts['J_true'] - kin_ts['J_true']) / max(1e-9, kin_ts['J_true'])
        out['delta_Jtrue_vs_replay_pct'] = 100.0 * (ts['J_true'] - rep_ts['J_true']) / max(1e-9, rep_ts['J_true'])
        if c is not None:
            out['collisions_airsim'] = R6.collision_details(c, names_v)
            out['collisions_at_rest'] = coll_rest
            out['collisions_in_flight'] = {nm: v for nm, v in out['collisions_airsim'].items()
                                           if v is not None and (coll_rest.get(nm) is None or v['t_ns'] != coll_rest[nm]['t_ns'])}
        out['collision_events'] = coll_events
        json.dump(dict(px4=track_px4, alt=a.alt, origin_ne=list(a.origin_ne), L=a.L), open(os.path.join(a.out, 'tracks.json'), 'w'))
        json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
        log(f"done: J_true flown {ts['J_true']:.2f} kin {kin_ts['J_true']:.2f} replay {rep_ts['J_true']:.2f}; seen {int(seen.sum())}")
    finally:
        fleet.close()
        if c is not None and not a.free_run:
            c.simPause(False)


if __name__ == '__main__':
    main()
