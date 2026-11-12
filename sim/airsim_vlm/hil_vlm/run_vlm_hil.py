#!/usr/bin/env python3
"""Closed-loop HIL with the onboard damage VLM (VLM paper, IEEE TVT; prepared 12 Nov 2026, not yet flown).

UAV 0 of a 2-UAV fleet is flown by PX4 on the Pixhawk 6C in HITL with AirSim CityEnviron on the laptop supplying physics,
sensors and UAV 0's downward camera (the AirSim city hardware-in-the-loop set-up: W2's p27_router.py on the Jetson, TCP 4560 to the laptop at
192.168.55.100). The Jetson Orin Nano runs the network model, every UAV's decisions (1 s rounds, real time) AND the int8
damage VLM on UAV 0's frames (one every 5 s of mission time) in a separate process (vlm_worker.py), so the 250 Hz HIL
loop and the decision rounds never wait for it.

Chain per frame: capture -> int8 VLM letter logits -> calibrated belief P (vector scaling, Belief class of step 6) ->
alpha = 0.1 conformal set C -> set-midpoint weight w_hat (Lemma 2) -> buildings seen in the frame (>= 2 % of pixels)
-> planner (--vlm-planner):
  off   : as step 6 (sec:flights): the VLM reads every frame live and is logged, the planner does NOT use it (it plans on
          the network model's onboard perception of users). Decisions are then those of plain MAQAC-S.
  prior : closed loop [NEEDS A HUMAN DECISION before it is pre-registered]: the MAQAC prior density of unseen users
          (agents' rho0 and arrival rho_arr) is multiplied by w_hat / g(no damage) = w_hat in the grid cells within
          --bld-radius m of every building whose latest frame gave a weight; applied after round t, used from round t+1.
          --vlm-share all: every UAV's planner gets UAV 0's building weights (assumes they travel with the relayed
          decision list; not counted as messages); own: only UAV 0's planner.
Each mission writes <out>/<name>/result.json, rounds.jsonl (flight + decision timing, the earlier MAQAC work format), vlm_frames.jsonl
(per frame: capture time, capture/VLM ms, logits, P, set, w_hat, seen buildings, when applied, belief age at use),
vlm_rounds.jsonl (per round: newest/max belief age, beliefs in use, hook ms, system memory in use).
After each mission: kinematic twin (no HIL, no VLM) and, for 'prior', a replay twin (no HIL, the same building weights
applied at the same rounds), so flight effects and VLM effects can be separated.

  # dry run on any PC (no AirSim, no Jetson, no Pixhawk): fake HIL autopilot, step-6 frames, stored Jetson logits
  python3 run_vlm_hil.py --dry-run --T 30 --seeds 20000 --vlm-planner prior --out /tmp/vlm_hil_dry
  # Jetson, set-up 2 (through hil_campaign_vlm.sh only)
  python3 run_vlm_hil.py --router --vlm-model int8 --vlm-frames airsim --vlm-python ~/vlm_damage_p3/venv/bin/python --T 300 \
      --seeds 20000 20001 --vlm-planner off prior --out results/hil_vlm
"""
import argparse, json, math, os, platform, sys, time
import multiprocessing as mp
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_maqac_hil as RH                                       # noqa: E402  (the earlier MAQAC work runner, netsim on sys.path)
from run_maqac_hil import HilMover, MissionAbort, TIMING, timed_methods, transfer   # noqa: E402
import episode as EP                                             # noqa: E402
from sim import World, realistic                                 # noqa: E402
from hil_link import HilClient                                   # noqa: E402
import vlm_worker as VW                                          # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STEP6_KEEPOUT = [(-80.0, 36.0, -339.0, -235.0, 240.0)]          # plan_step6.txt --keepout


def make_cfg(a):
    """Step-6 configuration (box/plan_step6.txt): disaster, n = 2, 600 m, T = 300 s, onboard perception, users per km^2
    as the 1 km scenarios, takeoff flown by PX4; + relayed list and chain rule (the planner of record, sec:chain)."""
    s = (a.L / 1000.0) ** 2
    cfg = realistic(name=f'{a.scenario}_{a.n}uav', scenario=a.scenario, n_uav=a.n, T=a.T, perception=a.perception,
                    altitude=a.alt, world=a.L, gateway=(a.L / 2, a.L / 2), n_users=int(round(200 * s)),
                    arrival_rate=0.1 * s, takeoff=False)
    if a.chain:
        cfg = replace(cfg, seq_relay=True, seq_chain=True)
    return cfg


def mem_used_mb():
    d = {}
    for l in open('/proc/meminfo'):
        k, v = l.split(':', 1)
        d[k] = int(v.split()[0])
    return (d['MemTotal'] - d['MemAvailable']) / 1024.0


class VLMLoop:
    """step_hook(W, agents, step) of run_episode: capture requests, result polling, belief -> planner weights, logs."""

    def __init__(self, a, req_q, res_q, frame, centres_area, blds, mode, out_dir, share='all', replay=None):
        self.a, self.req_q, self.res_q, self.fr, self.mode, self.share = a, req_q, res_q, frame, mode, share
        self.cent = centres_area                         # building centres in the area frame, by name
        self.blds = blds
        self.ff = open(os.path.join(out_dir, 'vlm_frames.jsonl'), 'w') if req_q is not None else None
        self.fr_round = open(os.path.join(out_dir, 'vlm_rounds.jsonl'), 'w') if req_q is not None else None
        self.w = {}                                      # building -> (w_hat, capture mission time)
        self.base = None
        self.applied = []                                # (step, building, w_hat, t_cap) for the replay twin
        self.replay = replay                             # list of the above: no queues, apply at the same steps
        self.pending, self.n_req, self.n_res, self.n_err, self.dropped = {}, 0, 0, 0, 0
        self.lat, self.lag, self.age_use, self.hook_ms, self.mem = [], [], [], [], []
        self.worker_hwm = None
        self.errors = []

    def _apply(self, agents):
        if self.mode != 'prior' or agents is None:
            return 0
        if self.base is None:
            self.base = [(ag.rho0, ag.rho_arr) for ag in agents]
            C = agents[0].centres
            self.cell_of = {}
            for b, xy in self.cent.items():
                self.cell_of[b] = np.where(np.hypot(C[:, 0] - xy[0], C[:, 1] - xy[1]) <= self.a.bld_radius)[0]
            self.G2 = len(C)
        m = np.ones(self.G2)
        for b, (wb, _) in self.w.items():
            idx = self.cell_of.get(b)
            if idx is not None and len(idx):
                m[idx] = np.maximum(m[idx], wb / VW.G_W[1])
        for i, ag in enumerate(agents):
            if self.share == 'own' and i != 0:
                continue
            r0, ra = self.base[i]
            ag.rho0, ag.rho_arr = r0 * m, ra * m
        return int((m > 1).sum())

    def __call__(self, W, agents, step):
        h0 = time.perf_counter()
        t = int(step)
        new = []
        if self.replay is not None:                      # twin: same weights at the same rounds, no VLM
            for st, b, wb, tc in self.replay:
                if st == t:
                    self.w[b] = (wb, tc)
            self._apply(agents)
            return
        # 1) capture request for this round (same timing as the step-6 Recorder: after the move of round t)
        if t % self.a.frame_every == 0:
            ne = self.fr.to_world(W.upos[0]).tolist()
            self.req_q.put(dict(kind='cap', t=t, wall=time.time(), pose_ne=ne))
            self.pending[t] = time.time(); self.n_req += 1
        # 2) results that arrived (non-blocking)
        while True:
            try:
                r = self.res_q.get_nowait()
            except Exception:
                break
            if r.get('kind') == 'error':
                self.n_err += 1; self.errors.append(r)
                continue
            if r.get('kind') != 'frame':
                continue
            self.n_res += 1
            self.pending.pop(r['t'], None)
            for k in [k for k in self.pending if k < r['t']]:       # superseded in the worker (kept newest only)
                self.pending.pop(k)
            self.dropped += r.get('dropped_before', 0)
            self.worker_hwm = r.get('vm_hwm_mb') or self.worker_hwm
            self.lat.append(r['vlm_ms']); self.lag.append(1e3 * (time.time() - r['wall_req']))
            r['applied_after_round'] = t
            r['age_at_use_s'] = (t + 1) - r['t']        # mission seconds from capture to the first decision that can use it
            r['lag_ms_req_to_apply'] = round(1e3 * (time.time() - r['wall_req']), 1)
            self.age_use.append(r['age_at_use_s'])
            if r['w_hat'] is not None:
                for b in r['seen']:
                    if b in self.cent:
                        self.w[b] = (r['w_hat'], r['t'])
                        self.applied.append((t, b, r['w_hat'], r['t']))
            new.append(r)
            self.ff.write(json.dumps(r, default=float) + '\n')
        # 3) planner weights for round t + 1
        ncell = self._apply(agents) if new else (int(self.n_cells) if hasattr(self, 'n_cells') else 0)
        self.n_cells = ncell
        ages = [(t + 1) - tc for _, tc in self.w.values()]
        mu = mem_used_mb()
        self.mem.append(mu)
        hm = 1e3 * (time.perf_counter() - h0)
        self.hook_ms.append(hm)
        self.fr_round.write(json.dumps(dict(
            t=t, new_frames=len(new), beliefs=len(self.w), cells_weighted=ncell,
            newest_age_s=min(ages) if ages else None, max_age_s=max(ages) if ages else None,
            pending=sorted(self.pending), hook_ms=round(hm, 3), mem_used_mb=round(mu, 1))) + '\n')

    def drain(self, wait_s):
        """End of mission: collect results still in flight (not applied; logged with applied_after_round = None)."""
        t_end = time.time() + wait_s
        while self.pending and time.time() < t_end:
            try:
                r = self.res_q.get(timeout=0.2)
            except Exception:
                continue
            if r.get('kind') == 'frame':
                self.pending.pop(r['t'], None); self.n_res += 1
                self.lat.append(r['vlm_ms']); self.lag.append(1e3 * (time.time() - r['wall_req']))
                r['applied_after_round'] = None; r['age_at_use_s'] = None
                self.ff.write(json.dumps(r, default=float) + '\n')
            elif r.get('kind') == 'error':
                self.n_err += 1; self.errors.append(r)

    def close(self):
        for f in (self.ff, self.fr_round):
            if f:
                f.close()

    def stats(self):
        def q(x):
            v = np.array([y for y in x if y is not None], float)
            return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()), n=int(v.size)) \
                if v.size else None
        return dict(mode=self.mode, share=self.share, requests=self.n_req, results=self.n_res, errors=self.n_err,
                    dropped=self.dropped, unanswered=len(self.pending), vlm_ms=q(self.lat), req_to_apply_ms=q(self.lag),
                    belief_age_at_use_s=q(self.age_use), hook_ms=q(self.hook_ms), mem_used_peak_mb=max(self.mem) if self.mem else None,
                    worker_vm_hwm_mb=self.worker_hwm, buildings_weighted=len(self.w),
                    weights_applied=len(self.applied), first_errors=self.errors[:3])


def run_mission(a, hil, cfg, method, seed, out_dir, city, vlm_mode, req_q, res_q, centres, blds):
    os.makedirs(out_dir, exist_ok=True)
    up0 = World(cfg, seed).upos.copy()
    z0 = city.start_height(0, up0[0]) if city is not None else cfg.altitude
    p_start = np.array([*up0[0], z0])
    if city is not None:
        city.hold_s[:] = 0; city.climb_m[:] = 0; city.clear.clear(); city.clear_r.clear()
    if not transfer(hil, city, p_start) or not hil.wait_at(p_start, tol=1.0, vtol=0.3, settle=5.0, timeout=600):
        return dict(method=method, seed=seed, vlm_planner=vlm_mode, aborted=True, note='did not reach the start point',
                    hil_state=str(hil.state()))
    TIMING['cur'] = {}
    if req_q is not None:
        req_q.put(dict(kind='outdir', out_dir=out_dir))
    log_f = open(os.path.join(out_dir, 'rounds.jsonl'), 'w')
    mover = HilMover(hil, cfg, log_f, a.v_cap, city)
    vl = VLMLoop(a, req_q, res_q, city.fr, centres, blds, vlm_mode, out_dir, a.vlm_share)
    hil.mark()
    mover.start(p_start, up0)
    t_wall = time.time()
    note, res = '', None
    try:
        res = EP.run_episode(cfg, method, seed, log_rounds=False, move=mover, step_hook=vl)
    except MissionAbort as e:
        note = str(e)
    log_f.close()
    link = hil.stats()
    wall = time.time() - t_wall
    hil.hold(hil.state()['p'])                           # city: hold here; the next transfer climbs first
    vl.drain(a.drain_s)
    vl.close()
    keys = ('J_session', 'qos_score', 'users_served', 'mean_wait_s', 'energy_wh', 'distance_m', 'continuity',
            'session_frac_mean', 'coord_msgs', 'coord_iters', 'decide_ms', 'decide_ms_p95', 'collisions', 'rtb',
            'disconnected_uav_s')
    t_k = time.perf_counter()
    kin = EP.run_episode(cfg, method, seed, log_rounds=False)
    t_kin = time.perf_counter() - t_k
    rep = None
    if vlm_mode == 'prior':                              # replay twin: no HIL, same building weights at the same rounds
        rv = VLMLoop(a, None, None, city.fr, centres, blds, 'prior', out_dir, a.vlm_share, replay=list(vl.applied))
        rep = EP.run_episode(cfg, method, seed, log_rounds=False, step_hook=rv)
    fl = mover.stats()
    out = dict(method=method, chain=a.chain, seed=seed, vlm_planner=vlm_mode, vlm_share=a.vlm_share, vlm_model=a.vlm_model,
               vlm_frames=a.vlm_frames, frame_every_s=a.frame_every, scenario=cfg.scenario, n_uav=cfg.n_uav, T=cfg.T,
               L=cfg.world, n_users=cfg.n_users, arrival_rate=cfg.arrival_rate, altitude=cfg.altitude, v_cap=a.v_cap,
               city=a.city, origin_ne=a.origin_ne, keepout=a.keepout, backend=a.backend, aborted=res is None, note=note,
               host=f'{platform.node()} {platform.machine()}', hil=res, kinematic={k: kin.get(k) for k in keys},
               replay_twin=None if rep is None else {k: rep.get(k) for k in keys}, kin_wall_s=round(t_kin, 2),
               flight=fl, link=link, vlm=vl.stats(), wall_s=round(wall, 1),
               hil_server_vm_hwm_mb=VW.vm_hwm_mb(hil.proc.pid), runner_vm_hwm_mb=VW.vm_hwm_mb())
    if res is not None:
        out['delta_J_pct'] = 100.0 * (res['J_session'] - kin['J_session']) / max(1e-9, kin['J_session'])
        if rep is not None:
            out['delta_J_vs_replay_pct'] = 100.0 * (res['J_session'] - rep['J_session']) / max(1e-9, rep['J_session'])
            out['delta_J_replay_vs_kin_pct'] = 100.0 * (rep['J_session'] - kin['J_session']) / max(1e-9, kin['J_session'])
    json.dump(out, open(os.path.join(out_dir, 'result.json'), 'w'), indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=['maqac_seq'])
    ap.add_argument('--no-chain', dest='chain', action='store_false', help='plain MAQAC-S (step-6 planner) instead of the chain rule')
    ap.add_argument('--seeds', nargs='+', type=int, default=[20000])
    ap.add_argument('--vlm-planner', nargs='+', default=['off'], choices=('off', 'prior'))
    ap.add_argument('--vlm-share', default='all', choices=('all', 'own'))
    ap.add_argument('--vlm-model', default='int8', choices=('int8', 'int8x4', 'fake'))
    ap.add_argument('--vlm-frames', default='airsim', choices=('airsim', 'relay', 'fake'))
    ap.add_argument('--model-dir', default=os.path.expanduser('~/vlm_damage_p3/model'), help='int8 model on the Jetson (step 8 bundle)')
    ap.add_argument('--ens-adapters', default=os.path.expanduser('~/vlm_damage_p3/adapters'))
    ap.add_argument('--vlm-python', default=None, help='interpreter of the VLM process, e.g. ~/vlm_damage_p3/venv/bin/python (torch); the runner may use the system python3 with pymavlink')
    ap.add_argument('--calib', default=None, help='default: int8_v2_calibration.npz next to this file (1 seed); int8x4 needs a 4-seed file')
    ap.add_argument('--buildings', default=os.path.join(HERE, 'buildings.json'))
    ap.add_argument('--fake-run', default=os.path.join(ROOT, 'results/vlm_damage/city_step6/runs/s6_n2_maqac_seq_20000'))
    ap.add_argument('--fake-logits', default=os.path.join(ROOT, 'results/vlm_damage/jetson/20261110_173516/city.jsonl'))
    ap.add_argument('--airsim-ip', default='192.168.55.100', help='laptop on the USB network (the AirSim city hardware-in-the-loop set-up)')
    ap.add_argument('--airsim-port', type=int, default=41451)
    ap.add_argument('--relay-port', type=int, default=41460)
    ap.add_argument('--frame-every', type=int, default=5)
    ap.add_argument('--bld-radius', type=float, default=25.0)
    ap.add_argument('--drain-s', type=float, default=8.0)
    ap.add_argument('--save-frames', action='store_true', help='keep the captured RGB frames (about 50 kB each)')
    ap.add_argument('--scenario', default='disaster')
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--L', type=float, default=600.0)
    ap.add_argument('--alt', type=float, default=100.0)
    ap.add_argument('--perception', default='onboard')
    ap.add_argument('--v-cap', type=float, default=14.0)
    ap.add_argument('--dev', default='/dev/ttyACM0')
    ap.add_argument('--warmup', type=float, default=40.0)
    ap.add_argument('--dry-run', action='store_true', help='fake HIL autopilot + fake frames + fake VLM')
    ap.add_argument('--dry-from-pad', action='store_true', help='dry run: start on the pad and fly the climb-out too')
    ap.add_argument('--router', action='store_true')
    ap.add_argument('--pad-ne', type=float, nargs=2, default=(0.0, -4.0))
    ap.add_argument('--city', default=None, help='default: survey/heightmap.npz of the VLM snapshot')
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(40.0, -10.0))
    ap.add_argument('--layer', type=float, default=3.0)
    ap.add_argument('--keepout', type=float, nargs=5, action='append', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.dry_run:
        a.vlm_model, a.vlm_frames = 'fake', 'fake'
    a.keepout = a.keepout if a.keepout is not None else STEP6_KEEPOUT
    if a.city is None:
        for c in (os.path.join(HERE, 'heightmap.npz'), os.path.join(HERE, '..', 'survey', 'heightmap.npz')):
            if os.path.exists(c):
                a.city = c; break
    if a.calib is None:
        a.calib = os.path.join(HERE, 'int8_v2_calibration.npz')
    os.makedirs(a.out, exist_ok=True)
    cfg = make_cfg(a)
    timed_methods(a.methods)
    from city import City
    city = City(a.city, L=a.L, origin_ne=a.origin_ne, alt=a.alt, layer=a.layer, n=a.n, keepout=a.keepout)
    blds = json.load(open(a.buildings))
    cxy = city.fr.to_area(np.array([[b['N'], b['E']] for b in blds]))
    centres = {b['name']: cxy[k] for k, b in enumerate(blds)
               if 0 <= cxy[k][0] <= a.L and 0 <= cxy[k][1] <= a.L}           # buildings inside the mission area
    missions = [(s, m, v) for s in a.seeds for m in a.methods for v in a.vlm_planner]   # complete seeds first
    a.backend = 'fake' if a.dry_run else ('router' if a.router else 'px4')
    if a.router and a.dev == '/dev/ttyACM0':
        a.dev = 'udpout:127.0.0.1:14560'
    pad = np.asarray(a.pad_ne, float)
    p_ground = np.array([*city.fr.to_area(pad), float(city.ground(city.fr.to_area(pad))[0])])
    if a.dry_run and not a.dry_from_pad:                 # dry run: put the fake vehicle on the first start point
        u = World(cfg, a.seeds[0]).upos[0]
        p_ground = np.array([*u, city.start_height(0, u)])
    print(f'VLM HIL missions {len(missions)}: {missions}; ground {np.round(p_ground, 1).tolist()}; buildings in area '
          f'{len(centres)}; chain {a.chain}; VLM {a.vlm_model}/{a.vlm_frames}', flush=True)
    # ---- VLM worker first (spawn: CUDA must not be forked), then the HIL server (fork, as the earlier MAQAC work)
    ctx = mp.get_context('spawn')
    vpath = []
    if a.vlm_python:
        ctx.set_executable(os.path.expanduser(a.vlm_python))
        # spawn hands the child the PARENT's sys.path, so the venv's site-packages (torch) would be missing (12 Nov fix)
        import subprocess
        vpath = subprocess.run([os.path.expanduser(a.vlm_python), '-c', 'import sys; print("\\n".join(p for p in sys.path if p))'],
                               capture_output=True, text=True, check=True).stdout.split()
    req_q, res_q = ctx.Queue(), ctx.Queue()
    wcfg = dict(frames=a.vlm_frames, model=a.vlm_model, calib=a.calib, out_dir=a.out, ip=a.airsim_ip,
                port=a.relay_port if a.vlm_frames == 'relay' else a.airsim_port, vehicle='UAV0', camera='down',
                fake_run=a.fake_run, fake_logits=a.fake_logits, model_dir=a.model_dir, ens_adapters=a.ens_adapters,
                buildings=a.buildings, save_frames=a.save_frames, sys_path=vpath)
    wp = ctx.Process(target=VW.worker, args=(req_q, res_q, wcfg), daemon=True)
    t0 = time.time()
    if vpath:    # spawn copies the parent's sys.path into the child BEFORE it imports vlm_worker (numpy, torch):
        import sys as _s; _keep = list(_s.path)       # hand it the VLM interpreter's paths, then restore ours (12 Nov fix)
        _s.path[:] = [os.path.dirname(os.path.abspath(__file__))] + vpath
        try:
            wp.start()
        finally:
            _s.path[:] = _keep
    else:
        wp.start()
    r = res_q.get(timeout=600)
    print(f'{time.strftime("%T")} VLM worker: {json.dumps(r, default=float)[:400]} ({time.time() - t0:.1f} s)', flush=True)
    if r.get('kind') != 'ready':
        raise SystemExit('VLM worker failed to load: stopping before the HIL link starts')
    hil = HilClient(a.backend, a.dev, p_ground, warmup=a.warmup if not a.dry_run else 2.0, verbose=a.verbose)
    summ = open(os.path.join(a.out, 'missions.jsonl'), 'a')
    try:
        if not hil.wait_ready(timeout=a.warmup + 200):
            raise SystemExit('HIL not ready (EKF / link): stopping')
        print(f'{time.strftime("%T")} HIL ready: {hil.state()}', flush=True)
        for seed, method, vmode in missions:
            name = f'{a.tag}{method}{"_chain" if a.chain else ""}_vlm{vmode}_s{seed}' + (f'_T{a.T}' if a.T != 300 else '')
            print(f'{time.strftime("%T")} mission {name}', flush=True)
            r = run_mission(a, hil, cfg, method, seed, os.path.join(a.out, name), city, vmode, req_q, res_q, centres, blds)
            r['name'] = name
            short = {k: r.get(k) for k in ('name', 'aborted', 'note', 'delta_J_pct', 'delta_J_vs_replay_pct', 'wall_s')}
            if r.get('hil'):
                v = r['vlm']
                short.update(J_hil=r['hil']['J_session'], J_kin=r['kinematic']['J_session'],
                             J_replay=(r['replay_twin'] or {}).get('J_session'),
                             err_h=r['flight']['track_err_h_m'], overruns=r['flight']['round_overruns'],
                             fleet_ms=r['flight']['fleet_decide_ms'], round_ms=r['flight']['round_compute_ms'],
                             vlm=dict(req=v['requests'], res=v['results'], dropped=v['dropped'], err=v['errors'],
                                      vlm_ms=v['vlm_ms'], age=v['belief_age_at_use_s'], mem_peak_mb=v['mem_used_peak_mb'],
                                      bld=v['buildings_weighted']),
                             link={k: r['link'].get(k) for k in ('setpoint_hz', 'actuator_hz', 'hil_sensor_hz', 'rtt_ms_mean',
                                                                  'mode_drops', 'failsafe_msgs', 'geofence_msgs')})
            print('  RESULT ' + json.dumps(short, default=float), flush=True)
            summ.write(json.dumps(short, default=float) + '\n'); summ.flush()
            if r.get('aborted') and r.get('note', '').startswith('did not reach'):
                print('  vehicle not at the start point: stopping the batch', flush=True)
                break
        if not a.dry_run or a.dry_from_pad:
            u_end = p_ground[:2]
            p_end = np.array([*u_end, city.start_height(0, u_end)])
            transfer(hil, city, p_end)
            hil.wait_at(p_end, tol=2.0, vtol=0.5, settle=2.0, timeout=300)
        if not a.dry_run or a.dry_from_pad:             # dry run on the start point: nothing to land (saves ~80 s)
            print(f'{time.strftime("%T")} landing', flush=True)
            hil.land()
    finally:
        summ.close()
        hil.close()
        req_q.put(None)
        wp.join(20)
        if wp.is_alive():
            wp.terminate()
    print('VLM HIL done', flush=True)


if __name__ == '__main__':
    main()
