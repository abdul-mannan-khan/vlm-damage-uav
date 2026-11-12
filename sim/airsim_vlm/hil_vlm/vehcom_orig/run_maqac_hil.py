#!/usr/bin/env python3
"""MAQAC hardware-in-the-loop missions: UAV 0 of the fleet is flown by the real PX4 firmware on a Pixhawk (HITL),
the decisions of the whole fleet are computed on the companion computer (Jetson, timed), UAVs 1..n-1 stay on the
network simulator's kinematic model. Real time: one 1 s decision round = 1 s of wall time.

Network model: src/netsim (v3, sim.realistic(), takeoff=False as in the AirSim bridge). The move hook follows
sim/airsim/maqac_px4/run_maqac_airsim.py (class Mover): W.move_uavs() gives the kinematic reference, UAV 0's
reference for the coming round goes to PX4 as a smooth segment (OFFBOARD position + velocity + acceleration
setpoints at 50 Hz), and UAV 0's true position from the HIL physics replaces the reference in W.upos[0] / W.ualt[0].

  python3 run_maqac_hil.py --methods maqac_seq maqac rag --seeds 20000 20001 --out results/hil        # Pixhawk
  python3 run_maqac_hil.py --dry-run --T 30 --methods maqac_seq --seeds 20000 --out /tmp/hil_dry         # no hardware
  python3 run_maqac_hil.py --router --L 600 --city survey/heightmap.npz --out results/hil_airsim           # AirSim
With --router, UAV 0's physics and sensors come from AirSim CityEnviron on the laptop through W2's p27_router.py
(started separately on the Jetson); the vehicle starts on its AirSim pad (--pad-ne) and every transfer between
pad, start points and the end climbs vertically above the highest surface on the straight line first.
Each mission writes <out>/<method>_s<seed>[_T<T>]/result.json and rounds.jsonl; <out>/missions.jsonl collects them.
The kinematic twin (same config, method and seed, no HIL) is run on the same computer after each mission.
"""
import argparse, json, math, os, platform, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, '..', '..', 'src', 'netsim'), os.path.join(HERE, 'netsim')):
    if os.path.isdir(p):
        sys.path.insert(0, p)
import agents as AG                                              # noqa: E402
import episode as EP                                             # noqa: E402
from sim import World, propulsion_power, realistic               # noqa: E402
from hil_link import HilClient, hermite                          # noqa: E402

TIMING = dict(cur={}, rounds=[])                                 # per-round decide() times by UAV index


def timed_methods(methods):
    """Wrap the agents' decide() (class swap in METHODS; src/netsim unchanged) to time every call by UAV."""
    for m in methods:
        cls, flags, cons, sep = AG.METHODS[m]
        if getattr(cls, '_timed', False):
            continue

        class Timed(cls):
            _timed = True

            def decide(self, prev_events=()):
                t0 = time.perf_counter()
                r = super().decide(prev_events)
                TIMING['cur'][self.i] = TIMING['cur'].get(self.i, 0.0) + time.perf_counter() - t0
                return r
        Timed.__name__ = 'Timed' + cls.__name__
        AG.METHODS[m] = (Timed, flags, cons, sep)
        EP.METHODS[m] = AG.METHODS[m]


def make_cfg(a):
    s = (a.L / 1000.0) ** 2
    return realistic(name=f'{a.scenario}_{a.n}uav', scenario=a.scenario, n_uav=a.n, T=a.T, perception=a.perception,
                     altitude=a.alt, world=a.L, gateway=(a.L / 2, a.L / 2), n_users=int(round(200 * s)),
                     arrival_rate=0.1 * s, takeoff=False)


class HilMover:
    """move(W, targets, sep, t) for run_episode: UAV 0 flown in HITL, the others kinematic."""

    def __init__(self, hil, cfg, log_f, v_cap, city=None):
        self.h, self.cfg, self.log_f, self.v_cap, self.city = hil, cfg, log_f, v_cap, city
        self.act_z = None
        self.PW = dict(W=cfg.weight_n, A=cfg.rotor_area)
        self.t0 = None                     # monotonic time of round 0
        self.seg = None
        self.err_h, self.err_3d, self.sep = [], [], []
        self.round_ms, self.fleet_ms, self.uav_ms, self.overrun = [], [], [], 0
        self.t_hook_end = None
        self.z_err = []

    def start(self, p_start, upos0=None):
        if self.city is not None:                       # commanded heights from the start points (climb rule)
            self.city.z = np.array([self.city.start_height(i, upos0[i]) for i in range(self.cfg.n_uav)])
            self.city.z[0] = p_start[2]
        self.act_z = float(p_start[2])
        now = time.monotonic()
        self.t0 = now
        self.seg = (now - 1.0, now, p_start, np.zeros(3), p_start, np.zeros(3))
        self.t_hook_end = now

    def __call__(self, W, targets, sep, t):
        c = self.cfg
        t_in = time.monotonic()
        # timing of this round's decisions (everything since the previous hook ended = network step + decisions)
        rc = 1e3 * (t_in - self.t_hook_end)
        cur = TIMING['cur']
        TIMING['cur'] = {}
        fleet = 1e3 * sum(cur.values())
        self.round_ms.append(rc)
        if cur:
            self.fleet_ms.append(fleet)
            self.uav_ms.extend(1e3 * v for v in cur.values())
        n = c.n_uav
        air0 = W.airborne().copy()
        mode0 = W.mode.copy()
        old = W.upos.copy()
        W.move_uavs(targets, sep, t)                     # kinematic reference, separation rule, battery logic
        ref = W.upos.copy()
        W.upos[:] = old
        # ---- UAV 0: segment over [t0 + t, t0 + t + 1] ending at the kinematic reference
        ta, tb = self.t0 + t, self.t0 + t + 1.0
        C = self.city
        held = np.zeros(n, bool)
        dup = np.zeros(n)
        if C is not None:                               # tower-climb rule for every UAV (AirSim bridge's Mover)
            tg = np.asarray(targets, float)
            for i in range(n):
                swap = W.mode[i] == 2 or not W.alive[i]
                dst = W.pad[i] if swap else ref[i]
                up_now = self.act_z if i == 0 else C.z[i]
                up_req, hold = C.required(i, old[i], dst, W.pad[i] if swap else tg[i], up_now)
                dz = C.step_z(i, up_req)
                if i > 0:
                    dup[i] = max(0.0, dz)
                if hold and not swap:
                    held[i] = True
                    C.hold_s[i] += 1.0
                    if i > 0:
                        ref[i] = old[i]
        alt0 = C.z[0] if C is not None else c.altitude
        if W.mode[0] == 2 or not W.alive[0]:
            pb, vb = np.array([*W.pad[0], alt0]), np.zeros(3)
        elif held[0]:
            pb, vb = np.array([*old[0], alt0]), np.zeros(3)
        else:
            pb = np.array([*ref[0], alt0])
            vb = np.array([*W.uvel[0], 0.0])
        sp_v = np.linalg.norm(vb[:2])
        if sp_v > self.v_cap:
            vb[:2] *= self.v_cap / sp_v
        pa, va, _ = hermite(self.seg, ta)
        self.seg = (ta, tb, pa, va, pb, vb)
        self.h.seg(*self.seg)
        # ---- wait for the end of the round (real time), then read the flown state
        late = time.monotonic() - tb
        if late > 0:
            self.overrun += 1
        while time.monotonic() < tb:
            time.sleep(min(0.01, max(0.0, tb - time.monotonic())))
        s = self.h.state()
        act = s['p']
        # the model starts a battery swap when its reference reached the gateway; require the real vehicle there
        if mode0[0] == 1 and W.mode[0] == 2 and np.hypot(*(act[:2] - W.gw)) >= 5.0:
            W.mode[0] = 1
        W.upos[1:] = ref[1:]
        if C is not None:
            dup[0] = max(0.0, float(act[2]) - self.act_z)
            for i in range(1, n):                       # kinematic UAVs held below a tower: energy of the distance not flown
                if held[i] and air0[i]:
                    mr = float(np.hypot(*(W.upos[i] - old[i])))
                    d = (propulsion_power(0.0, **self.PW) - propulsion_power(mr, **self.PW)) / c.prop_eff
                    W.energy[i] -= d
                    W.m['energy_j'] += d
                    W.m['dist'] -= mr
            # NB: W.upos[i] = old[i] for held kinematic UAVs (ref[i] was set to old[i] above)
        self.act_z = float(act[2])
        e_h = e_3 = float('nan')
        if W.alive[0] and W.mode[0] != 2:
            W.upos[0] = np.clip(act[:2], 0, c.world)
            W.ualt[0] = max(10.0, float(act[2]) - (float(C.ground(act[:2])[0]) if C is not None else 0.0))
            e_h = float(np.hypot(*(act[:2] - ref[0])))
            e_3 = float(np.linalg.norm(act - pb))
            if air0[0]:                                   # energy and distance from the distance actually flown
                mr, ma = float(np.hypot(*(ref[0] - old[0]))), float(np.hypot(*(W.upos[0] - old[0])))
                d = (propulsion_power(ma, **self.PW) - propulsion_power(mr, **self.PW)) / c.prop_eff
                W.energy[0] -= d
                W.m['energy_j'] += d
                W.m['dist'] += ma - mr
        else:
            W.upos[0] = ref[0]
        if C is not None:
            for i in range(1, n):                       # radio height above the local ground
                W.ualt[i] = max(10.0, C.z[i] - float(C.ground(W.upos[i])[0]))
            for i in range(n):                          # climb energy (as the AirSim bridge: W * dh)
                if air0[i] and dup[i] > 0:
                    W.energy[i] -= 20.0 * dup[i]
                    W.m['energy_j'] += 20.0 * dup[i]
                    C.climb_m[i] += dup[i]
            xy = np.vstack([act[None, :2], W.upos[1:]])
            clr, clr_r = C.log_clearance(xy, np.concatenate([[act[2]], C.z[1:]]))
            others = np.column_stack([W.upos[1:], C.z[1:]])
        else:
            clr = clr_r = None
            others = np.column_stack([W.upos[1:], W.ualt[1:]])
        d3 = np.linalg.norm(others - act[None, :], axis=1) if n > 1 else np.array([np.inf])
        self.err_h.append(e_h)
        self.err_3d.append(e_3)
        self.z_err.append(float(act[2] - pb[2]))
        self.sep.append(float(d3.min()))
        self.log_f.write(json.dumps(dict(
            t=int(t), ref0=np.round(pb, 2).tolist(), act0=np.round(act, 2).tolist(), v0=np.round(s['v'], 2).tolist(),
            err_h=None if math.isnan(e_h) else round(e_h, 3), err_3d=None if math.isnan(e_3) else round(e_3, 3),
            sep0_3d=np.round(d3, 2).tolist(), upos=np.round(W.upos, 1).tolist(), mode=W.mode.tolist(),
            armed=s['armed'], px4_mode=list(s['mode']), round_ms=round(rc, 3), fleet_decide_ms=round(fleet, 3),
            uav_decide_ms={int(k): round(1e3 * v, 3) for k, v in cur.items()}, late_s=round(late, 4),
            served=int(np.sum(~np.isnan(W.first_t))),
            z_cmd=None if C is None else np.round(C.z, 1).tolist(), held=held.tolist(),
            clear=None if clr is None else round(clr, 2), clear_r=None if clr_r is None else round(clr_r, 2))) + '\n')
        self.t_hook_end = time.monotonic()
        if not s['armed'] and self.h.backend == 'px4':
            raise MissionAbort(f'PX4 disarmed at t={t}')
        if not math.isnan(e_3) and e_3 > 50.0:
            raise MissionAbort(f'tracking error {e_3:.1f} m > 50 m at t={t}')

    def stats(self):
        def q(x):
            v = np.array([y for y in x if not math.isnan(y)], float)
            return dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(v.size)) \
                if v.size else None
        return dict(track_err_h_m=q(self.err_h), track_err_3d_m=q(self.err_3d), alt_err_m=q([abs(z) for z in self.z_err]),
                    min_sep_3d_m=float(np.min(self.sep)) if self.sep else None,
                    round_compute_ms=q(self.round_ms[1:]), fleet_decide_ms=q(self.fleet_ms), uav_decide_ms=q(self.uav_ms),
                    round_overruns=self.overrun, rounds=len(self.sep),
                    city=self.city.stats() if self.city is not None else None)


def transfer(hil, city, p_to, timeout=600):
    """Fly to p_to. In the city: climb vertically to clear every surface on the straight line (15 m ceiling map + margin
    + 10 m), cross at that height, then descend to p_to. Without a city: straight hold."""
    p_to = np.asarray(p_to, float)
    legs = [p_to]
    if city is not None:
        p = hil.state()['p']
        need = city.cl.need(city.w(p[:2]), city.w(p_to[:2]))
        zt = max(p[2], p_to[2], need + city.margin + 10.0)
        legs = [np.array([p[0], p[1], zt]), np.array([p_to[0], p_to[1], zt]), p_to]
    for q in legs[:-1]:
        hil.hold(q)
        if not hil.wait_at(q, tol=2.0, vtol=0.6, settle=1.0, timeout=timeout):
            return False
    hil.hold(legs[-1])
    return True


class MissionAbort(Exception):
    pass


def run_mission(a, hil, cfg, method, seed, out_dir, city=None):
    os.makedirs(out_dir, exist_ok=True)
    up0 = World(cfg, seed).upos.copy()
    z0 = city.start_height(0, up0[0]) if city is not None else cfg.altitude
    p_start = np.array([*up0[0], z0])
    if city is not None:
        city.hold_s[:] = 0; city.climb_m[:] = 0; city.clear.clear(); city.clear_r.clear()
    if not transfer(hil, city, p_start) or not hil.wait_at(p_start, tol=1.0, vtol=0.3, settle=5.0, timeout=600):
        return dict(method=method, seed=seed, aborted=True, note='did not reach the start point', hil_state=str(hil.state()))
    TIMING['cur'] = {}
    log_f = open(os.path.join(out_dir, 'rounds.jsonl'), 'w')
    mover = HilMover(hil, cfg, log_f, a.v_cap, city)
    hil.mark()
    mover.start(p_start, up0)
    t_wall = time.time()
    note, res = '', None
    try:
        res = EP.run_episode(cfg, method, seed, log_rounds=False, move=mover)
    except MissionAbort as e:
        note = str(e)
    log_f.close()
    link = hil.stats()
    wall = time.time() - t_wall
    if city is None:
        hil.hold(p_start)                                # fly back while the twin runs
    else:
        hil.hold(hil.state()['p'])                       # city: hold here; the next transfer climbs first
    t_k = time.perf_counter()
    kin = EP.run_episode(cfg, method, seed, log_rounds=False)
    t_kin = time.perf_counter() - t_k
    keys = ('J_session', 'qos_score', 'users_served', 'mean_wait_s', 'energy_wh', 'distance_m', 'continuity',
            'session_frac_mean', 'coord_msgs', 'coord_iters', 'decide_ms', 'decide_ms_p95', 'collisions', 'rtb')
    out = dict(method=method, seed=seed, scenario=cfg.scenario, n_uav=cfg.n_uav, T=cfg.T, L=cfg.world,
               n_users=cfg.n_users, arrival_rate=cfg.arrival_rate, altitude=cfg.altitude, speed=cfg.speed,
               v_cap=a.v_cap, city=a.city, origin_ne=a.origin_ne if a.city else None, backend=a.backend, aborted=res is None, note=note,
               host=f'{platform.node()} {platform.machine()} {platform.processor() or ""}'.strip(),
               hil=res, kinematic={k: kin.get(k) for k in keys}, kin_wall_s=round(t_kin, 2),
               flight=mover.stats(), link=link, wall_s=round(wall, 1))
    if res is not None:
        out['delta_J_pct'] = 100.0 * (res['J_session'] - kin['J_session']) / max(1e-9, kin['J_session'])
        out['delta_qos_pct'] = 100.0 * (res['qos_score'] - kin['qos_score']) / max(1e-9, kin['qos_score'])
    json.dump(out, open(os.path.join(out_dir, 'result.json'), 'w'), indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=['maqac_seq', 'maqac', 'rag'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[20000])
    ap.add_argument('--scenario', default='urban')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--L', type=float, default=1000.0)
    ap.add_argument('--alt', type=float, default=100.0)
    ap.add_argument('--perception', default='onboard')
    ap.add_argument('--v-cap', type=float, default=14.0, help='cap on the reference speed sent to PX4 (m/s)')
    ap.add_argument('--dev', default='/dev/ttyACM0')
    ap.add_argument('--warmup', type=float, default=40.0)
    ap.add_argument('--dry-run', action='store_true', help='no Pixhawk: first-order autopilot model')
    ap.add_argument('--router', action='store_true', help='AirSim physics via p27_router (UDP 14560 on the Jetson)')
    ap.add_argument('--pad-ne', type=float, nargs=2, default=(0.0, -4.0), help='AirSim spawn pad of UAV 0 (world N, E)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--city', default=None, help='height map (.npz) of the AirSim CityEnviron survey: tower-climb rule as in the AirSim campaign')
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(40.0, -10.0))
    ap.add_argument('--layer', type=float, default=3.0)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = make_cfg(a)
    timed_methods(a.methods)
    city = None
    if a.city:
        from city import City
        city = City(a.city, L=a.L, origin_ne=a.origin_ne, alt=a.alt, layer=a.layer, n=a.n)
    missions = [(s, m) for s in a.seeds for m in a.methods]     # complete seeds first (all methods per seed)
    a.backend = 'fake' if a.dry_run else ('router' if a.router else 'px4')
    if a.router:
        if city is None:
            raise SystemExit('--router needs --city (the AirSim map)')
        if a.dev == '/dev/ttyACM0':
            a.dev = 'udpout:127.0.0.1:14560'
        pad = np.asarray(a.pad_ne, float)               # PX4 local origin = UAV 0 resting on its AirSim pad, so its
        p_ground = np.array([*city.fr.to_area(pad), float(city.ground(city.fr.to_area(pad))[0])])   # z is the ground there
    else:
        p_ground = np.array([*World(cfg, a.seeds[0]).upos[0], 0.0])
    print(f'HIL missions {len(missions)}: {missions}; start/ground {p_ground.tolist()}; cfg speed {cfg.speed} '
          f'a_max {cfg.a_max} alt {cfg.altitude} users {cfg.n_users}+{cfg.arrival_rate}/s', flush=True)
    hil = HilClient(a.backend, a.dev, p_ground, warmup=a.warmup, verbose=a.verbose)
    summ = open(os.path.join(a.out, 'missions.jsonl'), 'a')
    try:
        if not hil.wait_ready(timeout=a.warmup + 200):
            raise SystemExit('HIL not ready (EKF / link): stopping')
        print(f'{time.strftime("%T")} HIL ready: {hil.state()}', flush=True)
        for seed, method in missions:
            name = f'{a.tag}{method}_s{seed}' + (f'_T{a.T}' if a.T != 300 else '')
            print(f'{time.strftime("%T")} mission {name}', flush=True)
            r = run_mission(a, hil, cfg, method, seed, os.path.join(a.out, name), city)
            r['name'] = name
            short = {k: r.get(k) for k in ('name', 'aborted', 'note', 'delta_J_pct', 'wall_s')}
            if r.get('hil'):
                short.update(J_hil=r['hil']['J_session'], J_kin=r['kinematic']['J_session'],
                             clear=(r['flight'].get('city') or {}).get('min_clearance_m'),
                             err_h=r['flight']['track_err_h_m'], sep=r['flight']['min_sep_3d_m'],
                             fleet_ms=r['flight']['fleet_decide_ms'], link={k: r['link'].get(k) for k in
                             ('setpoint_hz', 'actuator_hz', 'hil_sensor_hz', 'rtt_ms_mean', 'mode_drops', 'failsafe_msgs', 'geofence_msgs')})
            print('  RESULT ' + json.dumps(short, default=float), flush=True)
            summ.write(json.dumps(short, default=float) + '\n')
            summ.flush()
            if r.get('aborted') and r.get('note', '').startswith('did not reach'):
                print('  vehicle not at the start point: stopping the batch', flush=True)
                break
        u_end = p_ground[:2] if a.router else World(cfg, a.seeds[0]).upos[0]
        p_end = np.array([*u_end, city.start_height(0, u_end) if city is not None else cfg.altitude])
        transfer(hil, city, p_end)
        hil.wait_at(p_end, tol=2.0, vtol=0.5, settle=2.0, timeout=300)
        print(f'{time.strftime("%T")} landing', flush=True)
        hil.land()
    finally:
        summ.close()
        hil.close()
    print('HIL done', flush=True)


if __name__ == '__main__':
    main()
