#!/usr/bin/env python3
"""Onboard damage VLM for the closed-loop HIL: frame grab + int8 VLM + calibrated belief + conformal set, in its own process.

The mission runner (run_vlm_hil.py) sends a capture request every --frame-every s of mission time; this process grabs
UAV 0's downward frame (Scene + Segmentation, as the step-6 Recorder in run_maqac_airsim.py), runs the VLM, applies the
saved belief model with the Belief class of scripts/vlm_damage_live.py (vector scaling per seed, mean, ensemble
temperature, alpha = 0.1 split-conformal set), turns the set into a planner weight with the set-midpoint rule of
Lemma 2 (g = {1: 1.0, 2: 1.5, 3: 2.5, 4: 3.0}, scripts/verify_perception_bound.py; class A 'no building' excluded) and
attributes the frame to the buildings that fill >= 2 % of it (segmentation, buildings.Resolver; the step-6 'seen' rule).
The runner never waits for it: requests and results go through multiprocessing queues; if a request arrives while the
model is busy, only the newest waiting request is kept (the older ones are counted as dropped).

Backends
  frames: 'airsim' (AirSim RPC on the laptop, default 192.168.55.100:41451, vehicle UAV0, camera 'down'),
          'relay'  (frame_relay.py on the laptop serves the same two images over HTTP; no airsim package on the Jetson),
          'fake'   (step-6 frames results/vlm_damage/city_step6/runs/<run>/frames/<t>_uav0_rgb.jpg + labels.jsonl)
  model:  'int8'   (bench_vlm.load / run_one: seed 0 merged, int8 LM weights; ~1.8 s per 640x480 frame on the Orin Nano)
          'int8x4' (bench_vlm_ens: 4 seeds on one int8 model, ~3.6 s; RAM peak 7.5 of 7.6 GB -> see PLAN.md)
          'fake'   (stored Jetson logits results/vlm_damage/jetson/20261110_173516/city.jsonl; sleeps the stored time)
"""
import glob
import io
import json
import os
import queue
import time

import numpy as np

G_W = {1: 1.0, 2: 1.5, 3: 2.5, 4: 3.0}         # priority weight per damage class (verify_perception_bound.G)
SEEN_FRAC = 0.02                                # a building counts as seen when it fills >= 2 % of the frame (step 6)


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


class Belief:
    """Copied from scripts/vlm_damage_live.py (step 6) unchanged; with one seed and te = 1 it is plain vector scaling."""
    def __init__(self, path):
        c = np.load(path)
        self.W, self.B, self.te, self.qhat = c['W'], c['B'], float(c['te']), float(c['qhat'])

    def __call__(self, L):                      # L: seeds x 5 letter logits
        P = np.stack([softmax((L[s] - L[s].max()) * self.W[s] + self.B[s]) for s in range(len(L))])
        Pm = softmax(np.log(P.mean(0)) / self.te)
        return Pm, [int(k) for k in np.where(1 - Pm <= self.qhat)[0]]


def midpoint_weight(cset):
    """Lemma 2 set-midpoint rule over the damage classes in the set; None if the set holds no damage class."""
    g = [G_W[c] for c in cset if c in G_W]
    return None if not g else 0.5 * (min(g) + max(g))


def vm_hwm_mb(pid='self'):
    try:
        for l in open(f'/proc/{pid}/status'):
            if l.startswith('VmHWM'):
                return int(l.split()[1]) / 1024.0
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ frame sources
class AirSimFrames:
    def __init__(self, ip, port, vehicle, camera, buildings=None):
        import airsim
        self.A = airsim
        self.c = airsim.MultirotorClient(ip=ip, port=port, timeout_value=10)
        self.c.confirmConnection()
        self.veh, self.cam = vehicle, camera
        if buildings is not None:
            from buildings import apply_ids
            apply_ids(self.c, buildings)       # segmentation IDs of the step-6 inventory (needs --no-users visuals)
        # the first segmentation capture per camera is stale (step-6 pilot, 9 Nov): discard it
        self.c.simGetImages([airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False)], vehicle_name=vehicle)

    def grab(self, t):
        A = self.A
        rs = self.c.simGetImages([A.ImageRequest(self.cam, A.ImageType.Scene, False, False),
                                  A.ImageRequest(self.cam, A.ImageType.Segmentation, False, False)], vehicle_name=self.veh)
        f = [np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1] for r in rs]
        return f[0], f[1], None


class RelayFrames:
    """GET http://host:port/frame -> .npz with rgb, seg (uint8 HxWx3), served by frame_relay.py on the laptop."""
    def __init__(self, ip, port, vehicle, camera):
        import urllib.request
        self.u, self.url = urllib.request, f'http://{ip}:{port}/frame?vehicle={vehicle}&camera={camera}'

    def grab(self, t):
        with self.u.urlopen(self.url, timeout=5) as r:
            z = np.load(io.BytesIO(r.read()))
        return z['rgb'], z['seg'], None


class FakeFrames:
    """Step-6 frames of one mission: frame at mission time t = <t>_uav0_rgb.jpg; buildings from its labels.jsonl."""
    def __init__(self, run_dir):
        from PIL import Image
        self.Image, self.dir = Image, os.path.join(run_dir, 'frames')
        self.run = os.path.basename(os.path.normpath(run_dir))
        self.lab = {}
        p = os.path.join(self.dir, 'labels.jsonl')
        if os.path.exists(p):
            self.lab = {r['file']: r for r in map(json.loads, open(p)) if r['uav'] == 0}
        self.files = sorted(glob.glob(os.path.join(self.dir, '*_uav0_rgb.jpg')))
        assert self.files, f'no step-6 frames in {self.dir}'

    def grab(self, t):
        base = f'{int(t):04d}_uav0'
        f = os.path.join(self.dir, base + '_rgb.jpg')
        if not os.path.exists(f):
            f = self.files[(int(t) // 5) % len(self.files)]
            base = os.path.basename(f)[:-8]
        rgb = np.asarray(self.Image.open(f).convert('RGB'))
        r = self.lab.get(base)
        return rgb, None, dict(file=f'{self.run}__{base}_rgb.jpg', bld=r['bld'] if r else None)


# ------------------------------------------------------------------ models
class FakeModel:
    """Stored int8 logits from the Jetson run (same frame if it was benchmarked, else the next stored one in a fixed
    cycle); sleeps the stored model time so the runner sees the real latency."""
    def __init__(self, path, speed=1.0):
        self.rows = [json.loads(l) for l in open(path)]
        self.by_id = {r['id']: r for r in self.rows}
        self.k, self.speed = 0, speed

    def __call__(self, rgb, info):
        r = self.by_id.get((info or {}).get('file'))
        src = 'stored_same_frame'
        if r is None:
            r = self.rows[self.k % len(self.rows)]; self.k += 1; src = 'stored_other_frame'
        time.sleep(r['total_ms'] / 1e3 / self.speed)
        return np.array([r['logits']], float), dict(stored_id=r['id'], logit_source=src)


class Int8Model:
    def __init__(self, model_dir, ens_adapters=None):
        import torch
        from PIL import Image
        self.torch, self.Image = torch, Image
        if ens_adapters:
            import bench_vlm_ens as E
            self.m, self.p, self.letters, info = E.load(model_dir)
            info['n_delta'] = E.attach(self.m, ens_adapters)
            self.run = lambda im: E.run_ens(self.m, self.p, self.letters, im)[0]
        else:
            import bench_vlm as B
            self.m, self.p, self.letters, info = B.load(model_dir)
            self.run = lambda im: B.run_one(self.m, self.p, self.letters, im)[0][None, :]
        self.info = info
        self.info['cuda_alloc_mb_after_load'] = torch.cuda.memory_allocated() / 2 ** 20

    def __call__(self, rgb, info):
        return np.asarray(self.run(self.Image.fromarray(rgb)), float), {}


# ------------------------------------------------------------------ worker process
def worker(req_q, res_q, cfg):
    """Process body. cfg: dict(frames, model, calib, out_dir, ip, port, vehicle, camera, fake_run, fake_logits,
    model_dir, ens_adapters, buildings, save_frames, warmup_frame)."""
    import sys
    for q in reversed(cfg.get('sys_path') or []):      # the VLM interpreter's own paths first (venv with torch)
        if q not in sys.path:
            sys.path.insert(0, q)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t_load = time.time()
    try:
        bel = Belief(cfg['calib'])
        blds = json.load(open(cfg['buildings'])) if cfg.get('buildings') else None
        if cfg['model'] == 'fake':
            model = FakeModel(cfg['fake_logits'], cfg.get('fake_speed', 1.0))
        else:
            model = Int8Model(cfg['model_dir'], cfg.get('ens_adapters') if cfg['model'] == 'int8x4' else None)
        n_seeds = len(bel.W)
        if cfg['frames'] == 'fake':
            src = FakeFrames(cfg['fake_run'])
        elif cfg['frames'] == 'relay':
            src = RelayFrames(cfg['ip'], cfg['port'], cfg['vehicle'], cfg['camera'])
        else:
            src = AirSimFrames(cfg['ip'], cfg['port'], cfg['vehicle'], cfg['camera'], blds)
        resolver = None
        if blds is not None and cfg['frames'] != 'fake':
            from buildings import Resolver
            resolver = Resolver(blds)
        rgb0, _, i0 = src.grab(0)                       # warm-up (not recorded): CUDA kernels, first capture
        for _ in range(cfg.get('warmup', 2)):
            L0, _ = model(rgb0, i0)
        assert L0.shape == (n_seeds, 5), f'model gives {L0.shape} logits, calibration has {n_seeds} seed(s)'
        res_q.put(dict(kind='ready', load_s=round(time.time() - t_load, 2), seeds=n_seeds,
                       info=getattr(model, 'info', None), vm_hwm_mb=vm_hwm_mb()))
    except Exception as e:                              # report and stop: the runner flies without the VLM
        res_q.put(dict(kind='error', where='load', err=repr(e)))
        return
    frames_dir = os.path.join(cfg['out_dir'], 'frames') if cfg.get('save_frames') else None
    while True:
        try:
            r = req_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if r is None:
            break
        if r.get('kind') == 'outdir':                   # new mission
            frames_dir = os.path.join(r['out_dir'], 'frames') if cfg.get('save_frames') else None
            if frames_dir:
                os.makedirs(frames_dir, exist_ok=True)
            continue
        dropped = 0
        while True:                                     # keep only the newest waiting request
            try:
                nxt = req_q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                req_q.put(None); break
            if nxt.get('kind') == 'outdir':
                req_q.put(nxt); break
            dropped += 1; r = nxt
        t_cap0 = time.time()
        try:
            rgb, seg, info = src.grab(r['t'])
            t_cap1 = time.time()
            L, extra = model(rgb, info)
            t_inf = time.time()
            P, C = bel(L)
            w_hat = midpoint_weight(C)
            if seg is not None and resolver is not None:
                bld, unk = resolver.counts(seg, r.get('pose_ne') or (0.0, 0.0))
            else:
                bld, unk = (info or {}).get('bld'), None
            npx = rgb.shape[0] * rgb.shape[1]
            seen = [b for b, n in (bld or []) if n >= SEEN_FRAC * npx]
            name = f"{int(r['t']):04d}_uav0"
            if frames_dir:
                from PIL import Image
                Image.fromarray(rgb).save(os.path.join(frames_dir, name + '_rgb.jpg'), quality=92)
            res_q.put(dict(kind='frame', t=r['t'], wall_req=r['wall'], wall_cap0=t_cap0, wall_cap1=t_cap1, wall_done=time.time(),
                           capture_ms=round(1e3 * (t_cap1 - t_cap0), 1), vlm_ms=round(1e3 * (t_inf - t_cap1), 1),
                           logits=np.round(L, 4).tolist(), P=np.round(P, 5).tolist(), top=int(P.argmax()), set=C,
                           w_hat=w_hat, seen=seen, seg_unknown_px=unk, dropped_before=dropped, pose_ne=r.get('pose_ne'),
                           file=(info or {}).get('file', name), vm_hwm_mb=vm_hwm_mb(), **extra))
        except Exception as e:
            res_q.put(dict(kind='error', where='frame', t=r['t'], err=repr(e)))
    res_q.put(dict(kind='bye', vm_hwm_mb=vm_hwm_mb()))
