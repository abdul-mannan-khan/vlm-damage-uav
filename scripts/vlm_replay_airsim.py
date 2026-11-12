"""Replay video of one logged step-6 mission (PX4 SITL in AirSim CityEnviron, onboard VLM damage reading), 10 Nov 2026.
Adapted from a copy of maqac-vehcom/sim/hil/replay_hil_airsim.py (that file is not changed).

Logged data used (results/vlm_damage/city_step6/runs/<run>/):
  tracks.json        PX4 position estimate of each UAV every 1 s (world N, E, D)
  frames/*_rgb.jpg   downward camera frame of each UAV every 5 s (640 x 480), as fed to the VLM
  frames/vlm.jsonl   the VLM reading of each frame (calibrated class probabilities, top class, conformal label set)
  frames/labels.jsonl buildings in each frame (AirSim segmentation)
The script places the UAVs at the logged positions (interpolated, smoothed over 1 s), films UAV 0 with a chase camera
and composes each frame with the map, the logged downward frame with its VLM reading, altitude and the buildings-seen
count. It is a replay of logged simulator data, not a live recording, and every frame says so.

On a Vast box with abdulmannan617/airsim-city-wine, after
  city-settings --mode simpleflight --n 2 --overview > s.json (PhysicsEngineName = ExternalPhysicsEngine); city-start s.json
  python3 vlm_replay_airsim.py render  --run RUN --hm heightmap.npz --out OUT
  python3 vlm_replay_airsim.py compose --run RUN --hm heightmap.npz --out OUT    # -> OUT/vlm_replay.mp4
Frames: 25/SPEED per logged second; video 25 fps = SPEED x real time (--speed, default 2 since 11 Nov 2026; was 5).
Chase camera height: UAV 0's height smoothed over CAM_SMOOTH_S and kept within CAM_MAX_OFF m of it, camera pitched at
UAV 0, so the tower climbs show as the UAV rising in the frame instead of the whole city lurching (11 Nov 2026).
"""
import argparse, json, os, subprocess
import numpy as np

FPS_SAMPLE, FPS_VIDEO = 5, 25        # FPS_SAMPLE is reset from --speed in __main__
CAM_SMOOTH_S, CAM_MAX_OFF, CAM_BACK, CAM_HIGH = 15.0, 15.0, 30.0, 10.0
BG, FG, MUTED = '#0f1420', '#e8ecf3', '#9aa4b5'
UAV_COL = ['#7ee081', '#c792ea']
CLS = ['A  no building', 'B  intact', 'C  minor', 'D  major', 'E  destroyed']


def load(run):
    d = json.load(open(os.path.join(run, 'tracks.json')))
    P = np.array(d['px4'], float)                          # rounds x UAV x (N, E, D)
    P[..., 2] = -P[..., 2]                                 # up
    t = np.arange(len(P), dtype=float)
    fr = os.path.join(run, 'frames')
    V = {r['file']: r for r in map(json.loads, open(os.path.join(fr, 'vlm.jsonl')))}
    Lb = [json.loads(l) for l in open(os.path.join(fr, 'labels.jsonl'))]
    return t, P, V, Lb


def interp(t, P, T, smooth_s=1.0):
    out = np.empty((len(T),) + P.shape[1:])
    w = max(1, int(round(smooth_s * FPS_SAMPLE))); k = np.ones(w) / w
    for i in range(P.shape[1]):
        for d in range(3):
            x = np.interp(T, t, P[:, i, d])
            out[:, i, d] = np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode='edge'), k, mode='valid') if w > 1 else x
    return out


def headings(Q, max_rate_deg_s=8.0):
    """Camera (and UAV 0) heading: direction of UAV 0's horizontal motion over the last ~5 s, kept while it hovers,
    then rate-limited to max_rate_deg_s of mission time and smoothed, so the chase view turns slowly (an unlimited
    heading swung the camera within a few frames and the replay appeared to jump, 10 Nov 2026)."""
    lag = int(round(5 * FPS_SAMPLE))
    v = Q[:, 0, :2] - Q[np.maximum(np.arange(len(Q)) - lag, 0), 0, :2]
    yaw = np.zeros(len(Q)); last = float(np.arctan2(*(Q[min(len(Q) - 1, lag), 0, :2] - Q[0, 0, :2])[::-1]))
    for i, (dn, de) in enumerate(v):
        if np.hypot(dn, de) > 5.0:                      # moved > 5 m in 5 s: follow the direction of travel
            last = np.arctan2(de, dn)
        yaw[i] = last
    step = np.radians(max_rate_deg_s) / FPS_SAMPLE
    out = np.empty_like(yaw); out[0] = yaw[0]
    for i in range(1, len(yaw)):
        d = (yaw[i] - out[i - 1] + np.pi) % (2 * np.pi) - np.pi
        out[i] = out[i - 1] + np.clip(d, -step, step)
    w3 = int(round(3 * FPS_SAMPLE)); k = np.ones(w3) / w3
    return np.convolve(np.pad(out, (len(k) // 2, len(k) - 1 - len(k) // 2), mode='edge'), k, mode='valid')


def cam_heights(Q):
    """Camera height above UAV 0's track: moving average over CAM_SMOOTH_S, clipped to +-CAM_MAX_OFF of the UAV."""
    z = Q[:, 0, 2]; w = max(1, int(round(CAM_SMOOTH_S * FPS_SAMPLE))); k = np.ones(w) / w
    zs = np.convolve(np.pad(z, (w // 2, w - 1 - w // 2), mode='edge'), k, mode='valid')
    return np.clip(zs, z - CAM_MAX_OFF, z + CAM_MAX_OFF) + CAM_HIGH


def render(a):
    import airsim
    from PIL import Image
    t, P, _, _ = load(a.run)
    T = np.arange(0, t[-1] + 1e-9, 1.0 / FPS_SAMPLE)
    Q = interp(t, P, T); yaw = headings(Q); cz = cam_heights(Q)
    c = airsim.MultirotorClient('127.0.0.1', timeout_value=60); c.confirmConnection()
    names = [f'UAV{i}' for i in range(P.shape[1])]
    od = os.path.join(a.out, 'airsim'); os.makedirs(od, exist_ok=True)
    start = len([f for f in os.listdir(od) if f.endswith('.jpg')])          # resumable
    for k in range(start, len(T)):
        for i, nm in enumerate(names):
            n, e, up = Q[k, i]
            c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(float(n), float(e), float(-up)),
                                            airsim.to_quaternion(0, 0, float(yaw[k] if i == 0 else 0.0))), True, vehicle_name=nm)
        n, e, up = Q[k, 0]; back = CAM_BACK
        pitch = -float(np.arctan2(cz[k] - up, back))             # look at UAV 0
        c.simSetCameraPose('overview', airsim.Pose(
            airsim.Vector3r(float(n - back * np.cos(yaw[k])), float(e - back * np.sin(yaw[k])), float(-cz[k])),
            airsim.to_quaternion(pitch, 0, float(yaw[k]))), external=True)
        r = c.simGetImages([airsim.ImageRequest('overview', airsim.ImageType.Scene, False, False)], external=True)[0]
        img = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1]
        if k % 100 == 0:
            q = c.simGetVehiclePose(vehicle_name='UAV0').position
            print(f'frame {k}/{len(T)} set {Q[k, 0].round(1).tolist()} readback {q.x_val:.1f} {q.y_val:.1f} {-q.z_val:.1f}', flush=True)
            if np.hypot(q.x_val - Q[k, 0, 0], q.y_val - Q[k, 0, 1]) > 2.0:
                raise SystemExit('pose not applied (AirSim paused or physics on?)')
        Image.fromarray(img).save(os.path.join(od, f'{k:05d}.jpg'), quality=90)
    print('render done', flush=True)


def compose(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    t, P, V, Lb = load(a.run)
    T = np.arange(0, t[-1] + 1e-9, 1.0 / FPS_SAMPLE)
    Q = interp(t, P, T)
    z = np.load(a.hm); n0, e0, res, H = float(z['n0']), float(z['e0']), float(z['res']), z['h']
    ctr = P[:, :, :2].reshape(-1, 2).mean(0); half = 330
    lo_n, lo_e = ctr - half
    i0, j0 = int((lo_n - n0) / res), int((lo_e - e0) / res); i1, j1 = i0 + int(2 * half / res), j0 + int(2 * half / res)
    hmap = np.clip(H[max(i0, 0):i1, max(j0, 0):j1], 0, 200)
    ext = [lo_e, lo_e + 2 * half, lo_n, lo_n + 2 * half]
    # buildings seen so far by either UAV (AirSim segmentation of the logged frames), keyed by frame time
    seen_t, seen, cum = [], set(), []
    for r in sorted(Lb, key=lambda r: (r['t'], r['uav'])):
        seen |= {b for b, _ in r['bld']}; seen_t.append(r['t']); cum.append(len(seen))
    seen_t, cum = np.array(seen_t), np.array(cum)
    meta = json.load(open(os.path.join(a.run, 'result.json')))
    method = {'maqac_seq': 'MAQAC-S', 'maqac': 'MAQAC', 'rag': 'RAG'}.get(meta.get('method'), meta.get('method'))
    fd = os.path.join(a.out, 'frames'); os.makedirs(fd, exist_ok=True)
    f = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    axi = f.add_axes([0.012, 0.25, 0.58, 0.58]); axm = f.add_axes([0.615, 0.50, 0.175, 0.31])
    axc = f.add_axes([0.805, 0.50, 0.185, 0.31]); axp = f.add_axes([0.87, 0.29, 0.12, 0.16])
    axz = f.add_axes([0.645, 0.29, 0.14, 0.16]); axs = f.add_axes([0.645, 0.075, 0.345, 0.13])
    for k in ([a.only] if a.only is not None else range(len(T))):
        src = os.path.join(a.out, 'airsim', f'{k:05d}.jpg')
        if not os.path.exists(src):
            break
        for ax in (axi, axm, axc, axp, axz, axs):
            ax.cla()
        axi.imshow(Image.open(src)); axi.axis('off')
        axm.imshow(hmap, origin='lower', extent=ext, cmap='bone', vmin=-20, vmax=220); axm.set_xticks([]); axm.set_yticks([])
        for i in range(P.shape[1]):
            tr = Q[max(0, k - 300):k + 1, i]
            axm.plot(tr[:, 1], tr[:, 0], color=UAV_COL[i], lw=1.8 if i == 0 else 1.0)
            axm.plot(Q[k, i, 1], Q[k, i, 0], marker='^', ms=10 if i == 0 else 7, color=UAV_COL[i], mec='black', mew=0.6)
        tf = int(T[k] // 5 * 5); key = f'{tf:04d}_uav0'                  # latest logged downward frame of UAV 0
        axc.axis('off'); axp.set_facecolor(BG)
        if os.path.exists(os.path.join(a.run, 'frames', key + '_rgb.jpg')):
            axc.imshow(Image.open(os.path.join(a.run, 'frames', key + '_rgb.jpg')))
        v = V.get(key); vt = ''
        if v:
            cols = ['#ff6b6b' if j in (3, 4) else ('#ffd166' if j in v['set'] else '#3a4a66') for j in range(5)]
            axp.barh(range(5), v['P'], color=cols, height=0.7); axp.set_xlim(0, 1); axp.invert_yaxis()
            axp.set_yticks(range(5)); axp.set_yticklabels(CLS, color=FG, fontsize=9)
            vt = (f'VLM: {CLS[v["top"]][3:]}, set {{{",".join("ABCDE"[j] for j in v["set"])}}}, {v["compute_s"]:.1f} s on cloud GPU')
        axp.tick_params(colors=MUTED, labelsize=8); axp.set_xlabel('probability (yellow: in 90 % set)', color=MUTED, fontsize=9)
        for ax_ in (axz, axs, axp):
            ax_.set_facecolor(BG); ax_.tick_params(colors=MUTED, labelsize=9)
            for s_ in ax_.spines.values():
                s_.set_color('#2a3448')
        for i in range(P.shape[1]):
            axz.plot(T[:k + 1], Q[:k + 1, i, 2], color=UAV_COL[i], lw=1.6)
        axz.set_xlim(0, T[-1]); axz.set_ylim(0, float(np.ceil(P[..., 2].max() / 50) * 50)); axz.set_ylabel('altitude [m]', color=MUTED, fontsize=10)
        m = seen_t <= T[k]
        axs.step(seen_t[m], cum[m], where='post', color='#7ee081', lw=1.8)
        axs.set_xlim(0, T[-1]); axs.set_ylim(0, max(cum) * 1.1); axs.set_ylabel('buildings seen', color=MUTED, fontsize=10)
        axs.set_xlabel('mission time [s]', color=MUTED, fontsize=10)
        f.texts.clear()
        f.text(0.805, 0.83, f'UAV 0 downward camera, t = {tf} s', color=MUTED, fontsize=10.5, va='top')
        if v:
            f.text(0.805, 0.485, vt, color=FG, fontsize=11, va='top')
        f.text(0.012, 0.965, f'REPLAY  |  Live VLM damage reading, AirSim CityEnviron ({method}, two UAVs)    t = {T[k]:5.1f} s',
               color=FG, fontsize=20, va='top', weight='bold')
        f.text(0.012, 0.915, 'UAVs flown by PX4 SITL (software autopilot) in AirSim; each downward frame read during the flight by the fine-tuned '
               'Qwen2.5-VL-3B (4 seeds, calibrated, conformal set) on the cloud GPU,\nnot on the UAV. Intact city: no damage expected.', color=MUTED, fontsize=12, va='top')
        f.text(0.012, 0.225, 'Chase camera behind UAV 0 (green). Replay of logged simulator data (step 6, 9 Nov 2026):\n'
               f'positions logged every 1 s, interpolated and smoothed over 1 s; video at {a.speed:g}x real time.\n'
               'UAVs climb over towers taller than the 100 m flight altitude, then return to it.', color=MUTED, fontsize=12, va='top')
        f.text(0.615, 0.83, 'Map: building heights; tracks of the last 60 s', color=MUTED, fontsize=10.5, va='top')
        f.text(0.012, 0.10, 'Buildings seen: distinct buildings in either UAV\'s logged\nframes (AirSim segmentation), cumulative.',
               color=MUTED, fontsize=10)
        f.savefig(os.path.join(fd, f'{k:05d}.png'), facecolor=BG)
        if k % 100 == 0:
            print(f'composed {k}/{len(T)}', flush=True)
    plt.close(f)
    if a.only is not None:
        return
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS_VIDEO), '-i', os.path.join(fd, '%05d.png'),
                    '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', os.path.join(a.out, 'vlm_replay.mp4')], check=True)
    print('video done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('render', 'compose'))
    ap.add_argument('--run', required=True); ap.add_argument('--hm', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--only', type=int, default=None, help='compose one frame only (layout preview)')
    ap.add_argument('--speed', type=float, default=2.0, help='video speed vs mission time (25 fps video)')
    a = ap.parse_args()
    FPS_SAMPLE = FPS_VIDEO / a.speed
    render(a) if a.stage == 'render' else compose(a)
