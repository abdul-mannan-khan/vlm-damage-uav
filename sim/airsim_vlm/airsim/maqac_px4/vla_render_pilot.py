#!/usr/bin/env python3
"""VLA camera pilot: 20 nadir frames in the CityEnviron mission core, without PX4.

Five open-ground sites in the core x two heights (100 m, 150 m above local ground) x two cameras:
  vla      proposed camera, 1536 x 1536 px, 50 deg FOV (settings: make_settings_city.py --render)
  vla_old  current on-board camera, 640 x 480 px, 90 deg FOV
At each site ~20 users (BicycleMan) are spawned on open ground within 60 m, classes drawn 25/30/45 % (emergency /
real-time / best effort). Emergency users stand around an incident point marked by a context asset (a police
vehicle or similar, if CityEnviron offers one to simSpawnObject) so urgency has a visible cue.
Each frame stores RGB + segmentation PNGs and a label file: every user's class, world position, projected pixel and
measured size in pixels (segmentation blob around the projection). summary.json gives person size per camera/height:
the number that decides whether a VLA can see people or needs the detector front end.

    python3 vla_render_pilot.py --ip 10.142.0.5 --heightmap /work/maqac/survey_wide/heightmap.npz --out /work/maqac/pilot
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
from PIL import Image

from heightmap import HeightMap, AreaFrame

CLS_P = [0.25, 0.30, 0.45]
CLS_NAME = ['emergency', 'realtime', 'besteffort']
SEG_USER = 100           # + class
SEG_CUE = 150
CUE_RE = r'(police|ambul|fire|rescue|cone|barrier|barricade|tent|truck)'


def img(resp, ch=3):
    a = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    return a.reshape(resp.height, resp.width, -1)[:, :, :ch].copy()


def to_pixel(N, E, cam, d, W, H, fov, orient):
    """Inverse of heightmap.project for a nadir camera with yaw 0: world (N, E) at depth d -> pixel (u, v)."""
    f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
    swap, sN, sE = orient
    p, q = (N - cam[0]) / sN, (E - cam[1]) / sE          # = a', b'
    a, b = (q, p) if swap else (p, q)
    return W / 2.0 + b * f / d - 0.5, H / 2.0 + a * f / d - 0.5


def blob_px(seg, u, v, colour, r=12):
    """Pixels of `colour` in a (2r+1)^2 window around (u, v), and the bounding-box size (width, height)."""
    H, W = seg.shape[:2]
    i0, i1 = max(0, int(v) - r), min(H, int(v) + r + 1)
    j0, j1 = max(0, int(u) - r), min(W, int(u) + r + 1)
    if i0 >= i1 or j0 >= j1:
        return 0, [0, 0]
    m = np.all(seg[i0:i1, j0:j1] == colour, -1)
    if not m.any():
        return 0, [0, 0]
    ii, jj = np.where(m)
    return int(m.sum()), [int(jj.max() - jj.min() + 1), int(ii.max() - ii.min() + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', required=True)
    ap.add_argument('--heightmap', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', type=float, default=600.0)
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(40.0, -10.0))
    ap.add_argument('--sites', type=int, default=5)
    ap.add_argument('--heights', type=float, nargs='+', default=[100.0, 150.0])
    ap.add_argument('--users', type=int, default=20)
    ap.add_argument('--seed', type=int, default=31)
    ap.add_argument('--vla-fov', type=float, default=50.0)
    ap.add_argument('--old-fov', type=float, default=90.0)
    a = ap.parse_args()
    import airsim as A
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    c = A.MultirotorClient(ip=a.ip, timeout_value=120)
    c.confirmConnection()
    veh = c.listVehicles()
    print('vehicles', veh, flush=True)

    # asset survey (what CityEnviron lets us spawn)
    assets = c.simListAssets()
    cues = sorted(x for x in assets if re.search(CUE_RE, x, re.I))
    humans = sorted(x for x in assets if re.search(r'(human|person|people|man|woman|ped|char|body|mannequin|citizen)', x, re.I))
    json.dump({'n_assets': len(assets), 'cue_like': cues, 'human_like': humans, 'all': sorted(assets)},
              open(os.path.join(a.out, 'assets.json'), 'w'), indent=1)
    cue_asset = next((x for x in cues if re.search('police', x, re.I)), cues[0] if cues else None)
    print(f'assets {len(assets)}; cue-like {len(cues)} -> using {cue_asset}; human-like {len(humans)}', flush=True)

    # segmentation: everything 0 except our users (100 + class) and the cue (150), so person blobs can be measured
    c.simSetSegmentationObjectID(r'[\w]*', 0, True)
    hm = HeightMap.load(a.heightmap)
    fr = AreaFrame(a.L, a.origin_ne)
    n_lo, n_hi, e_lo, e_hi = fr.window(-60.0)
    cams = {'vla': a.vla_fov, 'vla_old': a.old_fov}
    for nm, fv in cams.items():           # CaptureSettings FOV is not reported for external cameras: set it explicitly
        c.simSetCameraFov(nm, fv, external=True)
        cams[nm] = float(c.simGetCameraInfo(nm, external=True).fov)
    print('camera fov', cams, flush=True)

    # sites: open ground, at least 120 m apart
    sites, tries = [], 0
    while len(sites) < a.sites and tries < 5000:
        tries += 1
        N, E = rng.uniform(n_lo, n_hi), rng.uniform(e_lo, e_hi)
        i, j = hm._clip(*hm.cell(N, E))
        if hm.obst[i, j] or not hm.valid[i, j]:
            continue
        if all(math.hypot(N - s[0], E - s[1]) > 120 for s in sites):
            sites.append((float(N), float(E)))

    sizes, bbs = {}, {}   # (cam, height) -> person pixel areas / longest bbox side
    frames = []
    k = 0
    for si, (Ns, Es) in enumerate(sites):
        # users: emergency around an incident point, others spread within 60 m
        cls = rng.choice(3, size=a.users, p=CLS_P)
        inc = np.array([Ns, Es]) + rng.uniform(-25, 25, 2)
        pos = []
        for cl in cls:
            if cl == 0:
                pos.append(inc + rng.normal(0, 5, 2))
            else:
                r, th = 60 * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
                pos.append([Ns + r * math.cos(th), Es + r * math.sin(th)])
        pos = np.array(pos)
        N2, E2, g, moved = hm.snap(pos[:, 0], pos[:, 1])
        names = []
        for u in range(a.users):
            nm = c.simSpawnObject(f'pu_{si}_{u}', 'BicycleMan',
                                  A.Pose(A.Vector3r(float(N2[u]), float(E2[u]), float(-g[u])),
                                         A.to_quaternion(0, 0, float(rng.uniform(0, 2 * math.pi)))),
                                  A.Vector3r(1, 1, 1), False)
            c.simSetSegmentationObjectID(nm, SEG_USER + int(cls[u]), False)
            names.append(nm)
        cue_nm, cue_ne = None, None
        if cue_asset:
            cn, ce, cg, _ = hm.snap(np.array([inc[0] + 8]), np.array([inc[1]]))
            try:
                cue_nm = c.simSpawnObject(f'cue_{si}', cue_asset,
                                          A.Pose(A.Vector3r(float(cn[0]), float(ce[0]), float(-cg[0])), A.to_quaternion(0, 0, 0.6)),
                                          A.Vector3r(1, 1, 1), False)
                c.simSetSegmentationObjectID(cue_nm, SEG_CUE, False)
                cue_ne = [float(cn[0]), float(ce[0])]
            except Exception as ex:
                print('cue spawn failed', cue_asset, ex, flush=True)
                cue_asset = None
        time.sleep(1.0)
        g0 = float(np.median(hm.ground[hm._clip(*hm.cell(Ns, Es))]))
        for h in a.heights:
            for cam, fov in cams.items():
                pose = A.Pose(A.Vector3r(Ns, Es, -(g0 + h)), A.to_quaternion(-math.pi / 2, 0, 0))
                c.simSetCameraPose(cam, pose, external=True)
                time.sleep(0.5)
                rs = c.simGetImages([A.ImageRequest(cam, A.ImageType.Scene, False, False),
                                     A.ImageRequest(cam, A.ImageType.Segmentation, False, False)], external=True)
                rgb, seg = img(rs[0]), img(rs[1])
                H, W = rgb.shape[:2]
                tag = f'{k:02d}_site{si}_{cam}_{int(h)}m'
                Image.fromarray(rgb).save(os.path.join(a.out, tag + '_rgb.png'))
                Image.fromarray(seg).save(os.path.join(a.out, tag + '_seg.png'))
                from scipy import ndimage
                bg = np.array(np.unravel_index(np.argmax(np.bincount(
                    (seg.reshape(-1, 3).astype(np.int64) @ np.array([65536, 256, 1])))), (256, 256, 256)))
                fg = ~np.all(seg == bg, -1)
                lab_im, nb = ndimage.label(fg)
                comp = ndimage.find_objects(lab_im)
                cen = np.array(ndimage.center_of_mass(fg, lab_im, range(1, nb + 1))) if nb else np.zeros((0, 2))
                area = ndimage.sum(fg, lab_im, range(1, nb + 1)) if nb else np.zeros(0)
                users = []
                for u in range(a.users):
                    d = g0 + h - g[u] - 0.9            # camera to person's mid-height
                    pu, pv = to_pixel(N2[u], E2[u], (Ns, Es), d, W, H, fov, hm.orient)
                    inside = 0 <= pu < W and 0 <= pv < H
                    px, bb, err = 0, [0, 0], None
                    if inside and nb:
                        dist = np.hypot(cen[:, 1] - pu, cen[:, 0] - pv)
                        m = int(np.argmin(dist)); err = float(dist[m])
                        if err < max(8.0, 0.02 * W):
                            sl = comp[m]
                            px, bb = int(area[m]), [sl[1].stop - sl[1].start, sl[0].stop - sl[0].start]
                    users.append(dict(cls=CLS_NAME[cls[u]], N=round(float(N2[u]), 2), E=round(float(E2[u]), 2),
                                      u=round(pu, 1), v=round(pv, 1), in_view=bool(inside), px=px, bbox_px=bb,
                                      match_err_px=None if err is None else round(err, 1)))
                    if inside:
                        sizes.setdefault(f'{cam}_{int(h)}m', []).append(px)
                        bbs.setdefault(f'{cam}_{int(h)}m', []).append(max(bb))
                gsd = 2 * (h + 0.0) * math.tan(math.radians(fov) / 2) / W
                lab = dict(frame=tag, cam=cam, fov=fov, W=W, H=H, height_m=h, gsd_m_per_px=round(gsd, 4),
                           cam_world=[Ns, Es, -(g0 + h)], cue_asset=cue_asset, cue_world=cue_ne, users=users,
                           counts={n: int((cls == i).sum()) for i, n in enumerate(CLS_NAME)})
                json.dump(lab, open(os.path.join(a.out, tag + '.json'), 'w'), indent=1)
                frames.append(tag)
                print(f'{tag}: gsd {gsd:.3f} m/px, users in view {sum(x["in_view"] for x in users)}, '
                      f'matched {sum(x["px"] > 0 for x in users)}, median area {np.median([x["px"] for x in users if x["px"]] or [0]):.0f} px, '
                      f'median bbox {np.median([max(x["bbox_px"]) for x in users if x["px"]] or [0]):.0f} px', flush=True)
                k += 1
        # park this site's objects far outside the city (destroying spawned objects hung AirSim once)
        for q, nm in enumerate(names + ([cue_nm] if cue_nm else [])):
            c.simSetObjectPose(nm, A.Pose(A.Vector3r(5000.0 + 10 * si, 5000.0 + 3 * q, 0.0), A.to_quaternion(0, 0, 0)), True)
    summ = {key: dict(n=len(v), median_area_px=float(np.median(v)), p10_area=float(np.percentile(v, 10)),
                      p90_area=float(np.percentile(v, 90)), median_bbox_side_px=float(np.median(bbs[key])),
                      unmatched_frac=float(np.mean(np.array(v) == 0)))
            for key, v in sizes.items()}
    json.dump(dict(frames=frames, sites=sites, person_px=summ, cue_asset=cue_asset), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
