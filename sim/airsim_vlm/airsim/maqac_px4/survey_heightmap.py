#!/usr/bin/env python3
"""Survey the CityEnviron mission area with the external downward camera 'mapcam' and build the height map.

1. Orientation check: with the simulation paused, spawn tall markers (BicycleMan x3, 5 m) at known world positions,
   find each in the depth difference (depth drops by > 3 m; cars and pedestrians are lower) and fit the image-to-world
   orientation (heightmap.fit_orientation). The fitted error must be below 2 m.
2. Tiles: DepthPlanar images from ALT m above PlayerStart on a grid covering the mission area + margin; every pixel is
   projected to a 3-D point and rasterised to a 1 m height map.
3. Report: obstacle heights in the area against the flight altitude, map extent seen, PNG of the map.

    python3 survey_heightmap.py --ip 10.142.0.5 --out survey/   [--origin-ne 0 0] [--alt 400]
"""
import argparse
import json
import os
import time

import numpy as np
import airsim

from heightmap import HeightMap, AreaFrame, project, fit_orientation

MARKERS = [(40.0, 6.0), (36.0, -8.0), (2.0, 2.0)]    # world N, E (sidewalk spots of GROUND_CHECK pass 4; plaza)
CHECK_CAM = (20.0, 0.0, -120.0)
FOV = 90.0


def set_cam(c, ned):
    c.simSetCameraPose('mapcam', airsim.Pose(airsim.Vector3r(*ned), airsim.to_quaternion(-np.pi / 2, 0, 0)),
                       external=True)
    time.sleep(0.4)


def depth(c):
    r = c.simGetImages([airsim.ImageRequest('mapcam', airsim.ImageType.DepthPlanar, True, False)], external=True)[0]
    return np.array(r.image_data_float, np.float32).reshape(r.height, r.width)


def scene(c):
    r = c.simGetImages([airsim.ImageRequest('mapcam', airsim.ImageType.Scene, False, False)], external=True)[0]
    return np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1]


def orientation_check(c, out):
    from scipy import ndimage
    set_cam(c, CHECK_CAM)
    base = depth(c)
    g0 = CHECK_CAM[2] + float(np.median(base[base.shape[0] // 2 - 20:base.shape[0] // 2 + 20,
                                             base.shape[1] // 2 - 20:base.shape[1] // 2 + 20]))
    blobs, rows = [], []
    for k, (N, E) in enumerate(MARKERS):
        nm = c.simSpawnObject(f'mark_{k}', 'BicycleMan', airsim.Pose(airsim.Vector3r(N, E, g0), airsim.to_quaternion(0, 0, 0)),
                              airsim.Vector3r(3, 3, 3), False)
        time.sleep(0.5)
        d = depth(c)
        c.simDestroyObject(nm)
        time.sleep(0.3)
        lab, nl = ndimage.label((base - d) > 3.0)
        if nl == 0:
            rows.append(dict(marker=[N, E], found=False))
            continue
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, nl + 1))
        L = int(np.argmax(sz)) + 1
        v, u = ndimage.center_of_mass(lab == L)
        dd = float(np.median(d[lab == L]))
        blobs.append((v, u, dd))
        rows.append(dict(marker=[N, E], found=True, px=[float(v), float(u)], depth=dd, n_px=int(sz[L - 1])))
    found = [r['marker'] for r in rows if r['found']]
    if len(found) < 2:
        raise SystemExit(f'orientation check failed: markers found {len(found)}/3 ({rows})')
    best, errs = fit_orientation(blobs, CHECK_CAM, FOV, base.shape, found)
    rep = dict(markers=rows, ground_guess_D=g0, best=list(best), err_best_m=errs[best],
               errs={str(k): v for k, v in errs.items()})
    json.dump(rep, open(os.path.join(out, 'orientation_check.json'), 'w'), indent=1)
    if errs[best] > 2.0:
        raise SystemExit(f'orientation check: best error {errs[best]:.2f} m > 2 m; see orientation_check.json')
    return best, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default='10.142.0.5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument('--L', type=float, default=1000.0)
    ap.add_argument('--margin', type=float, default=60.0)
    ap.add_argument('--alt', type=float, default=400.0, help='camera height above PlayerStart for the tiles')
    ap.add_argument('--flight-alt', type=float, default=100.0)
    ap.add_argument('--skip-check', action='store_true', help='use the default orientation (only for debugging)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = airsim.MultirotorClient(ip=a.ip)
    c.confirmConnection()
    c.simPause(True)
    t0 = time.time()
    try:
        orient, chk = ((False, -1, 1), None) if a.skip_check else orientation_check(c, a.out)
        fr = AreaFrame(a.L, a.origin_ne)
        n_lo, n_hi, e_lo, e_hi = fr.window(a.margin)
        hm = HeightMap(n_lo, e_lo, n_hi - n_lo, e_hi - e_lo, res=1.0)
        hm.orient = orient
        foot = 2 * a.alt * np.tan(np.radians(FOV / 2))          # footprint at PlayerStart ground level
        step = 0.7 * foot
        ns = np.arange(n_lo + step / 2, n_hi + step / 2, step)
        es = np.arange(e_lo + step / 2, e_hi + step / 2, step)
        thumbs = []
        for N in ns:
            for E in es:
                cam = (float(N), float(E), -a.alt)
                set_cam(c, cam)
                hm.add_points(project(depth(c), cam, FOV, orient, max_depth=a.alt + 300))
                thumbs.append(dict(cam=cam, img=scene(c)[::8, ::8]))
        hm.finish()
        hm.meta = dict(origin_ne=list(a.origin_ne), L=a.L, alt=a.alt, tiles=len(ns) * len(es), orient=list(orient))
        hm.save(os.path.join(a.out, 'heightmap.npz'))
        st = hm.stats(*fr.window(0.0), a.flight_alt)
        rep = dict(meta=hm.meta, area_stats=st, orientation_check=chk, wall_s=round(time.time() - t0, 1))
        json.dump(rep, open(os.path.join(a.out, 'survey_report.json'), 'w'), indent=1)
        np.savez_compressed(os.path.join(a.out, 'thumbs.npz'), cams=np.array([t['cam'] for t in thumbs]),
                            imgs=np.stack([t['img'] for t in thumbs]))
        print(json.dumps(rep, indent=1))
    finally:
        c.simPause(False)


if __name__ == '__main__':
    main()
