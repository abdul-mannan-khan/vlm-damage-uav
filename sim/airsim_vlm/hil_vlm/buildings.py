#!/usr/bin/env python3
"""Building inventory and per-frame building visibility for the VLM-paper city flights (8-9 Nov 2026).

AirSim segmentation IDs are 0..255, fewer than the city's buildings, so IDs are REUSED for buildings at least SEP metres
apart; a camera footprint (radius ~58 m at 100 m altitude, 60 deg FOV) is far smaller, so an ID seen in a frame is resolved to the nearest
building with that ID to the UAV's position. Everything that is not an inventoried building gets ID 0.

    # on the box, CityEnviron running (pilot):
    python3 buildings.py dump  --ip 127.0.0.1 --out /work/vlm/objects_all.json          # every scene object + pose
    python3 buildings.py build --objects /work/vlm/objects_all.json --regex 'Building|BP_House|...' \
        --origin-ne 40 -10 --L 600 --margin 150 --out /work/vlm/buildings.json            # inventory + IDs
"""
import argparse
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = 400.0          # m between buildings that share a segmentation ID
MAX_ID = 250         # IDs 1..250 for buildings; 0 = everything else


def palette(path=os.path.join(HERE, 'seg_rgbs.txt')):
    """AirSim's fixed ID -> RGB table (docs/seg_rgbs.txt). Returns a dict packed RGB int -> ID."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        k, rgb = line.split('\t', 1)
        r, g, b = json.loads(rgb)
        out[(r << 16) | (g << 8) | b] = int(k)
    return out


def dump(c, out):
    names = c.simListSceneObjects('.*')
    objs = []
    for nm in names:
        try:
            p = c.simGetObjectPose(nm).position
            if math.isfinite(p.x_val):
                objs.append(dict(name=nm, N=round(p.x_val, 2), E=round(p.y_val, 2), D=round(p.z_val, 2)))
        except Exception:
            pass
    json.dump(objs, open(out, 'w'), indent=0)
    return objs


def build(objs, regex, origin_ne, L, margin):
    """Keep objects whose name matches regex and whose position lies in the mission square (+ margin), then give IDs
    with reuse only beyond SEP metres (greedy, nearest-first order)."""
    rx = re.compile(regex)
    n0, e0 = origin_ne
    keep = [o for o in objs if rx.search(o['name'])
            and -L / 2 - margin <= o['N'] - n0 <= L / 2 + margin and -L / 2 - margin <= o['E'] - e0 <= L / 2 + margin]
    keep.sort(key=lambda o: (o['N'] - n0) ** 2 + (o['E'] - e0) ** 2)
    by_id = {k: [] for k in range(1, MAX_ID + 1)}
    for o in keep:
        for k in range(1, MAX_ID + 1):
            if all(math.hypot(o['N'] - q['N'], o['E'] - q['E']) >= SEP for q in by_id[k]):
                o['seg_id'] = k
                by_id[k].append(o)
                break
        else:
            raise SystemExit(f'more than {MAX_ID} IDs needed; raise SEP margin or narrow the regex')
    return keep


def apply_ids(c, blds):
    """Set every object to 0, then each inventoried building to its ID. Returns how many calls succeeded."""
    c.simSetSegmentationObjectID('.*', 0, True)
    ok = 0
    for b in blds:
        ok += bool(c.simSetSegmentationObjectID(re.escape(b['name']), b['seg_id'], False))
    return ok


class Resolver:
    """Per-frame building pixel counts from a segmentation image (H x W x 3, RGB)."""

    def __init__(self, blds, pal=None):
        self.pal = pal or palette()
        self.by_id = {}
        for b in blds:
            self.by_id.setdefault(b['seg_id'], []).append(b)

    def counts(self, seg_rgb, ne_uav):
        """Tries both channel orders (AirSim's raw byte order differs between versions) and keeps the one whose colours
        match the palette better."""
        r1 = self._counts(seg_rgb, ne_uav)
        if r1[1] > 0.05 * seg_rgb.shape[0] * seg_rgb.shape[1]:
            r2 = self._counts(seg_rgb[:, :, ::-1], ne_uav)
            if r2[1] < r1[1]:
                return r2
        return r1

    def _counts(self, seg_rgb, ne_uav):
        a = seg_rgb.astype(np.int64)
        packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
        vals, cnt = np.unique(packed, return_counts=True)
        out, unknown = [], 0
        for v, n in zip(vals.tolist(), cnt.tolist()):
            k = self.pal.get(v)
            if k is None:
                unknown += n
                continue
            if k == 0 or k not in self.by_id:
                continue
            cand = self.by_id[k]
            b = min(cand, key=lambda q: math.hypot(q['N'] - ne_uav[0], q['E'] - ne_uav[1]))
            out.append([b['name'], int(n)])
        out.sort(key=lambda x: -x[1])
        return out, unknown


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('dump', 'build'))
    ap.add_argument('--ip', default='127.0.0.1')
    ap.add_argument('--objects')
    ap.add_argument('--regex', default='(?i)build')
    ap.add_argument('--origin-ne', type=float, nargs=2, default=(40.0, -10.0))
    ap.add_argument('--L', type=float, default=600.0)
    ap.add_argument('--margin', type=float, default=150.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.mode == 'dump':
        import airsim
        c = airsim.MultirotorClient(ip=a.ip, timeout_value=120)
        c.confirmConnection()
        o = dump(c, a.out)
        print(len(o), 'objects ->', a.out)
    else:
        b = build(json.load(open(a.objects)), a.regex, a.origin_ne, a.L, a.margin)
        json.dump(b, open(a.out, 'w'), indent=0)
        print(len(b), 'buildings,', len({x['seg_id'] for x in b}), 'IDs ->', a.out)
