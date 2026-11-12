#!/usr/bin/env python3
"""Bench photo for the TVT draft (v2, 11 Nov 2026): top view 20261110_162024.jpg of the 10 Nov bench (X500 with Pixhawk 6C
and GPS, props removed; Jetson Orin Nano Super; host laptop). The laptop screen (an idle terminal in the photo) is
replaced by a frame of this study's AirSim replay video (t = 200 s, results/vlm_damage/video/vlmdmg_airsim_replay.mp4),
warped onto the screen; the caption states that the screen content was inserted. Wall, sockets, paper and floor are
removed (kept: dark or coloured pixels plus the laptop by a traced outline). Labels sit in white strips above and below
the photo; the script checks that no label box overlaps another label or the equipment.
v1 (side view 155452, laptop showing another project's view) is kept in git history / scratch.
Outputs: paper/vlm_tvt/figures/fig_setup_cutout.png (RGBA), fig_setup.pdf, fig_setup_preview.png.
    python3 scripts/vlm_setup_fig.py"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'vlm_tvt' / 'figures'
SRC = 'data/photos/20261110_162024.jpg'
VID = ROOT / 'results' / 'vlm_damage' / 'video' / 'vlmdmg_airsim_replay.mp4'
K = 2                                                   # work at 4000 x 3000; coordinates below are in the 2000 x 1500 grid
X0, Y0, X1, Y1 = 20, 285, 1840, 1000
SCREEN = [(1229, 381), (1787, 378), (1702, 683), (1222, 684)]   # display area: TL, TR, BR, BL
LAPTOP = [(1229, 372), (1800, 368), (1713, 688), (1742, 700), (1809, 992), (1226, 997), (1207, 694), (1219, 689)]


def screen_frame(t=40.0):
    """Frame of the replay video (video time 40 s = mission time 200 s), letterboxed to the 16:10 screen."""
    png = OUT / '_replay_frame.png'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(VID), '-frames:v', '1', str(png)], check=True)
    f = Image.open(png).convert('RGB'); png.unlink()
    can = Image.new('RGB', (1920, 1200), f.getpixel((5, 5))); can.paste(f, (0, 60))
    return ImageEnhance.Brightness(can).enhance(0.93)


def perspective_coeffs(dst, src):
    """Coefficients mapping output (dst) points to input (src) points for Image.transform(PERSPECTIVE)."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def cutout():
    im = Image.open(SRC); im.draft('RGB', (4100, 3100))
    im = ImageOps.exif_transpose(im).convert('RGB').resize((2000 * K, 1500 * K), Image.LANCZOS)
    scr = screen_frame(); sw, sh = scr.size                       # insert the replay frame on the screen
    dst = [(x * K, y * K) for x, y in SCREEN]
    warped = scr.transform(im.size, Image.PERSPECTIVE, perspective_coeffs(dst, [(0, 0), (sw, 0), (sw, sh), (0, sh)]),
                           Image.BICUBIC)
    sm = Image.new('L', im.size, 0); ImageDraw.Draw(sm).polygon(dst, fill=255)
    im.paste(warped, (0, 0), sm.filter(ImageFilter.GaussianBlur(0.6 * K)))
    im = im.crop((X0 * K, Y0 * K, X1 * K, Y1 * K))
    A = np.asarray(im).astype(float); W, H = im.size
    lum = A.mean(2); sat = A.max(2) - A.min(2)
    P = lambda pts: [((x - X0) * K, (y - Y0) * K) for x, y in pts]
    fg = (lum < 75) | (sat > 50)                        # black frame, cables, blue arms, boards, coloured wires
    lap = Image.new('L', (W, H), 0); ImageDraw.Draw(lap).polygon(P(LAPTOP), fill=255); lap = np.asarray(lap) > 0
    fg &= ~lap                                          # laptop handled by its outline
    fg = ndi.binary_closing(fg, iterations=2 * K)
    holes = ndi.binary_fill_holes(fg) & ~fg             # fill only small holes, not paper areas enclosed by cables
    hl, hn = ndi.label(holes); hs = ndi.sum(holes, hl, range(1, hn + 1))
    fg |= np.isin(hl, [i + 1 for i, z in enumerate(hs) if z < 400 * K * K])
    drop = np.zeros_like(fg)
    drop[: (380 - Y0) * K, (1400 - X0) * K: (1500 - X0) * K] = True                    # laptop mains lead up the wall
    drop[(700 - Y0) * K:, (1810 - X0) * K:] = True                                     # dark floor, right
    band = np.zeros_like(fg); band[(755 - Y0) * K:(805 - Y0) * K, :(360 - X0) * K] = True
    thick = ndi.binary_opening(fg, structure=np.ones((7 * K, 7 * K)))
    drop |= band & ~thick                               # thin grey table edge left of the drone; the thick arm stays
    fg &= ~drop
    bl, bn = ndi.label(fg)                              # drop blobs touching the photo edge (wall, floor)
    edge = set(np.unique(np.concatenate([bl[0], bl[-1], bl[:, -1]]))) - {0}   # not the left edge: a motor is there
    fg &= ~np.isin(bl, list(edge))
    keep = Image.new('L', (W, H), 0); d = ImageDraw.Draw(keep)
    for cx, cy, r in [(268, 338, 34), (836, 312, 32), (73, 818, 38), (812, 798, 36)]:   # light motor caps / prop adapters
        d.ellipse([(cx - X0 - r) * K, (cy - Y0 - r) * K, (cx - X0 + r) * K, (cy - Y0 + r) * K], fill=255)
    for poly in ([(285, 323), (325, 330), (290, 342)], [(860, 314), (893, 326), (862, 333)],
                 [(50, 772), (75, 772), (70, 800)]):                                  # paper prop-direction tags
        d.polygon(P(poly), fill=255)
    fg |= (np.asarray(keep) > 0) & (lum < 236)
    mask = fg | lap
    lab, n = ndi.label(mask); sz = ndi.sum(mask, lab, range(1, n + 1))
    mask = np.isin(lab, [i + 1 for i, z in enumerate(sz) if z > 900 * K * K])
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8 * K))
    rgba = im.copy(); rgba.putalpha(alpha)
    rgba.save(OUT / 'fig_setup_cutout.png')
    return rgba, mask


def figure(rgba, mask):
    W, H = rgba.size
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(3.5, 2.0))
    ax = fig.add_axes([0, 0.16, 1, 0.68]); ax.imshow(rgba); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    px = lambda x, y: ((x - X0) * K, (y - Y0) * K)
    lab = [  # text, text x (figure fraction), top/bottom, target in the 2000 x 1500 grid
        ('X500 with Pixhawk 6C (PX4),\nprops removed', 0.02, 'top', (520, 470)),
        ('GPS', 0.20, 'bottom', (392, 598)),
        ('Jetson Orin Nano Super\n(int8 VLM, Sec. VI-E)', 0.37, 'bottom', (985, 765)),
        ('USB-C', 0.47, 'top', (1150, 600)),
        ('laptop: AirSim replay\nof this study (inserted)', 0.63, 'top', (1460, 540)),
    ]
    texts = []
    for t, tx, pos, tgt in lab:
        ty = 0.99 if pos == 'top' else 0.01
        txt = fig.text(tx, ty, t, fontsize=6.5, va='top' if pos == 'top' else 'bottom', ha='left', color='#1f1f1e',
                       linespacing=1.1)
        texts.append((txt, pos, tgt))
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    boxes = []
    for txt, pos, tgt in texts:
        bb = txt.get_window_extent(r); boxes.append(bb)
        b = inv.transform(bb)
        tp = inv.transform(ax.transData.transform(px(*tgt)))
        x = np.clip(tp[0], b[0][0] + 0.01, b[1][0] - 0.01)
        y = b[0][1] - 0.008 if pos == 'top' else b[1][1] + 0.008
        fig.add_artist(plt.Line2D([x, tp[0]], [y, tp[1]], color='#c8321e', lw=0.6, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([tp[0]], [tp[1]], marker='o', ms=2.2, color='#c8321e', transform=fig.transFigure))
    fb = fig.bbox
    for i, a in enumerate(boxes):                       # labels inside the figure, no overlap, not on the equipment
        assert a.x0 >= fb.x0 and a.x1 <= fb.x1 and a.y0 >= fb.y0 and a.y1 <= fb.y1, f'label {i} leaves the figure'
        for j in range(i):
            assert not a.overlaps(boxes[j]), f'labels {i} and {j} overlap'
        c = ax.transData.inverted().transform([[a.x0, a.y0], [a.x1, a.y1]])
        xa, xb = sorted(np.clip(c[:, 0], 0, W).astype(int)); ya, yb = sorted(np.clip(c[:, 1], 0, H).astype(int))
        assert not mask[ya:yb, xa:xb].any(), f'label {i} covers equipment'
    fig.savefig(OUT / 'fig_setup.pdf', dpi=400); fig.savefig(OUT / 'fig_setup_preview.png', dpi=300)
    print('labels ok:', len(boxes))


if __name__ == '__main__':
    figure(*cutout())
