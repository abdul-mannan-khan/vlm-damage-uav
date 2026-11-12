#!/usr/bin/env python3
"""Graphical abstract for the TVT submission (11 Nov 2026). Images are real outputs of this study (UAV downward frame and
VLM probability bars from the AirSim replay video at t = 200 s; bench cut-out from scripts/vlm_setup_fig.py). Numbers are
copied from paper/vlm_tvt/main.tex (abstract and Section VI) and must change only with the paper.
Output: paper/vlm_tvt/submission/graphical_abstract.{png,pdf} (1980 x 885 px at 270 dpi; IEEE minimum 660 x 295).
    python3 scripts/vlm_graphical_abstract.py"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'vlm_tvt' / 'submission'; OUT.mkdir(parents=True, exist_ok=True)
VID = ROOT / 'results' / 'vlm_damage' / 'video' / 'vlmdmg_airsim_replay.mp4'
INK, MUTED, BLUE, ORANGE, GREEN, PALE = '#1f1f1e', '#55544e', '#2a78d6', '#eb6834', '#1baf7a', '#f3f2ee'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'dejavuserif'})


def frame():
    p = OUT / '_f.png'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '40', '-i', str(VID), '-frames:v', '1', str(p)], check=True)
    f = Image.open(p).convert('RGB'); p.unlink()
    return np.asarray(f.crop((1546, 239, 1902, 506))), np.asarray(f.crop((1560, 590, 1915, 812)))


def box(ax, x, y, w, h, title, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012', fc=PALE, ec=col, lw=1.2))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=9, fontweight='bold', color=col, va='top')


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=12, lw=1.4, color=INK))


def check(fig, ax):
    """Every text of the main axes lies inside the figure and inside at most one panel; no two texts overlap."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); fb = fig.bbox
    T = [(t.get_text()[:30], t.get_window_extent(r)) for t in ax.texts]
    for n, b in T:
        assert b.x0 >= fb.x0 and b.x1 <= fb.x1 and b.y0 >= fb.y0 and b.y1 <= fb.y1, f'outside figure: {n}'
    for i in range(len(T)):
        for j in range(i):
            assert not T[i][1].overlaps(T[j][1]), f'overlap: {T[i][0]} / {T[j][0]}'
    panels = [(0.012, 0.322), (0.356, 0.656), (0.689, 0.989)]
    W = fb.width
    for n, b in T:
        xs = [(p0 * W <= b.x0 and b.x1 <= p1 * W) for p0, p1 in panels]
        straddle = any(p0 * W < b.x1 and b.x0 < p1 * W for p0, p1 in panels) and not any(xs)
        if b.y1 > 0.3 * fb.height and b.y1 < 0.9 * fb.height:
            assert b.y0 >= 0.245 * fb.height, f'text runs below the panel: {n}'
        if straddle and 0.24 * fb.height < b.y0 and b.y1 < 0.9 * fb.height:
            raise AssertionError(f'text crosses a panel edge: {n}')


def main():
    cam, bars = frame()
    bench = Image.open(ROOT / 'paper' / 'vlm_tvt' / 'figures' / 'fig_setup_cutout.png')
    fig = plt.figure(figsize=(7.333, 3.278)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.5, 0.965, 'Trustworthy onboard VLM damage assessment for multi-UAV networks', ha='center', va='top',
            fontsize=11.5, fontweight='bold', color=INK)
    # panel 1: perceive
    box(ax, 0.012, 0.24, 0.31, 0.65, '1  Perceive', BLUE)
    fig.add_axes([0.022, 0.50, 0.135, 0.30]).imshow(cam); fig.axes[-1].axis('off')
    fig.add_axes([0.165, 0.50, 0.152, 0.30]).imshow(bars); fig.axes[-1].axis('off')
    ax.text(0.022, 0.488, 'UAV frame (AirSim)', fontsize=6.5, color=MUTED, va='top')
    ax.text(0.165, 0.488, 'belief; yellow = in set', fontsize=6.5, color=MUTED, va='top')
    ax.text(0.022, 0.43, 'Qwen2.5-VL-3B + LoRA, calibrated,\nsplit-conformal label set', fontsize=7.2, color=INK, va='top',
            linespacing=1.15)
    ax.text(0.022, 0.335, 'F1 0.27 → 0.70, ECE 0.046,\ncoverage 93.3 % (bound 90.6 %)', fontsize=7.2, color=BLUE, va='top',
            fontweight='bold', linespacing=1.15)
    arrow(ax, 0.327, 0.352, 0.565)
    # panel 2: guarantee
    box(ax, 0.356, 0.24, 0.30, 0.65, '2  Guarantee', ORANGE)
    ax.text(0.506, 0.70, r'$f_w(\hat S)\ \geq\ \beta\, f_w(O) - \beta\,\delta(O) - \delta(\hat S)$', ha='center',
            fontsize=10, color=INK)
    ax.text(0.506, 0.615, r'$\delta$: weight errors on served users only;' '\n'
            r'conformal sets bound it (Theorem 1)', ha='center', fontsize=7, color=MUTED, va='top', linespacing=1.15)
    ax.text(0.506, 0.49, 'Relaying the decision list keeps\n' r'$\beta = 1/2$ on any connected backhaul;' '\n'
            r'one-hop coordination can fall to $1/k$', ha='center', fontsize=7.2, color=INK, va='top', linespacing=1.15)
    ax.text(0.506, 0.335, 'Test bench, sets instead of top label:\nloss 2.1 → 1.1 points of the optimum', ha='center', fontsize=7.2,
            color=ORANGE, fontweight='bold', va='top', linespacing=1.15)
    arrow(ax, 0.66, 0.685, 0.565)
    # panel 3: fleet + onboard
    box(ax, 0.689, 0.24, 0.30, 0.65, '3  Coordinate, run onboard', GREEN)
    a3 = fig.add_axes([0.70, 0.53, 0.12, 0.25]); a3.set_xlim(0, 1); a3.set_ylim(0, 1); a3.axis('off')
    pts = [(0.12, 0.2), (0.4, 0.42), (0.68, 0.62), (0.9, 0.86)]
    a3.plot(*zip(*pts), color=GREEN, lw=1.2, ls='--')
    a3.add_patch(plt.Rectangle((0.02, 0.1), 0.18, 0.18, color=INK)); a3.text(0.11, 0.0, 'gateway', fontsize=5.5, ha='center', color=MUTED)
    a3.text(0.62, 0.12, 'targets chained\nin radio range', fontsize=5.5, ha='center', color=MUTED)
    for x, y in pts[1:]:
        a3.add_patch(Circle((x, y), 0.07, color=GREEN))
    fig.add_axes([0.825, 0.53, 0.158, 0.30]).imshow(bench); fig.axes[-1].axis('off')
    ax.text(0.904, 0.50, 'Jetson + Pixhawk bench', fontsize=6.5, color=MUTED, ha='center', va='top')
    ax.text(0.70, 0.46, 'Best-first relaying beats one-hop RAG\nby 2 to 25 % (8 to 16 UAVs, 200 m links)\nwith fewer messages',
            fontsize=7, color=INK, va='top', linespacing=1.15)
    ax.text(0.70, 0.325, 'Closed loop, Jetson + Pixhawk HIL:\n12 missions, 0 failsafes, 2.2 s/frame', fontsize=7.2, color=GREEN,
            fontweight='bold', va='top', linespacing=1.15)
    # bottom strip
    ax.text(0.5, 0.15, '48 AirSim CityEnviron missions with the live VLM (PX4 SITL): no collision; no false major/total '
            'damage alarm in 9,734 frames', ha='center', fontsize=7.2, color=INK)
    ax.text(0.5, 0.075, 'All tests pre-registered; failed criteria reported (e.g. single-seed ECE, no in-flight gain shown for damage weights).',
            ha='center', fontsize=6.5, color=MUTED)
    check(fig, ax)
    fig.savefig(OUT / 'graphical_abstract.png', dpi=270); fig.savefig(OUT / 'graphical_abstract.pdf')
    print(Image.open(OUT / 'graphical_abstract.png').size)


if __name__ == '__main__':
    main()
