#!/usr/bin/env python3
"""Figures for the TVT draft (paper/vlm_tvt/figures/*.pdf), from saved results only.
  fig_links.pdf       step 7 idea 1: f/OPT per backhaul regime (n = 6) and MAQAC-S - RAG margin vs fleet size
  fig_perception.pdf  step 7 idea 2: planning loss vs perception level, with the tight guarantee's mean floor
  fig_reliability.pdf calibrated 4-seed ensemble (V3) on the RescueNet test set: reliability diagram
  fig_jetson.pdf      step 8: per-frame answer time and memory over the Jetson Orin Nano run (int8)
  fig_visibility.pdf  step 6: flown share of buildings seen vs calibrated visibility model and 58 m disc
Palette: categorical slots 1-3 of the dataviz reference palette (validated, light mode); markers differ per series.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / 'results' / 'vlm_damage', ROOT / 'paper' / 'vlm_tvt' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
C = {'maqac_s': '#2a78d6', 'rag': '#eb6834', 'indep': '#1baf7a'}
MK = {'maqac_s': 'o', 'rag': 's', 'indep': '^'}
LAB = {'maqac_s': 'MAQAC-S (relay)', 'rag': 'RAG (one-hop)', 'indep': 'No coordination'}
INK, MUTED = '#1f1f1e', '#77766f'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'lines.linewidth': 1.5})
W1, W2 = 3.5, 7.16   # IEEE column / page width (in)


def fig_links():
    rows = json.load(open(R / 'step7_rows.json'))['idea1']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    regs = ['complete', 'medium', 'sparse']
    for j, p in enumerate(['maqac_s', 'rag', 'indep']):
        for i, reg in enumerate(regs):
            v = np.array([r[p]['ratio'] for r in rows if r['n'] == 6 and r['regime'] == reg])
            x = i + (j - 1) * 0.22
            ax[0].plot([x, x], np.quantile(v, [0.05, 0.95]), color=C[p], lw=1.5, solid_capstyle='round')
            ax[0].plot(x, v.mean(), MK[p], color=C[p], ms=5, mec='white', mew=0.8, label=LAB[p] if i == 0 else None)
            ax[0].plot(x, v.min(), '_', color=C[p], ms=7, mew=1.5)
    ax[0].axhline(0.5, color=MUTED, lw=0.8, ls='--'); ax[0].text(-0.45, 0.51, 'guarantee 1/2', ha='left', va='bottom', color=MUTED, fontsize=7)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['complete', '350 m', '200 m'])
    ax[0].set_xlabel('backhaul range (n = 6)'); ax[0].set_ylabel('f / OPT'); ax[0].set_ylim(0.45, 1.2); ax[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax[0].legend(loc='upper center', fontsize=7, ncol=3, handletextpad=0.3, columnspacing=0.8, bbox_to_anchor=(0.5, 1.03))
    ax[0].set_title('(a) mean (marker), 5\u201395 % range (bar), worst (dash)', fontsize=8, color=INK, loc='left')
    S = json.load(open(R / 'step7_summary.json'))['idea1']['cells']
    for reg, ls, mk in (('complete', ':', 'o'), ('medium', '--', 's'), ('sparse', '-', '^')):
        ns = [4, 6, 8, 10]
        ax[1].plot(ns, [S[f'{n}|{reg}']['diff_mean_pts'] for n in ns], ls, marker=mk, color=INK, ms=4.5, mfc='white',
                   label={'complete': 'complete', 'medium': '350 m', 'sparse': '200 m'}[reg])
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel('number of UAVs'); ax[1].set_ylabel('MAQAC-S \u2212 RAG (points)'); ax[1].set_xticks([4, 6, 8, 10])
    ax[1].legend(fontsize=7, loc='upper left')
    ax[1].set_title('(b) paired margin; n = 8, 10 vs upper bound', fontsize=8, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(OUT / 'fig_links.pdf'); plt.close(fig)


def fig_perception():
    S = json.load(open(R / 'step7_summary.json'))['idea2']['cells']
    lev = ['argmax', 'conformal', 'inj_0.05', 'inj_0.10', 'inj_0.20']
    names = ['top label', 'conformal\nmidpoint', '+5 %', '+10 %', '+20 %']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    for k, reg in enumerate(('complete', 'sparse')):
        c = S[reg]
        for j, p in enumerate(('maqac_s', 'rag')):
            x = np.arange(len(lev)) + (j - 0.5) * 0.3
            ax[k].bar(x, [100 * c[f'{p}|{l}']['loss_vs_oracle'] for l in lev], 0.28, color=C[p], label=LAB[p],
                      edgecolor='white', linewidth=1)
        ax[k].set_xticks(range(len(lev))); ax[k].set_xticklabels(names, fontsize=7)
        ax[k].set_ylabel('loss (points of OPT)'); ax[k].set_ylim(0, 4.2)
        ax[k].set_title(f'({"ab"[k]}) {"complete" if reg == "complete" else "200 m"} backhaul, n = 6', fontsize=8, color=INK, loc='left')
        fl = [100 * c[f'maqac_s|{l}']['tight_floor_mean'] for l in lev]
        for i, v in enumerate(fl):
            ax[k].text(i, 3.85, f'floor {v:.0f} %', ha='center', fontsize=6.5, color='#4a4943')
    ax[0].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.9))
    fig.tight_layout(); fig.savefig(OUT / 'fig_perception.pdf'); plt.close(fig)


def fig_reliability():
    B = np.load(R / 'step3_obs_bundle.npz'); P, y = B['P'], B['y']
    conf, pred = P.max(1), P.argmax(1); ok = pred == y
    edges = np.linspace(0, 1, 16); idx = np.clip(np.digitize(conf, edges) - 1, 0, 14)
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls='--')
    xs, ys, ns = [], [], []
    for b in range(15):
        m = idx == b
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(m.sum())
    ax.plot(xs, ys, '-o', color=C['maqac_s'], ms=4, mec='white', mew=0.6)
    for x_, n_ in zip(xs, ns):
        ax.text(x_, 0.04, str(n_), ha='center', va='bottom', fontsize=6, color='#4a4943')
    ax.text(0.02, 0.04, 'images per bin:', fontsize=6, color='#4a4943', va='bottom')
    e = sum(abs(conf[idx == b].mean() - ok[idx == b].mean()) * (idx == b).mean() for b in range(15) if (idx == b).any())
    ax.text(0.04, 0.92, f'ECE = {e:.3f}, n = {len(y)}', fontsize=7.5, color=INK, transform=ax.transAxes)
    ax.set_xlabel('confidence (calibrated top-class probability)'); ax.set_ylabel('accuracy'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(OUT / 'fig_reliability.pdf'); plt.close(fig)
    return e


def fig_visibility():
    S = json.load(open(R / 'step6_summary.json'))['cells']
    T = ['60', '120', '180', '300']; t = [1, 2, 3, 5]
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    flown = np.mean([[S[k]['flown'][x] for x in T] for k in S], 0)
    calib = np.mean([[S[k]['calib'][x] for x in T] for k in S], 0)
    disc = np.mean([[S[k]['disc'][x] for x in T] for k in S], 0)
    ax.plot(t, 100 * flown, '-o', color=INK, ms=4, label='flown (AirSim camera)')
    ax.plot(t, 100 * calib, '--s', color=C['maqac_s'], ms=4, mfc='white', label='measured visibility model')
    ax.plot(t, 100 * disc, ':^', color=C['rag'], ms=4, mfc='white', label='58 m disc footprint')
    ax.set_xlabel('mission time (min)'); ax.set_ylabel('buildings seen (%)'); ax.set_xticks(t); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT / 'fig_visibility.pdf'); plt.close(fig)
    return dict(flown=flown.tolist(), calib=calib.tolist(), disc=disc.tolist())


def fig_jetson(run='results/vlm_damage/jetson/20261110_173516', batch_start=1794335672.0):
    """Jetson Orin Nano, int8 seed-0 model (step 8). (a) answer time of every frame over the run, with the median per
    input type and the network-simulator batch shaded; (b) memory in use (tegrastats RAM, shared CPU+GPU) vs 7.6 GB."""
    import re, datetime as dt
    run = ROOT / run
    rec = {k: [json.loads(l) for l in open(run / f'{k}.jsonl')] for k in ('val', 'test', 'city', 'load')}
    t0 = min(r['t'] for r in rec['val'])
    ph = {'RescueNet val': rec['val'], 'RescueNet test': rec['test'],
          'city frames': rec['city'] + [r for r in rec['load'] if r['t'] < batch_start],
          'city frames + planner': [r for r in rec['load'] if r['t'] >= batch_start]}
    col = {'RescueNet val': C['rag'], 'RescueNet test': C['rag'], 'city frames': C['maqac_s'], 'city frames + planner': C['maqac_s']}
    mn = lambda t: (np.asarray(t, float) - t0) / 60
    tend = mn(max(r['t'] for r in rec['load']))
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    D = '#4a4943'
    for k, rs in ph.items():
        t = mn([r['t'] for r in rs]); v = np.array([r['total_ms'] for r in rs]) / 1000
        ax[0].plot(t, v, '.', color=col[k], ms=1.8, mew=0, zorder=3)
    for a in ax:
        a.axvspan(mn(batch_start), tend, color='#ebeae5', lw=0, zorder=0)
        a.set_xlim(0, tend + 1)
    ax[0].axhline(5.0, color=MUTED, lw=0.8, ls='--')
    ax[0].text(1, 5.1, 'frame period 5 s', fontsize=6.5, color=D, va='bottom')
    ann = (('RescueNet val', 'RescueNet\nval + test\nmedian 2.61 s', 3.3, 'left'),
           ('city frames', 'median\n1.79 s', 2.2, 'left'), ('city frames + planner', 'median\n1.81 s', 2.2, 'left'))
    xs = {'RescueNet val': 1.0, 'city frames': mn(rec['city'][0]['t']) - 0.5, 'city frames + planner': mn(batch_start) + 1.5}
    for k, txt, y, ha in ann:
        ax[0].text(xs[k], y, txt, fontsize=6.5, color=D, ha=ha, va='bottom')
    ax[0].text(tend - 0.5, 4.75, 'netsim batch\n(360 episodes, CPU)', fontsize=6.5, color=D, ha='right', va='top')
    ax[0].set_ylim(0, 5.9); ax[0].set_yticks([0, 1, 2, 3, 4, 5]); ax[0].set_ylabel('answer time (s)')
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=C['rag'], label='RescueNet images (1024 px)'),
                          Line2D([], [], ls='', marker='o', ms=4, color=C['maqac_s'], label='AirSim city frames (640\u00d7480)')],
                 loc='lower left', fontsize=6.5, frameon=True, framealpha=0.95, edgecolor='none', borderpad=0.3, handletextpad=0.2)
    ax[0].set_title('(a) answer time of every frame', fontsize=8, color=INK, loc='left')
    T, M = [], []
    for line in open(run / 'tegrastats.log'):
        m = re.match(r'(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) RAM (\d+)/', line)
        if m:
            T.append(dt.datetime.strptime(m[1], '%m-%d-%Y %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()); M.append(int(m[2]) / 1000)
    T, M = mn(T), np.array(M); keep = (T >= 0) & (T <= tend + 1)
    ax[1].plot(T[keep], M[keep], color=INK, lw=1, label='in use')
    ax[1].axhline(7.62, color=MUTED, lw=0.9, ls='--')
    ax[1].text(1, 7.75, 'total 7.6 GB (shared CPU + GPU)', fontsize=6.5, color=D, va='bottom')
    ax[1].text(1, 6.65, f'peak {M[keep].max():.1f} GB', fontsize=6.5, color=D, va='top')
    ax[1].set_ylim(0, 9.0); ax[1].set_yticks([0, 2, 4, 6, 8]); ax[1].set_ylabel('memory (GB)')
    ax[1].set_xlabel('time since start of run (min)')
    ax[1].set_title('(b) memory in use', fontsize=8, color=INK, loc='left')
    fig.tight_layout(h_pad=0.6); fig.savefig(OUT / 'fig_jetson.pdf'); plt.close(fig)
    out = {k: (float(np.median(v)), float(np.quantile(v, 0.9)), float(v.max()), len(v))
           for k, v in ((k, np.array([r['total_ms'] for r in rs]) / 1000) for k, rs in ph.items())}
    out['ram_peak_gb'] = float(M[keep].max())
    return out


def fig_training():
    """Fine-tuning (step 4): (a) validation damage macro-F1 every 200 steps, 4 LoRA seeds, mean and min-max band, best untuned
    model; (b) validation NLL; (c) test confusion matrix of the 4-seed ensemble (row-normalised, counts in cells)."""
    T = R / 'tune'
    cur = [json.load(open(T / f'sft_s{k}' / 'curve.json'))['curve'] for k in range(4)]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    grid = sorted({c['step'] for cc in cur for c in cc})
    for key, a, lab in (('val_f1', ax[0], 'validation damage macro-F1'), ('val_nll', ax[1], 'validation NLL (nats)')):
        M = np.full((4, len(grid)), np.nan)
        for k, cc in enumerate(cur):
            for c in cc:
                M[k, grid.index(c['step'])] = c[key]
            st = [c['step'] for c in cc]; v = [c[key] for c in cc]
            a.plot(st, v, color=C['maqac_s'], lw=0.6, alpha=0.45)
        ok = (~np.isnan(M)).sum(0) >= 2
        g = np.array(grid)[ok]
        a.fill_between(g, np.nanmin(M[:, ok], 0), np.nanmax(M[:, ok], 0), color=C['maqac_s'], alpha=0.15, lw=0)
        a.plot(g, np.nanmean(M[:, ok], 0), color=C['maqac_s'], lw=1.6, label='mean, 4 seeds (band: range)')
        a.set_xlabel('training step (batch 16)'); a.set_ylabel(lab)
    ax[0].axhline(0.271, color=C['rag'], lw=1, ls='--'); ax[0].text(80, 0.235, 'best untuned (Qwen2.5-VL-7B) 0.27', color='#4a4943', fontsize=6.5)
    for cc in cur:
        b = max(cc, key=lambda c: c['val_f1']); ax[0].plot(b['step'], b['val_f1'], 'o', ms=3.5, color=INK, zorder=4)
    ax[0].plot([], [], 'o', ms=3.5, color=INK, label='checkpoint kept')
    ax[0].set_ylim(0.2, 0.78); ax[0].legend(fontsize=6.3, loc='lower right', bbox_to_anchor=(1.0, 0.13), frameon=False)
    ax[0].set_title('(a) learning curves', fontsize=8, color=INK, loc='left')
    ax[1].set_title('(b) validation loss', fontsize=8, color=INK, loc='left')
    # confusion matrix, 4-seed ensemble on the test set
    P, Y = {}, {}
    for k in range(4):
        for l in open(T / f'eval_sft_s{k}_test.jsonl'):
            r = json.loads(l)
            if r['q'] == 'damage':
                P.setdefault(r['id'], []).append(r['p']); Y[r['id']] = r['y']
    ids = sorted(P); yt = np.array([Y[i] for i in ids]); yp = np.array([np.mean(P[i], 0).argmax() for i in ids])
    cm = np.zeros((5, 5), int)
    for a_, b_ in zip(yt, yp):
        cm[a_, b_] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax[2].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    names = ['none', 'no dmg', 'minor', 'major', 'total']
    for i in range(5):
        for j in range(5):
            ax[2].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6.5, color='white' if cmn[i, j] > 0.55 else INK)
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(names, fontsize=6.5, rotation=30, ha='right')
    ax[2].set_yticks(range(5)); ax[2].set_yticklabels(names, fontsize=6.5)
    ax[2].set_xlabel('predicted'); ax[2].set_ylabel('true (RescueNet mask)'); ax[2].grid(False)
    ax[2].set_title(f'(c) test confusion (n = {len(ids)})', fontsize=8, color=INK, loc='left')
    fig.tight_layout(w_pad=0.8); fig.savefig(OUT / 'fig_training.pdf'); plt.close(fig)
    ks = np.union1d(yt, yp)                       # macro-F1 over labels present, as sklearn and scripts/vlm_damage_analyse.py
    f1 = np.mean([0.0 if ((yp == k) & (yt == k)).sum() == 0 else 2 * ((yp == k) & (yt == k)).sum() / ((yp == k).sum() + (yt == k).sum()) for k in ks])
    return dict(n=len(ids), acc=float((yt == yp).mean()), macro_f1=float(f1), cm=cm.tolist())


def fig_trust():
    """Step 10 (trust and belief, results/vlm_damage/step10_trust.json): (a) risk-coverage of the 4 seeds and the
    ensemble (ranked by max probability and by mutual information); (b) aleatoric vs epistemic uncertainty per test
    image, errors marked; (c) belief in the true class per true class; (d) conformal coverage per class, marginal vs
    class-conditional (Mondrian) sets, with 95 % Clopper-Pearson lower bounds."""
    import sys; sys.path.insert(0, str(ROOT / 'scripts'))
    from vlm_damage_trust_eval import risk_cov
    d = json.load(open(R / 'step10_trust.json')); A = np.load(R / 'step10_trust_arrays.npz')
    PT, Pt, yt, ale, mi = A['PT'], A['Pt'], A['yt'], A['ale'], A['mi']
    err = PT.argmax(1) != yt
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.3)); ax = ax.ravel()
    a = ax[0]
    for s in range(4):
        c, r, _ = risk_cov(Pt[s].max(1), Pt[s].argmax(1) != yt)
        a.plot(c, r, color=MUTED, lw=0.7, label='single seeds' if s == 0 else None)
    c, r, au = risk_cov(PT.max(1), err); a.plot(c, r, color=C['maqac_s'], label=f'ensemble, max p (AURC {au:.3f})')
    c, r, au2 = risk_cov(-mi, err); a.plot(c, r, color=C['rag'], ls='--', lw=1.1, label=f'ensemble, MI ({au2:.3f})')
    a.axvline(0.9, color=MUTED, lw=0.6, ls=':')
    a.set_xlabel('coverage (share of images kept)'); a.set_ylabel('error rate on kept images'); a.set_xlim(0, 1); a.set_ylim(0, 0.32)
    a.legend(loc='upper left', fontsize=7, handlelength=1.8); a.set_title('(a) risk vs coverage', fontsize=8)
    a = ax[1]
    a.scatter(ale[~err], mi[~err], s=4, color=C['maqac_s'], alpha=0.55, lw=0, label='correct')
    a.scatter(ale[err], mi[err], s=6, marker='x', color=C['rag'], lw=0.7, label='wrong')
    m3 = d['M3']
    a.set_xlabel('aleatoric  $E_s H(p_s)$ (nats)'); a.set_ylabel('epistemic  MI (nats)')
    a.set_title('(b) uncertainty split', fontsize=8)
    a.text(0.02, 0.97, f"error AUROC: aleatoric {m3['auroc_aleatoric']:.2f}\nMI {m3['auroc_mi']:.2f}; MI share {100 * m3['mi_share']:.0f} %",
           transform=a.transAxes, va='top', fontsize=7, color=INK)
    a.legend(loc='upper right', fontsize=7, markerscale=1.5)
    a = ax[2]
    names = ['none', 'intact', 'minor', 'major', 'destr.']
    vals = [PT[yt == k, k] for k in range(5)]
    bp = a.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for b in bp['boxes']:
        b.set(facecolor='#dfe9f7', edgecolor=C['maqac_s'], lw=0.8)
    for k, v in enumerate(vals):
        a.scatter(np.full(len(v), k + 1) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v, s=1.5, color=C['maqac_s'], alpha=0.35, lw=0)
    a.set_xticks(range(1, 6)); a.set_xticklabels(names); a.set_ylim(0, 1.02)
    a.set_ylabel('belief in the true class'); a.set_title('(c) belief per true class', fontsize=8)
    a = ax[3]
    x = np.arange(5); w = 0.38
    for j, (key, col, lab) in enumerate((('marginal', C['maqac_s'], 'marginal'), ('mondrian', C['indep'], 'per class (Mondrian)'))):
        pc = d['M4'][key]['per_class']; cov = [r['cov'] for r in pc]; lo = [r['cp_lo'] for r in pc]
        a.bar(x + (j - 0.5) * w, cov, w * 0.92, color=col, alpha=0.85, label=f"{lab}, set {d['M4'][key]['size']:.2f}")
        a.errorbar(x + (j - 0.5) * w, cov, yerr=[np.subtract(cov, lo), np.zeros(5)], fmt='none', ecolor=INK, lw=0.7, capsize=1.5)
    a.axhline(0.9, color=INK, lw=0.7, ls='--'); a.text(4.62, 0.9, ' 1−α', fontsize=7, ha='left', va='center')
    a.set_xticks(x); a.set_xticklabels(names); a.set_ylim(0.70, 1.0); a.set_xlim(-0.6, 4.9); a.set_ylabel('coverage (whisker: 95 % CP low)')
    a.legend(loc='lower center', fontsize=7, ncol=2, columnspacing=1.0); a.set_title('(d) conformal coverage per class', fontsize=8)
    fig.tight_layout(w_pad=1.2, h_pad=1.0); fig.savefig(OUT / 'fig_trust.pdf'); plt.close(fig)
    return dict(aurc_maxp=au, aurc_mi=au2)


if __name__ == '__main__':
    fig_links(); fig_perception()
    print('ECE in reliability figure', round(fig_reliability(), 4))
    print('visibility means', fig_visibility())
    print('jetson', fig_jetson())
    print('training', fig_training())
    print('trust', fig_trust())
