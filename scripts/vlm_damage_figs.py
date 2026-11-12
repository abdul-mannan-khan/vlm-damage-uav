"""Figures for the tuned damage VLM: learning curves with seed band, reliability diagrams, risk-coverage."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts')
from vlm_damage_analyse import temp_scale, softmax

R = 'results/vlm_damage/tune'; O = 'results/vlm_damage/figs'; S = (0, 1, 2, 3)

def load(stage, s, split, q):
    rows = [json.loads(l) for l in open(f'{R}/eval_{stage}_s{s}_{split}.jsonl')]
    rows = [r for r in rows if r['q'] == q]
    return np.array([r['logits'] for r in rows], float), np.array([r['y'] for r in rows])

# 1. learning curves
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
for i, (k, name) in enumerate([('val_f1', 'damage'), ('val_f1_flood', 'flood'), ('val_f1_road', 'road')]):
    cs = [json.load(open(f'{R}/sft_s{s}/curve.json'))['curve'] for s in S]
    n = min(len(c) for c in cs); st = [c['step'] for c in cs[0][:n]]
    M = np.array([[c[k] for c in cc[:n]] for cc in cs])
    for cc in cs: ax[i].plot([c['step'] for c in cc], [c[k] for c in cc], lw=.6, alpha=.5)
    ax[i].fill_between(st, M.min(0), M.max(0), alpha=.2, color='k', label='seed range')
    ax[i].plot(st, M.mean(0), 'k', lw=1.6, label='mean of 4 seeds')
    ax[i].set_title(f'{name}: validation macro-F1'); ax[i].set_xlabel('SFT step'); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{O}/learning_curves.png', dpi=160); fig.savefig(f'{O}/learning_curves.pdf')

# 2. reliability (damage), seed ensemble, raw vs temperature-scaled
def rel(P, y, bins=15):
    c = P.max(1); a = (P.argmax(1) == y); e = np.linspace(0, 1, bins + 1); xs, ys, ns = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (c > lo) & (c <= hi)
        if m.sum(): xs.append(c[m].mean()); ys.append(a[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for i, q in enumerate(['damage', 'flood', 'road']):
    Praw, Pts = [], []
    for s in S:
        Lv, yv = load('sft', s, 'val', q); Lt, yt = load('sft', s, 'test', q)
        T = temp_scale(Lv, yv); Praw.append(softmax(Lt)); Pts.append(softmax(Lt, T))
    for P, lab, mk in [(np.mean(Praw, 0), 'raw', 'o'), (np.mean(Pts, 0), 'temperature-scaled', 's')]:
        x, a, n = rel(P, yt); ax[i].plot(x, a, mk + '-', ms=4, label=lab)
    ax[i].plot([0, 1], [0, 1], 'k--', lw=.8); ax[i].set_title(f'{q}: seed ensemble, test'); ax[i].set_xlabel('confidence'); ax[i].grid(alpha=.3)
ax[0].set_ylabel('accuracy'); ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{O}/reliability.png', dpi=160); fig.savefig(f'{O}/reliability.pdf')

# 3. risk-coverage (damage, ensemble), selective by max prob
fig, ax = plt.subplots(figsize=(4.2, 3.4))
for q in ['damage', 'flood', 'road']:
    Pts = []
    for s in S:
        Lv, yv = load('sft', s, 'val', q); Lt, yt = load('sft', s, 'test', q); Pts.append(softmax(Lt, temp_scale(Lv, yv)))
    P = np.mean(Pts, 0); o = np.argsort(-P.max(1)); err = (P.argmax(1) != yt)[o]
    cov = np.arange(1, len(o) + 1) / len(o); ax.plot(cov, np.cumsum(err) / np.arange(1, len(o) + 1), label=q)
ax.set_xlabel('coverage (answers kept)'); ax.set_ylabel('error rate'); ax.set_title('risk-coverage, test'); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{O}/risk_coverage.png', dpi=160); fig.savefig(f'{O}/risk_coverage.pdf')
print('ok')
