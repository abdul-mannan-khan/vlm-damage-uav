#!/usr/bin/env python3
"""Step 8 / P3 (10 Nov 2026): the fine-tuned damage VLM (seed s0, LoRA merged, int8 language-model weights) ON the
Jetson Orin Nano of the X500 kit. Same prompt, image limits and letter-logit read-out as training (vlm_damage_tune.py).

Phases (each writes <out>/<phase>.jsonl, one line per image: id, letter logits A..E, preprocess ms, model ms, total ms):
  val, test : RescueNet val / test images (1024 px thumbnails, as cached for training) -> re-calibration on the PC
  city      : 640 x 480 CityEnviron frames from the step-6 flights -> the latency bar (p90 <= 5 s)
  load      : the city frames again, looped, until <out>/STOP exists (VLM load while the planner batch runs)
tegrastats is sampled every 500 ms into <out>/tegrastats.log; summary in <out>/summary.json.

    python bench_vlm.py --model DIR --data DIR --frames DIR --out DIR [--phases val,test,city,load]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QUESTION = ('This is an aerial drone photo taken after a hurricane. What is the most severe building damage visible?\n'
            'A) No buildings visible\nB) Buildings with no damage\nC) Minor damage (some roof shingles or panels missing)\n'
            'D) Major damage (large parts of the roof missing, walls partly collapsed)\n'
            'E) Total destruction (building flattened or completely collapsed)\nAnswer with one letter.')
MAX_PIXELS, MIN_PIXELS = 640 * 28 * 28, 128 * 28 * 28


class Int8Linear(nn.Module):
    """y = x W^T + b with W stored as int8 rows and one float scale per row; dequantised per call (bf16)."""
    def __init__(self, i, o, bias, dtype):
        super().__init__()
        self.in_features, self.out_features = i, o
        self.register_buffer('qweight', torch.empty(o, i, dtype=torch.int8, device='meta'))
        self.register_buffer('qscale', torch.empty(o, dtype=torch.float32, device='meta'))
        self.bias = nn.Parameter(torch.empty(o, dtype=dtype, device='meta'), requires_grad=False) if bias else None

    def forward(self, x):
        return F.linear(x, self.qweight.to(x.dtype) * self.qscale.to(x.dtype)[:, None], self.bias)


def load(model_dir, device='cuda', dtype=torch.bfloat16):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration as QV
    cfg = AutoConfig.from_pretrained(model_dir)
    with init_empty_weights():
        m = QV._from_config(cfg, torch_dtype=dtype, attn_implementation='sdpa')
    conv = getattr(QV, '_checkpoint_conversion_mapping', None) or {
        r'^visual': 'model.visual', r'^model(?!\.(language_model|visual))': 'model.language_model'}
    names = dict(m.named_modules())

    def to_model(k):
        for pat, rep in conv.items():
            if re.search(pat, k):
                k2 = re.sub(pat, rep, k)
                return k2
        return k
    files = sorted(glob.glob(os.path.join(model_dir, 'model-int8-*.safetensors')))
    qmods = set()
    for fn in files:
        with safe_open(fn, 'pt') as f:
            qmods |= {to_model(k[:-len('.qweight')]) for k in f.keys() if k.endswith('.qweight')}
    for q in qmods:                                   # swap nn.Linear -> Int8Linear before loading
        parent, child = q.rsplit('.', 1)
        lin = getattr(names[parent], child)
        setattr(names[parent], child, Int8Linear(lin.in_features, lin.out_features, lin.bias is not None, dtype))
    loaded = 0
    for fn in files:
        with safe_open(fn, 'pt') as f:
            for k in f.keys():
                set_module_tensor_to_device(m, to_model(k), device, value=f.get_tensor(k))
                loaded += 1
    m.tie_weights()
    left = [n for n, p in list(m.named_parameters()) + list(m.named_buffers()) if p.device.type == 'meta']
    assert not left, f'not loaded: {left[:5]}'
    m.to(device).eval()
    p = AutoProcessor.from_pretrained(model_dir, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, use_fast=True)
    letters = [p.tokenizer.encode(L, add_special_tokens=False)[0] for L in 'ABCDE']
    return m, p, letters, dict(tensors=loaded, int8_modules=len(qmods))


class Tegra(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.ram, self.gpu, self.pw, self.stop = path, [], [], [], False

    def run(self):
        try:
            pr = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            return
        with open(self.path, 'a') as lg:
            for line in pr.stdout:
                lg.write(line)
                r = re.search(r'RAM (\d+)/(\d+)MB', line); g = re.search(r'GR3D_FREQ (\d+)%', line)
                w = re.search(r'VDD_IN (\d+)mW', line)
                if r: self.ram.append(int(r.group(1)))
                if g: self.gpu.append(int(g.group(1)))
                if w: self.pw.append(int(w.group(1)))
                if self.stop:
                    pr.terminate(); break


@torch.inference_mode()
def run_one(m, p, letters, im):
    t0 = time.perf_counter()
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': QUESTION}]}]
    x = p(text=[p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[im],
          return_tensors='pt').to('cuda')
    torch.cuda.synchronize(); t1 = time.perf_counter()
    lg = m(**x, logits_to_keep=1).logits[0, -1].float()
    z = lg[letters].cpu().numpy()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return z, 1e3 * (t1 - t0), 1e3 * (t2 - t1), int(x['input_ids'].shape[1])


def stats(v):
    v = np.asarray(v, float)
    return dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)),
                mean=float(v.mean()), max=float(v.max())) if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True, help='folder with val/*.jpg, test/*.jpg, val_labels.json, test_labels.json')
    ap.add_argument('--frames', required=True, help='folder of 640x480 CityEnviron frames (*.jpg)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--phases', default='val,test,city,load')
    a = ap.parse_args()
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    tg = Tegra(os.path.join(a.out, 'tegrastats.log')); tg.start()
    t0 = time.time()
    m, p, letters, info = load(a.model)
    info['load_s'] = time.time() - t0
    info['cuda_alloc_mb_after_load'] = torch.cuda.memory_allocated() / 2 ** 20
    try:
        info['nvpmodel'] = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        info['nvpmodel'] = None
    frames = sorted(glob.glob(os.path.join(a.frames, '*.jpg')))
    for im in [Image.open(frames[0]).convert('RGB')] * 3:
        run_one(m, p, letters, im)                            # warm-up (not recorded)
    summ = dict(info=info, phases={})
    for ph in a.phases.split(','):
        out = os.path.join(a.out, f'{ph}.jsonl')
        done = set()
        if os.path.exists(out):
            done = {json.loads(l)['id'] for l in open(out)}
        if ph in ('val', 'test'):
            lab = json.load(open(os.path.join(a.data, f'{ph}_labels.json')))
            todo = [(k, os.path.join(a.data, ph, f'{k}.jpg'), lab[k]['damage']) for k in sorted(lab)]
        else:
            todo = [(os.path.basename(f), f, None) for f in frames]
        ms, pre = [], []
        with open(out, 'a') as fo:
            loop = 0
            while True:
                for k, f, y in todo:
                    kid = k if ph != 'load' else f'{loop}:{k}'
                    if kid in done:
                        continue
                    z, tp, tm, ntok = run_one(m, p, letters, Image.open(f).convert('RGB'))
                    fo.write(json.dumps(dict(id=kid, y=y, logits=z.tolist(), pre_ms=tp, model_ms=tm, total_ms=tp + tm,
                                             tokens=ntok, t=time.time())) + '\n'); fo.flush()
                    ms.append(tp + tm); pre.append(tp)
                    if ph == 'load' and os.path.exists(os.path.join(a.out, 'STOP')):
                        break
                if ph != 'load' or os.path.exists(os.path.join(a.out, 'STOP')):
                    break
                loop += 1
        summ['phases'][ph] = dict(total_ms=stats(ms), pre_ms=stats(pre),
                                  cuda_max_alloc_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        print(ph, json.dumps(summ['phases'][ph]['total_ms']), flush=True)
    tg.stop = True
    summ['tegrastats'] = dict(ram_peak_mb=max(tg.ram) if tg.ram else None, gr3d_mean_pct=float(np.mean(tg.gpu)) if tg.gpu else None,
                              vdd_in_mean_mw=float(np.mean(tg.pw)) if tg.pw else None, n=len(tg.ram))
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ['tegrastats']))


if __name__ == '__main__':
    main()
