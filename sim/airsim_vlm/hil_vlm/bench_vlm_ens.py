#!/usr/bin/env python3
"""Step 11 (12 Nov 2026): four LoRA seeds on the Jetson from one int8 model (prereg results/vlm_damage/step11_*).
The int8 model holds seed 0 merged (W0 + D0). Seed s adds D_s - D_0 = [B_s, -B_0] [A_s; A_0] (x scale) at run time.
The 4 seeds run as one batch; the frozen vision tower runs once per image and its output is repeated.

    python bench_vlm_ens.py --model DIR --adapters DIR(with s0..s3 adapter_model.safetensors) --data DIR --frames DIR
                            --out DIR [--phases city,test] [--limit 40]
Writes <out>/<phase>.jsonl: id, y, logits (4 x 5), total_ms, vision_ms; <out>/summary.json; tegrastats log."""
import argparse, glob, json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_vlm import QUESTION, Int8Linear, load, Tegra, stats

NS = 4


def attach(m, adapters_dir, scale=2.0):
    """Give every Int8Linear that has a LoRA a stacked low-rank delta per batch element (seed 0 = zero)."""
    from safetensors import safe_open
    names = dict(m.named_modules())
    T = []
    for s in range(NS):
        with safe_open(os.path.join(adapters_dir, f's{s}', 'adapter_model.safetensors'), 'pt') as f:
            T.append({k: f.get_tensor(k) for k in f.keys()})
    keys = sorted({k.replace('base_model.model.', '').rsplit('.lora_', 1)[0] for k in T[0]})
    n = 0
    for k in keys:
        mod = names.get(re.sub(r'^model\.layers', 'model.language_model.layers', k)) or names.get(k)
        assert isinstance(mod, Int8Linear), k
        A0, B0 = T[0][f'base_model.model.{k}.lora_A.weight'], T[0][f'base_model.model.{k}.lora_B.weight']
        As, Bs = [], []                                              # seeds 1..3 only (seed 0 adds nothing)
        for s in range(1, NS):
            A, B = T[s][f'base_model.model.{k}.lora_A.weight'], T[s][f'base_model.model.{k}.lora_B.weight']
            As.append(torch.cat([A, A0], 0)); Bs.append(torch.cat([B, -B0], 1) * scale)
        mod.register_buffer('dA', torch.stack(As).to('cuda', torch.bfloat16))     # (NS-1) x 2r x in
        mod.register_buffer('dB', torch.stack(Bs).to('cuda', torch.bfloat16))     # (NS-1) x out x 2r
        n += 1
    return n


def fwd(self, x):
    y = F.linear(x, self.qweight.to(x.dtype) * self.qscale.to(x.dtype)[:, None], self.bias)
    if hasattr(self, 'dA') and x.dim() == 3 and x.shape[0] == NS:
        d = torch.bmm(torch.bmm(x[1:], self.dA.transpose(1, 2)), self.dB.transpose(1, 2))
        y = torch.cat([y[:1], y[1:] + d], 0)
    return y


Int8Linear.forward = fwd


@torch.inference_mode()
def run_ens(m, p, letters, im):
    t0 = time.perf_counter()
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': QUESTION}]}]
    txt = p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    x = p(text=[txt] * NS, images=[im] * NS, return_tensors='pt').to('cuda')
    vis = m.model.visual; orig = vis.forward; tv = {}

    def once(pixel_values, grid_thw=None, **kw):                   # vision once, repeated for the 4 seeds
        torch.cuda.synchronize(); a = time.perf_counter()
        n1 = pixel_values.shape[0] // NS
        o = orig(pixel_values[:n1], grid_thw=grid_thw[:1], **kw)
        o = torch.cat([o] * NS, 0) if torch.is_tensor(o) else o
        torch.cuda.synchronize(); tv['ms'] = 1e3 * (time.perf_counter() - a)
        return o
    vis.forward = once
    try:
        lg = m(**x, logits_to_keep=1).logits[:, -1].float()
    finally:
        vis.forward = orig
    z = lg[:, letters].cpu().numpy()
    torch.cuda.synchronize()
    return z, 1e3 * (time.perf_counter() - t0), tv.get('ms'), int(x['input_ids'].shape[1])


def main():
    ap = argparse.ArgumentParser()
    for k in ('model', 'adapters', 'data', 'frames', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--phases', default='city,test'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    tg = Tegra(os.path.join(a.out, 'tegrastats.log')); tg.start()
    t0 = time.time(); m, p, letters, info = load(a.model); info['n_delta'] = attach(m, a.adapters)
    info['load_s'] = time.time() - t0; info['cuda_alloc_mb_after_load'] = torch.cuda.memory_allocated() / 2 ** 20
    frames = sorted(glob.glob(os.path.join(a.frames, '*.jpg')))
    for _ in range(2):
        run_ens(m, p, letters, Image.open(frames[0]).convert('RGB'))
    summ = dict(info=info, phases={})
    for ph in a.phases.split(','):
        if ph in ('val', 'test'):
            lab = json.load(open(os.path.join(a.data, f'{ph}_labels.json')))
            todo = [(k, os.path.join(a.data, ph, f'{k}.jpg'), lab[k]['damage']) for k in sorted(lab)]
        else:
            todo = [(os.path.basename(f), f, None) for f in frames]
        if a.limit:
            todo = todo[:a.limit]
        out = os.path.join(a.out, f'{ph}.jsonl'); done = {json.loads(l)['id'] for l in open(out)} if os.path.exists(out) else set()
        ms, vm = [], []
        with open(out, 'a') as fo:
            for k, f, y in todo:
                if k in done or os.path.exists(os.path.join(a.out, 'STOP')):
                    continue
                z, t, v, ntok = run_ens(m, p, letters, Image.open(f).convert('RGB'))
                fo.write(json.dumps(dict(id=k, y=y, logits=z.tolist(), total_ms=t, vision_ms=v, tokens=ntok, t=time.time())) + '\n'); fo.flush()
                ms.append(t); vm.append(v)
        summ['phases'][ph] = dict(total_ms=stats(ms), vision_ms=stats(vm), cuda_max_alloc_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        print(ph, json.dumps(summ['phases'][ph]['total_ms']), flush=True)
    tg.stop = True
    summ['tegrastats'] = dict(ram_peak_mb=max(tg.ram) if tg.ram else None, n=len(tg.ram))
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ['tegrastats']))


if __name__ == '__main__':
    main()
