#!/usr/bin/env python3
"""Step 8 / P3 (10 Nov 2026): merge one fine-tuned LoRA adapter (seed s0) into Qwen2.5-VL-3B-Instruct and store the
language-model linear layers as int8 (symmetric, per output channel), so the model fits the Jetson Orin Nano (7.6 GB
shared). Streaming, one tensor at a time (peak RAM ~2.5 GB, at the 151936 x 2048 embedding). Everything else stays bf16.

    W' = W + (alpha / r) * B @ A          (alpha = 32, r = 16, from adapter_config.json)
    q  = round(W' / s), s = max|W'_row| / 127,  W' ~ q * s

Output: <out>/model-int8-0000k.safetensors (+ configs copied from <base>, manifest.json with error stats and md5s).
Keys: quantised weights as '<name>.qweight' (int8) and '<name>.qscale' (float32, one per output row); others unchanged.

    python quantize_merge.py --base DIR --adapter results/vlm_damage/tune/sft_s0/best --out DIR [--vision-int8]
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LLM = re.compile(r'^model\.layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$')
VIS = re.compile(r'^visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$')
SHARD = 1.5e9


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--vision-int8', action='store_true')
    a = ap.parse_args()
    base, ad, out = Path(a.base), Path(a.adapter), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(ad / 'adapter_config.json'))
    scale = cfg['lora_alpha'] / cfg['r']
    lora = {}
    with safe_open(ad / 'adapter_model.safetensors', 'pt') as f:
        for k in f.keys():
            mod, kind = re.match(r'^base_model\.model\.(.+)\.lora_([AB])\.weight$', k).groups()
            lora.setdefault(mod, {})[kind] = f.get_tensor(k).float()
    idx = json.load(open(base / 'model.safetensors.index.json'))['weight_map']
    merged, quant, stats, buf, nbuf, shards = set(), [], [], {}, 0, []

    def flush():
        nonlocal buf, nbuf
        if buf:
            p = out / f'model-int8-{len(shards) + 1:05d}.safetensors'
            save_file(buf, str(p), metadata={'format': 'pt'})
            shards.append(p.name)
            buf, nbuf = {}, 0

    for shard in sorted(set(idx.values())):
        with safe_open(base / shard, 'pt') as f:
            for k in f.keys():
                t = f.get_tensor(k)
                mod = k[:-len('.weight')] if k.endswith('.weight') else None
                if mod in lora:
                    W = t.float() + scale * (lora[mod]['B'] @ lora[mod]['A'])
                    merged.add(mod)
                else:
                    W = None
                if LLM.match(k) or (a.vision_int8 and VIS.match(k)):
                    W = t.float() if W is None else W
                    s = (W.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12)
                    q = torch.round(W / s).clamp(-127, 127).to(torch.int8)
                    err = (q.float() * s - W).abs()
                    stats.append((k, float(err.max()), float(err.mean() / W.abs().mean())))
                    buf[k[:-len('.weight')] + '.qweight'] = q
                    buf[k[:-len('.weight')] + '.qscale'] = s.squeeze(1).contiguous()
                    nbuf += q.numel() + 4 * s.numel()
                    quant.append(k)
                else:
                    buf[k] = (W.to(t.dtype) if W is not None else t).contiguous()
                    nbuf += buf[k].numel() * buf[k].element_size()
                if nbuf > SHARD:
                    flush()
    flush()
    missing = sorted(set(lora) - merged)
    assert not missing, f'LoRA modules not found in base: {missing[:5]}'
    for n in ('config.json', 'generation_config.json', 'preprocessor_config.json', 'chat_template.json',
              'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
        if (base / n).exists():
            shutil.copy(base / n, out / n)
    for n in ('adapter_config.json',):
        shutil.copy(ad / n, out / ('source_' + n))
    rel = [s[2] for s in stats]
    man = dict(base='Qwen/Qwen2.5-VL-3B-Instruct', adapter=str(ad), adapter_md5=md5(ad / 'adapter_model.safetensors'),
               lora_scale=scale, lora_modules_merged=len(merged), quantised_tensors=len(quant), vision_int8=a.vision_int8,
               rel_mean_abs_err_mean=sum(rel) / len(rel), rel_mean_abs_err_max=max(rel),
               shards={s: md5(out / s) for s in shards},
               base_shards={s: md5(base / s) for s in sorted(set(idx.values()))})
    json.dump(man, open(out / 'manifest.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k not in ('shards', 'base_shards')}, indent=1))
    print('size GB', round(sum((out / s).stat().st_size for s in shards) / 1e9, 2))


if __name__ == '__main__':
    main()
