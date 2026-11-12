#!/usr/bin/env python3
"""Live onboard VLM for the CityEnviron flights (step 5, 9 Nov 2026).

Watches the missions' frame folders (<root>/<mission>/frames/*_rgb.jpg) while they fly, asks the damage question of every
new frame with the 4 fine-tuned LoRA adapters (Qwen2.5-VL-3B, the same prompt and letter-logit read-out as training),
and applies the saved V3 belief model exactly (results/vlm_damage/v3_calibration.npz: per-seed vector scaling, mean,
ensemble temperature) plus the alpha = 0.1 conformal set. One line per frame in <mission>/frames/vlm.jsonl:
  file, per-seed letter logits, calibrated posterior P (A..E), argmax, conformal set, compute_s, lag_s (frame file
  written -> result), queue (frames waiting).
Runs until <root>/STOP_VLM exists and the queue is empty. Resumable (frames already in vlm.jsonl are skipped).

    python3 vlm_damage_live.py --root /work/vlm/runs --adapters /work/vlm/adapters --calib /work/vlm/v3_calibration.npz
"""
import argparse
import glob
import json
import os
import time

import numpy as np

BASE = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_PIXELS = 640 * 28 * 28          # as in training (vlm_damage_tune.CFG)
QUESTION = ('This is an aerial drone photo taken after a hurricane. What is the most severe building damage visible?\n'
            'A) No buildings visible\nB) Buildings with no damage\nC) Minor damage (some roof shingles or panels missing)\n'
            'D) Major damage (large parts of the roof missing, walls partly collapsed)\n'
            'E) Total destruction (building flattened or completely collapsed)\nAnswer with one letter.')


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


class Belief:
    def __init__(self, path):
        c = np.load(path)
        self.W, self.B, self.te, self.qhat = c['W'], c['B'], float(c['te']), float(c['qhat'])

    def __call__(self, L):                      # L: seeds x 5 letter logits
        P = np.stack([softmax((L[s] - L[s].max()) * self.W[s] + self.B[s]) for s in range(len(L))])
        Pm = softmax(np.log(P.mean(0)) / self.te)
        return Pm, [int(k) for k in np.where(1 - Pm <= self.qhat)[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--adapters', required=True, help='folder with sft_s0..sft_s3 (each the saved best adapter)')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--poll', type=float, default=0.5)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    bel = Belief(a.calib)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cuda',
                                                           attn_implementation='sdpa')
    seeds = [0, 1, 2, 3]
    m = PeftModel.from_pretrained(m, os.path.join(a.adapters, 'sft_s0'), adapter_name='s0', is_trainable=False)
    for s in seeds[1:]:
        m.load_adapter(os.path.join(a.adapters, f'sft_s{s}'), adapter_name=f's{s}')
    m.eval()
    p = AutoProcessor.from_pretrained(BASE, min_pixels=128 * 28 * 28, max_pixels=MAX_PIXELS, use_fast=True)
    letters = [p.tokenizer.encode(L, add_special_tokens=False)[0] for L in 'ABCDE']
    msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': QUESTION}]}]
    prompt = p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    print('VLM ready', flush=True)
    done = {}
    while True:
        todo = []
        for fr in sorted(glob.glob(os.path.join(a.root, '*', 'frames'))):
            log = os.path.join(fr, 'vlm.jsonl')
            if fr not in done:
                done[fr] = {json.loads(l)['file'] for l in open(log)} if os.path.exists(log) else set()
            for f in sorted(glob.glob(os.path.join(fr, '*_rgb.jpg'))):
                base = os.path.basename(f)[:-8]
                if base not in done[fr]:
                    todo.append((os.path.getmtime(f), fr, base, f))
        todo.sort()
        if not todo:
            if os.path.exists(os.path.join(a.root, 'STOP_VLM')):
                print('stop file and empty queue: exit', flush=True)
                return
            time.sleep(a.poll)
            continue
        mt, fr, base, f = todo[0]
        try:
            im = Image.open(f).convert('RGB')
        except Exception:                     # frame still being written
            time.sleep(0.2)
            continue
        t0 = time.time()
        x = p(text=[prompt], images=[im], return_tensors='pt').to('cuda')
        L = []
        with torch.no_grad():
            for s in seeds:
                m.set_adapter(f's{s}')
                try:
                    lg = m(**x, logits_to_keep=1).logits[0, -1].float()
                except TypeError:                # older Qwen2.5-VL forward (same fallback as training)
                    lg = m(**x).logits[0, -1].float()
                L.append(lg[letters].cpu().numpy())
        torch.cuda.synchronize()
        L = np.array(L); P, C = bel(L)
        rec = dict(file=base, logits=np.round(L, 4).tolist(), P=np.round(P, 5).tolist(), top=int(P.argmax()), set=C,
                   compute_s=round(time.time() - t0, 3), lag_s=round(time.time() - mt, 3), queue=len(todo) - 1)
        with open(os.path.join(fr, 'vlm.jsonl'), 'a') as g:
            g.write(json.dumps(rec) + '\n')
        done[fr].add(base)


if __name__ == '__main__':
    main()
