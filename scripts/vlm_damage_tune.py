#!/usr/bin/env python3
"""Step 4 (5 Nov 2026): tune Qwen2.5-VL-3B to read damage from UAV images (RescueNet, CC0). Pre-registered plan:
paper/adhocnet/LLM_INTENT_PLAN_2026-11-05_damage.md, "Step 4".

Stage A  SFT-LoRA: cross-entropy on the option letter (proper scoring rule); belief = softmax over the option-letter logits.
Stage B  GRPO from the SFT model: short reasoning, then "Answer: X (confidence NN%)"; reward = correct + 0.2 format
         - 0.5 * Brier(stated confidence); group size 8; KL 0.04 to the SFT model. Runs only if the live-group probe passes.

    python3 scripts/vlm_damage_tune.py prep DATA                 # resize images, labels per split
    python3 scripts/vlm_damage_tune.py sft SEED
    python3 scripts/vlm_damage_tune.py probe SEED                # live-group fraction of the SFT model
    python3 scripts/vlm_damage_tune.py grpo SEED
    python3 scripts/vlm_damage_tune.py eval STAGE SEED           # val + test probabilities of the best checkpoint
"""
import json
import math
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_damage_step2 import QUESTIONS, label_one   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'vlm_damage' / 'tune'
CACHE = Path('/root/cache')
BASE = 'Qwen/Qwen2.5-VL-3B-Instruct'
KEYS = list(QUESTIONS)
CFG = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, lr=1e-4, warmup=0.03, batch=16, micro=8, max_pixels=640 * 28 * 28,
           eval_every=200, min_steps_sft=675, max_epochs_sft=3, plateau_window=3, plateau_eps=0.005,
           grpo_group=8, grpo_prompts=4, grpo_lr=2e-5, grpo_kl=0.04, grpo_min=300, grpo_max=1500, grpo_new_tokens=96)
COT = ('\nFirst think briefly (at most two sentences) inside <think></think>. Then write exactly: '
       'Answer: <letter> (confidence <0-100>%)')


# ---------------------------------------------------------------- data
def _prep_one(a):
    k, ip, mp, split = a
    o = CACHE / split / f'{k}.jpg'
    if not o.exists():
        im = Image.open(ip).convert('RGB')
        im.thumbnail((1024, 1024))
        im.save(o, quality=92)
    return k, label_one(mp)


def prep(data):
    data = Path(data)
    for split in ('train', 'val', 'test'):
        (CACHE / split).mkdir(parents=True, exist_ok=True)
        dirs = [d for d in data.iterdir() if d.is_dir() and d.name.lower().startswith(split)]
        imgs = {p.stem: p for d in dirs for p in d.rglob('*.jpg')}
        masks = {p.stem.replace('_lab', ''): p for d in dirs for p in d.rglob('*_lab.png')}
        ks = sorted(set(imgs) & set(masks))
        with Pool(16) as p:
            lab = dict(p.map(_prep_one, [(k, imgs[k], masks[k], split) for k in ks]))
        json.dump(lab, open(CACHE / f'{split}_labels.json', 'w'))
        print(split, len(lab), np.bincount([v['damage'] for v in lab.values()], minlength=5).tolist(), flush=True)


def items(split):
    lab = json.load(open(CACHE / f'{split}_labels.json'))
    return [(k, q, lab[k][q]) for k in sorted(lab) for q in KEYS]


def img(split, k):
    return Image.open(CACHE / split / f'{k}.jpg').convert('RGB')


# ---------------------------------------------------------------- model
def load(seed, adapter=None, merge_from=None, new_lora=True):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cuda',
                                                           attn_implementation='sdpa')
    p = AutoProcessor.from_pretrained(BASE, min_pixels=128 * 28 * 28, max_pixels=CFG['max_pixels'], use_fast=True)
    p.tokenizer.padding_side = 'left'
    if merge_from:
        m = PeftModel.from_pretrained(m, merge_from).merge_and_unload()
    if adapter:
        m = PeftModel.from_pretrained(m, adapter, is_trainable=False)
    elif new_lora:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.enable_input_require_grads()
        tgt = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        cfg = LoraConfig(r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'], lora_dropout=CFG['lora_dropout'],
                         target_modules=r'^model\.layers\.\d+\..*(' + '|'.join(tgt) + ')$', task_type='CAUSAL_LM')
        m = get_peft_model(m, cfg)
    letters = {L: p.tokenizer.encode(L, add_special_tokens=False)[0] for L in 'ABCDE'}
    return m, p, letters


def prompts(p, qs, cot=False):
    out = []
    for q in qs:
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': QUESTIONS[q][0] + (COT if cot else '')}]}]
        out.append(p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return out


def letter_logits(m, p, letters, ims, qs):
    x = p(text=prompts(p, qs), images=ims, return_tensors='pt', padding=True).to('cuda')
    try:
        lg = m(**x, logits_to_keep=1).logits[:, -1].float()
    except TypeError:
        lg = m(**x).logits[:, -1].float()
    return lg


def probs_batch(m, p, letters, split, its):
    import torch
    out = []
    for i in range(0, len(its), 16):
        b = its[i:i + 16]
        with torch.no_grad():
            lg = letter_logits(m, p, letters, [img(split, k) for k, _, _ in b], [q for _, q, _ in b])
        for j, (k, q, y) in enumerate(b):
            Ls = QUESTIONS[q][1]
            z = lg[j, [letters[L] for L in Ls]]
            out.append(dict(id=k, q=q, y=y, logits=z.cpu().tolist(), p=torch.softmax(z, 0).cpu().tolist()))
    return out


def macro_f1(rows, q='damage'):
    from sklearn.metrics import f1_score
    r = [x for x in rows if x['q'] == q]
    return float(f1_score([x['y'] for x in r], [int(np.argmax(x['p'])) for x in r], average='macro'))


def plateau(curve):
    w = CFG['plateau_window']
    if len(curve) <= w:
        return False
    best = np.maximum.accumulate([c['val_f1'] for c in curve])
    return best[-1] - best[-1 - w] < CFG['plateau_eps']


# ---------------------------------------------------------------- stage A: SFT
def sft(seed):
    import torch
    d = OUT / f'sft_s{seed}'
    d.mkdir(parents=True, exist_ok=True)
    m, p, letters = load(seed)
    m.print_trainable_parameters()
    tr, va = items('train'), items('val')
    steps_ep = len(tr) // CFG['batch']
    max_steps = steps_ep * CFG['max_epochs_sft']
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=CFG['lr'], weight_decay=0.0)
    warm = int(CFG['warmup'] * max_steps)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(warm, 1)) *
                                            0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps)))
    rng = random.Random(seed)
    curve, best, step, t0 = [], -1, 0, time.time()
    order = []
    while step < max_steps:
        if len(order) < CFG['batch']:
            nxt = list(range(len(tr))); rng.shuffle(nxt); order += nxt
        b = [tr[i] for i in order[:CFG['batch']]]; order = order[CFG['batch']:]
        m.train(); opt.zero_grad(); tot = 0.0
        for i in range(0, len(b), CFG['micro']):
            mb = b[i:i + CFG['micro']]
            lg = letter_logits(m, p, letters, [img('train', k) for k, _, _ in mb], [q for _, q, _ in mb])
            tgt = torch.tensor([letters[QUESTIONS[q][1][y]] for _, q, y in mb], device='cuda')
            loss = torch.nn.functional.cross_entropy(lg, tgt) * len(mb) / len(b)
            loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step(); step += 1
        if step % CFG['eval_every'] == 0 or step == max_steps:
            m.eval()
            rows = probs_batch(m, p, letters, 'val', va)
            f1 = macro_f1(rows)
            nll = float(np.mean([-math.log(max(r['p'][r['y']], 1e-9)) for r in rows]))
            curve.append(dict(step=step, epoch=step / steps_ep, train_loss=tot, val_f1=f1, val_nll=nll,
                              val_f1_flood=macro_f1(rows, 'flood'), val_f1_road=macro_f1(rows, 'road'), wall_s=time.time() - t0))
            print(json.dumps(curve[-1]), flush=True)
            if f1 > best:
                best = f1; m.save_pretrained(d / 'best')
            json.dump(dict(cfg=CFG, seed=seed, curve=curve), open(d / 'curve.json', 'w'), indent=1)
            if step >= CFG['min_steps_sft'] and plateau(curve):
                print('plateau stop at', step, flush=True); break
    print('finished sft', seed, flush=True)


# ---------------------------------------------------------------- stage B: GRPO
ANS = re.compile(r'Answer:\s*([A-E])\s*\(confidence\s*(\d{1,3})\s*%\)')


LETTER = re.compile(r'(?:Answer:\s*)?\(?\b([A-E])\b\)?')
CONF = re.compile(r'(\d{1,3})\s*%')


def parse(text, Ls):
    """Answer letter (after 'Answer:' if present, else the first option letter) and stated confidence (None if absent)."""
    tail = text.split('</think>')[-1]
    j = tail.find('Answer:')
    mt = LETTER.search(tail[j:] if j >= 0 else tail)
    if not mt or mt.group(1) not in Ls:
        return None, None, False
    mc = CONF.search(tail[mt.end():mt.end() + 40])
    strict = bool(ANS.search(tail)) and '<think>' in text and '</think>' in text
    return mt.group(1), (min(int(mc.group(1)), 100) / 100.0 if mc else None), strict


def reward(text, q, y):
    """1[correct] + 0.2 full format - 0.5 Brier(stated confidence; missing = 100 %); no readable letter = -0.5."""
    Ls = QUESTIONS[q][1]
    a, c, strict = parse(text, Ls)
    if a is None:
        return -0.5, False, None
    ok = Ls.index(a) == y
    cc = 1.0 if c is None else c
    return float(ok) + 0.2 * strict - 0.5 * (cc - float(ok)) ** 2, ok, c


def sample_groups(m, p, its, split, G):
    import torch
    # each image is encoded once and sampled G times (rows come out as [item0 x G, item1 x G, ...]); memory fix 6 Nov
    ims = [img(split, k) for k, _, _ in its]
    qs = [q for _, q, _ in its]
    x = p(text=prompts(p, qs, cot=True), images=ims, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        g = m.generate(**x, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=CFG['grpo_new_tokens'],
                       num_return_sequences=G)
    comp = g[:, x['input_ids'].shape[1]:]
    texts = p.batch_decode(comp, skip_special_tokens=True)
    return x, comp, texts


def probe(seed, n=64):
    m, p, letters = load(seed, merge_from=str(OUT / f'sft_s{seed}' / 'best'), new_lora=False)
    m.eval()
    tr = items('train'); random.Random(seed).shuffle(tr)
    live, fmt, rows = 0, 0, []
    for i in range(0, n, 4):
        b = tr[i:i + 4]
        _, _, texts = sample_groups(m, p, b, 'train', CFG['grpo_group'])
        for j, (k, q, y) in enumerate(b):
            rs = [reward(t, q, y) for t in texts[j * 8:(j + 1) * 8]]
            r = [x[0] for x in rs]
            live += np.std(r) > 1e-6; fmt += sum(x[1] is not None and x[0] != -0.5 for x in rs)
            rows.append(dict(id=k, q=q, y=y, rewards=r, example=texts[j * 8][:300]))
    o = dict(seed=seed, groups=n, live_frac=live / n, format_frac=fmt / (8 * n), rule='>= 0.5 to run GRPO', rows=rows)
    json.dump(o, open(OUT / f'probe_s{seed}.json', 'w'), indent=1)
    print('probe', seed, 'live', o['live_frac'], 'format', o['format_frac'], flush=True)


def seq_logp(m, x, comp, pad):
    import torch
    ids = torch.cat([x['input_ids'], comp], 1)
    att = torch.cat([x['attention_mask'], (comp != pad).long()], 1)
    kw = {k: v for k, v in x.items() if k not in ('input_ids', 'attention_mask')}
    lg = m(input_ids=ids, attention_mask=att, **kw).logits[:, x['input_ids'].shape[1] - 1:-1].float()
    return torch.log_softmax(lg, -1).gather(-1, comp[..., None])[..., 0]


def cot_eval(m, p, letters, split, its):
    """Greedy reasoning, then the option-letter distribution right after 'Answer: ' (the belief)."""
    import torch
    out = []
    for i in range(0, len(its), 8):          # batch 8 (was 16): memory only, 6 Nov
        b = its[i:i + 8]
        ims = [img(split, k) for k, _, _ in b]
        x = p(text=prompts(p, [q for _, q, _ in b], cot=True), images=ims, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            g = m.generate(**x, do_sample=False, max_new_tokens=CFG['grpo_new_tokens'])
        texts = p.batch_decode(g[:, x['input_ids'].shape[1]:], skip_special_tokens=True)
        pre = []
        for t, base in zip(texts, prompts(p, [q for _, q, _ in b], cot=True)):
            j = t.find('Answer:')
            pre.append(base + (t[:j] if j >= 0 else t.rstrip() + '\n') + 'Answer: ')
        x2 = p(text=pre, images=ims, return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            lg = m(**x2).logits[:, -1].float()
        for j, (k, q, y) in enumerate(b):
            z = lg[j, [letters[L] for L in QUESTIONS[q][1]]]
            a, c, strict = parse(texts[j], QUESTIONS[q][1])
            out.append(dict(id=k, q=q, y=y, logits=z.cpu().tolist(), p=torch.softmax(z, 0).cpu().tolist(),
                            answer=a, stated_conf=c, strict=strict, text=texts[j][:240]))
    return out


def grpo(seed):
    import torch
    d = OUT / f'grpo_s{seed}'
    d.mkdir(parents=True, exist_ok=True)
    m, p, letters = load(seed, merge_from=str(OUT / f'sft_s{seed}' / 'best'))
    tr, va = items('train'), items('val')
    rng = random.Random(seed + 100)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=CFG['grpo_lr'])
    G, P = CFG['grpo_group'], CFG['grpo_prompts']
    curve, best, t0, hist = [], -1, time.time(), []
    for step in range(1, CFG['grpo_max'] + 1):
        b = rng.sample(tr, P)
        m.eval()
        x, comp, texts = sample_groups(m, p, b, 'train', G)
        r = np.array([reward(t, q, y)[0] for (k, q, y) in b for t in texts[b.index((k, q, y)) * G:(b.index((k, q, y)) + 1) * G]])
        r = r.reshape(P, G)
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)
        live = float((r.std(1) > 1e-6).mean())
        mask = (comp != p.tokenizer.pad_token_id).float()
        m.train(); opt.zero_grad()
        A = torch.tensor(adv.ravel(), device='cuda', dtype=torch.float32)
        for i in range(0, P * G, 8):
            sl = slice(i, i + 8)
            xs = p(text=prompts(p, [q for _, q, _ in b for _ in range(G)][sl], cot=True),
                   images=[img('train', k) for k, _, _ in b for _ in range(G)][sl], return_tensors='pt', padding=True).to('cuda')
            c = comp[sl]
            lp = seq_logp(m, xs, c, p.tokenizer.pad_token_id)
            with torch.no_grad(), m.disable_adapter():
                lr_ = seq_logp(m, xs, c, p.tokenizer.pad_token_id)
            kl = torch.exp(lr_ - lp) - (lr_ - lp) - 1
            mk = mask[sl]
            tok = -(A[sl, None] * torch.exp(lp - lp.detach())) + CFG['grpo_kl'] * kl
            loss = ((tok * mk).sum(1) / mk.sum(1).clamp(min=1)).sum() / (P * G)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        hist.append(dict(step=step, reward=float(r.mean()), live=live))
        if step % CFG['eval_every'] == 0 or step == CFG['grpo_max']:
            m.eval()
            rows = cot_eval(m, p, letters, 'val', va)
            f1 = macro_f1(rows)
            curve.append(dict(step=step, val_f1=f1, val_brier=float(np.mean([sum((pi - (j == x['y'])) ** 2 for j, pi in enumerate(x['p']))
                                                                            for x in rows])),
                              train_reward=float(np.mean([h['reward'] for h in hist[-CFG['eval_every']:]])),
                              live=float(np.mean([h['live'] for h in hist[-CFG['eval_every']:]])), wall_s=time.time() - t0))
            print(json.dumps(curve[-1]), flush=True)
            if f1 > best:
                best = f1; m.save_pretrained(d / 'best')
            json.dump(dict(cfg=CFG, seed=seed, curve=curve, hist=hist), open(d / 'curve.json', 'w'))
            if step >= CFG['grpo_min'] and plateau(curve):
                print('plateau stop at', step, flush=True); break
    print('finished grpo', seed, flush=True)


def evaluate(stage, seed):
    if stage == 'base':
        m, p, letters = load(seed, new_lora=False)
    elif stage == 'sft':
        m, p, letters = load(seed, adapter=str(OUT / f'sft_s{seed}' / 'best'))
    else:
        import torch  # noqa: F401
        from peft import PeftModel
        m, p, letters = load(seed, merge_from=str(OUT / f'sft_s{seed}' / 'best'), new_lora=False)
        m = PeftModel.from_pretrained(m, str(OUT / f'grpo_s{seed}' / 'best'))
    m.eval()
    for split in ('val', 'test'):
        its = items(split)
        rows = cot_eval(m, p, letters, split, its) if stage == 'grpo' else probs_batch(m, p, letters, split, its)
        with open(OUT / f'eval_{stage}_s{seed}_{split}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        print('eval', stage, seed, split, 'damage macro-F1', round(macro_f1(rows), 4), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'prep':
        prep(sys.argv[2])
    elif mode == 'sft':
        sft(int(sys.argv[2]))
    elif mode == 'probe':
        probe(int(sys.argv[2]))
    elif mode == 'grpo':
        grpo(int(sys.argv[2]))
    elif mode == 'eval':
        evaluate(sys.argv[2], int(sys.argv[3]))
