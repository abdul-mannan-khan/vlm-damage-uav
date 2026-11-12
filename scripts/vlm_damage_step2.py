#!/usr/bin/env python3
"""Step 2 (5 Nov 2026): untuned VLMs read damage from real UAV images (RescueNet validation set, CC0).

Plan + pre-registered rule: paper/adhocnet/LLM_INTENT_PLAN_2026-11-05_damage.md.
Per image, three questions with lettered options; the answer probabilities come from the option-letter logits
(no free-text parsing). Models: Qwen2.5-VL-3B, Qwen2.5-VL-7B (untuned), CLIP ViT-L/14 zero-shot baseline.

    python3 scripts/vlm_damage_step2.py labels DATA_DIR            # ground truth from the masks
    python3 scripts/vlm_damage_step2.py predict DATA_DIR MODEL [N]  # MODEL in qwen3b qwen7b clip; N = first N (smoke)
Output: results/vlm_damage/preds_<model>.jsonl, labels.json (resumable)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'vlm_damage'
# RescueNet classes (descriptor note): 0 bg, 1 water, 2 bldg no dmg, 3 minor, 4 major, 5 total, 6 vehicle,
# 7 road clear, 8 road blocked, 9 tree, 10 pool
DMG = ['no_building', 'no_damage', 'minor', 'major', 'total']
QUESTIONS = {
    'damage': ('This is an aerial drone photo taken after a hurricane. What is the most severe building damage visible?\n'
               'A) No buildings visible\nB) Buildings with no damage\nC) Minor damage (some roof shingles or panels missing)\n'
               'D) Major damage (large parts of the roof missing, walls partly collapsed)\n'
               'E) Total destruction (building flattened or completely collapsed)\nAnswer with one letter.', 'ABCDE'),
    'flood': ('This is an aerial drone photo taken after a hurricane. Is there flood water (standing water on land or roads; '
              'swimming pools do not count)?\nA) Yes\nB) No\nAnswer with one letter.', 'AB'),
    'road': ('This is an aerial drone photo taken after a hurricane. Is any road blocked by debris, fallen trees or water?\n'
             'A) Yes\nB) No\nAnswer with one letter.', 'AB'),
}
CLIP_PROMPTS = {
    'damage': ['an aerial photo with no buildings', 'an aerial photo of undamaged houses',
               'an aerial photo of houses with minor roof damage', 'an aerial photo of houses with major damage, roofs torn off',
               'an aerial photo of completely destroyed, flattened houses'],
    'flood': ['an aerial photo of a flooded area with standing water', 'an aerial photo with no flooding'],
    'road': ['an aerial photo of a road blocked by debris or fallen trees', 'an aerial photo of clear roads'],
}


def pairs(data):
    data = Path(data)
    masks = {p.stem.replace('_lab', ''): p for p in data.rglob('*_lab.png')}
    imgs = {p.stem: p for p in data.rglob('*.jpg')}
    keys = sorted(set(masks) & set(imgs))
    return [(k, imgs[k], masks[k]) for k in keys]


def label_one(mask_path):
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    fr = np.bincount(m.ravel(), minlength=11)[:11] / m.size
    b = fr[2:6]
    if b.sum() < 0.01:
        dmg = 0
    else:
        ok = np.where(b >= 0.01)[0]
        dmg = 1 + (int(ok.max()) if len(ok) else int(b.argmax()))
    return dict(damage=dmg, flood=0 if fr[1] >= 0.05 else 1, road=0 if fr[8] >= 0.005 else 1,   # option index (A = yes)
                frac=[float(x) for x in fr])


def load_img(p, side=1344):
    im = Image.open(p).convert('RGB')
    im.thumbnail((side, side))
    return im


class Qwen:
    def __init__(self, name):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='cuda').eval()
        self.p = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        tok = self.p.tokenizer
        self.ids = {L: tok.encode(L, add_special_tokens=False) for L in 'ABCDE'}
        assert all(len(v) == 1 for v in self.ids.values()), self.ids

    def probs(self, im, q, letters):
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}]
        text = self.p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        x = self.p(text=[text], images=[im], return_tensors='pt').to('cuda')
        with self.torch.no_grad():
            lg = self.m(**x).logits[0, -1].float()
        z = lg[[self.ids[L][0] for L in letters]]
        top = self.p.tokenizer.decode(int(lg.argmax()))
        return self.torch.softmax(z, 0).cpu().numpy().tolist(), top


class Clip:
    def __init__(self, name='openai/clip-vit-large-patch14'):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.m = CLIPModel.from_pretrained(name).cuda().eval()
        self.p = CLIPProcessor.from_pretrained(name)

    def probs(self, im, key):
        x = self.p(text=CLIP_PROMPTS[key], images=im, return_tensors='pt', padding=True).to('cuda')
        with self.torch.no_grad():
            lg = self.m(**x).logits_per_image[0].float()
        return self.torch.softmax(lg, 0).cpu().numpy().tolist(), ''


MODELS = {'qwen3b': 'Qwen/Qwen2.5-VL-3B-Instruct', 'qwen7b': 'Qwen/Qwen2.5-VL-7B-Instruct', 'clip': None}

if __name__ == '__main__':
    mode, data = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    P = pairs(data)
    if mode == 'labels':
        lab = {k: label_one(m) for k, _, m in P}
        json.dump(lab, open(OUT / 'labels.json', 'w'))
        d = np.bincount([v['damage'] for v in lab.values()], minlength=5)
        print(len(lab), 'images; damage', dict(zip(DMG, d.tolist())),
              'flood', sum(v['flood'] == 0 for v in lab.values()), 'road_blocked', sum(v['road'] == 0 for v in lab.values()))
        sys.exit(0)
    if mode == 'predict':
        name = sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else None
        out = OUT / f'preds_{name}.jsonl'
        done = {json.loads(l)['id'] for l in open(out)} if out.exists() else set()
        model = Clip() if name == 'clip' else Qwen(MODELS[name])
        with open(out, 'a') as f:
            for k, ip, _ in P[:n]:
                if k in done:
                    continue
                im = load_img(ip, 336 * 4 if name != 'clip' else 1024)
                row = dict(id=k, model=name)
                for key, (q, letters) in QUESTIONS.items():
                    row[key], row[key + '_top'] = model.probs(im, key) if name == 'clip' else model.probs(im, q, letters)
                f.write(json.dumps(row) + '\n'); f.flush()
        print('finished', name, flush=True)
