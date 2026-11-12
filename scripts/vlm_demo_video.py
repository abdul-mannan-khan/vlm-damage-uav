#!/usr/bin/env python3
"""Narrated demo video for the TVT submission (11 Nov 2026), 1920x1080.
Scenes are still cards (matplotlib) built from the paper's figures, plus one live segment of this study's AirSim replay
(results/vlm_damage/video/vlmdmg_airsim_replay.mp4). Narration: local piper TTS (en_GB-alan-medium). Every number in the
narration is copied from paper/vlm_tvt/main.tex. Writes paper/vlm_tvt/submission/vlmdmg_demo.mp4 and vlmdmg_demo.srt.
Run through resman (TTS + ffmpeg):  resman run -m 2G -n vlmdemo -- python3 scripts/vlm_demo_video.py"""
import json, subprocess, wave
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'paper' / 'vlm_tvt' / 'figures'
OUT = ROOT / 'paper' / 'vlm_tvt' / 'submission'
VID = ROOT / 'results' / 'vlm_damage' / 'video' / 'vlmdmg_airsim_replay.mp4'
WORK = ROOT / 'build' / 'demo'
PIPER = Path.home() / '.venvs/maqac/bin/piper'
VOICE = Path.home() / '.local/share/piper-voices/en_GB-alan-medium/en_GB-alan-medium.onnx'
DARK, LIGHT, BLUE, INK, SOFT = '#14213D', '#FBFBF8', '#2A78D6', '#14213D', '#3A4558'
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})


def png(name):
    p = WORK / f'{name}.png'
    if not p.exists():
        subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', str(FIG / f'{name}.pdf'), str(WORK / name)], check=True)
    return np.asarray(Image.open(p).convert('RGB'))


def fit_lines(fig, text, size, width_frac):
    """Wrap each line so it fits width_frac of the figure width (measured, not guessed)."""
    r = fig.canvas.get_renderer(); maxw = width_frac * fig.bbox.width; out = []
    for para in text.split('\n'):
        cur = ''
        for w in para.split(' '):
            trial = (cur + ' ' + w).strip()
            t = fig.text(0, 0, trial, fontsize=size); wpx = t.get_window_extent(r).width; t.remove()
            if wpx > maxw and cur:
                out.append(cur); cur = w
            else:
                cur = trial
        out.append(cur)
    return '\n'.join(out)


def card(path, title, body=None, img=None, img_box=(0.06, 0.12, 0.88, 0.62), dark=False, kicker=None, lines=None):
    bg, fg, sub = (DARK, '#F4F1EA', '#C9D3E3') if dark else (LIGHT, INK, SOFT)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=bg)
    if kicker:
        fig.text(0.06, 0.90, kicker, fontsize=22, color='#8FB8EC' if dark else BLUE, va='top')
    fig.text(0.06, 0.86 if kicker else 0.90, title, fontsize=44, fontweight='bold', color=fg, va='top', wrap=True)
    if img is not None:
        ax = fig.add_axes(img_box); ax.imshow(img); ax.axis('off')
    if body:
        fig.text(0.06, 0.08, fit_lines(fig, body, 26, 0.88), fontsize=26, color=sub, va='bottom', linespacing=1.35)
    for (x, y, t, kw) in (lines or []):
        fig.text(x, y, t, **kw)
    fig.savefig(path, facecolor=bg); plt.close(fig)


def tts(text, wav):
    subprocess.run([str(PIPER), '-m', str(VOICE), '-f', str(wav), '--sentence_silence', '0.25'], input=text.encode(),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def scenes():
    eq = r'$f_w(\hat S)\ \geq\ \beta\, f_w(O) - \beta\,\delta(O) - \delta(\hat S)$'
    return [
        dict(id='title', card=dict(title='Trustworthy onboard vision-language damage\nassessment for multi-UAV networks', dark=True,
                                   kicker='Demo for the IEEE Transactions on Vehicular Technology submission',
                                   body='Abdul Manan Khan, Vishwanath Bijalwan, Babar Jamil\nAll results: pre-registered tests; failures reported.'),
             say='Hello, I am Abdul Manan Khan. This is a short demo of our paper on trustworthy onboard vision language '
                 'damage assessment for multi UAV networks, with perception aware coordination guarantees.'),
        dict(id='problem', card=dict(title='The problem', body=(
            'UAVs that restore connectivity after a disaster can also map building damage.\n'
            'Off-the-shelf VLMs grade damage poorly: macro-F1 0.18 (3B) and 0.27 (7B) on RescueNet.\n'
            'Coordination guarantees assume the objective is known exactly; a perception error voids them.'),
            img=None),
             say='After a disaster, UAVs that restore connectivity can also map building damage. The problem is that off the '
                 'shelf vision language models grade damage poorly, and coordination guarantees assume we know the objective '
                 'exactly. One wrong reading can quietly void the guarantee. We set out to close that gap.'),
        dict(id='pipeline', card=dict(title='Per-UAV pipeline', img='sys', img_box=(0.04, 0.30, 0.92, 0.42),
                                      body='Camera frame → tuned VLM → calibrated belief → conformal set → weight interval → '
                                           'relayed sequential greedy → PX4 autopilot'),
             say='On each UAV, every camera frame goes to our fine tuned Qwen 2.5 VL 3B model. We turn its answer into a '
                 'calibrated belief and a conformal label set with ninety percent coverage. The planner uses the midpoint of '
                 'each set, and relays its decisions to the other UAVs over the backhaul.'),
        dict(id='vlm', card=dict(title='A calibrated VLM with conformal sets', img='fig_reliability', img_box=(0.52, 0.16, 0.44, 0.66),
                                 lines=[(0.06, 0.70, 'Damage macro-F1: 0.27 → 0.70', dict(fontsize=34, color=INK)),
                                        (0.06, 0.60, 'ECE 0.046 (4-seed ensemble)', dict(fontsize=34, color=INK)),
                                        (0.06, 0.50, 'Coverage 93.3 %\n(lower bound 90.6 %)', dict(fontsize=34, color=INK, va='top')),
                                        (0.06, 0.30, '450 RescueNet test images', dict(fontsize=24, color=SOFT))]),
             say='Fine tuning raised damage macro F1 from 0.27 to 0.70. After calibration, the expected calibration error is '
                 '0.046, and our label sets cover 93 percent of test images, with a lower bound of 90.6 percent.'),
        dict(id='bound', card=dict(title='The perception-aware bound', dark=True,
                                   lines=[(0.5, 0.58, eq, dict(fontsize=56, color='#F4F1EA', ha='center'))],
                                   body='δ counts weight errors only on the users a plan serves.\n'
                                        'Conformal sets bound it in expectation and with high probability (Theorem 1).'),
             say='Our main result is this bound. A plan computed on perceived weights keeps beta times the optimum, minus a '
                 'term that counts weight errors only on the users the plans serve, and the conformal sets bound that term.'),
        dict(id='links', card=dict(title='Range-limited links: relay the decision list', img='fig_links', img_box=(0.04, 0.22, 0.92, 0.55),
                                   body='Relaying keeps the 1/2 guarantee on any connected backhaul; one-hop coordination can fall to 1/k.'),
             say='We also show that relaying the decision list keeps the one half guarantee on any connected backhaul, while '
                 'one hop coordination can fall to one over k. On our test bench with 200 metre links, the relayed planner '
                 'leads by 2.3 points at six UAVs; with complete links the one hop baseline is slightly better.'),
        dict(id='perception', card=dict(title='Test bench: planning on the sets halves the loss', img='fig_perception', img_box=(0.04, 0.22, 0.92, 0.55),
                                        body='1.1 points of the optimum lost with the conformal midpoint vs 2.1 with the top label.'),
             say='On an exact optimum test bench, planning on the conformal sets instead of the top label halved the loss '
                 'caused by perception: 1.1 versus 2.1 points of the optimum. In our city scenario, damage barely changed the '
                 'plan, so we do not claim this gain in flight.'),
        dict(id='bestfirst', card=dict(title='Best-first relaying: the gap grows with the fleet', img='fig_bestfirst',
                                       img_box=(0.22, 0.17, 0.56, 0.62),
                                       body='Network simulator, 30 paired seeds per point: up to +25 % over one-hop RAG at 16 UAVs, fewer messages.'),
             say='In a network simulator, we let the UAV with the largest gain decide next. This best first relaying beats '
                 'one hop coordination by 2 percent at eight UAVs and by 25 percent at sixteen UAVs on 200 metre links, '
                 'while sending fewer messages. Its weak point is that some UAVs lose the gateway path for longer.'),
        dict(id='replay', live=dict(start=20.0),   # 2x replay since 11 Nov: mission t = 40 s, as before
             say='Here is a replay of one AirSim City Environ mission, flown by PX4 software autopilots. The model read every '
                 'downward frame live on a cloud GPU. In 48 missions we saw no collision, and no false major or total damage '
                 'alarm in 9,734 frames of the intact city.'),
        dict(id='onboard', card=dict(title='Closed loop on the Jetson and the Pixhawk', img='fig_setup_preview',
                                     img_box=(0.03, 0.18, 0.55, 0.62),
                                     lines=[(0.62, 0.72, '12 HIL missions, 0 failsafes', dict(fontsize=36, color=INK)),
                                            (0.62, 0.64, 'Pixhawk 6C flies UAV 0 (props off)', dict(fontsize=26, color=SOFT)),
                                            (0.62, 0.53, 'VLM 2.2 s per frame', dict(fontsize=36, color=INK)),
                                            (0.62, 0.45, 'all 720 frames answered; 5 s period', dict(fontsize=26, color=SOFT)),
                                            (0.62, 0.34, 'int8 model alone: 1.8 s', dict(fontsize=26, color=SOFT))],
                                     body='Bench photo: Jetson not mounted on the airframe; laptop screen inserted from the replay.'),
             say='Finally, we closed the loop on flight hardware. In 12 hardware in the loop missions, a Pixhawk 6C flew one UAV '
                 'through AirSim, while the Jetson Orin Nano ran both the model and the fleet decisions. There was no failsafe, '
                 'and the model answered every frame in about 2.2 seconds, inside the five second frame period.'),
        dict(id='end', card=dict(title='Summary', dark=True, body=(
            'Calibrated VLM perception, a guarantee that carries its uncertainty into the plan,\n'
            'best-first relayed coordination, and a closed loop on a Jetson and a Pixhawk.\n'
            'Code, results and videos: github.com/abdul-mannan-khan/vlm-damage-uav')),
             say='To sum up: a compact vision language model can be a trustworthy input to UAV fleet coordination, when its '
                 'uncertainty is calibrated and carried into the guarantee. Every test was pre registered, and we report the '
                 'ones that failed. The code, results and videos are public. Thank you for watching.'),
    ]


def fmt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sysimg = WORK / 'sys.png'
    if not sysimg.exists():
        subprocess.run(['pdftoppm', '-r', '300', '-png', '-f', '3', '-l', '3', str(ROOT / 'paper/vlm_tvt/main.pdf'), str(WORK / 'p3')], check=True)
        im = Image.open(sorted(WORK.glob('p3-*.png'))[0]); W, H = im.size
        im.crop((int(W * 0.08), int(H * 0.065), int(W * 0.92), int(H * 0.205))).save(sysimg)
    clips, srt, t = [], [], 0.0
    for k, sc in enumerate(scenes()):
        wav = WORK / f'{k:02d}_{sc["id"]}.wav'; dur = tts(sc['say'], wav) + 0.9
        mp4 = WORK / f'{k:02d}_{sc["id"]}.mp4'
        if 'card' in sc:
            c = dict(sc['card']); im = c.pop('img', None)
            if im == 'sys':
                c['img'] = np.asarray(Image.open(sysimg).convert('RGB'))
            elif im == 'fig_setup_preview':
                c['img'] = np.asarray(Image.open(FIG / 'fig_setup_preview.png').convert('RGB'))
            elif im:
                c['img'] = png(im)
            pngp = WORK / f'{k:02d}_{sc["id"]}.png'; card(pngp, **c)
            vin = ['-loop', '1', '-framerate', '25', '-t', f'{dur:.2f}', '-i', str(pngp)]
        else:
            vin = ['-ss', str(sc['live']['start']), '-t', f'{dur:.2f}', '-i', str(VID)]
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *vin, '-i', str(wav), '-filter_complex',
                        '[0:v]scale=1920:1080,fps=25,format=yuv420p,fade=t=in:st=0:d=0.4[v];[1:a]apad[a]',
                        '-map', '[v]', '-map', '[a]', '-t', f'{dur:.2f}', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                        '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', str(mp4)], check=True)
        clips.append(mp4)
        srt.append(f'{k + 1}\n{fmt(t + 0.1)} --> {fmt(t + dur - 0.3)}\n{sc["say"]}\n')
        t += dur
    lst = WORK / 'list.txt'; lst.write_text(''.join(f"file '{c}'\n" for c in clips))
    out = OUT / 'vlmdmg_demo.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy',
                    '-movflags', '+faststart', str(out)], check=True)
    (OUT / 'vlmdmg_demo.srt').write_text('\n'.join(srt))
    print(json.dumps(dict(out=str(out), seconds=round(t, 1), scenes=len(clips))))


if __name__ == '__main__':
    main()
