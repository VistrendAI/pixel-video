#!/usr/bin/env python3
"""Reverse-engineering kit for pixel-art videos (the method used to learn the reference style).

  analyze.py scale   VIDEO                         integer pixel scale (edge positions mod k)
  analyze.py lo      VIDEO --k 4 [--fps 30] OUT.npy   native-grid frame stack (block centres)
  analyze.py sheet   STACK.npy --fps 30 --t0 0 --t1 10 --step 0.25 OUT.png [--zoom 2 --cols 4]
  analyze.py grid    STACK.npy --fps 30 --t 5.2 --box 0,0,160,90 OUT.png [--zoom 8]
  analyze.py palette STACK.npy --fps 30 [--k 32] [--times 1,5,9]    k-means palette of non-background pixels
  analyze.py notes   VIDEO [--t0 0 --t1 30]      audio onsets → pitch, note name, pulse-duty guess
  analyze.py motion  VIDEO [--k 6]               whole-frame shake / pan per frame (phase correlation)
  analyze.py ocr     STACK.npy --fps 30 --box x0,y0,x1,y1 [--font 57|35]   read a pixel-font label per frame
  analyze.py compare SCENE.py:Class STACK.npy --fps 60 --times 1,2,3 OUT.png   reference | replica rows

A 'stack' is the video decoded to its native pixel grid (e.g. 480×270), so every analysis works on
the real art pixels instead of the upscaled, compressed frames.
"""
import argparse
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))


def probe(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate', '-of', 'csv=p=0', video],
                         capture_output=True, text=True).stdout.strip().split(',')
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split('/')
    return w, h, float(num) / float(den)


def frames(video, fps=None, t0=None, t1=None, max_frames=None):
    w, h, vfps = probe(video)
    cmd = ['ffmpeg', '-v', 'quiet']
    if t0 is not None:
        cmd += ['-ss', str(t0)]
    cmd += ['-i', video]
    if t1 is not None:
        cmd += ['-t', str(t1 - (t0 or 0))]
    if fps:
        cmd += ['-vf', f'fps={fps}']
    if max_frames:
        cmd += ['-frames:v', str(max_frames)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def cmd_scale(a):
    """Edges of nearest-neighbour upscaled art sit on multiples of the scale → histogram of edge x mod k."""
    import itertools
    best, nread = {}, 0
    for i, f in enumerate(itertools.islice(frames(a.video, fps=2, max_frames=6), 0, 6)):
        nread += 1
        L = f.astype(int).mean(2)
        xs = np.nonzero(np.abs(np.diff(L, axis=1)) > 40)[1] + 1
        ys = np.nonzero(np.abs(np.diff(L, axis=0)) > 40)[0] + 1
        for k in range(2, 13):
            hx = np.bincount(xs % k, minlength=k)
            hy = np.bincount(ys % k, minlength=k)
            conc = (hx.max() + hy.max()) / max(1, len(xs) + len(ys))
            best[k] = best.get(k, 0) + conc
    w, h, fps = probe(a.video)
    nread = max(1, nread)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    # prefer the largest k whose concentration is ~1 (multiples of the true scale also score 1)
    full = [k for k, v in ranked if v / nread > 0.97 and w % k == 0 and h % k == 0]
    k = max(full) if full else ranked[0][0]
    print(f'{a.video}: {w}×{h} @ {fps:g} fps → pixel scale ×{k} → native {w // k}×{h // k}  ({nread} frames)')
    for kk, v in sorted(best.items()):
        print(f'   k={kk:2d}  edge concentration {v / nread:.2f}')


def cmd_lo(a):
    import json
    k = a.k
    o = k // 2
    w, h, vfps = probe(a.video)
    stack = [f[o::k, o::k].copy() for f in frames(a.video, fps=a.fps)]
    arr = np.stack(stack)
    np.save(a.out, arr)
    meta = dict(fps=float(a.fps or vfps), k=k, video=os.path.abspath(a.video), frames=len(arr))
    with open(a.out + '.json', 'w') as f:
        json.dump(meta, f)
    print(a.out, arr.shape, f'fps={meta["fps"]:g}  (sidecar {a.out}.json — later commands read the fps from it)')


def _load(stack):
    return np.load(stack, mmap_mode='r')


def _fps(a):
    """--fps if given, else the fps recorded next to the stack by `lo`."""
    import json
    if getattr(a, 'fps', None):
        return float(a.fps)
    side = a.stack + '.json'
    if os.path.exists(side):
        return float(json.load(open(side))['fps'])
    sys.exit(f'✗ {a.stack}: no sidecar {side}; pass --fps (the fps you used for `lo`)')


def _idx(arr, t, fps):
    i = int(round(t * fps))
    if i >= len(arr) or i < 0:
        print(f'  note: t={t:.2f}s is outside the stack (0–{(len(arr) - 1) / fps:.2f}s); using the nearest frame')
    return max(0, min(len(arr) - 1, i))


def cmd_sheet(a):
    from PIL import Image, ImageDraw, ImageFont
    arr = _load(a.stack)
    fps = _fps(a)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    thumbs = []
    t1 = min(a.t1, (len(arr) - 1) / fps)
    for t in np.arange(a.t0, t1 + 1e-9, a.step):
        i = _idx(arr, t, fps)
        im = Image.fromarray(np.asarray(arr[i])).resize((arr.shape[2] * a.zoom, arr.shape[1] * a.zoom), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 64, 19], fill=(0, 0, 0))
        d.text((3, 1), f'{t:5.2f}', fill=(255, 255, 0), font=font)
        thumbs.append(im)
    w, h = thumbs[0].size
    rows = math.ceil(len(thumbs) / a.cols)
    sheet = Image.new('RGB', (a.cols * w + (a.cols + 1) * 3, rows * h + (rows + 1) * 3), (90, 90, 90))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, a.cols)
        sheet.paste(im, (3 + c * (w + 3), 3 + r * (h + 3)))
    sheet.save(a.out)
    print(a.out, sheet.size)


def cmd_grid(a):
    from PIL import Image, ImageDraw, ImageFont
    arr = _load(a.stack)
    fps = _fps(a)
    x0, y0, x1, y1 = map(int, a.box.split(','))
    H, W = arr.shape[1:3]
    x0, x1, y0, y1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    f = np.asarray(arr[_idx(arr, a.t, fps)])[y0:y1, x0:x1]
    k = a.zoom
    im = Image.fromarray(f).resize(((x1 - x0) * k, (y1 - y0) * k), Image.NEAREST)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    for x in range(x0 + (-x0) % 10, x1, 10):
        X = (x - x0) * k
        d.line([(X, 0), (X, im.height)], fill=(255, 255, 0) if x % 50 == 0 else (90, 90, 0))
        if x % 20 == 0:
            d.text((X + 1, 1), str(x), fill=(255, 255, 0), font=font)
    for y in range(y0 + (-y0) % 10, y1, 10):
        Y = (y - y0) * k
        d.line([(0, Y), (im.width, Y)], fill=(0, 255, 255) if y % 50 == 0 else (0, 90, 90))
        if y % 20 == 0:
            d.text((1, Y + 1), str(y), fill=(0, 255, 255), font=font)
    im.save(a.out)
    print(a.out, im.size)


def cmd_palette(a):
    import cv2
    arr = _load(a.stack)
    fps = _fps(a)
    times = [float(x) for x in a.times.split(',')] if a.times else np.linspace(0, (len(arr) - 1) / fps, 12)
    px = np.concatenate([np.asarray(arr[_idx(arr, t, fps)]).reshape(-1, 3) for t in times]).astype(np.float32)
    bg = np.median(px, axis=0)
    fg = px[np.abs(px - bg).sum(1) > 30]
    if len(fg) > 200000:
        fg = fg[np.random.default_rng(0).choice(len(fg), 200000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, lab, cen = cv2.kmeans(fg, a.k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    d = np.sqrt(((fg - cen[lab.ravel()]) ** 2).sum(1))
    cnt = np.bincount(lab.ravel(), minlength=a.k)
    verdict = ('≈ compression noise → the video really uses about k colours' if d.mean() < 10 else
               'well above compression noise → more colours than k, or glow / gradients (try a larger k)')
    print(f'background ≈ {tuple(int(v) for v in bg)}   k={a.k}: mean residual {d.mean():.1f} ({verdict})')
    for k in np.argsort(-cnt):
        c = tuple(int(round(v)) for v in cen[k])
        print(f'  #{c[0]:02x}{c[1]:02x}{c[2]:02x}  {c}  {100 * cnt[k] / len(fg):5.1f}%')


NOTE_NAMES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def note_name(f):
    k = round(12 * math.log2(f / 440.0))
    return f'{NOTE_NAMES[k % 12]}{4 + (k + 9) // 12}'


def cmd_notes(a):
    """Onsets + FFT peaks. Harmonic pattern tells the waveform: 1/8 pulse = 0,-0.7,-1.9,-3.7 dB;
    1/4 pulse = 2nd -3 dB and no 4th; square = odd only (3rd -9.5 dB); noise = no peaks."""
    sr = 48000
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', a.video, '-ac', '1', '-ar', str(sr), '-f', 's16le', '-'],
                         capture_output=True).stdout
    x = np.frombuffer(raw, '<i2').astype(float) / 32768
    if not len(x) or np.abs(x).max() < 1e-4:
        print('silent audio track')
        return
    env = np.convolve(np.abs(x), np.ones(240) / 240, 'same')
    th = max(0.005, 0.08 * env.max())
    end = min(len(x) - int(0.05 * sr), int((a.t1 or len(x) / sr) * sr))
    i, on = min(int(a.t0 * sr), max(0, end)), []
    while i < end:
        if env[i] > th and env[max(0, i - 240)] < th * 0.5:
            on.append(i / sr)
            i += int(0.03 * sr)
        else:
            i += 48
    print(f'{len(on)} onsets')
    for t in on:
        N = int(0.03 * sr)
        seg = x[int((t + 0.01) * sr): int((t + 0.01) * sr) + N]
        if len(seg) < N:
            continue
        F = np.abs(np.fft.rfft(seg * np.hanning(N), 1 << 16))
        f = np.fft.rfftfreq(1 << 16, 1 / sr)
        m = (f > 80) & (f < 8000)
        F, f = F[m], f[m]
        idx = [j for j in range(1, len(F) - 1) if F[j] > F[j - 1] and F[j] > F[j + 1] and F[j] > F.max() * 0.2]
        pk = sorted((f[j], F[j]) for j in idx)[:5]
        if not pk:
            print(f'{t:7.3f}  noise')
            continue
        f0 = pk[0][0]
        prof = ' '.join(f'{int(fr)}({20 * np.log10(A / max(p[1] for p in pk)):+.0f})' for fr, A in pk)
        print(f'{t:7.3f}  {note_name(f0):>4s} {int(f0):5d} Hz   {prof}')


def cmd_motion(a):
    """Whole-frame motion (screen shake, pans) by phase correlation between consecutive frames.
    Per-pixel colour tracking CANNOT see a shake — it only shows up as every probe glitching on one frame."""
    import cv2
    w, h, fps = probe(a.video)
    W, H = w // 2, h // 2
    cmd = ['ffmpeg', '-v', 'quiet', '-i', a.video, '-vf', f'scale={W}:{H}:flags=area', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    prev, i, hits = None, 0, 0
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        f = np.frombuffer(b, np.uint8).reshape(H, W).astype(np.float32)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, f, win)
            if (abs(dx) > 0.6 or abs(dy) > 0.6) and resp > 0.15:
                k = a.k or 1
                print(f'{i / fps:8.3f}s  dx={dx * 2:+6.1f} dy={dy * 2:+6.1f} px'
                      + (f'  (= {dx * 2 / k:+.1f}, {dy * 2 / k:+.1f} canvas px)' if a.k else '') + f'  resp {resp:.2f}')
                hits += 1
        prev = f
        i += 1
    print(f'{hits} frames with whole-frame motion')


def cmd_ocr(a):
    from pixelkit.fonts import FONT35, FONT57
    font = FONT35 if a.font == '35' else FONT57
    glyphs = {c: font.glyph(c) for c in '0123456789'}
    arr = _load(a.stack)
    fps = _fps(a)
    x0, y0, x1, y1 = map(int, a.box.split(','))
    prev = None
    for i in range(len(arr)):
        reg = np.asarray(arr[i][y0:y1, x0:x1]).astype(int).max(2) > a.thr
        s, x, cols = '', 0, reg.any(0)
        while x < reg.shape[1]:
            if not cols[x]:
                x += 1
                continue
            best = None
            for c, g in glyphs.items():
                w = g.shape[1]
                p = reg[:g.shape[0], x:x + w]
                if p.shape != g.shape:
                    continue
                e = (p != g).sum()
                if best is None or e < best[0]:
                    best = (e, c, w)
            if best and best[0] <= 1:
                s += best[1]
                x += best[2]
            else:
                x += 1
        if s != prev:
            print(f'{i / fps:7.3f}s  {s}')
            prev = s


def cmd_compare(a):
    from PIL import Image, ImageDraw, ImageFont
    path, cls = a.scene.split(':')
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    C = getattr(m, cls)
    if a.lang:
        C.LANG = a.lang
    sc = C()
    sc.setup()
    arr = _load(a.stack)
    fps = _fps(a)
    k = a.zoom
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    rows = []
    for t in [float(x) for x in a.times.split(',')]:
        r = Image.fromarray(np.asarray(arr[_idx(arr, t, fps)])).resize(
            (arr.shape[2] * k, arr.shape[1] * k), Image.NEAREST)
        mine = Image.fromarray(np.clip(sc.frame_lo(t), 0, 255).astype(np.uint8)).resize((sc.W * k, sc.H * k), Image.NEAREST)
        row = Image.new('RGB', (r.width + mine.width + 6, max(r.height, mine.height)), (120, 120, 120))
        row.paste(r, (0, 0))
        row.paste(mine, (r.width + 6, 0))
        d = ImageDraw.Draw(row)
        d.rectangle([0, 0, 110, 16], fill=(0, 0, 0))
        d.text((2, 0), f'REF {t:5.2f}', fill=(255, 255, 0), font=font)
        d.rectangle([r.width + 6, 0, r.width + 116, 16], fill=(0, 0, 0))
        d.text((r.width + 8, 0), f'MINE {t:5.2f}', fill=(0, 255, 255), font=font)
        rows.append(row)
    sheet = Image.new('RGB', (rows[0].width, sum(r.height + 4 for r in rows)), (60, 60, 60))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    sheet.save(a.out)
    print(a.out, sheet.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('scale'); p.add_argument('video'); p.set_defaults(fn=cmd_scale)
    p = sp.add_parser('lo'); p.add_argument('video'); p.add_argument('out'); p.add_argument('--k', type=int, required=True)
    p.add_argument('--fps', type=float, default=None); p.set_defaults(fn=cmd_lo)
    p = sp.add_parser('sheet'); p.add_argument('stack'); p.add_argument('out'); p.add_argument('--fps', type=float)
    p.add_argument('--t0', type=float, default=0); p.add_argument('--t1', type=float, required=True)
    p.add_argument('--step', type=float, default=0.25); p.add_argument('--zoom', type=int, default=2)
    p.add_argument('--cols', type=int, default=4); p.set_defaults(fn=cmd_sheet)
    p = sp.add_parser('grid'); p.add_argument('stack'); p.add_argument('out'); p.add_argument('--fps', type=float)
    p.add_argument('--t', type=float, required=True); p.add_argument('--box', required=True)
    p.add_argument('--zoom', type=int, default=8); p.set_defaults(fn=cmd_grid)
    p = sp.add_parser('palette'); p.add_argument('stack'); p.add_argument('--fps', type=float)
    p.add_argument('--k', type=int, default=32); p.add_argument('--times', default=None); p.set_defaults(fn=cmd_palette)
    p = sp.add_parser('notes'); p.add_argument('video'); p.add_argument('--t0', type=float, default=0)
    p.add_argument('--t1', type=float, default=None); p.set_defaults(fn=cmd_notes)
    p = sp.add_parser('motion'); p.add_argument('video'); p.add_argument('--k', type=int, default=None)
    p.set_defaults(fn=cmd_motion)
    p = sp.add_parser('ocr'); p.add_argument('stack'); p.add_argument('--fps', type=float)
    p.add_argument('--box', required=True); p.add_argument('--font', default='35'); p.add_argument('--thr', type=int, default=55)
    p.set_defaults(fn=cmd_ocr)
    p = sp.add_parser('compare'); p.add_argument('scene'); p.add_argument('stack'); p.add_argument('out')
    p.add_argument('--fps', type=float); p.add_argument('--times', required=True)
    p.add_argument('--zoom', type=int, default=2); p.add_argument('--lang'); p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
