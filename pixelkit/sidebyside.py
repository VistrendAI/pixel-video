"""Side-by-side A/B card: reference | replica, each area-downscaled by exactly 2 (1920×1080 → 960×540, so every
native pixel stays a crisp block) on a 1920×1080 card, labels drawn with the pixel font.

    python3 -m pixelkit.sidebyside ref.mp4 mine.mp4 out.mp4 [--title T] [--left L] [--right R] [--audio mine|ref|none]

The card lasts as long as the SHORTER input. --audio picks whose soundtrack plays (silently skipped if that
input has none).
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

from .pk import Canvas, upscale


def label_card(title, left, right, bg=(9, 11, 25)):
    """1920×1080 RGBA overlay drawn on a 480×270 pixel canvas (×4), transparent where the videos go."""
    cv = Canvas(480, 270, bg)
    cv.text(240, 22, title, (240, 242, 249), align='c')
    cv.text(120, 48, left, (114, 121, 160), align='c')
    cv.text(360, 48, right, (237, 104, 68), align='c')
    cv.hline(8, 471, 62, (52, 60, 96), dash=(1, 1))
    cv.hline(8, 471, 202, (52, 60, 96), dash=(1, 1))
    rgb = upscale(cv.a, 4)
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    alpha[270:810, 0:1920] = 0
    return np.dstack([rgb, alpha])


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                          '-of', 'default=nw=1', path], capture_output=True, text=True).stdout
    dur = float(next((l.split('=')[1] for l in out.splitlines() if l.startswith('duration=')), 0) or 0)
    return dur, 'codec_type=audio' in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('mine')
    ap.add_argument('out')
    ap.add_argument('--title', default='REFERENCE  VS  REPLICA')
    ap.add_argument('--left', default='REFERENCE')
    ap.add_argument('--right', default='PIXELKIT REPLICA')
    ap.add_argument('--audio', default='mine', choices=['mine', 'ref', 'none'])
    a = ap.parse_args()
    d0, a0 = probe(a.ref)
    d1, a1 = probe(a.mine)
    dur = min(d0, d1)
    card = label_card(a.title, a.left, a.right)
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, 'card.png')
    Image.fromarray(card, 'RGBA').save(png)
    fc = ('[0:v]scale=960:540:flags=area,setsar=1[l];[1:v]scale=960:540:flags=area,setsar=1[r];'
          '[2:v]format=rgba[c];'
          'color=c=0x090b19:s=1920x1080:r=60[bg];[bg][l]overlay=0:270[b1];[b1][r]overlay=960:270[b2];'
          '[b2][c]overlay=0:0,format=yuv420p[v]')
    maps = ['-map', '[v]']
    if a.audio == 'mine' and a1:
        maps += ['-map', '1:a']
    elif a.audio == 'ref' and a0:
        maps += ['-map', '0:a']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.ref, '-i', a.mine, '-loop', '1', '-i', png,
                    '-filter_complex', fc] + maps + ['-t', f'{dur:.3f}', '-c:v', 'libx264', '-crf', '16',
                                                     '-preset', 'medium', '-c:a', 'aac', '-b:a', '192k',
                                                     '-movflags', '+faststart', a.out], check=True)
    print('✓', a.out, f'{dur:.2f}s')


if __name__ == '__main__':
    main()
