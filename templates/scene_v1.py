"""<TITLE> — pixel mechanism loop in the v1 'neon' style (320×180 ×6, no text beyond 3×5 numerals,
8-bit SFX, seamless loop: the last frame dissolves back to the first).

    ~/.claude/skills/pixel-video/pxv stills THIS.py:Loop --times 0.2,1,2,3
    ~/.claude/skills/pixel-video/pxv render THIS.py:Loop
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/.claude/skills/pixel-video'))
from pixelkit import (FONT35, NEON, NEON32, Path, Scene, ghost_layer, materialize, particles,  # noqa: E402,F401
                      quantize, seg, sfx)
from pixelkit.widgets import rbox  # noqa: E402

P = NEON
BOOT = (0.05, 0.4)          # dithered ghost → solid
OUTRO = (5.4, 5.8)          # dissolve back to the empty layout (loop point)
DURATION = 6.0


class Loop(Scene):
    W, H, SCALE = 320, 180, 6
    FPS = 60
    PAL = NEON
    DURATION = DURATION

    def setup(self):
        rng = np.random.default_rng(0)
        # every image goes through the 32-colour palette, so data and UI share one look
        self.img = quantize(rng.uniform(0, 255, (32, 32, 3)), NEON32)
        self.wire = Path([(60, 90), (140, 90), (140, 60), (220, 60)])

    def layout(self, cv):
        rbox(cv, 20, 74, 34, 34, (50, 70, 123), fill=(8, 12, 27))
        rbox(cv, 226, 44, 34, 34, (50, 70, 123), fill=(8, 12, 27))
        cv.path(self.wire, P['wire'], dash=(1, 1))
        cv.text(37, 112, '0', (47, 66, 113), font=FONT35, align='c')

    def draw(self, cv, t):
        if t < BOOT[0]:
            return
        if seg(t, *BOOT) < 0.55:
            lay = cv.layer()
            self.layout(lay)
            ghost_layer(cv, lay, alpha=0.7)
            return
        self.layout(cv)
        if t < OUTRO[0]:
            materialize(cv, self.img, 21, 75, seg(t, 0.5, 0.9), seed=1)
        else:
            materialize(cv, self.img, 21, 75, seg(t, *OUTRO), seed=2, mode='out', sparkle=0)
        if 1.2 <= t < 2.2:
            cv.path(self.wire, P['cyan'], u1=seg(t, 1.2, 1.6))
        particles(cv, self.wire, t, 1.2, 0.5, n=3, gap=0.06, color=P['white'], trail=2, trail_color=P['cyan'])

    def sfx(self):
        n = sfx.note
        return [(0.45, sfx.boot(0.3), 0.7), (1.2, sfx.pew(n('A#4')), 0.7), (1.7, sfx.chime(), 0.7)]
