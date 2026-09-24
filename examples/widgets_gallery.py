"""Every reusable part on one v2 frame — the visual index of pixelkit/widgets.py (and a smoke test).

    ~/.claude/skills/pixel-video/pxv still widgets_gallery.py --t 3 --full --out gallery.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pixelkit import PHOSPHOR, Scene, particles, pixelate, NEON32, patch_grid, seg  # noqa: E402
from pixelkit import chrome  # noqa: E402
from pixelkit.fonts import FONT35  # noqa: E402
from pixelkit.widgets import (arrow, block, bracket, cup, hbars, hist_fill, label_box, layer_stack,  # noqa: E402
                              line_plot, link, matrix, mlp_symbol, node, rbox, robot_arm, stopgrad, table,
                              tile_stack, token_grid, vec_cells)

P = PHOSPHOR
TITLE = 'PIXELKIT PARTS'


class Gallery(Scene):
    W, H, SCALE = 480, 270, 4
    PAL = PHOSPHOR
    BLOOM = dict(strength=0.5, sigma=2.6, threshold=20)
    DURATION = 6.0
    TITLE = TITLE

    def setup(self):
        rng = np.random.default_rng(0)
        self.stages = chrome.Stages([dict(name='PARTS', t=0.0, status='WIDGETS.PY',
                                          cap=('EVERY PART, DRAWN ONCE', 'LABELS UNDER EACH PART ARE THEIR FUNCTION NAMES'))])
        self.vec = rng.normal(size=8)
        self.M = rng.normal(size=(6, 6))
        self.curves = [np.cumsum(rng.normal(size=20)) for _ in range(3)]
        self.probs = np.exp(rng.normal(size=6) * 1.5)
        self.probs /= self.probs.sum()
        gx, gy = np.meshgrid(np.linspace(0, 255, 64), np.linspace(255, 0, 64)); grad = np.dstack([gx, gy, np.full((64, 64), 128.0)])
        self.tiles = [pixelate(np.roll(grad, 16 * i, 1), 24, 24, NEON32) for i in range(4)]
        self.state = (rng.random((6, 6)) > 0.45).astype(int)

    def cap(self, cv, x, y, s):
        cv.text(x, y, s, P['label'], font=FONT35)

    def draw(self, cv, t):
        chrome.background(cv)
        u = seg(t, 0.3, 2.5)
        # row 1
        layer_stack(cv, 14, 36, 34, 24, P['blue'], P['blue_dk'], lit=u, pitch=2, taps=(6, 12, 18, 24))
        self.cap(cv, 14, 90, 'LAYER_STACK')
        mlp_symbol(cv, 84, 52, 14, 16, P['blue'])
        self.cap(cv, 70, 98, 'MLP_SYMBOL')
        node(cv, 120, 44, P['dim'])
        node(cv, 120, 62, P['blue'], fill=(20, 40, 110))
        self.cap(cv, 112, 90, 'NODE')
        vec_cells(cv, 150, 34, self.vec, P['orange'], (40, 48, 84), cell=5, gap=1, vmax=2.0)
        self.cap(cv, 142, 90, 'VEC_CELLS')
        matrix(cv, 190, 36, self.M, 6, (30, 30, 44), P['yellow'], mask=np.tril(np.ones((6, 6), bool)))
        self.cap(cv, 190, 90, 'MATRIX')
        token_grid(cv, 246, 36, 6, 6, 5, P['cream'], state=self.state, masked_color=P['orange_md'])
        self.cap(cv, 244, 90, 'TOKEN_GRID')
        tile_stack(cv, self.tiles, 300, 44, dx=3, dy=-3, border=P['dim2'])
        patch_grid(cv, 300, 44, 24, 24, 6, P['bg'])
        self.cap(cv, 298, 90, 'TILE_STACK')
        hbars(cv, 350, 38, self.probs, 30, P['blue'], pitch=7, hot=int(np.argmax(self.probs)), hot_color=P['text'])
        self.cap(cv, 348, 90, 'HBARS')
        robot_arm(cv, (412, 76), [1.25 - 0.25 * u, -1.55, -0.95], [22, 20, 7], P['text'], width=3, grip=1 - u)
        table(cv, 398, 470, 77, P['tan_dk'])
        cup(cv, 452, 76, P['cream'])
        block(cv, 440, 76, P['orange'])
        self.cap(cv, 402, 90, 'ROBOT_ARM')
        # row 2
        line_plot(cv, 16, 110, 90, 40, self.curves, [P['orange'], P['blue'], P['cream']], u=u, axis=P['dim2'])
        self.cap(cv, 14, 158, 'LINE_PLOT')
        x = np.linspace(0, 1, 40)
        hist_fill(cv, 124, 112, 80, 36, x ** 0.5 * (1 - x) ** 0 + 0.02, P['yellow'], marker=u, marker_color=P['text'])
        self.cap(cv, 124, 158, 'HIST_FILL')
        w = label_box(cv, 222, 112, 'LABEL_BOX', P['text'], border=P['blue'])
        rbox(cv, 222, 132, w, 16, P['orange'], dash=(1, 1))
        self.cap(cv, 222, 158, 'RBOX DASH')
        arrow(cv, 290, 120, 330, 120, P['cream'], frac=u)
        p = link(cv, (290, 132), (340, 146), P['wire'], dash=None)
        particles(cv, p, t, 0.2, 1.0, n=6, gap=0.25, color=P['text'], trail=3, trail_color=P['blue'])
        self.cap(cv, 290, 158, 'ARROW LINK')
        cv.hline(356, 400, 124, P['cream'], dash=(2, 1))
        stopgrad(cv, 376, 124, P['cream'])
        bracket(cv, 412, 112, 148, P['orange'], side='r')
        self.cap(cv, 356, 158, 'STOPGRAD BRACKET')
        # row 3: text
        cv.text(16, 176, 'FONT57  π_{0.5}  x_{t}  √d  × → · …  0.99925', P['text'])
        cv.text(16, 190, 'Lowercase too: put the marker in the cup', P['text2'])
        cv.text(16, 204, 'FONT35 0123456789 ABC', P['label'], font=FONT35)
        self.stages.draw(cv, t, TITLE)
