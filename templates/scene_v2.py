"""{NAME} — pixel explainer in the v2 'phosphor' style (480×270 ×4: header + progress squares,
two-line typewriter captions, dot grid, bloom). English captions.

COMPOSITION (the reference rule): the whole system map — every module, wire and small size label — is on
screen, dim, from t≈0.3 s. Each stage only LIGHTS UP its part (module 'active' → 'done', wires carry
particles), and fills it with real data. Nothing appears out of an empty frame except the data itself.

    python3 prep.py                                   # real numbers → data/data.npz
    PXV=~/.claude/skills/pixel-video/pxv
    $PXV stills scene.py --times 0.5,3,6,9            # layout at a glance
    $PXV check  scene.py                              # copy / glyphs / header / bands / text overlaps
    $PXV render scene.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/.claude/skills/pixel-video'))
from pixelkit import PHOSPHOR, Canvas, Path, Scene, ghost_layer, reveal_layer, seg  # noqa: E402
from pixelkit import chrome  # noqa: E402
from pixelkit.widgets import label_box, layer_stack, module, vec_cells, wire  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = PHOSPHOR
TITLE = 'MODEL'

# ── storyboard: one dict per stage (see SKILL.md §0.3). status = the equation / size of this step
#    (str, [(t, str), …], or a function of t for live counters); cap = (what happens, hard facts + caveats).
T = [0.0, 3.5, 7.0, 11.0]        # stage start times
DURATION = 14.0
STAGES = {
    'en': [
        dict(name='INPUT', t=T[0], status='6 TOKENS · d = 8',
             cap=('WHAT THE MODEL DOES, IN ONE SENTENCE', 'TOY SIZES FROM FACTS.MD; REAL MATH, ILLUSTRATIVE WORDS')),
        dict(name='EMBED', t=T[1], status='6 × 8 → 6 × 8', cap=('WHAT THIS STEP DOES', 'THE NUMBERS THAT MAKE IT CONCRETE')),
        dict(name='MODEL', t=T[2], status=[(T[2], 'BLOCK 1 / 12'), (T[2] + 2, 'BLOCK 12 / 12')],
             cap=('THE CORE MECHANISM (GIVE IT TWO STAGES)', 'WHAT IS REAL AND WHAT IS ILLUSTRATIVE')),
        dict(name='OUTPUT', t=T[3], status='6 × 8', cap=('THE RESULT', 'SOURCE: PAPER TABLE / FIGURE')),
    ],
}

# ── the system map (canvas px). key: (x, y, w, h, label, size label)
MODULES = {
    'input': (16, 44, 52, 70, 'INPUT', '6 × 8'),
    'embed': (96, 44, 56, 70, 'EMBED', 'W: 8 × 8'),
    'model': (182, 36, 96, 86, 'MODEL', '12 BLOCKS · 8-D'),
    'out': (308, 44, 60, 70, 'OUTPUT', '6 × 8'),
}
WIRES = [('input', 'embed'), ('embed', 'model'), ('model', 'out')]
ACTIVE = {'input': 0, 'embed': 1, 'model': 2, 'out': 3}      # module → the stage that runs it


def port(key, side):
    x, y, w, h = MODULES[key][:4]
    return (x + w, y + h // 2) if side == 'out' else (x - 1, y + h // 2)


class Explainer(Scene):
    W, H, SCALE = 480, 270, 4
    FPS = 60
    PAL = PHOSPHOR
    BLOOM = dict(strength=0.5, sigma=2.6, threshold=20)   # fitted to the V-JEPA 2.1 reference
    DURATION = DURATION
    LANG = 'en'
    TITLE = TITLE

    def setup(self):
        self.stages = chrome.Stages(STAGES[self.LANG], lang=self.LANG)   # pxv check / render check the copy
        f = os.path.join(HERE, 'data', 'data.npz')
        self.d = dict(np.load(f)) if os.path.exists(f) else {}
        self.x = self.d.get('x', np.random.default_rng(0).normal(size=(6, 8)))
        self.paths = {(a, b): Path([port(a, 'out'), port(b, 'in')]) for a, b in WIRES}

    def state(self, key, t):
        cur, _ = self.stages.at(t)
        k = ACTIVE[key]
        return 'active' if cur == k else ('done' if cur > k else 'idle')

    def draw_map(self, cv, t):
        """Every module + wire + size label, in its current state."""
        for key, (x, y, w, h, label, size) in MODULES.items():
            module(cv, x, y, w, h, self.state(key, t), P, label=label, sub=size, font=self.lf)
        for (a, b), p in self.paths.items():
            t0 = T[ACTIVE[b]] + 0.2                                    # data flows in when b's stage starts
            wire(cv, p, t, t0, 0.45, P)

    def draw(self, cv, t):
        chrome.background(cv)
        # the map: a checker ghost for 0.3 s (boot), then solid for the whole video
        if t >= 0.05:
            lay = cv.layer()
            self.draw_map(lay, t)
            if t < 0.3:
                ghost_layer(cv, lay, alpha=0.8)
            else:
                cv.composite(lay)
        # ── data inside the active parts (real values from prep.py)
        if t >= T[0] + 0.4:
            lay = cv.layer()
            for i in range(6):
                vec_cells(lay, 22 + 7 * i, 58, self.x[i], P['cream'], (40, 48, 84), cell=5, gap=1, vmax=2.5)
            reveal_layer(cv, lay, seg(t, T[0] + 0.4, T[0] + 1.0), seed=1, sparkle_color=P['text'])
        if t >= T[2]:
            layer_stack(cv, 196, 58, 50, 12, P['fwd'], P['blue_dk'], lit=seg(t, T[2] + 0.3, T[2] + 3.0), pitch=6)
        if t >= T[3] + 0.3:
            label_box(cv, 314, 76, '6 × 8', P['text'], border=P['fwd'], font=self.lf)
        self.stages.draw(cv, t, TITLE, P)             # chrome on top, always last

    def sfx(self):
        # the v2 references are SILENT. To add 8-bit sound, return events here (see SKILL.md §3), e.g.
        #   n = sfx.note; return [(T[1], sfx.blip(n('E6')), 0.6), (T[3], sfx.chime(), 0.7)]
        return []
