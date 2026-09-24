"""Replica of the reference 'diffusion U-Net' pixel video (OjpcnMY_mBfwLpc3.mp4, 21.5 s).

Story (all timings measured from the reference):
  0.0–0.9   boot: layout appears as a dithered ghost, then solid; x0 sparkles into box 0
  0.9–3.3   forward process: T ramps 0→1000, each chain box fills with q(x_t|x0) every 0.4 s
  3.8–4.3   training step: random t — slot-machine ticks settle on T=380
  4.4–5.0   x_t = √ᾱ·x0 + √(1-ᾱ)·ε is sent down into the U-Net input
  5.0–7.0   U-Net forward pass (blocks fire left→right, skips + time embedding light up)
  7.0–7.4   predicted ε̂ vs true ε → loss ◆
  7.5–9.0   backward pass (blocks go red right→left), 9.0–9.3 weight-update flash
  9.5–10    reset; only x_1000 stays
 10.0–18.6  sampling: 12 DDIM steps 1000→0 with the (oracle) denoiser, x_{t-1} = x_t ⊖ ε̂,
            every second step lands in the chain box
 18.6–19.4  x0 recovered — white flash + chime; hold; 21.0 dissolve back to the start (loops)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pixelkit import (FONT35, NEON, NEON32, Canvas, Path, Scene, clamp, ease_out, materialize,  # noqa: E402
                      mixc, particles, quantize, render, seg, sfx, shake_offset, stills)

P = NEON
BOX_BORDER = (50, 70, 123)
BOX_FILL = (8, 12, 27)
WIRE = (27, 35, 65)
BAR_OUT = (53, 67, 119)
BAR_IN = (16, 24, 44)
LABEL = (47, 66, 113)
ARROW = (48, 64, 116)
BLUE = (16, 55, 180)
BLUE_IN = (9, 24, 80)
BLUE_HI = (30, 130, 252)
CYAN = (85, 211, 213)
SKY = (32, 177, 251)
WHITE = (240, 240, 248)
PINK = (253, 90, 172)
HOT = (252, 28, 118)
RED_OUT = (123, 9, 44)
RED_IN = (57, 6, 18)
YEL = (252, 192, 28)
YEL_DK = (137, 98, 32)
YEL_PALE = (253, 239, 169)

# ─────────────────────────────────────── layout (measured on the reference, 320×180)
CHAIN_X = [11 + 44 * i for i in range(7)]
CHAIN_Y = 3
CHAIN_T = [0, 167, 333, 500, 667, 833, 1000]
IN_BOX = (13, 83)
OUT_BOX = (257, 83)
EPS_BOX = (257, 131)
T_BOX = (13, 137, 34, 16)
# blocks: (x of first bar, n bars, top y, height)
BLOCKS = [(58, 2, 80, 40), (84, 3, 100, 28), (112, 4, 118, 18), (142, 6, 134, 12),
          (178, 4, 118, 18), (204, 3, 100, 28), (232, 2, 80, 40)]
NB = len(BLOCKS)


def bcy(b):
    return b[2] + b[3] // 2


def bleft(b):
    return b[0] - 1


def bright(b):
    return b[0] + 4 * b[1] - 1


def bcx(b):
    return b[0] + (4 * b[1] - 1) // 2


# connections along the U: input → b0 → … → b6 → output
CONN = [Path([(47, 100), (bleft(BLOCKS[0]), 100)])]
for i in range(NB - 1):
    a, b = BLOCKS[i], BLOCKS[i + 1]
    CONN.append(Path([(bright(a), bcy(a)), (bleft(b), bcy(b))]))
CONN.append(Path([(bright(BLOCKS[-1]), 100), (256, 100)]))
# skips (encoder i → decoder 6-i), drawn as dotted rectangles above the blocks
SKIP = [Path([(61, 79), (61, 74), (236, 74), (236, 79)]),
        Path([(89, 99), (89, 96), (209, 96), (209, 99)]),
        Path([(119, 117), (119, 114), (185, 114), (185, 117)])]
# time-embedding bus: box → down → along y=168 → up into every block
T_TRUNK = Path([(47, 145), (50, 145), (50, 168), (235, 168)])
T_DROPS = [Path([(bcx(b), 168), (bcx(b), b[2] + b[3] + 1)]) for b in BLOCKS]
# buses
BUS_A = Path([(20, 58), (291, 58)])
DROPS_A = [Path([(x + 16, 45), (x + 16, 57)]) for x in CHAIN_X]
FEED_A = Path([(20, 59), (20, 82)])                      # bus A → input box
X0_FEED = Path([(27, 37), (27, 82)])                      # chain box 0 → input box
BUS_B = Path([(30, 64), (274, 64)])
FEED_B_L = Path([(30, 65), (30, 82)])
FEED_B_R = Path([(274, 65), (274, 82)])
MINUS = (150, 64)
DIAMOND = (301, 124)
LOSS_A = Path([(291, 104), (299, 122)])
LOSS_B = Path([(291, 144), (299, 126)])

# ─────────────────────────────────────── timeline (seconds, measured on the reference)
BOOT = (0.08, 0.42)                     # ghost → solid
X0_IN = (0.45, 0.85)
FWD0, FWD_DT = 0.883, 0.4               # T ramps 0→1000 over 2.4 s; box k fills at FWD0 + (k-1)·0.4
SLOT = [(3.8, 741), (3.85, 647), (3.917, 931), (3.983, 819), (4.05, 848), (4.117, 448), (4.183, 136),
        (4.25, 295), (4.283, 380)]
T_TRAIN = 380
XT_FEED = (4.42, 4.72)
XT_IN = (4.62, 4.95)
TEMB = (5.08, 5.66)                     # yellow front sweeps the time bus (full at 5.7)
TEMB_OFF = 6.07
F_ACT = [5.18 + 0.255 * i for i in range(7)]    # forward: block i fires (sound onsets)
OUT_IN = (6.85, 7.15)
EPS_IN = (7.0, 7.3)
LOSS_T = 7.25
TBOX_TRAIN = (3.78, 6.9)
B_ACT = [7.52 + 0.205 * r for r in range(7)]    # backward: block 6-r fires
UPD = [9.0 + 0.045 * i for i in range(7)]       # weight-update sweep
RESET = (9.45, 9.85)
XT_IN_S = (9.8, 10.05)                  # x_1000 drops into the input box
S_T = [11.367, 11.9, 12.933, 13.45, 14.367, 14.9, 15.667, 16.2, 16.883, 17.42, 18.083, 18.583]
S_VAL = [1000, 920, 833, 753, 667, 587, 500, 420, 333, 253, 167, 87, 0]
P_START = [10.35, 12.017, 13.567, 14.983, 16.283, 17.5]   # enc1 turns cyan (measured) — landing + ~0.1 s
S_START = P_START[0]
MINUS_LIT = (10.23, 18.97)             # ⊖ glows cyan through the whole sampling phase
FINAL = (18.9, 19.75)                   # white frames on the recovered x0
BLOCKS_OFF = 19.45
OUTRO = (20.82, 21.28)
DURATION = 21.5


# ─────────────────────────────────────── the pixel-art sunset (32×32)
def sunset32():
    img = np.zeros((32, 32, 3), np.float32)
    sky = [(34, 20, 52)] * 3 + [(52, 28, 76)] * 2 + [(80, 40, 100)] * 2 + [(124, 52, 110)] * 2 + \
          [(178, 64, 108)] * 2 + [(226, 100, 96)] * 2 + [(246, 152, 92)] * 2 + [(252, 196, 118)] * 2 + [(254, 226, 160)]
    for y, c in enumerate(sky):
        img[y, :] = c
    # sun (upper disc, sitting on the horizon at y=18)
    cy, cx = 18, 24
    for y in range(9, 18):
        for x in range(16, 32):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= 7.2:
                img[y, x] = (255, 244, 196) if d < 5.2 else (255, 226, 150)
    # cloud / dune on the left
    dome = [(10, 7, 12), (11, 5, 14), (12, 4, 15), (13, 3, 16), (14, 3, 17), (15, 2, 17), (16, 2, 18), (17, 1, 18)]
    for y, x0, x1 in dome:
        img[y, x0:x1] = (196, 150, 92)
        img[y, x0] = (70, 34, 26)
        img[y, x1 - 1] = (70, 34, 26)
    img[9, 8:11] = (70, 34, 26)
    img[10, 6:8] = (70, 34, 26)
    img[10, 11:13] = (70, 34, 26)
    for (y, x) in [(12, 7), (13, 9), (14, 6), (15, 11), (13, 13), (16, 8), (14, 12)]:
        img[y, x] = (150, 102, 60)
    # horizon line
    img[18, :] = (250, 208, 130)
    img[18, 1:17] = (244, 238, 228)
    img[19, 1:17] = (222, 58, 46)
    img[19, 17:] = (238, 150, 84)
    # sea
    sea = [(20, 44, 72), (22, 58, 92), (24, 76, 108), (22, 62, 96), (20, 50, 84), (18, 44, 76),
           (18, 42, 72), (16, 38, 66), (18, 44, 76), (16, 36, 62), (16, 34, 60), (14, 32, 56)]
    for i, c in enumerate(sea):
        img[20 + i, :] = c
    # waves / foam
    for x in range(8, 17):
        img[20, x] = (250, 250, 250) if x in (9, 14) else (84, 200, 196)
    for x in range(2, 24):
        if (x // 3) % 2 == 0:
            img[21, x] = (70, 180, 184)
    for x in range(0, 32):
        if (x + 2) % 7 < 3:
            img[23, x] = (44, 140, 150)
        if (x + 5) % 9 < 4:
            img[26, x] = (36, 110, 132)
        if (x + 1) % 11 < 3:
            img[29, x] = (30, 92, 118)
    # sun glitter (orange checker under the sun)
    for y in range(20, 25):
        for x in range(21, 27):
            if (x + y) % 2 == 0 and abs(x - 23.5) <= 2.5 - (y - 20) * 0.35:
                img[y, x] = (250, 160, 80)
    return img


# ─────────────────────────────────────── diffusion maths (linear β schedule, DDPM)
# the reference's chain fits Stable Diffusion's scaled-linear schedule (max err 0.05 on √ᾱ),
# with the displayed noise at 0.6·N(0, 1) so the image survives visibly up to t≈500
BETAS = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
ABAR = np.cumprod(1 - BETAS)
NOISE = 0.6


def abar(t):
    return 1.0 if t <= 0 else float(ABAR[min(999, int(t) - 1)])


def to_disp(x):
    """[-1, 1] → display pixels, snapped to the 32-colour palette like every image in the reference"""
    return quantize(np.clip((x + 1) * 127.5, 0, 255), NEON32)


def to_norm(img):
    return img / 127.5 - 1


def lerp_t(a, b, u):
    return a + (b - a) * u


def blinky(t, hz=8.0):
    return (t * hz) % 1.0 < 0.5


class DiffusionUNet(Scene):
    W, H, SCALE = 320, 180, 6
    FPS = 60
    DURATION = DURATION
    PAL = NEON

    def setup(self):
        rng = np.random.default_rng(20260923)
        self.x0 = quantize(sunset32(), NEON32)
        x0n = to_norm(self.x0)
        self.chain = [self.x0]
        self.eps_chain = [None] + [rng.standard_normal((32, 32, 3)) for _ in range(6)]
        for k in range(1, 7):
            t = CHAIN_T[k]
            self.chain.append(to_disp(np.sqrt(abar(t)) * x0n + np.sqrt(1 - abar(t)) * NOISE * self.eps_chain[k]))
        # training example: x_t for the sampled t, the true ε and the network's (imperfect) ε̂
        self.eps_train = rng.standard_normal((32, 32, 3))
        a = abar(T_TRAIN)
        self.xt_train = to_disp(np.sqrt(a) * x0n + np.sqrt(1 - a) * NOISE * self.eps_train)
        self.eps_hat_train = to_disp(NOISE * (0.9 * self.eps_train + 0.45 * rng.standard_normal((32, 32, 3))))
        self.eps_disp = to_disp(NOISE * self.eps_train)
        # sampling: DDIM (η = 0) with an oracle ε̂, starting from the chain's own x_1000
        eps_T = self.eps_chain[6]
        self.samp, self.samp_eps = [], []
        for v in S_VAL:
            a = abar(v)
            self.samp.append(to_disp(np.sqrt(a) * x0n + np.sqrt(1 - a) * NOISE * eps_T))
            self.samp_eps.append(to_disp(NOISE * (eps_T + 0.25 * rng.standard_normal((32, 32, 3)) * (v / 1000))))
        # per-bar activation patterns (interior column brightness), one set per pass
        self.act = {}
        for key in range(40):
            r = np.random.default_rng(1000 + key)
            self.act[key] = [r.random((b[3] - 2, b[1])) for b in BLOCKS]
        # sampling: one animated U-Net pass per chain box (measured). Pass k starts 0.13 s after the
        # previous landing, fires the 7 blocks at an even spacing, ends at S_T[2k] (ε̂ out, T changes);
        # S_T[2k+1] is the landing: x_t ⊖ ε̂ on bus B, the result drops into the chain box.
        self.passes = []
        for k in range(6):
            start, end = P_START[k], S_T[2 * k]
            self.passes.append((start, end, (end - start - 0.1) / 6.0))

    # ─────────────────────────── time → state
    def T_value(self, t):
        if t < 0.5:
            return None
        if t < 3.8:
            return int(round(clamp((t - FWD0) * 1000 / 2.4, 0, 1000))) if t >= FWD0 else 0
        if t < 9.583:
            v = SLOT[0][1]
            for (ts, val) in SLOT:
                if t >= ts:
                    v = val
            return v
        v = S_VAL[0]
        for i, ts in enumerate(S_T):
            if t >= ts:
                v = S_VAL[i + 1]
        return v

    def cur_pass(self, t):
        """(k, start, end, spacing) of the sampling pass whose window (start → landing) contains t"""
        for k, (a, b, sp) in enumerate(self.passes):
            if a - 0.1 <= t < S_T[2 * k + 1]:
                return k, a, b, sp
        return None

    def pass_act(self, k, i):
        """time block i turns cyan in sampling pass k"""
        a, b, sp = self.passes[k]
        return a + sp * i

    def recv_time(self, i):
        """time chain box i (0..5) receives its sampled image; box 6 'receives' at the start."""
        if i >= 6:
            return XT_IN_S[0]
        return S_T[2 * (6 - i) - 1]

    # ─────────────────────────── primitives of this diagram
    def box(self, cv, x, y, w=34, h=34, border=BOX_BORDER, fill=BOX_FILL, ghost=False):
        if ghost:
            cv.checker(x + 1, y + 1, w - 2, h - 2, (22, 28, 58))
            cv.rect(x, y, w, h, (46, 62, 118), dash=(2, 1))
            return
        if fill is not None:
            cv.fill(x + 1, y + 1, w - 2, h - 2, fill)
        cv.hline(x + 1, x + w - 2, y, border)
        cv.hline(x + 1, x + w - 2, y + h - 1, border)
        cv.vline(x, y + 1, y + h - 2, border)
        cv.vline(x + w - 1, y + 1, y + h - 2, border)

    # bar looks, measured on the reference: (outline, interior, bright dot, mid dot, filled?)
    LOOK = {
        'idle': (BAR_OUT, BAR_IN, None, None, False),
        'fcur': (CYAN, (40, 120, 150), WHITE, CYAN, True),
        'fsky': (SKY, (20, 90, 170), WHITE, SKY, True),
        'fdone': (BLUE, (14, 40, 140), CYAN, BLUE_HI, False),
        'bpink': (PINK, (150, 40, 90), WHITE, PINK, True),
        'bhot': (HOT, (140, 10, 60), WHITE, PINK, True),
        'bdone': (RED_OUT, RED_IN, HOT, (160, 20, 64), False),
        'flash': (WHITE, (40, 44, 70), None, None, False),
        'post_sky': (SKY, (20, 50, 90), None, None, False),
        'post_hot': (HOT, (60, 10, 30), None, None, False),
    }

    def bars(self, cv, b, state, pat=None, t=0.0, ghost=False):
        x0, n, y0, h = b
        if ghost:
            for j in range(n):
                cv.checker(x0 + 4 * j, y0, 3, h, (46, 62, 118), parity=j)
            return
        out, inn, hi, lo, filled = self.LOOK[state]
        for j in range(n):
            x = x0 + 4 * j
            if filled:
                cv.fill(x, y0, 3, h, out)
            cv.rect(x, y0, 3, h, out)
            cv.vline(x + 1, y0 + 1, y0 + h - 2, inn)
            if pat is not None and hi is not None:
                col = pat[:, j]
                v = col + 0.12 * np.sin(t * 23 + np.arange(len(col)) * 1.7 + j)
                ys = np.arange(len(col)) + y0 + 1
                k1 = v > 0.78
                k2 = (v > 0.55) & ~k1
                cv.put(np.full(k1.sum(), x + 1), ys[k1], hi)
                cv.put(np.full(k2.sum(), x + 1), ys[k2], lo)
                if filled:
                    k3 = v > 0.35
                    cv.put(np.full(k3.sum(), x), ys[k3], hi, 0.6)
                    cv.put(np.full(k3.sum(), x + 2), ys[k3], hi, 0.6)

    def arrow(self, cv, x0, x1, y, c, right=True):
        cv.hline(x0, x1, y, c)
        if right:
            cv.put([x1 - 1, x1 - 2, x1 - 1, x1 - 2], [y - 1, y - 2, y + 1, y + 2], c)
        else:
            cv.put([x0 + 1, x0 + 2, x0 + 1, x0 + 2], [y - 1, y - 2, y + 1, y + 2], c)

    def tbox(self, cv, active, value, ghost=False):
        x, y, w, h = T_BOX
        border = YEL_DK if active else BOX_BORDER
        if ghost:
            cv.rect(x, y, w, h, (46, 62, 118), dash=(2, 1))
        else:
            self.box(cv, x, y, w, h, border=border, fill=None)
        tv = 0 if value is None else value
        # four sinusoid 'channels' of the time embedding, drawn as stepped 1-px lines
        for k in range(4):
            freq = [0.09, 0.18, 0.33, 0.6][k]
            ph = tv * [0.004, 0.011, 0.023, 0.05][k]
            base = y + 3 + k * 3
            prev = None
            for i in range(w - 2):
                yy = base + int(round(1.2 * np.sin(i * freq + ph)))
                c = (YEL_PALE if (i // 6 + k) % 2 == 0 else YEL) if active else (38, 52, 96)
                cv.px(x + 1 + i, yy, c, 0.6 if ghost else 1.0)
                if prev is not None and abs(prev - yy) > 1:
                    cv.px(x + 1 + i, (prev + yy) // 2, c, 0.6 if ghost else 1.0)
                prev = yy
        if value is not None and not ghost:
            col = YEL_PALE if active else LABEL
            cv.text(x + 1, y + h + 2, 'T', col, font=FONT35)
            cv.text(x + 7, y + h + 2, str(value), col, font=FONT35)

    def minus_node(self, cv, lit):
        c = CYAN if lit else BOX_BORDER
        cv.circle(MINUS[0], MINUS[1], 3, c)
        cv.hline(MINUS[0] - 1, MINUS[0] + 1, MINUS[1], WHITE if lit else (120, 130, 170))

    # ─────────────────────────── the frame
    def draw(self, cv, t):
        if t < BOOT[0]:
            return
        if seg(t, *BOOT) < 0.55:
            self.draw_ghost(cv, t)
            return
        self.draw_structure(cv, t)
        self.draw_chain(cv, t)
        self.draw_io(cv, t)
        self.draw_blocks(cv, t)
        self.draw_tbox(cv, t)
        self.draw_loss(cv, t)
        self.draw_particles(cv, t)

    def draw_ghost(self, cv, t):
        lay = cv.layer()
        self.idle_wires(lay)
        for i, x in enumerate(CHAIN_X):
            lay.text(x + 17, 39, str(CHAIN_T[i]), LABEL, font=FONT35, align='c')
        for i in range(6):
            self.arrow(lay, CHAIN_X[i] + 35, CHAIN_X[i + 1] - 2, 13, ARROW, True)
            self.arrow(lay, CHAIN_X[i] + 35, CHAIN_X[i + 1] - 2, 25, ARROW, False)
        self.minus_node(lay, False)
        cv.composite(lay, alpha=0.7)
        for x in CHAIN_X:
            self.box(cv, x, CHAIN_Y, ghost=True)
        self.box(cv, *IN_BOX, ghost=True)
        self.box(cv, *OUT_BOX, ghost=True)
        for b in BLOCKS:
            self.bars(cv, b, 'idle', ghost=True)
        self.tbox(cv, False, None, ghost=True)

    def idle_wires(self, cv):
        cv.path(BUS_A, WIRE, dash=(1, 3), phase=3)
        for p in DROPS_A[1:]:
            cv.path(p, WIRE, dash=(1, 3), phase=1)
        cv.path(FEED_A, WIRE, dash=(1, 3), phase=2)
        cv.path(X0_FEED, WIRE, dash=(1, 2))
        cv.path(BUS_B, WIRE, dash=(1, 1), phase=1)
        cv.path(FEED_B_L, WIRE, dash=(1, 1))
        cv.path(FEED_B_R, WIRE, dash=(1, 1))
        for p in SKIP:
            cv.path(p, WIRE, dash=(1, 1))
        cv.path(T_TRUNK, WIRE, dash=(1, 1))
        for p in T_DROPS:
            cv.path(p, WIRE, dash=(1, 1))
        for p in CONN:
            cv.path(p, WIRE)

    def draw_structure(self, cv, t):
        self.idle_wires(cv)
        # ── training forward pass: wires light only while data is in transit
        for i, p in enumerate(CONN):
            ts = (F_ACT[i] if i < NB else OUT_IN[0] + 0.05) - 0.22
            if ts <= t < ts + 0.34:
                cv.path(p, CYAN if t < ts + 0.22 else BLUE, u1=seg(t, ts, ts + 0.2))
        for j, p in enumerate(SKIP):              # measured: solid for 0.3 s once the particles arrive
            ts = F_ACT[j] + 0.4
            if ts <= t < ts + 0.3:
                cv.path(p, (40, 110, 250))
        if TEMB[0] <= t < TEMB_OFF:
            self.temb(cv, t, *TEMB)
        # ── backward pass
        for r in range(NB + 1):
            i = NB - r                       # CONN index counted from the output side
            ts = B_ACT[0] - 0.2 + 0.205 * r
            if ts <= t < ts + 0.32:
                cv.path(CONN[i].reversed(), HOT if t < ts + 0.2 else RED_OUT, u1=seg(t, ts, ts + 0.18))
        for j, p in enumerate(SKIP):              # backward: solid for 0.2 s
            ts = B_ACT[j] + 0.45
            if ts <= t < ts + 0.2:
                cv.path(p, HOT)
        # ── x_t for training: box 0 → input
        if XT_FEED[0] <= t < F_ACT[0]:
            cv.path(X0_FEED, YEL if t < XT_IN[1] else YEL_DK, u1=seg(t, *XT_FEED))
        # ── sampling passes
        cp = self.cur_pass(t)
        if cp is not None:
            k, a, b, sp = cp
            for i, p in enumerate(CONN):
                ts = (self.pass_act(k, i) if i < NB else b - 0.02) - sp * 0.9
                if ts <= t < ts + sp * 1.3:
                    cv.path(p, CYAN, u1=seg(t, ts, ts + sp * 0.9))
            for j, p in enumerate(SKIP):
                ts = self.pass_act(k, j) + sp * (2.5 - 0.6 * j)
                if ts <= t < ts + 0.15:
                    cv.path(p, (40, 110, 250))
            self.temb(cv, t, a - 0.05, a + 1.6 * sp, off=b, hot=0.12)
        # landing: x_t ⊖ ε̂ on bus B, then the result climbs bus A into the chain box
        for k in range(6):
            land = S_T[2 * k + 1]
            if land - 0.21 <= t < land + 0.08:                 # measured window
                u = seg(t, land - 0.21, land - 0.1)
                for p in (Path([(30, 82), (30, 64), MINUS]), Path([(274, 82), (274, 64), MINUS])):
                    cv.path(p, (40, 110, 250), u1=u)
            if land - 0.05 <= t < land + 0.17:                 # measured window
                xc = CHAIN_X[5 - k] + 17
                p = Path([(20, 82), (20, 58), (xc, 58), (xc, 37)])
                cv.path(p, (40, 110, 250), u1=seg(t, land - 0.05, land + 0.02))

    def temb(self, cv, t, t0, t1, off=None, hot=0.3):
        if off is not None and t >= off:
            return
        u = seg(t, t0, t1)
        cv.path(T_TRUNK, YEL_DK, u1=u)
        if t < t1 + 0.08:                      # the bright front exists only while it travels
            cv.path(T_TRUNK, YEL, u0=max(0.0, u - 0.3), u1=u)
        for i, p in enumerate(T_DROPS):
            ui = (bcx(BLOCKS[i]) - 50) / (235 - 50)
            ts = lerp_t(t0, t1, 0.18 + 0.82 * ui)
            if t >= ts:
                cv.path(p, YEL if t < ts + hot else YEL_DK, u1=seg(t, ts, ts + 0.1))

    def draw_chain(self, cv, t):
        for i, x in enumerate(CHAIN_X):
            self.box(cv, x, CHAIN_Y, border=self.chain_border(t, i) or BOX_BORDER)
            cv.text(x + 17, 39, str(CHAIN_T[i]), LABEL, font=FONT35, align='c')
        for i in range(6):
            xa, xb = CHAIN_X[i] + 35, CHAIN_X[i + 1] - 2
            tf = FWD0 + i * FWD_DT
            self.arrow(cv, xa, xb, 13, YEL if (tf - 0.02 <= t < tf + 0.42) else ARROW, right=True)
            # ← is white while box i is the next landing target
            wa, wb = self.recv_time(i + 1), self.recv_time(i)
            self.arrow(cv, xa, xb, 25, WHITE if wa <= t < wb else ARROW, right=False)
        for i in range(7):
            self.chain_image(cv, t, i)

    def chain_border(self, t, i):
        if i == 0 and 3.78 <= t < 4.5:
            return YEL
        if i == 0 and FINAL[0] <= t < FINAL[1]:
            return WHITE
        if i < 6:
            rt = self.recv_time(i)
            if rt - 0.02 <= t < rt + 0.4:
                return WHITE if (i <= 1 and blinky(t, 6)) else CYAN
        return None

    def chain_image(self, cv, t, i):
        x, y = CHAIN_X[i] + 1, CHAIN_Y + 1
        img = self.chain[i]
        if i == 0:
            t_in = X0_IN
        else:
            ts = FWD0 + (i - 1) * FWD_DT
            t_in = (ts, ts + 0.38)
        # forward era: in → hold → (reset: out), box 6 survives the reset
        if t < RESET[0] or (i == 6 and t < OUTRO[0]):
            if t >= t_in[0]:
                materialize(cv, img, x, y, seg(t, *t_in), 100 + i)
            return
        if t < RESET[1] and i < 6:
            materialize(cv, img, x, y, seg(t, *RESET), 150 + i, mode='out', sparkle=0)
            return
        if i == 6:
            materialize(cv, img, x, y, seg(t, *OUTRO), 170, mode='out', sparkle=0)
            return
        rt = self.recv_time(i)
        if t >= rt - 0.03:
            simg = self.samp[2 * (6 - i)]
            if t < OUTRO[0]:
                materialize(cv, simg, x, y, seg(t, rt - 0.03, rt + 0.15), 120 + i, sparkle=0)
            else:
                materialize(cv, simg, x, y, seg(t, *OUTRO), 190 + i, mode='out', sparkle=0)

    def draw_io(self, cv, t):
        xi, yi = IN_BOX[0] + 1, IN_BOX[1] + 1
        xo, yo = OUT_BOX[0] + 1, OUT_BOX[1] + 1
        xe, ye = EPS_BOX[0] + 1, EPS_BOX[1] + 1
        in_border = WHITE if FINAL[0] <= t < FINAL[1] else BOX_BORDER
        self.box(cv, *IN_BOX, border=in_border)
        self.box(cv, *OUT_BOX)
        if EPS_IN[0] - 0.05 <= t < RESET[1]:
            self.box(cv, *EPS_BOX)
        # ── training images
        for (img, tin, seed, (bx, by)) in ((self.xt_train, XT_IN, 300, (xi, yi)),
                                            (self.eps_hat_train, OUT_IN, 302, (xo, yo)),
                                            (self.eps_disp, EPS_IN, 304, (xe, ye))):
            if tin[0] <= t < RESET[0]:
                materialize(cv, img, bx, by, seg(t, *tin), seed)
            elif RESET[0] <= t < RESET[1]:
                materialize(cv, img, bx, by, seg(t, *RESET), seed + 1, mode='out', sparkle=0)
        # ── sampling: the current x_t in the input box (updates at every T change)
        if t >= XT_IN_S[0]:
            j = 0
            for k, ts in enumerate(S_T):
                if t >= ts:
                    j = k + 1
            if t >= OUTRO[0]:
                materialize(cv, self.samp[j], xi, yi, seg(t, *OUTRO), 330, mode='out', sparkle=0)
            elif j == 0:
                materialize(cv, self.samp[0], xi, yi, seg(t, *XT_IN_S), 310)
            else:
                tj = S_T[j - 1]
                cv.blit(self.samp[j - 1], xi, yi)
                materialize(cv, self.samp[j], xi, yi, seg(t, tj - 0.1, tj + 0.12), 311 + j, sparkle=0)
            # ε̂ in the output box: produced at the end of each pass, held until the next one
            last = None
            for k, (a, b, sp) in enumerate(self.passes):
                if t >= b - 0.06:
                    last = (k, b - 0.06)
            if last is not None:
                k, ta = last
                img = self.samp_eps[2 * k]
                if t >= OUTRO[0]:
                    materialize(cv, img, xo, yo, seg(t, *OUTRO), 360, mode='out', sparkle=0)
                else:
                    if k > 0:
                        cv.blit(self.samp_eps[2 * k - 2], xo, yo)
                    materialize(cv, img, xo, yo, seg(t, ta, ta + 0.14), 340 + k, sparkle=0)

    # weight-update sparks after backprop (measured): (time of the white flash, colour after it)
    UPD_SPARK = {0: (9.0, 'post_hot'), 1: (8.95, 'post_hot'), 2: (9.1, 'post_sky'), 3: (9.1, 'post_sky'),
                 4: (9.0, 'post_sky'), 5: (9.05, 'post_hot'), 6: (9.15, 'post_sky')}

    def block_state(self, t, i):
        """Per-block state machine, timings measured frame by frame on the reference."""
        # weight-update sparks override whatever the block shows
        u, after = self.UPD_SPARK[i]
        if u <= t < u + 0.05:
            return 'flash', None
        if u + 0.05 <= t < u + 0.15:
            return after, None
        # training forward: cyan → sky → blue, then the blues switch off left→right at ~7.3
        f = F_ACT[i]
        if f - 0.09 <= t < 7.28 + 0.045 * i:
            if t < f - 0.02:
                return 'fcur', self.act[0][i]
            if t < f + 0.05:
                return 'fsky', self.act[0][i]
            return 'fdone', self.act[0][i]
        # backward: pink → hot → dark red for ~1 s
        bi = B_ACT[NB - 1 - i]
        if bi - 0.06 <= t < bi + 1.08:
            if t < bi:
                return 'bpink', self.act[1][i]
            if t < bi + 0.07:
                return 'bhot', self.act[1][i]
            return 'bdone', self.act[1][i]
        # sampling passes: cyan → sky → blue, held until the landing
        cp = self.cur_pass(t)
        if cp is not None:
            k, a, b, sp = cp
            f = self.pass_act(k, i)
            if t < f:
                return 'idle', None
            if t < f + 0.067:
                return 'fcur', self.act[2 + k][i]
            if t < f + 0.133:
                return 'fsky', self.act[2 + k][i]
            if k == 5 or t < S_T[2 * k + 1] + 0.03:
                return 'fdone', self.act[2 + k][i]
        if S_T[-1] <= t < BLOCKS_OFF:                 # the last pass stays lit through the finale
            return 'fdone', self.act[7][i]
        return 'idle', None

    def draw_blocks(self, cv, t):
        for i, b in enumerate(BLOCKS):
            st, pat = self.block_state(t, i)
            self.bars(cv, b, st, pat, t)

    def draw_tbox(self, cv, t):
        v = self.T_value(t)
        cp = self.cur_pass(t)
        active = (TBOX_TRAIN[0] <= t < TBOX_TRAIN[1]) or (cp is not None) or (S_T[-1] <= t < FINAL[0] + 0.1)
        self.tbox(cv, active, v)
        self.minus_node(cv, MINUS_LIT[0] <= t < MINUS_LIT[1])

    def draw_loss(self, cv, t):
        if not (LOSS_T - 0.12 <= t < RESET[0]):
            return
        u = seg(t, LOSS_T - 0.12, LOSS_T + 0.1)
        hot = t < B_ACT[0]
        cv.path(LOSS_A, HOT if hot else (120, 20, 60), dash=(1, 1), u1=u)
        cv.path(LOSS_B, HOT if hot else (120, 20, 60), dash=(1, 1), u1=u)
        if u >= 1:
            pulse = 0.5 + 0.5 * np.sin((t - LOSS_T) * 14)
            cv.diamond(*DIAMOND, 3, HOT)
            cv.diamond(*DIAMOND, 2, mixc((120, 10, 50), HOT, pulse))
            cv.px(*DIAMOND, WHITE)

    def draw_particles(self, cv, t):
        # training forward: along the U and down the skips
        for i, p in enumerate(CONN):
            ts = (F_ACT[i] if i < NB else OUT_IN[0] + 0.05) - 0.22
            particles(cv, p, t, ts, 0.2, n=3, gap=0.045, color=WHITE, trail=2, trail_color=CYAN)
        for j, p in enumerate(SKIP):
            particles(cv, p, t, F_ACT[j] - 0.06, 0.42, n=3, gap=0.05, color=WHITE, trail=2, trail_color=CYAN)
        # backward
        for r in range(NB + 1):
            i = NB - r
            ts = B_ACT[0] - 0.2 + 0.205 * r
            particles(cv, CONN[i].reversed(), t, ts, 0.18, n=3, gap=0.04, color=WHITE, trail=2, trail_color=HOT)
        for j, p in enumerate(SKIP):
            particles(cv, p.reversed(), t, B_ACT[j] - 0.02, 0.44, n=3, gap=0.05, color=WHITE, trail=2,
                      trail_color=HOT)
        # training x_t feed
        particles(cv, X0_FEED, t, XT_FEED[0], 0.3, n=3, gap=0.06, color=YEL_PALE, trail=3, trail_color=YEL)
        # sampling
        cp = self.cur_pass(t)
        if cp is not None:
            k, a, b, sp = cp
            for i, p in enumerate(CONN):
                ts = (self.pass_act(k, i) if i < NB else b - 0.02) - sp * 0.9
                particles(cv, p, t, ts, sp * 0.9, n=2, gap=sp * 0.25, color=WHITE, trail=2, trail_color=CYAN)
            for j, p in enumerate(SKIP):
                ts = self.pass_act(k, j)
                particles(cv, p, t, ts, sp * (2.5 - 0.6 * j), n=2, gap=sp * 0.3, color=WHITE, trail=2,
                          trail_color=CYAN)
        for k in range(6):
            land = S_T[2 * k + 1]
            particles(cv, BUS_B, t, land - 0.22, 0.14, n=2, gap=0.04, color=WHITE, trail=2, trail_color=CYAN)
            particles(cv, BUS_B.reversed(), t, land - 0.22, 0.14, n=2, gap=0.04, color=WHITE, trail=2,
                      trail_color=CYAN)

    def shake(self, t):
        """measured on the reference: nudge at 6.80 (ε̂ out), hit at 7.25 (loss), recoil at 18.983 (x0 back)"""
        return shake_offset(t, [(6.80, 'nudge'), (7.25, 'hit'), (18.983, 'recoil')]), 0

    # ─────────────────────────── soundtrack — the reference's score, recovered note by note with FFTs
    FWD_NOTES = ['A#4', 'D#5', 'G#5', 'C6', 'F6', 'C6', 'G#5']      # 1/8 pulse motif, one note per block
    BWD_NOTES = ['C#5', 'G#4', 'D#4', 'B3', 'F#3', 'C#3', 'G#2']    # square wave, falling
    SLOT_NOTES = ['E6', 'C6', 'D6', 'A6', 'G6', 'C7', 'E6', 'D6', 'E6']
    UPD_NOTES = [('G6', 8.976), ('D7', 9.009), ('E7', 9.042), ('B6', 9.109), ('E6', 9.193)]
    COMBINE_NOTES = ['D6', 'E6', 'G6', 'A6', 'C7', 'D7']            # ⊖ fires: rising pentatonic
    LAND_NOTES = ['E7', 'D7', 'B6', 'A6', 'G6', 'E6']               # lands in the chain: falling

    def sfx(self):
        n = sfx.note
        ev = [(0.55, sfx.boot(0.36), 0.8)]
        for k in range(6):                                           # forward process: noise channel
            ev.append((0.95 + k * FWD_DT, sfx.hiss(0.38, seed=k + 1), 0.7))
        for (ts, _), nm in zip(SLOT, self.SLOT_NOTES):               # random t: pentatonic ticks
            ev.append((ts, sfx.blip(n(nm), 0.045), 0.8))
        ev.append((XT_IN[0], sfx.hiss(0.26, seed=9), 0.6))
        for i, nm in enumerate(self.FWD_NOTES):                      # U-Net forward motif
            ev.append((5.125 + 0.25 * i, sfx.pew(n(nm)), 0.7))
        ev.append((6.825, sfx.blip(n('C7'), 0.06), 0.7))             # ε̂ comes out
        ev.append((7.04, sfx.glide(900, 440, 0.3, duty=0.5), 0.45))  # ε̂ and ε converge on the loss
        ev.append((7.04, sfx.glide(200, 440, 0.3, duty=0.5), 0.45))
        for r, nm in enumerate(self.BWD_NOTES):                      # backprop: falling square notes
            ev.append((7.492 + 0.2 * r, sfx.pew(n(nm), 0.12, duty=0.5, bend=1.0), 0.75))
        for nm, ts in self.UPD_NOTES:                                # weight-update sparkles
            ev.append((ts, sfx.blip(n(nm), 0.04), 0.5))
        ev.append((9.615, sfx.boot(0.34, f=n('G#3'), up=False), 0.7))
        for k in range(6):                                           # sampling: the same motif, faster
            a, b, sp = self.passes[k]
            for i, nm in enumerate(self.FWD_NOTES):
                ev.append((self.pass_act(k, i) + 0.04, sfx.pew(n(nm), min(0.11, sp * 0.8)), 0.6))
            land = S_T[2 * k + 1]
            ev.append((land - 0.17, sfx.blip(n(self.COMBINE_NOTES[k]), 0.07), 0.7))
            ev.append((land - 0.005, sfx.blip(n(self.LAND_NOTES[k]), 0.07), 0.7))
        ev.append((19.008, sfx.blip(n('E5'), 0.06, decay=None), 0.6))
        ev.append((19.07, sfx.chime(), 0.9))
        return ev


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'out', 'diffusion_unet.mp4'))
    args = ap.parse_args()
    sc = DiffusionUNet()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.stills:
        stills(sc, [0.25, 0.75, 2.0, 3.5, 4.3, 4.8, 5.62, 6.5, 7.3, 8.05, 9.1, 9.7, 10.4, 11.0, 12.0, 15.0, 17.5,
                    19.0, 20.0, 21.2], args.out.replace('.mp4', '_stills.png'), cols=4)
    else:
        sc.setup()
        tr = sfx.Track(sc.DURATION)
        for (ts, snd, g) in sc.sfx():
            tr.add(ts, snd, g)
        wav = args.out.replace('.mp4', '.wav')
        tr.save(wav, lufs=-24)
        render(sc, args.out, audio=wav)
