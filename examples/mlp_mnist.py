"""Replica of the reference 'MLP learns MNIST' pixel video (aGZ99qYbSkEyuXqq.mp4, 7 digits × 8 s = 56 s).

Every number on screen is real: a 784→16→16→10 sigmoid MLP trained in numpy on MNIST (93 % test),
then one real SGD step per showcased digit (mlp_mnist_train.py writes the activations, dL/dz per layer,
the input saliency dL/dx and the weight updates ΔW to an .npz this scene reads).

Per-digit cycle (seconds from the cycle start, measured on the reference):
  0.00–0.45  digit sparkles in           1.05–1.35  scan bar + sampled patches → input nodes
  1.35 input · 2.00 hidden 1 · 2.45 hidden 2 · 2.65–2.85 output    (cyan → blue)
  3.25 prediction box                    3.95 loss ◆ (true class vs. top wrong class)
  4.40 out · 5.00 h2 · 5.45 h1 · 5.95 input turn red (backprop)     6.10 saliency on the digit
  6.45 weight update flash (cyan = ΔW > 0, pink = ΔW < 0)          7.30–7.85 dissolve

    python3 mlp_mnist.py --data mlp_mnist_data.npz [--stills] [--out out/mlp_mnist.mp4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pixelkit import (FONT35, FONT57, NEON, Path, Scene, circle_mask, materialize, mixc, particles,  # noqa: E402
                      render, seg, sfx, shake_offset, stills)

BG = (0, 0, 0)
BOX = (28, 34, 66)
RING = (51, 67, 121)
MESH_LO, MESH_HI = (22, 28, 58), (40, 52, 100)
BLUE, BLUE2, SKY, CYAN, WHITE = (16, 54, 179), (29, 106, 251), (32, 177, 251), (91, 239, 247), (240, 244, 252)
HOT, PINK, RED_DK, RED_BG = (252, 28, 118), (253, 90, 172), (123, 9, 44), (57, 6, 18)
LABEL = (91, 100, 136)

# ─────────────────────────────── layout (320×180, measured)
IN_BOX = (14, 60, 60, 60)                       # digit drawn ×2 at (16, 62)
IN_X, H1_X, H2_X, OUT_X = 100, 150, 200, 250
IN_Y = [15, 25, 35, 45, 55, 65, 75, 105, 115, 125, 135, 145, 155, 165]   # 14 shown, '⋮' between
IN_PATCH = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]              # 4×4 patches of 7×7 px
H_Y = [15 + 10 * j for j in range(16)]
OUT_Y = [45 + 10 * k for k in range(10)]
PRED_BOX = (288, 76, 23, 29)
DIAMOND = (298, 128)

# ─────────────────────────────── per-digit timeline (s from cycle start)
CYCLE = 8.0
MAT = (0.0, 0.45)
SCAN = (1.05, 1.35)
F_IN, F_H1, F_H2, F_OUT = 1.35, 2.0, 2.45, 2.65
PRED = (3.2, 3.4)
LOSS = 3.95
FWD_OFF = 4.05
B_OUT, B_H2, B_H1, B_IN = 4.4, 5.0, 5.45, 5.95
SAL = (6.1, 6.35)
RED_OFF = (6.35, 6.6)
UPD = (6.35, 6.6)
OUTRO = (7.3, 7.85)


def patch_means(img):
    return img.reshape(4, 7, 4, 7).mean(axis=(1, 3)).ravel()


class MLPMnist(Scene):
    W, H, SCALE = 320, 180, 6
    FPS = 60
    PAL = dict(NEON, bg=BG)
    DATA = None

    def setup(self):
        path = self.DATA or os.environ.get('MLP_MNIST_DATA') or os.path.join(os.getcwd(), 'data', 'mlp_mnist_data.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} not found. Make it first: python3 {os.path.dirname(os.path.abspath(__file__))}/'
                f'mlp_mnist_train.py --mnist <dir with the 4 MNIST idx files> --out {path}   '
                f'(or set MLP_MNIST_DATA / pass --data)')
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        self.n = len(self.d['label'])
        self.DURATION = CYCLE * self.n
        # per-digit derived data
        self.meshes = []
        for k in range(self.n):
            self.meshes.append((self.mesh(k, before=True), self.mesh(k, before=False)))
        self.ring = circle_mask(4)
        self.disc = circle_mask(3, fill=True)

    # effective input→h1 weights for the 14 displayed patch nodes
    def w1_eff(self, W1):
        w = W1.reshape(28, 28, 16).reshape(4, 7, 4, 7, 16).mean(axis=(1, 3)).reshape(16, 16)
        return w[IN_PATCH]                                     # (14, 16)

    def mesh(self, k, before=True):
        """the faint weight mesh as an image (weights before / after this digit's update)"""
        from pixelkit import Canvas
        W1 = self.d['W1'][k] if before else self.d['W1'][k] + self.d['dW1'][k]
        W2 = self.d['W2'][k] if before else self.d['W2'][k] + self.d['dW2'][k]
        W3 = self.d['W3'][k] if before else self.d['W3'][k] + self.d['dW3'][k]
        cv = Canvas(self.W, self.H, BG)
        for (xa, ya_list, xb, yb_list, w) in self.pairs(W1, W2, W3):
            for (i, j, strong) in self.mesh_edges(w):
                if strong:
                    cv.line(xa + 5, ya_list[i], xb - 5, yb_list[j], MESH_HI)
                else:
                    cv.line(xa + 5, ya_list[i], xb - 5, yb_list[j], MESH_LO, dash=(1, 1))
        return cv.a.copy()

    def pairs(self, W1, W2, W3):
        return ((IN_X, IN_Y, H1_X, H_Y, self.w1_eff(W1)), (H1_X, H_Y, H2_X, H_Y, W2), (H2_X, H_Y, OUT_X, OUT_Y, W3))

    @staticmethod
    def mesh_edges(w, solid=0.08, dotted=0.25):
        """the reference draws only the strongest weights: top ~8 % solid, next 25 % dotted"""
        order = np.argsort(-np.abs(w).ravel())
        n = len(order)
        out = []
        for r, flat in enumerate(order[:int(n * (solid + dotted))]):
            i, j = divmod(flat, w.shape[1])
            out.append((i, j, r < n * solid))
        return out[::-1]                                       # weak first, strong on top

    # ─────────── helpers
    def node(self, cv, x, y, state, v=1.0):
        """state: idle | f (forward, v = activation 0..1) | fcur | b (backward, v = |grad| 0..1)"""
        ys, xs = np.nonzero(self.ring)
        if state == 'idle':
            cv.put(xs - 4 + x, ys - 4 + y, RING)
            return
        if state in ('f', 'fcur'):
            ring = CYAN if state == 'fcur' else mixc(RING, BLUE2, 0.4 + 0.6 * v)
            fill = mixc((8, 14, 40), BLUE, min(1, 0.3 + v))
            core = mixc(BLUE, WHITE, v ** 2)
        else:
            ring = mixc(RED_DK, HOT, v)
            fill = mixc(RED_BG, RED_DK, min(1, 0.3 + v))
            core = mixc(RED_DK, PINK, v ** 2)
        dy, dx = np.nonzero(self.disc)
        cv.put(dx - 3 + x, dy - 3 + y, fill)
        cv.put(xs - 4 + x, ys - 4 + y, ring)
        if v > 0.5:
            cv.fill(x - 1, y - 1, 2, 2, core)
        if v > 0.85 and state != 'b':                         # the winner glows
            cv.put([x - 5, x + 5, x, x], [y, y, y - 5, y + 5], SKY)

    def draw_digit(self, cv, img, u, mode, seed):
        big = np.repeat(np.repeat(img, 2, 0), 2, 1)
        rgb = np.stack([big * 238 + 6, big * 240 + 6, big * 250 + 8], -1)
        mask = (big > 0.08).astype(np.float32)
        materialize(cv, rgb, IN_BOX[0] + 2, IN_BOX[1] + 2, u, seed, mode=mode,
                    sparkle=0.1 if mode == 'in' else 0.0, mask=mask)

    def pred_glyph(self, cv, digit, x, y, u, wrong):
        g = FONT57.glyph(str(digit))
        big = np.repeat(np.repeat(g, 3, 0), 3, 1)
        h, w = big.shape
        stripes = [PINK, HOT, RED_DK] if wrong else [CYAN, SKY, BLUE2]
        img = np.zeros((h, w, 3), np.float32)
        for r in range(h):
            img[r, :] = stripes[r % 3]
        materialize(cv, img, x - w // 2, y - h // 2, u, seed=77 + digit, mask=big.astype(np.float32), sparkle=0.15)

    # ─────────── frame
    def draw(self, cv, t):
        k = min(self.n - 1, int(t // CYCLE))
        c = t - k * CYCLE
        D = {key: v[k] for key, v in self.d.items()}
        img = D['img']
        label = int(D['label'])
        p = D['p']
        pred = int(np.argmax(p))
        wrong = pred != label
        # mesh (switches to the updated weights at the flash)
        mesh = self.meshes[k][1] if c >= UPD[1] else self.meshes[k][0]
        cv.a[:] = mesh
        # frames
        x, y, w, h = IN_BOX
        cv.rect(x, y, w, h, BOX)
        px, py, pw, ph = PRED_BOX
        cv.rect(px, py, pw, ph, BOX)
        for yy in (86, 90, 94):                               # the '⋮' of the 784 inputs
            cv.px(IN_X, yy, (150, 130, 80))
        # labels + bars
        for j in range(10):
            is_pred = c >= F_OUT and c < OUTRO[0] and j == pred
            cv.text(OUT_X + 7, OUT_Y[j] - 2, str(j), WHITE if is_pred else LABEL, font=FONT35)
            if F_OUT <= c < OUTRO[0] + 0.3:
                ub = seg(c, F_OUT, F_OUT + 0.25)
                L = max(1, int(round(ub * 9 * p[j] ** 0.8)))
                bar_c = CYAN if j == pred else BLUE2
                cv.fill(OUT_X + 13, OUT_Y[j] - 1, L, 2, bar_c)
                cv.hline(OUT_X + 13 + L + 1, OUT_X + 24, OUT_Y[j], (24, 30, 60), dash=(1, 1))
        # saliency sits UNDER the digit (the white stroke stays on top, as in the reference)
        if SAL[0] <= c < OUTRO[1]:
            self.saliency(cv, c, D['dx'], img, k)
        # digit
        if c < OUTRO[0]:
            self.draw_digit(cv, img, seg(c, *MAT), 'in', 10 + k)
        else:
            self.draw_digit(cv, img, seg(c, *OUTRO), 'out', 20 + k)
        # ── forward
        a_in = patch_means(img)[IN_PATCH]
        a_in = a_in / max(1e-6, a_in.max())
        a1, a2 = D['a1'], D['a2']
        self.scan(cv, c, img, a_in)
        layers = [(IN_X, IN_Y, a_in, F_IN), (H1_X, H_Y, a1, F_H1), (H2_X, H_Y, a2, F_H2), (OUT_X, OUT_Y, p / p.max(), F_OUT)]
        fwd_on = c < FWD_OFF
        for li, (lx, ly, act, tf) in enumerate(layers):
            for i, yy in enumerate(ly):
                if fwd_on and c >= tf + 0.02 * i * 0:
                    st = 'fcur' if c < tf + 0.1 else 'f'
                    self.node(cv, lx, yy, st, float(np.clip(act[i], 0, 1)))
                elif li == 3 and FWD_OFF <= c < B_OUT:              # outputs stay lit until they turn red
                    self.node(cv, lx, yy, 'f', float(act[i]))
                else:
                    self.node(cv, lx, yy, 'idle')
        # forward flows between layers (top contributions light up, particles run)
        if SCAN[1] <= c < FWD_OFF:
            W1e = self.w1_eff(D['W1'])
            self.flow(cv, c, IN_X, IN_Y, H1_X, H_Y, a_in[:, None] * W1e, F_IN + 0.15, F_H1, fwd=True)
            self.flow(cv, c, H1_X, H_Y, H2_X, H_Y, a1[:, None] * D['W2'], F_H1 + 0.05, F_H2, fwd=True)
            self.flow(cv, c, H2_X, H_Y, OUT_X, OUT_Y, a2[:, None] * D['W3'], F_H2 + 0.02, F_OUT, fwd=True)
        # prediction
        if PRED[0] <= c < OUTRO[0] + 0.2:
            u = seg(c, *PRED) if c < OUTRO[0] else 1 - seg(c, OUTRO[0], OUTRO[0] + 0.2)
            self.pred_glyph(cv, pred, px + pw // 2, py + ph // 2, u, wrong)
        # loss: true class vs top wrong class
        if LOSS <= c < RED_OFF[1]:
            others = np.argsort(-p)
            rival = int(others[0] if others[0] != label else others[1])
            u = seg(c, LOSS, LOSS + 0.15)
            spine = Path([(282, OUT_Y[0]), (282, DIAMOND[1]), (DIAMOND[0] - 4, DIAMOND[1])])
            cv.path(spine, (40, 30, 70), dash=(1, 1), u1=u)
            for j in range(10):
                hot = j in (label, rival)
                cv.hline(OUT_X + 26, 281, OUT_Y[j], HOT if hot else (40, 30, 70), dash=(1, 1))
            pulse = 0.5 + 0.5 * np.sin((c - LOSS) * 16)
            cv.diamond(*DIAMOND, 3, HOT if pulse > 0.3 else CYAN)
            cv.diamond(*DIAMOND, 2, mixc(RED_DK, HOT, pulse))
            cv.px(*DIAMOND, WHITE)
        # ── backward: nodes turn red layer by layer, gradients flow right → left
        if B_OUT - 0.05 <= c < RED_OFF[1]:
            dim = 1 - seg(c, *RED_OFF)
            grads = [(OUT_X, OUT_Y, np.abs(D['d3']), B_OUT), (H2_X, H_Y, np.abs(D['d2']), B_H2),
                     (H1_X, H_Y, np.abs(D['d1']), B_H1), (IN_X, IN_Y, np.abs(patch_means(np.abs(D['dx']))[IN_PATCH]), B_IN)]
            for (lx, ly, g, tb) in grads:
                if c >= tb:
                    g = g / max(1e-9, g.max())
                    for i, yy in enumerate(ly):
                        self.node(cv, lx, yy, 'b', float(g[i]) * dim + 0.05)
            d3n, d2n, d1n = np.abs(D['d3']), np.abs(D['d2']), np.abs(D['d1'])
            for (pair, tl) in zip(self.pairs(D['W1'], D['W2'], D['W3'])[::-1], (B_H2, B_H1, B_IN)):
                if tl - 0.05 <= c < min(tl + 0.6, RED_OFF[0]):          # measured: tint when the left layer reddens
                    xa, ya, xb, yb, w = pair
                    for (i, j, strong) in self.mesh_edges(w):
                        cv.line(xa + 5, ya[i], xb - 5, yb[j], RED_DK if strong else RED_BG,
                                dash=None if strong else (1, 1))
            self.flow(cv, c, H2_X, H_Y, OUT_X, OUT_Y, np.abs(D['W3']) * d3n[None, :], B_OUT + 0.05, B_H2, fwd=False)
            self.flow(cv, c, H1_X, H_Y, H2_X, H_Y, np.abs(D['W2']) * d2n[None, :], B_H2 + 0.05, B_H1, fwd=False)
            self.flow(cv, c, IN_X, IN_Y, H1_X, H_Y, np.abs(self.w1_eff(D['W1'])) * d1n[None, :], B_H1 + 0.05, B_IN, fwd=False)
        if SAL[0] <= c < SAL[1] + 0.5:
            self.saliency_marks(cv, c, D['dx'], img)
        # weight update flash
        if UPD[0] <= c < UPD[1]:
            self.update_flash(cv, D)
        # the loss diamond lingers, dimmed, until the digit dissolves
        if RED_OFF[1] <= c < OUTRO[0] + 0.3:
            cv.diamond(*DIAMOND, 3, RED_DK)
            cv.diamond(*DIAMOND, 2, RED_BG)

    def scan(self, cv, c, img, a_in):
        x, y, w, h = IN_BOX
        if SCAN[0] <= c < SCAN[1]:
            yy = y + 2 + int(seg(c, *SCAN) * (h - 5))
            cv.fill(x + 2, yy, w - 4, 2, (10, 24, 82))
        # sampled patches pop as hollow squares, lines run to their input nodes
        if SCAN[0] + 0.1 <= c < F_IN + 0.45:
            for n, pi in enumerate(IN_PATCH):
                if a_in[n] < 0.35:
                    continue
                r, q = divmod(pi, 4)
                sx, sy = x + 2 + q * 14 + 5, y + 2 + r * 14 + 5
                t_on = SCAN[0] + 0.1 + (r / 4) * 0.2
                if c < t_on:
                    continue
                col = CYAN if (n % 3) else WHITE
                cv.rect(sx, sy, 4, 4, col)
                pth = Path([(sx + 4, sy + 1), (IN_X - 5, IN_Y[n])])
                cv.path(pth, (12, 30, 90), u1=seg(c, t_on, t_on + 0.15))
                particles(cv, pth, c, t_on + 0.05, 0.25, n=2, gap=0.06, color=WHITE, trail=2, trail_color=BLUE2)

    def flow(self, cv, c, xa, ya, xb, yb, contrib, t0, t1, fwd=True, top=14):
        """light the strongest contributions between two layers and run particles along them"""
        if not (t0 - 0.05 <= c < t1 + 0.25):
            return
        flat = np.argsort(-np.abs(contrib).ravel())[:top]
        m = np.abs(contrib).ravel()[flat[0]] + 1e-9
        for r, f in enumerate(flat):
            i, j = divmod(f, contrib.shape[1])
            s = abs(contrib[i, j]) / m
            pa, pb = (xa + 5, ya[i]), (xb - 5, yb[j])
            pth = Path([pa, pb]) if fwd else Path([pb, pa])
            lit = seg(c, t0, t1)
            col = mixc(BLUE, BLUE2, s) if fwd else mixc(RED_DK, HOT, s)
            if c < t1 + 0.2:
                cv.path(pth, col, u1=lit, dash=None if s > 0.5 else (1, 1))
            particles(cv, pth, c, t0 + 0.02 * r, (t1 - t0) * 0.8, n=1, color=WHITE if fwd else PINK, trail=2,
                      trail_color=SKY if fwd else HOT)

    def saliency(self, cv, c, dx, img, k):
        x, y, w, h = IN_BOX
        import cv2
        near = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
        s = np.abs(dx) * (0.15 + near / max(1e-6, near.max()))      # gradient, concentrated around the ink
        s = s / max(1e-9, s.max())
        big = np.repeat(np.repeat(s, 2, 0), 2, 1)
        u = seg(c, *SAL)
        out = seg(c, *OUTRO)
        mask = ((big > 0.12).astype(np.float32))
        col = np.zeros(big.shape + (3,), np.float32)
        col[:] = RED_DK
        col[big > 0.3] = HOT
        col[big > 0.65] = PINK
        if c < OUTRO[0]:
            materialize(cv, col, x + 2, y + 2, u, 40 + k, sparkle=0.0, mask=mask * 0.85)
        else:
            materialize(cv, col, x + 2, y + 2, out, 50 + k, mode='out', sparkle=0.0, mask=mask * 0.85)
    def saliency_marks(self, cv, c, dx, img):
        x, y, w, h = IN_BOX
        import cv2
        near = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
        s = np.abs(dx) * (0.15 + near / max(1e-6, near.max()))
        s = s / max(1e-9, s.max())
        # pink hollow squares on the strongest patches, red lines from the input nodes
        if SAL[0] <= c < SAL[1] + 0.5:
            pm = s.reshape(4, 7, 4, 7).mean(axis=(1, 3))
            order = np.argsort(-pm.ravel())[:6]
            for n, pi in enumerate(order):
                r, q = divmod(pi, 4)
                sx, sy = x + 2 + q * 14 + 5, y + 2 + r * 14 + 5
                cv.rect(sx, sy, 4, 4, PINK if n % 2 else HOT)
                if pi in IN_PATCH:
                    ni = IN_PATCH.index(pi)
                    pth = Path([(IN_X - 5, IN_Y[ni]), (sx + 4, sy + 1)])
                    cv.path(pth, mixc(RED_DK, HOT, 0.8), u1=seg(c, SAL[0] - 0.1, SAL[0] + 0.1))

    def update_flash(self, cv, D):
        for (xa, ya, xb, yb, dW) in ((H1_X, H_Y, H2_X, H_Y, D['dW2']), (H2_X, H_Y, OUT_X, OUT_Y, D['dW3']),
                                     (IN_X, IN_Y, H1_X, H_Y, self.w1_eff(D['dW1']))):
            flat = np.argsort(-np.abs(dW).ravel())[:8]
            for f in flat:
                i, j = divmod(f, dW.shape[1])
                col = CYAN if dW[i, j] > 0 else HOT
                cv.line(xa + 5, ya[i], xb - 5, yb[j], col)
                cv.line(xa + 5, ya[i] + 1, xb - 5, yb[j] + 1, mixc(BG, col, 0.5))

    def shake(self, t):
        """measured: +1 px nudge at 2.867 s (output lit), −2/−1 px hit at 4.10 s (loss), every cycle"""
        c = t % CYCLE
        return shake_offset(c, [(2.865, 'nudge'), (4.098, 'hit')]), 0

    # ─────────── 8-bit score (recovered from the reference with FFTs)
    def sfx(self):
        n = sfx.note
        ev = []
        for k in range(self.n):
            c0 = k * CYCLE
            wrong = int(np.argmax(self.d['p'][k])) != int(self.d['label'][k])
            ev += [(c0 + 0.05, sfx.boot(0.35), 0.5),
                   (c0 + SCAN[0] + 0.5, sfx.hiss(0.18, rate0=6000, rate1=1500, seed=k), 0.35),
                   (c0 + F_IN + 0.2, sfx.pew(n('C5')), 0.6),
                   (c0 + F_H1 - 0.1, sfx.pew(n('A#5')), 0.6),
                   (c0 + F_H2 - 0.1, sfx.pew(n('F6')), 0.6),
                   (c0 + F_OUT + 0.24, sfx.pew(n('F6')), 0.55),
                   (c0 + 3.275, sfx.blip(n('E5') if not wrong else n('D#5'), 0.12, decay=0.08), 0.7),
                   (c0 + LOSS, sfx.glide(n('G4'), n('G4') * 0.98, 0.25, duty=0.5), 0.45)]
            for tb, nm in ((4.444, 'F#4'), (4.992, 'C4'), (5.459, 'F#3'), (5.958, 'C3')):   # tritones down
                ev.append((c0 + tb, sfx.pew(n(nm), 0.14, duty=0.5, bend=1.0), 0.7))
            ev.append((c0 + UPD[0], sfx.arp(['E6', 'G6', 'B6', 'D7'], step=0.035, duty=0.25), 0.35))
        return ev


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--out', default=os.path.join(os.getcwd(), 'out', 'mlp_mnist.mp4'))
    a = ap.parse_args()
    MLPMnist.DATA = os.path.abspath(a.data)
    sc = MLPMnist()
    sc.setup()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.stills:
        stills(sc, [0.3, 1.25, 1.6, 2.1, 2.5, 3.5, 4.2, 4.7, 5.2, 5.7, 6.2, 6.5, 7.0, 7.6, 51.5, 53.5],
               a.out.replace('.mp4', '_stills.png'), cols=4)
    else:
        tr = sfx.Track(sc.DURATION)
        for (ts, snd, g) in sc.sfx():
            tr.add(ts, snd, g)
        wav = a.out.replace('.mp4', '.wav')
        tr.save(wav, lufs=-23)
        render(sc, a.out, audio=wav)
