"""pixelkit core — low-res pixel-art mechanism animations, rendered to 1920×1080 @ 60 fps.

Model of the reference videos (see STYLE.md):
  * every frame is drawn on a small canvas (320×180 → ×6, or 480×270 → ×4) and upscaled
    with nearest-neighbour, so every edge sits on the big-pixel grid;
  * motion is continuous at 60 fps, but positions are always snapped to whole canvas pixels;
  * optional "phosphor" post: soft bloom computed at low res and upscaled bilinearly;
  * scenes are pure functions of time t → any frame can be rendered alone, in any order,
    in parallel.
"""
import math
import os
import subprocess
import sys
import time as _time

import cv2
import numpy as np

from .fonts import FONT35, FONT57

# ════════════════════════════════════════════════════════════ timing helpers

def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def seg(t, t0, t1):
    """Progress of t through [t0, t1] as 0..1 (clamped)."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / (t1 - t0))


def lerp(a, b, u):
    return a + (b - a) * u


def lerp_pt(p0, p1, u):
    """Integer point between p0 and p1 (for the rare element that really moves; prefer dissolving)."""
    return (int(round(p0[0] + (p1[0] - p0[0]) * u)), int(round(p0[1] + (p1[1] - p0[1]) * u)))


def ease_in(u):
    return u * u * u


def ease_out(u):
    return 1 - (1 - u) ** 3


def ease_io(u):
    return 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def smooth(u):
    return u * u * (3 - 2 * u)


def quant(u, n):
    """Quantise a 0..1 progress into n discrete steps (pixel-y, stepped motion)."""
    return math.floor(clamp(u) * n + 1e-9) / n


def hexc(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def mixc(c0, c1, u):
    u = clamp(u)
    return tuple(a + (b - a) * u for a, b in zip(c0, c1))


def scalec(c, k):
    return tuple(min(255.0, v * k) for v in c)


def blink(t, hz=2.0, duty=0.5):
    return (t * hz) % 1.0 < duty


def hash01(*args):
    """Deterministic pseudo-random 0..1 from integers (stable across processes)."""
    h = 2166136261
    for a in args:
        h ^= int(a) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0


# ════════════════════════════════════════════════════════════ palettes

# v1 "neon" — the 320×180 videos (MLP on MNIST, diffusion U-Net): near-black ground,
# saturated arcade colours, no bloom, no text beyond 3×5 numerals.
NEON = dict(
    bg=(3, 3, 11), black=(0, 0, 0),
    box=(50, 70, 123), box_fill=(8, 12, 27), wire=(27, 35, 65), dim=(48, 69, 122),
    bar=(53, 67, 119), bar_in=(16, 24, 44),
    label=(47, 66, 113), label2=(91, 100, 136),
    blue=(16, 55, 180), blue2=(29, 106, 251), blue3=(30, 130, 252), cyan=(85, 211, 213), cyan2=(91, 239, 247),
    sky=(32, 177, 251),
    red=(252, 28, 118), magenta=(252, 28, 118), red_md=(197, 13, 79), red_dk=(123, 9, 44), red_bg=(57, 6, 18),
    pink=(253, 90, 172),
    yellow=(252, 192, 28), yellow_dk=(137, 98, 32), yellow_pale=(253, 239, 169),
    white=(240, 240, 248), white2=(254, 254, 254),
    idle=(50, 70, 123), active=(85, 211, 213), done=(16, 55, 180), fwd=(85, 211, 213), bwd=(253, 90, 172),
)

# v2 "phosphor" — the 480×270 explainer videos (V-JEPA 2.1, π0.5): navy ground with a dot grid,
# softer CRT colours, bloom, HD44780 text header + typewriter captions.
PHOSPHOR = dict(
    bg=(9, 11, 25), grid=(26, 30, 52), rule=(52, 60, 96),
    wire=(31, 39, 68), dim=(50, 70, 112), dim2=(55, 65, 108), todo=(33, 39, 79), label=(114, 121, 160),
    text=(240, 242, 249), text2=(195, 201, 232),
    blue=(98, 144, 244), blue_md=(55, 130, 180), blue_dk=(40, 52, 96),
    orange=(237, 104, 68), orange_md=(167, 77, 58), orange_dk=(115, 51, 42),
    cream=(213, 211, 194), tan=(213, 182, 153), tan_dk=(141, 128, 106),
    yellow=(245, 199, 112), teal=(86, 168, 147), pink=(201, 102, 140), grey=(91, 89, 95),
    white=(240, 242, 249),
    # three-state elements in v2 (idle → computing → done); see widgets.module / widgets.wire
    idle=(55, 65, 108), active=(240, 242, 249), done=(40, 52, 96),
    fwd=(98, 144, 244), bwd=(201, 102, 140),
)


# The 32-colour palette of the v1 videos, recovered by clustering every image pixel of the diffusion
# reference (UI colours and image colours share one palette — that is why the noise looks 'on-brand').
NEON32 = [
    (8, 12, 26), (15, 27, 66), (62, 120, 57), (141, 207, 105), (106, 62, 160), (35, 86, 127), (53, 38, 84),
    (16, 55, 179), (46, 142, 110), (138, 97, 36), (110, 45, 103), (240, 193, 132), (221, 208, 224),
    (253, 91, 172), (178, 69, 111), (81, 195, 192), (252, 29, 119), (196, 14, 76), (252, 193, 29),
    (92, 101, 135), (51, 12, 16), (123, 10, 42), (238, 124, 91), (191, 137, 80), (215, 62, 46), (41, 141, 165),
    (30, 106, 252), (253, 240, 174), (88, 49, 33), (233, 253, 253), (32, 178, 252), (92, 239, 253),
]


def quantize(img, palette, dither=0.0, seed=0):
    """Map every pixel to the nearest palette colour (redmean-weighted RGB distance).
    dither > 0 adds that much uniform noise before the lookup (breaks up banding)."""
    pal = np.asarray(palette, np.float32)
    x = np.asarray(img, np.float32)
    if dither:
        x = x + np.random.default_rng(seed).uniform(-dither, dither, x.shape).astype(np.float32)
    flat = x.reshape(-1, 3)
    idx = np.empty(len(flat), np.int64)
    for s in range(0, len(flat), 1 << 15):          # chunked: ~0.5 KB/pixel peak otherwise
        f = flat[s:s + (1 << 15)]
        rm = (f[:, None, 0] + pal[None, :, 0]) / 2
        d = f[:, None, :] - pal[None, :, :]
        dist = (2 + rm / 256) * d[..., 0] ** 2 + 4 * d[..., 1] ** 2 + (2 + (255 - rm) / 256) * d[..., 2] ** 2
        idx[s:s + len(f)] = np.argmin(dist, axis=1)
    return pal[idx].reshape(x.shape)


# ════════════════════════════════════════════════════════════ rasterisation helpers

def raster_line(x0, y0, x1, y1):
    """Pixel centres of a line (DDA, symmetric rounding). Returns (xs, ys) int arrays."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    n = max(abs(x1 - x0), abs(y1 - y0))
    if n == 0:
        return np.array([x0]), np.array([y0])
    k = np.arange(n + 1) / n
    xs = np.floor(x0 + (x1 - x0) * k + 0.5).astype(np.int32)
    ys = np.floor(y0 + (y1 - y0) * k + 0.5).astype(np.int32)
    return xs, ys


class Path:
    """A rasterised polyline with arc-length (pixel-count) parameterisation.

    p = Path([(10, 20), (40, 20), (40, 60)])
    p.n            number of pixels
    p.at(u)        (x, y) at fraction u (0..1)
    p.xs, p.ys     pixel coordinates in order
    """

    def __init__(self, pts):
        xs, ys = [], []
        for (a, b) in zip(pts[:-1], pts[1:]):
            lx, ly = raster_line(a[0], a[1], b[0], b[1])
            if xs:
                lx, ly = lx[1:], ly[1:]
            xs.append(lx)
            ys.append(ly)
        if not xs:
            xs, ys = [np.array([int(pts[0][0])])], [np.array([int(pts[0][1])])]
        self.xs = np.concatenate(xs)
        self.ys = np.concatenate(ys)
        self.n = len(self.xs)
        self.pts = pts

    def idx(self, u):
        return int(clamp(u) * (self.n - 1) + 0.5)

    def at(self, u):
        i = self.idx(u)
        return int(self.xs[i]), int(self.ys[i])

    def reversed(self):
        return Path(list(reversed(self.pts)))


def circle_mask(r, fill=False):
    key = (r, fill)
    m = _CIRCLES.get(key)
    if m is None:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d = np.sqrt(xx * xx + yy * yy)
        if fill:
            m = d <= r + 0.35
        else:
            inside = d <= r + 0.35
            # 4-neighbour erosion → a clean 1-px ring
            er = inside.copy()
            er[1:, :] &= inside[:-1, :]
            er[:-1, :] &= inside[1:, :]
            er[:, 1:] &= inside[:, :-1]
            er[:, :-1] &= inside[:, 1:]
            m = inside & ~er
        _CIRCLES[key] = m
    return m


_CIRCLES = {}


# ════════════════════════════════════════════════════════════ canvas

class Canvas:
    """Float RGB canvas. With layer=True it also tracks coverage (alpha) so it can be
    composited later with a reveal / dissolve / ghost effect."""

    def __init__(self, w, h, bg=(0, 0, 0), layer=False):
        self.w, self.h = w, h
        self.bg = np.array(bg, np.float32)
        self.a = np.empty((h, w, 3), np.float32)
        self.a[:] = self.bg
        self.m = np.zeros((h, w), np.float32) if layer else None

    def clear(self, c=None):
        self.a[:] = self.bg if c is None else np.array(c, np.float32)
        if self.m is not None:
            self.m[:] = 0

    def layer(self):
        """A transparent layer of the same size (draw into it, then composite())."""
        return Canvas(self.w, self.h, (0, 0, 0), layer=True)

    @staticmethod
    def _mix(dst, m0, src, a):
        """'src over dst' with coverage a (…, ) → (colour, coverage). Plain canvas: m0 is None.
        Layers keep straight (non-premultiplied) colour, so a half-transparent stroke drawn into a
        layer composites exactly like the same stroke drawn directly (no double alpha)."""
        if m0 is None:
            return dst + (src - dst) * a[..., None], None
        m1 = a + m0 * (1.0 - a)
        col = (src * a[..., None] + dst * (m0 * (1.0 - a))[..., None]) / np.maximum(m1, 1e-6)[..., None]
        return col, m1

    def over(self, src, cov):
        """Composite a full-size straight-colour image with per-pixel coverage onto this canvas."""
        cov = np.asarray(cov, np.float32)
        col, m1 = self._mix(self.a, self.m, np.asarray(src, np.float32), cov)
        self.a = col.astype(np.float32)
        if m1 is not None:
            self.m = m1.astype(np.float32)

    # ─────────────────────────── low level
    def put(self, xs, ys, c, alpha=1.0):
        """Plot pixels. c: rgb tuple or (N,3) array; alpha: scalar or (N,) array."""
        xs = np.asarray(xs, np.int32).ravel()
        ys = np.asarray(ys, np.int32).ravel()
        ok = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h)
        if not ok.all():
            xs, ys = xs[ok], ys[ok]
        if xs.size == 0:
            return
        c = np.asarray(c, np.float32)
        if c.ndim == 2 and not ok.all():
            c = c[ok]
        if np.ndim(alpha) == 0:
            al = float(alpha)
            if al <= 0:
                return
            if al >= 0.999:
                self.a[ys, xs] = c
                if self.m is not None:
                    self.m[ys, xs] = 1.0
                return
            al = np.full(xs.shape, min(al, 1.0), np.float32)
        else:
            al = np.clip(np.asarray(alpha, np.float32).ravel(), 0, 1)
            if not ok.all():
                al = al[ok]
        dst = self.a[ys, xs]
        col, m1 = self._mix(dst, None if self.m is None else self.m[ys, xs], np.broadcast_to(c, dst.shape), al)
        self.a[ys, xs] = col
        if m1 is not None:
            self.m[ys, xs] = m1

    def px(self, x, y, c, alpha=1.0):
        self.put([x], [y], c, alpha)

    # ─────────────────────────── lines & shapes
    @staticmethod
    def _dash(n, dash, phase):
        if not dash:
            return np.ones(n, bool)
        on, off = dash
        return ((np.arange(n) + int(math.floor(phase))) % (on + off)) < on

    def line(self, x0, y0, x1, y1, c, alpha=1.0, dash=None, phase=0, frac=1.0):
        xs, ys = raster_line(x0, y0, x1, y1)
        k = self._dash(len(xs), dash, phase)
        if frac < 1.0:
            k &= np.arange(len(xs)) < int(round(clamp(frac) * len(xs)))
        self.put(xs[k], ys[k], c, alpha)

    def hline(self, x0, x1, y, c, alpha=1.0, dash=None, phase=0):
        self.line(x0, y, x1, y, c, alpha, dash, phase)

    def vline(self, x, y0, y1, c, alpha=1.0, dash=None, phase=0):
        self.line(x, y0, x, y1, c, alpha, dash, phase)

    def path(self, p, c, alpha=1.0, dash=None, phase=0, u0=0.0, u1=1.0):
        """Draw a Path (or a pixel range of it: fractions u0..u1)."""
        i0 = int(round(clamp(u0) * p.n))
        i1 = int(round(clamp(u1) * p.n))
        if i1 <= i0:
            return
        k = self._dash(p.n, dash, phase)[i0:i1]
        self.put(p.xs[i0:i1][k], p.ys[i0:i1][k], c, alpha)

    def rect(self, x, y, w, h, c, alpha=1.0, dash=None, phase=0):
        """1-px outline of the w×h box whose top-left pixel is (x, y)."""
        if w <= 0 or h <= 0:
            return
        x2, y2 = x + w - 1, y + h - 1
        p = Path([(x, y), (x2, y), (x2, y2), (x, y2), (x, y)])
        self.path(p, c, alpha, dash, phase)

    def fill(self, x, y, w, h, c, alpha=1.0):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x + w)), min(self.h, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return
        c = np.array(c, np.float32)
        if alpha <= 0:
            return
        if alpha >= 0.999:
            self.a[y0:y1, x0:x1] = c
            if self.m is not None:
                self.m[y0:y1, x0:x1] = 1.0
            return
        a = np.full((y1 - y0, x1 - x0), float(alpha), np.float32)
        col, m1 = self._mix(self.a[y0:y1, x0:x1], None if self.m is None else self.m[y0:y1, x0:x1],
                            np.broadcast_to(c, (y1 - y0, x1 - x0, 3)), a)
        self.a[y0:y1, x0:x1] = col
        if m1 is not None:
            self.m[y0:y1, x0:x1] = m1

    def checker(self, x, y, w, h, c, alpha=1.0, parity=0, density=0.5):
        """Ordered-dither fill: density 0.5 = checkerboard, 0.25 = sparse grid dots."""
        yy, xx = np.mgrid[int(y):int(y + h), int(x):int(x + w)]
        if density >= 0.5:
            k = ((xx + yy + parity) % 2) == 0
        else:
            k = ((xx + parity) % 2 == 0) & ((yy + parity) % 2 == 0)
        self.put(xx[k], yy[k], c, alpha)

    def circle(self, cx, cy, r, c, alpha=1.0, fill=False):
        m = circle_mask(int(r), fill)
        yy, xx = np.nonzero(m)
        self.put(xx - int(r) + int(cx), yy - int(r) + int(cy), c, alpha)

    def diamond(self, cx, cy, r, c, alpha=1.0, fill=False):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d = np.abs(xx) + np.abs(yy)
        k = d <= r if fill else d == r
        self.put(xx[k] + cx, yy[k] + cy, c, alpha)

    # ─────────────────────────── text
    def text(self, x, y, s, c, alpha=1.0, font=FONT57, align='l', n=None, sub_dy=3, sub_color=None):
        """Draw text with its cap-top at y. align: 'l' | 'c' | 'r' (x is left / centre / right).
        n: show only the first n characters (typewriter; markup does not count).
        Inline scripts: 'x_{t}', 'π_{0.5}' (down sub_dy px), 'ℝ^{d×k}', 'Aᵀ' / 'A^{T}' (up sub_dy px).
        Returns the drawn width in px."""
        runs = _parse_sub(s)
        total = sum(font.width(r) for r, _ in runs) + font.spacing * (len(runs) - 1 if runs else 0)
        if align == 'c':
            x = x - total // 2
        elif align == 'r':
            x = x - total + 1
        left = n if n is not None else 10 ** 9
        cx = int(x)
        for i, (r, kind) in enumerate(runs):
            if i:
                cx += font.spacing
            for ch in r:
                if left <= 0:
                    return total
                g = font.glyph(ch)
                gy, gx = np.nonzero(g)
                dy = sub_dy if kind == 1 else (-sub_dy if kind == 2 else 0)
                col = sub_color if (kind and sub_color is not None) else c
                self.put(gx + cx, gy + int(y) + dy, col, alpha)
                cx += g.shape[1] + font.spacing
                left -= 1
            cx -= font.spacing
        return total

    def text_width(self, s, font=FONT57):
        runs = _parse_sub(s)
        return sum(font.width(r) for r, _ in runs) + font.spacing * (len(runs) - 1 if runs else 0)

    # ─────────────────────────── images
    def blit(self, img, x, y, alpha=1.0, mask=None):
        """Paste an (h, w, 3) image (0..255) at (x, y). mask: optional (h, w) 0..1 coverage."""
        img = np.asarray(img, np.float32)
        h, w = img.shape[:2]
        x, y = int(x), int(y)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        src = img[y0 - y:y1 - y, x0 - x:x1 - x]
        m = np.full(src.shape[:2], float(alpha), np.float32)
        if mask is not None:
            m *= np.asarray(mask, np.float32)[y0 - y:y1 - y, x0 - x:x1 - x]
        col, m1 = self._mix(self.a[y0:y1, x0:x1], None if self.m is None else self.m[y0:y1, x0:x1], src, m)
        self.a[y0:y1, x0:x1] = col
        if m1 is not None:
            self.m[y0:y1, x0:x1] = m1

    def composite(self, layer, alpha=1.0, mask=None):
        """Blend a layer (Canvas(layer=True)) onto this canvas. mask: extra (h,w) 0..1 factor."""
        m = layer.m * alpha
        if mask is not None:
            m = m * mask
        self.over(layer.a, m)


def _parse_sub(s):
    """Inline scripts: 'π_{0.5}' subscript, 'ℝ^{d×k}' superscript.
    'x_{t}^{2}' → [('x', 0), ('t', 1), ('2', 2)]   (kind 0 = normal, 1 = sub, 2 = sup)
    An unclosed '_{' / '^{' is kept as literal text; an empty one adds nothing."""
    out, i, buf = [], 0, ''
    while i < len(s):
        if s.startswith('_{', i) or s.startswith('^{', i):
            j = s.find('}', i)
            if j < 0:                       # unclosed: keep it as literal text
                buf += s[i:]
                break
            if j > i + 2:
                if buf:
                    out.append((buf, 0))
                    buf = ''
                out.append((s[i + 2:j], 1 if s[i] == '_' else 2))
            i = j + 1
        else:
            buf += s[i]
            i += 1
    if buf or not out:
        out.append((buf, 0))
    return out


# ════════════════════════════════════════════════════════════ effects

def reveal_mask(h, w, u, seed, mode='in', sparkle=0.08):
    """Per-pixel random dissolve. Returns (visible, sparkle) boolean masks.

    mode 'in': pixels appear in random order as u goes 0→1; a pixel that appeared within the
    last `sparkle` of progress is flagged as sparkle (draw it bright for a couple of frames).
    mode 'out': the reverse (pixels vanish; the ones about to vanish sparkle)."""
    rng = np.random.default_rng(seed)
    r = rng.random((h, w), dtype=np.float32)
    if mode == 'in':
        vis = r < u
        sp = vis & (r > u - sparkle) & (u < 1.0)
    else:
        vis = r >= u
        sp = vis & (r < u + sparkle) & (u > 0.0)
    return vis, sp


_RMASK = {}


def _rand_field(h, w, seed):
    key = (h, w, seed)
    r = _RMASK.get(key)
    if r is None:
        r = np.random.default_rng(seed).random((h, w), dtype=np.float32)
        _RMASK[key] = r
    return r


def materialize(cv, img, x, y, u, seed, sparkle_color=(255, 255, 255), mode='in', sparkle=0.1, mask=None):
    """Blit img with a random per-pixel dissolve (the reference 'sparkle-in' / 'sparkle-out').
    u: 0..1 progress. In 'in' mode pixels pop in with a one-beat bright sparkle."""
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    r = _rand_field(h, w, seed)
    if mode == 'in':
        vis = r < u
        sp = vis & (r > u - sparkle) & (u < 1.0)
    else:
        vis = r >= u
        sp = vis & (r < u + sparkle) & (u > 0.0)
    m = vis.astype(np.float32)
    if mask is not None:
        m *= mask
    out = img.copy()
    out[sp] = sparkle_color
    cv.blit(out, x, y, 1.0, m)


def reveal_layer(cv, layer, u, seed, mode='in', sparkle_color=None, sparkle=0.1):
    """Composite a drawn layer with the same random per-pixel dissolve."""
    r = _rand_field(cv.h, cv.w, seed)
    if mode == 'in':
        vis = r < u
        sp = vis & (r > u - sparkle) & (u < 1.0)
    else:
        vis = r >= u
        sp = vis & (r < u + sparkle) & (u > 0.0)
    m = vis.astype(np.float32)
    if sparkle_color is not None:
        L = layer.a.copy()
        L[sp & (layer.m > 0)] = sparkle_color
        cv.over(L, layer.m * m)
    else:
        cv.composite(layer, mask=m)


def ghost_layer(cv, layer, alpha=0.5, parity=0):
    """Boot-up 'ghost' look: only every other pixel (checkerboard) of the layer, dimmed."""
    yy, xx = np.mgrid[0:cv.h, 0:cv.w]
    k = (((xx + yy + parity) % 2) == 0).astype(np.float32)
    cv.composite(layer, alpha=alpha, mask=k)


def particles(cv, path, t, t0, travel, n=1, gap=0.08, color=(255, 255, 255), trail=3,
              trail_color=None, reverse=False, alpha=1.0):
    """Dots running along a Path. Particle k leaves at t0 + k*gap and takes `travel` seconds.
    Head pixel in `color`, a fading trail of `trail` pixels behind it."""
    tc = trail_color if trail_color is not None else color
    for k in range(n):
        u = seg(t, t0 + k * gap, t0 + k * gap + travel)
        if u <= 0 or u >= 1:
            continue
        if reverse:
            u = 1 - u
        i = int(u * (path.n - 1))
        cv.px(path.xs[i], path.ys[i], color, alpha)
        for j in range(1, trail + 1):
            ii = i + j if reverse else i - j
            if 0 <= ii < path.n:
                cv.px(path.xs[ii], path.ys[ii], tc, alpha * (1 - j / (trail + 1)))


# screen shake, measured on the v1 references: whole-frame horizontal jolts of whole canvas pixels
SHAKE = {
    'nudge': [(0.0, +1), (0.1, 0)],                 # a result lands (output layer lit, ε̂ produced)
    'hit': [(0.0, -2), (1 / 60, -1), (0.1, 0)],     # the loss is computed
    'recoil': [(0.0, -1), (0.1, 0)],                # success / final result
}


def shake_offset(t, events):
    """events: [(t0, 'nudge'|'hit'|'recoil'|[(dt, dx), …]), …] → integer x offset at time t."""
    dx = 0
    for t0, pat in events:
        seq = SHAKE[pat] if isinstance(pat, str) else pat
        if t < t0 or t >= t0 + seq[-1][0] + 1e-9:
            continue
        for dt, v in seq:
            if t >= t0 + dt - 1e-9:
                dx = v
    return dx


def shift_frame(a, dx, dy, bg):
    """Shift the whole low-res frame by whole pixels; the exposed edge is background."""
    if not dx and not dy:
        return a
    out = np.empty_like(a)
    out[:] = np.asarray(bg, np.float32)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def typewriter(s, t, t0, cps=55.0):
    """Number of visible characters of s (markup like '_{…}' not counted) at time t when typing starts at t0."""
    if t < t0:
        return 0
    return min(sum(len(r) for r, _ in _parse_sub(s)), int((t - t0) * cps))


# ════════════════════════════════════════════════════════════ phosphor chrome (v2 style)

def dot_grid(cv, color, step=6, ox=5, oy=2, alpha=1.0):
    """The faint background dot grid of the 480×270 videos (one pixel every `step`)."""
    ys = np.arange(oy, cv.h, step)
    xs = np.arange(ox, cv.w, step)
    xx, yy = np.meshgrid(xs, ys)
    cv.put(xx.ravel(), yy.ravel(), color, alpha)


def dotted_rule(cv, y, x0, x1, color, alpha=1.0):
    cv.hline(x0, x1, y, color, alpha, dash=(1, 1))


def progress_squares(cv, x_right, y, n, cur, pal, size=4, gap=2, done_color=None, cur_color=None, todo_color=None):
    """Stage indicator: n little squares ending at x_right; past = filled dim, current = accent,
    future = hollow outline. Returns left x."""
    done_color = done_color or pal['dim2']
    cur_color = cur_color or pal['orange']
    todo_color = todo_color or pal['dim2']
    x = x_right - n * size - (n - 1) * gap + 1
    for i in range(n):
        xi = x + i * (size + gap)
        if i < cur:
            cv.fill(xi, y, size, size, done_color)
        elif i == cur:
            cv.fill(xi, y, size, size, cur_color)
        else:
            cv.rect(xi, y, size, size, todo_color)
    return x


# ════════════════════════════════════════════════════════════ post & output

def upscale(lo, k):
    """Nearest-neighbour integer upscale of an (h, w, 3) float/uint8 image → uint8."""
    a = np.clip(lo, 0, 255).astype(np.uint8) if lo.dtype != np.uint8 else lo
    return cv2.resize(a, (a.shape[1] * k, a.shape[0] * k), interpolation=cv2.INTER_NEAREST)


def bloom(lo, k, bg, strength=0.5, sigma=2.6, threshold=20.0):
    """Phosphor bloom: blur the bright part of the low-res frame, upscale it smoothly, add.
    Defaults fitted to the V-JEPA 2.1 reference (mean abs error 1.8/255 on the halo bands).
    Returns the final uint8 full-res frame."""
    base = upscale(lo, k).astype(np.float32)
    lum = lo.max(axis=2)
    w = np.clip((lum - float(np.max(bg)) - threshold) / 120.0, 0, 1)[..., None]
    src = (lo - np.asarray(bg, np.float32)) * w
    # blur at 2× low res for a smooth halo, then bilinear up to full res
    src2 = cv2.resize(src, (lo.shape[1] * 2, lo.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    g = cv2.GaussianBlur(src2, (0, 0), sigmaX=sigma * 2, sigmaY=sigma * 2)
    g = cv2.resize(g, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    # the halo lands AROUND the ink, not on it: the palette values were measured on already-glowing
    # reference frames, so adding glow onto the ink would over-brighten text and wash out panels
    ink = cv2.resize((w[..., 0] > 0).astype(np.float32), (base.shape[1], base.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    out = base + strength * g * (1.0 - ink)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class Scene:
    """Subclass and implement draw(cv, t). Set W, H, SCALE, FPS, DURATION, PAL, BLOOM.

    setup() runs once per process (keep it deterministic: seed every RNG) and may precompute
    anything. sfx() returns a list of (time, sound ndarray) for the soundtrack."""
    W, H, SCALE = 320, 180, 6
    FPS = 60
    DURATION = 10.0
    PAL = NEON
    BLOOM = None          # v2: dict(strength=0.5, sigma=2.6, threshold=20)
    LANG = 'en'

    def setup(self):
        pass

    @property
    def lf(self):
        """LCD font for labels."""
        return FONT57

    def draw(self, cv, t):
        raise NotImplementedError

    def sfx(self):
        return []

    def shake(self, t):
        """Whole-frame offset (dx, dy) in canvas pixels at time t. Default: none.
        Typical: return shake_offset(t, [(2.867, 'nudge'), (4.10, 'hit')]), 0"""
        return 0, 0

    # ─── helpers used by the render pipeline
    def frame_lo(self, t):
        cv = Canvas(self.W, self.H, self.PAL['bg'])
        self.draw(cv, t)
        dx, dy = self.shake(t)
        if dx or dy:
            return shift_frame(cv.a, int(dx), int(dy), self.PAL['bg'])
        return cv.a

    def frame(self, t):
        lo = self.frame_lo(t)
        if self.BLOOM:
            return bloom(lo, self.SCALE, self.PAL['bg'], **self.BLOOM)
        return upscale(lo, self.SCALE)


# multiprocessing plumbing (fork: the scene object is inherited by the workers)
_SCENE = None


def ensure_setup(scene):
    """Run scene.setup() once. Workers are forked from a parent that already ran it, so they inherit the
    precomputed state instead of recomputing it N times (and holding N copies in memory)."""
    if not getattr(scene, '_pk_setup_done', False):
        scene.setup()
        scene._pk_setup_done = True
    return scene


def _init_worker(scene):
    global _SCENE
    _SCENE = ensure_setup(scene)


def _render_one(i):
    t = i / _SCENE.FPS
    return _SCENE.frame(t).tobytes()


def frame_range(scene, t0=0.0, t1=None):
    """(first, last+1) frame indices for [t0, t1), clamped to the scene; shared by video and audio."""
    dur = float(scene.DURATION)
    t1 = dur if t1 is None else min(float(t1), dur)
    t0 = max(0.0, min(float(t0), t1))
    return int(round(t0 * scene.FPS)), int(round(t1 * scene.FPS))


def render(scene, out, t0=0.0, t1=None, workers=None, crf=14, preset='medium', audio=None, log=True):
    """Render scene frames [t0, t1) to an H.264 mp4 at W*SCALE × H*SCALE (clamped to DURATION).
    audio: optional wav, muxed as AAC and padded / trimmed to exactly the video length."""
    import multiprocessing as mp
    ensure_setup(scene)                       # in the parent, before forking
    if t1 is not None and t1 > scene.DURATION + 1e-9 and log:
        print(f'  note: t1={t1} is past DURATION={scene.DURATION}; clamped', flush=True)
    f0, f1 = frame_range(scene, t0, t1)
    W, H = scene.W * scene.SCALE, scene.H * scene.SCALE
    workers = workers or max(1, min(16, (os.cpu_count() or 4) - 4))
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    tmp = out + '.video.mp4' if audio else out
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(scene.FPS), '-i', '-', '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ctx = mp.get_context('fork')
    t_start = _time.time()
    n = f1 - f0
    with ctx.Pool(workers, initializer=_init_worker, initargs=(scene,)) as pool:
        for k, buf in enumerate(pool.imap(_render_one, range(f0, f1), chunksize=4)):
            ff.stdin.write(buf)
            if log and (k % 120 == 0 or k == n - 1):
                el = _time.time() - t_start
                print(f'  frame {k + 1}/{n}  {el:5.1f}s  ({(k + 1) / max(el, 1e-6):4.1f} fps)', flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError('ffmpeg failed')
    def count(path):
        return int(subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
                                   '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', path],
                                  capture_output=True, text=True).stdout.strip() or 0)
    got = count(tmp)
    if got != n:
        raise RuntimeError(f'frame count mismatch: wrote {n}, {tmp} has {got}')
    if audio:
        # the audio is padded with silence / cut to exactly n frames — never '-shortest' (drops video)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-i', audio, '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-af', 'apad', '-c:a', 'aac', '-b:a', '192k', '-t', f'{n / scene.FPS:.6f}',
                        '-movflags', '+faststart', out], check=True)
        got = count(out)
        if got != n:
            raise RuntimeError(f'frame count mismatch after mux: wrote {n}, {out} has {got} ({tmp} kept)')
        os.remove(tmp)
    if log:
        print(f'✓ {out}  {n} frames  {n / scene.FPS:.2f}s', flush=True)
    return out


def stills(scene, times, out, cols=4, thumb_scale=None):
    """Render selected times into a labelled contact sheet (for quick visual checks)."""
    from PIL import Image, ImageDraw, ImageFont
    ensure_setup(scene)
    ims = []
    k = thumb_scale or 2
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = None
    for t in times:
        lo = scene.frame_lo(t)
        pic = Image.fromarray(upscale(lo, k))
        im = Image.new('RGB', (pic.width, pic.height + 18), (0, 0, 0))   # label strip ABOVE the frame
        im.paste(pic, (0, 18))
        ImageDraw.Draw(im).text((3, 1), f'{t:6.2f} s', fill=(255, 255, 0), font=font)
        ims.append(im)
    w, h = ims[0].size
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new('RGB', (cols * w + (cols + 1) * 3, rows * h + (rows + 1) * 3), (90, 90, 90))
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        sheet.paste(im, (3 + c * (w + 3), 3 + r * (h + 3)))
    sheet.save(out)
    return out


def still(scene, t, out, full=False):
    """One frame as PNG (full=True → final 1920×1080 with post; else low-res ×2)."""
    from PIL import Image
    ensure_setup(scene)
    if full:
        Image.fromarray(scene.frame(t)).save(out)
    else:
        Image.fromarray(upscale(scene.frame_lo(t), 2)).save(out)
    return out


# ════════════════════════════════════════════════════════════ real imagery → pixel art

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16 - 0.5


def pixelate(img, w, h, palette=None, dither=0.0, contrast=1.0, saturation=1.0, crop='center'):
    """A photo / paper figure / video frame → a w×h pixel-art tile (how the references show camera views
    and video frames). Centre-crops to the target aspect, area-downsamples, optional saturation/contrast,
    optional palette snap with 4×4 Bayer ordered dithering (dither = amplitude in 0..255 units, ~24–48)."""
    a = as_rgb(img)
    H, W = a.shape[:2]
    if crop == 'center':
        r = w / h
        if W / H > r:
            nw = int(round(H * r))
            x0 = (W - nw) // 2
            a = a[:, x0:x0 + nw]
        else:
            nh = int(round(W / r))
            y0 = (H - nh) // 2
            a = a[y0:y0 + nh]
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if saturation != 1.0:
        g = a.mean(2, keepdims=True)
        a = g + (a - g) * saturation
    if contrast != 1.0:
        a = (a - 128.0) * contrast + 128.0
    a = np.clip(a, 0, 255)
    if palette is not None:
        if dither:
            yy, xx = np.mgrid[0:h, 0:w]
            a = np.clip(a + BAYER4[yy % 4, xx % 4][..., None] * dither, 0, 255)
        a = quantize(a, palette)
    return a


def as_rgb(img, matte=(255, 255, 255)):
    """Any PIL image / array (L, LA, P, RGB, RGBA; uint8 or float) → float RGB (h, w, 3).
    Transparency is flattened onto `matte` (white, like a paper page — a black matte turns soft-masked
    figures into black boxes)."""
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            img = img.convert('RGBA') if ('A' in img.getbands() or img.mode == 'P') else img.convert('RGB')
    except ImportError:
        pass
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[2] == 2:                                  # LA
        g, al = a[..., :1], a[..., 1:2] / 255.0
        a = np.concatenate([g] * 3, -1) * al + np.asarray(matte, np.float32) * (1 - al)
    elif a.shape[2] == 4:                                # RGBA
        al = a[..., 3:4] / 255.0
        a = a[..., :3] * al + np.asarray(matte, np.float32) * (1 - al)
    return a[..., :3].astype(np.float32)


def patch_grid(cv, x, y, w, h, p, color, alpha=0.55):
    """Token/patch grid over an image tile: a 1-px line every p pixels (ViT patches, 16×16 tokens …)."""
    for k in range(0, w + 1, p):
        cv.vline(x + k, y, y + h - 1, color, alpha)
    for k in range(0, h + 1, p):
        cv.hline(x, x + w - 1, y + k, color, alpha)


def video_frames(path, times, w=None, h=None):
    """Grab frames of a video at the given times (s) as float RGB arrays (optionally resized).
    Times past the end give the last frame (the container can outlast the video stream)."""
    info = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'stream=width,height:format=duration', '-of', 'default=nw=1', path],
                          capture_output=True, text=True).stdout
    kv = dict(line.split('=', 1) for line in info.strip().splitlines() if '=' in line)
    ww, hh = (w, h) if (w and h) else (int(kv['width']), int(kv['height']))
    dur = float(kv.get('duration', 0) or 0)
    out = []
    for t in times:
        t = max(0.0, min(float(t), dur))
        frame = None
        for back in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            cmd = ['ffmpeg', '-v', 'error', '-ss', f'{max(0.0, t - back):.3f}', '-i', path, '-frames:v', '1',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24']
            if w and h:
                cmd += ['-vf', f'scale={w}:{h}:flags=area']
            raw = subprocess.run(cmd + ['-'], capture_output=True).stdout
            if len(raw) >= ww * hh * 3:
                frame = np.frombuffer(raw[:ww * hh * 3], np.uint8).reshape(hh, ww, 3).astype(np.float32)
                break
        if frame is None:
            raise RuntimeError(f'could not read a frame near t={t:.3f}s from {path}')
        out.append(frame)
    return out
