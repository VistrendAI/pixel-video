"""Reusable diagram parts in the reference vocabulary (all sizes in canvas pixels).

Built from the same rules as the references: 1-px lines, whole pixels, 'rounded' boxes with missing
corner pixels, checker dither instead of transparency, colour = meaning.
`examples/widgets_gallery.py` draws every part on one frame (and is the visual index of this file).
"""
import math

import numpy as np

from .fonts import FONT35, FONT57
from .pk import Path, circle_mask, mixc

__all__ = ['rbox', 'label', 'tiny', 'label_box', 'token_box', 'arrow', 'link', 'bracket', 'stopgrad', 'bar_v',
           'module', 'wire', 'op_node', 'lock', 'magnify', 'grid_table',
           'layer_stack', 'mlp_symbol', 'node', 'value_color', 'vec_cells', 'matrix', 'row_scan', 'token_grid', 'tile',
           'tile_stack', 'line_plot', 'hist_fill', 'hbars', 'robot_arm', 'table', 'cup', 'block']


# ─────────────────────────────── boxes, labels, arrows

def rbox(cv, x, y, w, h, border, fill=None, alpha=1.0, dash=None):
    """Box with 1-px 'rounded' corners (corner pixels left out), like every box in the references."""
    if fill is not None:
        cv.fill(x + 1, y + 1, w - 2, h - 2, fill, alpha)
    if dash:                                   # dashed, but still with the missing corner pixels
        cv.hline(x + 1, x + w - 2, y, border, alpha, dash=dash)
        cv.hline(x + 1, x + w - 2, y + h - 1, border, alpha, dash=dash)
        cv.vline(x, y + 1, y + h - 2, border, alpha, dash=dash)
        cv.vline(x + w - 1, y + 1, y + h - 2, border, alpha, dash=dash)
        return
    cv.hline(x + 1, x + w - 2, y, border, alpha)
    cv.hline(x + 1, x + w - 2, y + h - 1, border, alpha)
    cv.vline(x, y + 1, y + h - 2, border, alpha)
    cv.vline(x + w - 1, y + 1, y + h - 2, border, alpha)


def label(cv, x, y, s, color, align='l', font=FONT57, alpha=1.0, n=None):
    return cv.text(x, y, s, color, alpha, font=font, align=align, n=n)


def tiny(cv, x, y, s, color, align='l', alpha=1.0):
    return cv.text(x, y, s, color, alpha, font=FONT35, align=align)


def label_box(cv, x, y, text, color, border=None, fill=None, pad=3, font=FONT57, n=None):
    """Text in a rounded box (a token, a module name). Returns the box width."""
    w = cv.text_width(text, font) + 2 * pad
    h = font.cap + 2 * pad
    rbox(cv, x, y, w, h, border or color, fill)
    cv.text(x + pad, y + pad, text, color, font=font, n=n)
    return w


def token_box(cv, x, y, w, text, pal, color, border=None, fill=None, n=None, alpha=1.0):
    rbox(cv, x, y, w, 13, border or color, fill, alpha)
    cv.text(x + w // 2, y + 3, text, color, alpha, align='c', n=n)


def arrow(cv, x0, y0, x1, y1, color, head=2, alpha=1.0, dash=None, frac=1.0):
    """Straight arrow; the head appears when the shaft is complete (frac = 1)."""
    cv.line(x0, y0, x1, y1, color, alpha, dash=dash, frac=frac)
    if frac < 1:
        return
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (+1, -1):
        a = ang + math.pi - s * math.pi / 4
        for k in range(1, head + 1):
            cv.px(round(x1 + k * math.cos(a)), round(y1 + k * math.sin(a)), color, alpha)


def link(cv, a, b, color, dash=(1, 1), via='hv', u=1.0, alpha=1.0):
    """Orthogonal connector a → b ('hv' = horizontal first, 'vh' = vertical first). Returns the Path
    (use it for particles)."""
    (x0, y0), (x1, y1) = a, b
    mid = (x1, y0) if via == 'hv' else (x0, y1)
    p = Path([a, mid, b])
    cv.path(p, color, alpha, dash=dash, u1=u)
    return p


def bracket(cv, x, y0, y1, color, side='r', depth=3, alpha=1.0):
    """A square bracket spanning y0..y1 at x. side='r' draws ']', side='l' draws '['."""
    d = -depth if side == 'r' else depth
    cv.vline(x, y0, y1, color, alpha)
    cv.hline(min(x, x + d), max(x, x + d), y0, color, alpha)
    cv.hline(min(x, x + d), max(x, x + d), y1, color, alpha)


def stopgrad(cv, x, y, color, alpha=1.0):
    """The '//' stop-gradient mark across a horizontal line at (x, y)."""
    for dx in (0, 3):
        cv.line(x + dx - 1, y + 2, x + dx + 1, y - 2, color, alpha)


def bar_v(cv, x, y_base, w, h, color, alpha=1.0):
    """Vertical bar standing on y_base (grows upward)."""
    h = int(round(h))
    if h > 0:
        cv.fill(x, y_base - h + 1, w, h, color, alpha)


# ─────────────────────────────── three-state parts (idle → computing → done)

def module(cv, x, y, w, h, state, pal, label=None, sub=None, font=FONT57, fill=None, alpha=1.0):
    """A module box in one of the three states the references use:
       'idle'   dim outline (the whole diagram is drawn like this from t≈0.3 s);
       'active' bright frame + inner fill while it computes (~0.2–0.4 s);
       'done'   stays tinted after it ran.
    label: name inside (top-left); sub: a small size label under the box ('27 × 1152')."""
    col = {'idle': pal['idle'], 'active': pal['active'], 'done': pal.get('done', pal['idle'])}[state]
    inner = fill
    if inner is None:
        inner = {'idle': None, 'active': mixc(pal['bg'], pal['fwd'], 0.35), 'done': mixc(pal['bg'], pal['fwd'], 0.18)}[state]
    rbox(cv, x, y, w, h, col, inner, alpha)
    if label:
        cv.text(x + 3, y + 3, label, pal['text'] if state == 'active' else pal.get('label', col), alpha, font=font)
    if sub:
        cv.text(x, y + h + 2, sub, pal.get('label', col), alpha, font=font)


def wire(cv, path, t, t0, travel, pal, hold=0.3, dash=(1, 1), n=3, gap=0.06, color=None, reverse=False):
    """A connection in the reference's three states: dim dashed while idle; particles run t0 … t0+travel;
    then the whole line is lit for `hold` s; then back to idle. color: the lit / trail colour (default fwd)."""
    from .pk import particles
    lit = color or pal['fwd']
    cv.path(path, pal['wire'], dash=dash)
    if t0 + travel <= t < t0 + travel + hold:
        cv.path(path, lit)
    particles(cv, path, t, t0, travel, n=n, gap=gap, color=pal['text'], trail=3, trail_color=lit, reverse=reverse)


def op_node(cv, x, y, op, color, r=4, alpha=1.0):
    """A circled operator: '+' (⊕ sum), '*' (⊗ product), '-' (⊖), '.' (⊙). (x, y) is the centre."""
    ys, xs = np.nonzero(circle_mask(r))
    cv.put(xs - r + x, ys - r + y, color, alpha)
    k = r - 2
    if op in '+-':
        cv.hline(x - k, x + k, y, color, alpha)
    if op == '+':
        cv.vline(x, y - k, y + k, color, alpha)
    if op == '*':
        cv.line(x - k, y - k, x + k, y + k, color, alpha)
        cv.line(x - k, y + k, x + k, y - k, color, alpha)
    if op == '.':
        cv.px(x, y, color, alpha)


def lock(cv, x, y, color, alpha=1.0):
    """A 5×7 padlock ('frozen weights'). (x, y) = top-left."""
    for yy, row in enumerate(['.###.', '#...#', '#...#', '#####', '##.##', '#####', '#####']):
        for xx, ch in enumerate(row):
            if ch == '#':
                cv.px(x + xx, y + yy, color, alpha)


def magnify(cv, x, y, w, h, X, Y, k, border, alpha=1.0):
    """Zoom inset: copy the canvas region (x, y, w, h) scaled by integer k to (X, Y), frame both, and join
    them with a dotted leader. For 'this one pixel is the whole LoRA update' moments."""
    src = cv.a[max(0, y):y + h, max(0, x):x + w].copy()
    big = np.repeat(np.repeat(src, k, 0), k, 1)
    cv.blit(big, X, Y, alpha)
    cv.rect(x - 1, y - 1, w + 2, h + 2, border, alpha)
    cv.rect(X - 1, Y - 1, big.shape[1] + 2, big.shape[0] + 2, border, alpha)
    cv.line(x + w, y + h // 2, X - 2, Y + big.shape[0] // 2, border, alpha, dash=(1, 1))


def grid_table(cv, x, y, rows, widths, colors, font=FONT57, line_h=None, align=None, header_color=None, alpha=1.0):
    """A small text table. rows: list of rows (strings); widths: column widths in px; align: 'l'/'r' per
    column (numbers right-aligned); colors: one colour per column (or per row if a list of lists);
    header_color: colour for row 0. Returns the table height."""
    lh = line_h or font.height + 3
    align = align or ['l'] + ['r'] * (len(widths) - 1)
    for i, row in enumerate(rows):
        cx = x
        for j, cell in enumerate(row):
            col = header_color if (i == 0 and header_color) else (colors[i][j] if isinstance(colors[0], list) else colors[j])
            if align[j] == 'r':
                cv.text(cx + widths[j] - 1, y + i * lh, str(cell), col, alpha, font=font, align='r')
            else:
                cv.text(cx, y + i * lh, str(cell), col, alpha, font=font)
            cx += widths[j]
    return len(rows) * lh


# ─────────────────────────────── networks

def layer_stack(cv, x, y, h, n, color, dim, lit=1.0, pitch=2, taps=(), tap_color=None, tap_labels=True,
                alpha=1.0):
    """A deep network as n thin vertical bars (V-JEPA's 48-block ViT). Bars 0 … lit·n are lit.
    taps: 1-based block indices marked above and labelled below. Returns the stack width."""
    k = int(round(lit * n))
    for i in range(n):
        cv.vline(x + i * pitch, y, y + h - 1, color if i < k else dim, alpha)
    for b in taps:
        xx = x + (b - 1) * pitch
        cv.vline(xx, y - 3, y - 1, tap_color or color, alpha)
        if tap_labels:
            cv.text(xx, y + h + 3, str(b), tap_color or color, alpha, font=FONT35, align='c')
    return (n - 1) * pitch + 1


def mlp_symbol(cv, cx, cy, w, h, color, alpha=1.0):
    """The checker-filled funnel the references draw for 'an MLP' (wide end on the left)."""
    for i in range(w):
        half = int(round((h / 2) * (1 - 0.55 * i / max(1, w - 1))))
        x = cx - w // 2 + i
        cv.px(x, cy - half, color, alpha)
        cv.px(x, cy + half, color, alpha)
        for yy in range(cy - half + 1, cy + half):
            if (x + yy) % 2 == 0:
                cv.px(x, yy, color, alpha)
    cv.vline(cx - w // 2, cy - h // 2, cy + h // 2, color, alpha)


def node(cv, x, y, color, fill=None, r=4, alpha=1.0):
    """A neuron: 1-px ring of radius r, optionally filled."""
    if fill is not None:
        ys, xs = np.nonzero(circle_mask(r - 1, fill=True))
        cv.put(xs - (r - 1) + x, ys - (r - 1) + y, fill, alpha)
    ys, xs = np.nonzero(circle_mask(r))
    cv.put(xs - r + x, ys - r + y, color, alpha)


# ─────────────────────────────── data

def value_color(v, accent, dim, vmax=1.0):
    """Magnitude → brightness of the accent colour (dim at 0)."""
    u = min(1.0, abs(float(v)) / vmax)
    return mixc(dim, accent, 0.25 + 0.75 * u)


def vec_cells(cv, x, y, values, accent, dim, cell=4, gap=1, vmax=1.0, vertical=True, alpha=1.0, mask=None, zero=None):
    """A vector as a strip of cells. Positive = solid cell, negative = checker-dithered cell
    (sign without a second hue). mask: optional boolean per cell (False = not drawn yet).
    zero=eps: |value| ≤ eps drawn as a hollow cell ('exactly zero')."""
    for i, v in enumerate(values):
        if mask is not None and not mask[i]:
            continue
        cx = x if vertical else x + i * (cell + gap)
        cy = y + i * (cell + gap) if vertical else y
        if zero is not None and abs(float(v)) <= zero:
            cv.rect(cx, cy, cell, cell, mixc(dim, accent, 0.35), alpha)
            continue
        c = value_color(v, accent, dim, vmax)
        if v >= 0:
            cv.fill(cx, cy, cell, cell, c, alpha)
        else:
            cv.fill(cx, cy, cell, cell, dim, alpha)
            cv.checker(cx, cy, cell, cell, c, alpha, parity=0)


def matrix(cv, x, y, M, cell, lo, hi, gap=1, mask=None, vmin=None, vmax=None, alpha=1.0, signed=False,
           zero=None, reveal=None):
    """Heat map of a 2-D array.
    signed=False: colour ramps lo → hi over [vmin, vmax].
    signed=True : brightness = |value| / max|value| (like vec_cells), negative cells checker-dithered.
    zero=eps    : cells with |value| ≤ eps drawn as a hollow outline — 'exactly zero' (LoRA's B at init).
    mask        : boolean array; False cells become a dim checker (e.g. the causal mask).
    reveal      : optional boolean array; False cells are not drawn (animate a fill-in)."""
    M = np.asarray(M, np.float64)
    a = M.min() if vmin is None else vmin
    b = M.max() if vmax is None else vmax
    amax = np.abs(M).max() if vmax is None else max(abs(a), abs(b))
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if reveal is not None and not reveal[r, c]:
                continue
            xx, yy = x + c * (cell + gap), y + r * (cell + gap)
            if mask is not None and not mask[r, c]:
                cv.checker(xx, yy, cell, cell, lo, alpha, parity=(r + c) % 2)
                continue
            v = M[r, c]
            if zero is not None and abs(v) <= zero:
                cv.rect(xx, yy, cell, cell, mixc(lo, hi, 0.35), alpha)
                continue
            if signed:
                u = 0.0 if amax <= 0 else min(1.0, abs(v) / amax)
                col = mixc(lo, hi, 0.2 + 0.8 * u)
                if v >= 0:
                    cv.fill(xx, yy, cell, cell, col, alpha)
                else:
                    cv.fill(xx, yy, cell, cell, lo, alpha)
                    cv.checker(xx, yy, cell, cell, col, alpha)
            else:
                u = 0.0 if b <= a else (v - a) / (b - a)
                cv.fill(xx, yy, cell, cell, mixc(lo, hi, u), alpha)


def row_scan(cv, x, y, rows, cols, cell, u, color, gap=1, done_color=None, alpha=1.0):
    """Matrix × vector read-out: a frame sweeps row by row (u: 0→1); finished rows get a dim tick on the right.
    Draw it over matrix(); pair with vec_cells for the result filling in one cell per row."""
    k = min(rows - 1, int(u * rows)) if u < 1 else rows
    w = cols * (cell + gap) - gap
    for r in range(min(k, rows)):
        cv.px(x + w + 1, y + r * (cell + gap) + cell // 2, done_color or color, alpha)
    if u < 1:
        cv.rect(x - 1, y + k * (cell + gap) - 1, w + 2, cell + 2, color, alpha)


def token_grid(cv, x, y, cols, rows, cell, color, state=None, masked_color=None, gap=1, alpha=1.0):
    """Grid of tokens / patches. state: (rows, cols) array, 1 = visible (solid), 0 = masked (hollow)."""
    for r in range(rows):
        for c in range(cols):
            xx, yy = x + c * (cell + gap), y + r * (cell + gap)
            if state is None or state[r, c]:
                cv.fill(xx, yy, cell, cell, color, alpha)
            else:
                cv.rect(xx, yy, cell, cell, masked_color or color, alpha)


def tile(cv, img, x, y, border=None, alpha=1.0):
    """An image tile (already pixel-sized, e.g. from pixelate()) with an optional 1-px frame."""
    cv.blit(img, x, y, alpha)
    if border is not None:
        h, w = np.asarray(img).shape[:2]
        cv.rect(x - 1, y - 1, w + 2, h + 2, border, alpha)


def tile_stack(cv, imgs, x, y, dx=3, dy=-3, border=None, alpha=1.0):
    """Stacked feature maps (V-JEPA's '4 levels'): drawn back to front with an offset."""
    for i in range(len(imgs) - 1, -1, -1):
        tile(cv, imgs[i], x + i * dx, y + i * dy, border, alpha)


def line_plot(cv, x, y, w, h, series, colors, u=1.0, ylim=None, axis=None, alpha=1.0):
    """Pixel line chart. series: list of 1-D arrays, drawn left→right up to fraction u (animate u)."""
    allv = np.concatenate([np.asarray(s, np.float64).ravel() for s in series])
    lo, hi = (allv.min(), allv.max()) if ylim is None else ylim
    if hi <= lo:
        hi = lo + 1
    if axis is not None:
        cv.hline(x, x + w - 1, y + h, axis, alpha)
        cv.vline(x - 1, y, y + h, axis, alpha)
    for s, col in zip(series, colors):
        s = np.asarray(s, np.float64)
        n = len(s)
        pts = [(x + int(round(i * (w - 1) / max(1, n - 1))),
                y + h - 1 - int(round((min(hi, max(lo, v)) - lo) / (hi - lo) * (h - 1)))) for i, v in enumerate(s)]
        cv.path(Path(pts), col, alpha, u1=u)


def hist_fill(cv, x, y, w, h, values, color, marker=None, marker_color=None, alpha=1.0):
    """Filled distribution / area chart: checker-dithered body, solid top edge (π0.5's Beta(1.5, 1)).
    marker: 0..1 position of a vertical marker line."""
    v = np.asarray(values, np.float64)
    v = v / max(1e-12, v.max())
    xs = np.linspace(0, len(v) - 1, w)
    tops = []
    for i in range(w):
        hh = int(round(np.interp(xs[i], np.arange(len(v)), v) * (h - 1)))
        top = y + h - 1 - hh
        tops.append(top)
        for yy in range(top + 1, y + h):
            if (x + i + yy) % 2 == 0:
                cv.px(x + i, yy, color, alpha * 0.8)
    cv.path(Path([(x + i, t) for i, t in enumerate(tops)]), color, alpha)
    cv.hline(x, x + w - 1, y + h, color, alpha)
    if marker is not None:
        mx = x + int(round(marker * (w - 1)))
        cv.vline(mx, y, y + h, marker_color or color, alpha)


def hbars(cv, x, y, values, w_max, color, pitch=10, height=2, hot=None, hot_color=None, alpha=1.0):
    """Horizontal bars (class probabilities), one per row, length ∝ value; row `hot` highlighted."""
    v = np.asarray(values, np.float64)
    for i, val in enumerate(v):
        L = max(1, int(round(w_max * val / max(1e-12, v.max()))))
        cv.fill(x, y + i * pitch, L, height, hot_color if (hot == i and hot_color) else color, alpha)


# ─────────────────────────────── robots (robot-learning papers)

def robot_arm(cv, base, q, lengths, color, width=3, joint=None, grip=1.0, grip_color=None, alpha=1.0):
    """Planar serial arm by forward kinematics, drawn like π0.5's: thick straight links, round joints,
    a two-finger gripper. base = shoulder (x, y); q = relative joint angles in rad (0 = +x, positive =
    counter-clockwise on screen); grip = 0 closed … 1 open. Returns the tool point (x, y)."""
    x, y = float(base[0]), float(base[1])
    ang = 0.0
    pts = [(x, y)]
    for qi, L in zip(q, lengths):
        ang += qi
        x += L * math.cos(ang)
        y -= L * math.sin(ang)
        pts.append((x, y))
    half = width // 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        nx, ny = -(y1 - y0), (x1 - x0)
        nrm = math.hypot(nx, ny) or 1
        for o in range(-half, half + 1):
            ox, oy = o * nx / nrm, o * ny / nrm
            cv.line(round(x0 + ox), round(y0 + oy), round(x1 + ox), round(y1 + oy), color, alpha)
    ys, xs = np.nonzero(circle_mask(max(1, half + 1), fill=True))
    for (px_, py_) in pts[:-1]:
        cv.put(xs - (half + 1) + round(px_), ys - (half + 1) + round(py_), joint or color, alpha)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = math.hypot(x1 - x0, y1 - y0) or 1
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    px_, py_ = -uy, ux
    spread = 1.5 + 2.5 * grip
    gc = grip_color or color
    cv.line(round(x1 - 3 * px_), round(y1 - 3 * py_), round(x1 + 3 * px_), round(y1 + 3 * py_), gc, alpha)  # palm
    for s in (+1, -1):
        fx, fy = x1 + s * spread * px_, y1 + s * spread * py_
        cv.line(round(fx), round(fy), round(fx + 5 * ux), round(fy + 5 * uy), gc, alpha)            # jaw
    return (x1 + 5 * ux, y1 + 5 * uy)


def table(cv, x0, x1, y, color, leg=None, alpha=1.0):
    cv.hline(x0, x1, y, color, alpha)
    cv.hline(x0, x1, y + 1, mixc(color, (0, 0, 0), 0.4), alpha)
    if leg:
        cv.vline(x0 + 3, y + 2, y + leg, color, alpha)
        cv.vline(x1 - 3, y + 2, y + leg, color, alpha)


def cup(cv, x, y, color, h=9, w=7, alpha=1.0):
    """A mug standing on y (x = left edge)."""
    cv.fill(x, y - h + 1, w, h, color, alpha)
    cv.vline(x + w, y - h + 3, y - 3, color, alpha)
    cv.vline(x + w + 1, y - h + 3, y - 3, color, alpha)


def block(cv, x, y, color, s=5, alpha=1.0):
    """A small cube / object standing on y."""
    cv.fill(x, y - s + 1, s, s, color, alpha)
    cv.hline(x, x + s - 1, y - s + 1, mixc(color, (255, 255, 255), 0.35), alpha)
