"""KV CACHE — an original piece in the v2 'phosphor' style (480×270 ×4, header + typewriter captions,
dot grid, bloom). Real attention maths on a toy model (d = 8, 1 head, random weights); the words
are illustrative, and the captions say so.

    python3 examples/kv_cache.py                 # English, with 8-bit SFX
    python3 examples/kv_cache.py --stills        # contact sheet for checking
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pixelkit import (PHOSPHOR, Path, Scene, ghost_layer, mixc, particles, render, reveal_layer, seg, sfx,  # noqa: E402
                      stills)
from pixelkit import chrome  # noqa: E402
from pixelkit.widgets import rbox, tiny, token_box  # noqa: E402

P = PHOSPHOR
BLUE, ORANGE, CREAM, YEL = P['blue'], P['orange'], P['cream'], P['yellow']
DIM = (40, 48, 84)
TEXT, TEXT2, LABEL = P['text'], P['text2'], P['label']

# ─────────────────────────────── the toy model (real maths)
D = 8
TOKENS = ['THE', 'ROBOT', 'PICKS', 'UP', 'THE', 'RED', 'BLOCK', '.']
N_PROMPT = 5
rng = np.random.default_rng(7)
E = rng.normal(size=(len(TOKENS), D)) * 0.9
WQ, WK, WV = (rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(3))
Q, K, V = E @ WQ, E @ WK, E @ WV
VMAX = float(np.percentile(np.abs(np.concatenate([Q, K, V])), 95))


def attn_row(p):
    """weights of query p over keys 0..p (causal)"""
    s = Q[p] @ K[:p + 1].T / np.sqrt(D)
    w = np.exp(s - s.max())
    return s, w / w.sum()


OUT = {p: attn_row(p)[1] @ V[:p + 1] for p in range(len(TOKENS))}

# ─────────────────────────────── layout (480×270)
def tok_x(i):
    return 48 + 40 * i


TOK_Y, TOK_W = 24, 36
Q_Y, K_Y, V_Y = 46, 94, 142          # three rows of 8-cell columns (8×(4+1) = 40 px tall)
COL_DX = 14                          # column x offset inside a token slot (cells 8 px wide)
SCORE_Y, ATTN_Y = 188, 202           # score bars / attention-weight cells (under V)
OUT_X = 382                          # the attention output vector o
MAT_X, MAT_Y, CELL = 404, 46, 8      # right panel: the growing n×n attention matrix
CACHE_BOX = (42, 91, 332, 95)        # frame around the K and V rows (2 px below the q→k scan wire at y=89)


def col_x(i):
    return tok_x(i) + COL_DX


# ─────────────────────────────── timeline
T_PREFILL, T_CACHE, T_DECODE, T_COST, T_END = 3.2, 8.6, 11.0, 19.4, 25.2
DURATION = 26.4
PROJ = (3.4, 4.2)                     # prefill projections: q, k, v for all prompt tokens
ROW0, ROW_DT = 4.4, 0.4               # attention matrix rows fill
OUT0 = (6.6, 7.4)                     # last prompt row → output o
NEXT0 = (7.5, 8.2)                    # o → next token (RED)
DROP_Q = (8.7, 9.3)
CACHE_FRAME = (9.0, 9.8)
STEPS = [dict(p=5, t=11.0, k=1.0), dict(p=6, t=15.3, k=0.75)]   # decode steps (k = tempo scale)


def step_times(s):
    t, k = s['t'], s['k']
    return dict(hl=t, proj=(t + 0.3 * k, t + 0.9 * k), scan=(t + 1.1 * k, t + 2.1 * k),
                soft=(t + 2.2 * k, t + 2.7 * k), mix=(t + 2.8 * k, t + 3.4 * k), nxt=(t + 3.5 * k, t + 4.1 * k))


TXT = {
    'en': dict(
        stages=['PROMPT', 'PREFILL', 'CACHE', 'DECODE', 'COST'],
        status=[[(0.0, '5 TOKENS · d = 8 · 1 HEAD')],
                [(T_PREFILL, '5 TOKENS → q, k, v'), (ROW0, '5 × 5 SCORES · CAUSAL MASK'),
                 (OUT0[0], 'LAST ROW → NEXT TOKEN')],
                [(T_CACHE, 'KEEP k, v · DROP q')],
                [(STEPS[0]['t'], 'STEP 1 · 1 NEW k, v · 1 × 6 SCORES'),
                 (STEPS[1]['t'], 'STEP 2 · 1 NEW k, v · 1 × 7 SCORES')],
                [(T_COST, '24 STEPS · 300 VS 24 k, v')]],
        caps=[('A PROMPT OF 5 TOKENS. EACH ONE IS AN 8-D VECTOR',
               'TOY MODEL: 1 HEAD, d = 8, RANDOM WEIGHTS. REAL MATH, ILLUSTRATIVE WORDS'),
              ('PREFILL: THE WHOLE PROMPT GOES THROUGH IN ONE PASS',
               'q, k, v = x·W FOR EVERY TOKEN. SCORES q·k/√8, CAUSAL MASK, SOFTMAX PER ROW'),
              ('k AND v ARE KEPT: PAST TOKENS NEVER CHANGE, SO NEITHER DO THEIR k, v',
               'q IS DROPPED. A PAST TOKEN NEVER ASKS AGAIN. THIS IS THE KV CACHE'),
              ('EACH NEW TOKEN ADDS ONE k, v AND ONE ROW OF SCORES',
               'ITS q MEETS EVERY CACHED k. THE WEIGHTS MIX THE CACHED v INTO ITS OUTPUT'),
              ('WITHOUT A CACHE, STEP n RECOMPUTES k, v FOR ALL n TOKENS',
               'WITH IT, ONE PER STEP. THE PRICE IS MEMORY: 2 × LAYERS × HEADS × d × n')],
        lab=dict(q='q', k='k', v='v', score='q·k', attn='w', out='o', cache='KV CACHE', nocache='NO CACHE',
                 withcache='KV CACHE', axis='STEP n', perstep='k, v COMPUTED PER STEP', mem='MEMORY 2·n·d'),
    ),
}


class KVCache(Scene):
    W, H, SCALE = 480, 270, 4
    FPS = 60
    DURATION = DURATION
    PAL = PHOSPHOR
    BLOOM = dict(strength=0.5, sigma=2.6, threshold=20)
    LANG = 'en'

    def setup(self):
        T = TXT[self.LANG]
        self.T = T
        stage_t = [0.0, T_PREFILL, T_CACHE, T_DECODE, T_COST]
        self.stages = chrome.Stages([dict(name=n, t=t0, status=st, cap=c)
                                     for n, t0, st, c in zip(T['stages'], stage_t, T['status'], T['caps'])],
                                    lang=self.LANG)
        # copy is checked by `pxv check` / `pxv render` (Stages.check) — no assert here, so check can report
        # Labels use the 5×7 LCD font.
        self.idle = self.build_idle()

    def build_idle(self):
        """The whole map, dim, as it sits on screen from t≈0.3 s (the reference rule): every token slot,
        every q/k/v column's 8 empty cells, the cache frame, the full 8×8 attention grid, the output column.
        Stages only fill and light parts of it."""
        from pixelkit import Canvas
        lay = Canvas(self.W, self.H, (0, 0, 0), layer=True)
        faint = (30, 36, 64)
        for i in range(len(TOKENS)):
            rbox(lay, tok_x(i), TOK_Y, TOK_W, 13, P['idle'], dash=(1, 1))
            for y in (Q_Y, K_Y, V_Y):
                for j in range(D):
                    lay.rect(col_x(i), y + j * 5, 8, 4, faint)
        for y, lbl in ((Q_Y + 16, self.T['lab']['q']), (K_Y + 16, self.T['lab']['k']), (V_Y + 16, self.T['lab']['v'])):
            lay.text(20, y, lbl, LABEL, align='c')
        x, y, w, h = CACHE_BOX
        rbox(lay, x, y, w, h, P['idle'], dash=(1, 1))
        n = len(TOKENS)
        rbox(lay, MAT_X - 2, MAT_Y - 2, n * (CELL + 1) + 3, n * (CELL + 1) + 3, P['idle'])
        for r in range(n):
            for c in range(n):
                lay.rect(MAT_X + c * (CELL + 1), MAT_Y + r * (CELL + 1), CELL, CELL, faint)
        lay.text(MAT_X - 2, MAT_Y - 12, 'n × n', LABEL)
        for j in range(D):
            lay.rect(OUT_X, V_Y + j * 5, 8, 4, faint)
        lay.text(OUT_X + 4, V_Y - 10, self.T['lab']['out'], LABEL, align='c')
        return lay

    # ─────────── helpers
    def tok_state(self, t, i):
        """(visible fraction, border colour, text colour) for token i"""
        if i < N_PROMPT:
            t0 = 0.25 + 0.18 * i
            return seg(t, t0, t0 + 0.25), CREAM, TEXT
        if i == 5:
            return seg(t, *NEXT0), ORANGE, TEXT
        if i == 6:
            return seg(t, *step_times(STEPS[0])['nxt']), ORANGE, TEXT
        return seg(t, *step_times(STEPS[1])['nxt']), ORANGE, TEXT

    def qkv_visible(self, t, i):
        """reveal fraction of token i's k, v columns (0..1) and of its q column"""
        if i < N_PROMPT:
            u = seg(t, *PROJ)
            q = u if t < DROP_Q[0] else 1 - seg(t, *DROP_Q)
            return u, q
        for s in STEPS:
            if s['p'] == i:
                st = step_times(s)
                u = seg(t, *st['proj'])
                end = st['nxt'][1] + 0.1
                q = u if t < end else 1 - seg(t, end, end + 0.3)
                return u, q
        return 0.0, 0.0

    def active_query(self, t):
        """(position, times) of the query currently being used in the decode loop, else None"""
        for s in STEPS:
            st = step_times(s)
            if st['hl'] <= t < st['nxt'][1] + 0.1:
                return s['p'], st
        return None

    # ─────────── frame
    def draw(self, cv, t):
        chrome.background(cv)
        cost_u = seg(t, T_COST, T_COST + 0.6)
        end_u = seg(t, T_END, T_END + 0.6)
        if cost_u < 1:
            lay = cv.layer()
            self.draw_diagram(lay, t)
            if cost_u > 0:
                reveal_layer(cv, lay, cost_u, seed=11, mode='out')
            else:
                cv.composite(lay)
        if t >= T_COST + 0.6:                     # the chart only starts once the diagram has fully dissolved
            lay = cv.layer()
            self.draw_cost(lay, t)
            u = seg(t, T_COST + 0.6, T_COST + 1.1)
            if end_u > 0:
                reveal_layer(cv, lay, end_u, seed=12, mode='out')
            elif u < 1:
                reveal_layer(cv, lay, u, seed=13, mode='in', sparkle_color=TEXT)
            else:
                cv.composite(lay)
        self.stages.draw(cv, t, 'KV CACHE', P)

    # the main mechanism view
    def draw_diagram(self, cv, t):
        lab = self.T['lab']
        # the dim map: a checker ghost while booting, then always there
        if t >= 0.05:
            if t < 0.3:
                ghost_layer(cv, self.idle, alpha=0.8)
            else:
                cv.composite(self.idle)
        # cache frame: the dim dashed frame is part of the map; the CACHE stage lights it
        cu = seg(t, *CACHE_FRAME)
        x, y, w, h = CACHE_BOX
        if cu >= 0.5:
            rbox(cv, x, y, w, h, ORANGE if t < CACHE_FRAME[1] + 0.4 else P['dim2'])
        hot = CACHE_FRAME[0] <= t < CACHE_FRAME[1] + 0.4
        if t >= 0.3:                                   # one label, dim → orange while the cache forms → label grey
            cv.text(x + w - 3, y - 9, lab['cache'], ORANGE if hot else (LABEL if cu > 0 else P['idle']),
                    align='r', font=self.lf)
        aq = self.active_query(t)
        # tokens + their q/k/v columns
        for i, tok in enumerate(TOKENS):
            u, border, tc = self.tok_state(t, i)
            x = tok_x(i)
            if u > 0:
                lay = cv.layer()
                hl = aq is not None and aq[0] == i
                fill = (46, 30, 28) if hl else None
                token_box(lay, x, TOK_Y, TOK_W, tok, P, tc, border=border, fill=fill)
                reveal_layer(cv, lay, u, seed=30 + i, mode='in', sparkle_color=TEXT)
            kv_u, q_u = self.qkv_visible(t, i)
            if kv_u > 0:
                cells = int(np.ceil(kv_u * D))
                mask = [j < cells for j in range(D)]
                bright = self.kv_bright(t, i)
                self.cells(cv, col_x(i), K_Y, K[i], ORANGE, bright, mask)
                self.cells(cv, col_x(i), V_Y, V[i], CREAM, bright, mask)
            if q_u > 0:
                cells = int(np.ceil(q_u * D))
                self.cells(cv, col_x(i), Q_Y, Q[i], BLUE, 1.0, [j < cells for j in range(D)])
        # projection particles: token → its q, k, v
        for i in range(N_PROMPT):
            self.proj_particles(cv, t, i, PROJ[0] - 0.05)
        for s in STEPS:
            self.proj_particles(cv, t, s['p'], step_times(s)['proj'][0] - 0.05)
        # prefill: attention rows fill in the right panel; last row mixes v into o → next token
        self.draw_matrix(cv, t)
        if OUT0[0] <= t < T_CACHE + 0.3:
            self.mix_to_out(cv, t, N_PROMPT - 1, OUT0, NEXT0, fade=(T_CACHE, T_CACHE + 0.3))
        # decode steps
        if aq is not None:
            p, st = aq
            self.decode_step(cv, t, p, st)

    def cells(self, cv, x, y, vals, accent, bright, mask):
        """an 8-D vector as 8 cells (8×4 px): brightness = |value|, checker = negative"""
        acc = mixc(mixc(DIM, accent, 0.45), accent, bright)
        for j, v in enumerate(vals):
            if not mask[j]:
                continue
            cy = y + j * 5
            a = min(1.0, abs(v) / VMAX)
            c = mixc(DIM, acc, 0.25 + 0.75 * a)
            if v >= 0:
                cv.fill(x, cy, 8, 4, c)
            else:
                cv.fill(x, cy, 8, 4, DIM)
                cv.checker(x, cy, 8, 4, c)

    def kv_bright(self, t, i):
        """1 = freshly computed (bright), 0 = sitting in the cache (dimmer)"""
        if i < N_PROMPT:
            if t < T_CACHE:
                return 1.0
            return 1.0 - seg(t, T_CACHE, T_CACHE + 0.8) * 0.55
        for s in STEPS:
            if s['p'] == i:
                st = step_times(s)
                return 1.0 - seg(t, st['nxt'][1], st['nxt'][1] + 0.6) * 0.55
        return 1.0

    def proj_particles(self, cv, t, i, t0):
        x = tok_x(i) + TOK_W // 2
        for y1, col in ((Q_Y, BLUE), (K_Y, ORANGE), (V_Y, CREAM)):
            p = Path([(x, TOK_Y + 13), (x, y1 - 1)])
            particles(cv, p, t, t0, 0.35, n=2, gap=0.08, color=TEXT, trail=3, trail_color=col)

    def draw_matrix(self, cv, t):
        """Right panel: the n×n causal attention matrix, one row per query."""
        rows = []
        for r in range(N_PROMPT):
            rows.append((r, ROW0 + ROW_DT * r))
        for s in STEPS:
            rows.append((s['p'], step_times(s)['soft'][0]))
        n_vis = 0
        for r, tr in rows:
            if t >= tr - 0.05:
                n_vis = r + 1
        if n_vis == 0 and t < ROW0 - 0.3:
            return
        n_frame = max(N_PROMPT, n_vis)            # the 8×8 frame and grid are part of the dim map
        for r, tr in rows:
            if t < tr - 0.05:
                continue
            _, wts = attn_row(r)
            u = seg(t, tr - 0.05, tr + 0.2)
            newest = r == n_vis - 1 and t < tr + (0.9 if r >= N_PROMPT else 0.5)
            prefill_live = r < N_PROMPT and t < T_CACHE
            for c in range(n_frame):
                x = MAT_X + c * (CELL + 1)
                y = MAT_Y + r * (CELL + 1)
                if c > r:                          # causal mask: never computed
                    cv.checker(x, y, CELL, CELL, (30, 36, 64), parity=(r + c) % 2)
                    continue
                if c > u * (r + 1):
                    continue
                hot = wts[c] / wts.max()
                base = YEL if (newest or prefill_live) else mixc(DIM, YEL, 0.45)
                cv.fill(x, y, CELL, CELL, mixc((34, 34, 40), base, 0.2 + 0.8 * hot))
            if newest:
                cv.rect(MAT_X - 1, MAT_Y + r * (CELL + 1) - 1, (r + 1) * (CELL + 1) + 1, CELL + 2, TEXT)

    def mix_to_out(self, cv, t, p, mix, nxt, fade=None):
        """weights row under V, particles from v columns into o, then o → next token"""
        _, wts = attn_row(p)
        alpha = 1.0 - (seg(t, *fade) if fade else 0.0)
        # weights row
        for c in range(p + 1):
            x = col_x(c) - 3
            hot = wts[c] / wts.max()
            cv.fill(x, ATTN_Y, 14, 9, mixc((34, 34, 40), YEL, 0.2 + 0.8 * hot), alpha)
            pct = int(round(wts[c] * 100))
            tiny(cv, x + 7, ATTN_Y + 2, f'{pct}', (20, 16, 10) if hot > 0.55 else TEXT2, align='c', alpha=alpha)
        cv.text(20, ATTN_Y + 1, self.T['lab']['attn'], LABEL, align='c', alpha=alpha)
        # v → o particles (count ∝ weight)
        for c in range(p + 1):
            k = max(1, int(round(wts[c] * 8)))
            path = Path([(col_x(c) + 4, V_Y + 40), (col_x(c) + 4, V_Y + 44), (OUT_X + 4, V_Y + 44),
                         (OUT_X + 4, V_Y + 41)])
            particles(cv, path, t, mix[0] + 0.02 * c, (mix[1] - mix[0]) * 0.7, n=k, gap=0.03, color=TEXT, trail=2,
                      trail_color=YEL)
        u = seg(t, mix[0] + 0.25, mix[1])
        if u > 0:
            cells = int(np.ceil(u * D))
            self.cells(cv, OUT_X, V_Y, OUT[p], TEXT2, 1.0, [j < cells and alpha > 0.5 for j in range(D)])
        # o → next token slot
        tgt = tok_x(p + 1) + TOK_W // 2
        path = Path([(OUT_X + 4, V_Y - 12), (OUT_X + 4, TOK_Y + 20), (tgt, TOK_Y + 20), (tgt, TOK_Y + 14)])
        if nxt[0] - 0.1 <= t < nxt[1] + 0.2:
            cv.path(path, mixc(DIM, ORANGE, 0.8), u1=seg(t, nxt[0] - 0.1, nxt[0] + 0.25))
        particles(cv, path, t, nxt[0] - 0.1, 0.35, n=3, gap=0.05, color=TEXT, trail=3, trail_color=ORANGE)

    def decode_step(self, cv, t, p, st):
        # scan: q_p meets every cached k (0..p)
        s, wts = attn_row(p)
        u = seg(t, *st['scan'])
        n_done = int(u * (p + 1) + 1e-6)
        qx = col_x(p) + 4
        for c in range(p + 1):
            if c < n_done or (u >= 1):
                # score bar under V (height ∝ softmax score before normalisation)
                h = max(1, int(round(9 * (np.exp(s[c] - s.max())))))
                x = col_x(c)
                if t < st['soft'][0] + 0.15:
                    cv.fill(x, SCORE_Y + 9 - h, 8, h, BLUE)
        if st['scan'][0] <= t < st['scan'][1]:
            c = min(p, n_done)
            x = col_x(c)
            cv.rect(x - 1, K_Y - 1, 10, 41, TEXT)                  # the key being read right now
            path = Path([(qx, Q_Y + 40), (qx, Q_Y + 43), (x + 4, Q_Y + 43), (x + 4, K_Y - 2)])
            cv.path(path, mixc(DIM, BLUE, 0.9))
        if st['scan'][0] <= t < st['soft'][0] + 0.15:
            cv.text(20, SCORE_Y + 1, self.T['lab']['score'], LABEL, align='c')
        if t >= st['soft'][0]:
            self.mix_to_out(cv, t, p, st['mix'], st['nxt'], fade=(st['nxt'][1] + 0.05, st['nxt'][1] + 0.35))

    # the closing chart: k, v projections per step with and without a cache
    def draw_cost(self, cv, t):
        lab = self.T['lab']
        x0, y0, w, h = 60, 40, 360, 150       # plot area (y0 = top)
        base = y0 + h
        n_steps = 24
        bw = 6
        pitch = w // n_steps
        grow = seg(t, T_COST + 0.8, T_COST + 3.3)
        k_now = int(grow * n_steps + 1e-6)
        cv.hline(x0 - 4, x0 + w, base + 1, P['dim2'])
        cv.text(x0 + w, base + 14, lab['axis'], LABEL, align='r', font=self.lf)
        cv.text(x0 - 4, y0 - 16, lab['perstep'], LABEL, font=self.lf)
        tot_no = tot_yes = 0
        for n in range(1, n_steps + 1):
            if n > k_now:
                break
            x = x0 + (n - 1) * pitch
            hn = n * (h / n_steps)
            cv.fill(x, base - int(hn) + 1, bw, int(hn), mixc(DIM, ORANGE, 0.85))
            cv.fill(x + 1, base - 5 + 1, bw - 2, 5, BLUE)   # with a cache: one k, v per step
            tot_no += n
            tot_yes += 1
            if n in (1, 8, 16, 24):
                tiny(cv, x + bw // 2, base + 5, str(n), LABEL, align='c')
        # legend + running totals
        lx, ly = 70, 52
        lh = self.lf.height + 5
        cv.fill(lx, ly + 1, 6, 6, mixc(DIM, ORANGE, 0.85))
        cv.text(lx + 10, ly, f"{lab['nocache']}  {tot_no}", TEXT2, font=self.lf)
        cv.fill(lx, ly + lh + 1, 6, 6, BLUE)
        cv.text(lx + 10, ly + lh, f"{lab['withcache']}  {tot_yes}", TEXT2, font=self.lf)
        if t >= T_COST + 3.6:
            cv.text(lx, ly + 2 * lh + 6, lab['mem'], LABEL, n=int((t - T_COST - 3.6) * 60), font=self.lf)

    # ─────────── 8-bit soundtrack (the v2 references are silent; use --silent to match them)
    def sfx(self):
        n = sfx.note
        ev = []
        penta = ['C5', 'D5', 'E5', 'G5', 'A5']
        for i in range(N_PROMPT):
            ev.append((0.25 + 0.18 * i, sfx.blip(n(penta[i]), 0.06), 0.6))
        for i in range(3):
            ev.append((PROJ[0] + 0.12 * i, sfx.pew(n(['A4', 'C5', 'E5'][i])), 0.5))
        for r in range(N_PROMPT):
            ev.append((ROW0 + ROW_DT * r, sfx.blip(n(['E5', 'G5', 'A5', 'C6', 'D6'][r]), 0.05), 0.55))
        ev.append((OUT0[0] + 0.3, sfx.glide(n('A5'), n('E5'), 0.25), 0.4))
        ev.append((NEXT0[0] + 0.2, sfx.chime(('E5', 'G5', 'C6'), step=0.05, hold=0.12, drone=False), 0.6))
        ev.append((DROP_Q[0], sfx.boot(0.3, f=n('E4'), up=False), 0.4))
        ev.append((CACHE_FRAME[0] + 0.3, sfx.blip(n('G4'), 0.08, duty=0.5), 0.6))
        for s in STEPS:
            st = step_times(s)
            ev.append((st['proj'][0], sfx.pew(n('C5')), 0.5))
            for c in range(s['p'] + 1):
                ts = st['scan'][0] + (st['scan'][1] - st['scan'][0]) * c / (s['p'] + 1)
                ev.append((ts, sfx.blip(n(['C6', 'D6', 'E6', 'G6', 'A6', 'C7', 'D7'][c]), 0.035), 0.45))
            ev.append((st['mix'][0], sfx.glide(n('G5'), n('C5'), 0.3), 0.35))
            ev.append((st['nxt'][0] + 0.2, sfx.chime(('C5', 'E5', 'G5', 'C6'), step=0.045, hold=0.12,
                                                     drone=False), 0.6))
        for k in range(24):
            ev.append((T_COST + 0.8 + 2.5 * k / 24, sfx.tick(f=n('C6') * (1 + k / 24)), 0.25))
        ev.append((T_END, sfx.boot(0.4, up=False), 0.5))
        return ev


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stills', action='store_true')
    ap.add_argument('--silent', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sc = KVCache()
    out = a.out or os.path.join(os.getcwd(), 'out', 'kv_cache_en.mp4')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if a.stills:
        stills(sc, [0.6, 2.5, 3.9, 5.5, 7.0, 7.9, 9.4, 10.6, 11.7, 12.6, 13.5, 14.2, 15.0, 17.6, 19.7, 21.0, 23.5,
                    25.5], out.replace('.mp4', '_stills.png'), cols=3)
    else:
        sc.setup()
        wav = None
        if not a.silent:
            tr = sfx.Track(sc.DURATION)
            for (ts, snd, g) in sc.sfx():
                tr.add(ts, snd, g)
            wav = out.replace('.mp4', '.wav')
            tr.save(wav, lufs=-24)
        render(sc, out, audio=wav)
