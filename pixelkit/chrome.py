"""The v2 'phosphor' explainer chrome (480×270 → ×4), measured on the V-JEPA 2.1 / π0.5 references.

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │ TITLE (x=10)          STATUS LINE (centred at 240)      STAGE ■■■■■□□□ (→467) │  cap-top y=6
    │                                                                              │
    │                        … the mechanism …                                     │
    │ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·  │  rule y=243
    │ CAPTION LINE 1 — what happens (bright)                                       │  y=249
    │ caption line 2 — the hard facts: sizes, dims, source caveats (dim)           │  y=259
    └──────────────────────────────────────────────────────────────────────────────┘
  * dot grid: one pixel every 6 px at x≡3, y≡3 (mod 6)
  * captions clear at a stage change, then type at ~120 chars/s; line 2 starts 0.2 s after line 1
  * progress squares 5×5, pitch 7: past = filled dim blue, current = accent, future = hollow
"""
from .fonts import FONT57
from .pk import PHOSPHOR, dot_grid

W, H = 480, 270
CAP_Y1, CAP_Y2, RULE_Y = 249, 259, 243
CPS = 120.0
LINE2_DELAY = 0.2


def background(cv, pal=PHOSPHOR):
    dot_grid(cv, pal['grid'], step=6, ox=3, oy=3)


def rule(cv, y, pal=PHOSPHOR, x0=10, x1=469):
    cv.hline(x0, x1, y, pal['rule'], dash=(1, 1))


def squares(cv, n, cur, pal=PHOSPHOR, x_right=467, y=7, size=5, pitch=7, accent=None):
    x0 = x_right - (n - 1) * pitch - size + 1
    for i in range(n):
        x = x0 + i * pitch
        if i < cur:
            cv.fill(x, y, size, size, pal['dim2'])
        elif i == cur:
            cv.fill(x, y, size, size, accent or pal['orange'])
        else:
            cv.rect(x, y, size, size, pal['todo'])
    return x0


# Caption and header typography.
LAYOUT = {
    'en': dict(font=None, head_y=6, rule_y=243, y1=249, y2=259, cps=CPS),
}


def _font(spec):
    return FONT57


def title_font(title):
    """Use the LCD font for titles."""
    return FONT57, 6


def header(cv, title, status, stage, n_stages, cur, pal=PHOSPHOR, status_n=None, accent=None, lang='en'):
    """Title left, status centred, stage name + progress squares right."""
    L = LAYOUT[lang]
    f = _font(L['font'])
    tf, ty = title_font(title)
    cv.text(10, ty, title, pal['text'], font=tf)
    if status:
        cv.text(240, L['head_y'], status, pal['text2'], align='c', n=status_n, font=f)
    x0 = squares(cv, n_stages, cur, pal, accent=accent)
    if stage:
        cv.text(x0 - 7, L['head_y'], stage, pal['text2'], align='r', font=f)


def caption(cv, line1, line2, t, t0, pal=PHOSPHOR, cps=None, delay=LINE2_DELAY, lang='en'):
    """Two-line typewriter caption that starts typing at t0 (both lines type in parallel,
    line 2 lagging by `delay`)."""
    L = LAYOUT[lang]
    f = _font(L['font'])
    cps = cps or L['cps']
    rule(cv, L['rule_y'], pal)
    if t < t0:
        return
    n1 = int((t - t0) * cps)
    n2 = int((t - t0 - delay) * cps)
    if line1:
        cv.text(10, L['y1'], line1, pal['text2'], n=max(0, n1), font=f)
    if line2 and n2 > 0:
        cv.text(10, L['y2'], line2, pal['label'], n=n2, font=f)


GAP = 12        # min blank px between title | status | stage name (less reads as one run of words)


def cv_width(text, font):
    """Pixel width of text (with '_{…}' subscripts) in a font, without a canvas."""
    from .pk import _parse_sub
    runs = _parse_sub(text or '')
    return sum(font.width(r) for r, _ in runs) + font.spacing * (len(runs) - 1 if runs else 0)


def fits(text, lang='en', width=460):
    """True if a caption/status line fits the frame (use it to assert copy length)."""
    return _font(LAYOUT[lang]['font']).width(text) <= width


class Stages:
    """Stage list → what the chrome shows at time t.

    Stages([
        dict(name='INPUT', t=0.0, status='16 FRAMES · 4 FPS', cap=('LINE 1', 'LINE 2')),
        dict(name='TOKENS', t=3.0, status=[(3.0, 'A'), (4.0, 'B')], cap=(…, …)),
    ], lang='en')
    """

    def __init__(self, stages, lang='en'):
        self.stages = stages
        self.lang = lang

    def at(self, t):
        cur = 0
        for i, s in enumerate(self.stages):
            if t >= s['t']:
                cur = i
        return cur, self.stages[cur]

    def status(self, t):
        """status: a string, [(t, str), …] (changes inside the stage) or a function f(t) → str (live counters:
        'STEP {n} / 150 · LOSS {l:.4f}'). For values only known after setup(), pass status= to draw() instead."""
        _, s = self.at(t)
        st = s.get('status', '')
        if callable(st):
            return st(t)
        if isinstance(st, list):
            txt = st[0][1]
            for (ts, v) in st:
                if t >= ts:
                    txt = v
            return txt
        return st

    def caption_at(self, t):
        """(line1, line2, t_typing_start). cap may be a tuple, or [(t, (l1, l2)), …] to change the caption
        inside a stage (the references do: V-JEPA's TOKENS stage switches from video to images at 5 s)."""
        _, s = self.at(t)
        cap = s.get('cap', ('', ''))
        if isinstance(cap, list):
            c, t0 = cap[0][1], cap[0][0]
            for (ts, v) in cap:
                if t >= ts:
                    c, t0 = v, ts
            return c[0], c[1], t0
        return cap[0], cap[1], s['t'] + s.get('cap_delay', 0.0)

    def draw(self, cv, t, title, pal=PHOSPHOR, accent=None, status=None):
        """Header + captions. status: optional override of the status line for this frame (live values)."""
        cur, s = self.at(t)
        header(cv, title, status if status is not None else self.status(t), s['name'], len(self.stages), cur, pal,
               accent=accent, lang=self.lang)
        c1, c2, t0 = self.caption_at(t)
        caption(cv, c1, c2, t, t0, pal, lang=self.lang)

    def check(self, title=None):
        """Everything that would not render cleanly: captions wider than the frame, header parts that
        collide (title | status | stage name + squares, measured with the real fonts), and characters
        the chosen font cannot draw. Returns a list of (problem, text); empty = OK."""
        bad = []
        f = _font(LAYOUT[self.lang]['font'])
        n = len(self.stages)
        x_sq = 467 - (n - 1) * 7 - 5 + 1
        t_right = 10 + (cv_width(title, title_font(title)[0]) if title else 0)
        for s in self.stages:
            sts = s.get('status', '')
            if callable(sts):                             # sample a live status at a few times in the stage
                sts = [sts(s['t'] + dt) for dt in (0.0, 0.5, 1.0, 2.0, 4.0)]
            elif isinstance(sts, list):
                sts = [v for _, v in sts]
            else:
                sts = [sts]
            name = s.get('name', '')
            n_left = x_sq - 7 - cv_width(name, f) + 1
            for v in sts:
                w = cv_width(v, f)
                s_left, s_right = 240 - w // 2, 240 - w // 2 + w - 1
                if v and (s_left < t_right + GAP or s_right > n_left - GAP):
                    bad.append(('header collision (title | status | stage)', f'{title} | {v} | {name}'))
                if f.missing(v):
                    bad.append((f'status glyphs missing {f.missing(v)}', v))
            if f.missing(name):
                bad.append((f'stage-name glyphs missing {f.missing(name)}', name))
            cap = s.get('cap', ())
            lines = [x for _, pair in cap for x in pair] if isinstance(cap, list) else list(cap)
            for c in lines:
                if c and not fits(c, self.lang):
                    bad.append(('caption too wide', c))
                if c and f.missing(c):
                    bad.append((f'caption glyphs missing {f.missing(c)}', c))
            if not fits(s.get('name', ''), self.lang, 90):
                bad.append(('stage name', s.get('name')))
        ts = [s['t'] for s in self.stages]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            bad.append(('stage times not increasing', str(ts)))
        return list(dict.fromkeys(bad))                  # de-duplicated, order kept
