"""8-bit (NES-APU-style) sound effects, modelled on the reference soundtracks.

What the reference videos actually use (measured with FFTs on their audio):
  * layer / block 'pew'   — 12.5 %-duty pulse at A4 (≈442 Hz), 110 ms, hard stop, tiny pitch drop at onset.
                            Harmonic levels 0/-0.8/-2/-4/-6.7/-11 dB = exactly a 1/8 pulse.
  * counter / slot blips  — 25 %-duty pulse notes (C6, D6, E6 …): 2nd harmonic -4 dB, 4th missing.
  * success chime         — E-major arpeggio G#5 B5 E6 G#6 B6 in 25 % pulse, ~60 ms per note, last note held,
                            over a soft A4 drone.
  * noise being added     — NES noise channel (15-bit LFSR) whose clock sweeps down → descending comb stripes.
  * boot / reset          — two 1/8 pulses that detune away from each other (the 'fan' in the spectrogram).

All generators return float64 mono arrays at SR. Place them with Track.add(t, sound, gain) and save()
(two-pass EBU R128 loudness normalisation).
"""
import json
import os
import subprocess

import numpy as np

SR = 48000
A4 = 440.0


def note(name):
    """'A4', 'C#6', 'Bb5' → Hz (equal temperament)."""
    names = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}
    n = names[name[0].upper()]
    rest = name[1:]
    if rest and rest[0] in '#b':
        n += 1 if rest[0] == '#' else -1
        rest = rest[1:]
    octave = int(rest)
    return A4 * 2 ** ((n + 12 * (octave - 4)) / 12)


def _n(dur):
    return max(1, int(round(dur * SR)))


def _freq_track(f0, f1, n, curve=1.0):
    if f1 is None or f1 == f0:
        return np.full(n, float(f0))
    k = np.linspace(0, 1, n) ** curve
    return f0 * (f1 / f0) ** k


def pulse(freq, duty=0.125, fmax=14000.0):
    """Band-limited pulse wave (additive: a_n = 2/(nπ)·sin(nπ·duty)). freq: per-sample array (Hz)."""
    freq = np.asarray(freq, np.float64)
    ph = 2 * np.pi * np.cumsum(freq) / SR
    y = np.zeros_like(ph)
    top = int(fmax / max(1.0, float(freq.max())))
    for k in range(1, max(2, top) + 1):
        a = 2.0 / (k * np.pi) * np.sin(k * np.pi * duty)
        if abs(a) < 1e-4:
            continue
        y += a * np.cos(k * ph)
    return y / (2.0 / np.pi * np.sin(np.pi * duty) + 1e-9) * 0.5


def triangle(freq):
    freq = np.asarray(freq, np.float64)
    ph = (np.cumsum(freq) / SR) % 1.0
    return 2 * np.abs(2 * ph - 1) - 1


def lfsr_noise(rate, short=False, seed=1, n=None):
    """NES noise channel: 15-bit LFSR clocked `rate` times per second. rate: per-sample array, or a
    scalar together with n (number of samples). short=True uses the 93-step mode (metallic, pitched)."""
    rate = np.asarray(rate, np.float64)
    if rate.ndim == 0:
        rate = np.full(int(n if n is not None else SR), float(rate))
    n = len(rate)
    out = np.empty(n)
    reg = seed & 0x7FFF or 1
    tap = 6 if short else 1
    acc = 0.0
    bit = 1.0
    for i in range(n):
        acc += rate[i] / SR
        while acc >= 1.0:
            fb = (reg ^ (reg >> tap)) & 1
            reg = (reg >> 1) | (fb << 14)
            bit = 1.0 if reg & 1 else -1.0
            acc -= 1.0
        out[i] = bit
    return out


def adsr(n, a=0.002, d=None, r=0.004, sustain=1.0):
    """Linear attack, optional exponential decay (time constant d), linear release at the end."""
    e = np.ones(n)
    if n <= 0:
        return e
    na = min(n, max(1, int(a * SR)))
    e[:na] = np.linspace(0, 1, na)
    if d is not None and n > na:
        k = np.arange(n - na) / SR
        e[na:] = sustain + (1 - sustain) * np.exp(-k / d)
    nr = min(n, max(1, int(r * SR)))
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ──────────────────────────────────────────────── the vocabulary

def pew(f=442.0, dur=0.11, duty=0.125, bend=1.03, amp=0.5):
    """A block/layer fires: 1/8 pulse with a quick 3 % pitch drop, flat, hard stop."""
    n = _n(dur)
    fr = np.full(n, float(f))
    nb = min(n, _n(0.02))
    if nb > 0:
        fr[:nb] = f * np.linspace(bend, 1.0, nb)
    return amp * pulse(fr, duty) * adsr(n, 0.002, None, 0.004)


def blip(f=1318.5, dur=0.05, duty=0.25, amp=0.45, decay=0.03):
    """A counter tick / note: 25 % pulse with a fast decay."""
    n = _n(dur)
    return amp * pulse(np.full(n, float(f)), duty) * adsr(n, 0.001, decay, 0.003, sustain=0.0)


def tick(f=3520.0, amp=0.3):
    n = _n(0.012)
    return amp * pulse(np.full(n, f), 0.5) * adsr(n, 0.0005, 0.003, 0.001, sustain=0.0)


def glide(f0, f1, dur=0.3, duty=0.25, amp=0.35, curve=0.6):
    n = _n(dur)
    return amp * pulse(_freq_track(f0, f1, n, curve), duty) * adsr(n, 0.005, None, dur * 0.3)


def hiss(dur=0.4, rate0=24000.0, rate1=3000.0, short=False, amp=0.3, seed=1):
    """Noise being added: LFSR noise whose clock sweeps down (descending comb), decaying."""
    n = _n(dur)
    rate = _freq_track(rate0, rate1, n, 0.7)
    y = lfsr_noise(rate, short, seed)
    return amp * y * adsr(n, 0.004, dur * 0.55, 0.03, sustain=0.0)


def boot(dur=0.36, f=442.0, spread=0.18, amp=0.35, up=True):
    """Two 1/8 pulses detuning away from (or back into) unison — the materialise 'fan'."""
    n = _n(dur)
    k = np.linspace(0, 1, n)
    d = spread * (k if up else 1 - k) ** 1.4
    y = pulse(f * (1 + d), 0.125) + pulse(f * (1 - 0.5 * d), 0.125)
    return amp * 0.6 * y * adsr(n, 0.004, None, dur * 0.35)


def chime(notes=('G#5', 'B5', 'E6', 'G#6', 'B6'), step=0.06, hold=0.25, amp=0.45, drone=True):
    """Success: rising 25 % pulse arpeggio into a held top note, over a soft A4 drone."""
    total = step * (len(notes) - 1) + hold
    n = _n(total)
    y = np.zeros(n)
    for i, nm in enumerate(notes):
        s = _n(step * i)
        dur = step if i < len(notes) - 1 else hold
        m = min(n - s, _n(dur))
        if m <= 0:
            continue
        y[s:s + m] += pulse(np.full(m, note(nm)), 0.25) * adsr(m, 0.002, None, 0.005)
    if drone:
        y += 0.35 * pulse(np.full(n, A4), 0.125) * adsr(n, 0.004, None, 0.02)
    return amp * y


def arp(notes, step=0.05, duty=0.25, amp=0.4):
    """Plain note sequence (e.g. a scale run)."""
    parts = [pulse(np.full(_n(step), note(nm) if isinstance(nm, str) else nm), duty) * adsr(_n(step), 0.002, None, 0.004)
             for nm in notes]
    return amp * np.concatenate(parts)


PENTA = ['C6', 'D6', 'E6', 'G6', 'A6', 'C7']


# ──────────────────────────────────────────────── timeline

class Track:
    """Stereo timeline of sounds."""

    def __init__(self, duration):
        self.n = int(duration * SR) + SR
        self.duration = duration
        self.buf = np.zeros((self.n, 2))

    def add(self, t, snd, gain=1.0, pan=0.0):
        """Place a sound at time t (a negative t crops the start). Returns self (chainable)."""
        snd = np.asarray(snd, np.float64)
        s = int(round(t * SR))
        if s < 0:
            snd, s = snd[-s:], 0
        e = min(self.n, s + len(snd))
        if e <= s:
            return self
        l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        r = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        self.buf[s:e, 0] += snd[:e - s] * gain * l
        self.buf[s:e, 1] += snd[:e - s] * gain * r
        return self

    def save(self, path, lufs=-24.0, peak_db=-3.0):
        """Write wav, then loudness-normalise (EBU R128, two-pass, linear) with a true-peak ceiling."""
        y = self.buf[:int(round(self.duration * SR))]
        pk = float(np.abs(y).max()) if len(y) else 0.0
        raw = path + '.raw.wav'
        _write_wav(raw, y / pk * 0.5 if pk > 0 else y)
        try:
            if lufs is None or pk == 0 or len(y) < int(0.4 * SR):
                os.replace(raw, path)                   # silent / too short to measure: keep as is
                return path
            r = subprocess.run(['ffmpeg', '-hide_banner', '-i', raw, '-af',
                                f'loudnorm=I={lufs}:TP={peak_db}:LRA=20:print_format=json', '-f', 'null', '-'],
                               capture_output=True, text=True)
            m = json.loads(r.stderr[r.stderr.rfind('{'):r.stderr.rfind('}') + 1])
            if any(str(m.get(k, '')).lstrip('-') in ('inf', 'nan', '') for k in ('input_i', 'input_tp', 'target_offset')):
                os.replace(raw, path)                   # nearly silent: loudnorm cannot measure it
                return path
            af = (f'loudnorm=I={lufs}:TP={peak_db}:LRA=20:measured_I={m["input_i"]}:measured_TP={m["input_tp"]}:'
                  f'measured_LRA={m["input_lra"]}:measured_thresh={m["input_thresh"]}:'
                  f'offset={m["target_offset"]}:linear=true')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-af', af, '-ar', str(SR), path], check=True)
            return path
        finally:
            if os.path.exists(raw):
                os.remove(raw)


def _write_wav(path, y):
    import wave
    y = np.clip(y, -1, 1)
    pcm = (y * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
