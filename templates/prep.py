"""{NAME} — compute the REAL numbers the animation shows, save them to data/data.npz.

Preference order:
  1. run the released code (code/) on a tiny input and dump intermediate tensors with hooks;
  2. reproduce the core maths in numpy at toy scale (d = 8, 16 tokens, 10 steps …);
  3. copy numbers from the paper's tables — and cite them in FACTS.md.
The scene reads this file; nothing that looks like data is typed into scene.py by hand.

    python3 prep.py            # → data/data.npz
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)          # every random draw seeded — renders are multi-process
out = {}

# ── example (replace): a toy attention step, the kind of real values a stage can colour cells with
d, n = 8, 6
x = rng.normal(size=(n, d))
Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
q, k, v = x @ Wq, x @ Wk, x @ Wv
s = q @ k.T / np.sqrt(d)
s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
w = np.exp(s - s.max(1, keepdims=True))
w /= w.sum(1, keepdims=True)
out.update(x=x, q=q, k=k, v=v, attn=w, y=w @ v)

os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
np.savez_compressed(os.path.join(HERE, 'data', 'data.npz'), **out)
print('✓ data/data.npz:', {k_: np.asarray(v_).shape for k_, v_ in out.items()})
