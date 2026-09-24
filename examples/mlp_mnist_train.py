"""Train the tiny MLP (784 → 16 → 16 → 10, sigmoid hidden, softmax out) that drives mlp_mnist.py,
then run ONE real SGD step per showcased digit and save everything the animation needs.

    python3 mlp_mnist_train.py --mnist DIR --out mlp_mnist_data.npz
"""
import argparse
import os

import numpy as np


def load_idx(path):
    with open(path, 'rb') as f:
        data = f.read()
    magic = int.from_bytes(data[2:3], 'big')
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], 'big') for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def sig(z):
    return 1 / (1 + np.exp(-z))


def forward(W, x):
    z1 = x @ W['W1'] + W['b1']; a1 = sig(z1)
    z2 = a1 @ W['W2'] + W['b2']; a2 = sig(z2)
    z3 = a2 @ W['W3'] + W['b3']
    z3 = z3 - z3.max(-1, keepdims=True)
    p = np.exp(z3); p /= p.sum(-1, keepdims=True)
    return dict(x=x, a1=a1, a2=a2, p=p)


def backward(W, c, y):
    """cross-entropy grads; also returns dL/dz per layer and dL/dx (saliency)"""
    n = c['x'].shape[0]
    Y = np.eye(10)[y]
    d3 = (c['p'] - Y) / n
    g = dict(W3=c['a2'].T @ d3, b3=d3.sum(0))
    d2 = (d3 @ W['W3'].T) * c['a2'] * (1 - c['a2'])
    g.update(W2=c['a1'].T @ d2, b2=d2.sum(0))
    d1 = (d2 @ W['W2'].T) * c['a1'] * (1 - c['a1'])
    g.update(W1=c['x'].T @ d1, b1=d1.sum(0))
    dx = d1 @ W['W1'].T
    return g, dict(d3=d3, d2=d2, d1=d1, dx=dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mnist', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    X = load_idx(os.path.join(a.mnist, 'train-images-idx3-ubyte')).reshape(-1, 784) / 255.0
    Y = load_idx(os.path.join(a.mnist, 'train-labels-idx1-ubyte')).astype(int)
    Xt = load_idx(os.path.join(a.mnist, 't10k-images-idx3-ubyte')).reshape(-1, 784) / 255.0
    Yt = load_idx(os.path.join(a.mnist, 't10k-labels-idx1-ubyte')).astype(int)
    rng = np.random.default_rng(0)
    W = dict(W1=rng.normal(0, 1 / 28, (784, 16)), b1=np.zeros(16), W2=rng.normal(0, 0.25, (16, 16)), b2=np.zeros(16),
             W3=rng.normal(0, 0.25, (16, 10)), b3=np.zeros(10))
    # Adam, 3 epochs of minibatches
    m = {k: np.zeros_like(v) for k, v in W.items()}
    v = {k: np.zeros_like(v_) for k, v_ in W.items()}
    step = 0
    for ep in range(3):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), 64):
            b = idx[s:s + 64]
            c = forward(W, X[b])
            g, _ = backward(W, c, Y[b])
            step += 1
            for k in W:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** step); vh = v[k] / (1 - 0.999 ** step)
                W[k] -= 3e-3 * mh / (np.sqrt(vh) + 1e-8)
        acc = (forward(W, Xt)['p'].argmax(1) == Yt).mean()
        print(f'epoch {ep + 1}: test accuracy {acc:.3f}')
    pred = forward(W, Xt)['p'].argmax(1)
    # showcase: labels 1 2 2 3 2 8 then a 6 the model gets wrong (as in the reference)
    want = [1, 2, 2, 3, 2, 8]
    chosen, used = [], set()
    for lab in want:
        cand = [i for i in np.nonzero((Yt == lab) & (pred == lab))[0] if i not in used]
        i = cand[3 + len(chosen)]
        chosen.append(i); used.add(i)
    wrong6 = np.nonzero((Yt == 6) & (pred != 6))[0]
    chosen.append(wrong6[0])
    # one real SGD step per showcased digit (online), record everything
    rec = {k: [] for k in ('img', 'label', 'a1', 'a2', 'p', 'd3', 'd2', 'd1', 'dx', 'dW1', 'dW2', 'dW3', 'W1', 'W2', 'W3')}
    lr = 0.5
    for i in chosen:
        x = Xt[i:i + 1]
        c = forward(W, x)
        g, d = backward(W, c, Yt[i:i + 1])
        for k in ('W1', 'W2', 'W3'):
            rec[k].append(W[k].copy())
        for k in W:
            W[k] = W[k] - lr * g[k]
        rec['img'].append(Xt[i].reshape(28, 28)); rec['label'].append(Yt[i])
        rec['a1'].append(c['a1'][0]); rec['a2'].append(c['a2'][0]); rec['p'].append(c['p'][0])
        rec['d3'].append(d['d3'][0]); rec['d2'].append(d['d2'][0]); rec['d1'].append(d['d1'][0])
        rec['dx'].append(d['dx'][0].reshape(28, 28))
        rec['dW1'].append(-lr * g['W1']); rec['dW2'].append(-lr * g['W2']); rec['dW3'].append(-lr * g['W3'])
        print(f'digit {Yt[i]} → predicted {c["p"][0].argmax()}  p={c["p"][0].max():.2f}')
    np.savez_compressed(a.out, **{k: np.array(v_) for k, v_ in rec.items()})
    print('✓', a.out)


if __name__ == '__main__':
    main()
