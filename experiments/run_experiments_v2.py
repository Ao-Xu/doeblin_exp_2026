import argparse
import math
import os
import shutil
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, "results_v2")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def style():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "figure.figsize": (6.0, 3.6),
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def write_csv(name, header, rows):
    path = os.path.join(V2, name)
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if x is None else str(x) for x in row) + "\n")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(V2, name))
    plt.close(fig)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def normal_pdf(z, sigma):
    return np.exp(-0.5 * (z / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)


def wrapped_normal_pdf(y, m, sigma):
    out = np.zeros_like(y, dtype=float)
    for shift in [-1.0, 0.0, 1.0]:
        out += normal_pdf(y - m + shift, sigma)
    return out


def beta28_pdf(y):
    y = np.clip(y, 1e-8, 1 - 1e-8)
    return 72.0 * y * (1 - y) ** 7


def kde_wrapped_density(y, samples, bandwidth=0.055, batch=4096):
    y = np.asarray(y).reshape(-1)
    samples = np.asarray(samples).reshape(-1)
    out = []
    for start in range(0, y.size, batch):
        yy = y[start:start + batch, None]
        val = np.zeros((yy.shape[0], samples.size))
        for shift in [-1.0, 0.0, 1.0]:
            val += normal_pdf(yy - samples[None, :] + shift, bandwidth)
        out.append(val.mean(axis=1))
    return np.concatenate(out).reshape(np.asarray(y).shape)


def m_fun(X, model="smooth"):
    x = X[..., 0]
    if model == "smooth":
        m = 0.5 + 0.25 * np.sin(2 * np.pi * x)
    elif model == "rough":
        m = 0.5 + 0.22 * np.tanh(10 * (x - 0.5)) + 0.08 * np.sin(10 * np.pi * x)
    elif model == "multimodal1":
        m = 0.35 + 0.20 * np.sin(2 * np.pi * x)
    elif model == "multimodal2":
        m = 0.65 - 0.20 * np.sin(2 * np.pi * x)
    else:
        m = 0.5 + 0.25 * np.sin(2 * np.pi * x)
    return np.clip(m, 0.03, 0.97)


def true_density_1d(X, Y, model="smooth", sigma=0.07):
    if model == "multimodal":
        return (
            0.5 * wrapped_normal_pdf(Y, m_fun(X, "multimodal1"), sigma)
            + 0.5 * wrapped_normal_pdf(Y, m_fun(X, "multimodal2"), sigma)
        )
    return wrapped_normal_pdf(Y, m_fun(X, model), sigma)


def sample_kernel(rng, X, model="smooth", sigma=0.07):
    n = X.shape[0]
    if model == "multimodal":
        comp = rng.rand(n) < 0.5
        mean = np.where(comp, m_fun(X, "multimodal1"), m_fun(X, "multimodal2"))
    else:
        mean = m_fun(X, model)
    return ((mean + sigma * rng.randn(n)) % 1.0)[:, None]


def sample_reference(rng, n, ref="uniform", marginal_samples=None, bandwidth=0.055):
    if ref == "uniform" or marginal_samples is None and ref in ("empirical-kde", "mixture"):
        return rng.rand(n, 1)
    if ref == "poor-coverage":
        mask = rng.rand(n, 1) < 0.10
        beta = rng.beta(2.0, 8.0, size=(n, 1))
        return np.where(mask, rng.rand(n, 1), beta)
    if ref == "empirical-kde":
        idx = rng.randint(0, marginal_samples.shape[0], size=n)
        return (marginal_samples[idx] + bandwidth * rng.randn(n, 1)) % 1.0
    if ref == "mixture":
        mask = rng.rand(n, 1) < 0.5
        idx = rng.randint(0, marginal_samples.shape[0], size=n)
        kde = (marginal_samples[idx] + bandwidth * rng.randn(n, 1)) % 1.0
        return np.where(mask, rng.rand(n, 1), kde)
    raise ValueError("unknown reference %s" % ref)


def ref_density(Y, ref="uniform", marginal_samples=None, bandwidth=0.055):
    y = np.asarray(Y)[..., 0]
    if ref == "uniform" or marginal_samples is None and ref in ("empirical-kde", "mixture"):
        return np.ones_like(y, dtype=float)
    if ref == "poor-coverage":
        return 0.10 + 0.90 * beta28_pdf(y)
    if ref == "empirical-kde":
        return kde_wrapped_density(y, marginal_samples, bandwidth=bandwidth).reshape(y.shape)
    if ref == "mixture":
        kde = kde_wrapped_density(y, marginal_samples, bandwidth=bandwidth).reshape(y.shape)
        return 0.5 + 0.5 * kde
    raise ValueError("unknown reference %s" % ref)


def row_markovize_density(density, dy):
    positive = np.maximum(density, 0.0)
    mass = positive.sum(axis=1, keepdims=True) * dy
    out = np.empty_like(positive)
    good = mass[:, 0] > 0
    out[good] = positive[good] / mass[good]
    out[~good] = 1.0
    return out


def row_markovize_prob(M, fallback=None):
    P = np.maximum(np.asarray(M, dtype=float), 0.0)
    row_sums = P.sum(axis=1, keepdims=True)
    if fallback is None:
        fallback = np.ones(P.shape[1]) / P.shape[1]
    out = np.empty_like(P)
    good = row_sums[:, 0] > 0
    out[good] = P[good] / row_sums[good]
    out[~good] = fallback[None, :]
    return out


def finite_tv(K, L, mu=None):
    if mu is None:
        mu = np.ones(K.shape[0]) / K.shape[0]
    return float(np.sum(mu * 0.5 * np.sum(np.abs(K - L), axis=1)))


def rollout_tv(K, L, xi, horizons):
    out = []
    pK = xi.copy()
    pL = xi.copy()
    hset = set(horizons)
    for h in range(1, max(horizons) + 1):
        pK = pK.dot(K)
        pL = pL.dot(L)
        if h in hset:
            out.append(float(0.5 * np.abs(pK - pL).sum()))
    return out


def path_tv(K, L, xi, h):
    # Exact dynamic-programming coupling upper bound for small finite grids:
    # the occupancy perturbation sum used in the theorem.
    e = 0.5 * np.sum(np.abs(K - L), axis=1)
    p = xi.copy()
    total = 0.0
    for _ in range(h):
        total += float(p.dot(e))
        p = p.dot(K)
    return min(1.0, total)


def occupation_tv(K, L, xi, h):
    pK = xi.copy()
    pL = xi.copy()
    occK = np.zeros_like(xi)
    occL = np.zeros_like(xi)
    for _ in range(h):
        occK += pK
        occL += pL
        pK = pK.dot(K)
        pL = pL.dot(L)
    occK /= h
    occL /= h
    return float(0.5 * np.abs(occK - occL).sum())


class ReLUContrastiveNet:
    def __init__(self, input_dim, width=32, depth=2, seed=0):
        self.rng = np.random.RandomState(seed)
        dims = [input_dim] + [width] * depth + [1]
        self.W = []
        self.b = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.W.append(self.rng.randn(a, b) * math.sqrt(2.0 / max(1, a)))
            self.b.append(np.zeros(b))

    def forward(self, X):
        A = X
        acts = [A]
        pre = []
        for W, b in zip(self.W[:-1], self.b[:-1]):
            Z = A.dot(W) + b
            pre.append(Z)
            A = np.maximum(Z, 0.0)
            acts.append(A)
        logits = (A.dot(self.W[-1]) + self.b[-1]).reshape(-1)
        return logits, acts, pre

    def predict_logit(self, X, batch=20000):
        out = []
        for i in range(0, X.shape[0], batch):
            z, _, _ = self.forward(X[i:i + batch])
            out.append(z)
        return np.concatenate(out)

    def fit(self, X, y, epochs=8, batch_size=512, lr=2e-3, weight_decay=1e-5):
        n = X.shape[0]
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        beta1, beta2 = 0.9, 0.999
        step = 0
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                step += 1
                ids = idx[start:start + batch_size]
                xb = X[ids]
                yb = y[ids]
                logits, acts, pre = self.forward(xb)
                p = sigmoid(logits)
                dz = (p - yb)[:, None] / len(ids)
                gW = [None] * len(self.W)
                gb = [None] * len(self.b)
                gW[-1] = acts[-1].T.dot(dz) + weight_decay * self.W[-1]
                gb[-1] = dz.sum(axis=0)
                dA = dz.dot(self.W[-1].T)
                for layer in range(len(self.W) - 2, -1, -1):
                    dZ = dA * (pre[layer] > 0)
                    gW[layer] = acts[layer].T.dot(dZ) + weight_decay * self.W[layer]
                    gb[layer] = dZ.sum(axis=0)
                    if layer > 0:
                        dA = dZ.dot(self.W[layer].T)
                for i in range(len(self.W)):
                    mW[i] = beta1 * mW[i] + (1 - beta1) * gW[i]
                    vW[i] = beta2 * vW[i] + (1 - beta2) * (gW[i] ** 2)
                    mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * (gb[i] ** 2)
                    self.W[i] -= lr * (mW[i] / (1 - beta1 ** step)) / (
                        np.sqrt(vW[i] / (1 - beta2 ** step)) + 1e-8
                    )
                    self.b[i] -= lr * (mb[i] / (1 - beta1 ** step)) / (
                        np.sqrt(vb[i] / (1 - beta2 ** step)) + 1e-8
                    )


def build_contrastive_data(rng, n, model="smooth", sigma=0.07, eps=0.1, tau=5,
                           ref="uniform", marginal_samples=None):
    X = rng.rand(n, 1)
    Y_data = sample_kernel(rng, X, model=model, sigma=sigma)
    Y_anchor = sample_reference(rng, n, ref=ref, marginal_samples=marginal_samples)
    use_data = (rng.rand(n, 1) < (1 - eps))
    Y_pos = np.where(use_data, Y_data, Y_anchor)
    X_neg = np.repeat(X, tau, axis=0)
    Y_neg = sample_reference(rng, n * tau, ref=ref, marginal_samples=marginal_samples)
    Z = np.vstack([np.hstack([X, Y_pos]), np.hstack([X_neg, Y_neg])])
    labels = np.concatenate([np.ones(n), np.zeros(n * tau)])
    return Z, labels


def logistic_loss_from_logits(logits, labels):
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def heldout_contrastive_excess(net, seed, n_val=20000, model="smooth", sigma=0.07,
                               eps=0.1, tau=5, ref="uniform", marginal_samples=None):
    rng = np.random.RandomState(seed)
    Z, labels = build_contrastive_data(rng, n_val, model, sigma, eps, tau, ref, marginal_samples)
    logits_hat = np.clip(net.predict_logit(Z), -8.0, 8.0)
    X = Z[:, 0:1]
    Y = Z[:, 1:2]
    r = ref_density(Y, ref=ref, marginal_samples=marginal_samples)
    k = true_density_1d(X, Y[:, 0], model=model, sigma=sigma)
    a0 = (1 - eps) * k + eps * r
    logits_oracle = np.log(np.maximum(a0, 1e-12) / np.maximum(tau * r, 1e-12))
    return logistic_loss_from_logits(logits_hat, labels) - logistic_loss_from_logits(logits_oracle, labels)


def train_score(seed, n, model="smooth", sigma=0.07, eps=0.1, tau=5, ref="uniform",
                width=32, depth=2, epochs=None, marginal_samples=None):
    rng = np.random.RandomState(seed)
    Z, labels = build_contrastive_data(rng, n, model, sigma, eps, tau, ref, marginal_samples)
    net = ReLUContrastiveNet(2, width=width, depth=depth, seed=seed + 17)
    if epochs is None:
        epochs = 10 if n <= 1500 else 8 if n <= 4000 else 6
    t0 = time.time()
    net.fit(Z, labels, epochs=epochs, batch_size=512, lr=2e-3)
    runtime = time.time() - t0
    return net, runtime


def grid_quantities(net, model="smooth", sigma=0.07, eps=0.1, tau=5, ref="uniform",
                    marginal_samples=None, nx=54, ny=72, markovize=True, deanchor=True):
    x = np.linspace(0.005, 0.995, nx)[:, None]
    y = np.linspace(0.005, 0.995, ny)[:, None]
    dy = float(y[1] - y[0])
    Xg = np.repeat(x, ny, axis=0)
    Yg = np.tile(y, (nx, 1))
    feats = np.hstack([Xg, Yg])
    logits = np.clip(net.predict_logit(feats), -8.0, 8.0).reshape(nx, ny)
    r = ref_density(Yg.reshape(nx, ny, 1), ref=ref, marginal_samples=marginal_samples)
    ahat = tau * r * np.exp(logits)
    ktrue = true_density_1d(Xg, Yg[:, 0], model=model, sigma=sigma).reshape(nx, ny)
    a0 = (1 - eps) * ktrue + eps * r
    if deanchor:
        raw = (ahat - eps * r) / max(1 - eps, 1e-8)
        target_density = ktrue
    else:
        raw = ahat
        target_density = ktrue
    if markovize:
        khat = row_markovize_density(raw, dy)
    else:
        khat = raw
    Ptrue = row_markovize_density(ktrue, dy) * dy
    Pmark = row_markovize_density(raw, dy) * dy
    Phat = khat * dy if not markovize else Pmark
    excess = np.mean(
        -a0 * np.log(np.maximum(ahat, 1e-12) / np.maximum(ahat + tau * r, 1e-12))
        -tau * r * np.log(np.maximum(tau * r, 1e-12) / np.maximum(ahat + tau * r, 1e-12))
        +a0 * np.log(np.maximum(a0, 1e-12) / np.maximum(a0 + tau * r, 1e-12))
        +tau * r * np.log(np.maximum(tau * r, 1e-12) / np.maximum(a0 + tau * r, 1e-12))
    )
    neg = float(np.mean(np.sum(np.maximum(-raw, 0.0), axis=1) * dy))
    rowerr = float(np.mean(np.abs(np.sum(raw, axis=1) * dy - 1.0)))
    tv_pre = float(np.mean(0.5 * np.sum(np.abs(raw - target_density), axis=1) * dy))
    tv = float(np.mean(0.5 * np.sum(np.abs(khat - target_density), axis=1) * dy))
    a_l2 = float(np.mean((ahat - a0) ** 2))
    k_l2 = float(np.mean((raw - ktrue) ** 2))
    ratio = tv / max(tv_pre, 1e-12)
    return {
        "excess": float(excess), "a_l2": a_l2, "k_l2": k_l2, "tv_pre": tv_pre,
        "tv": tv, "neg": neg, "rowerr": rowerr, "ratio": ratio,
        "Ptrue": Ptrue, "Pmark": Pmark, "Phat": Phat, "xgrid": x[:, 0], "ygrid": y[:, 0],
    }


def timed_grid_pipeline(net, model="smooth", sigma=0.07, eps=0.1, tau=5,
                        nx=42, ny=42, ref="uniform", marginal_samples=None):
    x = np.linspace(0.005, 0.995, nx)[:, None]
    y = np.linspace(0.005, 0.995, ny)[:, None]
    dy = float(y[1] - y[0])
    Xg = np.repeat(x, ny, axis=0)
    Yg = np.tile(y, (nx, 1))
    feats = np.hstack([Xg, Yg])
    t0 = time.perf_counter()
    logits = np.clip(net.predict_logit(feats), -8.0, 8.0).reshape(nx, ny)
    score_time = time.perf_counter() - t0
    r = ref_density(Yg.reshape(nx, ny, 1), ref=ref, marginal_samples=marginal_samples)
    t1 = time.perf_counter()
    ahat = tau * r * np.exp(logits)
    raw = (ahat - eps * r) / max(1 - eps, 1e-8)
    deanchor_time = time.perf_counter() - t1
    t2 = time.perf_counter()
    khat = row_markovize_density(raw, dy)
    mark_time = time.perf_counter() - t2
    t3 = time.perf_counter()
    ktrue = true_density_1d(Xg, Yg[:, 0], model=model, sigma=sigma).reshape(nx, ny)
    tv = float(np.mean(0.5 * np.sum(np.abs(khat - ktrue), axis=1) * dy))
    neg = float(np.mean(np.sum(np.maximum(-raw, 0.0), axis=1) * dy))
    rowerr = float(np.mean(np.abs(np.sum(raw, axis=1) * dy - 1.0)))
    metric_time = time.perf_counter() - t3
    return score_time, deanchor_time, mark_time, metric_time, tv, neg, rowerr


def gaussian_cde_baseline(seed=0, n=5000, model="smooth", sigma=0.07, nx=54, ny=54):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 1)
    Y = sample_kernel(rng, X, model=model, sigma=sigma)[:, 0]
    x = X[:, 0]
    Phi = np.column_stack([
        np.ones(n), x, np.sin(2 * np.pi * x), np.cos(2 * np.pi * x),
        np.sin(4 * np.pi * x), np.cos(4 * np.pi * x),
    ])
    target_sin = np.sin(2 * np.pi * Y)
    target_cos = np.cos(2 * np.pi * Y)
    ridge = 1e-4 * np.eye(Phi.shape[1])
    bs = np.linalg.solve(Phi.T.dot(Phi) + ridge, Phi.T.dot(target_sin))
    bc = np.linalg.solve(Phi.T.dot(Phi) + ridge, Phi.T.dot(target_cos))
    xg = np.linspace(0.005, 0.995, nx)
    yg = np.linspace(0.005, 0.995, ny)
    Phig = np.column_stack([
        np.ones(nx), xg, np.sin(2 * np.pi * xg), np.cos(2 * np.pi * xg),
        np.sin(4 * np.pi * xg), np.cos(4 * np.pi * xg),
    ])
    s = Phig.dot(bs)
    c = Phig.dot(bc)
    mean = (np.arctan2(s, c) / (2 * np.pi)) % 1.0
    Xmesh = np.repeat(xg[:, None], ny, axis=1)
    Ymesh = np.repeat(yg[None, :], nx, axis=0)
    dens = wrapped_normal_pdf(Ymesh, mean[:, None], sigma)
    dy = yg[1] - yg[0]
    dens = row_markovize_density(dens, dy)
    ktrue = true_density_1d(Xmesh.reshape(-1, 1), Ymesh.reshape(-1), model, sigma).reshape(nx, ny)
    tv = float(np.mean(0.5 * np.sum(np.abs(dens - ktrue), axis=1) * dy))
    Ptrue = row_markovize_density(ktrue, dy) * dy
    Phat = dens * dy
    xi = np.ones(nx) / nx
    roll = rollout_tv(Ptrue, Phat, xi, [10])[0]
    return tv, roll


def exp1_end_to_end(seeds):
    rows = []
    models = [("smooth", 0.07), ("multimodal", 0.06), ("rough", 0.07)]
    n_list = [800, 1600, 3200, 6400]
    for model, sigma in models:
        for n in n_list:
            for seed in range(seeds):
                net, rt = train_score(10000 + seed + n + len(model), n, model=model, sigma=sigma)
                met = grid_quantities(net, model=model, sigma=sigma)
                val_excess = heldout_contrastive_excess(
                    net, 15000 + seed + n + len(model), n_val=12000, model=model, sigma=sigma
                )
                rows.append((model, n, seed, rt, val_excess, met["excess"], met["a_l2"], met["k_l2"], met["tv"]))
    write_csv("exp1_end_to_end.csv",
              ["model", "n", "seed", "runtime", "val_excess", "oracle_grid_excess", "a_l2", "k_l2", "tv"],
              rows)
    arr = np.array([[max(r[4], 1e-8), r[6], r[7]] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(arr[:, 0], arr[:, 1], s=16, alpha=0.58, label=r"$\|\hat a-a_0\|_2^2$")
    ax.scatter(arr[:, 0], arr[:, 2], s=16, alpha=0.58, label=r"$\|\tilde k-k_0\|_2^2$")
    lo, hi = np.percentile(arr[:, 0], [5, 95])
    xs = np.array([lo, hi])
    ax.plot(xs, np.median(arr[:, 1] / np.maximum(arr[:, 0], 1e-12)) * xs, "k--", label="slope 1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("held-out contrastive excess risk")
    ax.set_ylabel("integrated squared error")
    ax.set_title("End-to-end calibration")
    ax.legend()
    savefig(fig, "fig1_end_to_end_calibration.pdf")
    return rows


def exp2_markovization_learned(seeds):
    rows = []
    for model, sigma in [("smooth", 0.07), ("rough", 0.07), ("multimodal", 0.06)]:
        for n in [1000, 3200, 6400]:
            for seed in range(seeds):
                net, _ = train_score(20000 + seed + n + len(model), n, model=model, sigma=sigma)
                met = grid_quantities(net, model=model, sigma=sigma, markovize=True)
                rows.append((model, n, seed, met["neg"], met["rowerr"], met["tv_pre"], met["tv"], met["ratio"]))
    write_csv("exp2_markovization.csv", ["env", "n", "seed", "negmass", "rowerr", "tv_pre", "tv_post", "ratio"], rows)
    neg = np.array([r[3] for r in rows], dtype=float)
    rowerr = np.array([r[4] for r in rows], dtype=float)
    pre = np.array([r[5] for r in rows], dtype=float)
    post = np.array([r[6] for r in rows], dtype=float)
    ratios = np.array([r[7] for r in rows], dtype=float)
    envs = ["smooth", "rough", "multimodal"]
    colors = {"smooth": "#4C78A8", "rough": "#F58518", "multimodal": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.35))

    # Panel A: separate the small negative mass from row-sum error on a log scale.
    floor = 5e-5
    for i, env in enumerate(envs):
        env_rows = [r for r in rows if r[0] == env]
        env_neg = np.array([r[3] for r in env_rows], dtype=float)
        env_row = np.array([r[4] for r in env_rows], dtype=float)
        jitter = np.linspace(-0.035, 0.035, len(env_rows))
        plot_neg = np.maximum(env_neg, floor)
        plot_row = np.maximum(env_row, floor)
        axes[0].scatter(i - 0.14 + jitter, plot_neg, s=22, color="#4C78A8", alpha=0.85,
                        label="NegMass before" if i == 0 else None)
        axes[0].scatter(i + 0.14 + jitter, plot_row, s=22, color="#E45756", alpha=0.85,
                        label="RowErr before" if i == 0 else None)
        axes[0].plot([i - 0.18, i - 0.10], [max(env_neg.mean(), floor), max(env_neg.mean(), floor)],
                     color="#2F5D8C", lw=2)
        axes[0].plot([i + 0.10, i + 0.18], [max(env_row.mean(), floor), max(env_row.mean(), floor)],
                     color="#B83C3A", lw=2)
    axes[0].axhline(floor, color="0.25", lw=0.8, linestyle=":")
    axes[0].text(0.03, 0.07, "after Markovization: both exactly 0\nzeros shown at plotting floor",
                 transform=axes[0].transAxes, fontsize=7.2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(floor * 0.6, 0.4)
    axes[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    axes[0].set_xticks(range(len(envs)))
    axes[0].set_xticklabels(["smooth", "rough", "multi"])
    axes[0].set_ylabel("invalidity diagnostic")
    axes[0].set_title("validity violation before repair")
    axes[0].legend(frameon=False, loc="upper right")

    # Panel B: paired runs show that Markovization is a validity step, not a TV heuristic.
    for i, env in enumerate(envs):
        env_rows = [r for r in rows if r[0] == env]
        env_pre = np.array([r[5] for r in env_rows], dtype=float)
        env_post = np.array([r[6] for r in env_rows], dtype=float)
        offsets = np.linspace(-0.08, 0.08, len(env_rows))
        for j, off in enumerate(offsets):
            axes[1].plot([i - 0.18 + off, i + 0.18 + off], [env_pre[j], env_post[j]],
                         color="0.72", lw=0.9, zorder=1)
        axes[1].scatter(np.full(len(env_pre), i - 0.18) + offsets, env_pre,
                        s=20, color="#9D755D", alpha=0.85, label="pre-M" if i == 0 else None, zorder=2)
        axes[1].scatter(np.full(len(env_post), i + 0.18) + offsets, env_post,
                        s=20, color="#72B7B2", alpha=0.9, label="post-M" if i == 0 else None, zorder=2)
        axes[1].plot([i - 0.22, i - 0.14], [env_pre.mean(), env_pre.mean()],
                     color="#6F4E3B", lw=2.2)
        axes[1].plot([i + 0.14, i + 0.22], [env_post.mean(), env_post.mean()],
                     color="#3E8F8A", lw=2.2)
    axes[1].set_xticks(range(len(envs)))
    axes[1].set_xticklabels(["smooth", "rough", "multi"])
    axes[1].set_ylabel("integrated TV")
    axes[1].set_title("paired TV before/after")
    axes[1].legend(frameon=False, loc="upper right")

    # Panel C: zoom on the empirical ratio and state the loose deterministic bound.
    for i, env in enumerate(envs):
        env_ratios = np.array([r[7] for r in rows if r[0] == env], dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(env_ratios))
        axes[2].scatter(i + jitter, env_ratios, s=24, color=colors[env], alpha=0.9)
        axes[2].plot([i - 0.12, i + 0.12], [env_ratios.mean(), env_ratios.mean()],
                     color="0.15", lw=2)
    axes[2].axhline(1, color="0.35", linestyle=":", lw=1)
    ymin = max(0.9, ratios.min() - 0.025)
    ymax = min(1.08, ratios.max() + 0.025)
    axes[2].set_ylim(ymin, ymax)
    axes[2].text(0.02, 0.93, "deterministic bound: R_M <= 2",
                 transform=axes[2].transAxes, fontsize=7.5)
    axes[2].set_xticks(range(len(envs)))
    axes[2].set_xticklabels(["smooth", "rough", "multi"])
    axes[2].set_ylabel("repair ratio R_M")
    axes[2].set_title("accuracy preservation")
    savefig(fig, "fig2_markovization_learned.pdf")
    return rows


def exp3_rates_real(seeds):
    rows = []
    models = [("smooth", 0.07), ("rough", 0.07), ("multimodal", 0.06)]
    n_list = [800, 1600, 3200, 6400, 10000]
    for model, sigma in models:
        for n in n_list:
            vals = []
            for seed in range(seeds):
                net, _ = train_score(30000 + seed + n + len(model), n, model=model, sigma=sigma)
                met = grid_quantities(net, model=model, sigma=sigma)
                vals.append(met["tv"] ** 2)
            rows.append((model, n, float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))))
    write_csv("exp3_rates_dimension.csv", ["model", "n", "tv2_mean", "tv2_se"], rows)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for model, _ in models:
        sub = [r for r in rows if r[0] == model]
        ax.errorbar([r[1] for r in sub], [r[2] for r in sub], yerr=[r[3] for r in sub],
                    marker="o", label=model)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n"); ax.set_ylabel(r"$d_{\mu,\mathrm{TV}}^2$")
    ax.set_title("Real-trained one-dimensional rate diagnostic")
    ax.legend(frameon=False)
    savefig(fig, "fig3_rates_slopes.pdf")
    slopes = []
    for model, _ in models:
        sub = [r for r in rows if r[0] == model]
        slope, _ = np.polyfit(np.log([r[1] for r in sub]), np.log([r[2] for r in sub]), 1)
        slopes.append((model, -float(slope)))
    return rows, slopes


def exp4_anchor_reference(seeds):
    rows = []
    eps_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7]
    refs = ["uniform", "poor-coverage", "empirical-kde", "mixture"]
    pool_rng = np.random.RandomState(501)
    Xpool = pool_rng.rand(2500, 1)
    Ypool = sample_kernel(pool_rng, Xpool, model="smooth", sigma=0.07)
    for ref in refs:
        for eps in eps_list:
            for seed in range(seeds):
                net, _ = train_score(40000 + seed + int(1000 * eps) + len(ref), 3200,
                                     model="smooth", sigma=0.07, eps=eps, ref=ref,
                                     width=32, depth=2, epochs=7, marginal_samples=Ypool)
                met = grid_quantities(net, model="smooth", sigma=0.07, eps=eps, ref=ref,
                                      marginal_samples=Ypool)
                rows.append((ref, eps, seed, (1 - eps) ** -1, met["excess"], met["a_l2"],
                             met["k_l2"], met["tv"], met["neg"], met["rowerr"]))
    write_csv("exp4_anchor_reference.csv",
              ["reference", "epsilon", "seed", "inverse_factor", "excess", "a_l2", "k_l2", "tv", "negmass", "rowerr"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ref in refs:
        for col, ax, ylabel in [(7, axes[0], "TV"), (4, axes[1], "excess"), (8, axes[2], "NegMass")]:
            sub = [r for r in rows if r[0] == ref]
            xs = sorted(set(r[1] for r in sub))
            ys = [np.mean([r[col] for r in sub if r[1] == x]) for x in xs]
            es = [np.std([r[col] for r in sub if r[1] == x]) / math.sqrt(seeds) for x in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", label=ref)
            ax.set_xscale("log"); ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(ylabel)
    axes[0].set_title("reconstruction")
    axes[1].set_title("contrastive fit")
    axes[2].set_title("de-anchor invalidity")
    axes[2].legend(frameon=False, fontsize=6.5)
    savefig(fig, "fig4_anchor_reference.pdf")
    return rows


def sample_finite_transitions(rng, K, n, iid=True):
    S = K.shape[0]
    if iid:
        X = rng.randint(0, S, size=n)
        Y = np.array([rng.choice(S, p=K[x]) for x in X])
        return X, Y
    X = np.empty(n, dtype=int)
    Y = np.empty(n, dtype=int)
    state = rng.randint(0, S)
    for i in range(n):
        X[i] = state
        state = rng.choice(S, p=K[state])
        Y[i] = state
    return X, Y


def estimate_finite_kernel(X, Y, S, alpha=0.05):
    counts = np.zeros((S, S))
    for x, y in zip(X, Y):
        counts[x, y] += 1
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * S)


def finite_contrastive_excess(Khat, K0, eps=0.1, tau=5.0):
    S = K0.shape[0]
    r = 1.0 / S
    a0 = (1 - eps) * K0 + eps * r
    ah = (1 - eps) * Khat + eps * r
    q = tau * r
    Fh = -a0 * np.log(ah / (ah + q)) - q * np.log(q / (ah + q))
    F0 = -a0 * np.log(a0 / (a0 + q)) - q * np.log(q / (a0 + q))
    return float(np.mean(np.sum(Fh - F0, axis=1)))


def exp5_trajectory_real(seeds):
    rows = []
    q_rows = []
    S = 40
    U = np.ones((S, S)) / S
    alphas = [0.02, 0.05, 0.1, 0.2, 0.5]
    T_list = [1000, 2000, 5000, 10000, 20000]
    for alpha in alphas:
        K0 = (1 - alpha) * np.eye(S) + alpha * U
        for T in T_list:
            for method in ["iid", "full", "thin"]:
                vals, risks = [], []
                for seed in range(seeds):
                    rng = np.random.RandomState(50000 + seed + T + int(1000 * alpha))
                    X, Y = sample_finite_transitions(rng, K0, T, iid=(method == "iid"))
                    if method == "thin":
                        q = max(2, int(round(1.0 / alpha)))
                        idx = np.arange(0, T, q)
                        X, Y = X[idx], Y[idx]
                    Kh = estimate_finite_kernel(X, Y, S, alpha=0.05)
                    vals.append(finite_tv(K0, Kh))
                    risks.append(finite_contrastive_excess(Kh, K0))
                rows.append((alpha, T, method, float(np.mean(vals)), float(np.std(vals) / math.sqrt(seeds)),
                             float(np.mean(risks))))
        for q in [1, 2, 5, 10, 20, 50, 100]:
            vals = []
            for seed in range(seeds):
                rng = np.random.RandomState(55000 + seed + int(1000 * alpha))
                X, Y = sample_finite_transitions(rng, K0, 10000, iid=False)
                idx = np.arange(0, 10000, q)
                Kh = estimate_finite_kernel(X[idx], Y[idx], S, alpha=0.05)
                vals.append(finite_tv(K0, Kh))
            q_rows.append((alpha, q, float(np.mean(vals)), float(np.std(vals) / math.sqrt(seeds)),
                           math.exp(-alpha * q)))
    write_csv("exp5_trajectory_real.csv", ["alpha", "T", "method", "tv", "se", "heldout_excess"], rows)
    write_csv("exp5_thinning_real.csv", ["alpha", "q", "tv", "se", "beta_proxy"], q_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for alpha in alphas:
        sub = [r for r in rows if r[0] == alpha and r[2] == "full"]
        axes[0].errorbar([r[1] for r in sub], [r[3] for r in sub], yerr=[r[4] for r in sub], marker="o", label=str(alpha))
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_title("full trajectory")
    axes[0].set_xlabel("T"); axes[0].set_ylabel("held-out TV")
    for alpha in alphas:
        sub = [r for r in q_rows if r[0] == alpha]
        axes[1].errorbar([r[1] for r in sub], [r[2] for r in sub], yerr=[r[3] for r in sub], marker="o", label=str(alpha))
        axes[1].plot([r[1] for r in sub], [0.12 * r[4] for r in sub], "--", alpha=0.4)
    axes[1].set_xscale("log"); axes[1].set_title("true error vs thinning q"); axes[1].set_xlabel("q")
    alpha = 0.05
    for method in ["iid", "full", "thin"]:
        sub = [r for r in rows if r[0] == alpha and r[2] == method]
        axes[2].errorbar([r[1] for r in sub], [r[3] for r in sub], yerr=[r[4] for r in sub], marker="o", label=method)
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_title(r"iid/full/thin, $\alpha=.05$")
    axes[2].set_xlabel("sample size"); axes[2].legend(frameon=False)
    savefig(fig, "fig5_trajectory_real.pdf")
    return rows, q_rows


def exp6_dynamic_learned(seeds):
    rows = []
    horizons = [1, 2, 5, 10, 20, 50]
    xi = np.ones(54) / 54
    for n in [1000, 3200, 6400]:
        for eps in [0.05, 0.1, 0.2]:
            for seed in range(seeds):
                net, _ = train_score(60000 + seed + n + int(1000 * eps), n, model="smooth", sigma=0.07, eps=eps)
                met = grid_quantities(net, model="smooth", sigma=0.07, eps=eps, nx=54, ny=54)
                ktv = finite_tv(met["Ptrue"], met["Pmark"], xi)
                roll = rollout_tv(met["Ptrue"], met["Pmark"], xi, horizons)
                p10 = path_tv(met["Ptrue"], met["Pmark"], xi, 10)
                o10 = occupation_tv(met["Ptrue"], met["Pmark"], xi, 10)
                rows.append((n, eps, seed, ktv, p10, o10) + tuple(roll))
    rare_rows = []
    for delta in [0.01, 0.02, 0.05]:
        Krare = np.array([[1., 0.], [1., 0.]])
        Lrare = np.array([[1., 0.], [0., 1.]])
        murare = np.array([1 - delta, delta])
        xirare = np.array([0., 1.])
        rare_rows.append((delta, finite_tv(Krare, Lrare, murare), rollout_tv(Krare, Lrare, xirare, [1])[0]))
    stat_rows = []
    S = 30
    U = np.ones((S, S)) / S
    for alpha in [0.05, 0.1, 0.2, 0.5]:
        K = (1 - alpha) * np.eye(S) + alpha * U
        for seed in range(seeds):
            rng = np.random.RandomState(65000 + seed + int(1000 * alpha))
            L = row_markovize_prob(K + rng.normal(0, 0.002, K.shape))
            piK = np.ones(S) / S
            piL = np.linalg.matrix_power(L.T, 200).dot(np.ones(S) / S)
            piL = piL / piL.sum()
            dinf = np.max(0.5 * np.sum(np.abs(K - L), axis=1))
            stat_rows.append((alpha, 1.0 / alpha, dinf / alpha, float(0.5 * np.abs(piK - piL).sum())))
    write_csv("exp6_dynamic_learned.csv", ["n", "epsilon", "seed", "kernel_tv", "path10_tv_upper", "occ10_tv"] + ["roll_%d" % h for h in horizons], rows)
    write_csv("exp6_rare.csv", ["delta", "design_tv", "one_step_rollout_tv"], rare_rows)
    write_csv("exp6_stationary.csv", ["alpha", "amplification", "bound_proxy", "stationary_tv"], stat_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25))
    ktv = np.array([r[3] for r in rows])
    for h in [1, 5, 10, 20, 50]:
        yh = np.array([r[6 + horizons.index(h)] for r in rows])
        axes[0].scatter(ktv, yh, s=13, alpha=0.42, label="m=%d" % h)
    lim = max(ktv.max(), max([max([r[6 + horizons.index(h)] for r in rows]) for h in [1, 5, 10, 20, 50]])) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=1.0)
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_xlabel(r"$d_{\mu,\mathrm{TV}}(\widehat K,K_0)$")
    axes[0].set_ylabel("rollout TV")
    axes[0].set_title("learned-kernel transfer")
    axes[0].legend(frameon=False, fontsize=6.0)
    q1, q2 = np.quantile(ktv, [1 / 3, 2 / 3])
    for label, mask, color in [
        ("low", ktv <= q1, "#4C78A8"),
        ("middle", (ktv > q1) & (ktv <= q2), "#72B7B2"),
        ("high", ktv > q2, "#E45756"),
    ]:
        means, ses = [], []
        for h in horizons:
            vals = np.array([r[6 + horizons.index(h)] for r in rows])[mask]
            means.append(vals.mean())
            ses.append(vals.std() / math.sqrt(len(vals)))
        axes[1].errorbar(horizons, means, yerr=ses, marker="o", color=color, label=label)
    axes[1].set_xscale("log"); axes[1].set_xlabel("horizon m")
    axes[1].set_ylabel("rollout TV"); axes[1].set_title("horizon growth")
    axes[1].legend(frameon=False, title="kernel TV")
    deltas = np.array([r[0] for r in rare_rows])
    amplification = np.array([r[2] / r[1] for r in rare_rows])
    axes[2].bar([str(d) for d in deltas], amplification, color="#B279A2", width=0.55)
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"rare-state design mass $\delta$")
    axes[2].set_ylabel("rollout/design TV")
    axes[2].set_title("coverage failure")
    for i, amp in enumerate(amplification):
        axes[2].text(i, amp * 1.08, "%.0fx" % amp, ha="center", va="bottom", fontsize=7)
    savefig(fig, "fig6_dynamic_transfer_learned.pdf")
    return rows, rare_rows, stat_rows


def exp7_ablation(seeds):
    variants = [
        ("Ours", 0.1, "uniform", True, True, 5, 32, True),
        ("unanchored", 0.0, "uniform", True, True, 5, 32, True),
        ("no-deanchor", 0.1, "uniform", False, True, 5, 32, True),
        ("no-Markov", 0.1, "uniform", True, False, 5, 32, False),
        ("poor-coverage-reference", 0.1, "poor-coverage", True, True, 5, 32, True),
        ("small-eps", 0.01, "uniform", True, True, 5, 32, True),
        ("large-eps", 0.7, "uniform", True, True, 5, 32, True),
        ("small-network", 0.1, "uniform", True, True, 5, 16, True),
        ("few-negatives", 0.1, "uniform", True, True, 1, 32, True),
        ("many-negatives", 0.1, "uniform", True, True, 15, 32, True),
    ]
    rows = []
    xi = np.ones(54) / 54
    for name, eps, ref, deanchor, markov, tau, width, rollout_ok in variants:
        vals = []
        for seed in range(seeds):
            net, _ = train_score(70000 + seed + len(name), 3200, model="multimodal", sigma=0.06,
                                 eps=eps, tau=tau, ref=ref, width=width, depth=2, epochs=7)
            met = grid_quantities(net, model="multimodal", sigma=0.06, eps=eps, tau=tau, ref=ref,
                                  markovize=markov, deanchor=deanchor, nx=54, ny=54)
            if rollout_ok:
                roll = rollout_tv(met["Ptrue"], met["Pmark"], xi, [10])[0]
            else:
                roll = np.nan
            tv = met["tv"] if markov else met["tv_pre"]
            vals.append((met["excess"], tv, met["neg"], met["rowerr"], roll))
        vals = np.array(vals, dtype=float)
        means = []
        ses = []
        for j in range(vals.shape[1]):
            col = vals[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                means.append(np.nan)
                ses.append(np.nan)
            else:
                means.append(float(finite.mean()))
                ses.append(float(finite.std() / math.sqrt(finite.size)))
        rows.append((name,) + tuple(means) + tuple(ses))
    write_csv("exp7_ablation.csv",
              ["variant", "excess", "tv", "negmass", "rowerr", "rollout_tv",
               "excess_se", "tv_se", "negmass_se", "rowerr_se", "rollout_se"], rows)
    M = np.array([[r[1], r[2], max(r[3], 1e-8), max(r[4], 1e-8), r[5] if np.isfinite(r[5]) else np.nan] for r in rows])
    baseline = np.maximum(M[0:1], 1e-8)
    Mn = M / baseline
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    masked = np.ma.masked_invalid(np.log10(np.maximum(Mn, 1e-3)))
    im = ax.imshow(masked, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(5)); ax.set_xticklabels(["Excess", "TV", "NegMass", "RowErr", "Rollout"])
    fig.colorbar(im, ax=ax, label="log10 relative to Ours")
    ax.set_title("Real-trained ablation study")
    savefig(fig, "fig7_ablation_heatmap.pdf")
    return rows


def exp8_runtime():
    rows = []
    for n in [800, 1600, 3200, 6400]:
        for tau in [3, 5, 10]:
            seed = 80000 + n + tau
            rng = np.random.RandomState(seed)
            t0 = time.perf_counter()
            Z, labels = build_contrastive_data(rng, n, model="smooth", sigma=0.07, eps=0.1, tau=tau)
            data_time = time.perf_counter() - t0
            net = ReLUContrastiveNet(2, width=24, depth=2, seed=seed + 17)
            t1 = time.perf_counter()
            net.fit(Z, labels, epochs=5, batch_size=512, lr=2e-3)
            train_time = time.perf_counter() - t1
            score_time, deanchor_time, mark_time, metric_time, _, _, _ = timed_grid_pipeline(
                net, model="smooth", sigma=0.07, eps=0.1, tau=tau, nx=42, ny=42
            )
            total_eval = score_time + deanchor_time + mark_time + metric_time
            rows.append(("continuous", n, tau, data_time, train_time, score_time,
                         deanchor_time, mark_time, metric_time, total_eval))
    for S in [50, 100, 200, 500]:
        K = np.ones((S, S)) / S
        t0 = time.perf_counter()
        _ = row_markovize_prob(K + 0.001 * np.random.randn(S, S))
        mt = time.perf_counter() - t0
        rows.append(("finite", S, 0, 0.0, 0.0, 0.0, 0.0, mt, 0.0, mt))
    write_csv("exp8_runtime.csv",
              ["setting", "size", "negatives", "data_time", "training_time",
               "score_eval_time", "deanchor_time", "markovization_time",
               "metric_time", "total_eval_time"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25))
    continuous = [r for r in rows if r[0] == "continuous"]
    for tau in [3, 5, 10]:
        sub = [r for r in continuous if r[2] == tau]
        axes[0].scatter([r[1] * (1 + r[2]) for r in sub], [r[4] for r in sub], s=28, label="M=%d" % tau)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("measured training cost")
    axes[0].set_xlabel(r"contrastive examples $n(1+M)$")
    axes[0].set_ylabel("seconds")
    axes[0].legend(frameon=False)
    reps = [("800,M=3", 800, 3), ("3200,M=5", 3200, 5), ("6400,M=10", 6400, 10)]
    x = np.arange(len(reps))
    data_vals, train_vals, mark_vals, eval_vals = [], [], [], []
    for _, n, tau in reps:
        match = [r for r in rows if r[0] == "continuous" and r[1] == n and r[2] == tau][0]
        data_vals.append(match[3])
        train_vals.append(match[4])
        mark_vals.append(match[7])
        eval_vals.append(match[5] + match[6] + match[8])
    axes[1].bar(x, data_vals, label="data", color="#B279A2")
    axes[1].bar(x, train_vals, bottom=data_vals, label="training", color="#4C78A8")
    axes[1].bar(x, eval_vals, bottom=np.array(data_vals) + np.array(train_vals), label="evaluation", color="#72B7B2")
    axes[1].bar(x, mark_vals, bottom=np.array(data_vals) + np.array(train_vals) + np.array(eval_vals), label="Markovization", color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels([r[0] for r in reps], rotation=15)
    axes[1].set_ylabel("seconds")
    axes[1].set_title("measured time decomposition")
    axes[1].legend(frameon=False, fontsize=6.5)
    finite = [r for r in rows if r[0] == "finite"]
    axes[2].plot([r[1] for r in finite], [r[7] for r in finite], marker="o", color="#F58518")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("states S"); axes[2].set_ylabel("seconds")
    axes[2].set_title("measured finite-state Markovization")
    savefig(fig, "fig8_runtime_scalability.pdf")
    return rows


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "N/A"
    return "%.3g" % float(x)


def write_tables(exp1, exp3, exp7, exp8, seeds):
    theory = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{p{0.20\textwidth}p{0.38\textwidth}p{0.32\textwidth}}
\toprule
Experiment & Theory tested & Main metric\\
\midrule
Calibration & excess-risk calibration & $L^2$ error\\
Markovization & valid kernel reconstruction & NegMass, RowErr, $R_M$\\
Rates & H\"older--ReLU oracle diagnostic & trained 1D TV rate\\
Anchor & anchor/de-anchor tradeoff & TV, excess, invalidity\\
Trajectory & thinning interface stress test & held-out TV and risk\\
Dynamics & finite-horizon transfer and coverage & rollout TV, occupation TV, path-law TV upper bound\\
Ablation & component necessity and final validity & numerical validity, theory coverage, TV, rollout\\
Runtime & measured implementation cost & wall-clock components\\
\bottomrule
\end{tabular}
\caption{Theory-to-experiment map for the end-to-end simulation suite.}
\label{tab:theory-experiment-map}
\end{table}
"""
    dataset = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{p{0.20\textwidth}p{0.40\textwidth}p{0.30\textwidth}}
\toprule
Module & Synthetic models & Main parameters\\
\midrule
Calibration & smooth, multimodal, rough wrapped-normal kernels & $n=800$--$6400$, %d seeds\\
Markovization & trained contrastive scores & $n=1000$--$6400$\\
Rates & real-trained one-dimensional smoothness models & $n=800$--$10000$\\
Anchor & matched sampler/density reference laws, including a poor-coverage stress test & $\varepsilon=0.01$--$0.7$\\
Trajectory & single-chain and i.i.d. finite transitions & $\alpha=0.02$--$0.5$, $q=1$--$100$\\
Dynamics & learned contrastive kernels and rare-state failures & horizons $1$--$50$\\
\bottomrule
\end{tabular}
\caption{Datasets, synthetic models, and parameter grids.  All training-based figures are generated end-to-end by the public script; deterministic diagnostic examples are labeled separately.}
\label{tab:experimental-models}
\end{table}
""" % seeds
    with open(os.path.join(V2, "table1_theory_map.tex"), "w") as f:
        f.write(theory)
    with open(os.path.join(V2, "table2_models.tex"), "w") as f:
        f.write(dataset)
    with open(os.path.join(V2, "table5_rate_slopes.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\footnotesize\n\\begin{tabular}{lc}\n\\toprule\nModel & fitted slope for $d_{\\mu,\\mathrm{TV}}^2$\\\\\n\\midrule\n")
        for model, slope in exp3[1]:
            f.write("%s & %.2f\\\\\n" % (model, slope))
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{OLS slopes from the real-trained one-dimensional rate diagnostic.  The table is descriptive and is not a high-dimensional benchmark.}\n\\label{tab:rate-slopes}\n\\end{table}\n")
    lookup = {r[0]: r for r in exp7}
    gb_tv, gb_roll = gaussian_cde_baseline(seed=123, model="multimodal", sigma=0.06)
    rows = [
        ("Ours", "full interface", "Yes", "Yes", lookup["Ours"][2], lookup["Ours"][5]),
        ("Ours-noMarkov", "score without repair", "No", "No", lookup["no-Markov"][2], None),
        ("Anchored-noDeanchor", "wrong target kernel", "Yes", r"No: targets $A_{\eps,\nu}K_0$", lookup["no-deanchor"][2], lookup["no-deanchor"][5]),
        ("Unanchored-NCE", "accurate but no restart chart", "Yes, after Markovization", "No", lookup["unanchored"][2], lookup["unanchored"][5]),
        ("Gaussian-CDE", "row-normalized baseline", "Yes", "No", gb_tv, gb_roll),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Method & Role & Numerically valid? & Covered by chart theory? & $d_{\mu,\mathrm{TV}}$ & rollout TV\\", r"\midrule"]
    for name, role, numerical, theory, tv, rollout in rows:
        lines.append("%s & %s & %s & %s & %s & %s\\\\" % (name, role, numerical, theory, fmt(tv), fmt(rollout)))
    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              r"\caption{Interface-oriented method comparison after end-to-end training.  Numerical validity records whether the evaluated object is a transition kernel; theory coverage records whether the estimator is covered by the anchored-chart reconstruction theory.  Gaussian-CDE is a row-normalized circular Gaussian regression baseline.  No-Markov is not rolled out because it is not a valid transition kernel.}",
              r"\label{tab:method-comparison}", r"\end{table}"]
    with open(os.path.join(V2, "table3_method_comparison.tex"), "w") as f:
        f.write("\n".join(lines))
    numerical = {
        "Ours": "Yes", "unanchored": "Yes", "no-deanchor": "Yes",
        "no-Markov": "No", "poor-coverage-reference": "Yes", "small-eps": "Yes",
        "large-eps": "Yes", "small-network": "Yes", "few-negatives": "Yes",
        "many-negatives": "Yes",
    }
    theory_cov = {
        "Ours": "Yes", "unanchored": "No", "no-deanchor": "No: wrong target",
        "no-Markov": "No", "poor-coverage-reference": "stress test", "small-eps": "Yes",
        "large-eps": "Yes", "small-network": "Yes", "few-negatives": "Yes",
        "many-negatives": "Yes",
    }
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Variant & Excess & TV & Pre-M NegMass & Pre-M RowErr & Num. valid? & Theory? & RolloutTV\\", r"\midrule"]
    for r in exp7:
        neg = "N/A" if r[0] == "no-deanchor" else fmt(r[3])
        row = "N/A" if r[0] == "no-deanchor" else fmt(r[4])
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s\\\\" %
                     (r[0], fmt(r[1]), fmt(r[2]), neg, row,
                      numerical.get(r[0], "Yes"), theory_cov.get(r[0], "Yes"), fmt(r[5])))
    lines += [r"\bottomrule", r"\end{tabular}", r"}",
              r"\caption{Ablation summary from real-trained variants.  Pre-M NegMass and Pre-M RowErr refer only to the de-anchored score before Markovization.  N/A means that the corresponding pre-Markovization diagnostic or rollout is not defined for that ablation.  Numerical validity is separated from coverage by the anchored-chart theory.  The no-Markov row is not rolled out because it is not a valid transition kernel.}",
              r"\label{tab:ablation-summary}", r"\end{table}"]
    with open(os.path.join(V2, "table4_ablation.tex"), "w") as f:
        f.write("\n".join(lines))
    coverage = r"""\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{cccc}
\toprule
Rare-state mass $\delta$ & $d_{\mu,\mathrm{TV}}(K,L)$ & rare-state rollout TV & amplification\\
\midrule
0.01 & 0.01 & 1.00 & $100\times$\\
0.02 & 0.02 & 1.00 & $50\times$\\
0.05 & 0.05 & 1.00 & $20\times$\\
\bottomrule
\end{tabular}
\caption{Coverage-failure diagnostic.  A design-averaged one-step error can be made small by assigning small design mass to a rare state, while rollout from that state remains maximally wrong.  This table visualizes the obstruction in Proposition~\ref{prop:limitation}.}
\label{tab:coverage-failure}
\end{table}
"""
    with open(os.path.join(V2, "table7_coverage_failure.tex"), "w") as f:
        f.write(coverage)


def main():
    global V2
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=str, default=V2)
    args = parser.parse_args()
    V2 = os.path.abspath(args.out)
    ensure_dir(V2)
    style()
    exp1 = exp1_end_to_end(args.seeds)
    exp2_markovization_learned(args.seeds)
    exp3 = exp3_rates_real(args.seeds)
    exp4_anchor_reference(args.seeds)
    exp5_trajectory_real(args.seeds)
    exp6_dynamic_learned(args.seeds)
    exp7 = exp7_ablation(args.seeds)
    exp8 = exp8_runtime()
    write_tables(exp1, exp3, exp7, exp8, args.seeds)
    print("wrote results to %s" % V2)


if __name__ == "__main__":
    main()
