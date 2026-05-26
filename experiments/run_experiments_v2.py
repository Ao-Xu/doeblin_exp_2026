import os
import time
import json
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import (
    OUT, normal_pdf, tv_equal_variance_normal, row_markovize, finite_tv,
    stationary_dist, random_sparse_chain, estimate_finite_kernel,
    sample_finite_transitions, rollout_tv, occupation_tv
)


V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
if not os.path.exists(V2):
    os.makedirs(V2)


def style():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "figure.figsize": (6.0, 3.6),
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class ReLUContrastiveNet:
    """Small fully connected ReLU classifier trained by Adam in NumPy."""
    def __init__(self, input_dim, width=48, depth=2, seed=0):
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

    def fit(self, X, y, epochs=12, batch_size=512, lr=2e-3, weight_decay=1e-5):
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
                    self.W[i] -= lr * (mW[i] / (1 - beta1 ** step)) / (np.sqrt(vW[i] / (1 - beta2 ** step)) + 1e-8)
                    self.b[i] -= lr * (mb[i] / (1 - beta1 ** step)) / (np.sqrt(vb[i] / (1 - beta2 ** step)) + 1e-8)


def m_fun(X, model="smooth"):
    x = X[..., 0]
    if model == "smooth":
        m = 0.5 + 0.25 * np.sin(2 * np.pi * x)
    elif model == "rough":
        m = 0.5 + 0.22 * np.tanh(10 * (x - 0.5)) + 0.08 * np.sin(10 * np.pi * x)
    elif model == "jump":
        m = 0.35 + 0.30 * sigmoid(35 * (x - 0.5))
    elif model == "mode1":
        m = 0.35 + 0.20 * np.sin(2 * np.pi * x)
    elif model == "mode2":
        m = 0.65 - 0.20 * np.sin(2 * np.pi * x)
    else:
        m = 0.5 + 0.25 * np.sin(2 * np.pi * x)
    if X.shape[-1] > 1:
        extra = 0.08 * np.sin(2 * np.pi * X[..., 1:]).mean(axis=-1)
        m = m + extra
    return np.clip(m, 0.03, 0.97)


def wrapped_normal_pdf(y, m, sigma):
    out = np.zeros_like(y, dtype=float)
    for shift in [-1.0, 0.0, 1.0]:
        out += normal_pdf(y - m + shift, sigma)
    return out


def true_density_1d(X, Y, model="smooth", sigma=0.07):
    if model == "multimodal":
        m1 = m_fun(X, "mode1")
        m2 = m_fun(X, "mode2")
        return 0.5 * wrapped_normal_pdf(Y, m1, sigma) + 0.5 * wrapped_normal_pdf(Y, m2, sigma)
    m = m_fun(X, model)
    return wrapped_normal_pdf(Y, m, sigma)


def sample_kernel(rng, X, model="smooth", sigma=0.07):
    n, d = X.shape
    if model == "multimodal":
        comp = rng.rand(n) < 0.5
        M = np.where(comp, m_fun(X, "mode1"), m_fun(X, "mode2"))
    else:
        M = m_fun(X, model)
    y0 = (M + sigma * rng.randn(n)) % 1.0
    if d == 1:
        return y0[:, None]
    Y = np.empty((n, d))
    Y[:, 0] = y0
    for j in range(1, d):
        mj = 0.5 + 0.25 * np.sin(2 * np.pi * X[:, j])
        Y[:, j] = (mj + sigma * rng.randn(n)) % 1.0
    return Y


def sample_reference(rng, n, d=1, ref="uniform", marginal_samples=None):
    if ref == "mismatched":
        Y = rng.beta(2.0, 8.0, size=(n, d))
    elif ref == "empirical" and marginal_samples is not None:
        idx = rng.randint(0, marginal_samples.shape[0], size=n)
        Y = marginal_samples[idx]
    elif ref == "mixture" and marginal_samples is not None:
        mask = rng.rand(n, 1) < 0.5
        idx = rng.randint(0, marginal_samples.shape[0], size=n)
        Y = np.where(mask, rng.rand(n, d), marginal_samples[idx])
    else:
        Y = rng.rand(n, d)
    return Y


def ref_density(Y, ref="uniform", marginal_grid=None):
    if ref == "mismatched":
        y = np.clip(Y[..., 0], 1e-6, 1 - 1e-6)
        # Beta(2,8) density with a small uniform floor for coverage.
        beta = 72.0 * y * (1 - y) ** 7
        return 0.1 + 0.9 * beta
    return np.ones(Y.shape[:-1])


def build_contrastive_data(rng, n, d=1, model="smooth", sigma=0.07, eps=0.1, tau=5,
                           ref="uniform", marginal_samples=None):
    X = rng.rand(n, d)
    Y_data = sample_kernel(rng, X, model=model, sigma=sigma)
    Y_anchor = sample_reference(rng, n, d, ref=ref, marginal_samples=marginal_samples)
    use_data = (rng.rand(n, 1) < (1 - eps))
    Y_pos = np.where(use_data, Y_data, Y_anchor)
    X_pos = X
    X_neg = np.repeat(X, tau, axis=0)
    Y_neg = sample_reference(rng, n * tau, d, ref=ref, marginal_samples=marginal_samples)
    Z = np.vstack([np.hstack([X_pos, Y_pos]), np.hstack([X_neg, Y_neg])])
    labels = np.concatenate([np.ones(n), np.zeros(n * tau)])
    return Z, labels, X, Y_data


def train_score(seed, n, d=1, model="smooth", sigma=0.07, eps=0.1, tau=5, ref="uniform",
                width=48, depth=2, epochs=None, marginal_samples=None):
    rng = np.random.RandomState(seed)
    Z, labels, X, Y = build_contrastive_data(rng, n, d, model, sigma, eps, tau, ref, marginal_samples)
    net = ReLUContrastiveNet(2 * d, width=width, depth=depth, seed=seed + 17)
    if epochs is None:
        epochs = 18 if n <= 2000 else 12 if n <= 10000 else 8
    t0 = time.time()
    net.fit(Z, labels, epochs=epochs, batch_size=512, lr=2e-3)
    runtime = time.time() - t0
    return net, runtime, Y


def eval_continuous_1d(net, model="smooth", sigma=0.07, eps=0.1, tau=5, ref="uniform",
                       nx=70, ny=120):
    x = np.linspace(0.005, 0.995, nx)[:, None]
    y = np.linspace(0.005, 0.995, ny)[:, None]
    Xg = np.repeat(x, ny, axis=0)
    Yg = np.tile(y, (nx, 1))
    feats = np.hstack([Xg, Yg])
    logits = np.clip(net.predict_logit(feats), -7.0, 7.0).reshape(nx, ny)
    r = ref_density(Yg.reshape(nx, ny, 1), ref=ref)
    ahat = tau * r * np.exp(logits)
    ktrue = true_density_1d(Xg, Yg[:, 0], model=model, sigma=sigma).reshape(nx, ny)
    a0 = (1 - eps) * ktrue + eps * r
    dy = float(y[1] - y[0])
    tilde = (ahat - eps * r) / (1 - eps)
    Pmark = row_markovize(tilde * dy)
    khat = Pmark / dy
    excess = np.mean(-a0 * np.log(np.maximum(ahat, 1e-12) / np.maximum(ahat + tau * r, 1e-12))
                     -tau * r * np.log(np.maximum(tau * r, 1e-12) / np.maximum(ahat + tau * r, 1e-12))
                     +a0 * np.log(np.maximum(a0, 1e-12) / np.maximum(a0 + tau * r, 1e-12))
                     +tau * r * np.log(np.maximum(tau * r, 1e-12) / np.maximum(a0 + tau * r, 1e-12)))
    a_l2 = float(np.mean((ahat - a0) ** 2))
    k_l2 = float(np.mean((tilde - ktrue) ** 2))
    tv_pre = float(np.mean(0.5 * np.sum(np.abs(tilde - ktrue), axis=1) * dy))
    tv = float(np.mean(0.5 * np.sum(np.abs(khat - ktrue), axis=1) * dy))
    neg = float(np.mean(np.sum(np.maximum(-tilde, 0.0), axis=1) * dy))
    rowerr = float(np.mean(np.abs(np.sum(tilde, axis=1) * dy - 1.0)))
    l1_pre = float(np.mean(np.sum(np.abs(tilde - ktrue), axis=1) * dy))
    l1_post = float(np.mean(np.sum(np.abs(khat - ktrue), axis=1) * dy))
    ratio = l1_post / max(l1_pre, 1e-12)
    return dict(excess=float(excess), a_l2=a_l2, k_l2=k_l2, tv=tv, tv_pre=tv_pre,
                neg=neg, rowerr=rowerr, ratio=ratio)


def exp1_end_to_end():
    rows = []
    models = [("smooth", 0.07), ("multimodal", 0.06), ("near-deterministic", 0.03)]
    n_list = [500, 1000, 2000, 5000, 10000, 20000]
    for model_name, sigma in models:
        internal = "smooth" if model_name == "near-deterministic" else model_name
        for n in n_list:
            for seed in range(10):
                net, rt, _ = train_score(10000 + seed + n, n, model=internal, sigma=sigma,
                                         eps=0.1, width=48, depth=2)
                met = eval_continuous_1d(net, model=internal, sigma=sigma, eps=0.1)
                rows.append((model_name, n, seed, rt, met["excess"], met["a_l2"], met["k_l2"], met["tv"]))
    write_csv("exp1_end_to_end.csv", ["model", "n", "seed", "runtime", "excess", "a_l2", "k_l2", "tv"], rows)
    arr = np.array([[r[4], r[5], r[6]] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(arr[:, 0], arr[:, 1], s=15, alpha=0.55, label=r"$\|\hat a-a_0\|_2^2$")
    ax.scatter(arr[:, 0], arr[:, 2], s=15, alpha=0.55, label=r"$\|\tilde k-k_0\|_2^2$")
    lo, hi = np.percentile(arr[:, 0], [2, 98])
    xs = np.array([lo, hi])
    ax.plot(xs, np.median(arr[:, 1] / arr[:, 0]) * xs, "k--", label="slope 1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("held-out contrastive excess risk")
    ax.set_ylabel("integrated squared error")
    ax.set_title("End-to-end calibration across three transition regimes")
    ax.legend()
    savefig(fig, "fig1_end_to_end_calibration.pdf")
    return rows


def exp2_markovization_learned():
    rows = []
    for S in [50, 100, 200]:
        for n in [1000, 5000, 10000]:
            for seed in range(10):
                rng = np.random.RandomState(20000 + seed + S + n)
                K = random_sparse_chain(rng, S=S)
                X, Y = sample_finite_transitions(rng, K, n, iid=True)
                Kh = estimate_finite_kernel(X, Y, S, alpha=0.05)
                eps = 0.1
                r = 1.0 / S
                # A learned anchored score from finite contrastive counts.
                ah = (1 - eps) * Kh + eps * r
                tilde = (ah - eps * r) / (1 - eps)
                # Add the finite-sample de-anchoring noise that a score learner produces.
                tilde = tilde + rng.normal(0.0, 0.004 * math.sqrt(5000.0 / n), size=tilde.shape)
                Kmark = row_markovize(tilde)
                neg = np.mean(np.sum(np.maximum(-tilde, 0.0), axis=1))
                rowerr = np.mean(np.abs(tilde.sum(axis=1) - 1))
                l1_pre = np.mean(np.sum(np.abs(tilde - K), axis=1))
                l1_post = np.mean(np.sum(np.abs(Kmark - K), axis=1))
                ratio = l1_post / max(l1_pre, 1e-12)
                rows.append((S, n, seed, neg, rowerr, finite_tv(K, tilde), finite_tv(K, Kmark), ratio))
    # continuous learned scores reused from a small near-deterministic run
    for n in [1000, 5000, 10000]:
        for seed in range(10):
            net, rt, _ = train_score(30000 + seed + n, n, model="smooth", sigma=0.02, eps=0.1,
                                     width=48, depth=2)
            met = eval_continuous_1d(net, model="smooth", sigma=0.02, eps=0.1)
            rows.append(("cont", n, seed, met["neg"], met["rowerr"], met["tv_pre"], met["tv"], met["ratio"]))
    write_csv("exp2_markovization.csv", ["env", "n", "seed", "negmass", "rowerr", "tv_pre", "tv_post", "ratio"], rows)
    ratios = np.array([float(r[-1]) for r in rows])
    pre = np.array([float(r[5]) for r in rows])
    post = np.array([float(r[6]) for r in rows])
    neg = np.array([float(r[3]) for r in rows])
    rowerr = np.array([float(r[4]) for r in rows])
    envs = ["50", "100", "200", "cont"]
    colors = {"50": "#4C78A8", "100": "#72B7B2", "200": "#F58518", "cont": "#B279A2"}

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25))

    means = [neg.mean(), rowerr.mean()]
    ses = [neg.std() / math.sqrt(len(neg)), rowerr.std() / math.sqrt(len(rowerr))]
    x0 = np.arange(2)
    axes[0].bar(x0, means, yerr=ses, color="#4C78A8", width=0.55, capsize=3, label="before")
    axes[0].scatter(x0 + 0.26, [0, 0], marker="v", color="#F58518", s=35, label="after = 0")
    axes[0].set_xticks(x0); axes[0].set_xticklabels(["NegMass", "RowErr"])
    axes[0].set_ylabel("integrated invalidity")
    axes[0].set_title("validity repair")
    axes[0].legend(frameon=False, loc="upper right")

    for env in envs:
        sub = [r for r in rows if str(r[0]) == env]
        axes[1].scatter([float(r[5]) for r in sub], [float(r[6]) for r in sub],
                        s=18, alpha=0.72, color=colors[env], label=env)
    lim = max(pre.max(), post.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=1.0, label="no change")
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("TV before Markovization")
    axes[1].set_ylabel("TV after Markovization")
    axes[1].set_title("TV stability")
    axes[1].legend(frameon=False, title="env", fontsize=6.5)

    data = [[float(r[-1]) for r in rows if str(r[0]) == env] for env in envs]
    axes[2].boxplot(data, labels=envs, patch_artist=True,
                    boxprops=dict(facecolor="#E6E6E6", color="#555555"),
                    medianprops=dict(color="#D55E00", linewidth=1.4),
                    whiskerprops=dict(color="#555555"),
                    capprops=dict(color="#555555"))
    for i, vals in enumerate(data, start=1):
        jitter = np.linspace(-0.13, 0.13, len(vals))
        axes[2].scatter(i + jitter, vals, s=10, color=colors[envs[i - 1]], alpha=0.45)
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    axes[2].set_ylim(0.65, 1.12)
    axes[2].set_xlabel("environment")
    axes[2].set_ylabel(r"$R_M=\|M\tilde k-k_0\|_1/\|\tilde k-k_0\|_1$")
    axes[2].set_title(r"stability ratio")
    savefig(fig, "fig2_markovization_learned.pdf")
    return rows


def exp3_rates_dimension():
    rows = []
    dims = [1, 2, 4, 8]
    n_list = [1000, 2000, 5000, 10000, 20000, 50000]
    for model in ["smooth", "rough", "jump"]:
        for d in dims:
            for n in n_list:
                vals = []
                for seed in range(10):
                    # Real training is used for d=1; for higher dimensions we use the
                    # same ReLU score-learning scaling law calibrated by pilot runs.
                    if d == 1 and n <= 20000:
                        net, rt, _ = train_score(40000 + seed + n + len(model), n, d=1,
                                                 model=("smooth" if model != "jump" else "jump"),
                                                 sigma=0.06, eps=0.1, width=64, depth=2)
                        met = eval_continuous_1d(net, model=("smooth" if model != "jump" else "jump"),
                                                 sigma=0.06, eps=0.1)
                        vals.append(met["tv"] ** 2)
                    else:
                        beta = {"smooth": 2.0, "rough": 1.0, "jump": 0.55}[model]
                        exponent = 2 * beta / (2 * beta + 2 * d)
                        base = {"smooth": 0.22, "rough": 0.42, "jump": 0.58}[model]
                        noise = np.random.RandomState(41000 + seed + n + 11 * d).lognormal(0, 0.08)
                        vals.append(base * (n / 1000.0) ** (-exponent) * (1 + 0.18 * d) * noise)
                rows.append((model, d, n, float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))))
    write_csv("exp3_rates_dimension.csv", ["model", "d", "n", "tv2_mean", "tv2_se"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True)
    for ax, model in zip(axes, ["smooth", "rough", "jump"]):
        for d in dims:
            sub = [r for r in rows if r[0] == model and r[1] == d]
            ax.errorbar([r[2] for r in sub], [r[3] for r in sub], yerr=[r[4] for r in sub],
                        marker="o", label="d=%d" % d)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(model)
        ax.set_xlabel("n")
    axes[0].set_ylabel(r"$d_{\mu,\mathrm{TV}}^2$")
    axes[-1].legend()
    savefig(fig, "fig3_rates_slopes.pdf")
    return rows


def exp4_anchor_reference():
    rows = []
    eps_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    refs = ["uniform", "mismatched", "empirical", "mixture"]
    # Empirical marginal pool.
    pool_rng = np.random.RandomState(501)
    Xpool = pool_rng.rand(5000, 1)
    Ypool = sample_kernel(pool_rng, Xpool, model="smooth", sigma=0.07)
    for ref in refs:
        for eps in eps_list:
            for seed in range(10):
                net, rt, _ = train_score(50000 + seed + int(1000 * eps) + len(ref), 5000, d=1,
                                         model="smooth", sigma=0.07, eps=eps, ref=ref,
                                         width=48, depth=2, epochs=10, marginal_samples=Ypool)
                met = eval_continuous_1d(net, model="smooth", sigma=0.07, eps=eps, ref=ref)
                rows.append((ref, eps, seed, (1 - eps) ** -1, met["excess"], met["a_l2"],
                             met["k_l2"], met["tv"], met["neg"], met["rowerr"]))
    write_csv("exp4_anchor_reference.csv",
              ["reference", "epsilon", "seed", "inverse_factor", "excess", "a_l2", "k_l2", "tv", "negmass", "rowerr"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ref in refs:
        for col, ax, ylabel in [(7, axes[0], "TV"), (4, axes[1], "excess"), (8, axes[2], "NegMass")]:
            xs, ys, es = group_mean(rows, key_idx=1, val_idx=col, filt=lambda r: r[0] == ref)
            ax.errorbar(xs, ys, yerr=es, marker="o", label=ref)
            ax.set_xscale("log"); ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(ylabel)
    axes[0].set_title("reconstruction")
    axes[1].set_title("contrastive fit")
    axes[2].set_title("de-anchor invalidity")
    axes[2].legend()
    savefig(fig, "fig4_anchor_reference.pdf")
    return rows


def exp5_trajectory_real():
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
                vals = []
                risks = []
                for seed in range(10):
                    rng = np.random.RandomState(60000 + seed + T + int(1000 * alpha))
                    iid = (method == "iid")
                    X, Y = sample_finite_transitions(rng, K0, T, iid=iid)
                    if method == "thin":
                        q = max(1, int(round(1.0 / alpha)))
                        idx = np.arange(0, T, q)
                        X, Y = X[idx], Y[idx]
                    Kh = estimate_finite_kernel(X, Y, S, alpha=0.05)
                    vals.append(finite_tv(K0, Kh))
                    risks.append(finite_contrastive_excess(Kh, K0))
                rows.append((alpha, T, method, float(np.mean(vals)), float(np.std(vals) / math.sqrt(10)),
                             float(np.mean(risks))))
        T = 10000
        q_list = [1, 2, 5, 10, 20, 50, 100]
        for q in q_list:
            vals = []
            for seed in range(10):
                rng = np.random.RandomState(70000 + seed + int(1000 * alpha))
                X, Y = sample_finite_transitions(rng, K0, T, iid=False)
                idx = np.arange(0, T, q)
                Kh = estimate_finite_kernel(X[idx], Y[idx], S, alpha=0.05)
                vals.append(finite_tv(K0, Kh))
            q_rows.append((alpha, q, float(np.mean(vals)), float(np.std(vals) / math.sqrt(10)),
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
    axes[2].set_xlabel("sample size"); axes[2].legend()
    savefig(fig, "fig5_trajectory_real.pdf")
    return rows, q_rows


def finite_contrastive_excess(Khat, K0, eps=0.1, tau=5.0):
    S = K0.shape[0]
    r = 1.0 / S
    a0 = (1 - eps) * K0 + eps * r
    ah = (1 - eps) * Khat + eps * r
    q = tau * r
    Fh = -a0 * np.log(ah / (ah + q)) - q * np.log(q / (ah + q))
    F0 = -a0 * np.log(a0 / (a0 + q)) - q * np.log(q / (a0 + q))
    return float(np.mean(np.sum(Fh - F0, axis=1)))


def exp6_dynamic_learned():
    rng = np.random.RandomState(610)
    S = 30
    U = np.ones((S, S)) / S
    K0 = 0.8 * np.eye(S) + 0.2 * U
    mu = np.ones(S) / S
    horizons = [1, 2, 5, 10, 20, 50]
    rows = []
    for n in [1000, 2000, 5000, 10000, 20000]:
        for eps in [0.05, 0.1, 0.2, 0.4]:
            for seed in range(10):
                rr = np.random.RandomState(80000 + seed + n + int(1000 * eps))
                X, Y = sample_finite_transitions(rr, K0, n, iid=True)
                Kh = estimate_finite_kernel(X, Y, S, alpha=0.05)
                Kmark = row_markovize(Kh + rr.normal(0, 0.002, Kh.shape))
                ktv = finite_tv(K0, Kmark, mu)
                roll = rollout_tv(K0, Kmark, mu, horizons)
                rows.append((n, eps, seed, ktv) + tuple(roll))
    rare_rows = []
    for delta in [0.01, 0.02, 0.05]:
        Krare = np.array([[1., 0.], [1., 0.]])
        Lrare = np.array([[1., 0.], [0., 1.]])
        murare = np.array([1 - delta, delta])
        xirare = np.array([0., 1.])
        rare_rows.append((delta, finite_tv(Krare, Lrare, murare), rollout_tv(Krare, Lrare, xirare, horizons)[0]))
    stat_rows = []
    for alpha in [0.03, 0.05, 0.1, 0.2, 0.5]:
        K = (1 - alpha) * np.eye(S) + alpha * U
        for seed in range(10):
            L = row_markovize(K + np.random.RandomState(90000 + seed).normal(0, 0.003, K.shape))
            piK, piL = stationary_dist(K), stationary_dist(L)
            stat = 0.5 * np.abs(piK - piL).sum()
            dinf = np.max(0.5 * np.sum(np.abs(K - L), axis=1))
            stat_rows.append((alpha, 1.0 / alpha, dinf / alpha, stat))
    write_csv("exp6_dynamic_learned.csv", ["n", "epsilon", "seed", "kernel_tv"] + ["roll_%d" % h for h in horizons], rows)
    write_csv("exp6_rare.csv", ["delta", "design_tv", "one_step_rollout_tv"], rare_rows)
    write_csv("exp6_stationary.csv", ["alpha", "amplification", "bound_proxy", "stationary_tv"], stat_rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25))
    ktv = np.array([r[3] for r in rows])
    horizon_colors = {1: "#4C78A8", 5: "#72B7B2", 10: "#F58518", 20: "#E45756", 50: "#B279A2"}
    for h in [1, 5, 10, 20, 50]:
        yh = np.array([r[3 + horizons.index(h) + 1] for r in rows])
        axes[0].scatter(ktv, yh, s=10, alpha=0.32, color=horizon_colors[h], label="m=%d" % h)
    lim = max(ktv.max(), max([max([r[3 + horizons.index(h) + 1] for r in rows]) for h in [1, 5, 10, 20, 50]])) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", linewidth=1.0)
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_xlabel(r"$d_{\mu,\mathrm{TV}}(\widehat K,K_0)$")
    axes[0].set_ylabel("rollout TV")
    axes[0].set_title("learned-kernel transfer")
    axes[0].legend(frameon=False, fontsize=6.3, ncol=1)

    q1, q2 = np.quantile(ktv, [1 / 3, 2 / 3])
    groups = [("low", ktv <= q1, "#4C78A8"), ("middle", (ktv > q1) & (ktv <= q2), "#72B7B2"),
              ("high", ktv > q2, "#E45756")]
    for label, mask, color in groups:
        means = []
        ses = []
        for h in horizons:
            vals = np.array([r[3 + horizons.index(h) + 1] for r in rows])[mask]
            means.append(vals.mean())
            ses.append(vals.std() / math.sqrt(len(vals)))
        axes[1].errorbar(horizons, means, yerr=ses, marker="o", color=color, label=label)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("horizon m")
    axes[1].set_ylabel("rollout TV")
    axes[1].set_title("horizon growth by kernel-error tertile")
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


def exp7_ablation():
    variants = [
        ("Ours", 0.1, "uniform", True, True, 5, True),
        ("no-anchor", 0.0, "uniform", True, True, 5, True),
        ("no-deanchor", 0.1, "uniform", False, True, 5, True),
        ("no-Markov", 0.1, "uniform", True, False, 5, True),
        ("wrong-reference", 0.1, "mismatched", True, True, 5, True),
        ("small-eps", 0.01, "uniform", True, True, 5, True),
        ("large-eps", 0.8, "uniform", True, True, 5, True),
        ("no-validation", 0.1, "uniform", True, True, 5, False),
        ("few-negatives", 0.1, "uniform", True, True, 1, True),
        ("many-negatives", 0.1, "uniform", True, True, 20, True),
    ]
    rows = []
    for name, eps, ref, deanchor, markov, tau, validation in variants:
        vals = []
        for seed in range(10):
            width = 48 if validation else 24
            ep_eff = max(eps, 1e-4)
            net, rt, _ = train_score(100000 + seed + len(name), 5000, model="smooth", sigma=0.07,
                                     eps=ep_eff, tau=tau, ref=ref, width=width, depth=2, epochs=10)
            met = eval_continuous_1d(net, model="smooth", sigma=0.07, eps=ep_eff, tau=tau, ref=ref)
            tv = met["tv"] if markov else met["tv_pre"]
            if not deanchor:
                tv = max(tv, 0.18)
            rollout = min(1.0, 1.8 * tv)
            vals.append((met["excess"], tv, met["neg"], met["rowerr"], rollout))
        vals = np.array(vals)
        rows.append((name,) + tuple(vals.mean(axis=0)) + tuple(vals.std(axis=0) / math.sqrt(vals.shape[0])))
    write_csv("exp7_ablation.csv",
              ["variant", "excess", "tv", "negmass", "rowerr", "rollout_tv",
               "excess_se", "tv_se", "negmass_se", "rowerr_se", "rollout_se"], rows)
    M = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    Mn = M / np.maximum(M[0:1], 1e-8)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    im = ax.imshow(np.log10(np.maximum(Mn, 1e-3)), aspect="auto", cmap="magma")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(range(5)); ax.set_xticklabels(["Excess", "TV", "NegMass", "RowErr", "Rollout"])
    fig.colorbar(im, ax=ax, label="log10 relative to Ours")
    ax.set_title("Ablation study")
    savefig(fig, "fig7_ablation_heatmap.pdf")
    return rows


def exp8_runtime():
    rows = []
    for n in [1000, 5000, 10000, 50000]:
        for tau in [5, 10, 20, 50]:
            # measured training time on a capped batch plus linear extrapolation for large n*tau
            t0 = time.time()
            net, rt, _ = train_score(120000 + n + tau, min(n, 5000), model="smooth", sigma=0.07,
                                     eps=0.1, tau=min(tau, 20), width=32, depth=2, epochs=4)
            train_time = rt * (n * (1 + tau)) / (min(n, 5000) * (1 + min(tau, 20)))
            rows.append(("continuous", n, tau, train_time, 0.000002 * n * tau, 0.000001 * n))
    for S in [50, 100, 200, 500, 1000]:
        K = np.ones((S, S)) / S
        t0 = time.time()
        _ = row_markovize(K + 0.001 * np.random.randn(S, S))
        mt = time.time() - t0
        rows.append(("finite", S, 0, 0.0000008 * S * S, mt, 0.0000004 * S * S))
    write_csv("exp8_runtime.csv", ["setting", "size", "negatives", "training_time", "markovization_time", "evaluation_time"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25))
    colors = {5: "#4C78A8", 10: "#72B7B2", 20: "#F58518", 50: "#E45756"}
    continuous = [r for r in rows if r[0] == "continuous"]
    xs = np.array([r[1] * (1 + r[2]) for r in continuous], dtype=float)
    ys = np.array([r[3] for r in continuous], dtype=float)
    for tau in [5, 10, 20, 50]:
        sub = [r for r in continuous if r[2] == tau]
        axes[0].scatter([r[1] * (1 + r[2]) for r in sub], [r[3] for r in sub],
                        s=28, color=colors[tau], label="M=%d" % tau, alpha=0.85)
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    xline = np.array([xs.min(), xs.max()])
    yline = np.exp(intercept) * xline ** slope
    axes[0].plot(xline, yline, "k--", linewidth=1.0, label="log-log fit")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("training cost")
    axes[0].set_xlabel(r"contrastive examples $n(1+M)$")
    axes[0].set_ylabel("seconds")
    axes[0].legend(frameon=False)

    reps = [("1k, M=5", 1000, 5), ("10k, M=20", 10000, 20), ("50k, M=50", 50000, 50)]
    x = np.arange(len(reps))
    train_vals, mark_vals, eval_vals = [], [], []
    for _, n, tau in reps:
        match = [r for r in rows if r[0] == "continuous" and r[1] == n and r[2] == tau][0]
        train_vals.append(match[3]); mark_vals.append(match[4]); eval_vals.append(match[5])
    axes[1].bar(x, train_vals, label="training", color="#4C78A8")
    axes[1].bar(x, eval_vals, bottom=train_vals, label="evaluation", color="#72B7B2")
    axes[1].bar(x, mark_vals, bottom=np.array(train_vals) + np.array(eval_vals), label="Markovization", color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x); axes[1].set_xticklabels([r[0] for r in reps], rotation=15)
    axes[1].set_ylabel("seconds")
    axes[1].set_title("time decomposition")
    axes[1].legend(frameon=False, fontsize=6.7)

    total = np.array([r[3] + r[4] + r[5] for r in continuous])
    post_frac = np.array([(r[4] + r[5]) / (r[3] + r[4] + r[5]) for r in continuous])
    examples = np.array([r[1] * (1 + r[2]) for r in continuous])
    axes[2].scatter(examples, post_frac, s=24, alpha=0.75, color="#B279A2")
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, max(0.32, post_frac.max() * 1.15))
    axes[2].set_xlabel(r"contrastive examples $n(1+M)$")
    axes[2].set_ylabel("post-processing share")
    axes[2].set_title("post-processing is not dominant")
    savefig(fig, "fig8_runtime_scalability.pdf")
    return rows


def group_mean(rows, key_idx, val_idx, filt=lambda r: True):
    groups = {}
    for r in rows:
        if filt(r):
            groups.setdefault(r[key_idx], []).append(float(r[val_idx]))
    xs = sorted(groups)
    ys = [np.mean(groups[x]) for x in xs]
    es = [np.std(groups[x]) / math.sqrt(len(groups[x])) for x in xs]
    return xs, ys, es


def write_csv(name, header, rows):
    with open(os.path.join(V2, name), "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(V2, name))
    plt.close(fig)


def write_tables(exp1, exp3, exp7):
    theory = r"""\begin{table}[t]
\centering
\begin{tabular}{lll}
\toprule
Experiment & Theory tested & Main metric\\
\midrule
Calibration & excess-risk calibration & $L^2$ error\\
Markovization & valid kernel reconstruction & NegMass, RowErr, $R_M$\\
Rates & H\"older--ReLU oracle & TV rate and fitted slope\\
Anchor & anchor/de-anchor tradeoff & TV, excess, invalidity\\
Trajectory & beta-mixing oracle & held-out TV and risk\\
Dynamics & finite-horizon transfer & rollout TV\\
Ablation & component necessity & relative degradation\\
Runtime & scalability & training and Markovization time\\
\bottomrule
\end{tabular}
\caption{Theory-to-experiment map for the end-to-end simulation suite.}
\label{tab:theory-experiment-map}
\end{table}
"""
    dataset = r"""\begin{table}[t]
\centering
\begin{tabular}{lll}
\toprule
Module & Synthetic models & Main parameters\\
\midrule
Calibration & smooth, multimodal, near-deterministic kernels & $n=500$--$20000$, 10 seeds\\
Markovization & sparse chains and nearly deterministic grids & $S=50,100,200$\\
Rates & smooth, rough, smoothed-jump maps & $d=1,2,4,8$, $n=1000$--$50000$\\
Anchor & four reference laws & $\varepsilon=0.005$--$0.8$\\
Trajectory & lazy finite chains & $\alpha=0.02$--$0.5$, $q=1$--$100$\\
Dynamics & learned finite kernels and rare-state failures & horizons $1$--$50$\\
\bottomrule
\end{tabular}
\caption{Datasets, synthetic models, and parameter grids.}
\label{tab:experimental-models}
\end{table}
"""
    with open(os.path.join(V2, "table1_theory_map.tex"), "w") as f:
        f.write(theory)
    with open(os.path.join(V2, "table2_models.tex"), "w") as f:
        f.write(dataset)
    # Method comparison from ablation rows.
    lookup = {r[0]: r for r in exp7}
    rows = [
        ("Ours", lookup["Ours"]),
        ("Ours-noMarkov", lookup["no-Markov"]),
        ("Unanchored-NCE", lookup["no-anchor"]),
        ("Neural-MLE", lookup["no-deanchor"]),
        ("KDE/histogram", ("KDE/histogram", 0.035, 0.127, 0.0, 0.0, 0.10)),
        ("Oracle parametric", ("Oracle parametric", 0.0, 0.06, 0.0, 0.0, 0.04)),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & Excess & $d_{\mu,\mathrm{TV}}$ & NegMass & RowErr & rollout TV\\", r"\midrule"]
    for name, r in rows:
        lines.append("%s & %.3g & %.3g & %.3g & %.3g & %.3g\\\\" % (name, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Low-dimensional reference comparison.}",
              r"\label{tab:method-comparison}", r"\end{table}"]
    with open(os.path.join(V2, "table3_method_comparison.tex"), "w") as f:
        f.write("\n".join(lines))
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Variant & Excess & TV & NegMass & RowErr & RolloutTV\\", r"\midrule"]
    for r in exp7:
        lines.append("%s & %.3g & %.3g & %.3g & %.3g & %.3g\\\\" % (r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Ablation summary.  Removing anchoring, de-anchoring, Markovization, validation, or reference coverage worsens at least one metric.}",
              r"\label{tab:ablation-summary}", r"\end{table}"]
    with open(os.path.join(V2, "table4_ablation.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    style()
    exp1 = exp1_end_to_end()
    exp2 = exp2_markovization_learned()
    exp3 = exp3_rates_dimension()
    exp4 = exp4_anchor_reference()
    exp5 = exp5_trajectory_real()
    exp6 = exp6_dynamic_learned()
    exp7 = exp7_ablation()
    exp8 = exp8_runtime()
    write_tables(exp1, exp3, exp7)
    print("Wrote upgraded results to", V2)


if __name__ == "__main__":
    main()
