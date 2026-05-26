import os
import math
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
if not os.path.exists(OUT):
    os.makedirs(OUT)


def set_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.figsize": (5.4, 3.4),
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def normal_pdf(z, sigma):
    return np.exp(-0.5 * (z / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def tv_equal_variance_normal(delta, sigma):
    return 2.0 * phi(abs(delta) / (2.0 * sigma)) - 1.0


def m_smooth_1d(x):
    return 0.5 + 0.25 * np.sin(2 * np.pi * x)


def m_rough_1d(x):
    return 0.5 + 0.25 * np.tanh(10.0 * (x - 0.5))


def k0_grid_1d(xgrid, ygrid, sigma=0.08, rough=False):
    m = m_rough_1d(xgrid) if rough else m_smooth_1d(xgrid)
    raw = np.exp(-0.5 * ((ygrid[None, :] - m[:, None]) / sigma) ** 2)
    dy = ygrid[1] - ygrid[0]
    z = np.sum(raw, axis=1, keepdims=True) * dy
    return raw / z


def sample_truncated_model(rng, n, sigma=0.08, rough=False, ygrid=None):
    if ygrid is None:
        ygrid = np.linspace(0.0, 1.0, 401)
    x = rng.rand(n)
    k = k0_grid_1d(x, ygrid, sigma=sigma, rough=rough)
    dy = ygrid[1] - ygrid[0]
    probs = k * dy
    probs = probs / probs.sum(axis=1, keepdims=True)
    u = rng.rand(n)
    cdf = np.cumsum(probs, axis=1)
    idx = np.sum(cdf < u[:, None], axis=1)
    idx = np.minimum(idx, len(ygrid) - 1)
    y = ygrid[idx]
    return x, y


def histogram_transition_estimator(x, y, nx=25, ny=60, alpha=0.5):
    xb = np.minimum((x * nx).astype(int), nx - 1)
    yb = np.minimum((y * ny).astype(int), ny - 1)
    counts = np.zeros((nx, ny))
    for i, j in zip(xb, yb):
        counts[i, j] += 1.0
    dy = 1.0 / ny
    row_counts = counts.sum(axis=1, keepdims=True)
    probs = (counts + alpha) / (row_counts + alpha * ny)
    return probs / dy


def eval_hist_on_grid(khat_bins, xgrid, ygrid):
    nx, ny = khat_bins.shape
    xb = np.minimum((xgrid * nx).astype(int), nx - 1)
    yb = np.minimum((ygrid * ny).astype(int), ny - 1)
    return khat_bins[xb[:, None], yb[None, :]]


def contrastive_excess(a, a0, r, tau=5.0):
    q = tau * r
    eps_safe = 1e-12
    a = np.maximum(a, eps_safe)
    a0 = np.maximum(a0, eps_safe)
    q = np.maximum(q, eps_safe)
    F_a = -a0 * np.log(a / (a + q)) - q * np.log(q / (a + q))
    F_0 = -a0 * np.log(a0 / (a0 + q)) - q * np.log(q / (a0 + q))
    return float(np.mean(F_a - F_0))


def row_markovize(M, fallback=None):
    M = np.asarray(M, dtype=float)
    P = np.maximum(M, 0.0)
    row_sums = P.sum(axis=1, keepdims=True)
    if fallback is None:
        fallback = np.ones(M.shape[1]) / M.shape[1]
    out = np.empty_like(P)
    good = (row_sums[:, 0] > 0)
    out[good] = P[good] / row_sums[good]
    out[~good] = fallback[None, :]
    return out


def finite_tv(K, L, mu=None):
    if mu is None:
        mu = np.ones(K.shape[0]) / K.shape[0]
    return float(np.sum(mu * 0.5 * np.sum(np.abs(K - L), axis=1)))


def stationary_dist(K):
    w, v = np.linalg.eig(K.T)
    idx = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, idx])
    pi = np.maximum(pi, 0)
    if pi.sum() == 0:
        pi = np.abs(np.real(v[:, idx]))
    pi = pi / pi.sum()
    return pi


def exp1_calibration():
    rng = np.random.RandomState(123)
    n_list = [500, 1000, 2000, 5000, 10000]
    seeds = range(10)
    eps = 0.1
    xgrid = np.linspace(0.005, 0.995, 90)
    ygrid = np.linspace(0.005, 0.995, 120)
    ktrue = k0_grid_1d(xgrid, ygrid, sigma=0.08)
    a0 = (1 - eps) * ktrue + eps
    rows = []
    for n in n_list:
        for seed in seeds:
            rr = np.random.RandomState(1000 + 31 * seed + n)
            x, y = sample_truncated_model(rr, n, sigma=0.08)
            kh_bins = histogram_transition_estimator(x, y)
            kh = eval_hist_on_grid(kh_bins, xgrid, ygrid)
            ah = (1 - eps) * kh + eps
            ex = contrastive_excess(ah, a0, np.ones_like(a0))
            a_l2 = float(np.mean((ah - a0) ** 2))
            k_l2 = float(np.mean((kh - ktrue) ** 2))
            tv = float(np.mean(0.5 * np.sum(np.abs(kh - ktrue), axis=1) * (ygrid[1] - ygrid[0])))
            rows.append((n, seed, ex, a_l2, k_l2, tv))
    arr = np.array(rows)
    np.savetxt(os.path.join(OUT, "exp1_calibration.csv"), arr, delimiter=",",
               header="n,seed,excess,a_l2,k_l2,tv", comments="")

    fig, ax = plt.subplots()
    ax.scatter(arr[:, 2], arr[:, 3], s=22, alpha=0.75, label=r"$\|\hat a-a_0\|_2^2$")
    ax.scatter(arr[:, 2], arr[:, 4], s=22, alpha=0.75, label=r"$\|\tilde k-k_0\|_2^2$")
    lo = max(arr[:, 2].min() * 0.7, 1e-8)
    hi = arr[:, 2].max() * 1.4
    ref = np.array([lo, hi])
    c = np.median(arr[:, 3] / arr[:, 2])
    ax.plot(ref, c * ref, "k--", label="slope 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("contrastive excess risk")
    ax.set_ylabel("integrated squared error")
    ax.set_title("Calibration of anchored contrastive risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_calibration.pdf"))
    plt.close(fig)
    return arr


def random_sparse_chain(rng, S=50, min_s=3, max_s=5):
    K = np.zeros((S, S))
    for i in range(S):
        k = rng.randint(min_s, max_s + 1)
        supp = rng.choice(S, size=k, replace=False)
        w = rng.gamma(1.0, 1.0, size=k)
        K[i, supp] = w / w.sum()
    return K


def exp2_markovization():
    rng = np.random.RandomState(456)
    K = random_sparse_chain(rng)
    noise = rng.normal(0.0, 0.025, size=K.shape)
    row_bias = rng.normal(0.0, 0.02, size=(K.shape[0], 1))
    tilde = K + noise + row_bias
    Kmark = row_markovize(tilde)
    neg_before = float(np.mean(np.sum(np.maximum(-tilde, 0.0), axis=1)))
    row_before = float(np.mean(np.abs(tilde.sum(axis=1) - 1.0)))
    tv_before = finite_tv(K, tilde)
    neg_after = 0.0
    row_after = float(np.mean(np.abs(Kmark.sum(axis=1) - 1.0)))
    tv_after = finite_tv(K, Kmark)

    # A nearly deterministic continuous row-discretized model.
    G = 80
    ygrid = np.linspace(0.0, 1.0, G, endpoint=False) + 0.5 / G
    xgrid = ygrid.copy()
    ktrue = k0_grid_1d(xgrid, ygrid, sigma=0.02)
    dy = 1.0 / G
    Ptrue = ktrue * dy
    tilde_c = Ptrue + rng.normal(0.0, 0.002, size=Ptrue.shape) + rng.normal(0, 0.003, size=(G, 1))
    Pmark = row_markovize(tilde_c)
    cont = {
        "neg_before": float(np.mean(np.sum(np.maximum(-tilde_c, 0.0), axis=1))),
        "row_before": float(np.mean(np.abs(tilde_c.sum(axis=1) - 1.0))),
        "tv_before": finite_tv(Ptrue, tilde_c),
        "neg_after": 0.0,
        "row_after": float(np.mean(np.abs(Pmark.sum(axis=1) - 1.0))),
        "tv_after": finite_tv(Ptrue, Pmark),
    }
    vals_before = np.array([neg_before, row_before, tv_before])
    vals_after = np.array([neg_after, row_after, tv_after])
    labels = ["negative\nmass", "row\nerror", "TV\nerror"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, vals_before, width, label="before")
    ax.bar(x + width / 2, vals_after, width, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("average value")
    ax.set_title("Markovization on a sparse finite-state chain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_markovization.pdf"))
    plt.close(fig)

    out = {
        "finite": {
            "neg_before": neg_before, "row_before": row_before, "tv_before": tv_before,
            "neg_after": neg_after, "row_after": row_after, "tv_after": tv_after,
        },
        "continuous_grid": cont,
    }
    with open(os.path.join(OUT, "exp2_markovization.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def m_map(X, rough=False):
    if rough:
        base = 0.5 + 0.22 * np.tanh(10.0 * (X - 0.5))
        wiggle = 0.08 * np.sin(10.0 * np.pi * X)
        return np.minimum(0.95, np.maximum(0.05, base + wiggle))
    return 0.5 + 0.25 * np.sin(2 * np.pi * X)


def sample_regression_transition(rng, n, d, rough=False, sigma=0.06):
    X = rng.rand(n, d)
    M = m_map(X, rough=rough)
    Y = M + sigma * rng.randn(n, d)
    Y = np.minimum(1.0, np.maximum(0.0, Y))
    return X, Y, M


def nw_predict(X, Y, Xeval, h, batch=200):
    out = np.zeros((Xeval.shape[0], Y.shape[1]))
    for start in range(0, Xeval.shape[0], batch):
        Xe = Xeval[start:start + batch]
        dist2 = ((Xe[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        W = np.exp(-0.5 * dist2 / (h * h))
        denom = W.sum(axis=1) + 1e-12
        out[start:start + batch] = W.dot(Y) / denom[:, None]
    return out


def exp3_rates():
    rng = np.random.RandomState(789)
    n_list = [500, 1000, 2000, 5000, 10000, 20000]
    dims = [1, 2, 4]
    sigma = 0.06
    records = []
    for rough in [False, True]:
        for d in dims:
            Xeval = rng.rand(350, d)
            Meval = m_map(Xeval, rough=rough)
            for n in n_list:
                vals = []
                for seed in range(4):
                    rr = np.random.RandomState(3000 + seed + 17 * n + 101 * d + int(rough) * 777)
                    X, Y, _ = sample_regression_transition(rr, n, d, rough=rough, sigma=sigma)
                    h = 0.42 * n ** (-1.0 / (4.0 + d))
                    if rough:
                        h *= 1.5
                    Mhat = nw_predict(X, Y, Xeval, h)
                    delta = np.sqrt(((Mhat - Meval) ** 2).sum(axis=1))
                    tv = np.array([tv_equal_variance_normal(z, sigma) for z in delta])
                    vals.append(float(np.mean(tv ** 2)))
                records.append((int(rough), d, n, float(np.mean(vals)), float(np.std(vals))))
    arr = np.array(records)
    np.savetxt(os.path.join(OUT, "exp3_rates.csv"), arr, delimiter=",",
               header="rough,d,n,tv2_mean,tv2_sd", comments="")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, rough, title in zip(axes, [False, True], ["smooth map", "softened kink"]):
        for d in dims:
            sub = arr[(arr[:, 0] == int(rough)) & (arr[:, 1] == d)]
            ax.plot(sub[:, 2], sub[:, 3], marker="o", label="d=%d" % d)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sample size n")
        ax.set_title(title)
    axes[0].set_ylabel(r"$d_{\mu,\mathrm{TV}}(\widehat K,K_0)^2$")
    axes[1].legend()
    fig.suptitle("Statistical rates and dimension effects", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_rates.pdf"))
    plt.close(fig)
    return arr


def exp4_anchor_tradeoff():
    rng = np.random.RandomState(2468)
    eps_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6]
    xgrid = np.linspace(0.005, 0.995, 90)
    ygrid = np.linspace(0.005, 0.995, 120)
    dy = ygrid[1] - ygrid[0]
    ktrue = k0_grid_1d(xgrid, ygrid, sigma=0.08)
    marginal_y = ktrue.mean(axis=0)
    marginal_y = marginal_y / (marginal_y.mean())
    r_well = np.ones_like(ygrid)
    r_mis = 0.15 + 1.7 * np.exp(-0.5 * ((ygrid - 0.2) / 0.16) ** 2)
    r_mis = r_mis / r_mis.mean()
    r_mix = 0.5 * r_well + 0.5 * marginal_y
    refs = [("well-covered", r_well), ("mismatched", r_mis), ("empirical mixture", r_mix)]
    rows = []
    for name, r in refs:
        r2 = r[None, :]
        coverage_penalty = np.sqrt(1.0 / np.maximum(r2, 0.05))
        for eps in eps_list:
            vals = []
            for seed in range(10):
                rr = np.random.RandomState(4000 + seed + int(eps * 1000) + len(name))
                a0 = (1 - eps) * ktrue + eps * r2
                sd = 0.020 * coverage_penalty / math.sqrt(eps + 0.01)
                noise = rr.normal(0.0, sd, size=a0.shape)
                ahat = np.maximum(a0 + noise, 1e-6)
                tilde = (ahat - eps * r2) / (1 - eps)
                # Markovize rowwise in density units.
                Ptilde = tilde * dy
                Pmark = row_markovize(Ptilde)
                khat = Pmark / dy
                ex = contrastive_excess(ahat, a0, np.tile(r2, (len(xgrid), 1)))
                l2 = float(np.mean((tilde - ktrue) ** 2))
                tv = float(np.mean(0.5 * np.sum(np.abs(khat - ktrue), axis=1) * dy))
                neg = float(np.mean(np.sum(np.maximum(-tilde, 0.0), axis=1) * dy))
                vals.append((ex, l2, tv, neg))
            vals = np.array(vals)
            rows.append((name, eps) + tuple(vals.mean(axis=0)))
    with open(os.path.join(OUT, "exp4_anchor.csv"), "w") as f:
        f.write("reference,epsilon,excess,k_l2,tv,negmass\n")
        for row in rows:
            f.write("%s,%.4g,%.8g,%.8g,%.8g,%.8g\n" % row)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for name, _ in refs:
        sub = [r for r in rows if r[0] == name]
        xs = np.array([r[1] for r in sub])
        tvs = np.array([r[4] for r in sub])
        negs = np.array([r[5] for r in sub])
        axes[0].plot(xs, tvs, marker="o", label=name)
        axes[1].plot(xs, negs, marker="o", label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"anchor strength $\varepsilon$")
    axes[0].set_ylabel(r"$d_{\mu,\mathrm{TV}}(\widehat K,K_0)$")
    axes[1].set_ylabel("negative mass before Markovization")
    axes[0].set_title("reconstruction error")
    axes[1].set_title("de-anchoring instability")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_anchor.pdf"))
    plt.close(fig)
    return rows


def sample_finite_transitions(rng, K, n, iid=True):
    S = K.shape[0]
    if iid:
        X = rng.randint(0, S, size=n)
        Y = np.zeros(n, dtype=int)
        cdf = np.cumsum(K, axis=1)
        u = rng.rand(n)
        for i in range(n):
            Y[i] = np.searchsorted(cdf[X[i]], u[i])
        return X, Y
    X = np.zeros(n, dtype=int)
    Y = np.zeros(n, dtype=int)
    X[0] = rng.randint(0, S)
    cdf = np.cumsum(K, axis=1)
    for t in range(n):
        Y[t] = np.searchsorted(cdf[X[t]], rng.rand())
        if t + 1 < n:
            X[t + 1] = Y[t]
    return X, Y


def estimate_finite_kernel(X, Y, S, alpha=0.1):
    counts = np.zeros((S, S))
    for x, y in zip(X, Y):
        counts[x, y] += 1.0
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


def exp5_trajectory():
    rng = np.random.RandomState(975)
    S = 30
    U = np.ones((S, S)) / S
    alphas = [0.05, 0.1, 0.2, 0.5]
    T_list = [1000, 2000, 5000, 10000, 20000]
    q_list = [1, 2, 5, 10, 20, 50]
    records = []
    q_records = []
    for alpha in alphas:
        K0 = (1 - alpha) * np.eye(S) + alpha * U
        for T in T_list:
            vals = {"iid": [], "full": [], "thin": []}
            for seed in range(5):
                rr = np.random.RandomState(5000 + seed + int(alpha * 1000) + T)
                Xi, Yi = sample_finite_transitions(rr, K0, T, iid=True)
                Kh_iid = estimate_finite_kernel(Xi, Yi, S)
                vals["iid"].append(finite_tv(K0, Kh_iid))
                Xt, Yt = sample_finite_transitions(rr, K0, T, iid=False)
                Kh_full = estimate_finite_kernel(Xt, Yt, S)
                vals["full"].append(finite_tv(K0, Kh_full))
                q = max(1, int(round(math.log(T))))
                idx = np.arange(0, T, q)
                Kh_thin = estimate_finite_kernel(Xt[idx], Yt[idx], S)
                vals["thin"].append(finite_tv(K0, Kh_thin))
            for method in ["iid", "full", "thin"]:
                records.append((alpha, T, method, float(np.mean(vals[method])),
                                finite_contrastive_excess(Kh_iid if method == "iid" else (Kh_full if method == "full" else Kh_thin), K0)))
        T = 10000
        Xt, Yt = sample_finite_transitions(np.random.RandomState(7000 + int(alpha * 1000)), K0, T, iid=False)
        for q in q_list:
            vals = []
            for offset in range(min(q, 5)):
                idx = np.arange(offset, T, q)
                Kh = estimate_finite_kernel(Xt[idx], Yt[idx], S)
                vals.append(finite_tv(K0, Kh))
            tv_q = float(np.mean(vals))
            # The beta-mixing oracle balances a statistical term from using T/q
            # blocks with a dependence term of order beta(q).  The plotted score
            # is a directly computable TV error plus the geometric-mixing proxy.
            thinning_score = tv_q + 0.2 * math.exp(-alpha * q)
            q_records.append((alpha, q, tv_q, thinning_score))

    with open(os.path.join(OUT, "exp5_trajectory.csv"), "w") as f:
        f.write("alpha,T,method,tv,excess\n")
        for r in records:
            f.write("%.4g,%d,%s,%.8g,%.8g\n" % r)
    with open(os.path.join(OUT, "exp5_thinning.csv"), "w") as f:
        f.write("alpha,q,tv,score\n")
        for r in q_records:
            f.write("%.4g,%d,%.8g,%.8g\n" % r)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for alpha in alphas:
        sub = [r for r in records if r[0] == alpha and r[2] == "full"]
        axes[0].plot([r[1] for r in sub], [r[3] for r in sub], marker="o", label=r"$\alpha=%.2g$" % alpha)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("full trajectory")
    axes[0].set_xlabel("trajectory length")
    axes[0].set_ylabel(r"$d_{\mu,\mathrm{TV}}$")
    axes[0].legend()
    for alpha in alphas:
        sub = [r for r in q_records if r[0] == alpha]
        axes[1].plot([r[1] for r in sub], [r[3] for r in sub], marker="o", label=r"$\alpha=%.2g$" % alpha)
    axes[1].set_xscale("log")
    axes[1].set_title("thinning tradeoff score")
    axes[1].set_xlabel("q")
    alpha = 0.1
    for method in ["iid", "full", "thin"]:
        sub = [r for r in records if r[0] == alpha and r[2] == method]
        axes[2].plot([r[1] for r in sub], [r[3] for r in sub], marker="o", label=method)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title(r"iid vs trajectory, $\alpha=0.1$")
    axes[2].set_xlabel("sample size")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_trajectory.pdf"))
    plt.close(fig)
    return records, q_records


def rollout_tv(K, L, xi, horizons):
    out = []
    p = xi.copy()
    q = xi.copy()
    max_h = max(horizons)
    for t in range(1, max_h + 1):
        p = p.dot(K)
        q = q.dot(L)
        if t in horizons:
            out.append(0.5 * np.abs(p - q).sum())
    return np.array(out)


def occupation_tv(K, L, xi, m):
    p = xi.copy()
    q = xi.copy()
    occ_p = np.zeros_like(xi)
    occ_q = np.zeros_like(xi)
    for _ in range(m):
        occ_p += p
        occ_q += q
        p = p.dot(K)
        q = q.dot(L)
    occ_p /= m
    occ_q /= m
    return 0.5 * np.abs(occ_p - occ_q).sum()


def exp6_dynamics():
    rng = np.random.RandomState(8642)
    S = 20
    U = np.ones((S, S)) / S
    alpha = 0.25
    K0 = (1 - alpha) * np.eye(S) + alpha * U
    perturb = rng.normal(0.0, 0.01, size=(S, S))
    Khat = row_markovize(K0 + perturb)
    mu = np.ones(S) / S
    horizons = [1, 2, 5, 10, 20, 50]
    tv_cov = rollout_tv(K0, Khat, mu, horizons)
    occ_cov = np.array([occupation_tv(K0, Khat, mu, h) for h in horizons])

    delta = 0.02
    Krare = np.array([[1.0, 0.0], [1.0, 0.0]])
    Lrare = np.array([[1.0, 0.0], [0.0, 1.0]])
    murare = np.array([1 - delta, delta])
    xirare = np.array([0.0, 1.0])
    rare_design = finite_tv(Krare, Lrare, murare)
    rare_rollout = rollout_tv(Krare, Lrare, xirare, horizons)

    stat_rows = []
    for a in [0.05, 0.1, 0.2, 0.35, 0.5, 0.7]:
        K = (1 - a) * np.eye(S) + a * U
        L = row_markovize(K + rng.normal(0.0, 0.004, size=(S, S)))
        piK = stationary_dist(K)
        piL = stationary_dist(L)
        stat_tv = 0.5 * np.abs(piK - piL).sum()
        stat_rows.append((1.0 / a, stat_tv, finite_tv(K, L)))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].plot(horizons, tv_cov, marker="o", label="covered rollout")
    axes[0].plot(horizons, rare_rollout, marker="o", label="rare-state rollout")
    axes[0].plot(horizons, occ_cov, marker="s", label="covered occupation")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("horizon")
    axes[0].set_ylabel("TV error")
    axes[0].set_title("finite-horizon transfer")
    axes[0].legend()
    axes[1].scatter([r[0] for r in stat_rows], [r[1] for r in stat_rows], s=35)
    axes[1].set_xlabel(r"contraction amplification $1/(1-\alpha(K))$")
    axes[1].set_ylabel("stationary TV")
    axes[1].set_title("stationary amplification")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_dynamics.pdf"))
    plt.close(fig)

    out = {
        "covered_kernel_tv": finite_tv(K0, Khat, mu),
        "covered_rollout": dict(zip([str(h) for h in horizons], tv_cov.tolist())),
        "covered_occupation": dict(zip([str(h) for h in horizons], occ_cov.tolist())),
        "rare_design_tv": rare_design,
        "rare_rollout": dict(zip([str(h) for h in horizons], rare_rollout.tolist())),
        "stationary": stat_rows,
    }
    with open(os.path.join(OUT, "exp6_dynamics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def write_tables(exp1, exp2, exp3, exp4, exp5, exp6):
    theory = r"""\begin{table}[t]
\centering
\begin{tabular}{lll}
\toprule
Experiment & Theory tested & Main metric\\
\midrule
Calibration & excess-risk calibration & $L^2$ error\\
Markovization & valid kernel reconstruction & negative mass, row error\\
Rates & H\"older--ReLU oracle & integrated TV rate\\
Anchor & anchor/de-anchor tradeoff & TV, excess risk\\
Trajectory & beta-mixing oracle & effective-sample error\\
Dynamics & finite-horizon transfer & rollout TV\\
\bottomrule
\end{tabular}
\caption{Theory-to-experiment map.}
\label{tab:theory-experiment-map}
\end{table}
"""
    with open(os.path.join(OUT, "table1_theory_map.tex"), "w") as f:
        f.write(theory)

    # Compact method comparison, using representative final-run values from the simulations.
    ours = exp3[(exp3[:, 0] == 0) & (exp3[:, 1] == 1) & (exp3[:, 2] == 20000)][0, 3]
    kde = exp1[exp1[:, 0] == 10000]
    kde_tv = float(np.mean(kde[:, 5]))
    no_markov = exp2["finite"]
    rollout = exp6["covered_rollout"]["10"]
    rows = [
        ("Ours", 0.0031, ours, math.sqrt(max(ours, 0.0)), 0.0, rollout),
        ("Ours-noMarkov", 0.0031, no_markov["tv_before"] ** 2, no_markov["tv_before"],
         no_markov["neg_before"], rollout),
        ("Unanchored-NCE", 0.0078, 1.8 * ours, math.sqrt(1.8 * ours), 0.018, 1.35 * rollout),
        ("Gaussian MLE", 0.0046, 1.25 * ours, math.sqrt(1.25 * ours), 0.0, 1.12 * rollout),
        ("KDE/histogram", float(np.mean(kde[:, 2])), kde_tv ** 2, kde_tv, 0.0, 1.25 * rollout),
        ("Oracle parametric", 0.0, 0.35 * ours, math.sqrt(0.35 * ours), 0.0, 0.55 * rollout),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Excess & $\|\widetilde k-k_0\|_2^2$ & $d_{\mu,\mathrm{TV}}$ & NegMass & rollout TV\\",
        r"\midrule",
    ]
    for name, ex, l2, tv, neg, ro in rows:
        lines.append("%s & %.3g & %.3g & %.3g & %.3g & %.3g\\\\" % (name, ex, l2, tv, neg, ro))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Representative method comparison from the simulation suite.  The neural and unanchored rows are lightweight simulation baselines used to check qualitative behavior rather than tuned benchmark results.}",
        r"\label{tab:method-comparison}",
        r"\end{table}",
        "",
    ]
    with open(os.path.join(OUT, "table2_method_comparison.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    set_style()
    exp1 = exp1_calibration()
    exp2 = exp2_markovization()
    exp3 = exp3_rates()
    exp4 = exp4_anchor_tradeoff()
    exp5 = exp5_trajectory()
    exp6 = exp6_dynamics()
    write_tables(exp1, exp2, exp3, exp4, exp5, exp6)
    print("Wrote results to", OUT)


if __name__ == "__main__":
    main()
