#!/usr/bin/env python3
"""Figures for the revised paper (revision 2). Palette: validated categorical slots 1-3 (blue, orange, aqua)."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

R = "../results"; OUT = "../paper_r2/figures"; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
SERIES = [("logit_all_platt", "Logit, all 24 + Platt", BLUE, "o"), ("ens5_platt", "Deep ensemble + Platt", ORANGE, "s"),
          ("vids_platt", "VIDS + Platt", AQUA, "^")]


def eo_figure():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (d, title) in zip(axs, (("main_r2", "(a) Fixed design"), ("rolling_r2", "(b) Rolling design"))):
        by = pd.read_csv(f"{R}/{d}/eval_r2/metrics_by_year.csv")
        band = by[by.model == "logit_all_platt"].set_index("t")
        yrs = list(range(2019, 2024))
        for i, t in enumerate(yrs):
            ax.add_patch(plt.Rectangle((i - 0.32, band.loc[t, "band_lo"]), 0.64, band.loc[t, "band_hi"] - band.loc[t, "band_lo"],
                                       color=GRID, lw=0, zorder=0))
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--", zorder=1)
        for j, (k, lab, col, mk) in enumerate(SERIES):
            s = by[by.model == k].set_index("t")
            x = np.arange(len(yrs)) + (j - 1) * 0.16
            ax.plot(x, [s.loc[t, "eo"] for t in yrs], color=col, lw=1.5, marker=mk, ms=4.5, label=lab, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(0.18, 3.2)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.25", "0.5", "1", "2"]))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("Expected / observed failures (log scale)")
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_eo_by_year.pdf"); plt.close(fig)


def load_seed_mean(model, field):
    files = sorted(glob.glob(f"{R}/main_r2/fixed/seed_*.pkl"))
    return np.mean([pickle.load(open(f, "rb"))[model][field] for f in files], axis=0)


def unc_figure():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (k, lab, field) in zip(axs, (("vids", "(a) VIDS", "u_epi"), ("ens5", "(b) Deep ensemble", "u_epi"),
                                          ("llla", "(c) Last-layer Laplace", "u_logit"))):
        p = load_seed_mean(k, "p"); u = load_seed_mean(k, field)
        hb = ax.hexbin(np.log10(p), np.log10(u), gridsize=45, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        ax.set_title(lab, loc="left", color=INK)
        ax.set_xlabel("Forecast $\\hat p$ (log$_{10}$)")
        ax.set_ylabel("s.d. of $\\hat p$ (log$_{10}$)" if field == "u_epi" else "s.d. of logit $\\hat p$ (log$_{10}$)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_unc_vs_risk.pdf"); plt.close(fig)


def reliability_figure():
    y = pd.read_parquet(f"{R}/main_r2/fixed/test_index.parquet")["y"].to_numpy().astype(float)
    import evaluate_r2 as E
    idx, vidx, P = E.load_dir(f"{R}/main_r2", ["fixed"])
    P = E.add_platt(P, idx, ["ens5", "vids"])
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    lim = (1e-3, 0.6)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    for k, lab, col, mk in SERIES:
        p = np.clip(P[k]["p"].mean(0), 1e-6, 1 - 1e-6)
        o = np.argsort(p); ps, ys = p[o], y[o]
        cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ps, ys).predict(ps)
        ok = cal > 0
        ax.plot(ps[ok], cal[ok], color=col, lw=1.5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Forecast probability"); ax.set_ylabel("Recalibrated (observed) probability")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_reliability.pdf"); plt.close(fig)


def toy_figure():
    sys.path.insert(0, "../src")
    import nn_jax as N

    def sg(z): return 1 / (1 + np.exp(-z))
    rng = np.random.default_rng(100)
    X = rng.beta(0.5, 0.5, 500).reshape(-1, 1).astype(np.float32); Xn = rng.beta(0.5, 0.5, 500).reshape(-1, 1).astype(np.float32)
    keep = ((X < 0.3) | (X > 0.7)).ravel(); X = X[keep]
    rho = sg(-5 + 10 * X.ravel()); Y = rng.binomial(1, rho).astype(np.float32)
    mlp = N.fit_mlp(X, Y, X, Y, seed=0, hidden=(), rep=8, lr=1e-3, wd=0.0, epochs=100, batch=32, patience=1000)
    vm = N.fit_vids(mlp, X, Y, X, Y, seed=0, J=30, m=20, n=500, K=30, tau=0.001, lam=0.005, lr=1e-3, widths=(512, 256, 128, 64, 32),
                    init_from_pretrained=False, keras_init=True, n_samples=1000, checkpoint="none", clip=None)
    grid = np.linspace(0, 1, 201).reshape(-1, 1).astype(np.float32)
    out = vm.predict(grid); pm = mlp.predict(grid)["p"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.axvspan(0.3, 0.7, color=GRID, lw=0, zorder=0)
    ax.plot(grid.ravel(), sg(-5 + 10 * grid.ravel()), color=MUTED, lw=1.2, ls="--", label="True probability")
    ax.fill_between(grid.ravel(), out["p"] - out["u_epi"], out["p"] + out["u_epi"], color=BLUE, alpha=0.18, lw=0, label="VIDS $\\pm$1 s.d.")
    ax.plot(grid.ravel(), out["p"], color=BLUE, lw=1.5, label="VIDS (our port)")
    ax.scatter(X.ravel()[:150], Y[:150], marker="x", s=10, color=INK, lw=0.6, label="Training data")
    ax.set_xlabel("$x$"); ax.set_ylabel("Probability"); ax.set_ylim(-0.05, 1.05)
    ax.text(0.5, 0.06, "no training data", ha="center", color=MUTED, fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_toy.pdf"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["eo", "unc", "reliability", "toy"]
    for w in which:
        {"eo": eo_figure, "unc": unc_figure, "reliability": reliability_figure, "toy": toy_figure}[w]()
        print("done", w, flush=True)
