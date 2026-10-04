"""Evaluation metrics for rare-event probability forecasts."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-6


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def logloss(y, p):
    p = clip(p); y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins=10, strategy="quantile"):
    """Expected calibration error with equal-mass (quantile) or equal-width bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def calib_slope_intercept(y, p):
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: a=0, b=1."""
    import statsmodels.api as sm
    y = np.asarray(y); lp = np.log(clip(p) / (1 - clip(p)))
    if y.sum() < 5:
        return np.nan, np.nan
    try:
        res = sm.Logit(y, sm.add_constant(lp)).fit(disp=0, maxiter=100)
        return float(res.params[0]), float(res.params[1])
    except Exception:
        return np.nan, np.nan


def citl_ratio(y, p):
    """Calibration-in-the-large: mean predicted / observed rate."""
    return float(np.mean(p) / max(np.mean(y), 1e-12))


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan


def all_metrics(y, p):
    a, b = calib_slope_intercept(y, p)
    return dict(auc=auc(y, p), ap=ap(y, p), brier=brier(y, p), logloss=logloss(y, p),
                ece=ece(y, p), cal_int=a, cal_slope=b, citl=citl_ratio(y, p))


def selective_gain(y, p, u, fracs=(0.05, 0.10, 0.20)):
    """Brier score on the retained set after dropping the top-q most uncertain forecasts,
    minus Brier after dropping q at random (negative = uncertainty helps)."""
    y = np.asarray(y); p = np.asarray(p); u = np.asarray(u)
    out = {}
    order = np.argsort(-u)
    n = len(y)
    base = brier(y, p)
    for q in fracs:
        k = int(round(q * n))
        keep = np.ones(n, bool); keep[order[:k]] = False
        out[f"sel_brier_{int(q*100)}"] = brier(y[keep], p[keep])
        # random-drop reference (average of 50 draws); gain<0 means uncertainty ranking beats random removal
        rng = np.random.default_rng(0)
        rb = []
        for _ in range(50):
            kp = np.ones(n, bool); kp[rng.permutation(n)[:k]] = False
            rb.append(brier(y[kp], p[kp]))
        out[f"sel_rand_{int(q*100)}"] = float(np.mean(rb))
        out[f"sel_gain_{int(q*100)}"] = out[f"sel_brier_{int(q*100)}"] - float(np.mean(rb))
    return out


def delong_test(y, p1, p2):
    """Paired DeLong test for the difference of two correlated AUCs. Returns (auc1, auc2, z, p)."""
    from scipy import stats
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    def comps(p):
        p = np.asarray(p)
        px, py = p[pos], p[neg]
        # placement values
        V10 = np.empty(m); V01 = np.empty(n)
        order = np.argsort(py); pys = py[order]
        lo = np.searchsorted(pys, px, "left"); hi = np.searchsorted(pys, px, "right")
        V10 = (lo + 0.5 * (hi - lo)) / n
        order2 = np.argsort(px); pxs = px[order2]
        lo2 = np.searchsorted(pxs, py, "left"); hi2 = np.searchsorted(pxs, py, "right")
        V01 = 1.0 - (lo2 + 0.5 * (hi2 - lo2)) / m
        return V10, V01
    a1, b1 = comps(p1); a2, b2 = comps(p2)
    auc1, auc2 = a1.mean(), a2.mean()
    s10 = np.cov(np.vstack([a1, a2])); s01 = np.cov(np.vstack([b1, b2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (auc1 - auc2) / np.sqrt(var) if var > 0 else np.nan
    return float(auc1), float(auc2), float(z), float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan


def cluster_bootstrap_diff(y, pA, pB, groups, metric, n_boot=1000, seed=0):
    """Paired cluster bootstrap (by firm) of metric(A) - metric(B). Returns (diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pA = np.asarray(pA); pB = np.asarray(pB)
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    d0 = metric(y, pA) - metric(y, pB)
    ds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        ds[b] = metric(y[idx], pA[idx]) - metric(y[idx], pB[idx])
    ds = ds[np.isfinite(ds)]
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975)), float(min(p, 1.0))
