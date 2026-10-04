"""Classical and tree-based baselines + post-hoc recalibration wrappers."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from nn_jax import _entropy, _np_logloss, sigmoid_np


class SKModel:
    def __init__(self, est, cols_idx=None):
        self.est, self.idx = est, cols_idx

    def predict(self, X, **kw):
        X = X if self.idx is None else X[:, self.idx]
        p = self.est.predict_proba(X)[:, 1]
        return dict(p=p, u_epi=np.zeros_like(p), u_logit=np.zeros_like(p), u_tot=_entropy(p))


def fit_logit(Xtr, ytr, Xva, yva, cols_idx=None, C=1e4):
    X = Xtr if cols_idx is None else Xtr[:, cols_idx]
    est = LogisticRegression(C=C, max_iter=2000).fit(X, ytr)
    m = SKModel(est, cols_idx)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m


def fit_rf(Xtr, ytr, Xva, yva, seed=0, min_samples_leaf=20, max_features="sqrt", n_estimators=500):
    est = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                 n_jobs=2, random_state=seed).fit(Xtr, ytr)
    m = SKModel(est)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m


def fit_lgbm(Xtr, ytr, Xva, yva, seed=0, num_leaves=8, learning_rate=0.03, min_child_samples=50,
             reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8, n_estimators=2000):
    est = lgb.LGBMClassifier(num_leaves=num_leaves, learning_rate=learning_rate, min_child_samples=min_child_samples,
                             reg_lambda=reg_lambda, subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
                             n_estimators=n_estimators, random_state=seed, n_jobs=2, verbose=-1)
    est.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    m = SKModel(est)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m


class Recal:
    """Post-hoc recalibration fitted on the validation year: platt | isotonic | temp."""

    def __init__(self, base, method, Xva, yva):
        self.base, self.method = base, method
        p = np.clip(base.predict(Xva)["p"], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        if method == "platt":
            self.est = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yva)
        elif method == "isotonic":
            self.est = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, yva)
        elif method == "temp":
            Ts = np.exp(np.linspace(-1.5, 1.5, 121))
            ll = [_np_logloss(yva, sigmoid_np(z / T)) for T in Ts]
            self.T = Ts[int(np.argmin(ll))]
        else:
            raise ValueError(method)

    def predict(self, X, **kw):
        out = self.base.predict(X)
        p = np.clip(out["p"], 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
        ul = out.get("u_logit", np.zeros_like(p))
        if self.method == "platt":
            q = self.est.predict_proba(z[:, None])[:, 1]
            ul = ul * abs(float(self.est.coef_[0][0]))
        elif self.method == "isotonic":
            q = np.clip(self.est.predict(p), 1e-4, 1 - 1e-4)
        else:
            q = sigmoid_np(z / self.T)
            ul = ul / self.T
        return dict(p=q, u_epi=out["u_epi"], u_logit=ul, u_tot=_entropy(q))
