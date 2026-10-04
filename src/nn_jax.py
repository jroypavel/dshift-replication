"""
Neural-network models in JAX (CPU): MLP, deep ensemble, MC-dropout, SNGP-style GP head,
last-layer Laplace, and VIDS (Slavutsky & Blei, 2025; binary-classification variant).

The VIDS loss is a line-by-line port of the authors' released TensorFlow code
(uncertainty.py: train_posterior / get_phi, binary-classification branch), with ONE optional
stabiliser that is OFF in the 'strict' variant:  init_from_pretrained=True initialises the
output bias of the posterior-mean network at the pre-trained head weights.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import optax

LOG2PI = math.log(2.0 * math.pi)


# ------------------------------------------------------------------ helpers
def init_mlp(key, sizes, last_scale=1.0):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, (a, b)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        W = jax.random.normal(k, (a, b)) * math.sqrt(2.0 / a)
        if i == len(sizes) - 2:
            W = W * last_scale
        params.append((W, jnp.zeros(b)))
    return params


def init_glorot(key, sizes):
    """Keras default initialisation (Glorot-uniform weights, zero biases), as in the authors' TensorFlow code."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (a, b) in zip(keys, zip(sizes[:-1], sizes[1:])):
        lim = math.sqrt(6.0 / (a + b))
        params.append((jax.random.uniform(k, (a, b), minval=-lim, maxval=lim), jnp.zeros(b)))
    return params


def mlp_apply(params, x, drop_p=0.0, key=None):
    h = x
    for (W, b) in params[:-1]:
        h = jax.nn.relu(h @ W + b)
        if drop_p > 0.0 and key is not None:
            key, sub = jax.random.split(key)
            h = h * jax.random.bernoulli(sub, 1.0 - drop_p, h.shape) / (1.0 - drop_p)
    W, b = params[-1]
    return h @ W + b


def bce_logits(logits, y):
    return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _np_logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))


def l2_penalty(params, wd):
    return wd * sum(jnp.sum(W ** 2) for (W, _) in params)


# ------------------------------------------------------------------ plain MLP (and MC dropout)
class MLPModel:
    def __init__(self, g, w, drop_p=0.0):
        self.g, self.w, self.drop_p = g, w, drop_p

    def embed(self, X):
        return np.asarray(mlp_apply(self.g, jnp.asarray(X)))

    def logits(self, X, key=None, drop=False):
        z = mlp_apply(self.g, jnp.asarray(X), self.drop_p if drop else 0.0, key)
        return np.asarray(z @ self.w)

    def predict(self, X, n_mc=100, seed=0):
        if self.drop_p > 0:
            zs = np.stack([self.logits(X, jax.random.PRNGKey(seed * 1000 + s), True) for s in range(n_mc)])
            ps = sigmoid_np(zs)
            p = ps.mean(0)
            return dict(p=p, u_epi=ps.std(0), u_logit=zs.std(0), u_tot=_entropy(p))
        p = sigmoid_np(self.logits(X))
        return dict(p=p, u_epi=np.zeros_like(p), u_logit=np.zeros_like(p), u_tot=_entropy(p))


def _entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def fit_mlp(Xtr, ytr, Xva, yva, seed=0, hidden=(64, 64), rep=16, lr=1e-3, wd=1e-4, epochs=150,
            batch=512, drop_p=0.0, patience=15):
    key = jax.random.PRNGKey(seed)
    k1, k2, key = jax.random.split(key, 3)
    d = Xtr.shape[1]
    g = init_mlp(k1, [d, *hidden, rep])
    w = jax.random.normal(k2, (rep,)) * 0.1
    params = {"g": g, "w": w}
    opt = optax.adam(lr)
    st = opt.init(params)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr, dtype=jnp.float32)

    @jax.jit
    def step(params, st, xb, yb, k):
        def loss(p):
            z = mlp_apply(p["g"], xb, drop_p, k if drop_p > 0 else None)
            return bce_logits(z @ p["w"], yb) + l2_penalty(p["g"], wd)
        l, gr = jax.value_and_grad(loss)(params)
        up, st = opt.update(gr, st, params)
        return optax.apply_updates(params, up), st, l

    @jax.jit
    def val_logits(params, X):
        return mlp_apply(params["g"], X) @ params["w"]

    Xva_j = jnp.asarray(Xva)
    rng = np.random.default_rng(seed)
    best, best_params, bad = np.inf, params, 0
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            key, sub = jax.random.split(key)
            params, st, _ = step(params, st, Xtr_j[idx], ytr_j[idx], sub)
        v = _np_logloss(yva, sigmoid_np(np.asarray(val_logits(params, Xva_j))))
        if v < best - 1e-5:
            best, best_params, bad = v, params, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m = MLPModel(best_params["g"], best_params["w"], drop_p)
    m.val_logloss = best
    return m


class Ensemble:
    def __init__(self, members):
        self.members = members

    def predict(self, X, **kw):
        ps = np.stack([m.predict(X)["p"] for m in self.members])
        zs = np.log(np.clip(ps, 1e-7, 1 - 1e-7) / (1 - np.clip(ps, 1e-7, 1 - 1e-7)))
        p = ps.mean(0)
        return dict(p=p, u_epi=ps.std(0), u_logit=zs.std(0), u_tot=_entropy(p))


def fit_ensemble(Xtr, ytr, Xva, yva, seed=0, n_members=5, **kw):
    ms = [fit_mlp(Xtr, ytr, Xva, yva, seed=seed * 100 + j, **kw) for j in range(n_members)]
    e = Ensemble(ms)
    e.val_logloss = _np_logloss(yva, e.predict(Xva)["p"])
    return e


# ------------------------------------------------------------------ last-layer Laplace
class LLLA:
    def __init__(self, base, prior_prec, Xtr, n_samples=300, seed=0):
        self.base, self.n_samples, self.seed = base, n_samples, seed
        G = base.embed(Xtr)
        p = sigmoid_np(G @ np.asarray(base.w))
        H = (G * (p * (1 - p))[:, None]).T @ G + prior_prec * np.eye(G.shape[1])
        self.cov = np.linalg.inv(H)
        self.cov = 0.5 * (self.cov + self.cov.T)

    def predict(self, X, **kw):
        G = self.base.embed(X)
        mu = G @ np.asarray(self.base.w)
        var = np.einsum("ij,jk,ik->i", G, self.cov, G)
        rng = np.random.default_rng(self.seed)
        f = mu[None, :] + np.sqrt(np.maximum(var, 1e-12))[None, :] * rng.standard_normal((self.n_samples, len(mu)))
        ps = sigmoid_np(f)
        p = ps.mean(0)
        return dict(p=p, u_epi=ps.std(0), u_logit=np.sqrt(np.maximum(var, 0)), u_tot=_entropy(p))


def fit_llla(base, Xtr, Xva, yva, prior_prec=10.0, seed=0):
    m = LLLA(base, prior_prec, Xtr, seed=seed)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m


# ------------------------------------------------------------------ SNGP-style
def _spec_norm(params, c):
    out = []
    for (W, b) in params:
        s = jnp.linalg.norm(W, 2)
        out.append((W * jnp.minimum(1.0, c / (s + 1e-8)), b))
    return out


class SNGPModel:
    def __init__(self, g, Wr, br, beta, b0, cov, scale):
        self.g, self.Wr, self.br, self.beta, self.b0, self.cov, self.scale = g, Wr, br, beta, b0, cov, scale

    def _phi(self, X):
        h = mlp_apply(self.g, jnp.asarray(X))
        D = self.Wr.shape[1]
        return np.asarray(jnp.sqrt(2.0 / D) * jnp.cos(h @ self.Wr * self.scale + self.br))

    def predict(self, X, **kw):
        phi = self._phi(X)
        logit = phi @ np.asarray(self.beta) + float(self.b0)
        var = np.einsum("ij,jk,ik->i", phi, self.cov, phi)
        p = sigmoid_np(logit / np.sqrt(1.0 + (math.pi / 8.0) * var))
        sd = np.sqrt(np.maximum(var, 0))
        rng = np.random.default_rng(12345)
        ps = sigmoid_np(logit[None, :] + sd[None, :] * rng.standard_normal((300, len(logit))))
        return dict(p=p, u_epi=ps.std(0), u_logit=sd, u_tot=_entropy(p))


def fit_sngp(Xtr, ytr, Xva, yva, seed=0, hidden=(64, 64), rep=32, n_rff=256, norm_c=2.0, scale=1.0,
             ridge=1.0, lr=1e-3, epochs=150, batch=512, patience=15):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, key = jax.random.split(key, 5)
    d = Xtr.shape[1]
    g = init_mlp(k1, [d, *hidden, rep])
    Wr = jax.random.normal(k2, (rep, n_rff)) / math.sqrt(rep)
    br = jax.random.uniform(k3, (n_rff,), minval=0, maxval=2 * math.pi)
    beta = jnp.zeros(n_rff); b0 = jnp.array(0.0)
    params = {"g": g, "beta": beta, "b0": b0}
    opt = optax.adam(lr); st = opt.init(params)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr, dtype=jnp.float32)

    def phi_fn(g, x):
        h = mlp_apply(g, x)
        return jnp.sqrt(2.0 / n_rff) * jnp.cos(h @ Wr * scale + br)

    @jax.jit
    def step(params, st, xb, yb):
        def loss(p):
            z = phi_fn(p["g"], xb) @ p["beta"] + p["b0"]
            return bce_logits(z, yb) + 0.5 * ridge * jnp.sum(p["beta"] ** 2) / Xtr.shape[0]
        l, gr = jax.value_and_grad(loss)(params)
        up, st = opt.update(gr, st, params)
        params = optax.apply_updates(params, up)
        params = {**params, "g": _spec_norm(params["g"], norm_c)}
        return params, st, l

    @jax.jit
    def logits_fn(params, X):
        return phi_fn(params["g"], X) @ params["beta"] + params["b0"]

    rng = np.random.default_rng(seed); n = len(Xtr)
    Xva_j = jnp.asarray(Xva)
    best, best_p, bad = np.inf, params, 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            params, st, _ = step(params, st, Xtr_j[idx], ytr_j[idx])
        v = _np_logloss(yva, sigmoid_np(np.asarray(logits_fn(params, Xva_j))))
        if v < best - 1e-5:
            best, best_p, bad = v, params, 0
        else:
            bad += 1
            if bad >= patience:
                break
    # Laplace precision on training data
    phi = np.asarray(phi_fn(best_p["g"], Xtr_j))
    z = phi @ np.asarray(best_p["beta"]) + float(best_p["b0"])
    p = sigmoid_np(z)
    P = (phi * (p * (1 - p))[:, None]).T @ phi + ridge * np.eye(n_rff)
    cov = np.linalg.inv(P)
    m = SNGPModel(best_p["g"], Wr, br, best_p["beta"], best_p["b0"], 0.5 * (cov + cov.T), scale)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m


# ------------------------------------------------------------------ VIDS
def _vids_split_loss(hp, gtr, ytr, gte, yte, eps, lam, rep, adaptive=True, prior_form="code"):
    """Loss for ONE synthetic environment (binary case), port of train_posterior()."""
    gmean = jnp.mean(gtr, axis=0, keepdims=True)
    inp = jnp.concatenate([gte, jnp.tile(gmean, (gte.shape[0], 1))], axis=1)
    out = mlp_apply(hp, inp)
    mu, raw = out[:, :rep], out[:, rep:]
    sigma = jax.nn.softplus(raw) + 1e-6
    theta = mu + sigma * eps
    e = 1e-8
    logq = -0.5 * jnp.sum(LOG2PI + 2 * jnp.log(sigma + e) + jnp.square(theta - mu) / (jnp.square(sigma) + e), axis=-1)

    theta_avg = jnp.mean(theta, axis=0)
    tr_logit = gtr @ theta_avg
    te_logit = jnp.sum(gte * theta, axis=-1)

    LOGE = math.log(1e-8)                      # same floor as the authors' log(p + 1e-8)

    def log_sig(z):                            # numerically stable log sigmoid(z)
        return jnp.maximum(-jax.nn.softplus(-z), LOGE)

    def log_1m_sig(z):                         # numerically stable log(1 - sigmoid(z))
        return jnp.maximum(-jax.nn.softplus(z), LOGE)

    def loglik(logit, y):
        return y * log_sig(logit) + (1 - y) * log_1m_sig(logit)

    def int_ll(logit):                         # 'integrated' log-likelihood of the prior energy
        if prior_form == "paper":              # Eq. (4) of the paper: sum over y in {0,1} with equal weight
            v = log_sig(logit) + log_1m_sig(logit)
        else:                                  # authors' code: weighted by the predicted probabilities (negative entropy)
            p = jnp.clip(jax.nn.sigmoid(logit), 0.0, 1.0)
            v = p * log_sig(logit) + (1 - p) * log_1m_sig(logit)
        return v - jnp.log(jnp.mean(jnp.exp(v)) + e)  # normalize_integrated_ll

    obs = jnp.mean(loglik(tr_logit, ytr)) + jnp.mean(loglik(te_logit, yte))
    prior_lp = jnp.mean(int_ll(tr_logit)) + (jnp.mean(int_ll(te_logit)) if adaptive else 0.0)
    kl = jnp.mean(logq) - prior_lp
    return -(obs - lam * kl)


class VIDSModel:
    def __init__(self, base, hp, G_train, rep, n_samples=300, seed=0):
        self.base, self.hp, self.rep, self.S, self.seed = base, hp, rep, n_samples, seed
        self.gmean = jnp.mean(jnp.asarray(G_train), axis=0, keepdims=True)

    def predict(self, X, **kw):
        G = jnp.asarray(self.base.embed(X))
        inp = jnp.concatenate([G, jnp.tile(self.gmean, (G.shape[0], 1))], axis=1)
        out = np.asarray(mlp_apply(self.hp, inp))
        mu, sigma = out[:, :self.rep], np.logaddexp(0.0, out[:, self.rep:]) + 1e-6   # numerically stable softplus
        rng = np.random.default_rng(self.seed)
        Gn = np.asarray(G)
        zs = []
        for _ in range(self.S):
            th = mu + sigma * rng.standard_normal(mu.shape)
            zs.append(np.sum(Gn * th, axis=1))
        zs = np.stack(zs)
        ps = sigmoid_np(zs)
        p = ps.mean(0)
        return dict(p=p, u_epi=ps.std(0), u_logit=zs.std(0), u_tot=_entropy(p))


def fit_vids(base, Xtr, ytr, Xva, yva, seed=0, m=200, n=1000, J=16, K=1500, lam=0.05, tau=0.01,
             lr=1e-3, widths=(128, 64, 32), init_from_pretrained=True, sigma_init=0.1, n_samples=300,
              adaptive=True, checkpoint="start", clip=5.0, keras_init=False, prior_form="code"):
    """checkpoint: 'start' = keep the best validation state among the starting state and periodic checkpoints;
    'train' = periodic checkpoints only; 'none' = final state after K steps (as in the authors' code)."""
    """base: a trained MLPModel supplying the frozen embedding g and (optionally) head init."""
    G = jnp.asarray(base.embed(Xtr)); y = jnp.asarray(ytr, dtype=jnp.float32)
    N, rep = G.shape
    key = jax.random.PRNGKey(10_000 + seed)
    if keras_init and not init_from_pretrained:
        # the authors' initialisation: Keras defaults for every layer of the posterior network (no special output bias)
        hp = init_glorot(key, [2 * rep, *widths, 2 * rep])
    else:
        hp = init_mlp(key, [2 * rep, *widths, 2 * rep], last_scale=0.1)
        # output-layer bias: [mu | raw-sigma]
        Wl, bl = hp[-1]
        mu0 = jnp.asarray(base.w) if init_from_pretrained else jnp.zeros(rep)
        raw0 = jnp.full((rep,), math.log(math.expm1(sigma_init)))
        if init_from_pretrained:
            # zero output weights: before training the posterior is exactly N(pretrained head, sigma_init^2 I) for every firm
            Wl = jnp.zeros_like(Wl)
        hp[-1] = (Wl, jnp.concatenate([mu0, raw0]))
    opt = (optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr)) if clip else optax.adam(lr)); st = opt.init(hp)
    from functools import partial
    batched = jax.vmap(partial(_vids_split_loss, adaptive=adaptive, prior_form=prior_form), in_axes=(None, 0, 0, 0, 0, 0, None, None))

    @jax.jit
    def step(hp, st, itr, ite, eps):
        def loss(hp_):
            Ls = batched(hp_, G[itr], y[itr], G[ite], y[ite], eps, lam, rep)
            return jnp.mean(Ls) + tau * jnp.var(Ls)
        l, gr = jax.value_and_grad(loss)(hp)
        up, st = opt.update(gr, st, hp)
        return optax.apply_updates(hp, up), st, l

    rng = np.random.default_rng(seed)
    best, best_hp, hist = np.inf, hp, []
    check = max(K // 15, 1)
    v0 = _np_logloss(yva, VIDSModel(base, hp, G, rep, n_samples=50, seed=seed).predict(Xva)["p"])
    if checkpoint == "start" and np.isfinite(v0):      # the starting state is a checkpoint candidate
        best, best_hp = v0, hp
    n_improve = 0
    for k in range(K):
        itr = rng.integers(0, N, (J, n)); ite = rng.integers(0, N, (J, m))
        eps = rng.standard_normal((J, m, rep)).astype(np.float32)
        hp, st, l = step(hp, st, itr, ite, eps)
        hist.append(float(l))
        if not np.isfinite(float(l)):
            break
        if checkpoint != "none" and ((k + 1) % check == 0 or k == K - 1):
            mdl = VIDSModel(base, hp, G, rep, n_samples=50, seed=seed)
            v = _np_logloss(yva, mdl.predict(Xva)["p"])
            if np.isfinite(v) and v < best:
                best, best_hp = v, hp; n_improve += 1
    if checkpoint == "none":
        best_hp = hp
        best = _np_logloss(yva, VIDSModel(base, hp, G, rep, n_samples=50, seed=seed).predict(Xva)["p"])
    mdl = VIDSModel(base, best_hp, G, rep, n_samples=n_samples, seed=seed)
    mdl.val_logloss = best; mdl.val_start = v0; mdl.n_improve = n_improve
    mdl.loss_hist = hist
    return mdl


# ------------------------------------------------------------------ SNGP with residual hidden layers (Liu et al. 2020)
class SNGPResModel(SNGPModel):
    def __init__(self, Win, blocks, Wr, br, beta, b0, cov, scale):
        self.Win, self.blocks, self.Wr, self.br, self.beta, self.b0, self.cov, self.scale = Win, blocks, Wr, br, beta, b0, cov, scale

    def _phi(self, X):
        h = _res_forward(self.Win, self.blocks, jnp.asarray(X))
        D = self.Wr.shape[1]
        return np.asarray(jnp.sqrt(2.0 / D) * jnp.cos(h @ self.Wr * self.scale + self.br))


def _res_forward(Win, blocks, x):
    W0, b0 = Win
    h = x @ W0 + b0
    for (W, b) in blocks:
        h = h + jax.nn.relu(h @ W + b)
    return h


def fit_sngp_res(Xtr, ytr, Xva, yva, seed=0, width=64, n_blocks=2, n_rff=256, norm_c=0.95, scale=1.0,
                 ridge=1.0, lr=1e-3, epochs=150, batch=512, patience=15):
    """Residual network whose hidden (residual) layers are spectrally normalised to norm <= norm_c, so the
    hidden map is approximately distance-preserving; random-Fourier-feature GP output layer with a Laplace
    posterior, mean-field logit adjustment for prediction (Liu et al. 2020)."""
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3, key = jax.random.split(key, 5)
    d = Xtr.shape[1]
    Win = (jax.random.normal(k0, (d, width)) * math.sqrt(1.0 / d), jnp.zeros(width))
    bk = jax.random.split(k1, n_blocks)
    blocks = [(jax.random.normal(bk[i], (width, width)) * math.sqrt(2.0 / width) * 0.5, jnp.zeros(width)) for i in range(n_blocks)]
    blocks = _spec_norm(blocks, norm_c)
    Wr = jax.random.normal(k2, (width, n_rff)) / math.sqrt(width)
    br = jax.random.uniform(k3, (n_rff,), minval=0, maxval=2 * math.pi)
    params = {"Win": Win, "blocks": blocks, "beta": jnp.zeros(n_rff), "b0": jnp.array(0.0)}
    opt = optax.adam(lr); st = opt.init(params)
    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr, dtype=jnp.float32)

    def phi_fn(p, x):
        h = _res_forward(p["Win"], p["blocks"], x)
        return jnp.sqrt(2.0 / n_rff) * jnp.cos(h @ Wr * scale + br)

    @jax.jit
    def step(params, st, xb, yb):
        def loss(p):
            z = phi_fn(p, xb) @ p["beta"] + p["b0"]
            return bce_logits(z, yb) + 0.5 * ridge * jnp.sum(p["beta"] ** 2) / Xtr.shape[0]
        l, gr = jax.value_and_grad(loss)(params)
        up, st = opt.update(gr, st, params)
        params = optax.apply_updates(params, up)
        params = {**params, "blocks": _spec_norm(params["blocks"], norm_c)}
        return params, st, l

    @jax.jit
    def logits_fn(params, X):
        return phi_fn(params, X) @ params["beta"] + params["b0"]

    rng = np.random.default_rng(seed); n = len(Xtr)
    Xva_j = jnp.asarray(Xva)
    best, best_p, bad = np.inf, params, 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            params, st, _ = step(params, st, Xtr_j[idx], ytr_j[idx])
        v = _np_logloss(yva, sigmoid_np(np.asarray(logits_fn(params, Xva_j))))
        if v < best - 1e-5:
            best, best_p, bad = v, params, 0
        else:
            bad += 1
            if bad >= patience:
                break
    phi = np.asarray(phi_fn(best_p, Xtr_j))
    z = phi @ np.asarray(best_p["beta"]) + float(best_p["b0"])
    pp = sigmoid_np(z)
    P = (phi * (pp * (1 - pp))[:, None]).T @ phi + ridge * np.eye(n_rff)
    cov = np.linalg.inv(P)
    m = SNGPResModel(best_p["Win"], best_p["blocks"], Wr, br, best_p["beta"], best_p["b0"], 0.5 * (cov + cov.T), scale)
    m.val_logloss = _np_logloss(yva, m.predict(Xva)["p"])
    return m
