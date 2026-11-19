"""Classifier score after CaloChallenge (arXiv:2410.21611): can a fixed classifier tell the fast arm from stock?

A logistic regression on quadratic features of per-cluster feature vectors, fixed seed, 50/50 train/test split of
equal-size stock and fast samples; the score is cls_score = clip(2 (1 - AUC), 0, 1) on the test half (1 = the
classifier cannot tell the arms apart, 0 = it separates them perfectly). NumPy only, so the stock Codabench image
(codalab/codalab-legacy:py312) runs it; it imports nothing from this repository.

Procedure, all deterministic for a given seed:
1. keep the clusters passing the selection (energy >= min_energy), draw min(n_stock, n_fast, max_per_class) of each
   arm without replacement (numpy default_rng(seed): stock first, then fast);
2. the first half of each arm trains, the second half tests;
3. standardise the features with the training mean and standard deviation, add every product x_i x_j (i <= j),
   standardise those terms the same way;
4. fit the logistic regression (intercept + one weight per term) by Newton's method with a small ridge (l2 per
   training sample on the weights, not on the intercept), at most 100 steps;
5. AUC = P(score_fast > score_stock) on the test half (Mann-Whitney, ties counted half).
"""
from __future__ import annotations

import numpy as np

DEFAULT_SEED = 20261118
DEFAULT_L2 = 1e-3
DEFAULT_MAX_PER_CLASS = 100_000
MIN_PER_CLASS = 50


def load_features(path, names) -> np.ndarray:
    """(n, len(names)) float64 matrix of the named arrays of a features.npz (no pickles)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [n for n in names if n not in z.files]
        if missing:
            raise ValueError(f"{path}: features {missing} missing")
        cols = [np.asarray(z[n], dtype=np.float64).ravel() for n in names]
    if len({c.size for c in cols}) != 1:
        raise ValueError(f"{path}: feature arrays differ in length")
    x = np.column_stack(cols) if cols else np.zeros((0, 0))
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: features must be finite")
    return x


def quadratic(z: np.ndarray) -> np.ndarray:
    """The linear terms followed by every product z_i z_j with i <= j."""
    i, j = np.triu_indices(z.shape[1])
    return np.hstack([z, z[:, i] * z[:, j]])


def _standardise(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(a - mu) / sd for a in (train, *others)]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = DEFAULT_L2, max_iter: int = 100) -> np.ndarray:
    """Weights (intercept first) of the ridge logistic regression, by Newton's method."""
    n, d = x.shape
    a = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2 * n)
    reg[0] = 0.0
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(a @ w, -40.0, 40.0)))
        g = a.T @ (p - y) + reg * w
        h = (a * (p * (1 - p))[:, None]).T @ a + np.diag(reg) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + P(pos == neg) / 2 (Mann-Whitney U / (n_pos n_neg)), from average ranks."""
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0           # 1-based average rank of each distinct value
    ranks = np.empty(s.size)
    ranks[order] = avg[inv]
    npos, nneg = pos.size, neg.size
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def classifier_score(stock: np.ndarray, fast: np.ndarray, seed: int = DEFAULT_SEED, l2: float = DEFAULT_L2,
                     max_per_class: int = DEFAULT_MAX_PER_CLASS) -> dict:
    """AUC and cls_score of the fixed classifier, fast (label 1) against stock (label 0)."""
    if stock.ndim != 2 or fast.ndim != 2 or stock.shape[1] != fast.shape[1] or stock.shape[1] == 0:
        raise ValueError("stock and fast need the same (n, d) feature layout")
    n = min(len(stock), len(fast), int(max_per_class))
    if n < MIN_PER_CLASS:
        raise ValueError(f"classifier needs >= {MIN_PER_CLASS} selected clusters per arm, has {n}")
    rng = np.random.default_rng(seed)
    s = stock[rng.permutation(len(stock))[:n]]
    f = fast[rng.permutation(len(fast))[:n]]
    h = n // 2
    tr, te_s, te_f = _standardise(np.vstack([s[:h], f[:h]]), s[h:], f[h:])
    q_tr, q_s, q_f = _standardise(quadratic(tr), quadratic(te_s), quadratic(te_f))
    y = np.r_[np.zeros(h), np.ones(h)]
    w = fit_logistic(q_tr, y, l2)
    score = lambda q: w[0] + q @ w[1:]  # noqa: E731
    a = auc(score(q_f), score(q_s))
    train_auc = auc(score(q_tr[h:]), score(q_tr[:h]))
    return {"auc": a, "cls_score": float(np.clip(2.0 * (1.0 - a), 0.0, 1.0)), "train_auc": train_auc,
            "n_per_class": n, "n_train": 2 * h, "n_test": 2 * (n - h), "n_features": stock.shape[1],
            "n_terms": q_tr.shape[1], "seed": int(seed)}
