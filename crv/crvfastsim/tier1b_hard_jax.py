"""Strict-v4 whole-muon donor selection with the reference rate and redraw rules."""
import numpy as np

from . import tier1b as T
from .tier1b_jax import _jax


def prepare(hard):
    _, j = _jax()
    out = {c: j.asarray(hard[c]) for c in ("start", "xid", "is_hard_xing")}
    for name, fields, keys in (("rate", ["nlib", "nhard"], ["pc", "lo", "hi"]),
                               ("cat", ["a", "cnt"], ["pc", "lo", "hi"]),
                               ("pn", ["rate_pn"], ["pc", "n"])):
        frame = hard[{"rate": "rate_key", "cat": "kcat", "pn": "rate_pn"}[name]].sort_values(keys, kind="stable")
        out[name + "_keys"] = j.asarray(frame[keys].to_numpy(), dtype=j.int64)
        for c in fields:
            out[c] = j.asarray(frame[c].to_numpy())
    return out


def lookup(keys, want):
    """Lexicographic lower bound over integer tuples, without lossy packed keys."""
    jax, j = _jax()
    if not len(keys):
        return j.zeros(len(want), j.int64), j.zeros(len(want), bool)
    lo, hi = j.zeros(len(want), j.int64), j.full(len(want), len(keys), j.int64)
    def body(_, state):
        a, b = state
        m = (a + b) // 2
        row = keys[j.minimum(m, len(keys) - 1)]
        less, equal = j.zeros(len(want), bool), j.ones(len(want), bool)
        for k in range(keys.shape[1]):
            less |= equal & (row[:, k] < want[:, k])
            equal &= row[:, k] == want[:, k]
        return j.where((a < b) & less, m + 1, a), j.where((a < b) & ~less, m, b)
    lo, _ = jax.lax.fori_loop(0, (len(keys) + 1).bit_length(), body, (lo, hi))
    idx = j.minimum(lo, len(keys) - 1)
    return idx, (lo < len(keys)) & j.all(keys[idx] == want, axis=1)


def _value(h, name, idx, found):
    _, j = _jax()
    return j.where(found, h[name][idx], 0) if len(h[name]) else j.zeros(len(idx))


def patterns(x, g, h):
    _, j = _jax()
    valid, mu = x["valid"], x["mu"]
    n, safe = len(mu), j.maximum(mu, 0)
    ii = j.arange(n, dtype=j.int64)
    order = j.lexsort((ii, x["t0"], mu, ~valid))
    ms, ok = safe[order], valid[order]
    first = j.full(n, n, j.int64).at[ms].min(j.where(ok, ii, n))
    rank = j.empty(n, j.int64).at[order].set(ii - first[ms])
    count = j.zeros(n, j.int64).at[safe].add(valid.astype(j.int64))
    head = order[j.minimum(first, n - 1)]
    p = j.sqrt(x["mom"][:, 0] ** 2 + x["mom"][:, 1] ** 2 + x["mom"][:, 2] ** 2)
    pc = j.searchsorted(j.asarray(T.HARD_PCLASS), p[head], side="left")
    sym = g["group"][x["bar"]] * 4 + j.minimum(g["rank"][x["bar"]], 3)
    powers = j.asarray(17 ** np.arange(12, dtype=np.int64))
    def half(offset):
        take = valid & (rank >= offset) & (rank < offset + 12)
        value = (sym + 1) * powers[j.clip(rank - offset, 0, 11)]
        return j.zeros(n, j.int64).at[safe].add(j.where(take, value, 0))
    want = j.stack((pc, half(0), half(12)), axis=1)
    ir, kr = lookup(h["rate_keys"], want)
    ic, kc = lookup(h["cat_keys"], want)
    ip, kp = lookup(h["pn_keys"], j.stack((pc, count), axis=1))
    nlib, nhard = _value(h, "nlib", ir, kr), _value(h, "nhard", ir, kr)
    rate = j.where(nlib >= T.HARD_MIN_KEY_MUONS, nhard / j.maximum(nlib, 1), _value(h, "rate_pn", ip, kp))
    rate = j.where((count > 0) & (count <= T.HARD_MAX_CROSSINGS), rate, 0.)
    cnt = _value(h, "cnt", ic, kc).astype(j.int64)
    return dict(rank=rank, count=count, rate=rate, have=(cnt > 0) & (count > 0),
                a=_value(h, "a", ic, kc).astype(j.int64), cnt=cnt,
                nmu=j.sum(count > 0), pattern=want)


def choose(h, x, pat, picks, u, v):
    _, j = _jax()
    mu = j.maximum(x["mu"], 0)
    hard = (u < pat["rate"]) & pat["have"]
    catalogue = pat["a"] + (v * pat["cnt"]).astype(j.int64)
    if len(h["start"]) and len(h["xid"]):
        row = h["start"][j.minimum(catalogue[mu], len(h["start"]) - 1)] + pat["rank"]
        donor = h["xid"][j.clip(row, 0, len(h["xid"]) - 1)]
        picks = j.where(x["valid"] & hard[mu], donor, picks)
    redo = x["valid"] & ~hard[mu] & pat["have"][mu] & h["is_hard_xing"][picks]
    return picks, redo, dict(fast_muons=pat["nmu"], hard=j.sum(hard), expected=j.sum(pat["rate"]),
                             no_key=j.sum(j.where(~pat["have"], pat["rate"], 0.)))


def redraw(h, order, picks, lo, hi, redo, u, active):
    _, j = _jax()
    idx = j.nonzero(redo, size=len(u), fill_value=0)[0]
    value = order[j.minimum(lo[idx] + (u * (hi[idx] - lo[idx])).astype(j.int64), len(order) - 1)]
    mask = j.arange(len(u)) < active
    picks = picks.at[j.where(mask, idx, len(picks))].set(value, mode="drop")
    redo = redo & h["is_hard_xing"][picks]
    return picks, redo
