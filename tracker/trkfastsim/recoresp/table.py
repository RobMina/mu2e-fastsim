"""Reco response tables of the conversion-electron track reconstruction (stock LoopHelix + CE selection), NumPy.

Built from a reference arm whose reconstruction ran (stock digitisation + LoopHelix on Geant4 tracker steps), per
generated event: the electron's true state at the tracker entry, the number of StrawGasStep rows it left (one row per
passage through a straw) and the first fitted track. The response is binned in the entry truth that drives it:

  * the number of step rows `nrow` (how much of the tracker the electron crossed) and
  * the true tan(dip) at the entry;

and holds per bin:

  * eff_fit: P(a fitted track, status > 0);
  * eff_q: P(fit quality and direction pass | fit): electron hypothesis, downstream at the tracker front (pz > 0,
    in-bounds front segment), fit consistency above the CE cut;
  * inverse CDFs (on a fixed probability grid, denser in both tails) of the residuals reco - truth of the momentum at
    the front, tan(dip), d0 and r_max, and of the fit-quality outputs (fit consistency, chi2/ndf, active hits), over
    the quality-passing tracks of the bin;
  * the inverse CDF of t0 at the front minus the entry time (drawn on its own stream);
  * a turn-on of the fit efficiency in the entry time (the digitisation window: no tracks below ~210 ns), relative
    to the plateau; the bin tables above are then built from plateau events (entry time >= T_PLATEAU) only;
  * the correlation of the seven residual and quality quantities as a Gaussian copula: the Cholesky factor of the correlation matrix of
    their normal scores (rank -> standard normal), so a draw keeps e.g. the tan(dip)-d0-r_max correlations that the
    fiducial cuts see;
  * optionally (`build(..., pools=True)`, opt-in): the pool of each bin itself, as indices into the quality tracks'
    residual vectors. A table with pools draws the seven quantities jointly as one reference track's vector (an
    empirical joint draw) instead of copula + inverse CDFs. It keeps the tail dependence the Gaussian copula does
    not: a large momentum outlier of the stock fit usually comes with an off-cut d0 / r_max / tan(dip) and fails the
    fiducial cuts.
  * optionally (`build(..., pools="all")`, opt-in): pools of every fitted track instead of the quality-passing
    ones, with `pool_dir` (electron hypothesis, downstream, a front segment). The quality selection (direction and
    the fit-consistency cut) is then applied at run time to the drawn track, so the whole CE selection, including
    the fit-consistency cut, can change without rebuilding the table;
  * optionally (`build(..., config=manifest)`): the stock-input manifest (`recoresp.config`) and its hash, so a
    table can be checked against the configuration it is used for.

A fast track is then: fit with eff_fit, quality with eff_q, correlated uniforms from the copula, residuals by inverse
CDF of those uniforms, added to its own entry
truth, and the CE fiducial cuts applied to the smeared tan(dip), d0 and r_max (device.py). Bins with fewer than
`min_eff` reference events (efficiencies) or `min_count` quality-passing tracks (distributions) widen to the +-1,
then +-2 neighbouring tan(dip) bins, then the whole nrow row, then everything.

Truth helix at the entry: uniform field bz along the tracker axis; circle radius R = pt / (K |bz|), centre on the side
the charge bends to; r_centre from the tracker axis; d0 = |r_centre - R|, r_max = r_centre + R (the EventNtuple
LoopHelix d0, which is unsigned, and maxr).
"""
from pathlib import Path

import numpy as np

KB = 0.299792458
NROW_EDGES = (0, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 60, 80, 120, 1_000_000)
TANDIP_EDGES = (-1e9, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1e9)
NQ = 129                                     # inverse-CDF nodes
RES = ("dp", "dtandip", "dd0", "drmax")      # residuals reco - truth
QUAL = ("fitcon", "chi2ndf", "nactive")      # fit-quality outputs
EXTRA = ("dt0",)                             # drawn on their own (outside the copula): t0 at the front - entry time
TIME_EDGES = (0.0, 150.0, 175.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 275.0, 300.0, 1e9)   # entry time [ns]
T_PLATEAU = 300.0       # entry times from here on are past the digitisation-window turn-on
CE_CUTS = dict(tandip=(0.5774, 1.0), d0=(-80.0, 105.0), rmax=(450.0, 680.0), fitcon=1e-3)
POOL_MAX = 20000        # pools=True: largest stored pool per bin (merged bins keep a subsample)
POOL_SEED = 20261109


def qnodes(nq=NQ):
    """Probabilities of the inverse-CDF nodes: 0.5 (1 - cos(pi k / (nq - 1))), denser in both tails."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(nq) / (nq - 1)))


def entry_truth(x, p, axis_xy, bz, q=-1.0):
    """(|p|, tan(dip), d0, r_max) of the helix through the entry states x [n, 3] (Mu2e frame, mm), p [n, 3] (MeV/c)."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    pt = np.hypot(p[:, 0], p[:, 1])
    R = pt / (KB * abs(bz))
    s = np.sign(q * bz)
    ptn = np.maximum(pt, 1e-12)
    cx = x[:, 0] - axis_xy[0] + s * R * p[:, 1] / ptn
    cy = x[:, 1] - axis_xy[1] - s * R * p[:, 0] / ptn
    rc = np.hypot(cx, cy)
    return np.linalg.norm(p, axis=1), p[:, 2] / ptn, np.abs(rc - R), rc + R


def _icdf(v, nq=NQ):
    """Inverse CDF of the sample v at qnodes (linear between the order statistics at (i - 0.5) / m, clamped)."""
    v = np.sort(np.asarray(v, float))
    m = len(v)
    pp = (np.arange(m) + 0.5) / m
    return np.interp(qnodes(nq), pp, v)


def _normal_scores(v):
    from scipy.special import ndtri
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return ndtri((r + 0.5) / len(v))


def _copula(rows):
    """Cholesky factor of the correlation of the normal scores of the columns of rows [m, K] (ridge 1e-6)."""
    K = rows.shape[1]
    if len(rows) < 3:
        return np.eye(K)
    z = np.stack([_normal_scores(rows[:, k]) for k in range(K)], 1)
    c = np.corrcoef(z, rowvar=False)
    c = np.where(np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    return np.linalg.cholesky(c + 1e-6 * np.eye(K))


def build(df, nrow_edges=NROW_EDGES, tandip_edges=TANDIP_EDGES, min_count=40, min_eff=10, axis_xy=(-3904.0, 0.0),
          bz=1.0, pools=False, pool_max=POOL_MAX, reference="", config=None):
    """Response tables from the reference events df (one row per generated event): has_entry, nrow, p_true, tandip_t,
    d0_t, rmax_t (entry truth), fit (bool), and for fitted tracks p_reco, pz, tandip, d0, rmax, pdg, fitcon, chi2ndf,
    nactive. Returns a dict of NumPy arrays (save with np.savez). pools (opt-in): also store each bin's pool for the
    empirical joint draw (pool_res [M, 7] in the order RES + QUAL, pool_idx, pool_start/pool_count [ne, nt]); a pool
    wider than pool_max (a merged bin) keeps a fixed-seed subsample of pool_max tracks. reference: a provenance text
    (the reference arm and events) stored as `reference` when given. pools="all": pools of every fitted track,
    with `pool_dir` [M] (the direction part of the quality selection), for the run-time CE selection. config: a
    recoresp.config manifest, stored as `config` (canonical JSON) and `config_hash`."""
    keep_pools = bool(pools)       # (the helper below is also called pools)
    all_tracks = isinstance(pools, str) and pools == "all"
    if isinstance(pools, str) and not all_tracks:
        raise ValueError(f"pools must be True, False or 'all', not {pools!r}")
    ne, nt = len(nrow_edges) - 1, len(tandip_edges) - 1
    d = df[df.has_entry.astype(bool)]
    teff = None
    if "t_t" in d:      # fit-efficiency turn-on in entry time (digitisation window), relative to the plateau
        tt = d.t_t.to_numpy(float)
        fit_all = d.fit.to_numpy(bool)
        plat = fit_all[tt >= T_PLATEAU].mean()
        it = np.clip(np.digitize(tt, TIME_EDGES) - 1, 0, len(TIME_EDGES) - 2)
        teff = np.array([min(fit_all[it == k].mean() / plat, 1.0) if (it == k).any() else 1.0
                         for k in range(len(TIME_EDGES) - 1)])
        teff[-1] = 1.0
        d = d[tt >= T_PLATEAU]                       # the bin tables: past the turn-on
    bn = np.clip(np.digitize(d.nrow.to_numpy(float), nrow_edges) - 1, 0, ne - 1)
    bt = np.clip(np.digitize(d.tandip_t.to_numpy(float), tandip_edges) - 1, 0, nt - 1)
    fit = d.fit.to_numpy(bool)
    direc = fit & np.isfinite(d.p_reco.to_numpy(float)) & (d.pdg.to_numpy(float) == 11) & (d.pz.to_numpy(float) > 0)
    qual = direc & (d.fitcon.to_numpy(float) > CE_CUTS["fitcon"])
    res = dict(dp=d.p_reco - d.p_true, dtandip=d.tandip - d.tandip_t, dd0=d.d0 - d.d0_t, drmax=d.rmax - d.rmax_t,
               fitcon=d.fitcon, chi2ndf=d.chi2ndf, nactive=d.nactive)
    if "t0" in d and "t_t" in d:
        res["dt0"] = d.t0 - d.t_t
    res = {k: np.asarray(v, float) for k, v in res.items()}
    extra = [k for k in EXTRA if k in res]
    eff_fit = np.zeros((ne, nt))
    eff_q = np.zeros((ne, nt))
    icdf = {k: np.zeros((ne, nt, NQ)) for k in RES + QUAL + tuple(extra)}
    chol = np.zeros((ne, nt, len(RES + QUAL), len(RES + QUAL)))
    level = np.zeros((ne, nt), np.int8)        # 0 own bin, 1/2 +-1/+-2 tan(dip) bins, 3 nrow row, 4 all
    count = np.zeros((ne, nt), np.int64)
    if keep_pools:
        prow = fit if all_tracks else qual         # the tracks a pool holds
        qrows = np.flatnonzero(prow)
        pos = np.full(len(d), -1, np.int64)
        pos[qrows] = np.arange(len(qrows))
        pool_res = np.stack([res[k][qrows] for k in RES + QUAL], 1)
        pool_dir = direc[qrows]                      # all-track pools: the direction part of the quality cut
        pool_idx, pool_start, pool_count = [], np.zeros((ne, nt), np.int64), np.zeros((ne, nt), np.int64)
        prng = np.random.default_rng(POOL_SEED)
    def pools(i, j):
        """Event selections of growing width: the bin, then +-1, +-2 neighbouring tan(dip) bins, the nrow row, all."""
        sel = [(bn == i) & (np.abs(bt - j) <= w) for w in (0, 1, 2)]
        return sel + [bn == i, np.ones(len(d), bool)]

    for i in range(ne):
        for j in range(nt):
            sel_e = pools(i, j)
            count[i, j] = int(sel_e[0].sum())
            lv = next(k for k, s_ in enumerate(sel_e) if s_.sum() >= min_eff or k == len(sel_e) - 1)
            s = sel_e[lv]
            eff_fit[i, j] = fit[s].mean() if s.any() else 0.0
            lq0 = next(k for k, s_ in enumerate(sel_e) if (s_ & fit).sum() >= min_eff or k == len(sel_e) - 1)
            eff_q[i, j] = qual[sel_e[lq0]].sum() / max(fit[sel_e[lq0]].sum(), 1)
            sel_q = [sq & qual for sq in sel_e]
            lq = next(k for k, s_ in enumerate(sel_q) if s_.sum() >= min_count or k == len(sel_q) - 1)
            level[i, j] = max(lv, lq)
            for k in RES + QUAL + tuple(extra):
                icdf[k][i, j] = _icdf(res[k][sel_q[lq]])
            chol[i, j] = _copula(np.stack([res[k][sel_q[lq]] for k in RES + QUAL], 1))
            if keep_pools:
                if all_tracks:
                    sel_p = [sq & fit for sq in sel_e]
                    lp = next(k for k, s_ in enumerate(sel_p) if s_.sum() >= min_count or k == len(sel_p) - 1)
                    ix = pos[np.flatnonzero(sel_p[lp])]
                else:
                    ix = pos[np.flatnonzero(sel_q[lq])]
                if len(ix) > pool_max:
                    ix = np.sort(prng.choice(ix, pool_max, replace=False))
                pool_start[i, j] = sum(len(a) for a in pool_idx)
                pool_count[i, j] = len(ix)
                pool_idx.append(ix)
    out = dict(nrow_edges=np.asarray(nrow_edges, float), tandip_edges=np.asarray(tandip_edges, float),
               qnodes=qnodes(), eff_fit=eff_fit, eff_q=eff_q, level=level, count=count, chol=chol,
               axis_xy=np.asarray(axis_xy, float), bz=np.float64(bz),
               cuts=np.array([*CE_CUTS["tandip"], *CE_CUTS["d0"], *CE_CUTS["rmax"], CE_CUTS["fitcon"]]))
    for k in icdf:
        out[f"icdf_{k}"] = icdf[k]
    if teff is not None:
        out["time_edges"], out["time_eff"] = np.asarray(TIME_EDGES, float), teff
    if reference:
        out["reference"] = np.array(str(reference))
    if config is not None:
        from . import config as RC
        out["config"], out["config_hash"] = np.array(RC.canonical(config)), np.array(RC.digest(config))
    if keep_pools:
        out["pool_res"] = pool_res
        out["pool_idx"] = np.concatenate(pool_idx).astype(np.int32) if pool_idx else np.zeros(0, np.int32)
        out["pool_start"], out["pool_count"] = pool_start, pool_count
        if all_tracks:
            out["pool_dir"] = pool_dir
    return out


def save(path, tab):
    np.savez_compressed(Path(path), **tab)


DEFAULT = Path(__file__).with_name("data") / "ce_loophelix_run1a.npz"
POOLS = Path(__file__).with_name("data") / "ce_loophelix_run1a_pools.npz"   # the same tables plus the pools (opt-in)
# opt-in: the response of the full Geant4 events (every StrawGasStep: delta rays, other secondaries, the calorimeter
# albedo), on the same development events and bins as DEFAULT, with pools
FULLDTS = Path(__file__).with_name("data") / "ce_loophelix_run1a_fulldts_pools.npz"
NAMED = {"default": DEFAULT, "pools": POOLS, "fulldts": FULLDTS}   # the shipped tables by name


# Tables of other stock configurations are not in git: a registry (data/registry.json) maps a stock-input hash
# (recoresp.config) to the table's file name, sha256 and download URL (release assets resp-<hash>), and a local
# cache holds the downloaded tables.
REGISTRY = Path(__file__).with_name("data") / "registry.json"
CACHE_ENV = "TRKFASTSIM_RESP_CACHE"


def cache_dir():
    """Local cache of downloaded tables: $TRKFASTSIM_RESP_CACHE, else ~/.cache/trkfastsim/recoresp."""
    import os
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "trkfastsim" / "recoresp")


def registry(path=REGISTRY):
    import json
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(h, reg=None):
    """Path of the registered table of stock-input hash h in the cache, downloaded from its URL if missing; the
    sha256 is checked either way."""
    import urllib.request
    reg = registry() if reg is None else reg
    if h not in reg:
        raise KeyError(f"stock-input hash {h} is not in the response table registry")
    e = reg[h]
    p = cache_dir() / e["file"]
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(e["url"]) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        tmp.replace(p)
    if _sha256(p) != e["sha256"]:
        raise ValueError(f"{p}: sha256 differs from the registry entry of {h}")
    return p


def resolve(response, reg=None):
    """Path of a response table: a shipped name (NAMED), an existing file, or a registered stock-input hash
    (fetched into the cache)."""
    from . import config as RC
    if isinstance(response, str) and response in NAMED:
        return NAMED[response]
    if Path(response).exists():
        return Path(response)
    s = str(response)
    if len(s) == RC.HASH_LEN and all(c in "0123456789abcdef" for c in s):
        return fetch(s, reg)
    raise FileNotFoundError(f"response table {response!r}: not a shipped name, a file or a registered hash")


def load(path=DEFAULT):
    with np.load(Path(path)) as f:
        return {k: f[k] for k in f.files}


def check_config(tab, c):
    """Raise unless tab was built with the stock inputs c: a recoresp.config manifest (dict, JSON text or path) or its
    hash. Tables without a stored hash (built without config) never match."""
    from . import config as RC
    want, have = RC.expected_hash(c), str(tab.get("config_hash", ""))
    if have != want:
        raise ValueError(f"response table built with stock-input hash {have or '(none)'}, requested {want}")


def interpolate(lo, hi, w, seed=POOL_SEED):
    """The response at a fraction w in [0, 1] between two tables built on the same events and bins with a continuous
    stock knob at two grid values (e.g. a threshold scale): efficiencies, inverse CDFs and copula factors linear in w;
    pools the mixture (1 - w) lo + w hi, as a fixed-seed resample of both pools of each bin (all-track pools stay
    all-track). The result has no config hash (it is not a stock configuration)."""
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError("w must be in [0, 1]")
    for k in ("nrow_edges", "tandip_edges", "qnodes", "count", "cuts", "axis_xy"):
        if not np.array_equal(lo[k], hi[k]):
            raise ValueError(f"tables differ in {k}: not built on the same events and bins")
    if ("pool_res" in lo) != ("pool_res" in hi) or ("pool_dir" in lo) != ("pool_dir" in hi):
        raise ValueError("both tables need the same pool kind")
    out = {k: v for k, v in lo.items() if k not in ("config", "config_hash", "reference")}
    for k in ["eff_fit", "eff_q", "chol", "time_eff"] + [k for k in lo if k.startswith("icdf_")]:
        if k in lo:
            out[k] = (1.0 - w) * np.asarray(lo[k], float) + w * np.asarray(hi[k], float)
    out["level"] = np.maximum(lo["level"], hi["level"])
    out["reference"] = np.array(f"interpolated, w = {w:g} between [{lo.get('config_hash', '')}] "
                                f"{lo.get('reference', '')} and [{hi.get('config_hash', '')}] {hi.get('reference', '')}")
    if "pool_res" not in lo:
        return out
    rng = np.random.default_rng(seed)
    na = len(lo["pool_res"])
    out["pool_res"] = np.concatenate([lo["pool_res"], hi["pool_res"]])
    if "pool_dir" in lo:
        out["pool_dir"] = np.concatenate([lo["pool_dir"], hi["pool_dir"]])
    idx, start, count = [], np.zeros_like(lo["pool_start"]), np.zeros_like(lo["pool_count"])
    for i in range(lo["pool_start"].shape[0]):
        for j in range(lo["pool_start"].shape[1]):
            a = lo["pool_idx"][lo["pool_start"][i, j]:lo["pool_start"][i, j] + lo["pool_count"][i, j]]
            b = hi["pool_idx"][hi["pool_start"][i, j]:hi["pool_start"][i, j] + hi["pool_count"][i, j]] + na
            n = len(a) + len(b)
            kb = int(round(w * n)) if len(b) else 0
            ka = n - kb if len(a) else 0
            kb = n - ka
            pick = [x[rng.permutation(len(x))[np.arange(k) % len(x)]] for x, k in ((a, ka), (b, kb)) if k]
            ix = np.sort(np.concatenate(pick)) if pick else np.zeros(0, np.int64)
            start[i, j], count[i, j] = sum(len(x) for x in idx), len(ix)
            idx.append(ix)
    out["pool_idx"] = np.concatenate(idx).astype(np.int32)
    out["pool_start"], out["pool_count"] = start, count
    return out


def at_knob(grid, value):
    """The response at a knob value from tables on a grid {knob value: table}: the two neighbouring grid tables,
    interpolated. Valid range: the grid's span; outside it, ValueError."""
    xs = sorted(float(x) for x in grid)
    v = float(value)
    if not xs[0] <= v <= xs[-1]:
        raise ValueError(f"knob value {v} outside the grid [{xs[0]}, {xs[-1]}]")
    tab = {float(k): t for k, t in grid.items()}
    if v in tab:
        return tab[v]
    k = int(np.searchsorted(xs, v))
    lo, hi = xs[k - 1], xs[k]
    return interpolate(tab[lo], tab[hi], (v - lo) / (hi - lo))


def with_cuts(tab, tandip=None, d0=None, rmax=None, fitcon=None):
    """A copy of tab with other CE selection cuts, applied at run time: tandip, d0, rmax as (low, high), fitcon as the
    fit-consistency threshold. The fiducial cuts act on the drawn values with any table; the fit-consistency cut is
    part of eff_q and of the pools unless the table has all-track pools (pools="all"), so changing it needs one.
    Valid range of fitcon: [0, 1); a tighter cut leaves fewer pool tracks per bin (see `pool_count`)."""
    c = np.array(tab["cuts"], float)
    for k, (i, v) in dict(tandip=(0, tandip), d0=(2, d0), rmax=(4, rmax)).items():
        if v is not None:
            lo, hi = map(float, v)
            if not lo < hi:
                raise ValueError(f"{k} cut must be (low, high) with low < high")
            c[i], c[i + 1] = lo, hi
    if fitcon is not None and float(fitcon) != c[6]:
        if "pool_dir" not in tab:
            raise ValueError("the fit-consistency cut is built into this table (eff_q); use a table with all-track "
                             "pools (table.build(..., pools='all')) to change it")
        if not 0.0 <= float(fitcon) < 1.0:
            raise ValueError("fitcon cut must be in [0, 1)")
        c[6] = float(fitcon)
    return {**tab, "cuts": c}


def _node(u, nq):
    """Fractional node index of the probability u on the qnodes grid."""
    return (nq - 1) * np.arccos(np.clip(1.0 - 2.0 * u, -1.0, 1.0)) / np.pi


def _sample(icdf, bn, bt, u):
    nq = icdf.shape[-1]
    f = np.clip(_node(u, nq), 0.0, nq - 1.000001)
    i = f.astype(np.int64)
    w = f - i
    return icdf[bn, bt, i] * (1 - w) + icdf[bn, bt, i + 1] * w


def apply_numpy(tab, rng, has_entry, nrow, x, p, q=-1.0, t=None):
    """The device response (device.apply) in NumPy, for checks: per event fit, ce and the track observables."""
    ptrue, td, d0, rmax = entry_truth(x, p, tab["axis_xy"], float(tab["bz"]), q)
    ne, nt = tab["eff_fit"].shape
    bn = np.clip(np.digitize(nrow, tab["nrow_edges"]) - 1, 0, ne - 1)
    bt = np.clip(np.digitize(td, tab["tandip_edges"]) - 1, 0, nt - 1)
    from scipy.special import ndtr
    u = rng.random((2, len(nrow)))
    z = np.einsum("nij,nj->in", tab["chol"][bn, bt], rng.standard_normal((len(nrow), len(RES + QUAL))))
    eff = tab["eff_fit"][bn, bt]
    if t is not None and "time_eff" in tab:
        eff = eff * tab["time_eff"][np.clip(np.digitize(t, tab["time_edges"]) - 1, 0, len(tab["time_eff"]) - 1)]
    fit = np.asarray(has_entry, bool) & (u[0] < eff)
    qual = fit & (u[1] < tab["eff_q"][bn, bt])
    if "pool_res" in tab:      # empirical joint draw: one reference track's vector of the bin pool (own stream)
        up = np.random.Generator(rng.bit_generator.jumped()).random(len(nrow))   # does not advance rng (t0 unchanged)
        cnt = tab["pool_count"][bn, bt]
        j = tab["pool_start"][bn, bt] + np.minimum((up * cnt).astype(np.int64), np.maximum(cnt - 1, 0))
        pr = tab["pool_idx"][np.minimum(j, len(tab["pool_idx"]) - 1)]
        row = tab["pool_res"][pr]
        draw = {k: row[:, i] for i, k in enumerate(RES + QUAL)}
        if "pool_dir" in tab:  # all-track pools: the quality selection on the drawn track, at run time
            qual = fit & tab["pool_dir"][pr] & (draw["fitcon"] > tab["cuts"][6])
    else:
        draw = {k: _sample(tab[f"icdf_{k}"], bn, bt, ndtr(z[i])) for i, k in enumerate(RES + QUAL)}
    c = tab["cuts"]
    obs = dict(fit=fit, qual=qual, p_true=ptrue, p_reco=ptrue + draw["dp"], tandip=td + draw["dtandip"],
               d0=d0 + draw["dd0"], rmax=rmax + draw["drmax"], fitcon=draw["fitcon"], chi2ndf=draw["chi2ndf"],
               nactive=draw["nactive"])
    obs["ce"] = qual & (obs["tandip"] >= c[0]) & (obs["tandip"] <= c[1]) & (obs["d0"] >= c[2]) & \
        (obs["d0"] <= c[3]) & (obs["rmax"] >= c[4]) & (obs["rmax"] <= c[5])
    if t is not None and "icdf_dt0" in tab:     # own stream, after the others: their draws are unchanged
        obs["t0"] = np.asarray(t, float) + _sample(tab["icdf_dt0"], bn, bt, rng.random(len(nrow)))
    for k in obs:
        if k not in ("fit", "qual", "ce", "p_true"):
            obs[k] = np.where(qual, obs[k], np.nan)
    return obs


def reference_events(gen_paths, truth_paths, steps_paths, nts_dir, axis_xy=(-3904.0, 0.0), bz=1.0):
    """Per generated event of the reference arm: entry truth (dumper tree `entries` of truth_paths), step rows
    (`sgsteps` of steps_paths) and the first fitted LoopHelix track of the EventNtuples under nts_dir
    (validation.cemom.read_arm). Denominator: the events of the `gen` trees of gen_paths."""
    import pandas as pd
    from mu2efastsim import tables
    from trkfastsim.validation import cemom
    KEY = tables.KEY

    def keyed(t):
        for k in KEY:
            t[k] = t[k].astype(np.uint64)
        return t
    den = keyed(tables.read_many(gen_paths, "gen", KEY))[KEY].drop_duplicates()
    tr = keyed(tables.read_many(truth_paths, "entries", KEY + ["x", "y", "z", "px", "py", "pz", "t"]))
    tr = tr.drop_duplicates(KEY)
    pt, td, d0, rm = entry_truth(tr[["x", "y", "z"]].to_numpy(), tr[["px", "py", "pz"]].to_numpy(), axis_xy, bz)
    tr = tr.rename(columns={c: f"e_{c}" for c in ("x", "y", "z", "px", "py", "pz")})
    tr = tr.rename(columns={"t": "t_t"}).assign(has_entry=True, p_true=pt, tandip_t=td, d0_t=d0, rmax_t=rm)
    st = keyed(tables.read_many(steps_paths, "sgsteps", KEY))
    nrow = st.groupby(KEY).size().rename("nrow").reset_index()
    arm, _ = cemom.read_arm(nts_dir)
    d = den.merge(tr, on=KEY, how="left").merge(nrow, on=KEY, how="left").merge(arm, on=KEY, how="left")
    d["has_entry"] = d.has_entry.fillna(False).astype(bool)
    d["nrow"] = d.nrow.fillna(0).astype(np.int64)
    d["fit"] = d.fit.fillna(False).astype(bool)
    d["ce"] = d.ce.fillna(False).astype(bool)
    return d


def main(argv=None):
    import argparse
    import glob
    import json
    from . import config as RC
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.recoresp.table", description="build CE reco response tables")
    ap.add_argument("--steps-dirs", nargs="+", required=True,
                    help="per-file dirs with states.root (gen), g4entries.root (truth) and g4prim_field.root (steps)")
    ap.add_argument("--nts", required=True, help="EventNtuple dir of the reference arm (nts.*.root below it)")
    ap.add_argument("--out", required=True, help="tables .npz")
    ap.add_argument("--events-out", help="optional per-event reference table (.csv.gz)")
    ap.add_argument("--bz", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=40)
    ap.add_argument("--reference", default="", help="provenance text stored in the table (reference arm and events)")
    ap.add_argument("--pools", nargs="?", const=True, default=False, choices=(True, "all"),
                    help="also store each bin's pool for the empirical joint draw (opt-in; default copula + inverse CDFs); "
                         "'--pools all': pools of every fitted track, for the run-time CE selection")
    ap.add_argument("--config", help="recoresp.config manifest of the stock inputs (stored with its hash)")
    a = ap.parse_args(argv)
    dirs = sorted(sum((glob.glob(d) for d in a.steps_dirs), []))
    d = reference_events([f"{x}/states.root" for x in dirs], [f"{x}/g4entries.root" for x in dirs],
                         [f"{x}/g4prim_field.root" for x in dirs], a.nts, bz=a.bz)
    tab = build(d, min_count=a.min_count, bz=a.bz, pools=a.pools, reference=a.reference,
                config=None if a.config is None else RC.load(a.config))
    save(a.out, tab)
    if a.events_out:
        d.to_csv(a.events_out, index=False)
    print(json.dumps(dict(events=len(d), with_entry=int(d.has_entry.sum()), fit=int(d.fit.sum()), ce=int(d.ce.sum()),
                          bins=list(tab["eff_fit"].shape), merged_bins=int((tab["level"] > 0).sum())), indent=1))


if __name__ == "__main__":
    main()
