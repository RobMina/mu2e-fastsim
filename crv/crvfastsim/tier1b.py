"""Tier 1b: G4-built library of the CRV deposits tier 1a does not make (docs/plan.md section 6).

Two parts, both resampled from G4 CrvStep tables (CrvStepTableDumper) of training shards:

1. Muon descendants (deltas above the production cut, decay electrons, ...). Each descendant step is attached to one
   primary-muon bar crossing of its event: the crossing in the same bar, else the crossing whose entry point is closest
   to the step (a library-building convention; the arms are never paired this way). A library crossing stores its
   descendants as offsets from its entry point. Sampling: each fast crossing draws a library crossing from the same
   (sector group, momentum, path-length) bin, with momentum and path-length edges at quantiles per sector group
   (P_QUANTILES, L_QUANTILES; the path-length bins are finer in the tail); the offsets are rotated from the G4 muon direction to the fast one and
   added to the fast entry point and time; same-bar steps that fall outside are clamped into the fast crossing's bar.
   Steps in another bar keep the bar structure instead: the library stores the centre offset between their bar and the
   crossing's bar and their position inside their bar; the fast step goes to the bar at that centre offset from the fast
   crossing's bar (same length axis), at the stored position, moved along the bar by the shift of the muon entry along
   the length. Those with no bar there (edge of the CRV, module gap) are dropped (counted).
2. Non-muon primaries (e, gamma, n, p, pi from CRY). A library shower is every CrvStep whose primary is that particle,
   with absolute positions and times relative to the primary's generation time. Sampling: each non-muon CRY primary of
   the event (gen table) reaches the CRV with the library probability of its (pdg, kinetic-energy) bin, and then gets a
   random shower of that bin. Known limit: shower bar indices are those of the library event, so a geometry move does
   not move them.

Descendant and shower particles are flattened to depth 1 (parent = the primary). Vectorised; no per-event loops.
"""
import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from . import tables
from .analyses.census import SECTOR_GROUP, classify

KEY = tables.KEY
GROUPS = ("EX", "T1", "T2", "M")
MASS = {11: 0.511, 13: 105.658, 22: 0.0, 211: 139.570, 321: 493.677, 2212: 938.272, 2112: 939.565}
KE_EDGES = np.geomspace(1.0, 1e6, 19)
P_QUANTILES = (0.25, 0.5, 0.75)
L_QUANTILES = (1 / 3, 2 / 3, 0.9, 0.97)  # finer in the long-path tail: long crossings carry more muon and delta energy
N_PBINS, N_LBINS = len(P_QUANTILES) + 1, len(L_QUANTILES) + 1
SHOWER_KEY_BASE = 10_000_000
STEP_COLS = ["x0", "y0", "z0", "x1", "y1", "z1"]
LIB_TREES = ("libbins", "libxing", "libdesc", "libshower", "libshowersteps", "libgen")


def _group_index(geom, bar):
    g = SECTOR_GROUP[geom.sector[np.asarray(bar, np.int64)]]
    return np.select([g == x for x in GROUPS], range(len(GROUPS)), -1)


def _bin(libbins, group, p, length):
    """Library bin of each crossing: (group, momentum bin, path-length bin) with the per-group edges of libbins."""
    out = np.zeros(len(group), np.int64)
    for gi in range(len(GROUPS)):
        m = group == gi
        pe = libbins.edge[(libbins.kind == 0) & (libbins.group == gi)].to_numpy()
        le = libbins.edge[(libbins.kind == 1) & (libbins.group == gi)].to_numpy()
        out[m] = (gi * N_PBINS + np.searchsorted(pe, p[m])) * N_LBINS + np.searchsorted(le, length[m])
    return out


def _ke_bin(ke):
    return np.clip(np.searchsorted(KE_EDGES, ke, "right") - 1, 0, len(KE_EDGES) - 2)


def _evkey(df):
    return (df.run.to_numpy().astype(np.int64) << 40) | (df.subrun.to_numpy().astype(np.int64) << 20) | \
        df.event.to_numpy().astype(np.int64)


def bar_containing(pos, geom, chunk=20000):
    """Bar index whose box contains each point (-1 if none)."""
    out = np.full(len(pos), -1, np.int64)
    for i in range(0, len(pos), chunk):
        inside = (np.abs(pos[i:i + chunk, None, :] - geom.center[None]) <= geom.halflen[None]).all(axis=2)
        hit = inside.any(axis=1)
        out[i:i + chunk][hit] = inside[hit].argmax(axis=1)
    return out


def rotate(vec, u_from, u_to):
    """Rotate each row of vec by the rotation taking unit vector u_from to u_to (Rodrigues; rows broadcast)."""
    v = np.cross(u_from, u_to)
    c = np.sum(u_from * u_to, axis=1, keepdims=True)
    k = 1.0 / np.maximum(1.0 + c, 1e-9)
    return vec * c + np.cross(v, vec) + v * np.sum(v * vec, axis=1, keepdims=True) * k


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------------------------------------- build
def _crossings(mu):
    """Primary-muon crossings with their first step's entry point, direction and |p|; summed edep and path."""
    mu = mu.sort_values(KEY + ["bar", "t0"], kind="stable")
    g = mu.groupby(KEY + ["bar"], sort=False)
    x = g.first()[["t0", "x0", "y0", "z0", "px0", "py0", "pz0"]]
    x["len"] = g.len.sum()
    x["edep"] = g.edep.sum()
    x = x.reset_index()
    x["p"] = np.sqrt(x.px0 ** 2 + x.py0 ** 2 + x.pz0 ** 2)
    return x


def build(step_paths, gen_paths, geom):
    """Library trees {name: DataFrame} from G4 CrvStep tables (step_paths) and the gen tables of the same shards."""
    steps = tables.read_many(step_paths, "steps")
    parts = tables.read_many(step_paths, "particles", KEY + ["key", "parent", "pdg", "isprim", "genidx"])
    s = classify(steps, parts, geom.sector)
    lib = {}
    # 1. muon descendants per crossing
    x = _crossings(s[s.cls == "mu_primary"])
    x["group"] = _group_index(geom, x.bar)
    edges = []
    for gi in range(len(GROUPS)):
        xg = x[x.group == gi]
        edges += [(0, gi, e) for e in np.quantile(xg.p, P_QUANTILES)]
        edges += [(1, gi, e) for e in np.quantile(xg.len, L_QUANTILES)]
    lib["libbins"] = pd.DataFrame(edges, columns=["kind", "group", "edge"]).astype({"kind": np.int64, "group": np.int64})
    x["bin"] = _bin(lib["libbins"], x.group.to_numpy(), x.p.to_numpy(), x.len.to_numpy())
    x["xid"] = np.arange(len(x))
    d = s[s.cls == "mu_descendant"].reset_index(drop=True)
    d["did"] = np.arange(len(d))
    m = d[KEY + ["did", "bar", "x0", "y0", "z0", "x1", "y1", "z1"]].merge(
        x[KEY + ["xid", "bar", "x0", "y0", "z0"]].rename(columns={"bar": "xbar", "x0": "ex", "y0": "ey", "z0": "ez"}),
        on=KEY)
    mid = 0.5 * (m[["x0", "y0", "z0"]].to_numpy() + m[["x1", "y1", "z1"]].to_numpy())
    dist = np.linalg.norm(mid - m[["ex", "ey", "ez"]].to_numpy(), axis=1)
    m["score"] = np.where(m.bar.to_numpy() == m.xbar.to_numpy(), -1.0, dist)
    best = m.loc[m.groupby("did").score.idxmin(), ["did", "xid", "score"]]
    d = d.merge(best, on="did")
    d = d.merge(x[["xid", "t0", "x0", "y0", "z0", "bar"]].rename(
        columns={"t0": "xt0", "x0": "ex", "y0": "ey", "z0": "ez", "bar": "xbar"}), on="xid")
    db, xb = d.bar.to_numpy(), d.xbar.to_numpy()
    dc = geom.center[db] - geom.center[xb]
    entry_len = (d[["ex", "ey", "ez"]].to_numpy() - geom.center[xb])[np.arange(len(d)), geom.local_to_world[xb, 2]]
    lib["libdesc"] = pd.DataFrame({
        "xid": d.xid.to_numpy().astype(np.int64), "sp": d.sp.to_numpy().astype(np.int64),
        "samebar": (d.bar.to_numpy() == d.xbar.to_numpy()).astype(np.int64),
        "pdg": d.merge(parts[KEY + ["key", "pdg"]].rename(columns={"key": "sp", "pdg": "spdg"}),
                       on=KEY + ["sp"], how="left").spdg.fillna(11).to_numpy().astype(np.int32),
        "dt0": (d.t0 - d.xt0).to_numpy(), "dt1": (d.t1 - d.xt0).to_numpy(),
        **{f"d{c}{e}": (d[f"{c}{e}"] - d[f"e{c}"]).to_numpy() for c in "xyz" for e in "01"},
        "px0": d.px0.to_numpy(), "py0": d.py0.to_numpy(), "pz0": d.pz0.to_numpy(), "p1": d.p1.to_numpy(),
        "len": d.len.to_numpy(), "edep": d.edep.to_numpy(),
        "dcx": dc[:, 0], "dcy": dc[:, 1], "dcz": dc[:, 2], "entrylen": entry_len,
        **{f"l{c}{e}": (d[f"{c}{e}"].to_numpy() - geom.center[db][:, j]) for j, c in enumerate("xyz") for e in "01"}})
    lib["libxing"] = pd.DataFrame({"xid": x.xid, "bin": x.bin.astype(np.int64),
                                   "ux": x.px0 / x.p, "uy": x.py0 / x.p, "uz": x.pz0 / x.p})
    n_unattached = int(len(s[s.cls == "mu_descendant"]) - len(d))
    # 2. non-muon showers, keyed by primary
    gen = tables.read_many(step_paths, "gen", KEY + ["idx", "pdg", "e", "t"])
    prim = parts[(parts.isprim == 1) & (parts.pdg.abs() != tables.PRIMARY_MUON_PDG)]
    prim = prim.merge(gen.rename(columns={"idx": "genidx", "pdg": "gpdg"}), on=KEY + ["genidx"], how="left")
    prim["apdg"] = prim.pdg.abs()
    prim["ke"] = prim.e - prim.apdg.map(MASS).fillna(0.0)
    nm = s[s.cls.str.startswith("nonmu")].merge(
        prim[KEY + ["key", "apdg", "ke", "t"]].rename(columns={"key": "prim_key"}), on=KEY + ["prim_key"])
    sh = nm[KEY + ["prim_key", "apdg", "ke"]].drop_duplicates(KEY + ["prim_key"]).reset_index(drop=True)
    sh["shid"] = np.arange(len(sh))
    sh["kebin"] = _ke_bin(sh.ke.to_numpy())
    nm = nm.merge(sh[KEY + ["prim_key", "shid"]], on=KEY + ["prim_key"])
    lib["libshower"] = pd.DataFrame({"shid": sh.shid.astype(np.int64), "apdg": sh.apdg.astype(np.int64),
                                     "kebin": sh.kebin.astype(np.int64), "ke": sh.ke.to_numpy()})
    lib["libshowersteps"] = pd.DataFrame({
        "shid": nm.shid.to_numpy().astype(np.int64), "sp": nm.sp.to_numpy().astype(np.int64),
        "isroot": (nm.sp.to_numpy() == nm.prim_key.to_numpy()).astype(np.int64),
        "pdg": nm.merge(parts[KEY + ["key", "pdg"]].rename(columns={"key": "sp", "pdg": "spdg"}),
                        on=KEY + ["sp"], how="left").spdg.to_numpy().astype(np.int32),
        "bar": nm.bar.to_numpy().astype(np.int64),
        "dt0": (nm.t0 - nm.t).to_numpy(), "dt1": (nm.t1 - nm.t).to_numpy(),
        **{c: nm[c].to_numpy().astype(float) for c in STEP_COLS + ["px0", "py0", "pz0", "p1", "len", "edep"]}})
    # generated non-muon primaries per (pdg, KE bin): the denominator of the reach probability
    g = tables.read_many(gen_paths, "gen", KEY + ["pdg", "e"])
    g = g[g.pdg.abs() != tables.PRIMARY_MUON_PDG]
    g["apdg"] = g.pdg.abs()
    g["kebin"] = _ke_bin((g.e - g.apdg.map(MASS).fillna(0.0)).to_numpy())
    ng = g.groupby(["apdg", "kebin"]).size().rename("ngen").reset_index()
    ns = sh.groupby(["apdg", "kebin"]).size().rename("nshower").reset_index()
    lg = ng.merge(ns, on=["apdg", "kebin"], how="left").fillna({"nshower": 0})
    lib["libgen"] = lg.astype({"apdg": np.int64, "kebin": np.int64, "ngen": np.int64, "nshower": np.int64})
    info = {"crossings": len(x), "desc_steps": len(d), "desc_steps_unattached": n_unattached,
            "showers": len(sh), "shower_steps": len(nm), "generated_nonmu": int(len(g)),
            "max_shower_sp": int(nm.sp.max()) if len(nm) else 0}
    return lib, info


def write_library(path, lib):
    core_tables.write(path, {k: lib[k] for k in LIB_TREES}, as_ttree=True)


def read_library(path):
    return {k: core_tables.read(path, k) for k in LIB_TREES}


# ----------------------------------------------------------------------------------------------------------- sample
def _desc_for_crossings(fx, lib, geom, rng):
    """Descendant steps for the fast crossings fx (one row per crossing). Returns (steps, n_dropped)."""
    p = np.sqrt(fx.px0 ** 2 + fx.py0 ** 2 + fx.pz0 ** 2).to_numpy()
    b = _bin(lib["libbins"], _group_index(geom, fx.bar), p, fx.len.to_numpy())
    lx = lib["libxing"].sort_values("bin", kind="stable")
    lbin = lx.bin.to_numpy()
    lo, hi = np.searchsorted(lbin, b, "left"), np.searchsorted(lbin, b, "right")
    assert np.all(hi > lo), "empty library bin"
    pick = lx.xid.to_numpy()[lo + (rng.random(len(b)) * (hi - lo)).astype(np.int64)]
    ld = lib["libdesc"].sort_values("xid", kind="stable")
    lxid = ld.xid.to_numpy()
    a, z = np.searchsorted(lxid, pick, "left"), np.searchsorted(lxid, pick, "right")
    cnt = z - a
    ix = np.repeat(np.arange(len(fx)), cnt)                        # fast crossing of each new step
    rows = np.repeat(a - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    dd = ld.iloc[rows]
    lu = lib["libxing"].set_index("xid").loc[pick[ix], ["ux", "uy", "uz"]].to_numpy()
    fu = _unit(fx[["px0", "py0", "pz0"]].to_numpy())[ix]
    entry = fx[["x0", "y0", "z0"]].to_numpy()[ix]
    r0 = entry + rotate(dd[["dx0", "dy0", "dz0"]].to_numpy(), lu, fu)
    r1 = entry + rotate(dd[["dx1", "dy1", "dz1"]].to_numpy(), lu, fu)
    mom = rotate(dd[["px0", "py0", "pz0"]].to_numpy(), lu, fu)
    bar = bar_containing(0.5 * (r0 + r1), geom)
    xbar = fx.bar.to_numpy()[ix]
    same = dd.samebar.to_numpy() == 1
    other = ~same
    if other.any():
        tb = bar_containing(geom.center[xbar[other]] + dd.loc[:, ["dcx", "dcy", "dcz"]].to_numpy()[other], geom)
        ok = tb >= 0
        ok[ok] = geom.local_to_world[tb[ok], 2] == geom.local_to_world[xbar[other][ok], 2]
        tbs = np.where(ok, tb, 0)
        la = geom.local_to_world[tbs, 2]
        rows_o = np.arange(ok.size)
        shift = (entry[other] - geom.center[xbar[other]])[rows_o, la] - dd.entrylen.to_numpy()[other]
        o0 = geom.center[tbs] + dd.loc[:, ["lx0", "ly0", "lz0"]].to_numpy()[other]
        o1 = geom.center[tbs] + dd.loc[:, ["lx1", "ly1", "lz1"]].to_numpy()[other]
        o0[rows_o, la] += shift
        o1[rows_o, la] += shift
        lo_, hi_ = geom.center[tbs] - geom.halflen[tbs], geom.center[tbs] + geom.halflen[tbs]
        ok &= (0.5 * (o0 + o1) >= lo_).all(1) & (0.5 * (o0 + o1) <= hi_).all(1)
        r0[other], r1[other] = np.clip(o0, lo_, hi_), np.clip(o1, lo_, hi_)
        bar[other] = np.where(ok, tb, -1)
        mom[other] = dd.loc[:, ["px0", "py0", "pz0"]].to_numpy()[other]
    clamp = same & (bar != xbar)
    if clamp.any():
        c, h = geom.center[xbar[clamp]], geom.halflen[xbar[clamp]]
        r0[clamp] = np.clip(r0[clamp], c - h, c + h)
        r1[clamp] = np.clip(r1[clamp], c - h, c + h)
        bar[clamp] = xbar[clamp]
    keep = bar >= 0
    out = pd.DataFrame({k: fx[k].to_numpy()[ix][keep] for k in KEY})
    out["bar"] = bar[keep]
    out["lsp"] = dd.sp.to_numpy()[keep]
    out["xi"] = ix[keep]
    out["pdg"] = dd.pdg.to_numpy()[keep]
    t0 = fx.t0.to_numpy()[ix]
    out["t0"] = (t0 + dd.dt0.to_numpy())[keep]
    out["t1"] = (t0 + dd.dt1.to_numpy())[keep]
    for j, c in enumerate("xyz"):
        out[f"{c}0"], out[f"{c}1"], out[f"p{c}0"] = r0[keep, j], r1[keep, j], mom[keep, j]
    for c in ("p1", "len", "edep"):
        out[c] = dd[c].to_numpy()[keep]
    return out, int((~keep).sum()), int(clamp.sum())


def _showers_for_gen(gen, lib, rng):
    """Shower steps for the non-muon CRY primaries in gen. Returns steps with a per-event shower index."""
    g = gen[gen.pdg.abs() != tables.PRIMARY_MUON_PDG].reset_index(drop=True)
    g["apdg"] = g.pdg.abs()
    g["kebin"] = _ke_bin((g.e - g.apdg.map(MASS).fillna(0.0)).to_numpy())
    lg = lib["libgen"]
    g = g.merge(lg, on=["apdg", "kebin"], how="left").fillna({"ngen": 0, "nshower": 0})
    prob = np.where(g.ngen > 0, g.nshower / np.maximum(g.ngen, 1), 0.0)
    g = g[rng.random(len(g)) < prob].reset_index(drop=True)
    ls = lib["libshower"].sort_values(["apdg", "kebin"], kind="stable")
    code = ls.apdg.to_numpy() * 100 + ls.kebin.to_numpy()
    want = g.apdg.to_numpy() * 100 + g.kebin.to_numpy()
    lo, hi = np.searchsorted(code, want, "left"), np.searchsorted(code, want, "right")
    g["shid"] = ls.shid.to_numpy()[lo + (rng.random(len(g)) * (hi - lo)).astype(np.int64)]
    g["k"] = g.groupby(KEY).cumcount()
    st = lib["libshowersteps"].merge(g[KEY + ["shid", "k", "t"]], on="shid")
    st["t0"] = st.t + st.dt0
    st["t1"] = st.t + st.dt1
    return st


def sample(fast_steps, fast_parts, gen, lib, geom, rng):
    """Fast tier-1a tables + tier 1b: (steps, particles, info). The muon keeps key 1; descendants 2..; showers k*1e7+sp."""
    fx = fast_steps.copy()
    desc, n_drop, n_clamp = _desc_for_crossings(fx, lib, geom, rng)
    # descendant particles: one per (event, crossing, library particle), keys 2.. within the event
    u = desc[KEY + ["xi", "lsp", "pdg"]].drop_duplicates(KEY + ["xi", "lsp"]).reset_index(drop=True)
    u["key"] = u.groupby(KEY).cumcount() + 2
    desc = desc.merge(u[KEY + ["xi", "lsp", "key"]], on=KEY + ["xi", "lsp"])
    desc["sp"] = desc.key
    dparts = pd.DataFrame({k: u[k].to_numpy() for k in KEY})
    dparts["key"], dparts["parent"], dparts["pdg"], dparts["isprim"] = u.key, 1, u.pdg, 0
    sh = _showers_for_gen(gen, lib, rng)
    sh["sp"] = (sh.k + 1) * SHOWER_KEY_BASE + sh.sp
    su = sh[KEY + ["sp", "pdg", "isroot", "k", "shid"]].drop_duplicates(KEY + ["sp"])
    # every shower gets its primary as a particle (parent of the others), even when only daughters have steps
    roots = sh[KEY + ["k", "shid"]].drop_duplicates()
    rootpdg = lib["libshower"].set_index("shid").apdg
    gsign = gen[gen.pdg.abs() != tables.PRIMARY_MUON_PDG]
    rparts = pd.DataFrame({k: roots[k].to_numpy() for k in KEY})
    rparts["key"] = (roots.k.to_numpy() + 1) * SHOWER_KEY_BASE
    rparts["parent"], rparts["isprim"] = -1, 1
    rparts["pdg"] = rootpdg.loc[roots.shid.to_numpy()].to_numpy()
    # shower steps of the root itself are re-pointed to the primary's key
    sh.loc[sh.isroot == 1, "sp"] = (sh.k[sh.isroot == 1] + 1) * SHOWER_KEY_BASE
    su = su[su.isroot == 0]
    sparts = pd.DataFrame({k: su[k].to_numpy() for k in KEY})
    sparts["key"] = su.sp.to_numpy()
    sparts["parent"] = (su.k.to_numpy() + 1) * SHOWER_KEY_BASE
    sparts["pdg"], sparts["isprim"] = su.pdg.to_numpy(), 0
    cols = list(fast_steps.columns)
    steps = pd.concat([fx[cols], desc.reindex(columns=cols), sh.reindex(columns=cols)], ignore_index=True)
    steps["nstep"] = steps.groupby(KEY).bar.transform("size")
    parts = pd.concat([fast_parts, dparts, rparts, sparts], ignore_index=True)
    for c in KEY:
        steps[c] = steps[c].astype(np.uint32)
        parts[c] = parts[c].astype(np.uint32)
    for c in ("nstep", "bar", "sp"):
        steps[c] = steps[c].astype(np.int64)
    for c in steps.columns:
        if c not in KEY and c not in ("nstep", "bar", "sp"):
            steps[c] = steps[c].astype(np.float64)
    parts = parts.astype({"key": np.int64, "parent": np.int64, "isprim": np.int64, "pdg": np.int32})
    steps = steps.sort_values(KEY + ["t0"], kind="stable").reset_index(drop=True)
    info = {"desc_steps": len(desc), "desc_dropped": n_drop, "desc_clamped": n_clamp,
            "showers": int(len(roots)), "shower_steps": len(sh), "nonmu_primaries": int(len(gsign))}
    return steps, parts, info
