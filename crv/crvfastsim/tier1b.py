"""Tier 1b: G4-built library of the CRV deposits tier 1a does not make (docs/plan.md section 6).

Two parts, both resampled from G4 CrvStep tables (CrvStepTableDumper) of training shards:

1. Muon descendants (deltas above the production cut, decay electrons, ...). Each descendant step is attached to one
   primary-muon bar crossing of its event: the crossing in the same bar, else the crossing whose entry point is closest
   to the step (a library-building convention; the arms are never paired this way). A library crossing stores its
   descendants as offsets from its entry point. Sampling: each fast crossing draws a library crossing from the same
   (sector group, momentum, path-length) bin, with momentum and path-length edges at quantiles per sector group
   (P_QUANTILES, L_QUANTILES; the path-length bins are finer in both tails); the offsets are rotated from the G4 muon direction to the fast one and
   added to the fast entry point and time. A same-bar step that falls outside the fast crossing's bar keeps its
   rotated position along the bar but takes its G4 position across the bar (thickness and width, both endpoints, from
   the library's in-bar coordinates); fit_inside then guards the box. Pushing such steps onto the nearest face (first
   version) or just inside it piled descendant energy at the faces, where the CRV lookup tables give less light.
   Steps in another bar keep the bar structure instead: the library stores the centre offset between their bar and the
   crossing's bar and their position inside their bar; the fast step goes to the bar at that centre offset from the fast
   crossing's bar (same length axis), at the stored position, moved along the bar by the shift of the muon entry along
   the length. With no bar of the same length axis there (module gap, sector edge, a bar along another axis) the
   step goes to the nearest such bar in the transverse plane (nearest_bar); a step shifted past a bar end is moved
   back inside (fit_inside). Dropping both cases (first versions) lost ~44 % of other-bar steps.
   Opt-in refinements (sample(context, contain, point); defaults off reproduce the earlier sampling bit for bit):
   - context: the descendants seen in a bar depend on what the muon crossed just before it (delta rays from the
     layers, absorbers and skins above build up with depth; a module stacked under another sector receives that
     sector's deltas), and the other-bar offsets of a library crossing point to the layers that exist around it.
     Libraries store crossing_context per crossing (layer rank in its module, and whether the muon crossed another
     module within UPSTREAM_MM before it); a fast crossing draws from the library crossings of the same (bin,
     context), falling back to the bin alone when that cell holds fewer than MIN_CTX_CROSSINGS. Pooling every layer
     flattens G4's depth build-up and puts other-bar offsets of deeper crossings (pointing up) on top-layer fast
     crossings, which have no bar there: nearest_bar piles them into the top layer, most at the module edges.
   - contain: a same-bar step whose midpoint is inside but whose rotated endpoints leave the bar is moved inside
     (fit_inside); without it ~0.2 % of fast steps start > 1 mm outside their bar.
   - point: an other-bar step whose target bar is missing or runs along another axis (a step in a stacked sector
     of the other orientation, e.g. EX above T1, T2 above EX, M above T2) goes to the bar containing its library
     position relative to the fast crossing, shifted with the entry along the crossing's bar; nearest_bar (same
     axis as the crossing) remains the fallback. Without it such steps move into the crossing's own sector.
   - mirror: other-bar offsets are not rotated (they keep the bar structure), so a library crossing whose muon leans
     the other way across (along) the bar than the fast muon puts its neighbour-bar steps on the wrong side: G4's
     other-bar steps lie off the library muon's path, and pasted unreflected they land on the fast muon's path (in
     bars it crosses). The offsets (bar-centre offset, in-bar position, momentum, and along the bar the offset from
     the entry) are reflected across the crossing bar's width and length planes where the signs differ.
   - translate: the bar-structure placement keeps the library's bar offset but not where the muon entered its bar
     across the width; the layers of a module are staggered by ~0.8 bar widths, so whether the bar at that offset in
     the next layer is one the fast muon crosses depends on the entry position. With translate an other-bar step goes
     to the bar containing (fast entry + its unrotated library offset from the library entry), as the G4 step sat
     relative to its muon; the bar-structure placement is the fallback when that point is in no bar.
2. Non-muon primaries (e, gamma, n, p, pi from CRY). A library shower is every CrvStep whose primary is that particle,
   with absolute positions and times relative to the primary's generation time. Sampling: each non-muon CRY primary of
   the event (gen table) reaches the CRV with the library probability of its (pdg, kinetic-energy) bin, and then gets a
   random shower of that bin. Known limit: shower bar indices are those of the library event, so a geometry move does
   not move them.

Descendant and shower particles are flattened to depth 1 (parent = the primary). Vectorised; no per-event loops.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from . import genassoc, tables
from .analyses.census import SECTOR_GROUP, classify

KEY = tables.KEY
GROUPS = ("EX", "T1", "T2", "M")
MASS = {11: 0.511, 13: 105.658, 22: 0.0, 211: 139.570, 321: 493.677, 2212: 938.272, 2112: 939.565}
KE_EDGES = np.geomspace(1.0, 1e6, 19)
P_QUANTILES = (0.25, 0.5, 0.75)
L_QUANTILES = (0.05, 0.15, 1 / 3, 2 / 3, 0.9, 0.97)  # finer in both tails: corner clips (small pulses) and long crossings
N_PBINS, N_LBINS = len(P_QUANTILES) + 1, len(L_QUANTILES) + 1
SHOWER_KEY_BASE = 10_000_000
UPSTREAM_MM = 300.0      # crossing context: another module crossed within this distance before the entry
N_CTX = 8                # context codes: layer rank (0 = top, clipped to 3) x 2 + upstream flag
MIN_CTX_CROSSINGS = 20   # fewer library crossings in a (bin, context) cell: draw from the bin
STEP_COLS = ["x0", "y0", "z0", "x1", "y1", "z1"]
LIB_TREES = ("libbins", "libxing", "libdesc", "libshower", "libshowersteps", "libgen")


def _group_index(geom, bar):
    """Library group (index into GROUPS) per bar: the run configuration's sector -> group map when the geometry carries
    one (geom.lib_group, set by runconfig.attach), else the extracted v04 sectors (EX, T1, T2, taggers)."""
    sg = getattr(geom, "lib_group", None)
    g = (np.asarray(sg, dtype=object) if sg is not None else SECTOR_GROUP)[geom.sector[np.asarray(bar, np.int64)]]
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


def layer_rank(geom):
    """Per bar, the rank of its layer within its module along the thickness axis: 0 = top (largest coordinate)."""
    ta = geom.local_to_world[:, 0]
    df = pd.DataFrame({"s": geom.sector, "m": geom.module, "l": geom.layer,
                       "t": geom.center[np.arange(geom.nbars), ta]})
    lp = df.groupby(["s", "m", "l"]).t.transform("mean")
    return lp.groupby([df.s, df.m]).rank(method="dense", ascending=False).to_numpy().astype(np.int64) - 1


def crossing_context(x, geom):
    """Context code per primary-muon crossing (rows of x: KEY, sp = muon key, bar, t0, x0, y0, z0 = entry):
    min(layer rank, 3) * 2 + 1 if the same muon entered a bar of another module earlier, within UPSTREAM_MM of this
    entry (a sector or module stacked above). Vectorised over all crossing pairs of each muon."""
    n = len(x)
    bar = x.bar.to_numpy().astype(np.int64)
    lay = np.minimum(layer_rank(geom)[bar], 3)
    if n == 0:
        return lay
    mod = (geom.sector.astype(np.int64) * 1000 + geom.module)[bar]
    mu = x.groupby(KEY + ["sp"], sort=False).ngroup().to_numpy()
    d = pd.DataFrame({"i": np.arange(n), "mu": mu, "t": x.t0.to_numpy(), "mod": mod,
                      "x": x.x0.to_numpy(), "y": x.y0.to_numpy(), "z": x.z0.to_numpy()})
    pr = d.merge(d, on="mu", suffixes=("", "_p"))
    pr = pr[(pr.t_p < pr.t) & (pr.mod_p != pr["mod"])]
    dist = np.sqrt((pr.x - pr.x_p) ** 2 + (pr.y - pr.y_p) ** 2 + (pr.z - pr.z_p) ** 2)
    up = np.zeros(n, np.int64)
    up[np.unique(pr.i.to_numpy()[dist.to_numpy() < UPSTREAM_MM])] = 1
    return lay * 2 + up


INSET_MM = 0.1


def fit_inside(r0, r1, lo, hi, inset=INSET_MM):
    """Move each step (r0 -> r1) into its box [lo, hi]: shift it so its midpoint is >= inset inside every face, then
    clip the endpoints. Clipping the endpoints alone put the step on a face, where the CRV lookup tables give no light."""
    mid = 0.5 * (r0 + r1)
    tgt = np.clip(mid, lo + inset, hi - inset)
    d = tgt - mid
    return np.clip(r0 + d, lo, hi), np.clip(r1 + d, lo, hi)


def nearest_bar(points, length_axis, geom):
    """Per point, the bar with the given length axis whose centre is nearest in the two transverse coordinates."""
    out = np.full(len(points), -1, np.int64)
    bar_axis = geom.local_to_world[:, 2]
    for a in np.unique(length_axis):
        m = length_axis == a
        cand = np.flatnonzero(bar_axis == a)
        if not len(cand):
            continue
        t = [k for k in range(3) if k != a]
        idx = np.flatnonzero(m)
        step = max(1000, int(1.2e7 // max(len(cand), 1)))
        for i in range(0, len(idx), step):
            j = idx[i:i + step]
            d = ((points[j][:, None, t] - geom.center[cand][None, :, t]) ** 2).sum(axis=2)
            out[j] = cand[np.argmin(d, axis=1)]
    return out


def _ke_bin(ke):
    return np.clip(np.searchsorted(KE_EDGES, ke, "right") - 1, 0, len(KE_EDGES) - 2)


def _evkey(df):
    return (df.run.to_numpy().astype(np.int64) << 40) | (df.subrun.to_numpy().astype(np.int64) << 20) | \
        df.event.to_numpy().astype(np.int64)


def bar_containing(pos, geom, chunk=None):
    """Bar index whose box contains each point (-1 if none; the lowest index if boxes touch). Chunks hold ~1e8
    point-bar pairs (extracted v04: 20,000 points x 576 bars, the earlier fixed chunk)."""
    chunk = chunk or max(1000, int(1.2e7 // max(geom.nbars, 1)))
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
    x = g.first()[["sp", "t0", "x0", "y0", "z0", "px0", "py0", "pz0"]]
    x["len"] = g.len.sum()
    x["edep"] = g.edep.sum()
    x = x.reset_index()
    x["p"] = np.sqrt(x.px0 ** 2 + x.py0 ** 2 + x.pz0 ** 2)
    return x


def _bin_edges(step_paths, geom):
    """Per-group momentum and path-length quantile edges of the primary-muon crossings over all step_paths (one file at
    a time: crossings only, so memory stays small)."""
    xs = []
    for f in step_paths:
        steps = tables.read(f, "steps", KEY + ["bar", "sp", "t0", "edep", "len", "x0", "y0", "z0", "px0", "py0", "pz0"])
        parts = tables.read(f, "particles", KEY + ["key", "pdg", "isprim"])
        prim = parts[(parts.isprim == 1) & (parts.pdg.abs() == tables.PRIMARY_MUON_PDG)]  # every primary muon, as classify
        mu = steps.merge(prim[KEY + ["key"]].rename(columns={"key": "sp"}), on=KEY + ["sp"])
        x = _crossings(mu)
        xs.append(pd.DataFrame({"group": _group_index(geom, x.bar), "p": x.p.to_numpy(), "len": x.len.to_numpy()}))
    x = pd.concat(xs, ignore_index=True)
    edges = []
    for gi in range(len(GROUPS)):
        xg = x[x.group == gi]
        edges += [(0, gi, e) for e in np.quantile(xg.p, P_QUANTILES)]
        edges += [(1, gi, e) for e in np.quantile(xg.len, L_QUANTILES)]
    return pd.DataFrame(edges, columns=["kind", "group", "edge"]).astype({"kind": np.int64, "group": np.int64})


def build(step_paths, gen_paths, geom, generated=None):
    """Library from G4 CrvStep tables, one file at a time (a GPVM holds ~5 production files at once, not 20).

    Bin edges come from all files first; each file's crossings, descendants and showers get ids offset past the previous
    files'; the reach probability uses the showers of all files over the scaled gen-table counts.
    """
    step_paths = list(step_paths)
    gen_paths = list(gen_paths)
    # Check each physical donor before binning; bootstrap repetitions retain their weight.
    validated = {str(Path(f).resolve()): None for f in step_paths}
    for f in validated:
        validated[f] = genassoc.validate_donor(f)
    generator_streams = [genassoc.validate_generator_stream(f)
                         for f in dict.fromkeys(str(Path(p).resolve()) for p in gen_paths)]
    libbins = _bin_edges(step_paths, geom)
    parts_out = {k: [] for k in ("libxing", "libdesc", "libshower", "libshowersteps")}
    info = {"crossings": 0, "desc_steps": 0, "desc_steps_unattached": 0, "showers": 0, "shower_steps": 0,
            "max_shower_sp": 0}
    xoff = soff = 0
    for f in step_paths:
        lib, inf = _build_one([f], geom, libbins, _already_validated=validated)
        lib["libxing"]["xid"] += xoff
        lib["libdesc"]["xid"] += xoff
        lib["libshower"]["shid"] += soff
        lib["libshowersteps"]["shid"] += soff
        xoff += inf["crossings"]
        soff += inf["showers"]
        for k in parts_out:
            parts_out[k].append(lib[k])
        for k in info:
            info[k] = max(info[k], inf[k]) if k == "max_shower_sp" else info[k] + inf[k]
    lib = {k: pd.concat(v, ignore_index=True) for k, v in parts_out.items()}
    lib["libbins"] = libbins
    lib["libgen"], info["generated_nonmu"] = _reach(lib["libshower"], gen_paths, generated)
    info["generated_scaled_to"] = generated
    info["files"] = len(step_paths)
    info["generator_associations"] = list(validated.values())
    info["generator_streams"] = generator_streams
    return lib, info


def _reach(libshower, gen_paths, generated):
    """Generated non-muon primaries per (pdg, KE bin) and the library showers of each: the reach probability."""
    g = tables.read_many(gen_paths, "gen", KEY + ["pdg", "e"])
    n_gen_tables = len(pd.concat([genassoc.generator_event_keys(p) for p in gen_paths], ignore_index=True).drop_duplicates())
    if n_gen_tables == 0:
        raise ValueError("generator denominator has no events")
    g = g[g.pdg.abs() != tables.PRIMARY_MUON_PDG]
    g["apdg"] = g.pdg.abs()
    g["kebin"] = _ke_bin((g.e - g.apdg.map(MASS).fillna(0.0)).to_numpy())
    ng = g.groupby(["apdg", "kebin"]).size().rename("ngen").reset_index()
    if generated:
        ng["ngen"] = np.rint(ng.ngen * (float(generated) / n_gen_tables)).astype(np.int64)
    ns = libshower.groupby(["apdg", "kebin"]).size().rename("nshower").reset_index()
    lg = ng.merge(ns, on=["apdg", "kebin"], how="left").fillna({"nshower": 0})
    return lg.astype({"apdg": np.int64, "kebin": np.int64, "ngen": np.int64, "nshower": np.int64}), int(len(g))


def _build_one(step_paths, geom, libbins, *, _already_validated=None):
    """Library trees {name: DataFrame} of the given G4 CrvStep tables with the given bin edges (libgen not included).


    gen_paths of build(): the generated non-muon primaries per (pdg, KE bin), the denominator of the reach probability.
    By default the gen streams of the same shards. generated: generated events behind step_paths when they are not
    those shards (production dts have no gen stream; dh.gencount per file): the gen-table counts are scaled by
    generated / (generated events in gen_paths), i.e. the same CRY configuration is assumed.
    """
    step_paths = list(step_paths)
    validated = {} if _already_validated is None else dict(_already_validated)
    donor_keys = list(dict.fromkeys(str(Path(f).resolve()) for f in step_paths))
    for f in donor_keys:
        if f not in validated:
            validated[f] = genassoc.validate_donor(f)
    steps = tables.read_many(step_paths, "steps")
    parts = tables.read_many(step_paths, "particles", KEY + ["key", "parent", "pdg", "isprim", "genidx"])
    s = classify(steps, parts, geom.sector)
    lib = {}
    # 1. muon descendants per crossing
    x = _crossings(s[s.cls == "mu_primary"])
    x["group"] = _group_index(geom, x.bar)
    x["bin"] = _bin(libbins, x.group.to_numpy(), x.p.to_numpy(), x.len.to_numpy())
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
                                   "ux": x.px0 / x.p, "uy": x.py0 / x.p, "uz": x.pz0 / x.p,
                                   "ctx": crossing_context(x, geom).astype(np.int64)})
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
    info = {"crossings": len(x), "desc_steps": len(d), "desc_steps_unattached": n_unattached,
            "showers": len(sh), "shower_steps": len(nm), "max_shower_sp": int(nm.sp.max()) if len(nm) else 0,
            "generator_associations": [validated[f] for f in donor_keys]}
    return lib, info


def write_library(path, lib):
    core_tables.write(path, {k: lib[k] for k in LIB_TREES}, as_ttree=True)


def read_library(path):
    return {k: core_tables.read(path, k) for k in LIB_TREES}


# ----------------------------------------------------------------------------------------------------------- sample
def _desc_for_crossings(fx, lib, geom, rng, context=False, contain=False, point=False, mirror=False,
                        translate=False):
    """Descendant steps for the fast crossings fx (one row per crossing). Returns (steps, n_dropped, n_clamped, why).
    Opt-in (all off reproduces the earlier sampling bit for bit): context = draw within the crossing-context cell
    (library with a ctx column); contain = same-bar endpoints into the bar; point = other-bar steps whose target bar
    is missing or along another axis go to the bar containing their library position relative to the fast crossing;
    mirror = other-bar offsets reflected across the crossing bar's width (length) plane where the library and fast muons
    point opposite ways along that axis; translate = other-bar steps go to the bar containing their library offset from
    the muon entry (unrotated) added to the fast entry, the bar-structure placement remaining the fallback."""
    p = np.sqrt(fx.px0 ** 2 + fx.py0 ** 2 + fx.pz0 ** 2).to_numpy()
    b = _bin(lib["libbins"], _group_index(geom, fx.bar), p, fx.len.to_numpy())
    lx = lib["libxing"]
    if context and "ctx" not in lx.columns:
        raise ValueError("crossing context needs a library built with the ctx column (tier1b-build after this change)")
    if context:   # draw within the crossing's (bin, context) cell, else within its bin
        lx = lx.sort_values(["bin", "ctx"], kind="stable")
        code = b * N_CTX + crossing_context(fx, geom)
        lcode = lx.bin.to_numpy() * N_CTX + lx.ctx.to_numpy()
        lo, hi = np.searchsorted(lcode, code, "left"), np.searchsorted(lcode, code, "right")
        few = (hi - lo) < MIN_CTX_CROSSINGS
        lbin = lx.bin.to_numpy()
        lo[few], hi[few] = np.searchsorted(lbin, b[few], "left"), np.searchsorted(lbin, b[few], "right")
    else:
        lx = lx.sort_values("bin", kind="stable")
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
    why = {"other": int(other.sum()), "no_bar": 0, "axis": 0, "off_end": 0}
    if other.any():
        dc = dd.loc[:, ["dcx", "dcy", "dcz"]].to_numpy()[other]
        l0 = dd.loc[:, ["lx0", "ly0", "lz0"]].to_numpy()[other]
        l1 = dd.loc[:, ["lx1", "ly1", "lz1"]].to_numpy()[other]
        elen = dd.entrylen.to_numpy()[other].copy()
        mo = dd.loc[:, ["px0", "py0", "pz0"]].to_numpy()[other]
        d0 = dd.loc[:, ["dx0", "dy0", "dz0"]].to_numpy()[other]
        d1 = dd.loc[:, ["dx1", "dy1", "dz1"]].to_numpy()[other]
        xa = geom.local_to_world[xbar[other], 2]
        if mirror:   # reflect across the crossing bar's width / length plane where the muons point opposite ways
            rr = np.arange(len(xa))
            for ax, length in ((geom.local_to_world[xbar[other], 1], False), (xa, True)):
                flip = np.flatnonzero(lu[other][rr, ax] * fu[other][rr, ax] < 0)
                fa = ax[flip]
                for arr in (dc, l0, l1, mo, d0, d1):
                    arr[flip, fa] *= -1
                if length:   # offset from the entry along the bar, dc + l - entrylen, changes sign
                    elen[flip] *= -1
            why["mirror_w"] = int((lu[other][rr, geom.local_to_world[xbar[other], 1]] *
                                   fu[other][rr, geom.local_to_world[xbar[other], 1]] < 0).sum())
        target = geom.center[xbar[other]] + dc
        tb = bar_containing(target, geom)
        why["no_bar"] = int((tb < 0).sum())
        bad = (tb < 0) | (geom.local_to_world[np.maximum(tb, 0), 2] != xa)
        why["axis"] = int(bad.sum() - why["no_bar"])
        pt = np.zeros(len(tb), bool)
        Q0, Q1 = np.zeros_like(l0), np.zeros_like(l1)
        if translate:   # library offset from the muon entry, unrotated, from the fast entry: bar containing it
            t0_, t1_ = entry[other] + d0, entry[other] + d1
            tt = bar_containing(0.5 * (t0_ + t1_), geom)
            use = tt >= 0
            pt[use], tb[use], bad[use] = True, tt[use], False
            Q0[use], Q1[use] = t0_[use], t1_[use]
            why["translate"] = int(use.sum())
        if point and bad.any():   # library position relative to the fast crossing (shifted along its bar), any axis
            ib = np.flatnonzero(bad)
            rb = np.arange(len(ib))
            sx = (entry[other][ib] - geom.center[xbar[other][ib]])[rb, xa[ib]] - elen[ib]
            q0, q1 = target[ib] + l0[ib], target[ib] + l1[ib]
            q0[rb, xa[ib]] += sx
            q1[rb, xa[ib]] += sx
            pb = bar_containing(0.5 * (q0 + q1), geom)
            hit = pb >= 0
            pt[ib[hit]], tb[ib[hit]], bad[ib[hit]] = True, pb[hit], False
            Q0[ib[hit]], Q1[ib[hit]] = q0[hit], q1[hit]
            why["point"] = int(hit.sum())
        tb[bad] = nearest_bar(target[bad], xa[bad], geom)
        ok = tb >= 0
        tbs = np.where(ok, tb, 0)
        la = geom.local_to_world[tbs, 2]
        rows_o = np.arange(ok.size)
        shift = (entry[other] - geom.center[xbar[other]])[rows_o, la] - elen
        o0 = geom.center[tbs] + l0
        o1 = geom.center[tbs] + l1
        o0[rows_o, la] += shift
        o1[rows_o, la] += shift
        if pt.any():
            o0[pt], o1[pt] = Q0[pt], Q1[pt]
        lo_, hi_ = geom.center[tbs] - geom.halflen[tbs], geom.center[tbs] + geom.halflen[tbs]
        inside = (0.5 * (o0 + o1) >= lo_).all(1) & (0.5 * (o0 + o1) <= hi_).all(1)
        why["off_end"] = int((ok & ~inside).sum())  # moved back inside by fit_inside, not dropped
        r0[other], r1[other] = fit_inside(o0, o1, lo_, hi_)
        bar[other] = np.where(ok, tb, -1)
        mom[other] = mo
    clamp = same & (bar != xbar)
    if clamp.any():
        xc = xbar[clamp]
        c, h = geom.center[xc], geom.halflen[xc]
        rc = np.arange(len(xc))
        for r, cols in ((r0, ["lx0", "ly0", "lz0"]), (r1, ["lx1", "ly1", "lz1"])):
            loc = r[clamp] - c
            g4 = dd.loc[:, cols].to_numpy()[clamp]
            for k in (0, 1):  # thickness and width axes from G4; length axis keeps the rotated placement
                ax = geom.local_to_world[xc, k]
                loc[rc, ax] = g4[rc, ax]
            r[clamp] = c + loc
        r0[clamp], r1[clamp] = fit_inside(r0[clamp], r1[clamp], c - h, c + h)
        bar[clamp] = xbar[clamp]
    # a same-bar step whose midpoint is inside but whose rotated endpoints leave the bar: endpoints into the bar
    # (G4 steps never extend outside their volume; ~0.2 % of fast steps started > 1 mm outside their bar)
    sb = same & (bar == xbar) if contain else np.zeros(len(bar), bool)
    if sb.any():
        c, h = geom.center[xbar[sb]], geom.halflen[xbar[sb]]
        out_ = ((np.abs(r0[sb] - c) > h) | (np.abs(r1[sb] - c) > h)).any(axis=1)
        if out_.any():
            idx = np.flatnonzero(sb)[out_]
            r0[idx], r1[idx] = fit_inside(r0[idx], r1[idx], c[out_] - h[out_], c[out_] + h[out_])
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
    return out, int((~keep).sum()), int(clamp.sum()), why


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


def sample(fast_steps, fast_parts, gen, lib, geom, rng, showers=True, context=False, contain=False, point=False,
           mirror=False, translate=False):
    """Fast tier-1a tables + tier 1b: (steps, particles, info). Muons keep their keys (1, or 1..n with all_muons);
    descendants follow them; showers k*1e7+sp. showers=False: descendants only (run configurations whose bars are not
    the library's: the showers carry absolute extracted v04 bar positions)."""
    fx = fast_steps.copy()
    desc, n_drop, n_clamp, why = _desc_for_crossings(fx, lib, geom, rng, context, contain, point, mirror, translate)
    # descendant particles: one per (event, crossing, library particle), keys after the event's muon keys (2.. with one
    # muon), parent = the muon of the crossing (tier-1a `sp`)
    nmu = fast_parts.groupby(KEY, sort=False).key.max().rename("kmax").reset_index() if len(fast_parts) else None
    u = desc[KEY + ["xi", "lsp", "pdg"]].drop_duplicates(KEY + ["xi", "lsp"]).reset_index(drop=True)
    base = (u[KEY].merge(nmu, on=KEY, how="left").kmax.fillna(1).to_numpy().astype(np.int64) if nmu is not None
            else np.ones(len(u), np.int64))
    u["key"] = u.groupby(KEY).cumcount().to_numpy() + 1 + base
    u["parent"] = fx.sp.to_numpy()[u.xi.to_numpy()] if "sp" in fx.columns else 1
    desc = desc.merge(u[KEY + ["xi", "lsp", "key"]], on=KEY + ["xi", "lsp"])
    desc["sp"] = desc.key
    dparts = pd.DataFrame({k: u[k].to_numpy() for k in KEY})
    dparts["key"], dparts["parent"], dparts["pdg"], dparts["isprim"] = u.key, u.parent, u.pdg, 0
    sh = _showers_for_gen(gen, lib, rng) if showers else _showers_for_gen(gen.iloc[:0], lib, rng)
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
    info = {"desc_steps": len(desc), "desc_dropped": n_drop, "desc_clamped": n_clamp, "other_bar": why,
            "showers": int(len(roots)), "shower_steps": len(sh), "nonmu_primaries": int(len(gsign))}
    return steps, parts, info
