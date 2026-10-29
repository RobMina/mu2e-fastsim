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
   - topology: an other-bar step keeps its layer topology relative to its own library muon. A muon's crossings in time
     order form visits of layer planes (consecutive crossings of one plane: the modules of a sector, and the side-by-
     side M sectors, share each plane). desc_topology stores, per other-bar row, the visit of its plane relative to the
     anchor crossing's, its bar's width-rank offset beyond the bars crossed in that visit (signed by the muon's lean
     along the width), and its in-bar and along-bar offsets from that visit's entry; the fast step goes to the same
     visit and offset around the receiving muon, so never into a bar that muon crosses there. Needs the per-crossing
     bar, muon and visit of the library (attach_topology; a sidecar rebuilt from the library's donors). Rows whose
     plane the library muon never visited, or whose visit or bar the receiving muon lacks, keep the other placements.
   Opt-in position-conditioned templates (sample(unvisited, nearmiss); a template sidecar, attach_templates):
   - unvisited: an other-bar descendant in a layer plane its own library muon never visits (mostly the decay electron
     of a muon that stops after its last crossing, and deltas into a sector stacked beside or below the crossing) is
     placed by where the crossing is, not by its offsets: the main draw leaves these rows out, and each fast crossing
     draws a second library crossing from its position cell (layer plane, width-rank and along-bar cell of the entry,
     the groups its muon crosses, last crossing of its muon or not, |p| class; coarser cells when a cell holds fewer
     than MIN_UV_CROSSINGS) and
     takes that crossing's unvisited rows, translated by the entry difference when the translated step lies in a bar
     of the same layer plane that the receiving muon does not visit, else at their library position.
   - nearmiss: a primary muon with no CRV crossing in its event deposits nothing in tier 1a; G4 muons that pass beside
     the CRV or stop above it do (decay electrons, deltas). Each such fast muon reaches the CRV with the probability
     of its cell (kinetic energy, the library group whose box its straight generated line comes closest to, and that
     distance), templates of the cell over the no-crossing muons of a generator census in the cell, and then gets a
     random template of the cell at its absolute positions, times relative to the muon's generation time.
   Same-bar options (sample(chord, follow); a template sidecar with the library chords, attach_templates; the default
   asset set strict_v3 turns chord on, follow stays opt-in):
   - chord: a same-bar step is placed at its distance along the library muon's direction from the entry by default, so
     when the library chord within the bin is shorter (longer) than the fast one the steps crowd toward the entry
     face (overshoot the exit and fall back to their library position across the bar). With chord the component
     along the library chord keeps its fraction of the chord (outside it, its distance from the nearer end) on the
     fast chord, and the rest is rotated chord to chord.
   - follow: G4 own-bar counts of one muon's crossings are correlated (the EM-cascade products of a hard delta reach
     the next bars the muon crosses), while independent draws per crossing are not. In time order along each fast
     muon, a crossing takes the next crossing of the previous pick's library muon when that crossing lies in its own
     draw cell, else its own uniform draw (the same random numbers in both cases).
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


TOPO_NONE = -128   # no layer topology: the descendant's sector layer is not one its library muon visited
TOPO_COLS = ("tvis", "tside", "tt0", "tt1", "tw0", "tw1", "tl0", "tl1")


def layer_slots(geom):
    """Per bar: its layer-plane id (library group * 100 + layer: the modules of a sector, and the side-by-side M
    sectors, share each layer plane; sector * 100 + layer outside the groups) and its rank along the width axis
    within that layer plane."""
    g = _group_index(geom, np.arange(geom.nbars))
    lid = np.where(g >= 0, 1000 + g, geom.sector.astype(np.int64)) * 100 + geom.layer.astype(np.int64)
    w = geom.center[np.arange(geom.nbars), geom.local_to_world[:, 1]]
    widx = pd.Series(w).groupby(lid).rank(method="dense").to_numpy().astype(np.int64) - 1
    return lid, widx


def _slot_lookup(geom):
    lid, widx = layer_slots(geom)
    code = lid * 10_000 + widx
    order = np.argsort(code, kind="stable")
    return code[order], order


def slot_bar(geom, lid, widx):
    """Bar at (sector layer, width rank), -1 where that layer has no such bar."""
    code, order = _slot_lookup(geom)
    want = np.asarray(lid, np.int64) * 10_000 + np.asarray(widx, np.int64)
    i = np.minimum(np.searchsorted(code, want), len(code) - 1)
    w = np.asarray(widx, np.int64)
    ok = (code[i] == want) & (w >= 0) & (w < 10_000)
    return np.where(ok, order[i], -1)


def muon_visits(x, geom):
    """Sector-layer visits of each primary muon (rows of x: KEY, sp = muon key, bar, t0, x0, y0, z0 = entry): its
    crossings ordered in time, consecutive crossings of one sector layer forming one visit. Returns (mu, vis) per
    crossing (muon id in x, visit index along that muon) and the visits (mu, vis, lid, wmin, wmax: width ranks of the
    bars crossed in it, ex, ey, ez: entry of its first crossing), sorted by (mu, vis)."""
    n = len(x)
    cols = ["mu", "vis", "lid", "wmin", "wmax", "ex", "ey", "ez"]
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), pd.DataFrame({c: np.zeros(0) for c in cols})
    lid, widx = layer_slots(geom)
    bar = x.bar.to_numpy().astype(np.int64)
    mu = x.groupby(KEY + ["sp"], sort=False).ngroup().to_numpy().astype(np.int64)
    order = np.lexsort((x.t0.to_numpy(), mu))
    m_s, l_s = mu[order], lid[bar[order]]
    first_mu = np.r_[True, m_s[1:] != m_s[:-1]]
    new = first_mu | np.r_[True, l_s[1:] != l_s[:-1]]
    vg = np.cumsum(new) - 1
    vis_s = vg - np.maximum.accumulate(np.where(first_mu, vg, 0))
    vis = np.empty(n, np.int64)
    vis[order] = vis_s
    e = x[["x0", "y0", "z0"]].to_numpy()[order]
    v = pd.DataFrame({"g": vg, "mu": m_s, "vis": vis_s, "lid": l_s, "w": widx[bar[order]],
                      "ex": e[:, 0], "ey": e[:, 1], "ez": e[:, 2]})
    a = v.groupby("g", sort=True).agg(mu=("mu", "first"), vis=("vis", "first"), lid=("lid", "first"),
                                      wmin=("w", "min"), wmax=("w", "max"), ex=("ex", "first"), ey=("ey", "first"),
                                      ez=("ez", "first")).reset_index(drop=True)
    return mu, vis, a[cols]


def _visit_of(visits, mu, vis):
    """Row of visits holding (mu, vis), -1 if none (visits sorted by mu, vis)."""
    key = visits.mu.to_numpy().astype(np.int64) * 100_000 + visits.vis.to_numpy().astype(np.int64)
    want = np.asarray(mu, np.int64) * 100_000 + np.asarray(vis, np.int64)
    if not len(key):
        return np.full(len(want), -1, np.int64)
    i = np.minimum(np.searchsorted(key, want), len(key) - 1)
    return np.where((key[i] == want) & (np.asarray(vis) >= 0), i, -1)


def _lean(u, axis):
    """Sign of each direction's component along its axis (+1 when zero)."""
    s = np.sign(u[np.arange(len(axis)), axis])
    return np.where(s == 0, 1.0, s)


def desc_topology(ld, lx, visits, geom):
    """Layer topology of each other-bar library descendant relative to the visits of its own library muon (lx: xid,
    bar, mu, vis, ux, uy, uz per library crossing, xid = row; visits from muon_visits on the library crossings).
    tvis: visit index of the descendant's sector layer minus the anchor crossing's; tside: its bar's width-rank offset
    beyond the bars crossed in that visit times the muon's lean along the width axis; tt/tw: in-bar thickness and width
    coordinates (width times the lean); tl: along-bar offset from the visit entry times the muon's lean along the bar.
    TOPO_NONE (tvis) for same-bar rows and where the sector layer was not visited. Returns a frame aligned with ld."""
    out = pd.DataFrame({c: np.zeros(len(ld)) for c in TOPO_COLS})
    out["tvis"] = TOPO_NONE
    other = np.flatnonzero(ld.samebar.to_numpy() == 0)
    if not len(other):
        return out
    xid = ld.xid.to_numpy()[other]
    xb = lx.bar.to_numpy().astype(np.int64)[xid]
    tgt = geom.center[xb] + ld[["dcx", "dcy", "dcz"]].to_numpy()[other]
    db = bar_containing(tgt, geom)
    if (db < 0).any() or not np.allclose(tgt, geom.center[np.maximum(db, 0)], rtol=0, atol=1e-6):
        raise ValueError("library other-bar offset is not an exact bar centre offset")
    lid, widx = layer_slots(geom)
    mu, va = lx.mu.to_numpy()[xid], lx.vis.to_numpy()[xid]
    key = visits.mu.to_numpy().astype(np.int64) * 10_000 + visits.lid.to_numpy()
    order = np.lexsort((visits.vis.to_numpy(), key))
    ks = key[order]
    want = mu.astype(np.int64) * 10_000 + lid[db]
    lo, hi = np.searchsorted(ks, want, "left"), np.searchsorted(ks, want, "right")
    found = hi > lo
    a, b = order[np.minimum(lo, len(order) - 1)], order[np.maximum(hi - 1, 0)]
    vv = visits.vis.to_numpy()
    pick = np.where(np.abs(vv[b] - va) < np.abs(vv[a] - va), b, a)   # the visit nearest the anchor
    w = widx[db]
    wmin, wmax = visits.wmin.to_numpy()[pick], visits.wmax.to_numpy()[pick]
    k = np.where(w > wmax, w - wmax, np.where(w < wmin, w - wmin, 0))
    ok = found & (k != 0)
    ax = geom.local_to_world[db]
    r = np.arange(len(db))
    u = lx[["ux", "uy", "uz"]].to_numpy()[xid]
    sw, sl = _lean(u, ax[:, 1]), _lean(u, ax[:, 2])
    e = visits[["ex", "ey", "ez"]].to_numpy()[pick]
    rows = other[ok]
    out.loc[rows, "tvis"] = (vv[pick] - va)[ok]
    out.loc[rows, "tside"] = (k * sw)[ok]
    for end in "01":
        loc = ld[[f"lx{end}", "ly" + end, "lz" + end]].to_numpy()[other]
        out.loc[rows, "tt" + end] = loc[r, ax[:, 0]][ok]
        out.loc[rows, "tw" + end] = (loc[r, ax[:, 1]] * sw)[ok]
        out.loc[rows, "tl" + end] = ((geom.center[db][r, ax[:, 2]] + loc[r, ax[:, 2]] - e[r, ax[:, 2]]) * sl)[ok]
    out["tvis"] = out.tvis.astype(np.int64)
    return out


def attach_topology(lib, crossings, visits, geom):
    """Add the desc_topology columns to lib['libdesc'] (in memory) from per-library-crossing (xid, bar, mu, vis) and the
    library muons' visits (a topology sidecar); the library trees themselves are not changed."""
    lx = lib["libxing"].drop(columns=[c for c in ("bar", "mu", "vis") if c in lib["libxing"].columns])
    c = crossings.sort_values("xid", kind="stable")
    if not np.array_equal(c.xid.to_numpy(), lx.xid.to_numpy()) or not np.array_equal(lx.xid, np.arange(len(lx))):
        raise ValueError("topology sidecar crossings do not match the library crossings")
    lx = lx.assign(bar=c.bar.to_numpy(), mu=c.mu.to_numpy(), vis=c.vis.to_numpy())
    vs = visits.sort_values(["mu", "vis"], kind="stable").reset_index(drop=True)
    topo = desc_topology(lib["libdesc"], lx, vs, geom)
    ld = lib["libdesc"].drop(columns=[c for c in TOPO_COLS if c in lib["libdesc"].columns])
    lib = dict(lib)
    lib["libdesc"] = pd.concat([ld.reset_index(drop=True), topo], axis=1)
    return lib


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
def _topology_targets(fx, dd, ix, fu, geom):
    """Target bars and endpoints of other-bar library rows dd (desc_topology columns) placed by their layer topology
    on the fast crossings ix: the visit tvis after the anchor's along the receiving muon, the bar tside ranks beyond
    the bars it crosses there (lean-signed), in-bar thickness/width and the along-bar offset from that visit's entry.
    Rows without topology, or whose visit or bar does not exist for the receiving muon, are not hit."""
    n = len(dd)
    q0, q1 = np.zeros((n, 3)), np.zeros((n, 3))
    tv = dd.tvis.to_numpy().astype(np.int64)
    mu, vis, visits = muon_visits(fx, geom)
    has = tv != TOPO_NONE
    row = np.full(n, -1, np.int64)
    if has.any():
        row[has] = _visit_of(visits, mu[ix[has]], vis[ix[has]] + tv[has])
    hit = row >= 0
    vr = np.maximum(row, 0)
    lid = visits.lid.to_numpy().astype(np.int64)[vr]
    wmin, wmax = visits.wmin.to_numpy().astype(np.int64)[vr], visits.wmax.to_numpy().astype(np.int64)[vr]
    ref = slot_bar(geom, lid, wmin)                     # a bar of that layer: its axes
    ax = geom.local_to_world[np.maximum(ref, 0)]
    sw, sl = _lean(fu, ax[:, 1]), _lean(fu, ax[:, 2])
    k = np.rint(dd.tside.to_numpy() * sw).astype(np.int64)
    tb = slot_bar(geom, lid, np.where(k > 0, wmax + k, wmin + k))
    hit &= (tb >= 0) & (ref >= 0)
    tbs = np.maximum(tb, 0)
    r = np.arange(n)
    e = visits[["ex", "ey", "ez"]].to_numpy()[vr]
    for q, end in ((q0, "0"), (q1, "1")):
        q[:] = geom.center[tbs]
        q[r, ax[:, 0]] += dd["tt" + end].to_numpy()
        q[r, ax[:, 1]] += dd["tw" + end].to_numpy() * sw
        q[r, ax[:, 2]] = e[r, ax[:, 2]] + dd["tl" + end].to_numpy() * sl
    return hit, tb, q0, q1


def _chord_map(d0, d1, mom, cl, cf, ok):
    """Same-bar offsets d0/d1 (from the library entry) and momenta moved from the library crossing's chord cl (entry to
    exit) onto the fast one cf, for rows ok: the component along the chord keeps its fraction of the chord inside it and
    its distance from the nearer end outside; the perpendicular part, the step vector and the momentum are rotated
    from chord to chord. Returns (offset of the start, offset of the end, momentum) from the fast entry."""
    ll, lf = np.linalg.norm(cl, axis=1), np.linalg.norm(cf, axis=1)
    ok = ok & (ll > 1e-6) & (lf > 1e-6)
    ul, uf = cl / np.maximum(ll, 1e-12)[:, None], cf / np.maximum(lf, 1e-12)[:, None]
    a = (d0 * ul).sum(1)
    perp = d0 - a[:, None] * ul
    an = np.where(a < 0, a, np.where(a > ll, lf + a - ll, a * lf / np.maximum(ll, 1e-12)))
    s0 = an[:, None] * uf + rotate(perp, ul, uf)
    s1 = s0 + rotate(d1 - d0, ul, uf)
    return ok, s0, s1, rotate(mom, ul, uf)


def _follow_picks(fx, lx_xid, lo, hi, pick, nxt, pos):
    """Chain draw: in time order along each fast muon, a crossing takes the next library crossing (in time) of the
    library muon of the previous crossing's pick when that crossing lies in its own draw cell [lo, hi) of the sorted
    library crossings (pos = position of each xid there); otherwise it keeps its own uniform pick. Returns the picks
    and how many crossings followed."""
    pick = pick.copy()
    n = len(fx)
    if n == 0:
        return pick, 0
    mu = fx.groupby(KEY + (["sp"] if "sp" in fx.columns else []), sort=False).ngroup().to_numpy()
    order = np.lexsort((fx.t0.to_numpy(), mu))
    first = np.r_[True, mu[order][1:] != mu[order][:-1]]
    rank = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))
    followed = 0
    for k in range(1, int(rank.max()) + 1 if n else 0):
        cur = order[rank == k]
        prev = order[np.flatnonzero(rank == k) - 1]
        cand = nxt[pick[prev]]
        p = np.where(cand >= 0, pos[np.maximum(cand, 0)], -1)
        ok = (cand >= 0) & (p >= lo[cur]) & (p < hi[cur])
        pick[cur[ok]] = cand[ok]
        followed += int(ok.sum())
    return pick, followed


def _desc_for_crossings(fx, lib, geom, rng, context=False, contain=False, point=False, mirror=False,
                        translate=False, topology=False, trace=None, drop_unvisited=False, chord=False, follow=False):
    """Descendant steps for the fast crossings fx (one row per crossing). Returns (steps, n_dropped, n_clamped, why).
    Opt-in (all off reproduces the earlier sampling bit for bit): context = draw within the crossing-context cell
    (library with a ctx column); contain = same-bar endpoints into the bar; point = other-bar steps whose target bar
    is missing or along another axis go to the bar containing their library position relative to the fast crossing;
    mirror = other-bar offsets reflected across the crossing bar's width (length) plane where the library and fast muons
    point opposite ways along that axis; translate = other-bar steps go to the bar containing their library offset from
    the muon entry (unrotated) added to the fast entry, the bar-structure placement remaining the fallback;
    topology = other-bar steps keep their library layer topology around the receiving muon's own sector-layer visits
    (library with the desc_topology columns, attach_topology), the other placements remaining the fallback.
    An optional trace dict records pre-placement donor labels and final bars without consuming random draws.
    drop_unvisited leaves out the rows in layer planes the library muon never visits (library with the uv column,
    attach_templates); sample(unvisited=True) places those rows by position instead. No random draw changes.
    chord (library chords from a template sidecar with them, attach_templates): same-bar rows are placed along the
    fast crossing's chord at their fraction of the library chord (_chord_map) instead of at their distance along the
    library muon's direction. follow (template sidecar with the next crossing of each library crossing): chain draw
    along each fast muon (_follow_picks). Both consume no other random numbers."""
    p = np.sqrt(fx.px0 ** 2 + fx.py0 ** 2 + fx.pz0 ** 2).to_numpy()
    b = _bin(lib["libbins"], _group_index(geom, fx.bar), p, fx.len.to_numpy())
    lx = lib["libxing"]
    if context and "ctx" not in lx.columns:
        raise ValueError("crossing context needs a library built with the ctx column (tier1b-build after this change)")
    if topology and "tvis" not in lib["libdesc"].columns:
        raise ValueError("topology placement needs the desc_topology columns (attach_topology with a sidecar)")
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
    if (chord or follow) and "xgeo" not in lib:
        raise ValueError("chord / follow need a template sidecar with the library chords (attach_templates)")
    if follow:
        if "nxt" not in lib["xgeo"]:
            raise ValueError("follow needs a template sidecar with the next crossing of each library crossing")
        pos = np.empty(len(lx), np.int64)
        pos[lx.xid.to_numpy()] = np.arange(len(lx))
        pick, nfollow = _follow_picks(fx, lx.xid.to_numpy(), lo, hi, pick, lib["xgeo"]["nxt"], pos)
    ld = lib["libdesc"].sort_values("xid", kind="stable")
    lxid = ld.xid.to_numpy()
    a, z = np.searchsorted(lxid, pick, "left"), np.searchsorted(lxid, pick, "right")
    cnt = z - a
    ix = np.repeat(np.arange(len(fx)), cnt)                        # fast crossing of each new step
    rows = np.repeat(a - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    if drop_unvisited:
        if "uv" not in ld.columns:
            raise ValueError("unvisited-layer templates need the uv column (attach_templates with a template sidecar)")
        keep_row = ~ld.uv.to_numpy()[rows]
        ix, rows = ix[keep_row], rows[keep_row]
    dd = ld.iloc[rows]
    lu = lib["libxing"].set_index("xid").loc[pick[ix], ["ux", "uy", "uz"]].to_numpy()
    fu = _unit(fx[["px0", "py0", "pz0"]].to_numpy())[ix]
    entry = fx[["x0", "y0", "z0"]].to_numpy()[ix]
    r0 = entry + rotate(dd[["dx0", "dy0", "dz0"]].to_numpy(), lu, fu)
    r1 = entry + rotate(dd[["dx1", "dy1", "dz1"]].to_numpy(), lu, fu)
    mom = rotate(dd[["px0", "py0", "pz0"]].to_numpy(), lu, fu)
    same = dd.samebar.to_numpy() == 1
    nchord = 0
    if chord:   # same-bar rows along the fast chord at their fraction of the library chord
        g = lib["xgeo"]
        cl = (g["exit"] - g["entry"])[pick[ix]]
        cf = (fx[["x1", "y1", "z1"]].to_numpy() - fx[["x0", "y0", "z0"]].to_numpy())[ix]
        ok, s0, s1, mo = _chord_map(dd[["dx0", "dy0", "dz0"]].to_numpy(), dd[["dx1", "dy1", "dz1"]].to_numpy(),
                                    dd[["px0", "py0", "pz0"]].to_numpy(), cl, cf, same)
        r0[ok], r1[ok], mom[ok] = entry[ok] + s0[ok], entry[ok] + s1[ok], mo[ok]
        nchord = int(ok.sum())
    bar = bar_containing(0.5 * (r0 + r1), geom)
    xbar = fx.bar.to_numpy()[ix]
    other = ~same
    why = {"other": int(other.sum()), "no_bar": 0, "axis": 0, "off_end": 0}
    if chord:
        why["chord"] = nchord
    if follow:
        why["follow"] = nfollow
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
        if topology:   # the library layer topology around the receiving muon's own visits (sector layer, side, along)
            hit, tq, q0, q1 = _topology_targets(fx, dd.iloc[np.flatnonzero(other)], ix[other], fu[other], geom)
            pt[hit], tb[hit], bad[hit] = True, tq[hit], False
            Q0[hit], Q1[hit] = q0[hit], q1[hit]
            why["topology"] = int(hit.sum())
            if trace is not None:
                th = np.zeros(len(dd), bool)
                th[np.flatnonzero(other)[hit]] = True
                trace["topology_hit"] = th
                trace["source_tvis"] = dd.tvis.to_numpy().copy()
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
    if trace is not None:
        trace.update(xi=ix.copy(), xid=pick[ix].copy(), source_samebar=same.copy(),
                     source_edep=dd.edep.to_numpy().copy(), target_bar=bar.copy(), kept=keep.copy())
        if "uv" in dd.columns:
            trace["source_uv"] = dd.uv.to_numpy().copy()
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


# ------------------------------------------------------------------------------- opt-in position-conditioned templates
UV_P_EDGES = np.array([150.0, 300.0, 1500.0])   # crossing |p| classes [MeV/c]: the lowest holds the muons that stop
UV_W_BARS = (8, 16)                              # width-rank size of the fine and the coarse position cell
UV_A_MM = (400.0, 800.0)                         # along-bar size [mm] of the fine and the coarse position cell
N_UV_LEVELS = 5                                  # fine cell, coarse cell, layer plane (each with the visit mask),
#                                                  layer plane, library group
MIN_UV_CROSSINGS = 20                            # fewer library crossings in a cell: the next coarser level
NM_KE_EDGES = np.array([300.0, 1000.0, 3000.0, 10000.0])   # near-miss muon kinetic energy classes [MeV]
NM_D_EDGES = np.array([100.0, 300.0, 1000.0, 3000.0])      # closest-approach distance classes [mm], after "enters"
NM_GOLDEN_ITER = 60
UV_STEP_COLS = ["x0", "y0", "z0", "x1", "y1", "z1"]


def last_crossing(x):
    """Per crossing (rows of x: KEY, sp, t0), whether it is the last crossing in time of its muon."""
    if not len(x):
        return np.zeros(0, bool)
    by = KEY + (["sp"] if "sp" in x.columns else [])
    return (x.t0 == x.groupby(by, sort=False).t0.transform("max")).to_numpy()


def visit_mask(x, geom):
    """Per crossing (rows of x: KEY, sp, bar), the library groups its muon crosses anywhere, as a bit mask."""
    if not len(x):
        return np.zeros(0, np.int64)
    by = KEY + (["sp"] if "sp" in x.columns else [])
    return group_mask(x.groupby(by, sort=False).ngroup().to_numpy(), x.bar.to_numpy(), geom)


def group_mask(mu, bar, geom):
    """Bit mask of the library groups of the bars sharing each muon id mu (one bit per group)."""
    bit = np.left_shift(1, np.maximum(_group_index(geom, np.asarray(bar, np.int64)), 0)).astype(np.int64)
    d = pd.DataFrame({"mu": np.asarray(mu, np.int64), "bit": bit})
    m = d.drop_duplicates().groupby("mu").bit.sum()
    return m.reindex(d.mu.to_numpy()).to_numpy().astype(np.int64)


def uv_cells(bar, entry, p, last, mask, geom):
    """Position cell codes of crossings for the unvisited-layer templates, finest first (columns = N_UV_LEVELS levels):
    (layer plane, width rank // UV_W_BARS, along-bar entry // UV_A_MM) at two sizes and the layer plane, each with the
    groups the muon crosses (visit_mask: a T2 deposit is unvisited only for a muon that misses T2), then the layer
    plane and the library group without it; every level with the last-crossing flag and the |p| class (UV_P_EDGES)."""
    bar = np.asarray(bar, np.int64)
    mask = np.asarray(mask, np.int64)
    lid, widx = layer_slots(geom)
    la = geom.local_to_world[bar, 2]
    along = np.asarray(entry, float)[np.arange(len(bar)), la]
    pc = np.searchsorted(UV_P_EDGES, np.asarray(p, float), "right")
    tail = np.asarray(last, np.int64) * (len(UV_P_EDGES) + 1) + pc
    nt = 2 * (len(UV_P_EDGES) + 1)
    out = np.zeros((len(bar), N_UV_LEVELS), np.int64)
    for j, (wb, am) in enumerate(zip(UV_W_BARS, UV_A_MM)):
        cell = (lid[bar] * 1000 + widx[bar] // wb) * 100_000 + np.floor(along / am).astype(np.int64) + 50_000
        out[:, j] = (cell * 16 + mask) * nt + tail
    out[:, 2] = (lid[bar] * 16 + mask) * nt + tail
    out[:, 3] = lid[bar] * nt + tail
    out[:, 4] = (_group_index(geom, bar) + 1) * nt + tail
    return out


def group_boxes(geom):
    """Axis-aligned box (lo [4, 3], hi [4, 3]) of every library group's bars."""
    g = _group_index(geom, np.arange(geom.nbars))
    lo, hi = geom.center - geom.halflen, geom.center + geom.halflen
    return (np.array([lo[g == i].min(0) for i in range(len(GROUPS))]),
            np.array([hi[g == i].max(0) for i in range(len(GROUPS))]))


def closest_approach(pos, u, geom):
    """For each straight ray pos + t u (t >= 0): the library group whose box it comes closest to, that distance (0 when
    it enters the box) and t there. Ties at 0 go to the box entered first. Exact slab test for entering; golden-section
    search of the convex distance otherwise (t between the box corners' projections)."""
    pos, u = np.asarray(pos, float), _unit(np.asarray(u, float))
    lo, hi = group_boxes(geom)
    n, ng = len(pos), len(GROUPS)
    dist, tbest = np.zeros((n, ng)), np.zeros((n, ng))
    with np.errstate(divide="ignore", invalid="ignore"):
        for gi in range(ng):
            t1, t2 = (lo[gi] - pos) / u, (hi[gi] - pos) / u
            par = u == 0
            inside_slab = (pos >= lo[gi]) & (pos <= hi[gi])
            tn = np.where(par, np.where(inside_slab, -np.inf, np.inf), np.minimum(t1, t2)).max(1)
            tf = np.where(par, np.where(inside_slab, np.inf, -np.inf), np.maximum(t1, t2)).min(1)
            enters = (tn <= tf) & (tf >= 0)
            corners = np.array([[a, b, c] for a in (lo[gi, 0], hi[gi, 0]) for b in (lo[gi, 1], hi[gi, 1])
                                for c in (lo[gi, 2], hi[gi, 2])])
            proj = np.einsum("nkj,nj->nk", corners[None, :, :] - pos[:, None, :], u)
            a, b = np.maximum(proj.min(1), 0.0), np.maximum(proj.max(1), 0.0)
            f = lambda t: np.linalg.norm(np.maximum(np.maximum(lo[gi] - (pos + t[:, None] * u), 0.0),
                                                    (pos + t[:, None] * u) - hi[gi]), axis=1)
            r = (np.sqrt(5.0) - 1) / 2
            c, d = b - r * (b - a), a + r * (b - a)
            fc, fd = f(c), f(d)
            for _ in range(NM_GOLDEN_ITER):
                left = fc < fd
                b = np.where(left, d, b)
                a = np.where(left, a, c)
                c_new, d_new = b - r * (b - a), a + r * (b - a)
                fc, fd = np.where(left, f(c_new), fd), np.where(left, fc, f(d_new))
                # keep the reused end point exact: the surviving interior point becomes the new d (left) or c (right)
                c, d = np.where(left, c_new, d), np.where(left, c, d_new)
            tm = 0.5 * (a + b)
            dist[:, gi] = np.where(enters, 0.0, f(tm))
            tbest[:, gi] = np.where(enters, np.maximum(tn, 0.0), tm)
    key = np.where(dist == 0, tbest, np.inf)
    first = np.argmin(key, axis=1)
    g = np.where((dist == 0).any(1), first, np.argmin(dist, axis=1))
    r = np.arange(n)
    return g, dist[r, g], tbest[r, g]


NM_KE_GROUPS = (0,)   # groups whose near-miss cells are split by kinetic energy (EX); the others are too sparse


def nearmiss_cell(ke, group, dist):
    """Near-miss cell code: (group, distance class: 0 enters the box, then NM_D_EDGES, kinetic-energy class; the
    kinetic-energy class only for NM_KE_GROUPS: near misses of the other groups are rare in a 500k-event census)."""
    group = np.asarray(group, np.int64)
    db = np.where(np.asarray(dist) <= 0, 0, 1 + np.searchsorted(NM_D_EDGES, dist, "right"))
    kb = np.where(np.isin(group, NM_KE_GROUPS), np.searchsorted(NM_KE_EDGES, ke, "right"), 0)
    return (group * 10 + db) * 10 + kb


def muon_rays(gen):
    """Every generated muon (the tier-1a order: event, gen index; key = 1, 2, ... within the event) with its ray."""
    mu = gen[gen.pdg.abs() == tables.PRIMARY_MUON_PDG]
    order = [c for c in KEY + ["idx"] if c in mu.columns]
    mu = mu.sort_values(order, kind="stable").reset_index(drop=True)
    mu["mukey"] = mu.groupby(KEY, sort=False).cumcount().to_numpy() + 1
    mu["ke"] = mu.e.to_numpy() - MASS[13]
    return mu


def attach_templates(lib, side, geom):
    """Add the opt-in template components to lib (in memory) from a template sidecar (dict of frames + meta, see
    valhist.descendant_templates): the uv column of libdesc, the unvisited-layer rows at absolute positions (uvdesc),
    the library crossings' position cells (uvxing), the near-miss templates and their reach table. The library trees
    themselves are not changed."""
    lx = lib["libxing"]
    xs = side["xing"].sort_values("xid", kind="stable").reset_index(drop=True)
    if not np.array_equal(xs.xid.to_numpy(), lx.xid.to_numpy()):
        raise ValueError("template sidecar crossings do not match the library crossings")
    ld = lib["libdesc"].drop(columns=[c for c in ("uv",) if c in lib["libdesc"].columns])
    uvr = side["uv"].row.to_numpy().astype(np.int64)
    if len(uvr) and (uvr.max() >= len(ld) or (ld.samebar.to_numpy()[uvr] != 0).any()):
        raise ValueError("template sidecar unvisited rows are not other-bar library rows")
    flag = np.zeros(len(ld), bool)
    flag[uvr] = True
    lib = dict(lib)
    lib["libdesc"] = ld.assign(uv=flag)
    d = ld.iloc[uvr]
    ub = side["uv"].bar.to_numpy().astype(np.int64)
    uvd = pd.DataFrame({"xid": d.xid.to_numpy(), "sp": d.sp.to_numpy(), "pdg": d.pdg.to_numpy(), "bar": ub,
                        "dt0": d.dt0.to_numpy(), "dt1": d.dt1.to_numpy()})
    for j, c in enumerate("xyz"):
        for e in "01":
            uvd[f"{c}{e}"] = geom.center[ub, j] + d[f"l{c}{e}"].to_numpy()
    for c in ("px0", "py0", "pz0", "p1", "len", "edep"):
        uvd[c] = d[c].to_numpy()
    lib["uvdesc"] = uvd.sort_values("xid", kind="stable").reset_index(drop=True)
    mask = group_mask(xs.mu.to_numpy(), xs.bar.to_numpy(), geom)
    entry = xs[["ex", "ey", "ez"]].to_numpy()
    cells = uv_cells(xs.bar.to_numpy(), entry, xs.p.to_numpy(), xs.last.to_numpy().astype(bool), mask, geom)
    if {"qx", "qy", "qz"} <= set(xs.columns):   # template sidecar v3: library chords and next crossings (chord, follow)
        lib["xgeo"] = {"entry": entry, "exit": xs[["qx", "qy", "qz"]].to_numpy()}
        if "nxt" in xs.columns:
            lib["xgeo"]["nxt"] = xs.nxt.to_numpy().astype(np.int64)
    lib["uvxing"] = {"entry": entry, "levels": [(np.sort(cells[:, j], kind="stable"),
                                                 np.argsort(cells[:, j], kind="stable")) for j in range(N_UV_LEVELS)]}
    t = side["nmt"]
    tc = nearmiss_cell(t.ke.to_numpy(), t.group.to_numpy(), t.dist.to_numpy())
    lib["nmtmpl"] = pd.DataFrame({"tid": t.tid.to_numpy().astype(np.int64), "cell": tc}).sort_values(
        ["cell", "tid"], kind="stable").reset_index(drop=True)
    lib["nmsteps"] = side["nms"].sort_values(["tid"], kind="stable").reset_index(drop=True)
    c = side["nmc"]
    cc = nearmiss_cell(c.ke.to_numpy(), c.group.to_numpy(), c.dist.to_numpy())
    scale = float(side["meta"]["generated"]) / float(side["meta"]["census_events"])
    den = pd.Series(cc).value_counts().rename("ncensus")
    num = pd.Series(tc).value_counts().rename("ntemp")
    r = pd.concat([den, num], axis=1).fillna(0).rename_axis("cell").reset_index()
    r["nden"] = r.ncensus * scale
    r["prob"] = np.where(r.nden > 0, np.minimum(r.ntemp / np.maximum(r.nden, 1e-300), 1.0), 0.0)
    lib["nmreach"] = r.astype({"cell": np.int64}).sort_values("cell").reset_index(drop=True)
    return lib


def _unvisited_for_crossings(fx, lib, geom, rng):
    """Unvisited-layer rows for the fast crossings fx, drawn from the library crossings of each crossing's position
    cell (uv_cells, the finest level with >= MIN_UV_CROSSINGS library crossings, else the coarsest non-empty one).
    One uniform draw per crossing. Returns (steps with xi = fast crossing, lsp = library particle; info)."""
    n = len(fx)
    entry = fx[["x0", "y0", "z0"]].to_numpy()
    p = np.sqrt(fx.px0 ** 2 + fx.py0 ** 2 + fx.pz0 ** 2).to_numpy()
    cells = uv_cells(fx.bar.to_numpy(), entry, p, last_crossing(fx), visit_mask(fx, geom), geom)
    u = rng.random(n)
    lo, hi = np.zeros(n, np.int64), np.zeros(n, np.int64)
    lev = np.full(n, -1, np.int64)
    pickx = np.full(n, -1, np.int64)
    for j, (codes, order) in reversed(list(enumerate(lib["uvxing"]["levels"]))):
        a, b = np.searchsorted(codes, cells[:, j], "left"), np.searchsorted(codes, cells[:, j], "right")
        use = ((b - a) >= MIN_UV_CROSSINGS) | ((lev < 0) & (b > a))
        lo[use], hi[use], lev[use] = a[use], b[use], j
        sel = lev == j
        pickx[sel] = order[lo[sel] + (u[sel] * (hi[sel] - lo[sel])).astype(np.int64)]
    uvd = lib["uvdesc"]
    ux = uvd.xid.to_numpy()
    a, z = np.searchsorted(ux, pickx, "left"), np.searchsorted(ux, pickx, "right")
    cnt = np.where(pickx >= 0, z - a, 0)
    ix = np.repeat(np.arange(n), cnt)
    rows = np.repeat(a - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    d = uvd.iloc[rows]
    lid, _ = layer_slots(geom)
    a0, a1 = d[["x0", "y0", "z0"]].to_numpy(), d[["x1", "y1", "z1"]].to_numpy()
    shift = entry[ix] - lib["uvxing"]["entry"][pickx[ix]]
    q0, q1 = a0 + shift, a1 + shift
    tb = bar_containing(0.5 * (q0 + q1), geom)
    lb = d.bar.to_numpy()
    mu, _, visits = muon_visits(fx, geom)
    vkey = np.unique(visits.mu.to_numpy().astype(np.int64) * 1_000_000 + visits.lid.to_numpy().astype(np.int64))
    tbs = np.maximum(tb, 0)
    ok = (tb >= 0) & (lid[tbs] == lid[lb]) & ~np.isin(mu[ix].astype(np.int64) * 1_000_000 + lid[tbs], vkey)
    bar = np.where(ok, tb, lb)
    r0, r1 = np.where(ok[:, None], q0, a0), np.where(ok[:, None], q1, a1)
    c, h = geom.center[bar], geom.halflen[bar]
    r0, r1 = fit_inside(r0, r1, c - h, c + h)
    out = pd.DataFrame({k: fx[k].to_numpy()[ix] for k in KEY})
    out["bar"], out["lsp"], out["xi"], out["pdg"] = bar, d.sp.to_numpy(), ix, d.pdg.to_numpy()
    t0 = fx.t0.to_numpy()[ix]
    out["t0"], out["t1"] = t0 + d.dt0.to_numpy(), t0 + d.dt1.to_numpy()
    for j, cc in enumerate("xyz"):
        out[f"{cc}0"], out[f"{cc}1"], out[f"p{cc}0"] = r0[:, j], r1[:, j], d[f"p{cc}0"].to_numpy()
    for cc in ("p1", "len", "edep"):
        out[cc] = d[cc].to_numpy()
    info = {"steps": len(out), "translated": int(ok.sum()),
            "level": np.bincount(lev[lev >= 0], minlength=N_UV_LEVELS).tolist(), "no_cell": int((lev < 0).sum())}
    return out, info


def _nearmiss_for_gen(fast_steps, gen, lib, geom, rng):
    """Near-miss templates for the generated muons of events without a fast crossing (fast_steps: tier-1a). Each muon
    reaches the CRV with its cell's probability (lib['nmreach']) and takes a uniform template of that cell. Returns
    (steps with mukey = the muon's key, xi = the muon's row, lsp = template particle; info)."""
    mu = muon_rays(gen)
    crossed = pd.MultiIndex.from_frame(fast_steps[KEY].drop_duplicates().astype(np.int64))
    cand = mu[~pd.MultiIndex.from_frame(mu[KEY].astype(np.int64)).isin(crossed)].reset_index(drop=True)
    g, dist, _ = closest_approach(cand[["x", "y", "z"]].to_numpy(), cand[["px", "py", "pz"]].to_numpy(), geom)
    cell = nearmiss_cell(cand.ke.to_numpy(), g, dist)
    r = lib["nmreach"]
    prob = np.zeros(len(cand))
    if len(r):
        i = np.minimum(np.searchsorted(r.cell.to_numpy(), cell), len(r) - 1)
        hit = r.cell.to_numpy()[i] == cell
        prob[hit] = r.prob.to_numpy()[i][hit]
    reach = rng.random(len(cand)) < prob
    sel = np.flatnonzero(reach)
    t = lib["nmtmpl"]
    tcell = t.cell.to_numpy()
    lo, hi = np.searchsorted(tcell, cell[sel], "left"), np.searchsorted(tcell, cell[sel], "right")
    if (hi <= lo).any():
        raise ValueError("near-miss cell with a reach probability but no template")
    tid = t.tid.to_numpy()[lo + (rng.random(len(sel)) * (hi - lo)).astype(np.int64)]
    ns = lib["nmsteps"]
    nt = ns.tid.to_numpy()
    a, z = np.searchsorted(nt, tid, "left"), np.searchsorted(nt, tid, "right")
    cnt = z - a
    im = np.repeat(sel, cnt)
    rows = np.repeat(a - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    d = ns.iloc[rows]
    out = pd.DataFrame({k: cand[k].to_numpy()[im] for k in KEY})
    out["bar"] = d.bar.to_numpy().astype(np.int64)
    out["lsp"], out["xi"], out["pdg"] = d.sp.to_numpy(), im, d.pdg.to_numpy()
    out["mukey"] = cand.mukey.to_numpy()[im]
    t0 = cand.t.to_numpy()[im]
    out["t0"], out["t1"] = t0 + d.dt0.to_numpy(), t0 + d.dt1.to_numpy()
    for c in UV_STEP_COLS + ["px0", "py0", "pz0", "p1", "len", "edep"]:
        out[c] = d[c].to_numpy()
    info = {"candidates": len(cand), "reached": int(len(sel)), "steps": len(out),
            "expected_reached": float(prob.sum())}
    return out, info


def sample(fast_steps, fast_parts, gen, lib, geom, rng, showers=True, context=False, contain=False, point=False,
           mirror=False, translate=False, topology=False, unvisited=False, nearmiss=False, chord=False, follow=False):
    """Fast tier-1a tables + tier 1b: (steps, particles, info). Muons keep their keys (1, or 1..n with all_muons);
    descendants follow them; showers k*1e7+sp. showers=False: descendants only (run configurations whose bars are not
    the library's: the showers carry absolute extracted v04 bar positions). unvisited / nearmiss (opt-in, a library
    with attach_templates): their random draws come after the showers', so every other component is unchanged.
    chord / follow (opt-in, a template sidecar with library chords): same-bar placement along the chord and the chain
    draw of the main descendant draw (_desc_for_crossings); no random number changes."""
    fx = fast_steps.copy()
    desc, n_drop, n_clamp, why = _desc_for_crossings(fx, lib, geom, rng, context, contain, point, mirror, translate,
                                                     topology, drop_unvisited=unvisited, chord=chord, follow=follow)
    sh = _showers_for_gen(gen, lib, rng) if showers else _showers_for_gen(gen.iloc[:0], lib, rng)
    sp_of = fx.sp.to_numpy() if "sp" in fx.columns else np.ones(len(fx), np.int64)
    desc["src"], desc["parent"] = 0, sp_of[desc.xi.to_numpy()]
    extra, tinfo = [desc], {}
    if unvisited:
        uvd, tinfo["unvisited"] = _unvisited_for_crossings(fx, lib, geom, rng)
        extra.append(uvd.assign(src=1, parent=sp_of[uvd.xi.to_numpy()]))
    if nearmiss:
        nmd, tinfo["nearmiss"] = _nearmiss_for_gen(fast_steps, gen, lib, geom, rng)
        extra.append(nmd.drop(columns="mukey").assign(src=2, parent=nmd.mukey.to_numpy()))
    if len(extra) > 1:
        desc = pd.concat(extra, ignore_index=True)
    # descendant particles: one per (event, crossing, library particle), keys after the event's muon keys (2.. with one
    # muon), parent = the muon of the crossing (tier-1a `sp`); template rows (src 1, 2) come after the event's others
    nmu = fast_parts.groupby(KEY, sort=False).key.max().rename("kmax").reset_index() if len(fast_parts) else None
    pk = KEY + ["xi", "src", "lsp"]
    u = desc[pk + ["pdg", "parent"]].drop_duplicates(pk).reset_index(drop=True)
    base = (u[KEY].merge(nmu, on=KEY, how="left").kmax.fillna(1).to_numpy().astype(np.int64) if nmu is not None
            else np.ones(len(u), np.int64))
    u["key"] = u.groupby(KEY).cumcount().to_numpy() + 1 + base
    desc = desc.merge(u[pk + ["key"]], on=pk)
    desc["sp"] = desc.key
    dparts = pd.DataFrame({k: u[k].to_numpy() for k in KEY})
    dparts["key"], dparts["parent"], dparts["pdg"], dparts["isprim"] = u.key, u.parent, u.pdg, 0
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
    if tinfo:
        info["templates"] = tinfo
    return steps, parts, info
