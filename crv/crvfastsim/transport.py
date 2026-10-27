"""Tier-1a muon transport (stage A): straight lines from the CRY primaries to the CRV bars, Geant4-matched energy loss.

Stage A world: air everywhere outside the CRV (no building, overburden, aluminium sheets or strongbacks, no multiple
scattering). It is the first rung of V1: where it agrees with G4 the CRV-internal physics is right; where it fails
(muons through concrete or overburden, about 5 %) the missing world material is the reason. Stage B adds the material.

Per primary muon (vectorised over all muons of a file):
  1. crossings of the straight line with the 576 bars (crossings.crossings), ordered by path parameter;
  2. |p| at each bar entry from remaining kinetic energy after air, passive-material and earlier bar losses;
  3. per bar: sampled restricted loss (eloss.sample_loss), visible energy with Birks (eloss.birks); start time from
     the path length at the muon's speed.
Output rows follow the CrvStep-table columns (tables.py) so v1compare reads them unchanged.
"""
import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from mu2efastsim import world as world_mod
from mu2efastsim import transport as core_tr
from mu2efastsim.transport import msc_to_plane, slab_loss  # noqa: F401  (shared pieces; re-exported)

from . import crossings, tables
from .tables import primary_muons

C_MM_PER_NS = 299.792458
TAGGER_FIRST_BAR = 512
Y_MSC_PLANE = 4720.0   # mm: just above the highest bar top (taggers, L3 top at 4719.2); MSC is applied up to here

# world material is traced only for muons whose straight line reaches the plane within this margin of the CRV footprint
# (MSC moves a track at most ~1 m at the plane, p99.9 at 0.2 GeV/c through 1 m of concrete); others cannot hit a bar
CRV_FOOTPRINT_MARGIN = 1500.0


def msc_plane(geom):
    """MSC plane height of a CRV geometry: Y_MSC_PLANE for the extracted CRV (bit-identical), else 0.8 mm above the
    highest bar top (the extracted plane's margin), rounded up to 1 mm."""
    top = float((geom.center[:, 1] + geom.halflen[:, 1]).max())
    if abs(top - 4719.2) < 0.05:
        return Y_MSC_PLANE
    return float(np.ceil(top + 0.8))


def band_edges(y_plane):
    """World height bands above the MSC plane: the extracted bands (15.4 m .. 5 m) with the lower edges below the
    plane dropped and the plane as the last edge (extracted: BAND_EDGES_Y unchanged)."""
    e = BAND_EDGES_Y[:-1]
    e = e[e > y_plane + 1.0]
    return np.r_[e, y_plane]


def near_crv_box(o, d, geom, margin=CRV_FOOTPRINT_MARGIN):
    """Downward tracks whose straight line enters the CRV's 3D bounding box grown by `margin` (CRVs with side walls:
    a steep muon can cross the MSC plane far outside the x-z footprint and still reach a wall below it)."""
    lo = (geom.center - geom.halflen).min(0) - margin
    hi = (geom.center + geom.halflen).max(0) + margin
    t_in, t_out = crossings.slab_crossings(o, d, lo[None, :], hi[None, :])
    return (d[:, 1] < 0) & (t_out[:, 0] > np.maximum(t_in[:, 0], 0.0))


def near_crv(o, d, geom, margin=CRV_FOOTPRINT_MARGIN, y_plane=None):
    """Downward tracks whose straight line crosses the MSC plane within `margin` of the bars' x-z footprint."""
    y_plane = Y_MSC_PLANE if y_plane is None else y_plane
    lo = (geom.center - geom.halflen).min(0)
    hi = (geom.center + geom.halflen).max(0)
    down = d[:, 1] < 0
    L = np.where(down, (o[:, 1] - y_plane) / -np.where(down, d[:, 1], -1.0), 0.0)
    x = o[:, 0] + d[:, 0] * L
    z = o[:, 2] + d[:, 2] * L
    return down & (x > lo[0] - margin) & (x < hi[0] + margin) & (z > lo[2] - margin) & (z < hi[2] + margin)


BAND_EDGES_Y = np.array([15400.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5500.0, 5000.0, Y_MSC_PLANE])


def world_loss(rng, o, d, p, world, edges=None):
    """Material between o and the MSC plane: (|p| at the plane, total x/X0, survived?, x/X0 per height band,
    path distance from each band's centre to the plane).

    Loss per material is integrated in slices (slab_loss); materials without constants are treated as concrete. Bands
    in height (BAND_EDGES_Y) locate where the scattering happens, so the lateral spread at the plane uses the right
    lever arm (concrete 1-3 m above the CRV scatters with a short arm; the air above it with a long one).
    """
    edges = BAND_EDGES_Y if edges is None else edges
    L = (o[:, 1] - edges[-1]) / -d[:, 1]
    nb = len(edges) - 1
    band_xx0 = np.zeros((len(p), nb))
    band_h = np.zeros((len(p), nb))
    loss = np.zeros(len(p))
    x0_air = eloss.radiation_length_mm(materials.AIR)
    for k in range(nb):
        y_top, y_bot = edges[k], edges[k + 1]
        s_top = np.clip((o[:, 1] - y_top) / -d[:, 1], 0, L)
        s_bot = np.clip((o[:, 1] - y_bot) / -d[:, 1], 0, L)
        seg = s_bot - s_top
        if not (seg > 0).any():
            continue
        if isinstance(world, core_tr.BandedWorld):   # pre-banded grid (same columns, vectorised over pieces)
            band_pieces = world.band(k)
        else:
            band_pieces = [q for q in world if q[3][1] >= y_bot and q[2][1] <= y_top]
        act = np.flatnonzero(seg > 0)
        col = {} if not len(band_pieces) else {
            k: np.bincount(act, weights=v, minlength=len(p))
            for k, v in world_mod.column(band_pieces, (o + d * s_top[:, None])[act], d[act], seg[act]).items()}
        solid = np.zeros(len(p))
        order = world.mat_order[k] if hasattr(world, "mat_order") else list(col)   # the piece list's order: same draws
        for name, c in ((m, col[m]) for m in order if m in col):
            if not c.any():
                continue
            mat = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
            hit = np.flatnonzero(c > 0)
            loss[hit] += slab_loss(rng, mat, core_tr._p_after(p[hit], loss[hit]), c[hit])
            band_xx0[:, k] += c / eloss.radiation_length_mm(mat)
            solid += c
        air = np.clip(seg - solid, 0, None)
        loss += eloss.dedx_restricted(materials.AIR, np.maximum(core_tr._p_after(p, loss), 1.0), 1e9) * air
        band_xx0[:, k] += air / x0_air
        band_h[:, k] = L - 0.5 * (s_top + s_bot)
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    p_plane = core_tr._p_after(p, loss)
    return p_plane, band_xx0.sum(1), tk > 0.5, band_xx0, band_h


def all_primary_muons(gen):
    """Every generated muon, one row per muon, ordered by event and gen index, with `mukey` = its SimParticle key:
    1 for the first muon of an event (as primary_muons gives), 2, 3, ... for further muons of the same event."""
    mu = gen[gen.pdg.abs() == tables.PRIMARY_MUON_PDG]
    order = [c for c in tables.KEY + ["idx"] if c in mu.columns]
    mu = mu.sort_values(order, kind="stable").reset_index(drop=True)
    mu["mukey"] = mu.groupby(tables.KEY, sort=False).cumcount().to_numpy() + 1
    return mu


def absorber_slabs(gdml_path, prefix=("CRSAbsorber", "CRSAluminumSheet")):
    """The CRV aluminium inside the module stack as axis-aligned boxes (lo [n, 3], hi [n, 3]) and their material, from
    the GDML world: the absorber sheets between scintillator layers (CRSAbsorber_<sector>_<module>_<gap>, 9.5 mm) and
    the module skins (CRSAluminumSheet_<sector>_<module>_<0 bottom 12.7 mm | 1 top 3.2 mm>); 44 mm of Al on a vertical
    path through EX. They sit below Y_MSC_PLANE, so the stage-C world column (which stops at the plane) never sees them."""
    names = []
    pieces = world_mod.load(gdml_path, y_min=0.0, names=names)
    sel = [(p[2], p[3], p[0]) for n, p in zip(names, pieces) if n.startswith(tuple(np.atleast_1d(prefix)))]
    if not sel:
        raise ValueError(f"no {prefix} pieces in {gdml_path}")
    mats = {m for _, _, m in sel}
    if len(mats) != 1:
        raise ValueError(f"absorbers of several materials: {mats}")
    return np.array([a for a, _, _ in sel]), np.array([b for _, b, _ in sel]), mats.pop()


def load_world(gdml_path, banded=True, y_plane=Y_MSC_PLANE):
    """(stage-C world above the MSC plane without the bars, CRV aluminium slabs) from one GDML parse. banded: the world
    as a core BandedWorld (grid per height band; same columns and steps as the piece list, ~5x faster transport).
    y_plane: the geometry's MSC plane (msc_plane(geom); the CRV aluminium is read from every height)."""
    names = []
    pieces = world_mod.load(gdml_path, y_min=min(0.0, y_plane - 6000.0), names=names)
    world = [q for q in pieces if q[3][1] >= y_plane and q[0] != "G4_POLYSTYRENE"]
    sel = [(q[2], q[3], q[0]) for n, q in zip(names, pieces) if n.startswith(("CRSAbsorber", "CRSAluminumSheet"))]
    ab = None
    if sel:
        mats = {m for _, _, m in sel}
        if len(mats) != 1:
            raise ValueError(f"absorbers of several materials: {mats}")
        ab = (np.array([a for a, _, _ in sel]), np.array([b for _, b, _ in sel]), mats.pop())
    return (banded_world(world, band_edges(y_plane)) if banded else world), ab


def load_below(gdml_path, geom, y_plane=None, margin=2000.0):
    """world.PieceSet (grid) of the material below the MSC plane inside the CRV's box (+ margin), bars excluded: the
    detector, DS, shielding and CRV aluminium a muon crosses before or between the bars of a CRV that wraps the DS
    (Run 1A/Run 2), and with a large margin (RunConfig.below_margin) the hall walls and dirt in front of the side walls. Materials without constants (StrawGas, Inconel718, ... ~1 % of the pieces, inside the DS) are
    treated as concrete by the loss, as in world_loss."""
    y_plane = msc_plane(geom) if y_plane is None else y_plane
    lo = (geom.center - geom.halflen).min(0) - margin
    hi = (geom.center + geom.halflen).max(0) + margin
    pieces = world_mod.load(gdml_path, y_min=lo[1])
    keep = [q for q in pieces if q[0] != "G4_POLYSTYRENE" and q[2][1] < y_plane and (q[3] > lo).all() and (q[2] < hi).all()]
    return world_mod.PieceSet(keep).build_grid(250.0)


def banded_world(world, edges=None):
    """core BandedWorld of a stage-C piece list, with each band's materials in first-appearance order (mat_order), the
    order the piece-list path draws its slab losses in, so both give the same steps for a seed."""
    edges = BAND_EDGES_Y if edges is None else np.asarray(edges, float)
    bw = core_tr.BandedWorld(world, edges)
    bw.mat_order = [list(dict.fromkeys(q[0] for q in world if q[3][1] >= edges[k + 1] and q[2][1] <= edges[k]))
                    for k in range(len(edges) - 1)]
    return bw


def _next_crossing(o, d, geom, exclude):
    """First bar each line enters (t >= 0), skipping bar `exclude` [n] (the bar just left, -1 none): (track, bar,
    t_in, t_out) with one row per track that has a crossing."""
    xs = crossings.crossings(o, d, geom)
    if not len(xs):
        z = np.zeros(0)
        return z.astype(np.int64), z.astype(np.int64), z, z
    it, bar = xs[:, 0].astype(np.int64), xs[:, 1].astype(np.int64)
    ok = (bar != exclude[it]) & (xs[:, 4] > 1e-9)
    it, bar, t_in, t_out = it[ok], bar[ok], xs[ok, 2], xs[ok, 3]
    order = np.lexsort((t_in, it))
    it, bar, t_in, t_out = it[order], bar[order], t_in[order], t_out[order]
    first = np.r_[True, it[1:] != it[:-1]] if len(it) else np.zeros(0, bool)
    return it[first], bar[first], t_in[first], t_out[first]


def _msc_sample(rng, var, e1, e2):
    """Correlated (angle, displacement) [n, 2] in two transverse planes for scattering of variance var [rad^2] spread
    along a segment with E[lever] = e1 and E[lever^2] = e2 (lever = distance from the scattering point to the segment
    end): theta ~ N(0, var), y = theta e1 + N(0, var (e2 - e1^2)) (the PDG slab case is e1 = L/2, e2 = L^2/3)."""
    z_t = rng.standard_normal((len(var), 2))
    z_y = rng.standard_normal((len(var), 2))
    a = np.sqrt(np.maximum(var, 0.0))
    dth = z_t * a[:, None]
    dy = dth * e1[:, None] + z_y * (a * np.sqrt(np.maximum(e2 - e1 ** 2, 0.0)))[:, None]
    return dth, dy


def _kick(pos, d, dth, dy):
    """Turn direction d by the plane angles dth and displace pos by dy, in two planes perpendicular to d."""
    ref = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    dn = d + u * np.tan(dth[:, [0]]) + v * np.tan(dth[:, [1]])
    return pos + u * dy[:, [0]] + v * dy[:, [1]], dn / np.linalg.norm(dn, axis=1)[:, None]


def scattered_crossings(rng, o, d, p_base, x_above, geom, below, cut, max_iter=1000):
    """Bar crossings below the MSC plane with multiple scattering in the material the muon crosses there.

    Each track is followed from bar to bar: the next bar on the current line; straggled loss in the below-plane material
    of the gap (below: world.PieceSet, transport.load_below) and Highland scattering of that material, applied at the
    gap's x/X0 centroid (angle, plus the displacement spread about the centroid), after which the next bar is found
    again on the turned line; then the bar's own loss (stopping as in transport) and scattering at its exit. Highland
    variances use the cumulative x/X0 of the track (above-plane column x_above included: highland_var_increment), so a
    stack of slabs sums to the Highland width of the stack. Air below the plane gives loss (Bethe mean at the plane |p|)
    but no scattering. Returns per crossing (track, bar, entry point, direction, |p| at entry, path, sampled loss,
    distance from the origin to the entry), crossings of each track in order.
    """
    from mu2efastsim.transport import highland_var_increment
    n = len(o)
    pos, dirn = np.array(o, float), np.array(d, float)
    carried, dist, xcum = np.zeros(n), np.zeros(n), np.array(x_above, float)
    last = np.full(n, -1, np.int64)
    dedx_air = eloss.dedx_restricted(materials.AIR, np.maximum(p_base, 1.0), 1e9)
    x0_poly = eloss.radiation_length_mm(materials.POLYSTYRENE)
    mats = {m: materials.BY_NAME.get(m, materials.CONCRETE_MARS) for m in below.mats}
    x0 = {m: eloss.radiation_length_mm(mat) for m, mat in mats.items()}
    rows = []
    active = np.flatnonzero(p_base > 1.0)
    for _ in range(max_iter):
        if not len(active):
            break
        k, bar, t_in, t_out = _next_crossing(pos[active], dirn[active], geom, last[active])
        a = active[k]
        if not len(a):
            break
        # gap: straggled loss per material in fixed order, x/X0 and its first two moments along the segment
        col = below.column(pos[a], dirn[a], t_in, moments=True) if len(below) else {}
        p_start = core_tr._p_after(p_base[a], dedx_air[a] * dist[a] + carried[a])
        xg, s1, s2 = np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a))
        live = np.ones(len(a), bool)
        for m in sorted(col):
            ln, m1, m2 = col[m]
            hit = np.flatnonzero((ln > 0) & live)
            if not len(hit):
                continue
            pa = core_tr._p_after(p_base[a[hit]], dedx_air[a[hit]] * dist[a[hit]] + carried[a[hit]])
            ok = pa > 1.0
            live[hit[~ok]] = False
            hit, pa = hit[ok], pa[ok]
            carried[a[hit]] += slab_loss(rng, mats[m], pa, ln[hit])
            xg += ln / x0[m]
            s1 += m1 / x0[m]
            s2 += m2 / x0[m]
        p_end = core_tr._p_after(p_base[a], dedx_air[a] * (dist[a] + t_in) + carried[a])
        live &= p_end > 1.0
        a, bar, t_in, t_out, xg, s1, s2 = (v[live] for v in (a, bar, t_in, t_out, xg, s1, s2))
        p_start, p_end = p_start[live], p_end[live]
        # scattering of the gap material at its x/X0 centroid, then the next bar on the turned line
        sc = np.flatnonzero(xg > 0)
        if len(sc):
            q = a[sc]
            xs = xg[sc]
            cen = s1[sc] / xs
            var = highland_var_increment(np.sqrt(p_start[sc] * p_end[sc]), xcum[q], xcum[q] + xs)
            dth, dy = _msc_sample(rng, var, np.zeros(len(q)), np.maximum(s2[sc] / xs - cen ** 2, 0.0))
            xcum[q] += xs
            dist[q] += cen
            pos[q], dirn[q] = _kick(pos[q] + dirn[q] * cen[:, None], dirn[q], dth, dy)
            k2, bar2, t_in2, t_out2 = _next_crossing(pos[q], dirn[q], geom, last[q])
            keep = np.ones(len(a), bool)
            keep[sc] = False
            keep[sc[k2]] = True
            bar[sc[k2]], t_in[sc[k2]], t_out[sc[k2]] = bar2, t_in2, t_out2
            a, bar, t_in, t_out = a[keep], bar[keep], t_in[keep], t_out[keep]
        # the bar: entry, loss (range-out caps the loss and the path), scattering at the exit
        entry = pos[a] + dirn[a] * t_in[:, None]
        dist[a] += t_in
        path = t_out - t_in
        p_here = core_tr._p_after(p_base[a], dedx_air[a] * dist[a] + carried[a])
        ok = p_here > 1.0
        a, bar, entry, path, p_here = a[ok], bar[ok], entry[ok], path[ok], p_here[ok]
        loss = eloss.sample_loss(rng, materials.POLYSTYRENE, p_here, path, cut[bar])
        tkin = np.sqrt(p_here ** 2 + eloss.MMU ** 2) - eloss.MMU
        stops = loss >= tkin
        loss[stops] = tkin[stops]
        path[stops] = np.minimum(path[stops], tkin[stops] / eloss.dedx_restricted(materials.POLYSTYRENE,
                                                                                  p_here[stops], 1e9))
        rows.append((a, bar, entry, dirn[a].copy(), p_here, path.copy(), loss, dist[a].copy()))
        carried[a] += loss
        go = ~stops
        a, bar, entry, path, p_here, loss = a[go], bar[go], entry[go], path[go], p_here[go], loss[go]
        xb = path / x0_poly
        p_out = core_tr._p_after(p_here, loss)
        var = highland_var_increment(np.sqrt(p_here * np.maximum(p_out, 1.0)), xcum[a], xcum[a] + xb)
        dth, dy = _msc_sample(rng, var, path / 2, path ** 2 / 3)
        xcum[a] += xb
        dist[a] += path
        pos[a], dirn[a] = _kick(entry + dirn[a] * path[:, None], dirn[a], dth, dy)
        last[a] = bar
        active = a[p_out > 1.0]
    else:
        raise RuntimeError(f"scattered_crossings: tracks still crossing bars after {max_iter} iterations")
    if not rows:
        z = np.zeros(0)
        return (z.astype(np.int64), z.astype(np.int64), np.zeros((0, 3)), np.zeros((0, 3)), z, z, z, z)
    out = [np.concatenate([r[j] for r in rows]) for j in range(8)]
    rank = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rows)])
    order = np.lexsort((rank, out[0]))
    return tuple(v[order] for v in out)


def scatter_in_crossings(rng, o, d, p_base, x_above, geom, below, cut, *, step_mm=50.0,
                         envelope_mm=1500.0, max_steps=20000, diagnostics=None):
    """Experimental finite-domain, angular-endpoint marcher; no initial bar is required.

    Every straight segment ends at a bar boundary, domain exit, or step_mm. Only its actually traversed
    material is charged, then an angular kick is applied and ALL bar intersections are recomputed. No lateral
    endpoint jump can skip a bar. This first-order angular diffusion needs step/envelope convergence studies;
    unlike scattered_crossings it does not reproduce the finite-slab displacement covariance in one step.
    Returns the legacy eight crossing arrays plus actual end position, end momentum, end distance and summed
    segment Birks energy. Multiple segments/visits are reduced to one ledger row per (track, bar).
    """
    from mu2efastsim.transport import highland_var_increment
    _validate_scatter_in(step_mm, envelope_mm, max_steps, diagnostics)
    n = len(o)
    pos, dirn = np.array(o, float), np.array(d, float)
    momentum, xcum = np.array(p_base, float), np.array(x_above, float)
    if (pos.shape != (n, 3) or dirn.shape != (n, 3) or momentum.shape != (n,)
            or xcum.shape != (n,) or not np.isfinite(pos).all() or not np.isfinite(dirn).all()
            or not np.isfinite(momentum).all() or not np.isfinite(xcum).all()
            or (momentum < 0).any() or (xcum < 0).any()
            or not np.allclose(np.linalg.norm(dirn, axis=1), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError("scatter_in_crossings: finite positions, unit directions and nonnegative p/x0 required")
    cut = np.asarray(cut, float)
    if cut.shape != (geom.nbars,) or not np.isfinite(cut).all() or (cut <= 0).any():
        raise ValueError("scatter_in_crossings: one finite positive cut per bar required")
    # The domain includes the approach from each admitted plane point to the detector. Per-track bounds avoid
    # dependence on other tracks in the batch. A track leaving it is absorbed, not silently allowed to return.
    lo = np.minimum((geom.center - geom.halflen).min(0) - envelope_mm, pos - envelope_mm)
    hi = np.maximum((geom.center + geom.halflen).max(0) + envelope_mm, pos + envelope_mm)
    hi[:, 1] = np.maximum(msc_plane(geom), pos[:, 1])  # never propagate back through the above-plane world
    dist, counts = np.zeros(n), np.zeros(n, np.int64)
    initial_miss = np.zeros(n, bool)
    active = np.flatnonzero(momentum > 1.0)
    rows = []
    stats = dict(step_mm=float(step_mm), envelope_mm=float(envelope_mm), max_steps=int(max_steps),
                 tracks_started=len(active), escaped=0, stopped=0, segments=0, bar_segments=0,
                 iterations=0, max_segments_per_track=0, max_path_mm=0., max_segment_mm=0.,
                 max_segment_xx0=0., max_scatter_sigma_rad=0.,
                 recovered_initial_misses=0, cap_hit=False, complete=False)
    if diagnostics is not None:
        diagnostics.update(stats)
    mats = {m: materials.BY_NAME.get(m, materials.CONCRETE_MARS) for m in below.mats}
    x0 = {m: eloss.radiation_length_mm(mat) for m, mat in mats.items()}
    x0_poly = eloss.radiation_length_mm(materials.POLYSTYRENE)
    for iteration in range(max_steps):  # vectorized across ALL active tracks, never a per-track loop
        if not len(active):
            break
        # Distance to first outward domain face. Origins on the top face moving down are admitted.
        a = active
        inv = 1. / np.where(dirn[a] == 0., 1e-300, dirn[a])
        exit_dist = np.maximum((lo[a] - pos[a]) * inv, (hi[a] - pos[a]) * inv).min(1)
        inside_domain = ((pos[a] >= lo[a] - 1e-9) & (pos[a] <= hi[a] + 1e-9)).all(1)
        go = inside_domain & (exit_dist > 1e-9)
        stats["escaped"] += int((~go).sum())
        a, exit_dist = a[go], exit_dist[go]
        active = a
        if not len(a):
            break
        k, bar, tin, tout = _next_crossing(pos[a], dirn[a], geom, np.full(len(a), -1))
        if iteration == 0:
            initial_miss[a] = True
            initial_miss[a[k]] = False
        ibar = np.full(len(a), -1, np.int64)
        seg = np.minimum(step_mm, exit_dist)
        in_bar = tin <= 1e-9
        ibar[k[in_bar]] = bar[in_bar]
        seg[k] = np.minimum(seg[k], np.where(in_bar, tout, tin))
        b = np.flatnonzero(ibar >= 0)
        g = np.flatnonzero(ibar < 0)
        pin = momentum[a].copy()
        loss, xx = np.zeros(len(a)), np.zeros(len(a))
        # Passive solids replace air; bars are excluded from below (load_below contract).
        col = below.column(pos[a[g]], dirn[a[g]], seg[g]) if len(below) and len(g) else {}
        solid = np.zeros(len(g))
        for m in sorted(col):
            ln = col[m]
            hit = np.flatnonzero((ln > 0) & (core_tr._p_after(pin[g], loss[g]) > 1.))
            q = g[hit]
            loss[q] += slab_loss(rng, mats[m], core_tr._p_after(pin[q], loss[q]), ln[hit])
            xx[g] += ln / x0[m]
            solid += ln
        loss[g] += eloss.dedx_restricted(materials.AIR, np.maximum(pin[g], 1.), 1e9) * np.maximum(seg[g] - solid, 0.)
        if len(b):
            loss[b] = eloss.sample_loss(rng, materials.POLYSTYRENE, pin[b], seg[b], cut[ibar[b]])
            xx[b] = seg[b] / x0_poly
        if not np.isfinite(loss).all() or (loss < 0).any():
            raise RuntimeError("scatter_in_crossings: invalid sampled energy loss")
        tk = np.sqrt(pin ** 2 + eloss.MMU ** 2) - eloss.MMU
        stops = loss >= tk
        # Uniform-within-segment range estimate. Stop once, cap the deposited energy, never restart.
        fraction = np.ones(len(a))
        fraction[stops] = tk[stops] / np.maximum(loss[stops], 1e-300)
        seg *= fraction
        xx *= fraction
        loss = np.minimum(loss, tk)
        pout = core_tr._p_after(pin, loss)
        end = pos[a] + dirn[a] * seg[:, None]
        if len(b):
            rows.append((a[b], ibar[b], pos[a[b]].copy(), dirn[a[b]].copy(), pin[b], seg[b], loss[b],
                         dist[a[b]].copy(), end[b], pout[b], dist[a[b]] + seg[b],
                         eloss.birks(materials.POLYSTYRENE, loss[b], seg[b])))
        pos[a], momentum[a] = end, pout
        dist[a] += seg
        counts[a] += 1
        stats["segments"] += len(a)
        stats["bar_segments"] += len(b)
        stats["iterations"] = iteration + 1
        stats["max_segment_mm"] = max(stats["max_segment_mm"], float(seg.max(initial=0.)))
        stats["max_segment_xx0"] = max(stats["max_segment_xx0"], float(xx.max(initial=0.)))
        live = (~stops) & (pout > 1.)
        stats["stopped"] += int((~live).sum())
        # A boundary exit is absorbing even if an endpoint kick would turn it back inward.
        escaped = live & (seg >= exit_dist - 1e-9)
        stats["escaped"] += int(escaped.sum())
        live &= ~escaped
        sc = np.flatnonzero(live & (xx > 0))
        q = a[sc]
        if len(sc):
            var = highland_var_increment(np.sqrt(pin[sc] * pout[sc]), xcum[q], xcum[q] + xx[sc])
            sigma = np.sqrt(np.maximum(var, 0.))
            stats["max_scatter_sigma_rad"] = max(stats["max_scatter_sigma_rad"], float(sigma.max(initial=0.)))
            dth = rng.standard_normal((len(sc), 2)) * sigma[:, None]
            pos[q], dirn[q] = _kick(pos[q], dirn[q], dth, np.zeros_like(dth))
        xcum[a] += xx
        active = a[live]
    stats["max_segments_per_track"] = int(counts.max(initial=0))
    stats["max_path_mm"] = float(dist.max(initial=0.))
    stats["cap_hit"] = bool(len(active))
    stats["complete"] = not len(active)
    if rows:
        stats["recovered_initial_misses"] = int(initial_miss[np.unique(np.concatenate([r[0] for r in rows]))].sum())
    if diagnostics is not None:
        diagnostics.update(stats)
    if len(active):
        raise RuntimeError(f"scatter_in_crossings: {len(active)} active tracks at cap {max_steps}; no partial result")
    if not rows:
        z = np.zeros(0)
        return (z.astype(np.int64), z.astype(np.int64), np.zeros((0, 3)), np.zeros((0, 3)),
                z, z, z, z, np.zeros((0, 3)), z, z, z)
    data = [np.concatenate([r[j] for r in rows]) for j in range(12)]
    # Preserve first entry and last exit, with a single energy/path ledger even after a genuine re-entry.
    _, first, group = np.unique(np.stack(data[:2], axis=1), axis=0, return_index=True, return_inverse=True)
    last = np.zeros(len(first), np.int64)
    np.maximum.at(last, group, np.arange(len(group)))
    out = [v[first] for v in data]
    for j in (5, 6, 11):
        out[j] = np.bincount(group, weights=data[j], minlength=len(first))
    for j in (8, 9, 10):
        out[j] = data[j][last]
    order = np.lexsort((out[7], out[0]))
    return tuple(v[order] for v in out)


def _validate_scatter_in(step_mm, envelope_mm, max_steps, diagnostics):
    for name, v in (("scatter_in_step_mm", step_mm), ("scatter_in_envelope_mm", envelope_mm)):
        if not np.isscalar(v) or isinstance(v, (str, bool, np.bool_)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError("scatter_in_max_steps must be a positive integer")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError("scatter_in_diagnostics must be a dict or None")


def transport(gen, geom, rng, air_loss=True, stage="A", world=None, msc_tail=False, tail_k=None, sliced_msc=False,
              all_muons=True, absorbers=None, absorber_straggling=True, tagger_bar=None, bar_cut=None, below=None,
              below_msc=False, below_scatter_in=False, scatter_in_step_mm=50.0,
              scatter_in_envelope_mm=1500.0, scatter_in_max_steps=20000, scatter_in_diagnostics=None):
    """Return (steps, particles) for the primary muons in `gen`.

    Stage A: air, no MSC. B: air + MSC to Y_MSC_PLANE. C: GDML world material (world = world.load(...)) + MSC.
    Stage C options (opt-in; defaults reproduce the earlier output bit for bit):
      msc_tail: single-scattering (screened Rutherford) tail per height band above the Gaussian core
        (mu2efastsim.transport.single_scatter_tail; tail_k = threshold in units of theta0, None = density crossing);
      sliced_msc: each band scatters with its own |p| (band_p) instead of the column's geometric mean.
    Both come from the shared core world_loss (the tracker's --msc-tail / --sliced-msc); the default path keeps the
    CRV's own world_loss.
    all_muons (default True since 2026-10-23): transport every generated muon, not only the first of each event (CRY
    makes 0.3 % multi-muon events, which carried 2.5 GeV of the 7.3 GeV G4-minus-fast CRV energy on g4ref_v1 11-20).
    Muon i of an event gets SimParticle key i (1 = the first muon, so single-muon events are unchanged); steps point to
    it through `sp`. all_muons=False (CLI --first-muon-only) reproduces the earlier tables.
    absorbers (opt-in; absorber_slabs(...)): mean (Bethe) energy loss in the CRV aluminium sheets crossed between
    successive bars; a muon whose kinetic energy runs out in a sheet stops there (no later crossings). Without it the
    fast muon sees no material between bars, which delivered 36 % too many soft (< 300 MeV/c at EX) muons to T1, under
    EX and its absorbers (V1 T1 +1.4 %, 2026-10-23). absorber_straggling (default True with absorbers): sample the
    sheet loss with the Urban fluctuation model instead of the mean, so soft muons range out with G4's spread.
    tagger_bar ([nbars] bool, runconfig.RunConfig.tagger_bar): bars with the 10 mm-cut delta threshold; None = the
    extracted v04 taggers (bars >= 512). The MSC plane and the world bands follow the geometry (msc_plane, band_edges);
    a BandedWorld carries its own edges. bar_cut ([nbars] MeV, RunConfig.bar_cut): the delta production threshold per
    bar (overrides tagger_bar). below (world.PieceSet of the material below the MSC plane, load_below): straggled
    loss in it between the plane and the first bar and between successive bars (CRV aluminium included, so use it
    instead of absorbers); a muon that ranges out there makes no later crossings.
    below_msc (with below; RunConfig.below_msc): follow each muon from bar to bar with Highland scattering in the
    below-plane material and in the bars (scattered_crossings) instead of one straight line from the plane. A CRV that
    wraps the DS (Run 1A, Run 2) is reached through metres of material below the plane, and the side walls take muons
    at grazing incidence, where the path in a bar follows the direction. Opt-in: material is only scattered in before a
    bar the current line enters, so muons can scatter out of the CRV but never into it.
    below_scatter_in (experimental, default False) additionally requires stage C, below and below_msc=True.
    It replaces that algorithm with scatter_in_crossings: bounded steps independent of an initial bar hit.
    scatter_in_step_mm=50, scatter_in_envelope_mm=1500 and scatter_in_max_steps=20000 control the approximation;
    an active track at the cap raises instead of returning partial tables. scatter_in_diagnostics may be a dict
    populated with admission, termination and step statistics for convergence scans. See docs/below-scatter-in.md.
    """
    if below_scatter_in:
        if stage != "C" or below is None or not below_msc:
            raise ValueError("below_scatter_in requires stage C, below, and below_msc=True")
        _validate_scatter_in(scatter_in_step_mm, scatter_in_envelope_mm, scatter_in_max_steps,
                             scatter_in_diagnostics)
    y_plane = msc_plane(geom)
    edges = world.edges if isinstance(world, core_tr.BandedWorld) else band_edges(y_plane)
    mu = all_primary_muons(gen) if all_muons else primary_muons(gen)
    mukey = mu.mukey.to_numpy() if all_muons else np.ones(len(mu), np.int64)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy()
    if below_scatter_in and (not np.isfinite(o).all() or not np.isfinite(p0).all() or (p0 <= 0).any()):
        raise ValueError("below_scatter_in requires finite origins and nonzero finite generated momentum")
    d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
    t_off = np.zeros(len(mu))   # path length already travelled before the origin used for the crossings
    p_plane = p0.copy()         # |p| at the MSC plane (stage C: after the world material)
    xx0 = None                  # x/X0 above the plane (stage C)
    if below_msc and below is not None and stage != "C":
        raise ValueError("below_msc needs stage C (the below-plane world starts at the MSC plane)")
    if stage in ("B", "C"):
        if below_scatter_in:
            # Never admit a wider below-plane population without processing its above-plane world too.
            down = near_crv_box(o, d, geom, margin=max(CRV_FOOTPRINT_MARGIN, scatter_in_envelope_mm))
        else:
            down = near_crv_box(o, d, geom) if below is not None else near_crv(o, d, geom, y_plane=y_plane)
        bxx0 = bh = None
        tail = None
        if stage == "C" and (msc_tail or sliced_msc):
            res = core_tr.world_loss(rng, o[down], d[down], p0[down], world, edges, return_band_p=True,
                                     return_tail=msc_tail)
            p_plane, xx0, alive_w, bxx0, bh, bp = res[:6]
            tail = dict(band_p=bp if sliced_msc else None, band_chic2=res[6] if msc_tail else None,
                        band_thn=res[7] if msc_tail else None, tail_k=tail_k)
            keep_mu = np.ones(len(mu), bool)
            keep_mu[np.flatnonzero(down)[~alive_w]] = False
        elif stage == "C":
            p_plane, xx0, alive_w, bxx0, bh = world_loss(rng, o[down], d[down], p0[down], world, edges)
            keep_mu = np.ones(len(mu), bool)
            keep_mu[np.flatnonzero(down)[~alive_w]] = False
        o2, d2, L = o.copy(), d.copy(), np.zeros(len(mu))
        p_msc = p0[down] if stage == "B" else np.maximum(np.sqrt(p0[down] * np.maximum(p_plane, 1.0)), 1.0)
        bands = (bxx0, bh) if stage == "C" else (None, None)
        o2[down], d2[down], L[down] = msc_to_plane(rng, o[down], d[down], p_msc, y_plane,
                                                   eloss.radiation_length_mm(materials.AIR), xx0, *bands,
                                                   **(tail or {}))
        o, d, t_off = o2, d2, L
        if stage == "C":
            pp = p0.copy()
            pp[down] = p_plane
            p_plane = np.where(keep_mu, pp, 0.0)
    if below_msc and below is not None:   # bar to bar, with scattering in the below-plane material and the bars
        if absorbers is not None:
            raise ValueError("below_msc: the below-plane world already holds the CRV aluminium; pass no absorbers")
        if bar_cut is not None:
            cut_bar = np.asarray(bar_cut, float)
        else:
            tag = (np.arange(geom.nbars) >= TAGGER_FIRST_BAR) if tagger_bar is None else np.asarray(tagger_bar, bool)
            cut_bar = np.where(tag, materials.POLYSTYRENE.cuts_e_MeV["tagger"], materials.POLYSTYRENE.cuts_e_MeV["crv"])
        p_base_mu = p_plane if stage == "C" else p0
        x_above = np.zeros(len(mu))
        if xx0 is not None:
            x_above[np.flatnonzero(down)] = xx0
        lo_all, hi_all = (geom.center - geom.halflen).min(0), (geom.center + geom.halflen).max(0)
        b_in, b_out = crossings.slab_crossings(o, d, lo_all[None, :], hi_all[None, :])
        near = np.flatnonzero((b_out[:, 0] > np.maximum(b_in[:, 0], 0.0)) & (p_base_mu > 1.0))
        if below_scatter_in:
            near = np.flatnonzero(down & near_crv_box(o, d, geom, margin=scatter_in_envelope_mm) & (p_base_mu > 1.0))
            if scatter_in_diagnostics is not None:
                scatter_in_diagnostics.update(generated_muons=len(mu), above_processed=int(down.sum()), admitted=len(near))
            result = scatter_in_crossings(rng, o[near], d[near], p_base_mu[near], x_above[near], geom, below, cut_bar,
                                         step_mm=scatter_in_step_mm, envelope_mm=scatter_in_envelope_mm,
                                         max_steps=scatter_in_max_steps, diagnostics=scatter_in_diagnostics)
            it, bar, pos0, dirs, pin, path, loss, dist, pos1, scatter_p1, scatter_dist1, scatter_vis = result
        else:
            it, bar, pos0, dirs, pin, path, loss, dist = scattered_crossings(
                rng, o[near], d[near], p_base_mu[near], x_above[near], geom, below, cut_bar)
        it = near[it]
        keep = loss > 0
        it, bar, pos0, dirs, pin, path, loss, dist = (v[keep] for v in (it, bar, pos0, dirs, pin, path, loss, dist))
        if below_scatter_in:
            pos1, scatter_p1, scatter_dist1, scatter_vis = (v[keep] for v in (pos1, scatter_p1, scatter_dist1, scatter_vis))
        else:
            pos1 = pos0 + dirs * path[:, None]
    else:
        # only tracks whose line enters the CRV bounding box can cross a bar (exact superset: one slab test per track)
        lo_all, hi_all = (geom.center - geom.halflen).min(0), (geom.center + geom.halflen).max(0)
        b_in, b_out = crossings.slab_crossings(o, d, lo_all[None, :], hi_all[None, :])
        near = np.flatnonzero(b_out[:, 0] > np.maximum(b_in[:, 0], 0.0))
        xs = crossings.crossings(o[near], d[near], geom)
        xs[:, 0] = near[xs[:, 0].astype(np.int64)]
        it = xs[:, 0].astype(np.int64)
        bar = xs[:, 1].astype(np.int64)
        t_in, t_out, path = xs[:, 2], xs[:, 3], xs[:, 4]
        order = np.lexsort((t_in, it))
        it, bar, t_in, t_out, path = it[order], bar[order], t_in[order], t_out[order], path[order]
        # air loss to each bar entry (Bethe mean; air thickness ~13 m, fluctuations negligible), CRV losses accumulated
        air = materials.AIR
        p_base = p_plane[it] if stage == "C" else p0[it]
        air_energy = np.zeros_like(path)
        if stage == "C":
            air_energy = eloss.dedx_restricted(air, p_base, 1e9) * t_in
        elif air_loss:
            air_energy = eloss.dedx_restricted(air, p_base, 1e9) * (t_in + t_off[it])
        pin = core_tr._p_after(p_base, air_energy)
        if bar_cut is not None:
            cut = np.asarray(bar_cut, float)[bar]
        else:
            is_tag = (bar >= TAGGER_FIRST_BAR) if tagger_bar is None else np.asarray(tagger_bar, bool)[bar]
            cut = np.where(is_tag, materials.POLYSTYRENE.cuts_e_MeV["tagger"],
                           materials.POLYSTYRENE.cuts_e_MeV["crv"])
        loss = np.zeros_like(path)
        alive = pin > 1.0
        # pass k handles each track's k-th bar (bounded by the max bars crossed per track, ~20)
        first = np.r_[True, it[1:] != it[:-1]]
        rank = np.arange(len(it)) - np.flatnonzero(first)[np.cumsum(first) - 1]
        carried = np.zeros(len(mu))   # kinetic-energy loss [MeV] in earlier bars and passive material
        alive_after = np.ones(len(mu), bool)   # False once the muon has stopped in a bar or gap
        ab_len = None
        if absorbers is not None:   # absorber path between the previous crossing's exit and this crossing's entry
            alo, ahi, amat = absorbers
            amat = materials.BY_NAME.get(amat, amat) if isinstance(amat, str) else amat
            o_it, d_it = o[it], d[it]
            a_in, a_out = crossings.slab_crossings(o_it, d_it, alo, ahi)            # [n crossings, n sheets]
            prev_out = np.where(first, -np.inf, np.r_[0.0, t_out[:-1]])
            lo_t = np.maximum(a_in, prev_out[:, None])
            hi_t = np.minimum(a_out, t_in[:, None])
            ab_len = np.clip(hi_t - lo_t, 0, None).sum(1)
            ab_len[~np.isfinite(ab_len)] = 0.0
        gap_col = None
        if below is not None and len(it):   # material between the previous exit (or the plane) and each bar entry
            g0 = np.where(first, 0.0, np.r_[0.0, t_out[:-1]])
            seg = np.clip(t_in - g0, 0.0, None)
            has = np.flatnonzero(seg > 0)
            gap_col = {m: np.zeros(len(it)) for m in below.mats}
            for m, v in world_mod.column(below, o[it][has] + d[it][has] * g0[has, None], d[it][has], seg[has]).items():
                gap_col[m][has] = v
            gap_col = {m: v for m, v in gap_col.items() if v.any()}
        for k in range(int(rank.max()) + 1 if len(rank) else 0):
            s = np.flatnonzero((rank == k) & alive & alive_after[it])
            if gap_col:   # straggled loss in every material of the gap before this bar, in fixed material order
                for m in sorted(gap_col):
                    c = gap_col[m][s]
                    hit = np.flatnonzero(c > 0)
                    if not len(hit):
                        continue
                    q = s[hit]
                    pa = core_tr._p_after(p_base[q], air_energy[q] + carried[it[q]])
                    live = pa > 1.0
                    q, pa = q[live], pa[live]
                    dE = slab_loss(rng, materials.BY_NAME.get(m, materials.CONCRETE_MARS), pa, c[hit][live])
                    carried[it[q]] += dE
            if ab_len is not None:   # loss in the absorber sheets before this bar (mean dE/dx at the current |p|)
                pa = core_tr._p_after(p_base[s], air_energy[s] + carried[it[s]])
                hit = (ab_len[s] > 0) & (pa > 1.0)
                dl = np.zeros(len(s))
                if hit.any():
                    if absorber_straggling:   # Urban fluctuations of the full loss (large cut: deltas leave the sheet's muon)
                        dl[hit] = eloss.sample_loss(rng, amat, pa[hit], ab_len[s][hit], 1e9)
                    else:
                        dl[hit] = eloss.dedx_restricted(amat, pa[hit], 1e9) * ab_len[s][hit]
                carried[it[s]] += dl
            p_here = core_tr._p_after(p_base[s], air_energy[s] + carried[it[s]])
            dead = p_here <= 1.0
            alive[s[dead]] = False
            alive_after[it[s[dead]]] = False
            s, p_here = s[~dead], p_here[~dead]
            loss[s] = eloss.sample_loss(rng, materials.POLYSTYRENE, p_here, path[s], cut[s])
            tkin = np.sqrt(p_here ** 2 + eloss.MMU ** 2) - eloss.MMU
            stops = loss[s] >= tkin
            if stops.any():   # the muon ranges out in this bar: deposit its kinetic energy over its residual range
                k = s[stops]
                loss[k] = tkin[stops]
                path[k] = np.minimum(path[k], tkin[stops] / eloss.dedx_restricted(materials.POLYSTYRENE, p_here[stops],
                                                                                  1e9))
                alive_after[it[k]] = False
            pin[s] = p_here
            carried[it[s]] += loss[s]   # one bar per track per pass, so no duplicate indices
        keep = alive & (loss > 0)
        it, bar, t_in, t_out, path, loss, pin = (a[keep] for a in (it, bar, t_in, t_out, path, loss, pin))
        dist = t_in
        dirs = d[it]
        pos0 = o[it] + d[it] * t_in[:, None]
        pos1 = o[it] + d[it] * t_out[:, None]
    vis = scatter_vis if below_scatter_in else eloss.birks(materials.POLYSTYRENE, loss, path)
    e0 = np.sqrt(p0[it] ** 2 + eloss.MMU ** 2)
    beta = p0[it] / e0
    t0 = mu.t.to_numpy()[it] + (dist + t_off[it]) / (beta * C_MM_PER_NS)
    steps = pd.DataFrame({k: mu[k].to_numpy()[it] for k in tables.KEY})
    steps["nstep"] = np.bincount(it, minlength=len(mu))[it]
    steps["bar"] = bar
    steps["sp"] = mukey[it]
    steps["t0"] = t0
    steps["t1"] = t0 + ((scatter_dist1 - dist) if below_scatter_in else path) / (beta * C_MM_PER_NS)
    for j, c in enumerate("xyz"):
        steps[f"{c}0"] = pos0[:, j]
        steps[f"{c}1"] = pos1[:, j]
        steps[f"p{c}0"] = dirs[:, j] * pin
    steps["p1"] = scatter_p1 if below_scatter_in else core_tr._p_after(pin, loss)
    steps["len"] = path
    steps["edep"] = vis
    parts = pd.DataFrame({k: mu[k].to_numpy() for k in tables.KEY})
    parts["key"] = mukey
    parts["parent"] = -1
    parts["pdg"] = mu.pdg.to_numpy()
    parts["isprim"] = 1
    return steps, parts
