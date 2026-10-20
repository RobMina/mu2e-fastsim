"""Tier-1a muon transport (stage A): straight lines from the CRY primaries to the CRV bars, Geant4-matched energy loss.

Stage A world: air everywhere outside the CRV (no building, overburden, aluminium sheets or strongbacks, no multiple
scattering). It is the first rung of V1: where it agrees with G4 the CRV-internal physics is right; where it fails
(muons through concrete or overburden, about 5 %) the missing world material is the reason. Stage B adds the material.

Per primary muon (vectorised over all muons of a file):
  1. crossings of the straight line with the 576 bars (crossings.crossings), ordered by path parameter;
  2. |p| at each bar entry = |p0| minus the air loss to that point minus the restricted losses in earlier bars;
  3. per bar: sampled restricted loss (eloss.sample_loss), visible energy with Birks (eloss.birks); start time from
     the path length at the muon's speed.
Output rows follow the CrvStep-table columns (tables.py) so v1compare reads them unchanged.
"""
import numpy as np
import pandas as pd

from mu2efastsim import eloss, materials
from mu2efastsim import world as world_mod
from mu2efastsim.transport import msc_to_plane, slab_loss  # noqa: F401  (shared pieces; re-exported)

from . import crossings, tables
from .tables import primary_muons

C_MM_PER_NS = 299.792458
TAGGER_FIRST_BAR = 512
Y_MSC_PLANE = 4720.0   # mm: just above the highest bar top (taggers, L3 top at 4719.2); MSC is applied up to here

# world material is traced only for muons whose straight line reaches the plane within this margin of the CRV footprint
# (MSC moves a track at most ~1 m at the plane, p99.9 at 0.2 GeV/c through 1 m of concrete); others cannot hit a bar
CRV_FOOTPRINT_MARGIN = 1500.0


def near_crv(o, d, geom, margin=CRV_FOOTPRINT_MARGIN):
    """Downward tracks whose straight line crosses y = Y_MSC_PLANE within `margin` of the bars' x-z footprint."""
    lo = (geom.center - geom.halflen).min(0)
    hi = (geom.center + geom.halflen).max(0)
    down = d[:, 1] < 0
    L = np.where(down, (o[:, 1] - Y_MSC_PLANE) / -np.where(down, d[:, 1], -1.0), 0.0)
    x = o[:, 0] + d[:, 0] * L
    z = o[:, 2] + d[:, 2] * L
    return down & (x > lo[0] - margin) & (x < hi[0] + margin) & (z > lo[2] - margin) & (z < hi[2] + margin)


BAND_EDGES_Y = np.array([15400.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5500.0, 5000.0, Y_MSC_PLANE])


def world_loss(rng, o, d, p, world):
    """Material between o and the MSC plane: (|p| at the plane, total x/X0, survived?, x/X0 per height band,
    path distance from each band's centre to the plane).

    Loss per material is integrated in slices (slab_loss); materials without constants are treated as concrete. Bands
    in height (BAND_EDGES_Y) locate where the scattering happens, so the lateral spread at the plane uses the right
    lever arm (concrete 1-3 m above the CRV scatters with a short arm; the air above it with a long one).
    """
    L = (o[:, 1] - Y_MSC_PLANE) / -d[:, 1]
    nb = len(BAND_EDGES_Y) - 1
    band_xx0 = np.zeros((len(p), nb))
    band_h = np.zeros((len(p), nb))
    loss = np.zeros(len(p))
    x0_air = eloss.radiation_length_mm(materials.AIR)
    for k in range(nb):
        y_top, y_bot = BAND_EDGES_Y[k], BAND_EDGES_Y[k + 1]
        s_top = np.clip((o[:, 1] - y_top) / -d[:, 1], 0, L)
        s_bot = np.clip((o[:, 1] - y_bot) / -d[:, 1], 0, L)
        seg = s_bot - s_top
        if not (seg > 0).any():
            continue
        band_pieces = [q for q in world if q[3][1] >= y_bot and q[2][1] <= y_top]
        act = np.flatnonzero(seg > 0)
        col = {} if not band_pieces else {
            k: np.bincount(act, weights=v, minlength=len(p))
            for k, v in world_mod.column(band_pieces, (o + d * s_top[:, None])[act], d[act], seg[act]).items()}
        solid = np.zeros(len(p))
        for name, c in col.items():
            if not c.any():
                continue
            mat = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
            hit = np.flatnonzero(c > 0)
            loss[hit] += slab_loss(rng, mat, p[hit] - loss[hit], c[hit])
            band_xx0[:, k] += c / eloss.radiation_length_mm(mat)
            solid += c
        air = np.clip(seg - solid, 0, None)
        loss += eloss.dedx_restricted(materials.AIR, np.maximum(p - loss, 1.0), 1e9) * air
        band_xx0[:, k] += air / x0_air
        band_h[:, k] = L - 0.5 * (s_top + s_bot)
    tk = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU - loss
    p_plane = np.sqrt(np.clip((tk + eloss.MMU) ** 2 - eloss.MMU ** 2, 0, None))
    return p_plane, band_xx0.sum(1), tk > 0.5, band_xx0, band_h


def transport(gen, geom, rng, air_loss=True, stage="A", world=None):
    """Return (steps, particles) for the primary muons in `gen`.

    Stage A: air, no MSC. B: air + MSC to Y_MSC_PLANE. C: GDML world material (world = world.load(...)) + MSC.
    """
    mu = primary_muons(gen)
    p0 = np.sqrt(mu.px ** 2 + mu.py ** 2 + mu.pz ** 2).to_numpy()
    o = mu[["x", "y", "z"]].to_numpy()
    d = mu[["px", "py", "pz"]].to_numpy() / p0[:, None]
    t_off = np.zeros(len(mu))   # path length already travelled before the origin used for the crossings
    p_plane = p0.copy()         # |p| at the MSC plane (stage C: after the world material)
    if stage in ("B", "C"):
        down = near_crv(o, d, geom)
        xx0 = None
        bxx0 = bh = None
        if stage == "C":
            p_plane, xx0, alive_w, bxx0, bh = world_loss(rng, o[down], d[down], p0[down], world)
            keep_mu = np.ones(len(mu), bool)
            keep_mu[np.flatnonzero(down)[~alive_w]] = False
        o2, d2, L = o.copy(), d.copy(), np.zeros(len(mu))
        p_msc = p0[down] if stage == "B" else np.maximum(np.sqrt(p0[down] * np.maximum(p_plane, 1.0)), 1.0)
        bands = (bxx0, bh) if stage == "C" else (None, None)
        o2[down], d2[down], L[down] = msc_to_plane(rng, o[down], d[down], p_msc, Y_MSC_PLANE,
                                                   eloss.radiation_length_mm(materials.AIR), xx0, *bands)
        o, d, t_off = o2, d2, L
        if stage == "C":
            pp = p0.copy()
            pp[down] = p_plane
            p_plane = np.where(keep_mu, pp, 0.0)
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
    pin = p0[it].copy()
    if stage == "C":
        pin = p_plane[it] - eloss.dedx_restricted(air, p_plane[it], 1e9) * t_in   # world loss already applied
    elif air_loss:
        pin = pin - eloss.dedx_restricted(air, p0[it], 1e9) * (t_in + t_off[it])  # unrestricted: no deltas in air
    cut = np.where(bar >= TAGGER_FIRST_BAR, materials.POLYSTYRENE.cuts_e_MeV["tagger"],
                   materials.POLYSTYRENE.cuts_e_MeV["crv"])
    loss = np.zeros_like(path)
    alive = pin > 1.0
    # pass k handles each track's k-th bar (bounded by the max bars crossed per track, ~20)
    first = np.r_[True, it[1:] != it[:-1]]
    rank = np.arange(len(it)) - np.flatnonzero(first)[np.cumsum(first) - 1]
    carried = np.zeros(len(mu))
    alive_after = np.ones(len(mu), bool)   # False once the muon has stopped in a bar
    for k in range(int(rank.max()) + 1 if len(rank) else 0):
        s = np.flatnonzero((rank == k) & alive & alive_after[it])
        p_here = pin[s] - carried[it[s]]
        dead = p_here <= 1.0
        alive[s[dead]] = False
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
    vis = eloss.birks(materials.POLYSTYRENE, loss, path)
    e0 = np.sqrt(p0[it] ** 2 + eloss.MMU ** 2)
    beta = p0[it] / e0
    t0 = mu.t.to_numpy()[it] + (t_in + t_off[it]) / (beta * C_MM_PER_NS)
    pos0 = o[it] + d[it] * t_in[:, None]
    pos1 = o[it] + d[it] * t_out[:, None]
    steps = pd.DataFrame({k: mu[k].to_numpy()[it] for k in tables.KEY})
    steps["nstep"] = np.bincount(it, minlength=len(mu))[it]
    steps["bar"] = bar
    steps["sp"] = 1
    steps["t0"] = t0
    steps["t1"] = t0 + path / (beta * C_MM_PER_NS)
    for j, c in enumerate("xyz"):
        steps[f"{c}0"] = pos0[:, j]
        steps[f"{c}1"] = pos1[:, j]
        steps[f"p{c}0"] = d[it, j] * pin
    steps["p1"] = np.sqrt(np.clip((np.sqrt(pin ** 2 + eloss.MMU ** 2) - loss) ** 2 - eloss.MMU ** 2, 0, None))
    steps["len"] = path
    steps["edep"] = vis
    parts = pd.DataFrame({k: mu[k].to_numpy() for k in tables.KEY})
    parts["key"] = 1
    parts["parent"] = -1
    parts["pdg"] = mu.pdg.to_numpy()
    parts["isprim"] = 1
    return steps, parts
