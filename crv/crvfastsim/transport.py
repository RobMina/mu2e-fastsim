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

from . import crossings, eloss, materials, tables
from . import world as world_mod

C_MM_PER_NS = 299.792458
TAGGER_FIRST_BAR = 512
Y_MSC_PLANE = 4720.0   # mm: just above the highest bar top (taggers, L3 top at 4719.2); MSC is applied up to here


def msc_to_plane(rng, o, d, p, y_plane, x0_mm, x_over_x0=None):
    """Move each track from o to the plane y = y_plane through a uniform slab, with PDG (eq. 34.20) correlated MSC.

    Two independent transverse planes per track: y_plane displacement and angle from (z1, z2) unit Gaussians,
    theta_plane = z2 * theta0, y_plane = z1 * L * theta0 / sqrt(12) + z2 * L * theta0 / 2. Returns new (origin,
    direction, path length L). Tracks are assumed downward (d_y < 0).
    """
    L = (o[:, 1] - y_plane) / -d[:, 1]
    th0 = eloss.highland_theta0(p, L / x0_mm if x_over_x0 is None else x_over_x0)
    # orthonormal transverse basis (u, v) to d
    ref = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    u = np.cross(d, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    z1 = rng.standard_normal((len(p), 2))
    z2 = rng.standard_normal((len(p), 2))
    dy = z1 * (L * th0 / np.sqrt(12))[:, None] + z2 * (L * th0 / 2)[:, None]
    dth = z2 * th0[:, None]
    end = o + d * L[:, None] + u * dy[:, [0]] + v * dy[:, [1]]
    dn = d + u * np.tan(dth[:, [0]]) + v * np.tan(dth[:, [1]])
    dn /= np.linalg.norm(dn, axis=1)[:, None]
    # the transverse shift has a y component: slide back along the new direction onto the plane, so no track starts
    # inside or below the topmost bars
    back = (end[:, 1] - y_plane) / dn[:, 1]
    end = end - dn * back[:, None]
    return end, dn, L - back


def primary_muons(gen):
    """The CRY primary muon of each event: one row per event (events with no muon are dropped)."""
    mu = gen[gen.pdg.abs() == tables.PRIMARY_MUON_PDG]
    return mu.drop_duplicates(tables.KEY).reset_index(drop=True)


def world_loss(rng, o, d, p, world):
    """Material column from o along d down to the MSC plane: (|p| at the plane, total x/X0, survived?).

    Mean loss per material (unrestricted Bethe, densities from materials.BY_NAME; air from the leftover path), with
    Urban fluctuations on the solid part. Materials without constants are treated as concrete (logged once).
    """
    L = (o[:, 1] - Y_MSC_PLANE) / -d[:, 1]
    col = world_mod.column(world, o, d, L)
    solid = np.zeros(len(p))
    xx0 = np.zeros(len(p))
    loss = np.zeros(len(p))
    for name, c in col.items():
        if not c.any():
            continue
        mat = materials.BY_NAME.get(name, materials.CONCRETE_MARS)
        hit = c > 0
        loss[hit] += eloss.sample_loss(rng, mat, p[hit], c[hit], 1e9)
        xx0 += c / eloss.radiation_length_mm(mat)
        solid += c
    air_len = np.clip(L - solid, 0, None)
    loss += eloss.dedx_restricted(materials.AIR, p, 1e9) * air_len
    xx0 += air_len / eloss.radiation_length_mm(materials.AIR)
    p_plane = p - loss
    return np.clip(p_plane, 0, None), xx0, p_plane > 1.0


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
        down = d[:, 1] < 0
        xx0 = None
        if stage == "C":
            p_plane, xx0, alive_w = world_loss(rng, o[down], d[down], p0[down], world)
            keep_mu = np.ones(len(mu), bool)
            keep_mu[np.flatnonzero(down)[~alive_w]] = False
        o2, d2, L = o.copy(), d.copy(), np.zeros(len(mu))
        p_msc = p0[down] if stage == "B" else np.maximum(np.sqrt(p0[down] * np.maximum(p_plane, 1.0)), 1.0)
        o2[down], d2[down], L[down] = msc_to_plane(rng, o[down], d[down], p_msc, Y_MSC_PLANE,
                                                   eloss.radiation_length_mm(materials.AIR), xx0)
        o, d, t_off = o2, d2, L
        if stage == "C":
            pp = p0.copy()
            pp[down] = p_plane
            p_plane = np.where(keep_mu, pp, 0.0)
    xs = crossings.crossings(o, d, geom)
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
