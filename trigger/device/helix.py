"""Helix seeds on the device, one per time cluster, with the HelixFilter quantities defined as the stock finder does.

The fit is not a port of RobustHelixFinder's search (triplets, median circle fit, phi-z histograms); it is a compact
fit chosen for the device:
- circle: weighted algebraic fit, then geometric Gauss-Newton steps, with hit selection as RobustHelixFinder's final
  XY pass: at most one hit per plane face, the one with the smallest radial pull, pull <= MaxRPull,
  |dr| <= MaxRadiusDiff, radial error from the wire and transverse resolutions;
- phi-z: a scan of signed lambda (|lambda| from fitFZMinLambda to fitFZMaxLambda in fitFZStepLambda) maximising the
  phase coherence of phi - z/lambda, 2-pi unwrapping, a weighted line fit, then RobustHelixFinder's final ZPhi pass:
  per face the hit with the smallest chisq (wire and transverse residuals to the helix at the hit z, including the
  Central{Radial,Perp}Resolution terms), chisq <= MaxChisquared.
Helix quantities then follow HelixFilter / HelixTool: momentum = 0.3 sqrt(R^2 + lambda^2), pT = 0.3 R,
d0 = rcent - R, chi2dXY and chi2dZPhi as the per-face means, nStrawHits over the selected hits, nLoops from the
first and last selected z, and the hit ratio against the faces the helix crosses inside the tracker envelope.
Both helicities are fitted and the better one (more straw hits, then lower chi2dZPhi) is kept.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

F32, F64 = jnp.float32, jnp.float64
NFACES = 72                   # StrawId::_ntotalfaces
TRACKER_RIN, TRACKER_ROUT = 377.0, 810.0   # inner tracker envelope (tracker_v7 / detailed support model)
MM2MEV = 0.3                  # HelixFilter's mm -> MeV/c (1 T)
BIG = 1e9


def settings(m):
    f = m["HelixFitter"]
    return {"maxrpull": float(m["MaxRPull"]), "maxdr": float(m["MaxRadiusDiff"]), "maxchisq": float(m["MaxChisquared"]),
            "cradres": float(m["CentralRadialResolution"]), "cperpres": float(m["CentralPerpResolution"]),
            "minnsh": int(m["MinNStrawHits"]), "lmin": float(f["fitFZMinLambda"]), "lmax": float(f["fitFZMaxLambda"]),
            "lstep": float(f["fitFZStepLambda"]), "minR": float(f["minR"]), "maxR": float(f["maxR"]),
            "minCR": float(f["minCenterR"]), "maxCR": float(f["maxCenterR"]),
            "minabsl": float(f["minAbsLambda"]), "maxabsl": float(f["maxAbsLambda"]),
            "helicities": tuple(int(h) for h in m["Helicities"])}


# Nominal tracker of the stock geometry file (geom_run1_a.txt -> tracker_v7.txt, Offline v13_39): the parameters
# TrackerMaker places the straws with along z.
TRACKER_GEOM = {"nstations": 18, "planeSpacing": 174.0, "planeHalfSeparation": 27.992, "supportHalfThickness": 2.767,
                "manifoldHalfLengthZ": 8.4454, "strawOuterRadius": 2.5235, "strawGap": 1.203, "layerZPattern": 1,
                "panelZPattern": 1, "spacingPattern": 1}
ALIGNED_TOLERANCE = 5.0      # mm: largest |nominal - aligned| face z accepted (catches a geometry that is not the dump's)


def nominal_straw_z(geom, plane, panel, layer):
    """z of a straw midpoint in the nominal tracker, as TrackerMaker::makeStraws / makeLayer compute it."""
    g = geom
    if g["spacingPattern"] != 1:
        raise ValueError("nominal tracker: spacingPattern 1 only")
    R, gap = g["strawOuterRadius"], g["strawGap"]
    rspacing = 2 * R + gap
    lhs = 0.25 * math.sqrt(3.0) * rspacing                              # _layerHalfSpacing
    mzx = g["manifoldHalfLengthZ"] - R - (2 - 1) * lhs                  # _manifoldZEdgeExcessSpace (2 layers)
    z0 = -(float(g["nstations"] - 1) / 2.0 * g["planeSpacing"])         # -findFirstPlaneZ0()
    station, k = plane // 2, plane % 2
    dz = station * g["planeSpacing"] + (g["planeHalfSeparation"] if k else -g["planeHalfSeparation"])
    factor = layer if g["layerZPattern"] == 0 else 2 - layer - 1
    zoff = g["supportHalfThickness"] + mzx + R + factor * 2. * lhs
    side = -1.0 if panel % 2 == 0 else 1.0                              # _panelZSide: -1, +1, -1, ...
    if g["panelZPattern"] == 1:
        side *= -1.0 if plane % 4 in (0, 3) else 1.0
    return zoff * side + (z0 + dz)


def face_z(cond, geom=TRACKER_GEOM):
    """z of the two faces per plane that HelixTool counts for the hit ratio: `float z = (straw 0 z + straw 1 z) / 2.`
    of panels 0 and 1, on the nominal tracker (HelixFilter's GeomHandle<Tracker>, not the aligned one the hits are
    made with). Checked against the aligned straw midpoints of the conditions."""
    mz = np.asarray(cond["straw"]["midz"], np.float64)
    planes = len(mz) // (6 * 96)
    us = lambda pl, pn, st: (pl * 6 + pn) * 96 + st
    nom = np.array([[np.float32((nominal_straw_z(geom, p, f, 0) + nominal_straw_z(geom, p, f, 1)) / 2.) for f in (0, 1)]
                    for p in range(planes)]).ravel()
    aligned = np.array([[0.5 * (mz[us(p, f, 0)] + mz[us(p, f, 1)]) for f in (0, 1)] for p in range(planes)]).ravel()
    if np.max(np.abs(nom - aligned)) > ALIGNED_TOLERANCE:
        raise ValueError(f"nominal face z differs from the aligned straws by up to {np.max(np.abs(nom - aligned)):.3f} mm: "
                         "the tracker geometry is not the dump's")
    return nom.astype(np.float64)


def _circle_lsq(x, y, w):
    """Weighted algebraic circle fit x^2+y^2 = 2a x + 2b y + c."""
    A = jnp.stack([2 * x, 2 * y, jnp.ones_like(x)], -1) * jnp.sqrt(w)[:, None]
    b = (x * x + y * y) * jnp.sqrt(w)
    ata = A.T @ A + 1e-9 * jnp.eye(3)
    sol = jnp.linalg.solve(ata, A.T @ b)
    a, bb, c = sol
    return a, bb, jnp.sqrt(jnp.maximum(c + a * a + bb * bb, 1.0))


def _face_best(score, ok, uf):
    """Per face, the hit with the smallest score among ok hits: mask [H]."""
    H = score.shape[0]
    s = jnp.where(ok, score, BIG)
    best = jnp.full(NFACES, BIG).at[uf].min(s)
    cand = ok & (s == best[uf])
    # first hit in collection order among ties
    first = jnp.full(NFACES, H).at[jnp.where(cand, uf, NFACES - 1)].min(jnp.where(cand, jnp.arange(H), H))
    return cand & (jnp.arange(H) == first[uf])


SCAN_WIDTHS = (16, 32, NFACES)   # helices_pairs: lambda-scan slots, the narrowest holding every pair's XY-selected hits


def _fit(hit, m0, s, lam_sign, fz):
    return _phiz(hit, s, lam_sign, fz, _circle(hit, m0, s))


def _circle(hit, m0, s):
    """The XY part of the fit: circle and its hit selection. It does not depend on the helicity (one call serves both)."""
    x, y = hit["x"].astype(F64), hit["y"].astype(F64)
    ux, uy = hit["udx"].astype(F64), hit["udy"].astype(F64)
    werr2, terr2 = hit["uvar"].astype(F64), hit["vvar"].astype(F64)
    uf, nsh = hit["uface"], hit["nsh"].astype(F64)
    w0 = jnp.where(m0, nsh, 0.0)
    cx, cy, R = _circle_lsq(x, y, w0)

    def xy_select(cx, cy, R):
        dx, dy = x - cx, y - cy
        rr = jnp.sqrt(dx * dx + dy * dy)
        cdx, cdy = dx / rr, dy / rr
        rwdot = ux * cdx + uy * cdy
        dr = rr - R
        rres = jnp.sqrt(werr2 * rwdot ** 2 + terr2 * (1 - rwdot ** 2))
        pull = jnp.abs(dr / rres)
        ok = m0 & (jnp.abs(dr) <= s["maxdr"]) & (pull <= s["maxrpull"])
        sel = _face_best(pull, ok, uf)
        return sel, pull, rres

    def gn(carry, _):
        cx, cy, R = carry
        sel, _, rres = xy_select(cx, cy, R)
        dx, dy = x - cx, y - cy
        rr = jnp.sqrt(dx * dx + dy * dy)
        r = rr - R
        J = jnp.stack([-dx / rr, -dy / rr, -jnp.ones_like(rr)], -1)
        wt = jnp.where(sel, 1.0 / rres ** 2, 0.0)
        JtW = J.T * wt
        step = jnp.linalg.solve(JtW @ J + 1e-6 * jnp.eye(3), -(JtW @ r))
        good = jnp.sum(sel) >= 3
        new = (cx + step[0], cy + step[1], R + step[2])
        return tuple(jnp.where(good, n, o) for n, o in zip(new, (cx, cy, R))), None

    (cx, cy, R), _ = jax.lax.scan(gn, (cx, cy, R), None, length=8, unroll=True)
    selxy, pull, _ = xy_select(cx, cy, R)
    nxy = jnp.sum(selxy)
    chi2xy = jnp.sum(jnp.where(selxy, pull ** 2, 0.0)) / jnp.maximum(nxy, 1e-10)
    return cx, cy, R, selxy, nxy, chi2xy


def _phiz(hit, s, lam_sign, fz, circle, width=None):
    """The phi-z part of the fit for one helicity (sign of lambda), from the circle of _circle. With `width`, the lambda
    scan sums over the XY-selected hits only, gathered in collection order into `width` slots (width >= their count:
    the caller's choice); the other hits carry weight 0, so the sums and the fit are the same."""
    cx, cy, R, selxy, nxy, chi2xy = circle
    x, y, z = hit["x"].astype(F64), hit["y"].astype(F64), hit["z"].astype(F64)
    ux, uy = hit["udx"].astype(F64), hit["udy"].astype(F64)
    werr2, terr2 = hit["uvar"].astype(F64), hit["vvar"].astype(F64)
    uf, nsh = hit["uface"], hit["nsh"].astype(F64)

    # phi-z: signed-lambda scan of phase coherence, then unwrapped weighted line fit
    phi = jnp.arctan2(y - cy, x - cx)
    lams = lam_sign * jnp.arange(s["lmin"], s["lmax"] + 0.5 * s["lstep"], s["lstep"])
    if width is None or width >= x.shape[0]:
        wz, phs, zs = jnp.where(selxy, nsh, 0.0), phi, z
    else:
        slot = jnp.nonzero(selxy, size=width, fill_value=0)[0]
        wz, phs, zs = jnp.where(jnp.arange(width) < nxy, nsh[slot], 0.0), phi[slot], z[slot]
    ph0 = phs[None, :] - zs[None, :] / lams[:, None]
    coh = jnp.abs(jnp.sum(wz[None, :] * jnp.exp(1j * ph0), axis=1))
    k = jnp.argmax(coh)
    lam = lams[k]
    fz0 = jnp.angle(jnp.sum(wz * jnp.exp(1j * ph0[k])))
    tx, ty = -(y - cy) / jnp.maximum(R, 1.0), (x - cx) / jnp.maximum(R, 1.0)        # tangent (unit)
    pdot = ux * tx + uy * ty
    sphi2 = (werr2 * pdot ** 2 + terr2 * (1 - pdot ** 2)) / jnp.maximum(R, 1.0) ** 2

    def line(carry, _):
        fz0, invl = carry
        n = jnp.round((fz0 + z * invl - phi) / (2 * jnp.pi))
        pu = phi + 2 * jnp.pi * n
        wt = jnp.where(selxy, 1.0 / sphi2, 0.0)
        S, Sz, Szz = jnp.sum(wt), jnp.sum(wt * z), jnp.sum(wt * z * z)
        Sp, Szp = jnp.sum(wt * pu), jnp.sum(wt * z * pu)
        det = S * Szz - Sz * Sz
        ok = (nxy >= 3) & (det > 0)
        invl_new = (S * Szp - Sz * Sp) / jnp.where(ok, det, 1.0)
        fz_new = (Sp - invl_new * Sz) / jnp.where(ok, S, 1.0)
        return (jnp.where(ok, fz_new, fz0), jnp.where(ok, invl_new, invl)), None

    (fz0, invl), _ = jax.lax.scan(line, (fz0, 1.0 / lam), None, length=3, unroll=True)
    lam = 1.0 / invl

    # final ZPhi pass (RobustHelixFinder): per face the best non-outlier hit by chisq
    ph = fz0 + z * invl
    hx, hy = cx + R * jnp.cos(ph), cy + R * jnp.sin(ph)
    dhx, dhy = x - hx, y - hy
    wtx, wty = -uy, ux
    dtrans, dwire = jnp.abs(dhx * wtx + dhy * wty), jnp.abs(dhx * ux + dhy * uy)
    dx, dy = x - cx, y - cy
    rr = jnp.sqrt(dx * dx + dy * dy)
    cdx, cdy = dx / rr, dy / rr
    cpx, cpy = -cdy, cdx
    wres2 = werr2 + (s["cradres"] * (cdx * ux + cdy * uy)) ** 2 + (s["cperpres"] * (cpx * ux + cpy * uy)) ** 2
    wtres2 = terr2 + (s["cradres"] * (cdx * wtx + cdy * wty)) ** 2 + (s["cperpres"] * (cpx * wtx + cpy * wty)) ** 2
    chisq = dwire ** 2 / wres2 + dtrans ** 2 / wtres2
    sel = _face_best(chisq, selxy & (chisq <= s["maxchisq"]), uf)
    nf = jnp.sum(sel)
    chi2zphi = jnp.sum(jnp.where(sel, chisq, 0.0)) / jnp.maximum(nf, 1e-10)
    nstraw = jnp.sum(jnp.where(sel, nsh, 0.0))
    zsel_min, zsel_max = jnp.min(jnp.where(sel, z, BIG)), jnp.max(jnp.where(sel, z, -BIG))
    nloops = jnp.abs((zsel_max - zsel_min) / (lam * 2 * jnp.pi))
    rcent = jnp.hypot(cx, cy)
    # expected faces (HelixTool::hitRatio)
    phf = fz0 + fz * invl
    rf = jnp.hypot(cx + R * jnp.cos(phf), cy + R * jnp.sin(phf))
    expected = 1e-6 + jnp.sum((rf >= TRACKER_RIN) & (rf <= TRACKER_ROUT))
    ok = (nstraw >= s["minnsh"]) & (R >= s["minR"]) & (R <= s["maxR"]) & (rcent >= s["minCR"]) & (rcent <= s["maxCR"]) & \
        (jnp.abs(lam) >= s["minabsl"]) & (jnp.abs(lam) <= s["maxabsl"]) & jnp.isfinite(lam) & jnp.isfinite(R)
    return {"ok": ok, "helicity": jnp.sign(lam), "R": R, "rcent": rcent, "fcent": jnp.arctan2(cy, cx), "lambda": lam,
            "p": MM2MEV * jnp.sqrt(R * R + lam * lam), "pt": MM2MEV * R, "d0": rcent - R, "chi2xy": chi2xy,
            "chi2zphi": chi2zphi, "nsh": nstraw, "nloops": nloops, "hitratio": nstraw / expected,
            "tandip": jnp.abs(lam) / R}


def _helix_hits(ph, bkg_mask):
    sel_bits = 1 << 10                                     # HitSelectionBits: TimeDivision
    usable = ph["valid"] & ((ph["flag"] & sel_bits) == sel_bits) & ((ph["flag"] & bkg_mask) == 0)
    hit = {k: ph[k] for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar")}
    hit["uface"] = ph["plane"] * 2 + ph["face"]
    hit["nsh"] = ph["ncombo"]
    return hit, usable


def _best_helicity(s, fz, width=None):
    fzj = jnp.asarray(fz)

    def best_helicity(h, circle, valid):
        fits = [_phiz(h, s, float(hel), fzj, circle, width) for hel in s["helicities"]]
        best = fits[0]
        for f in fits[1:]:
            better = f["ok"] & (~best["ok"] | (f["nsh"] > best["nsh"]) |
                                ((f["nsh"] == best["nsh"]) & (f["chi2zphi"] < best["chi2zphi"])))
            best = {k: jnp.where(better, f[k], best[k]) for k in best}
        best["ok"] = best["ok"] & valid
        return best
    return best_helicity


def _per_cluster(s, fz):
    finish = _best_helicity(s, fz)

    def per_cluster(h, use, mem, valid):
        return finish(h, _circle(h, use & mem & valid, s), valid)
    return per_cluster


def helices(ph, tc, s, fz, bkg_mask=0):
    """Panel hits [E, H] + clusters [E, NTC] -> best-helicity helix per cluster [E, NTC]."""
    hit, usable = _helix_hits(ph, bkg_mask)
    per_event = jax.vmap(_per_cluster(s, fz), in_axes=(None, None, 0, 0))
    return jax.vmap(per_event)(hit, usable, tc["members"], tc["valid"])


def helices_multi(ph, tcs, s, fz, bkg_mask=0):
    """helices of several cluster sets with the same helix settings (stock TrigHelixFinder on TrigTimeClusterFinder and
    TrigHelixFinderUe on TrigTimeClusterFinderUe), in one batched call: the same fit per cluster."""
    mem = jnp.stack([tc["members"] for tc in tcs])
    val = jnp.stack([tc["valid"] for tc in tcs])
    out = jax.vmap(lambda m, v: helices(ph, {"members": m, "valid": v}, s, fz, bkg_mask))(mem, val)
    return [{k: v[i] for k, v in out.items()} for i in range(len(tcs))]


def helices_multi_pairs(ph, tcs, s, fz, K, bkg_mask=0):
    """helices_multi on the valid (cluster set, event, slot) clusters only, gathered up to K; when a batch has more
    than K valid clusters, the batched full path runs instead. The same fit per cluster; clusters that are not valid
    get ok False and zero quantities (helices gives them ok False too; every consumer reads helices through `ok`)."""
    mem = jnp.stack([tc["members"] for tc in tcs])
    val = jnp.stack([tc["valid"] for tc in tcs])
    P, E, C = val.shape
    hit, usable = _helix_hits(ph, bkg_mask)

    def full():
        return jax.vmap(lambda m, v: helices(ph, {"members": m, "valid": v}, s, fz, bkg_mask))(mem, val)

    def pairs():
        flat = jnp.nonzero(val.reshape(-1), size=K, fill_value=P * E * C)[0]
        use = flat < P * E * C
        f = jnp.minimum(flat, P * E * C - 1)
        p, e, c = f // (E * C), (f // C) % E, f % C
        best = jax.vmap(_per_cluster(s, fz))({k: v[e] for k, v in hit.items()}, usable[e], mem[p, e, c],
                                            val[p, e, c] & use)
        dst = jnp.where(use, flat, P * E * C)
        out = {k: jnp.zeros(P * E * C + 1, v.dtype).at[dst].set(v)[:P * E * C].reshape(P, E, C) for k, v in best.items()}
        out["ok"] = out["ok"] & val
        return out

    out = jax.lax.cond(jnp.sum(val) <= K, pairs, full)
    return [{k: v[i] for k, v in out.items()} for i in range(P)]


def helices_pairs(ph, tc, s, fz, K, bkg_mask=0):
    """helices on the first K valid (event, cluster) pairs of tc only (K >= their count: the caller reads it), the
    same fit per cluster, scattered back to [E, NTC]. Clusters that are not valid get ok False and zero quantities
    (helices gives them ok False too; every consumer reads helices through `ok`)."""
    hit, usable = _helix_hits(ph, bkg_mask)
    E, C = tc["valid"].shape
    flat = jnp.nonzero(tc["valid"].reshape(-1), size=K, fill_value=E * C)[0]
    use = flat < E * C
    e, c = jnp.minimum(flat // C, E - 1), flat % C
    h = {k: v[e] for k, v in hit.items()}
    valid = tc["valid"][e, c] & use
    circle = jax.vmap(lambda hh, m0: _circle(hh, m0, s))(h, usable[e] & tc["members"][e, c] & valid[:, None])
    H = hit["x"].shape[1]
    widths = [w for w in SCAN_WIDTHS if w < H] + [None]   # None: every hit (as helices)
    branches = [lambda a, w=w: jax.vmap(_best_helicity(s, fz, w))(*a) for w in widths]
    edges = jnp.array([w for w in widths if w is not None], jnp.int32)
    branch = jnp.searchsorted(edges, jnp.max(circle[4]).astype(jnp.int32), side="left")
    best = jax.lax.switch(branch, branches, (h, circle, valid))
    dst = jnp.where(use, flat, E * C)
    out = {k: jnp.zeros(E * C + 1, v.dtype).at[dst].set(v)[:E * C].reshape(E, C) for k, v in best.items()}
    out["ok"] = out["ok"] & tc["valid"]
    return out
