"""Opt-in production-target + PS chain (host NumPy reference): primary protons -> target cascade from the thin-target
bank (`ptgt_device`) -> PS vacuum in the field with decays in flight -> states at the TS entrance plane, written as a
tsdevice reference (the same format as `ptgt_g4bl.build`), so stops per POT come from the unchanged W1 chain.

Configuration-agnostic: the target is a run-time `ptgt_device.Target` (any CSG of coaxial frusta and placement), the
PS is an absorbing envelope (piecewise-linear radius about the solenoid axis, upstream kill plane) and a field bank;
none of them is in the bank.

Model, per lane (mm, MeV/c, ns, Mu2e frame):
  * in material: the `ptgt_device` step (inelastic / elastic from the bank, continuous loss with a fractional step
    cap, Highland scattering, cuts with local deposit), plus a first-order field kick and a decay-length limit;
  * in vacuum: RK4 in the field (charged) or straight lines (neutral), step `h_vac`, `h_near` near the target; a
    chord that meets the target re-enters it at the crossing point;
  * decays in flight with a proper time sampled at birth: pi -> mu nu; K -> mu nu, pi pi0, 3 pi, pi pi0 pi0,
    pi0 mu nu (three-body decays: uniform phase space, an approximation), pi0 e nu (no charged mu/pi: dropped);
    muons are kept to the plane (their decay in the PS is a loss);
  * the envelope absorbs a lane whose radius about the axis exceeds r(z) (no scattering back from the HRS: an
    approximation), lanes going upstream past z_min are dropped, and every lane that crosses the record plane is
    recorded there (inside r_plane) and stopped.
"""
from dataclasses import dataclass, field

import numpy as np

from beamfastsim import ptgt_device as D

C_MM_NS = 299.792458
K_FIELD = 0.299792458                 # (MeV/c) per (T mm)
MASS = dict(D.MASS)
MASS.update({111: 134.9768, 130: 497.611, 11: 0.51099895, -11: 0.51099895})
TAU_NS = {211: 26.033, -211: 26.033, 321: 12.380, -321: 12.380, 13: 2196.981, -13: 2196.981}
CHARGE = {2212: 1, 2112: 0, 211: 1, -211: -1, 321: 1, -321: -1, 13: -1, -13: 1}
RECORD = (13, -13, 211, -211)
K_CHANNELS = (                        # (BR, products for K+; K- is the charge conjugate)
    (0.6356, (-13,)),                 # mu+ nu   (charge-conjugate codes are built below)
    (0.2067, (211, 111)),
    (0.05583, (211, 211, -211)),
    (0.01760, (211, 111, 111)),
    (0.03352, (111, -13)),
    (0.0507, ()),                      # pi0 e+ nu: nothing recorded downstream
)


def mass(pdg):
    pdg = np.asarray(pdg)
    out = np.zeros(pdg.shape)
    for k, v in MASS.items():
        out[pdg == k] = v
    return out


def charge(pdg):
    pdg = np.asarray(pdg)
    out = np.zeros(pdg.shape)
    for k, v in CHARGE.items():
        out[pdg == k] = v
    return out


def lifetime(pdg):
    pdg = np.asarray(pdg)
    out = np.full(pdg.shape, np.inf)
    for k, v in TAU_NS.items():
        out[pdg == k] = v
    return out


@dataclass(frozen=True)
class Envelope:
    """Absorbing PS envelope about the solenoid axis (x = axis_x, y = 0, along z) and the record plane."""
    z_nodes: np.ndarray                 # increasing z [mm]
    r_nodes: np.ndarray                 # absorbing radius at the nodes (linear between, flat outside)
    z_min: float                        # upstream kill plane
    z_plane: float = -4044.45           # record plane (G4beamline Coll_01_DetIn)
    r_plane: float = np.inf             # record radius about the axis
    axis_x: float = 3904.0

    def radius(self, x):
        return np.hypot(x[:, 0] - self.axis_x, x[:, 1])

    def r_max(self, z):
        return np.interp(z, self.z_nodes, self.r_nodes)


@dataclass(frozen=True)
class Stepping:
    h_vac: float = 20.0                 # vacuum step [mm]
    h_near: float = 2.0                 # vacuum step within `near` mm of the target's bounding cylinder
    near: float = 30.0
    h_mat: float = 2.0                  # largest step in material
    neutral_step: float = 200.0
    max_iter: int = 200_000
    cuts: D.Cuts = field(default_factory=D.Cuts)


def _rk4(xx, u, p, q, h, bfield):
    """One RK4 step of length h for unit directions u (du/ds = K q / p u x B)."""
    k = K_FIELD * q / p

    def f(xp, up):
        return up, k[:, None] * np.cross(up, bfield(xp))

    x1, u1 = f(xx, u)
    x2, u2 = f(xx + 0.5 * h[:, None] * x1, u + 0.5 * h[:, None] * u1)
    x3, u3 = f(xx + 0.5 * h[:, None] * x2, u + 0.5 * h[:, None] * u2)
    x4, u4 = f(xx + h[:, None] * x3, u + h[:, None] * u3)
    xn = xx + h[:, None] / 6.0 * (x1 + 2 * x2 + 2 * x3 + x4)
    un = u + h[:, None] / 6.0 * (u1 + 2 * u2 + 2 * u3 + u4)
    return xn, un / np.linalg.norm(un, axis=1, keepdims=True)


def _two_body(M, m1, m2, n, rng):
    """Momenta [n, 2, 3] of an isotropic two-body decay at rest."""
    pst = np.sqrt(max((M ** 2 - (m1 + m2) ** 2) * (M ** 2 - (m1 - m2) ** 2), 0.0)) / (2 * M)
    d = _iso(n, rng) * pst
    return np.stack([d, -d], axis=1)


def _three_body(M, ms, n, rng):
    """Uniform phase-space three-body decay at rest (accept-reject on the Dalitz plane); momenta [n, 3, 3]."""
    m1, m2, m3 = ms
    out = np.zeros((n, 3, 3))
    todo = np.arange(n)
    while todo.size:
        k = todo.size
        lo, hi = (m1 + m2) ** 2, (M - m3) ** 2
        m12 = np.sqrt(lo + (hi - lo) * rng.random(k))
        # momenta in the 12 frame and of 3 in the rest frame
        q = np.sqrt(np.maximum((m12 ** 2 - (m1 + m2) ** 2) * (m12 ** 2 - (m1 - m2) ** 2), 0)) / (2 * m12)
        p3 = np.sqrt(np.maximum((M ** 2 - (m12 + m3) ** 2) * (M ** 2 - (m12 - m3) ** 2), 0)) / (2 * M)
        pmax = np.sqrt(((M ** 2 - (m1 + m2 + m3) ** 2) * (M ** 2 - (m1 + m2 - m3) ** 2)) ) / (2 * M) + 1e-12
        qmax = np.sqrt(((M - m3) ** 2 - (m1 + m2) ** 2) * ((M - m3) ** 2 - (m1 - m2) ** 2)) / (2 * (M - m3)) + 1e-12
        ok = rng.random(k) < (q * p3) / (qmax * pmax)
        i = todo[ok]
        m12, q, p3 = m12[ok], q[ok], p3[ok]
        kk = i.size
        d3 = _iso(kk, rng)
        P3 = d3 * p3[:, None]
        dq = _iso(kk, rng)
        e1 = np.sqrt(q ** 2 + m1 ** 2)
        e2 = np.sqrt(q ** 2 + m2 ** 2)
        # boost the 12 pair (momentum -P3, mass m12) from its rest frame
        P12 = -P3
        E12 = np.sqrt(m12 ** 2 + p3 ** 2)
        for j, (e, pv) in enumerate(((e1, dq * q[:, None]), (e2, -dq * q[:, None]))):
            out[i, j] = _boost(e, pv, P12, E12, m12)
        out[i, 2] = P3
        todo = todo[~ok]
    return out


def _iso(n, rng):
    c = 2 * rng.random(n) - 1
    s = np.sqrt(1 - c * c)
    ph = 2 * np.pi * rng.random(n)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], 1)


def _boost(e, pv, P, E, M):
    """Boost (e, pv) from the rest frame of a system of momentum P, energy E, mass M; returns the momentum."""
    b = P / E[:, None]
    g = E / M
    bp = np.sum(b * pv, axis=1)
    b2 = np.sum(b * b, axis=1)
    coef = np.where(b2 > 0, (g - 1) * bp / np.where(b2 > 0, b2, 1.0), 0.0) + g * e
    return pv + coef[:, None] * b


def decay(pdg, x, u, p, t, rng):
    """Daughters (pdg, x, u, p, t, parent row) of decaying pi+- and K+- lanes (muons are not decayed here)."""
    out = dict(pdg=[], x=[], u=[], p=[], t=[], parent=[])
    m0 = mass(pdg)
    E = np.hypot(p, m0)
    P = u * p[:, None]

    def emit(idx, codes, moms):
        for j, c in enumerate(codes):
            if c not in RECORD and abs(c) not in (211, 321):
                continue
            pv = _boost(np.hypot(np.linalg.norm(moms[:, j], axis=1), MASS[c]), moms[:, j], P[idx], E[idx], m0[idx])
            pa = np.linalg.norm(pv, axis=1)
            out["pdg"].append(np.full(idx.size, c)); out["x"].append(x[idx]); out["t"].append(t[idx])
            out["u"].append(pv / np.maximum(pa, 1e-300)[:, None]); out["p"].append(pa); out["parent"].append(idx)

    pi = np.flatnonzero(np.abs(pdg) == 211)
    if pi.size:
        s = np.sign(pdg[pi])
        moms = _two_body(MASS[211], MASS[13], 0.0, pi.size, rng)
        for sign in (1, -1):
            k = pi[s == sign]
            if k.size:
                emit(k, (-13 * sign, 0), moms[s == sign])
    ka = np.flatnonzero(np.abs(pdg) == 321)
    if ka.size:
        cum = np.cumsum([br for br, _ in K_CHANNELS])
        ch = np.searchsorted(cum / cum[-1], rng.random(ka.size), side="right")
        for c, (_, codes) in enumerate(K_CHANNELS):
            for sign in (1, -1):
                k = ka[(ch == c) & (np.sign(pdg[ka]) == sign)]
                if not k.size or not codes:
                    continue
                cc = tuple(sign * a if a != 111 else 111 for a in codes)
                ms = [MASS[a] for a in cc]
                if len(cc) == 1:          # mu nu
                    moms = _two_body(MASS[321], ms[0], 0.0, k.size, rng)
                    emit(k, cc, moms)
                elif len(cc) == 2:
                    moms = _two_body(MASS[321], ms[0], ms[1], k.size, rng)
                    emit(k, cc, moms)
                else:
                    emit(k, cc, _three_body(MASS[321], ms, k.size, rng))
    empty = dict(x=np.zeros((0, 3)), u=np.zeros((0, 3)), pdg=np.zeros(0, np.int64), parent=np.zeros(0, np.int64),
                 p=np.zeros(0), t=np.zeros(0))
    return {kk: (np.concatenate(v) if v else empty[kk]) for kk, v in out.items()}


def primaries(n, target, rng, sigma=1.0, ke=8000.0, offset=0.5):
    """G4beamline beam: protons 0.5 mm outside the target's local +z face moving along local -z, Gaussian sigma
    1 mm in the transverse plane, no divergence, t = 0."""
    zl = 0.5 * (target.z_edges[-1] - target.z_edges[0]) + offset + 0.5 * (target.z_edges[-1] + target.z_edges[0])
    xl = np.stack([sigma * rng.standard_normal(n), sigma * rng.standard_normal(n), np.full(n, zl)], 1)
    dl = np.tile([0.0, 0.0, -1.0], (n, 1))
    x, u = target.to_mu2e(xl, dl)
    m = MASS[2212]
    return dict(pdg=np.full(n, 2212), x=x, u=u, p=np.full(n, np.sqrt((ke + m) ** 2 - m * m)), t=np.zeros(n),
                event=np.arange(n))


def run(bank, target, bfield, env, prim, rng, st=Stepping(), enclosure=None):
    """Transport primaries to the record plane. Returns recorded states (event, pdg, x, p, t), deposit per target
    segment and counters.

    enclosure=(R, H) (diagnostic): instead of the plane, record every lane (all species) where it first leaves the
    cylinder of radius R and half-length H about the target axis (target frame), and stop it there."""
    n0 = len(prim["pdg"])
    L = dict(pdg=np.asarray(prim["pdg"], np.int64), x=np.asarray(prim["x"], float).copy(),
             u=np.asarray(prim["u"], float).copy(), p=np.asarray(prim["p"], float).copy(),
             t=np.asarray(prim["t"], float).copy(), ev=np.asarray(prim["event"], np.int64))
    L["tau"] = -np.log(1 - rng.random(n0)) * lifetime(L["pdg"])
    L["seg"] = target.locate(target.to_local(L["x"]))
    rec = dict(ev=[], pdg=[], x=[], p=[], t=[])
    edep = np.zeros(len(target.mat))
    cnt = dict(inelastic=0, elastic=0, stopped=0, decays=0, absorbed=0, upstream=0, recorded=0, outside_plane=0,
               unknown=0, iterations=0)
    zc = 0.5 * (target.z_edges[0] + target.z_edges[-1])
    hl = 0.5 * (target.z_edges[-1] - target.z_edges[0])
    rb = float(np.max(target.r_out))

    def keep(mask):
        for k in L:
            L[k] = L[k][mask]

    def add(new):
        n = len(new["pdg"])
        if not n:
            return
        new = dict(new)
        new["tau"] = -np.log(1 - rng.random(n)) * lifetime(new["pdg"])
        for k in L:
            L[k] = np.concatenate([L[k], new[k]])

    for it in range(st.max_iter):
        if not len(L["pdg"]):
            break
        cnt["iterations"] = it + 1
        m = mass(L["pdg"])
        ke = np.hypot(L["p"], m) - m
        hadron = np.isin(np.abs(L["pdg"]), (2212, 2112))
        known = np.isin(L["pdg"], D.SPECIES)
        low = np.where(hadron, ke < st.cuts.ke_min_hadron, ke < st.cuts.ke_min_pion)
        stop = low | ~known
        if np.any(stop):
            inmat = stop & (L["seg"] >= 0)
            np.add.at(edep, L["seg"][inmat], ke[inmat])
            cnt["stopped"] += int(stop.sum())
            cnt["unknown"] += int((~known & ~low).sum())
            keep(~stop)
            m, ke = m[~stop], ke[~stop]
            if not len(L["pdg"]):
                break
        q = charge(L["pdg"])
        beta = L["p"] / np.hypot(L["p"], m)
        d_dec = np.where(np.isfinite(L["tau"]), L["tau"] * C_MM_NS * L["p"] / np.maximum(m, 1e-300), np.inf)
        inm = L["seg"] >= 0
        h = np.zeros(len(m))
        new_lanes = []
        # ---------------------------------------------------------------- material lanes
        if np.any(inm):
            i = np.flatnonzero(inm)
            xl, dl = target.to_local(L["x"][i], L["u"][i])
            seg = L["seg"][i]
            sp = D.species_index(L["pdg"][i])
            cell = bank.cell(sp, target.mat[seg], ke[i])
            s_in, s_el = bank.per_cell("sigma_inel", cell), bank.per_cell("sigma_el", cell)
            dedx, x0 = bank.per_cell("dedx", cell), bank.per_cell("x0", cell)
            s_tot = s_in + s_el
            with np.errstate(divide="ignore"):
                t_int = np.where(s_tot > 0, -np.log(1.0 - rng.random(i.size)) / s_tot, np.inf)
                t_loss = np.where(dedx > 0, st.cuts.max_loss_fraction * ke[i] / dedx, np.inf)
            t_bnd = target.distance_out(xl, dl, seg)
            step = np.minimum.reduce([t_int, t_loss, t_bnd, d_dec[i], np.full(i.size, st.h_mat)])
            # field kick (first order, B at the start point) and straight move
            B = bfield(L["x"][i])
            u = L["u"][i] + (K_FIELD * q[i] / L["p"][i] * step)[:, None] * np.cross(L["u"][i], B)
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            xnew = L["x"][i] + step[:, None] * L["u"][i]
            de = np.minimum(dedx * step, ke[i])
            np.add.at(edep, seg, de)
            e_new = ke[i] - de + m[i]
            pn = np.sqrt(np.maximum(e_new ** 2 - m[i] ** 2, 0.0))
            bet = np.where(e_new > 0, pn / np.maximum(e_new, 1e-300), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                th0 = 13.6 / np.maximum(bet * pn, 1e-300) * np.sqrt(step / x0) * \
                    (1 + 0.038 * np.log(np.maximum(step / x0, 1e-12)))
            th0 = np.where((step > 0) & (q[i] != 0), np.abs(th0), 0.0)
            th = np.minimum(th0 * np.sqrt(-2.0 * np.log(1.0 - rng.random(i.size))), np.pi)
            u = D.rotate_secondaries(np.stack([np.sin(th), np.zeros(i.size), np.cos(th)], 1), u,
                                     rng.random(i.size) * 2 * np.pi)
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            L["t"][i] += step / (beta[i] * C_MM_NS)
            L["tau"][i] -= np.where(np.isfinite(L["tau"][i]), step * m[i] / (L["p"][i] * C_MM_NS), 0.0)
            L["x"][i], L["u"][i], L["p"][i] = xnew, u, pn
            hit = (t_int <= step) & (t_int <= t_bnd)
            crossed = (t_bnd <= step) & ~hit
            L["x"][i[crossed]] += 1e-6 * L["u"][i[crossed]]
            L["seg"][i] = target.locate(target.to_local(L["x"][i]))
            inel = hit & (rng.random(i.size) * s_tot < s_in)
            el = hit & ~inel
            cnt["elastic"] += int(el.sum())
            if np.any(el):
                k = i[el]
                ct = bank.elastic_cos(cell[el], rng.random(k.size))
                s = np.sqrt(np.maximum(1 - ct * ct, 0.0))
                L["u"][k] = D.rotate_secondaries(np.stack([s, np.zeros_like(s), ct], 1), L["u"][k],
                                                 rng.random(k.size) * 2 * np.pi)
            if np.any(inel):
                cnt["inelastic"] += int(inel.sum())
                k = i[inel]
                j = bank.draw(cell[inel], rng)
                off = bank["sec_offsets"]
                ns = off[j + 1] - off[j]
                rows = np.repeat(off[j], ns) + (np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns))
                par = np.repeat(k, ns)
                np.add.at(edep, L["seg"][k].clip(0), bank["edep_local"][j].astype(float))
                pl = D.rotate_secondaries(bank["sec_p"][rows].astype(float), L["u"][par], rng.random(rows.size) * 2 * np.pi)
                pa = np.linalg.norm(pl, axis=1)
                new_lanes.append(dict(pdg=bank["sec_pdg"][rows].astype(np.int64), x=L["x"][par].copy(),
                                      u=pl / np.maximum(pa, 1e-300)[:, None], p=pa, t=L["t"][par].copy(),
                                      ev=L["ev"][par].copy(), seg=L["seg"][par].copy()))
                L["p"][k] = 0.0                                  # parent consumed: removed below
        # ---------------------------------------------------------------- vacuum lanes
        vac = ~inm
        if np.any(vac):
            i = np.flatnonzero(vac)
            xl = target.to_local(L["x"][i])
            rloc = np.hypot(xl[:, 0], xl[:, 1])
            near = (rloc < rb + st.near) & (np.abs(xl[:, 2] - zc) < hl + st.near)
            hv = np.where(near, st.h_near, np.where(q[i] != 0, st.h_vac, st.neutral_step))
            hv = np.minimum(hv, d_dec[i])
            x0v = L["x"][i]
            ch = q[i] != 0
            neu = np.flatnonzero(~ch)                      # neutral: straight to the next target entry, or dropped
            if neu.size:
                a_, b_ = target.to_local(x0v[neu], L["u"][i[neu]])
                tn = target.distance_in(a_, b_)
                hv[neu] = np.where(np.isfinite(tn), np.minimum(tn + 1e-6, d_dec[i[neu]]), 0.0)
                near[neu] = False
                cnt["neutral_out"] = cnt.get("neutral_out", 0) + int(np.sum(~np.isfinite(tn)))
            xn, un = x0v + hv[:, None] * L["u"][i], L["u"][i].copy()
            if np.any(ch):
                c = np.flatnonzero(ch)
                xn[c], un[c] = _rk4(x0v[c], L["u"][i[c]], L["p"][i[c]], q[i[c]], hv[c], bfield)
            # target re-entry along the chord
            if np.any(near):
                nn = np.flatnonzero(near)
                chord = xn[nn] - x0v[nn]
                clen = np.linalg.norm(chord, axis=1)
                cd = chord / np.maximum(clen, 1e-300)[:, None]
                a, b = target.to_local(x0v[nn], cd)
                ti = target.distance_in(a, b)
                enter = ti < clen
                if np.any(enter):
                    e = nn[enter]
                    f = (ti[enter] + 1e-6) / clen[enter]
                    xn[e] = x0v[e] + (ti[enter] + 1e-6)[:, None] * cd[enter]
                    hv[e] = hv[e] * f
            L["t"][i] += hv / (beta[i] * C_MM_NS)
            L["tau"][i] -= np.where(np.isfinite(L["tau"][i]), hv * m[i] / (L["p"][i] * C_MM_NS), 0.0)
            # record plane crossing (downstream), absorbers
            zp = env.z_plane
            cross = (x0v[:, 2] < zp) & (xn[:, 2] >= zp)
            if enclosure is not None:
                xe = target.to_local(xn)
                cross = (np.hypot(xe[:, 0], xe[:, 1]) >= enclosure[0]) | (np.abs(xe[:, 2] - zc) >= enclosure[1])
                if neu.size:
                    cross[neu[hv[neu] == 0.0]] = True
                c = np.flatnonzero(cross)
                rec["ev"].append(L["ev"][i[c]]); rec["pdg"].append(L["pdg"][i[c]])
                rec["x"].append(np.where((hv[c] == 0)[:, None], x0v[c], xn[c]))
                rec["p"].append(un[c] * L["p"][i[c], None]); rec["t"].append(L["t"][i[c]].copy())
                cnt["recorded"] += c.size
            elif np.any(cross):
                c = np.flatnonzero(cross)
                f = (zp - x0v[c, 2]) / (xn[c, 2] - x0v[c, 2])
                xp = x0v[c] + f[:, None] * (xn[c] - x0v[c])
                up = L["u"][i[c]] + f[:, None] * (un[c] - L["u"][i[c]])
                up /= np.linalg.norm(up, axis=1, keepdims=True)
                tp = L["t"][i[c]] - (1 - f) * hv[c] / (beta[i[c]] * C_MM_NS)
                inside = env.radius(xp) < env.r_plane
                sel = inside & np.isin(L["pdg"][i[c]], RECORD)
                rec["ev"].append(L["ev"][i[c]][sel]); rec["pdg"].append(L["pdg"][i[c]][sel])
                rec["x"].append(xp[sel]); rec["p"].append(up[sel] * L["p"][i[c]][sel, None]); rec["t"].append(tp[sel])
                cnt["recorded"] += int(sel.sum())
                cnt["outside_plane"] += int((~inside).sum())
            L["x"][i], L["u"][i] = xn, un
            L["seg"][i] = target.locate(target.to_local(xn))
            gone = cross.copy()
            if neu.size:
                gone[neu[hv[neu] == 0.0]] = True
            ab = (env.radius(xn) > env.r_max(xn[:, 2])) & ~gone
            up_ = (xn[:, 2] < env.z_min) & ~gone & ~ab
            cnt["absorbed"] += int(ab.sum()); cnt["upstream"] += int(up_.sum())
            L["p"][i[gone | ab | up_]] = 0.0
        # ---------------------------------------------------------------- decays, removal, new lanes
        dec = np.isfinite(L["tau"]) & (L["tau"] <= 1e-9) & (L["p"] > 0) & np.isin(np.abs(L["pdg"]), (211, 321))
        if np.any(dec):
            k = np.flatnonzero(dec)
            cnt["decays"] += k.size
            dd = decay(L["pdg"][k], L["x"][k], L["u"][k], L["p"][k], L["t"][k], rng)
            # parent index per daughter: decay() emits per channel blocks; rebuild by matching positions
            par = dd["parent"]
            new_lanes.append(dict(pdg=dd["pdg"].astype(np.int64), x=dd["x"], u=dd["u"], p=dd["p"], t=dd["t"],
                                  ev=L["ev"][k][par], seg=L["seg"][k][par]))
            L["p"][k] = 0.0
        mu_dec = np.isfinite(L["tau"]) & (L["tau"] <= 1e-9) & (np.abs(L["pdg"]) == 13)
        L["p"][mu_dec] = 0.0
        keep(L["p"] > 0)
        for nl in new_lanes:
            add(nl)
    else:
        raise RuntimeError("chain did not finish within max_iter")
    out = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k in ("x", "p") else 0)) for k, v in rec.items()}
    return dict(record=out, edep=edep, counts=cnt, pot=n0)


def reference(rec, pot, point, shard, z_target=-4004.0):
    """tsdevice reference (ptgt_g4bl.build format, mode translate) from recorded plane states of one shard."""
    n = len(rec["pdg"])
    x = np.asarray(rec["x"], float).copy()
    zs = x[:, 2].copy()
    x[:, 2] = z_target
    order = np.lexsort((np.arange(n), rec["ev"]))
    ref = dict(id=np.stack([np.full(n, point), np.full(n, shard), rec["ev"], np.arange(n)], 1).astype(np.int64)[order],
               x=x[order], p=np.asarray(rec["p"], float)[order], t=np.asarray(rec["t"], float)[order],
               pdg=np.asarray(rec["pdg"], np.int64)[order], shard=np.full(n, shard, np.int64),
               parent_shard_ids=np.array([[point, shard]], np.int64),
               parent_shard_event_counts=np.array([pot], np.int64))
    yc = int(np.sum((ref["pdg"] == 13) | (ref["pdg"] == -211)))
    meta = dict(mode="translate", point=int(point), z_target=float(z_target),
                source_plane_z=float(np.median(zs)) if n else None, files=1, pot=int(pot), rows_read=n, entrances=n,
                helix_returned=0, species={str(s): int(np.sum(ref["pdg"] == s)) for s in RECORD},
                yield_per_pot=yc / float(pot), yield_count=yc, generator="ptgt_chain (device model, host reference)")
    return ref, meta


def rod_target(length, r_up, r_mid, r_dn, z_center=-6164.5, x_axis=3904.0, tilt_deg=14.0, mat=0):
    """The G4beamline rod placed as in the deck: centre on the PS axis at z_center, rotated by tilt_deg about y."""
    a = np.deg2rad(tilt_deg)
    rot = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    return D.Target.rod(length, r_up, r_mid, r_dn, mat=mat, origin=np.array([x_axis, 0.0, z_center]), rot=rot)


def main(argv=None):
    import argparse
    import json
    import time

    from beamfastsim import ptgt_g4bl, tsfield

    ap = argparse.ArgumentParser(description="device target + PS chain for one rod design point (host reference)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rod", nargs=4, type=float, metavar=("TLENGTH", "R_UP", "R_MID", "R_DN"), required=True)
    ap.add_argument("--material", default="G4_W")
    ap.add_argument("--maps", nargs="+", required=True, help="field map .header files, first-containing order")
    ap.add_argument("--env-z", nargs="+", type=float, required=True)
    ap.add_argument("--env-r", nargs="+", type=float, required=True)
    ap.add_argument("--z-min", type=float, required=True)
    ap.add_argument("--r-plane", type=float, required=True)
    ap.add_argument("--z-plane", type=float, default=-4044.45)
    ap.add_argument("--pot", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--h-vac", type=float, default=Stepping.h_vac)
    ap.add_argument("--h-near", type=float, default=Stepping.h_near)
    ap.add_argument("--enclosure", nargs=2, type=float, metavar=("R", "H"),
                    help="diagnostic: record all species leaving this cylinder about the target, instead of the plane")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    bank = D.Bank.load(a.bank)
    if a.material not in bank.materials:
        raise SystemExit(f"material {a.material} not in the bank {bank.materials}")
    tgt = rod_target(*a.rod, mat=bank.materials.index(a.material))
    fb = tsfield.FieldBank(a.maps)
    env = Envelope(z_nodes=np.asarray(a.env_z), r_nodes=np.asarray(a.env_r), z_min=a.z_min, z_plane=a.z_plane,
                   r_plane=a.r_plane)
    st = Stepping(h_vac=a.h_vac, h_near=a.h_near)
    rng = np.random.default_rng(a.seed)
    recs, counts, edep = [], {}, np.zeros(len(tgt.mat))
    t0 = time.time()
    done = 0
    while done < a.pot:
        n = min(a.chunk, a.pot - done)
        prim = primaries(n, tgt, rng)
        prim["event"] = prim["event"] + done
        out = run(bank, tgt, fb, env, prim, rng, st, enclosure=a.enclosure)
        recs.append(out["record"])
        edep += out["edep"]
        for k, v in out["counts"].items():
            counts[k] = counts.get(k, 0) + v
        done += n
    wall = time.time() - t0
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    ref, meta = reference(rec, a.pot, a.point, a.shard)
    meta.update(rod=a.rod, material=a.material, seed=a.seed, counts=counts, edep_per_pot=(edep / a.pot).tolist(),
                wall_s=wall, s_per_pot=wall / a.pot, maps=[str(m) for m in a.maps],
                envelope=dict(z=a.env_z, r=a.env_r, z_min=a.z_min, r_plane=a.r_plane, z_plane=a.z_plane),
                stepping=dict(h_vac=st.h_vac, h_near=st.h_near), bank=dict(path=str(a.bank),
                                                                            content_hash=bank.lib.content_hash))
    ptgt_g4bl.save(a.out, ref, meta)
    print(json.dumps({k: meta[k] for k in ("pot", "entrances", "yield_per_pot", "species", "s_per_pot", "counts")}))


if __name__ == "__main__":
    main()
