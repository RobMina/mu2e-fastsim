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
    annuli: tuple = ()                  # absorbing annuli (z, r_in, r_out) crossed in either direction (thin rings)
    slabs: tuple = ()                   # passive thin rings (z, r_in, r_out, thickness) crossed in either direction:
                                        # loss, scattering and absorption from the tables of a slab material (slab_cross)
    tally: tuple = ()                   # recording-only planes (z, r) (diagnostic): every mu/pi crossing at radius < r,
                                        # in either direction, is recorded with its plane index; the lane goes on

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
    elastic: bool = True                # hadron elastic scattering from the bank (diagnostic switch)
    msc_scale: float = 1.0              # scale on the Highland width (diagnostic switch)
    max_iter: int = 200_000
    cuts: D.Cuts = field(default_factory=D.Cuts)


SLAB_SUBSTEPS = 4
SLAB_MIN_COS = 0.05                     # path through a slab capped at thickness / 0.05 (grazing crossings)


def slab_cross(sbank, mat, pdg, p, u, thickness, rng, n_sub=SLAB_SUBSTEPS):
    """Lanes crossing a thin slab of material `mat` (index in the bank `sbank`, which may hold cross sections only):
    path thickness / |u_z|; inelastic interaction = absorption (no secondaries); mean ionisation loss in n_sub
    substeps; Highland scattering over the path with the geometric mean of the entry and exit p*beta. Lateral
    displacement inside the slab is neglected. Returns (p, u, absorbed)."""
    pdg, p, u = np.asarray(pdg), np.asarray(p, float), np.asarray(u, float)
    m, q = mass(pdg), charge(pdg)
    sp = D.species_index(pdg)
    path = thickness / np.maximum(np.abs(u[:, 2]), SLAB_MIN_COS)
    e = sbank["log10_ke_edges"]
    lo, hi = 10.0 ** e[0], 10.0 ** e[-1]

    def cell(ke):
        return sbank.cell(sp, np.full(sp.shape, mat), np.clip(ke, lo * (1 + 1e-9), hi * (1 - 1e-9)))

    ke0 = np.hypot(p, m) - m
    c0 = cell(ke0)
    absorbed = rng.random(p.size) < -np.expm1(-sbank.per_cell("sigma_inel", c0) * path)
    x0 = sbank.per_cell("x0", c0)
    ke = ke0.copy()
    for _ in range(n_sub):
        ke = ke - sbank.per_cell("dedx", cell(np.maximum(ke, lo))) * path / n_sub
    stopped = ke <= 0
    ke = np.maximum(ke, 0.0)
    pn = np.sqrt(np.maximum((ke + m) ** 2 - m ** 2, 0.0))
    b0 = p / np.hypot(p, m)
    b1 = pn / np.maximum(np.hypot(pn, m), 1e-300)
    pb, bet = np.sqrt(np.maximum(p * b0 * pn * b1, 1e-300)), np.sqrt(b0 * b1)
    with np.errstate(divide="ignore", invalid="ignore"):
        th0 = 13.6 / pb * np.sqrt(path / x0) * (1 + 0.038 * np.log(np.maximum(path / (x0 * np.maximum(bet, 1e-6) ** 2),
                                                                                1e-12)))
    th0 = np.where((q != 0) & (pn > 0), np.abs(th0), 0.0)
    th = np.minimum(th0 * np.sqrt(-2.0 * np.log(1.0 - rng.random(p.size))), np.pi)
    un = D.rotate_secondaries(np.stack([np.sin(th), np.zeros(p.size), np.cos(th)], 1), u, rng.random(p.size) * 2 * np.pi)
    un /= np.linalg.norm(un, axis=1, keepdims=True)
    return pn, un, absorbed | stopped


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
    """G4beamline beam: protons 0.5 mm outside the target's local +z face (z_beam) moving along local -z, Gaussian sigma
    1 mm in the transverse plane, no divergence, t = 0."""
    zl = target.z_beam + offset
    xl = np.stack([sigma * rng.standard_normal(n), sigma * rng.standard_normal(n), np.full(n, zl)], 1)
    dl = np.tile([0.0, 0.0, -1.0], (n, 1))
    x, u = target.to_global(xl, dl)
    m = MASS[2212]
    return dict(pdg=np.full(n, 2212), x=x, u=u, p=np.full(n, np.sqrt((ke + m) ** 2 - m * m)), t=np.zeros(n),
                event=np.arange(n))


def check_tally(env, h_vac):
    """Tally planes and the record plane must be more than one vacuum step apart (one record per lane and step)."""
    z = np.sort(np.r_[[t[0] for t in env.tally], env.z_plane]) if len(env.tally) else np.zeros(0)
    if z.size > 1 and np.min(np.diff(z)) <= h_vac:
        raise ValueError(f"tally planes closer than the vacuum step {h_vac} mm: {z.tolist()}")


def run(bank, target, bfield, env, prim, rng, st=Stepping(), enclosure=None, slab=None):
    """Transport primaries to the record plane. Returns recorded states (event, pdg, x, p, t), deposit per target
    segment and counters. slab=(bank, material index): tables for env.slabs.

    enclosure=(R, H) (diagnostic): instead of the plane, record every lane (all species) where it first leaves the
    cylinder of radius R and half-length H about the target axis (target frame), and stop it there.

    env.tally (diagnostic): mu/pi crossings of recording-only planes, returned as `tally` (with `plane`)."""
    check_tally(env, st.h_vac)
    n0 = len(prim["pdg"])
    L = dict(pdg=np.asarray(prim["pdg"], np.int64), x=np.asarray(prim["x"], float).copy(),
             u=np.asarray(prim["u"], float).copy(), p=np.asarray(prim["p"], float).copy(),
             t=np.asarray(prim["t"], float).copy(), ev=np.asarray(prim["event"], np.int64))
    L["tau"] = -np.log(1 - rng.random(n0)) * lifetime(L["pdg"])
    L["seg"] = target.locate(target.to_local(L["x"]))
    rec = dict(ev=[], pdg=[], x=[], p=[], t=[])
    tal = dict(ev=[], pdg=[], plane=[], x=[], p=[], t=[])
    edep = np.zeros(len(target.mat))
    cnt = dict(inelastic=0, elastic=0, stopped=0, decays=0, absorbed=0, upstream=0, recorded=0, outside_plane=0,
               unknown=0, iterations=0, tallied=0)
    zc = 0.5 * (target.z_lo + target.z_hi)
    hl = 0.5 * (target.z_hi - target.z_lo)
    rb = target.r_max

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
            if not st.elastic:
                s_el = np.zeros_like(s_el)
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
            with np.errstate(divide="ignore", invalid="ignore"):   # Highland (PDG), log term with x / (X0 beta^2)
                th0 = st.msc_scale * 13.6 / np.maximum(bet * pn, 1e-300) * np.sqrt(step / x0) * \
                    (1 + 0.038 * np.log(np.maximum(step / (x0 * np.maximum(bet, 1e-6) ** 2), 1e-12)))
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
            near = (rloc < rb + st.near) & (np.abs(xl[:, 2] - zc) < hl + st.near) | target.cylinder_near(xl, st.near)
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
            # passive slabs: the step ends just past the first slab plane crossed inside the ring, where the lane
            # loses energy, scatters or is absorbed
            sl_h, sl_abs = np.zeros(0, np.int64), np.zeros(0, bool)
            if env.slabs:
                if slab is None:
                    raise ValueError("envelope slabs need slab=(bank, material index)")
                fs, sid = np.full(i.size, np.inf), np.full(i.size, -1)
                for k_, (za, ra, rb_, _th) in enumerate(env.slabs):
                    hz = ((x0v[:, 2] - za) * (xn[:, 2] - za) <= 0) & (xn[:, 2] != x0v[:, 2])
                    if not np.any(hz):
                        continue
                    fa = np.where(hz, (za - x0v[:, 2]) / np.where(hz, xn[:, 2] - x0v[:, 2], 1.0), np.inf)
                    ra_ = env.radius(x0v + np.where(hz, fa, 0.0)[:, None] * (xn - x0v))
                    hit = hz & (ra_ >= ra) & (ra_ < rb_) & (fa < fs)
                    fs[hit], sid[hit] = fa[hit], k_
                sl_h = np.flatnonzero(sid >= 0)
                if sl_h.size:
                    f = fs[sl_h]
                    us = L["u"][i[sl_h]] + f[:, None] * (un[sl_h] - L["u"][i[sl_h]])
                    us /= np.linalg.norm(us, axis=1, keepdims=True)
                    xs = x0v[sl_h] + f[:, None] * (xn[sl_h] - x0v[sl_h])
                    xs[:, 2] += np.where(us[:, 2] >= 0, 1e-6, -1e-6)
                    hv[sl_h] = hv[sl_h] * f
                    th_ = np.asarray([env.slabs[k][3] for k in sid[sl_h]])
                    p_sl, u_sl, sl_abs = slab_cross(slab[0], slab[1], L["pdg"][i[sl_h]], L["p"][i[sl_h]], us, th_, rng)
                    xn[sl_h], un[sl_h] = xs, u_sl
                    cnt["slab_crossed"] = cnt.get("slab_crossed", 0) + int(sl_h.size)
                    cnt["slab_absorbed"] = cnt.get("slab_absorbed", 0) + int(sl_abs.sum())
            L["t"][i] += hv / (beta[i] * C_MM_NS)
            L["tau"][i] -= np.where(np.isfinite(L["tau"][i]), hv * m[i] / (L["p"][i] * C_MM_NS), 0.0)
            if sl_h.size:
                L["p"][i[sl_h]] = p_sl
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
            for k_, (zt, rt) in enumerate(env.tally if enclosure is None else ()):
                c = np.flatnonzero((x0v[:, 2] < zt) != (xn[:, 2] < zt))
                if not c.size:
                    continue
                f = (zt - x0v[c, 2]) / (xn[c, 2] - x0v[c, 2])
                xp = x0v[c] + f[:, None] * (xn[c] - x0v[c])
                up = L["u"][i[c]] + f[:, None] * (un[c] - L["u"][i[c]])
                up /= np.linalg.norm(up, axis=1, keepdims=True)
                tp = L["t"][i[c]] - (1 - f) * hv[c] / (beta[i[c]] * C_MM_NS)
                sel = (env.radius(xp) < rt) & np.isin(L["pdg"][i[c]], RECORD)
                tal["ev"].append(L["ev"][i[c]][sel]); tal["pdg"].append(L["pdg"][i[c]][sel])
                tal["plane"].append(np.full(int(sel.sum()), k_, np.int64))
                tal["x"].append(xp[sel]); tal["p"].append(up[sel] * L["p"][i[c]][sel, None]); tal["t"].append(tp[sel])
                cnt["tallied"] += int(sel.sum())
            L["x"][i], L["u"][i] = xn, un
            L["seg"][i] = target.locate(target.to_local(xn))
            gone = cross.copy()
            if neu.size:
                gone[neu[hv[neu] == 0.0]] = True
            ab = (env.radius(xn) > env.r_max(xn[:, 2])) & ~gone
            if sl_h.size:
                ab[sl_h[sl_abs]] = True
            for za, ra, rb_ in env.annuli:
                hitz = ((x0v[:, 2] - za) * (xn[:, 2] - za) <= 0) & (xn[:, 2] != x0v[:, 2])
                if np.any(hitz):
                    fz = np.where(hitz, (za - x0v[:, 2]) / np.where(hitz, xn[:, 2] - x0v[:, 2], 1.0), 0.0)
                    ra_ = env.radius(x0v + fz[:, None] * (xn - x0v))
                    ab |= hitz & (ra_ >= ra) & (ra_ < rb_) & ~gone
                    cnt["annulus"] = cnt.get("annulus", 0) + int((hitz & (ra_ >= ra) & (ra_ < rb_) & ~gone).sum())
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
    cat = lambda d: {k: (np.concatenate(v) if v else np.zeros((0, 3) if k in ("x", "p") else 0)) for k, v in d.items()}
    return dict(record=cat(rec), edep=edep, counts=cnt, pot=n0, tally=cat(tal))


def save_tally(path, tally, pot, env, meta):
    """Tally crossings (Mu2e frame) with the planes and POT, for the plane-by-plane comparison."""
    import json
    np.savez_compressed(path, ev=tally["ev"], pdg=tally["pdg"], plane=tally["plane"], x=tally["x"], p=tally["p"],
                        t=tally["t"], pot=np.int64(pot), planes=np.asarray(env.tally, float).reshape(-1, 2),
                        meta=json.dumps(meta, default=str))


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


def placement(z_center=-6164.5, x_axis=3904.0, tilt_deg=14.0, offset_x=0.0, tilt_x_deg=0.0):
    """Target placement of the deck: centre on the PS axis at z_center (moved by offset_x along x), rotated by
    tilt_x_deg about x and then tilt_deg about y (fixed axes: rot = Ry @ Rx, the deck's rotation=X..,Y..)."""
    a, b = np.deg2rad(tilt_deg), np.deg2rad(tilt_x_deg)
    ry = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
    return dict(origin=np.array([x_axis + offset_x, 0.0, z_center]), rot=ry @ rx)


def rod_target(length, r_up, r_mid, r_dn, z_center=-6164.5, x_axis=3904.0, tilt_deg=14.0, mat=0, offset_x=0.0,
               tilt_x_deg=0.0):
    """The G4beamline rod placed as in the deck (placement; offset_x, tilt_x_deg: a misalignment)."""
    return D.Target.rod(length, r_up, r_mid, r_dn, mat=mat,
                        **placement(z_center, x_axis, tilt_deg, offset_x, tilt_x_deg))


def add_design_arguments(ap):
    """Target design options beyond --rod (shared by the host and device chains)."""
    ap.add_argument("--offset-x", type=float, default=0.0, help="target moved along x [mm]; the beam stays on the "
                    "nominal axis (the deck's Toffset_x)")
    ap.add_argument("--tilt-x", type=float, default=0.0, help="target rotated about x [deg] before the y tilt; the beam "
                    "stays on the nominal axis (the deck's Ttilt_x)")
    ap.add_argument("--deck-target", metavar="FILE", help="the target group (pTarget) of a generated G4beamline target "
                    "file (e.g. Geometry/Proton_Target_Hayman2.txt), placed on the deck's target centre (--rod unused)")
    ap.add_argument("--deck-rotation", default="Y14", help="rotation of that group in the deck (G4beamline string, "
                    "e.g. 'Z-30,Y14' for Hayman2)")
    ap.add_argument("--ral", action="store_true", help="the deck's RAL target (Use_Proton_Target=6): rod, hubs, hub "
                    "balls and spokes in --material (ptgt_deck.ral_target; --rod unused)")
    ap.add_argument("--clad", nargs=3, metavar=("R_CORE", "T_CLAD", "MATERIAL"),
                    help="a cylinder of radius R_CORE (--material) in a sleeve of thickness T_CLAD of MATERIAL, the "
                         "length of --rod (its radii unused): the deck's Use_Proton_Target=10")


def design_targets(a, bank, rod):
    """(target, beam target) of the command-line design: the beam is aimed at the nominal placement of the same
    shape, so a misalignment moves the target only."""
    off = dict(offset_x=a.offset_x, tilt_x_deg=a.tilt_x)
    if getattr(a, "deck_target", None):
        from beamfastsim import ptgt_deck
        if a.offset_x or a.tilt_x or a.clad:
            raise SystemExit("--deck-target takes no --offset-x, --tilt-x or --clad")
        p0 = placement(tilt_deg=0.0)
        t = ptgt_deck.target(a.deck_target, list(bank.materials), origin=p0["origin"],
                             rot=ptgt_deck.rotation(a.deck_rotation))
        return t, t
    mat = bank.materials.index(a.material)
    if getattr(a, "ral", False):
        from beamfastsim import ptgt_deck
        if a.offset_x or a.tilt_x or a.clad:
            raise SystemExit("--ral takes no --offset-x, --tilt-x or --clad")
        t = ptgt_deck.ral_target(mat, **placement())
        return t, t
    if a.clad:
        r_core, t_clad, cmat = float(a.clad[0]), float(a.clad[1]), a.clad[2]
        if cmat not in bank.materials:
            raise SystemExit(f"clad material {cmat} not in the bank {bank.materials}")

        def make(**kw):
            return D.Target.clad_rod(rod[0], r_core, t_clad, mat, bank.materials.index(cmat), **placement(**kw))
        return make(**off), make()
    return rod_target(*rod, mat=mat, **off), rod_target(*rod, mat=mat)


def load_slab(slabs, path, material):
    """(bank, material index) for --slab, or None without slabs."""
    if not slabs:
        return None
    if not path:
        raise SystemExit("--slab needs --slab-bank")
    sb = D.Bank.load(path)
    mat = material or sb.materials[0]
    if mat not in sb.materials:
        raise SystemExit(f"slab material {mat} not in {sb.materials}")
    return sb, sb.materials.index(mat)


def slab_meta(slab, path):
    if slab is None:
        return None
    return dict(path=str(path), material=slab[0].materials[slab[1]], content_hash=slab[0].lib.content_hash)


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
    ap.add_argument("--no-elastic", action="store_true", help="diagnostic: no hadron elastic scattering")
    ap.add_argument("--msc-scale", type=float, default=1.0, help="diagnostic: scale on the Highland width")
    ap.add_argument("--h-mat", type=float, default=Stepping.h_mat)
    ap.add_argument("--ke-min-hadron", type=float, default=D.Cuts.ke_min_hadron,
                    help="p, n transport cut [MeV] (the bank must hold interactions down to it)")
    ap.add_argument("--annulus", nargs=3, type=float, action="append", default=[], metavar=("Z", "R_IN", "R_OUT"),
                    help="absorbing thin annulus at z (repeatable)")
    ap.add_argument("--slab", nargs=4, type=float, action="append", default=[], metavar=("Z", "R_IN", "R_OUT", "THICK"),
                    help="passive thin ring at z (repeatable): loss, scattering, absorption from --slab-bank")
    ap.add_argument("--slab-bank", help="bank with the slab material's tables (cross sections only is enough)")
    ap.add_argument("--slab-material", help="slab material in --slab-bank (default: its first)")
    add_design_arguments(ap)
    ap.add_argument("--enclosure", nargs=2, type=float, metavar=("R", "H"),
                    help="diagnostic: record all species leaving this cylinder about the target, instead of the plane")
    ap.add_argument("--tally", nargs=2, type=float, action="append", default=[], metavar=("Z", "R"),
                    help="diagnostic: recording-only plane (repeatable); mu/pi crossings both ways go to --tally-out")
    ap.add_argument("--tally-out", help="npz for the --tally crossings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    slab = load_slab(a.slab, a.slab_bank, a.slab_material)
    bank = D.Bank.load(a.bank)
    if a.material not in bank.materials:
        raise SystemExit(f"material {a.material} not in the bank {bank.materials}")
    tgt, beam_tgt = design_targets(a, bank, a.rod)
    fb = tsfield.FieldBank(a.maps)
    env = Envelope(z_nodes=np.asarray(a.env_z), r_nodes=np.asarray(a.env_r), z_min=a.z_min, z_plane=a.z_plane,
                   r_plane=a.r_plane, annuli=tuple(tuple(x) for x in a.annulus), slabs=tuple(tuple(x) for x in a.slab),
                   tally=tuple(tuple(x) for x in a.tally))
    if a.tally and not a.tally_out:
        raise SystemExit("--tally needs --tally-out")
    st = Stepping(h_vac=a.h_vac, h_near=a.h_near, h_mat=a.h_mat, elastic=not a.no_elastic, msc_scale=a.msc_scale,
                  cuts=D.Cuts(ke_min_hadron=a.ke_min_hadron))
    rng = np.random.default_rng(a.seed)
    recs, tals, counts, edep = [], [], {}, np.zeros(len(tgt.mat))
    t0 = time.time()
    done = 0
    while done < a.pot:
        n = min(a.chunk, a.pot - done)
        prim = primaries(n, beam_tgt, rng)
        prim["event"] = prim["event"] + done
        out = run(bank, tgt, fb, env, prim, rng, st, enclosure=a.enclosure, slab=slab)
        recs.append(out["record"]); tals.append(out["tally"])
        edep += out["edep"]
        for k, v in out["counts"].items():
            counts[k] = counts.get(k, 0) + v
        done += n
    wall = time.time() - t0
    rec = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    ref, meta = reference(rec, a.pot, a.point, a.shard)
    meta.update(rod=a.rod, material=a.material, seed=a.seed, counts=counts, edep_per_pot=(edep / a.pot).tolist(),
                wall_s=wall, s_per_pot=wall / a.pot, maps=[str(m) for m in a.maps],
                envelope=dict(z=a.env_z, r=a.env_r, z_min=a.z_min, r_plane=a.r_plane, z_plane=a.z_plane, annuli=a.annulus,
                              slabs=a.slab, slab_bank=slab_meta(slab, a.slab_bank)),
                stepping=dict(h_vac=st.h_vac, h_near=st.h_near, h_mat=st.h_mat, elastic=st.elastic,
                              msc_scale=st.msc_scale), bank=dict(path=str(a.bank),
                                                                            content_hash=bank.lib.content_hash))
    ptgt_g4bl.save(a.out, ref, meta)
    if a.tally:
        save_tally(a.tally_out, {k: np.concatenate([r[k] for r in tals]) for k in tals[0]}, a.pot, env,
                   dict(generator="ptgt_chain", seed=a.seed, rod=a.rod, counts=counts))
    print(json.dumps({k: meta[k] for k in ("pot", "entrances", "yield_per_pot", "species", "s_per_pot", "counts")}))


if __name__ == "__main__":
    main()
