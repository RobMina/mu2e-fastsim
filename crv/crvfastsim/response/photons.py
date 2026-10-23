"""CRV photons at the SiPMs from CrvSteps: a vectorised port of Offline CRVResponse MakeCrvPhotons + CrvPhotonGenerator.

Per step and simulated SiPM (stock loop, MakeCrvPhotons::MakePhotons):
  1. local coordinates in the bar (thickness, width, length); the table describes SiPM 0, the other SiPMs by mirroring
     (1: z -> -z, 2: y -> -y, 3: both). With a reflector only SiPMs 0 and 2 are simulated; reflector < 0 flips z first
     and stores SiPM s's photons at s + 1 (its photon-yield deviation is still that of channel s, as in the stock code).
  2. photon numbers: scintillation Y * edep_visible, Cerenkov (scintillator and fibre) photons/mm(beta) * |q| * path;
     one Gauss (mean > 10) or Poisson draw per type, divided by the number of 0.1 mm substeps and rounded (lrint).
  3. each substep: in the scintillator (box, minus the elliptical fibre holes and rounded corners) the scintillation
     and scintillator-Cerenkov bins; else in a fibre the fibre-Cerenkov bin (beta, theta, phi, r, z).
  4. each photon arrives with the bin probability x (deviation * variationScale + 1) * photonYieldScaleFactor; its time
     is the substep time + fibre-emission decays (decay time 0 in the v6 tables) + a delay drawn from the bin.
  5. light propagation per bar relative to its table (Offline CRVLightPropagation / DbTable CRVPropagation,
     MakeCrvPhotons::SetLightPropagation): speed ratio r_v adds d / v_table (1 / r_v - 1) to the arrival time, d the
     distance to the SiPM end; attenuation-length ratio r_a multiplies the probability by
     exp(-z (1 / r_a - 1) / L_table), z along the bar from its centre. v_table and L_table are measured from the
     table (table_propagation).
Arrivals are drawn as Binomial(n, p) per substep and photon type (the stock code draws n Bernoullis: same law).
Vectorised over steps x SiPMs x substeps in chunks; no per-step loops.
"""
from dataclasses import dataclass, field

import numpy as np

from . import lookup

PRECISION_MM = 0.1


@dataclass
class SectorConfig:
    table: str
    reflector: int
    yield_per_mev: float


# Offline CRVResponse/fcl/epilog_extracted_v05.fcl (included by Production digitize/Extracted.fcl)
EXTRACTED_V05 = {
    "EX": SectorConfig("6000_0", 0, 39794.0), "T1": SectorConfig("2370_1", -1, 28610.0),
    "T2": SectorConfig("3200_0", 0, 28610.0),
    **{f"M{i}": SectorConfig("2370_1", 1, 28610.0) for i in range(1, 9)},
}


@dataclass
class ResponseConfig:
    sectors: dict = field(default_factory=lambda: dict(EXTRACTED_V05))
    photon_yield_scale: float = 0.94
    photon_yield_variation_scale: float = 1.0
    # light propagation relative to the lookup tables, {key: ratio}; key = sector (EX, T1, T2, M1..M8; TAG = all M*) or
    # module (sector name + module index for sectors with several modules: EX0..EX3, T10, T11, T20, T21); module wins
    speed_ratio: dict = field(default_factory=dict)
    atten_ratio: dict = field(default_factory=dict)


# pdg -> (mass MeV, charge e); ions 100ZZZAAAI handled in particle_props
_PDG = {11: (0.51099895, -1), -11: (0.51099895, 1), 13: (105.6583755, -1), -13: (105.6583755, 1),
        22: (0.0, 0), 211: (139.57039, 1), -211: (139.57039, -1), 111: (134.9768, 0), 321: (493.677, 1),
        -321: (493.677, -1), 130: (497.611, 0), 310: (497.611, 0), 2212: (938.272088, 1), -2212: (938.272088, -1),
        2112: (939.565420, 0), -2112: (939.565420, 0), 12: (0, 0), -12: (0, 0), 14: (0, 0), -14: (0, 0),
        3122: (1115.683, 0), 3222: (1189.37, 1), 3112: (1197.449, -1)}
AMU = 931.49410242


def particle_props(pdg):
    """Mass [MeV] and charge [e] per pdg code (ions: 100ZZZAAAI -> charge Z, mass A * amu). Unknown -> (0, 0)."""
    pdg = np.asarray(pdg, np.int64)
    m, q = np.zeros(len(pdg)), np.zeros(len(pdg))
    for code, (mm, qq) in _PDG.items():
        s = pdg == code
        m[s], q[s] = mm, qq
    ion = np.abs(pdg) >= 1_000_000_000
    a = (np.abs(pdg[ion]) // 10) % 1000
    z = (np.abs(pdg[ion]) // 10000) % 1000
    m[ion], q[ion] = a * AMU, z * np.sign(pdg[ion])
    known = np.isin(pdg, list(_PDG)) | ion
    return m, q, known


def beta_of(p0, p1, mass):
    """Stock beta: from the mean of the start and end energies; massless -> 1."""
    e = 0.5 * (np.sqrt(p0 ** 2 + mass ** 2) + np.sqrt(p1 ** 2 + mass ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(mass > 0, e / np.where(mass > 0, mass, 1.0), np.inf)
        return np.sqrt(np.clip(1.0 - 1.0 / g ** 2, 0.0, 1.0))


def cerenkov_per_mm(beta, charge, cmap):
    """Stock GetAverageNumberOfCerenkovPhotons: linear in beta from (0, 0) through the map, flat beyond its end."""
    b = np.r_[0.0, cmap[:, 0]]
    n = np.r_[0.0, cmap[:, 1]]
    return np.interp(beta, b, n, right=n[-1]) * np.abs(charge)


def find_bin(edges, v):
    """Stock findBin: the first i with edges[i] <= v <= edges[i+1] (both inclusive); -1 if outside."""
    i = np.searchsorted(edges, v, side="left")
    i = np.where(v == edges[0], 1, i)
    return np.where((i >= 1) & (i <= len(edges) - 1), i - 1, -1)


def _draw_counts(avg, rng):
    """Stock GetNumberOfPhotonsFromAverage before the division: Gauss(avg, sqrt avg) if avg > 10, else Poisson."""
    out = np.zeros(len(avg))
    big = avg > 10.0
    out[big] = rng.normal(avg[big], np.sqrt(avg[big]))
    out[~big] = rng.poisson(np.maximum(avg[~big], 0.0))
    return out


def _per_substep(avg, nsub, rng):
    return np.maximum(np.rint(_draw_counts(avg, rng) / nsub), 0).astype(np.int64)


def _sample_delays(T, bins, rng):
    """Delay index per arrival: first j with u * sum <= cumsum_j (stock GetRandomTime), u ~ U[0, 1)."""
    cum = np.cumsum(T.delays[bins].astype(np.int32), axis=1)
    thr = rng.random(len(bins)) * T.delay_sum[bins]
    return (cum < thr[:, None]).sum(axis=1).astype(float)


def _sample_emissions(T, bins, rng):
    cum = np.cumsum(T.emis[bins].astype(np.int32), axis=1)
    thr = rng.random(len(bins)) * T.emis_sum[bins]
    return (cum < thr[:, None]).sum(axis=1)


def _inside_scint(L, x, y, z):
    ax, ay = np.abs(x), np.abs(y)
    ok = (ax < L.half_thickness) & (ay < L.half_width) & (np.abs(z) < L.half_length)
    ell = (ax / L.hole_radius_x) ** 2 + ((ay - L.fiber_separation / 2) / L.hole_radius_y) ** 2
    ok &= ell >= 1.0
    cx, cy = L.half_thickness - L.corner_radius, L.half_width - L.corner_radius
    corner = (ax > cx) & (ay > cy) & (np.hypot(ax - cx, ay - cy) > L.corner_radius)
    return ok & ~corner


def _inside_fiber(L, x, y, z, dx, dy):
    """In a fibre: (flag, r, phi); phi is the angle between the step direction and the radius vector in (x, y)."""
    ok = np.abs(z) < L.half_length
    r0 = np.hypot(x, y + L.fiber_separation / 2)
    r1 = np.hypot(x, y - L.fiber_separation / 2)
    in0, in1 = r0 <= L.fiber_radius, r1 <= L.fiber_radius
    ry = np.where(in1, y - L.fiber_separation / 2, y + L.fiber_separation / 2)
    r = np.where(in1, r1, r0)
    norm = np.hypot(dx, dy) * r
    cosang = np.where(norm > 0, (dx * x + dy * ry) / np.where(norm > 0, norm, 1.0), 1.0)
    phi = np.where(norm > 0, np.arccos(np.clip(cosang, -1.0, 1.0)), 0.0)
    return ok & (in0 | in1), r, phi


def _substep_photons(L, rows, rng, max_sub=4_000_000):
    """rows: dict of per-(step, SiPM) arrays (local start/end, t0, t1, n per substep per type, beta, devfactor, out
    channel, step index). Returns (step index, channel, time) of the arriving photons."""
    nsub = rows["nsub"]
    out_s, out_c, out_t = [], [], []
    starts = np.r_[0, np.cumsum(nsub)]
    # chunk boundaries on rows so each chunk has <= max_sub substeps
    cuts = [0]
    while cuts[-1] < len(nsub):
        nxt = int(np.searchsorted(starts, starts[cuts[-1]] + max_sub, side="right")) - 1
        cuts.append(max(nxt, cuts[-1] + 1))
    E = L.edges
    nb = {k: len(v) - 1 for k, v in E.items()}
    Ts, Tc, Tf = (L.tables[k] for k in lookup.TABLES)
    mu = rows.get("mu")
    if mu is not None and not np.any(mu):
        mu = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = nsub[a:b]
        ri = np.repeat(np.arange(a, b), n)
        k = np.arange(len(ri)) - np.repeat(starts[a:b] - starts[a], n)
        frac = (k + 0.5) / nsub[ri]
        p = rows["s"][ri] + rows["d"][ri] * frac[:, None]
        t = rows["t0"][ri] + (rows["t1"][ri] - rows["t0"][ri]) * frac
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        insc = _inside_scint(L, x, y, z)
        infb, r, phi = _inside_fiber(L, x, y, z, rows["d"][ri, 0], rows["d"][ri, 1])
        infb &= ~insc
        bx, by, bz = find_bin(E["x"], np.abs(x)), find_bin(E["y"], y), find_bin(E["z"], z)
        bb = find_bin(E["beta"], rows["beta"][ri])
        geo = (bx >= 0) & (by >= 0) & (bz >= 0)
        sbin = np.where(insc & geo, bz + by * nb["z"] + bx * nb["y"] * nb["z"], -1)
        cbin = np.where(insc & geo & (bb >= 0), bb + bz * nb["beta"] + by * nb["z"] * nb["beta"]
                        + bx * nb["y"] * nb["z"] * nb["beta"], -1)
        bth, bph, br = find_bin(E["theta"], rows["theta"][ri]), find_bin(E["phi"], phi), find_bin(E["r"], r)
        fok = infb & (bb >= 0) & (bth >= 0) & (bph >= 0) & (br >= 0) & (bz >= 0)
        fbin = np.where(fok, bz + br * nb["z"] + bph * nb["r"] * nb["z"] + bth * nb["phi"] * nb["r"] * nb["z"]
                        + bb * nb["theta"] * nb["phi"] * nb["r"] * nb["z"], -1)
        for T, bins, ncol in ((Ts, sbin, "n_scint"), (Tc, cbin, "n_ck_s"), (Tf, fbin, "n_ck_f")):
            m = bins >= 0
            if not m.any():
                continue
            nn = rows[ncol][ri[m]]
            pr = T.prob[bins[m]].astype(float) * rows["dev"][ri[m]]
            if mu is not None:
                pr = pr * np.exp(-mu[ri[m]] * z[m])
            pr = np.minimum(pr, 1.0)
            narr = rng.binomial(nn, pr)
            if not narr.any():
                continue
            sel = np.repeat(np.flatnonzero(m), narr)
            bsel = np.repeat(bins[m], narr)
            tt = t[sel] + _sample_delays(T, bsel, rng)
            if L.fiber_decay_time > 0:
                ne = _sample_emissions(T, bsel, rng)
                tt += rng.gamma(np.maximum(ne, 1e-12), L.fiber_decay_time) * (ne > 0)
            out_s.append(rows["step"][ri[sel]])
            out_c.append(rows["chan"][ri[sel]])
            out_t.append(tt)
    if not out_s:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(out_s), np.concatenate(out_c), np.concatenate(out_t)


_TABLE_PROP = {}


def table_propagation(L):
    """(inverse signal speed [ns/mm], attenuation length [mm]) of a table, as MakeCrvPhotons::FindTablePropagation:
    slope of the 10 % quantile of the scintillation delays vs the distance d to the SiPM end, and -1 / slope of the log
    of the summed arrival probability vs d (0.2 < d / half length < 1.8)."""
    key = (L.name, L.half_length)
    if key in _TABLE_PROP:
        return _TABLE_PROP[key]
    E, T = L.edges, L.tables["scint"]
    nx, ny, nz = len(E["x"]) - 1, len(E["y"]) - 1, len(E["z"]) - 1
    p = T.prob.astype(np.float64).reshape(nx, ny, nz)
    ds = T.delay_sum.astype(np.float64).reshape(nx, ny, nz)
    w = np.where(ds > 0, p / np.where(ds > 0, ds, 1.0), 0.0)
    h = np.einsum("xyz,xyzj->zj", w, T.delays.astype(np.float64).reshape(nx, ny, nz, -1))
    d = 0.5 * (E["z"][1:] + E["z"][:-1]) + L.half_length
    tot = h.sum(1)
    ok = tot > 0
    q = np.argmax(np.cumsum(h, 1) >= 0.1 * tot[:, None], axis=1)
    inv_v = np.polyfit(d[ok], q[ok].astype(float), 1)[0]
    P = p.sum((0, 1))
    a = (P > 0) & (d > 0.2 * L.half_length) & (d < 1.8 * L.half_length)
    sl = np.polyfit(d[a], np.log(P[a]), 1)[0]
    if not sl < 0:
        raise ValueError(f"cannot determine the attenuation length of {L.name}")
    _TABLE_PROP[key] = (float(inv_v), float(-1.0 / sl))
    return _TABLE_PROP[key]


def bar_keys(geom):
    """Per bar: (sector key, module key); module key = sector name + module index if the sector has several modules."""
    names = np.array(geom.sector_names, dtype=object)
    nmod = np.zeros(len(names), np.int64)
    np.maximum.at(nmod, geom.sector, geom.module + 1)
    sk = names[geom.sector]
    mk = np.where(nmod[geom.sector] > 1, sk + geom.module.astype(str).astype(object), sk)
    return sk, mk


def bar_ratios(values, geom):
    """Per-bar ratio array from {key: ratio} (keys as ResponseConfig.speed_ratio); unlisted bars 1."""
    out = np.ones(geom.nbars)
    if not values:
        return out
    sk, mk = bar_keys(geom)
    known = set(sk) | set(mk) | {"TAG"}
    bad = sorted(set(values) - known)
    if bad:
        raise KeyError(f"unknown propagation keys {bad}; known: {sorted(known)}")
    tag = np.array([str(n).startswith("M") for n in sk])
    for k, v in values.items():
        if k == "TAG":
            out[tag] = v
    for k, v in values.items():
        if k != "TAG" and k in set(sk):
            out[sk == k] = v
    for k, v in values.items():
        if k in set(mk) and k not in set(sk):
            out[mk == k] = v
    if not np.all(out > 0):
        raise ValueError("propagation ratios must be positive")
    return out


def step_inputs(steps, pdg, geom):
    """Per step: local start and end (thickness, width, length) in the bar, beta, charge, and the known-pdg flag."""
    bar = steps.bar.to_numpy().astype(np.int64)
    l2w = geom.local_to_world[bar]
    r = np.arange(len(bar))
    c = geom.center[bar]
    s_loc = np.stack([(steps[f"{a}0"].to_numpy() - c[:, j]) for j, a in enumerate("xyz")], 1)[r[:, None], l2w]
    e_loc = np.stack([(steps[f"{a}1"].to_numpy() - c[:, j]) for j, a in enumerate("xyz")], 1)[r[:, None], l2w]
    mass, charge, known = particle_props(pdg)
    p0 = np.sqrt(steps.px0.to_numpy() ** 2 + steps.py0.to_numpy() ** 2 + steps.pz0.to_numpy() ** 2)
    return s_loc, e_loc, beta_of(p0, steps.p1.to_numpy(), mass), charge, known


def iter_rows(steps, pdg, geom, cfg, tables, deviation, rng):
    """Yield (lookup table, rows) per (table, reflector, simulated SiPM): the per-(step, SiPM) inputs of the substep
    loop, with the photon numbers per substep already drawn (rng). Shared by the numpy and JAX substep kernels."""
    bar = steps.bar.to_numpy().astype(np.int64)
    s_loc, e_loc, beta, charge, _ = step_inputs(steps, pdg, geom)
    edep, length = steps.edep.to_numpy(), steps.len.to_numpy()
    t0, t1 = steps.t0.to_numpy(), steps.t1.to_numpy()
    names = list(geom.sector_names)
    sec = geom.sector[bar]
    good = np.isfinite(t0) & np.isfinite(t1)
    r_speed, r_atten = bar_ratios(cfg.speed_ratio, geom)[bar], bar_ratios(cfg.atten_ratio, geom)[bar]
    for tname in sorted({cfg.sectors[n].table for n in names}):
        L = tables[tname]
        for refl in sorted({cfg.sectors[n].reflector for n in names if cfg.sectors[n].table == tname}):
            sids = [i for i, n in enumerate(names) if cfg.sectors[n].table == tname and cfg.sectors[n].reflector == refl]
            m = np.isin(sec, sids) & good
            if not m.any():
                continue
            if abs(refl) != L.reflector:
                raise ValueError(f"reflector {refl} does not match lookup table {tname} ({L.reflector})")
            Y = np.array([cfg.sectors[names[i]].yield_per_mev for i in range(len(names))])[sec[m]]
            idx = np.flatnonzero(m)
            inv_v, att = table_propagation(L)
            k = inv_v * (1.0 / r_speed[idx] - 1.0)                  # extra delay per mm to the SiPM end [ns/mm]
            mu = (1.0 / r_atten[idx] - 1.0) / att                   # extra attenuation, pivot at the bar centre [1/mm]
            sl, el = s_loc[m].copy(), e_loc[m].copy()
            if refl < 0:
                sl[:, 2], el[:, 2] = -sl[:, 2], -el[:, 2]
            for s in ((0, 1, 2, 3) if refl == 0 else (0, 2)):
                a, b = sl.copy(), el.copy()
                if s in (1, 3):
                    a[:, 2], b[:, 2] = -a[:, 2], -b[:, 2]
                if s in (2, 3):
                    a[:, 1], b[:, 1] = -a[:, 1], -b[:, 1]
                d = b - a
                dl = np.linalg.norm(d, axis=1)
                theta = np.where(dl > 0, np.arccos(np.clip(d[:, 2] / np.where(dl > 0, dl, 1.0), -1, 1)), 0.0)
                nsub = np.maximum((dl / PRECISION_MM).astype(np.int64), 1)
                ch_dev = bar[idx] * 4 + s
                # speed: in the table frame (z flipped for sides 1, 3) the readout is at z = -half_length, so the delay
                # k d is linear in z: exact through t0/t1
                ta = t0[idx] + k * (a[:, 2] + L.half_length)
                tb = t1[idx] + k * (b[:, 2] + L.half_length)
                dev = (deviation[ch_dev] * cfg.photon_yield_variation_scale + 1.0) * cfg.photon_yield_scale
                yield L, {"s": a, "d": d, "t0": ta, "t1": tb, "nsub": nsub, "beta": beta[idx],
                          "theta": theta, "dev": dev, "mu": mu, "step": idx,
                          "chan": bar[idx] * 4 + (s + 1 if refl < 0 else s),
                          "n_scint": _per_substep(Y * edep[idx], nsub, rng),
                          "n_ck_s": _per_substep(cerenkov_per_mm(beta[idx], charge[idx], L.ck_scint) * length[idx],
                                                 nsub, rng),
                          "n_ck_f": _per_substep(cerenkov_per_mm(beta[idx], charge[idx], L.ck_fiber) * length[idx],
                                                 nsub, rng)}


def make_photons(steps, pdg, geom, cfg, tables, deviation, rng):
    """Photons at the SiPMs for a CrvStep table.

    steps: DataFrame with bar, edep, t0, t1, x0..z1, px0, py0, pz0, p1, len (Mu2e frame, mm, ns, MeV);
    pdg: [nsteps] particle code of each step; geom: crvfastsim CRVGeometry; cfg: ResponseConfig;
    tables: {table name: lookup.LookupTable}; deviation: [nbars * 4] photon-yield deviation (CRVPhoton table).
    Returns (step index, channel = bar*4 + SiPM, arrival time ns) arrays. Steps with a NaN start or end time are
    skipped, as CrvPhotonGenerator does (G4 tables contain a few).
    """
    outs = [_substep_photons(L, rows, rng) for L, rows in iter_rows(steps, pdg, geom, cfg, tables, deviation, rng)]
    if not outs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return tuple(np.concatenate([o[i] for o in outs]) for i in range(3))


def load_tables(cfg, directory=None):
    return {t: lookup.load(t, directory) for t in sorted({s.table for s in cfg.sectors.values()})}
