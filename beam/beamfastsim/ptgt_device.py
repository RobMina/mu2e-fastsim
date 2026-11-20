"""Opt-in production-target model (skeleton): hadron transport through a target described at run time, from a
geometry-free thin-target interaction bank.

Configuration-agnostic by construction. The only Geant4-made input is the bank: per (projectile species, material,
kinetic energy) cell, the macroscopic inelastic and elastic cross sections, stopping power and radiation length, and a
pool of single inelastic interactions whose secondaries are stored in the projectile frame (z along the projectile
momentum at the interaction, azimuth randomised at use), plus a per-cell inverse CDF of the elastic scattering
cos(theta). Secondaries outside SPECIES (pi0, photons, electrons, nuclear fragments, other hadrons) are summed into
the interaction's local energy deposit by the bank builder. It contains no target, solenoid or field geometry, so any
target family (radius profile, plates, clad, hollow, multi-material, offset or tilt) is a run-time `Target` and never
needs a new bank. The bank is a `mu2efastsim.library` file (kind "pool", subsystem "beam").

This module is the NumPy host reference: straight steps inside the target (no field), continuous loss at the step's
start energy with a fractional step cap, Highland multiple scattering, inelastic and elastic interactions, tallies of
deposited energy per segment, and the states that leave the target (no field inside the target). It and the bank
builder (`ptgt_bank`) name no experiment geometry; the solenoid stage and the device (JAX) chain are the experiment
layer (`ptgt_chain`, `ptgt_chain_jax`). Nothing imports it from a default chain.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mu2efastsim import library as L
from mu2efastsim.library import Axis

SPECIES = (2212, 2112, 211, -211, 321, -321, 13, -13)  # bank axis index = position (muons: loss and MSC only)
MASS = {2212: 938.27208816, 2112: 939.56542052, 211: 139.57039, -211: 139.57039, 321: 493.677, -321: 493.677,
        13: 105.6583755, -13: 105.6583755}           # MeV/c^2
BANK_FRAME = "projectile frame at the interaction: z along the projectile momentum, azimuth randomised at use"
CELL_ARRAYS = ("sigma_inel", "sigma_el", "dedx", "x0")   # per cell, 1/mm, 1/mm, MeV/mm, mm (x0 per material only)
N_EL = 33                                            # nodes of the elastic cos(theta) inverse CDF per cell
TRIAL_ARRAYS = ("trial_cell", "sec_offsets", "edep_local")
SEC_ARRAYS = ("sec_pdg", "sec_p")


def species_index(pdg):
    pdg = np.asarray(pdg)
    idx = np.full(pdg.shape, -1, dtype=np.int64)
    for i, s in enumerate(SPECIES):
        idx[pdg == s] = i
    return idx


def mass(pdg):
    pdg = np.asarray(pdg)
    return np.select([np.abs(pdg) == a for a in (2212, 2112, 211, 321, 13)],
                     [MASS[2212], MASS[2112], MASS[211], MASS[321], MASS[13]], default=0.0)


# ------------------------------------------------------------------------------------------------ bank (library)

def bank_axes(materials, log10_ke_edges):
    """Species x material x log10(KE/MeV) axes of the bank; the material order is the bank's material index."""
    return (Axis("species", "1", np.arange(len(SPECIES) + 1, dtype=float), frame="PDG code by SPECIES index",
                 array="species_edges", scale="categorical", policy="refuse"),
            Axis("material", "1", np.arange(len(materials) + 1, dtype=float), frame=",".join(materials),
                 array="material_edges", scale="categorical", policy="refuse"),
            Axis("log10_ke", "log10(MeV)", np.asarray(log10_ke_edges, float), frame=BANK_FRAME,
                 array="log10_ke_edges", scale="log", policy="refuse"))


def save_bank(path, materials, log10_ke_edges, cells, trials, secondaries, fields):
    """Write a bank. cells: CELL_ARRAYS shaped (species, material, ke bins); trials: trial_cell (flat cell, sorted),
    sec_offsets (trials + 1), edep_local [MeV] (pi0/gamma/e and fragments deposited at the vertex); secondaries:
    sec_pdg, sec_p [N, 3] MeV/c in the projectile frame. fields: manifest fields (name 'beam.thin_target.<list>',
    source, generator, release, seeds, split, ...). Returns the manifest."""
    axes = bank_axes(materials, log10_ke_edges)
    shape = tuple(a.nbins for a in axes)
    arrays = {a.array: a.edges for a in axes}
    for k in CELL_ARRAYS:
        v = np.asarray(cells[k], dtype=np.float64)
        if v.shape != shape:
            raise ValueError(f"{k}: shape {v.shape}, expected {shape}")
        arrays[k] = v
    el = np.asarray(cells["el_cos_icdf"], dtype=np.float64)
    if el.shape != shape + (N_EL,) or np.any(np.diff(el, axis=-1) < 0) or np.any(np.abs(el) > 1):
        raise ValueError(f"el_cos_icdf must be {shape + (N_EL,)}, non-decreasing, within [-1, 1]")
    arrays["el_cos_icdf"] = el
    tc = np.asarray(trials["trial_cell"], dtype=np.int64)
    if np.any(np.diff(tc) < 0) or (tc.size and (tc[0] < 0 or tc[-1] >= np.prod(shape))):
        raise ValueError("trial_cell must be sorted flat cell indices")
    off = np.asarray(trials["sec_offsets"], dtype=np.int64)
    if off.shape != (tc.size + 1,) or off[0] != 0 or np.any(np.diff(off) < 0):
        raise ValueError("sec_offsets must be (trials + 1,), start at 0, non-decreasing")
    arrays.update(trial_cell=tc, sec_offsets=off,
                  edep_local=np.asarray(trials["edep_local"], dtype=np.float32),
                  cell_offsets=np.searchsorted(tc, np.arange(np.prod(shape) + 1)).astype(np.int64),
                  sec_pdg=np.asarray(secondaries["sec_pdg"], dtype=np.int32),
                  sec_p=np.asarray(secondaries["sec_p"], dtype=np.float32))
    if arrays["sec_p"].shape != (off[-1], 3) or arrays["sec_pdg"].shape != (off[-1],):
        raise ValueError("secondary arrays must have sec_offsets[-1] rows")
    base = dict(schema=L.manifest.SCHEMA, subsystem="beam", kind="pool", version=1,
                description="Single hadronic interactions per (projectile, material, KE) in the projectile frame, with "
                            "macroscopic cross sections, stopping power and radiation length: no geometry.",
                axes=[a.describe() for a in axes],
                outputs=[dict(name="secondaries", unit="MeV/c", dtype="float32",
                              description="pdg and momentum of each secondary in the projectile frame"),
                         dict(name="edep_local", unit="MeV", dtype="float32", description="energy left at the vertex")],
                geometry=dict(tag="none (component-local thin target)", hash=None),
                dtype=dict(storage={"sec_p": "float32", "edep_local": "float32"}, compute=["float32", "float64"],
                           notes="host reference computes in float64"),
                rng=dict(host="numpy.random.Generator", device="jax.random key", streams={}, draws_per_row=None),
                domain=dict(coverage="log10_ke edges; refuse outside", fallbacks=[]),
                validation=dict(status="not validated", record=None), unrecorded={})
    base.update(fields)
    return L.save(path, arrays, base)


@dataclass(frozen=True)
class Bank:
    lib: object
    materials: tuple

    @classmethod
    def load(cls, path, expect=None):
        lib = L.load(path, expect=expect) if expect else L.load(path)
        return cls(lib, tuple(lib.axis("material").frame.split(",")))

    def __getitem__(self, k):
        return self.lib.arrays[k]

    def cell(self, sp, mat, ke):
        """Flat cell index per lane; raises on an energy outside the bank (policy refuse)."""
        e = self["log10_ke_edges"]
        lk = np.log10(ke)
        bad = np.flatnonzero((lk < e[0]) | (lk >= e[-1]))
        if bad.size:
            raise L.OutOfDomain(self.lib.name, "log10_ke", bad)
        k = L.axes.locate(e, lk)
        nm, nk = len(self.materials), e.size - 1
        return (np.asarray(sp) * nm + np.asarray(mat)) * nk + k

    def per_cell(self, name, cell):
        return self[name].reshape(-1)[cell]

    def elastic_cos(self, cell, u):
        """cos(theta) of an elastic scatter per lane: linear interpolation of the cell's inverse CDF at u in [0, 1)."""
        icdf = self["el_cos_icdf"].reshape(-1, N_EL)[cell]
        f = u * (N_EL - 1)
        k = np.minimum(f.astype(np.int64), N_EL - 2)
        w = f - k
        rows = np.arange(len(cell))
        return (1 - w) * icdf[rows, k] + w * icdf[rows, k + 1]

    def draw(self, cell, rng):
        """One stored interaction per lane from its cell (uniform within the cell); raises on an empty cell."""
        co = self["cell_offsets"]
        n = co[cell + 1] - co[cell]
        if np.any(n == 0):
            raise L.OutOfDomain(self.lib.name, "cell (no stored interactions)", np.flatnonzero(n == 0))
        return co[cell] + np.floor(rng.random(cell.shape) * n).astype(np.int64)


# ------------------------------------------------------------------------------------------------ target geometry

@dataclass(frozen=True)
class Target:
    """Coaxial frusta along the target's local z (the beam enters at the +z face and travels along local -z, see
    ptgt_chain.primaries): segment i spans z_edges[i]..z_edges[i+1] (or z_ranges[i], for segments that share a z
    range, e.g. a core and its sleeve) with inner/outer radii linear between r_in/r_out[i, 0] and [i, 1], material
    mat[i] (bank material index). Segments must not overlap; a point inside several is in the first. Placement: local
    -> global (the caller's frame) is x_global = origin + rot @ x_local."""
    z_edges: np.ndarray
    r_in: np.ndarray
    r_out: np.ndarray
    mat: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    z_ranges: Optional[np.ndarray] = None

    def __post_init__(self):
        z = np.asarray(self.z_edges, float)
        if self.z_ranges is None:
            if z.size < 2 or np.any(np.diff(z) <= 0):
                raise ValueError("z_edges must increase")
        elif np.asarray(self.z_ranges).ndim != 2 or np.any(np.diff(np.asarray(self.z_ranges, float), axis=1) <= 0):
            raise ValueError("z_ranges must be (segments, 2) with z0 < z1")
        n = len(self.zr)
        for k in ("r_in", "r_out"):
            if np.asarray(getattr(self, k)).shape != (n, 2):
                raise ValueError(f"{k} must be (segments, 2)")
        if np.any(np.asarray(self.r_out) <= np.asarray(self.r_in)) or np.any(np.asarray(self.r_in) < 0):
            raise ValueError("need 0 <= r_in < r_out")
        if not np.allclose(np.asarray(self.rot) @ np.asarray(self.rot).T, np.eye(3), atol=1e-12):
            raise ValueError("rot must be orthonormal")

    @property
    def zr(self):
        """(segments, 2): the z range of each segment."""
        if self.z_ranges is not None:
            return np.asarray(self.z_ranges, float)
        z = np.asarray(self.z_edges, float)
        return np.stack([z[:-1], z[1:]], 1)

    @property
    def z_lo(self):
        return float(self.zr[:, 0].min())

    @property
    def z_hi(self):
        return float(self.zr[:, 1].max())

    @classmethod
    def clad_rod(cls, length, r_core, t_clad, mat_core=0, mat_clad=1, **placement):
        """A cylindrical rod of radius r_core in a sleeve r_core..r_core + t_clad (no end caps), both of length."""
        h, r1 = 0.5 * length, r_core + t_clad
        return cls(np.array([-h, h]), np.array([[0.0, 0.0], [r_core, r_core]]), np.array([[r_core, r_core], [r1, r1]]),
                   np.array([mat_core, mat_clad]), z_ranges=np.array([[-h, h], [-h, h]]), **placement)

    @classmethod
    def rod(cls, length, r_up, r_mid, r_dn, mat=0, **placement):
        """A solid three-plane polycone rod (z = -L/2, 0, L/2; outer radii r_up, r_mid, r_dn): the beam enters at
        the r_dn end (+L/2) and travels towards r_up; the radius is linear between the planes."""
        h = 0.5 * length
        return cls(np.array([-h, 0.0, h]), np.zeros((2, 2)), np.array([[r_up, r_mid], [r_mid, r_dn]]),
                   np.array([mat, mat]), **placement)

    def to_local(self, x, d=None):
        xl = (np.asarray(x) - self.origin) @ np.asarray(self.rot)
        return xl if d is None else (xl, np.asarray(d) @ np.asarray(self.rot))

    def to_global(self, xl, dl=None):
        x = np.asarray(xl) @ np.asarray(self.rot).T + self.origin
        return x if dl is None else (x, np.asarray(dl) @ np.asarray(self.rot).T)

    def _radii(self, i, z):
        z0, z1 = self.zr[i, 0], self.zr[i, 1]
        f = (z - z0) / (z1 - z0)
        return (self.r_in[i, 0] + f * (self.r_in[i, 1] - self.r_in[i, 0]),
                self.r_out[i, 0] + f * (self.r_out[i, 1] - self.r_out[i, 0]))

    def locate(self, xl):
        """Segment index per local point, -1 outside the material."""
        xl = np.asarray(xl, float)
        z, r = xl[:, 2], np.hypot(xl[:, 0], xl[:, 1])
        seg = np.full(len(z), -1)
        zr = self.zr
        for i in range(len(self.mat) - 1, -1, -1):          # the first segment wins where several hold the point
            rin, rout = self._radii(i, z)
            inside = (z >= zr[i, 0]) & (z < zr[i, 1]) & (r >= rin) & (r < rout)
            seg = np.where(inside, i, seg)
        return seg

    @staticmethod
    def _cone_hits(xl, dl, z0, z1, ra, rb):
        """Smallest t > 0 where the ray meets the cone r(z) = ra + (rb-ra)(z-z0)/(z1-z0) with z0 <= z <= z1."""
        s = (rb - ra) / (z1 - z0)
        c0 = ra - s * z0                                   # r(z) = c0 + s z
        x, y, z = xl[:, 0], xl[:, 1], xl[:, 2]
        dx, dy, dz = dl[:, 0], dl[:, 1], dl[:, 2]
        a = dx * dx + dy * dy - s * s * dz * dz
        b = 2 * (x * dx + y * dy - s * dz * (c0 + s * z))
        c = x * x + y * y - (c0 + s * z) ** 2
        disc = b * b - 4 * a * c
        best = np.full(len(x), np.inf)
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        lin = np.abs(a) < 1e-15
        roots = [np.where(lin, -c / np.where(b == 0, np.inf, b), (-b - sq) / np.where(lin, 1.0, 2 * a)),
                 np.where(lin, np.inf, (-b + sq) / np.where(lin, 1.0, 2 * a))]
        for t in roots:
            zt = z + t * dz
            good = ok & (t > 1e-9) & (zt >= z0) & (zt <= z1) & (c0 + s * zt >= 0)
            best = np.where(good & (t < best), t, best)
        return best

    def _plane_hits(self, xl, dl, zp):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (zp - xl[:, 2]) / dl[:, 2]
        return np.where(np.isfinite(t) & (t > 1e-9), t, np.inf)

    def distance_out(self, xl, dl, seg):
        """Distance along dl from a point inside segment seg to that segment's boundary."""
        t = np.full(len(xl), np.inf)
        for i in np.unique(seg):
            m = seg == i
            z0, z1 = self.zr[i]
            ti = np.minimum(self._plane_hits(xl[m], dl[m], z0), self._plane_hits(xl[m], dl[m], z1))
            ti = np.minimum(ti, self._cone_hits(xl[m], dl[m], z0, z1, *self.r_out[i]))
            if np.any(self.r_in[i] > 0):
                ti = np.minimum(ti, self._cone_hits(xl[m], dl[m], z0, z1, *self.r_in[i]))
            t[m] = ti
        return t

    def distance_in(self, xl, dl):
        """Distance to the first entry into any segment for points outside the material (inf: never)."""
        t = np.full(len(xl), np.inf)
        for i in range(len(self.mat)):
            z0, z1 = self.zr[i]
            for tc in (self._cone_hits(xl, dl, z0, z1, *self.r_out[i]),
                       self._cone_hits(xl, dl, z0, z1, *self.r_in[i]) if np.any(self.r_in[i] > 0) else t + np.inf):
                t = np.minimum(t, tc)
            for zp, k in ((z0, 0), (z1, 1)):
                tp = self._plane_hits(xl, dl, zp)
                p = xl + np.where(np.isfinite(tp), tp, 0.0)[:, None] * dl
                r = np.hypot(p[:, 0], p[:, 1])
                t = np.where((r >= self.r_in[i, k]) & (r < self.r_out[i, k]) & (tp < t), tp, t)
        return t


# ------------------------------------------------------------------------------------------------ transport

def frame_to_lab(d):
    """Rotation matrices [N, 3, 3] whose third column is the unit direction d (projectile frame -> lab)."""
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    helper = np.where(np.abs(d[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    return np.stack([u, v, d], axis=2)


def rotate_secondaries(p_local, d, phi):
    """Secondaries [N, 3] from the projectile frame to the lab, with an azimuth phi [N] about the projectile axis."""
    c, s = np.cos(phi), np.sin(phi)
    pr = np.stack([c * p_local[:, 0] - s * p_local[:, 1], s * p_local[:, 0] + c * p_local[:, 1], p_local[:, 2]], 1)
    return np.einsum("nij,nj->ni", frame_to_lab(d), pr)


@dataclass(frozen=True)
class Cuts:
    ke_min_hadron: float = 50.0           # p, n below: deposit, stop (the bank stores nucleons from 50 MeV)
    ke_min_pion: float = 1.0              # pi, K, mu below: deposit, stop
    max_loss_fraction: float = 0.05       # continuous-loss step cap
    max_lanes: int = 10_000_000


def transport(bank, target, pdg, x, p, rng, cuts=Cuts(), max_iter=100_000):
    """Host reference: transport lanes (global frame, mm, MeV/c) through the target until they leave or stop.

    Returns dict(exit_pdg, exit_x, exit_p (global frame, at the target boundary), edep [segments] MeV,
    counts {inelastic, elastic, stopped, unknown_species})."""
    xl, dl = target.to_local(x, p)
    pdg = np.asarray(pdg, np.int64)
    pabs = np.linalg.norm(dl, axis=1)
    dl = dl / pabs[:, None]
    seg = target.locate(xl)
    edep = np.zeros(len(target.mat))
    out = {"pdg": [], "x": [], "p": []}
    counts = dict(inelastic=0, elastic=0, stopped=0, unknown_species=0)
    for _ in range(max_iter):
        if not len(pdg):
            break
        if len(pdg) > cuts.max_lanes:
            raise RuntimeError("lane cap exceeded")
        outside = seg < 0
        if np.any(outside):                                    # move outside lanes to their next entry or out
            ti = target.distance_in(xl[outside], dl[outside])
            gone = np.isinf(ti)
            idx = np.flatnonzero(outside)
            for k, v in (("pdg", pdg[idx[gone]]), ("x", xl[idx[gone]]), ("p", dl[idx[gone]] * pabs[idx[gone], None])):
                out[k].append(v)
            xl[idx[~gone]] += (ti[~gone] + 1e-7)[:, None] * dl[idx[~gone]]
            seg[idx[~gone]] = target.locate(xl[idx[~gone]])
            keep = np.ones(len(pdg), bool)
            keep[idx[gone]] = False
            pdg, xl, dl, pabs, seg = pdg[keep], xl[keep], dl[keep], pabs[keep], seg[keep]
            continue
        m = mass(pdg)
        ke = np.hypot(pabs, m) - m
        hadron = np.isin(np.abs(pdg), (2212, 2112))
        low = np.where(hadron, ke < cuts.ke_min_hadron, ke < cuts.ke_min_pion)
        sp = species_index(pdg)
        unknown = (sp < 0) & ~low
        counts["unknown_species"] += int(unknown.sum())
        stop = low | unknown
        if np.any(stop):
            np.add.at(edep, seg[stop], ke[stop])
            counts["stopped"] += int(stop.sum())
            keep = ~stop
            pdg, xl, dl, pabs, seg, ke, sp, m = (a[keep] for a in (pdg, xl, dl, pabs, seg, ke, sp, m))
            if not len(pdg):
                break
        cell = bank.cell(sp, target.mat[seg], ke)
        s_in = bank.per_cell("sigma_inel", cell)
        s_el = bank.per_cell("sigma_el", cell)
        dedx = bank.per_cell("dedx", cell)
        x0 = bank.per_cell("x0", cell)
        s_tot = s_in + s_el
        with np.errstate(divide="ignore"):
            t_int = np.where(s_tot > 0, -np.log(1.0 - rng.random(len(pdg))) / s_tot, np.inf)
            t_loss = np.where(dedx > 0, cuts.max_loss_fraction * ke / dedx, np.inf)
        t_bnd = target.distance_out(xl, dl, seg)
        step = np.minimum(np.minimum(t_int, t_loss), t_bnd)
        xl = xl + step[:, None] * dl
        de = np.minimum(dedx * step, ke)
        np.add.at(edep, seg, de)
        e_new = ke - de + m
        pabs = np.sqrt(np.maximum(e_new ** 2 - m ** 2, 0.0))
        beta = np.where(e_new > 0, pabs / np.maximum(e_new, 1e-300), 0.0)
        theta0 = 13.6 / np.maximum(beta * pabs, 1e-300) * np.sqrt(step / x0) * (1 + 0.038 * np.log(np.maximum(step / x0, 1e-12)))
        theta0 = np.where(step > 0, np.abs(theta0), 0.0)
        th = np.minimum(theta0 * np.sqrt(-2.0 * np.log(1.0 - rng.random(len(pdg)))), np.pi)  # space angle, Highland
        dl = rotate_secondaries(np.stack([np.sin(th), np.zeros(len(pdg)), np.cos(th)], 1), dl,
                                rng.random(len(pdg)) * 2 * np.pi)
        dl /= np.linalg.norm(dl, axis=1, keepdims=True)
        hit = (t_int <= t_loss) & (t_int <= t_bnd)
        crossed = (t_bnd < t_int) & (t_bnd <= t_loss)
        xl[crossed] += 1e-7 * dl[crossed]
        seg = target.locate(xl)
        inel = hit & (rng.random(len(pdg)) * s_tot < s_in)
        el = hit & ~inel
        counts["elastic"] += int(el.sum())
        if np.any(el):                                         # elastic: angle from the bank, |p| kept (no recoil)
            ct = bank.elastic_cos(cell[el], rng.random(int(el.sum())))
            st = np.sqrt(np.maximum(1.0 - ct * ct, 0.0))
            dl[el] = rotate_secondaries(np.stack([st, np.zeros_like(st), ct], 1), dl[el],
                                        rng.random(int(el.sum())) * 2 * np.pi)
        if np.any(inel):
            counts["inelastic"] += int(inel.sum())
            j = bank.draw(cell[inel], rng)
            off = bank["sec_offsets"]
            n_sec = off[j + 1] - off[j]
            rows = np.repeat(off[j], n_sec) + (np.arange(n_sec.sum()) - np.repeat(np.cumsum(n_sec) - n_sec, n_sec))
            parent = np.repeat(np.flatnonzero(inel), n_sec)
            np.add.at(edep, seg[np.flatnonzero(inel)], bank["edep_local"][j].astype(float))
            p_new = rotate_secondaries(bank["sec_p"][rows].astype(float), dl[parent],
                                       rng.random(len(rows)) * 2 * np.pi)
            keep = ~inel
            pn = np.linalg.norm(p_new, axis=1)
            pdg = np.concatenate([pdg[keep], bank["sec_pdg"][rows].astype(np.int64)])
            xl = np.concatenate([xl[keep], xl[parent]])
            dl = np.concatenate([dl[keep], p_new / np.maximum(pn, 1e-300)[:, None]])
            pabs = np.concatenate([pabs[keep], pn])
            seg = np.concatenate([seg[keep], seg[parent]])
    else:
        raise RuntimeError("transport did not finish within max_iter")
    cat = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k != "pdg" else 0)) for k, v in out.items()}
    ex, ep = target.to_global(cat["x"], cat["p"]) if len(cat["pdg"]) else (cat["x"], cat["p"])
    return dict(exit_pdg=cat["pdg"].astype(np.int64), exit_x=ex, exit_p=ep, edep=edep, counts=counts)
