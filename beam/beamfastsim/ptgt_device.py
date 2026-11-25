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
    mat[i] (bank material index). Optional boxes in the local frame (fins, plates): boxes[j] =
    (x0, x1, y0, y1, z0, z1), segment index n_cone + j, material mat[n_cone + j]; box_phi[j] (radians, default 0)
    rotates box j about the local z axis (its bounds hold in the frame turned by box_phi[j]: a point x is inside when
    Rz(-box_phi[j]) x is within them; e.g. a radial fin at angle phi is the box (0, L, -w/2, w/2, z0, z1), box_phi phi). Optional spheres[k] = (cx, cy, cz,
    R) and solid cylinders of any axis cylinders[m] = (ax, ay, az, bx, by, bz, r) (end centres a, b; flat ends) in the
    local frame (hub balls, spokes), indexed after the boxes in that order. mat covers the coaxial segments, then the
    boxes, spheres and cylinders. Solids of different materials must not overlap; a point inside several is in the
    first (coaxial segments, then boxes, spheres, cylinders). beam_z: the local z of the beam's entrance plane
    (default z_hi; the deck starts the beam at its nominal rod face, inside hubs that reach further). heat_edges:
    increasing local z edges of the target-heat slices (e.g. one per plate pitch; default EDEP_SLICE_MM). Placement:
    local -> global (the caller's frame) is x_global = origin + rot @ x_local."""
    z_edges: np.ndarray
    r_in: np.ndarray
    r_out: np.ndarray
    mat: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    z_ranges: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    box_phi: Optional[np.ndarray] = None
    spheres: Optional[np.ndarray] = None
    cylinders: Optional[np.ndarray] = None
    beam_z: Optional[float] = None
    heat_edges: Optional[np.ndarray] = None     # target-heat slice edges along the local z (default: EDEP_SLICE_MM)

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
        b = self.bx
        if b.ndim != 2 or b.shape[1] != 6 or np.any(b[:, 1::2] <= b[:, 0::2]):
            raise ValueError("boxes must be (n, 6) = x0, x1, y0, y1, z0, z1 with lo < hi")
        if self.box_phi is not None and np.asarray(self.box_phi, float).shape != (len(b),):
            raise ValueError("box_phi must have one angle per box")
        sp, cy = self.sp, self.cy
        if sp.ndim != 2 or sp.shape[1] != 4 or np.any(sp[:, 3] <= 0):
            raise ValueError("spheres must be (n, 4) = cx, cy, cz, R with R > 0")
        if cy.ndim != 2 or cy.shape[1] != 7 or np.any(cy[:, 6] <= 0) or np.any(
                np.linalg.norm(cy[:, 3:6] - cy[:, 0:3], axis=1) <= 0):
            raise ValueError("cylinders must be (n, 7) = ax, ay, az, bx, by, bz, r with a != b and r > 0")
        if len(self.mat) != n + len(b) + len(sp) + len(cy):
            raise ValueError(f"mat must have one entry per segment, box, sphere and cylinder ({n} + {len(b)} + "
                             f"{len(sp)} + {len(cy)}), got {len(self.mat)}")

    @property
    def n_cone(self):
        return len(np.asarray(self.r_in))

    @property
    def bx(self):
        """(boxes, 6): x0, x1, y0, y1, z0, z1 of each box (local frame); (0, 6) without boxes."""
        return np.zeros((0, 6)) if self.boxes is None else np.asarray(self.boxes, float)

    @property
    def sp(self):
        """(spheres, 4): cx, cy, cz, R (local frame); (0, 4) without spheres."""
        return np.zeros((0, 4)) if self.spheres is None else np.asarray(self.spheres, float)

    @property
    def cy(self):
        """(cylinders, 7): ax, ay, az, bx, by, bz, r (local frame); (0, 7) without cylinders."""
        return np.zeros((0, 7)) if self.cylinders is None else np.asarray(self.cylinders, float)

    @property
    def bphi(self):
        """(boxes,): rotation of each box about the local z axis [rad]; zeros without box_phi."""
        return np.zeros(len(self.bx)) if self.box_phi is None else np.asarray(self.box_phi, float)

    @staticmethod
    def box_frame(v, phi):
        """v (..., 3) in the frame of boxes turned by phi (broadcast against v[..., 0]): Rz(-phi) v."""
        c, s = np.cos(phi), np.sin(phi)
        x, y = v[..., 0], v[..., 1]
        return np.stack(np.broadcast_arrays(c * x + s * y, c * y - s * x, v[..., 2] + 0 * c), -1)

    @property
    def n_box(self):
        return len(self.bx)

    @property
    def n_sphere(self):
        return len(self.sp)

    @property
    def z_beam(self):
        return float(self.z_hi if self.beam_z is None else self.beam_z)

    @property
    def r_max(self):
        """Radius of the bounding cylinder about the local z axis of the coaxial segments, boxes and spheres (the
        cylinders, e.g. long spokes, have their own: cylinder_near)."""
        b, sp = self.bx, self.sp
        rb = np.hypot(np.maximum(np.abs(b[:, 0]), np.abs(b[:, 1])), np.maximum(np.abs(b[:, 2]), np.abs(b[:, 3])))
        rs = np.hypot(sp[:, 0], sp[:, 1]) + sp[:, 3]
        return float(max(np.max(self.r_out), rb.max() if len(b) else 0.0, rs.max() if len(sp) else 0.0))

    def cylinder_near(self, xl, margin):
        """Points within margin of a cylinder (distance to its axis segment < r + margin)."""
        c = self.cy
        if not len(c):
            return np.zeros(len(xl), bool)
        a, ab = c[None, :, 0:3], (c[:, 3:6] - c[:, 0:3])[None]
        w = np.asarray(xl, float)[:, None, :] - a
        s = np.clip(np.sum(w * ab, -1) / np.sum(ab * ab, -1), 0.0, 1.0)
        return np.any(np.linalg.norm(w - s[..., None] * ab, axis=-1) < c[None, :, 6] + margin, axis=1)

    @property
    def zr(self):
        """(segments, 2): the z range of each segment."""
        if self.z_ranges is not None:
            return np.asarray(self.z_ranges, float)
        z = np.asarray(self.z_edges, float)
        return np.stack([z[:-1], z[1:]], 1)

    @property
    def z_lo(self):
        """Lowest local z of the coaxial segments, boxes and spheres (as r_max)."""
        sp = self.sp
        return float(min(self.zr[:, 0].min(), self.bx[:, 4].min() if len(self.bx) else np.inf,
                         (sp[:, 2] - sp[:, 3]).min() if len(sp) else np.inf))

    @property
    def z_hi(self):
        sp = self.sp
        return float(max(self.zr[:, 1].max(), self.bx[:, 5].max() if len(self.bx) else -np.inf,
                         (sp[:, 2] + sp[:, 3]).max() if len(sp) else -np.inf))

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
        b = self.bx
        o_sp, o_cy = self.n_cone + len(b), self.n_cone + len(b) + self.n_sphere
        xb = self.box_frame(xl[:, None, :], self.bphi[None, :])
        for off, inside in ((o_cy, self._in_cylinders(xl)), (o_sp, self._in_spheres(xl)),
                            (self.n_cone, np.all((xb >= b[None, :, 0::2]) & (xb < b[None, :, 1::2]), axis=2))):
            if inside.shape[1]:                              # later groups first, so earlier ones win; the first of a group
                seg = np.where(inside.any(axis=1), off + np.argmax(inside, axis=1), seg)
        zr = self.zr
        for i in range(self.n_cone - 1, -1, -1):             # the first segment wins where several hold the point
            rin, rout = self._radii(i, z)
            inside = (z >= zr[i, 0]) & (z < zr[i, 1]) & (r >= rin) & (r < rout)
            seg = np.where(inside, i, seg)
        return seg

    @staticmethod
    def _box_slabs(xl, dl, lo, hi):
        """(t_enter, t_exit) of rays against boxes: xl, dl (N, 3); lo, hi (N or 1, 3)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - xl) / dl, (hi - xl) / dl
        par = dl == 0                                        # parallel to a pair of faces: inside the slab or never
        inside = (xl >= lo) & (xl < hi)
        t1 = np.where(par, np.where(inside, -np.inf, np.inf), t1)
        t2 = np.where(par, np.where(inside, np.inf, -np.inf), t2)
        return np.max(np.minimum(t1, t2), axis=-1), np.min(np.maximum(t1, t2), axis=-1)

    def _in_spheres(self, xl):
        c = self.sp
        return np.sum((xl[:, None, :] - c[None, :, 0:3]) ** 2, -1) < c[None, :, 3] ** 2

    def _in_cylinders(self, xl):
        c = self.cy
        ab = c[:, 3:6] - c[:, 0:3]
        L = np.linalg.norm(ab, axis=1)
        w = xl[:, None, :] - c[None, :, 0:3]
        s = np.sum(w * (ab / L[:, None])[None], -1)
        q = w - s[..., None] * (ab / L[:, None])[None]
        return (s >= 0) & (s < L[None]) & (np.sum(q * q, -1) < c[None, :, 6] ** 2)

    @staticmethod
    def _sphere_span(xl, dl, c, R):
        """(t_enter, t_exit) of rays against spheres (broadcast); (inf, -inf) for a miss."""
        w = xl - c
        b = np.sum(w * dl, -1)
        disc = b * b - (np.sum(w * w, -1) - R * R)
        sq = np.sqrt(np.maximum(disc, 0.0))
        return np.where(disc > 0, -b - sq, np.inf), np.where(disc > 0, -b + sq, -np.inf)

    @staticmethod
    def _cylinder_span(xl, dl, a, b, r):
        """(t_enter, t_exit) of rays against finite solid cylinders (broadcast): the axial slab 0 <= s <= L
        intersected with the infinite cylinder |q| <= r; (inf, -inf) or t_enter > t_exit for a miss."""
        ab = b - a
        L = np.linalg.norm(ab, axis=-1)
        u = ab / L[..., None]
        w = xl - a
        s0, ds = np.sum(w * u, -1), np.sum(dl * u, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = -s0 / ds, (L - s0) / ds
        par = ds == 0
        ins = (s0 >= 0) & (s0 < L)
        lo = np.where(par, np.where(ins, -np.inf, np.inf), np.minimum(t1, t2))
        hi = np.where(par, np.where(ins, np.inf, -np.inf), np.maximum(t1, t2))
        q0, qd = w - s0[..., None] * u, dl - ds[..., None] * u
        A, B, Cq = np.sum(qd * qd, -1), np.sum(q0 * qd, -1), np.sum(q0 * q0, -1) - r * r
        disc = B * B - A * Cq
        axial = A < 1e-15                                    # along the axis: inside the radius or never
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(np.maximum(disc, 0.0))
            c1, c2 = (-B - sq) / A, (-B + sq) / A
        c1 = np.where(axial, np.where(Cq < 0, -np.inf, np.inf), np.where(disc > 0, c1, np.inf))
        c2 = np.where(axial, np.where(Cq < 0, np.inf, -np.inf), np.where(disc > 0, c2, -np.inf))
        return np.maximum(lo, c1), np.minimum(hi, c2)

    def _spans(self, xl, dl):
        """(t_enter, t_exit) of rays (N, 3) against every sphere and cylinder: (N, n_sphere + n_cylinder)."""
        sp, cy = self.sp, self.cy
        x3, d3 = xl[:, None, :], dl[:, None, :]
        e1, x1 = self._sphere_span(x3, d3, sp[None, :, 0:3], sp[None, :, 3])
        e2, x2 = self._cylinder_span(x3, d3, cy[None, :, 0:3], cy[None, :, 3:6], cy[None, :, 6])
        return np.concatenate([e1, e2], 1), np.concatenate([x1, x2], 1)

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
        b = self.bx
        for i in np.unique(seg):
            m = seg == i
            if i >= self.n_cone + len(b):                    # sphere or cylinder: the exit of its span
                t[m] = self._spans(xl[m], dl[m])[1][:, i - self.n_cone - len(b)]
                continue
            if i >= self.n_cone:
                bj, ph = b[i - self.n_cone], self.bphi[i - self.n_cone]
                t[m] = self._box_slabs(self.box_frame(xl[m], ph), self.box_frame(dl[m], ph), bj[0::2][None],
                                       bj[1::2][None])[1]
                continue
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
        b = self.bx
        if len(b):
            ph = self.bphi[None, :]
            te, tx = self._box_slabs(self.box_frame(xl[:, None, :], ph), self.box_frame(dl[:, None, :], ph),
                                     b[None, :, 0::2], b[None, :, 1::2])
            ok = (te > 1e-9) & (te <= tx)
            t = np.min(np.where(ok, te, np.inf), axis=1)
        if self.n_sphere + len(self.cy):
            te, tx = self._spans(xl, dl)
            t = np.minimum(t, np.min(np.where((te > 1e-9) & (te <= tx), te, np.inf), axis=1))
        for i in range(self.n_cone):
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

# ------------------------------------------------------------------------------------------------ target heating

# Densities [g/cm3] of the bank materials (NIST values for G4_*; the G4beamline deck's composites as defined there).
DENSITY = {"G4_W": 19.30, "G4_Ir": 22.42, "G4_Ta": 16.654, "G4_Mo": 10.22, "G4_Cu": 8.96, "G4_Fe": 7.874,
           "G4_STAINLESS-STEEL": 8.00, "WL10": 18.75, "Inconel718": 8.19}
EDEP_SLICE_MM = 10.0     # default slice width along the target's local z
EDEP_SLICE_MAX = 64      # fixed number of slice bins (a new design length needs no recompile on the device)
EDEP_BATCHES = 10        # batch means (by event number mod EDEP_BATCHES) for the statistical errors
MEV_PER_G_TO_GY = 1.602176634e-13 / 1e-3  # MeV/g -> J/kg (1 MeV = 1.602e-13 J, 1 g = 1e-3 kg)


def slice_edges(target, width=EDEP_SLICE_MM, nmax=EDEP_SLICE_MAX):
    """(nmax + 1) slice edges along the local z from target.z_lo in steps of width, and the number of slices that
    cover z_lo..z_hi (bins beyond carry no mass). Deposits outside the slices (e.g. spokes) are not sliced. A target
    with heat_edges uses those (padded past its last edge to nmax + 1 edges)."""
    he = getattr(target, "heat_edges", None)
    if he is not None:
        he = np.asarray(he, float)
        n = he.size - 1
        if n < 1 or np.any(np.diff(he) <= 0):
            raise ValueError("heat_edges must increase")
        if n > nmax:
            raise ValueError(f"{n} heat slices > {nmax}")
        return np.concatenate([he, he[-1] + width * np.arange(1, nmax - n + 1)]), n
    n = int(np.ceil((target.z_hi - target.z_lo) / width - 1e-9))
    if n > nmax:
        raise ValueError(f"target length {target.z_hi - target.z_lo} mm needs {n} slices > {nmax}: widen the slices")
    return target.z_lo + width * np.arange(nmax + 1), n


def slice_masses(target, edges, densities, n_per_slice=200_000, seed=12345):
    """Target mass [g] per slice: Monte Carlo over the cylinder about the local z axis that holds every solid in the
    slice (points located with Target.locate, density per material). densities: g/cm3 per bank material index."""
    rng = np.random.default_rng(seed)
    R = target.r_max
    cy = target.cy
    if len(cy):
        R = max(R, float(np.max(np.maximum(np.hypot(cy[:, 0], cy[:, 1]), np.hypot(cy[:, 3], cy[:, 4])) + cy[:, 6])))
    rho = np.asarray(densities, float)[np.asarray(target.mat)]
    out = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        if edges[k] >= target.z_hi:
            break
        r = R * np.sqrt(rng.random(n_per_slice)); ph = 2 * np.pi * rng.random(n_per_slice)
        z = edges[k] + (edges[k + 1] - edges[k]) * rng.random(n_per_slice)
        seg = target.locate(np.stack([r * np.cos(ph), r * np.sin(ph), z], 1))
        dens = np.where(seg >= 0, rho[np.clip(seg, 0, None)], 0.0)
        out[k] = dens.mean() * np.pi * R * R * (edges[k + 1] - edges[k]) * 1e-3     # g/cm3 * mm3 -> g
    return out


def material_densities(materials, overrides=None):
    """g/cm3 per bank material; overrides: {name: density}."""
    d = dict(DENSITY, **(overrides or {}))
    missing = [m for m in materials if m not in d]
    if missing:
        raise ValueError(f"no density for {missing}: pass --density NAME=G_PER_CM3")
    return [d[m] for m in materials]


def peak_heat(batches, masses, pot, n_used):
    """Peak heat from batch sums: batches (EDEP_BATCHES, nbins) MeV deposited per slice, masses (nbins) g, pot (total).
    Returns per-slice MeV/g/POT, the peak (MeV/g/POT and Gy/POT) with the batch-mean error of the peak slice."""
    b = np.asarray(batches, float)[:, :n_used]
    m = np.asarray(masses, float)[:n_used]
    nb = b.shape[0]
    per = b.sum(0) / m / pot
    k = int(np.argmax(per))
    bm = b[:, k] / m[k] / (pot / nb)                   # per-batch estimates (batches hold ~pot / nb POT each)
    err = float(np.std(bm, ddof=1) / np.sqrt(nb)) if nb > 1 else float("nan")
    return dict(slice_MeV_per_g_per_pot=per.tolist(), peak_slice=k, peak_MeV_per_g_per_pot=float(per[k]),
                peak_err=err, peak_Gy_per_pot=float(per[k] * MEV_PER_G_TO_GY), peak_Gy_err=err * MEV_PER_G_TO_GY,
                total_MeV_per_pot=float(b.sum() / pot), mass_g=m.tolist())


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
