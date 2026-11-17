"""Expected photoelectrons per counter end for a crossing of a designed counter (the "scaled optics" model).

The stock response (MakeCrvPhotons) reads photon arrival probabilities from a lookup table made for one counter
cross-section and fiber (v6: 19.28 x 50.84 mm scintillator core, two 1.4 mm fibers 26 mm apart) and one length.
For a designed counter this model:
  1. maps the local point into the table: thickness and width scaled by the ratio of scintillator cores (the coating
     around the core gives no light, as in stock), length by the distance to the read-out end; beyond the table's
     length the last bin is extended with the table's own attenuation length (photons.table_propagation);
  2. multiplies the arrival probability by a capture factor (d / d_stock)^alpha_d x (P_table / P_core)^alpha_g:
     alpha_d from the stock pair of fiber yields (CRV-T 1.8 mm 39794 vs 1.4 mm 30482 photons/MeV, the way stock
     itself models the fiber), P the core perimeter (light reaching the fibers falls with the reflecting surface);
  3. mean number of photons: yield x photon_yield_scale x visible energy (Bethe mean x path), the visible energy
     spread evenly along the chord; photoelectrons per arrival from the SiPM model (fire probability at the
     overvoltage, the 4 x 4 dead centre, mean cross-talk), times a calibration scale fitted to the full chain at the
     nominal design (pe_scale, default 1).
Cherenkov light and fluctuations of the visible energy are not modelled (folded into pe_scale at nominal).
Steps 1-2 are an approximation with no stock counterpart: new lookup tables from the stock optical simulation are
the stock alternative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mu2efastsim import eloss, materials
from ..response import photons as PH
from ..response import sipm as SI

FIBER_ALPHA = math.log(39794.0 / 30482.0) / math.log(1.8 / 1.4)
COATING = (0.26, 0.25)   # mm: counter half thickness / half width minus the v6 table's scintillator core (19.80 x 51.34)
N_SAMPLES = 8             # points along a chord


@dataclass(frozen=True)
class OpticsConfig:
    alpha_d: float = FIBER_ALPHA
    alpha_g: float = 1.0
    photon_yield_scale: float = 0.84     # Run 2 CrvPhotons photonYieldScaleFactor
    pe_scale: float = 1.0                # calibration to the full chain at the nominal design
    aging: float = 1.0                   # light-yield factor (1 = new)


def pe_per_arrival(s=SI.SiPMConfig()):
    """Mean PE per photon reaching a SiPM: fire probability at Vov, active pixel fraction, mean cross-talk."""
    fire = s.p1 * (1.0 - math.exp(-s.overvoltage / s.p2))
    active = 1.0 - 16.0 / (s.n_pixels_x * s.n_pixels_y)
    return fire * active / (1.0 - s.cross_talk)


@dataclass
class TableModel:
    """Arrival probability at SiPM 0 of one lookup table on its (|x|, y, z) scintillation grid."""
    name: str
    prob: np.ndarray       # [nx, ny, nz]
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    half: tuple            # table half thickness, width, length
    atten: float           # mm
    reflector: int

    @classmethod
    def from_lookup(cls, L):
        E = L.edges
        nx, ny, nz = len(E["x"]) - 1, len(E["y"]) - 1, len(E["z"]) - 1
        p = L.tables["scint"].prob.astype(np.float64).reshape(nx, ny, nz)
        return cls(L.name, p, E["x"], E["y"], E["z"], (L.half_thickness, L.half_width, L.half_length),
                   PH.table_propagation(L)[1], int(L.reflector))

    def perimeter(self):
        return 4.0 * (self.half[0] + self.half[1])

    def prob_at(self, x, y, d, dtype=np.float64):
        """Arrival probability at the SiPM of the -z end for table-frame |x|, y and distance d from that end."""
        zt = d - self.half[2]
        zlast = 0.5 * (self.ez[-2] + self.ez[-1])
        beyond = zt > zlast
        zq = np.where(beyond, zlast, zt)
        bx, by, bz = (PH.find_bin(self.ex, np.abs(x)), PH.find_bin(self.ey, y), PH.find_bin(self.ez, zq))
        ok = (bx >= 0) & (by >= 0) & (bz >= 0)
        p = np.where(ok, self.prob[np.maximum(bx, 0), np.maximum(by, 0), np.maximum(bz, 0)], 0.0)
        p = np.where(beyond, p * np.exp(-(zt - zlast) / self.atten), p)
        return p.astype(dtype)


def capture_factor(t, w, d, d_stock, tab, cfg):
    core_t, core_w = t / 2 - COATING[0], w / 2 - COATING[1]
    if core_t <= 0 or core_w <= 0:
        raise ValueError("counter thinner than its coating")
    geo = (tab.perimeter() / (4.0 * (core_t + core_w))) ** cfg.alpha_g
    return (d / d_stock) ** cfg.alpha_d * geo


def expected_pe(local_in, local_out, halflen, p_mu, table, yield_per_mev, d, d_stock, sides, cfg=OpticsConfig(),
                dtype=np.float64):
    """[n, 2] expected PE at the -l end (side 0) and the +l end (side 1), summed over the two fibers, for chords from
    local_in to local_out (counter-local thickness, width, length, mm) of counters with half lengths halflen [n, 3];
    p_mu [n] muon momentum [MeV/c]; sides: the read-out sides (module.readout_sides). Unread sides are 0."""
    local_in, local_out = np.asarray(local_in, dtype), np.asarray(local_out, dtype)
    halflen = np.asarray(halflen, dtype)
    n = len(local_in)
    out = np.zeros((n, 2), dtype)
    if n == 0:
        return out
    path = np.linalg.norm(local_out - local_in, axis=1)
    dedx = eloss.dedx_restricted(materials.POLYSTYRENE, np.asarray(p_mu, np.float64), np.inf)
    nphot = (yield_per_mev * cfg.photon_yield_scale * dedx * path).astype(dtype)
    core_t = halflen[:, 0] - COATING[0]
    core_w = halflen[:, 1] - COATING[1]
    cap = capture_factor(2 * float(halflen[0, 0]), 2 * float(halflen[0, 1]), d, d_stock, table, cfg)
    scale = nphot * dtype(cap * pe_per_arrival() * cfg.pe_scale * cfg.aging) / N_SAMPLES
    f = (np.arange(N_SAMPLES, dtype=dtype) + 0.5) / N_SAMPLES
    pts = local_in[:, None, :] + (local_out - local_in)[:, None, :] * f[None, :, None]   # [n, K, 3]
    inside = (np.abs(pts[..., 0]) < core_t[:, None]) & (np.abs(pts[..., 1]) < core_w[:, None])
    xt = pts[..., 0] * (table.half[0] / core_t[:, None])
    yt = pts[..., 1] * (table.half[1] / core_w[:, None])
    for side in sides:
        dist = (pts[..., 2] + halflen[:, None, 2]) if side == 0 else (halflen[:, None, 2] - pts[..., 2])
        p = table.prob_at(xt, yt, dist, dtype) + table.prob_at(xt, -yt, dist, dtype)   # SiPMs of both fibers
        out[:, side] = scale * np.where(inside, p, 0).sum(1)
    return out


def light_yield_profile(halflen, table, yield_per_mev, d, d_stock, sides, cfg=OpticsConfig(), npos=25,
                        p_mu=1000.0, dtype=np.float64):
    """Expected PE of the better-read end for a muon crossing the counter at normal incidence through its width centre,
    at npos positions along the length (50 mm from each end): [npos]."""
    halflen = np.asarray(halflen, float)
    z = np.linspace(-halflen[2] + 50.0, halflen[2] - 50.0, npos)
    a = np.stack([np.full(npos, -halflen[0]), np.zeros(npos), z], 1)
    b = np.stack([np.full(npos, halflen[0]), np.zeros(npos), z], 1)
    pe = expected_pe(a, b, np.broadcast_to(halflen, (npos, 3)), np.full(npos, p_mu), table, yield_per_mev, d,
                     d_stock, sides, cfg, dtype)
    return pe.max(1)
