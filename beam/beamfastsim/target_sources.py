"""Opt-in target sources following Offline DIOGenerator, RPCGun and BinnedSpectrum.

Energies are total MeV, positions mm and times ns. DIO draws are conditional on
an explicit table interval, not normalized to all muon decays. RPC needs pion
stops, not beam_point muon stops. Internal RPC retains the correlated positron;
external RPC emits a photon and needs a separate material-conversion transport.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cechain as C

ME = 0.510998910
MPI = 139.57018
M_AL = 26.981539 * 931.494028


@dataclass(frozen=True)
class BinnedSpectrum:
    """Piecewise uniform density, as CLHEP RandGeneral plus Offline sample()."""
    edges: np.ndarray
    pdf: np.ndarray

    def __post_init__(self):
        edges, pdf = np.asarray(self.edges, float), np.asarray(self.pdf, float)
        if (edges.ndim != 1 or pdf.ndim != 1 or pdf.size < 1 or edges.size != pdf.size + 1
                or not np.isfinite(edges).all() or not np.isfinite(pdf).all()
                or np.any(np.diff(edges) <= 0) or np.any(pdf < 0) or not np.isfinite(pdf.sum()) or not pdf.sum() > 0):
            raise ValueError("invalid binned spectrum")
        if not np.allclose(np.diff(edges), edges[1] - edges[0], rtol=1e-8, atol=1e-12):
            raise ValueError("Offline BinnedSpectrum requires uniform bins")
        object.__setattr__(self, "edges", edges.copy())
        object.__setattr__(self, "pdf", pdf.copy())

    @classmethod
    def table(cls, path, elow, ehi, bin_center=False):
        """Offline whole-bin restriction, including loadTable's EOF drop without final whitespace."""
        raw = Path(path).read_bytes()
        a = np.loadtxt(path, ndmin=2)
        if raw and not chr(raw[-1]).isspace():
            a = a[:-1]
        if a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError("expected at least two finite energy/density rows")
        x, pdf = a.T
        width = x[1] - x[0]
        if width <= 0 or not np.allclose(np.diff(x), width, rtol=1e-8, atol=1e-12):
            raise ValueError("spectrum table must have increasing uniform abscissae")
        if not (np.isfinite(elow) and np.isfinite(ehi) and elow < ehi):
            raise ValueError("invalid spectrum interval")
        if np.any(pdf < 0):
            raise ValueError("negative table density")
        centres = x if bin_center else x + width / 2
        keep = (centres - width / 2 >= elow) & (centres + width / 2 <= ehi)
        if keep.sum() < 2:
            raise ValueError("spectrum restriction must retain at least two whole bins")
        centres, pdf = centres[keep], pdf[keep]
        edges = np.linspace(centres[0] - width / 2, centres[-1] + width / 2, pdf.size + 1)
        return cls(edges, pdf)

    def sample(self, rng, n):
        cdf = np.cumsum(self.pdf / self.pdf.sum())
        cdf[-1] = 1.0
        u = rng.random(n)
        i = np.minimum(np.searchsorted(cdf, u, side="right"), self.pdf.size - 1)
        below = np.r_[0.0, cdf[:-1]][i]
        fraction = (u - below) / (cdf[i] - below)
        return self.edges[i] + np.diff(self.edges)[i] * fraction


def bistirlich(energy):
    """Offline PionCaptureSpectrum's Mg fit used for Al RPC; no fitted correction."""
    e = np.asarray(energy, float)
    return np.maximum(134.530 - e, 0) ** 1.29931 * np.exp(-(134.530 - 0.928705 * e) / 9.39676) * (
        0.0477611 - 0.000326349 * e)


def rpc_spectrum(elow=50.0, ehi=139.5, resolution=0.1):
    """Offline's midpoint-binned Bistirlich spectrum, without a partial final bin."""
    if not (np.isfinite([elow, ehi, resolution]).all() and ehi > elow >= 2 * ME and resolution > 0):
        raise ValueError("invalid RPC energy interval or resolution")
    n = int(np.floor((ehi - elow) / resolution + 1e-10))
    edges = elow + np.arange(n + 1) * resolution
    return BinnedSpectrum(edges, bistirlich((edges[1:] + edges[:-1]) / 2))


def kwj_density(energy, mass, asymmetry, nuclear_mass=M_AL):
    """Offline Kroll-Wada-Joseph density in pair mass and energy asymmetry."""
    E, x, y = np.broadcast_arrays(np.asarray(energy, float), np.asarray(mass, float), np.asarray(asymmetry, float))
    valid = (x >= 2 * ME) & (x <= E) & (E > 2 * ME)
    x = np.maximum(x, 2 * ME)
    eta = np.sqrt(np.maximum(0, 1 - (2 * ME / x) ** 2))
    valid &= np.abs(y) <= eta
    M = nuclear_mass + MPI - E
    xe = x / E
    rt = 1 + 0.57 ** 2 / 3 - 2 * 0.064 * xe ** 2
    xl = (1 / (1 - 0.466 * xe ** 2) + 1.88 * (0.57 ** 2 / 6 + 0.064 * (1 - xe ** 2))) / 1.064
    rl = 0.142 * xl ** 2
    den = 2 * E * M + E ** 2
    kr2 = (den ** 2 - 2 * x ** 2 * (2 * M ** 2 + 2 * E * M + E ** 2) + x ** 4) / den ** 2
    prefactor = ((E + M) ** 2 + M ** 2 - x ** 2) / ((E + M) ** 2 + M ** 2)
    trans = rt * ((1 + y ** 2) / x + 4 * ME ** 2 / x ** 3)
    longit = rl * (1 - y ** 2) * 8 * (E + M) ** 2 * x / (den + x ** 2) ** 2
    return np.where(valid, np.sqrt(np.maximum(0, kr2)) * prefactor * (trans + longit), 0.0)


def internal_pair(rng, energy, nuclear_mass=M_AL):
    """Batched rejection draws and correlated e-/e+ four-vectors, as RPCGun."""
    E = np.asarray(energy, float)
    if E.ndim != 1 or not np.isfinite(E).all() or np.any(E <= 2 * ME) or np.any(E >= nuclear_mass + MPI):
        raise ValueError("RPC photon energies must be finite and above pair threshold")
    x, y = np.zeros_like(E), np.zeros_like(E)
    pending = np.arange(E.size)
    envelope = kwj_density(E, 2 * ME, 0.0, nuclear_mass)
    while pending.size:
        e = E[pending]
        xx = rng.uniform(2 * ME, e)
        yy = rng.uniform(-1, 1, pending.size)
        density = kwj_density(e, xx, yy, nuclear_mass)
        if np.any(density > envelope[pending] * (1 + 1e-12)):
            raise ValueError("Offline KWJ rejection envelope exceeded")
        accept = rng.random(pending.size) * envelope[pending] <= density
        x[pending[accept]], y[pending[accept]] = xx[accept], yy[accept]
        pending = pending[~accept]
    ee = (E - y * np.sqrt(np.maximum(0, E ** 2 - x ** 2))) / 2
    ep = E - ee
    pe, pp = np.sqrt(np.maximum(0, ee ** 2 - ME ** 2)), np.sqrt(np.maximum(0, ep ** 2 - ME ** 2))
    ue = C.isotropic(rng, E.size)
    axis = np.eye(3)[np.argmin(np.abs(ue), axis=1)]
    v = np.cross(ue, axis)
    v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(ue, v)
    phi = rng.uniform(0, 2 * np.pi, E.size)
    ct = np.clip((E ** 2 - x ** 2 - pe ** 2 - pp ** 2) / (2 * pe * pp), -1, 1)
    up = ct[:, None] * ue + np.sqrt(np.maximum(0, 1 - ct ** 2))[:, None] * (
        np.cos(phi)[:, None] * v + np.sin(phi)[:, None] * w)
    return dict(p=pe, u=ue, positron_p=pp, positron_u=up, pair_mass=x, asymmetry=y, photon_energy=E)


def canonical_stops(stops):
    """Accept G4 stop rows or beam_point --stops-out rows, without inventing pion stops."""
    if "pdg" in stops:
        return stops
    required = ("x", "t", "q", "foil", "muons")
    if not all(k in stops for k in required):
        raise ValueError("unknown stop schema: need G4 stop columns or beam_point --stops-out")
    x = np.asarray(stops["x"], float)
    q = np.asarray(stops["q"], float)
    if x.shape != (q.size, 3) or not np.isfinite(q).all() or not np.isin(q, [-1, 1]).all():
        raise ValueError("invalid beam_point stop positions or charges")
    out = dict(pdg=np.where(q < 0, 13, -13), at_rest=np.ones(q.size, bool), foil=stops["foil"],
               xstop=x[:, 0], ystop=x[:, 1], zstop=x[:, 2], tstop=stops["t"])
    if "weight" in stops:
        out["weight"] = stops["weight"]
    return out


def from_stops(rng, stops, n, process, spectrum, tau=C.TAU_MU_AL):
    """Draw one primary per stop resample; physical pion times have no muon delay."""
    if process not in ("dio", "rpc-internal", "rpc-external"):
        raise ValueError("unknown target process")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("invalid muon lifetime")
    stops = canonical_stops(stops)
    required = ("pdg", "at_rest", "foil", "xstop", "ystop", "zstop", "tstop")
    if any(k not in stops for k in required):
        raise ValueError("missing required target stop columns")
    shape = np.shape(stops["pdg"])
    if len(shape) != 1 or any(np.shape(stops[k]) != shape for k in required):
        raise ValueError("target stop columns must be equal-length vectors")
    if np.asarray(stops["at_rest"]).dtype.kind != "b":
        raise ValueError("at_rest must be a boolean stop flag")
    pdg = 13 if process == "dio" else -211
    keep = (stops["pdg"] == pdg) & stops["at_rest"] & (stops["foil"] >= 0)
    eligible = np.flatnonzero(keep)
    if not eligible.size:
        raise ValueError(f"no target stops with PDG {pdg}; RPC requires pion stops, not beam muons")
    if "weight" in stops:
        raise ValueError("weighted stop sampling needs an event-normalization adapter; use physical unweighted stops")
    if process == "dio":
        idx = rng.choice(eligible, n)
    else:
        event_keys = ("run", "subrun", "evt")
        if not all(k in stops and np.shape(stops[k]) == shape for k in event_keys):
            raise ValueError("RPC stop sampling needs run/subrun/evt: RPCGun chooses an event, then one pion")
        event = np.column_stack([stops[k][eligible] for k in event_keys])
        if "rpc_event_occurrence" in stops:
            occurrence = np.asarray(stops["rpc_event_occurrence"])
            if occurrence.shape != shape or not np.isfinite(occurrence).all() or np.any(occurrence != np.floor(occurrence)):
                raise ValueError("RPC event occurrence must be a finite integer per stop")
            event = np.column_stack([event, occurrence[eligible]])
        _, groups, counts = np.unique(event, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(groups, kind="stable")
        offset = np.r_[0, np.cumsum(counts)[:-1]]
        chosen = rng.integers(0, counts.size, n)
        slot = offset[chosen] + np.floor(rng.random(n) * counts[chosen]).astype(int)
        idx = eligible[order[slot]]
    x = np.column_stack([stops[k][idx] for k in ("xstop", "ystop", "zstop")]).astype(float)
    t = np.asarray(stops["tstop"][idx], float)
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("nonfinite stop position or time")
    E = spectrum.sample(rng, n)
    if process == "dio":
        if np.any(E <= ME):
            raise ValueError("DIO table must describe total electron energy above its rest mass")
        t = t + rng.exponential(tau, n)
        particle = dict(p=np.sqrt(E ** 2 - ME ** 2), u=C.isotropic(rng, n))
    elif process == "rpc-internal":
        particle = internal_pair(rng, E)
    else:
        particle = dict(p=E, u=C.isotropic(rng, n))
    out = dict(x=x[:, 0], y=x[:, 1], z=x[:, 2], time=t, stop_idx=idx,
               pdg=np.full(n, 22 if process == "rpc-external" else 11, dtype=np.int32), **particle)
    for j, k in enumerate(("px", "py", "pz")):
        out[k] = particle["p"] * particle["u"][:, j]
    return out


def electron_starts(table):
    """Validate explicit sources for the electron-only CE transport; never treat photons as e-."""
    if "pdg" not in table:
        raise ValueError("source table must explicitly identify pdg=11 electron starts")
    pdg = np.asarray(table["pdg"])
    if pdg.ndim != 1 or not pdg.size or not np.all(pdg == 11):
        raise ValueError("CE transport accepts electron starts only; external RPC needs photon conversion")
    keys = ("x", "y", "z", "px", "py", "pz", "time")
    if any(k not in table or np.shape(table[k]) != pdg.shape or not np.isfinite(table[k]).all() for k in keys):
        raise ValueError("invalid source columns")
    x = np.column_stack([table[k] for k in ("x", "y", "z")]).astype(float)
    pv = np.column_stack([table[k] for k in ("px", "py", "pz")]).astype(float)
    p = np.linalg.norm(pv, axis=1)
    if np.any(p <= 0):
        raise ValueError("source momentum must be positive")
    if "weight" in table and (np.shape(table["weight"]) != pdg.shape or not np.all(table["weight"] == 1)):
        raise ValueError("weighted sources require a weighted-yield transport adapter")
    return x, pv / p[:, None], p, np.asarray(table["time"], float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--process", choices=("dio", "rpc-internal", "rpc-external"), required=True)
    ap.add_argument("--stops", required=True, help="target stops: pdg, at_rest, foil, xstop/ystop/zstop, tstop")
    ap.add_argument("--spectrum-table", help="DIO total-energy table from the selected Offline release")
    ap.add_argument("--elow", type=float, required=True)
    ap.add_argument("--ehi", type=float, required=True)
    ap.add_argument("--bin-center", action="store_true", help="Offline BinCenter=true (default false)")
    ap.add_argument("--resolution", type=float, default=0.1, help="RPC photon energy bin width [MeV]")
    ap.add_argument("--physical-pion-times", action="store_true", help="certify already-decayed pion stop ensemble")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.process == "dio" and not a.spectrum_table:
        ap.error("DIO needs --spectrum-table from the reference release")
    if a.process != "dio" and not a.physical_pion_times:
        ap.error("RPC needs --physical-pion-times; decay-off stops need ancestry survival weights and a separate adapter")
    spectrum = (BinnedSpectrum.table(a.spectrum_table, a.elow, a.ehi, a.bin_center) if a.process == "dio"
                else rpc_spectrum(a.elow, a.ehi, a.resolution))
    with np.load(a.stops) as stops:
        out = from_stops(np.random.default_rng(a.seed), stops, a.n, a.process, spectrum)
    meta = dict(process=a.process, seed=a.seed, n=a.n, elow=float(spectrum.edges[0]), ehi=float(spectrum.edges[-1]),
                bin_center=a.bin_center, resolution=a.resolution,
                normalization="per generated electron in interval" if a.process == "dio" else "per generated RPC capture in interval",
                stops_sha256=hashlib.sha256(Path(a.stops).read_bytes()).hexdigest(),
                spectrum_sha256=hashlib.sha256(Path(a.spectrum_table).read_bytes()).hexdigest() if a.spectrum_table else None,
                physical_pion_times=a.physical_pion_times, external_conversion_included=False)
    np.savez_compressed(a.out, **out, metadata=json.dumps(meta))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
