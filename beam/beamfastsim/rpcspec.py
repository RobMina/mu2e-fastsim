"""Internal RPC source following Offline RPCGun and PionCaptureSpectrum.

Total energies and momenta are MeV, positions mm and times ns. The Bistirlich
photon spectrum is midpoint-binned as in BinnedSpectrum; the correlated pair
uses the Kroll-Wada-Joseph density. Input is a pool of physical negative pion
stops in target foils, not decay-disabled weighted pions or stopped muons. This
module generates both particles but performs no transport or absolute-rate prediction.
"""
import argparse
from dataclasses import dataclass
import json

import numpy as np

ME = 0.510998910
MPI = 139.57018
M_AL = 26.981539 * 931.494028


def isotropic(rng, n):
    """Uniform unit vectors on the sphere."""
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(np.maximum(0, 1 - z * z))
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def bistirlich(energy):
    """The Mg photon-spectrum fit used for Al in Offline PionCaptureSpectrum."""
    e = np.asarray(energy, float)
    return np.maximum(134.530 - e, 0) ** 1.29931 * np.exp(-(134.530 - 0.928705 * e) / 9.39676) * (
        0.0477611 - 0.000326349 * e)


@dataclass(frozen=True)
class PhotonSpectrum:
    """Piecewise-uniform Bistirlich bins, matching Offline RandGeneral sampling."""
    elow: float = 50.0
    ehi: float = 139.5
    resolution: float = 0.1

    def __post_init__(self):
        if not (np.isfinite([self.elow, self.ehi, self.resolution]).all()
                and self.ehi > self.elow >= 2 * ME and self.resolution > 0):
            raise ValueError("invalid RPC spectrum interval or resolution")
        if self.edges.size < 2 or not self.pdf.sum() > 0:
            raise ValueError("RPC spectrum has no positive complete bins")

    @property
    def edges(self):
        n = int(np.floor((self.ehi - self.elow) / self.resolution + 1e-10))
        return self.elow + np.arange(n + 1) * self.resolution

    @property
    def pdf(self):
        edges = self.edges
        return bistirlich((edges[1:] + edges[:-1]) / 2)

    def sample(self, rng, n):
        pdf, edges = self.pdf, self.edges
        cdf = np.cumsum(pdf / pdf.sum())
        cdf[-1] = 1.0
        u = rng.random(n)
        i = np.minimum(np.searchsorted(cdf, u, side="right"), pdf.size - 1)
        lower = np.r_[0.0, cdf[:-1]][i]
        return edges[i] + np.diff(edges)[i] * (u - lower) / (cdf[i] - lower)


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
    """Vectorised rejection sampling and correlated electron/positron momenta."""
    E = np.asarray(energy, float)
    if (not np.isfinite(nuclear_mass) or nuclear_mass <= 0 or E.ndim != 1
            or not np.isfinite(E).all() or np.any(E <= 2 * ME) or np.any(E >= nuclear_mass + MPI)):
        raise ValueError("RPC energies must be finite, above pair threshold and below the recoil mass")
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
        accept = rng.random(pending.size) * envelope[pending] < density
        x[pending[accept]], y[pending[accept]] = xx[accept], yy[accept]
        pending = pending[~accept]
    ee = (E - y * np.sqrt(np.maximum(0, E ** 2 - x ** 2))) / 2
    ep = E - ee
    pe, pp = np.sqrt(np.maximum(0, ee ** 2 - ME ** 2)), np.sqrt(np.maximum(0, ep ** 2 - ME ** 2))
    ue = isotropic(rng, E.size)
    axis = np.eye(3)[np.argmin(np.abs(ue), axis=1)]
    v = np.cross(ue, axis)
    v /= np.linalg.norm(v, axis=1)[:, None]
    w = np.cross(ue, v)
    phi = rng.uniform(0, 2 * np.pi, E.size)
    ct = np.clip((E ** 2 - x ** 2 - pe ** 2 - pp ** 2) / (2 * pe * pp), -1, 1)
    up = ct[:, None] * ue + np.sqrt(np.maximum(0, 1 - ct ** 2))[:, None] * (
        np.cos(phi)[:, None] * v + np.sin(phi)[:, None] * w)
    return dict(p=pe, u=ue, positron_p=pp, positron_u=up, pair_mass=x, asymmetry=y, photon_energy=E)


def from_stops(rng, stops, n, spectrum=None, *, tmin=350.0, stopping_code=78):
    """Resample an eligible event uniformly, then one of its selected pion stops.

    Input is a target-stop pool: at_rest and foil >= 0 restrict that scope,
    beyond the stock species, stopping-process and minimum-time selection.
    StopSelector has already conditioned the production on an eligible stop.
    Times are copied directly: physical pions get neither a muon lifetime delay
    nor an extra pion-survival weight. Output normalisation is per generated
    capture after this stop selection, not per proton or incident pion.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not np.isfinite(tmin):
        raise ValueError("tmin must be finite")
    keys = ("run", "subrun", "evt", "pdg", "at_rest", "foil", "codestop", "xstop", "ystop", "zstop", "tstop")
    if any(k not in stops for k in keys):
        raise ValueError("missing event-keyed physical pion stop columns")
    a = {k: np.asarray(stops[k]) for k in keys}
    shape = a["pdg"].shape
    if len(shape) != 1 or any(v.shape != shape or not np.isfinite(v).all() for v in a.values()):
        raise ValueError("stop columns must be finite equal-length vectors")
    if a["at_rest"].dtype.kind != "b":
        raise ValueError("at_rest must be boolean")
    if "weight" in stops:
        raise ValueError("use physical pion stops, not decay-disabled weighted stops")
    for k in ("run", "subrun", "evt"):
        if np.any(a[k] != np.floor(a[k])):
            raise ValueError("event identities must be integers")
    keep = ((a["pdg"] == -211) & a["at_rest"] & (a["foil"] >= 0)
            & (a["codestop"] == stopping_code) & (a["tstop"] >= tmin))
    eligible = np.flatnonzero(keep)
    if not eligible.size:
        raise ValueError("no selected negative pion stops")
    events = np.column_stack([a[k][eligible] for k in ("run", "subrun", "evt")])
    _, groups, counts = np.unique(events, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(groups, kind="stable")
    offset = np.r_[0, np.cumsum(counts)[:-1]]
    chosen = rng.integers(0, counts.size, n)
    slot = offset[chosen] + np.floor(rng.random(n) * counts[chosen]).astype(int)
    idx = eligible[order[slot]]
    pair = internal_pair(rng, (spectrum or PhotonSpectrum()).sample(rng, n))
    out = dict(x=a["xstop"][idx], y=a["ystop"][idx], z=a["zstop"][idx], time=a["tstop"][idx],
               stop_idx=idx, pdg=np.full(n, 11, dtype=np.int32), **pair)
    for j, k in enumerate(("px", "py", "pz")):
        out[k] = pair["p"] * pair["u"][:, j]
        out["positron_" + k] = pair["positron_p"] * pair["positron_u"][:, j]
    for k in ("run", "subrun", "evt"):
        out["stop_" + k] = a[k][idx].astype(np.int64)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stops", required=True, help="NPZ of event-keyed physical pion stops in target foils")
    ap.add_argument("--n", type=int, required=True, help="generated captures, one pair each")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--elow", type=float, default=50.0)
    ap.add_argument("--ehi", type=float, default=139.5)
    ap.add_argument("--resolution", type=float, default=0.1)
    ap.add_argument("--tmin", type=float, default=350.0)
    ap.add_argument("--stopping-code", type=int, default=78)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spectrum = PhotonSpectrum(a.elow, a.ehi, a.resolution)
    with np.load(a.stops, allow_pickle=False) as stops:
        out = from_stops(np.random.default_rng(a.seed), stops, a.n, spectrum,
                         tmin=a.tmin, stopping_code=a.stopping_code)
    meta = dict(process="rpc-internal", seed=a.seed, captures=a.n, stops=a.stops,
                elow=float(spectrum.edges[0]), ehi=float(spectrum.edges[-1]), resolution=a.resolution,
                tmin=a.tmin, stopping_code=a.stopping_code, normalization="per generated capture after stop selection")
    np.savez_compressed(a.out, **out, metadata=json.dumps(meta))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
