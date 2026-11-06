"""Opt-in TS field bank: released map order, trilinear lookup and header Y reflection.

Inputs use mm and tesla. Selection is first-containing inner map, then first-containing
outer map; fields are never summed. Unlike Offline's shared last-map cache, this bank
has fixed ordering in overlap regions. This difference is a development limitation.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
from mu2efastsim import cvmfsget, ehelix

from . import simpleconfig


def _grid_field(n, grid):
    """Unit-grid core arithmetic with ordered corners and N*8*3 float scratch.

    The gather and accumulation share storage (192 bytes/row for float64 fields).
    Weights and indices require additional O(N*8) scratch; no scratch is retained
    between queries. Accumulation includes the core's initial positive zero.
    """
    n = np.asarray(n)
    corners = np.array([[dx, dy, dz] for dx in (0, 1)
                        for dy in (0, 1) for dz in (0, 1)])
    offsets = (corners[:, 0] * n[1] + corners[:, 1]) * n[2] + corners[:, 2]
    flat = grid.reshape(-1, 3)

    def field(x):
        uu = np.asarray(x, dtype=float)
        i = np.floor(uu).astype(int)
        inside = np.all((i >= 0) & (i < n - 1), axis=1)
        i = np.clip(i, 0, n - 2)
        f = uu - i
        wx, wy, wz = np.stack((1 - f, f), axis=1).transpose(2, 0, 1)
        weights = ((wx[:, :, None, None] * wy[:, None, :, None])
                   * wz[:, None, None, :]).reshape(-1, 8)
        base = (i[:, 0] * n[1] + i[:, 1]) * n[2] + i[:, 2]
        values = flat[base[:, None] + offsets]
        values *= weights[:, :, None]
        values[:, 0, :] += 0.0
        np.add.accumulate(values, axis=1, out=values)
        return np.where(inside[:, None], values[:, -1, :], 0.0)

    return field


class FieldBank:
    """Callable field with explicit domain masks; missing coverage is not silently valid vacuum."""

    def __init__(self, headers, scale=1.0):
        if not np.isfinite(scale):
            raise ValueError("field scale must be finite")
        self.maps = []
        self._grid_max = []
        self._domains = []
        self.max_field_t = 0.0
        self.scale = float(scale)
        for header in headers:
            header = Path(header)
            lo, spacing, n, grid = ehelix.read_g4bl(str(header))
            bound = abs(self.scale) * np.linalg.norm(np.max(np.abs(grid), axis=(0, 1, 2)))
            self.max_field_t = max(self.max_field_t, bound * (1 + 1e-12)) if np.isfinite(bound) else np.inf
            hi = lo + spacing * (n - 1)
            reflect = "extendY flip=By" in header.read_text()
            if reflect and lo[1] != 0:
                raise ValueError("Y-reflected map must start at y=0")
            interp = _grid_field(n, grid)
            self.maps.append((lo, hi, spacing, n, reflect, interp))
            domain_lo = lo.copy()
            if reflect:
                domain_lo[1] = -hi[1]
            self._domains.append((domain_lo, hi))
            # Core interpolation uses half-open cells; retain closed physical faces.
            self._grid_max.append(np.nextafter((n - 1).astype(float), 0))
        if not self.maps:
            raise ValueError("field bank needs at least one map")

    def evaluate(self, x):
        """Return B[N,3] and covered[N]; preserve priority and flip only By below y=0."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
            raise ValueError("positions must be finite N by 3")
        out, covered = np.zeros_like(x), np.zeros(len(x), bool)
        remaining = np.arange(len(x))
        for (lo, hi, spacing, n, reflect, interp), grid_max in zip(self.maps, self._grid_max):
            if not remaining.size:
                break
            # First-containing precedence: later maps only see unclaimed rows.
            pos = x[remaining]
            if reflect:
                pos[:, 1] = np.abs(pos[:, 1])
            mask = np.all((pos >= lo) & (pos <= hi), axis=1)
            if mask.any():
                selected = remaining[mask]
                grid_position = (pos[mask] - lo) / spacing
                out[selected] = interp(np.clip(grid_position, 0, grid_max))
                if reflect:
                    out[selected, 1] *= np.where(x[selected, 1] < 0, -1.0, 1.0)
                covered[selected] = True
                remaining = remaining[~mask]
        return self.scale * out, covered

    def safety(self, x):
        """Lower bound [mm] to uncovered space, zero outside every map.

        The largest containing-map rectangular clearance certifies a sphere in
        one map, without assuming a union of overlaps is rectangular. Reflected
        maps extend continuously across y=0. This does not certify constant map
        selection inside the sphere; field evaluation keeps its usual priority.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
            raise ValueError("positions must be finite N by 3")
        clearance = np.zeros(len(x))
        for lo, hi in self._domains:
            clearance = np.maximum(clearance, np.min(np.minimum(x - lo, hi - x), axis=1))
        return np.nextafter(clearance, 0.0)

    def __call__(self, x):
        return self.evaluate(x)[0]


def from_config(config, offline_root, maps_dir):
    """Load the ordered inner/outer map lists of an explicitly selected field config."""
    params = simpleconfig.read(str(config), str(offline_root))
    names = params["bfield.innerMaps"] + params["bfield.outerMaps"]
    return FieldBank([Path(maps_dir) / Path(n).name for n in names], params["bfield.scaleFactor"])


def fetch(out, release, field_config):
    """Fetch a released field config and precisely its named maps; no geometry or jobs created."""
    out = Path(out)
    repo = cvmfsget.Repo("mu2e.opensciencegrid.org")
    config = out / "offline" / field_config
    fetched = set()

    def fetch_config(name):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "Offline":
            raise ValueError(f"expected release-relative Offline config: {name}")
        if name in fetched:
            return
        fetched.add(name)
        dest = out / "offline" / name
        repo.fetch(f"/Musings/Offline/{release}/{name}", str(dest))
        text = re.sub(r'//[^\n]*', '', dest.read_text())
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', text, re.M):
            fetch_config(include)

    fetch_config(field_config)
    params = simpleconfig.read(str(config), str(out / "offline"))
    names = params["bfield.innerMaps"] + params["bfield.outerMaps"]
    for name in dict.fromkeys(names):
        for suffix in (".header", ".bin"):
            src = str(Path(name).with_suffix(suffix))
            dest = out / "maps" / Path(src).name
            repo.fetch(f"/DataFiles/{src}", str(dest))
            print(dest, flush=True)
    return {"release": release, "field_config": field_config, "map_names": names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--release", required=True)
    parser.add_argument("--field-config", default="Offline/Mu2eG4/geom/bfgeom_DSOff.txt")
    args = parser.parse_args(argv)
    print(json.dumps(fetch(args.out, args.release, args.field_config), indent=2))


if __name__ == "__main__":
    main()
