"""G4BL-format Mu2e field maps (BFieldMaps/<set>/<Map>.header + .bin) and a batched trilinear lookup in JAX.

Binary layout (Offline BFieldManagerMaker::readG4BLBinary): 4-byte 0xDEADBEEF marker, then Hep3Vector doubles
(Bx, By, Bz in T) for every grid point, index ix*ny*nz + iy*nz + iz. Interpolation as BFGridMap::interpolateTriLinear;
points outside the grid get B = 0.
"""
import os
import re
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

MARKER = 0xDEADBEEF


class FieldMap(NamedTuple):
    lo: object      # (3,) grid origin [mm]
    d: object       # (3,) spacing [mm]
    n: object       # (3,) int points per axis
    B: object       # (nx, ny, nz, 3) [T]


def read_header(path):
    with open(path) as f:
        text = f.read()
    g = dict((k, float(v)) for k, v in re.findall(r'(\w+)=([-\d.eE+]+)', text.split("grid", 1)[1].split("\n")[0]))
    lo = np.array([g["X0"], g["Y0"], g["Z0"]])
    d = np.array([g["dX"], g["dY"], g["dZ"]])
    n = np.array([int(g["nX"]), int(g["nY"]), int(g["nZ"])])
    return lo, d, n


def load(header_path, dtype=np.float64):
    """FieldMap from a .header and its sibling .bin (numpy arrays; move to device with to_device)."""
    lo, d, n = read_header(header_path)
    binf = header_path[: -len(".header")] + ".bin"
    npts = int(np.prod(n))
    size = os.path.getsize(binf)
    if size != 4 + 24 * npts:
        raise ValueError(f"{binf}: size {size} is not marker + {npts} Hep3Vectors (old format with coordinates?)")
    marker = np.fromfile(binf, dtype=np.uint32, count=1)[0]
    if marker != MARKER:
        raise ValueError(f"{binf}: endian marker {marker:#x}")
    B = np.fromfile(binf, dtype=np.float64, offset=4).reshape(n[0], n[1], n[2], 3).astype(dtype)
    return FieldMap(lo, d, n, B)


def to_device(fm):
    return FieldMap(jnp.asarray(fm.lo), jnp.asarray(fm.d), jnp.asarray(fm.n), jnp.asarray(fm.B))


def field(fm, pos):
    """B [T] at positions pos (N, 3) [mm], Mu2e frame; zero outside the grid."""
    u = (pos - fm.lo) / fm.d
    i = jnp.floor(u).astype(jnp.int32)
    inside = jnp.all((i >= 0) & (i < fm.n - 1), axis=-1)
    i = jnp.clip(i, 0, fm.n - 2)
    f = u - i
    out = jnp.zeros(pos.shape, fm.B.dtype)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out = out + (wx * wy * wz)[:, None] * fm.B[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return jnp.where(inside[:, None], out, 0.0)
