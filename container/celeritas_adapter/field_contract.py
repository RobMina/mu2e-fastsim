"""Validate diagnostic map JSON and generate an asymmetric synthetic field; no DSMap assumptions."""

import argparse
import json
from pathlib import Path
import numpy as np


CONTRACT = {"format": "m2fs-cartmap-v1", "length_unit": "mm", "field_unit": "tesla",
            "coordinates": "gdml-world", "outside_policy": "world-contained", "order": "xyz-component"}


def validate(data):
    if any(data.get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("Wrong map format, units, coordinates, order or boundary policy")
    axes = data["axes"]
    if len(axes) != 3:
        raise ValueError("Expected three axes")
    n = []
    for a in axes:
        if type(a["num"]) is not int or not 3 <= a["num"] <= 1000000:
            raise ValueError("Invalid grid count")
        lo, hi = a["min_mm"], a["max_mm"]
        if (type(lo) not in (int, float) or type(hi) not in (int, float)
                or not np.isfinite([lo, hi]).all() or hi <= lo):
            raise ValueError("Invalid axis extent")
        spacing = (hi - lo) / (a["num"] - 1)
        if not np.isfinite(spacing) or spacing <= 8 * float(np.finfo(np.float32).eps) * max(abs(lo), abs(hi)):
            raise ValueError("Spacing unresolved by covfie float coordinates")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            flo, fhi = np.float32(lo / 10), np.float32(hi / 10)
            scale = np.float32(10 / spacing)
            good = (np.isfinite([flo, fhi, scale, scale * flo, scale * fhi]).all()
                    and flo < fhi and scale > 0)
        if not good:
            raise ValueError("Grid affine transform is not representable in CGS covfie floats")
        n.append(a["num"])
    count = 3 * int(np.prod(n, dtype=np.int64))
    if count > 30000000:
        raise ValueError("Map exceeds diagnostic size limit")
    raw = np.asarray(data["field"])
    if raw.dtype.kind not in "fiu" or raw.shape != (count,):
        raise ValueError("Field must be a flat numeric xyz-component array")
    b = raw.astype(np.float64)
    if not np.isfinite(b).all() or np.max(np.abs(b)) > np.finfo(np.float32).max / 10000:
        raise ValueError("Field not representable in CGS float storage")
    return b.reshape(*n, 3)


def require_world_inside(data, lower_mm, upper_mm):
    validate(data)
    lo = np.array([a["min_mm"] for a in data["axes"]])
    hi = np.array([a["max_mm"] for a in data["axes"]])
    n = np.array([a["num"] for a in data["axes"]])
    lower, upper = np.asarray(lower_mm), np.asarray(upper_mm)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(lower >= upper) or np.any(lower < lo + (hi - lo) / (n - 1))
            or np.any(upper > hi - (hi - lo) / (n - 1))):
        raise ValueError("World must fit inside grid with a one-cell margin")


def interpolate(data, points_mm):
    """Double-precision analytic oracle, not a replacement for covfie host/CUDA checks."""
    b = validate(data)
    p = np.asarray(points_mm, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite N x 3 points")
    lo = np.array([a["min_mm"] for a in data["axes"]])
    hi = np.array([a["max_mm"] for a in data["axes"]])
    n = np.array(b.shape[:3])
    if np.any(p < lo) or np.any(p >= hi):
        raise ValueError("Point outside half-open support")
    u = (p - lo) / ((hi - lo) / (n - 1))
    i = np.floor(u).astype(int)
    f = u - i
    result = np.zeros_like(p)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz])
                weight = np.prod(np.where(offset, f, 1 - f), axis=1)
                idx = i + offset
                result += weight[:, None] * b[idx[:, 0], idx[:, 1], idx[:, 2]]
    return result


def synthetic(constant=False):
    axes = [{"min_mm": -2000, "max_mm": 2000, "num": n} for n in (5, 6, 7)]
    grid = np.stack(np.meshgrid(*(np.linspace(a["min_mm"], a["max_mm"], a["num"])
                                 for a in axes), indexing="ij"), axis=-1)
    field = np.zeros_like(grid)
    field[..., 2] = 1
    if not constant:
        field += grid @ np.array([[1e-5, 2e-5, 3e-5], [4e-5, -2e-5, 6e-5], [7e-5, 8e-5, 1e-5]])
    return dict(CONTRACT, axes=axes, field=field.ravel(order="C").tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("synthetic", "check"))
    p.add_argument("path", type=Path)
    p.add_argument("--constant", action="store_true")
    args = p.parse_args()
    if args.action == "synthetic":
        data = synthetic(args.constant)
        validate(data)
        with args.path.open("x") as out:
            json.dump(data, out, indent=2, allow_nan=False)
    else:
        data = json.loads(args.path.read_text())
        print("shape:", validate(data).shape)


if __name__ == "__main__":
    main()
