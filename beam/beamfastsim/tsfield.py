"""Opt-in NumPy CPU reference for G4BL grid fields; not a transport implementation.

Coordinates are Mu2e mm and fields are tesla. The header origin shift is subtracted
from the G4BL grid origin, with Offline's pre-Mau7 default (-3904, 0, 7929).
Only modern .header/.bin pairs (marker plus packed float64 field triples) are
supported. Both marker-declared byte orders are decoded; coordinate-bearing old
binaries, text maps, parametric maps and other extension commands are rejected.

``load(path).field(points)`` and ``FieldMapBank(inner=..., outer=...).field(points)``
raise CoverageError for uncovered points. ``return_coverage=True`` explicitly
requests (B, covered), with NaN fields for uncovered lanes, never silent zeros.
Bounds are closed, including the last grid node; touching inner maps are rejected
to avoid Offline's history-dependent shared-face selection. No JAX import is used.

Audit without transport: ``python -m beamfastsim.tsfield --inner TS.header
--outer Outer.header --point 0 0 0``. Asset hashes are observations, not release
approval, and the CPU reference still requires upstream/physics validation.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np


MARKER = 0xDEADBEEF
LEGACY_ORIGIN_SHIFT = (-3904.0, 0.0, 7929.0)
_GRID_KEYS = frozenset(("X0", "Y0", "Z0", "nX", "nY", "nZ", "dX", "dY", "dZ"))
_ORIGIN = re.compile(r"#\s*Origin\s+shift\s+for\s+Mu2e:\s*(.*?)\s*$")


class CoverageError(ValueError):
    """A field query contains uncovered lanes; indices refer to flattened batches."""

    def __init__(self, covered):
        self.covered = np.array(covered, dtype=bool, copy=True)
        self.indices = np.flatnonzero(~self.covered.reshape(-1))
        super().__init__(f"{self.indices.size} uncovered field point(s); flat indices "
                         f"{self.indices[:10].tolist()}")


def _readonly(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_triplet(value, label):
    result = _readonly(value, np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite length-3 array")
    return result


def _dimension(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("grid dimensions must be exact integers >= 2") from exc
    limit = np.iinfo(np.intp).max // 24
    if not number.is_finite() or number < 2 or number > limit or number != number.to_integral_value():
        raise ValueError("grid dimensions must be exact integers >= 2 within addressable size")
    return int(number)


@dataclass(frozen=True, eq=False)
class GridHeader:
    """Validated grid metadata; lo/hi are in Mu2e coordinates."""

    path: Path
    grid_origin: np.ndarray
    origin_shift: np.ndarray
    n: np.ndarray
    d: np.ndarray
    extend_y: bool = False
    explicit_origin: bool = False

    def __post_init__(self):
        origin = _finite_triplet(self.grid_origin, "grid origin")
        shift = _finite_triplet(self.origin_shift, "origin shift")
        spacing = _finite_triplet(self.d, "spacing")
        dims = np.asarray(self.n)
        if dims.shape != (3,):
            raise ValueError("grid dimensions must have shape (3,)")
        dims = _readonly([_dimension(v) for v in dims], np.intp)
        npts = math.prod(int(v) for v in dims)
        if npts > (np.iinfo(np.intp).max - 4) // 24:
            raise ValueError("grid npts exceeds addressable binary size")
        if np.any(spacing <= 0):
            raise ValueError("grid spacing must be positive")
        with np.errstate(over="ignore", invalid="ignore"):
            lo = origin - shift
            hi = lo + (dims - 1) * spacing
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(hi <= lo):
            raise ValueError("grid bounds must be finite and increasing")
        if self.extend_y and lo[1] < 0:
            raise ValueError("extendY requires a nonnegative Mu2e Y grid")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "grid_origin", origin)
        object.__setattr__(self, "origin_shift", shift)
        object.__setattr__(self, "n", dims)
        object.__setattr__(self, "d", spacing)

    @property
    def lo(self):
        return self.grid_origin - self.origin_shift

    @property
    def hi(self):
        return self.lo + (self.n - 1) * self.d

    @property
    def npts(self):
        return math.prod(int(v) for v in self.n)

    @property
    def boxes(self):
        """Exact coverage boxes, retaining any gap about the mirror plane."""
        lo, hi = self.lo, self.hi
        if not self.extend_y:
            return ((lo, hi),)
        mirror_lo, mirror_hi = lo.copy(), hi.copy()
        mirror_lo[1], mirror_hi[1] = -hi[1], -lo[1]
        return ((lo, hi), (mirror_lo, mirror_hi))


def read_header(path):
    """Parse a strict binary-map header; reject ambiguous or unsupported records."""
    path = Path(path)
    if path.suffix != ".header":
        raise ValueError("only .header with a sibling .bin is supported")
    grid = None
    shift = None
    extend_y = False
    data_seen = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        origin = _ORIGIN.fullmatch(line)
        if origin:
            if shift is not None:
                raise ValueError(f"{path}:{lineno}: duplicate origin shift")
            tokens = origin.group(1).split("#", 1)[0].split()
            if len(tokens) != 3:
                raise ValueError(f"{path}:{lineno}: malformed origin shift")
            shift = _finite_triplet([float(v) for v in tokens], "origin shift")
            continue
        if line.startswith("#"):
            if re.match(r"#\s*Origin\s+shift", line):
                raise ValueError(f"{path}:{lineno}: malformed origin shift comment")
            continue
        line = line.split("#", 1)[0].strip()
        tokens = line.split()
        command = tokens[0]
        if command == "grid" and not data_seen:
            if grid is not None:
                raise ValueError(f"{path}:{lineno}: duplicate grid")
            grid = {}
            for token in tokens[1:]:
                if token.count("=") != 1:
                    raise ValueError(f"{path}:{lineno}: malformed grid token {token!r}")
                key, value = token.split("=")
                if key not in _GRID_KEYS or key in grid or not value:
                    raise ValueError(f"{path}:{lineno}: unknown or duplicate grid key {key!r}")
                grid[key] = value
            if grid.keys() != _GRID_KEYS:
                raise ValueError(f"{path}:{lineno}: missing grid keys")
        elif command == "param" and not data_seen:
            params = dict(token.split("=", 1) for token in tokens[1:] if token.count("=") == 1)
            if len(params) != len(tokens) - 1 or set(params) - {"maxline", "current", "gradient", "normB", "normE"}:
                raise ValueError(f"{path}:{lineno}: unsupported param record")
            if "maxline" in params:
                maxline = float(params["maxline"])
                if not math.isfinite(maxline) or maxline < 1 or maxline != int(maxline):
                    raise ValueError(f"{path}:{lineno}: invalid maxline")
            for key in set(params) - {"maxline"}:
                if float(params[key]) != 1.0:
                    raise ValueError(f"{path}:{lineno}: only unity field parameters are supported")
        elif command.startswith("extend"):
            if tokens != ["extendY", "flip=By"] or extend_y or data_seen:
                raise ValueError(f"{path}:{lineno}: only one extendY flip=By is supported")
            extend_y = True
        elif tokens == ["data"] and not data_seen and grid is not None:
            data_seen = True
        else:
            raise ValueError(f"{path}:{lineno}: unsupported header record {line!r}")
    if grid is None:
        raise ValueError(f"{path}: missing grid")
    return GridHeader(path, [float(grid[k]) for k in ("X0", "Y0", "Z0")],
                      LEGACY_ORIGIN_SHIFT if shift is None else shift,
                      [_dimension(grid[k]) for k in ("nX", "nY", "nZ")],
                      [float(grid[k]) for k in ("dX", "dY", "dZ")],
                      extend_y, shift is not None)


def _positions(pos):
    result = np.asarray(pos, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ValueError("positions must have shape (..., 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("positions must be finite")
    return result


def _finish(values, covered, return_coverage):
    if return_coverage:
        return values, covered
    if not np.all(covered):
        raise CoverageError(covered)
    return values


@dataclass(frozen=True, eq=False)
class FieldMap:
    """An immutable-by-convention, validated NumPy field grid."""

    header: GridHeader
    B: np.ndarray
    byte_order: str = "native"

    def __post_init__(self):
        if not isinstance(self.header, GridHeader):
            raise TypeError("header must be a GridHeader")
        data = _readonly(self.B, np.float64)
        if data.shape != tuple(self.header.n) + (3,) or not np.all(np.isfinite(data)):
            raise ValueError("field array must be finite with shape (nx, ny, nz, 3)")
        object.__setattr__(self, "B", data)

    @property
    def lo(self):
        return self.header.lo

    @property
    def hi(self):
        return self.header.hi

    @property
    def d(self):
        return self.header.d

    @property
    def n(self):
        return self.header.n

    def coverage(self, pos):
        points = _positions(pos).copy()
        if self.header.extend_y:
            points[..., 1] = np.abs(points[..., 1])
        return np.all((points >= self.lo) & (points <= self.hi), axis=-1)

    def field(self, pos, *, return_coverage=False):
        """Batched trilinear interpolation, including exact lower/upper endpoints."""
        points = _positions(pos)
        shape = points.shape
        points = points.reshape(-1, 3)
        covered = self.coverage(points)
        values = np.full(points.shape, np.nan, dtype=np.float64)
        query = points[covered].copy()
        if self.header.extend_y:
            query[:, 1] = np.abs(query[:, 1])
        u = np.clip((query - self.lo) / self.d, 0, self.n - 1)
        u = np.where(query == self.hi, self.n - 1, u)
        index = np.minimum(np.floor(u).astype(np.intp), self.n - 2)
        fraction = np.clip(u - index, 0, 1)
        interpolated = np.zeros(query.shape, dtype=np.float64)
        for corner in itertools.product((0, 1), repeat=3):
            offset = np.asarray(corner)
            weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
            node = index + offset
            interpolated += weight[:, None] * self.B[node[:, 0], node[:, 1], node[:, 2]]
        if self.header.extend_y:
            interpolated[:, 1] *= np.where(points[covered, 1] < 0, -1.0, 1.0)
        values[covered] = interpolated
        return _finish(values.reshape(shape), covered.reshape(shape[:-1]), return_coverage)


def load(header_path):
    """Load marker + nx*ny*nz float64 triples; z index varies fastest."""
    header = read_header(header_path)
    binary = header.path.with_suffix(".bin")
    expected_size = 4 + 24 * header.npts
    with binary.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size != expected_size:
            old = " (old coordinate-bearing layout is unsupported)" if size == 4 + 48 * header.npts else ""
            raise ValueError(f"{binary}: size {size}; expected {expected_size}{old}")
        marker = stream.read(4)
        if marker == struct.pack("<I", MARKER):
            order = "<"
        elif marker == struct.pack(">I", MARKER):
            order = ">"
        else:
            raise ValueError(f"{binary}: invalid endian marker {marker.hex()}")
        data = np.fromfile(stream, dtype=order + "f8", count=3 * header.npts)
        if data.size != 3 * header.npts or stream.read(1):
            raise ValueError(f"{binary}: binary changed size or is truncated")
    return FieldMap(header, data.reshape(tuple(header.n) + (3,)),
                    "little" if order == "<" else "big")


class FieldMapBank:
    """Ordered inner priority then first outer; overlapping or touching inner maps fail."""

    def __init__(self, inner=(), outer=()):
        self.inner = tuple(inner)
        self.outer = tuple(outer)
        if not all(isinstance(item, FieldMap) for item in self.inner + self.outer):
            raise TypeError("bank entries must be loaded FieldMap objects")
        for left, right in itertools.combinations(self.inner, 2):
            for (alo, ahi), (blo, bhi) in itertools.product(left.header.boxes, right.header.boxes):
                if np.all(np.maximum(alo, blo) <= np.minimum(ahi, bhi)):
                    raise ValueError(f"overlapping inner coverage (including shared boundaries): "
                                     f"{left.header.path} and {right.header.path}")

    def map_indices(self, pos):
        """Map indices in inner+outer order, or -1 for uncovered coordinates."""
        points = _positions(pos)
        selected = np.full(points.shape[:-1], -1, dtype=np.intp)
        for index, item in enumerate(self.inner + self.outer):
            selected = np.where((selected < 0) & item.coverage(points), index, selected)
        return selected

    def coverage(self, pos):
        return self.map_indices(pos) >= 0

    def field(self, pos, *, return_coverage=False):
        points = _positions(pos)
        selected = self.map_indices(points).reshape(-1)
        flat = points.reshape(-1, 3)
        values = np.full(flat.shape, np.nan, dtype=np.float64)
        for index, item in enumerate(self.inner + self.outer):
            lanes = selected == index
            values[lanes] = item.field(flat[lanes])
        return _finish(values.reshape(points.shape), (selected >= 0).reshape(points.shape[:-1]),
                       return_coverage)


def _asset(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def audit_map(item):
    """Hash source assets and report validated grid and exact coverage boxes."""
    header = item.header
    return {"header": _asset(header.path), "binary": _asset(header.path.with_suffix(".bin")),
            "shape": list(item.B.shape), "npts": header.npts, "spacing_mm": item.d.tolist(),
            "grid_origin_mm": header.grid_origin.tolist(),
            "origin_shift_mm": header.origin_shift.tolist(), "explicit_origin": header.explicit_origin,
            "bounds_mm": {"lo": item.lo.tolist(), "hi": item.hi.tolist()},
            "coverage_boxes_mm": [{"lo": lo.tolist(), "hi": hi.tolist()} for lo, hi in header.boxes],
            "symmetry": "extendY flip=By about Mu2e y=0" if header.extend_y else "none",
            "byte_order": item.byte_order}


def main(argv=None):
    """Write JSON audit to stdout; uncovered audit points return exit status 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="append", default=[], metavar="HEADER")
    parser.add_argument("--outer", action="append", default=[], metavar="HEADER")
    parser.add_argument("--point", action="append", nargs=3, type=float, default=[], metavar=("X", "Y", "Z"))
    args = parser.parse_args(argv)
    if not args.inner and not args.outer:
        parser.error("supply at least one --inner or --outer .header")
    try:
        inner = tuple(load(path) for path in args.inner)
        outer = tuple(load(path) for path in args.outer)
        bank = FieldMapBank(inner, outer)
        report = {"schema": "beamfastsim.tsfield.audit.v1", "backend": "numpy-cpu-reference",
                  "transport_validated": False, "inner": [audit_map(item) for item in inner],
                  "outer": [audit_map(item) for item in outer]}
        status = 0
        if args.point:
            points = _positions(args.point)
            values, covered = bank.field(points, return_coverage=True)
            report["points"] = {"positions_mm": points.tolist(), "covered": covered.tolist(),
                                "map_indices": bank.map_indices(points).tolist(),
                                "field_tesla": np.where(covered[:, None], values.astype(object), None).tolist()}
            status = 0 if np.all(covered) else 2
        print(json.dumps(report, indent=2, allow_nan=False))
        return status
    except (OSError, ValueError) as exc:
        print(f"tsfield: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
