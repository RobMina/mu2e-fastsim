"""Stock table lookup with supplied uniforms, not an analytic propagation model."""
import numpy as np


def build_cdf(contents):
    """Build float32 sequential depth-slice CDFs from regular TH2 bin contents.

    Axis 0 is depth; axis 1 is propagation time. No flow bins. The initial
    1e-6 floor and float32 accumulation match CaloPhotonPropagation::buildTable.
    This host preparation is deliberately sequential along the small time axis.
    """
    counts = np.asarray(contents, dtype=np.float32)
    if counts.ndim != 2 or 0 in counts.shape:
        raise ValueError("contents must be a nonempty depth-by-time matrix")
    if np.any(~np.isfinite(counts) | (counts < 0)):
        raise ValueError("contents must be finite and nonnegative")
    floor = np.full((len(counts), 1), np.float32(1e-6))
    cdf = np.cumsum(np.concatenate((floor, counts), axis=1), axis=1, dtype=np.float32)[:, 1:]
    if np.any(~np.isfinite(cdf[:, -1])):
        raise ValueError("float32 CDF accumulation overflow")
    return cdf / cdf[:, -1:]


def validate_table(cdf, times, dz):
    """Validate exported tables on the host before device lookup."""
    cdf = np.asarray(cdf, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    dz = np.float32(dz)
    if cdf.ndim != 2 or 0 in cdf.shape or times.shape != (cdf.shape[1],):
        raise ValueError("CDF must be nonempty depth-by-time with matching times")
    if not np.isfinite(dz) or dz <= 0:
        raise ValueError("dz must be finite and positive")
    if (np.any(~np.isfinite(cdf)) or np.any(~np.isfinite(times)) or
            np.any((cdf < 0) | (cdf > 1)) or np.any(np.diff(cdf, axis=1) < 0) or
            np.any(cdf[:, -1] != 1)):
        raise ValueError("invalid normalized monotone CDF or time centers")
    return cdf, times, dz


def propagation_time(depth, uniforms, cdf, times, dz):
    """Look up stock bin centers using float32 supplied draws and L-z depth.

    The depth-axis origin is deliberately not subtracted; no interpolation.
    Uniforms must be in [0,1]. Side='left' semantics retain the first equal bin.
    The truncated float32 depth/dz quotient must fit signed int32 before clamping;
    stock's out-of-range float-to-int conversion has no portable result.
    This consumes no random numbers and does not reproduce CLHEP by seed.
    """
    cdf, times, dz = validate_table(cdf, times, dz)
    depth, u = np.broadcast_arrays(np.asarray(depth, dtype=np.float32),
                                   np.asarray(uniforms, dtype=np.float32))
    if np.any(~np.isfinite(depth)) or np.any(~np.isfinite(u) | (u < 0) | (u > 1)):
        raise ValueError("finite depth and uniforms in [0,1] required")
    with np.errstate(over="ignore", invalid="ignore"):
        quotient = depth / dz
    truncated = np.trunc(quotient).astype(np.float64)
    if np.any(~np.isfinite(truncated) | (truncated < -(2**31)) | (truncated >= 2**31)):
        raise ValueError("stock depth quotient must be representable as signed int32")
    row = np.clip(truncated, 0, len(cdf) - 1).astype(np.int64)
    lo = np.zeros(row.shape, dtype=np.int64)
    hi = np.full(row.shape, cdf.shape[1], dtype=np.int64)
    for _ in range(cdf.shape[1].bit_length()):
        mid = (lo + hi) // 2
        less = cdf[row, np.minimum(mid, cdf.shape[1] - 1)] < u
        active = lo < hi
        lo = np.where(active & less, mid + 1, lo)
        hi = np.where(active & ~less, mid, hi)
    return times[np.minimum(lo, len(times) - 1)]
