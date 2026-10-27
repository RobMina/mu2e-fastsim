"""Exact additive, time-independent shower feature moments; no occupancy or detector-response model."""

import warnings

import numpy as np
from scipy import sparse

__all__ = ["conditional_moments", "file_bootstrap", "bootstrap_probability_guard", "paired_covariance", "ProbabilityOverflowError"]


class ProbabilityOverflowError(ValueError):
    """An invalid bootstrap probability, with complete guard diagnostics in report."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"probability overflow in {report['invalid_replicas']}/{report['replicas']} bootstrap replicas "
            f"({report['invalid_cell_replica_pairs']} replica/cell pairs); no replicas discarded"
        )


def _array(value, name, ndim, *, nonnegative=False, integral=False):
    """Validate real finite numerical arrays without rounding or replacing values."""
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}-dimensional array")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or (nonnegative and np.any(result < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    if integral and np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain integer-valued counts; no rounding is performed")
    return result


def _indices(value, name, size):
    """Validate integer indices, allowing an untyped empty list."""
    result = np.asarray(value)
    if result.ndim != 1 or (result.size and result.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer index array")
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError(f"{name} contains unsupported indices outside [0, {size})")
    return result.astype(np.int64, copy=False)


def _support(cell, ngen):
    """Check the persisted denominator and nominal template support."""
    ngen = _array(ngen, "ngen", 1, nonnegative=True, integral=True)
    cell = _indices(cell, "template_cell", ngen.size)
    counts = np.bincount(cell, minlength=ngen.size)
    if np.any(counts > ngen):
        raise ValueError("template count exceeds persisted ngen in at least one cell")
    return cell, ngen


def _matrix(value, name, shape, *, nonnegative=False):
    """Validate a dense or sparse sufficient-statistic matrix."""
    if sparse.issparse(value):
        if value.ndim != 2 or value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        _array(value.data, name, 1, nonnegative=nonnegative)
        return value.astype(np.float64)
    result = _array(value, name, 2, nonnegative=nonnegative)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _dense(value):
    """Materialize only the small feature-space result when sparse."""
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def conditional_moments(features, cell, ngen, neval):
    """Return mean_per_cell, total_mean and sampling_cov for independent primary draws."""
    if not sparse.issparse(features) or features.format != "csr" or features.ndim != 2:
        raise ValueError("features must be a CSR matrix or array of shape (nTemplates, D)")
    _array(features.data, "features", 1, nonnegative=True)
    features = features.astype(np.float64)
    cell, ngen = _support(cell, ngen)
    neval = _array(neval, "neval", 1, nonnegative=True, integral=True)
    if cell.size != features.shape[0] or neval.shape != ngen.shape:
        raise ValueError("features, cell, ngen and neval have incompatible shapes")
    incidence = sparse.csr_matrix((np.ones(cell.size), (cell, np.arange(cell.size))), shape=(ngen.size, cell.size))
    # Each template has probability 1/ngen; remaining mass is a zero-feature outcome.
    sums = (incidence @ features).toarray()
    mean = np.divide(sums, ngen[:, None], out=np.zeros_like(sums), where=ngen[:, None] > 0)
    weights = np.divide(neval, ngen, out=np.zeros_like(neval), where=ngen > 0)
    second = (features.T @ features.multiply(weights[cell, None])).toarray()
    covariance = second - mean.T @ (neval[:, None] * mean)
    return {"mean_per_cell": mean, "total_mean": neval @ mean, "sampling_cov": (covariance + covariance.T) / 2}


def bootstrap_probability_guard(template_file, template_cell, ngen, file_weights, *, strict=True):
    """Check global equal-exposure file weights; raise with report on any overflow unless strict=False."""
    weights = _array(file_weights, "file_weights", 2, nonnegative=True, integral=True)
    replicas, nfiles = weights.shape
    if replicas < 1 or nfiles < 1 or np.any(weights.sum(axis=1) != nfiles):
        raise ValueError("file_weights must contain at least one replica, each with exactly F file draws")
    cell, ngen = _support(template_cell, ngen)
    files = _indices(template_file, "template_file", nfiles)
    if files.shape != cell.shape:
        raise ValueError("template_file and template_cell must have equal shapes")
    filecell = sparse.csr_matrix((np.ones(cell.size), (files, cell)), shape=(nfiles, ngen.size))
    max_counts = filecell.max(axis=0).toarray().ravel() if ngen.size else np.empty(0)
    # F * max_file(template_count_in_cell) / ngen bounds every possible resampled cell probability.
    bound = np.divide(nfiles * max_counts, ngen, out=np.zeros_like(ngen), where=ngen > 0)
    risky = np.flatnonzero(bound > 1)
    invalid_indices = np.empty(0, dtype=np.int64)
    invalid_pairs = 0
    invalid_cells = np.empty(0, dtype=np.int64)
    if risky.size:
        sampled_counts = np.asarray((filecell[:, risky].T @ weights.T).T)
        overflow = sampled_counts > ngen[risky]
        invalid_indices = np.flatnonzero(overflow.any(axis=1))
        invalid_cells = risky[np.any(overflow, axis=0)]
        invalid_pairs = int(np.count_nonzero(overflow))
    report = {
        "upper_bound_per_cell": bound,
        "certified_all_replicas": not bool(risky.size),
        "replicas": replicas,
        "invalid_replicas": int(invalid_indices.size),
        "invalid_cell_replica_pairs": invalid_pairs,
        "invalid_replica_indices": invalid_indices,
        "invalid_cell_indices": invalid_cells,
    }
    if strict and invalid_indices.size:
        raise ProbabilityOverflowError(report)
    return report


def file_bootstrap(contributions, template_file, template_cell, ngen, replicas=2000, seed=20261027):
    """Globally resample F equal-exposure files; return exact covariance and unfiltered sampled summaries."""
    contributions = _array(contributions, "contributions", 2, nonnegative=True)
    nfiles, dimension = contributions.shape
    if nfiles < 1:
        raise ValueError("contributions must contain at least one file, including zero-contribution files")
    if isinstance(replicas, (bool, np.bool_)) or not isinstance(replicas, (int, np.integer)) or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    weights = np.random.default_rng(seed).multinomial(nfiles, np.full(nfiles, 1 / nfiles), size=int(replicas))
    guard = bootstrap_probability_guard(template_file, template_cell, ngen, weights)
    # Contributions use the unchanged persisted ngen; one global weight multiplies every template in a file.
    centered = contributions - contributions.mean(axis=0)
    covariance = centered.T @ centered
    sampled = weights @ contributions
    levels = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    if dimension > nfiles - 1:
        warnings.warn(f"file bootstrap covariance rank is at most F-1={nfiles - 1}, below D={dimension}", RuntimeWarning, stacklevel=2)
    return {
        "nominal": contributions.sum(axis=0),
        "bootstrap_cov": covariance,
        "sampled_mean": sampled.mean(axis=0),
        "quantile_levels": levels,
        "quantiles": np.quantile(sampled, levels, axis=0),
        "replica_totals": sampled,
        "probability_guard": guard,
        "covariance_rank_upper_bound": min(dimension, nfiles - 1),
        "replicas": int(replicas),
        "seed": seed,
    }


def paired_covariance(mean_per_cell, hth, htr, rtr, counts, rsum, nEvents):
    """Return residual_total and fixed-N covariance of sum(H @ mean_per_cell + R), not sampling covariance."""
    # Inputs are H.T@H, H.T@R, R.T@R, sum(H), sum(R), and the full event census including zeros.
    # R may be -G_nonmu or fixedMuon-Gall; expected-versus-G4 comparisons do not add sampling_cov.
    mean = _array(mean_per_cell, "mean_per_cell", 2, nonnegative=True)
    cells, dimension = mean.shape
    hth = _matrix(hth, "hth", (cells, cells), nonnegative=True)
    htr = _matrix(htr, "htr", (cells, dimension))
    rtr = _matrix(rtr, "rtr", (dimension, dimension))
    counts = _array(counts, "counts", 1, nonnegative=True, integral=True)
    rsum = _array(rsum, "rsum", 1)
    if counts.shape != (cells,) or rsum.shape != (dimension,):
        raise ValueError("counts and rsum have incompatible shapes")
    if isinstance(nEvents, (bool, np.bool_)) or not isinstance(nEvents, (int, np.integer)) or nEvents < 2:
        raise ValueError("nEvents must be an integer >= 2, including zero-activity events")
    if sparse.issparse(hth):
        symmetric_hth = np.allclose((hth - hth.T).data, 0)
    else:
        symmetric_hth = np.allclose(hth, hth.T)
    rtr = _dense(rtr)
    if not symmetric_hth or not np.allclose(rtr, rtr.T):
        raise ValueError("hth and rtr must be symmetric Gram matrices")
    total = counts @ mean + rsum
    cross = _dense(htr.T @ mean)
    second = mean.T @ (hth @ mean) + cross + cross.T + rtr
    covariance = nEvents / (nEvents - 1) * (second - np.outer(total, total) / nEvents)
    return {"residual_total": total, "covariance": (covariance + covariance.T) / 2}
