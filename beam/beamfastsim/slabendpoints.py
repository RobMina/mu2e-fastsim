"""All-entry endpoint-energy identification for the opt-in slab diagnostic.

These intervals describe an algorithmic endpoint-energy convention under
nonnegative, monotone continuation of primary-energy debit. They are neither
physical-stop identification nor a distribution conditional on exiting the
slab. Positive resolved exits are exact. FLOOR/CAP permit [0, residual]; a
NUMERICAL lane permits [0, entry] without trusting its residual. RAWZERO is
exactly zero ONLY as the sampled-loss algorithmic diagnostic, not a certified
physical stop. Every lane remains in the denominator, including unresolved
lanes and RAWZERO lanes that reached a face.

The threshold map is E -> 0 for E < threshold, otherwise E; equality is not
thresholded. CDF envelopes are empirical identification envelopes, not sampling
confidence bands. No deterministic CSDA range is used as a stochastic bound.
"""
import numpy as np

from . import slabdiffusion as sd


SCOPE = __doc__


def _threshold(value):
    x = np.asarray(value, dtype=float)
    if x.ndim or isinstance(value, (bool, np.bool_)) or not np.isfinite(x) or x < 0:
        raise ValueError('threshold must be a nonnegative finite scalar')
    return float(x)


def _broadcast_positive(value, shape, name):
    try:
        x = np.broadcast_to(np.asarray(value, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f'{name} must broadcast to response energy shape') from exc
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError(f'{name} must be positive and finite')
    return x


def endpoint_intervals(response, entry_energy, threshold=.1):
    """Return four energy-bound arrays with exactly response.raw_energy.shape.

    Entry energy broadcasts to that shape; status must have exactly that shape.
    Numerical residuals may be NaN, infinite, negative, or inconsistent with
    entry: none narrows the conservative entry-energy interval. Other residuals
    must be finite, nonnegative, and no greater than entry. Active/unknown
    statuses and empty responses are rejected. Inputs are never changed.
    """
    threshold = _threshold(threshold)
    energy = np.asarray(response.raw_energy, dtype=float)
    status = np.asarray(response.status)
    if not energy.size or status.shape != energy.shape:
        raise ValueError('nonempty response energy and matching status shape required')
    if status.dtype.kind not in 'iu' or np.any(~np.isin(status, list(sd.STATUS_NAMES))):
        raise ValueError('unknown, active, or noninteger endpoint status')
    entry = _broadcast_positive(entry_energy, energy.shape, 'entry_energy')
    numerical = status == sd.UNRESOLVED_NUMERICAL
    invalid = ~np.isfinite(energy) | (energy < 0) | (energy > entry)
    if np.any(~numerical & invalid):
        raise ValueError('nonnumerical residual must be finite and in [0, entry_energy]')
    exited = np.isin(status, [sd.TRANSMITTED, sd.REFLECTED])
    rawzero = status == sd.RAWZERO
    if np.any(exited & (energy <= 0)) or np.any(rawzero & (energy != 0)):
        raise ValueError('resolved exit must be positive and RAWZERO residual must be zero')
    lower = np.where(exited, energy, 0.)
    upper = np.where(numerical, entry, energy)
    return dict(raw_lower=lower, raw_upper=upper,
                threshold_lower=np.where(lower < threshold, 0., lower),
                threshold_upper=np.where(upper < threshold, 0., upper))


def cdf_envelope(lower, upper, grid):
    """Return dict(lower=F_lower, upper=F_upper), with the shape of grid.

    For F(x)=P(E<=x), the lower CDF counts upper energies and the upper CDF
    counts lower energies. All lanes, flattened together, form the denominator.
    Sorting/searchsorted avoids per-sample loops or an N-by-grid allocation.
    Grid order is preserved; +/- infinity is allowed, NaN is not.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    grid = np.asarray(grid, float)
    if lower.shape != upper.shape or not lower.size:
        raise ValueError('nonempty, same-shape interval arrays required')
    if (np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
            or np.any(lower < 0) or np.any(lower > upper)):
        raise ValueError('finite nonnegative ordered energy intervals required')
    if np.any(np.isnan(grid)):
        raise ValueError('CDF grid must not contain NaN')
    return dict(lower=np.searchsorted(np.sort(upper.ravel()), grid, side='right') / lower.size,
                upper=np.searchsorted(np.sort(lower.ravel()), grid, side='right') / lower.size)


def _stats(values):
    x = np.asarray(values, float).ravel()
    finite = x[np.isfinite(x)]
    return dict(n=int(x.size), finite=int(finite.size),
                mean=float(np.sum(finite / finite.size)) if finite.size else None,
                min=float(finite.min()) if finite.size else None,
                max=float(finite.max()) if finite.size else None,
                quantiles=np.quantile(finite, [0., .01, .5, .99, 1.]).tolist() if finite.size else None)


def _identification(lower, upper):
    width = upper - lower
    identified = int(np.count_nonzero(lower == upper))
    return dict(fully_identified=identified == lower.size, identified_count=identified,
                unidentified_count=int(lower.size) - identified,
                interval_width_MeV=_stats(width),
                mean_bounds_MeV=[float(np.sum(lower / lower.size)), float(np.sum(upper / upper.size))])


def summarize_endpoints(response, entry_energy, thickness, tables, threshold=.1):
    """Return a JSON-safe all-entry report, without changing the response.

    Raw and thresholded mean bounds retain all entries, rather than selecting
    resolved exits. Floor normal distances are geometric distances along z,
    not remaining path lengths. Missing/nonfinite floor positions are counted
    explicitly. ``tables`` is accepted for the study API but is not evaluated:
    this report makes no deterministic-CSDA comparison and no range inference.
    """
    intervals = endpoint_intervals(response, entry_energy, threshold)
    status = np.asarray(response.status)
    thickness = _broadcast_positive(thickness, status.shape, 'thickness')
    position = np.asarray(response.position, float)
    if position.shape != status.shape + (3,):
        raise ValueError('position must have response energy shape plus (3,)')
    try:
        face = np.asarray(response.ledger['face_reached'])
    except (KeyError, TypeError) as exc:
        raise ValueError('face_reached ledger is required') from exc
    if face.shape != status.shape or np.any(~np.isin(face, [0, 1])):
        raise ValueError('face_reached must have response energy shape and boolean values')
    face = face.astype(bool)
    census = {name: dict(false=int(np.count_nonzero((status == code) & ~face)),
                         true=int(np.count_nonzero((status == code) & face)))
              for code, name in sd.STATUS_NAMES.items()}
    floor = status == sd.UNRESOLVED_FLOOR
    z, depth = position[..., 2][floor], thickness[floor]
    valid_position = np.isfinite(z)
    if np.any(valid_position & ((z < 0) | (z > depth))):
        raise ValueError('finite floor position must lie between slab faces')
    z = np.where(valid_position, z, np.nan)
    rawzero = status == sd.RAWZERO
    return dict(n=int(status.size), denominator=int(status.size), threshold_MeV=float(threshold),
                scope=SCOPE, raw=_identification(intervals['raw_lower'], intervals['raw_upper']),
                thresholded=_identification(intervals['threshold_lower'], intervals['threshold_upper']),
                status_face_reached_counts=census,
                rawzero=dict(count=int(rawzero.sum()), face_reached_count=int(np.count_nonzero(rawzero & face)),
                             scope='Sampled-loss algorithmic zero only; not certified physical stopping'),
                floor=dict(n=int(floor.sum()), residual_energy_MeV=_stats(np.asarray(response.raw_energy)[floor]),
                           position_unavailable_count=int(np.count_nonzero(~valid_position)),
                           normal_distance_to_front_mm=_stats(z),
                           normal_distance_to_back_mm=_stats(depth - z),
                           normal_distance_to_nearest_face_mm=_stats(np.minimum(z, depth - z))),
                csda_comparison=None)
