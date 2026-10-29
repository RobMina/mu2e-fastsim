"""Bounded saved-pilot track-grid coverage, not full F or CE factorisation closure."""
from __future__ import annotations

from numbers import Integral, Real

import awkward as ak
import numpy as np
import pandas as pd

from . import deadtime
from .apply import coinc_mask, gate
from .card import VetoCard
from .frameaudit import track_histogram

KEYS = ["input_sha256", "run", "subrun", "event", "frame_id"]
FIELDS = {"time": "avgtime", "timeStart": "start", "timeEnd": "end",
          "PEs": "pe", "nLayers": "nLayers", "sectorType": "sectortype"}
CONTRACT = "historical-difference-array-edges-v1"


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _real(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be real and finite")
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be real and finite") from exc
    if not np.isfinite(out):
        raise ValueError(f"{label} must be real and finite")
    return out


def _array(value, label, integer=False):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in ("iu" if integer else "iuf") or not np.isfinite(a).all():
        raise ValueError(f"{label} must contain finite {'integer' if integer else 'real'} scalars")
    return a


def _card(config):
    if not isinstance(config, dict):
        raise ValueError("reference card must be a dictionary")
    config = dict(config)
    sha = config.pop("sha", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError("reference card hash is mandatory")
    for name in ("dt_lo", "dt_hi", "min_pe", "gate_start_lo", "gate_end_hi"):
        if name in config and not (name.startswith("gate_") and config[name] is None):
            _real(config[name], f"card {name}")
    if "min_layers" in config:
        _integer(config["min_layers"], "card min_layers")
    if config.get("sectors") is not None:
        if not isinstance(config["sectors"], (list, tuple)):
            raise ValueError("card sectors must be an integer sequence")
        for sector in config["sectors"]:
            _integer(sector, "card sector")
    thresholds = config.get("min_pe_by_sector", {})
    if not isinstance(thresholds, dict):
        raise ValueError("card sector thresholds must be a dictionary")
    for sector, threshold in thresholds.items():
        if isinstance(sector, str) and sector.isascii() and sector.isdecimal():
            sector = int(sector)
        _integer(sector, "card threshold sector")
        if _real(threshold, "card sector threshold") < 0:
            raise ValueError("negative card sector threshold")
    try:
        card = VetoCard.from_dict(config)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid reference card schema") from exc
    if card.dt_lo >= card.dt_hi or card.min_pe < 0:
        raise ValueError("invalid card window or PE threshold")
    if card.gate_start_lo is not None and card.gate_end_hi is not None and card.gate_start_lo >= card.gate_end_hi:
        raise ValueError("invalid card gate bounds")
    if card.sha() != sha:
        raise ValueError("reference card hash mismatch")
    return card


def frozen_grid(reference: dict, selection: ak.Array, *, max_tracks=3000000, max_bins=20000):
    """Validate a frozen reference and selection before the unchanged histogram allocation."""
    max_tracks = _integer(max_tracks, "max_tracks", 1)
    max_bins = _integer(max_bins, "max_bins", 1)
    if not isinstance(reference, dict):
        raise ValueError("reference must be a dictionary")
    card = _card(reference.get("card"))
    selected = _integer(reference.get("selected"), "reference selected", 1)
    pmin, pmax = reference.get("pmin"), reference.get("pmax")
    if pmin is not None:
        pmin = _real(pmin, "pmin")
    if pmax is not None:
        pmax = _real(pmax, "pmax")
    if pmin is not None and pmax is not None and pmin >= pmax:
        raise ValueError("pmin must be less than pmax")
    if not isinstance(reference.get("F"), dict):
        raise ValueError("reference lacks F.grid_ns")
    step = _real(reference["F"].get("grid_ns"), "grid step")
    if step <= 0:
        raise ValueError("grid step must be positive")
    if not isinstance(selection, ak.Array) or not {"t_mid", "p_mid"}.issubset(selection.fields):
        raise ValueError("selection requires jagged t_mid and p_mid")
    counts, values = {}, {}
    for name in ("t_mid", "p_mid"):
        column = selection[name]
        if column.layout.purelist_depth != 2:
            raise ValueError(f"{name} must be one jagged list per event")
        try:
            counts[name] = _array(ak.to_numpy(ak.num(column, axis=1), allow_missing=False), name, True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing or malformed {name} lists") from exc
    if not np.array_equal(counts["t_mid"], counts["p_mid"]):
        raise ValueError("per-event t_mid/p_mid lengths differ")
    total = int(counts["t_mid"].sum())
    if total > max_tracks:
        raise ValueError("total tracks exceed max_tracks")
    for name in ("t_mid", "p_mid"):
        try:
            values[name] = _array(ak.to_numpy(ak.flatten(selection[name], axis=1), allow_missing=False), name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} requires finite real track values") from exc
    mask = np.ones(total, dtype=bool)
    if pmin is not None:
        mask &= values["p_mid"] > pmin
    if pmax is not None:
        mask &= values["p_mid"] < pmax
    if int(mask.sum()) != selected:
        raise ValueError("track selection count mismatch")
    times = values["t_mid"][mask].astype(float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lo, hi = np.floor(times.min()) - step, np.ceil(times.max()) + step
        length = (hi + step - lo) / step
    if not np.isfinite(length) or not np.isfinite([lo, hi, hi + step]).all() or lo + step <= lo:
        raise ValueError("unrepresentable grid")
    bins = int(np.ceil(length)) - 1
    if bins < 1 or bins > max_bins:
        raise ValueError("grid bins exceed max_bins or are empty")
    hist, edges = track_histogram(times, step)
    if len(hist) != bins:
        raise ValueError("historical histogram grid length changed")
    _grid(hist, edges)
    metadata = {"selected": selected, "total_tracks": total, "selection_events": len(selection),
                "grid_ns": step, "bins": bins, "card_sha": card.sha(), "contract": CONTRACT,
                "boundary_semantics": "ceil((time + dt - edges[0]) / step), clipped [lo,hi) integer cells; not continuous strict centre-exact",
                "scope": "Per-frame historical track-grid coverage only; not full F or CE factorisation closure."}
    return card, hist, edges, metadata


def _grid(hist, edges):
    with np.errstate(over="ignore", invalid="ignore"):
        hist = _array(hist, "hist").astype(float)
        edges = _array(edges, "edges").astype(float)
        spacing = np.diff(edges)
    if not len(hist) or len(edges) != len(hist) + 1 or np.any(hist < 0) or not np.isfinite(hist).all():
        raise ValueError("invalid histogram/edge schema")
    if not np.isclose(hist.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("histogram must be normalized")
    tolerance = np.finfo(float).eps * max(1.0, float(np.abs(edges).max())) * 4
    if (not np.isfinite(edges).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=tolerance)):
        raise ValueError("grid edges must increase uniformly")
    return hist, edges


def _identities(table, label):
    if not isinstance(table, pd.DataFrame) or not table.columns.is_unique or not set(KEYS).issubset(table.columns):
        raise ValueError(f"{label} lacks unique source-qualified key columns")
    sha = table["input_sha256"]
    try:
        valid_sha = sha.str.fullmatch(r"[0-9a-f]{64}", na=False).all()
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"invalid {label} input_sha256") from exc
    if sha.isna().any() or not valid_sha:
        raise ValueError(f"invalid {label} input_sha256")
    for name in KEYS[1:]:
        a = _array(table[name].to_numpy(), f"{label} {name}", True)
        if np.any(a < 0) or np.any(a > np.iinfo(np.int64).max):
            raise ValueError(f"invalid {label} {name}")
    for name in table.columns:
        if table[name].dtype.kind in "iufc":
            _array(table[name].to_numpy(), f"{label} {name}")


def _indices(times, card, edges):
    h, g0, bins = edges[1] - edges[0], edges[0], len(edges) - 1
    indices = []
    for delta in (card.dt_lo, card.dt_hi):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            index = np.ceil((times + delta - g0) / h)
        if not np.isfinite(index).all() or np.any(index < -(2.0**63)) or np.any(index >= 2.0**63):
            raise ValueError("coincidence window exceeds historical int64 grid indices")
        indices.append(np.clip(index.astype(np.int64), 0, bins))
    return indices


def _independent(coincs, card, hist, edges, passing, eligible):
    out = np.zeros(len(coincs))
    out[~passing] = 1.0
    active = np.flatnonzero(passing & (ak.to_numpy(ak.sum(eligible, axis=1)) > 0))
    times = coincs["time"][eligible][active]
    counts = ak.to_numpy(ak.num(times, axis=1))
    lo, hi = _indices(ak.to_numpy(ak.flatten(times)), card, edges)
    cells = np.arange(len(hist))
    occupied = (cells[None, :] >= lo[:, None]) & (cells[None, :] < hi[:, None])
    cov = np.zeros((len(active), len(hist)), dtype=bool)
    np.logical_or.at(cov, np.repeat(np.arange(len(active)), counts), occupied)
    out[active] = cov @ hist
    return out


def evaluate(frames: pd.DataFrame, clusters: pd.DataFrame, card, hist, edges, *, max_cells=80000) -> pd.DataFrame:
    """Preserve every frame and cross-check the unchanged helper using bounded integer-cell masks."""
    max_cells = _integer(max_cells, "max_cells", 1)
    if not isinstance(card, VetoCard):
        raise ValueError("card must be a VetoCard")
    _card(dict(card.__dict__, sha=card.sha()))
    hist, edges = _grid(hist, edges)
    _identities(frames, "frames")
    _identities(clusters, "clusters")
    if frames.duplicated(KEYS[:-1]).any() or frames.frame_id.duplicated().any():
        raise ValueError("duplicate frame identity")
    if not {"idx", *FIELDS.values()}.issubset(clusters.columns):
        raise ValueError("clusters lack scalar coincidence schema")
    for name in ("idx", "nLayers", "sectortype"):
        if np.any(_array(clusters[name].to_numpy(), name, True) < 0):
            raise ValueError(f"negative {name}")
    for name in ("avgtime", "start", "end", "pe"):
        _array(clusters[name].to_numpy(), name)
    if np.any(clusters.start > clusters.end) or np.any(clusters.pe < 0):
        raise ValueError("invalid cluster timing or PE")
    if clusters.duplicated(["frame_id", "idx"]).any():
        raise ValueError("duplicate cluster frame_id/idx")
    location = pd.MultiIndex.from_frame(frames[KEYS]).get_indexer(pd.MultiIndex.from_frame(clusters[KEYS]))
    if np.any(location < 0):
        raise ValueError("cluster key missing from frames")
    counts = np.bincount(location, minlength=len(frames))
    for name in ("n_coincidences", "clusters"):
        if name in frames:
            saved = _array(frames[name].to_numpy(), "saved " + name, True)
            if not np.array_equal(saved, counts):
                raise ValueError("inconsistent frame coincidence counts")
    bins = len(hist)
    if bins + 1 > max_cells or np.any(counts > (max_cells - bins - 1) // bins):
        raise ValueError("frame/grid/coincidence cells exceed max_cells")
    order = np.lexsort((clusters.idx.to_numpy(), location))
    ordered = clusters.iloc[order]
    _indices(ordered.avgtime.to_numpy(), card, edges)
    coincs = ak.zip({name: ak.unflatten(ordered[column].to_numpy(), counts) for name, column in FIELDS.items()})
    passing = gate(coincs, card)
    eligible = coinc_mask(coincs, card)
    coverage = np.zeros(len(frames))
    costs = (bins + 1) + counts * bins
    cumulative = np.concatenate(([0], np.cumsum(costs)))
    start = 0
    while start < len(frames):
        stop = int(np.searchsorted(cumulative, cumulative[start] + max_cells, side="right") - 1)
        stop = min(stop, len(frames))
        block = coincs[start:stop]
        actual = deadtime.frame_coverage(block, card, hist, edges, chunk=stop - start)
        independent = _independent(block, card, hist, edges, passing[start:stop], eligible[start:stop])
        if not np.array_equal(actual, independent):
            raise ValueError("historical grid coverage parity failed")
        coverage[start:stop] = actual
        start = stop
    out = frames.copy()
    out["n_coincidences"] = counts
    out["n_veto_eligible"] = ak.to_numpy(ak.sum(eligible, axis=1)).astype(np.int64)
    out["gate_pass"] = passing
    out["track_grid_coverage"] = coverage
    return out
