"""Compact descriptive summaries of the one frozen saved CE joint JSON only.

No physics inputs, diagnostic rerun, calibration, uncertainty or fitted weights.
Run with REPORT --expected-report-sha256 SHA --out NEW_JSON. The output parent
must already exist, outside the input's parent and this repository's code tree.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

REPORT_SHA256 = "09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831"
REPORT_BYTES = 138519
INPUT_SCHEMA = "ce-joint-diagnostic-v1"
SCHEMA = "ce-joint-summary-v1"
INPUT_CAP = OUTPUT_CAP = 2 * 1024**2
SOURCE_CAP = 64 * 1024
PROXIES = ("nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1", "caloEnergy")
PB, MC, TIME = "measured_pbtime", "MC_truth_nprotons", "raw_stored_t_mid"
PB_EDGES = [-240, -230, -220, -210, -200, -190]
TIME_EDGES = list(range(0, 2001, 100))
TC_EDGES = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 9.5]
HIT_EDGES = [0, 1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
COUNT_FIELDS = ("N_all_retained", "A_events_with_tracks", "T_tracks")
MULT = "multiplicity_0_1_2_3_ge4"
HIST = "selected_track_raw_time_histogram"
# Conservative arithmetic bound: every fixed-grid reduction fits signed int64.
MAX_COUNT = (2**63 - 1) // 2**20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local_path(value):
    require(isinstance(value, (str, Path)) and bool(str(value)) and ":" not in str(value)
            and not str(value).startswith("//"), "local files only")
    # Inspect lexical components before abspath can erase '..' traversal.
    raw = Path(value)
    require(".." not in raw.parts, "parent traversal forbidden")
    path = Path(os.path.abspath(raw))
    require(not {p.lower() for p in path.parts} & {"pnfs", "net", "nfs", "afs", "cvmfs"}
            and not any(p.lower().endswith(".root") for p in path.parts), "ROOT/network paths forbidden")
    return path


def no_symlinks(path):
    for component in [*reversed(path.parents), path]:
        require(not stat.S_ISLNK(component.lstat().st_mode), "symlink forbidden")


def bounded_bytes(path, cap):
    no_symlinks(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= cap, "regular file byte cap exceeded")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require((info.st_dev, info.st_ino, info.st_size) ==
                (opened.st_dev, opened.st_ino, opened.st_size), "file changed at open")
        data = stream.read(cap + 1)
    require(len(data) == info.st_size and len(data) <= cap, "file changed/byte cap exceeded")
    return data


def outside(out, report, code):
    require(not os.path.lexists(out), "new output required")
    no_symlinks(out.parent)
    require(out.parent.is_dir(), "existing output directory required")
    for directory in (report.parent, code.parent, code.parents[2]):
        require(not out.is_relative_to(directory), "output under input/code directory forbidden")


def json_object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(type(result) is dict, "JSON object required")
    return result


def integer(value, name):
    require(type(value) is int and 0 <= value <= MAX_COUNT, "nonnegative bounded integer required: " + name)
    return value


def counts(value, shape, name):
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid shape: " + name) from exc
    require(array.shape == tuple(shape) and array.dtype.kind in "iu", "integer grid shape differs: " + name)
    require(np.all(array >= 0) and np.all(array <= MAX_COUNT), "count domain differs: " + name)
    return array.astype(np.int64, copy=False)


def validate_counts(n, a, t, mult, hist):
    require(np.all(a <= n) and np.all(t >= a) and np.all((a == 0) == (t == 0)), "N/A/T inconsistent")
    require(np.array_equal(hist.sum(axis=-1), t), "histogram does not match T")
    require(np.array_equal(mult.sum(axis=-1), n) and np.array_equal(n - mult[..., 0], a),
            "multiplicity does not match N/A")
    lower = (mult * np.arange(5)).sum(axis=-1)
    require(np.all(t >= lower) and np.all((mult[..., 4] > 0) | (t == lower)), "multiplicity does not match T")


def validate_axes(axes):
    require(type(axes) is dict and set(axes) == {*PROXIES, PB, MC, TIME}, "axes differ")
    for name in (*PROXIES, PB, MC, TIME):
        axis = axes[name]
        require(type(axis) is dict, "axis object required")
        edges = np.asarray(axis.get("edges"))
        expected = (TC_EDGES if name == PROXIES[0] else HIT_EDGES if name in PROXIES
                    else PB_EDGES if name == PB else TIME_EDGES if name == TIME else None)
        require(edges.ndim == 1 and edges.dtype.kind in "iuf" and np.isfinite(edges).all()
                and np.all(np.diff(edges) > 0), "finite increasing edges required")
        if expected is not None:
            require(np.array_equal(edges, expected), "frozen edges differ: " + name)
        else:
            require(len(edges) == 6 and np.all(edges >= 0), "five inherited MC strata required")
        require(type(axis.get("bins")) is int and axis["bins"] == len(edges) + 1
                and type(axis.get("underflow_index")) is int and axis["underflow_index"] == 0
                and type(axis.get("overflow_index")) is int and axis["overflow_index"] == len(edges)
                and axis.get("last_finite_upper_inclusive") is (name == MC)
                and axis.get("convention") == "left closed, right open; explicit underflow/overflow",
                "flow/convention differs: " + name)


def read_grid(grid, shape, totals):
    require(type(grid) is dict and grid.get("shape") == list(shape), "declared grid shape differs")
    arrays = tuple(counts(grid.get(key), shape, key) for key in COUNT_FIELDS)
    arrays += (counts(grid.get(MULT), (*shape, 5), MULT), counts(grid.get(HIST), (*shape, 22), HIST))
    validate_counts(*arrays)
    require(tuple(int(v.sum()) for v in arrays[:3]) == totals, "grid totals do not match report counts")
    return arrays


def reduce_grid(arrays, axis):
    return tuple(v.sum(axis=axis) for v in arrays)


def equal_grid(left, right):
    require(all(np.array_equal(a, b) for a, b in zip(left, right)), "cross-grid marginals differ")


def ratio(numerator, denominator, mask=True):
    n, d, mask = np.broadcast_arrays(numerator, denominator, mask)
    result = np.full(n.shape, None, dtype=object)
    np.divide(n, d, out=result, where=(d != 0) & mask, casting="unsafe")
    return result.tolist()


def support(arrays):
    n, a, t = arrays[:3]
    return (n >= 100) & (a >= 20) & (t >= 30)


def total_variation(hist, reference, supported):
    """Half L1 between normalized raw selected-track counts, including flows."""
    t, total = hist.sum(axis=-1), int(reference.sum())
    normalized = np.divide(hist, t[..., None], out=np.zeros(hist.shape, dtype=float), where=t[..., None] != 0)
    ref = reference / total if total else np.zeros(reference.shape, dtype=float)
    tv = 0.5 * np.abs(normalized - ref).sum(axis=-1)
    return np.where(supported & (t > 0) & (total > 0), tv.astype(object), None).tolist()


def describe(arrays, reference, *, histogram=True):
    n, a, t, mult, hist = arrays
    joint = support(arrays)
    result = dict(N=n.tolist(), A=a.tolist(), T=t.tolist(),
        A_over_N=ratio(a, n), T_over_N=ratio(t, n), T_over_A=ratio(t, a),
        support_N_ge100=(n >= 100).tolist(), support_A_ge20_and_T_ge30=((a >= 20) & (t >= 30)).tolist(),
        joint_support=joint.tolist(),
        all_event_bin_percent=ratio(100 * n, n.sum()), selected_event_bin_percent=ratio(100 * a, a.sum()),
        total_variation_vs_all_selected=total_variation(hist, reference, joint),
        unsupported=dict(N=int(n[~joint].sum()), A=int(a[~joint].sum()), T=int(t[~joint].sum())))
    if histogram:
        result[MULT] = mult.tolist()
        result[HIST] = hist.tolist()
    return result


def row_range(arrays):
    n, a, _ = arrays[:3]
    mask = support(arrays)
    rate = np.divide(a, n, out=np.zeros(n.shape, dtype=float), where=n != 0)
    low = np.where(mask, rate, np.inf).min(axis=1)
    high = np.where(mask, rate, -np.inf).max(axis=1)
    have = mask.any(axis=1)
    # Avoid evaluating inf-inf for empty rows; unsupported rows stay null.
    low, high = np.where(have, low, 0), np.where(have, high, 0)
    return dict(supported_cell_count=mask.sum(axis=1).tolist(),
        min_A_over_N=np.where(have, low.astype(object), None).tolist(),
        max_A_over_N=np.where(have, high.astype(object), None).tolist(),
        max_minus_min_A_over_N=np.where(have, (high - low).astype(object), None).tolist(),
        supported_N=np.where(mask, n, 0).sum(axis=1).tolist(),
        supported_A=np.where(mask, a, 0).sum(axis=1).tolist(),
        supported_T=np.where(mask, arrays[2], 0).sum(axis=1).tolist())


def summarize(report):
    """Pure reduction of decoded saved counts; file admission is exclusively run()."""
    require(report.get("schema") == INPUT_SCHEMA, "joint report schema differs")
    axes, grids, reported = report.get("axes"), report.get("grids"), report.get("counts")
    validate_axes(axes)
    require(type(reported) is dict, "report counts required")
    totals = tuple(integer(reported.get(k), k) for k in
                   ("retained_events", "selected_events_matched", "selected_tracks_matched"))
    require(totals[1] <= totals[0] and totals[2] >= totals[1], "report totals inconsistent")
    require(integer(reported.get("retained_events_without_window_tracks"), "zero-track events") == totals[0] - totals[1],
            "report zero-track count differs")
    names = (*PROXIES, MC)
    key3 = "__".join((PROXIES[0], MC, PB))
    require(type(grids) is dict and set(grids) == {*(name + "__" + PB for name in names), key3}, "grid allowlist differs")
    parsed = {name: read_grid(grids[name + "__" + PB], (axes[name]["bins"], axes[PB]["bins"]), totals)
              for name in names}
    pb = reduce_grid(parsed[PROXIES[0]], 0)
    for name in names:
        equal_grid(reduce_grid(parsed[name], 0), pb)
    three = read_grid(grids[key3], (axes[PROXIES[0]]["bins"], axes[MC]["bins"], axes[PB]["bins"]), totals)
    equal_grid(reduce_grid(three, 1), parsed[PROXIES[0]])
    equal_grid(reduce_grid(three, 0), parsed[MC])
    reference = pb[4].sum(axis=0)
    marginal = {name: describe(reduce_grid(parsed[name], 1), reference) for name in names}
    marginal[PB] = describe(pb, reference)
    ntc_truth = reduce_grid(three, 2)
    sources = report.get("sources")
    require(type(sources) is list and 0 < len(sources) <= 100, "source records required")
    source_totals = np.zeros(3, dtype=np.int64)
    seen, source_result = set(), []
    for source in sources:  # bounded source-file records, never events or tracks
        require(type(source) is dict, "source object required")
        index = integer(source.get("source_index"), "source_index")
        require(index not in seen, "duplicate source index")
        seen.add(index)
        n, a, t = (integer(source.get(k), k) for k in ("retained_events", "events_with_tracks", "selected_tracks"))
        mult = counts(source.get(MULT), (5,), MULT)
        # Sources have no time histogram. Validate their N/A/T and multiplicities.
        validate_counts(np.asarray(n), np.asarray(a), np.asarray(t), mult, np.asarray([t]))
        source_totals += (n, a, t)
        row = copy.deepcopy(source)
        row.update(A_over_N=ratio(a, n), T_over_N=ratio(t, n), T_over_A=ratio(t, a),
                   joint_support=bool(n >= 100 and a >= 20 and t >= 30))
        source_result.append(row)
    require(tuple(source_totals) == totals, "source totals differ")
    return dict(schema=SCHEMA, input_schema=INPUT_SCHEMA, axes=copy.deepcopy(axes),
        counts=copy.deepcopy(reported), totals=dict(N=totals[0], A=totals[1], T=totals[2]),
        all_selected_raw_time_histogram=reference.tolist(), marginals=marginal,
        conditional_on_measured_pbtime={name: describe(parsed[name], reference, histogram=False) for name in names},
        nProtonTCs_by_MC_truth_nprotons=dict(axes=[PROXIES[0], MC], **describe(ntc_truth, reference),
            within_nProtonTCs_MC_truth_rate_range=row_range(ntc_truth),
            row_all_event_bin_percent=ratio(100 * ntc_truth[0], ntc_truth[0].sum(axis=1, keepdims=True)),
            row_selected_event_bin_percent=ratio(100 * ntc_truth[1], ntc_truth[1].sum(axis=1, keepdims=True))),
        within_nProtonTCs_measured_pbtime_rate_range=row_range(parsed[PROXIES[0]]),
        sources=sorted(source_result, key=lambda s: s["source_index"]),
        definitions=dict(N="all retained events", A="events with at least one selected track", T="selected tracks",
            rates="A/N is retained-event selection rate; T/N and T/A are multiplicities, not probabilities. Nonzero-denominator raw rates remain visible even below support; zero denominators are null.",
            support="Joint support requires N>=100 AND A>=20 AND T>=30; thresholds are not precision guarantees.",
            percentages="100*N/sum(N) versus 100*A/sum(A) across marginal bins; row_* percentages condition on each nProtonTCs row.",
            total_variation="0.5*sum(abs(h/T - H/sum(H))), across all 22 raw stored t_mid bins including under/overflow. H is the overall all-selected-track marginal, not a proxy-specific or pbtime-specific reference. Only jointly supported cells; others null.",
            ranges="Unweighted max-minus-min A/N over jointly supported reduced cells only. Zero supported cells gives null; one gives zero, not evidence of invariance."),
        scope="Descriptive coarse-bin retained, trigger/output-conditioned sample only; not generated-CE efficiency or a measured intensity correction.",
        interpretation=[
            "Frozen bins, flows and raw times preserved; no refitted edges, rewrapping, rebasing, pseudocounts or sparse-cell zero filling.",
            "MC truth PBI is diagnostic only. Raw activity proxies are not calibrated beam intensity; signal activity can contribute.",
            "Measured pbtime is MC-smeared truth in the named recipe, not independent detector timing. Per-source rates/ranges and within-nProtonTCs pbtime rate ranges expose possible source/clock-phase confounding, not adjustment for it.",
            "Total variation describes selected-track shapes relative to the overall marginal. Coarse binning can conceal dependence; small TV is not an independence certificate, causal result, significance test or sufficiency result.",
            "Tracks are not independent trials. No uncertainty, physical exposure, independent-history count, fitted weights, correction or new F is inferred.",
            "Saved-report/source hashes bind bytes and this summary implementation, not independent historical execution, ancestry, runtime provenance or filesystem locality. No physics inputs or report-listed paths are opened."],
        calibration=None, weight_model=None, F=None, finite_library_uncertainty=None, uncertainty="not estimated")


def run(report, out, *, expected_report_sha256):
    require(type(expected_report_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_report_sha256)
            and expected_report_sha256 == REPORT_SHA256, "external pin must equal frozen joint report SHA256")
    path, out, code = map(local_path, (report, out, __file__))
    outside(out, path, code)
    source = bounded_bytes(code, SOURCE_CAP)
    data = bounded_bytes(path, INPUT_CAP)
    require(sha(data) == expected_report_sha256, "report SHA256 mismatch")
    require(len(data) == REPORT_BYTES, "frozen report byte size differs")
    result = summarize(json_object(data))
    result.update(report=dict(path=str(path), sha256=sha(data), bytes=len(data), external_expected_sha256=expected_report_sha256),
        source=dict(path=str(code), sha256=sha(source), bytes=len(source), snapshot=source.decode("utf-8")),
        software=dict(numpy=np.__version__), bounds=dict(input_bytes=INPUT_CAP, source_bytes=SOURCE_CAP, output_bytes=OUTPUT_CAP),
        publication="Input and actual source snapshot rechecked before exclusive creation. Byte caps, not whole-process RSS, malicious-decoder allocation, atomic publication or crash-durability guarantees.")
    payload = json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    require(len(payload) <= OUTPUT_CAP, "output byte cap exceeded")
    require(sha(bounded_bytes(path, INPUT_CAP)) == expected_report_sha256, "input changed before output")
    require(sha(bounded_bytes(code, SOURCE_CAP)) == sha(source), "source changed before output")
    outside(out, path, code)
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.report, args.out, expected_report_sha256=args.expected_report_sha256)


if __name__ == "__main__":
    main()
