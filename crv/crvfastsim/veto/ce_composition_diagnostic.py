"""Bounded saved-input source/stored-PBT composition diagnostic, never a correction.

Four explicit report paths are required. Publish exclusively to --out NEW_JSON
(existing distinct parent) or --outdir NEW_DIRECTORY (report.json). No ROOT,
report-listed joint/summary paths, coincs, old consumer, fit or uncertainty run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import ce_joint_diagnostic as J

SCHEMA = "ce-composition-diagnostic-v1"
CONTRACT_SHA256 = "ca69360f65abea85e12381601c299850636903fb57358804dd610711667655a1"
HELPER_SHA256 = "1d9c4dafc450df68c72d365c82fd46116090a807838a0b07fdb055a410eab4b4"
SELECTION_SHA256 = "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
ROWS_REPORT_SHA256 = "af675694a4a997295d42d84a98bc8fb0aedac160aa83c2cc69002e3cb4e2b2e8"
ROWS_SHA256 = "20f012be65a9487226923f0388a72f06eca8ee02c95bcdcadc822b8b68f45f3a"
SELECTED_SHA256 = "7f60b88899f35bea7cf0a438aa18250ff0021ad229bc3c000bfd180fa386a2ba"
LEDGER_SHA256 = "b9f94c948099cdeddb1e27634bb4f2f232619aa11d854ae5a96d4d5339e955cf"
JOINT_SHA256 = "09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831"
SUMMARY_SHA256 = "63fc77886c3febb00ba6120ee9dfd6140bb3fedf12260312544f3bb574b68e0a"
MC_EDGES = [2412587, 11844973, 15682329, 19931318, 26354159, 93225507]
REPORT_CAP, SOURCE_CAP, OUTPUT_CAP = 8 * 1024**2, 64 * 1024, 2 * 1024**2
PB, MC, TC = "measured_pbtime", "MC_truth_nprotons", "nProtonTCs"
NAT = ("N_all_retained", "A_events_with_tracks", "T_tracks")
MULT, HIST = "multiplicity_0_1_2_3_ge4", "selected_track_raw_time_histogram"
RAW = (*NAT, MULT, HIST)
require = J.require


def local_path(value):
    raw = Path(value)
    require(not {p.lower() for p in raw.parts} & {"pnfs", "net", "nfs", "afs", "cvmfs"}
            and not any(p.lower().endswith(".root") for p in raw.parts), "ROOT/network paths forbidden")
    return J.local_path(value)


def raw_grid(grid, shape):
    require(grid.get("shape") == list(shape), "declared grid shape differs")
    arrays = []
    for name, extra in zip(RAW, ((), (), (), (5,), (22,))):
        v = np.asarray(grid.get(name))
        require(v.shape == (*shape, *extra) and v.dtype.kind in "iu"
                and np.all(v >= 0) and np.all(v <= J.MAX_TRACKS), "invalid raw grid: " + name)
        arrays.append(v.astype(np.int64, copy=False))
    n, a, t, mult, hist = arrays
    require(np.all(a <= n) and np.all(t >= a) and np.all((a == 0) == (t == 0)), "N/A/T conservation")
    require(np.array_equal(hist.sum(axis=-1), t), "time conservation")
    require(np.array_equal(mult.sum(axis=-1), n) and np.array_equal(n - mult[..., 0], a),
            "multiplicity conservation")
    lower = (mult * np.arange(5)).sum(axis=-1)
    require(np.all(t >= lower) and np.all(t <= lower + 60 * mult[..., 4]), "track multiplicity conservation")
    return tuple(arrays)


def reduced(arrays, axes):
    return tuple(v.sum(axis=axes) for v in arrays)


def equal(left, right, label):
    require(len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right)),
            "inherited marginal differs: " + label)


def pack(arrays):
    return dict(shape=list(arrays[0].shape), **{k: v.tolist() for k, v in zip(RAW, arrays)})


def track_binding(rows, track_event, positions):
    binding = np.empty(len(track_event), dtype=[(k, "<i8") for k in (*J.IDS, "position")])
    for name in J.IDS:
        binding[name] = rows[name][track_event]
    binding["position"] = positions
    binding.sort(order=[*J.IDS, "position"])
    return dict(sha256=J.sha(binding.tobytes()), count=len(binding),
        encoding="C bytes of lexicographically sorted run,subrun,event,original_list_position; four little-endian signed int64")


def join_accounting(original, rows, ledger, selected):
    require(J.edges_from_report(original) == MC_EDGES, "frozen MC edges differ")
    keys, lkeys = J.identities(rows), J.identities(ledger)
    order = J.lookup(keys, lkeys)
    require(len(keys) == len(lkeys) and np.all(order >= 0), "proxy/ledger keyset anti-join")
    ledger = {k: v[order] for k, v in ledger.items()}
    for name in ("measured_nprotons", "measured_pbtime"):
        J.exact_measurement_equal(rows[name], ledger[name], name)
    join = J.lookup(selected["keys"], keys)
    matched, active = join >= 0, selected["multiplicity"] > 0
    mult = np.zeros(len(keys), dtype=np.int64)
    mult[join[matched]] = selected["multiplicity"][matched]
    tm = matched[selected["events"]]
    te = join[selected["events"][tm]]
    times = selected["times"][tm]
    counts = dict(selected["counts"], retained_events=len(keys),
        selected_rows_matched_including_empty=int(matched.sum()),
        selected_rows_unmatched_including_empty=int((~matched).sum()),
        full_selected_tracks_matched_before_window=int(selected["original_multiplicity"][matched].sum()),
        full_selected_tracks_unmatched_before_window=int(selected["original_multiplicity"][~matched].sum()),
        selected_events_matched=int(np.count_nonzero(matched & active)), selected_tracks_matched=int(tm.sum()),
        selected_events_unmatched=int(np.count_nonzero(~matched & active)), selected_tracks_unmatched=int((~tm).sum()),
        retained_events_without_selected_row=int(len(keys) - matched.sum()),
        retained_events_without_window_tracks=int(np.count_nonzero(mult == 0)), proxy_not_in_ledger=0, ledger_not_in_proxy=0)
    window, diag = original["momentum_window"], original["diagnostic"]
    require(window.get("pmin") == 85 and window.get("pmax") == 200 and window.get("strict") is True,
            "original strict momentum window differs")
    for observed, inherited in (("momentum_window_events", "selected_events"), ("momentum_window_tracks", "selected_tracks")):
        require(counts[observed] == window[inherited], "original global window counts differ")
    for observed, inherited in (("retained_events", "ntuple_events"), ("selected_events_matched", "selected_events_matched"),
            ("selected_tracks_matched", "selected_tracks_matched"), ("selected_events_unmatched", "selected_events_not_in_supplied_ntuples")):
        require(counts[observed] == diag[inherited], "original matched counts differ")
    require(counts["selected_tracks_unmatched"] == window["selected_tracks"] - diag["selected_tracks_matched"],
            "original unmatched tracks differ")
    mc = J.bins(ledger["nprotons"], MC_EDGES, True)
    for i, inherited in enumerate(diag["pbi_quantile_strata"], 1):
        mask = mc == i
        require((int(mask.sum()), int(np.count_nonzero(mult[mask])), int(mult[mask].sum())) ==
                tuple(inherited[k] for k in ("ntuple_events", "selected_events", "selected_tracks")),
                "original MC stratum counts differ")
    binding = track_binding(rows, te, selected["positions"][tm])
    return counts, mult, te, J.bins(times, J.TIME_EDGES), mc, binding


def source_bindings(original, extracted, rows, mult):
    sources, files = original.get("ntuples"), extracted.get("files")
    require(type(sources) is list and 0 < len(sources) <= 100 and type(files) is list
            and len(files) == len(sources), "source file records required")
    originals = {r["path"]: r for r in sources}
    require(len(originals) == len(sources), "duplicate original source paths")
    indices, paths, summaries = set(), set(), []
    source = J.integer_array(rows["source_index"], "source_index")
    for record in sorted(files, key=lambda r: r["index"]):
        i, path = record["index"], record["path"]
        require(type(i) is int and 0 <= i <= 2**31 - 1 and i not in indices
                and path in originals and path not in paths, "source index/path binding differs")
        indices.add(i)
        paths.add(path)
        old = originals[path]
        count = old["trees"]["EventNtuple/ntuple"]["entries"]
        require(record["entries"] == count and record["uuid"] == old["uuid"], "source identity differs")
        mask = source == i
        require(int(mask.sum()) == count, "source retained entry count differs")
        clock = rows[PB][mask]
        summaries.append(dict(source_index=i, path=path, uuid=record["uuid"], retained_events=int(mask.sum()),
            events_with_tracks=int(np.count_nonzero(mult[mask])), selected_tracks=int(mult[mask].sum()),
            multiplicity_0_1_2_3_ge4=np.bincount(np.minimum(mult[mask], 4), minlength=5).tolist(),
            measured_pbtime_min=float(clock.min()) if len(clock) else None,
            measured_pbtime_max=float(clock.max()) if len(clock) else None))
    require(np.isin(source, list(indices)).all(), "unbound source index")
    axis = sorted(indices)
    return axis, np.searchsorted(axis, source), summaries


def validate_axes(axes):
    expected = {name: J.axis(J.TC_EDGES if name == TC else J.HIT_EDGES) for name in J.PROXIES}
    expected.update({PB: J.axis(J.PB_EDGES), MC: J.axis(MC_EDGES, True), "raw_stored_t_mid": J.axis(J.TIME_EDGES)})
    require(axes == expected, "frozen axes differ")


def reconcile(full, counts, sources, binding, joint, summary):
    require(joint.get("schema") == "ce-joint-diagnostic-v1" and summary.get("schema") == "ce-joint-summary-v1",
            "completed report schema differs")
    require(summary.get("input_schema") == "ce-joint-diagnostic-v1", "summary input schema differs")
    require(summary.get("report", {}).get("sha256") == JOINT_SHA256, "summary joint binding differs")
    require(joint.get("counts") == counts and summary.get("counts") == counts, "completed counts differ")
    require(joint.get("matched_track_identity") == binding, "matched tuple/list-position digest differs")
    require(joint.get("sources") == sources, "completed source bindings differ")
    validate_axes(joint.get("axes"))
    validate_axes(summary.get("axes"))
    grids = joint.get("grids")
    key3 = "__".join((TC, MC, PB))
    require(type(grids) is dict and set(grids) == {*(p + "__" + PB for p in (*J.PROXIES, MC)), key3},
            "completed grid allowlist differs")
    three = reduced(full, 0)
    equal(three, raw_grid(grids[key3], (8, 7, 7)), "TC/MC/PBT")
    tc, mc, pb = reduced(three, 1), reduced(three, 0), reduced(three, (0, 1))
    for name in (*J.PROXIES, MC):
        parsed = raw_grid(grids[name + "__" + PB], (8 if name == TC else 7 if name == MC else 11, 7))
        equal(reduced(parsed, 0), pb, name + " PBT/time/multiplicity")
        if name in (TC, MC):
            equal(parsed, tc if name == TC else mc, name)
    by_source = reduced(full, (1, 2, 3))
    for i, source in enumerate(sources):
        equal(tuple(v[i] for v in by_source[:4]),
              tuple(np.asarray(source[k]) for k in ("retained_events", "events_with_tracks", "selected_tracks", MULT)), "source")
    totals = tuple(int(v.sum()) for v in full[:3])
    require(summary.get("totals") == dict(zip(("N", "A", "T"), totals)), "summary totals differ")
    equal((pb[4].sum(axis=0),), (np.asarray(summary.get("all_selected_raw_time_histogram")),), "summary time")
    saved = summary.get("nProtonTCs_by_MC_truth_nprotons", {})
    endpoint = reduced(three, 2)
    summary_arrays = dict(shape=[8, 7], **{k: saved.get(alias) for k, alias in zip(RAW, ("N", "A", "T", MULT, HIST))})
    equal(endpoint, raw_grid(summary_arrays, (8, 7)), "summary TC/MC raw counts")
    require(saved.get("A_over_N") == J.divide(endpoint[1], endpoint[0]), "summary endpoint raw rates differ")
    return {str(c): contrast(endpoint[0][c + 1, [1, 5]], endpoint[1][c + 1, [1, 5]]) for c in (0, 1)}


def fraction(n, d):
    return float(n / d) if d else None


def contrast(n, a):
    return 100.0 * (float(a[0] / n[0]) - float(a[1] / n[1])) if np.all(np.asarray(n) > 0) else None


def describe_endpoint(values):
    n, a, t, mult, hist = values
    return dict(N=int(n), A=int(a), T=int(t), A_over_N=fraction(a, n), T_over_N=fraction(t, n), T_over_A=fraction(t, a),
        support_N_ge100=bool(n >= 100), support_A_ge20_and_T_ge30=bool(a >= 20 and t >= 30),
        **{MULT: mult.tolist(), HIST: hist.tolist()})


def endpoint_contrasts(arrays, source_axis):
    """Fixed-cell reductions only; B uses N alone, q is count-specific bookkeeping."""
    result = {}
    for c in (0, 1):
        ep = tuple(np.take(v[:, c + 1], [1, 5], axis=1) for v in arrays)
        n, a, t = ep[:3]
        common = (n[:, 0] >= 100) & (n[:, 1] >= 100)
        pooled = n.sum(axis=1)
        mass = int(pooled[common].sum())
        q = np.where(common, pooled / mass, 0.0) if mass else np.zeros(pooled.shape)
        supported = tuple(np.where(common[:, None, :, *([None] * (v.ndim - 3))], v, 0) for v in ep)
        all_totals, common_totals = reduced(ep, (0, 2)), reduced(supported, (0, 2))
        rates = np.divide(a, n, out=np.zeros(a.shape, dtype=float), where=n != 0)
        cell_delta = 100 * (rates[:, 0] - rates[:, 1])
        endpoints = {}
        for k, label in enumerate(("low", "high")):
            total = tuple(v[k] for v in all_totals)
            kept = tuple(v[k] for v in common_totals)
            excluded = tuple(x - y for x, y in zip(total, kept))
            endpoints[label] = dict(total=describe_endpoint(total), supported=describe_endpoint(kept),
                excluded=describe_endpoint(excluded),
                supported_fractions={name: fraction(v, d) for name, v, d in zip(("N", "A", "T"), kept[:3], total[:3])},
                excluded_fractions={name: fraction(v, d) for name, v, d in zip(("N", "A", "T"), excluded[:3], total[:3])})
        cells, per_source = [], []
        for s, source in enumerate(source_axis):
            for b in range(7):
                reasons = []
                for k, label in enumerate(("low", "high")):
                    if n[s, k, b] < 100:
                        reasons.append(label + ("_N_zero" if n[s, k, b] == 0 else "_N_lt100"))
                cells.append(dict(source_index=source, stored_PBT_bin=b, common_support=bool(common[s, b]),
                    unsupported_reasons=reasons, q=float(q[s, b]) if mass else None,
                    low=describe_endpoint(tuple(v[s, 0, b] for v in ep)),
                    high=describe_endpoint(tuple(v[s, 1, b] for v in ep)),
                    D_cell_pp=contrast(n[s, :, b], a[s, :, b]),
                    standardized_contribution_pp=float(q[s, b] * cell_delta[s, b]) if mass else None))
            sn, sa = n[s].sum(axis=1), a[s].sum(axis=1)
            bn, ba = supported[0][s].sum(axis=1), supported[1][s].sum(axis=1)
            weight = float(q[s].sum())
            contribution = float(np.sum(q[s] * cell_delta[s])) if mass else None
            per_source.append(dict(source_index=source, D_all_pp=contrast(sn, sa), D_common_pp=contrast(bn, ba),
                common_block_count=int(common[s].sum()), common_weight=weight if mass else None,
                standardized_contribution_pp=contribution,
                D_standardized_within_source_pp=fraction(contribution, weight) if mass else None,
                low=dict(total=describe_endpoint(tuple(v[s, 0].sum(axis=0) for v in ep)),
                         supported=describe_endpoint(tuple(v[s, 0].sum(axis=0) for v in supported))),
                high=dict(total=describe_endpoint(tuple(v[s, 1].sum(axis=0) for v in ep)),
                          supported=describe_endpoint(tuple(v[s, 1].sum(axis=0) for v in supported)))))
        result[str(c)] = dict(raw_count=c, endpoint_MC_bins=[1, 5], common_block_count=int(common.sum()),
            common_pooled_N=mass, D_all_pp=contrast(all_totals[0], all_totals[1]),
            D_common_pp=contrast(common_totals[0], common_totals[1]),
            D_standardized_pp=float(np.sum(q * cell_delta)) if mass else None,
            endpoints=endpoints, cells=cells, per_source=per_source)
    return result


def compose(original, extracted, rows, ledger, selected, joint, summary):
    counts, mult, te, time, mc, binding = join_accounting(original, rows, ledger, selected)
    source_axis, source, sources = source_bindings(original, extracted, rows, mult)
    tc, pb = J.bins(rows[TC], J.TC_EDGES), J.bins(rows[PB], J.PB_EDGES)
    require(np.array_equal(tc == 1, rows[TC] == 0) and np.array_equal(tc == 2, rows[TC] == 1),
            "exact raw count bins differ")
    shape = (len(source_axis), 8, 7, 7)
    full = raw_grid(J.grid([source, tc, mc, pb], shape, mult, te, time), shape)
    inherited = reconcile(full, counts, sources, binding, joint, summary)
    endpoints = endpoint_contrasts(full, source_axis)
    require(all(endpoints[c]["D_all_pp"] == inherited[c] for c in ("0", "1")), "summary D_all differs")
    return dict(schema=SCHEMA, label="source/stored-PBT composition-standardized retained association",
        counts=counts, matched_track_identity=binding, sources=sources,
        axes=dict(source_index=source_axis, nProtonTCs=J.axis(J.TC_EDGES), MC_truth_nprotons=J.axis(MC_EDGES, True),
                  measured_pbtime=J.axis(J.PB_EDGES), raw_stored_t_mid=J.axis(J.TIME_EDGES)),
        full_grid=dict(axis_order=["source_index", TC, MC, PB], **pack(full)), contrasts=endpoints,
        reconciliation=dict(completed_joint_counts_identity_sources_and_raw_marginals=True,
            completed_summary_raw_counts_and_rates=True, completed_summary_D_all_pp=inherited),
        momentum_window=dict(pmin=85, pmax=200, strict=True, track_identity="full tuple plus frozen list position; no deduplication"),
        definitions=dict(N="all retained events", A="events with at least one selected track", T="selected tracks",
            common_support="N_low>=100 AND N_high>=100 only, separately for counts 0 and 1",
            shape_support="A>=20 AND T>=30, reported separately; never selects B or q",
            q="(N_low+N_high)/sum_B(N_low+N_high); each count has its own retained endpoint mixture, not correction weights",
            D_all_pp="100*(sum_all A_low/sum_all N_low - sum_all A_high/sum_all N_high)",
            D_common_pp="100*(sum_B A_low/sum_B N_low - sum_B A_high/sum_B N_high); endpoints retain their own composition",
            D_standardized_pp="100*sum_B q*(A_low/N_low-A_high/N_high)",
            nulls="zero-denominator fractions and no-common-support contrasts are null; unsupported cells retained",
            accounting="full grid includes other counts, middle MC bins, both flows, every source and empty cells"),
        interpretation=[
            "Exploratory development evidence: endpoint counts 0/1 chosen after observed contrasts; no holdout or independent validation.",
            "Retained output denominator remains trigger-conditioned; not generated-CE efficiency or physical exposure.",
            "MC PBI is truth diagnostic only. Stored PBT is an MC truth-smeared surrogate, not independent detector timing.",
            "Source is not ancestry. Coarse bins and flows do not remove within-bin composition differences.",
            "D_all to D_common describes support restriction; D_common to D_standardized describes composition change.",
            "q is descriptive retained-mixture bookkeeping, not deployable correction weights; count-specific targets are different.",
            "Selected-time histograms are selection-conditional; unselected events have no selected-track time probability.",
            "No significance, causal, sufficiency, factorization, lack-of-signal-contamination or transportability inference.",
            "Neither attenuation nor persistence validates calibration. Support thresholds certify neither precision nor independence.",
            "No fit, bootstrap, inferential interval, uncertainty estimate or new F. Stock C4 and historical F are unchanged."],
        calibration=None, weight_model=None, F=None, finite_library_uncertainty=None, total_uncertainty=None,
        uncertainty="not estimated", contract_sha256=CONTRACT_SHA256)


def run(selection_report, rows_report, joint_report, summary_report, out=None, *, outdir=None):
    require((out is None) != (outdir is None), "specify exactly one of out and outdir")
    sp, rp, jp, up = map(local_path, (selection_report, rows_report, joint_report, summary_report))
    code, helper = map(local_path, (__file__, J.__file__))
    destination = local_path(out if out is not None else outdir)
    paths = [sp, rp, jp, up, code, helper, code.parents[2] / "__code_directory__"]
    J.outside(destination, paths)
    evidence = J.Evidence()
    source = evidence.read(code, SOURCE_CAP)
    helper_source = evidence.read(helper, SOURCE_CAP, HELPER_SHA256)
    original = J.json_object(evidence.read(sp, REPORT_CAP, SELECTION_SHA256))
    extracted = J.json_object(evidence.read(rp, REPORT_CAP, ROWS_REPORT_SHA256))
    joint = J.json_object(evidence.read(jp, REPORT_CAP, JOINT_SHA256))
    summary = J.json_object(evidence.read(up, REPORT_CAP, SUMMARY_SHA256))
    require(extracted.get("schema") == "ce-proxy-rows-v1", "rows report schema differs")
    require(extracted.get("selection_report", {}).get("sha256") == SELECTION_SHA256, "rows selection binding differs")
    require(extracted.get("availability_report", {}).get("sha256") == J.AVAILABILITY_SHA256, "rows availability binding differs")
    require(original["selection"]["sha256"] == SELECTED_SHA256
            and original["exposure_ledger"]["sha256"] == LEDGER_SHA256
            and extracted["rows"]["sha256"] == ROWS_SHA256, "hardbound artifact hash differs")
    selected_path = local_path(original["selection"]["path"])
    ledger_path = local_path(sp.parent / original["exposure_ledger"]["path"])
    row_path = local_path(extracted["rows"]["path"])
    require(row_path == rp.parent / "rows.npz", "rows path must be report parent/rows.npz")
    paths += [selected_path, ledger_path, row_path]
    J.outside(destination, paths)
    rows = J.load_rows(evidence.read(row_path, J.ROWS_CAP, ROWS_SHA256), extracted["rows"])
    ledger = J.read_ledger(evidence.read(ledger_path, J.LEDGER_CAP, LEDGER_SHA256))
    selected = J.read_selected(evidence.read(selected_path, J.SELECTED_CAP, SELECTED_SHA256))
    result = compose(original, extracted, rows, ledger, selected, joint, summary)
    result.update(source=dict(path=str(code), sha256=J.sha(source), bytes=len(source), snapshot=source.decode("utf-8")),
        helper_source=dict(path=str(helper), sha256=J.sha(helper_source), bytes=len(helper_source), snapshot=helper_source.decode("utf-8")),
        inputs=list(evidence.records.values()), software=dict(numpy=np.__version__, pyarrow=J.pa.__version__),
        input_column_allowlists=dict(selected=list(J.SELECTED_KEYS), ledger=list(J.LEDGER_KEYS), rows=list(J.ROW_KEYS)),
        bounds=dict(report_bytes=REPORT_CAP, source_bytes=SOURCE_CAP, output_bytes=OUTPUT_CAP,
            selected_bytes=J.SELECTED_CAP, ledger_bytes=J.LEDGER_CAP, npz_bytes=J.ROWS_CAP,
            npz_uncompressed_bytes=J.ROWS_CAP, footer_bytes=J.FOOTER_CAP, selected_rows=J.MAX_ROWS,
            selected_tracks=J.MAX_TRACKS, tracks_per_row=J.MAX_TRACKS_PER_ROW, ledger_rows=J.MAX_LEDGER_ROWS,
            decoded_selected_columns_bytes=J.DECODED_CAP),
        publication="Input/source hashes and snapshots rechecked before exclusive report-last publication. Byte caps are not whole-process RSS, malicious-decoder allocation, filesystem-locality, atomicity or crash-durability guarantees.")
    payload = json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    require(len(payload) <= OUTPUT_CAP, "output byte cap exceeded; no outcomes trimmed")
    evidence.recheck()
    J.outside(destination, paths)
    if outdir is not None:
        destination.mkdir()
        destination = destination / "report.json"
    J.outside(destination, paths)
    with destination.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection_report", "rows_report", "joint_report", "summary_report"):
        parser.add_argument(name)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--out")
    outputs.add_argument("--outdir")
    args = parser.parse_args(argv)
    run(args.selection_report, args.rows_report, args.joint_report, args.summary_report, args.out, outdir=args.outdir)


if __name__ == "__main__":
    main()
