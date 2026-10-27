"""Frozen-capture same-bar/end distinct-SiPM diagnostics on common measured support."""
import argparse
from pathlib import Path

import numpy as np

from . import joint_compare as J, joint_timing as T, validation as V

KEY_NAMES = ("body", "side", "along_bin", "path_bin", "pe_low_sipm_bin", "pe_high_sipm_bin", "multiplicity", "layer")
MODES = ("none", "band", "exact", "layer_band")
FIELDS = ("mean", "second_moment", "pe_low_mean", "pe_high_mean")


def _reduce(ids, values, size):
    n = np.bincount(ids, minlength=size)
    mean = T._mean(np.bincount(ids, weights=values, minlength=size), n)
    var = T._mean(np.bincount(ids, weights=(values-mean[ids])**2, minlength=size), n)
    return n, mean, var


def _sensor_stats(base, h, valid=None, pebin=None):
    if valid is None:
        valid = np.ones(len(base), bool)
    index = base[valid]*2 + h["sipm_id"][valid]//2
    if pebin is not None:
        index = index*5 + pebin[valid]
    keys, inv = np.unique(index, return_inverse=True)
    n, mean, var = _reduce(inv, h["time_residual"][valid], len(keys))
    pe_values = h["pe"][valid]
    pe = T._mean(np.bincount(inv, weights=pe_values, minlength=len(keys)), n) if pebin is not None else np.zeros(len(keys))
    return keys, dict(n=n, mean=mean, var=var, pe=pe)


def _cross(left_ids, left, right_ids, right):
    base, li, ri = np.intersect1d(left_ids, right_ids, assume_unique=True, return_indices=True)
    n = left["n"][li]*right["n"][ri]
    mean = left["mean"][li]-right["mean"][ri]
    second = left["var"][li]+right["var"][ri]+mean**2
    return base, dict(count=n, mean=mean, second_moment=second, variance=left["var"][li]+right["var"][ri],
                      pe_low_mean=left["pe"][li], pe_high_mean=right["pe"][ri])


def _take(values, mask):
    return {k: v[mask] for k, v in values.items()}


def pair_rows(h):
    """Compress sensor-bin cross products; repeated pulses contribute all distinct-sensor pairs."""
    base_keys, first, inv = np.unique(np.column_stack((h["group"], h["bar_id"], h["side"])),
                                      axis=0, return_index=True, return_inverse=True)
    b = {k: h[k][first] for k in ("group", "body", "side", "along", "path", "group_count", "layer")}
    for name in ("body", "side", "along", "path", "group_count", "layer", "distance"):
        values = h[name]
        J.require(np.allclose(values, values[first][inv], rtol=0, atol=1e-10, equal_nan=True),
                  f"same-bar/end pulses disagree in {name}")
    full_ids, full = _sensor_stats(inv, h)
    lo = full_ids % 2 == 0
    full_base, full_pairs = _cross(full_ids[lo]//2, _take(full, lo), full_ids[~lo]//2, _take(full, ~lo))
    pebin, pe_ok = J._bins(h["pe"], J.PE)
    ids, stats = _sensor_stats(inv, h, pe_ok, pebin)
    groups, pieces, keys = [], [], []
    along, aok = J._bins(b["along"], J.ALONG)
    path, pok = J._bins(b["path"], J.PATH)
    for name, edges in (("along", J.ALONG), ("path", J.PATH)):
        index, valid = J._bins(h[name], edges)
        J.require(np.array_equal(index, index[first][inv]) and np.array_equal(valid, valid[first][inv]),
                  f"same-bar/end {name} bin or validity mismatch")
    pair_pe_inrange = np.zeros(len(base_keys), np.int64)
    for p0 in range(5):
        for p1 in range(5):
            low, high = ids % 10 == p0, ids % 10 == 5+p1
            selected, values = _cross(ids[low]//10, _take(stats, low), ids[high]//10, _take(stats, high))
            pair_pe_inrange[selected] += values["count"]
            ok = aok[selected] & pok[selected]
            selected, values = selected[ok], _take(values, ok)
            keys.append(np.column_stack((b["body"][selected], b["side"][selected], along[selected], path[selected],
                np.full(len(selected), p0, int), np.full(len(selected), p1, int),
                b["group_count"][selected], b["layer"][selected])))
            groups.append(b["group"][selected])
            pieces.append(values)
    rows = dict(keys=np.concatenate(keys), group=np.concatenate(groups),
                **{k: np.concatenate([piece[k] for piece in pieces]) for k in pieces[0]})
    flow = {}
    for body, name in enumerate(T.G.BODY_NAMES):
        m = b["body"][full_base] == body
        all_n = full_pairs["count"][m]
        coord_ok = aok[full_base[m]] & pok[full_base[m]]
        valid_pe = int(pair_pe_inrange[b["body"] == body].sum())
        inrange = int(rows["count"][rows["keys"][:, 0] == body].sum())
        flow[name] = dict(all_pairs=int(all_n.sum()),
                         excluded_pe_pairs=int(all_n.sum())-valid_pe,
                         excluded_coordinates_after_pe_pairs=valid_pe-inrange,
                         distinct_event_slots=int(len(np.unique(b["group"][full_base[m]]))),
                         pairs_with_both_pe_inrange=int(pair_pe_inrange[b["body"] == body].sum()),
                         coordinate_excluded_pairs=int(all_n[~coord_ok].sum()),
                         inrange_pairs=int(rows["count"][rows["keys"][:, 0] == body].sum()),
                         all_second_moment=float(np.dot(all_n, full_pairs["second_moment"][m])/all_n.sum())
                         if all_n.sum() else None)
    flow["pulse_pe"] = J._flow(h["pe"], J.PE, np.ones(len(inv), bool))
    return rows, flow


def cell_table(rows, mode):
    J.require(mode in MODES, "unknown pair conditioning mode")
    key = rows["keys"][:, :6].copy()
    if mode != "none":
        n = rows["keys"][:, 6]
        mult = n if mode == "exact" else np.searchsorted(T.MULT, n, side="right")-1
        key = np.column_stack((key, mult))
    if mode == "layer_band":
        key = np.column_stack((key, rows["keys"][:, 7]))
    unique, inv = np.unique(key, axis=0, return_inverse=True)
    size = len(unique)
    count = np.bincount(inv, weights=rows["count"], minlength=size).astype(np.int64)
    cg, ci = np.unique(np.column_stack((inv, rows["group"])), axis=0, return_inverse=True)
    groups = np.bincount(cg[:, 0], minlength=size)
    gpairs = np.bincount(ci, weights=rows["count"], minlength=len(cg))
    gsecond = T._mean(np.bincount(ci, weights=rows["count"]*rows["second_moment"], minlength=len(cg)), gpairs)
    group_second = T._mean(np.bincount(cg[:, 0], weights=gsecond, minlength=size), groups)
    values = {f: T._mean(np.bincount(inv, weights=rows["count"]*rows[f], minlength=size), count) for f in FIELDS}
    variance = T._mean(np.bincount(inv, weights=rows["count"]*(rows["variance"]+
                      (rows["mean"]-values["mean"][inv])**2), minlength=size), count)
    result = [dict(key=k.tolist(), count=int(count[i]), groups=int(groups[i]), variance=float(variance[i]),
                   group_mean_second_moment=float(group_second[i]), **{f: float(v[i]) for f, v in values.items()})
              for i, k in enumerate(unique)]
    return result


def legacy_diagonal(h):
    base_keys, first, inv = np.unique(np.column_stack((h["group"], h["bar_id"], h["side"])),
                                      axis=0, return_index=True, return_inverse=True)
    pe, valid = J._bins(h["pe"], J.PE)
    ids, stats = _sensor_stats(inv, h, valid, pe)
    out = []
    for p in range(5):
        lo, hi = ids % 10 == p, ids % 10 == p+5
        index, pairs = _cross(ids[lo]//10, _take(stats, lo), ids[hi]//10, _take(stats, hi))
        body = h["body"][first][index]
        groups = base_keys[index, 0]
        for b in range(16):
            mask = body == b
            n = int(pairs["count"][mask].sum())
            out.append(dict(body=b, bin=p, pairs=n, groups=int(len(np.unique(groups[mask]))),
                            difference_second=float(np.dot(pairs["count"][mask], pairs["second_moment"][mask])/n)
                            if n else None))
    return sorted(out, key=lambda row: (row["body"], row["bin"]))


def check_legacy(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        J.require(all(a[k] == e[k] for k in ("body", "bin", "pairs", "groups")), "legacy pair census mismatch")
        J.require(a["difference_second"] is None and e["difference_second"] is None or
                  a["difference_second"] is not None and e["difference_second"] is not None and
                  np.isclose(a["difference_second"], e["difference_second"], rtol=1e-10, atol=1e-8),
                  "legacy pair moment mismatch")


def _moment(cells, weights):
    if not len(cells):
        return dict.fromkeys(FIELDS + ("variance",))
    values = {f: float(np.dot(weights, [c[f] for c in cells])) for f in FIELDS}
    values["variance"] = float(np.dot(weights, [c["variance"]+(c["mean"]-values["mean"])**2 for c in cells]))
    return values


def support_tables(samples):
    tables = {name: {tuple(row["key"]): row for row in rows} for name, rows in samples.items()}
    J.require(set(tables) == set((*J.ARMS, "data")), "common support needs four arms and data")
    supported = [{k for k, c in table.items() if c["count"] >= J.MIN_SUPPORT and c["groups"] >= T.MIN_GROUPS}
                 for table in tables.values()]
    return tables, sorted(set.intersection(*supported))


def standardize(tables, common, body, diagonal=None):
    select = lambda k: k[0] == body and (diagonal is None or k[4] == k[5] == diagonal)
    keys = [k for k in common if select(k)]
    dn = np.array([tables["data"][k]["count"] for k in keys], float)
    weights = dn/dn.sum() if dn.sum() else dn
    out = {}
    for name, table in tables.items():
        available = [c for k, c in table.items() if select(k)]
        cells = [table[k] for k in keys]
        counts = np.array([c["count"] for c in cells], float)
        total = sum(c["count"] for c in available)
        count = int(counts.sum())
        own = counts/count if count else counts
        out[name] = dict(inrange_pairs=total, common_pairs=count, coverage=count/total if total else None,
                         marginal=_moment(available, np.array([c["count"] for c in available])/total)
                         if total else _moment([], []),
                         support_own_weights=_moment(cells, own), data_weights=_moment(cells, weights),
                         group_cell_incidences=sum(c["groups"] for c in cells),
                         data_weighted_within_cell_group_second_moment=float(np.dot(weights,
                            [c["group_mean_second_moment"] for c in cells])) if cells else None)
    return dict(common_cells=len(keys), keys=[list(k) for k in keys], data_pair_weights=weights.tolist(), samples=out)


def fixed_support_multiplicity(tables, common, body):
    keys = [k for k in common if k[0] == body]
    if not keys:
        return dict(common_exact_cells=0, coarse_cells=0, samples={})
    coarse, inv = np.unique(np.array(keys)[:, :6], axis=0, return_inverse=True)
    data_count = np.array([tables["data"][k]["count"] for k in keys], float)
    weights = data_count/data_count.sum()
    coarse_weights = np.bincount(inv, weights=weights, minlength=len(coarse))
    result = {}
    for name, table in tables.items():
        cells = [table[k] for k in keys]
        n = np.array([c["count"] for c in cells], float)
        cn = np.bincount(inv, weights=n, minlength=len(coarse))
        coarsened_weights = coarse_weights[inv]*n/cn[inv]
        fine = _moment(cells, weights)
        collapsed = _moment(cells, coarsened_weights)
        result[name] = dict(pairs=int(n.sum()), fine_data_weights=fine,
                            collapsed_N_own_mix_data_coarse_weights=collapsed,
                            second_moment_reweighting=fine["second_moment"]-collapsed["second_moment"])
    return dict(common_exact_cells=len(keys), coarse_cells=len(coarse), samples=result,
                meaning="Identical fine-supported pair rows; only within-coarse-cell multiplicity fractions change. Not causal.")


def analyze(h, flow, previous):
    diagonal = legacy_diagonal(h)
    check_legacy(diagonal, previous["pairs"])
    J.require(flow == previous["flow"], "legacy selected-hit flow mismatch")
    rows, pair_flow = pair_rows(h)
    tables = {mode: cell_table(rows, mode) for mode in MODES}
    return dict(selected_flow=flow, pair_flow=pair_flow, strata=tables,
                legacy_diagonal=diagonal, legacy_pair_census_and_moments_equal=True)


def bind_captures(previous_files, receipts, previous_events, events):
    prior = {p: sha for p, sha in previous_files.items() if p.endswith(".npz")}
    current = {p: r["sha256"] for p, r in receipts.items() if p.endswith(".npz")}
    J.require(prior == current, "capture set differs from previous timing report")
    J.require(events == previous_events, "data exposure differs from previous timing report")


def compare(directory, data, previous, out):
    J.require(not Path(out).exists(), "output already exists")
    receipts = {}
    modules = (Path(__file__), Path(T.__file__), Path(J.__file__), Path(J.joint.__file__), Path(V.__file__),
               Path(J.RC.__file__), Path(J.RC.DEFAULT_PATH), Path(T.G.__file__), Path(T.R.__file__), Path(T.R.CL.__file__))
    for source in modules:
        J._receipt(source, receipts)
    old = J._json(previous, receipts)
    J.require(old.get("passed") is True and old.get("comparison") == "joint-timing", "invalid prior timing report")
    prior = {name: dict(flow=s["flow"], pairs=s["pairs"]["same_bar_same_end_distinct_sipm_equal_pe_bin"])
             for name, s in old["samples"].items()}
    previous_files = {str(Path(f["path"]).resolve()): f["sha256"] for f in old["files"]}
    previous_events = old["data"]["events"]
    del old
    points, gates = J.load_block(directory, receipts)
    asset = points["nominal"]["provenance"]["manifest"]["assets"]["calib"]
    mapping = T.channel_map(asset["path"], receipts, asset["sha256"])
    samples = {}
    for name in J.ARMS:
        point = points.pop(name)
        h, flow = T.prepare(point.pop("columns"), mapping)
        del point
        samples[name] = analyze(h, flow, prior[name])
        del h
    columns, summary, provenance = J.load_data(data, receipts)
    h, flow = T.prepare(columns, mapping)
    del columns
    samples["data"] = analyze(h, flow, prior["data"])
    del h
    bind_captures(previous_files, receipts, previous_events, summary["events"])
    standardized, fixed = {}, {}
    for mode in MODES:
        tables, common = support_tables({name: s["strata"][mode] for name, s in samples.items()})
        standardized[mode] = {name: dict(all_pe=standardize(tables, common, b),
                            diagonal_pe=[standardize(tables, common, b, p) for p in range(5)])
                              for b, name in enumerate(T.G.BODY_NAMES)}
        if mode == "exact":
            fixed = {name: fixed_support_multiplicity(tables, common, b) for b, name in enumerate(T.G.BODY_NAMES)}
    report = dict(schema=1, comparison="joint-pairs", passed=True, samples=samples,
                  standardization=standardized, fixed_support_multiplicity=fixed, replay_gates=gates,
                  data=dict(events=summary["events"], provenance=provenance),
                  map=dict(asset=asset, policy="projected frozen MC map, not certified historical P4 cabling"),
                  axes=dict(bodies=list(T.G.BODY_NAMES), along=J.ALONG.tolist(), path=J.PATH.tolist(), pe=J.PE.tolist(),
                            multiplicity_lower_edges=T.MULT[:-1].tolist(), key_names=list(KEY_NAMES)),
                  definitions=dict(population="complete finite selected event/slot spread groups, both lines, >=3 members; same bar/end distinct SiPM pairs",
                    orientation="lower SiPM index minus higher: 0-2 at side 0, 1-3 at side 1; PE bins follow sensor, not PE order",
                    pairing="within sample/event/slot only; every low/high sensor pulse combination, never cross-arm pulse matching",
                    statistic="mean signed time difference and mean difference squared [ns, ns^2]; not intrinsic sensor resolution",
                    coordinates="same measured along/path checked within each bar/end; both PE bins in [0,200], interior bins left-closed/right-open, final upper edge inclusive",
                    flow="exclusive losses: excluded_pe_pairs then excluded_coordinates_after_pe_pairs; coordinate_excluded_pairs is an overlapping marginal",
                    support="at least 30 pairs AND 10 distinct event/slot groups per cell in all four arms and data",
                    weights="fixed data pair fractions on common support; also each arm's own fractions on same support",
                    multiplicity="full event/slot spread group_count, not pair count or per-body count",
                    group_weighting="within-cell mean of per-event/slot pair moments, then data pair cell fractions; groups may occupy multiple cells",
                    coverage="common / in-range eligible pairs within body (or diagonal PE); exclusions reported separately",
                    independent_units="pair counts and sums of cell group counts are NOT independent event counts"),
                  cautions=["No fitting, time calibration, background/response changes, rescore or acceptance/significance claim.",
                    "Modes use different common support. Fixed-support contrast holds exact-N-supported rows and coarse target fractions fixed.",
                    "Coarse measured bins leave residual position/path/PE composition; sparse excluded populations are unresolved.",
                    "Same-bar/end differences cancel line propagation and any common group/module offset, not sensor-specific effects.",
                    "Neither a difference moment nor half that moment is an intrinsic sensor variance without additional assumptions.",
                    "Historical P4 map/cabling not certified. Photon arrivals, fit shapes/quality and phase samples are absent.",
                    "Data are a scoring-reference subset; two MC blocks show consistency, not calibrated uncertainty."])
    for record in receipts.values():
        J.require(V.digest(record["path"]) == record["sha256"], f"file changed: {record['path']}")
    report["files"] = list(receipts.values())
    return V._report(out, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "data", "previous", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    try:
        compare(args.directory, args.data, args.previous, args.out)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"joint-pairs: PASS; descriptive report {args.out}")


if __name__ == "__main__":
    main()
