"""Saved-array multiplicity, hardware and measured pulse-timing diagnostics; no refit."""
import argparse
from pathlib import Path

import numpy as np

from . import bars as G, crvresponse as R, joint_compare as J, validation as V

MULT = np.array([3, 5, 9, 13, 17, 25, np.iinfo(np.int64).max])
MIN_GROUPS = 10


def _mean(total, count):
    return np.divide(total, count, out=np.zeros_like(total, dtype=float), where=count > 0)


def channel_map(path, receipts, expected_sha):
    """Read only channel/ROC/FEB labels from the immutable MC conditions asset."""
    J.require(J._receipt(path, receipts)["sha256"] == expected_sha, "channel map receipt mismatch")
    table = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    J.require(table.shape[1] == 9, "channel map needs nine columns")
    labels = table[:, [0, 6, 7]]
    J.require(np.all(np.isfinite(labels)) and np.all(labels == np.floor(labels)) and
              np.all(labels[:, 0] >= 0) and np.all(labels[:, 1:] >= -1) and np.all(labels < 2**31) and
              np.all((labels[:, 1] == -1) == (labels[:, 2] == -1)), "invalid channel/ROC/FEB labels")
    labels = labels.astype(np.int64)
    J._unique(labels[:, 0], "channel map")
    order = np.argsort(labels[:, 0])
    return labels[order]


def prepare(columns, mapping, extra_fields=()):
    """Retain complete finite, two-projection spread groups, with exact int64 keys."""
    h = {k[4:]: v for k, v in columns.items() if k.startswith("hit_")}
    eligible = h["both_lines"] & h["spread_member"] & h["spread_eligible"]
    finite = np.isfinite(h["time_residual"]) & np.isfinite(h["time"]) & np.isfinite(h["distance"])
    keys = J._keys(columns, "hit")
    _, inv = np.unique(keys[eligible], return_inverse=True)
    bad = np.bincount(inv, weights=~finite[eligible]) > 0
    keep = eligible.copy()
    keep[eligible] &= ~bad[inv]
    needed = ("body", "slot", "layer", "bar_id", "sipm_id", "side", "group_count", "time_residual",
              "time", "distance", "pe", "along", "path")
    J.require(not set(extra_fields) & set(needed), 'supplemental fields must not replace timing coordinates')
    selected = {k: h[k][keep] for k in (*needed, *extra_fields)}
    _, groups = np.unique(keys[keep], return_inverse=True)
    n = np.bincount(groups)
    J.require(np.array_equal(n[groups], selected["group_count"]), "incomplete timing group membership")
    J.require(np.all(n >= 3), "ineligible timing group")
    bar, sipm = selected["bar_id"], selected["sipm_id"]
    J.require(bar.dtype.kind in "iu" and sipm.dtype.kind in "iu" and
              np.all((bar >= 0) & (bar < 576)) and np.all((sipm >= 0) & (sipm < 4)), "invalid channel")
    slot, body, layer = G.bar_slot_body_layer(bar)
    J.require(all(np.array_equal(a, selected[k]) for a, k in
                  ((slot, "slot"), (body, "body"), (layer, "layer"), (sipm % 2, "side"))),
              "hardware labels contradict bar/SiPM identity")
    channel = 4 * bar + sipm
    mi = np.searchsorted(mapping[:, 0], channel)
    J.require(np.all(mi < len(mapping)), "unmapped channel")
    J.require(np.array_equal(mapping[mi, 0], channel), "unmapped channel")
    J.require(np.all(mapping[mi, 1:] >= 0), "selected pulse has unavailable ROC/FEB mapping")
    r = selected["time_residual"]
    J.require(np.all(np.abs(np.bincount(groups, weights=r)) < 1e-6 * n), "residuals are not group centered")
    prop = selected["distance"] / R.V_FIBER
    centered_prop = prop - _mean(np.bincount(groups, weights=prop), n)[groups]
    raw = selected["time"] - _mean(np.bincount(groups, weights=selected["time"]), n)[groups]
    J.require(np.allclose(r, raw-centered_prop, rtol=1e-10, atol=1e-7),
              "residual disagrees with saved time and propagation")
    selected.update(group=groups, channel=channel, roc=mapping[mi, 1], feb=mapping[mi, 2],
                    propagation_centered=centered_prop, raw_centered=raw,
                    multiplicity_bin=np.searchsorted(MULT, selected["group_count"], side="right") - 1)
    return selected, dict(selected_first_pulses=len(keys), eligible_members=int(eligible.sum()),
                         nonfinite_members=int((eligible & ~finite).sum()),
                         excluded_nonfinite_group_members=int(eligible.sum() - keep.sum()),
                         retained_members=int(keep.sum()), retained_event_slots=len(n))


def moments(keys, h):
    """Sparse cells with distinct group counts and centered raw/propagation decomposition."""
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    n = np.bincount(inv, minlength=len(unique))
    pairs = np.unique(np.column_stack((inv, h["group"])), axis=0)
    ng = np.bincount(pairs[:, 0], minlength=len(unique)) if len(pairs) else np.zeros(len(unique), int)
    r, raw, prop = h["time_residual"], h["raw_centered"], h["propagation_centered"]
    values = dict(mean=r, second_moment=r*r, raw_second_moment=raw*raw,
                  propagation_second_moment=prop*prop, raw_propagation=raw*prop)
    reduced = {k: _mean(np.bincount(inv, weights=v, minlength=len(unique)), n) for k, v in values.items()}
    return [dict(key=key.tolist(), count=int(n[i]), groups=int(ng[i]),
                 **{k: float(v[i]) for k, v in reduced.items()}) for i, key in enumerate(unique)]


def subset(h, mask):
    return {k: v[mask] for k, v in h.items()}


def cells(h, mode):
    keep = np.ones(len(h["body"]), bool)
    coords, flows = [h["body"]], {}
    for name, edges in (("along", J.ALONG), ("path", J.PATH), ("pe", J.PE)):
        index, valid = J._bins(h[name], edges)
        coords.append(index)
        keep &= valid
        flows[name] = J._flow(h[name], edges, np.ones(len(keep), bool))
    if mode != "none":
        coords.append(h["group_count"] if mode == "exact" else h["multiplicity_bin"])
    return dict(flows=flows, coordinate_excluded=int((~keep).sum()),
                cells=moments(np.column_stack(coords)[keep], subset(h, keep)))


def standardize(samples, body, pe=None):
    tables = {name: {tuple(c["key"]): c for c in sample["cells"]
                     if c["key"][0] == body and (pe is None or c["key"][3] == pe)}
              for name, sample in samples.items()}
    supports = [{k for k, c in t.items() if c["count"] >= J.MIN_SUPPORT and c["groups"] >= MIN_GROUPS}
                for t in tables.values()]
    common = sorted(set.intersection(*supports))
    weights = np.array([tables["data"][k]["count"] for k in common], float)
    weights = weights / weights.sum() if len(weights) else weights
    out = {}
    fields = ("mean", "second_moment", "raw_second_moment", "propagation_second_moment", "raw_propagation")
    for name, table in tables.items():
        total = sum(c["count"] for c in table.values())
        retained = sum(table[k]["count"] for k in common)
        measured = {f: float(np.dot(weights, [table[k][f] for k in common])) if common else None for f in fields}
        out[name] = dict(inrange_hits=total, common_hits=retained, coverage=retained/total if total else None,
                         data_weighted=measured,
                         support_matched_second_moment=sum(table[k]["count"]*table[k]["second_moment"]
                                                          for k in common)/retained if retained else None)
    return dict(min_hits=J.MIN_SUPPORT, min_distinct_event_slots=MIN_GROUPS, common_cells=len(common),
                keys=[list(k) for k in common], data_weights=weights.tolist(), samples=out)


def _pair_sums(h, keys):
    """Unordered-pair sums via subgroup sufficient statistics, never a pulse cross product."""
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    n = np.bincount(inv, minlength=len(unique)).astype(np.int64)
    r, a, b = h["time_residual"], h["raw_centered"], h["propagation_centered"]
    sums = {k: np.bincount(inv, weights=v, minlength=len(unique)) for k, v in
            dict(r=r, rr=r*r, a=a, aa=a*a, b=b, bb=b*b, ab=a*b).items()}
    s = sums
    rc, ac, bc = (v-_mean(s[k], n)[inv] for v, k in ((r, "r"), (a, "a"), (b, "b")))
    centered = {k: np.bincount(inv, weights=v, minlength=len(unique)) for k, v in
                dict(rr=rc*rc, aa=ac*ac, bb=bc*bc, ab=ac*bc).items()}
    return unique, dict(pairs=n*(n-1)//2, endpoint_sum=(n-1)*s["r"]/2,
                        endpoint_second=(n-1)*s["rr"]/2,
                        product=(s["r"]**2-s["rr"])/2,
                        difference_second=n*centered["rr"],
                        raw_difference_second=n*centered["aa"],
                        propagation_difference_second=n*centered["bb"],
                        raw_propagation_difference=n*centered["ab"])


def pair_report(h):
    """Inclusive hardware relationships, per body and multiplicity band, plus equal-PE-bin controls."""
    base = np.column_stack((h["body"], h["multiplicity_bin"], h["group"]))
    definitions = dict(all=[], layer=[h["layer"]], sipm_index=[h["sipm_id"]],
                       feb=[h["roc"], h["feb"]], bar=[h["bar_id"]],
                       bar_side=[h["bar_id"], h["side"]], channel=[h["channel"]])
    totals = {}
    for name, extra in definitions.items():
        keys = np.column_stack([base] + extra)
        unique, values = _pair_sums(h, keys)
        groups, totals[name] = _reduce_groups(unique, values)
    relations = dict(all=("all", None), same_layer=("layer", None), different_layer=("all", "layer"),
                     same_sipm_index=("sipm_index", None), different_sipm_index=("all", "sipm_index"),
                     same_mapped_feb=("feb", None), different_mapped_feb=("all", "feb"),
                     same_channel=("channel", None), same_bar_same_end_distinct_sipm=("bar_side", "channel"),
                     same_bar_opposite_end=("bar", "bar_side"))
    out = {}
    for label, (plus, minus) in relations.items():
        sums = {k: v-(totals[minus][k] if minus else 0) for k, v in totals[plus].items()}
        out[label] = _pair_rows(_aggregate_pairs(groups, sums, 6))
    pe, valid = J._bins(h["pe"], J.PE)
    hs = subset(h, valid)
    keys = np.column_stack((hs["body"], pe[valid], hs["group"], hs["bar_id"], hs["side"]))
    unique, same = _pair_sums(hs, keys)
    unique_ch, channel = _pair_sums(hs, np.column_stack((keys, hs["channel"])))
    groups, same = _reduce_groups(unique, same)
    groups_ch, channel = _reduce_groups(unique_ch, channel)
    J.require(np.array_equal(groups, groups_ch), "pair groups disagree")
    sums = {k: v-channel[k] for k, v in same.items()}
    out["same_bar_same_end_distinct_sipm_equal_pe_bin"] = _pair_rows(_aggregate_pairs(groups, sums, 5), bins=5)
    return out


def _reduce_groups(keys, values):
    groups, inv = np.unique(keys[:, :3], axis=0, return_inverse=True)
    return groups, {k: np.bincount(inv, weights=v, minlength=len(groups)) for k, v in values.items()}


def _aggregate_pairs(groups, values, bins):
    cell = groups[:, 0]*bins + groups[:, 1]
    sums = {k: np.bincount(cell, weights=v, minlength=16*bins) for k, v in values.items()}
    n = values["pairs"]
    sums["groups"] = np.bincount(cell[n > 0], minlength=16*bins)
    sums["group_weighted_difference_second_sum"] = np.bincount(
        cell, weights=_mean(values["difference_second"], n), minlength=16*bins)
    return sums


def _pair_rows(sums, bins=6):
    n = sums["pairs"]
    means = {k: _mean(v, n) for k, v in sums.items()
             if k not in ("pairs", "groups", "group_weighted_difference_second_sum")}
    mean, second, prod = means["endpoint_sum"], means["endpoint_second"], means["product"]
    variance, cov = second-mean**2, prod-mean**2
    return [dict(body=i//bins, bin=i % bins, pairs=int(n[i]), groups=int(sums["groups"][i]),
                 group_weighted_difference_second=float(sums["group_weighted_difference_second_sum"][i]/sums["groups"][i])
                 if sums["groups"][i] else None,
                 **{k: float(v[i]) if n[i] else None for k, v in means.items()},
                 covariance=float(cov[i]) if n[i] else None,
                 correlation=float(cov[i]/variance[i]) if n[i] and variance[i] > 0 else None)
            for i in range(len(n))]


def analyze(columns, mapping):
    h, flow = prepare(columns, mapping)
    events = columns["event_selected"] & columns["event_both_lines"] & columns["event_spread_eligible"]
    event_report = {}
    for body, name in enumerate(G.BODY_NAMES):
        mask = events & (columns["event_body"] == body) & np.isfinite(columns["event_group_rms"])
        event_report[name] = dict(counts=J.count_distribution(columns["event_group_count"][mask]),
                                  rms=J.stats(columns["event_group_rms"][mask]),
                                  by_multiplicity=[J.stats(columns["event_group_rms"][mask &
                                     (columns["event_group_count"] >= lo) & (columns["event_group_count"] < hi)])
                                     for lo, hi in zip(MULT[:-1], MULT[1:])])
    hardware = {}
    for name, coords in dict(layer=["layer"], sipm_index=["sipm_id"], mapped_feb=["roc", "feb"]).items():
        hardware[name] = moments(np.column_stack([h[k] for k in ["body"] + coords]), h)
    return dict(flow=flow, event_multiplicity=event_report, hardware=hardware,
                strata={mode: cells(h, mode) for mode in ("none", "band", "exact")}, pairs=pair_report(h))


def compare(directory, data, out):
    J.require(not Path(out).exists(), "output already exists")
    receipts = {}
    sources = (Path(__file__), Path(J.__file__), Path(J.joint.__file__), Path(V.__file__),
               Path(J.RC.__file__), Path(J.RC.DEFAULT_PATH), Path(G.__file__), Path(R.__file__),
               Path(R.CL.__file__))
    for source in sources:
        J._receipt(source, receipts)
    points, gates = J.load_block(directory, receipts)
    asset = points["nominal"]["provenance"]["manifest"]["assets"]["calib"]
    mapping = channel_map(asset["path"], receipts, asset["sha256"])
    samples = {}
    for arm in J.ARMS:
        point = points.pop(arm)
        samples[arm] = analyze(point["columns"], mapping)
        del point
    columns, summary, provenance = J.load_data(data, receipts)
    samples["data"] = analyze(columns, mapping)
    del columns
    standardized = {}
    for mode in ("none", "band", "exact"):
        strata = {k: v["strata"][mode] for k, v in samples.items()}
        standardized[mode] = {name: dict(all_pe=standardize(strata, i),
                                  by_pe=[standardize(strata, i, p) for p in range(5)])
                              for i, name in enumerate(G.BODY_NAMES)}
    report = dict(schema=1, comparison="joint-timing", passed=True, samples=samples,
                  standardization=standardized, replay_gates=gates,
                  data=dict(events=summary["events"], provenance=provenance),
                  axes=dict(bodies=list(G.BODY_NAMES), along=J.ALONG.tolist(), path=J.PATH.tolist(),
                            pe=J.PE.tolist(), multiplicity_lower_edges=MULT[:-1].tolist()),
                  map=dict(asset=asset, data_policy="projected frozen MC channel map, not certified P4 cabling",
                           clock_policy="MC v05 uses one phase per event, not independent physical FEB phases"),
                  definitions=dict(population="complete finite selected event/slot spread groups with both projections; >=3 members",
                    cells="body, measured along bin, path bin, PE bin, optional multiplicity band or exact N",
                    weights="finite hit counts; common support requires >=30 hits AND >=10 distinct event/slot groups per sample",
                    pairs="unordered within-event/slot pairs in the same body; inclusive relations; pulse pairs never matched across arms",
                    pair_bins="multiplicity bands except equal_pe_bin, where both pulse PEs must lie in that bin",
                    sipm_index="0..3 within a bar, not a globally unique sensor; channel=4*bar+SiPM",
                    decomposition="r=a-b; a=time minus full-group mean(time), b=distance/140 minus full-group mean(distance/140)",
                    pair_difference="E[(r_i-r_j)^2]=raw_difference_second+propagation_difference_second-2*raw_propagation_difference",
                    pair_correlation="symmetric pooled endpoint covariance/variance; residual centering induces anticorrelation",
                    event_body="representative body of event/slot, not a per-body timing group",
                    propagation_speed_mm_per_ns=float(R.V_FIBER),
                    units="times ns; second moments and covariances ns^2"),
                  cautions=["Descriptive only; no uncertainty, intrinsic resolution, significance, calibration or acceptance claim.",
                    "Multiplicity modes have different support/weights; cross-mode changes cannot be attributed solely to multiplicity.",
                    "Multiplicity and measured-cell conditioning change support; report coverage, including sparse/empty sectors.",
                    "Hardware moments and pair tables are unstandardized; pair weights grow as N(N-1)/2.",
                    "Centered residual correlations include the fitted shared mean; iid full-group expectation is -1/(N-1).",
                    "Same-bar/same-end distinct-SiPM differences cancel line propagation, but retain optical, sensor, fitting and selection effects.",
                    "Photon arrivals, fit beta/chi2, waveform and clock phase were not captured; PE trends cannot separate those mechanisms.",
                    "All sectors/sides retained; data channel labels are projected through the frozen MC map, not verified data electronics.",
                    "Background, timecal, response, reconstruction and GP noise unchanged; no response rerun.",
                    "Data subset belongs to the scoring reference, not an independent calibration holdout; A/B are consistency blocks."])
    for record in receipts.values():
        J.require(V.digest(record["path"]) == record["sha256"], f"file changed: {record['path']}")
    report["files"] = list(receipts.values())
    return V._report(out, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        compare(args.directory, args.data, args.out)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"joint-timing: PASS; descriptive report {args.out}")


if __name__ == "__main__":
    main()
