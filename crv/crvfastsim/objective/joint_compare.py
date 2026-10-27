"""Read-only four-arm measured joint diagnostics; no simulation, fitting or pulse pairing.

python -m crvfastsim.objective.joint_compare --directory BLOCK --out NEW.json [--data DATA]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import response_config as RC
from . import bars as G, joint, validation as V

ARMS = ("nominal", "atten", "yield", "joint")
ATTEN = 0.4367132104009329
YIELD_NOMINAL, YIELD_LOW = 28610., 20027.
ALONG = np.array([-1., -.6, -.2, .2, .6, 1.])
PATH = np.array([1., 1.25, 1.5, 2., 3.])
PE = np.array([0., 20., 40., 60., 100., 200.])
MIN_SUPPORT = 30


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _receipt(path, receipts):
    path = Path(path).resolve()
    record = dict(path=str(path), sha256=V.digest(path))
    require(str(path) not in receipts or receipts[str(path)] == record, f"file changed during comparison: {path}")
    receipts[str(path)] = record
    return record


def _json(path, receipts):
    _receipt(path, receipts)
    return json.loads(Path(path).read_text())


def _keys(columns, table, extra=("slot",)):
    names = ("run", "subrun", "event") + tuple(extra)
    arrays = [columns[f"{table}_{name}"] for name in names]
    require(all(a.ndim == 1 and a.dtype.kind in "iu" and np.all(a >= 0) and
                np.all(a <= np.iinfo(np.int64).max) for a in arrays), "invalid integer identity")
    result = np.empty(len(arrays[0]), dtype=[(k, "<i8") for k in names])
    for name, array in zip(names, arrays):
        result[name] = array
    return result


def _unique(keys, label):
    require(len(np.unique(keys)) == len(keys), f"duplicate {label} identities")


def _audit(columns, summary):
    require(summary["schema_version"] == joint.SCHEMA_VERSION, "unsupported joint schema")
    require(summary["slots"] == list(G.SLOT_NAME) and summary["body_names"] == list(G.BODY_NAMES),
            "joint slot/body schema mismatch")
    for table in ("event", "hit", "candidate"):
        n = len(columns[table + "_slot"])
        for name, values in columns.items():
            if name.startswith(table + "_"):
                require(values.ndim >= 1 and len(values) == n and values.dtype.kind in "biuf",
                        f"invalid column {name}")
        body, slot = columns[table + "_body"], columns[table + "_slot"]
        require(body.dtype.kind in "iu" and np.all((body >= -1) & (body < 16)), "invalid body")
        require(slot.dtype.kind in "iu" and np.all((slot >= (-1 if table == "candidate" else 0)) &
                                                  (slot < 4)), "invalid slot")
    keys = _keys(columns, "event")
    _unique(keys, "event/slot")
    event_keys, counts = np.unique(_keys(columns, "event", ()), return_counts=True)
    require(np.all(counts == 4), "each exposure identity needs all four slots")
    require(summary["events"] == len(event_keys), "joint event census mismatch")
    # Pulse ordinals are used ONLY to audit unique rows within one capture; never
    # to match arms. Distinct pulses may share bar/SiPM/sequence membership.
    for table, extra in (("hit", ("pulse_index",)),
                         ("candidate", ("cluster_index",))):
        _unique(_keys(columns, table, extra), table)
        require(np.all(np.isin(_keys(columns, table, ()), event_keys)), f"orphan {table} event identity")
    for table, names in (("event", ("selected", "spread_eligible")), ("candidate", ("selected",)),
                         ("hit", ("both_lines", "spread_member", "spread_eligible",
                                  "pe_along_eligible", "pe_path_eligible"))):
        require(all(columns[f"{table}_{name}"].dtype.kind == "b" and
                    columns[f"{table}_{name}"].ndim == 1 for name in names), f"invalid {table} flags")
    require(columns["candidate_pe_layer"].shape == (len(columns["candidate_slot"]), 4),
            "layer PE must have four columns")
    require(joint.summarize(columns, summary["chunks"]) == summary, "joint summary/counts mismatch")
    order = np.argsort(keys)
    return keys[order], order


def _npz(path, receipts):
    _receipt(path, receipts)
    with np.load(path, allow_pickle=False) as archive:
        summary = json.loads(str(archive["summary_json"]))
        columns = {k: archive[k] for k in archive.files if k != "summary_json"}
    return columns, summary


def _config(point, arm):
    prov, result = point["provenance"], point["r3"]
    record = prov["response_config"]
    config = record["values"]
    legacy = 'add_opposite_side_pulses' not in config['finder']
    compatible = json.loads(json.dumps(config))
    if legacy:
        compatible['finder']['add_opposite_side_pulses'] = False
    RC.validate(compatible)
    expected = RC.defaults()
    if legacy:
        expected['finder'].pop('add_opposite_side_pulses')
    expected["timing"].update(phase_scope="per_event", timeoffset_mode="zero")
    for key in ("calib", "deviation", "background", "time_shift"):
        expected["assets"][key] = prov["manifest"]["assets"][key]["path"]
    if arm in ("atten", "joint"):
        expected["response"]["atten_ratio"]["T1"] = ATTEN
    if arm in ("yield", "joint"):
        expected["response"]["sectors"]["T1"]["yield_per_mev"] = YIELD_LOW
    require(config == expected, f"{arm}: unexpected effective response coordinates/config")
    require(RC.defaults()["response"]["sectors"]["T1"]["yield_per_mev"] == YIELD_NOMINAL,
            "nominal yield preset changed")
    canonical = RC.canonical(config)
    require(record["canonical"] == canonical and record["sha256"] == RC.record(config)["sha256"],
            f"{arm}: effective config hash mismatch")
    resolved = result["resolved_config"]
    require(resolved["values"] == config and resolved["sha256"] == record["sha256"],
            f"{arm}: R3/provenance effective config mismatch")
    require(RC.apply_knobs(compatible, prov["knobs"]) == compatible, f"{arm}: knobs contradict effective config")
    window = config["timing"]["window"]
    require(result["response_state"]["exposure_ns"] == prov["manifest"]["events"] * (window[1] - window[0]),
            f"{arm}: time exposure mismatch")


def _frozen(prov):
    # Compare every field, including future additions, not a hand-picked identity subset.
    return {k: v for k, v in prov.items() if k not in ("knobs", "response_config", "joint_files")}


def _gate(root, arm, point, receipts):
    gate = _json(root / f"{arm}_capture_gate.json", receipts)
    require(gate["comparison"] == "compare-replay" and gate["passed"] is True and
            gate["response_state_equal"] is True, f"{arm}: capture replay gate is not PASS")
    candidate = root / arm / "r3.json"
    require(gate["candidate"]["sha256"] == V.digest(candidate) and
            Path(gate["candidate"]["path"]).resolve() == candidate.resolve(),
            f"{arm}: replay candidate receipt mismatch")
    reference_path = Path(gate["reference"]["path"])
    require(_receipt(reference_path, receipts)["sha256"] == gate["reference"]["sha256"],
            f"{arm}: replay reference changed")
    reference = _json(reference_path, receipts)
    result = point["r3"]
    counts = V._result_counts(result)
    require(counts == V._result_counts(reference) == gate["counts"], f"{arm}: replay counts mismatch")
    for name in ("seed", "knobs"):
        require(reference[name] == result[name] == gate[name], f"{arm}: replay {name} mismatch")
    require(reference["response_state"] == result["response_state"], f"{arm}: replay not exact")
    return gate


def load_block(directory, receipts):
    root = Path(directory).resolve()
    points = {}
    for arm in ARMS:
        point = {k: _json(root / arm / f"{k}.json", receipts) for k in ("r3", "comparison", "provenance")}
        prov = point["provenance"]
        require(prov["status"] == "complete" and prov["integrity_errors"] == [] and
                "execution_error" not in prov, f"{arm}: incomplete arm/integrity failure")
        require(prov["joint_capture"] is True and prov["code"] == prov["code_after"],
                f"{arm}: capture/code integrity failure")
        require(prov["diagnostic_code"] is True and prov["code_policy"] == "unpublished-diagnostic-snapshot",
                f"{arm}: joint comparison requires diagnostic snapshot provenance")
        for code in prov["code"].values():
            require(code["mode"] == "diagnostic-worktree" and code["published"] is False and
                    bool(code["sources"]), f"{arm}: missing diagnostic code snapshot")
            sha = hashlib.sha256(json.dumps(code["sources"], sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
            require(sha == code["source_sha256"], f"{arm}: source code hash mismatch")
        # Apply the shared census/score contract after the distinct diagnostic-source audit.
        counts = V._validate_measurement(point["r3"], point["comparison"], prov)
        _config(point, arm)
        require(set(prov["joint_files"]) == {"joint.json", "joint.npz"}, "missing joint file receipts")
        for name, record in prov["joint_files"].items():
            path = root / arm / name
            require(Path(record["path"]).resolve() == path and
                    _receipt(path, receipts)["sha256"] == record["sha256"],
                    f"{arm}: joint receipt mismatch")
        columns, summary = _npz(root / arm / "joint.npz", receipts)
        require(summary == _json(root / arm / "joint.json", receipts), f"{arm}: joint summaries differ")
        keys, order = _audit(columns, summary)
        require(summary["events"] == counts["events"], f"{arm}: incomplete capture exposure")
        point.update(columns=columns, summary=summary, keys=keys, order=order, counts=counts)
        points[arm] = point
    nominal = points["nominal"]
    for arm, point in points.items():
        require(point['provenance']['response_config']['values']['finder'] ==
                nominal['provenance']['response_config']['values']['finder'],
                f'{arm}: finder schema/settings differ across arms')
        require(_frozen(point["provenance"]) == _frozen(nominal["provenance"]),
                f"{arm}: frozen provenance/manifest mismatch")
        require(np.array_equal(point["keys"], nominal["keys"]), f"{arm}: exposure identities differ")
        for field in ("jax", "noise", "v05", "batch_events", "background", "time_shift"):
            require(point["r3"].get(field) == nominal["r3"].get(field), f"{arm}: result {field} mismatch")
        require(point["r3"]["resolved_config"].get("assets") == nominal["r3"]["resolved_config"].get("assets"),
                f"{arm}: resolved asset mismatch")
        require(point["comparison"].get("n0") == nominal["comparison"].get("n0"), "score normalization mismatch")
    gates = {arm: _gate(root, arm, points[arm], receipts) for arm in ("nominal", "atten")}
    return points, gates


def _data_policy(provenance):
    require(provenance.get("sample_role") == "measured_data", "data sample_role must be measured_data")
    require("time_shift_ns" in provenance and provenance["time_shift_ns"] is None and
            provenance.get("time_policy") in ("unshifted", "unshifted stock data; no recalibration"),
            "data must use unshifted time_policy and null time_shift_ns")
    sources = provenance.get("inputs", [])
    require(provenance.get("full_files") is True and bool(sources),
            "data require full-file saved reference equality; partial diagnostics are not admitted")
    fields = {"events", "files", "exposure_ns", "lines_z", "lines_x", "lines_xz", "h", "oot_n"}
    for source in sources:
        reference = source.get("reference_comparison", {})
        require(source.get("complete") is True and source.get("full_file") is True and
                reference.get("status") == "exact" and bool(reference.get("path")) and
                _sha256(reference.get("sha256")) and
                set(reference.get("matched_scoring_fields", [])) == fields,
                "data require exact saved per-file reference equality and hashed reference provenance")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_data(directory, receipts):
    """Preallocate once; retain only final arrays plus one audited raw shard.

    The metadata pass reads only summary_json, not the numeric tables. Full-array
    audits still reject cross-shard duplicates; raw artifacts remain untouched.
    """
    root = Path(directory).resolve()
    provenance = _json(root / "provenance.json", receipts)
    require(provenance.get("complete") is True or provenance.get("status") == "complete", "incomplete data")
    _data_policy(provenance)
    if (root / "joint.npz").is_file():
        recorded = provenance.get("joint_files", {}).get("joint.npz", {})
        records = [dict(recorded, path="joint.npz")]
    else:
        manifest = _json(root / "summary.json", receipts)
        require(manifest["complete"] is True and manifest["provenance"]["sha256"] ==
                V.digest(root / "provenance.json"), "data completion/provenance mismatch")
        records = manifest["raw_shards"]
    require(bool(records), "no data capture files")
    summaries = []
    for record in records:
        path, sha = root / record["path"], record.get("sha256")
        require(_sha256(sha), "data shard requires hashed provenance")
        require(_receipt(path, receipts)["sha256"] == sha, "data shard hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            summary = json.loads(str(archive["summary_json"]))
        require("summary" not in record or record["summary"] == summary, "data shard summary receipt mismatch")
        summaries.append(summary)
    tables = ("event", "hit", "candidate")
    totals = {table: sum(s["tables"][table]["all"]["rows"] for s in summaries) for table in tables}
    offsets = dict.fromkeys(tables, 0)
    columns = None
    for record, expected in zip(records, summaries):
        shard, summary = _npz(root / record["path"], receipts)
        require(summary == expected, "data shard summary changed")
        _audit(shard, summary)
        if columns is None:
            require(all(k.split("_", 1)[0] in tables for k in shard), "invalid data column table")
            if len(records) == 1:
                # A monolithic capture already owns the final arrays; no copy.
                columns = shard
                offsets = {table: len(shard[table + "_slot"]) for table in tables}
                del shard
                break
            columns = {k: np.empty((totals[k.split("_", 1)[0]],) + v.shape[1:], dtype=v.dtype)
                       for k, v in shard.items()}
        require(set(shard) == set(columns), "data shard schemas differ")
        for name in columns:
            values, target = shard[name], columns[name]
            table = name.split("_", 1)[0]
            require(values.dtype == target.dtype and values.shape[1:] == target.shape[1:],
                    "data shard schemas differ")
            start = offsets[table]
            target[start:start + len(values)] = values
        for table in tables:
            offsets[table] += len(shard[table + "_slot"])
        del shard, values, target
    require(offsets == totals, "data shard row census mismatch")
    summary = joint.summarize(columns, sum(s["chunks"] for s in summaries))
    _audit(columns, summary)  # Also reject duplicates across shards.
    require(provenance["counts"]["events"] == summary["events"], "incomplete data exposure")
    if "event_key_count" in provenance:
        require(provenance["event_key_count"] == summary["events"] and provenance["duplicate_event_keys"] == 0,
                "data event identity census mismatch")
    return columns, summary, provenance


def stats(values):
    values = np.asarray(values, float).ravel()
    finite = values[np.isfinite(values)]
    result = dict(rows=len(values), count=len(finite), nonfinite=int(len(values) - len(finite)))
    if len(finite):
        mean = np.mean(finite)
        result.update(mean=float(mean), rms=float(np.sqrt(np.mean((finite - mean) ** 2))),
                      second_moment=float(np.mean(finite ** 2)), median=float(np.median(finite)),
                      min=float(np.min(finite)), max=float(np.max(finite)))
    else:
        result.update(dict.fromkeys(("mean", "rms", "second_moment", "median", "min", "max")))
    return result


def _bins(values, edges):
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = len(edges) - 2
    return index, np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])


def _flow(values, edges, mask):
    values = values[mask]
    return dict(rows=len(values), finite=int(np.isfinite(values).sum()),
                nonfinite=int((~np.isfinite(values)).sum()), underflow=int((values < edges[0]).sum()),
                overflow=int((values > edges[-1]).sum()), inrange=int(_bins(values, edges)[1].sum()))


def grouped_stats(ids, values, size):
    """Linear reductions plus one lexicographic sort for exact unbinned medians."""
    rows = np.bincount(ids, minlength=size)
    finite = np.isfinite(values)
    ids, values = ids[finite], values[finite]
    n = np.bincount(ids, minlength=size)
    sums = np.bincount(ids, weights=values, minlength=size)
    second = np.bincount(ids, weights=values ** 2, minlength=size) / np.maximum(n, 1)
    mean = sums / np.maximum(n, 1)
    variance = np.bincount(ids, weights=(values - mean[ids]) ** 2, minlength=size) / np.maximum(n, 1)
    med = np.full(size, np.nan)
    populated = np.flatnonzero(n)
    if len(values):
        ordered = values[np.lexsort((values, ids))]
        starts = np.cumsum(n) - n
        med[populated] = .5 * (ordered[(starts + (n - 1) // 2)[populated]] +
                               ordered[(starts + n // 2)[populated]])
    return [dict(rows=int(rows[i]), count=int(n[i]), nonfinite=int(rows[i] - n[i]),
                 mean=float(mean[i]) if n[i] else None, rms=float(np.sqrt(variance[i])) if n[i] else None,
                 second_moment=float(second[i]) if n[i] else None,
                 median=float(med[i]) if n[i] else None) for i in range(size)]


def count_distribution(values):
    """Exact integer frequencies (no bin centers or pulse weighting inferred)."""
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind in "iu" and np.all(values >= 0),
            "invalid timing group count")
    support, counts = np.unique(values, return_counts=True)
    return dict(rows=len(values), values=support.tolist(), counts=counts.tolist())


def _cell_multiplicity(columns, keep, cell, cells):
    # Full int64 IDs, including run and subrun: never float IDs or local entry IDs.
    names = ("cell", "run", "subrun", "event", "slot")
    keys = np.empty(int(keep.sum()), dtype=[(name, "<i8") for name in names])
    keys["cell"] = cell[keep]
    for name in names[1:]:
        keys[name] = columns["hit_" + name][keep]
    unique, inverse = np.unique(keys, return_inverse=True)
    finite = np.isfinite(columns["hit_time_residual"][keep])
    finite_groups = np.bincount(inverse, weights=finite, minlength=len(unique)) > 0
    del keys, inverse
    size = len(cells)
    counts = {}
    for label, groups in (("", unique), ("finite_", unique[finite_groups])):
        counts[label + "distinct_event_slots"] = np.bincount(groups["cell"], minlength=size)
        events = np.unique(groups[["cell", "run", "subrun", "event"]])
        counts[label + "distinct_events"] = np.bincount(events["cell"], minlength=size)
    del unique, finite_groups, groups, events, finite
    group_counts = columns["hit_group_count"][keep]
    require(group_counts.ndim == 1 and group_counts.dtype.kind in "iu" and np.all(group_counts >= 0) and
            np.all(group_counts <= np.iinfo(np.int64).max),
            "invalid timing group count")
    pairs = np.empty(len(group_counts), dtype=[("cell", "<i8"), ("group_count", "<i8")])
    pairs["cell"], pairs["group_count"] = cell[keep], group_counts
    support, frequency = np.unique(pairs, return_counts=True)
    starts = np.searchsorted(support["cell"], np.arange(size + 1))
    for i, result in enumerate(cells):  # Fixed residual cells, not events/pulses.
        result.update({name: int(values[i]) for name, values in counts.items()})
        lo, hi = starts[i:i + 2]
        result["hit_group_count"] = dict(rows=result["rows"], values=support["group_count"][lo:hi].tolist(),
                                         counts=frequency[lo:hi].tolist())


def residual_cells(columns):
    body = columns["hit_body"]
    eligible = columns["hit_both_lines"] & columns["hit_spread_member"] & columns["hit_spread_eligible"]
    keep = eligible & (body >= 0)
    indices, flows = [], {}
    for name, edges in (("along", ALONG), ("path", PATH), ("pe", PE)):
        index, valid = _bins(columns["hit_" + name], edges)
        indices.append(index)
        keep &= valid
        flows[name] = _flow(columns["hit_" + name], edges, eligible)
    cell = ((body * 5 + indices[0]) * 4 + indices[1]) * 5 + indices[2]
    del indices, index, valid
    values = columns["hit_time_residual"]
    cells = grouped_stats(cell[keep], values[keep], 16 * 5 * 4 * 5)
    _cell_multiplicity(columns, keep, cell, cells)
    return dict(eligible_rows=int(eligible.sum()), eligible_unknown_body=int((eligible & (body < 0)).sum()),
                coordinate_inrange_rows=int(keep.sum()), coordinate_excluded_rows=int((eligible & ~keep).sum()),
                flows=flows, residual_flow=_flow(values, np.array([-20., 20.]), eligible),
                marginal=stats(values[eligible]), inrange_marginal=stats(values[keep]), cells=cells)


def measured(columns):
    out = {}
    for body in range(-1, 16):  # Fixed, bounded body groups; never loop over events/pulses.
        name = G.BODY_NAMES[body] if body >= 0 else "unavailable"
        e = columns["event_body"] == body
        q = (columns["candidate_body"] == body) & columns["candidate_selected"]
        h = columns["hit_body"] == body
        layer = columns["candidate_pe_layer"][q]
        out[name] = dict(event_group_count=count_distribution(columns["event_group_count"][e]),
                         selected_event_group_count=count_distribution(columns["event_group_count"][e & columns["event_selected"]]),
                         valid_event_group_count=count_distribution(columns["event_group_count"][e & columns["event_selected"] &
                                                                                                  columns["event_spread_eligible"]]),
                         event_selected_rms=stats(columns["event_group_rms"][e & columns["event_selected"]]),
                         event_valid_rms=stats(columns["event_group_rms"][e & columns["event_selected"] &
                                                                         columns["event_spread_eligible"]]),
                         selected_layer_pe=[stats(layer[:, i]) for i in range(4)],
                         positive_selected_layer_pe=[stats(layer[:, i][layer[:, i] > 0]) for i in range(4)],
                         hit_pe=stats(columns["hit_pe"][h]), hit_pe_path=stats(columns["hit_pe_path"][h]))
    # Conditional PE/path vs measured along and path, preserving readout side.
    conditional = {}
    for variable, edges, flag in (("along", ALONG, "pe_along_eligible"), ("path", PATH, "pe_path_eligible")):
        index, valid = _bins(columns["hit_" + variable], edges)
        keep = columns["hit_" + flag] & valid & (columns["hit_body"] >= 0)
        side = columns["hit_side"]
        require(np.all((side == 0) | (side == 1)), "invalid readout side")
        size = 16 * 2 * (len(edges) - 1)
        ids = (columns["hit_body"] * 2 + side) * (len(edges) - 1) + index
        conditional[variable] = dict(edges=edges.tolist(), shape=[16, 2, len(edges) - 1],
            flow=_flow(columns["hit_" + variable], edges, columns["hit_" + flag]),
            pe=grouped_stats(ids[keep], columns["hit_pe"][keep], size),
            pe_path=grouped_stats(ids[keep], columns["hit_pe_path"][keep], size))
    return dict(by_body=out, conditional_pe=conditional)


def cohorts(points):
    aligned = {arm: {key: p["columns"]["event_" + key][p["order"]] for key in
                     ("slot", "body", "selected", "spread_eligible", "group_rms")} for arm, p in points.items()}
    base = aligned["nominal"]
    all_selected = np.logical_and.reduce([a["selected"] for a in aligned.values()])
    all_valid = all_selected & np.logical_and.reduce([a["spread_eligible"] & np.isfinite(a["group_rms"])
                                                    for a in aligned.values()])
    masks = {"all": np.ones(len(base["slot"]), bool)}
    masks.update({name: base["slot"] == i for i, name in enumerate(G.SLOT_NAME)})
    masks.update({"body_" + name: base["body"] == i for i, name in enumerate(G.BODY_NAMES)})
    masks["body_unavailable"] = base["body"] < 0
    report = {}
    for label, mask in masks.items():
        groups = {}
        for arm, sample in aligned.items():
            selected = mask & base["selected"] & sample["selected"]
            finite = selected & np.isfinite(base["group_rms"]) & np.isfinite(sample["group_rms"])
            valid = finite & base["spread_eligible"] & sample["spread_eligible"]
            def shift(keep):
                return dict(rows=int(keep.sum()), nominal=stats(base["group_rms"][keep]),
                            arm=stats(sample["group_rms"][keep]),
                            paired_difference=stats(sample["group_rms"][keep] - base["group_rms"][keep]))
            groups[arm] = dict(selected=int((mask & sample["selected"]).sum()),
                gained=int((mask & ~base["selected"] & sample["selected"]).sum()),
                lost=int((mask & base["selected"] & ~sample["selected"]).sum()),
                common_selected=int(selected.sum()), common_selected_finite=shift(finite),
                common_valid=shift(valid), body_changed=int((selected & (base["body"] != sample["body"])).sum()),
                all_arm_common_valid=stats(sample["group_rms"][mask & all_valid]))
        report[label] = dict(exposure_rows=int(mask.sum()), all_arm_common_selected=int((mask & all_selected).sum()),
                             all_arm_common_valid=int((mask & all_valid).sum()), arms=groups)
    return dict(identity="run/subrun/event/slot; sorted exact integer join", duplicate_identities=0,
                equal_exposure_identities=True, body_policy="nominal representative body anchors paired cohorts",
                groups=report)


def support_report(samples):
    counts = {name: np.array([c["count"] for c in sample["cells"]]) for name, sample in samples.items()}
    support = np.logical_and.reduce([n >= MIN_SUPPORT for n in counts.values()])
    weights = counts["data"][support].astype(float) if "data" in counts else None
    if weights is not None and weights.sum():
        weights /= weights.sum()
    out = {}
    for name, sample in samples.items():
        n = counts[name]
        moments = np.array([c["second_moment"] if c["count"] else 0. for c in sample["cells"]])
        total, retained = int(n.sum()), int(n[support].sum())
        marginal = float(np.dot(n, moments) / total) if total else None
        matched = float(np.dot(n[support], moments[support]) / retained) if retained else None
        standardized = float(np.dot(weights, moments[support])) if weights is not None and retained else None
        out[name] = dict(inrange_finite_rows=total, common_support_finite_rows=retained,
                         coverage=retained / total if total else None, inrange_second_moment=marginal,
                         support_matched_second_moment=matched, data_weighted_second_moment=standardized)
    baseline = out["nominal"]
    for name, result in out.items():
        for field in ("support_matched_second_moment", "data_weighted_second_moment"):
            value, base = result[field], baseline[field]
            result[field + "_minus_nominal"] = value - base if value is not None and base is not None else None
        data_moment = out.get("data", {}).get("data_weighted_second_moment")
        value = result["data_weighted_second_moment"]
        result["data_weighted_minus_data"] = value - data_moment if value is not None and data_moment is not None else None
    return dict(min_finite_rows_per_cell=MIN_SUPPORT, common_cell_count=int(support.sum()),
                common_cell_indices=np.flatnonzero(support).tolist(),
                data_cell_weights=weights.tolist() if weights is not None else None,
                status="data_weighted" if weights is not None and support.any() else
                       "no_common_support" if weights is not None else "no_data_standardization", samples=out)


def compare(directory, out, data=None):
    require(not Path(out).exists(), "output already exists")
    receipts = {}
    for module in (Path(__file__), Path(joint.__file__), Path(V.__file__), Path(RC.__file__), Path(RC.DEFAULT_PATH), Path(G.__file__)):
        _receipt(module, receipts)
    points, gates = load_block(directory, receipts)
    nominal = points["nominal"]["comparison"]
    cohort_report = cohorts(points)
    arms, residuals = {}, {}
    for arm, point in points.items():
        score = point["comparison"]
        arms[arm] = dict(objective=score["objective"], objective_minus_nominal=score["objective"] - nominal["objective"],
            terms={name: dict(value=term["chi2ndf"], nominal=nominal["terms"][name]["chi2ndf"],
                             minus_nominal=term["chi2ndf"] - nominal["terms"][name]["chi2ndf"])
                   for name, term in sorted(score["terms"].items())},
            counts=point["counts"], provenance=point["provenance"], capture_summary=point["summary"],
            measured=measured(point["columns"]))
        residuals[arm] = residual_cells(point["columns"])
        for name in ("columns", "keys", "order"):
            del point[name]
    measured_data = None
    if data is not None:
        columns, summary, provenance = load_data(data, receipts)
        measured_data = dict(sample_role="independent measured population; not event matched to MC",
                             provenance=provenance, capture_summary=summary, measured=measured(columns))
        residuals["data"] = residual_cells(columns)
        del columns
    standardized = support_report(residuals)
    standardized["by_body"] = {
        body: support_report({name: dict(cells=sample["cells"][i * 100:(i + 1) * 100])
                              for name, sample in residuals.items()})
        for i, body in enumerate(G.BODY_NAMES)}
    report = dict(schema=1, comparison="joint-compare", passed=True, arms=arms, data=measured_data,
        coordinates=dict(atten_T1=ATTEN, yield_nominal=YIELD_NOMINAL, yield_low=YIELD_LOW, yield_fraction=.7),
        replay_gates=gates, cohorts=cohort_report,
        residual_diagnostics=dict(axes=dict(body=list(G.BODY_NAMES), along=ALONG.tolist(), path=PATH.tolist(), pe=PE.tolist()),
            shape=[16, 5, 4, 5], cell_order="C order: body, along, path, PE; final upper edges inclusive",
            population="selected first pulses with both measured lines, spread_member and spread_eligible; finite residuals for moments",
            samples=residuals, standardization=standardized),
        definitions=dict(rms="centered population RMS (standard deviation), not RMS about zero",
                         second_moment="mean(time_residual**2); units ns**2 for residuals",
                         units="time/RMS ns; PE photoelectrons; along and path dimensionless",
                         distributions="exact unbinned moments/medians, not raw-row duplication or bin-center estimates",
                         multiplicity="cell distinct_events use run/subrun/event; distinct_event_slots also include slot; "
                                      "finite_ counts require a finite residual; hit_group_count is hit-weighted over "
                                      "coordinate-inrange rows, including nonfinite residuals",
                         event_group_count="event-weighted timing event/slot rows by representative body; all, selected, "
                                           "and selected spread-eligible populations; not separately fitted body groups"),
        cautions=["Descriptive diagnostics only; no causal or significance claim, refit, rescore or acceptance decision.",
                  "Common seeds do not imply the same photon draws after response-dependent stochastic branches.",
                  "Pulses are never matched across arms, including by ordinal; only event/slot cohorts are paired.",
                  "Timing groups are event/slot; body labels are representative, not separately fitted body groups.",
                  "Cell membership and selection may change with response. Compare support-matched and data-weighted moments, not unrestricted marginals.",
                  "Input/runtime/code identities use full recorded provenance; original ROOT inputs and runtime assets are not reopened.",
                  "Data, if supplied, are independent, unshifted measured data with exact saved per-file reference equality; not event-aligned to MC.",
                  "MC numeric tables are released before data loading; data use preallocated final arrays plus one raw shard, with full identity audits and exact reductions."])
    for module in (Path(__file__), Path(joint.__file__), Path(V.__file__), Path(RC.__file__), Path(RC.DEFAULT_PATH), Path(G.__file__)):
        _receipt(module, receipts)
    # Hash-before/read/hash-after every saved artifact, including gates and references.
    for record in receipts.values():
        require(V.digest(record["path"]) == record["sha256"], f"file changed during comparison: {record['path']}")
    report["files"] = list(receipts.values())
    return V._report(out, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--data", help="optional unshifted measured joint_data capture; full-file exact saved references required")
    parser.add_argument("--out", required=True, help="new JSON file, never overwritten")
    args = parser.parse_args(argv)
    try:
        compare(args.directory, args.out, args.data)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"joint-compare: PASS; descriptive report {args.out}")


if __name__ == "__main__":
    main()
