"""Verify hash-bound saved response evidence without regenerating optics or downstream products."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import stat

import numpy as np
import pandas as pd

from . import saved_qualified_response as V
from . import bounded_response as R, batch_artifacts as A, frame_registry as F
from .replay import sha256

SCHEMA = "saved-qualified-response-report-v1"
SCOPE = ("Observed, input-hash-bound saved diagnostic response and exact physical-field partition comparisons. "
         "Source/census/conditions and immutable phases are checked; optics, avalanches, waveform synthesis and fits "
         "are not regenerated. No stock equivalence, coincidence, F, timing closure, exposure or uncertainty certification.")


def _require(test, message):
    if not test:
        raise ValueError(message)


def _equal(actual, expected, message):
    _require(V.json_bytes(actual) == V.json_bytes(expected), message)


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def _regular(path):
    _require(stat.S_ISREG(path.lstat().st_mode), "saved artifact is not a regular file")
    return path.stat().st_size


def _batch(directory, context, expected, cap):
    """Validate retained transaction markers and count every persisted pathname, including receipt copies."""
    _require(stat.S_ISDIR(directory.lstat().st_mode), "saved batch is not a directory")
    _require(set(p.name for p in directory.iterdir()) == {*expected, A.INCOMPLETE, A.COMPLETE, A.PENDING},
             "failed, missing or undeclared batch artifact")
    sizes = {p.name: _regular(p) for p in directory.iterdir()}
    _require(sum(sizes.values()) <= cap, "persisted byte cap exceeded")
    complete, initial = _json(directory / A.COMPLETE), _json(directory / A.INCOMPLETE)
    _, digest = A._provenance(context)
    common = dict(schema=A.SCHEMA, context_sha256=digest, execution_admitted=False,
                  physics_truncated=False, denominator_renormalized=False)
    reserve = len(A._canonical(dict(common, state="failed")))
    wanted_initial = dict(common, state="incomplete", context=context, max_bytes=cap,
                          expected_artifacts=sorted(expected), failure_reserve_bytes=reserve)
    _equal(initial, wanted_initial, "incomplete context, expected artifacts or cap changed")
    artifacts = complete["artifacts"]
    _require(set(artifacts) == set(expected), "completion artifact domain changed")
    for name in expected:
        _equal(artifacts[name], {"bytes": sizes[name], "sha256": sha256(directory / name)},
               "artifact hash or byte count changed")
    wanted_complete = dict(common, state="complete", context=context, max_bytes=cap, artifacts=artifacts,
                           artifact_bytes=sum(sizes[name] for name in expected),
                           incomplete_sha256=sha256(directory / A.INCOMPLETE), receipt_staging_retained=True)
    _equal(complete, wanted_complete, "completion receipt changed")
    _require((directory / A.INCOMPLETE).read_bytes() == A._canonical(initial), "noncanonical incomplete receipt")
    _require((directory / A.COMPLETE).read_bytes() == A._canonical(complete), "noncanonical complete receipt")
    _require((directory / A.PENDING).read_bytes() == (directory / A.COMPLETE).read_bytes(), "pending receipt changed")
    total = sum(sizes.values())
    _require(total + reserve <= cap, "persisted byte cap exceeded")
    return complete, total


def _table_equal(actual, expected, keys):
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  expected.sort_values(keys).reset_index(drop=True), check_exact=True)


def _bindings(mapping):
    _require(type(mapping) is dict and bool(mapping), "missing mandatory hash bindings")
    for path, digest in mapping.items():
        V.M.P.bound(path, digest)


def _source(summary, context):
    inputs, code = summary["inputs"], summary["code_sha256"]
    paths = V.code_paths()
    _require(set(code) == {str(p) for p in paths}, "mandatory runner code bindings changed")
    _bindings(code)
    _require(code.get(str(Path(R.W.W.__file__))) == V.FROZEN_WAVEFORM_SHA, "unapproved frozen waveform helper")
    V.M.P.bound(R.W.W.__file__, V.FROZEN_WAVEFORM_SHA)
    _bindings(inputs)
    _require(all(inputs[path] == code[path] for path in inputs.keys() & code.keys()), "conflicting input/code hash binding")
    roots = []
    for label in ("plan", "smoke"):
        digest = context["batch_identity"][label + "_sha256"]
        matches = [Path(path) for path, value in inputs.items() if value == digest and Path(path).name == "summary.json"]
        _require(len(matches) == 1, "missing unique source summary binding")
        roots.append(matches[0].parent)
    admitted = V.admission(*roots, context["batch_identity"]["plan_sha256"], context["batch_identity"]["smoke_sha256"])
    mandatory = admitted[4]
    _require(all(inputs.get(path) == digest for path, digest in mandatory.items()), "mandatory source input binding missing")
    template = summary["waveform_config"]["single_pe_file"]
    _require(inputs.get(template) == R.D.TEMPLATE_SHA, "waveform template is not hash-bound")
    maps = [path for path, digest in inputs.items() if digest == R.D.MAP_SHA]
    _require(bool(maps), "photon map is not hash-bound")
    return admitted, paths


def _wave_receipt(products, wcfg, limits):
    clusters = products["clusters"]
    _require(int(clusters.charges.sum()) == len(products["charges"]), "saved cluster charge count changed")
    reach = int(np.ceil(wcfg.single_pe_max_time / 12.5)) + 2
    start = np.ceil((clusters.first_charge.to_numpy() - 12.5 - clusters.phase.to_numpy()) / 12.5) * 12.5 + clusters.phase.to_numpy()
    estimate = np.floor((clusters.last_charge.to_numpy() - clusters.first_charge.to_numpy() + wcfg.single_pe_max_time) / 12.5)
    jmax = np.maximum(np.ceil((clusters.last_charge.to_numpy() - start) / 12.5), 0)
    upper = np.maximum(estimate, jmax + reach)
    length = int(upper.max(initial=0)) + wcfg.samples_zs if len(clusters) else 0
    _require(length <= limits.max_length and len(clusters) * length <= limits.waveform_cells,
             "saved waveform allocation cap exceeded")
    _require(len(products["charges"]) <= limits.charges and len(products["charges"]) * reach <= limits.reach_cells,
             "saved charge reach cap exceeded")
    _require(len(products["samples"]) <= len(clusters) * length, "saved sample matrix exceeds admission")
    return dict(contract=R.W.CONTRACT, charges=len(products["charges"]), clusters=len(clusters), reach=reach,
                charge_reach_cells=len(products["charges"]) * reach, waveform_length_upper=length,
                waveform_cells_upper=len(clusters) * length, limits=asdict(limits), peak_memory_certified=False)


BOOL_COLUMNS = {"dark", "primary", "inactive", "fired", "failed", "zerondf", "pedfromdb", "channel_ignored",
                "valid", "cluster_prekeep", "above_zs", "prekeep", "near_threshold"}
INT_COLUMNS = {"run", "subrun", "event", "frame_id", "chan", "root", "generation", "px", "py", "multiplicity", "photon",
               "root_px", "root_py", "starttdc", "sample", "adc", "wf", "seq", "cluster", "tdc", "length", "n", "status", "roc", "feb", "label"}


def _schema(products):
    for stage, table in products.items():
        required = R.REQUIRED[stage] | set(F.FRAME_KEYS) | {"frame_id"}
        _require(required <= set(table), "missing required stage fields: " + stage)
        for name in required - {"input_sha256"}:
            values = table[name].to_numpy()
            integer = name in INT_COLUMNS or (stage == "frames" and name in R.STAGES) or (stage == "clusters" and name == "charges")
            kinds = "b" if name in BOOL_COLUMNS else "iu" if integer else "iuf"
            _require(values.dtype.kind in kinds and np.isfinite(values).all(), "invalid numeric stage schema: " + stage + "." + name)
        for name in required & {"p", "uniform"}:
            _require(table[name].between(0., 1.).all(), "probability/uniform outside numeric domain")
        if "multiplicity" in table:
            _require(table.multiplicity.ge(1).all(), "invalid attempt multiplicity")


def _products(products, registry, selected, calibration, q, seed, arm, wcfg):
    """Check physical domains, authoritative conditions, saved lineage and observable admission counts."""
    _schema(products)
    digest, limits = F.digest(registry), R.Limits()
    expected_frames = R.selected_frames(registry, selected, digest)
    _table_equal(products["frames"][list(F.COLUMNS)], expected_frames, ["frame_id"])
    conditions = R.Q.conditions(registry, selected.drop(columns="frame_id"), calibration, digest)
    eq, ec = R.Q.admit(q, conditions, registry, digest, seed)
    authoritative = F.restore(ec, registry)
    _table_equal(products["conditions"], authoritative, ["frame_id", "chan"])
    helper = F.helper_seed(registry, seed)
    phases = F.restore(R.W.W.phase_table(ec, helper).set_index("label").reset_index(), registry)
    _table_equal(products["phases"], phases, ["frame_id", "chan"])
    live = pd.MultiIndex.from_frame(authoritative[["frame_id", "chan"]])
    for name in R.STAGES:
        table = products[name]
        _require(table.frame_id.dtype.kind in "iu" and table.frame_id.isin(selected.frame_id).all(),
                 "stage outside declared frame domain")
        encoded = F.encode(table.drop(columns="frame_id"), registry)
        np.testing.assert_array_equal(encoded.event, table.frame_id)
        for column in table:
            if table[column].dtype.kind in "iufb":
                _require(np.isfinite(table[column]).all(), "nonfinite saved stage value")
        R.canonical(products, name)
        if name == "frames":
            continue
        _require(table.chan.dtype.kind in "iu", "noninteger stage channel")
        _require(pd.MultiIndex.from_frame(table[["frame_id", "chan"]]).isin(live).all(), "stage outside all-live channel domain")
        physical = table[table.valid] if name == "samples" else table
        counts = physical.groupby("frame_id").size()
        np.testing.assert_array_equal(products["frames"][name], products["frames"].frame_id.map(counts).fillna(0).astype(np.int64))
    candidates, attempts, charges = (products[n] for n in ("candidates", "attempts", "charges"))
    for table in (candidates, attempts, charges):
        _require(table.dark.dtype == bool and table.root.dtype.kind in "iu", "invalid candidate root or dark flag")
        _require(((table.root < 0) == table.dark).all(), "optical/thermal root domains changed")
        _require(table[["px", "py"]].ge(0).all().all() and table[["px", "py"]].le(40).all().all(), "pixel domain changed")
    keys = [*F.FRAME_KEYS, "frame_id", "chan", "root"]
    time_key = "time_historical_relative" if arm in ("historical", "rejection_only") else "time_raw_wrapped_relative"
    optical = eq.loc[eq[arm + "_kept"], ["event", "chan", "photon_id", time_key]].rename(columns={"photon_id": "root", time_key: "time"})
    optical = F.restore(optical, registry)
    _table_equal(candidates.loc[~candidates.dark, keys + ["time"]], optical[keys + ["time"]], keys)
    dark = candidates[candidates.dark]
    _require(dark.time.between(100., 1500.).all(), "thermal candidate outside declared time domain")
    _require(candidates[["px", "py"]].le(39).all().all(), "candidate pixel outside fixed-integer domain")
    _require(attempts.primary.dtype == bool and attempts.fired.dtype == bool, "invalid attempt masks")
    _require(attempts.generation.dtype.kind in "iu" and attempts.generation.ge(0).all(), "invalid attempt generation")
    _require(np.array_equal(attempts.primary, attempts.generation == 0), "primary generation changed")
    cols = keys + ["time", "px", "py", "dark"]
    _table_equal(attempts.loc[attempts.primary, cols], candidates.loc[candidates.time <= 1500., cols], keys)
    ancestry = attempts.merge(candidates[keys + ["time", "px", "py", "dark"]], on=keys, how="left",
                              validate="many_to_one", suffixes=("", "_candidate"), indicator=True)
    _require(ancestry._merge.eq("both").all(), "attempt outside saved candidate ancestry")
    for name, root_name in (("time", "root_time"), ("px", "root_px"), ("py", "root_py")):
        np.testing.assert_array_equal(ancestry[root_name], ancestry[name + "_candidate"])
    np.testing.assert_array_equal(ancestry.dark, ancestry.dark_candidate)
    _table_equal(charges, attempts[attempts.fired], keys + ["generation", "px", "py"])
    _require(len(selected) <= limits.frames and len(ec) <= limits.live_channels and len(q) <= limits.photons,
             "saved response input cap exceeded")
    sipm = limits.sipm
    active_channels = candidates[candidates.time <= 1500.][["frame_id", "chan"]].drop_duplicates()
    _require(len(optical) <= sipm.photons and len(dark) <= sipm.thermal_candidates and len(candidates) <= sipm.candidates
             and len(attempts) <= sipm.attempts and len(ec) <= sipm.live_channels
             and len(active_channels) * 41 * 41 * 8 <= sipm.state_bytes
             and (not len(attempts) or int(attempts.generation.max()) < sipm.generations), "saved SiPM cap exceeded")
    waveform_receipt = _wave_receipt(products, wcfg, limits.waveform)
    waveforms, fits, digis = (products[n] for n in ("waveforms", "fit_samples", "digis"))
    _require(len(fits) <= limits.fit.sample_rows and len(digis) <= limits.fit.sample_rows
             and (not len(waveforms) or (waveforms.n.ge(1).all() and waveforms.n.max() <= limits.fit.max_length
                  and len(waveforms) * int(waveforms.n.max()) <= limits.fit.matrix_cells)), "saved fit cap exceeded")
    _require(len(fits) == int(waveforms.n.sum()), "saved fit sample count changed")
    starts = digis[["frame_id", "chan", "starttdc"]].drop_duplicates()
    _require(len(digis) == len(starts) * wcfg.samples_zs and digis["sample"].between(0, wcfg.samples_zs - 1).all(),
             "saved digi sample coverage changed")
    _require(digis.adc.between(wcfg.min_adc, wcfg.max_adc).all(), "saved ADC domain changed")
    return dict(contract=R.CONTRACT, registry_sha256=digest, optical_seed=seed, response_seed=seed, helper_seed=helper,
                arm=arm, frames=len(selected), live_channels=len(ec), candidate_photons=len(q), photons=len(optical),
                dark_candidates=len(dark), attempts=len(attempts), charges=len(charges), digis=len(starts),
                pulses=len(products["pulses"]), limits=asdict(limits), waveform_admission=waveform_receipt,
                fit_contract=R.P.CONTRACT, thermal_domain="all status-live detector channels, including quiet frames",
                execution_admitted=False, pulse_fit_partition_exact_certified=False)


def verify(directory, out):
    """Write one exclusive JSON report after saved-only checks; never overwrite an existing output."""
    directory, out = Path(directory), Path(out)
    _require(not out.exists() and not out.is_symlink(), "choose a new saved-report file")
    root = directory / "run"
    _require(stat.S_ISDIR(directory.lstat().st_mode) and stat.S_ISDIR(root.lstat().st_mode), "invalid response directory")
    _require(_regular(root / "summary.json") <= V.ROOT_CAP and _regular(root / A.COMPLETE) <= V.ROOT_CAP,
             "run metadata byte cap exceeded")
    summary = _json(root / "summary.json")
    context = _json(root / A.COMPLETE)["context"]
    _require(set(context["batch_identity"]) == {"plan_sha256", "smoke_sha256", "contract"}, "unexpected run identity")
    expected_context = dict(registry_sha256=summary["registry_sha256"], batch_identity={
        "plan_sha256": context["batch_identity"]["plan_sha256"], "smoke_sha256": context["batch_identity"]["smoke_sha256"],
        "contract": R.CONTRACT}, arms=list(V.M.P.ARMS), seeds=list(V.SEEDS))
    _equal(context, expected_context, "run context changed")
    tags = [f"seed{seed}_{arm}" for seed in V.SEEDS for arm in V.M.P.ARMS]
    _require(set(p.name for p in directory.iterdir()) == {"run", *tags}, "missing or undeclared response batch")
    names = ["summary.json", "registry.parquet", "selected.parquet", "calibration.parquet", "admission.json"]
    names += [tag + ".json" for tag in tags] + [f"code{i}.txt" for i in range(len(V.code_paths()))]
    top, total = _batch(root, context, names, V.ROOT_CAP)
    for key, expected in dict(schema=V.SCHEMA, scope=V.SCOPE, frames=4, registry_frames=9000, seeds=list(V.SEEDS),
                              arms=list(V.M.P.ARMS), thermal_rate=V.THERMAL_RATE,
                              all_tested_physical_partition_fields_exact=True, execution_admitted=False, F=None,
                              finite_library_uncertainty=None, timing_closure=False).items():
        _equal(summary[key], expected, "saved summary contract changed: " + key)
    caps = dict(run=V.ROOT_CAP, each_arm_seed=V.BATCH_CAP, maximum_total=V.ROOT_CAP + len(tags) * V.BATCH_CAP)
    _equal(summary["byte_caps"], caps, "saved byte budgets changed")
    cfg = replace(V.SI.SiPMConfig(), thermal_rate=V.THERMAL_RATE)
    wcfg = replace(V.WF.WaveformConfig(), single_pe_file=summary["waveform_config"]["single_pe_file"])
    _equal(summary["sipm_config"], asdict(cfg), "SiPM configuration changed")
    _equal(summary["waveform_config"], asdict(wcfg), "waveform configuration changed")
    admitted, paths = _source(summary, context)
    registry, selected, calibration, photons, _, evidence = admitted
    _require(len(registry) == 9000 and len(selected) == 4 and F.digest(registry) == summary["registry_sha256"],
             "source registry or selected domain changed")
    for name, actual, keys in (("registry", registry, ["frame_id"]), ("selected", selected, ["frame_id"]),
                               ("calibration", calibration, ["input_sha256", "chan"])):
        saved = pd.read_parquet(root / (name + ".parquet"))
        if name == "registry":
            F.validate(saved)
            _require(F.digest(saved) == summary["registry_sha256"], "saved registry digest changed")
        if name == "selected":
            R.selected_frames(registry, saved, summary["registry_sha256"])
        _table_equal(saved, actual, keys)
    _equal(_json(root / "admission.json"), evidence, "saved source admission changed")
    for i, path in enumerate(paths):
        _require(sha256(root / f"code{i}.txt") == summary["code_sha256"][str(path)], "code snapshot changed")
    _require(set(photons) == set(V.SEEDS) and sum(map(len, photons.values())) <= V.MAX_PHOTONS_UNION, "saved photon union changed")
    union = pd.concat([q[list(V.M.K.KEYS) + ["photon_id"]] for q in photons.values()], ignore_index=True)
    np.testing.assert_array_equal(union.photon_id, V.M.roots(union))
    _require(len(summary["records"]) == len(tags), "wrong arm/seed record count")
    comparisons, records, stage_counts, frame_counts, operation_comparisons = {}, [], {}, {}, {}
    reference, rejection = None, None
    domains = {"full": selected, "part0": selected.iloc[::2], "part1": selected.iloc[1::2]}
    for record, tag, (seed, arm) in zip(summary["records"], tags, ((s, a) for s in V.SEEDS for a in V.M.P.ARMS)):
        _equal({k: record[k] for k in ("path", "seed", "arm")}, dict(path=tag, seed=seed, arm=arm), "arm/seed schedule changed")
        _require(type(record["elapsed_seconds"]) in (int, float) and np.isfinite(record["elapsed_seconds"])
                 and record["elapsed_seconds"] >= 0, "invalid measured elapsed time")
        child = directory / tag
        child_context = {**context, "batch_identity": {**context["batch_identity"], "frame_ids": selected.frame_id.tolist(),
                          "variants": list(V.VARIANTS)}, "arms": [arm], "seeds": [seed]}
        expected = [f"{variant}_{stage}.parquet" for variant in V.VARIANTS for stage in R.STAGES] + ["validation.json"]
        receipt, size = _batch(child, child_context, expected, V.BATCH_CAP)
        total += size
        _require(record["receipt_sha256"] == sha256(child / A.COMPLETE), "child receipt hash changed")
        _equal(_json(root / (tag + ".json")), receipt, "copied child receipt changed")
        validation = _json(child / "validation.json")
        _require(set(validation) == {"partition", "subset", "receipts"} and set(validation["receipts"]) == set(V.VARIANTS),
                 "saved validation domain changed")
        products, receipts = {}, {}
        for variant in V.VARIANTS:
            products[variant] = {stage: pd.read_parquet(child / f"{variant}_{stage}.parquet") for stage in R.STAGES}
            q = V._photon_subset(photons[seed], domains[variant])
            receipts[variant] = _products(products[variant], registry, domains[variant], calibration, q, seed, arm, wcfg)
        _equal(validation["receipts"], receipts, "saved response receipt changed")
        _equal(record["counts"], receipts["full"], "summary response counts changed")
        partition = R.compare_partition(products["full"], [products["part0"], products["part1"]])
        subset = R.compare_partition(products["full"], [products["part0"]])
        for key, actual in (("partition", partition), ("subset", subset)):
            _equal(validation[key], actual, "saved partition validation changed")
            _equal(record[key], actual, "summary partition validation changed")
        if arm == "historical":
            reference = {stage: products["full"][stage].copy() for stage in R.D.STAGES}
        else:
            comparisons[f"seed{seed}_historical_vs_{arm}"] = V.structural_comparisons(reference, products["full"], registry)
        if arm == "rejection_only":
            rejection = {stage: products["full"][stage].copy() for stage in R.D.STAGES}
        elif arm == "stock_timing":
            operation_comparisons[f"seed{seed}_rejection_only_vs_stock_timing"] = V.structural_comparisons(rejection, products["full"], registry)
        frame_counts[tag] = products["full"]["frames"].to_dict(orient="records")
        records.append(dict(seed=seed, arm=arm, counts=receipts["full"], partition=partition, subset=subset))
        stage_counts[tag] = {variant: {stage: len(table) for stage, table in product.items()} for variant, product in products.items()}
        del products
    _equal(summary["structural_comparisons"], comparisons, "historical arm structural comparisons changed")
    _require(total <= caps["maximum_total"], "total persisted cap exceeded")
    _bindings(summary["inputs"])
    _bindings(summary["code_sha256"])
    _require(sha256(root / "summary.json") == top["artifacts"]["summary.json"]["sha256"], "summary changed during verification")
    result = dict(schema=SCHEMA, summary_sha256=sha256(root / "summary.json"), registry_sha256=F.digest(registry),
                  registry_frames=len(registry), frames=len(selected), seeds=list(V.SEEDS), arms=list(V.M.P.ARMS),
                  records=records, stage_rows=stage_counts, frame_counts=frame_counts, structural_comparisons=comparisons,
                  operation_comparisons=operation_comparisons, persisted_path_bytes=total,
                  byte_caps=caps, saved_artifact_protocol_exact=True, source_admission_exact=True, root_union_collision_free=True,
                  authoritative_all_live_conditions_exact=True, all_tested_physical_partition_fields_exact=True,
                  observed_input_hash_bound_response=True, regenerated_optics=False, regenerated_response=False,
                  execution_admitted=False, F=None, finite_library_uncertainty=None, timing_closure=False,
                  code_sha256=sha256(__file__), scope=SCOPE)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def compact(result):
    """Small numeric view of already verified saved evidence, not an additional inference."""
    counts = [{k: record["counts"][k] for k in ("optical_seed", "arm", "photons", "dark_candidates", "attempts", "charges", "digis", "pulses")}
              for record in result["records"]]
    comparisons = {label: {stage: {k: value[k] for k in ("rows_a", "rows_b", "only_a", "only_b", "changed_common", "exact", "max_abs_delta_common")}
                          for stage, value in stages.items()}
                   for label, stages in (result["structural_comparisons"] | result["operation_comparisons"]).items()}
    return {"summary_sha256": result["summary_sha256"], "frames": result["frames"], "counts": counts,
            "comparisons": comparisons, "frame_counts": result["frame_counts"], "persisted_path_bytes": result["persisted_path_bytes"],
            "all_tested_physical_partition_fields_exact": result["all_tested_physical_partition_fields_exact"],
            "execution_admitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.directory, args.out)
    print(json.dumps(compact(result) if args.compact else result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
