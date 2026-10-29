"""Bounded finder-only diagnostic on already-saved stock-ADC refits; no new fitting."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import refit_pulse_bridge as B
from . import boundary_pulse_inventory as L
from . import pilot_membership_report as V

I, M, P, F, A, SR = L.I, L.M, L.P, L.F, L.A, L.SR
require = SR._require
SCHEMA = "saved-refit-membership-v1"
CAP = 32 * 1024**2
REGISTRY_SHA = "6f08a5f6eefdebfb389249c51b145e4c197bca3516c3e8d9d6a08dc598078a31"
TARGETS = {0: [2505], 1: [281, 2755]}
INPUTS = {0: "5cfee976c73b5890ff6d4b83cfc258f81b616c661965e8f62363ebc322dc16f7",
          1: "ce3e20570319c1e25026c35feca9213ad1132c0da9ad1f0313d09f071bbdc8d7"}
SCOPE = ("Three selected complete frames of retained stock-ADC refits, not identical stock pulse values. "
         "Finder-only integration; no response, refit, stock, raw input, coverage or estimator execution. "
         "Native links refer to this refit collection; no stock coincidence comparison is performed.")


class Evidence:
    """Hash-bound consumed files and completed transactions, checked again before dispatch and publication."""

    def __init__(self):
        self.bindings = {}
        self.transactions = []

    def bound(self, path, digest):
        path = Path(path)
        require(SR._regular(path) <= I.FILE_CAP, "refit input file cap exceeded")
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest, "conflicting refit binding")
        M.bind(path, digest)
        self.bindings[str(path)] = digest
        return path

    def json(self, path, digest):
        return M.read_json(self.bound(path, digest))

    def table(self, root, report, name, rows):
        require(Path(name).name == name and name in report["output_sha256"], "missing refit payload binding")
        return I.table(self.bound(Path(root) / name, report["output_sha256"][name]), rows)

    def batch(self, root, context, names, cap):
        receipt, _ = SR._batch(root, context, names, cap)
        self.bindings.update(M.transaction_bindings(root, receipt))
        self.transactions.append((root, context, names, cap, receipt))
        return receipt

    def recheck(self):
        M.check_bindings(self.bindings)
        for root, context, names, cap, original in self.transactions:
            receipt, _ = SR._batch(root, context, names, cap)
            require(receipt == original, "refit inherited transaction changed late")


def authority(evidence, membership, verified, expected_membership_sha256, expected_membership_verified_sha256):
    """Reuse completed registry/calibration provenance without repeating upstream verification."""
    s = evidence.json(membership / "summary.json", expected_membership_sha256)
    v = evidence.json(verified, expected_membership_verified_sha256)
    require(s["schema"] == M.SCHEMA and set(s) == V.SUMMARY_KEYS, "invalid inherited membership summary")
    require(v["schema"] == V.SCHEMA and v["summary_sha256"] == expected_membership_sha256,
            "membership verification belongs to another output")
    for flag in ("saved_artifact_protocol_exact", "qualified_frame_census_exact", "source_transactions_exact",
                 "canonical_pulses_exact", "native_member_structure_exact"):
        require(v[flag] is True, "missing inherited membership gate: " + flag)
    require(s["registry_sha256"] == v["registry_sha256"] == REGISTRY_SHA and v["registry_frames"] == 9000,
            "unexpected inherited full registry")
    require(v["code_sha256"] == M.S.C.sha256(V.__file__), "inherited verifier source changed")
    context = dict(registry_sha256=s["registry_sha256"], batch_identity={"contract": M.SCHEMA, **s["hashes"]},
                   arms=list(M.S.ARMS), seeds=list(M.S.SEEDS))
    names = ["summary.json", "registry.parquet", "selected.parquet", "grid.parquet", "pulses.parquet",
             "clusters.parquet", "members.parquet", "coverage.parquet"]
    names += [f"code{i}.txt" for i in range(len(s["code_sha256"]))]
    receipt = evidence.batch(membership, context, names, M.OUTPUT_CAP)
    require(receipt["artifacts"]["summary.json"]["sha256"] == expected_membership_sha256, "late membership summary change")
    require({receipt["artifacts"][f"code{i}.txt"]["sha256"] for i in range(len(s["code_sha256"]))}
            == set(s["code_sha256"].values()), "inherited code snapshots differ")
    for path, digest in s["code_sha256"].items():
        evidence.bound(path, digest)
    registry = I.table(membership / "registry.parquet", 9000)
    require(len(registry) == 9000 and F.digest(registry) == REGISTRY_SHA, "complete registry changed")
    def unique(suffix):
        matches = [(path, sha) for path, sha in s["inputs"].items() if path.endswith(suffix)]
        require(len(matches) == 1, "missing unique authority: " + suffix)
        return evidence.bound(*matches[0])
    calibration = I.table(unique("/calibration.parquet"), 15360)
    require(len(calibration) == 15360 and set(calibration.input_sha256) == set(registry.input_sha256),
            "complete per-input calibration domain differs")
    counters = unique("/crv_counters_run1a_v01.txt")
    require(SR._regular(counters) <= 2 * 1024**2, "geometry text cap exceeded")
    geom = M.geometry_only({"counters": counters})
    require(len(geom.sector) == 1280, "Run1A bar domain differs")
    for digest in registry.input_sha256.unique():
        P._status(calibration, digest, 5120)
    for path, digest in M.ignore_bindings().items():
        evidence.bound(path, digest)
    return registry, calibration, geom


def admission(evidence, downstream, boundary, verified, registry, calibration, geom, *,
              expected_downstream_sha256, expected_boundary_sha256, expected_verified_sha256):
    """Admit both complete retained refit populations before any finder dispatch."""
    d = evidence.json(downstream / "summary.json", expected_downstream_sha256)
    b = evidence.json(boundary / "summary.json", expected_boundary_sha256)
    v = evidence.json(verified, expected_verified_sha256)
    require(d["schema"] == "thread-d-saved-photon-downstream-v1" and v["schema"] == "thread-d-downstream-report-v1",
            "invalid retained downstream report")
    require(d["input_summary_sha256"] == v["boundary_summary_sha256"] == expected_boundary_sha256
            and v["input_summary_sha256"] == expected_downstream_sha256 and v["original_inputs_verified"] is True,
            "retained downstream provenance differs")
    require(len(d["blocks"]) == len(b["blocks"]) == 2
            and [x["block"] for x in d["blocks"]] == [0, 1] == [x["block"] for x in b["blocks"]],
            "retained block schedule differs")
    blocks = []
    for number in (0, 1):
        old, prior = d["blocks"][number], b["blocks"][number]
        require(old["targets"] == prior["targets"] == TARGETS[number], "retained target schedule differs")
        shas = [sha for path, sha in prior["input_sha256"].items() if path.endswith("/mixed.root")]
        require(shas == [INPUTS[number]], "retained input identity differs")
        def read(root, report, suffix, rows):
            return evidence.table(root, report, f"block{number}_{suffix}.parquet", rows)
        events = read(boundary, b, "events", 3)
        require(sorted(events.event) == TARGETS[number], "retained event census differs")
        stock = read(downstream, d, "stock_pulses", 2048)
        conditions = read(downstream, d, "conditions", 20480)
        p, w, comparison = (read(downstream, d, "stock_adc_refit_" + suffix, 2048)
                             for suffix in ("pulses", "waveforms", "comparison"))
        prepared = B.prepare(p, w, conditions, events, stock, comparison, registry, calibration, geom, shas[0])
        blocks.append(dict(block=number, **prepared))
    require(sum(len(x["selected"]) for x in blocks) == 3 and sum(len(x["pulses"]) for x in blocks) == 87
            and sum(len(x["waveforms"]) for x in blocks) == 80, "retained aggregate census differs")
    require([len(x["pulses"]) for x in blocks] == [25, 62], "retained block pulse census differs")
    return blocks


def dispatch(blocks, registry, calibration, geom):
    """Call the frozen finder once per admitted input, preserving all saved pulse flags and indices."""
    require(len(blocks) == 2, "requires two admitted refit blocks")
    tables, records = {}, []
    keys = [*P.FRAME_KEYS, *P.PULSE_KEYS]
    expected_maps = []
    for block in blocks:
        expected = block["pulses"][keys + ["refit_idx"]].rename(columns={"refit_idx": "idx"}).copy()
        for name in keys[1:] + ["idx"]:
            expected[name] = P._integer(expected[name], "admitted native " + name)
        expected_maps.append(expected.reset_index(drop=True))
    for block, expected in zip(blocks, expected_maps):
        number = block["block"]
        products = P.run(block["pulses"], registry, block["selected"], calibration, geom, registry_digest=REGISTRY_SHA)
        saved = block["pulses"]
        native = products["pulses"]
        require(len(native) == len(saved), "finder pulse census changed")
        require(native.idx.dtype == np.dtype("int64")
                and all(np.array_equal(native[n], expected[n]) for n in keys + ["idx"]),
                "finder changed admitted native index bridge")
        flags = saved[keys + ["failed", "zerondf"]]
        products["pulses"] = native.merge(flags, on=keys, how="left", validate="one_to_one", sort=False)
        for name, table in products.items():
            tables[f"block{number}_{name}"] = table
        for name in ("bridge", "waveforms", "pulses"):
            tables[f"block{number}_saved_{name}"] = block[name]
        records.append(dict(block=number, diagnostics=block["diagnostics"],
                            frames=products["frames"].to_dict("records"),
                            counts={name: len(t) for name, t in products.items()}))
    return tables, records


def run(downstream, boundary, verified, membership, membership_verified, out, *, expected_downstream_sha256,
        expected_boundary_sha256, expected_verified_sha256, expected_membership_sha256,
        expected_membership_verified_sha256):
    downstream, boundary, verified, membership, membership_verified, out = map(
        Path, (downstream, boundary, verified, membership, membership_verified, out))
    M._outside(out, [downstream, boundary, verified.parent, membership, membership_verified.parent])
    begin = time.perf_counter()
    evidence = Evidence()
    code_paths = list(dict.fromkeys([Path(__file__), Path(B.__file__), Path(L.__file__), Path(I.__file__),
                                    Path(V.__file__), *M.code_paths()]))
    code = {str(path): M.S.C.sha256(path) for path in code_paths}
    registry, calibration, geom = authority(evidence, membership, membership_verified, expected_membership_sha256,
                                          expected_membership_verified_sha256)
    blocks = admission(evidence, downstream, boundary, verified, registry, calibration, geom,
                       expected_downstream_sha256=expected_downstream_sha256,
                       expected_boundary_sha256=expected_boundary_sha256, expected_verified_sha256=expected_verified_sha256)
    M._outside(out, [Path(path).parent for path in evidence.bindings])
    context = dict(registry_sha256=REGISTRY_SHA, batch_identity={"contract": SCHEMA,
                   "downstream_sha256": expected_downstream_sha256, "membership_sha256": expected_membership_sha256,
                   "seed_role": "protocol-only label; deterministic saved ADC refit; no random draw"},
                   arms=["saved_adc_refit"], seeds=[0])
    names = ["summary.json", "registry.parquet", "calibration.parquet"]
    names += [f"block{i}_{name}.parquet" for i in (0, 1) for name in
              ("frames", "pulses", "clusters", "members", "saved_bridge", "saved_waveforms", "saved_pulses")]
    names += [f"code{i}.txt" for i in range(len(code))]
    evidence.recheck()
    M.check_bindings(code)
    with A.BatchArtifacts(out, max_bytes=CAP, context=context, expected_artifacts=names) as tx:
        tx.write_parquet("registry.parquet", registry)
        tx.write_parquet("calibration.parquet", calibration)
        for i, path in enumerate(code):
            snapshot = Path(path).read_bytes()
            require(hashlib.sha256(snapshot).hexdigest() == code[path], "refit source snapshot hash differs")
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(snapshot)
        evidence.recheck()
        M.check_bindings(code)
        tables, records = dispatch(blocks, registry, calibration, geom)
        for name, table in tables.items():
            tx.write_parquet(name + ".parquet", table)
        result = dict(schema=SCHEMA, scope=SCOPE, inputs=evidence.bindings, code_sha256=code,
                      registry_sha256=REGISTRY_SHA, registry_frames=len(registry), unique_frames=3,
                      saved_pulses=87, saved_waveforms=80, finder_calls=2, records=records,
                      finder_config=asdict(M.C.finder_config()), coincidence_limits=asdict(P.Limits()),
                      nonempty_real_refit_membership=any(x["counts"]["members"] > 0 for x in records),
                      full_registry_admitted=True, stock_refit_pulse_bridge_exact=True,
                      stock_native_membership_compared=False, same_pulse_stock_validation=False,
                      regenerated_response=False, regenerated_fit=False, stock_rerun=False,
                      inherited_raw_admission_repeated=False, prior_verifier_rerun=False,
                      coverage_computed=False, execution_admitted=False, full_retained_campaign_admitted=False,
                      timing_closure=False, F=None, finite_library_uncertainty=None,
                      output_cap_bytes=CAP, seconds=time.perf_counter() - begin, memory=M.S.C.memory())
        evidence.recheck()
        M.check_bindings(code)
        tx.write_json("summary.json", result)
    _, size = SR._batch(out, context, names, CAP)
    return result | {"persisted_path_bytes": size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("downstream", "boundary", "verified", "membership", "membership_verified", "out"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in ("downstream", "boundary", "verified", "membership", "membership_verified"):
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "records", "nonempty_real_refit_membership", "seconds",
                                           "memory", "persisted_path_bytes", "scope")}, indent=2))


if __name__ == "__main__":
    main()
