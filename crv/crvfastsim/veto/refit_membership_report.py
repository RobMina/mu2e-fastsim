"""Verify saved refit bridges and native structure without a finder, fitting, response or coverage."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import refit_membership as N

B, I, M, P, F, A, SR, V = N.B, N.I, N.M, N.P, N.F, N.A, N.SR, N.V
require = SR._require
SCHEMA = "saved-refit-membership-report-v1"
CAP = 2 * 1024**2
FALSE_FIELDS = {"stock_native_membership_compared", "same_pulse_stock_validation", "regenerated_response",
                "regenerated_fit", "stock_rerun", "inherited_raw_admission_repeated", "prior_verifier_rerun",
                "coverage_computed", "execution_admitted", "full_retained_campaign_admitted", "timing_closure"}
NULL_FIELDS = {"F", "finite_library_uncertainty"}
SUMMARY_KEYS = {"schema", "scope", "inputs", "code_sha256", "registry_sha256", "registry_frames", "unique_frames",
                "saved_pulses", "saved_waveforms", "finder_calls", "records", "finder_config", "coincidence_limits",
                "nonempty_real_refit_membership", "full_registry_admitted", "stock_refit_pulse_bridge_exact",
                "output_cap_bytes", "seconds", "memory", *FALSE_FIELDS, *NULL_FIELDS}
FRAME_TYPES = dict(V.REGISTRY, pulses="int64", clusters="int64", members="int64")
NATIVE_PULSES = {name: V.PULSES[name] for name in [*P.FRAME_KEYS, "idx", *P.PULSE_FIELDS, "failed", "zerondf"]}
ROLES = {"downstream": "thread-d-saved-photon-downstream-v1", "boundary": "thread-d-optical-boundary-v1",
         "verified": "thread-d-downstream-report-v1", "membership": M.SCHEMA, "membership_verified": V.SCHEMA}
SCOPE = ("Saved-output structural verification of three selected ADC-refit frames. No finder, fitting, response, "
         "stock, raw-input, prior-verifier or coverage execution. Exact source/index/link/census checks do not "
         "independently establish cluster scalar physics, stock equivalence, efficiency, timing closure or F.")


def generation_paths():
    return list(dict.fromkeys([Path(N.__file__), Path(B.__file__), Path(N.L.__file__), Path(I.__file__),
                               Path(V.__file__), *M.code_paths()]))


def summary_contract(s):
    require(set(s) == SUMMARY_KEYS and s["schema"] == N.SCHEMA and s["scope"] == N.SCOPE,
            "refit summary schema/scope changed")
    for name in FALSE_FIELDS:
        require(s[name] is False, "refit scope changed: " + name)
    for name in NULL_FIELDS:
        require(s[name] is None, "refit null scope changed: " + name)
    for name in ("full_registry_admitted", "stock_refit_pulse_bridge_exact"):
        require(s[name] is True, "missing refit admission: " + name)
    for name, value in (("registry_frames", 9000), ("unique_frames", 3), ("saved_pulses", 87),
                        ("saved_waveforms", 80), ("finder_calls", 2), ("output_cap_bytes", N.CAP)):
        require(type(s[name]) is int and s[name] == value, "refit census/cap changed: " + name)
    require(s["registry_sha256"] == N.REGISTRY_SHA, "full registry digest changed")
    require(type(s["nonempty_real_refit_membership"]) is bool, "invalid nonempty observation")
    SR._equal(s["finder_config"], asdict(M.C.finder_config()), "refit finder configuration changed")
    SR._equal(s["coincidence_limits"], asdict(P.Limits()), "refit resource limits changed")
    require(type(s["records"]) is list and len(s["records"]) == 2, "refit block record census differs")
    require(type(s["inputs"]) is dict and 0 < len(s["inputs"]) <= 256, "refit source binding cap exceeded")
    for path, digest in s["inputs"].items():
        require(type(path) is str and Path(path).is_absolute(), "absolute bound source path required")
        M.S.I._digest(digest)
    require(type(s["seconds"]) in (int, float) and np.isfinite(s["seconds"]) and s["seconds"] >= 0,
            "invalid generation timing observation")
    require(type(s["memory"]) is dict and set(s["memory"]) == {"peak_rss_bytes_cumulative", "current_rss_bytes"},
            "invalid generation memory observation")
    for name, amount in s["memory"].items():
        require((name == "current_rss_bytes" and amount is None) or (type(amount) is int and amount >= 0),
                "invalid generation memory value")


def sources(bindings, out):
    """Resolve unique schema-qualified reports from inherited pins, never raw file contents."""
    candidates = [(Path(p), sha) for p, sha in bindings.items()
                  if Path(p).suffix == ".json" and Path(p).name not in (A.COMPLETE, A.INCOMPLETE, A.PENDING)]
    require(len(candidates) == len(ROLES), "source JSON report domain differs")
    roles = {}
    for path, sha in candidates:
        M._outside(out, [path.parent])
        require(SR._regular(path) <= 8 * 1024**2, "source JSON byte cap exceeded before hash")
        value = M.read_json(M.bind(path, sha))
        matches = [role for role, schema in ROLES.items() if value.get("schema") == schema]
        require(len(matches) == 1 and matches[0] not in roles, "missing or ambiguous source report schema")
        roles[matches[0]] = (path, sha)
    require(set(roles) == set(ROLES), "source report roles incomplete")
    return roles


def member_side_counts(products):
    """Check integer positive-PE side-hit counts; do not recompute floating cluster properties."""
    c, m, p = (products[name] for name in ("clusters", "members", "pulses"))
    links = m.merge(p[[*P.FRAME_KEYS, "idx", "pe"]].rename(columns={"idx": "pulse_idx"}),
                    on=[*P.FRAME_KEYS, "pulse_idx"], how="left", validate="many_to_one", indicator=True)
    require(links._merge.eq("both").all(), "side census pulse link missing")
    links["side"] = links.chan.to_numpy() % 2
    keys = pd.MultiIndex.from_frame(c[["frame_id", "idx"]])
    for side in (0, 1):
        count = links.loc[links.side.eq(side) & links.pe.gt(0)].groupby(["frame_id", "cluster_idx"]).size()
        expected = count.reindex(keys, fill_value=0).to_numpy(dtype=np.int64)
        require(np.array_equal(c[f"sidehits{side}"], expected), "cluster positive-PE side count differs")
        absent = expected == 0
        require((c.loc[~absent, f"sidepe{side}"] > 0).all(), "populated side has no positive PE")
        for name in (f"sidepe{side}", f"sidetime{side}"):
            values = c.loc[absent, name].to_numpy()
            require(np.all(values == 0) and not np.signbit(values).any(), "absent side sentinel differs")
    require((~c.poscalc | (c.sidehits0.gt(0) & c.sidehits1.gt(0))).all(), "position method requires both sides")


def products(directory, block, registry, calibration, geom):
    """Read bounded products and validate native links against independently rebuilt source rows."""
    number = block["block"]
    for name in ("pulses", "waveforms", "bridge"):
        saved = I.table(directory / f"block{number}_saved_{name}.parquet", 2048)
        V._exact(saved, block[name], "saved " + name)
    saved = {name: V._table(directory / f"block{number}_{name}.parquet", types, 2048)
             for name, types in (("pulses", NATIVE_PULSES), ("clusters", V.CLUSTERS), ("members", V.MEMBERS))}
    require(np.array_equal(saved["pulses"].idx, block["pulses"].refit_idx), "admitted refit index bridge changed")
    saved["pulses"] = saved["pulses"][list(V.PULSES)]
    frame_table = V._table(directory / f"block{number}_frames.parquet", FRAME_TYPES, 4)
    rebuilt = V._products(block["pulses"], block["selected"], registry, calibration, geom, saved)
    V._exact(frame_table, rebuilt["frames"], "complete refit frame census")
    cfg = M.C.finder_config()
    eligible = np.array([name in cfg.cuts for name in geom.sector_names], dtype=bool)
    require(eligible[rebuilt["members"].sector.to_numpy()].all(), "native member sector is disabled")
    require(all(cut.coincidence_layers == 3 for cut in cfg.cuts.values()), "unsupported layer consistency contract")
    require((rebuilt["clusters"].nLayers >= 3).all(), "native cluster violates minimum layer requirement")
    member_side_counts(rebuilt)
    return dict(block=number, diagnostics=block["diagnostics"], frames=rebuilt["frames"].to_dict("records"),
                counts={name: len(table) for name, table in rebuilt.items()})


def verify(directory, out, *, expected_summary_sha256):
    directory, out = Path(directory), Path(out)
    M._outside(out, [directory])
    begin = time.perf_counter()
    summary = M.read_json(M.bind(directory / "summary.json", expected_summary_sha256))
    summary_contract(summary)
    code = {str(path): M.S.C.sha256(path) for path in generation_paths()}
    SR._equal(summary["code_sha256"], code, "mandatory refit generation sources changed")
    verifier_code = {str(Path(__file__)): M.S.C.sha256(__file__), **code}
    M._outside(out, [Path(path).parent for path in summary["inputs"]] + [Path(path).parent for path in verifier_code])
    roles = sources(summary["inputs"], out)
    context = dict(registry_sha256=N.REGISTRY_SHA, batch_identity={"contract": N.SCHEMA,
                   "downstream_sha256": roles["downstream"][1], "membership_sha256": roles["membership"][1],
                   "seed_role": "protocol-only label; deterministic saved ADC refit; no random draw"},
                   arms=["saved_adc_refit"], seeds=[0])
    names = ["summary.json", "registry.parquet", "calibration.parquet"]
    names += [f"block{i}_{name}.parquet" for i in (0, 1) for name in
              ("frames", "pulses", "clusters", "members", "saved_bridge", "saved_waveforms", "saved_pulses")]
    names += [f"code{i}.txt" for i in range(len(code))]
    receipt, size = SR._batch(directory, context, names, N.CAP)
    require(receipt["artifacts"]["summary.json"]["sha256"] == expected_summary_sha256, "late refit summary mutation")
    for i, digest in enumerate(code.values()):
        M.bind(directory / f"code{i}.txt", digest)
    evidence = N.Evidence()
    registry, calibration, geom = N.authority(evidence, roles["membership"][0].parent,
        roles["membership_verified"][0], roles["membership"][1], roles["membership_verified"][1])
    ancestor = M.read_json(roles["membership"][0])
    ancestor_code = {str(path): M.S.C.sha256(path) for path in M.code_paths()}
    SR._equal(ancestor["code_sha256"], ancestor_code, "mandatory inherited generation sources changed")
    for i, digest in enumerate(ancestor_code.values()):
        evidence.bound(roles["membership"][0].parent / f"code{i}.txt", digest)
    blocks = N.admission(evidence, roles["downstream"][0].parent, roles["boundary"][0].parent,
        roles["verified"][0], registry, calibration, geom,
        expected_downstream_sha256=roles["downstream"][1], expected_boundary_sha256=roles["boundary"][1],
        expected_verified_sha256=roles["verified"][1])
    SR._equal(evidence.bindings, summary["inputs"], "refit source bindings differ from admitted evidence")
    M._outside(out, [Path(path).parent for path in evidence.bindings])
    V._exact(I.table(directory / "registry.parquet", 9000), registry, "full registry")
    V._exact(I.table(directory / "calibration.parquet", 15360), calibration, "complete calibration")
    records = [products(directory, block, registry, calibration, geom) for block in blocks]
    SR._equal(records, summary["records"], "saved refit diagnostics/counts/frames changed")
    nonempty = any(row["counts"]["members"] > 0 for row in records)
    require(summary["nonempty_real_refit_membership"] is nonempty, "saved nonempty observation differs")
    snapshots = {}
    for path, digest in verifier_code.items():
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest, "verifier source snapshot hash differs")
        snapshots[path] = payload.decode("utf-8")
    result = dict(schema=SCHEMA, scope=SCOPE, summary_sha256=expected_summary_sha256,
                  code_sha256=verifier_code, code_snapshots=snapshots, source_reports={k: {"path": str(p), "sha256": d}
                      for k, (p, d) in roles.items()}, registry_sha256=N.REGISTRY_SHA, registry_frames=len(registry),
                  unique_frames=3, counts=[dict(block=r["block"], **r["counts"]) for r in records],
                  frames=[frame for r in records for frame in r["frames"]],
                  stock_refit_differences=[dict(block=r["block"], **r["diagnostics"]["stock_refit_differences"]) for r in records],
                  saved_artifact_protocol_exact=True, inherited_membership_transaction_exact=True,
                  source_bridge_rechecked=True, source_tables_exact=True, canonical_pulses_exact=True,
                  native_member_structure_exact=True, positive_pe_side_counts_exact=True, qualified_frame_census_exact=True,
                  nonempty_real_refit_membership=nonempty, regenerated_finder=False, cluster_scalars_independently_recomputed=False,
                  native_member_order_independently_recomputed=False, persisted_path_bytes=size, output_cap_bytes=CAP,
                  generation_measurements={"seconds": summary["seconds"], "memory": summary["memory"],
                      "interpretation": "Hash-bound generation observations; not independently measured by verification"},
                  verification_seconds=time.perf_counter() - begin, verification_memory=M.S.C.memory(),
                  publication="exclusive JSON; no atomic-completion or crash-durability claim",
                  **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(NULL_FIELDS))
    payload = A._canonical(result)
    require(len(payload) <= CAP, "refit verification report byte cap exceeded")
    evidence.recheck()
    M.check_bindings(M.S.C.merge_bindings(verifier_code, M.transaction_bindings(directory, receipt)))
    final, final_size = SR._batch(directory, context, names, N.CAP)
    require(final == receipt and final_size == size, "saved refit transaction changed late")
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def compact(result):
    return {name: result[name] for name in ("schema", "summary_sha256", "unique_frames", "counts", "frames",
            "stock_refit_differences", "nonempty_real_refit_membership", "persisted_path_bytes", "verification_seconds",
            "verification_memory", "generation_measurements", "scope", "F")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    args = vars(parser.parse_args(argv))
    print(json.dumps(compact(verify(**args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
