"""Per-frame frozen-grid coverage from pinned saved refit coincidences, never a selected-frame F."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import refit_membership_report as R

N, M, V, F, A, SR = R.N, R.M, R.V, R.F, R.A, R.SR
G = M.G
require = SR._require
SCHEMA = "saved-refit-coverage-v1"
CAP = 2 * 1024**2
MAX_CELLS = 80000
GRID_COUNTS = dict(selected=1929421, total_tracks=1930163, selection_events=1930163, bins=4042)
SCOPE = ("Conditional per-frame coverage of three deliberately selected saved ADC-refit frames. "
         "Reuse pinned native structure and the existing historical track grid; no finder, fit, response, stock, "
         "raw input, selected-track read or prior verifier. No mean, F, uncertainty or promotion inference.")
FALSE_FIELDS = {"regenerated_finder", "regenerated_fit", "regenerated_response", "regenerated_optics", "stock_rerun",
                "selected_tracks_read", "selected_tracks_rehashed", "track_grid_rebuilt", "prior_verifier_rerun",
                "source_bridge_rechecked", "native_structure_rechecked", "cluster_scalars_independently_recomputed",
                "native_member_order_independently_recomputed", "same_pulse_stock_validation", "stock_native_membership_compared",
                "inherited_raw_admission_repeated", "execution_admitted", "full_retained_campaign_admitted", "timing_closure"}


def code_paths():
    return list(dict.fromkeys([Path(__file__), Path(R.__file__), *R.generation_paths()]))


def bounded_bytes(path, cap=N.CAP):
    require(SR._regular(Path(path)) <= cap, "input byte cap exceeded before read")
    with Path(path).open("rb") as stream:
        payload = stream.read(cap + 1)
    require(len(payload) <= cap, "input byte cap exceeded during read")
    return payload


def file_sha(path):
    return hashlib.sha256(bounded_bytes(path)).hexdigest()


def bounded_json(path, expected_digest=None):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    def bad(value):
        raise ValueError("nonfinite JSON value")
    payload = bounded_bytes(path, 8 * 1024**2)
    if expected_digest is not None:
        require(hashlib.sha256(payload).hexdigest() == expected_digest, "consumed JSON bytes differ from pin")
    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=bad)
    require(type(value) is dict, "JSON object required")
    return value


class Evidence(N.Evidence):
    """Bound every fresh hash and compare transaction receipts with independently inherited pins."""

    def bound(self, path, digest):
        path = Path(path)
        M.S.I._digest(digest)
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest, "conflicting coverage binding")
        require(file_sha(path) == digest, "changed bound coverage input: " + str(path))
        self.bindings[str(path)] = digest
        return path

    def json(self, path, digest):
        return bounded_json(self.bound(path, digest), digest)

    def batch(self, root, context, names, cap):
        receipt = self.check_batch(root, context, names, cap)
        for path, digest in M.transaction_bindings(root, receipt).items():
            self.bound(path, digest)
        self.transactions.append((root, context, names, cap, receipt))
        return receipt

    @staticmethod
    def check_batch(root, context, names, cap):
        require(not root.is_symlink() and root.is_dir(), "saved batch must be a directory")
        require(set(p.name for p in root.iterdir()) == {*names, A.INCOMPLETE, A.COMPLETE, A.PENDING},
                "failed, missing or undeclared coverage input artifact")
        sizes = {p.name: SR._regular(p) for p in root.iterdir()}
        require(sum(sizes.values()) <= cap, "saved transaction byte cap exceeded")
        complete, initial = (bounded_json(root / n) for n in (A.COMPLETE, A.INCOMPLETE))
        _, digest = A._provenance(context)
        common = dict(schema=A.SCHEMA, context_sha256=digest, execution_admitted=False,
                      physics_truncated=False, denominator_renormalized=False)
        reserve = len(A._canonical(dict(common, state="failed")))
        SR._equal(initial, dict(common, state="incomplete", context=context, max_bytes=cap,
                               expected_artifacts=sorted(names), failure_reserve_bytes=reserve), "initial receipt differs")
        artifacts = {name: {"bytes": sizes[name], "sha256": file_sha(root / name)} for name in names}
        SR._equal(complete, dict(common, state="complete", context=context, max_bytes=cap, artifacts=artifacts,
                  artifact_bytes=sum(sizes[name] for name in names), incomplete_sha256=file_sha(root / A.INCOMPLETE),
                  receipt_staging_retained=True), "complete receipt differs")
        require(bounded_bytes(root / A.INCOMPLETE) == A._canonical(initial), "noncanonical initial receipt")
        require(bounded_bytes(root / A.COMPLETE) == A._canonical(complete)
                and bounded_bytes(root / A.PENDING) == A._canonical(complete), "completion marker bytes differ")
        require(sum(sizes.values()) + reserve <= cap, "saved transaction reserve byte cap exceeded")
        return complete

    def recheck(self):
        for path, digest in dict(self.bindings).items():
            self.bound(path, digest)
        for root, context, names, cap, original in self.transactions:
            SR._equal(self.check_batch(root, context, names, cap), original, "coverage input transaction changed late")


def transaction(evidence, directory, summary, *, refit):
    """Hash complete saved transactions without decoding pulse, waveform or member payloads."""
    paths = R.generation_paths() if refit else M.code_paths()
    code = {str(path): file_sha(path) for path in paths}
    SR._equal(summary["code_sha256"], code, "saved generation code map changed")
    if refit:
        context = dict(registry_sha256=N.REGISTRY_SHA, batch_identity={"contract": N.SCHEMA,
            "downstream_sha256": evidence.roles["downstream"]["sha256"],
            "membership_sha256": evidence.roles["membership"]["sha256"],
            "seed_role": "protocol-only label; deterministic saved ADC refit; no random draw"},
            arms=["saved_adc_refit"], seeds=[0])
        names = ["summary.json", "registry.parquet", "calibration.parquet"]
        names += [f"block{i}_{name}.parquet" for i in (0, 1) for name in
                  ("frames", "pulses", "clusters", "members", "saved_bridge", "saved_waveforms", "saved_pulses")]
    else:
        context = dict(registry_sha256=N.REGISTRY_SHA, batch_identity={"contract": M.SCHEMA, **summary["hashes"]},
                       arms=list(M.S.ARMS), seeds=list(M.S.SEEDS))
        names = ["summary.json", "registry.parquet", "selected.parquet", "grid.parquet", "pulses.parquet",
                 "clusters.parquet", "members.parquet", "coverage.parquet"]
    names += [f"code{i}.txt" for i in range(len(code))]
    pinned = evidence.bindings[str(directory / "summary.json")]
    receipt = evidence.batch(directory, context, names, N.CAP)
    require(receipt["artifacts"]["summary.json"]["sha256"] == pinned, "late saved summary mutation")
    for i, (path, digest) in enumerate(code.items()):
        evidence.bound(path, digest)
        evidence.bound(directory / f"code{i}.txt", digest)
    return receipt


def refit_admission(evidence, directory, verified, summary_sha, verified_sha, complete_sha):
    evidence.bound(directory / A.COMPLETE, complete_sha)
    s = evidence.json(directory / "summary.json", summary_sha)
    R.summary_contract(s)
    v = evidence.json(verified, verified_sha)
    require(v["schema"] == R.SCHEMA and v["scope"] == R.SCOPE and v["summary_sha256"] == summary_sha,
            "refit verification binding/scope differs")
    for name in ("saved_artifact_protocol_exact", "inherited_membership_transaction_exact", "source_bridge_rechecked",
                 "source_tables_exact", "canonical_pulses_exact", "native_member_structure_exact",
                 "positive_pe_side_counts_exact", "qualified_frame_census_exact"):
        require(v[name] is True, "missing inherited refit gate: " + name)
    for name in R.FALSE_FIELDS | {"regenerated_finder", "cluster_scalars_independently_recomputed",
                                  "native_member_order_independently_recomputed"}:
        require(v[name] is False, "inherited refit scope changed: " + name)
    require(v["F"] is None and v["finite_library_uncertainty"] is None, "inherited refit promotion differs")
    require(v["registry_sha256"] == N.REGISTRY_SHA and v["registry_frames"] == 9000 and v["unique_frames"] == 3,
            "inherited refit registry/census differs")
    code = {str(p): file_sha(p) for p in [Path(R.__file__), *R.generation_paths()]}
    SR._equal(v["code_sha256"], code, "inherited refit verification code changed")
    require(set(v["code_snapshots"]) == set(code), "inherited verification snapshot domain differs")
    for path, digest in code.items():
        snapshot = v["code_snapshots"][path]
        require(type(snapshot) is str and hashlib.sha256(snapshot.encode()).hexdigest() == digest,
                "inherited verification snapshot differs")
        evidence.bound(path, digest)
    roles = v["source_reports"]
    require(set(roles) == set(R.ROLES), "inherited refit source report roles differ")
    for role in roles.values():
        require(set(role) == {"path", "sha256"} and s["inputs"].get(role["path"]) == role["sha256"],
                "inherited report not bound by refit summary")
    evidence.roles = roles
    evidence.original_inputs = s["inputs"]
    transaction(evidence, directory, s, refit=True)
    return s, v


def saved_grid(evidence):
    """Inherit grid provenance and check saved bytes/card; never access the selected-track path."""
    roles = evidence.roles
    membership = Path(roles["membership"]["path"]).parent
    s = evidence.json(membership / "summary.json", roles["membership"]["sha256"])
    v = evidence.json(**dict(path=roles["membership_verified"]["path"], digest=roles["membership_verified"]["sha256"]))
    require(s["schema"] == M.SCHEMA and set(s) == V.SUMMARY_KEYS, "saved grid summary schema differs")
    require(v["schema"] == V.SCHEMA and v["summary_sha256"] == roles["membership"]["sha256"],
            "saved grid verification binding differs")
    require(v["registry_sha256"] == s["registry_sha256"] == N.REGISTRY_SHA and v["registry_frames"] == 9000,
            "saved grid registry differs")
    for name in ("saved_artifact_protocol_exact", "source_transactions_exact", "qualified_frame_census_exact", "frozen_grid_exact"):
        require(v[name] is True, "missing inherited grid gate: " + name)
    for name in V.FALSE_FIELDS | {"regenerated_finder", "cluster_scalars_independently_recomputed"}:
        require(v[name] is False, "inherited grid scope changed: " + name)
    require(v["F"] is None and v["finite_library_uncertainty"] is None, "inherited grid promotion differs")
    evidence.bound(V.__file__, v["code_sha256"])
    receipt = transaction(evidence, membership, s, refit=False)
    for path, digest in M.transaction_bindings(membership, receipt).items():
        require(evidence.original_inputs.get(path) == digest, "grid transaction differs from pinned refit lineage")
    registry = V._table(membership / "registry.parquet", V.REGISTRY, 9000)
    require(len(registry) == 9000 and F.digest(registry) == N.REGISTRY_SHA, "saved full registry differs")
    SR._equal(s["grid"], v["grid"], "inherited grid metadata differs")
    meta = s["grid"]
    for name, value in GRID_COUNTS.items():
        require(type(meta[name]) is int and meta[name] == value, "frozen grid census differs: " + name)
    require(meta["contract"] == G.CONTRACT and meta["grid_ns"] == 0.25, "frozen grid semantics differ")
    grid = V._table(membership / "grid.parquet", V.TABLE_TYPES["grid"], 20000)
    require(len(grid) == meta["bins"], "saved grid length differs")
    require(grid.right.to_numpy()[:-1].tobytes() == grid.left.to_numpy()[1:].tobytes(), "saved grid discontinuity")
    hist, edges = G._grid(grid.weight.to_numpy(), np.r_[grid.left, grid.right.iloc[-1]])
    require(np.all(np.diff(edges) == meta["grid_ns"]), "saved grid spacing differs")
    pair_path = V._bound_path(s["inputs"], s["hashes"]["expected_paired_sha256"])
    pair = evidence.json(pair_path, s["hashes"]["expected_paired_sha256"])
    require(s["inputs"].get(pair["reference"]) == pair["reference_sha256"]
            and s["inputs"].get(pair["selection"]) == pair["selection_sha256"], "saved grid input lineage differs")
    M.S.I._digest(pair["selection_sha256"])
    require(Path(pair["selection"]).is_absolute(), "absolute selected-track provenance path required")
    reference = evidence.json(pair["reference"], pair["reference_sha256"])
    card = G._card(reference["card"])
    require(card.sha() == meta["card_sha"] and card.gate_mode == "event", "saved event card differs")
    require(reference["selected"] == meta["selected"] and reference["pmin"] == 85 and reference["pmax"] == 200
            and reference["F"]["grid_ns"] == meta["grid_ns"], "historical track selection metadata differs")
    return registry, card, hist, edges, meta, {"path": pair["selection"], "sha256": pair["selection_sha256"],
        "interpretation": "Inherited provenance only; not opened, hashed or re-histogrammed in this check"}


def saved_products(directory, summary, verified, registry):
    V._exact(V._table(directory / "registry.parquet", V.REGISTRY, 9000), registry, "full refit registry")
    fs, cs = [], []
    for number in (0, 1):
        frames = V._table(directory / f"block{number}_frames.parquet", R.FRAME_TYPES, 4)
        clusters = V._table(directory / f"block{number}_clusters.parquet", V.CLUSTERS, 2048)
        selected = registry.loc[registry.input_sha256.eq(N.INPUTS[number]) & registry.event.isin(N.TARGETS[number])]
        require(len(selected) == len(N.TARGETS[number]), "target identity is missing or ambiguous in full registry")
        V._exact(frames[list(F.COLUMNS)], selected, "selected full frame identity")
        SR._equal(frames.to_dict("records"), summary["records"][number]["frames"], "saved frame records differ")
        record = summary["records"][number]
        require(record["block"] == number, "saved block order differs")
        expected = dict(frames=len(frames), pulses=int(frames.pulses.sum()), clusters=len(clusters), members=int(frames.members.sum()))
        require((frames[["pulses", "clusters", "members"]] >= 0).all().all(), "negative frame census")
        SR._equal(expected, record["counts"], "saved block count differs")
        SR._equal(dict(block=number, **expected), verified["counts"][number], "verified block counts differ")
        fs.append(frames)
        cs.append(clusters)
    frames, clusters = pd.concat(fs, ignore_index=True), pd.concat(cs, ignore_index=True)
    require(len(frames) == 3 and frames.pulses.sum() == 87, "saved aggregate frame/pulse census differs")
    SR._equal(frames.to_dict("records"), verified["frames"], "verified frame census differs")
    order = pd.MultiIndex.from_frame(frames[G.KEYS]).get_indexer(pd.MultiIndex.from_frame(clusters[G.KEYS]))
    require(np.all(order >= 0) and np.all(np.diff(order) >= 0), "saved cluster frame identity/order differs")
    require(np.array_equal(clusters.idx, clusters.groupby(G.KEYS, sort=False).cumcount()), "saved cluster indices/order differ")
    nonempty = bool(len(clusters))
    require(summary["nonempty_real_refit_membership"] is nonempty and verified["nonempty_real_refit_membership"] is nonempty,
            "inherited nonempty observation differs")
    return frames, clusters


def evaluate(frames, clusters, card, hist, edges):
    """Keep official card coverage separate from diagnostic eligible-window occupancy."""
    require(card.gate_mode == "event", "coverage diagnostic requires event gate semantics")
    actual = G.evaluate(frames, clusters, card, hist, edges, max_cells=MAX_CELLS)
    ungated = replace(card, gate_start_lo=None, gate_end_hi=None)
    windows = G.evaluate(frames, clusters, ungated, hist, edges, max_cells=MAX_CELLS)
    require(np.array_equal(actual.n_veto_eligible, windows.n_veto_eligible), "event gate changed coincidence eligibility")
    require(np.array_equal(actual.track_grid_coverage, np.where(actual.gate_pass, windows.track_grid_coverage, 1.0)),
            "event gate decomposition differs")
    actual["eligible_window_coverage"] = windows.track_grid_coverage.to_numpy()
    actual["event_gate_override"] = ~actual.gate_pass
    return actual


def run(directory, verified, out, *, expected_summary_sha256, expected_verified_sha256, expected_complete_sha256):
    directory, verified, out = map(Path, (directory, verified, out))
    M._outside(out, [directory, verified.parent])
    begin = time.perf_counter()
    evidence = Evidence()
    code = {str(p): file_sha(p) for p in code_paths()}
    for path, digest in code.items():
        evidence.bound(path, digest)
    summary, prior = refit_admission(evidence, directory, verified, expected_summary_sha256,
                                     expected_verified_sha256, expected_complete_sha256)
    registry, card, hist, edges, metadata, selection = saved_grid(evidence)
    excluded = [Path(p).parent for p in evidence.bindings] + [Path(p).parent for p in summary["inputs"]]
    excluded += [Path(p).parent for p in code] + [Path(selection["path"]).parent]
    M._outside(out, excluded)
    frames, clusters = saved_products(directory, summary, prior, registry)
    evidence.recheck()
    coverage = evaluate(frames, clusters, card, hist, edges)
    snapshots = {}
    for path, digest in code.items():
        payload = bounded_bytes(path)
        require(hashlib.sha256(payload).hexdigest() == digest, "coverage source snapshot differs")
        snapshots[path] = payload.decode("utf-8")
    result = dict(schema=SCHEMA, scope=SCOPE, summary_sha256=expected_summary_sha256, verified_sha256=expected_verified_sha256,
        complete_sha256=expected_complete_sha256,
        inputs=evidence.bindings, code_sha256=code, code_snapshots=snapshots, registry_sha256=N.REGISTRY_SHA, registry_frames=9000,
        unique_frames=len(frames), coincidences=len(clusters), native_links=int(frames.members.sum()),
        rows=coverage.to_dict("records"), grid=metadata, card=dict(asdict(card), sha=card.sha()),
        selected_track_provenance=selection, inherited_native_structure_verified=True, saved_transactions_exact=True,
        qualified_frame_census_exact=True, historical_integer_grid_parity=True, eligible_window_grid_parity=True,
        event_gate_decomposition_exact=True, window_diagnostic="Same eligibility and integer bins; event gate disabled only for this diagnostic",
        max_cells=MAX_CELLS, output_cap_bytes=CAP, seconds=time.perf_counter() - begin, memory=M.S.C.memory(),
        publication="exclusive JSON; no atomic-completion or crash-durability claim",
        F=None, finite_library_uncertainty=None, **dict.fromkeys(FALSE_FIELDS, False))
    payload = A._canonical(result)
    require(len(payload) <= CAP, "saved coverage report byte cap exceeded")
    evidence.recheck()
    M._outside(out, [directory, verified.parent, *excluded])
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    for name in ("verified", "out", "expected_summary_sha256", "expected_verified_sha256", "expected_complete_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "rows", "grid", "seconds", "memory", "scope", "F")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
