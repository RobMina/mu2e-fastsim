"""Synthetic candidate tests only: never open production physics or invoke art."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers import production_candidate as candidate
from test_showers_training_audit import inventory  # authenticated synthetic fixed-ten inventory fixture
from test_showers_training_coverage import part, parts, sha, save_json


def geometry():
    result = dict(disk_disk=np.array([0, 1]), crystal_crystal=np.array([0, 1]),
                  crystal_disk=np.array([0, 1]), vd_id=np.array([73, 75]))
    for i, axis in enumerate("xyz"):
        for j, component in enumerate("xyz"):
            result[f"disk_axis_{axis}_{component}"] = np.full(2, float(i == j))
    return result


def ledger(n, fid="10", subrun=1, source_start=1):
    expected = dict(output_fid=np.full(n, fid), run=np.ones(n, np.int64),
                    subrun=np.full(n, subrun, np.int64), event=np.arange(1, n + 1, dtype=np.int64))
    full = dict({k: v.copy() for k, v in expected.items()}, parent_fid=np.full(n, candidate.PARENT_FID),
                source_run=np.ones(n, np.int64), source_subrun=np.ones(n, np.int64),
                source_event=np.arange(source_start, source_start + n, dtype=np.int64),
                source_count=np.ones(n, np.int64), source_present=np.ones(n, bool),
                source_file_resolved=np.ones(n, bool))
    return expected, full


def pack_case():
    e, s, c, total = candidate.training_coverage.merge(parts())
    # Nontrivial frame, times and deliberately interleaved entry rows.
    e["direction"][:] = [.6, 0, .8]
    e["position"][:] = [400., -20., 12000.]
    e["time"][:] = 300.
    s["position"][:] = [[401., -19., 12001.], [404., -22., 12003.], [402., -18., 12006.]]
    s["time"][:] = [301., 305., 307.]
    for key in s:
        s[key] = s[key][[1, 0, 2]]
    expected, full = ledger(7)
    return e, s, geometry(), full, expected, dict(file_ordinal=np.array([1] * 4 + [2] * 3),
                                                file_event_row=np.array([0, 1, 2, 3, 0, 1, 2]))


def test_pack_preserves_whole_events_zeros_other_disk_offsets_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate.library, "sample", lambda *a, **k: pytest.fail("no sampling"))
    e, s, g, full, expected, kwargs = pack_case()
    originals = deepcopy((e, s, g, full, expected))
    packed, provenance = candidate.pack(e, s, g, full, expected, **kwargs)
    np.testing.assert_array_equal(packed["offsets"], [0, 1, 3, 3])
    np.testing.assert_array_equal(provenance["global_event_row"], [1, 3, 6])
    np.testing.assert_array_equal(provenance["file_event_row"], [1, 3, 2])
    np.testing.assert_array_equal(provenance["file_ordinal"], [1, 1, 2])
    np.testing.assert_array_equal(provenance["donor_row"], [0, 1, 2])
    np.testing.assert_array_equal(packed["step_disk"], [1, 0, 1])
    np.testing.assert_array_equal(packed["step_energy_birks"], [0, 4, 7])
    for key in candidate.source_split.OUTPUT_KEY + candidate.source_split.PARENT_KEY:
        np.testing.assert_array_equal(provenance[key], full[key][[1, 3, 6]])
    candidate.library.save(tmp_path / "library.npz", packed, {"training_only": True})
    saved, metadata = candidate.library.load(tmp_path / "library.npz")
    assert metadata == {"training_only": True}
    candidate.validate_packing(saved, e, s, g)
    for current, before in zip((e, s, g, full, expected), originals):
        for key in before:
            np.testing.assert_array_equal(current[key], before[key])


@pytest.mark.parametrize("defect", ["parent", "duplicate-parent", "partial-ledger", "source-count", "source-present",
                                  "wrong-output", "ledger-order", "event-order", "bad-geometry", "bad-frame",
                                  "coordinate-shape", "coordinate-duplicate"])
def test_pack_rejects_identity_geometry_and_partial_census(defect):
    e, s, g, full, expected, kwargs = pack_case()
    if defect == "parent": full["parent_fid"][0] = "999"
    elif defect == "duplicate-parent": full["source_event"][1] = full["source_event"][0]
    elif defect == "partial-ledger": full = {k: v[:-1] for k, v in full.items()}
    elif defect == "source-count": full["source_count"][0] = 2
    elif defect == "source-present": full["source_present"][0] = False
    elif defect == "wrong-output": full["event"][0] = 99
    elif defect == "ledger-order": full = {k: v[::-1] for k, v in full.items()}
    elif defect == "event-order": e["event_row"] = e["event_row"][::-1]
    elif defect == "bad-geometry": g["disk_disk"][1] = 3
    elif defect == "bad-frame": g["disk_axis_x_x"][0] = 0
    elif defect == "coordinate-shape": kwargs["file_ordinal"] = np.ones(3, np.int64)
    else: kwargs["file_event_row"][1] = 0
    with pytest.raises(ValueError):
        candidate.pack(e, s, g, full, expected, **kwargs)


@pytest.mark.parametrize("key", ["offsets", "entry_energy", "entry_time", "step_disk", "step_crystal",
                                "step_energy_birks", "step_relative_time", "step_relative_position", "geometry_disk_disk"])
def test_roundtrip_detects_packing_mutations(key):
    e, s, g, full, expected, kwargs = pack_case()
    packed, _ = candidate.pack(e, s, g, full, expected, **kwargs)
    packed[key] = packed[key].copy()
    packed[key].flat[0] += 1
    with pytest.raises(ValueError):
        candidate.validate_packing(packed, e, s, g)


@pytest.fixture
def bundle(tmp_path, monkeypatch, inventory):
    args = SimpleNamespace(**{k: tmp_path / k for k in
                             ("coverage_report", "tables", "bound", "parent_summary", "out")})
    args.inventory = inventory.path
    inv = inventory.data
    parent_meta = tmp_path / "parent.metadata.json"
    parent_sha = save_json(parent_meta, {"file_name": candidate.source_audit.PARENT_NAME,
                                         "file_id": int(candidate.PARENT_FID)})
    inv.update(parent_metadata_sha256=parent_sha,
               original_parent=dict(fid=candidate.PARENT_FID, metadata_filename=parent_meta.name,
                                    metadata_sha256=parent_sha))
    inv_sha = save_json(args.inventory, inv)
    proofs, per_file, validations, cases, source_results = {}, [], [], {}, {}
    source_start = 1
    for row in inventory.rows:
        name, n = row["name"], row["event_count"]
        piece = part([1, n - 1], np.r_[0., 0., np.zeros(n - 3), 11.], [(1, 0, 4), (1, 1, 7)])
        cases[name] = piece
        directory = args.tables / name
        table = directory / "export/calo_library_tables.root"
        table.parent.mkdir(parents=True)
        table.write_bytes(name.encode())
        evidence = {}
        for receipt in candidate.RECEIPTS:
            path = directory / receipt
            path.write_bytes((name + receipt).encode())
            evidence[receipt] = sha(path.read_bytes())
        expected, full = ledger(n, row["fid"], row["subrun"], source_start)
        expected["run"][:] = row["run"]
        full["run"][:] = row["run"]
        source_start += n
        bound = args.bound / name / "bound"
        bound.mkdir(parents=True)
        for filename, value in (("expected.npz", expected), ("ledger.npz", full)):
            candidate.source_overlap_report._save_npz(bound / filename, value)
            proofs["saved/" + name + "/bound/" + filename] = candidate.source_events.file_digest(bound / filename)
        hashes = dict.fromkeys(candidate.PROOF_KEYS, "a" * 64)
        hashes.update(inventory=inv_sha, metadata=row["metadata_sha256"], parent_metadata=parent_sha)
        binding = dict(role="training", item=name, ordinal=row["ordinal"], events=n, source_rows=n,
                       parent_fid=candidate.PARENT_FID, ordered_full_census_exact=True, input_sha256=hashes,
                       art_sha256=row["sha256"], census_receipt_checked=True, manifest_sha256=inv["manifest_sha256"],
                       output_sha256={f: proofs["saved/" + name + "/bound/" + f]
                                      for f in ("expected.npz", "ledger.npz")})
        proofs["saved/" + name + "/bound/summary.json"] = save_json(bound / "summary.json", binding)
        identity = {"events": n, "ordered_event_census_exact": True}
        source_results[name] = (full, identity)
        validations.append(dict(item=name, fid=row["fid"], events=n, art_sha256=row["sha256"],
                                tables_sha256=sha(table.read_bytes()), evidence_sha256=evidence,
                                source_proof={"input_sha256": hashes}, source_identity=identity,
                                raw_reconciliation={"census": {k: int(v) for k, v in piece[2].items()}},
                                status={"events": n, "failed_transport_events": 0}))
        per_file.append(dict(item=name, fid=row["fid"], **candidate.coverage.census(*piece)[1]))
    parent_sha = save_json(args.parent_summary, {"input_sha256": proofs})
    cells, report = candidate.coverage.census(*candidate.training_coverage.merge(list(cases.values())))
    prior = dict(input_sha256={"inventory": inv_sha, "parent_summary": parent_sha},
                 source_sha256={m.__name__: candidate.source_events.file_digest(m.__file__) for m in
                                (candidate.coverage, candidate.export, candidate.library, candidate.production_check,
                                 candidate.source_audit, candidate.source_events, candidate.training_audit,
                                 candidate.training_coverage)}, validation=validations, per_file=per_file,
                 training=report, comparison=candidate.coverage.compare(cells, cells))
    args.coverage_sha256 = save_json(args.coverage_report, prior)
    calls = []
    def read(path, *, input_mode):
        assert input_mode == "primary"
        name = path.parent.parent.name
        calls.append(name)
        e, s, c, total = cases[name]
        return e, s, geometry(), c, total
    def numeric(path, expected, **kwargs):
        result, identity = source_results[path.parent.parent.name]
        return deepcopy(result), deepcopy(identity)
    monkeypatch.setattr(candidate.export, "read", read)
    monkeypatch.setattr(candidate.source_events, "read", numeric)
    monkeypatch.setattr(candidate.library, "sample", inventory.forbidden)
    monkeypatch.setattr(candidate.training_audit, "validate_saved", inventory.forbidden)
    # Only the frozen production population sizes are replaced for this tiny-CSS oracle.
    monkeypatch.setattr(candidate, "_authorized", lambda report: None)
    return SimpleNamespace(args=args, prior=prior, calls=calls, rows=inventory.rows, cases=cases)


def repin(case):
    case.args.coverage_sha256 = save_json(case.args.coverage_report, case.prior)


def test_run_authenticated_ten_files_once_complete_ledger_and_all_outputs(bundle):
    report = candidate.run(bundle.args)
    assert bundle.calls == [r["name"] for r in bundle.rows]
    assert report["training"]["stored_events"] == 28265
    assert report["training"]["front_entry_events"] == 20
    assert report["sampled"] is False and report["legacy_donors_included"] is False
    assert report["source_independence_certified"] is False
    assert set(p.name for p in bundle.args.out.iterdir()) == set(candidate.ARTIFACTS.values())
    for name, digest in report["output_sha256"].items():
        assert candidate.source_events.file_digest(bundle.args.out / name) == digest
    with np.load(bundle.args.out / "training_ledger.npz", allow_pickle=False) as full:
        assert len(full["event"]) == 28265
    with np.load(bundle.args.out / "donor_provenance.npz", allow_pickle=False) as donors:
        assert len(donors["event"]) == 20
        np.testing.assert_array_equal(donors["file_ordinal"], np.repeat(np.arange(1, 11), 2))
    assert json.loads((bundle.args.out / "summary.json").read_text()) == report
    with pytest.raises(FileExistsError):
        candidate.run(bundle.args)


@pytest.mark.parametrize("defect", ["unanchored", "partial-proof", "partial-evidence", "missing-bound", "ambiguous-bound",
                                  "wrong-bound", "wrong-table", "wrong-receipt", "stale-source", "per-file", "aggregate",
                                  "cells", "numeric-id", "geometry", "mutate-table", "mutate-inventory", "order"])
def test_run_fails_closed_before_output(bundle, monkeypatch, defect):
    args, prior = bundle.args, bundle.prior
    name = bundle.rows[0]["name"]
    if defect == "unanchored": args.coverage_sha256 = "0" * 64
    elif defect == "partial-proof":
        del prior["validation"][0]["source_proof"]["input_sha256"]["printer_log"]
        repin(bundle)
    elif defect == "partial-evidence":
        del prior["validation"][0]["evidence_sha256"]["source_config.txt"]
        repin(bundle)
    elif defect in ("missing-bound", "ambiguous-bound"):
        parent = json.loads(args.parent_summary.read_text())
        key = "saved/" + name + "/bound/ledger.npz"
        if defect == "missing-bound": del parent["input_sha256"][key]
        else: parent["input_sha256"]["other/" + key] = parent["input_sha256"][key]
        prior["input_sha256"]["parent_summary"] = save_json(args.parent_summary, parent)
        repin(bundle)
    elif defect == "wrong-bound": (args.bound / name / "bound/ledger.npz").write_bytes(b"changed")
    elif defect == "wrong-table": (args.tables / name / "export/calo_library_tables.root").write_bytes(b"changed")
    elif defect == "wrong-receipt": (args.tables / name / "source_check.json").write_bytes(b"changed")
    elif defect == "stale-source":
        prior["source_sha256"][candidate.export.__name__] = "0" * 64
        repin(bundle)
    elif defect == "per-file":
        prior["per_file"][0]["front"]["empty"] += 1
        repin(bundle)
    elif defect == "aggregate":
        prior["training"]["front"]["empty"] += 1
        repin(bundle)
    elif defect == "cells":
        prior["comparison"]["cells"][0]["training"]["counts"] += 1
        repin(bundle)
    elif defect == "order":
        prior["validation"] = prior["validation"][::-1]
        repin(bundle)
    elif defect == "numeric-id":
        original = candidate.source_events.read
        def numeric(*a, **k):
            numeric, identity = original(*a, **k)
            numeric["source_event"][0] += 1
            return numeric, identity
        monkeypatch.setattr(candidate.source_events, "read", numeric)
    else:
        original = candidate.export.read
        def read(path, **kwargs):
            result = original(path, **kwargs)
            if defect == "geometry" and path.parent.parent.name != name:
                result[2]["disk_axis_x_x"][0] = 0
            elif defect == "mutate-table": path.write_bytes(b"changed")
            elif defect == "mutate-inventory": args.inventory.write_text(args.inventory.read_text() + " ")
            return result
        monkeypatch.setattr(candidate.export, "read", read)
    with pytest.raises(ValueError):
        candidate.run(args)
    assert not args.out.exists()


def test_authorized_exact_population():
    report = dict(stored_events=28265, front_entry_events=24120, front=dict(empty=259, weak=835, css_rows=1034261))
    candidate._authorized(report)
    report["front"]["empty"] -= 1
    with pytest.raises(ValueError, match="authorized"):
        candidate._authorized(report)


def test_inputs_detect_mutation_and_external_anchor(tmp_path):
    path = tmp_path / "proof.json"
    digest = save_json(path, {"proof": "saved"})
    inputs = candidate.Inputs()
    assert inputs.checked_json(path, digest) == {"proof": "saved"}
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        inputs.unchanged()
    with pytest.raises(ValueError, match="hash mismatch"):
        candidate.Inputs().checked_json(path, digest)


def test_cli_requires_external_pin(tmp_path):
    with pytest.raises(SystemExit):
        candidate.main(["--coverage-report", str(tmp_path / "report")])


def test_coverage_cli_module_hash_alias(bundle):
    hashes = bundle.prior['source_sha256']
    hashes['__main__'] = hashes.pop(candidate.training_coverage.__name__)
    repin(bundle)
    report = candidate.run(bundle.args)
    assert 'calofastsim.showers.production_candidate' in report['source_sha256']


def test_mutation_while_packing_stops_before_directory(bundle, monkeypatch):
    original = candidate.library.build
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        bundle.args.parent_summary.write_text('{}')
        return result
    monkeypatch.setattr(candidate.library, 'build', build)
    with pytest.raises(ValueError, match='changed'):
        candidate.run(bundle.args)
    assert not bundle.args.out.exists()


def test_mutation_during_publication_never_publishes_success_receipt(bundle, monkeypatch):
    original = candidate.library.save
    def save(*args, **kwargs):
        original(*args, **kwargs)
        bundle.args.coverage_report.write_text('{}')
    monkeypatch.setattr(candidate.library, 'save', save)
    with pytest.raises(ValueError, match='changed'):
        candidate.run(bundle.args)
    assert not (bundle.args.out / 'summary.json').exists()
    with pytest.raises(FileExistsError):
        candidate.run(bundle.args)
