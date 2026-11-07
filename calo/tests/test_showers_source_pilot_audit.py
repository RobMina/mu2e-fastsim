"""Synthetic copy/receipt/identity contracts; no external files or remote execution."""
from copy import deepcopy
import io
import json

import numpy as np
import pytest
import uproot

from calofastsim.showers import source_pilot_audit as pilot


def _pack(events):
    stream = io.BytesIO()
    np.savez(stream, **events)
    return stream.getvalue()


def fixture():
    n = 2000
    ids = dict(run=np.full(n, 1430), subrun=np.full(n, 6000), event=np.arange(1, n + 1))
    copied = dict(run=np.full(n, 1430), subrun=np.arange(n) // 200, event=np.arange(n) * 13 + 2130)
    proof = dict(events=n, parent_size=7076347412, parent_checksums={"adler32": "51b0fb96"},
                 pilot_sha256="a" * 64, copy_sha256="b" * 64,
                 original_source_uri="xroot://server/data/" + pilot.audit.PARENT_NAME,
                 pilot_pool_path="/fixture/test/stop_pool.art", approved_config_path="/fixture/test/approved.fcl")
    copycfg = '''outputs: { pool: { fileName: "stop_pool.art" module_type: "RootOutput"
outputCommands: ["keep *_*_*_*"] } } physics: {e1: ["pool"] end_paths: ["e1"]}
process_name: "CaloLibraryPool" source: {module_type: "RootInput"}'''
    config = '''TargetStopResampler: { fileNames: ["/fixture/test/stop_pool.art"] module_type: "ResamplingMixer"
mu2e: { MaxEventsToSkip: 0 writeEventIDs: true } readMode: "sequential" wrapFiles: true }'''
    log = []
    for i in range(n):
        log.extend([f"Begin processing the {i+1}th record. run: 1430 subRun: 6000 event: {i+1} at today",
                    "EventIDSequence InputTag: label = 'TargetStopResampler', instance = '', process = 'CaloLibraryCE' : "
                    f"( run: 1430 subRun: {copied['subrun'][i]} event: {copied['event'][i]} )"])
    log += ["TrigReport Events total = 2000 passed = 2000 failed = 0", pilot.check.COMPLETION]
    approved, executed = b'# approved\nprocess_name: "CaloLibraryCE"\n', b'# executed\nprocess_name: "CaloLibraryCE"\n'
    blobs = dict(parent_metadata=json.dumps(dict(file_name=pilot.audit.PARENT_NAME, file_id=102778799,
                 file_size=proof["parent_size"], checksum=["adler32:51b0fb96"])).encode(),
                 pilot_events=_pack(ids), copy_events=_pack(copied), printer=("\n".join(log) + "\n").encode(),
                 copy_source=(proof["original_source_uri"] + "\n").encode(), copy_config=copycfg.encode(),
                 copy_log=(f'Opened input file "{proof["original_source_uri"]}"\n'
                           'TrigReport Events total = 2000 passed = 2000 failed = 0\n' + pilot.check.COMPLETION).encode(),
                 copy_manifest=(proof["copy_sha256"] + "  stop_pool.art\n").encode(),
                 approved_config=approved, executed_config=executed,
                 pilot_provenance=(pilot.audit._sha(executed) + "  executed.expanded.fcl\n").encode(),
                 pilot_config=("====\nProcessing file: /fixture/pilot.art\n# header\n\n" + config).encode())
    blobs["pilot_inputs"] = (proof["copy_sha256"] + "  " + proof["pilot_pool_path"] + "\n" +
                             pilot.audit._sha(approved) + "  " + proof["approved_config_path"] + "\n" +
                             "c" * 64 + "  /fixture/runtime\n" + "d" * 64 + "  /fixture/runner\n").encode()
    for name in ("pilot", "copy"):
        blobs[name + "_census"] = json.dumps(dict(events=n, expected_count=n, ordered_full_event_census=True,
            input_sha256=proof[name + "_sha256"], output_sha256=pilot.audit._sha(blobs[name + "_events"]))).encode()
    return blobs, proof


def test_full_pilot_identity_and_original_copy():
    blobs, proof = fixture()
    expected, ledger, report = pilot.validate_evidence(blobs, proof)
    assert len(expected["event"]) == report["events"] == 2000
    assert report["source_file_resolution"] == "verified_original_copy"
    assert report["actual_source_sequence_equals_full_copy"]
    assert not report["source_independence_certified"]
    assert np.all(ledger["parent_fid"] == "102778799")
    assert np.all(ledger["output_fid"] == "sha256-" + "a" * 64)
    np.testing.assert_array_equal(ledger["source_event"][:3], [2130, 2143, 2156])


@pytest.mark.parametrize("defect", ["count", "parent", "receipt", "copy-id", "source-uri", "filter",
    "renumber", "wrong-copy-input", "copy-failed", "copy-manifest", "pool-hash", "executed-config",
    "provenance", "pilot-pool", "skip", "missing-ids", "wrong-process", "incomplete-log", "double-input"])
def test_provenance_failures_are_not_zero_overlap(defect):
    blobs, proof = fixture()
    if defect == "count": proof["events"] = 1999
    elif defect == "parent": proof["parent_size"] += 1
    elif defect == "receipt": blobs["pilot_census"] = blobs["pilot_census"].replace(b'"expected_count": 2000', b'"expected_count": 1999')
    elif defect == "copy-id": blobs["printer"] = blobs["printer"].replace(b"event: 2130 )", b"event: 9999 )", 1)
    elif defect == "source-uri": proof["original_source_uri"] += "other"
    elif defect in ("filter", "renumber"):
        blobs["copy_config"] = blobs["copy_config"].replace(b'module_type: "RootInput"', b'module_type: "RootInput" firstEvent: 1')
    elif defect == "wrong-copy-input": blobs["copy_log"] = blobs["copy_log"].replace(b"xroot://server", b"xroot://wrong")
    elif defect == "copy-failed": blobs["copy_log"] = blobs["copy_log"].replace(b"status 0", b"status 1")
    elif defect == "copy-manifest": blobs["copy_manifest"] = b"c" * 64 + b"  stop_pool.art\n"
    elif defect == "pool-hash": blobs["pilot_inputs"] = blobs["pilot_inputs"].replace(b"b" * 64, b"c" * 64)
    elif defect == "executed-config": blobs["executed_config"] += b"wrong: true\n"
    elif defect == "provenance": blobs["pilot_provenance"] = b"f" * 64 + b"  executed.expanded.fcl\n"
    elif defect == "pilot-pool": blobs["pilot_config"] = blobs["pilot_config"].replace(b"/fixture/test/", b"/fixture/other/")
    elif defect == "skip": blobs["pilot_config"] = blobs["pilot_config"].replace(b"MaxEventsToSkip: 0", b"MaxEventsToSkip: 10")
    elif defect == "missing-ids": blobs["pilot_config"] = blobs["pilot_config"].replace(b"writeEventIDs: true", b"writeEventIDs: false")
    elif defect == "wrong-process": blobs["printer"] = blobs["printer"].replace(b"'CaloLibraryCE'", b"'Primary'")
    elif defect == "incomplete-log": blobs["printer"] = blobs["printer"].split(b"TrigReport")[0]
    else: blobs["copy_log"] = b'Opened input file "other.art"\n' + blobs["copy_log"]
    with pytest.raises(ValueError):
        pilot.validate_evidence(blobs, proof)


def persisted(tmp_path):
    blobs, proof = fixture()
    for name, blob in blobs.items():
        (tmp_path / name).write_bytes(blob)
    proof["evidence"] = {name: dict(path=name, sha256=pilot.audit._sha(blob)) for name, blob in blobs.items()}
    old, lib = tmp_path / "old.root", tmp_path / "library.npz"
    with uproot.recreate(old) as output:
        output.mktree("dump/events", pilot._npz(blobs["pilot_events"]))
    np.savez(lib, entry_event_row=np.arange(505))
    proof.update(old_tables_sha256=pilot.audit.file_digest(old), library_sha256=pilot.audit.file_digest(lib),
                 donors=505, split_row=1000)
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    return path, old, lib


def test_frozen_donors_bound_to_full_old_event_order(tmp_path):
    proof, old, lib = persisted(tmp_path)
    expected, ledger, rows, report = pilot.validate(proof, tmp_path, old, lib)
    assert report["donor_events"] == len(rows) == 505
    assert report["old_event_rows_exact"]
    out = tmp_path / "bound"
    args = ["--proof", str(proof), "--base", str(tmp_path), "--old-tables", str(old), "--library", str(lib), "--out", str(out)]
    pilot.main(args)
    assert (out / "donor_ledger.npz").exists()
    with pytest.raises(FileExistsError): pilot.main(args)


@pytest.mark.parametrize("defect", ["row-range", "duplicate", "float", "old-order", "evidence-hash", "library-hash"])
def test_frozen_artifact_and_row_binding_rejects_mutations(tmp_path, defect):
    path, old, lib = persisted(tmp_path)
    proof = json.loads(path.read_text())
    if defect in ("row-range", "duplicate", "float"):
        rows = np.arange(505, dtype=float if defect == "float" else int)
        rows[-1] = 1000 if defect == "row-range" else 0 if defect == "duplicate" else 504
        np.savez(lib, entry_event_row=rows)
        proof["library_sha256"] = pilot.audit.file_digest(lib)
    elif defect == "old-order":
        events = pilot._npz((tmp_path / "pilot_events").read_bytes())
        with uproot.recreate(old) as output:
            output.mktree("dump/events", {k: v[::-1] for k, v in events.items()})
        proof["old_tables_sha256"] = pilot.audit.file_digest(old)
    elif defect == "evidence-hash": (tmp_path / "copy_source").write_text("changed")
    else: proof["library_sha256"] = "f" * 64
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError): pilot.validate(path, tmp_path, old, lib)
