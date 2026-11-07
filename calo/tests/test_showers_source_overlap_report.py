"""Synthetic exact-key oracle; no remote access, real provenance, or art execution."""
from collections import Counter
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from calofastsim.showers import source_audit as audit, source_overlap_report as overlap, source_split


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def alter_json(path, **changes):
    value = json.loads(Path(path).read_text())
    value.update(changes)
    write_json(path, value)


def tables(fid, subrun, sources, run=1201):
    n = len(sources)
    expected = dict(output_fid=np.full(n, fid), run=np.full(n, run, np.int64),
                    subrun=np.full(n, subrun, np.int64), event=np.arange(1, n + 1, dtype=np.int64))
    ledger = dict(expected, parent_fid=np.full(n, audit.PARENT_FID),
                  source_run=np.full(n, 1430, np.int64), source_subrun=np.zeros(n, np.int64),
                  source_event=np.asarray(sources, dtype=np.int64), source_count=np.ones(n, np.int64),
                  source_present=np.ones(n, bool), source_file_resolved=np.ones(n, bool))
    return expected, ledger


def save_bound(path, expected, ledger, summary):
    path.mkdir(parents=True)
    np.savez(path / "expected.npz", **expected)
    np.savez(path / "ledger.npz", **ledger)
    summary["output_sha256"] = {name: audit.file_digest(path / name) for name in ("expected.npz", "ledger.npz")}
    write_json(path / "summary.json", summary)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: pytest.fail("remote/subprocess access"))
    monkeypatch.setattr(audit.uproot, "open", lambda *a, **k: pytest.fail("art reread"))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    parent_path = metadata / "parent.json"
    write_json(parent_path, dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID)))
    parent = dict(name=audit.PARENT_NAME, fid=audit.PARENT_FID, did="mu2e:" + audit.PARENT_NAME,
                  metadata_filename=parent_path.name, metadata_sha256=audit.file_digest(parent_path))
    rows = []
    for i in range(20):
        name = f"dts.mu2e.Synthetic.Test.001201_{i:08d}.art"
        frozen = dict(name=name, did="mu2e:" + name, fid=str(1000 + i), size=42,
                      checksums={"enstore": "42"}, role="training" if i < 10 else "development",
                      ordinal=i % 10 + 1, run=1201, subrun=i)
        md = dict(file_name=name, file_id=1000 + i, file_size=42, checksum=["enstore:42"],
                  event_count=3, parents=[dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID))],
                  locations=[dict(full_path="enstore:/data/synthetic/" + name)], **{"dh.sha256": "a" * 64})
        path = metadata / (name + ".json")
        write_json(path, md)
        rows.append(dict(audit.validate_metadata(md, frozen), metadata_filename=path.name,
                         metadata_sha256=audit.file_digest(path)))
    inventory = metadata / "inventory.json"
    write_json(inventory, dict(schema_version=1, rows=rows, original_parent=parent,
                              parent_metadata_sha256=parent["metadata_sha256"], manifest_sha256="b" * 64))
    production, pilot = tmp_path / "production", tmp_path / "pilot"
    directories = []
    for i, row in enumerate(rows):
        sources = {0: [1, 1, 2], 1: [2, 5, 6], 10: [2, 2, 3]}.get(i, [100 + i * 3, 101 + i * 3, 102 + i * 3])
        expected, ledger = tables(row["fid"], i, sources)
        hashes = dict(inventory=audit.file_digest(inventory), metadata=row["metadata_sha256"],
                      parent_metadata=parent["metadata_sha256"], expected_events="c" * 64,
                      printer_log="d" * 64, config="e" * 64, census_receipt="f" * 64)
        summary = dict(item=row["name"], role=row["role"], ordinal=row["ordinal"], events=3, source_rows=3,
                       parent_fid=audit.PARENT_FID, manifest_sha256="b" * 64, art_sha256="a" * 64,
                       ordered_full_census_exact=True, census_receipt_checked=True, input_sha256=hashes,
                       source_independence_certified=False)
        path = production / row["name"] / "bound"
        save_bound(path, expected, ledger, summary)
        directories.append(path)
    expected, ledger = tables("sha256-" + "9" * 64, 0, np.r_[2, 3, np.arange(100000, 101998)], run=9000)
    summary = dict(item="legacy-pilot.art", role="pilot", events=2000, source_rows=2000,
                   output_fid="sha256-" + "9" * 64, art_sha256="9" * 64, parent_fid=audit.PARENT_FID,
                   source_file_resolution="verified_original_copy", ordered_full_census_exact=True,
                   input_sha256={"original_copy": "8" * 64}, source_independence_certified=False)
    save_bound(pilot, expected, ledger, summary)
    file0 = tmp_path / "source_events.npz"
    with np.load(directories[10] / "ledger.npz") as archive:
        np.savez(file0, **{field: archive[field] for field in
                          ("run", "subrun", "event", "source_run", "source_subrun", "source_event", "source_count")})
    return dict(inventory=inventory, production=production, pilot=pilot, file0_sources=file0,
                out=tmp_path / "report", dirs=directories, rows=rows)


def run(bundle):
    return overlap.report(**{key: bundle[key] for key in
                             ("inventory", "production", "pilot", "file0_sources", "out")})


def change_npz(path, update):
    with np.load(path, allow_pickle=False) as archive:
        table = dict(archive)
    update(table)
    np.savez(path, **table)


def oracle_keys(table):
    return list(zip(*(table[field].tolist() for field in source_split.PARENT_KEY)))


def test_exact_oracle_reuse_all_pairs_and_hashes(bundle):
    result = run(bundle)
    assert result["production_files"] == 20
    assert result["production_expected_events"] == 60
    assert result["pilot_reference_events"] == 2000
    assert result["all_complete_expected_events"] == 2060
    assert result["cross_file_pairs_checked"] == 210
    assert result["file0_reuse_parity"]["source_ids_exact"]
    assert result["file0_reuse_parity"]["art_reread"] is False
    assert result["source_independence_certified"] is False and result["no_pruning"]
    keys = {}
    for role in overlap.ROLES:
        with np.load(bundle["out"] / role / "ledger.npz") as archive:
            keys[role] = oracle_keys(archive)
        counter = Counter(keys[role])
        stats = result["roles"][role]
        assert stats["unique_parent_events"] == len(counter)
        assert stats["reused_parent_events"] == sum(n > 1 for n in counter.values())
        assert stats["repeated_output_events"] == sum(n - 1 for n in counter.values())
    for name, pair in result["role_pairs"].items():
        left, right = name.split("--")
        shared = set(keys[left]) & set(keys[right])
        assert pair["shared_parent_events"] == len(shared)
        assert pair["left_events_with_shared_parent"] == sum(key in shared for key in keys[left])
        assert pair["right_events_with_shared_parent"] == sum(key in shared for key in keys[right])
        with np.load(bundle["out"] / pair["shared_event_families_npz"]) as archive:
            assert set(oracle_keys(archive)) == shared
    per_file_keys = []
    for path in bundle["dirs"] + [bundle["pilot"]]:
        with np.load(path / "ledger.npz") as archive:
            per_file_keys.append(oracle_keys(archive))
    collided = {(pair["left"], pair["right"]): pair
                for pair in result["cross_file_pairs_with_parent_overlap"]}
    expected_pairs = 0
    for i, left in enumerate(per_file_keys):
        for j in range(i + 1, len(per_file_keys)):
            right = per_file_keys[j]
            shared = set(left) & set(right)
            names = (result["per_file"][i]["item"], result["per_file"][j]["item"])
            if not shared:
                assert names not in collided
                continue
            expected_pairs += 1
            assert collided[names]["shared_parent_events"] == len(shared)
            assert collided[names]["left_events_with_shared_parent"] == sum(key in shared for key in left)
            assert collided[names]["right_events_with_shared_parent"] == sum(key in shared for key in right)
    assert expected_pairs == len(collided) == 6
    assert result["per_file"][0]["reused_parent_events"] == 1
    assert result["per_file"][10]["repeated_output_events"] == 1
    for path, sha in result["input_sha256"].items():
        assert audit.file_digest(path) == sha
    for path, sha in result["output_sha256"].items():
        assert audit.file_digest(bundle["out"] / path) == sha


def test_full_keys_and_valid_zero(bundle):
    # Identical event numbers cannot collide across different source run/subrun.
    for i, path in enumerate(bundle["dirs"] + [bundle["pilot"]]):
        change_npz(path / "ledger.npz", lambda t, i=i: t.update(source_run=np.full(len(t["event"]), i + 1)))
        summary = json.loads((path / "summary.json").read_text())
        summary["output_sha256"]["ledger.npz"] = audit.file_digest(path / "ledger.npz")
        write_json(path / "summary.json", summary)
    change_npz(bundle["file0_sources"], lambda t: t.update(source_run=np.full(3, 11)))
    result = run(bundle)
    assert result["cross_file_pairs_with_parent_overlap"] == []
    assert result["cross_file_pairs_without_parent_overlap"] == 210
    for pair in result["role_pairs"].values():
        assert pair["shared_parent_events"] == 0 and pair["parent_events_disjoint"]
        assert pair["source_independence_certified"] is False
        with np.load(bundle["out"] / pair["shared_event_families_npz"]) as archive:
            assert len(archive["parent_fid"]) == 0


@pytest.mark.parametrize("filename", ["expected.npz", "ledger.npz", "summary.json"])
def test_missing_file(bundle, filename):
    (bundle["dirs"][19] / filename).unlink()
    with pytest.raises(FileNotFoundError):
        run(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("changes", [dict(events=2), dict(events=3.0), dict(role="confirmation"),
    dict(item="wrong"), dict(parent_fid="999"), dict(manifest_sha256="0" * 64),
    dict(census_receipt_checked=False), dict(census_receipt_checked=1),
    dict(source_rows=4), dict(ordered_full_census_exact=False), dict(input_sha256={}),
    dict(source_independence_certified=True)])
def test_binding_summary_fail_closed(bundle, changes):
    alter_json(bundle["dirs"][0] / "summary.json", **changes)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("defect", ["wrong-fid", "wrong-parent", "count", "resolved", "present", "incomplete", "order"])
def test_bound_table_fail_closed(bundle, defect):
    def update(t):
        if defect == "wrong-fid": t["output_fid"] = np.full(3, "999")
        elif defect == "wrong-parent": t["parent_fid"] = np.full(3, "999")
        elif defect == "count": t["source_count"][1] = 2
        elif defect == "resolved": t["source_file_resolved"][1] = False
        elif defect == "present": t["source_present"][1] = False
        elif defect == "incomplete":
            for key in t: t[key] = t[key][:-1]
        else:
            for key in t: t[key] = t[key][::-1]
    change_npz(bundle["dirs"][0] / "ledger.npz", update)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("defect", ["count", "role", "duplicate", "changed-metadata", "bad-metadata-hash", "identity"])
def test_inventory_fail_closed(bundle, defect):
    inv = json.loads(bundle["inventory"].read_text())
    if defect == "count": inv["rows"].pop()
    elif defect == "role": inv["rows"][0]["role"] = "confirmation"
    elif defect == "duplicate": inv["rows"][1] = inv["rows"][0]
    elif defect == "identity": inv["rows"][0]["event_count"] = 3.0
    elif defect == "bad-metadata-hash": inv["rows"][0]["metadata_sha256"] = "0" * 64
    else:
        path = bundle["inventory"].parent / inv["rows"][0]["metadata_filename"]
        alter_json(path, event_count=4)
    write_json(bundle["inventory"], inv)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("changes", [dict(events=1999), dict(output_fid="1234"),
    dict(art_sha256="a" * 64), dict(parent_fid="999"), dict(parent_fid=int(audit.PARENT_FID)),
    dict(source_file_resolution="unverified")])
def test_pilot_contract(bundle, changes):
    alter_json(bundle["pilot"] / "summary.json", **changes)
    with pytest.raises(ValueError):
        run(bundle)
    assert not bundle["out"].exists()


@pytest.mark.parametrize("field", ["event", "subrun", "run", "source_event", "source_subrun", "source_run", "source_count"])
def test_file0_parity_rejects_changes(bundle, field):
    change_npz(bundle["file0_sources"], lambda t: t[field].__setitem__(0, t[field][0] + 1))
    with pytest.raises(ValueError, match="file0"):
        run(bundle)


def test_input_mutation_rejected(bundle, monkeypatch):
    original = overlap.Inputs.unchanged
    def changed(self):
        path = bundle["dirs"][0] / "summary.json"
        path.write_text(path.read_text() + "\n")
        original(self)
    monkeypatch.setattr(overlap.Inputs, "unchanged", changed)
    with pytest.raises(ValueError, match="changed during audit"):
        run(bundle)
    assert not bundle["out"].exists()


def test_no_overwrite_or_dangling_symlink(bundle):
    bundle["out"].symlink_to(bundle["out"].parent / "missing")
    with pytest.raises(FileExistsError):
        run(bundle)
    bundle["out"].unlink()
    bundle["out"].mkdir()
    with pytest.raises(FileExistsError):
        run(bundle)


def test_duplicate_json_and_npz_schema(bundle):
    path = bundle["dirs"][0] / "summary.json"
    path.write_text(path.read_text().replace('"events": 3', '"events": 3, "events": 3'))
    with pytest.raises(ValueError, match="duplicate JSON"):
        run(bundle)


def test_changed_inventory_proof_hash(bundle):
    path = bundle["dirs"][0] / "summary.json"
    value = json.loads(path.read_text())
    value["input_sha256"]["inventory"] = "0" * 64
    write_json(path, value)
    with pytest.raises(ValueError, match="inventory mismatch"):
        run(bundle)


def test_cli(bundle, capsys):
    argv = []
    for key in ("inventory", "production", "pilot", "file0_sources", "out"):
        argv.extend(["--" + key.replace("_", "-"), str(bundle[key])])
    overlap.main(argv)
    result = json.loads(capsys.readouterr().out)
    assert result["all_complete_expected_events"] == 2060


def test_production_uses_no_event_iteration():
    tree = ast.parse(inspect.getsource(overlap))
    # Every loop is a schema, file, role, metadata or digest loop; no range over events,
    # conversion to Python records, vectorize, or per-event callback machinery.
    forbidden = {"tolist", "vectorize", "frompyfunc", "iterrows", "apply_along_axis"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree))
    loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.comprehension))]
    assert all("range(" not in ast.unparse(node.iter) for node in loops)
    assert all(ast.unparse(node.iter) not in {"ledger", "expected", "parents", "shared", "a", "b"}
               for node in loops)


def test_consistently_wrong_row_fid_is_not_inferred(bundle):
    for name in ("expected.npz", "ledger.npz"):
        change_npz(bundle["dirs"][0] / name, lambda t: t.update(output_fid=np.full(3, "9876")))
    path = bundle["dirs"][0]
    summary = json.loads((path / "summary.json").read_text())
    summary["output_sha256"] = {name: audit.file_digest(path / name) for name in ("expected.npz", "ledger.npz")}
    write_json(path / "summary.json", summary)
    with pytest.raises(ValueError, match="FID mismatch"):
        run(bundle)


@pytest.mark.parametrize("defect", ["missing", "unresolved", "multiple", "truncated", "sha-fid"])
def test_full_pilot_binding_required(bundle, defect):
    path = bundle["pilot"] / "ledger.npz"
    if defect == "missing":
        path.unlink()
    else:
        def update(t):
            if defect == "unresolved": t["source_file_resolved"][-1] = False
            elif defect == "multiple": t["source_count"][-1] = 2
            elif defect == "sha-fid": t["output_fid"] = np.full(2000, "sha256-" + "8" * 64)
            else:
                for field in t: t[field] = t[field][:-1]
        change_npz(path, update)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(bundle)
    assert not bundle["out"].exists()


def test_npz_schema_rejects_unexpected_field(bundle):
    change_npz(bundle["dirs"][0] / "ledger.npz", lambda t: t.update(unexpected=np.zeros(3)))
    with pytest.raises(ValueError, match="schema mismatch"):
        run(bundle)


def test_mutation_during_output_never_publishes_success_summary(bundle, monkeypatch):
    original = overlap._save_npz
    def mutate(path, table):
        original(path, table)
        if Path(path).name == "ledger.npz":
            source = bundle["dirs"][0] / "summary.json"
            source.write_text(source.read_text() + "\n")
    monkeypatch.setattr(overlap, "_save_npz", mutate)
    with pytest.raises(ValueError, match="changed during audit"):
        run(bundle)
    assert not (bundle["out"] / "summary.json").exists()
    with pytest.raises(FileExistsError):
        run(bundle)


@pytest.mark.parametrize("replacement", ["coordinated-output", "source-ids", "pilot-source"])
def test_summary_authenticates_generated_bundle(bundle, replacement):
    path = bundle["pilot"] if replacement == "pilot-source" else bundle["dirs"][0]
    if replacement == "coordinated-output":
        for name in ("expected.npz", "ledger.npz"):
            change_npz(path / name, lambda t: t["event"].__setitem__(0, 99999))
    else:
        change_npz(path / "ledger.npz", lambda t: t["source_event"].__setitem__(0, 99999))
    with pytest.raises(ValueError, match="bound output hash"):
        run(bundle)
    assert not bundle["out"].exists()
