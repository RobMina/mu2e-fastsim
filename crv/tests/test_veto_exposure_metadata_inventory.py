"""Synthetic saved SAM + real Parquet metadata fixtures, not retained-data evidence."""
import base64
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crvfastsim.veto import exposure_metadata_inventory as I


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value))
    return sha(path)


def forbidden(*args, **kwargs):
    pytest.fail("selected tracks, payload decoding, namespace discovery or prior audit executed")


class Saved:
    def __init__(self, base):
        self.root = base / "metadata"
        self.root.mkdir()
        self.selected = base / "unread_selection" / "selected.parquet"
        self.selected.parent.mkdir()
        self.selected.write_bytes(b"deliberately invalid selection: never open")
        self.ledger = self.root / "selection_window_v2.parquet"
        self.table = pa.table(dict(run=[1, 1], subrun=[2, 2], event=[3, 4],
                                  nprotons=[1.0, 2.0], pbtime=[0.0, 0.0],
                                  measured_nprotons=[-1.0, -1.0]))
        pq.write_table(self.table, self.ledger, row_group_size=1)
        self.provenance = {"sources": {}, "parent_overlaps": [{"inherited_only": True}]}
        self.paths = []
        for sequence in range(744, 756):
            name = f"dts.mu2e.NeutralsFlashCat.MDC2025ad.001430_{sequence:08d}.art"
            parents = [f"dts.mu2e.NeutralsFlash.MDC2025ac.001430_{j:08d}.art" for j in range(5)]
            text = (f"                 File Name: {name}\n"
                    f"                   File Id: {sequence}\n"
                    "               Event Count: 137\n"
                    f"              dh.sequencer: 001430_{sequence:08d}\n"
                    "               dh.gencount: 6700000000\n"
                    "          dh.configuration: synthetic:configuration\n"
                    "                dh.dataset: synthetic_dataset\n"
                    f"                 dh.sha256: {'a' * 64}\n"
                    "                   Comment: synthetic saved metadata only\n"
                    "                              continued prose without a colon\n"
                    "                   Parents:\n" + "".join(f"                              {p}\n" for p in reversed(parents)))
            path = self.root / f"sam_{sequence}.txt"
            path.write_text(text)
            self.paths.append(path)
            self.provenance["sources"][name] = {"sam": {
                "file_name": name, "file_id": str(sequence), "source_sha256": "a" * 64,
                "event_count": 137, "metadata_sha256": sha(path), "parent_count": len(parents),
                "parents_sha256": hashlib.sha256(("\n".join(parents) + "\n").encode()).hexdigest()}}
        self.selection = dict(selection=dict(path=str(self.selected), sha256="d" * 64, fields=["t_mid"], has_pbi=False),
            ntuples=[{"path": "/mnt/pnfs/forbidden/input.root", "trees": {"never_read": {}}}],
            exposure_ledger=dict(path=str(self.ledger), sha256=sha(self.ledger)),
            diagnostic={"pbi_mean_ntuple_events": 1.5, "scope": "synthetic MC diagnostic"},
            measured_field_checks={"nprotons": {"min": -1.0, "max": -1.0, "equals_mc": False}})
        self.args = dict(root=self.root, out=base / "inventory.json")
        self.repin()

    def repin(self):
        self.args["expected_provenance_sha256"] = dump(self.root / "provenance_final.json", self.provenance)
        self.args["expected_selection_sha256"] = dump(self.root / "selection_window_v2.json", self.selection)

    @property
    def first(self):
        return next(iter(self.provenance["sources"].values()))["sam"]

    def mutate_sam(self, before, after):
        path = self.paths[0]
        path.write_text(path.read_text().replace(before, after))
        self.first["metadata_sha256"] = sha(path)
        self.repin()

    def run(self, **changes):
        return I.run(**(self.args | changes))


@pytest.fixture
def saved(tmp_path):
    return Saved(tmp_path)


def test_inventory_real_footer_only_synthetic(saved, monkeypatch):
    # Explicitly forbid every old execution entrypoint or payload reader in reach.
    for module, names in ((I.C, ("run", "refit_admission", "saved_grid", "evaluate", "saved_products")),
                           (I.C.N, ("run", "authority", "dispatch")),
                           (I.C.R, ("verify",)), (I.C.M, ("run",))):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(pq.ParquetFile, "read", forbidden)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", forbidden)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    original_open, original_stat = Path.open, Path.stat
    def checked_open(path, *args, **kwargs):
        assert path != saved.selected and "pnfs" not in path.parts
        return original_open(path, *args, **kwargs)
    def checked_stat(path, *args, **kwargs):
        assert path != saved.selected and "pnfs" not in path.parts
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(Path, "stat", checked_stat)
    report = saved.run()
    assert len(report["sources"]) == 12
    assert all(r["gencount_catalog_observation"] == 6700000000 for r in report["sources"].values())
    row = next(iter(report["sources"].values()))
    assert row["scalar_fields"]["dh.configuration"] == "synthetic:configuration"
    assert row["scalar_fields"]["dh.dataset"] == "synthetic_dataset"
    assert row["scalar_fields"]["Comment"].endswith("continued prose without a colon")
    assert "Parents" not in row["scalar_fields"]
    ledger = report["exposure_ledger"]
    assert (ledger["nrows"], ledger["ncols"], ledger["row_groups"]) == (2, 6, 2)
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(ledger["arrow_schema_serialized_base64"])))
    assert schema.equals(saved.table.schema)
    assert ledger["per_row_source_identity"]["available_schema_field_names"] == saved.table.schema.names
    assert report["original_selection_report"]["saved_fields"] == saved.selection
    assert "inherited" in report["original_selection_report"]["status"]
    assert report["interpretation"]["validated_measured_pbi"] == I.NOT_ESTABLISHED
    assert report["weight_model"] is None and report["F"] is None
    assert str(saved.selected) not in report["inputs"]
    assert str(saved.ledger) in report["inputs"]
    assert saved.args["out"].stat().st_size < I.OUTPUT_CAP
    for path, digest in report["code_sha256"].items():
        assert hashlib.sha256(report["code_snapshots"][path].encode()).hexdigest() == digest


@pytest.mark.parametrize("pin", ["expected_provenance_sha256", "expected_selection_sha256"])
@pytest.mark.parametrize("value", ["bad", "f" * 64])
def test_bad_external_pins(saved, pin, value):
    with pytest.raises(ValueError):
        saved.run(**{pin: value})
    assert not saved.args["out"].exists()


@pytest.mark.parametrize("field,value", [("metadata_sha256", "f" * 64), ("parents_sha256", "e" * 64),
    ("parent_count", 4), ("parent_count", True), ("event_count", 136), ("file_name", "wrong.art"),
    ("file_id", "different"), ("source_sha256", "b" * 64)])
def test_bad_compact_bindings(saved, field, value):
    saved.first[field] = value
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


@pytest.mark.parametrize("value", ["-1", "0", "1.5", "nan", "inf", "1e9", str(2**63), ""])
def test_gencount_domain(saved, value):
    saved.mutate_sam("dh.gencount: 6700000000", "dh.gencount: " + value)
    with pytest.raises(ValueError, match="integer domain"):
        saved.run()


@pytest.mark.parametrize("before,after", [("dh.sequencer: 001430_00000744", "dh.sequencer: 001430_00000745"),
    ("001430_00000744.art", "001430_00000999.art"),
    ("dh.dataset: synthetic_dataset", "dh.dataset: a\n dh.dataset: b"),
    ("dh.gencount: 6700000000", "dh.gencount: 6700000000\n dh.gencount: 2"),
    ("Event Count: 137", "Event Count: -1")])
def test_filename_sequencer_duplicates_and_counts(saved, before, after):
    saved.mutate_sam(before, after)
    with pytest.raises(ValueError):
        saved.run()


def test_full_5000_parent_binding_and_no_overlap_analysis(saved):
    parents = [f"dts.mu2e.NeutralsFlash.MDC2025ac.001430_{i:08d}.art" for i in range(5000)]
    path = saved.paths[0]
    prefix = path.read_text().split("                   Parents:")[0]
    path.write_text(prefix + " Parents:\n" + "".join("    " + p + "\n" for p in reversed(parents)))
    saved.first.update(parent_count=5000, metadata_sha256=sha(path),
                       parents_sha256=hashlib.sha256(("\n".join(parents) + "\n").encode()).hexdigest())
    saved.repin()
    report = saved.run()
    assert next(iter(report["sources"].values()))["parent_count"] == 5000
    assert report["parent_overlap_recomputed"] is False


def test_parent_line_cap_before_comparison(saved):
    path = saved.paths[0]
    prefix = path.read_text().split("                   Parents:")[0]
    path.write_text(prefix + " Parents:\n" + "".join(f"    p{i}.art\n" for i in range(5001)))
    saved.first["metadata_sha256"] = sha(path)
    saved.repin()
    with pytest.raises(ValueError, match="parent count cap"):
        saved.run()


def test_duplicate_parent(saved):
    saved.mutate_sam("001430_00000001.art", "001430_00000000.art")
    with pytest.raises(ValueError, match="duplicate SAM parents"):
        saved.run()


@pytest.mark.parametrize("target,cap", [("sam", I.SAM_CAP), ("json", I.JSON_CAP)])
def test_byte_caps_before_hash_or_decode(saved, monkeypatch, target, cap):
    path = saved.paths[0] if target == "sam" else saved.root / "provenance_final.json"
    with path.open("wb") as stream:
        stream.truncate(cap)
    monkeypatch.setattr(I.C, "file_sha", forbidden)
    monkeypatch.setattr(I, "parse_sam", forbidden)
    with pytest.raises(ValueError, match="byte cap"):
        saved.run()


def test_total_cap_before_any_hash(saved, monkeypatch):
    monkeypatch.setattr(I, "TOTAL_CAP", 10000)
    monkeypatch.setattr(I.C, "file_sha", forbidden)
    with pytest.raises(ValueError, match="total byte cap"):
        saved.run()


def test_ledger_file_cap_before_footer(saved, monkeypatch):
    with saved.ledger.open("wb") as stream:
        stream.truncate(I.LEDGER_CAP + 1)
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    with pytest.raises(ValueError, match="byte cap"):
        saved.run()


@pytest.mark.parametrize("limit,value,message", [("MAX_ROWS", 1, "row cap"),
    ("MAX_COLUMNS", 5, "column cap"), ("MAX_GROUPS", 1, "row-group cap"),
    ("MAX_UNCOMPRESSED", 1, "uncompressed cap")])
def test_ledger_footer_caps_without_data_decode(saved, monkeypatch, limit, value, message):
    monkeypatch.setattr(I, limit, value)
    monkeypatch.setattr(pq.ParquetFile, "read", forbidden)
    with pytest.raises(ValueError, match=message):
        saved.run()


def test_ledger_bad_pin_before_footer(saved, monkeypatch):
    saved.selection["exposure_ledger"]["sha256"] = "f" * 64
    saved.repin()
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    with pytest.raises(ValueError, match="changed bound"):
        saved.run()


def test_duplicate_json(saved):
    path = saved.root / "provenance_final.json"
    path.write_text('{"sources":{},"sources":{}}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        saved.run()


@pytest.mark.parametrize("mutation", ["sam", "ledger", "provenance", "selection", "code"])
def test_late_changes_fail_before_output(saved, monkeypatch, tmp_path, mutation):
    code = tmp_path / "code" / "snapshot.py"
    code.parent.mkdir()
    code.write_text("# synthetic code snapshot\n")
    monkeypatch.setattr(I, "code_paths", lambda: [code])
    original = I.ledger_metadata
    def changing(path):
        result = original(path)
        target = {"sam": saved.paths[0], "ledger": saved.ledger,
                  "provenance": saved.root / "provenance_final.json",
                  "selection": saved.root / "selection_window_v2.json", "code": code}[mutation]
        with target.open("ab") as stream:
            stream.write(b"\n")
        return result
    monkeypatch.setattr(I, "ledger_metadata", changing)
    with pytest.raises(ValueError, match="changed"):
        saved.run()
    assert not saved.args["out"].exists()


def test_overwrite_and_late_destination(saved, monkeypatch):
    out = saved.args["out"]
    out.write_text("keep")
    with pytest.raises(ValueError, match="overwrite"):
        saved.run()
    assert out.read_text() == "keep"
    out.unlink()
    original = I.ledger_metadata
    def create(path):
        result = original(path)
        out.write_text("late winner")
        return result
    monkeypatch.setattr(I, "ledger_metadata", create)
    with pytest.raises(ValueError, match="overwrite"):
        saved.run()
    assert out.read_text() == "late winner"


@pytest.mark.parametrize("place", ["root", "selection", "code"])
def test_output_outside_protected_parents(saved, place):
    parent = {"root": saved.root, "selection": saved.selected.parent,
              "code": Path(I.__file__).parent}[place]
    with pytest.raises(ValueError, match="outside"):
        saved.run(out=parent / "new_inventory.json")


def test_output_cap(saved, monkeypatch):
    monkeypatch.setattr(I, "OUTPUT_CAP", 100000)
    saved.mutate_sam("Comment: synthetic saved metadata only", "Comment: " + "a" * 120000)
    with pytest.raises(ValueError, match="output byte cap"):
        saved.run()
    assert not saved.args["out"].exists()


@pytest.mark.parametrize("path", ["/mnt/pnfs/secret/ledger.parquet", "/elsewhere/not-in-root.parquet", "https://invalid/ledger.parquet"])
def test_no_external_ledger_access(saved, path, monkeypatch):
    saved.selection["exposure_ledger"]["path"] = path
    saved.repin()
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    with pytest.raises(ValueError):
        saved.run()


def test_selected_path_cannot_be_ledger(saved, monkeypatch):
    saved.selection["selection"]["path"] = str(saved.ledger)
    saved.repin()
    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    with pytest.raises(ValueError, match="must not be consumed"):
        saved.run()


def test_symlink_sam_rejected(saved, tmp_path):
    path = saved.paths[0]
    real = tmp_path / "external.txt"
    path.rename(real)
    path.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        saved.run()


def test_missing_fixed_file_does_not_search(saved, monkeypatch):
    saved.paths[0].rename(saved.root / "sam_000744.txt")
    monkeypatch.setattr(Path, "glob", forbidden)
    with pytest.raises(FileNotFoundError):
        saved.run()


def test_missing_gencount_is_available_field_observation_not_negative_claim(saved):
    saved.mutate_sam("               dh.gencount: 6700000000\n", "")
    report = saved.run()
    row = next(iter(report["sources"].values()))
    assert row["gencount_catalog_observation"] is None
    assert "dh.gencount" not in row["scalar_fields"]
    assert report["interpretation"]["gencount_semantics_and_additivity"] == I.NOT_ESTABLISHED


def test_cli(saved, capsys):
    I.main([str(saved.root), "--out", str(saved.args["out"]),
            "--expected-provenance-sha256", saved.args["expected_provenance_sha256"],
            "--expected-selection-sha256", saved.args["expected_selection_sha256"]])
    assert json.loads(capsys.readouterr().out)["schema"] == I.SCHEMA


@pytest.mark.parametrize("mode", ["missing", "ambiguous", "traversal"])
def test_exact_source_domain(saved, mode):
    names = list(saved.provenance["sources"])
    if mode == "missing":
        saved.provenance["sources"].pop(names[0])
    else:
        record = saved.provenance["sources"].pop(names[0])
        name = names[1].replace("001430", "001431") if mode == "ambiguous" else "../" + names[0]
        record["sam"]["file_name"] = name
        saved.provenance["sources"][name] = record
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


def test_all_scalar_fields_preserved_without_keyword_absence_inference(saved):
    saved.mutate_sam("Comment: synthetic saved metadata only",
                     "custom.reuse: intended as described elsewhere\n"
                     " custom.measured: documentation pointer only\n"
                     " deeper.parent: opaque saved scalar\n"
                     " Comment: synthetic saved metadata only")
    report = saved.run()
    row = next(iter(report["sources"].values()))
    assert row["scalar_fields"]["custom.reuse"] == "intended as described elsewhere"
    assert row["scalar_fields"]["custom.measured"] == "documentation pointer only"
    assert row["scalar_fields"]["deeper.parent"] == "opaque saved scalar"
    assert report["interpretation"]["intended_reuse_declaration"] == I.NOT_ESTABLISHED


def test_evidence_final_recheck_after_serialization(saved, monkeypatch):
    original = I.json.dumps
    def mutate(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(value, dict) and value.get("schema") == I.SCHEMA:
            with saved.paths[0].open("ab") as stream:
                stream.write(b"\n")
        return result
    monkeypatch.setattr(I.json, "dumps", mutate)
    with pytest.raises(ValueError, match="changed bound"):
        saved.run()
    assert not saved.args["out"].exists()


def test_exclusive_create_survives_late_destination_race(saved, monkeypatch):
    original = I.C.Evidence.recheck
    checks = 0
    def competing_writer(evidence):
        nonlocal checks
        original(evidence)
        checks += 1
        if checks == 2:
            saved.args["out"].write_text("competing exclusive publication")
    monkeypatch.setattr(I.C.Evidence, "recheck", competing_writer)
    with pytest.raises(FileExistsError):
        saved.run()
    assert saved.args["out"].read_text() == "competing exclusive publication"


def test_declared_footer_row_cap_real_parquet(saved, monkeypatch):
    pq.write_table(pa.table({"event": pa.array(range(I.MAX_ROWS + 1))}), saved.ledger)
    saved.selection["exposure_ledger"]["sha256"] = sha(saved.ledger)
    saved.repin()
    monkeypatch.setattr(pq.ParquetFile, "read", forbidden)
    with pytest.raises(ValueError, match="row cap"):
        saved.run()


def test_declared_footer_column_cap_real_parquet(saved, monkeypatch):
    pq.write_table(pa.table({f"field{i}": [1] for i in range(I.MAX_COLUMNS + 1)}), saved.ledger)
    saved.selection["exposure_ledger"]["sha256"] = sha(saved.ledger)
    saved.repin()
    monkeypatch.setattr(pq.ParquetFile, "read", forbidden)
    with pytest.raises(ValueError, match="column cap"):
        saved.run()


def test_actual_sam_checksum_continuation_preserved(saved):
    saved.mutate_sam("               Event Count: 137", "                 Checksum: enstore:632078868\n"
                     "                           adler32:525ec215\n               Event Count: 137")
    report = saved.run()
    fields = next(iter(report["sources"].values()))["scalar_fields"]
    assert fields["Checksum"] == "enstore:632078868\nadler32:525ec215"
    assert "adler32" not in fields
    assert fields["dh.sequencer"] == "001430_00000744"


@pytest.mark.parametrize("sequencer", ["744", "00000744", "001431_00000744"])
def test_full_sequencer_cannot_be_replaced_by_partial_or_wrong_run(saved, sequencer):
    saved.mutate_sam("dh.sequencer: 001430_00000744", "dh.sequencer: " + sequencer)
    with pytest.raises(ValueError, match="sequencer"):
        saved.run()


def test_footer_reader_uses_portable_constructor_keywords(saved, monkeypatch):
    original = I.pq.ParquetFile
    calls = []
    def portable(path, *, thrift_string_size_limit, thrift_container_size_limit, pre_buffer):
        calls.append(path)
        return original(path, thrift_string_size_limit=thrift_string_size_limit,
                        thrift_container_size_limit=thrift_container_size_limit, pre_buffer=pre_buffer)
    monkeypatch.setattr(I.pq, "ParquetFile", portable)
    result = saved.run()
    assert calls == [saved.ledger]
    assert result["exposure_ledger"]["table_decoded"] is False
