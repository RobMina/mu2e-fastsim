"""Synthetic-only v2 extraction/denominator tests; no art jobs or physics validation."""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import uproot

from crvfastsim.veto import catsteps_v2 as v2


def provenance(tag="a"):
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return {"source_art": {"identity": f"art:{tag}", "uri": f"/sources/{tag}.art", "sha256": digest(tag)},
            "extraction": {"id": f"extract:{tag}", "command": f"mu2e -c frozen.fcl -s /sources/{tag}.art -T dump.root",
                           "software_version": "synthetic-only", "fcl_sha256": digest("fcl"),
                           "dumper_sha256": digest("dumper"), "write_steps": True, "source_count": 1,
                           "event_selection": "all"}}


def tables():
    events = pd.DataFrame({"run": [1, 2, 1, 1], "subrun": [1, 1, 0, 2],
                           "event": [1, 1, 1048577, 1], "ngen": [1, 1, 0, 0], "nsteps": [2, 1, 1, 0]})
    gen = events.iloc[:2][v2.IDS].copy()
    gen["idx"], gen["pdg"] = 0, 13
    particles = events.iloc[:3][v2.IDS].copy()
    particles["key"], particles["pdg"], particles["genidx"] = 7, 13, [0, 0, -1]
    steps = events.iloc[[0, 0, 1, 2]][v2.IDS].reset_index(drop=True)
    steps["nstep"], steps["bar"], steps["sp"] = [2, 2, 1, 1], 2, 7
    steps["t0"], steps["t1"] = [np.nan, 10., 20., 30.], [2., 11., 21., 31.]
    for name in v2.F32:
        steps[name] = np.ones(len(steps), dtype=np.float32)
    return {"events": events, "gen": gen, "steps": steps, "particles": particles}


def dump(path, data):
    with uproot.recreate(path) as root:
        for name, frame in data.items():
            arrays = {key: frame[key].to_numpy() for key in frame}
            tree = root.mktree(f"dump/{name}", {key: value.dtype for key, value in arrays.items()})
            tree.extend(arrays)
    return path


def converted(tmp_path, tag="a", data=None):
    root = dump(tmp_path / f"{tag}.root", tables() if data is None else data)
    out = tmp_path / f"{tag}.parquet"
    meta = v2.convert(root, "ExampleCat", out, **provenance(tag))
    return out, meta


def rewrite(path, *, metadata=None, frame=None):
    table = pq.read_table(path)
    if frame is not None:
        import pyarrow as pa
        table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(table.schema.metadata)
    if metadata is not None:
        table = table.replace_schema_metadata({**table.schema.metadata, b"crvfastsim": json.dumps(metadata).encode()})
    pq.write_table(table, path)


def test_full_tuple_all_event_census_and_pool_contract(tmp_path):
    out, meta = converted(tmp_path)
    pool = v2.PoolV2([out])
    assert meta["schema"] == "crv-cat-v2"
    assert pool.frames == 4 and pool.stepped == 3
    assert len(pool.steps) == 4
    assert pool.ids.names == v2.IDENTITY
    assert len(set(pool.ids)) == 3
    assert pool.count.tolist() == [1, 2, 1]
    assert pool.start.tolist() == [0, 1, 3]
    assert pool.steps.fid.tolist() == [0, 1, 1, 2]
    assert ("art:a", 1, 0, 1048577) in pool.ids
    assert ("art:a", 2, 1, 1) in pool.ids
    assert pool.events.ngen.sum() == 2 and (pool.events.ngen == 0).sum() == 2
    assert len(pool.hash_groups["source_sha256"]) == 1
    assert pool.manifest[0]["file_sha256"] == v2.sha256(out)
    assert meta["dump"]["sha256"] == v2.sha256(tmp_path / "a.root")
    assert meta["source_art"]["uri"] == "/sources/a.art"
    assert np.isnan(pool.steps.t0).sum() == 1
    assert pool.steps.edep.dtype == np.float32


def test_empty_steps_with_nonempty_census_is_usable(tmp_path):
    data = tables()
    data["steps"] = data["steps"].iloc[:0]
    data["particles"] = data["particles"].iloc[:0]
    data["events"]["nsteps"] = 0
    out, meta = converted(tmp_path, data=data)
    pool = v2.PoolV2([out])
    assert pool.frames == 4 and pool.stepped == 0 and meta["steps"] == 0
    assert not len(pool.steps) and not len(pool.start) and not len(pool.count)
    assert set(v2.STEP_COLUMNS) <= set(pool.steps)
    assert pool.start.dtype == pool.count.dtype == np.int64


def test_same_event_tuple_distinct_sources_not_merged(tmp_path):
    first, _ = converted(tmp_path)
    data = tables()
    data["steps"]["edep"] = 2.
    second, _ = converted(tmp_path, "b", data)
    pool = v2.PoolV2([first, second])
    assert pool.frames == 8 and pool.stepped == 6
    assert pool.count.sum() == 8
    assert set(pool.events.source_identity) == {"art:a", "art:b"}


@pytest.mark.parametrize("tree", ["events", "gen", "steps", "particles"])
def test_required_trees(tmp_path, tree):
    data = tables()
    del data[tree]
    with pytest.raises(ValueError, match="required dump/"):
        converted(tmp_path, data=data)
    assert not (tmp_path / "a.parquet").exists()
    assert not (tmp_path / "a.events.parquet").exists()


@pytest.mark.parametrize("tree,column,value", [
    ("events", "event", 0), ("events", "run", 0), ("events", "subrun", -1),
    ("events", "event", 2**32 - 1), ("events", "event", 2**32),
    ("steps", "sp", -1), ("particles", "key", -1), ("gen", "idx", -1),
    ("events", "nsteps", -1), ("events", "ngen", -1), ("steps", "bar", -1),
])
def test_invalid_integer_ids_and_counts(tmp_path, tree, column, value):
    data = tables()
    data[tree].loc[data[tree].index[0], column] = value
    with pytest.raises(ValueError):
        converted(tmp_path, data=data)


def test_float_id_not_coerced(tmp_path):
    data = tables()
    data["events"]["event"] = data["events"].event.astype(float)
    with pytest.raises(ValueError, match="invalid integer"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("tree", ["events", "particles", "gen"])
def test_duplicate_keys_rejected_even_with_equal_pdg(tmp_path, tree):
    data = tables()
    data[tree] = pd.concat([data[tree], data[tree].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("tree", ["steps", "particles", "gen"])
def test_foreign_event_rejected(tmp_path, tree):
    data = tables()
    data[tree].loc[data[tree].index[0], "run"] = 99
    with pytest.raises(ValueError, match="missing from census"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("column", ["ngen", "nsteps"])
def test_census_count_disagreement(tmp_path, column):
    data = tables()
    data["events"].loc[0, column] += 1
    with pytest.raises(ValueError, match="disagrees with census"):
        converted(tmp_path, data=data)


def test_missing_particle_is_not_pdg_zero(tmp_path):
    data = tables()
    data["steps"].loc[0, "sp"] = 999
    with pytest.raises(ValueError, match="missing particle"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("mode", ["step_pdg", "generator_key", "nstep", "duplicate_pdg"])
def test_particle_reconciliation(tmp_path, mode):
    data = tables()
    if mode == "step_pdg":
        data["steps"]["pdg"] = 211
    elif mode == "generator_key":
        data["particles"].loc[0, "genidx"] = 10
    elif mode == "nstep":
        data["steps"].loc[0, "nstep"] = 100
    else:
        row = data["particles"].iloc[:1].copy()
        row["pdg"] = 211
        data["particles"] = pd.concat([data["particles"], row])
    with pytest.raises(ValueError):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("field", ["frames", "steps", "stepped_frames", "census_sha256", "content_sha256", "schema"])
def test_metadata_disagreements(tmp_path, field):
    out, meta = converted(tmp_path)
    meta[field] = meta[field] + 1 if isinstance(meta[field], int) else "0" * 64
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError):
        v2.PoolV2([out])


def test_payload_mutation_and_source_mismatch(tmp_path):
    out, _ = converted(tmp_path)
    data = pd.read_parquet(out)
    data.loc[0, "edep"] = 3.
    rewrite(out, frame=data)
    with pytest.raises(ValueError, match="hash disagrees"):
        v2.PoolV2([out])
    data.loc[0, "source_identity"] = "different"
    rewrite(out, frame=data)
    with pytest.raises(ValueError, match="source identity disagrees"):
        v2.PoolV2([out])


@pytest.mark.parametrize("field", ["identity", "uri", "sha256"])
def test_explicit_source_provenance_required(tmp_path, field):
    p = provenance()
    del p["source_art"][field]
    root = dump(tmp_path / "dump.root", tables())
    with pytest.raises(ValueError):
        v2.convert(root, "Cat", tmp_path / "out.parquet", **p)


def test_basename_provenance_and_generator_only_extraction_rejected(tmp_path):
    root = dump(tmp_path / "dump.root", tables())
    p = provenance()
    p["source_art"]["uri"] = "dump.root"
    with pytest.raises(ValueError, match="absolute locator"):
        v2.convert(root, "Cat", tmp_path / "out.parquet", **p)
    p = provenance()
    p["extraction"]["write_steps"] = False
    with pytest.raises(ValueError, match="write_steps"):
        v2.convert(root, "Cat", tmp_path / "out.parquet", **p)


@pytest.mark.parametrize("mode", ["copy", "source", "source_hash", "source_uri", "dump_hash", "content", "extraction"])
def test_duplicates_fail_closed_without_deduplication(tmp_path, mode):
    first, meta1 = converted(tmp_path)
    if mode == "copy":
        second = tmp_path / "copied.parquet"
        shutil.copyfile(first, second)
    else:
        data = tables()
        if mode != "content":
            data["steps"]["edep"] = 3.
        second, meta2 = converted(tmp_path, "b", data)
        if mode == "source":
            meta2["source_art"]["identity"] = meta1["source_art"]["identity"]
            frame = pd.read_parquet(second)
            frame["source_identity"] = meta2["source_art"]["identity"]
            rewrite(second, frame=frame)
        elif mode in ("source_hash", "source_uri"):
            field = "sha256" if mode == "source_hash" else "uri"
            meta2["source_art"][field] = meta1["source_art"][field]
        elif mode == "dump_hash":
            meta2["dump"]["sha256"] = meta1["dump"]["sha256"]
        elif mode == "extraction":
            meta2["extraction"]["id"] = meta1["extraction"]["id"]
        rewrite(second, metadata=meta2)
    with pytest.raises(ValueError, match="duplicate"):
        v2.PoolV2([first, second])


def test_dataset_disagreement_no_empty_pool_and_no_overwrite(tmp_path):
    first, _ = converted(tmp_path)
    data = tables()
    data["steps"]["edep"] = 3.
    second, meta = converted(tmp_path, "b", data)
    meta["dataset"] = "OtherCat"
    rewrite(second, metadata=meta)
    with pytest.raises(ValueError, match="dataset"):
        v2.PoolV2([first, second])
    with pytest.raises(ValueError, match="at least one"):
        v2.PoolV2([])
    before = v2.sha256(first)
    with pytest.raises(FileExistsError):
        v2.convert(tmp_path / "a.root", "Cat", first, **provenance())
    assert v2.sha256(first) == before


def test_cli_uses_explicit_provenance(tmp_path, capsys):
    root = dump(tmp_path / "dump.root", tables())
    info = tmp_path / "provenance.json"
    info.write_text(json.dumps(provenance()))
    out = tmp_path / "out.parquet"
    v2.main([str(root), "--dataset", "ExampleCat", "--out", str(out), "--provenance", str(info)])
    assert json.loads(capsys.readouterr().out)["frames"] == 4
    assert v2.PoolV2([out]).frames == 4


@pytest.mark.parametrize("field", ["software_version", "fcl_sha256", "dumper_sha256", "converter_sha256"])
def test_extraction_contract_disagreement(tmp_path, field):
    first, _ = converted(tmp_path)
    data = tables()
    data["steps"]["edep"] = 3.
    second, meta = converted(tmp_path, "b", data)
    target = meta if field == "converter_sha256" else meta["extraction"]
    target[field] = "different" if field == "software_version" else "0" * 64
    rewrite(second, metadata=meta)
    with pytest.raises(ValueError, match="extraction metadata disagreement"):
        v2.PoolV2([first, second])


def test_reordered_content_duplicate_rejected(tmp_path):
    first, _ = converted(tmp_path)
    data = tables()
    data["steps"] = data["steps"].iloc[::-1].reset_index(drop=True)
    data["events"] = data["events"].iloc[::-1].reset_index(drop=True)
    second, _ = converted(tmp_path, "b", data)
    with pytest.raises(ValueError, match="duplicate content_sha256"):
        v2.PoolV2([first, second])


def test_all_empty_generator_and_step_collections(tmp_path):
    data = tables()
    for tree in ("gen", "steps", "particles"):
        data[tree] = data[tree].iloc[:0]
    data["events"][["ngen", "nsteps"]] = 0
    out, _ = converted(tmp_path, data=data)
    pool = v2.PoolV2([out])
    assert pool.frames == 4 and pool.stepped == 0
    assert len(pool.events) == 4


def test_zero_event_extraction_rejected(tmp_path):
    data = {name: table.iloc[:0] for name, table in tables().items()}
    with pytest.raises(ValueError, match="must not be empty"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("column,value", [("t0", np.inf), ("t1", -np.inf), ("edep", np.nan), ("x0", 1e100)])
def test_invalid_kinematics(tmp_path, column, value):
    data = tables()
    data["steps"][column] = data["steps"][column].astype(np.float64)
    data["steps"].loc[0, column] = value
    with pytest.raises(ValueError):
        converted(tmp_path, data=data)


def test_missing_census_metadata_and_old_schema_rejected(tmp_path):
    out, meta = converted(tmp_path)
    del meta["events"]
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError, match="event census"):
        v2.PoolV2([out])
    del meta["schema"]
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError, match="expected metadata schema"):
        v2.PoolV2([out])


@pytest.mark.parametrize("tree", ["events", "gen", "steps", "particles"])
def test_missing_id_branch_rejected(tmp_path, tree):
    data = tables()
    del data[tree]["run"]
    with pytest.raises(ValueError, match="missing columns"):
        converted(tmp_path, data=data)


def test_no_generator_links_required_for_stepless_event_particles(tmp_path):
    data = tables()
    extra = data["particles"].iloc[-1:].copy()
    extra[v2.IDS] = data["events"].iloc[3][v2.IDS].to_numpy()
    data["particles"] = pd.concat([data["particles"], extra], ignore_index=True)
    out, _ = converted(tmp_path, data=data)
    assert v2.PoolV2([out]).frames == 4


def test_generator_index_outside_collection_rejected(tmp_path):
    data = tables()
    data["gen"].loc[0, "idx"] = 3
    data["particles"].loc[0, "genidx"] = 3
    with pytest.raises(ValueError, match="index outside census"):
        converted(tmp_path, data=data)


@pytest.mark.parametrize("field,value", [("source_count", 1.0), ("source_count", True),
                                        ("source_count", 2), ("event_selection", "partial"), ("id", "")])
def test_invalid_extraction_declarations(tmp_path, field, value):
    root = dump(tmp_path / "dump.root", tables())
    p = provenance()
    p["extraction"][field] = value
    with pytest.raises(ValueError):
        v2.convert(root, "Cat", tmp_path / "out.parquet", **p)


def test_pool_payload_types_not_silently_narrowed(tmp_path):
    out, _ = converted(tmp_path)
    frame = pd.read_parquet(out)
    frame["edep"] = frame.edep.astype(np.float64)
    rewrite(out, frame=frame)
    with pytest.raises(ValueError, match="dtype disagrees"):
        v2.PoolV2([out])


def test_compressed_descendant_can_differ_from_generator_pdg(tmp_path):
    data = tables()
    # CompressDetStepMCs may retain a descendant and attach its ancestor's
    # generator pointer: an electron can validly refer to a generated muon.
    data["particles"].loc[0, "pdg"] = 11
    data["steps"]["pdg"] = [11, 11, 13, 13]
    out, _ = converted(tmp_path, data=data)
    pool = v2.PoolV2([out])
    assert sorted(pool.steps.pdg.tolist()) == [11, 11, 13, 13]


def test_companion_canonical_and_file_hashes_and_types(tmp_path):
    out, meta = converted(tmp_path)
    companion = out.parent / meta["events"]["path"]
    assert companion == out.with_suffix(".events.parquet")
    assert v2.sha256(companion) == meta["events"]["sha256"]
    census = pd.read_parquet(companion)
    assert list(census) == v2.CENSUS_COLUMNS
    assert all(dtype == np.dtype("int64") for dtype in census.dtypes)
    assert v2._table_hash(census) == meta["census_sha256"]
    assert "event_census" not in meta
    assert v2.PoolV2([out]).manifest[0]["metadata"] == meta == v2.read_meta(out)


@pytest.mark.parametrize("mode", ["missing", "bytes", "canonical", "types", "columns", "null"])
def test_companion_tampering_rejected(tmp_path, mode):
    out, meta = converted(tmp_path)
    companion = out.parent / meta["events"]["path"]
    if mode == "missing":
        companion.unlink()
    elif mode == "bytes":
        with companion.open("ab") as stream:
            stream.write(b"tamper")
    else:
        frame = pd.read_parquet(companion)
        if mode == "canonical":
            # Mutate a stepless event, keeping step counts and membership valid.
            frame.loc[frame.nsteps == 0, "ngen"] += 1
        elif mode == "types":
            frame["event"] = frame.event.astype("uint64")
        elif mode == "columns":
            frame["extra"] = 0
        else:
            frame.loc[0, "ngen"] = None
        frame.to_parquet(companion, index=False)
        meta["events"]["sha256"] = v2.sha256(companion)
        rewrite(out, metadata=meta)
    with pytest.raises(ValueError):
        v2.PoolV2([out])


@pytest.mark.parametrize("name", ["../a.events.parquet", "/tmp/a.events.parquet",
                                  "nested/a.events.parquet", r"..\a.events.parquet",
                                  "a.parquet", "C:a.events.parquet", "bad\0.events.parquet"])
def test_companion_reference_cannot_escape_sibling(tmp_path, name):
    out, meta = converted(tmp_path)
    meta["events"]["path"] = name
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError, match="relative sibling"):
        v2.PoolV2([out])


def test_companion_symlink_and_self_reference_rejected(tmp_path):
    out, meta = converted(tmp_path)
    companion = out.parent / meta["events"]["path"]
    moved = companion.with_name("moved.events.parquet")
    companion.rename(moved)
    companion.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        v2.PoolV2([out])
    self_path = out.with_name("self.events.parquet")
    shutil.copyfile(out, self_path)
    meta["events"]["path"] = self_path.name
    rewrite(self_path, metadata=meta)
    with pytest.raises(ValueError, match="collision"):
        v2.PoolV2([self_path])


@pytest.mark.parametrize("target,kind", [("steps", "file"), ("events", "file"),
                                        ("steps", "symlink"), ("events", "symlink")])
def test_orphan_and_destination_collision_never_overwritten(tmp_path, target, kind):
    root = dump(tmp_path / "a.root", tables())
    out = tmp_path / "a.parquet"
    events = out.with_suffix(".events.parquet")
    blocked = out if target == "steps" else events
    if kind == "file":
        blocked.write_bytes(b"pre-existing orphan")
    else:
        blocked.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        v2.convert(root, "Cat", out, **provenance())
    if kind == "file":
        assert blocked.read_bytes() == b"pre-existing orphan"
    else:
        assert blocked.is_symlink() and not blocked.exists()
    assert not (events if target == "steps" else out).exists()


@pytest.mark.parametrize("fail_on", [1, 2])
def test_failed_conversion_cleans_only_new_outputs(tmp_path, monkeypatch, fail_on):
    root = dump(tmp_path / "a.root", tables())
    out = tmp_path / "a.parquet"
    real_write, calls = pq.write_table, 0

    def broken_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        real_write(*args, **kwargs)
        if calls == fail_on:
            raise OSError("synthetic write failure")

    monkeypatch.setattr(pq, "write_table", broken_write)
    with pytest.raises(OSError, match="synthetic write"):
        v2.convert(root, "Cat", out, **provenance())
    assert not out.exists() and not out.with_suffix(".events.parquet").exists()


def test_mostly_empty_large_census_keeps_metadata_compact(tmp_path):
    _, small = converted(tmp_path)
    data = tables()
    n = 100_000
    extra = pd.DataFrame({"run": np.full(n, 3, dtype=np.int64),
                          "subrun": np.zeros(n, dtype=np.int64),
                          "event": np.arange(1, n + 1, dtype=np.int64),
                          "ngen": np.zeros(n, dtype=np.int64),
                          "nsteps": np.zeros(n, dtype=np.int64)})
    data["events"] = pd.concat([data["events"], extra], ignore_index=True)
    out, meta = converted(tmp_path, "b", data)
    pool = v2.PoolV2([out])
    assert pool.frames == n + 4 and pool.stepped == 3
    assert len(json.dumps(meta)) < 4096
    assert abs(len(json.dumps(meta)) - len(json.dumps(small))) < 32
    assert len(json.dumps(pool.manifest)) < 8192


@pytest.mark.parametrize("reference", [None, [], {}, {"path": "a.events.parquet"},
                                       {"path": "a.events.parquet", "sha256": "bad"},
                                       {"path": 123, "sha256": "0" * 64}])
def test_malformed_companion_reference_rejected(tmp_path, reference):
    out, meta = converted(tmp_path)
    meta["events"] = reference
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError):
        v2.PoolV2([out])


def test_inline_census_metadata_not_accepted(tmp_path):
    out, meta = converted(tmp_path)
    meta["event_census"] = tables()["events"].to_dict(orient="list")
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError, match="companion reference"):
        v2.PoolV2([out])


def test_companion_collision_after_preflight_preserves_other_file(tmp_path, monkeypatch):
    from pathlib import Path
    root = dump(tmp_path / "a.root", tables())
    out = tmp_path / "a.parquet"
    events = out.with_suffix(".events.parquet")
    real_open = Path.open

    def racing_open(path, *args, **kwargs):
        if path == events and args == ("xb",):
            # Another writer wins after the existence check, before creation.
            with real_open(events, "xb") as stream:
                stream.write(b"other writer")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        v2.convert(root, "Cat", out, **provenance())
    assert events.read_bytes() == b"other writer"
    assert not out.exists()


def test_output_stems_colliding_on_companion_fail_closed(tmp_path):
    first, meta = converted(tmp_path)
    other = first.with_suffix(".pq")
    before = v2.sha256(first), v2.sha256(first.parent / meta["events"]["path"])
    with pytest.raises(FileExistsError):
        v2.convert(tmp_path / "a.root", "Cat", other, **provenance())
    assert not other.exists()
    assert before == (v2.sha256(first), v2.sha256(first.parent / meta["events"]["path"]))


def test_relative_companion_pair_can_be_moved_together(tmp_path):
    out, meta = converted(tmp_path)
    destination = tmp_path / "moved"
    destination.mkdir()
    shutil.move(out, destination / out.name)
    companion = out.parent / meta["events"]["path"]
    shutil.move(companion, destination / companion.name)
    assert v2.PoolV2([destination / out.name]).frames == 4


@pytest.mark.parametrize("name", [r"bad\name.parquet", "bad:name.parquet"])
def test_converter_cannot_publish_unreadable_companion_reference(tmp_path, name):
    root = dump(tmp_path / "a.root", tables())
    out = tmp_path / name
    with pytest.raises(ValueError, match="relative sibling"):
        v2.convert(root, "Cat", out, **provenance())
    assert not out.exists() and not out.with_suffix(".events.parquet").exists()


def unusual_floats():
    # C++ NaNs may have a negative sign or arbitrary payload; Parquet need not
    # retain either, since pandas NaNs become Arrow nulls.
    return np.array([0xfff8000000000000, 0x7ff8000000000123,
                     0xfff8000000000456, 0x8000000000000000], dtype=np.uint64).view(np.float64)


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 65])
def test_numeric_hash_roundtrip_with_noncanonical_nan_and_index(tmp_path, length):
    import pyarrow as pa
    values = np.resize(np.r_[unusual_floats(), 0., 1., -1.], length)
    frame = pd.DataFrame({"t0": values, "x0": values.astype(np.float32),
                          "event": np.arange(length, dtype=np.int64)},
                         index=pd.Index(np.arange(length)[::-1] * 17 + 93, name="entry"))
    before = [frame[name].to_numpy().tobytes() for name in frame]
    path = tmp_path / "roundtrip.parquet"
    # Deliberately persist the real-style index; it must not enter numeric hashes.
    pq.write_table(pa.Table.from_pandas(frame), path, row_group_size=3)
    restored = pd.read_parquet(path)
    assert v2._table_hash(frame) == v2._table_hash(restored)
    assert v2._table_hash(frame) == v2._table_hash(frame.reset_index(drop=True))
    normalized = frame.copy()
    for name in ("t0", "x0"):
        normalized.loc[normalized[name].isna(), name] = np.nan
        normalized.loc[normalized[name] == 0, name] = 0.
    assert v2._table_hash(frame) == v2._table_hash(normalized)
    assert before == [frame[name].to_numpy().tobytes() for name in frame]


def test_hash_null_storage_is_explicit_and_deterministic(monkeypatch):
    import pyarrow as pa
    original = pa.ipc.new_stream
    streams = []

    def capture(sink, schema, *args, **kwargs):
        streams.append(sink)
        return original(sink, schema, *args, **kwargs)

    monkeypatch.setattr(pa.ipc, "new_stream", capture)
    frame = pd.DataFrame({"t0": unusual_floats()})
    v2._table_hash(frame)
    table = pa.ipc.open_stream(streams[0].getvalue()).read_all()
    array = table.column("t0").chunk(0)
    bitmap, data = array.buffers()
    assert bitmap.to_pybytes() == b"\x08"  # Three nulls, one valid -0, zero padding.
    assert data.to_pybytes() == bytes(4 * 8)  # Even null slots and -0 are +0.
    assert array.null_count == 3 and array[3].as_py() == 0.


def test_pool_roundtrip_nan_payloads_signed_zero_and_reordered_duplicate(tmp_path):
    data = tables()
    data["steps"]["t0"] = unusual_floats()
    data["steps"]["t1"] = [-0., 0., np.nan, -0.]
    for name in v2.F32:
        data["steps"][name] = np.array([-0., 0., 1., -1.], dtype=np.float32)
    first, meta1 = converted(tmp_path, data=data)
    pool = v2.PoolV2([first])
    assert meta1["content_hash_format"] == "arrow-ipc-stream-v2"
    assert pool.frames == 4 and pool.stepped == 3
    assert pool.steps.t0.isna().sum() == 3 and pool.steps.t1.isna().sum() == 1
    assert np.signbit(pool.steps.loc[pool.steps.t0.notna(), "t0"]).all()
    assert np.signbit(pool.steps.loc[pool.steps.edep == 0, "edep"]).sum() == 1
    for name in ["t0", "t1", *v2.F32]:
        column = data["steps"][name]
        data["steps"].loc[column.isna(), name] = np.nan
        data["steps"].loc[column == 0, name] = 0.
    data["steps"] = data["steps"].iloc[::-1]
    second, meta2 = converted(tmp_path, "b", data)
    assert meta1["content_sha256"] == meta2["content_sha256"]
    with pytest.raises(ValueError, match="duplicate content_sha256"):
        v2.PoolV2([first, second])


@pytest.mark.parametrize("column,value", [
    ("t0", 0.),  # Missing time is not zero.
    ("t1", np.nextafter(np.float64(2), np.float64(3))),
    ("edep", np.nextafter(np.float32(1), np.float32(2))),
    ("x0", np.nextafter(np.float32(0), np.float32(1))),
])
def test_logical_hash_still_detects_exact_numeric_changes(tmp_path, column, value):
    data = tables()
    data["steps"].loc[0, "x0"] = 0.
    out, _ = converted(tmp_path, data=data)
    frame = pd.read_parquet(out)
    frame.loc[0, column] = value
    rewrite(out, frame=frame)
    with pytest.raises(ValueError, match="hash disagrees"):
        v2.PoolV2([out])


def test_table_hash_retains_dtype_column_and_row_significance():
    frame = pd.DataFrame({"x": np.array([0., 1.], dtype=np.float32)})
    original = v2._table_hash(frame)
    assert original != v2._table_hash(frame.astype(np.float64))
    assert original != v2._table_hash(frame.rename(columns={"x": "y"}))
    assert original != v2._table_hash(frame.iloc[::-1])
    assert original != v2._table_hash(pd.concat([frame, frame]))


def test_old_noncanonical_hash_format_requires_reconversion(tmp_path):
    out, meta = converted(tmp_path)
    meta["content_hash_format"] = "arrow-ipc-stream-v1"
    rewrite(out, metadata=meta)
    with pytest.raises(ValueError, match="reconvert the ROOT dump"):
        v2.PoolV2([out])
