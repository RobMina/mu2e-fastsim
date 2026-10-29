"""Isolated synthetic replay-cache admission, exact identities and whole-frame selection."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import frame_registry as F, indexed_replay as I
from crvfastsim.veto.replay import EVENT_TYPES, IDS, PART_TYPES, STEP_TYPES, sha256, write_replay


def sample():
    events = pd.DataFrame({"run": [9]*4, "subrun": [3]*4, "event": [7, 8, 9, 10],
                           "pbtime": [-224.46]*4, "intensity": [1., 2., 3., 4.]}).astype(EVENT_TYPES)
    steps = pd.DataFrame({name: np.zeros(5, dtype=dtype) for name, dtype in STEP_TYPES.items()})
    steps["run"], steps["subrun"] = np.uint32(9), np.uint32(3)
    steps["event"] = np.array([7, 8, 7, 9, 8], dtype=np.uint32)
    steps["sp"] = np.array([1, 1, 2, 1, 1], dtype=np.int64)
    steps["edep"] = np.arange(5, dtype=float)
    steps["t0"] = [0., 2., np.nan, 4., -1.]
    steps["t1"] = [1., 3., np.nan, 5., 0.]
    particles = pd.DataFrame({"run": [9]*4, "subrun": [3]*4, "event": [8, 7, 9, 7],
                              "key": [1, 2, 1, 1], "pdg": [13, 2212, -11, 11],
                              "parent": [-1]*4, "isprim": [1]*4}).astype(PART_TYPES)
    return {"events": events, "steps": steps, "particles": particles}


def artifact(tmp_path, tables=None, **manifest_changes):
    tables = sample() if tables is None else tables
    path = tmp_path / "raw.root"
    counts = write_replay(path, tables)
    digest = sha256(path)
    manifest = {"schema": "crv-mixed-replay-v1", "raw_times": True, "output_sha256": digest,
                "entries": counts, "events": len(tables["events"]), "run": 9, "subrun": 3, "first_event": 7}
    manifest.update(manifest_changes)
    path.with_suffix(".json").write_text(json.dumps(manifest))
    local = tables["events"].assign(input_sha256=digest)
    foreign = tables["events"].assign(input_sha256="0"*64)
    registry = F.freeze(pd.concat([local, foreign], ignore_index=True))
    args = dict(expected_registry_sha256=F.digest(registry), expected_input_sha256=digest,
                expected_manifest_sha256=sha256(path.with_suffix(".json")))
    return path, registry, args, tables


def selected(registry, args, numbers=None):
    mask = registry.input_sha256 == args["expected_input_sha256"]
    if numbers is not None:
        mask &= registry.event.isin(numbers)
    return registry[mask].copy()


def take(cache, registry, args, numbers=None):
    return cache.select(selected(registry, args, numbers), expected_registry_sha256=args["expected_registry_sha256"])


@pytest.mark.parametrize("chunk_rows", [1, 2, 99])
def test_real_tiny_root_interleaved_source_order_join_and_empty_denominator(tmp_path, monkeypatch, chunk_rows):
    path, registry, args, tables = artifact(tmp_path)
    with I.load_input(path, registry, **args, limits=replace(I.Limits(), chunk_rows=chunk_rows)) as cache:
        monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("selection reread ROOT"))
        result = take(cache, registry, args, [7, 10])
        assert result.events.event.tolist() == [7, 10]
        assert result.events.source_row.tolist() == [0, 3]
        assert result.steps.source_row.tolist() == [0, 2]
        assert result.steps.pdg.tolist() == [11, 2212]
        assert result.particles.source_row.tolist() == [1, 3]
        assert result.steps.frame_id.tolist() == [4, 4]
        assert result.input_sha256 == args["expected_input_sha256"]
        pd.testing.assert_frame_equal(result.steps[list(STEP_TYPES)], tables["steps"].iloc[[0, 2]].reset_index(drop=True))
        empty = take(cache, registry, args, [10])
        assert len(empty.events) == 1 and empty.steps.empty and empty.particles.empty
        assert empty.steps.source_row.dtype == np.int64 and empty.steps.pdg.dtype == np.int64
        none = take(cache, registry, args, [])
        assert none.events.empty and none.steps.empty and none.particles.empty
        full = take(cache, registry, args)
        assert full.steps.source_row.tolist() == list(range(5))
        assert full.steps.pdg.tolist() == [11, 13, 2212, -11, 13]
        reverse = cache.select(selected(registry, args).iloc[::-1], expected_registry_sha256=args["expected_registry_sha256"])
        assert reverse.events.event.tolist() == [10, 9, 8, 7]
        pd.testing.assert_frame_equal(full.steps, reverse.steps)
        joined = pd.concat([take(cache, registry, args, [7, 9]).steps,
                            take(cache, registry, args, [8, 10]).steps]).sort_values("source_row").reset_index(drop=True)
        pd.testing.assert_frame_equal(joined, full.steps)
        full.steps.loc[0, "edep"] = 999
        assert take(cache, registry, args).steps.edep.iloc[0] == 0
        assert cache.binding["execution_admitted"] is False
        assert cache.resident_array_bytes == I.resident_bytes({k: len(v) for k, v in tables.items()})
    with pytest.raises(ValueError, match="closed"):
        take(cache, registry, args)


@pytest.mark.parametrize("case", ["duplicate", "missing", "foreign_run", "foreign_subrun", "foreign_event", "zero_pdg", "zero_key", "parent", "isprim"])
def test_bad_particle_identity_and_join(tmp_path, case):
    tables = sample()
    p = tables["particles"]
    if case == "duplicate":
        tables["particles"] = pd.concat([p, p.iloc[:1]], ignore_index=True)
    elif case == "missing":
        tables["particles"] = p.iloc[:-1].copy()
    elif case.startswith("foreign_"):
        p.loc[0, case.removeprefix("foreign_")] = 99
    else:
        column, value = {"zero_pdg": ("pdg", 0), "zero_key": ("key", 0), "parent": ("parent", 1), "isprim": ("isprim", 0)}[case]
        p.loc[0, column] = value
    path, registry, args, _ = artifact(tmp_path, tables)
    with pytest.raises(ValueError, match="particle|census"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("case", ["run", "subrun", "event", "negative_sp", "negative_bar", "inf_time", "nan_geometry"])
def test_bad_step_domain(tmp_path, case):
    tables = sample()
    column, value = {"run": ("run", 10), "subrun": ("subrun", 4), "event": ("event", 99),
                     "negative_sp": ("sp", -1), "negative_bar": ("bar", -1),
                     "inf_time": ("t0", np.inf), "nan_geometry": ("x0", np.nan)}[case]
    tables["steps"].loc[0, column] = value
    path, registry, args, _ = artifact(tmp_path, tables)
    with pytest.raises(ValueError, match="step|census"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("case", ["digest", "subset", "domain", "exposure", "missing_empty"])
def test_full_registry_admission_mismatch(tmp_path, monkeypatch, case):
    path, registry, args, _ = artifact(tmp_path)
    if case == "digest":
        args["expected_registry_sha256"] = "f"*64
    elif case == "subset":
        registry = F.freeze(selected(registry, args))
    else:
        rows = selected(registry, args)
        if case == "domain":
            rows.loc[rows.index[-1], "subrun"] = 4
        elif case == "exposure":
            rows.loc[rows.index[-1], "pbtime"] = -99.
        else:
            rows = rows.iloc[:-1]
        registry = F.freeze(pd.concat([registry[registry.input_sha256 == "0"*64], rows]))
        args["expected_registry_sha256"] = F.digest(registry)
    if case in ("digest", "subset"):
        monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("opened before registry admission"))
    with pytest.raises(ValueError, match="registry|census"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("case", ["digest", "foreign_input", "run", "subrun", "event", "label", "float", "phase", "intensity", "duplicate", "rebuilt"])
def test_selected_registry_mismatch(tmp_path, case):
    path, registry, args, _ = artifact(tmp_path)
    cache = I.load_input(path, registry, **args)
    sel = selected(registry, args, [7])
    digest = args["expected_registry_sha256"]
    if case == "digest":
        digest = "a"*64
    elif case == "duplicate":
        sel = pd.concat([sel, sel])
    elif case == "rebuilt":
        sel = F.freeze(sel)
    else:
        column, value = {"foreign_input": ("input_sha256", "0"*64), "run": ("run", 1), "subrun": ("subrun", 1),
                         "event": ("event", 99), "label": ("frame_id", 0), "float": ("frame_id", 4.),
                         "phase": ("pbtime", -1.), "intensity": ("intensity", 99.)}[case]
        sel[column] = value
    with pytest.raises(ValueError, match="selected|registry|census"):
        cache.select(sel, expected_registry_sha256=digest)


@pytest.mark.parametrize("cap", ["max_events", "max_steps", "max_particles", "max_resident_bytes", "max_basket_bytes"])
def test_caps_before_payload_allocation_or_read(tmp_path, monkeypatch, cap):
    path, registry, args, _ = artifact(tmp_path)
    monkeypatch.setattr(I, "_read", lambda *a: pytest.fail("payload allocated/read before cap admission"))
    with pytest.raises(ValueError, match="cap"):
        I.load_input(path, registry, **args, limits=replace(I.Limits(), **{cap: 1}))


def test_exact_cap_boundary_and_selection_cap_before_gather(tmp_path, monkeypatch):
    path, registry, args, tables = artifact(tmp_path)
    size = I.resident_bytes({k: len(v) for k, v in tables.items()})
    limits = replace(I.Limits(), max_events=4, max_steps=5, max_particles=4, max_resident_bytes=size, max_selection_bytes=1)
    cache = I.load_input(path, registry, **args, limits=limits)
    monkeypatch.setattr(I, "_gather", lambda *a: pytest.fail("gather allocated before selection cap"))
    with pytest.raises(ValueError, match="selected typed payload"):
        take(cache, registry, args)


@pytest.mark.parametrize("target", ["root", "sidecar"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_input_hash_admission_before_and_after(tmp_path, monkeypatch, target, when):
    path, registry, args, _ = artifact(tmp_path)
    changed = path if target == "root" else path.with_suffix(".json")
    if when == "before":
        with changed.open("ab") as stream:
            stream.write(b" ")
        monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("opened changed input"))
    else:
        read = I._read
        def mutate(*a):
            result = read(*a)
            if a[1] == "particles":
                with changed.open("ab") as stream:
                    stream.write(b" ")
            return result
        monkeypatch.setattr(I, "_read", mutate)
    with pytest.raises(ValueError, match="changed bound input"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("changes", [{"raw_times": False}, {"schema": "crv-mixed-replay-v2"},
                                      {"first_event": 8}, {"events": 3}, {"entries": {}}])
def test_manifest_semantics(tmp_path, changes):
    path, registry, args, _ = artifact(tmp_path, **changes)
    with pytest.raises(ValueError):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("case", ["dtype", "extra", "missing"])
def test_tree_branch_contract(tmp_path, case):
    tables = sample()
    if case == "dtype":
        tables["steps"]["bar"] = tables["steps"].bar.astype(np.int32)
    elif case == "extra":
        tables["steps"]["extra"] = 0
    else:
        tables["steps"] = tables["steps"].drop(columns="sp")
    path, registry, args, _ = artifact(tmp_path, tables)
    with pytest.raises(ValueError, match="branches|dtype"):
        I.load_input(path, registry, **args)


def test_entire_input_can_be_stepless(tmp_path):
    tables = sample()
    tables["steps"] = tables["steps"].iloc[:0]
    tables["particles"] = tables["particles"].iloc[:0]
    path, registry, args, _ = artifact(tmp_path, tables)
    cache = I.load_input(path, registry, **args)
    result = take(cache, registry, args)
    assert len(result.events) == 4 and result.steps.empty and result.particles.empty


class FakeBranch:
    """Mirror uproot's lazy embedded state; unsafe APIs must never admit baskets."""

    def __init__(self, dtype):
        self.interpretation = SimpleNamespace(numpy_dtype=dtype)
        self._num_normal_baskets = 1
        self._embedded_baskets = []
        self.header_calls = []

    @property
    def embedded_baskets(self):
        pytest.fail("embedded baskets materialized during admission")

    @property
    def num_baskets(self):
        return self._num_normal_baskets + len(self.embedded_baskets)

    def basket_uncompressed_bytes(self, index):
        pytest.fail("payload-reading size API used during admission")

    def basket_key(self, index):
        assert 0 <= index < self._num_normal_baskets
        self.header_calls.append(index)
        return SimpleNamespace(fKeylen=32, fObjlen=32)


class FakeTree:
    classname = "TTree"

    def __init__(self, table, chunk_rows=2):
        self.table, self.num_entries, self.chunk_rows = table, len(table), chunk_rows
        self.calls = []
        self.branches = {name: FakeBranch(table[name].dtype) for name in table}

    def keys(self):
        return list(self.table)

    def __getitem__(self, name):
        return self.branches[name]

    def iterate(self, names, **kwargs):
        self.calls.append((names, kwargs))
        for first in range(0, len(self.table), self.chunk_rows):
            yield {name: self.table[name].iloc[first:first + self.chunk_rows].to_numpy() for name in names}


def test_fake_baskets_scanned_once_selection_reuses_index(tmp_path, monkeypatch):
    path, registry, args, tables = artifact(tmp_path)
    trees = {f"dump/{name}": FakeTree(table) for name, table in tables.items()}
    class Root(dict):
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
    monkeypatch.setattr(I.uproot, "open", lambda *a, **k: Root(trees))
    cache = I.load_input(path, registry, **args, limits=replace(I.Limits(), chunk_rows=2))
    take(cache, registry, args, [7])
    take(cache, registry, args, [8, 10])
    for tree in trees.values():
        assert all(branch.header_calls == [0] for branch in tree.branches.values())
        assert len(tree.calls) == 1
        assert tree.calls[0][1] == {"step_size": 2, "library": "np"}


@pytest.mark.parametrize("case", ["too_large", "short", "dtype"])
def test_fake_chunk_rejects_wrong_shape_coverage_or_dtype(case):
    table = sample()["steps"]
    tree = FakeTree(table, 5 if case == "too_large" else 2)
    if case == "short":
        tree.num_entries += 1
    elif case == "dtype":
        tree.table["sp"] = tree.table.sp.astype(float)
    with pytest.raises(ValueError, match="cap|coverage|dtype"):
        I._read(tree, "steps", replace(I.Limits(), chunk_rows=2))


@pytest.mark.parametrize("field,value", [("max_events", True), ("max_steps", 0), ("max_particles", -1),
                                          ("chunk_rows", 2.5), ("max_resident_bytes", 2**53)])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError, match="positive integers"):
        replace(I.Limits(), **{field: value})


@pytest.mark.parametrize("case", ["duplicate", "nan_phase", "negative_intensity", "inf_intensity"])
def test_fake_all_event_census_is_validated_before_selection(tmp_path, monkeypatch, case):
    path, registry, args, tables = artifact(tmp_path)
    if case == "duplicate":
        tables["events"].iloc[1] = tables["events"].iloc[0]
    else:
        column, value = {"nan_phase": ("pbtime", np.nan), "negative_intensity": ("intensity", -1.),
                         "inf_intensity": ("intensity", np.inf)}[case]
        tables["events"].loc[0, column] = value
    class Root(dict):
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
    trees = {f"dump/{name}": FakeTree(table) for name, table in tables.items()}
    monkeypatch.setattr(I.uproot, "open", lambda *a, **k: Root(trees))
    with pytest.raises(ValueError, match="census|exposure"):
        I.load_input(path, registry, **args)


def test_selected_fulltuple_and_input_identity_survive_caller_mutations(tmp_path):
    path, registry, args, _ = artifact(tmp_path)
    original = registry.copy(deep=True)
    cache = I.load_input(path, registry, **args)
    registry.loc[:, "pbtime"] = 0.
    out = take(cache, original, args)
    for table in (out.events, out.steps, out.particles):
        assert (table.input_sha256 == args["expected_input_sha256"]).all()
        assert (table.run == 9).all() and (table.subrun == 3).all()
        joined = table.merge(original, on=list(F.FRAME_KEYS), validate="many_to_one", suffixes=("", "_registry"))
        assert len(joined) == len(table)
        np.testing.assert_array_equal(joined.frame_id, joined.frame_id_registry)
    assert set(out.events.frame_id) == {4, 5, 6, 7}


@pytest.mark.parametrize("reject", [True, False])
def test_real_uproot_header_admission_never_decodes_payload(tmp_path, monkeypatch, reject):
    path, registry, args, _ = artifact(tmp_path)
    # Read tree metadata first so metadata decompression is not confused with
    # basket payload decompression. load_input still performs its full admission.
    root = I.uproot.open(path, array_cache=None, object_cache=None, minimal_ttree_metadata=True)
    trees = {name: root[f"dump/{name}"] for name in I.TYPES}
    headers = [(branch, index, branch.basket_key(index))
               for tree in trees.values() for branch in tree.values()
               for index in range(branch._num_normal_baskets)]
    largest = max(key.fKeylen + key.fObjlen for _, _, key in headers)
    compressed = sum(key.fNbytes < key.fKeylen + key.fObjlen for _, _, key in headers)
    assert compressed > 0  # Exercise actual decompression, not just uncompressed baskets.
    expected = Counter((branch.object_path, index) for branch, index, _ in headers)
    decoded, interpreted = Counter(), []
    decompressed, header_reads, payload_reads = [], [], []
    reading = False

    basket_model = I.uproot.models.TBasket.Model_TBasket
    original_decode = basket_model.read_members
    def decode(self, chunk, cursor, context, file):
        assert reading, "basket decoded before all header caps admitted"
        decoded[(self._parent.object_path, context["basket_num"])] += 1
        return original_decode(self, chunk, cursor, context, file)
    monkeypatch.setattr(basket_model, "read_members", decode)

    original_decompress = I.uproot.compression.decompress
    def decompress(*a, **k):
        assert reading, "payload decompressed during header admission"
        decompressed.append(1)
        return original_decompress(*a, **k)
    decompress.__dict__.update(original_decompress.__dict__)  # uproot's block hooks
    monkeypatch.setattr(I.uproot.compression, "decompress", decompress)

    interpretation = I.uproot.interpretation.numerical.AsDtype
    original_interpret = interpretation.basket_array
    def interpret(*a, **k):
        assert reading, "basket interpreted during header admission"
        interpreted.append(1)
        return original_interpret(*a, **k)
    monkeypatch.setattr(interpretation, "basket_array", interpret)

    source = root.file.source
    original_chunk = source.chunk
    def chunk(start, stop):
        if not reading:
            header_reads.append((start, stop))
            assert stop - start == I.uproot.reading._key_format_big.size
        return original_chunk(start, stop)
    monkeypatch.setattr(source, "chunk", chunk)
    original_chunks = source.chunks
    def chunks(ranges, notifications):
        assert reading, "bulk payload read during header admission"
        payload_reads.extend(ranges)
        return original_chunks(ranges, notifications)
    monkeypatch.setattr(source, "chunks", chunks)

    original_read = I._read
    def read(*a):
        nonlocal reading
        reading = True
        return original_read(*a)
    monkeypatch.setattr(I, "_read", read)
    class MetadataRoot(dict):
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            root.close()
    def opened(*a, **k):
        assert k == dict(array_cache=None, object_cache=None, minimal_ttree_metadata=True)
        # With object_cache=None, repeated directory lookup reloads metadata;
        # reuse these real TTrees to isolate basket I/O from metadata I/O.
        return MetadataRoot({f"dump/{name}": tree for name, tree in trees.items()})
    monkeypatch.setattr(I.uproot, "open", opened)
    limits = replace(I.Limits(), chunk_rows=99, max_basket_bytes=largest - int(reject))
    if reject:
        with pytest.raises(ValueError, match="basket byte cap"):
            I.load_input(path, registry, **args, limits=limits)
        assert not reading and not decoded and not decompressed and not interpreted and not payload_reads
    else:
        with I.load_input(path, registry, **args, limits=limits):
            pass
        # Header-inclusive exact boundary admits; every payload is decoded only
        # by _read, once (one chunk here), never once extra by header admission.
        assert decoded == expected
        assert len(interpreted) == sum(expected.values())
        assert len(decompressed) == compressed
        assert len(header_reads) == len(headers)
        assert Counter(payload_reads) == Counter(
            (int(branch.member("fBasketSeek")[index]),
             int(branch.member("fBasketSeek")[index]) + key.fNbytes)
            for branch, index, key in headers
        )
    assert header_reads


@pytest.mark.parametrize("embedded", [None, [object()]])
def test_fake_embedded_baskets_rejected_without_materializing(embedded):
    tree = FakeTree(sample()["events"])
    branch = tree["run"]
    branch._embedded_baskets = embedded
    with pytest.raises(ValueError, match="embedded"):
        I._check_tree(tree, "events", I.Limits())
    assert not branch.header_calls


def test_real_uproot_pending_embedded_baskets_are_not_materialized(tmp_path, monkeypatch):
    path, _, _, _ = artifact(tmp_path)
    with I.uproot.open(path, minimal_ttree_metadata=True) as root:
        tree = root["dump/events"]
        # Recreate postprocess's lazy state for a branch whose normal baskets
        # do not cover all entries; no embedded payload fixture is materialized.
        branch = tree["run"]
        branch._embedded_baskets = None
        def forbidden(*a, **k):
            pytest.fail("embedded basket accessor materialized payload")
        monkeypatch.setattr(I.uproot.behaviors.TBranch.TBranch, "embedded_baskets", property(forbidden))
        monkeypatch.setattr(I.uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
        with pytest.raises(ValueError, match="embedded"):
            I._check_tree(tree, "events", I.Limits())


_MANIFEST_FIELDS = ("schema", "raw_times", "output_sha256", "entries", "events", "run", "subrun", "first_event")


@pytest.mark.parametrize("field", _MANIFEST_FIELDS)
def test_missing_manifest_fields_fail_before_legacy_check_or_root(tmp_path, monkeypatch, field):
    path, registry, args, _ = artifact(tmp_path)
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text())
    del manifest[field]
    sidecar.write_text(json.dumps(manifest))
    args["expected_manifest_sha256"] = sha256(sidecar)
    monkeypatch.setattr(I, "require_legacy_response", lambda *a: pytest.fail("unvalidated manifest"))
    monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("opened malformed manifest"))
    with pytest.raises(ValueError, match="manifest"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("replacement", [[], None, "manifest", 42, True])
def test_nonobject_manifest_is_valueerror(tmp_path, monkeypatch, replacement):
    path, registry, args, _ = artifact(tmp_path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(replacement))
    args["expected_manifest_sha256"] = sha256(sidecar)
    monkeypatch.setattr(I, "require_legacy_response", lambda *a: pytest.fail("unvalidated manifest"))
    monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("opened malformed manifest"))
    with pytest.raises(ValueError, match="manifest"):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("field,value", [
    ("schema", []), ("raw_times", 1), ("output_sha256", None),
    ("entries", None), ("entries", []),
    ("entries", {"events": 4, "steps": 5}),
    ("entries", {"events": 4, "steps": 5, "particles": 4, "extra": 0}),
    *[(field, value) for field in ("events", "run", "subrun", "first_event")
      for value in (None, "4", 4.0, True, -1, [])],
    *[("entries", {"events": 4, "steps": value, "particles": 4})
      for value in (None, "5", 5.0, True, -1, [])],
])
def test_manifest_field_types_before_legacy_check_or_root(tmp_path, monkeypatch, field, value):
    path, registry, args, _ = artifact(tmp_path, **{field: value})
    monkeypatch.setattr(I, "require_legacy_response", lambda *a: pytest.fail("unvalidated manifest"))
    monkeypatch.setattr(I.uproot, "open", lambda *a, **k: pytest.fail("opened malformed manifest"))
    with pytest.raises(ValueError):
        I.load_input(path, registry, **args)


@pytest.mark.parametrize("field,value", [
    ("_embedded_baskets", None), ("_embedded_baskets", 0),
    ("_num_normal_baskets", None), ("_num_normal_baskets", -1),
])
def test_unknown_basket_metadata_fails_closed(field, value):
    tree = FakeTree(sample()["events"])
    setattr(tree["run"], field, value)
    with pytest.raises(ValueError, match="unsupported"):
        I._check_tree(tree, "events", I.Limits())
    assert not tree["run"].header_calls
