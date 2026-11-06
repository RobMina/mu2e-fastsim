"""Integer tuple joins checked against an independent structured-array oracle."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

import mixing.frames as frames
from mixing.frames import FrameBatch, IDS, overlay, read_frames
from mixing.sampling import FrameSelection


def _structured_join(target, query):
    target = np.rec.fromarrays(target, names=[f"k{i}" for i in range(len(target))])
    query = np.rec.fromarrays(query, names=target.dtype.names)
    order = np.argsort(target)
    positions = np.searchsorted(target[order], query)
    if len(query) and (not len(target) or np.any(positions == len(target))):
        raise ValueError("missing tuple")
    if np.any(target[order[positions]] != query):
        raise ValueError("missing tuple")
    return order[positions].astype(np.int64)


def _rows(counts, **columns):
    return ak.unflatten(ak.zip({name: np.asarray(values) for name, values in columns.items()}), counts)


def _flat(values):
    return ak.to_numpy(ak.flatten(values))


def _batch():
    high = np.iinfo(np.int64).max
    ids = np.array([[1, 2, 2**63 + 1], [1, 2, 2**63],
                    [1, 3, 2**63 + 1], [2**64 - 1, 2, 2**63 + 1]], dtype=np.uint64)
    return FrameBatch(ids, {
        "particles": _rows([3, 0, 2, 1], key=[high, 2**53 + 1, 7, 7, 2**53 + 2, 0],
                           parent=[-1, 7, -1, -1, 7, -1], genidx=[0, 1, -1, 0, -1, -1]),
        "gen": _rows([2, 0, 1, 0], idx=[0, 1, 0]),
        "tracker": _rows([4, 0, 2, 0], sp=[7, high, high, 2**53 + 1, 2**53 + 2, 7]),
        "crv": _rows([1, 0, 1, 1], sp=[2**53 + 1, 7, 0]),
        "calo": _rows([0, 0, 1, 0], sp=[2**53 + 2]),
    })


def _reference_oracle(batch):
    parts = batch.collections["particles"]
    event = np.repeat(np.arange(len(batch.event_ids)), ak.to_numpy(ak.num(parts)))
    target = [event, _flat(parts.key)]
    parent = _flat(parts.parent)
    present = parent >= 0
    references = {"parent": np.full(len(parent), -1, dtype=np.int64)}
    references["parent"][present] = _structured_join(target, [event[present], parent[present]])
    for name in batch.collections.keys() - {"particles", "gen"}:
        rows = batch.collections[name]
        event = np.repeat(np.arange(len(batch.event_ids)), ak.to_numpy(ak.num(rows)))
        references[name] = _structured_join(target, [event, _flat(rows.sp)])
    return references


def _replace(batch, name, field, values):
    rows = batch.collections[name]
    batch.collections[name] = ak.with_field(rows, ak.unflatten(np.asarray(values), ak.num(rows)), field)


def test_reference_indices_match_structured_oracle_and_flat_order():
    batch = _batch()
    actual, expected = batch.validate_references(), _reference_oracle(batch)
    assert actual.keys() == expected.keys() == {"parent", "tracker", "crv", "calo"}
    for name in expected:
        assert actual[name].dtype == np.int64
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(actual["parent"], [-1, 2, -1, -1, 3, -1])
    np.testing.assert_array_equal(actual["tracker"], [2, 0, 0, 1, 4, 3])


@pytest.mark.parametrize("seed", [3, 18, 125])
def test_randomized_reference_indices_match_structured_oracle(seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 20, size=300)
    event = np.repeat(np.arange(len(counts)), counts)
    local = np.arange(len(event)) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    key = 2**53 + 1 + local * 100003
    parent = np.where(local, key - 100003, -1)
    shuffled = rng.permutation(len(event))
    selected = shuffled[np.argsort(event[shuffled], kind="stable")]
    batch = FrameBatch(np.column_stack([np.ones(len(counts), dtype=np.int64)] * 2 + [np.arange(len(counts))]), {
        "particles": _rows(counts, key=key, parent=parent, genidx=np.full(len(event), -1)),
        "gen": _rows(np.zeros(len(counts), dtype=np.int64), idx=np.array([], dtype=np.int64)),
        "tracker": _rows(counts, sp=key[selected]),
        "crv": _rows(counts, sp=key),
    })
    expected = _reference_oracle(batch)
    for name, indices in batch.validate_references().items():
        np.testing.assert_array_equal(indices, expected[name])


@pytest.mark.parametrize("empty_events", [False, True])
def test_empty_particle_and_query_indices(empty_events):
    batch = _batch()
    chosen = slice(1, 2) if empty_events else slice(0, 0)
    empty = FrameBatch(batch.event_ids[chosen], {name: rows[chosen] for name, rows in batch.collections.items()})
    for indices in empty.validate_references().values():
        assert indices.dtype == np.int64 and indices.size == 0
    if empty_events:
        empty.collections["tracker"] = _rows([1], sp=[7])
        with pytest.raises(ValueError, match="dangling tracker"):
            empty.validate_references()


def test_repeated_validation_observes_awkward_buffers_and_returned_indices_are_not_cached():
    batch = _batch()
    first = batch.validate_references()
    first["tracker"][:] = -123
    tracker = _flat(batch.collections["tracker"].sp)
    tracker[0] = 2**53 + 1
    assert batch.collections["tracker"].sp[0, 0] == 2**53 + 1
    second = batch.validate_references()
    assert second["tracker"][0] == 1
    np.testing.assert_array_equal(second["tracker"], _reference_oracle(batch)["tracker"])
    tracker[0] = -1
    with pytest.raises(ValueError, match="dangling tracker"):
        batch.validate_references()
    with pytest.raises(ValueError, match="dangling tracker"):
        overlay(batch, [])


@pytest.mark.parametrize("name,field,values,message", [
    ("particles", "key", [7, 2**53 + 1, 7, 7, 2**53 + 2, 0], "event-unique"),
    ("particles", "key", [-1, 2**53 + 1, 7, 7, 2**53 + 2, 0], "nonnegative"),
    ("particles", "parent", [-2, 7, -1, -1, 7, -1], "null parent"),
    ("particles", "parent", [-1, 8, -1, -1, 7, -1], "dangling parent"),
    ("particles", "parent", [-1, 7, -1, 2**53 + 1, 7, -1], "dangling parent"),
    ("particles", "genidx", [-2, 1, -1, 0, -1, -1], "generator index"),
    ("particles", "genidx", [2, 1, -1, 0, -1, -1], "generator index"),
    ("particles", "genidx", [0, 1, -1, 0, -1, 0], "generator index"),
    ("gen", "idx", [1, 0, 0], "collection order"),
    ("gen", "idx", [0., 1., 0.], "integer keys"),
    ("tracker", "sp", [-1, 7, 7, 7, 7, 7], "dangling tracker"),
    ("tracker", "sp", [7, 7, 7, 7, 2**53 + 1, 7], "dangling tracker"),
    ("crv", "sp", [7, 7, -1], "dangling crv"),
    ("crv", "sp", [7., 7., 0.], "integer keys"),
    ("calo", "sp", [-1], "dangling calo"),
    ("calo", "sp", [2**53 + 1], "dangling calo"),
    ("calo", "sp", np.array([2**63], dtype=np.uint64), "exceeds int64"),
])
def test_repeated_validation_rejects_mutated_references(name, field, values, message):
    batch = _batch()
    batch.validate_references()
    _replace(batch, name, field, values)
    with pytest.raises(ValueError, match=message):
        batch.validate_references()


def test_repeated_validation_rejects_event_and_collection_mutations():
    batch = _batch()
    batch.validate_references()
    batch.event_ids[0] = batch.event_ids[1]
    with pytest.raises(ValueError, match="duplicate event"):
        batch.validate_references()
    batch = _batch()
    batch.validate_references()
    batch.collections["tracker"] = batch.collections["tracker"][:1]
    with pytest.raises(ValueError, match="event count"):
        batch.validate_references()
    batch = _batch()
    batch.validate_references()
    del batch.collections["gen"]
    with pytest.raises(ValueError, match="required"):
        batch.validate_references()


class _Tree:
    def __init__(self, ids, **columns):
        self.data = ak.Array({**dict(zip(IDS, np.asarray(ids).T)), **columns})

    def arrays(self, names, library):
        assert library == "ak"
        return self.data[names]


@pytest.mark.parametrize("seed", [1, 99])
def test_census_join_and_stable_grouping_match_structured_oracle(seed):
    rng = np.random.default_rng(seed)
    ids = _batch().event_ids
    selected = rng.choice([0, 2, 3], 200)
    tree = _Tree(ids[selected], row=np.arange(len(selected)))
    event = _structured_join(ids.T, ids[selected].T)
    expected = ak.unflatten(tree.data[["row"]][np.argsort(event, kind="stable")],
                            np.bincount(event, minlength=len(ids)))
    actual = frames._group(tree, frames._event_index(ids), ("row",))
    assert ak.almost_equal(actual, expected, rtol=0, atol=0)
    assert ak.to_list(ak.num(actual))[1] == 0


@pytest.mark.parametrize("empty_census", [False, True])
def test_census_join_rejects_missing_tuples_including_empty_census(empty_census):
    ids = _batch().event_ids
    query = ids[:1].copy()
    query[0, 1] = 99
    census = ids[:0] if empty_census else ids
    with pytest.raises(ValueError, match="absent from census"):
        frames._group(_Tree(query, row=[1]), frames._event_index(census), ("row",))
    result = frames._group(_Tree(ids[:0], row=np.array([], dtype=np.int64)),
                           frames._event_index(census), ("row",))
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result)), np.zeros(len(census)))


def _dump_table(ids, fields, **overrides):
    columns = {name: np.zeros(len(ids), dtype=np.int64 if name in
               {"key", "parent", "genidx", "idx", "sp"} else np.float64) for name in fields}
    columns.update(dict(zip(IDS, ids.T)))
    columns["run"] = ids[:, 0].astype(np.int64)
    columns["subrun"] = ids[:, 1].astype(np.uint32)
    columns.update(overrides)
    return columns


def test_reader_reuses_census_index_and_preserves_mixed_signed_unsigned_ids(tmp_path, monkeypatch):
    ids = np.array([[1, 1, 2**63 + 1], [1, 1, 2**63], [1, 1, 2**53 + 1],
                    [1, 1, 2**64 - 1]], dtype=np.uint64)
    occupied = ids[[3, 0, 2]]
    path = tmp_path / "wide_ids.root"
    with uproot.recreate(path) as root:
        root.mktree("crv/events", _dump_table(ids, (), n_steps=np.array([1, 0, 1, 1])))
        root.mktree("crv/particles", _dump_table(occupied, frames.PARTICLE_FIELDS,
                     key=np.array([7, 7, 7]), parent=np.array([-1, -1, -1]), genidx=np.array([0, 0, 0])))
        root.mktree("crv/gen", _dump_table(occupied, frames.GEN_FIELDS))
        root.mktree("crv/steps", _dump_table(occupied, frames.CRV_FIELDS, sp=np.array([7, 7, 7])))
        root.mktree("tracker/sgsteps", _dump_table(occupied, frames.TRACKER_FIELDS, sp=np.array([7, 7, 7])))
    original = frames._group
    indices = []

    def record(tree, census, fields):
        assert isinstance(census, pd.MultiIndex)
        indices.append(census)
        return original(tree, census, fields)

    monkeypatch.setattr(frames, "_group", record)
    batch = read_frames(path)
    assert len(indices) == 4 and all(index is indices[0] for index in indices)
    assert batch.event_ids.dtype == np.uint64
    np.testing.assert_array_equal(batch.event_ids, ids)
    np.testing.assert_array_equal(ak.to_numpy(ak.num(batch.collections["tracker"])), [1, 0, 1, 1])
    np.testing.assert_array_equal(read_frames(path, event_ids=ids).event_ids, ids)
    duplicate = ids.copy()
    duplicate[0] = duplicate[1]
    with pytest.raises(ValueError, match="duplicate event"):
        read_frames(path, event_ids=duplicate)


def test_legacy_event_keys_stay_structured_for_timing_reader():
    ids = _batch().event_ids
    keys = frames._event_keys(ids)
    assert keys.dtype.names == IDS
    for i, name in enumerate(IDS):
        np.testing.assert_array_equal(keys[name], ids[:, i])


def _single_key_batch(keys, dtype=np.int64):
    keys = np.asarray(keys, dtype=dtype)
    counts = np.ones(len(keys), dtype=np.int64)
    return FrameBatch(np.column_stack([counts, counts, np.arange(len(keys))]), {
        "particles": _rows(counts, key=keys, parent=np.full(len(keys), -1), genidx=np.full(len(keys), -1)),
        "gen": _rows(counts * 0, idx=np.array([], dtype=np.int64)),
        "tracker": _rows(counts, sp=keys),
        "crv": _rows(counts, sp=keys),
    })


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_overlay_full_int64_sparse_keys_without_overflow(dtype):
    limit = np.iinfo(np.int64).max
    batch = _single_key_batch([limit, limit - 1], dtype)
    result = overlay(batch, [])
    np.testing.assert_array_equal(_flat(result.collections["particles"].key), [limit, limit - 1])
    np.testing.assert_array_equal(_flat(result.collections["tracker"].sp), [limit, limit - 1])
    assert _flat(result.collections["particles"].key).dtype.kind == "i"


def test_overlay_sparse_key_overflow_raises_and_boundary_sum_is_exact():
    limit = np.iinfo(np.int64).max
    signal = _single_key_batch([limit - 1])
    zero = _single_key_batch([0])
    selection = FrameSelection(np.array([1]), np.array([0]))
    result = overlay(signal, [(zero, selection)])
    np.testing.assert_array_equal(_flat(result.collections["particles"].key), [limit - 1, limit])
    with pytest.raises(ValueError, match="exceeds int64"):
        overlay(_single_key_batch([limit]), [(zero, selection)])
    with pytest.raises(ValueError, match="exceeds int64"):
        overlay(signal, [(_single_key_batch([1]), selection)])


def test_overlay_mixed_signed_unsigned_particle_keys_never_convert_to_float():
    signal = _single_key_batch([2**53 + 1], np.uint64)
    donor = _single_key_batch([0], np.int64)
    result = overlay(signal, [(donor, FrameSelection(np.array([1]), np.array([0])))])
    np.testing.assert_array_equal(_flat(result.collections["particles"].key), [2**53 + 1, 2**53 + 2])
    np.testing.assert_array_equal(_flat(result.collections["tracker"].sp), [2**53 + 1, 2**53 + 2])


def test_validation_reuses_one_particle_target_index(monkeypatch):
    batch = _batch()
    original = frames._check_reference
    targets = []

    def record(target, query, name):
        assert isinstance(target, pd.MultiIndex)
        targets.append(target)
        return original(target, query, name)

    monkeypatch.setattr(frames, "_check_reference", record)
    batch.validate_references()
    assert len(targets) == 4 and all(target is targets[0] for target in targets)
    batch.validate_references()
    assert targets[4] is not targets[0]


@pytest.mark.parametrize("name,field,index,value,message", [
    ("particles", "key", 0, 7, "event-unique"),
    ("particles", "parent", 1, 8, "dangling parent"),
    ("particles", "genidx", 0, 2, "generator index"),
    ("gen", "idx", 0, 1, "collection order"),
])
def test_repeated_validation_rejects_in_place_numpy_buffer_mutation(name, field, index, value, message):
    batch = _batch()
    batch.validate_references()
    _flat(batch.collections[name][field])[index] = value
    with pytest.raises(ValueError, match=message):
        batch.validate_references()


@pytest.mark.parametrize("ids", [np.array([[1., 2., 3.]]), np.array([[1, 2, -1]]), np.array([1, 2, 3])])
def test_event_index_rejects_noninteger_negative_or_malformed_ids(ids):
    with pytest.raises(ValueError, match="nonnegative integer"):
        frames._event_index(ids)


def test_overlay_limit_key_followed_by_empty_donor_keeps_exact_key():
    limit = np.iinfo(np.int64).max
    signal = _single_key_batch([limit])
    empty = FrameBatch(np.array([[1, 1, 1]]), {
        name: ak.unflatten(ak.flatten(rows)[:0], [0]) for name, rows in signal.collections.items()})
    selection = FrameSelection(np.array([1]), np.array([0]))
    result = overlay(signal, [(empty, selection)])
    np.testing.assert_array_equal(_flat(result.collections["particles"].key), [limit])
    np.testing.assert_array_equal(_flat(result.collections["tracker"].sp), [limit])
