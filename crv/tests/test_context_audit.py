"""Small synthetic context audits; no detector data, IO, or accelerator needed."""

from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses.context_audit import content_fingerprint as fingerprint
from crvfastsim.analyses.context_audit import freeze_arrays


@dataclass
class Geometry:
    center: np.ndarray


@dataclass(slots=True)
class Slotted:
    weights: np.ndarray


class _NB(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def payload():
    geom = Geometry(np.arange(12, dtype=np.float64).reshape(4, 3))
    geom.lib_group = np.array(["EX", "T1", "T2", "M"], dtype=object)
    table = pd.DataFrame({"weight": np.arange(4, dtype=np.float64),
                          "name": pd.Series(["a", "b", "c", "d"], dtype=object)})
    return {"geom": geom, "tables": [table],
            "cache": {(3, "prob"): SimpleNamespace(nb=_NB(x=2), weights=np.ones(3))},
            "scalar": (np.int32(2), np.float64(0.5), np.str_("label")),
            "slots": Slotted(np.ones(2))}


def test_independent_host_payloads_freezing_and_index_caches():
    a, b = payload(), payload()
    before = fingerprint(a)
    assert before == fingerprint(b)
    df = a["tables"][0]
    df.index.get_loc(2)
    df.columns.get_loc("name")
    _ = df.index.is_unique, df.columns.is_monotonic_increasing
    assert fingerprint(a) == before
    assert freeze_arrays(a) is a
    assert fingerprint(a) == before
    assert fingerprint(a) == fingerprint(a)
    assert not a["geom"].center.flags.writeable
    assert not a["geom"].lib_group.flags.writeable
    assert not a["slots"].weights.flags.writeable
    for column in (df.iloc[:, i] for i in range(len(df.columns))):
        assert not column.to_numpy(copy=False).flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        a["geom"].center[0, 0] = -1


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(extra=1),
    lambda x: x["tables"].append(None),
    lambda x: x["geom"].center.__setitem__((0, 0), -1),
    lambda x: setattr(x["geom"], "lib_group", np.array(["M"] * 4, dtype=object)),
    lambda x: setattr(x["geom"], "new_dynamic", {"v": np.ones(1)}),
    lambda x: x["cache"][(3, "prob")].nb.update(x=3),
    lambda x: setattr(x["cache"][(3, "prob")], "other", "new"),
])
def test_nested_structural_and_content_mutation(mutate):
    value = payload()
    before = fingerprint(value)
    mutate(value)
    assert fingerprint(value) != before


def test_dictionary_order_tuple_keys_and_subclass_attributes():
    assert fingerprint({(1, "x"): 2, 3: 4}) != fingerprint({3: 4, (1, "x"): 2})
    assert fingerprint([1, 2]) != fingerprint((1, 2))
    assert fingerprint([1, 2]) != fingerprint([2, 1])
    counts = _NB(x=2)
    counts.extra = np.ones(1)
    before = fingerprint(counts)
    counts.extra = np.zeros(1)
    assert fingerprint(counts) != before
    freeze_arrays(counts)
    assert not counts.extra.flags.writeable


@pytest.mark.parametrize("mutate", [
    lambda df: df.iloc.__setitem__((0, 0), 99),
    lambda df: df.__setitem__("a", np.array([7, 8], dtype=np.int64)),
    lambda df: setattr(df, "index", pd.Index([4, 5], name="row")),
    lambda df: df.__setitem__("a", df["a"].astype(np.float64)),
    lambda df: setattr(df, "columns", ["b", "a"]),
    lambda df: df.sort_index(ascending=False, inplace=True),
    lambda df: setattr(df.index, "name", "other"),
])
def test_dataframe_values_replacement_index_dtype_order(mutate):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index([0, 1], name="row"))
    before = fingerprint(df)
    mutate(df)
    assert fingerprint(df) != before


def test_frozen_dataframe_replacement_detected_and_backing_array_readonly():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    before = fingerprint(df)
    freeze_arrays(df)
    assert not df.to_numpy(copy=False).flags.writeable
    df["a"] = [9, 10]  # pandas replacement is legal, not silently accepted
    assert fingerprint(df) != before
    equal = df.copy(deep=True)
    assert fingerprint(equal) == fingerprint(df)


def test_series_and_multiindex_public_state_only():
    index = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["group", "row"])
    series = pd.Series(np.arange(4), index=index, name="weights")
    before = fingerprint(series)
    index.get_loc(("a", 1))
    _ = index.is_unique, index.is_monotonic_increasing
    freeze_arrays(series)
    assert before == fingerprint(series)
    assert not series.to_numpy(copy=False).flags.writeable
    changed = series.rename("other")
    assert fingerprint(changed) != before


@pytest.mark.parametrize("array", [
    np.arange(12).reshape(3, 4), np.arange(20)[::3],
    np.asfortranarray(np.arange(12).reshape(3, 4)), np.array(2.5),
    np.empty((0, 3)), np.array(["µ", "a\x00b"], dtype="U3"),
    np.array([b"a", b"b"]), np.array([1 + 2j]),
    np.array([["EX", "T1"], ["T2", "M"]], dtype=object),
    np.array(["label", None, np.nan], dtype=object),
])
def test_array_storage_and_primitive_content(array):
    before = fingerprint(array)
    assert fingerprint(array.copy(order="C")) == before
    freeze_arrays(array)
    assert not array.flags.writeable
    assert fingerprint(array) == before


def test_numpy_scalar_dtype_and_shape_are_bound():
    assert fingerprint(np.int32(2)) != fingerprint(np.int64(2))
    assert fingerprint(np.array([1, 2])) != fingerprint(np.array([[1, 2]]))
    assert fingerprint(np.array([1, 2])) != fingerprint(np.array([1., 2.]))
    assert fingerprint(np.float32(1.5)) == fingerprint(np.float32(1.5))


def test_contiguous_hash_does_not_request_array_copy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected contiguous copy")
    monkeypatch.setattr(np, "ascontiguousarray", forbidden)
    assert fingerprint(np.arange(10))


@pytest.mark.parametrize("make", [
    lambda: {1, 2}, lambda: iter([1]), lambda: (lambda: None),
    lambda: StringIO("hidden native state"), lambda: np.ma.array([1], mask=[True]),
    lambda: defaultdict(list), lambda: memoryview(b"hidden"),
    lambda: np.array([object()], dtype=object),
    lambda: np.array([1, 2], dtype=object),
    lambda: np.array([("a", 1)], dtype=[("x", "U1"), ("y", "i4")]),
    lambda: np.array(["2020-01-01"], dtype="datetime64[D]"),
    lambda: pd.Series([1, None], dtype="Int64"),
    lambda: pd.Series(["a", "b"], dtype="category"),
    lambda: pd.date_range("2020-01-01", periods=2),
])
def test_unknown_types_and_dtypes_fail_closed(make):
    for audit in (freeze_arrays, fingerprint):
        with pytest.raises(TypeError, match="unsupported"):
            audit(make())


def test_hidden_slots_fail_closed():
    class Hidden:
        __slots__ = ("__dict__", "secret")
    value = Hidden()
    value.secret = np.ones(1)
    for audit in (freeze_arrays, fingerprint):
        with pytest.raises(TypeError, match="slots"):
            audit(value)


@pytest.mark.parametrize("kind", ["list", "dict", "object"])
def test_cycles_fail_closed_but_shared_references_do_not(kind):
    shared = np.arange(3)
    value = {"a": shared, "b": [shared]}
    assert fingerprint(value) == fingerprint({"a": shared.copy(), "b": [shared.copy()]})
    freeze_arrays(value)
    if kind == "list":
        cyclic = []
        cyclic.append(cyclic)
    elif kind == "dict":
        cyclic = {}
        cyclic["self"] = cyclic
    else:
        cyclic = SimpleNamespace()
        cyclic.self = cyclic
    for audit in (freeze_arrays, fingerprint):
        with pytest.raises(ValueError, match="cycle"):
            audit(cyclic)


def test_jax_small_cpu_array_identity_placement_and_no_host_copy(monkeypatch):
    jax = pytest.importorskip("jax")
    cpu = jax.devices("cpu")[0]
    array = jax.device_put(np.arange(3, dtype=np.float32), cpu)
    replacement = jax.device_put(np.arange(3, dtype=np.float32), cpu)
    expected = fingerprint({"device": array})
    calls = []
    original = type(array).block_until_ready

    def ready(self):
        calls.append(id(self))
        return original(self)

    def forbidden(*args, **kwargs):
        raise AssertionError("device-to-host copy requested")

    monkeypatch.setattr(type(array), "block_until_ready", ready)
    monkeypatch.setattr(type(array), "__array__", forbidden)
    monkeypatch.setattr(jax, "device_get", forbidden)
    assert freeze_arrays(array) is array
    assert fingerprint({"device": array}) == expected
    assert fingerprint({"device": replacement}) != expected
    assert id(array) in calls
    assert array.devices() == {cpu}
    with pytest.raises(TypeError):
        array[0] = 4


def test_pandas_primitive_string_dtype_and_missing_values():
    series = pd.Series(["EX", None, "T1"], dtype=pd.StringDtype(storage="python", na_value=np.nan))
    before = fingerprint(series)
    assert fingerprint(series.copy(deep=True)) == before
    freeze_arrays(series)
    assert fingerprint(series) == before
    changed = pd.Series(["M", None, "T1"], dtype=series.dtype)
    assert fingerprint(changed) != before


def test_jax_placement_is_in_fingerprint(monkeypatch):
    jax = pytest.importorskip("jax")
    array = jax.device_put(np.arange(2), jax.devices("cpu")[0])
    before = fingerprint(array)
    # Alter only the reported placement; no second real device is required.
    placement = SimpleNamespace(platform="synthetic-other-placement",
                                process_index=0, id=0, device_kind="test")
    monkeypatch.setattr(type(array), "devices", lambda self: [placement])
    assert fingerprint(array) != before


def test_nullable_pandas_strings_are_validated_and_frozen():
    series = pd.Series(["EX", None, "T1"], dtype=pd.StringDtype(storage="python"))
    before = fingerprint(series)
    assert fingerprint(series.copy(deep=True)) == before
    freeze_arrays(series)
    assert not series.to_numpy(copy=False).flags.writeable
    assert fingerprint(series) == before


def test_index_dtype_and_duplicate_columns_are_bound():
    a = pd.DataFrame([[1, 2], [3, 4]], columns=["same", "same"])
    b = a.copy(deep=True)
    assert fingerprint(a) == fingerprint(b)
    b.iloc[0, 1] = 99
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(pd.Index([1, 2], dtype="int32")) != fingerprint(pd.Index([1, 2], dtype="int64"))
    freeze_arrays(a)
    assert not a.to_numpy(copy=False).flags.writeable
