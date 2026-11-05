"""Synthetic contracts for catalog and ntuple controls, no private data."""
import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Catalog:
    def get_dataset(self, *, did, exact_file_count):
        assert did == "mu2e:sample" and exact_file_count
        return {"file_count": 12}

    def query(self, query):
        assert query == "files from mu2e:sample limit 1"
        yield {"namespace": "mu2e", "name": "example"}

    def get_file(self, *, did, with_metadata):
        assert did == "mu2e:example" and with_metadata
        return {"name": "example", "metadata": {"events": 7}}


def test_readonly_census_does_not_certify_products():
    result = load_script("census").census(Catalog(), ["mu2e:sample"])
    row = result["datasets"][0]
    assert row["dataset"]["file_count"] == 12
    assert row["example"]["metadata"]["events"] == 7
    assert row["product_content_verified"] is False


def test_census_lazy_errors_surface():
    class Broken(Catalog):
        def query(self, query):
            yield from ()
            raise RuntimeError("catalog unavailable")
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        load_script("census").census(Broken(), ["mu2e:sample"])


class Branch:
    def __init__(self, data):
        self.data = ak.Array(data)

    def array(self):
        return self.data


class Tree:
    num_entries = 3

    def __init__(self):
        self.branches = {
            "calohits.": Branch([
                [{"nSiPMs_": 2, "time_": 12.5, "timeErr_": np.sqrt(.5),
                  "eDep_": 5, "eDepErr_": np.sqrt(2)}],
                [],
                [{"nSiPMs_": 1, "time_": 4, "timeErr_": 1,
                  "eDep_": 3, "eDepErr_": 2}],
            ]),
            "calorecodigis.": Branch([
                [{"caloHitIdx_": 0, "time_": 12, "timeErr_": 1, "eDep_": 4, "eDepErr_": 2},
                 {"caloHitIdx_": 0, "time_": 13, "timeErr_": 1, "eDep_": 6, "eDepErr_": 2}],
                [{"caloHitIdx_": -1, "time_": 2, "timeErr_": 1, "eDep_": 9, "eDepErr_": 2}],
                [{"caloHitIdx_": 0, "time_": 4, "timeErr_": 1, "eDep_": 3, "eDepErr_": 2}],
            ]),
        }
        for key in ("time_", "timeErr_", "eDep_", "eDepErr_"):
            hits = self.branches["calohits."].data
            self.branches["calohits."].data = ak.with_field(hits, ak.values_astype(hits[key], np.float32), key)

    def __getitem__(self, key):
        return self.branches[key]


class File:
    def __init__(self, tree):
        self.tree = tree

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getitem__(self, key):
        assert key == "EventNtuple/ntuple"
        return self.tree


def test_weight_control_empty_event_and_unlinked_digi(monkeypatch):
    script = load_script("check_hit_weights")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(Tree()))
    result = script.check("unused", "EventNtuple/ntuple", 3)
    assert result["events"] == 3 and result["hits"] == 2
    assert result["reco_digis"] == 4 and result["linked_reco_digis"] == 3
    assert all(q["values_float32_equal"] and q["errors_float32_equal"]
               for q in result["quantities"].values())


def test_control_refuses_wrong_entry_count(monkeypatch):
    script = load_script("check_hit_weights")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(Tree()))
    with pytest.raises(ValueError, match="expected 4 entries"):
        script.check("unused", "EventNtuple/ntuple", 4)


def test_control_accepts_split_record_names(monkeypatch):
    script = load_script("check_hit_weights")
    tree = Tree()
    tree.branches = {key.rstrip("."): Branch(ak.zip(
        {key.rstrip(".") + "." + field: branch.data[field]
         for field in ak.fields(branch.data)}, depth_limit=2))
        for key, branch in tree.branches.items()}
    monkeypatch.setattr(script.uproot, "open", lambda _: File(tree))
    result = script.check("unused", "EventNtuple/ntuple", 3)
    assert result["hits"] == 2
    assert all(q["values_float32_equal"] for q in result["quantities"].values())


def test_control_refuses_missing_member(monkeypatch):
    script = load_script("check_hit_weights")
    tree = Tree()
    reco = tree.branches["calorecodigis."].data
    idx = reco["caloHitIdx_"]
    tree.branches["calorecodigis."].data = ak.with_field(
        reco, ak.where(ak.local_index(idx) == 1, -1, idx), "caloHitIdx_")
    monkeypatch.setattr(script.uproot, "open", lambda _: File(tree))
    with pytest.raises(ValueError, match="bad_hits=1/2"):
        script.check("unused", "EventNtuple/ntuple", 3)
