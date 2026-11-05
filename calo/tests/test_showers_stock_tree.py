"""Split-parent TTree selection with real baskets and optional production-file replay.

The synthetic helper supplies the in-memory hierarchy because uproot's writer
flattens record branches. Only opening the fixture is patched: selection, basket
decoding and AsGrouped assembly use uproot. ROOT-produced TBranchElement metadata
is covered by the optional CALO_SHOWER_STOCK_NTUPLES path-list test, not the helper.
"""
import os

import awkward as ak
import numpy as np
import pytest
import uproot
from uproot.interpretation.grouped import AsGrouped

from calofastsim.showers import prepare_chain
from calofastsim.showers.export import EVENT, keys


FIELDS = dict(energy="energyDep_", time="time_", x="cog_.fCoordinates.fX",
              y="cog_.fCoordinates.fY", z="cog_.fCoordinates.fZ")


def attach_group(tree, name, leafnames):
    """Isolate the uproot-model internals used only to construct synthetic parents."""
    leaves = [tree[n] for n in leafnames]
    interpretations = {b.name: b.interpretation for b in leaves}
    parent = uproot.models.TBranch.Model_TBranch_v13.empty()
    parent._file = tree._file
    parent._parent = tree
    parent._members.update(fName=name, fTitle=name, fEntries=tree.num_entries, fBranches=leaves)
    parent._lookup = {b.name: b for b in leaves}
    parent._cache_key = None
    parent._interpretation = AsGrouped(parent, interpretations)
    for branch in leaves:
        branch._parent = parent
        branch._cache_key = None
    tree._members["fBranches"] = [b for b in tree.branches if all(b is not leaf for leaf in leaves)] + [parent]
    tree._lookup = {b.name: b for b in tree.branches}


def test_split_tree_parents_full_identity_highest_cycle(tmp_path, monkeypatch):
    ids = dict(run=np.array([1, 2, 2], np.int32), subrun=np.array([0, 0, 1], np.int32),
               event=np.ones(3, np.int32))
    values = dict(energy=ak.Array([[11., 30.], [], [20.]]), time=ak.Array([[2., 3.], [], [4.]]),
                  x=ak.Array([[12., 13.], [], [14.]]), y=ak.Array([[22., 23.], [], [24.]]),
                  z=ak.Array([[32., 33.], [], [34.]]))
    table = {k: v[[2, 0, 1]] for k, v in ids.items()}
    table.update({"caloclusters." + FIELDS[k]: v for k, v in values.items()})
    path = tmp_path / "split.root"
    with uproot.recreate(path) as out:
        out.mktree("prd/ntuple", {k: v[:1] for k, v in table.items()})
        out.mktree("prd/ntuple", table)
    real_open = uproot.open

    def grouped_open(filename):
        source = real_open(filename)
        tree = source["prd/ntuple"]
        assert tree.classname == "TTree" and tree.num_entries == 3
        assert source["prd/ntuple;1"].num_entries == 1
        attach_group(tree, "evtinfo", EVENT)
        attach_group(tree, "caloclusters", ["caloclusters." + field for field in FIELDS.values()])
        assert tree["evtinfo"].keys() == list(EVENT)
        return source

    monkeypatch.setattr(prepare_chain.uproot, "open", grouped_open)
    result = prepare_chain.stock_clusters(path, keys(ids, EVENT))
    np.testing.assert_array_equal(result["energy"], [0., 20., 30.])
    for name, wanted in (("time", [4., 3.]), ("x", [14., 13.]), ("y", [24., 23.]), ("z", [34., 33.])):
        assert np.isnan(result[name][0])
        np.testing.assert_array_equal(result[name][1:], wanted)


STOCK_PATHS = os.environ.get("CALO_SHOWER_STOCK_NTUPLES", "").split(os.pathsep)


@pytest.mark.parametrize("stock_path", STOCK_PATHS, ids=[f"stock-{i}" for i in range(len(STOCK_PATHS))])
def test_root_produced_split_tree(stock_path):
    """Replay caller-selected saved TTrees without shipping private event arrays."""
    if not stock_path:
        pytest.skip("set CALO_SHOWER_STOCK_NTUPLES to saved stock TTrees for metadata replay")
    with uproot.open(stock_path) as source:
        tree = source["prd/ntuple"]
        assert tree.classname == "TTree"
        assert tree["evtinfo"].classname == "TBranchElement"
        assert set(EVENT) <= set(tree["evtinfo"].keys())
        evt = tree["evtinfo"].array()
        clusters = tree["caloclusters"].array()
    event_ids = keys({k: ak.to_numpy(evt[k]) for k in EVENT}, EVENT)[::-1]
    result = prepare_chain.stock_clusters(stock_path, event_ids)
    order = ak.argsort(clusters["caloclusters.energyDep_"], axis=1, ascending=False, stable=True)
    for name, field in FIELDS.items():
        fill = 0. if name == "energy" else np.nan
        expected = ak.to_numpy(ak.fill_none(ak.firsts(clusters["caloclusters." + field][order]), fill))[::-1]
        np.testing.assert_array_equal(result[name], expected)
