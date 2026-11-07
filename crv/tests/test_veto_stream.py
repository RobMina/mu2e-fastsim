"""veto.stream: one setup for many batches, batch partition and seeds, the per-track veto decision."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import primary_chain, stream
from crvfastsim.veto.card import VetoCard


def selection(n=5):
    return ak.Array(dict(run=[1] * n, subrun=[2] * n, event=list(range(10, 10 + n)), shard=[0] * n,
                         t_mid=[[100.0]] * n, p_mid=[[100.0]] * n))


def primaries(events):
    return pd.DataFrame(dict(run=1, subrun=2, event=events, pdg=13, shard=0))


def fake_run(calls):
    def run(sel, prim, gdml, calib, deviation, seed, period, library_dir, input_kind, census, setup=None):
        calls.append(dict(events=ak.to_list(sel.event), prim=prim.event.tolist(), seed=seed, setup=setup))
        n = len(sel)
        coincs = ak.zip({"time": ak.Array([[95.0]] * n), "timeStart": ak.Array([[500.0]] * n),
                         "timeEnd": ak.Array([[600.0]] * n), "PEs": ak.Array([[30.0]] * n),
                         "nLayers": ak.Array([[3]] * n), "sectorType": ak.Array([[1]] * n)})
        fast = ak.zip({"run": sel.run, "subrun": sel.subrun, "event": sel.event, "coincs": coincs}, depth_limit=1)
        return fast, dict(timers={"setup_s": 0.0}, steps=1, clusters=n, limitations=[])
    return run


def test_batches_partition_primaries_in_selection_order():
    sel = selection(5)
    prim = primaries([14, 10, 10, 12, 13, 11])
    parts = list(stream.batches(sel, prim, None, 2))
    assert [ak.to_list(p[1].event) for p in parts] == [[10, 11], [12, 13], [14]]
    assert [sorted(p[2].event) for p in parts] == [[10, 10, 11], [12, 13], [14]]
    assert len(list(stream.batches(sel, prim, None, None))) == 1


def test_primary_outside_selection_is_refused():
    with pytest.raises(ValueError, match="absent"):
        list(stream.batches(selection(2), primaries([10, 99]), None, None))


def test_single_batch_keeps_the_run_seed_and_batches_get_distinct_seeds():
    assert stream.batch_seed(7, 0, 0, True) == 7
    seeds = {stream.batch_seed(7, f, b, False) for f in range(2) for b in range(3)}
    assert len(seeds) == 6 and 7 not in seeds


def test_stream_reuses_one_setup_and_concatenates_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(primary_chain, "run", fake_run(calls))
    setup = object()
    inputs = [(selection(3), primaries([10, 11, 12]), None), (selection(2), primaries([10, 11]), None)]
    fast, reports, veto = stream.stream(inputs, setup, "g", "c", "d", seed=5, batch_events=2,
                                        card=VetoCard(dt_lo=0, dt_hi=150))
    assert [c["events"] for c in calls] == [[10, 11], [12], [10, 11]]
    assert all(c["setup"] is setup for c in calls)
    assert len({c["seed"] for c in calls}) == 3
    assert ak.to_list(fast.event) == [10, 11, 12, 10, 11]
    assert [(r["file"], r["batch"]) for r in reports] == [(0, 0), (0, 1), (1, 0)]
    v, g = veto
    assert ak.to_list(v) == [[True]] * 5 and g.all()


def test_one_file_one_batch_is_the_plain_run(monkeypatch):
    calls = []
    monkeypatch.setattr(primary_chain, "run", fake_run(calls))
    stream.stream([(selection(3), primaries([10, 11, 12]), None)], None, "g", "c", "d", seed=11)
    assert len(calls) == 1 and calls[0]["seed"] == 11


def test_run_rejects_a_setup_for_other_inputs():
    setup = dict(inputs=("other.gdml", "c", "d", None))
    sel = ak.with_field(selection(1), [0.0], "pbtime_mc")
    prim = primaries([10]).assign(x=0., y=0., z=0., px=0., py=0., pz=1000., t=0., idx=0)
    with pytest.raises(ValueError, match="prepared C4 setup"):
        primary_chain.run(sel, prim, "g", "c", "d", setup=setup)
