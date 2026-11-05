"""Primary identity extraction and onspill time transforms for C4 placeholders."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto.primaries import bunch_times, extract
from crvfastsim.veto.primary_chain import bunch_frame, coincidence_rows, rekey


def lineage():
    vec = dict(fCoordinates=dict(fX=1., fY=2., fZ=3.))
    sim = dict(valid=True, gen=38, pdg=13, id=9, time=800., pos=vec, mom=vec, prirel={"_rel": -1})
    return ak.Array([dict(run=1, subrun=2, event=3, trkmcsim=[[sim, sim]], pbtime=-210., pbtime_mc=-212.),
                     dict(run=2, subrun=2, event=3, trkmcsim=[[sim]], pbtime=-220., pbtime_mc=-221.)])


def test_extract_deduplicates_only_within_event_identity():
    table, summary = extract(lineage(), [-3904, 0, 10171])
    assert len(table) == 2
    assert summary["duplicate_genealogy_rows"] == 1
    assert summary["represented_events"] == 2
    assert table.x.iloc[0] == -3903 and table.z.iloc[0] == 10174
    assert table.t.iloc[0] == 800
    assert summary["primary_relations"] == {"-1": 2}


def test_inconsistent_duplicate_identity_rejected():
    data = lineage()
    time = ak.Array([[[800., 801.]], [[800.]]])
    data = ak.with_field(data, ak.with_field(data.trkmcsim, time, "time"), "trkmcsim")
    with pytest.raises(ValueError, match="inconsistent"):
        extract(data, [0, 0, 0])


def test_dense_event_keys_and_time_transform():
    data = lineage()
    table, _ = extract(data, [0, 0, 0])
    gen = rekey(data, table)
    assert np.array_equal(gen.event, [1, 2])
    assert np.array_equal(gen.run, [1, 1])
    steps = pd.DataFrame(dict(event=[2, 1], t0=[900., 800.], t1=[901., 801.]))
    transformed = bunch_frame(steps, data)
    np.testing.assert_allclose(transformed.t0, [679., 588.])
    np.testing.assert_allclose(transformed.t1, [680., 589.])
    np.testing.assert_allclose(steps.t0, [900., 800.])


def test_bunch_times_exact_join(tmp_path):
    data = lineage()
    with uproot.recreate(tmp_path / "source.root") as root:
        root.mktree("EventNtuple/ntuple", {k: "float64" if "pbtime" in k else "int32" for k in
                     ("evtinfo/run", "evtinfo/subrun", "evtinfo/event", "evtinfo/pbtime", "evtinfomc/pbtime")})
        root["EventNtuple/ntuple"].extend({"evtinfo/run": [2, 1], "evtinfo/subrun": [2, 2], "evtinfo/event": [3, 3],
                                          "evtinfo/pbtime": [-220., -210.], "evtinfomc/pbtime": [-221., -212.]})
    times = bunch_times(tmp_path / "source.root", data)
    assert np.array_equal(times.pbtime, [-210, -220])


def test_empty_coincidences_keep_events():
    from types import SimpleNamespace
    geom = SimpleNamespace(layer=np.array([0]), center=np.zeros((1, 3)))
    fast = coincidence_rows([], geom, lineage())
    assert len(fast) == 2 and ak.sum(ak.num(fast.coincs)) == 0
    assert set(fast.coincs.fields) == {"time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType"}


def test_empty_charges_bypass_digi_and_reco(monkeypatch):
    from types import SimpleNamespace
    from crvfastsim import chain
    from crvfastsim.veto.primary_chain import reconstruct
    def unexpected(*args, **kwargs):
        raise AssertionError("empty charges must bypass digitisation")
    monkeypatch.setattr(chain, "digis", unexpected)
    charges = {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
    geom = SimpleNamespace(layer=np.array([0]), center=np.zeros((1, 3)))
    timers = {}
    result = reconstruct(charges, None, None, None, geom, None, lineage(), timers)
    assert len(result) == 2 and ak.sum(ak.num(result.coincs)) == 0
    assert timers == {"digis": 0, "pulses": 0}


def test_selected_lineage_uses_original_indices():
    from crvfastsim.veto.cosmic_ana import selected_lineage
    data = {"trkmc": {"trkmcsim": ak.Array([[[{"id": 7}], [{"id": 9}]], [[{"id": 3}]]])}}
    mask = ak.Array([[False, True], [False]])
    result = selected_lineage(data, mask, ak.any(mask, axis=1))
    assert np.array_equal(ak.flatten(result["track_index"]), [1])
    assert np.array_equal(ak.flatten(result["trkmcsim"].id, axis=None), [9])


def test_photon_folding_before_sipm(monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from crvfastsim import readout
    from crvfastsim.response import photons_jax, sipm_jax
    from crvfastsim.veto.primary_chain import onspill_charges
    steps = pd.DataFrame(dict(run=[1], subrun=[0], event=[1], sp=[1]))
    parts = pd.DataFrame(dict(run=[1], subrun=[0], event=[1], key=[1], pdg=[13]))
    monkeypatch.setattr(readout, "effective_status", lambda cond: np.array([0, 1]))
    monkeypatch.setattr(photons_jax, "make_photons", lambda *args: (
        np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([-10., 1705., 400., 900.])))
    def capture(labels, times, step, pmap, cfg, rng, key, t_end):
        np.testing.assert_allclose(times, [1685., 10., 400.])
        np.testing.assert_array_equal(labels, [32768, 32768, 32768])
        assert t_end == 1500
        return {"time": np.array([])}
    monkeypatch.setattr(sipm_jax, "simulate", capture)
    cfg = SimpleNamespace(window=(200., 1500.), response=None, sipm=None)
    onspill_charges(steps, parts, None, SimpleNamespace(deviation=None), cfg,
                   np.random.default_rng(1), None, None, {}, 1695.)


def test_coincidence_times_return_to_reco_frame(monkeypatch):
    from crvfastsim.reco import ntview
    co = ak.Array([{"crvcoincs.time": [500.], "crvcoincs.PEs": [100.],
                    "crvcoincs.nLayers": [3], "crvcoincs.sectorType": [1]},
                   {"crvcoincs.time": [], "crvcoincs.PEs": [],
                    "crvcoincs.nLayers": [], "crvcoincs.sectorType": []}])
    monkeypatch.setattr(ntview, "build", lambda *args, **kwargs: (co, None, None))
    outs = [dict(cl_event=np.array([1]), cl_start=np.array([490.]), cl_end=np.array([510.]))]
    result = coincidence_rows(outs, None, lineage())
    np.testing.assert_allclose(ak.flatten(result.coincs.time), [710.])
    np.testing.assert_allclose(ak.flatten(result.coincs.timeStart), [700.])
    np.testing.assert_allclose(ak.flatten(result.coincs.timeEnd), [720.])
