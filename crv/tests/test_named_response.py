"""Synthetic named V2/fasttable contracts, without conditions assets or geometry downloads."""
import copy
import dataclasses
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import geometry, named_conditions, runconfig
from crvfastsim.analyses import v2compare as V
from valhist import fasttables as FT


def synthetic_geometry(name):
    rc = runconfig.get(name)
    names = list(rc.sectors)
    if name.startswith("run1"):
        counts, types = [1088, 192], [1, 1]
    else:
        # Stock sector populations; synthetic positions only.
        modules = [5, 13, 1, 1, 2, 4, 4, 11, 4, 3, 2, 4, 13, 3, 1, 1, 4, 3, 1, 1, 1]
        counts = [64 * m for m in modules] + [80, 16]
        types = [1] * 6 + [2] * 3 + [3] * 5 + [4] * 2 + [5] + [6] * 4 + [7] * 2
    sector = np.repeat(np.arange(len(names)), counts)
    n = len(sector)
    g = geometry.CRVGeometry(names, sector, np.zeros(n, int), np.arange(n) % 4,
                             np.arange(n) % 16, np.zeros((n, 3)), np.ones((n, 3)),
                             np.tile([0, 1, 2], (n, 1)))
    cfg = {"crs.nLayers": 4, **{f"crs.sectorType{s}": t for s, t in zip(names, types)}}
    return g, cfg


@pytest.fixture
def named(monkeypatch):
    monkeypatch.setattr(runconfig.RunConfig, "geometry", lambda self: synthetic_geometry(self.name)[0])
    monkeypatch.setattr(runconfig.RunConfig, "counters_cfg", lambda self: synthetic_geometry(self.name)[1])
    return runconfig.get


def populated(rc, n=10000):
    S = V._new_state(rc)
    h = np.histogram(np.linspace(5.05, 149.95, n), V.PE_EDGES)[0].tolist()
    for arm in ("fast", "ref"):
        A = S[arm]
        A["events"] = 10000
        A["bar_rate"] = [10] * len(A["bar_rate"])
        for G in A["groups"].values():
            G.update(npulse=n, h_pe=h[:], nclus=1000)
    return S


@pytest.mark.parametrize("name,nbars,ngroups", [("run1a_v01", 1280, 1), ("run2_v10", 5344, 7)])
def test_dynamic_counts_all_sectors_and_immutable_identity(named, name, nbars, ngroups):
    rc = named(name)
    layout = V.response_layout(rc)
    assert layout.nbars == nbars
    assert len(layout.groups) == ngroups
    assert sorted(s for _, _, secs in layout.groups for s in secs) == list(range(len(rc.sectors)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.nbars = 576
    S = V._new_state(rc)
    assert len(S["fast"]["bar_rate"]) == nbars
    # State identity is an immutable serialized snapshot, not the mutable registry dictionaries.
    assert isinstance(S["configuration"], str)
    assert V._state_layout(json.loads(json.dumps(S))) == layout
    assert V._new_state() == {"fast": V._new_arm(), "ref": V._new_arm()}


def test_named_identical_supported_and_diagnostic_not_gate(named):
    S = populated(named("run2_v10"))
    first = next(iter(S["fast"]["groups"]))
    for arm, num in (("fast", 0), ("ref", 1000)):
        S[arm]["groups"][first].update(eff_den=[1000] * 4, eff_num=[num] * 4)
    out = V.summarise(S)
    assert out["metrics_pass"] and out["metrics_status"] == "pass"
    assert not out["pass"] and out["status"] == "unsupported"
    assert out["input_provenance_status"].startswith("unverified")
    assert not out["groups"][first]["diagnostic_layer_eff_ok"]
    assert "diagnostic_only" in out["layer_efficiency_status"]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("what", ["empty", "ten", "missing", "no_clusters", "missing_tree", "49", "1999"])
def test_named_unsupported_never_passes(named, what):
    rc = named("run1a_v01")
    S = V._new_state(rc) if what == "empty" else populated(rc, int(what) if what.isdigit() else 10000)
    g = next(iter(S["ref"]["groups"]))
    if what == "ten":
        S = populated(rc, 10)
        S["ref"]["events"] = S["fast"]["events"] = 10
    elif what == "missing":
        del S["ref"]["groups"][g]
    elif what == "no_clusters":
        S["ref"]["groups"][g]["nclus"] = 0
    elif what == "missing_tree":
        S["fast"]["missing_trees"] = ["pulses"]
    out = V.summarise(S)
    assert not out["pass"] and out["status"] == "unsupported"
    assert not out["groups"][g]["pass"]
    json.dumps(out, allow_nan=False)


def test_named_support_thresholds_and_failed_shift(named):
    S = populated(named("run1a_v01"), 50)
    r = next(iter(V.summarise(S)["groups"].values()))
    assert r["checks"]["pe_median"] and not r["checks"]["pe_ratio"]
    S = populated(named("run1a_v01"), 2000)
    assert V.summarise(S)["metrics_pass"] and not V.summarise(S)["pass"]
    S = populated(named("run1a_v01"), 100000)
    G = next(iter(S["fast"]["groups"].values()))
    G["h_pe"] = np.roll(G["h_pe"], 100).tolist()
    r = next(iter(V.summarise(S)["groups"].values()))
    assert not r["checks"]["pe_median"]


def test_merge_identity_and_shapes_checked_before_accumulation(named):
    rc = named("run1a_v01")
    A = populated(rc)
    T = V._new_state(rc)
    V._merge_into(T, json.loads(json.dumps(A)))
    V._merge_into(T, A)
    assert T["fast"]["events"] == 2 * A["fast"]["events"]
    changed = dataclasses.replace(rc, photon_yield_scale=0.1)
    moved = rc.geometry()
    moved.center[0, 0] = 1
    broken = copy.deepcopy(A)
    del broken["ref"]["groups"][next(iter(broken["ref"]["groups"]))]
    bad_shape = copy.deepcopy(A)
    bad_shape["ref"]["bar_rate"].pop()
    for incompatible in (V._new_state(), V._new_state(named("run1b_v01")), V._new_state(changed),
                         V._new_state(rc, moved), broken, bad_shape):
        before = copy.deepcopy(T)
        with pytest.raises(ValueError):
            V._merge_into(T, incompatible)
        assert T == before


def test_fill_counts_every_sector_once_per_event_bar(named, tmp_path):
    rc = named("run2_v10")
    geom = rc.geometry()
    bars = np.r_[0, np.flatnonzero(np.diff(geom.sector)) + 1]
    n = len(bars)
    p = tmp_path / "reco.root"
    with uproot.recreate(p) as f:
        f["crvdump/events"] = {k: np.array([1]) for k in ("run", "subrun", "event")}
        f["crvdump/pulses"] = {"run": np.ones(2*n, int), "subrun": np.ones(2*n, int),
                               "event": np.ones(2*n, int), "bar": np.repeat(bars, 2),
                               "pe": np.full(2*n, 25.0), "seq": np.zeros(2*n, int)}
        f["crvdump/clusters"] = {"sectortype": np.arange(1, 8)}
    S = V._new_state(rc)
    V._fill_arm(S["fast"], [str(p)], None, geom, V._state_layout(S))
    assert sum(S["fast"]["bar_rate"]) == n
    assert sum(G["npulse"] for G in S["fast"]["groups"].values()) == 2*n
    assert all(G["nclus"] == 1 for G in S["fast"]["groups"].values())
    assert S["fast"]["missing_trees"] == []


def test_missing_trees_and_empty_pulses_safe(named, tmp_path):
    rc = named("run1a_v01")
    p = tmp_path / "empty.root"
    with uproot.recreate(p) as f:
        f["crvdump/events"] = {k: np.array([1]) for k in ("run", "subrun", "event")}
    S = V._new_state(rc)
    V._fill_arm(S["fast"], [str(p)], None, rc.geometry(), V._state_layout(S))
    assert S["fast"]["missing_trees"] == ["clusters", "pulses"]
    assert not V.summarise(S)["pass"]
    # Single-group plot uses a two-dimensional axes array, even without entries.
    assert not V.report(S, str(tmp_path / "report"))["pass"]


def test_empty_efficiency_pulses_safe(named, monkeypatch):
    rc = named("run1a_v01")
    S = V._new_state(rc)
    p = ak.Array({k: np.zeros(0, int) for k in ("run", "subrun", "event", "bar", "pe", "seq")})
    monkeypatch.setattr(V.tables, "read_many", lambda *a: pd.DataFrame())
    x = pd.DataFrame({"run": [1], "subrun": [1], "event": [1], "bar": [0]})
    monkeypatch.setattr(V.tables, "primary_muon_crossings", lambda *a: x)
    V._fill_eff(S["fast"], p, np.array([(1 << 40) | (1 << 20) | 1]), ["fake"], rc.geometry(), V._state_layout(S))
    G = next(iter(S["fast"]["groups"].values()))
    assert sum(G["eff_den"]) == 1 and sum(G["eff_num"]) == 0


@pytest.mark.parametrize("name", ["run1a_v01", "run2_v10"])
def test_fasttable_named_config_helper(named, monkeypatch, name):
    rc = named(name)
    seen = []
    sentinel = object()
    def load(manifest, passed_rc, geom):
        seen.append((manifest, passed_rc, geom.nbars))
        return sentinel, {"verified": "synthetic"}
    monkeypatch.setattr(named_conditions, "load_conditions", load)
    geom, ccfg, cond, cfg, metadata = FT.configure(run_config=name, conditions_manifest="manifest.json")
    assert seen == [("manifest.json", rc, geom.nbars)]
    assert cond is sentinel and ccfg == rc.counters_cfg()
    assert cfg.response == rc.response() and cfg.finder == rc.finder()
    assert cfg.window == runconfig.OFFSPILL_WINDOW and rc.window == runconfig.ONSPILL_WINDOW
    assert not metadata["onspill_equivalence"]
    assert metadata["timing_policy"] == "fixed_absolute_crop_no_pbtime_rebase"
    _, _, _, on, meta = FT.configure(run_config=name, conditions_manifest="manifest.json",
                                     window="onspill", no_opposite_side=True)
    assert on.window == runconfig.ONSPILL_WINDOW and not on.finder.add_opposite_side_pulses
    assert not meta["onspill_equivalence"]
    for extra in ({"calib": "x"}, {"counters": "x"}, {"deviation": "x"}):
        with pytest.raises(ValueError):
            FT.configure(run_config=name, conditions_manifest="manifest.json", **extra)
    with pytest.raises(ValueError):
        FT.configure(run_config=name)


def test_fasttable_legacy_defaults(monkeypatch):
    geom, ccfg = synthetic_geometry("run1a_v01")
    monkeypatch.setattr(FT.geometry, "find_counters", lambda path: __file__)
    monkeypatch.setattr(FT.geometry, "from_offline_config", lambda path: geom)
    monkeypatch.setattr(FT.geometry, "parse_simpleconfig", lambda text: ccfg)
    monkeypatch.setattr(FT.chain.Conditions, "from_csv", lambda *a: "legacy")
    _, _, cond, cfg, metadata = FT.configure(calib="calib.csv")
    assert cond == "legacy" and metadata is None
    assert cfg.response == FT.chain.ChainConfig().response
    assert cfg.finder == FT.C.FinderConfig() and cfg.window == (0.0, 100000.0)
    with pytest.raises(ValueError):
        FT.configure(conditions_manifest="manifest.json")


def test_named_run_emit_merge_and_empty_group(named, monkeypatch, tmp_path):
    rc = named("run2_v10")
    reference = populated(rc)
    def fill(arm, paths, steps, geom, layout):
        assert geom.nbars == 5344 and layout == V.response_layout(rc)
        arm.update(copy.deepcopy(reference["ref"]))
    monkeypatch.setattr(V, "_fill_arm", fill)
    emit = tmp_path / "batch.json"
    V.run(["candidate"], str(tmp_path), compare=["reference"], run_config=rc, emit=str(emit))
    monkeypatch.setattr(V, "report", lambda S, *a: V.summarise(S))
    result = V.run([str(emit)], str(tmp_path), run_config="run2_v10", merge=True)
    assert result["metrics_pass"] and not result["pass"] and len(result["groups"]) == 7
    with pytest.raises(ValueError):
        V.run([str(emit)], str(tmp_path), merge=True)
    with pytest.raises(ValueError, match="positive integers"):
        V.run(["candidate"], str(tmp_path), compare=["reference"], run_config=rc, generated=0)
    # A wholly empty configured group is reported, not dropped, even with other populated groups.
    S = copy.deepcopy(reference)
    g = next(iter(S["ref"]["groups"]))
    S["ref"]["groups"][g] = V._new_state(rc)["ref"]["groups"][g]
    out = V.summarise(S)
    assert len(out["groups"]) == 7 and out["groups"][g]["status"] == "unsupported"
    assert not out["pass"]


def test_fasttable_cli_reco_wiring_and_sidecar(named, monkeypatch, tmp_path):
    import sys
    rc = named("run1a_v01")
    geom = rc.geometry()
    cfg = dataclasses.replace(rc.chain_config(), window=runconfig.OFFSPILL_WINDOW)
    seen = {}
    def configured(**kw):
        seen.update(kw)
        return geom, rc.counters_cfg(), "conditions", cfg, {"onspill_equivalence": False}
    monkeypatch.setattr(FT, "configure", configured)
    monkeypatch.setattr(FT, "stock_digis", lambda _: {"run": 1400, "sentinel": "digis"})
    def reco(D, cond, passed_geom, ccfg, **kw):
        assert D == {"sentinel": "digis"} and cond == "conditions" and passed_geom is geom
        assert kw["fcfg"] == rc.finder()
        return "pulses", "clusters"
    monkeypatch.setattr(FT, "fast_reco", reco)
    inp, out = tmp_path / "input.root", tmp_path / "output.root"
    with uproot.recreate(inp) as f:
        f.mktree("crvdump/events", {"subrun": "int32", "event": "int32"})
        f["crvdump/events"].extend({"subrun": np.array([1]), "event": np.array([7])})
    writes = []
    monkeypatch.setattr(FT, "write", lambda *args: writes.append(args))
    monkeypatch.setattr(sys, "argv", ["fasttables", "reco", str(inp), str(out), "--run-config", rc.name,
                                      "--conditions-manifest", "manifest.json"])
    FT.main()
    assert seen["window"] == "offspill" and seen["conditions_manifest"] == "manifest.json"
    assert writes[0][1] == 1400 and writes[0][3:5] == ("pulses", "clusters")
    metadata = json.loads((tmp_path / "output.root.metadata.json").read_text())
    assert metadata["stage"] == "reco" and not metadata["window_applied"]
    assert not metadata["onspill_equivalence"]
