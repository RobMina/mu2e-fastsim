"""W1 and W2-W4 approximation levels reach the chains' code paths (beamfastsim.approximations)."""
import json

import numpy as np
import pandas as pd
import pytest

from mu2efastsim import approximations as A
from beamfastsim import approximations as BA
from beamfastsim import cetracker as H


class Reached(Exception):
    pass


def test_declarations_load_by_workflow_and_alias():
    assert A.load("beamfastsim", "W1") is BA.W1
    for name in ("W2", "W3", "W4", "W2-W4"):
        assert A.load("beamfastsim", name) is BA.W2
    with pytest.raises(ValueError, match="name one"):
        A.load("beamfastsim")


def test_w2_levels_are_device_chain_keywords():
    import inspect
    params = inspect.signature(H.DeviceChain.__init__).parameters
    for a in BA.W2.approximations:
        for level in a.levels:
            for key in a.sets.get(level, {}):
                assert key in params, (a.name, key)


def _cetracker_main(monkeypatch, tmp_path, extra):
    """cetracker main up to the DeviceChain construction; returns its keywords and the parsed approximations."""
    seen = {}
    g = pd.DataFrame({"run": [1], "subrun": [1], "event": [1]})
    monkeypatch.setattr(H.tables, "expand", lambda paths: list(paths))
    monkeypatch.setattr(H, "starts", lambda paths: g)
    monkeypatch.setattr(H.cechain, "from_offline", lambda geom, root: None)

    def chain(*args, **kw):
        seen.update(kw)
        raise Reached
    monkeypatch.setattr(H, "DeviceChain", chain)
    argv = ["t.root", "--outdir", str(tmp_path), "--gdml", "g", "--maps", "m", "--offline-root", "o",
            "--backend", "device", "--tracker", "device", *extra]
    with pytest.raises(Reached):
        H.main(argv)
    return seen


def test_cetracker_default_passes_todays_arguments(monkeypatch, tmp_path):
    kw = _cetracker_main(monkeypatch, tmp_path, [])
    assert kw["albedo"] == "default" and kw["complete_outer_wall"] is False


def test_cetracker_fast_preset_turns_the_albedo_off(monkeypatch, tmp_path):
    kw = _cetracker_main(monkeypatch, tmp_path, ["--approx-preset", "fast"])
    assert kw["albedo"] is None and kw["complete_outer_wall"] is False


def test_cetracker_fidelity_preset_takes_the_complete_wall(monkeypatch, tmp_path):
    kw = _cetracker_main(monkeypatch, tmp_path, ["--approx-preset", "fidelity"])
    assert kw["albedo"] == "default" and kw["complete_outer_wall"] is True
    kw = _cetracker_main(monkeypatch, tmp_path, ["--approx", "straw_wall=complete", "--approx", "calo_albedo=none"])
    assert kw["albedo"] is None and kw["complete_outer_wall"] is True


def test_tracker_levels_leave_the_response_tables_alone():
    for preset in A.PRESETS:
        assert BA.W2.select(preset).apply_kwargs({"response": "default"})["response"] == "default"


def test_cetracker_refuses_an_option_against_the_level(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _cetracker_main(monkeypatch, tmp_path, ["--albedo", "lib.npz", "--approx", "calo_albedo=none"])
    with pytest.raises(SystemExit):
        _cetracker_main(monkeypatch, tmp_path, ["--approx", "primary_only_steps=off"])


def test_bench_run_forwards_levels_to_the_device_chain(monkeypatch):
    seen = {}

    def chain(**kw):
        seen.update(kw)
        raise Reached
    monkeypatch.setattr(H, "DeviceChain", chain)
    monkeypatch.setattr(H, "_CHAIN", {})
    kw = BA.W2.select("fast").apply_kwargs({"offline_root": "o"})
    with pytest.raises(Reached):
        H.bench_run(({"x": np.zeros(1)}, 1), **kw)
    assert seen == {"offline_root": "o", "albedo": None}
    seen.clear()
    monkeypatch.setattr(H, "_CHAIN", {})
    kw = BA.W2.select("fidelity").apply_kwargs({"offline_root": "o", "response": "default"})
    with pytest.raises(Reached):
        H.bench_run(({"x": np.zeros(1)}, 1), **kw)
    assert seen == {"offline_root": "o", "response": "default", "complete_outer_wall": True}


def test_tsdevice_stock_ds_entrance_reaches_the_ds_only_arm(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    from beamfastsim import tsdevice as dev
    n = 2
    ref = dict(id=np.arange(n), shard=np.zeros(n), pdg=np.full(n, 13), batch=np.zeros(n), x=np.zeros((n, 3)),
               p=np.ones((n, 3)), t=np.zeros(n), stock_x=np.zeros((n, 3)), stock_p=np.ones((n, 3)),
               stock_t=np.zeros(n), stock_pdg=np.full(n, 13), stock_reached=np.ones(n, bool))
    called = []
    monkeypatch.setattr(dev, "load_chain", lambda *a, **k: object())
    monkeypatch.setattr(dev, "load_references", lambda refs, limit: ref)

    def ds_only(*a, **k):
        called.append("ds_only")
        raise Reached

    def full(*a, **k):
        called.append("full")
        raise Reached
    monkeypatch.setattr(dev, "run_ds_only", ds_only)
    monkeypatch.setattr(dev, "run_chain", full)
    base = ["--reference", "r", "--gdml", "g", "--field-config", "f", "--offline-root", "o", "--maps", "m",
            "--ds-geom", "d", "--ds-map", "dm", "--out", str(tmp_path / "o.npz")]
    for extra, arm in (([], "full"), (["--approx-preset", "fidelity"], "ds_only"),
                       (["--approx", "ts_transport=stock_ds_entrance"], "ds_only")):
        with pytest.raises(Reached):
            dev.main(base + extra)
        assert called[-1] == arm
    with pytest.raises(SystemExit):
        dev.main(base + ["--arm", "saved-ts", "--approx-preset", "fidelity"])
    with pytest.raises(SystemExit):
        dev.main(base + ["--approx", "pion_nuclear_interactions=on"])


def test_records_are_json_and_list_the_fixed_approximations():
    for w in (BA.W1, BA.W2):
        for preset in A.PRESETS:
            r = w.select(preset).record()
            json.dumps(r)
            assert r["fixed"] == sorted(w.fixed)
    assert BA.W2.select("fidelity").record()["no_stock_path"] == ["calo_albedo", "straw_wall"]


def test_fidelity_preset_without_a_response_sets_only_the_wall():
    assert BA.W2.select("fidelity").apply_kwargs({}) == {"complete_outer_wall": True}
