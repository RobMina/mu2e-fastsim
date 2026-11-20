"""beamfastsim.ptgt_point: the design cache key, cache reuse, the point's metrics and the chain arguments (no JAX run)."""
import argparse
import json

import numpy as np
import pytest

from beamfastsim import ptgt_point as PP
from beamfastsim import stage

ROD = (160.0, 3.1495, 3.1495, 3.1495)


def _key(**kw):
    a = dict(rod=ROD, pot=1000, seed=1, dtype="float64", bank_hash="b" * 64, slab_hash="s" * 64, map_sha="m" * 64)
    a.update(kw)
    return PP.design_key(**a)


def test_design_key_is_stable_and_sees_every_input():
    k = _key()
    assert k == _key() and k == _key(rod=(160.0000001, 3.1495, 3.1495, 3.1495))      # rounded to 1e-6 mm
    for change in (dict(rod=(161.0, 3.1495, 3.1495, 3.1495)), dict(pot=2000), dict(seed=2), dict(dtype="float32"),
                   dict(bank_hash="c" * 64), dict(slab_hash="t" * 64), dict(map_sha="n" * 64),
                   dict(slabs=PP.SLABS[:2])):
        assert _key(**change) != k


def _args(tmp_path, platform="cpu", pool=None):
    return argparse.Namespace(rod=list(ROD), pot=1000, seed=3, dtype="float32", platform=platform, pool=pool,
                              bank="bank.npz", slab_bank="slab.npz", ps_map="ga05/PSMap.header", cache=str(tmp_path / "c"),
                              gdml="ts.gdml", field_config="cfg.txt", offline_root="off", maps="maps", ds_geom="ds.txt",
                              ds_offline_root=None, ds_map="DSMap.header")


def test_chain_argv_carries_the_validated_ps_configuration(tmp_path):
    argv = PP.chain_argv(_args(tmp_path), tmp_path / "r.npz")
    assert argv.count("--slab") == len(PP.SLABS) and "--slab-bank" in argv
    assert argv[argv.index("--pool") + 1] == str(PP.POOL["cpu"]) and argv[argv.index("--dtype") + 1] == "float32"
    assert argv[argv.index("--rod") + 1:argv.index("--rod") + 5] == [str(v) for v in ROD]
    assert argv[argv.index("--z-plane") + 1] == str(PP.ENVELOPE["z_plane"])
    gpu = PP.chain_argv(_args(tmp_path, "gpu"), tmp_path / "r.npz")
    assert gpu[gpu.index("--pool") + 1] == str(PP.POOL["gpu"])
    ts = PP.tsdevice_argv(_args(tmp_path), "ref.npz", "o.npz")
    assert ts[ts.index("--ds-offline-root") + 1] == "off" and ts[ts.index("--seed") + 1] == str(PP.TS_SEED)


def _write_reference(path, pdg, pot):
    np.savez(path, pdg=np.asarray(pdg), parent_shard_event_counts=np.array([pot]))


def test_states_runs_the_chain_once_per_design(tmp_path):
    calls = []

    def chain(argv):
        calls.append(argv)
        _write_reference(argv[argv.index("--out") + 1], [13, -211], 1000)

    a = _args(tmp_path)
    ref, cached, _ = PP.states(a, "k1", chain)
    assert not cached and ref.exists() and json.loads((ref.parent / "key.json").read_text())["key"] == "k1"
    ref2, cached2, dt2 = PP.states(a, "k1", chain)
    assert cached2 and ref2 == ref and dt2 == 0.0 and len(calls) == 1
    PP.states(a, "k2", chain)
    assert len(calls) == 2 and not list((tmp_path / "c").glob(".*"))      # no temporary directory left


def test_metrics_stops_and_yields(tmp_path):
    _write_reference(tmp_path / "r.npz", [13, -211, 211, -13, 13], 1000)
    np.savez(tmp_path / "t.npz", ds_fate=np.array([1, 0, 1, 2, 1]), pdg_ds=np.array([13, -211, 211, -13, 13]))
    m = PP.metrics(tmp_path / "r.npz", tmp_path / "t.npz", cached=False, chain_s=2.0, ts_s=1.0)
    assert m["stops"] == 3 and m["stops_per_pot"] == pytest.approx(3e-3)
    assert m["stops_per_pot_err"] == pytest.approx(3 ** 0.5 / 1000) and m["mu_minus_stops_per_pot"] == pytest.approx(2e-3)
    assert m["ts_mu_minus_pi_minus_per_pot"] == pytest.approx(3e-3) and m["ts_mu_pi_per_pot"] == pytest.approx(5e-3)
    assert m["ts_entrances"] == 5 and m["cached"] == 0.0 and all(isinstance(v, float) for v in m.values())


def test_stage_map_sets():
    assert stage.map_path("DSMap") == (f"{stage.MAPS}/DSMap", "DSMap")
    assert stage.map_path("GA05/PSMap") == ("/DataFiles/BFieldMaps/GA05/PSMap", "GA05/PSMap")
    for bad in ("../x", "GA05/../PSMap", "a/b/c", "", "PS Map"):
        with pytest.raises(ValueError):
            stage.map_path(bad)
