"""Synthetic physical readout contracts; no assets or accelerator execution."""
from dataclasses import fields
from types import ModuleType, SimpleNamespace
import copy
import sys

import numpy as np
import pandas as pd
import pytest

from crvfastsim import chain, geometry, readout, runconfig
from crvfastsim import response


def geom(names=("A", "B"), sectors=(1, 0, 1)):
    sectors = np.asarray(sectors)
    n = len(sectors)
    return geometry.CRVGeometry(list(names), sectors, np.zeros(n, int), np.zeros(n, int),
                                np.zeros(n, int), np.zeros((n, 3)), np.ones((n, 3)),
                                np.tile([0, 1, 2], (n, 1)))


def counters(names=("A", "B")):
    return {f"crs.sipmsAtSide{side}{name}": (i + side) % 2 == 0
            for i, name in enumerate(names) for side in (0, 1)}


def cond(n=8):
    z = np.zeros(n)
    return chain.Conditions(z + 100, z + 400, z + 11, z, np.zeros(n, np.int64),
                            np.ones(n, np.int64), np.arange(n), z)


def test_side_parity_uses_each_bar_sector_not_bar_order():
    g, cfg = geom(), counters()
    before = copy.deepcopy(cfg)
    mask = readout.sensor_mask(g, cfg)
    assert mask.dtype == np.dtype(bool) and mask.shape == (4 * g.nbars,)
    assert mask.reshape(-1, 4).tolist() == [[False, True, False, True], [True, False, True, False],
                                         [False, True, False, True]]
    assert cfg == before
    np.testing.assert_array_equal(g.sector, [1, 0, 1])


@pytest.mark.parametrize("name", ["run1a_v01", "run1b_v01", "run2_v10"])
def test_all_named_sectors_without_assets(name):
    names = list(runconfig.get(name).sectors)
    sector = np.repeat(np.arange(len(names))[::-1], 2)
    g, cfg = geom(names, sector), counters(names)
    cfg["crs.sectorNames"] = names
    mask = readout.sensor_mask(g, cfg)
    for b, isec in enumerate(sector):
        for s in range(4):
            assert mask[4 * b + s] == cfg[f"crs.sipmsAtSide{s % 2}{names[isec]}"]
    for name in names:
        for side in (0, 1):
            bad = dict(cfg)
            del bad[f"crs.sipmsAtSide{side}{name}"]
            with pytest.raises(ValueError, match="missing"):
                readout.sensor_mask(g, bad)


def test_unused_sector_flags_still_required_and_empty_geometry_supported():
    g, cfg = geom(sectors=(0,)), counters()
    del cfg["crs.sipmsAtSide1B"]
    with pytest.raises(ValueError, match="Side1B"):
        readout.sensor_mask(g, cfg)
    g.sector = np.zeros(0, int)
    assert readout.sensor_mask(g, counters()).shape == (0,)


@pytest.mark.parametrize("value", [0, 1, 2, -1, "true", "false", None, 0.0, np.nan, [], np.array(True)])
def test_counters_reject_non_boolean_values(value):
    cfg = counters()
    cfg["crs.sipmsAtSide0A"] = value
    with pytest.raises(ValueError, match="must be boolean"):
        readout.sensor_mask(geom(), cfg)


def test_numpy_booleans_and_both_or_neither_sides():
    cfg = counters()
    cfg.update({"crs.sipmsAtSide0A": np.bool_(True), "crs.sipmsAtSide1A": np.bool_(True),
                "crs.sipmsAtSide0B": False, "crs.sipmsAtSide1B": False})
    assert readout.sensor_mask(geom(), cfg).tolist() == [False] * 4 + [True] * 4 + [False] * 4


@pytest.mark.parametrize("sector", [np.array([-1]), np.array([2]), np.array([0.0]),
                                    np.array([True]), np.array([[0]]), np.array(0), None])
def test_invalid_sector_shape_type_or_unknown_index(sector):
    g = geom()
    g.sector = sector
    with pytest.raises(ValueError, match="geometry"):
        readout.sensor_mask(g, counters())


@pytest.mark.parametrize("names", [None, [], ["A", "A"], ["A", ""], [1, "B"], "AB", np.array("A"), np.array([["A"]])])
def test_invalid_geometry_names(names):
    g = geom()
    g.sector_names = names
    with pytest.raises(ValueError, match="geometry"):
        readout.sensor_mask(g, counters())


def test_missing_geometry_bad_bar_count_and_mismatched_counters_names():
    for g in (None, SimpleNamespace(sector_names=["A"]),
              SimpleNamespace(sector_names=["A"], sector=np.array([0]), nbars=2)):
        with pytest.raises(ValueError, match="geometry"):
            readout.sensor_mask(g, counters())
    with pytest.raises(ValueError, match="mapping"):
        readout.sensor_mask(geom(), None)
    with pytest.raises(ValueError, match="disagree"):
        readout.sensor_mask(geom(), {**counters(), "crs.sectorNames": ["B", "A"]})


@pytest.mark.parametrize("name,reflector,side", [("R3", -1, 1), ("R4", 0, 0), ("T1", 1, 0)])
def test_signed_reflector_and_no_reflector_have_distinct_presence_rules(name, reflector, side):
    sector_cfg = runconfig.get("run2_v10").sectors[name]
    assert sector_cfg.reflector == reflector
    g = geom((name,), (0,))
    cfg = {f"crs.sipmsAtSide0{name}": side == 0, f"crs.sipmsAtSide1{name}": side == 1}
    mask = readout.sensor_mask(g, cfg)
    assert mask.tolist() == [s % 2 == side for s in range(4)]
    cfg[f"crs.sipmsAtSide0{name}"] = cfg[f"crs.sipmsAtSide1{name}"] = True
    assert readout.sensor_mask(g, cfg).all()
    assert sector_cfg.reflector == reflector


def test_optional_last_field_default_status_identity_and_no_mutation():
    c = cond()
    assert fields(c)[-1].name == "readout_mask" and c.readout_mask is None
    assert readout.effective_status(c) is c.status
    c.status[:] = np.arange(8)
    before = c.status.copy()
    c.status.flags.writeable = False
    c.readout_mask = np.array([True, False] * 4)
    c.readout_mask.flags.writeable = False
    effective = readout.effective_status(c)
    assert effective is not c.status and not np.shares_memory(effective, c.status)
    np.testing.assert_array_equal(effective, [0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(c.status, before)
    c.readout_mask = np.zeros(8, bool)
    np.testing.assert_array_equal(readout.effective_status(c), before | 1)


@pytest.mark.parametrize("mask", [[True] * 8, np.ones(8, int), np.ones(8, float),
                                  np.array(["true"] * 8), np.ones(8, object), np.ones((2, 4), bool),
                                  np.ones(7, bool), np.ones(9, bool), np.array(True)])
def test_invalid_mask_shape_dtype_or_size(mask):
    c = cond()
    c.readout_mask = mask
    with pytest.raises(ValueError, match="readout_mask"):
        readout.effective_status(c)


def install_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    if name.startswith("crvfastsim.response."):
        monkeypatch.setattr(response, name.rsplit(".", 1)[1], module, raising=False)
    return module


@pytest.mark.parametrize("backend", ["numpy", "jax", "fused"])
@pytest.mark.parametrize("masked", [False, True])
def test_charge_boundary_preserves_status_rng_and_photon_inputs(monkeypatch, backend, masked):
    c, cfg = cond(), chain.ChainConfig(jax=backend != "numpy", fused_response=backend == "fused")
    c.status[:] = [0, 0, 1, 0, 2, 4, 0, 0]
    if masked:
        c.readout_mask = np.array([True, False] * 4)
    before = c.status.copy()
    photons = np.arange(8)
    steps = pd.DataFrame({"subrun": [0], "event": [1]})
    pdg, key = np.array([13]), object()
    rng = np.random.default_rng(123)
    rng_before = copy.deepcopy(rng.bit_generator.state)
    expected = np.array([0, 4, 6] if masked else [0, 1, 3, 4, 6, 7])

    def make_photons(*args):
        assert args[5] is c.deviation and args[6] is rng
        return np.zeros(8, int), photons.copy(), photons.astype(float)

    def simulate(labels, times, sources, *args, **kwargs):
        np.testing.assert_array_equal(labels, chain.CHAN_MULT + expected)
        np.testing.assert_array_equal(times, expected)
        np.testing.assert_array_equal(sources, np.zeros(len(expected), int))
        assert kwargs["t_end"] == cfg.window[1]
        return {"chan": labels, "time": times}

    if backend != "numpy":
        install_module(monkeypatch, "jax", random=SimpleNamespace(split=lambda k: (k, k)))
    if backend == "fused":
        def response_batched(*args):
            status = args[7]
            assert args[6] is c.deviation and args[9] is rng and args[10] is key
            if masked:
                assert status is not c.status
                np.testing.assert_array_equal(status, before | (~c.readout_mask).astype(int))
            else:
                assert status is c.status
            np.testing.assert_array_equal(photons[(status & 5) == 0], expected)
            return []
        install_module(monkeypatch, "crvfastsim.response.device", response_batched=response_batched)
    elif backend == "jax":
        install_module(monkeypatch, "crvfastsim.response.photons_jax", make_photons=make_photons)
        install_module(monkeypatch, "crvfastsim.response.sipm_jax", simulate=simulate)
    else:
        monkeypatch.setattr(chain.PH, "make_photons", make_photons)
        monkeypatch.setattr(chain.SI, "simulate", simulate)
    chain.charges(steps, None, geom(sectors=(0, 1)), c, cfg, rng, key=key, pdg=pdg)
    np.testing.assert_array_equal(c.status, before)
    assert rng.bit_generator.state == rng_before


@pytest.mark.parametrize("status", [np.zeros((2, 4), int), np.zeros(8, float), np.zeros(8, bool)])
def test_masked_status_requires_flat_integer_channels(status):
    c = cond()
    c.status, c.readout_mask = status, np.ones(8, bool)
    with pytest.raises(ValueError, match="readout status"):
        readout.effective_status(c)


def test_csv_loading_remains_unmasked_and_does_not_rewrite_input(tmp_path):
    path = tmp_path / "calib.csv"
    rows = np.column_stack((np.arange(8), np.full(8, 100), np.full(8, 400), np.full(8, 11),
                            np.zeros(8), np.zeros(8), np.ones(8), np.arange(8)))
    np.savetxt(path, rows, delimiter=",")
    before = path.read_bytes()
    c = chain.Conditions.from_csv(path)
    assert c.readout_mask is None and readout.effective_status(c) is c.status
    c.readout_mask = np.zeros(8, bool)
    np.testing.assert_array_equal(readout.effective_status(c), np.ones(8, int))
    np.testing.assert_array_equal(c.status, np.zeros(8, int))
    assert path.read_bytes() == before
