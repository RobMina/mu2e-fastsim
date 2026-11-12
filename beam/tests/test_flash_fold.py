"""flash_fold: target photons at the foil exit folded through hard and soft response libraries (synthetic)."""
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim import flash_fold as FF
from beamfastsim import flash_response_library as F
from beamfastsim.cechain import LIVE, STOPPED
from beamfastsim.flash_attenuation import uncollided_weight

X0 = F.DS_X
BEAM = SimpleNamespace(foil_z=np.array([5800.0, 5900.0, 6000.0]), foil_ht=np.array([0.05, 0.1, 0.2]),
                       foil_rin=np.array([20.0, 20.0, 20.0]), foil_rout=np.array([75.0, 75.0, 75.0]), axis_x=X0)
C_HARD, C_SOFT = 3.0e-5, 7.0e-5


def _lib(tmp_path, name, energy_edges, c):
    rng = np.random.default_rng(1)
    n = 4000
    rows = dict(energy=np.exp(rng.uniform(np.log(energy_edges[0]), np.log(energy_edges[-1] * 0.999), n)),
                z=rng.uniform(5000, 7000, n), r=rng.uniform(0, 299, n), cos_theta=rng.uniform(-1, 1, n),
                u_r=rng.uniform(-1, 1, n), edep=np.full(n, c))
    cfg = dict(name=name, version=1,
               edges=dict(energy=energy_edges, z=[4999.0, 7001.0], r=[0.0, 300.0], cos_theta=[-1.0, 0.0, 1.0],
                          u_r=[-1.0, 0.0, 1.0]),
               classes={"all": dict(cos_theta=[-1.0, 1.0])},
               files=[dict(path="synthetic", **{"class": "all"}, job=0, seed=0, events=n)],
               source_description="synthetic",
               generator=dict(name="numpy", release="test", physics_list=None, configuration="synthetic"),
               release=dict(repository="test", commit="0" * 7, entry_point="tests"),
               geometry=dict(tag="synthetic", hash="0" * 64), config_hashes=dict(cfg="x"), coverage="synthetic")
    F.build(cfg, tmp_path / f"{name}.npz", reader=lambda p: rows)
    return F.load(tmp_path / f"{name}.npz")


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    t = tmp_path_factory.mktemp("libs")
    return (_lib(t, "flash.response.synthetic_hard", [0.05, 1.0, 100.0], C_HARD),
            _lib(t, "flash.response.synthetic_soft", [0.005, 0.02, 0.05], C_SOFT))


def test_sphere_directions_are_unit_and_isotropic():
    d = FF.sphere_directions(4096)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(d.mean(0), 0, atol=2e-3)
    np.testing.assert_allclose((d * d).mean(0), 1 / 3, atol=2e-3)


def _births(x_stop, fates, pdgs, nan, ean, events=3):
    n = len(pdgs)
    ledger = np.stack([np.full(events, 1), np.full(events, 2), np.arange(events)], 1).astype(np.uint64)
    return dict(source_pdg=np.asarray(pdgs), state_fate=np.asarray(fates), state_x=np.asarray(x_stop, float),
                state_nan=np.asarray(nan), state_ean=np.asarray(ean, float), source_event_ledger=ledger,
                source_run=np.ones(n, np.int64), source_subrun=np.full(n, 2), source_event=np.arange(n) % events)


def test_annihilation_rays_weights_and_census():
    centre = [X0 + 50.0, 0.0, 5900.0]                       # inside foil 1 (r = 50 mm)
    outside = [X0 + 10.0, 0.0, 5900.0]                      # in the hole: not in a foil
    b = _births([centre, outside, centre], [STOPPED, STOPPED, LIVE], [-11, -11, 11], [2, 4, 0],
                [2 * FF.ME, 3 * FF.ME, 0.0])
    ph, census = FF.annihilation_photons(b, BEAM, n_dirs=64, seed=3)
    assert census["stopped"] == 2 and census["stopped_in_foil"] == 1 and census["stopped_outside_foils"] == 1
    assert census["inflight_photons"] == 2 and census["counter_mismatch"] == 0
    assert np.isclose(census["inflight_energy_mev"], FF.ME)
    assert len(ph["energy"]) == 128 and np.all(ph["energy"] == FF.ME) and np.all(ph["parent"] == 0)
    np.testing.assert_allclose(ph["direction"][:64], -ph["direction"][64:], atol=1e-12)
    # weights: 1/n_dirs times the uncollided transmission over each ray's own path to the foil exit
    assert 0 < ph["weight"].sum() <= 2.0 and ph["weight"].sum() > 2.0 * 0.95
    again, _ = FF.annihilation_photons(b, BEAM, n_dirs=64, seed=3)
    np.testing.assert_array_equal(ph["direction"], again["direction"])


def test_other_foil_path():
    p = np.array([[X0 + 50.0, 0.0, 5800.05], [X0 + 10.0, 0.0, 5800.05], [X0 + 50.0, 0.0, 5800.05]])
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    L = FF.other_foil_path_mm(p, d, np.array([0, 0, 0]), BEAM)
    np.testing.assert_allclose(L, [0.2 + 0.4, 0.0, 0.0])    # foils 1 and 2; the hole; backward: nothing
    c = np.cos(0.3)
    L2 = FF.other_foil_path_mm(p[:1], np.array([[np.sin(0.3), 0.0, c]]), np.array([0]), BEAM)
    np.testing.assert_allclose(L2, [(0.2 + 0.4) / c])


def _exits(energies, events=3):
    n = len(energies)
    pos = np.tile([X0 + 50.0, 0.0, 5800.05], (n, 1))
    d = np.tile([0.0, 0.0, 1.0], (n, 1))
    return dict(parent=np.arange(n) % 3, energy=np.asarray(energies, float), exit_position=pos, direction=d,
                weight=np.full(n, 0.9), foil_index=np.zeros(n, np.int64), foil_exit=np.ones(n, bool),
                attenuation_supported=np.ones(n, bool))


@pytest.mark.parametrize("seed", [0, 1])
def test_fold_constant_libraries(libs, seed):
    hard, soft = libs
    b = _births([[X0 + 50.0, 0.0, 5900.0]] * 3, [STOPPED, LIVE, LIVE], [-11, 11, 11], [2, 0, 0],
                [2 * FF.ME, 0, 0])
    e = _exits([10.0, 0.02, 0.003, 200.0])                 # hard, soft, below the soft table, above the hard table
    res = FF.fold(b, e, BEAM, hard, soft, n_dirs=32, seed=seed)
    t_soft = uncollided_weight(0.02, 0.6)["weight"]        # crosses foils 1 and 2 at normal incidence
    expect_brems = 0.9 * (C_HARD + C_SOFT * t_soft) / 3
    for dt in ("float32", "float64"):
        br = res["brems"][dt]
        assert br["refused"] == 1 and br["below_soft"] == 1
        np.testing.assert_allclose(br["mean"], expect_brems, rtol=1e-5 if dt == "float32" else 1e-12)
        np.testing.assert_allclose(br["soft"], 0.9 * C_SOFT * t_soft / 3, rtol=1e-5 if dt == "float32" else 1e-12)
    ann = res["annihilation"]["float64"]
    w = FF.annihilation_photons(b, BEAM, n_dirs=32, seed=seed)[0]["weight"].sum()
    np.testing.assert_allclose(ann["mean"], w * C_HARD / 3, rtol=1e-12)
    assert res["brems"]["host_equals_device_float64"] and res["annihilation"]["host_equals_device_float64"]
    np.testing.assert_allclose(res["total"]["float32"]["mean"], res["total"]["float64"]["mean"], rtol=1e-5)
    assert res["census"]["annihilation"]["stopped_in_foil"] == 1 and res["events"] == 3
    assert res["omitted"]["inflight_annihilation_photons_per_event"] == 0


def test_event_index_rejects_unknown_events():
    b = _births([[X0 + 50.0, 0.0, 5900.0]], [LIVE], [11], [0], [0.0])
    b["source_event"] = np.array([7])
    with pytest.raises(ValueError):
        FF.event_index(b, np.array([0]))
