"""tsdevice bench hooks: fixed inputs tile the references; seeded (fresh) inputs draw new rows and carry the seed."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from beamfastsim import tsdevice as dev  # noqa: E402


@pytest.fixture
def fake_resident(monkeypatch):
    loads = []
    ref = {"x": np.arange(30.0).reshape(10, 3), "p": np.arange(30.0).reshape(10, 3) + 100, "t": np.arange(10.0),
           "pdg": np.arange(10)}
    monkeypatch.setattr(dev, "load_references", lambda paths, limit=None: loads.append(("ref", paths)) or ref)
    monkeypatch.setattr(dev, "load_chain", lambda *a, **kw: loads.append(("chain", kw)) or "chain")
    for name in ("TSDEV_REFERENCE", "TSDEV_GDML", "TSDEV_FIELD_CONFIG", "TSDEV_OFFLINE_ROOT", "TSDEV_MAPS",
                 "TSDEV_DS_GEOM", "TSDEV_DS_OFFLINE_ROOT", "TSDEV_DS_MAP"):
        monkeypatch.setenv(name, name.lower())
    monkeypatch.setattr(dev, "_BENCH_RESIDENT", {})
    return loads


def test_fixed_inputs_tile_the_references(fake_resident):
    chain, (x, p, t, pdg) = dev.bench_prepare(25)
    assert chain == "chain" and np.array_equal(np.asarray(pdg), np.arange(25) % 10)


def test_fresh_inputs_draw_per_seed_and_load_once(fake_resident):
    a, b, a2 = (dev.bench_prepare(25, seed=s) for s in (1, 2, 1))
    assert [k for k, _ in fake_resident] == ["ref", "chain"]          # references and chain loaded once
    assert a[2] == 1 and b[2] == 2
    pa, pb, pa2 = (np.asarray(q[1][3]) for q in (a, b, a2))
    assert pa.shape == (25,) and not np.array_equal(pa, pb) and np.array_equal(pa, pa2)
    assert np.array_equal(np.asarray(a[1][2]), pa.astype(float))       # the columns stay row-aligned


def test_bench_chain_uses_the_pass_seed(fake_resident, monkeypatch):
    seeds = []

    class State:
        fate = x = t = foil = 0

    monkeypatch.setattr(dev, "run_chain", lambda chain, x, p, t, pdg, seed: seeds.append(seed) or (State, State))
    dev.bench_chain(dev.bench_prepare(5))
    dev.bench_chain(dev.bench_prepare(5), seed=3)
    dev.bench_chain(dev.bench_prepare(5, seed=9), seed=3)
    assert seeds == [20261107, 3, 9]


def test_hooks_take_a_seed_keyword():
    bench = pytest.importorskip("mu2efastsim.bench")
    if hasattr(bench, "accepts_seed"):
        assert bench.accepts_seed(dev.bench_prepare)
