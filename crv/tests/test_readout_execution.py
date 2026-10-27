"""Real chain.charges regressions with tiny, entirely synthetic optical inputs.

No photon/SiPM/response implementation is patched. LookupTable/Table objects
follow test_photons' synthetic-table pattern; they and a uniform 40x40 photon
map are passed directly, so even lookup.load need not be patched. Registered
sector names, yields, reflector signs and windows are retained, but table names,
geometry and side flags are synthetic (not a detector-conditions census). Run2
R4 deliberately has reflector 0 and no side-1 sensors. Run1A T2's missing side
is an artificial test case, not a statement about the real Run1A geometry.

One short step per sector, four longitudinal bins, and shared tables for the
three reflector signs keep real CPU JAX dispatch bounded. Production bucket
sizes and kernels are unchanged. Python call profiling observes execution only:
it detects an otherwise silent nonfused-JAX fallback to the NumPy SiPM kernel.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, replace
import sys

import numpy as np
import pandas as pd
import pytest

from crvfastsim import chain, geometry, readout, runconfig
from crvfastsim.response import lookup


SEED = 20261027
RUNS = ("run1a_v01", "run2_v10")
OUTPUT_FIELDS = {"chan", "time", "charge", "pe", "photon"}


def _optical_table(reflector):
    """Small v6 lookup with real attenuation/delay slopes, zero fibre decay."""
    edges = {"x": np.array([0., 10.]), "y": np.array([-26., 26.]),
             "z": np.linspace(-1000., 1000., 5), "beta": np.array([0., 1.01]),
             "theta": np.array([0., np.pi]), "phi": np.array([0., np.pi]),
             "r": np.array([0., .7])}
    tables = {}
    for kind in lookup.TABLES:
        delays = np.eye(5, dtype=np.uint8)[1:]
        emissions = np.ones((4, 1), np.uint8)
        tables[kind] = lookup.Table(
            np.array([.8, .7, .6, .5], np.float32), delays,
            delays.sum(1, dtype=np.int32), emissions, emissions.sum(1, dtype=np.int32))
    ck = np.array([[.5, 0.], [1., 0.]])
    return lookup.LookupTable(
        f"LookupTable_readout_execution_{reflector}", (6, 0), reflector,
        9.64, 25.42, 1000., 26., 2., 1., .7, 1.75, .126, 0.,
        ck, ck.copy(), edges, tables)


@pytest.fixture(scope="module", params=("numpy", "jax", "fused"))
def backend(request):
    name = request.param
    if name == "numpy":
        print(f"\nreadout execution backend={name}, device=host CPU, numpy={np.__version__}")
        yield name, None
        return
    jax = pytest.importorskip("jax")
    cpu = jax.devices("cpu")[0]
    assert cpu.platform == "cpu"
    # Explicit device context, independent of the machine's default accelerator.
    with jax.default_device(cpu):
        assert jax.numpy.asarray(1).devices() == {cpu}
        print(f"\nreadout execution backend={name}, device={cpu}, platform={cpu.platform}, jax={jax.__version__}")
        yield name, cpu


@pytest.fixture(params=RUNS)
def case(request, backend):
    name, cpu = backend
    rc = runconfig.get(request.param)
    names = list(rc.sectors)
    # Reverse bar order to catch sector-index vs bar-index mistakes.
    sector = np.arange(len(names), dtype=np.int64)[::-1].copy()
    n = len(sector)
    geom = geometry.CRVGeometry(
        names, sector, np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
        np.zeros((n, 3)), np.tile([9.64, 25.42, 1000.], (n, 1)), np.tile([0, 1, 2], (n, 1)))
    cfg = rc.chain_config(jax=name != "numpy", fused_response=name == "fused")
    cfg = replace(cfg, response=replace(cfg.response, sectors={
        s: replace(v, table=f"readout_execution_{abs(v.reflector)}") for s, v in rc.sectors.items()}))
    tabs = {f"readout_execution_{r}": _optical_table(r)
            for r in {abs(v.reflector) for v in rc.sectors.values()}}
    if cpu is not None:
        from crvfastsim.response.photons_jax import DeviceTable
        tabs = {k: DeviceTable(v) for k, v in tabs.items()}
        assert all(t.prob.devices() == {cpu} for t in tabs.values())
    steps = pd.DataFrame({
        "subrun": np.ones(n, np.int64), "event": np.full(n, 7, np.int64),
        "bar": np.arange(n), "edep": np.full(n, .003), "len": np.full(n, .05),
        "t0": np.full(n, 500.), "t1": np.full(n, 500.01),
        "x0": np.full(n, 2.), "x1": np.full(n, 2.05),
        "y0": np.zeros(n), "y1": np.zeros(n), "z0": np.zeros(n), "z1": np.zeros(n),
        "px0": np.full(n, 1000.), "py0": np.zeros(n), "pz0": np.zeros(n), "p1": np.full(n, 1000.)})
    counters = {"crs.sectorNames": names}
    counters.update({f"crs.sipmsAtSide{side}{s}": True for s in names for side in (0, 1)})
    counters[f"crs.sipmsAtSide1{'R4' if rc.name == 'run2_v10' else 'T2'}"] = False
    mask = readout.sensor_mask(geom, counters)
    routed = np.zeros(4 * n, bool)
    for bar, sec in enumerate(sector):
        refl = rc.sectors[names[sec]].reflector
        routed[4 * bar:4 * bar + 4] = [refl == 0 or sipm % 2 == (refl < 0) for sipm in range(4)]
    return dict(run=rc.name, backend=name, cpu=cpu, geom=geom, cfg=cfg, tabs=tabs, steps=steps,
                pdg=np.full(n, 13, np.int64), pmap=np.ones((40, 40)), mask=mask, routed=routed)


def _conditions(case, mask=None, with_status=False):
    cond = runconfig.nominal_conditions(case["geom"])
    if with_status:
        channels = np.flatnonzero(case["routed"])
        # Bit 1 and higher unrelated bits must remain live; bits 0/2 suppress.
        cond.status[channels] = np.resize(np.array([0, 1, 2, 4, 8, 3, 6, 16]), len(channels))
    cond = replace(cond, readout_mask=None if mask is None else mask.copy())
    for f in fields(cond):
        value = getattr(cond, f.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return cond


def _execute(case, cond):
    """Run untouched chain/kernels, check all stored condition arrays, trace calls."""
    stored = {f.name: (getattr(cond, f.name), deepcopy(getattr(cond, f.name))) for f in fields(cond)}
    rng = np.random.default_rng(SEED)
    key = None
    context = nullcontext()
    if case["cpu"] is not None:
        import jax
        context = jax.default_device(case["cpu"])
        with context:
            key = jax.random.PRNGKey(SEED)
            assert key.devices() == {case["cpu"]}
        context = jax.default_device(case["cpu"])
    calls = set()
    observed_names = {"make_photons", "_substep_photons", "substep_photons", "simulate",
                      "response_batched", "response", "device_photons", "run_device_sipm"}

    def observe(frame, event, arg):
        if event == "call" and frame.f_code.co_name in observed_names:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("crvfastsim.response."):
                calls.add((module.rsplit(".", 1)[1], frame.f_code.co_name))

    previous = sys.getprofile()
    timers = {}
    try:
        sys.setprofile(observe)
        with context:
            out = chain.charges(case["steps"], None, case["geom"], cond, case["cfg"], rng,
                                key=key, tabs=case["tabs"], pmap=case["pmap"], pdg=case["pdg"], timers=timers)
    finally:
        sys.setprofile(previous)
    for field, (original, saved) in stored.items():
        assert getattr(cond, field) is original
        if original is not None:
            np.testing.assert_array_equal(original, saved, err_msg=field)
            assert not original.flags.writeable
    assert set(out) == OUTPUT_FIELDS
    assert len({len(v) for v in out.values()}) == 1
    assert timers["charges"] == len(out["chan"])
    assert not timers.get("response_overflow_batches", 0)
    assert all(np.isfinite(v).all() for v in out.values())
    assert (out["charge"] > 0).all() and (out["pe"] > 0).all()
    if len(out["chan"]):
        np.testing.assert_array_equal(out["chan"] // chain.CHAN_MULT, chain.event_key(case["steps"])[0])
        assert ((out["time"] >= 500) & (out["time"] <= case["cfg"].window[1])).all()
        source = out["photon"].astype(np.int64)
        assert ((source >= 0) & (source < len(case["steps"]))).all()
        np.testing.assert_array_equal((out["chan"] % chain.CHAN_MULT) // 4,
                                      case["steps"].bar.to_numpy()[source])
    if case["backend"] == "numpy":
        assert {("photons", "_substep_photons"), ("sipm", "simulate")} <= calls
    else:
        # Refuse the nonfused wrapper's silent NumPy fallback on overflow.
        assert ("sipm", "simulate") not in calls
        if case["backend"] == "jax":
            assert {("photons_jax", "substep_photons"), ("sipm_jax", "simulate")} <= calls
        else:
            assert {("device", "response_batched"), ("device", "response"),
                    ("photons_jax", "device_photons")} <= calls
        if len(out["chan"]) or case["backend"] == "fused":
            assert ("sipm_jax", "run_device_sipm") in calls
    print(f"{case['run']} {case['backend']}: charges={len(out['chan'])}, "
          f"channels={len(np.unique(out['chan']))}, real_calls={sorted(calls)}")
    return out, deepcopy(rng.bit_generator.state), rng.random(8)


def _assert_channels(out, expected):
    np.testing.assert_array_equal(np.unique(out["chan"] % chain.CHAN_MULT), np.flatnonzero(expected))


def _assert_same_stream(left, right):
    a, state_a, next_a = left
    b, state_b, next_b = right
    for field in OUTPUT_FIELDS:
        np.testing.assert_array_equal(a[field], b[field], err_msg=field)
    assert state_a == state_b
    np.testing.assert_array_equal(next_a, next_b)


def test_named_side_presence_survivors_and_all_absent(case):
    baseline = _execute(case, _conditions(case))[0]
    masked = _execute(case, _conditions(case, case["mask"]))[0]
    empty = _execute(case, _conditions(case, np.zeros_like(case["mask"])))[0]
    _assert_channels(baseline, case["routed"])
    _assert_channels(masked, case["routed"] & case["mask"])
    assert all(v.size == 0 for v in empty.values())
    if case["run"] == "run2_v10":
        names = case["geom"].sector_names
        bar = int(np.flatnonzero(case["geom"].sector == names.index("R4"))[0])
        assert case["cfg"].response.sectors["R4"].reflector == 0
        assert case["mask"][4 * bar:4 * bar + 4].tolist() == [True, False, True, False]
        assert {4 * bar + 1, 4 * bar + 3} <= set(baseline["chan"] % chain.CHAN_MULT)
        assert {4 * bar, 4 * bar + 2} <= set(masked["chan"] % chain.CHAN_MULT)


def test_status_bits_compose_without_rewriting_conditions(case):
    cond = _conditions(case, case["mask"], with_status=True)
    effective = readout.effective_status(cond)
    np.testing.assert_array_equal(effective, cond.status | (~cond.readout_mask).astype(np.int64))
    expected = case["routed"] & cond.readout_mask & ((cond.status & 5) == 0)
    out = _execute(case, cond)
    _assert_channels(out[0], expected)
    # Equivalent transient status must execute identically, including RNG state.
    effective.flags.writeable = False
    direct = replace(cond, status=effective, readout_mask=None)
    _assert_same_stream(out, _execute(case, direct))
    live_status = cond.status[np.flatnonzero(expected)]
    assert 2 in live_status and 8 in live_status


def test_none_and_all_true_preserve_seeded_stream(case):
    cond = _conditions(case, with_status=True)
    assert cond.readout_mask is None
    all_true = np.ones_like(case["mask"])
    all_true.flags.writeable = False
    _assert_same_stream(_execute(case, cond), _execute(case, replace(cond, readout_mask=all_true)))
