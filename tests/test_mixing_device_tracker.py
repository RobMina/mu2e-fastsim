"""Physical tracker ports: real kernels compared to the staged JAX reference."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# The shared test hooks select float64 before precision-aware kernels are imported.
pytest.importorskip("mu2efastsim.testing")
trkfastsim = pytest.importorskip("trkfastsim")

from mixing.device_tracker import Capacities, prepare
from mixing.device_backend import _audit_program


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def helpers():
    path = Path(trkfastsim.__file__).parents[1] / "tests" / "test_digi.py"
    if not path.is_file() or not (path.parent / "data" / "digi_job_excerpt.fcl").is_file():
        pytest.skip("tracker checkout with tests/test_digi.py and its public conditions fixture required")
    spec = importlib.util.spec_from_file_location("tracker_fixture", path)
    module = importlib.util.module_from_spec(spec)
    previous_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


@pytest.fixture(scope="module")
def cond(helpers):
    from trkfastsim.digi import digicond as dc
    d = dc.parse_fhicl_dump(helpers.FIX.read_text())
    preamp = np.zeros((dc.NUSTRAWS, 6))
    preamp[:, 0] = np.arange(dc.NUSTRAWS)
    preamp[:, 3], preamp[:, 4], preamp[:, 5] = 11.996, 11.632, 1.5e6
    return dc.assemble(d, preamp, helpers._geom())


def collection(st, capacity=None):
    capacity = capacity or len(st["ev"])
    return {"columns": {name: np.pad(value, (0, capacity - len(value))) for name, value in st.items() if name != "ev"},
            "event": np.pad(st["ev"], (0, capacity - len(st["ev"]))),
            "mask": np.arange(capacity) < len(st["ev"])}


def timing(lengths, pbtime=None):
    return {"event_length": np.asarray(lengths, np.float64),
            "mc_pbtime": np.zeros(len(lengths)) if pbtime is None else np.asarray(pbtime, np.float64),
            "event_mask": np.ones(len(lengths), bool)}


def evaluate(model, rows, times, seed=3):
    args = jax.device_put((rows, times, model.assets, jax.random.key(seed)))
    _audit_program(jax.make_jaxpr(model.apply)(*args))
    compiled = jax.jit(model.apply)
    with jax.transfer_guard("disallow"):
        result = compiled(*args)
        jax.block_until_ready(result)
    output, errors = jax.device_get(result)
    return {name: value[output["mask"]] for name, value in output.items() if name != "mask"}, errors


@pytest.mark.parametrize("onspill", [False, True])
def test_physical_outputs_match_staged_digis(cond, helpers, onspill):
    from trkfastsim.digi import digi_jax as reference
    from trkfastsim.digi.digicond import Conditions
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = onspill
    st = helpers._steps([helpers._step(c, 520), helpers._step(c, 521, ion=1),
                         helpers._step(c, 92, t=100.), helpers._step(c, 523, t=2800.),
                         helpers._step(c, 524, eion=1e-6)], ev=[0, 0, 2, 2, 2])
    times = timing([1700., 1675., 1695.] if onspill else [100000.] * 3, [10., -5., 20.])
    model = prepare(c, Capacities(64, 1024, 8, 512, 8, 2), 3)
    got, errors = evaluate(model, collection(st, 8), times)
    assert not any(errors.values()), errors
    expected = reference.digitize(st, 3, c, jax.random.key(3),
                                  event_timing={"pbtime": times["mc_pbtime"], "event_length": times["event_length"]})
    assert len(got["sid"]) > 0
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=0, atol=1e-6, err_msg=name)


def test_cluster_compaction_and_electron_expansion_exact(cond, helpers):
    import jax.numpy as jnp
    from trkfastsim.digi import digitize as dg, digi_clusters_jax as dc
    st = helpers._steps([helpers._step(cond, 520), helpers._step(cond, 521, ion=1),
                         helpers._step(cond, 522), helpers._step(cond, 523, eion=1e-6)])
    st["shape"][1] = 3
    st["us"] = dg.unique_straw(st["sid"])
    selected = np.array([True, True, True, False])
    model = prepare(cond, Capacities(64, 1024, 4, 512, 4), 1)
    key = jax.random.fold_in(jax.random.key(5), 0)
    result, errors = jax.jit(model.clusters)(jax.device_put(st), jnp.asarray(selected), jax.device_put(model.assets), key)
    result, errors = jax.device_get((result, errors))
    assert not any(errors.values())
    expected = dc.make_clusters(key, cond, {**st, "_us": st["us"]}, np.flatnonzero(selected))
    for name, values in zip(("step", "pos", "ne"), expected):
        np.testing.assert_array_equal(result[name][result["mask"]], values)


@pytest.mark.parametrize("kind", ["drift_clusters", "clusters_per_step", "straws", "clusters_per_end", "pairs_per_straw"])
def test_no_silent_capacity_truncation(cond, helpers, kind):
    limits = dict(clusters_per_step=64, drift_clusters=1024, straws=4,
                  clusters_per_end=512, pairs_per_straw=8, electronics_batch=1)
    limits[kind] = 1
    times = [1000., 2000., 3000.]
    rows = [helpers._step(cond, 520, t=value) for value in times]
    if kind == "straws":
        rows[1] = helpers._step(cond, 521)
    st = helpers._steps(rows)
    model = prepare(cond, Capacities(**limits), 1)
    _, errors = evaluate(model, collection(st), timing([100000.]))
    assert any(errors.values()), errors
    target = "electronics_search_or_pairs" if kind == "pairs_per_straw" else kind
    assert errors[target]


def test_no_steps_empty_event_and_poisoned_padding(cond, helpers):
    st = helpers._steps([helpers._step(cond, 520)])
    rows = collection(st, 4)
    rows["mask"][:] = False
    rows["columns"]["t"][:] = np.nan
    model = prepare(cond, Capacities(16, 128, 4, 128, 4), 3)
    got, errors = evaluate(model, rows, timing([100000.] * 3))
    assert not any(errors.values())
    assert len(got["sid"]) == 0


@pytest.mark.parametrize("field,value", [("eion", np.nan), ("len", -1.), ("sid", 65535)])
def test_invalid_inputs_return_errors(cond, helpers, field, value):
    st = helpers._steps([helpers._step(cond, 520)])
    st[field][0] = value
    model = prepare(cond, Capacities(16, 128, 4, 128, 4), 1)
    _, errors = evaluate(model, collection(st), timing([100000.]))
    assert errors["input"]


@pytest.mark.parametrize("name,value", [("shape", np.nan), ("ion", np.nan), ("panel", 8), ("straw", 128)])
def test_selectors_and_component_ids_cannot_alias(cond, helpers, name, value):
    st = helpers._steps([helpers._step(cond, 520)])
    sid = st["sid"]
    st.update(plane=sid >> 10, panel=(sid >> 7) & 7, straw=sid & 127)
    st[name] = np.array([value])
    model = prepare(cond, Capacities(16, 128, 4, 128, 4), 1)
    _, errors = evaluate(model, collection(st), timing([100000.]))
    assert errors["input"]


@pytest.mark.parametrize("name", ["sid", "ev"])
def test_fractional_ids_rejected_before_cast(cond, helpers, name):
    st = helpers._steps([helpers._step(cond, 520)])
    st[name] = st[name].astype(float) + .5
    model = prepare(cond, Capacities(16, 128, 4, 128, 4), 1)
    with pytest.raises(ValueError, match="integer dtypes"):
        evaluate(model, collection(st), timing([100000.]))


def test_larger_capacity_and_batch_preserve_output(cond, helpers):
    st = helpers._steps([helpers._step(cond, 520), helpers._step(cond, 521)], ev=[0, 2])
    times = timing([100000.] * 3)
    a, ae = evaluate(prepare(cond, Capacities(32, 512, 4, 256, 4, 1), 3), collection(st, 4), times)
    b, be = evaluate(prepare(cond, Capacities(64, 1024, 8, 512, 8, 4), 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values())
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=0)


def test_ghost_packing_exact_at_boundaries(cond):
    from trkfastsim.digi.digicond import Conditions
    from trkfastsim.digi.timing import ghost_indices
    import jax.numpy as jnp
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    buf = (c.nADCSamples - c.nADCPresamples) * c.ADCPeriod
    gt = np.array([[buf, c.mbtime - buf], [np.nextafter(np.float32(buf), np.float32(0)),
                   np.nextafter(np.float32(c.mbtime - buf), np.float32(2000))],
                   [50., 50.], [50., 50.]], np.float32)
    cs = np.arange(4)
    st = {"sid": np.full(4, 1000, np.int64), "ev": np.zeros(4, np.int64)}
    cluster = {"step": cs, "mask": np.ones(4, bool)}
    charge, wd = np.arange(4.) + 1, np.arange(8.).reshape(4, 2) + 400
    model = prepare(c, Capacities(4, 4, 1, 16, 4, 1), 1)
    packed, errors = jax.jit(model.pack)(jax.device_put(st), jax.device_put(cluster), jnp.asarray(gt),
        jnp.asarray(charge), jnp.asarray(wd), jax.device_put(timing([1700.])), jax.random.key(3), jax.device_put(model.assets))
    packed, errors = jax.device_get((packed, errors))
    assert not any(errors.values())
    source, times = ghost_indices(gt.ravel(), c.mbtime, buf)
    for end in (0, 1):
        selected = np.flatnonzero(source % 2 == end)
        selected = selected[np.argsort(times[selected], kind="stable")]
        n = len(selected)
        assert packed[4][0, end] == n
        np.testing.assert_array_equal(packed[0][0, end, :n], times[selected])
        np.testing.assert_array_equal(packed[1][0, end, :n], charge[source[selected] // 2])
        np.testing.assert_array_equal(packed[2][0, end, :n], wd[source[selected] // 2, end])
        np.testing.assert_array_equal(packed[3][0, end, :n], source[selected] // 2)


def test_charge_block_keys_cross_4096_boundary(cond):
    import jax.numpy as jnp
    from trkfastsim.digi import digi_jax as dj
    n = 4100
    model = prepare(cond, Capacities(4, n, 1, 4, 1, 1), 1)
    us = np.full(n, 520, np.int64)
    position = np.tile(cond.straw_mid[520], (n, 1)) + [100., 1., 0.]
    cluster = {"step": np.arange(n), "pos": position, "ne": np.ones(n, np.int64), "mask": np.ones(n, bool)}
    st = {"us": us, "time": np.full(n, 500.)}
    key = jax.random.key(5)
    got = jax.device_get(jax.jit(model.drift)(jax.device_put(st), jax.device_put(cluster), key, jax.device_put(model.assets)))
    assert not got[-1]
    for first in (0, 4096):
        take = min(4096, n - first)
        pad = 4096 - take
        result = dj._charge_kernel(cond)(jax.random.fold_in(jax.random.fold_in(key, 1), first),
            jnp.asarray(np.pad(us[first:first + take], (0, pad))),
            jnp.asarray(np.pad(position[first:first + take], ((0, pad), (0, 0)))),
            jnp.ones(4096, jnp.int64), jnp.asarray(np.pad(st["time"][first:first + take], (0, pad))))
        for actual, expected in zip(got[:3], jax.device_get(result)):
            # Fused CUDA arithmetic can shift a rounded floating output by one ULP.
            np.testing.assert_array_max_ulp(actual[first:first + take], expected[:take], maxulp=1)


def test_marker_jitter_crosses_event_block_and_masked_timing(cond, helpers):
    from trkfastsim.digi import digi_jax as dj
    st = helpers._steps([helpers._step(cond, 520), helpers._step(cond, 521)], ev=[0, 128])
    model = prepare(cond, Capacities(32, 512, 4, 256, 4, 1), 130)
    times = timing([100000.] * 130)
    times["event_mask"][-1] = False
    times["mc_pbtime"][-1] = np.nan
    actual, errors = evaluate(model, collection(st, 4), times)
    assert not any(errors.values())
    expected = dj.digitize(st, 129, cond, jax.random.key(3), event_timing={"pbtime": times["mc_pbtime"][:-1],
                                                                      "event_length": times["event_length"][:-1]})
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-6, err_msg=name)
