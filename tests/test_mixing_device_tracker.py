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


@pytest.fixture(autouse=True, scope="module")
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


@pytest.fixture(scope="module")
def wide(cond):
    """wide(onspill): conditions and the (64, 1024, 8, 512, 8, 2) three-event model, compiled once per module."""
    from trkfastsim.digi.digicond import Conditions
    models = {}

    def get(onspill):
        if onspill not in models:
            c = Conditions(cond.s, cond.a)
            c.s["onspill"] = onspill
            models[onspill] = c, prepare(c, Capacities(64, 1024, 8, 512, 8, 2), 3)
        return models[onspill]
    return get


@pytest.fixture(scope="module")
def small(cond):
    """One single-event model for the input-validation tests: each input structure compiles once, not once per test."""
    return prepare(cond, Capacities(16, 128, 4, 128, 4), 1)


def evaluate(model, rows, times, seed=3):
    args = jax.device_put((rows, times, model.assets, jax.random.key(seed)))
    _audit_program(jax.make_jaxpr(model.apply)(*args))
    compiled = jax.jit(model.apply)   # same function object per model, so JAX reuses its compiled program
    with jax.transfer_guard("disallow"):
        result = compiled(*args)
        jax.block_until_ready(result)
    output, errors = jax.device_get(result)
    return {name: value[output["mask"]] for name, value in output.items() if name != "mask"}, errors


@pytest.mark.pr_params("True")   # onspill is the mixing mode
@pytest.mark.parametrize("onspill", [False, True])
def test_physical_outputs_match_staged_digis(wide, helpers, onspill):
    from trkfastsim.digi import digi_jax as reference
    c, model = wide(onspill)
    st = helpers._steps([helpers._step(c, 520), helpers._step(c, 521, ion=1),
                         helpers._step(c, 92, t=100.), helpers._step(c, 523, t=2800.),
                         helpers._step(c, 524, eion=1e-6)], ev=[0, 0, 2, 2, 2])
    times = timing([1700., 1675., 1695.] if onspill else [100000.] * 3, [10., -5., 20.])
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


@pytest.mark.pr_params("pairs_per_straw")   # one compile per limit; this one also checks the flag remap
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


@pytest.mark.slow   # empty events per PR: test_mixing_device all_empty test
def test_no_steps_empty_event_and_poisoned_padding(cond, helpers):
    st = helpers._steps([helpers._step(cond, 520)])
    rows = collection(st, 4)
    rows["mask"][:] = False
    rows["columns"]["t"][:] = np.nan
    model = prepare(cond, Capacities(16, 128, 4, 128, 4), 3)
    got, errors = evaluate(model, rows, timing([100000.] * 3))
    assert not any(errors.values())
    assert len(got["sid"]) == 0


@pytest.mark.slow   # input-error path per PR: selectors test [shape-nan]
@pytest.mark.parametrize("field,value", [("eion", np.nan), ("len", -1.), ("sid", 65535)])
def test_invalid_inputs_return_errors(small, cond, helpers, field, value):
    st = helpers._steps([helpers._step(cond, 520)])
    st[field][0] = value
    _, errors = evaluate(small, collection(st), timing([100000.]))
    assert errors["input"]


@pytest.mark.pr_params("shape-nan")   # the per-PR input-error case
@pytest.mark.parametrize("name,value", [("shape", np.nan), ("ion", np.nan), ("panel", 8), ("straw", 128)])
def test_selectors_and_component_ids_cannot_alias(small, cond, helpers, name, value):
    st = helpers._steps([helpers._step(cond, 520)])
    sid = st["sid"]
    st.update(plane=sid >> 10, panel=(sid >> 7) & 7, straw=sid & 127)
    st[name] = np.array([value])
    _, errors = evaluate(small, collection(st), timing([100000.]))
    assert errors["input"]


@pytest.mark.parametrize("name", ["sid", "ev"])
def test_fractional_ids_rejected_before_cast(small, cond, helpers, name):
    st = helpers._steps([helpers._step(cond, 520)])
    st[name] = st[name].astype(float) + .5
    with pytest.raises(ValueError, match="integer dtypes"):
        evaluate(small, collection(st), timing([100000.]))


@pytest.mark.slow   # two compiles; block independence per PR: charge block test
def test_larger_capacity_and_batch_preserve_output(cond, helpers):
    st = helpers._steps([helpers._step(cond, 520), helpers._step(cond, 521)], ev=[0, 2])
    times = timing([100000.] * 3)
    a, ae = evaluate(prepare(cond, Capacities(32, 512, 4, 256, 4, 1), 3), collection(st, 4), times)
    b, be = evaluate(prepare(cond, Capacities(64, 1024, 8, 512, 8, 4), 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values())
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=0)


@pytest.mark.slow   # ghost packing runs inside the onspill parity test per PR
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


@pytest.mark.slow   # the physical-output parity check at a 130-event shape
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


def _tier_steps(cond, helpers):
    st = helpers._steps([helpers._step(cond, 520), helpers._step(cond, 521, ion=1), helpers._step(cond, 522),
                         helpers._step(cond, 523), helpers._step(cond, 524, eion=1e-6), helpers._step(cond, 525)],
                        ev=[0, 0, 1, 1, 2, 2])
    st["shape"][1] = 3
    return st


@pytest.mark.slow   # tiered clusters per PR: test_long_step_and_end_tiers_digis_equal_wide
def test_tiered_cluster_capacity_exact(cond, helpers):
    """Long steps through the gathered tier give the same clusters, order and electrons as one wide capacity."""
    import jax.numpy as jnp
    from trkfastsim.digi import digitize as dg
    st = _tier_steps(cond, helpers)
    st["us"] = dg.unique_straw(st["sid"])
    selected = np.ones(len(st["sid"]), bool)
    wide = prepare(cond, Capacities(64, 2048, 8, 512, 4), 3)
    tiered = prepare(cond, Capacities(2, 2048, 8, 512, 4, long_steps=6, clusters_per_long_step=64), 3)
    key = jax.random.fold_in(jax.random.key(5), 0)
    args = (jax.device_put(st), jnp.asarray(selected))
    a, ae = jax.device_get(jax.jit(wide.clusters)(*args, jax.device_put(wide.assets), key))
    b, be = jax.device_get(jax.jit(tiered.clusters)(*args, jax.device_put(tiered.assets), key))
    assert not any(ae.values()) and not any(be.values()), (ae, be)
    np.testing.assert_array_equal(a["mask"], b["mask"])
    assert a["mask"].sum() > 20
    for name in ("step", "pos", "ne"):
        np.testing.assert_array_equal(a[name][a["mask"]], b[name][b["mask"]], err_msg=name)


@pytest.mark.slow   # four compiles; tiers per PR: test_long_step_and_end_tiers_digis_equal_wide
def test_tiered_capacity_digis_and_overflow(cond, helpers):
    st = _tier_steps(cond, helpers)
    times = timing([100000.] * 3)
    a, ae = evaluate(prepare(cond, Capacities(64, 2048, 8, 512, 8, 2), 3), collection(st, 8), times)
    b, be = evaluate(prepare(cond, Capacities(2, 2048, 8, 512, 8, 2, long_steps=6, clusters_per_long_step=64), 3),
                     collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values())
    assert len(a["sid"]) > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    _, errors = evaluate(prepare(cond, Capacities(2, 2048, 8, 512, 8, 2, long_steps=1, clusters_per_long_step=64), 3),
                         collection(st, 8), times)
    assert errors["long_steps"]
    _, errors = evaluate(prepare(cond, Capacities(2, 2048, 8, 512, 8, 2, long_steps=6, clusters_per_long_step=3), 3),
                         collection(st, 8), times)
    assert errors["clusters_per_step"]
    with pytest.raises(ValueError, match="together"):
        Capacities(2, 2048, 8, 512, 8, 2, long_steps=6)
    with pytest.raises(ValueError, match="exceed"):
        Capacities(8, 2048, 8, 512, 8, 2, long_steps=6, clusters_per_long_step=8)


@pytest.mark.slow   # six compiles per case (opt-in end tiers)
@pytest.mark.parametrize("onspill", [False, True])
def test_tiered_straw_electronics_exact(cond, helpers, onspill):
    """Busy straws through the wide electronics tier give the same digis as one wide clusters_per_end."""
    from trkfastsim.digi.digicond import Conditions
    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = onspill
    st = helpers._steps([helpers._step(c, 520), helpers._step(c, 520, t=40.), helpers._step(c, 521, ion=1),
                         helpers._step(c, 522), helpers._step(c, 92, t=100.), helpers._step(c, 523, t=900.)],
                        ev=[0, 0, 0, 1, 1, 2])
    times = timing([1700., 1675., 1695.] if onspill else [100000.] * 3, [10., -5., 20.])
    a, ae = evaluate(prepare(c, Capacities(64, 2048, 8, 512, 8, 2), 3), collection(st, 8), times)
    tiered = Capacities(64, 2048, 8, 4, 8, 2, end_tiers=[(8, 512)])
    b, be = evaluate(prepare(c, tiered, 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values()), (ae, be)
    assert len(a["sid"]) > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    both = Capacities(2, 2048, 8, 4, 8, 2, long_steps=6, clusters_per_long_step=64, end_tiers=[(8, 512)])
    d, de = evaluate(prepare(c, both, 3), collection(st, 8), times)
    assert not any(de.values())
    for name in a:
        np.testing.assert_array_equal(a[name], d[name], err_msg=name)
    _, errors = evaluate(prepare(c, Capacities(64, 2048, 8, 4, 8, 2, end_tiers=[(1, 512)]), 3),
                         collection(st, 8), times)
    assert errors["end_tier_1_straws"]
    three = Capacities(64, 2048, 8, 4, 8, 2, end_tiers=[(8, 16), (4, 512)])
    e, ee = evaluate(prepare(c, three, 3), collection(st, 8), times)
    assert not any(ee.values())
    for name in a:
        np.testing.assert_array_equal(a[name], e[name], err_msg=name)
    _, errors = evaluate(prepare(c, Capacities(64, 2048, 8, 4, 8, 2, end_tiers=[(8, 16)]), 3), collection(st, 8), times)
    assert errors["clusters_per_end"]
    with pytest.raises(ValueError, match="increase"):
        Capacities(64, 2048, 8, 512, 8, 2, end_tiers=[(4, 512)])
    with pytest.raises(ValueError, match="pairs"):
        Capacities(64, 2048, 8, 4, 8, 2, end_tiers=[(4,)])


def test_long_step_and_end_tiers_digis_equal_wide(wide, helpers):
    """Per-PR tier check: the long-step tier and two end tiers together give the wide model's onspill digis."""
    c, model = wide(True)
    st = helpers._steps([helpers._step(c, 520), helpers._step(c, 520, t=40.), helpers._step(c, 521, ion=1),
                         helpers._step(c, 522), helpers._step(c, 92, t=100.), helpers._step(c, 523, t=900.)],
                        ev=[0, 0, 0, 1, 1, 2])
    times = timing([1700., 1675., 1695.], [10., -5., 20.])
    a, ae = evaluate(model, collection(st, 8), times)
    tiered = Capacities(2, 1024, 8, 4, 8, 2, long_steps=6, clusters_per_long_step=64, end_tiers=[(8, 16), (4, 512)])
    b, be = evaluate(prepare(c, tiered, 3), collection(st, 8), times)
    assert not any(ae.values()) and not any(be.values()), (ae, be)
    assert len(a["sid"]) > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    # tier validation (also in the slow tier tests)
    with pytest.raises(ValueError, match="together"):
        Capacities(2, 1024, 8, 512, 8, 2, long_steps=6)
    with pytest.raises(ValueError, match="exceed"):
        Capacities(8, 1024, 8, 512, 8, 2, long_steps=6, clusters_per_long_step=8)
    with pytest.raises(ValueError, match="increase"):
        Capacities(64, 1024, 8, 512, 8, 2, end_tiers=[(4, 512)])
    with pytest.raises(ValueError, match="pairs"):
        Capacities(64, 1024, 8, 4, 8, 2, end_tiers=[(4,)])


def _flat_digit_steps(c, helpers):
    """Three pairs on one straw (digital dead time), a saturated non-minimum-ionising straw, a read-all straw."""
    rows = [helpers._step(c, 520, t=t) for t in (300., 700., 1100.)]
    rows += [helpers._step(c, 1000, eion=.04, ion=1), helpers._step(c, 92, t=100.), helpers._step(c, 523, t=900.)]
    return helpers._steps(rows, ev=[0, 0, 0, 1, 1, 2])


@pytest.mark.pr_params("True")   # onspill is the mixing mode
@pytest.mark.parametrize("onspill", [False, True])
def test_flat_digit_entries_equal_pair_loop(wide, helpers, onspill):
    """digit_entries_per_straw digitises the flat (straw, pair) entry list, also in the end tiers: the same digis."""
    c, model = wide(onspill)
    st = _flat_digit_steps(c, helpers)
    times = timing([1700., 1675., 1695.] if onspill else [100000.] * 3, [10., -5., 20.])
    a, ae = evaluate(model, collection(st, 8), times)
    assert not any(ae.values()), ae
    assert len(a["sid"]) > 0 and np.bincount(a["sid"]).max() >= 2
    for caps in (Capacities(64, 1024, 8, 512, 8, 2, digit_entries_per_straw=2),
                 Capacities(64, 1024, 8, 4, 8, 2, end_tiers=[(4, 512)], digit_entries_per_straw=2.5)):
        b, be = evaluate(prepare(c, caps, 3), collection(st, 8), times)
        assert not any(be.values()), be
        for name in a:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    _, errors = evaluate(prepare(c, Capacities(64, 1024, 8, 512, 8, 2, digit_entries_per_straw=0.5), 3),
                         collection(st, 8), times)
    assert errors["digit_entries_per_straw"]
    for bad in (0, 9, "1"):
        with pytest.raises(ValueError, match="digit_entries_per_straw"):
            Capacities(64, 1024, 8, 512, 8, 2, digit_entries_per_straw=bad)


def test_float32_tracker_mode_in_subprocess(request):
    """Run the float32 device-versus-staged tests in a process whose tracker device float is float32."""
    import os
    import subprocess
    here = Path(__file__).resolve().parent
    env = {**os.environ, "TRKFASTSIM_DEVICE_FLOAT": "core", "MU2EFS_DEVICE_FLOAT": "float32", "JAX_ENABLE_X64": "1"}
    runslow = ["--runslow"] if request.config.getoption("--runslow", False) else []   # its slow cases run nightly
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *runslow,
                             str(here / "test_mixing_device_tracker_float32.py")],
                            env=env, cwd=here.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    assert " passed" in result.stdout or " skipped" in result.stdout


@pytest.mark.pr_params("2")   # two blocks per launch leave a padded remainder launch
@pytest.mark.parametrize("group", [2, 3])
def test_batched_charge_blocks_equal_sequential(cond, group):
    """charge_blocks_per_launch vmaps blocks with their own keys: same outputs as one block per step."""
    n = 2 * 4096 + 17
    us = np.full(n, 520, np.int64)
    position = np.tile(cond.straw_mid[520], (n, 1)) + [100., 1., 0.]
    position[:, 0] += np.linspace(-50., 50., n)
    cluster = {"step": np.arange(n), "pos": position, "ne": 1 + np.arange(n) % 3, "mask": np.arange(n) < n - 5}
    st = {"us": us, "time": np.linspace(400., 900., n)}
    key = jax.random.key(7)
    def run(capacities):
        model = prepare(cond, capacities, 1)
        return jax.device_get(jax.jit(model.drift)(jax.device_put(st), jax.device_put(cluster), key,
                                                   jax.device_put(model.assets)))
    a = run(Capacities(4, n, 1, 4, 1, 1))
    b = run(Capacities(4, n, 1, 4, 1, 1, charge_blocks_per_launch=group))
    assert not a[-1] and not b[-1]
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="positive"):
        Capacities(4, n, 1, 4, 1, 1, charge_blocks_per_launch=0)
