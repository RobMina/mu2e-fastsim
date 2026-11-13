"""Actual-JAX contracts for resident physical CRV kernels and device stage packing."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
# Match the shared suites' float64 parity mode before importing device kernels.
pytest.importorskip("mu2efastsim.testing")
pytest.importorskip("crvfastsim")
import jax.numpy as jnp

from mixing import device_crv as D
from crvfastsim import chain as CH
from crvfastsim.geometry import find_counters
from crvfastsim.response import lookup, photons as PH, sipm as SI, sipm_jax as SJ
from crvfastsim.reco import waveform as WF, pulsefit as PF, coinc as CO


@pytest.fixture(autouse=True, scope="module")
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def template():
    # Resolve the actual public Offline template; tests must never fetch assets.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CRVFS_NO_FETCH", "1")
        try:
            return Path(find_counters(WF.WaveformConfig().single_pe_file))
        except FileNotFoundError:
            pytest.skip("public singlePEWaveform_v3.txt unavailable; set MU2E_SEARCH_PATH")


def tiny_lookup():
    edges = dict(x=np.array([0., 10.]), y=np.array([-25., 25.]), z=np.array([-100., -33., 33., 100.]),
                 beta=np.array([0., 1.]), theta=np.array([0., np.pi]), phi=np.array([0., np.pi]), r=np.array([0., 1.]))
    delay = np.zeros((3, 8), np.uint8)
    delay[np.arange(3), [2, 3, 4]] = 1
    def table():
        return lookup.Table(np.array([.8, .6, .4], np.float32), delay, delay.sum(1),
                            np.ones((3, 1), np.uint8), np.ones(3, np.int64))
    return lookup.LookupTable("LookupTable_tiny", (6, 0), 0, 10., 25., 100., 10., 1., 1., .5, 1., .126, 0.,
                              np.array([[1., 0.]]), np.array([[1., 0.]]), edges,
                              {name: table() for name in lookup.TABLES})


def capacity(**changes):
    values = dict(substeps=32, photons=128, thermal=64, ct1=64, ct2=32, ct3=16, pixel_short=8,
                  pixel_long_streams=8, pixel_long_hits=32, charge_clusters=32, charges_per_cluster=128,
                  waveform_samples=144, digis=32, reco_waveforms=32, reco_samples=48, pulse_rounds=4,
                  coinc_hits=16, coinc_combinations=4)
    return {**values, **changes}


@pytest.fixture(scope="module")
def setup(template):
    geom = SimpleNamespace(nbars=4, sector_names=["EX"], sector=np.zeros(4, np.int64),
                           module=np.zeros(4, np.int64), layer=np.arange(4, dtype=np.int64),
                           counter=np.zeros(4, np.int64), center=np.array([[30. * i, 0., 0.] for i in range(4)]),
                           local_to_world=np.tile(np.arange(3), (4, 1)), halflen=np.tile([10., 25., 100.], (4, 1)))
    counters = {"crs.sectorNames": ["EX"], "crs.sipmsAtSide0EX": True, "crs.sipmsAtSide1EX": True,
                "crs.sectorTypeEX": 1, "crs.precedingSectorForEX": -1,
                "crs.nCountersPerModuleEX": 1, "crs.nModulesEX": 1}
    z = np.zeros(16)
    cond = CH.Conditions(z + 100, z + 400, z + 11, z + 7, z.astype(int), z.astype(int), z.astype(int), z)
    cfg = CH.ChainConfig(response=PH.ResponseConfig(sectors={"EX": PH.SectorConfig("tiny", 0, 100.)}),
                         sipm=replace(SI.SiPMConfig(), thermal_rate=1e-4), window=(200., 1500.), noise=False,
                         waveform=replace(WF.WaveformConfig(), single_pe_file=str(template)))
    return geom, cond, cfg, counters, {"tiny": tiny_lookup()}, np.ones((40, 40))


def adapter(setup, capacities=None, cfg=None, cond=None):
    geom, conditions, config, counters, tables, pmap = setup
    return D.prepare(geom, cond if cond is not None else conditions, cfg if cfg is not None else config,
                     counters, capacity() if capacities is None else capacities, tables, pmap)


def inputs(events=2, rows=4, active=True):
    values = {name: np.zeros(rows) for name in D.STEP_FLOATS}
    values.update(bar=np.arange(rows, dtype=np.int64) % 4, pdg=np.full(rows, 13, np.int64),
                  t0=np.full(rows, 300.), t1=np.full(rows, 301.),
                  x0=(np.arange(rows) % 4) * 30., x1=(np.arange(rows) % 4) * 30. + .1,
                  pz0=np.full(rows, 1000.), p1=np.full(rows, 1000.), len=np.full(rows, .1), edep=np.full(rows, .5))
    collection = dict(columns=values, mask=np.full(rows, active), event=np.zeros(rows, np.int64))
    timing = dict(mc_pbtime=np.zeros(events), reco_pbtime=np.full(events, 10.),
                  event_length=np.full(events, 1675.), event_mask=np.ones(events, bool))
    return collection, timing


def host(x):
    return jax.device_get(x)


# One jitted callable per bound method (model.apply etc.): a bound method is a new object on every attribute access,
# so jax.jit(model.apply) alone would trace and compile the same program again on every call.
_JITTED = {}


@pytest.fixture(autouse=True, scope="module")
def _drop_jitted():
    yield
    _JITTED.clear()


def jitted(fn):
    owner = getattr(fn, "__self__", None)
    if owner is None:
        return jax.jit(fn)
    key = (id(owner), fn.__func__)
    if key not in _JITTED:
        _JITTED[key] = owner, jax.jit(fn)
    return _JITTED[key][1]


def run(fn, *args):
    resident = jax.device_put(args)
    compiled = jitted(fn)
    with jax.transfer_guard("disallow"):
        out = compiled(*resident)
        jax.block_until_ready(out)
    return host(out)


def test_stock_stepgate_unwrapped_and_period():
    a = np.array([99., 100., 1500., 1501., 1600., 1600., 1690., -5., 1795., 99., np.nan])
    b = np.array([99., 100., 1500., 1501., 50., 100., 1805., 110., 1795., 100., 300.])
    result = run(D._step_gate, a, b, np.zeros(len(a)))
    np.testing.assert_array_equal(np.flatnonzero(result), [1, 2, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(run(D._fold, np.array([-1695., -1., 1695., 1700.])), [0., 1694., 0., 5.])


@pytest.mark.slow   # row parity per PR: reflector row geometry test
def test_row_port_preserves_original_endpoints(setup):
    model = adapter(setup)
    collection, _ = inputs(rows=1)
    c = collection["columns"]
    c["t0"][:] = 1690.
    c["t1"][:] = 1805.
    fn = lambda columns, assets, key: model._rows(columns, columns["bar"], jnp.ones(1, bool), assets, model.blocks[0], key)
    rows, channels = run(fn, c, model.assets, jax.random.key(3))
    np.testing.assert_array_equal(rows["t0"], [1690.])
    np.testing.assert_array_equal(rows["t1"], [1805.])
    np.testing.assert_array_equal(channels, [0])
    np.testing.assert_allclose(rows["s"], [[0, 0, 0]])
    np.testing.assert_allclose(rows["d"], [[.1, 0, 0]])


def test_thermal_all_physical_empty_events_and_partial_mask(setup):
    cond = replace(setup[1], status=np.array([0, 1, 2, 4] * 4))
    model = adapter(setup, capacity(thermal=4096), cond=cond)
    _, timing = inputs(events=512, active=False)
    timing["event_mask"][-32:] = False
    dark, overflow = run(model._thermal, timing, model.assets, jax.random.key(21))
    assert not overflow
    valid = dark["mask"]
    assert valid.sum() > 500
    ch = dark["chan"][valid] % CH.CHAN_MULT
    assert set(ch) == set(range(0, 16, 2))
    assert dark["chan"][valid].max() // CH.CHAN_MULT < 480
    assert np.all((dark["time"][valid] >= 100) & (dark["time"][valid] < 1500))
    assert np.unique(dark["pixel"][valid]).size > 300
    mean = 480 * 8 * 1400e-4 / model.cfg.sipm.p_fire(model.cfg.sipm.overvoltage)
    assert abs(valid.sum() - mean) < 5 * np.sqrt(mean)


def test_real_physical_chain_resident_repeatable_and_tail(setup):
    model = adapter(setup)
    collection, timing = inputs()
    timing["event_mask"][1] = False
    args = collection, timing, model.assets, jax.random.key(19)
    out, errors = run(model.apply, *args)
    assert not any(errors.values()), errors
    assert out["digis"]["mask"].sum() > 0
    assert out["pulses"]["mask"].sum() > 0
    assert out["clusters"]["mask"].sum() > 0
    for table in out.values():
        assert np.all(table["ev"][table["mask"]] == 0)
    repeated = run(model.apply, *args)
    for a, b in zip(jax.tree_util.tree_leaves((out, errors)), jax.tree_util.tree_leaves(repeated)):
        np.testing.assert_array_equal(a, b)
    text = str(jax.make_jaxpr(model.apply)(*jax.device_put(args)))
    assert "callback" not in text


def test_waveform_device_packing_matches_reference(setup):
    model = adapter(setup, cfg=replace(setup[2], feb_phase_width=0.))
    _, timing = inputs(events=2)
    times = np.array([187.5] * 10 + [300.] * 10 + [1490.] * 10 + [310.] * 10)
    labels = np.array([0] * 30 + [CH.CHAN_MULT] * 10)
    q = dict(chan=labels, time=times, charge=np.full(40, model.wcfg.single_pe_reference_charge), mask=np.ones(40, bool))
    actual, errors = run(model.digis, q, timing, model.assets, jax.random.key(5))
    assert not any(errors.values()), errors
    expected = WF.synthesize(labels, times, q["charge"], lambda ch: np.zeros(len(ch)), model.wcfg,
                             window=(200., 1500.), rng=None)
    mask = actual["mask"]
    np.testing.assert_array_equal(actual["starttdc"][mask], expected["starttdc"])
    np.testing.assert_array_equal(actual["adc"][mask], expected["adc"])
    np.testing.assert_array_equal(actual["ev"][mask], expected["chan"] // CH.CHAN_MULT)


def test_contiguous_digis_reco_and_independent_bunch_correction(setup):
    model = adapter(setup, cfg=replace(setup[2], pulse_reco=replace(PF.RecoConfig(), use_time_offset_db=False)))
    _, timing = inputs(events=2)
    sample = np.arange(24) * 12.5
    adc = np.rint(100 + PF.gumbel(sample, 400., 125., 19.)).astype(np.int16)
    d = dict(mask=np.array([True, True, False]), ev=np.array([0, 0, 1]), bar=np.zeros(3, int), sipm=np.zeros(3, int),
             starttdc=np.array([20, 32, 20]), adc=np.stack([adc[:12], adc[12:], np.full(12, 100)]))
    p, errors = run(model.pulses, d, timing, model.assets)
    assert not any(errors.values()), errors
    assert p["mask"].sum() == 1
    changed = {**timing, "reco_pbtime": timing["reco_pbtime"] + 71., "mc_pbtime": timing["mc_pbtime"] + 999.}
    other, _ = run(model.pulses, d, changed, model.assets)
    np.testing.assert_allclose(other["time"][other["mask"]], p["time"][p["mask"]] - 71.)
    np.testing.assert_array_equal(other["pe"][other["mask"]], p["pe"][p["mask"]])


@pytest.mark.slow   # full-apply compile; stage capacity flags per PR
def test_capacities_reject_without_output_or_fallback(setup):
    model = adapter(setup, capacity(substeps=1, photons=1, thermal=1))
    collection, timing = inputs()
    out, errors = run(model.apply, collection, timing, model.assets, jax.random.key(9))
    assert errors["crv_substeps"] or errors["crv_photons"] or errors["crv_thermal"]
    assert all(not table["mask"].any() for table in out.values())


def test_unsupported_configuration_and_numeric_assets(setup):
    for cfg in (replace(setup[2], fused_response=True), replace(setup[2], background=object()),
                replace(setup[2], sipm=replace(setup[2].sipm, trap0_prob=.1))):
        with pytest.raises((ValueError, NotImplementedError)):
            adapter(setup, cfg=cfg)
    with pytest.raises(ValueError, match="capacities"):
        adapter(setup, {"photons": 12})
    model = adapter(setup)
    assert all(x.dtype.kind in "biuf" and not x.flags.writeable for x in jax.tree_util.tree_leaves(model.assets))


def test_coincidence_packing_matches_independent_numpy(setup):
    model = adapter(setup)
    _, timing = inputs(events=3)
    pulses = dict(mask=np.array([True] * 9 + [False]), ev=np.array([0] * 4 + [1] * 5 + [2]),
                  bar=np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 0]), sipm=np.array([0] * 8 + [1, 0]),
                  seq=np.zeros(10, int), idx=np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 0]),
                  time=np.array([300., 302., 304., 306., 700., 702., 704., 706., 701., 0.]),
                  pe=np.full(10, 12.))
    out, errors = run(model.clusters, pulses, timing, model.assets)
    assert not any(errors.values())
    geometry = CO.sector_geom(setup[0], setup[3])
    expected = []
    for ev in (0, 1):
        selected = pulses["mask"] & (pulses["ev"] == ev)
        p = {k: v[selected] for k, v in pulses.items() if k not in ("mask", "ev")}
        expected.extend((ev, c) for c in CO.find_event(p, geometry, setup[1].status, model.fcfg))
    valid = np.flatnonzero(out["mask"])
    assert len(valid) == len(expected) == 2
    for row, (ev, cluster) in zip(valid, expected):
        assert out["ev"][row] == ev
        np.testing.assert_allclose(out["pe"][row], cluster["pe"], rtol=2e-6)
        np.testing.assert_allclose(out["time"][row], cluster["avgtime"], rtol=2e-6)
        np.testing.assert_allclose([out[k][row] for k in ("x", "y", "z")], cluster["pos"], atol=2e-5)
    small = adapter(setup, capacity(coinc_hits=2, coinc_combinations=1))
    _, errors = run(small.clusters, pulses, timing, small.assets)
    assert errors["crv_coinc_hits"]


@pytest.mark.slow   # its own full-apply compile (thermal_rate=0)
def test_empty_and_malformed_live_lanes(setup):
    cfg = replace(setup[2], sipm=replace(setup[2].sipm, thermal_rate=0.))
    model = adapter(setup, cfg=cfg)
    collection, timing = inputs(active=False)
    collection["columns"]["t0"][:] = np.inf
    collection["columns"]["x0"][:] = np.nan
    out, errors = run(model.apply, collection, timing, model.assets, jax.random.key(81))
    assert not any(errors.values()), errors
    assert all(not table["mask"].any() for table in out.values())
    collection["mask"][0] = True
    out, errors = run(model.apply, collection, timing, model.assets, jax.random.key(81))
    assert errors["crv_input"]
    assert all(not table["mask"].any() for table in out.values())
    collection, timing = inputs()
    collection["columns"]["t0"][:] = np.nan
    out, errors = run(model.apply, collection, timing, model.assets, jax.random.key(81))
    assert not any(errors.values()), errors
    assert all(not table["mask"].any() for table in out.values())


@pytest.mark.slow   # crvfastsim SiPM kernel, tested in the crv suite
def test_shared_optical_dark_pixel_recovery():
    cum = np.ones(1)
    dead = np.zeros(SJ.NP2, bool)
    def simulate(key):
        return SJ.device_sipm(jnp.array([0, 0, 1]), jnp.array([300., 300., 300.]), jnp.ones(3, bool), key,
                             cum, 1, dead, 0., 3., 13.3, 1., .001, 1., 3, 1, 1, 1, 6, 8, 1, 8, 40, 40,
                             pixel_override=jnp.array([50, 50, 50]))
    result = run(simulate, jax.random.key(32))
    np.testing.assert_array_equal(result["fired"][:3], [True, False, True])
    assert not result["overflow"]


@pytest.mark.slow   # charges compile; fold arithmetic per PR: stepgate test
def test_fold_only_after_physical_optics_and_eventlength_independence(setup):
    model = adapter(setup, cfg=replace(setup[2], sipm=replace(setup[2].sipm, thermal_rate=0.)))
    collection, timing = inputs()
    collection["columns"]["t0"][:] = 1690.
    collection["columns"]["t1"][:] = 1805.
    q, errors = run(model.charges, collection, timing, model.assets, jax.random.key(77))
    assert not any(errors.values())
    assert q["mask"].any()
    assert np.all(q["time"][q["mask"]] < 100.)
    timing["event_length"][:] = 1700.
    q2, errors2 = run(model.charges, collection, timing, model.assets, jax.random.key(77))
    for name in q:
        np.testing.assert_array_equal(q[name], q2[name])
    assert not any(errors2.values())


def test_real_v6_optics_asset_resident_smoke(setup):
    asset_dir = os.environ.get("CRVFS_LOOKUP_DIR")
    if not asset_dir:
        pytest.skip("optional physical v6 smoke requires explicit CRVFS_LOOKUP_DIR")
    directory = Path(asset_dir)
    if not all((directory / name).is_file() for name in ("LookupTable_6000_0", "photonMap.root")):
        pytest.skip("physical v6 lookup table and photonMap.root unavailable in CRVFS_LOOKUP_DIR")
    geom, cond, cfg, counters, _, _ = setup
    response = PH.ResponseConfig(sectors={"EX": PH.SectorConfig("6000_0", 0, 39794.)})
    cfg = replace(cfg, response=response, sipm=replace(cfg.sipm, thermal_rate=0.))
    model = D.prepare(geom, cond, cfg, counters, capacity(photons=256),
                      {"6000_0": lookup.load("6000_0", str(directory))}, SI.photon_map(str(directory / "photonMap.root")))
    collection, timing = inputs(rows=1)
    collection["columns"]["edep"][:] = 2.
    q, errors = run(model.charges, collection, timing, model.assets, jax.random.key(231))
    assert not any(errors.values()), errors
    assert q["mask"].sum() > 0
    assert np.all(q["charge"][q["mask"]] > 0)
    assert np.all(np.isfinite(q["time"][q["mask"]]))


def test_reflector_row_geometry_matches_host_optics(setup):
    import pandas as pd
    geom, cond, cfg, counters, tables, pmap = setup
    table = replace(tables["tiny"], reflector=1)
    cfg = replace(cfg, response=replace(cfg.response, sectors={"EX": PH.SectorConfig("tiny", -1, 100.)},
                                       speed_ratio={"EX": .9}, atten_ratio={"EX": 1.1}))
    model = D.prepare(geom, cond, cfg, counters, capacity(), {"tiny": table}, pmap)
    collection, _ = inputs(rows=1)
    columns = collection["columns"]
    columns["y0"][:] = 3.
    columns["y1"][:] = 4.
    columns["z0"][:] = 7.
    columns["z1"][:] = 8.
    frame = pd.DataFrame(columns)
    reference = list(PH.iter_rows(frame, columns["pdg"], geom, cfg.response, {"tiny": table},
                                  cond.deviation, np.random.default_rng(5)))
    assert len(reference) == len(model.blocks) == 2
    for block, (_, expected) in zip(model.blocks, reference):
        fn = lambda c, a, k: model._rows(c, c["bar"], jnp.ones(1, bool), a, block, k)
        actual, channel = run(fn, columns, model.assets, jax.random.key(8))
        for name in ("s", "d", "t0", "t1", "beta", "theta", "dev", "mu", "nsub"):
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(channel, expected["chan"])


def test_each_waveform_and_reco_capacity_is_fatal(setup):
    cfg = replace(setup[2], feb_phase_width=0.)
    model = adapter(setup, capacity(charge_clusters=1, charges_per_cluster=1, waveform_samples=12, digis=1,
                                     reco_waveforms=1, reco_samples=12, pulse_rounds=1), cfg=cfg)
    _, timing = inputs(events=1)
    q = dict(mask=np.ones(4, bool), chan=np.array([0, 0, 0, 4]), time=np.array([300., 350., 400., 300.]),
             charge=np.full(4, 100 * model.wcfg.single_pe_reference_charge))
    _, errors = run(model.digis, q, timing, model.assets, jax.random.key(4))
    assert errors["crv_charge_clusters"]
    assert errors["crv_cluster_charges"]
    assert errors["crv_waveform_samples"]
    sample = np.arange(24) * 12.5
    adc = np.rint(100 + PF.gumbel(sample, 400., 75., 19.)).astype(np.int16)
    d = dict(mask=np.ones(3, bool), ev=np.zeros(3, int), bar=np.array([0, 0, 1]), sipm=np.zeros(3, int),
             starttdc=np.array([20, 32, 20]), adc=np.stack([adc[:12], adc[12:], adc[:12]]))
    _, errors = run(model.pulses, d, timing, model.assets)
    assert errors["crv_reco_waveforms"]
    assert errors["crv_reco_samples"]
    assert errors["crv_pulse_rounds"]


@pytest.mark.slow   # dark counts per PR: thermal all-physical test
def test_thermal_physical_readout_not_optical_reflector(setup):
    geom, cond, cfg, counters, tabs, pmap = setup
    counters = {**counters, "crs.sipmsAtSide1EX": False}
    cfg = replace(cfg, sipm=replace(cfg.sipm, thermal_rate=.001))
    model = D.prepare(geom, cond, cfg, counters, capacity(thermal=512), tabs, pmap)
    _, timing = inputs(events=8, active=False)
    dark, overflow = run(model._thermal, timing, model.assets, jax.random.key(717))
    assert not overflow
    assert dark["mask"].any()
    assert np.all(dark["chan"][dark["mask"]] % 2 == 0)


@pytest.mark.slow   # charges compile; dark counts per PR: thermal all-physical test
def test_all_empty_events_still_have_physical_dark_charges(setup):
    model = adapter(setup, capacity(thermal=128))
    collection, timing = inputs(events=4, active=False)
    q, errors = run(model.charges, collection, timing, model.assets, jax.random.key(22))
    assert not any(errors.values())
    assert q["mask"].any()
    assert np.all(q["photon"][q["mask"]] == -1)
    assert np.all((q["time"][q["mask"]] >= 100) & (q["time"][q["mask"]] < 1500))
    assert set(q["chan"][q["mask"]] // CH.CHAN_MULT) == {0, 1, 2, 3}


def _reference_charges(model, rng=None):
    times = np.array([187.5] * 10 + [300.] * 10 + [1490.] * 10 + [310.] * 10)
    labels = np.array([0] * 30 + [CH.CHAN_MULT] * 10)
    charge = np.full(40, model.wcfg.single_pe_reference_charge)
    if rng is not None:
        times = times + rng.random(40) * 6
        charge = charge * (1 + rng.random(40))
    return dict(chan=labels, time=times, charge=charge, mask=np.ones(40, bool))


@pytest.mark.slow   # scatter = dense per PR: the no-charge-capacity scatter test
@pytest.mark.parametrize("noise", [False, True])
def test_scatter_waveform_matches_dense_digis(setup, noise):
    """Opt-in scatter synthesis: same digis as the dense kernel (with and without noise, two events)."""
    cfg = replace(setup[2], feb_phase_width=0., noise=noise)
    dense = adapter(setup, cfg=cfg)
    scatter = adapter(setup, {**capacity(), "waveform": "scatter"}, cfg=cfg)
    assert dense.waveform == "dense" and scatter.waveform == "scatter"
    assert scatter.metadata["waveform"] == "scatter"
    _, timing = inputs(events=2)
    for q in (_reference_charges(dense), _reference_charges(dense, np.random.default_rng(11))):
        a, ea = run(dense.digis, q, timing, dense.assets, jax.random.key(5))
        b, eb = run(scatter.digis, q, timing, scatter.assets, jax.random.key(5))
        assert not any(ea.values()) and not any(eb.values()), (ea, eb)
        assert a["mask"].sum() > 0
        for name in a:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_scatter_waveform_needs_no_charge_capacity_and_validates_option(setup):
    cfg = replace(setup[2], feb_phase_width=0.)
    dense = adapter(setup, cfg=cfg)
    small = adapter(setup, {**capacity(charges_per_cluster=1), "waveform": "scatter"}, cfg=cfg)
    _, timing = inputs(events=2)
    q = _reference_charges(dense)
    a, _ = run(dense.digis, q, timing, dense.assets, jax.random.key(4))
    b, errors = run(small.digis, q, timing, small.assets, jax.random.key(4))
    assert not any(errors.values()), errors
    assert b["mask"].sum() > 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    with pytest.raises(ValueError, match="waveform"):
        adapter(setup, {**capacity(), "waveform": "sparse"})


def test_thermal_counters_hit_only_on_counters_with_steps(setup):
    """thermal_counters="hit" draws thermal noise only on the SiPMs of counters with an in-window step; the default
    draws it on every live SiPM."""
    collection, timing = inputs()
    stepped = {(0, int(b)) for b in collection["columns"]["bar"]}
    geom, cond, cfg, counters, tables, pmap = setup
    noisy = replace(cfg, sipm=replace(cfg.sipm, thermal_rate=1e-2))   # many dark counts per SiPM, both modes
    found = {}
    for mode in ("all", "hit"):
        model = adapter(setup, capacity(thermal=16384, ct1=4096, ct2=1024, ct3=256, thermal_counters=mode), cfg=noisy)
        fn = lambda c, t, a, k, model=model: model.charges(c, t, a, k)
        q, errors = run(fn, collection, timing, model.assets, jax.random.key(29))
        assert not any(errors.values()), [k for k, v in errors.items() if v]
        thermal = q["mask"] & (q["photon"] < 0)
        chan = q["chan"][thermal]
        found[mode] = {(int(c // CH.CHAN_MULT), int(c % CH.CHAN_MULT) // 4) for c in chan}
    assert found["hit"] and found["hit"] <= stepped
    assert found["all"] - stepped
    with pytest.raises(ValueError, match="thermal_counters"):
        adapter(setup, capacity(thermal_counters="some"))


def test_exact_restructure_options_match_default(setup):
    """digi_starts="rows" and pulse_scan="bounded" give the same kept digis, pulses and clusters as the defaults.

    photons 256: the float32 draws (MU2EFS_DEVICE_FLOAT=float32) need more than the helper's 128."""
    collection, timing = inputs()
    timing["event_mask"][1] = False
    outs = {}
    for name, options in (("default", {}), ("exact", {"digi_starts": "rows", "pulse_scan": "bounded"})):
        model = adapter(setup, capacity(waveform="scatter", photons=256, **options))
        out, errors = run(model.apply, collection, timing, model.assets, jax.random.key(23))
        assert not any(errors.values()), [k for k, v in errors.items() if v]
        outs[name] = out
    for table, cols in outs["default"].items():
        mask = cols["mask"]
        assert mask.sum() > 0, table
        np.testing.assert_array_equal(outs["exact"][table]["mask"], mask, err_msg=table)
        for col, value in cols.items():
            np.testing.assert_array_equal(outs["exact"][table][col][mask], value[mask], err_msg=f"{table}/{col}")
    with pytest.raises(ValueError, match="digi_starts"):
        adapter(setup, capacity(digi_starts="sparse"))
