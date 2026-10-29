"""Synthetic keyed-optical identities, distribution laws and unchanged geometry arithmetic."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma

from crvfastsim.response import lookup, photons as P
from crvfastsim.veto import keyed_optics as K
from test_propagation import _G1, _table

SHA = "0123456789abcdef" * 4


def fixture(n=48, reflector=0, decay=0.0):
    L = _table()
    L.name = f"LookupTable_keyed_{reflector}_{decay}"
    L.reflector, L.fiber_decay_time = abs(reflector), decay
    L.ck_scint = L.ck_fiber = np.array([[0.5, 40.0], [1.0, 400.0]])
    for kind in ("ckscint", "ckfiber"):
        T = L.tables[kind]
        T.prob[:] = 0.15
        T.delays = np.tile(np.array([0, 1, 3, 0, 2], np.uint8), (len(T.prob), 1))
        T.delay_sum = T.delays.sum(1)
        T.emis = np.tile(np.array([1, 0, 3], np.uint8), (len(T.prob), 1))
        T.emis_sum = T.emis.sum(1)
    row = np.arange(n, dtype=np.int64)
    fiber = row % 3 == 0
    st = pd.DataFrame({"run": 20 + row % 2, "subrun": row % 5, "event": row % 7 + 2**21,
                       "source_row": row * 3 + 10, "bar": np.zeros(n, np.int64),
                       "edep": np.where(row % 2, .003, .08), "len": np.full(n, .25),
                       "t0": 400.0 + row, "t1": 400.125 + row,
                       "x0": np.where(fiber, 13.0, 4.0), "y0": np.where(fiber, 0.0, 3.0),
                       "z0": np.linspace(-900, 900, n), "x1": np.where(fiber, 13.0, 4.0),
                       "y1": np.where(fiber, .25, 3.25), "z1": np.linspace(-900, 900, n),
                       "px0": np.zeros(n), "py0": np.full(n, 1e4), "pz0": np.zeros(n), "p1": np.full(n, 1e4)})
    cfg = P.ResponseConfig(sectors={"EX": P.SectorConfig("keyed", reflector, 1000.0)}, photon_yield_scale=1.0)
    return st, np.full(n, 13), _G1, cfg, {"keyed": L}, np.array([.01, -.02, .03, -.04])


def run(args, **kwargs):
    return K.make_photons(*args, input_sha256=kwargs.pop("input_sha256", SHA), seed=kwargs.pop("seed", 717), **kwargs)


def canonical(q):
    assert not q.duplicated(list(K.KEYS)).any()
    return q.sort_values(list(K.KEYS)).reset_index(drop=True)


def equal(a, b):
    pd.testing.assert_frame_equal(canonical(a), canonical(b), check_exact=True)


@pytest.mark.parametrize("max_sub", [1, 3, 17, 200000])
def test_internal_substep_chunk_invariance(max_sub):
    args = fixture()
    baseline = run(args)
    assert len(baseline) > 100 and set(baseline.photon_type) == {0, 1, 2}
    assert set(baseline.chan) == {0, 1, 2, 3}
    equal(baseline, run(args, max_substeps=max_sub))


def test_permutation_subset_and_external_partitions():
    st, pdg, *rest = args = fixture()
    base = run(args)
    order = np.random.default_rng(91).permutation(len(st))
    equal(base, run((st.iloc[order], pdg[order], *rest)))
    take = order[:17]
    equal(base[base.source_row.isin(st.iloc[take].source_row)], run((st.iloc[take], pdg[take], *rest)))
    chunks = [run((st.iloc[s], pdg[s], *rest), max_substeps=7) for s in np.array_split(order, 7)]
    equal(base, pd.concat(chunks, ignore_index=True))


@pytest.mark.parametrize("reflector", [-1, 0, 1])
def test_reflectors_and_nonzero_decay_invariance(reflector):
    st, pdg, *rest = args = fixture(reflector=reflector, decay=1.7)
    a = run(args)
    equal(a, run((st.iloc[::-1], pdg[::-1], *rest), max_substeps=7))
    assert set(a.chan) == ({1, 3} if reflector < 0 else {0, 2} if reflector else {0, 1, 2, 3})


def test_raw_folded_same_keys_enormous_times_actual_arithmetic():
    st, *rest = fixture()
    raw = st.assign(t0=2.6081538852432572e16, t1=2.6081538852432572e16)
    folded = st.assign(t0=567., t1=567.)
    a, b = canonical(run((raw, *rest))), canonical(run((folded, *rest)))
    pd.testing.assert_frame_equal(a[list(K.KEYS)], b[list(K.KEYS)])
    assert np.any(np.mod(a.time.to_numpy(), 4) == 0)
    assert np.any(a.time.to_numpy() - 2.6081538852432572e16 != b.time.to_numpy() - 567.)


def test_full_identity_and_seed_domains():
    draws = K.Draws(SHA, 717)
    base = tuple(np.arange(32, dtype=np.int64) for _ in range(8))
    u = draws.uniform("delay", *base)
    assert np.all((u >= 0) & (u < 1))
    for i in range(len(base)):
        keys = list(base)
        keys[i] = keys[i] + 1
        assert np.all(u != draws.uniform("delay", *keys))
    for i in range(4):
        changed = SHA[:16*i] + "fedcba9876543210" + SHA[16*(i+1):]
        assert np.all(u != K.Draws(changed, 717).uniform("delay", *base))
    assert np.all(u != K.Draws(SHA, 718).uniform("delay", *base))
    assert np.all(u != draws.uniform("emission", *base))
    huge = np.array([2**53, 2**53 + 1, 2**63 - 1], np.int64)
    assert len(np.unique(draws.uniform("large-identity", huge))) == 3


def test_seed_change_new_draw_not_new_exposure():
    args = fixture()
    a, b = run(args, seed=717), run(args, seed=718)
    assert set(a.draw_seed) == {717} and set(b.draw_seed) == {718}
    assert not a.drop(columns="draw_seed").equals(b.drop(columns="draw_seed"))
    assert a.source_row.isin(args[0].source_row).all() and b.source_row.isin(args[0].source_row).all()


@pytest.mark.parametrize("mode", ["empty", "nan", "outside"])
def test_empty_domains_and_nan_skip(mode):
    st, pdg, *rest = fixture()
    if mode == "empty":
        st, pdg = st.iloc[:0], pdg[:0]
    elif mode == "nan":
        st = st.assign(t0=np.nan)
    else:
        st = st.assign(x0=1e5, x1=1e5)
    out = run((st, pdg, *rest))
    assert len(out) == 0 and list(out) == [*K.KEYS, "time"]
    assert out.time.dtype == np.float64 and out.draw_seed.dtype == np.uint64


def test_partial_nan_skip_does_not_advance_other_draws():
    st, pdg, *rest = args = fixture()
    changed = st.copy()
    changed.loc[::3, "t1"] = np.nan
    a = run(args)
    equal(a[~a.source_row.isin(changed.loc[::3, "source_row"])], run((changed, pdg, *rest)))


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.0])
def test_bad_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        run(fixture(), seed=seed)


@pytest.mark.parametrize("digest", ["a", "Z" * 64, "A" * 64, None])
def test_bad_digest(digest):
    with pytest.raises(ValueError, match="SHA256"):
        run(fixture(), input_sha256=digest)


@pytest.mark.parametrize("column,value", [("source_row", -1), ("event", 1.0), ("bar", 1),
                                           ("t0", np.inf), ("t1", -np.inf), ("edep", -1.),
                                           ("len", np.nan), ("p1", -1.), ("x0", np.inf)])
def test_bad_step_columns(column, value):
    st, *rest = fixture()
    st[column] = value
    with pytest.raises(ValueError):
        run((st, *rest))


def test_duplicate_source_row_and_unknown_pdg():
    st, pdg, *rest = fixture()
    st.loc[1, "source_row"] = st.loc[0, "source_row"]
    with pytest.raises(ValueError, match="duplicate"):
        run((st, pdg, *rest))
    st, pdg, *rest = fixture()
    pdg[0] = 999
    with pytest.raises(ValueError, match="unknown"):
        run((st, pdg, *rest))


@pytest.mark.parametrize("option,value", [("max_substeps", 0), ("max_substeps", True),
                                          ("max_photons", -1), ("max_photons", 1)])
def test_resource_limits_fail_not_truncate(option, value):
    with pytest.raises(ValueError):
        run(fixture(), **{option: value})


@pytest.mark.parametrize("mode", ["deviation", "probability", "sum", "yield", "decay"])
def test_invalid_conditions(mode):
    st, pdg, geom, cfg, tables, dev = fixture()
    if mode == "deviation":
        dev[0] = -2
    elif mode == "probability":
        tables["keyed"].tables["scint"].prob[0] = np.nan
    elif mode == "sum":
        tables["keyed"].tables["scint"].delay_sum[0] += 1
    elif mode == "yield":
        cfg.sectors["EX"].yield_per_mev = -1
    elif mode == "decay":
        tables["keyed"].fiber_decay_time = -1
    with pytest.raises(ValueError):
        run((st, pdg, geom, cfg, tables, dev))


@pytest.mark.parametrize("mean", [0., .2, 3., 10., 10.0001, 100.])
def test_count_moments_and_quantiles(mean):
    n = 20000
    draws = K.Draws(SHA, 1717)
    values = K.count_draws(np.full(n, mean), np.ones(n, np.int64), draws, np.arange(n))
    if mean == 0:
        assert not values.any()
        return
    assert abs(values.mean() - mean) < 6 * np.sqrt(mean / n) + .003
    assert abs(values.var() - mean) < .10 * mean + .15
    if mean <= 10:
        expected = np.exp(-mean)
        assert abs(np.mean(values == 0) - expected) < 6 * np.sqrt(expected * (1 - expected) / n) + 1/n


@pytest.mark.parametrize("n,p", [(0, .3), (30, 0.), (30, 1.), (1, .3), (40, .07), (800, .8)])
def test_binomial_law(n, p):
    size = 20000
    draws = K.Draws(SHA, 1717)
    values = K.arrival_counts(np.full(size, n), np.full(size, p), draws, np.arange(size))
    mean, var = n*p, n*p*(1-p)
    assert np.all((values >= 0) & (values <= n))
    assert abs(values.mean() - mean) < 6*np.sqrt(var/size) + 1e-12
    assert abs(values.var() - var) < .1*var + 1e-12


def test_quantile_clipping_and_exact_rounding(monkeypatch):
    d = K.Draws(SHA, 1)
    monkeypatch.setattr(d, "uniform", lambda domain, *keys: np.array([0., 1.]))
    np.testing.assert_array_equal(d.quantile("test", np.arange(2)), [2**-53, 1 - 2**-53])
    monkeypatch.setattr(d, "normal", lambda domain, *keys: np.zeros(2))
    np.testing.assert_array_equal(K.count_draws(np.array([15., 25.]), np.array([10, 10]), d, np.arange(2)), [2, 2])


def test_weighted_delay_distribution_and_zero_sum():
    n = 20000
    w = np.array([[0, 1, 3, 0, 2]], np.uint8)
    u = K.Draws(SHA, 8).uniform("delay", np.arange(n))
    out = K._weighted_index(w, w.sum(1), np.zeros(n, int), u)
    freq = np.bincount(out, minlength=5) / n
    np.testing.assert_allclose(freq, w[0] / 6, atol=.015)
    assert K._weighted_index(w, w.sum(1), np.array([0]), np.array([0.]))[0] == 0
    assert K._weighted_index(w*0, np.array([0]), np.array([0]), np.array([.7]))[0] == 0


@pytest.mark.parametrize("reflector", [-1, 0, 1])
def test_deterministic_production_geometry_and_time_parity(monkeypatch, reflector):
    st, pdg, geom, cfg, tables, dev = fixture(reflector=reflector, decay=1.5)
    cfg = replace(cfg, speed_ratio={"EX": 1.2}, atten_ratio={"EX": .8})
    class Fixed:
        normal = staticmethod(lambda loc, scale: np.asarray(loc))
        poisson = staticmethod(lambda lam: np.asarray(lam))
        binomial = staticmethod(lambda n, p: np.floor(n*p).astype(np.int64))
        random = staticmethod(lambda n: np.full(n, .375))
        gamma = staticmethod(lambda shape, scale: gamma.ppf(.375, shape, scale=scale))
    monkeypatch.setattr(K.Draws, "normal", lambda self, domain, *keys: np.zeros(np.broadcast_arrays(*keys)[0].shape))
    monkeypatch.setattr(K.Draws, "uniform", lambda self, domain, *keys: np.full(np.broadcast_arrays(*keys)[0].shape, .375))
    monkeypatch.setattr(K, "count_draws", lambda mean, nsub, draws, *keys: np.maximum(np.rint(mean/nsub), 0).astype(np.int64))
    monkeypatch.setattr(K, "arrival_counts", lambda n, p, draws, *keys: Fixed.binomial(n, p))
    si, ch, time = P.make_photons(st, pdg, geom, cfg, tables, dev, Fixed())
    expected = pd.DataFrame({"source_row": st.source_row.to_numpy()[si], "chan": ch, "time": time})
    actual = run((st, pdg, geom, cfg, tables, dev), max_substeps=7)[list(expected)]
    cols = list(expected)
    pd.testing.assert_frame_equal(expected.sort_values(cols).reset_index(drop=True),
                                  actual.sort_values(cols).reset_index(drop=True), check_exact=True)
    assert len(actual) > 100


@pytest.mark.parametrize("name", ["speed_ratio", "atten_ratio"])
@pytest.mark.parametrize("value", [np.inf, np.nan, 0., -1.])
def test_bad_propagation_even_empty(name, value):
    st, pdg, geom, cfg, tables, dev = fixture()
    setattr(cfg, name, {"EX": value})
    with pytest.raises(ValueError, match="propagation"):
        run((st.iloc[:0], pdg[:0], geom, cfg, tables, dev))


@pytest.mark.parametrize("mode", ["edge_nan", "edge_order", "radius", "shape", "weights", "overflow"])
def test_bad_lookup_contract(mode):
    st, pdg, geom, cfg, tables, dev = fixture()
    L = tables["keyed"]
    T = L.tables["scint"]
    if mode == "edge_nan":
        L.edges["z"][0] = np.nan
    elif mode == "edge_order":
        L.edges["z"][1] = L.edges["z"][0]
    elif mode == "radius":
        L.hole_radius_x = 0.
    elif mode == "shape":
        T.prob = T.prob[:-1]
    elif mode == "weights":
        T.delays = T.delays.astype(np.int64)
    elif mode == "overflow":
        T.delays = np.full((len(T.prob), 2), 2**62, np.int64)
        T.delay_sum = T.delays.sum(1)
    with pytest.raises(ValueError, match="lookup"):
        run((st, pdg, geom, cfg, tables, dev))


@pytest.mark.parametrize("pdg", [2**64 - 13, 1000000000, 1000020010, 1100020040])
def test_invalid_pdg_encoding(pdg):
    st, _, *rest = fixture()
    with pytest.raises(ValueError, match="PDG"):
        run((st, np.full(len(st), pdg, np.uint64), *rest))


def test_work_guard_before_zero_signal_expansion(monkeypatch):
    st, pdg, *rest = fixture()
    st = st.assign(x1=1e6, edep=0., len=0.)
    def forbidden(*args, **kwargs):
        pytest.fail("work guard must fire before row generation")
    monkeypatch.setattr(K, "_rows", forbidden)
    with pytest.raises(ValueError, match="total substep work limit"):
        run((st, pdg, *rest), max_total_substeps=1000)


def test_content_bound_propagation_cache():
    args = list(fixture())
    args[3] = replace(args[3], speed_ratio={"EX": 1.3}, atten_ratio={"EX": .7})
    L = args[4]["keyed"]
    P._TABLE_PROP[(L.name, L.half_length)] = (12345., .00001)
    a = run(args)
    del P._TABLE_PROP[(L.name, L.half_length)]
    equal(a, run(args))
    changed = _table(inv_v=1/90., att=3000.)
    changed.name = L.name
    args[4] = {"keyed": changed}
    b = run(args)
    P._TABLE_PROP[(L.name, L.half_length)] = (98765., .01)
    equal(b, run(args))
    del P._TABLE_PROP[(L.name, L.half_length)]
    assert not a.equals(b)


def test_large_integer_keys_survive_entire_pipeline():
    st, pdg, *rest = fixture(n=6)
    st["run"] = np.int64(2**53 + 1)
    st["subrun"] = np.int64(2**53 + 3)
    st["event"] = np.int64(2**63 - 1)
    st["source_row"] = np.arange(6, dtype=np.int64) + (2**63 - 6)
    base = run((st, pdg, *rest), seed=2**64 - 1)
    assert len(base) > 0 and base.draw_seed.dtype == np.uint64
    assert np.all(base.draw_seed == np.uint64(2**64 - 1))
    equal(base, run((st.iloc[::-1], pdg[::-1], *rest), seed=2**64 - 1, max_substeps=1))
    for name in K.IDS:
        assert base[name].dtype == np.int64
        assert base[name].isin(st[name]).all()


def test_mixed_sector_table_reflector_partition_invariance():
    from types import SimpleNamespace
    st, pdg, _, _, _, _ = fixture(n=90)
    geo = SimpleNamespace(nbars=3, sector_names=["A", "B", "C"], sector=np.arange(3),
                          module=np.zeros(3, int), center=np.zeros((3, 3)),
                          local_to_world=np.tile([1, 0, 2], (3, 1)))
    st["bar"] = np.arange(len(st)) % 3
    st["y1"] = st.y0 + np.linspace(.01, 1., len(st))
    st["len"] = st.y1 - st.y0
    tables = {name: fixture(reflector=r, decay=1.2)[4]["keyed"] for name, r in zip("abc", [0, -1, 1])}
    cfg = P.ResponseConfig(sectors={name: P.SectorConfig(table, r, 1000 + 500*i)
                           for i, (name, table, r) in enumerate(zip("ABC", "abc", [0, -1, 1]))},
                           speed_ratio={"B": 1.2}, atten_ratio={"A": .8})
    rest = (geo, cfg, tables, np.linspace(-.1, .1, 12))
    a = run((st, pdg, *rest))
    assert set(a.chan // 4) == {0, 1, 2}
    order = np.random.default_rng(71).permutation(len(st))
    equal(a, pd.concat([run((st.iloc[s], pdg[s], *rest), max_substeps=11)
                        for s in np.array_split(order, 9)], ignore_index=True))
    take = order[:31]
    equal(a[a.source_row.isin(st.iloc[take].source_row)], run((st.iloc[take], pdg[take], *rest), max_substeps=3))


@pytest.mark.parametrize("mean,nsub", [(2., 4), (10., 7), (12., 3), (100., 50)])
def test_count_law_after_division_against_sequential_numpy(mean, nsub):
    n = 20000
    rng = np.random.default_rng(901)
    expected = P._per_substep(np.full(n, mean), np.full(n, nsub), rng)
    actual = K.count_draws(np.full(n, mean), np.full(n, nsub), K.Draws(SHA, 901), np.arange(n))
    edges = np.arange(max(expected.max(), actual.max()) + 2) - .5
    pa = np.histogram(actual, edges)[0] / n
    pe = np.histogram(expected, edges)[0] / n
    assert np.max(abs(pa - pe)) < .025


def test_emission_gamma_law_and_batch_invariance():
    n = 20000
    L = fixture(decay=1.5)[4]["keyed"]
    T = L.tables["ckscint"]
    T.delays = np.ones((len(T.prob), 1), np.uint8)
    T.delay_sum = T.delays.sum(1)
    T.emis = np.tile(np.array([1, 0, 3], np.uint8), (len(T.prob), 1))
    T.emis_sum = T.emis.sum(1)
    bins, base = np.zeros(n, int), np.zeros(n)
    keys = (np.arange(n),)
    d = K.Draws(SHA, 901)
    a = K._arrival_times(L, T, bins, base, d, keys, 8192)
    np.testing.assert_array_equal(a, K._arrival_times(L, T, bins, base, d, keys, 317))
    assert abs(np.mean(a == 0) - .25) < .015
    assert abs(a.mean() - 2.25) < .07
    assert abs(a[a > 0].var() - 4.5) < .25


def test_integrated_photon_counts_against_production_law():
    st, pdg, geom, cfg, tables, dev = fixture(n=6000)
    st = st.assign(edep=.05)
    cfg = replace(cfg, photon_yield_scale=.8, speed_ratio={"EX": 1.2}, atten_ratio={"EX": .7})
    actual = run((st, pdg, geom, cfg, tables, dev), max_substeps=1000)
    si, ch, time = P.make_photons(st, pdg, geom, cfg, tables, dev, np.random.default_rng(1947))
    assert len(actual) > 10000
    for channel in range(4):
        a, b = (actual.chan == channel).sum(), np.sum(ch == channel)
        assert abs(a - b) < 7 * np.sqrt(a + b)
    assert abs(actual.time.mean() - time.mean()) < 7 * np.sqrt(actual.time.var()/len(actual) + time.var()/len(time))


def test_probability_saturation_matches_unit_probability(monkeypatch):
    st, pdg, geom, cfg, tables, dev = fixture()
    L = tables["keyed"]
    for T in L.tables.values():
        T.prob[:] = 1.
    monkeypatch.setattr(P, "table_propagation", lambda L: (1/160., 6000.))
    a = run((st, pdg, geom, replace(cfg, photon_yield_scale=3.), tables, dev*0))
    b = run((st, pdg, geom, replace(cfg, photon_yield_scale=1.), tables, dev*0))
    equal(a, b)


def test_unsigned_large_identity_empty_partitions_preserve_int64():
    st, pdg, *rest = fixture(n=6)
    for name in K.IDS:
        st[name] = np.arange(6, dtype=np.uint64) + np.uint64(2**63 - 6)
    a = run((st, pdg, *rest))
    empty = run((st.iloc[:0], pdg[:0], *rest))
    quiet = run((st.assign(edep=0., len=0.), pdg, *rest))
    assert len(quiet) == 0
    joined = pd.concat([empty, a, quiet, empty], ignore_index=True)
    equal(a, joined)
    assert all(joined[name].dtype == np.int64 for name in K.IDS)
    assert joined.source_row.nunique() == a.source_row.nunique() > 1
