"""Compressed additive-observable checks against actual sampler and full-width event identity."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b
from valhist import shower_uncertainty as U


def setup_features():
    edges = {"y": np.array([0., 1., 2.]), "z": np.array([10., 20.])}
    sector = np.array([0, 1, 2, 3], dtype=np.int64)
    steps = pd.DataFrame({"bar": [0, 0, 1, 2, 3], "edep": [.1, .2, .3, .4, .5],
                          "y0": [-1., 0., 1., 2., 1.5], "z0": [9., 10., 19., 20., 21.]})
    return steps, sector, edges


def test_sparse_features_preserve_sectors_float32_and_root_flows():
    steps, sector, edges = setup_features()
    names, blocks = U.feature_layout(edges)
    x = U.features(steps, np.array([0, 0, 1, 1, 1]), 3, sector, edges).toarray()
    assert x.shape == (3, len(names))
    np.testing.assert_array_equal(x[:, 0], [2, 0, 0])
    np.testing.assert_allclose(x[0, 1], .3)
    np.testing.assert_array_equal(x[0, slice(*blocks["EX/y"])], [1, 1, 0, 0])
    np.testing.assert_array_equal(x[1, slice(*blocks["T2/y"])], [0, 0, 0, 1])
    np.testing.assert_array_equal(x[1, slice(*blocks["M/z"])], [0, 0, 1])
    assert not x[2].any()
    for g in U.GROUPS:
        np.testing.assert_array_equal(x[:, slice(*blocks[g + "/y"])].sum(axis=1), x[:, names.index(g + "/steps")])
    empty = U.features(steps.iloc[:0], np.array([], dtype=np.int64), 5, sector, edges)
    assert empty.shape == (5, len(names)) and empty.nnz == 0


def test_grouped_features_preserve_repeated_template_weight():
    steps, sector, edges = setup_features()
    x = U.features(steps, np.arange(5), 5, sector, edges)
    result = U.grouped_features(x, np.array([0, 0, 1, 1, 1]), 3).toarray()
    direct = U.features(steps, np.array([0, 0, 1, 1, 1]), 3, sector, edges).toarray()
    np.testing.assert_array_equal(result, direct)


def test_event_identity_full_uint32_no_packed_collision_and_missing_refusal():
    maximum = np.iinfo(np.uint32).max
    census = pd.DataFrame({"run": [maximum, maximum - 1, maximum], "subrun": [maximum, maximum, 1],
                           "event": [maximum, maximum, maximum]}, dtype=np.uint32)
    np.testing.assert_array_equal(U.event_rows(census.iloc[[2, 0, 1]], census), [2, 0, 1])
    invalid = census.iloc[:1].assign(event=0)
    with pytest.raises(ValueError, match="outside"):
        U.event_rows(invalid, census)
    with pytest.raises(ValueError, match="duplicate"):
        U.event_rows(census, pd.concat([census, census]))
    with pytest.raises(ValueError, match="full-width"):
        U.event_index(census.astype(float))


def test_cell_semantics_native_edges_abs_sign_and_invalid_ke():
    gen = pd.DataFrame({"pdg": [-11, 11, 22, -13],
                        "e": [.511, .511 + tier1b.KE_EDGES[5], 1e7, 200.]})
    got = U.bin_frame(gen)
    np.testing.assert_array_equal(got.apdg, [11, 11, 22])
    np.testing.assert_array_equal(got.kebin, [0, 5, 17])
    with pytest.raises(ValueError, match="negative"):
        U.bin_frame(pd.DataFrame({"pdg": [2212], "e": [1.]}))


def test_compact_draw_matches_actual_emitted_features_and_rng_state():
    sh = pd.DataFrame({"shid": [0, 1, 2], "apdg": [22, 22, 11], "kebin": [0, 0, 0], "ke": [1., 1., 1.]})
    st = pd.DataFrame({"shid": [0, 1, 1, 2], "bar": [0, 0, 1, 3], "edep": [1., 2., 3., 4.],
                       "y0": [.2, .4, .8, 1.2], "z0": [11., 12., 14., 15.], "dt0": 0., "dt1": 1.})
    lib = {"libshower": sh, "libshowersteps": st,
           "libgen": pd.DataFrame({"apdg": [11, 22], "kebin": [0, 0], "ngen": [5, 10], "nshower": [1, 2]})}
    gen = pd.DataFrame({"run": np.ones(100, dtype=np.uint32), "subrun": np.ones(100, dtype=np.uint32),
                        "event": np.repeat(np.arange(50, dtype=np.uint32), 2), "idx": np.tile([0, 1], 50),
                        "pdg": np.tile([22, -11], 50), "e": np.tile([1., 1.511], 50), "t": 100.})
    sector = np.array([0, 1, 2, 3]); edges = {"y": np.array([0., 1., 2.]), "z": np.array([10., 20.])}
    a, b = np.random.default_rng(123), np.random.default_rng(123)
    picked = U.compact_draw(gen, sh, lib["libgen"], a)
    emitted = tier1b._showers_for_gen(gen, lib, b)
    x = U.features(st, st.shid.to_numpy(), len(sh), sector, edges)
    compressed = np.asarray(x[picked].sum(axis=0)).ravel()
    actual = U.features(emitted, np.zeros(len(emitted), dtype=np.int64), 1, sector, edges).toarray()[0]
    np.testing.assert_array_equal(compressed, actual)
    assert a.bit_generator.state == b.bit_generator.state


def test_bad_support_not_reinterpreted_as_new_cell():
    with pytest.raises(ValueError, match="unaccounted"):
        U.cell_rows(pd.DataFrame({"apdg": [22], "kebin": [2]}), pd.DataFrame({"apdg": [22], "kebin": [1]}))
