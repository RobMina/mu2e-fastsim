"""Frozen-entry mean-loss and observational replay regressions."""
import jax
import numpy as np
import pytest

from beamfastsim import layerstudy as ls, transport
from test_transport import _geom, _uniform_map


def test_constant_stopping_exact(monkeypatch):
    monkeypatch.setattr(ls.eloss, "dedx_restricted", lambda mat, p, tcut: np.full_like(p, 2.0))
    e, loss = ls.mean_history([1., 2., 3.], [.5, .25, 0.], 4)
    np.testing.assert_allclose(e[-1], [0., 1.5, 3.])
    table = ls.range_table(10.)
    out, stop = ls.range_out([1., 2., 3.], [.5, .25, 0.], table)
    np.testing.assert_allclose(out, [0., 1.5, 3.])
    assert stop.tolist() == [True, False, False]
    np.testing.assert_allclose(table[1], (table[0] - .1) / 2)


def test_bethe_slicing_approaches_same_law_range():
    energy, length = np.array([1., 2., 5.]), np.array([.01, .05, .1])
    reference, stops = ls.range_out(energy, length, ls.range_table(10.))
    coarse, _ = ls.mean_history(energy, length, 4)
    fine, _ = ls.mean_history(energy, length, 128)
    assert not stops.any()
    assert np.all(fine[-1] > reference)
    assert np.all(fine[-1] - reference < coarse[-1] - reference)


@pytest.mark.parametrize("energy,length,slices", [([1.], [-1.], 4), ([np.nan], [1.], 4),
                                                  ([1.], [1.], 0), ([1.], [1.], 1.5)])
def test_bad_mean_inputs(energy, length, slices):
    with pytest.raises(ValueError):
        ls.mean_history(energy, length, slices)


def test_observation_preserves_outputs_and_compacted_rows():
    geom = _geom(nfoil=3, ht=.05, z0=5000.)
    prm = transport.Params(ds=5., max_steps=13, msc=True, decay=False)
    fm = _uniform_map(0.)
    state = transport.initial_state(jax.random.PRNGKey(1),
                                    [[-3904., 0., 4999.]] * 4,
                                    [[0., 0., 5.], [0., 0., 30.], [0., 0., 60.], [0., 0., 70.]],
                                    np.zeros(4), -np.ones(4), prm)
    key = jax.random.PRNGKey(2)
    ordinary, _ = transport.transport_all(fm, geom, prm, key, state, chunk=2, min_bucket=2)
    observed, _, history = ls.capture_transport(fm, geom, prm, key, state, chunk=2, min_bucket=2)
    ls.verify_replay(observed, ordinary._asdict())
    assert np.all(history["row"] >= 0)
    assert np.sum(history["row"] == 0) == 1
    assert set(history["row"][history["chunk"] > 0]) == {1, 2, 3}
    assert history["slice_loss_MeV"].shape[1] == 4
    energy = ls.kinetic(history["slice_p_in"])
    np.testing.assert_allclose(energy[:, 1:], np.maximum(energy[:, :-1] - history["slice_loss_MeV"][:, :-1], 0),
                               atol=5e-14)
    report, frozen = ls.frozen_study(history, {"foil": np.array([0, -1, -1, -1])}, geom)
    assert report["entrants"] == 4 and report["prior_material"] == 0
    assert frozen["row"].tolist() == [0, 1, 2, 3]
    from dataclasses import replace
    with pytest.raises(ValueError, match="bracket"):
        ls.frozen_study(history, {"foil": np.array([0, -1, -1, -1])}, replace(geom, foil_ht=np.full(3, 5.)))
    from beamfastsim.layerreport import inspect
    g = {"foil": np.array([0, -1, -1, -1]), "evt": np.arange(4), "run": np.ones(4),
         "subrun": np.zeros(4), "trk": np.arange(4)}
    detail = inspect(history, frozen, g, {s: ordinary._asdict() for s in (4, 8, 16, 32)})
    assert detail["range_grid_stop_disagreements"] == 0
    assert detail["entrants"] == 4
    corrupted = ordinary._asdict()
    corrupted["p"] = ordinary.p + 1
    with pytest.raises(ValueError, match="differs"):
        ls.verify_replay(observed, corrupted)


def test_exact_threshold_matches_strict_stop_semantics():
    table = ls.range_table(10.)
    out, stop = ls.range_out([.1, table[0][100]], [0., table[1][100]], table)
    np.testing.assert_allclose(out, [.1, .1])
    assert not stop.any()
