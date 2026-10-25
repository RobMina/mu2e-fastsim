"""Deterministic stepping and cap regressions, independent of loss fluctuations."""
import jax
import numpy as np
import pytest

from beamfastsim import transport
from test_transport import _geom, _uniform_map


def _state(prm, z=4000.0, direction=1.0):
    return transport.initial_state(
        jax.random.PRNGKey(0), [[-3904.0, 0.0, z]], [[0.0, 0.0, direction * 40.0]],
        [0.0], [-1.0], prm)


@pytest.mark.parametrize("cap,chunk", [(7, 4), (3, 8), (8, 4)])
def test_chunk_respects_exact_step_cap(cap, chunk):
    prm = transport.Params(ds=2.0, max_steps=cap, msc=False, decay=False)
    result, lane_steps = transport.transport_all(
        _uniform_map(0.0), _geom(nfoil=1, z0=1e6), prm, jax.random.PRNGKey(1),
        _state(prm, z=5000.0), chunk=chunk, min_bucket=1)
    assert lane_steps == cap
    assert result.fate[0] == 9
    assert result.x[0, 2] == 5000.0 + 2.0 * cap


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_exact_plane_endpoint_counts_once(monkeypatch, direction):
    def no_loss(key, mat, x0, p, u, length, active, n_slices, tcut, msc):
        return p, u, active & False

    monkeypatch.setattr(transport, "_layer", no_loss)
    prm = transport.Params(ds=2.0, max_steps=2, msc=False, decay=False)
    run = transport.make_stepper(_uniform_map(0.0), _geom(nfoil=1, z0=5000.0), prm)
    first, _ = run(jax.random.PRNGKey(1), _state(prm, z=5000.0 - 2.0 * direction, direction=direction))
    assert first.nfoil[0] == 1
    # One step must own arrival at the plane, not the following departure.
    one = transport.make_stepper(_uniform_map(0.0), _geom(nfoil=1, z0=5000.0), prm._replace(max_steps=1))
    arrival, _ = one(jax.random.PRNGKey(1), _state(prm, z=5000.0 - 2.0 * direction, direction=direction))
    assert arrival.nfoil[0] == 1


@pytest.mark.parametrize("chunk", [0, -1, 1.5])
def test_invalid_chunk_rejected(chunk):
    prm = transport.Params(msc=False, decay=False)
    with pytest.raises(ValueError, match="chunk"):
        transport.transport_all(_uniform_map(), _geom(nfoil=1), prm, jax.random.PRNGKey(1),
                                _state(prm), chunk=chunk)


@pytest.mark.parametrize("ds", [2.0, 5.0])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_foil_stop_time_matches_crossing_position(monkeypatch, ds, direction):
    def stop_at_layer(key, mat, x0, p, u, length, active, n_slices, tcut, msc):
        return p * ~active, u, active

    monkeypatch.setattr(transport, "_layer", stop_at_layer)
    prm = transport.Params(ds=ds, max_steps=1, msc=False, decay=False)
    g = _geom(nfoil=1, z0=5000.0)
    run = transport.make_stepper(_uniform_map(0.0), g, prm)
    initial = _state(prm, z=5000.0 - direction, direction=direction)
    result, _ = run(jax.random.PRNGKey(1), initial)
    energy = np.sqrt(40.0 ** 2 + transport.eloss.MMU ** 2)
    assert result.fate[0] == 1
    assert result.x[0, 2] == 5000.0
    assert float(result.t[0]) == pytest.approx(energy / (40.0 * transport.C_MM_NS))
    assert float(result.tau[0]) == pytest.approx(transport.eloss.MMU / (40.0 * transport.C_MM_NS))
