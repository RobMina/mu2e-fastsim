"""Unit tests: hand-off from the conversion-electron chain to the field-on tracker starts (beamfastsim.cetracker)."""
import numpy as np
import pandas as pd
import pytest

from beamfastsim import cechain as C
from beamfastsim import cetracker as H
from beamfastsim import geometry as G

AXIS = -3904.0


def _geom():
    """One foil upstream of the starts, IPA beyond the plane and OPA at large radius: vacuum between start and plane."""
    beam = G.BeamGeometry(AXIS, np.array([5000.0]), np.full(1, 21.5), np.full(1, 75.0), np.full(1, 0.0528), 4204.6,
                          25.4, 235.0, 525.0, 300.0, 0.5, 9000.0, 9001.0, 900.0, 4280.0, 8530.0)
    return C.CEGeometry(beam, 900.0, 900.0, 1.0, z_plane=8539.9)


def _starts(n, rng):
    """CE forward and backward (backward ones are lost upstream), large event numbers."""
    ct = np.where(rng.uniform(size=n) < 0.8, rng.uniform(0.5, 0.8, n), -0.8)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    pv = C.P_CE_AL * np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    r = rng.uniform(0.0, 20.0, n)
    a = rng.uniform(0, 2 * np.pi, n)
    z = np.full(n, 5800.0)
    return pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(7), "event": np.arange(n, dtype=np.uint64) + 2**33,
                         "pdg": 11, "x": AXIS + r * np.cos(a), "y": r * np.sin(a), "z": z, "px": pv[:, 0],
                         "py": pv[:, 1], "pz": pv[:, 2], "t": rng.exponential(864.0, n)})


def test_handoff_states_of_1000_ce_equal_the_cechain_plane_states():
    rng = np.random.default_rng(11)
    g = _starts(1000, rng)
    x =g[["x", "y", "z"]].to_numpy()
    pv = g[["px", "py", "pz"]].to_numpy()
    p = np.linalg.norm(pv, axis=1)
    res = C.transport(np.random.default_rng(3), _geom(), [0.0, 0.0, 1.0], x, pv / p[:, None], p, g.t.to_numpy())
    ent = H.handoff(g, res)
    ok = res.fate == C.PLANE
    assert 0 < ok.sum() < len(g)                                  # both outcomes occur in the sample
    assert len(ent) == ok.sum()
    pd.testing.assert_frame_equal(ent[H.KEY], g.loc[ok, H.KEY].reset_index(drop=True))
    assert ent[H.KEY].dtypes.eq(np.uint64).all()
    np.testing.assert_array_equal(ent[["x", "y", "z"]].to_numpy(), res.x[ok])
    np.testing.assert_allclose(ent.z, 8539.9, atol=1e-6)
    pe = ent[["px", "py", "pz"]].to_numpy()
    pmag = np.sqrt(res.T[ok] * (res.T[ok] + 2 * C.eioni.ME))
    np.testing.assert_allclose(np.linalg.norm(pe, axis=1), pmag, rtol=1e-12)
    np.testing.assert_allclose(pe / pmag[:, None], res.u[ok] / np.linalg.norm(res.u[ok], axis=1)[:, None],
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ent.t.to_numpy(), res.t[ok])
    assert (ent.pdg == 11).all() and (ent.sp == 1).all()
    assert (ent.pz > 0).all()                                     # crossing the entry plane downstream


def test_cli_help_has_no_gas_selector(capsys):
    with pytest.raises(SystemExit) as error:
        H.main(["--help"])
    assert error.value.code == 0
    assert "--gas-state" not in capsys.readouterr().out


@pytest.mark.parametrize("state", ["solid", "gas"])
def test_tracker_steps_rejects_a_gas_selector(state):
    with pytest.raises(TypeError, match="unexpected keyword argument 'gas_state'"):
        H.tracker_steps(None, "tracker.gdml", ["DS.header"], gas_state=state)


@pytest.mark.parametrize("state", ["solid", "gas"])
def test_cli_rejects_a_gas_selector(state, capsys, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        H.main(["starts.root", "--outdir", str(out), "--gdml", "tracker.gdml", "--maps", "DS.header",
                "--offline-root", "offline", "--gas-state", state])
    assert error.value.code == 2
    assert f"unrecognized arguments: --gas-state {state}" in capsys.readouterr().err
    assert not out.exists()


def test_handoff_refuses_a_result_of_another_length():
    rng = np.random.default_rng(12)
    g = _starts(10, rng)
    res = C.Result(*(np.zeros(9) if f in ("fate", "T", "t", "path") else np.zeros((9, 3))
                     for f in C.Result._fields[:9]))
    with pytest.raises(ValueError):
        H.handoff(g, res)


def test_starts_take_the_primary_electron_birth_when_gen_has_no_electron(tmp_path):
    from mu2efastsim import tables
    gen = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [5, 6], "pdg": [13, 13], "x": [0.0, 0.0],
                        "y": [0.0, 0.0], "z": [0.0, 0.0], "px": [0.0, 0.0], "py": [0.0, 0.0], "pz": [0.0, 0.0],
                        "t": [0.0, 0.0]})
    part = pd.DataFrame({"run": [1, 1, 1], "subrun": [0, 0, 0], "event": [6, 5, 5], "key": [1, 1, 2],
                         "pdg": [11, 11, 22], "isprim": [1, 1, 0], "ismcprim": [1, 1, 0],
                         "sx": [1.0, 2.0, 3.0], "sy": [0.0, 0.0, 0.0], "sz": [5800.0, 5822.0, 5822.0],
                         "spx": [0.0, 0.0, 1.0], "spy": [0.0, 0.0, 0.0], "spz": [104.97, 104.97, 1.0],
                         "st": [100.0, 200.0, 210.0]})
    f = tmp_path / "t.root"
    tables.write(str(f), {"gen": gen, "particles": part}, as_ttree=True)
    s = H.starts([str(f)])
    assert list(s.event) == [5, 6]
    assert list(s.x) == [2.0, 1.0] and list(s.t) == [200.0, 100.0]
