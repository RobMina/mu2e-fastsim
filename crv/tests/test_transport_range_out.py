"""Stage-C range_out: absorber range table, delta kinetic loss, bar sub-steps, the run-configuration default and the
JAX refusal (docs/transport-range-out.md)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import runconfig, transport as T
from mu2efastsim import eloss as E, materials as M


def _geometry():
    return SimpleNamespace(nbars=3, center=np.array([[0., 40., 0.], [0., 20., 0.], [0., 0., 0.]]),
                           halflen=np.tile([100., 5., 100.], (3, 1)))


def _gen(p):
    p = np.asarray(p, float)
    return pd.DataFrame(dict(run=1, subrun=1, event=np.arange(len(p)), idx=0, pdg=13,
                             x=0., y=60., z=0., px=0., py=-p, pz=0., t=0.))


def _run(monkeypatch, p, **kw):
    monkeypatch.setattr(T, "world_loss", lambda rng, o, d, p, world, edges:
                        (p, np.zeros(len(p)), np.ones(len(p), bool), None, None))
    monkeypatch.setattr(T, "msc_to_plane", lambda rng, o, d, p, *args, **k: (o, d, np.zeros(len(p))))
    return T.transport(_gen(p), _geometry(), np.random.default_rng(3), stage="C", world=[], **kw)[0]


def test_range_table_ranges_out_and_agrees_far_from_it():
    al = M.BY_NAME["G4_Al"]
    p = np.array([30., 60., 70., 1000.])
    tin = np.hypot(p, E.MMU) - E.MMU
    loss = T._range_loss(al, p, 9.525)
    assert loss[:2] == pytest.approx(tin[:2])                 # 4.2 and 15.8 MeV range out in 9.5 mm Al
    assert loss[2] < tin[2] and loss[2] > E.dedx_restricted(al, p[2:3], 1e9)[0] * 9.525   # dE/dx rises along it
    assert loss[3] == pytest.approx(E.dedx_restricted(al, p[3:], 1e9)[0] * 9.525, rel=1e-3)
    tg, rg = T._RANGE_TABLES["G4_Al"]
    assert np.interp(15.85, tg, rg) == pytest.approx(7.7, abs=0.2)   # CSDA range of a 60 MeV/c muon [mm]
    assert np.all(np.diff(rg) > 0) and rg[0] == 0.0


def test_deltas_take_kinetic_energy_but_not_the_deposit(monkeypatch):
    p = [3000.0] * 200
    a = _run(monkeypatch, p)
    b = _run(monkeypatch, p, range_out=True)
    assert len(a) == len(b)
    first = (a.bar == 0).to_numpy()                           # same main stream: the first bar's deposit is unchanged,
    assert np.array_equal(a.edep[first].to_numpy(), b.edep[first].to_numpy())
    assert b.edep.to_numpy() == pytest.approx(a.edep.to_numpy(), rel=1e-4)   # later bars at a slightly lower momentum
    drop_a = np.hypot(a.px0, a.py0) - a.p1
    drop_b = np.hypot(b.px0, b.py0) - b.p1
    assert drop_b.mean() > drop_a.mean() * 1.1                # deltas above the cut leave the muon as well


def test_slow_muon_stops_inside_a_bar_with_substeps(monkeypatch):
    st = _run(monkeypatch, [50.0] * 400, range_out=True)      # 11 MeV: stops in the 2nd bar
    stopped = st[st.p1 < 1.0]
    assert len(stopped) > 0
    assert (stopped.len < 10.0).all() and (stopped.len > 0).all()   # the path ends inside the 10 mm bar
    tin = np.hypot(np.hypot(stopped.px0, stopped.py0), E.MMU) - E.MMU
    assert (stopped.edep <= tin + 1e-9).all()


def test_off_by_default_and_extracted_default_on():
    import inspect
    assert inspect.signature(T.transport).parameters["range_out"].default is False
    assert runconfig.get("extracted_v04").range_out
    assert not any(runconfig.get(n).range_out for n in ("run1a_v01", "run1b_v01", "run2_v10"))


def test_jax_backend_refuses_range_out():
    with pytest.raises(ValueError, match="legacy-transport"):
        T.transport(_gen([1000.]), _geometry(), np.random.default_rng(1), stage="C", world=[],
                    transport_backend="jax", range_out=True)
