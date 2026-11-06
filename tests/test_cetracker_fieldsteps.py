"""Umbrella integration: cechain hand-off rows run through the field-on tracker (tracker fieldsteps) and the injector table."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("trkfastsim")
from beamfastsim import cechain as C                                      # noqa: E402
from beamfastsim import cetracker as H                                    # noqa: E402
from beamfastsim import geometry as G                                     # noqa: E402
from mu2efastsim import tables                                            # noqa: E402
from trkfastsim import geometry                                           # noqa: E402
from trkfastsim.sim import fieldsteps as fs                               # noqa: E402

AXIS = -3904.0
ORIGIN = np.array([AXIS, 0.0, 10171.0])


def _straws():
    """A plane of 60 x-axis straws 900 mm long at z = 9000 mm (Mu2e frame), y pitch 6 mm about the DS axis."""
    n = 60
    y = np.arange(n) * 6.0 - 177.0
    zeros = np.zeros(n, int)
    return geometry.StrawGeometry(
        plane=zeros, panel=zeros, straw=np.arange(n), mid=np.c_[np.full(n, AXIS), y, np.full(n, 9000.0)],
        dir=np.tile([1.0, 0, 0], (n, 1)), halflen=np.full(n, 450.0), radius=2.5085, panel_R=np.eye(3)[None],
        panel_t=np.zeros((1, 3)), plane_R=np.eye(3)[None], plane_t=np.zeros((1, 3)), wall_rmax=2.5235, metal_t=0.0001,
        wire_r=0.0125, wire_plate_t=0.0002)


def _no_loss(rng, m, p, ln, ms):
    return np.zeros(len(p))


def test_tracker_steps_uses_the_tracker_gas_material(monkeypatch):
    from dataclasses import replace
    from mu2efastsim import ehelix, world
    from trkfastsim.sim import steps as st

    geom = _straws()
    n = 8
    start = pd.DataFrame({"run": 1, "subrun": 0, "event": np.arange(n, dtype=np.uint64) + 2**33,
                          "sp": 1, "pdg": 11, "x": AXIS, "y": geom.mid[:n, 1] + 0.5, "z": 8980.0,
                          "px": 10.0, "py": 0.0, "pz": 104.0, "t": 100.0})
    map_data = (ORIGIN - 2000.0, np.full(3, 4000.0), np.full(3, 2),
                np.broadcast_to([0.0, 0.0, 1.0], (2, 2, 2, 3)))
    monkeypatch.setattr(geometry, "from_gdml", lambda path: geom)
    monkeypatch.setattr(world, "load_tubes", lambda *args, **kwargs: [])
    monkeypatch.setattr(ehelix, "read_g4bl", lambda path: map_data)
    actual, final = H.tracker_steps(start, "tracker.gdml", ["DS.header"], seed=23)

    field = ehelix.grid_field(*map_data)
    env = fs.Envelope(geom, axis_xy=ORIGIN[:2])
    support = fs.interior("tracker.gdml", geom, env, origin=ORIGIN)

    def transport(gas):
        return fs.transport(np.random.default_rng(np.random.SeedSequence([23, 1])), start, geom, field, support,
                            chord_mm=10.0, sagitta_mm=0.02, envelope=env, origin=ORIGIN,
                            bnom=ehelix.field_at(field, ORIGIN[None])[0], loss=fs.electron_loss, tail=False, gas=gas)

    assert st.STRAW_GAS.state == "gas"
    expected, expected_final = transport(st.STRAW_GAS)
    assert len(actual) >= n and (actual.eion > 0).all()
    assert len(final) == n and (final.status == 0).all()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(final, expected_final, check_exact=True)
    solid, _ = transport(replace(st.STRAW_GAS, state="solid"))
    assert not np.array_equal(actual.eion.to_numpy(), solid.eion.to_numpy())


def test_tracker_steps_refuses_a_first_map_without_the_ds_field(monkeypatch):
    from mu2efastsim import ehelix, world
    geom = _straws()
    weak = (ORIGIN - 2000.0, np.full(3, 4000.0), np.full(3, 2), np.broadcast_to([0.0, 0.0, 0.1], (2, 2, 2, 3)))
    monkeypatch.setattr(geometry, "from_gdml", lambda path: geom)
    monkeypatch.setattr(world, "load_tubes", lambda *args, **kwargs: [])
    monkeypatch.setattr(ehelix, "read_g4bl", lambda path: weak)
    start = pd.DataFrame({"run": 1, "subrun": 0, "event": np.arange(2, dtype=np.uint64), "sp": 1, "pdg": 11,
                          "x": AXIS, "y": 0.0, "z": 8980.0, "px": 10.0, "py": 0.0, "pz": 104.0, "t": 100.0})
    with pytest.raises(ValueError, match="DS map must come first"):
        H.tracker_steps(start, "tracker.gdml", ["TSd.header", "DS.header"])


def test_handoff_rows_start_the_field_on_tracker_and_fill_the_injector_table(tmp_path):
    rng = np.random.default_rng(21)
    beam = G.BeamGeometry(AXIS, np.array([5000.0]), np.full(1, 21.5), np.full(1, 75.0), np.full(1, 0.0528), 4204.6,
                          25.4, 235.0, 525.0, 300.0, 0.5, 9500.0, 9501.0, 900.0, 4280.0, 8530.0)
    cg = C.CEGeometry(beam, 900.0, 900.0, 1.0, z_plane=8539.9)
    n = 300
    ct = rng.uniform(0.5, 0.8, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    u = np.stack([np.sqrt(1 - ct ** 2) * np.cos(ph), np.sqrt(1 - ct ** 2) * np.sin(ph), ct], axis=1)
    x = np.tile([AXIS + 5.0, 0.0, 5800.0], (n, 1))
    g = pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(0), "event": np.arange(n, dtype=np.uint64)})
    res = C.transport(rng, cg, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0)
    ent = H.handoff(g, res)
    assert len(ent) == n
    geom = _straws()
    env = fs.Envelope(geom, pad_r=500.0, pad_z=600.0, axis_xy=ORIGIN[:2])
    assert (ent.z > env.z0).all() and (ent.z < env.z1).all()               # the entry plane is inside the envelope
    steps, final = fs.transport(np.random.default_rng(5), ent, geom, np.array([0.0, 0.0, 1.0]), envelope=env,
                                origin=ORIGIN, loss=_no_loss, msc=False, tail=False)
    assert len(final) == n and (final.status == 0).all()
    p0 = np.linalg.norm(ent[["px", "py", "pz"]].to_numpy(), axis=1)
    np.testing.assert_allclose(np.linalg.norm(final[["px", "py", "pz"]].to_numpy(), axis=1), p0, rtol=1e-9)
    assert len(steps) > 0 and set(steps.event) <= set(ent.event)
    first = steps.sort_values("t").drop_duplicates("event").merge(ent[["event", "t"]], on="event",
                                                                     suffixes=("", "_0"))
    assert (first.t > first.t_0).all()
    H.write_injector(tmp_path / "inj.root", steps, ent)
    s = tables.read(str(tmp_path / "inj.root"), "sgsteps")
    p = tables.read(str(tmp_path / "inj.root"), "particles")
    assert list(s.columns) == H.STEP_COLS and len(s) == len(steps) and (s.sp == 1).all()
    assert len(p) == n and (p.key == 1).all() and (p.pdg == 11).all()
