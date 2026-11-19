"""Per-entrant shower library: entrant attribution, entry faces, placement round trip, storage and ROOT export."""
import numpy as np
import pytest
import uproot

from calofastsim.showers import entrants as E
from mu2efastsim.library.hashing import content_hash


def _event(n, values):
    return dict(run=np.ones(n, dtype=np.int32), subrun=np.zeros(n, dtype=np.int32), event=np.asarray(values, dtype=np.int32))


def tables():
    """Event 1: CE 20 enters disk 0 front, its photon 21 enters disk 1 front (not an entrant), an unrelated photon 30
    enters disk 0 through the inner edge, electron 40 enters disk 0 through the back. Event 2: CE 20 enters disk 0
    through the inner edge only. Event 3: no entrant; particle 50 deposits."""
    rows = {"events": {**_event(3, [1, 2, 3]), "g4_status": np.zeros(3, dtype=np.int32)}}
    pev = [1, 1, 1, 1, 2, 3]
    rows["particles"] = {**_event(6, pev), "sim_product_id": np.full(6, 100), "sim_key": np.array([20, 21, 30, 40, 20, 50]),
                         "pdg": np.array([11, 22, 22, 11, 11, 22]), "parent_product_id": np.full(6, 100),
                         "parent_key": np.array([0, 20, 0, 0, 0, 0]), "parent_valid": np.array([0, 1, 0, 0, 0, 0], bool)}
    # VD steps: (event, key, volume, time, position, momentum)
    vd = [(1, 20, 73, 10., (300., 0., -10.), (0., 0., 100.)),
          (1, 21, 75, 12., (310., 0., -10.), (0., 0., 5.)),
          (1, 30, 77, 5., (380., 0., 20.), (3., 0., 1.)),
          (1, 40, 74, 15., (500., 0., 10.), (0., 0., -20.)),
          (2, 20, 77, 11., (0., 380., 30.), (0., 90., 40.)),
          (2, 20, 77, 13., (0., 381., 31.), (0., -1., 0.))]          # inward crossing: not an entry
    disk_origin = {0: 100., 1: 200.}
    steps = {**_event(len(vd), [v[0] for v in vd]), "sim_product_id": np.full(len(vd), 100),
             "sim_key": np.array([v[1] for v in vd]), "kind": np.zeros(len(vd), dtype=np.int32),
             "volume": np.array([v[2] for v in vd]), "time": np.array([v[3] for v in vd])}
    for k, a in enumerate("xyz"):
        steps["position_" + a] = np.array([v[4][k] + (disk_origin[E.VD_FACES[v[2]][0]] if a == "z" else 0.) for v in vd])
        steps["momentum_" + a] = np.array([v[5][k] for v in vd])
    rows["steps"] = steps
    disks = dict(run=np.ones(2, int), disk=np.array([0, 1]))
    for axis in "xyz":
        disks["origin_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        disks["size_" + axis] = np.full(2, 20.)
        for coord in "xyz":
            disks["axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    rows["disks"] = disks
    crystals = dict(run=np.ones(3, int), disk=np.array([0, 1, 0]), crystal=np.array([10, 11, 12]))
    for axis, values in zip("xyz", ([300., 310., 400.], [0., 0., 50.], [110., 210., 110.])):
        crystals["position_" + axis] = np.array(values)
        crystals["size_" + axis] = np.full(3, 20.)
    rows["crystals"] = crystals
    rows["virtualdetectors"] = dict(run=np.ones(2, int), id=np.array([73, 75]), position_x=np.zeros(2),
                                    position_y=np.zeros(2), position_z=np.array([100., 200.]))
    # CSS (event, key, crystal, time, e_g4, e_birks, crystal-local position)
    css = [(1, 20, 10, 11., 40., 39., (1., 2., 3.)), (1, 21, 11, 14., 5., 4.5, (0., 0., 1.)),
           (1, 30, 12, 6., 2., 1.9, (0., 1., 0.)), (1, 40, 10, 16., 3., 2.8, (0., 0., 0.)),
           (2, 20, 12, 12., 30., 29., (2., 0., 0.)), (3, 50, 10, 20., 7., 6.5, (0., 0., 0.))]
    sh = {**_event(len(css), [c[0] for c in css]), "sim_product_id": np.full(len(css), 100),
          "sim_key": np.array([c[1] for c in css]), "sim_valid": np.ones(len(css), dtype=np.int32),
          "crystal": np.array([c[2] for c in css]), "time": np.array([c[3] for c in css]),
          "energy_g4": np.array([c[4] for c in css]), "energy_birks": np.array([c[5] for c in css])}
    for k, a in enumerate("xyz"):
        sh["position_" + a] = np.array([c[6][k] for c in css])
    rows["showers"] = sh
    return rows


def write(path, data):
    with uproot.recreate(path) as out:
        for name, cols in data.items():
            out.mktree("dump/" + name, cols)


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "export.root"
    write(path, tables())
    return path


def test_entrant_attribution_and_census(export):
    entrants, css, geometry, census = E.read(export)
    # supported entrants: event-1 CE (front), event-1 photon 30 (inner), event-2 CE (inner); the back entrant is dropped
    np.testing.assert_array_equal(entrants["event_row"], [0, 0, 1])
    np.testing.assert_array_equal(entrants["face"], [E.FRONT, E.INNER, E.INNER])
    np.testing.assert_array_equal(entrants["pdg"], [11, 22, 11])
    np.testing.assert_array_equal(entrants["time"], [10., 5., 11.])
    # the CE owns its photon's disk-1 deposit; photon 30 owns its own; the back and orphan deposits are not kept
    np.testing.assert_array_equal(css["entrant"], [0, 0, 1, 2])
    np.testing.assert_array_equal(css["disk"], [0, 1, 0, 0])
    np.testing.assert_allclose(css["position"][0], [301., 2., 113.])
    total = 39. + 4.5 + 1.9 + 2.8 + 29. + 6.5
    assert census["entrants"] == 4
    assert census["by_face"]["back"]["entrants"] == 1
    assert census["by_face"]["front"]["energy_fraction"] == pytest.approx((39. + 4.5) / total)
    assert census["css_fraction_no_entrant"] == pytest.approx(6.5 / total)


def test_inward_inner_crossing_is_not_an_entry(export):
    entrants, _, _, _ = E.read(export)
    assert entrants["time"][2] == 11.        # the later inward crossing of the event-2 CE is ignored


def _self_queries(entrants):
    return dict(disk=entrants["disk"], face=entrants["face"], pdg=entrants["pdg"], time=entrants["time"],
                local_position=entrants["local_position"], local_momentum=entrants["local_momentum"])


def test_round_trip_places_donor_at_its_own_entry(export):
    entrants, css, geometry, _ = E.read(export)
    library = E.build(entrants, css, geometry)
    out = E.sample(library, _self_queries(entrants), np.random.default_rng(1))
    np.testing.assert_array_equal(out["donor"], [0, 1, 2])            # one entrant per occupied cell
    assert out["supported"].all()
    np.testing.assert_allclose(out["position"], css["position"], atol=1e-9)
    np.testing.assert_allclose(out["time"], css["time"], atol=1e-12)
    np.testing.assert_array_equal(out["energy_birks"], css["energy_birks"])


def test_rotation_about_the_disk_axis(export):
    entrants, css, geometry, _ = E.read(export)
    library = E.build(entrants, css, geometry)
    q = _self_queries(entrants)
    turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])        # +90 degrees about z
    q["local_position"] = q["local_position"] @ turn.T
    q["local_momentum"] = q["local_momentum"] @ turn.T
    out = E.sample(library, q, np.random.default_rng(1))
    origin = np.where(css["disk"][:, None] == 0, [0., 0., 100.], [0., 0., 200.])
    expected = (css["position"] - origin) @ turn.T + origin
    np.testing.assert_allclose(out["position"], expected, atol=1e-9)


def test_unsupported_group_and_float32(export):
    entrants, css, geometry, _ = E.read(export)
    library = E.build(entrants, css, geometry)
    q = _self_queries(entrants)
    q["pdg"] = np.array([-11, 22, 11])                                   # no positron entrants in the library
    out = E.sample(library, q, np.random.default_rng(1))
    np.testing.assert_array_equal(out["supported"], [False, True, True])
    np.testing.assert_array_equal(np.diff(out["offsets"]), [0, 1, 1])
    out32 = E.sample(library, _self_queries(entrants), np.random.default_rng(1), dtype=np.float32)
    assert out32["position"].dtype == np.float32
    np.testing.assert_allclose(out32["position"], css["position"], atol=1e-3)


def test_save_load_and_root_export(export, tmp_path):
    entrants, css, geometry, census = E.read(export)
    library = E.build(entrants, css, geometry)
    E.save(tmp_path / "lib.npz", library, dict(name="fixture", census=census))
    loaded, metadata = E.load(tmp_path / "lib.npz")
    assert metadata["name"] == "fixture"
    for k, v in library.items():
        np.testing.assert_array_equal(loaded[k], v)
    digest = E.export_root(library, tmp_path / "lib.root")
    assert digest == content_hash(library)
    with uproot.open(tmp_path / "lib.root") as f:
        assert str(f["content_hash"]) == digest
        e = f["entries"].arrays(library="np")
        s = f["steps"].arrays(library="np")
        np.testing.assert_array_equal(e["face"], library["entry_face"])
        np.testing.assert_array_equal(e["step_end"] - e["step_begin"], np.diff(library["offsets"]))
        np.testing.assert_allclose(s["x"], library["step_relative_position"][:, 0])
        np.testing.assert_allclose(f["bins_inner_depth"]["edge"].array(library="np"), library["bins_inner_depth"])
