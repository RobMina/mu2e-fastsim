"""Unit tests: EdepAna-format files of the fast CE chain (beamfastsim.edepana). No Mu2e inputs."""
import json
import math
import re

import numpy as np
import pytest

from beamfastsim import cesob as S
from beamfastsim import edepana as E

pytest.importorskip("uproot")
pytest.importorskip("awkward")

# The interpretations Offline EdepAna (v13_39_00) gives its branches, as uproot reports them on a stock file.
STOCK_TYPES = {
    "nprimaries": "AsDtype('>i4')", "primary_start_pdg": "AsJagged(AsDtype('>i4'))",
    "event_calo_edep": "AsDtype('>f4')", "event_calo_edep_vis": "AsDtype('>f4')", "event_trk_edep": "AsDtype('>f4')",
    "run": "AsDtype('>i4')", "subrun": "AsDtype('>i4')", "event": "AsDtype('>i4')", "weight": "AsDtype('>f4')",
    "ngen": "AsDtype('>i8')",
    **{k: "AsJagged(AsDtype('>f4'))" for k in E.PRIMARY if k != "primary_start_pdg"},
}

# The design studies' reader: the branches it asks for and its first-primary rule (analysis server tools/analyses/edep.py).
READER_EVENT = ("event_calo_edep", "event_calo_edep_vis", "event_trk_edep", "weight", "run", "subrun", "event",
                "nprimaries", "ngen")
READER_PRIMARY = ("primary_start_x", "primary_start_y", "primary_start_z", "primary_start_px", "primary_start_py",
                  "primary_start_pz", "primary_start_e", "primary_start_m", "primary_start_pdg", "primary_calo_edep",
                  "primary_calo_edep_vis", "primary_trk_front_p", "primary_trk_front_energy")
# Its summary-block regexes (same file)
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
SUMMARY_FIELDS = [
    ("n_events", rf"Saw\s+({_NUM})\s+events\b"), ("n_gen_events", rf"\(\s*({_NUM})\s+gen events\)"),
    ("event_rate", rf"output rate\s*=\s*({_NUM})\s+events / gen event"),
    ("avg_calo_edep_per_event_mev", rf"Average calo energy deposition per event:\s*({_NUM})\s*MeV"),
    ("avg_calo_edep_per_gen_event_mev", rf"Average calo energy deposition per gen event:\s*({_NUM})\s*MeV"),
    ("n_events_calo_edep_above_50mev", rf"Events with calo Edep > 50 MeV:\s*({_NUM})"),
    ("avg_trk_edep_per_event_mev", rf"Average tracker energy deposition per event:\s*({_NUM})\s*MeV"),
    ("avg_trk_edep_per_gen_event_mev", rf"Average tracker energy deposition per gen event:\s*({_NUM})\s*MeV"),
]


def _rows(n=50, seed=1, ngen0=1000):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    p = np.full(n, 104.9726)
    cols = dict(x=np.column_stack([rng.uniform(-4000, -3800, n), rng.uniform(-80, 80, n), rng.uniform(5400, 6300, n)]),
                u=u, p=p, t=rng.uniform(500, 1700, n), trk_e=rng.uniform(0, 0.2, n),
                trk_n=rng.integers(0, 40, n), p_front=np.where(rng.random(n) < 0.9, 104.0, 0.0),
                calo_g4=rng.uniform(0, 110, n), calo_vis=rng.uniform(0, 105, n))
    cols["e_front"] = np.where(cols["p_front"] > 0, np.hypot(cols["p_front"], E.ME), 0.0)
    return E.rows_from_columns(cols, run=1, subrun=0, ngen_running=ngen0), cols


def _reader(path):
    """The studies' read_edep_tree, branch for branch."""
    import uproot
    with uproot.open(path) as f:
        b = f[E.TREE].arrays(READER_EVENT + READER_PRIMARY, library="np")
    nprim = np.asarray(b["nprimaries"])
    first = {k: np.array([e[0] for e in b[k][nprim > 0]]) for k in READER_PRIMARY}
    return b, first


def test_filter_is_the_ceendpoint_detector_step_filter():
    keep = E.detector_filter([11, 12, 0, 0, 30], [44.9, 0.0, 45.0, 10.0, 0.0])
    assert keep.tolist() == [False, True, True, False, True]


def test_rows_keep_stored_events_with_start_front_and_running_count():
    rows, cols = _rows()
    keep = E.detector_filter(cols["trk_n"], cols["calo_vis"])
    idx = np.flatnonzero(keep)
    assert rows["event"].tolist() == (idx + 1).tolist()
    assert (rows["ngen"] == 1000).all() and (rows["weight"] == 1).all() and (rows["primary_start_pdg"] == 11).all()
    np.testing.assert_allclose(rows["primary_start_e"], math.hypot(104.9726, E.ME))
    np.testing.assert_allclose(np.hypot.reduce([rows[f"primary_start_p{a}"] for a in "xyz"]), 104.9726)
    np.testing.assert_array_equal(rows["event_calo_edep_vis"], rows["primary_calo_edep_vis"])
    np.testing.assert_array_equal(rows["primary_trk_front_p"], cols["p_front"][idx])


def test_tree_has_the_edepana_branches_types_and_shared_counter(tmp_path):
    import uproot
    rows, _ = _rows()
    path = E.write_tree(tmp_path / "f.root", rows, provenance={"k": 1})
    with uproot.open(path) as f:
        t = f[E.TREE]
        assert {k: repr(t[k].interpretation) for k in STOCK_TYPES} == STOCK_TYPES
        assert set(t.keys()) == set(STOCK_TYPES)
        assert t.num_entries == rows["event"].size
        assert json.loads(str(f[E.PROVENANCE])) == {"k": 1}
    b, first = _reader(path)
    assert (b["nprimaries"] == 1).all()
    for k in READER_PRIMARY:
        np.testing.assert_array_equal(first[k], np.asarray(rows[k]).astype(first[k].dtype))
    for k in ("event_calo_edep_vis", "ngen", "subrun", "event"):
        np.testing.assert_array_equal(b[k], np.asarray(rows[k]).astype(b[k].dtype))


def test_empty_file_still_has_the_tree(tmp_path):
    import uproot
    rows, _ = _rows()
    rows = {k: v[:0] for k, v in rows.items()}
    path = E.write_tree(tmp_path / "e.root", rows)
    with uproot.open(path) as f:
        assert f[E.TREE].num_entries == 0 and "nprimaries" in f[E.TREE].keys()


def test_falling_ngen_is_refused(tmp_path):
    rows, _ = _rows()
    rows["ngen"] = np.arange(rows["event"].size)[::-1]
    with pytest.raises(ValueError, match="ngen"):
        E.write_tree(tmp_path / "x.root", rows)


def test_summary_block_parses_with_the_studies_regexes_and_matches_the_tree(tmp_path):
    a, _ = _rows(n=80, seed=2, ngen0=75000)
    b, _ = _rows(n=60, seed=3, ngen0=150000)
    rows = E.concat_rows([a, b])
    paths = E.write_outputs(tmp_path, rows, {"test": True})
    text = paths["summary"].read_text()
    got = {k: float(re.search(rx, text).group(1)) for k, rx in SUMMARY_FIELDS}
    m = E.summary(rows)
    for k, v in got.items():
        assert v == pytest.approx(m[k], rel=1e-5), k
    assert got["n_gen_events"] == 150000 and got["n_events"] == rows["event"].size
    assert json.loads(paths["run_record"].read_text()) == {"max_events": None}
    assert paths["run_record"].name == paths["file"].name + ".edep.json"
    side = json.loads(paths["provenance"].read_text())
    assert side["test"] is True and side["summary"]["n_gen_events"] == 150000
    back = E.read_tree(paths["file"])
    np.testing.assert_allclose(back["event_calo_edep"], rows["event_calo_edep"], rtol=1e-6)


def _library():
    from calofastsim.showers import build
    entries = dict(energy=np.array([100., 100., 70.]), cosine=np.ones(3), radius=np.array([400., 400., 600.]),
                   time=np.zeros(3), disk=np.array([0, 0, 1]), event_row=np.arange(3),
                   position=np.array([[400., 0, 0], [400., 0, 0], [600., 0, 0]]), direction=np.tile([0., 0., 1.], (3, 1)))
    steps = dict(entry=np.array([0, 0, 2]), energy_g4=np.array([20., 30., 60.]), energy_birks=np.array([19., 29., 58.]),
                 time=np.ones(3), crystal=np.arange(3), disk=np.array([0, 1, 1]), position=np.zeros((3, 3)))
    lib = build(entries, steps, development=np.ones(3, bool))
    lib.update(geometry_disk_disk=np.array([0, 1]), **{f"geometry_disk_origin_{a}": np.zeros(2) for a in "xyz"},
               geometry_disk_axis_z_x=np.zeros(2), geometry_disk_axis_z_y=np.zeros(2), geometry_disk_axis_z_z=np.ones(2))
    return lib


def test_shower_energies_are_the_drawn_patterns_whole_sums():
    sh = E.ShowerEnergies(_library())
    np.testing.assert_allclose(sh.e_g4, [50., 0., 60.])
    np.testing.assert_allclose(sh.e_vis, [48., 0., 58.])
    n = 4000
    q = E._entries(np.full(n, 100.), np.tile([400., 0., 0.], (n, 1)), np.tile([0., 0., 1.], (n, 1)),
                   np.zeros(n, int), np.zeros(n), np.zeros((2, 3)))
    g, v, donor, _ = sh.draw(np.random.default_rng(1), q)
    assert set(donor.tolist()) == {0, 1}
    np.testing.assert_array_equal(g, np.where(donor == 0, 50., 0.))
    np.testing.assert_array_equal(v, np.where(donor == 0, 48., 0.))
    assert abs((donor == 0).mean() - 0.5) < 0.03
    z = sh.draw(np.random.default_rng(1), {k: x[:0] for k, x in q.items()})
    assert all(len(a) == 0 for a in z)


def test_contact_state_does_not_change_the_wall_flags():
    from test_cesob import CALO, _helix_starts, _wall_starts
    c = S.calo_from_config(CALO)
    w = S.calo_wall_from_config(CALO, c)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    cos = 0.15
    R = math.sqrt(1 - cos ** 2) * p / (0.299792458 * 1.0)
    xa, ua = _wall_starts(R, cos)
    xb, ub, _ = _helix_starts(0.0, R, n=4)
    x, u = np.concatenate([xa, xb]), np.concatenate([ua, ub])
    TT = np.full(len(x), T)
    plain = S.calo_accept([0.0, 0.0, 1.0], c, x, u, TT, wall=w, rng=np.random.default_rng(3))
    contact = {}
    flags = S.calo_accept([0.0, 0.0, 1.0], c, x, u, TT, wall=w, rng=np.random.default_rng(3), contact=contact)
    np.testing.assert_array_equal(plain, flags)
    assert (contact["found"] >= flags).all()           # every accepted electron has a contact
    assert not contact["found"][len(xa):].any()         # the bore helices never touch a crystal
    f = contact["found"]
    if f.any():
        r = np.hypot(contact["x"][f, 0] - c.axis_x, contact["x"][f, 1])
        assert (r >= c.r_in - 1).all() and (r <= c.r_out + 1).all()
        np.testing.assert_allclose(np.linalg.norm(contact["u"][f], axis=1), 1.0, atol=1e-9)
    with pytest.raises(ValueError, match="wall"):
        S.calo_accept([0.0, 0.0, 1.0], c, x, u, TT, contact={})


def test_stop_tables_in_both_layouts_give_the_mu_minus_stops():
    x = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    a = E._mu_minus_stops(dict(x=x, t=np.array([1., 2., 3.]), q=np.array([-1, 1, -1])))
    assert a["x"].tolist() == [1., 7.] and a["z"].tolist() == [3., 9.] and a["t"].tolist() == [1., 3.]
    b = E._mu_minus_stops(dict(x=x[:, 0], y=x[:, 1], z=x[:, 2], t=np.array([1., 2., 3.])))
    assert b["y"].tolist() == [2., 5., 8.]
    with pytest.raises(ValueError, match="no mu- stops"):
        E._mu_minus_stops(dict(x=x[:0], t=np.zeros(0)))


def test_edge_energies_draw_from_the_query_bin_or_the_nearest_full_one():
    rng = np.random.default_rng(0)
    dz = np.r_[np.full(30, 10.0), np.full(30, 60.0), np.full(5, 110.0), np.full(30, 210.0)]
    tab = dict(disk=np.r_[np.zeros(95, int)], dz=dz, e_vis=np.r_[np.full(30, 80.), np.full(30, 40.), np.full(5, 1.),
                                                                   np.full(30, 0.)],
               e_g4=np.r_[np.full(30, 81.), np.full(30, 41.), np.full(5, 2.), np.full(30, 0.5)])
    ee = E.EdgeEnergies(tab)
    g, v = ee.draw(rng, np.zeros(4, int), np.array([5.0, 70.0, 115.0, 230.0]))
    assert v.tolist() == [80.0, 40.0, 40.0, 0.0]     # 100-125 is sparse (5 < 20): its nearest full bin is 50-75
    assert g.tolist() == [81.0, 41.0, 41.0, 0.5]
    g, v = ee.draw(rng, np.array([1]), np.array([10.0]))  # a disk without entries: no energy
    assert v.tolist() == [0.0] and g.tolist() == [0.0]


def test_first_contact_separates_front_and_inner_edge_entries_in_a_uniform_field():
    from test_cesob import _helix_starts
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    cos = 0.6
    R = math.sqrt(1 - cos ** 2) * p / (0.299792458 * 1.0)
    fz = (11877.03, 12577.03)
    # starting at r = 500 mm upstream of disk 0 (r_max 500 mm): the front of disk 0 inside its annulus
    x, u, _ = _helix_starts(500.0 - R, R, n=1, cos=cos, z=11800.0)
    c = E.first_contact([0.0, 0.0, 1.0], x, u, np.full(1, T), fz, -3904.0)
    assert (c["kind"] == E.FRONT).all() and (c["disk"] == 0).all()
    np.testing.assert_allclose(c["x"][:, 2], fz[0], atol=3.0)
    # flat helix (cos 0.2) starting inside disk 0's length at r = 331 mm, r_max = 345 mm: disk 0's inner edge
    cos2 = 0.2
    R2 = math.sqrt(1 - cos2 ** 2) * p / (0.299792458 * 1.0)
    rc = 345.0 - R2
    x = np.tile([-3904.0 + rc - R2, 0.0, 11900.0], (4, 1))   # the helix point nearest the axis
    u = np.tile([0.0, -math.sqrt(1 - cos2 ** 2), cos2], (4, 1))
    c = E.first_contact([0.0, 0.0, 1.0], x, u, np.full(4, T), fz, -3904.0)
    assert (c["kind"] == E.EDGE).all() and (c["disk"] == 0).all()
    assert ((c["dz"] > 0) & (c["dz"] < E.DISK_LENGTH)).all()
    r = np.hypot(c["x"][:, 0] + 3904.0, c["x"][:, 1])
    assert ((r >= E.EDGE_R) & (r < E.EDGE_R + 5)).all()
    # r_max = 81 mm: never touches the calorimeter
    T3 = 30.0
    R3 = math.sqrt(1 - cos ** 2) * math.sqrt(T3 * (T3 + 2 * 0.51099895)) / 0.299792458
    x, u, _ = _helix_starts(0.0, R3, n=2, cos=cos, z=11800.0)
    c = E.first_contact([0.0, 0.0, 1.0], x, u, np.full(2, T3), fz, -3904.0)
    assert (c["kind"] == 0).all() and np.isnan(c["dz"]).all()


def test_host_stage_library_edge_on_synthetic_device_columns():
    """FastEdep.calo (calo library+edge) end to end on the host: front entries draw library patterns, edge entries
    the edge table, electrons that never left the tracker nothing; the rows keep the stored events only."""
    from test_cesob import _helix_starts
    fe = object.__new__(E.FastEdep)
    fe.showers = E.ShowerEnergies(_library())
    fe.origin = np.array([[-3904.0, 0.0, 12069.3], [-3904.0, 0.0, 12769.3]])
    fe.edge = E.EdgeEnergies(dict(disk=np.zeros(40, int), dz=np.full(40, 10.0), e_vis=np.full(40, 30.0),
                                  e_g4=np.full(40, 31.0)))
    fe.field, fe.front_z, fe.axis_x, fe.side = [0.0, 0.0, 1.0], np.array([11877.03, 12577.03]), -3904.0, None
    T = 103.0
    p = math.sqrt(T * (T + 2 * E.ME))
    R = 0.8 * p / 0.299792458
    xf, uf, _ = _helix_starts(500.0 - R, R, n=1, cos=0.6, z=11800.0)          # front of disk 0
    R2 = math.sqrt(1 - 0.04) * p / 0.299792458
    xe = np.array([[-3904.0 + (450.0 - R2) - R2, 0.0, 11900.0]])              # disk 0's inner edge, rmax 450 mm
    xl = np.array([[-3904.0 + (345.0 - R2) - R2, 0.0, 11900.0]])              # the same edge, rmax 345 mm
    ue = np.array([[0.0, -math.sqrt(1 - 0.04), 0.2]])
    x = np.concatenate([xf, xe, xf, xl])
    u = np.concatenate([uf, ue, uf, ue])
    n = 4
    h = dict(x0=np.zeros((n, 3)), u0=np.tile([0.0, 0.0, 1.0], (n, 1)), p0=np.full(n, 104.97), t0=np.zeros(n),
             has=np.array([True, True, False, True]), pf=np.tile([0.0, 0.0, 104.0], (n, 1)),
             trk_e=np.full(n, 0.05), trk_n=np.array([30, 30, 0, 0]), out=np.array([True, True, False, True]),
             found=np.zeros(n, bool), xe=x, ue=u, Te=np.full(n, T), te=np.zeros(n))
    cols, stats = fe.calo(h, 1, 0)
    assert stats["disk_front"] == 1 and stats["edge_entry"] == 1 and stats["below_rmax"] == 1
    assert cols["calo_vis"][0] in (48.0, 0.0) and cols["calo_vis"][1] == 30.0 and cols["calo_vis"][2] == 0.0
    assert cols["calo_vis"][3] == 0.0 and cols["calo_g4"][1] == 31.0
    assert cols["disk_front"].tolist() == [True, False, False, False]
    rows = E.rows_from_columns(cols, 1, 0, 4)               # the low-rmax electron: no steps, no energy, not stored
    assert rows["event"].tolist() == [1, 2] and (rows["ngen"] == 4).all()


def test_rmax_is_the_helix_largest_radius_about_the_axis():
    # e- in +1 T at (axis_x + 100, 0) moving +y with pt = 0.3 * 200 MeV/c: R = 200 mm, centre at (axis_x - 100, 0)
    pt = 0.299792458 * 200.0
    r = E.rmax_1t(np.array([-3904.0 + 100.0]), np.array([0.0]), np.array([0.0]), np.array([pt]))
    np.testing.assert_allclose(r, 300.0)
    tab = dict(disk=np.zeros(60, int), dz=np.full(60, 10.0), rmax=np.r_[np.full(30, 430.0), np.full(30, 600.0)],
               e_vis=np.r_[np.full(30, 20.0), np.full(30, 70.0)], e_g4=np.zeros(60))
    ee = E.EdgeEnergies(tab)
    _, v = ee.draw(np.random.default_rng(0), np.zeros(3, int), np.full(3, 10.0), np.array([420.0, 600.0, 480.0]))
    assert v.tolist() == [20.0, 70.0, 20.0]      # 450-520 is empty: its nearest radius bin (below 450) is used


def test_default_batch_follows_the_backend(monkeypatch):
    jax = pytest.importorskip("jax")
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert E.default_batch() == E.BATCH_CPU == 5000
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert E.default_batch() == E.BATCH_GPU == 65536
