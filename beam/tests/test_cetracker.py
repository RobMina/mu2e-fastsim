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


def test_starts_fall_back_per_event(tmp_path):
    """An event with a generated electron keeps it; only the event without one takes its SimParticle birth."""
    from mu2efastsim import tables
    gen = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [5, 6], "pdg": [11, 13], "x": [7.0, 0.0],
                        "y": [0.0, 0.0], "z": [5800.0, 0.0], "px": [0.0, 0.0], "py": [0.0, 0.0], "pz": [104.97, 0.0],
                        "t": [50.0, 0.0]})
    part = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [6, 5], "key": [2, 2], "pdg": [11, 11],
                         "isprim": [0, 0], "ismcprim": [1, 1], "sx": [1.0, 9.0], "sy": [0.0, 0.0],
                         "sz": [5822.0, 5822.0], "spx": [0.0, 0.0], "spy": [0.0, 0.0], "spz": [104.97, 104.97],
                         "st": [100.0, 999.0]})
    f = tmp_path / "t.root"
    tables.write(str(f), {"gen": gen, "particles": part}, as_ttree=True)
    s = H.starts([str(f)])
    assert list(s.event) == [5, 6]
    assert list(s.x) == [7.0, 1.0] and list(s.t) == [50.0, 100.0]


def test_births_from_stops_follow_the_ceendpoint_generator():
    rng = np.random.default_rng(31)
    n = 20000
    stops = {"x": np.full(5, AXIS + 30.0), "y": np.zeros(5), "z": np.linspace(5800, 5900, 5), "t": np.full(5, 300.0)}
    keys = pd.DataFrame({"run": np.uint64(1430), "subrun": np.uint64(9), "event": np.arange(n, dtype=np.uint64)})
    g = H.births_from_stops(rng, stops, keys)
    assert list(g.event) == list(keys.event) and (g.pdg == 11).all()
    p = np.sqrt(g.px ** 2 + g.py ** 2 + g.pz ** 2)
    np.testing.assert_allclose(p, C.P_CE_AL, rtol=1e-12)
    assert abs((g.pz / p).mean()) < 0.02 and abs((g.pz / p).var() - 1 / 3) < 0.01      # isotropic
    assert set(np.round(g.z, 6)) <= set(np.round(stops["z"], 6))
    assert (g.t > 300.0).all() and abs((g.t - 300.0).mean() - C.TAU_MU_AL) < 0.03 * C.TAU_MU_AL


def _gen_file(n_per, shards, pdg=None):
    """Generator rows in the cechain reference format, in shard order (as diospec starts writes them)."""
    rng = np.random.default_rng(5)
    n = n_per * len(shards)
    d = {k: rng.normal(size=n) for k in ("x", "y", "z", "px", "py", "pz")}
    d["time"] = rng.uniform(500.0, 1500.0, n)
    if shards != [None]:
        d["shard"] = np.repeat(shards, n_per)
    if pdg is not None:
        d["pdg"] = np.full(n, pdg, np.int32)
    return d


def test_births_from_starts_keep_rows_and_number_events_per_shard():
    src = _gen_file(3, [50, 51, 52])
    g = H.births_from_starts(src)
    assert len(g) == 9 and (g.pdg == 11).all() and (g.run == 0).all()
    assert list(g.subrun) == [50] * 3 + [51] * 3 + [52] * 3
    assert list(g.event) == [1, 2, 3] * 3
    np.testing.assert_array_equal(g.px, src["px"])
    np.testing.assert_array_equal(g.t, src["time"])
    assert not g.duplicated(H.KEY).any()
    h = H.births_from_starts(src, shards=(51, 52))
    assert list(h.subrun) == [51] * 3 + [52] * 3 and list(h.event) == [1, 2, 3] * 2
    np.testing.assert_array_equal(h.x, src["x"][3:])


def test_births_from_starts_take_rpc_electrons_without_shards():
    src = _gen_file(4, [None], pdg=11)
    g = H.births_from_starts(src)
    assert list(g.subrun) == [0] * 4 and list(g.event) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="other than electrons"):
        H.births_from_starts(_gen_file(4, [None], pdg=-11))
    with pytest.raises(ValueError, match="shard column"):
        H.births_from_starts(src, shards=(0, 1))
    bad = dict(src)
    del bad["time"]
    with pytest.raises(ValueError, match="'time'"):
        H.births_from_starts(bad)


def test_births_from_starts_number_interleaved_shards_in_row_order():
    src = _gen_file(1, [2, 3, 2, 3, 2])
    g = H.births_from_starts(src)
    assert list(g.subrun) == [2, 3, 2, 3, 2] and list(g.event) == [1, 1, 2, 2, 3]
    np.testing.assert_array_equal(g.x, src["x"])
    h = H.births_from_starts(src, shards=(3, 3))
    assert list(h.event) == [1, 2] and list(h.x) == list(src["x"][[1, 3]])


@pytest.mark.parametrize("args", [["t.root", "--starts", "s.npz"], ["--stops", "x.npz", "--starts", "s.npz"],
                                  [], ["t.root", "--shards", "0", "1"]])
def test_cli_starts_argument_rules(args, capsys, tmp_path):
    with pytest.raises(SystemExit):
        H.main(args + ["--outdir", str(tmp_path), "--gdml", "g", "--maps", "m", "--offline-root", "o"])


def test_resident_chunks_equal_the_padded_slices_of_the_flat_steps():
    """DeviceChain._digitize_resident's jitted flat steps and event chunks: the rows a per-chunk slice of the flat
    valid steps gives, ev shifted by the chunk's first event, zero beyond (digi_resident's padding), at fixed sizes."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    rng = np.random.default_rng(3)
    nev, slots = 9, 6
    valid = rng.uniform(size=(nev, slots)) < 0.5
    valid[4] = False                                              # an event without steps
    s = {k: rng.normal(size=(nev, slots)) for k in H.DIGI_COLS}
    s["shape"] = rng.integers(0, 3, (nev, slots)).astype(np.int32)
    s["ion"] = rng.integers(0, 2, (nev, slots)).astype(np.int32)
    for k, hi in (("plane", 36), ("panel", 6), ("straw", 96)):
        s[k] = rng.integers(0, hi, (nev, slots)).astype(np.int32)
    s["valid"] = valid
    tr, sl = np.nonzero(valid)
    ref = {k: s[k][tr, sl] for k in H.DIGI_COLS}
    ref["sid"] = ((s["plane"][tr, sl].astype(np.int64) << 10) | (s["panel"][tr, sl].astype(np.int64) << 7)
                  | s["straw"][tr, sl])
    ref["ev"] = tr.astype(np.int64)
    nv = len(tr)
    flat, chunk = H._resident_steps()
    st = flat({k: jnp.asarray(v) for k, v in s.items()}, 64)
    for k, v in ref.items():
        got = np.asarray(st[k])
        assert got.dtype == np.asarray(jnp.asarray(v)).dtype, k
        assert np.array_equal(got[:nv], v) and not got[nv:].any(), k
    start = np.concatenate([[0], np.cumsum(valid.sum(1))])
    for e0, ne in ((0, 3), (3, 2), (5, 4), (4, 1)):
        a, b = int(start[e0]), int(start[e0 + ne])
        sub = chunk(st, jnp.int64(a), jnp.int64(b - a), jnp.int64(e0), 32)
        for k, v in ref.items():
            want = (v[a:b] - e0) if k == "ev" else v[a:b]
            got = np.asarray(sub[k])
            assert got.shape == (32,) and np.array_equal(got[:b - a], want) and not got[b - a:].any(), (k, e0)


def test_bench_prepare_with_a_seed_draws_that_seeds_births(tmp_path, monkeypatch):
    """bench --fresh-inputs: bench_prepare(n, seed=s) draws the births run() would draw for seed s; without a seed it
    returns the stops as before."""
    rng = np.random.default_rng(4)
    stops = {"x": rng.normal(-3904, 20, 500), "y": rng.normal(0, 20, 500), "z": rng.uniform(5400, 6300, 500),
             "t": rng.exponential(300, 500)}
    np.savez(tmp_path / "stops.npz", **stops)
    monkeypatch.setenv("CETRACKER_STOPS", str(tmp_path / "stops.npz"))
    s, n = H.bench_prepare(64)
    assert n == 64 and set(s) == {"x", "y", "z", "t"} and np.array_equal(s["z"], stops["z"])
    births, n, seed = H.bench_prepare(64, seed=7)
    assert (n, seed) == (64, 7)
    for a, b in zip(births, H.DeviceChain.births(stops, 64, 7)):
        assert np.array_equal(a, b)
    other = H.bench_prepare(64, seed=8)[0]
    assert not np.array_equal(births[0], other[0])


def test_default_digi_electronics_are_valid_levers():
    """The default electronics levers of the resident digitiser validate (digi_resident.electronics_options) and keep
    the flat entries at most the pair capacity."""
    pytest.importorskip("jax")
    from trkfastsim.digi import digi_resident as DR
    opts = DR.electronics_options(H.DIGI_ELECTRONICS)
    assert 1 <= opts["entries_per_straw"] <= 4 and opts["saturation_pool"] and opts["saturation_tiles"] == "bounded"
