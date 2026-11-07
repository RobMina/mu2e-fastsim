"""Device chain against the host reference: library draw, placement, and the deterministic response-to-clusters path."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("scipy")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from calofastsim import device_chain as dc  # noqa: E402
from calofastsim.chain import run_chain  # noqa: E402
from calofastsim.cluster import Geometry, make_clusters  # noqa: E402,F401
from calofastsim.pulse import PulseCache  # noqa: E402
from calofastsim.readout import PropagationTable, ReadoutConfig  # noqa: E402
from calofastsim.showers.crystal_placement import place_crystals  # noqa: E402
from calofastsim.showers.library import build, sample  # noqa: E402
from calofastsim.showers.response import place  # noqa: E402

PITCH, SIZE, DEPTH, INNER, OUTER, DZ = 34.3, 34.0, 200.0, 374.0, 660.0, 700.0


def _grid():
    n = int(np.ceil(OUTER / PITCH)) + 1
    c = (np.arange(-n, n + 1)) * PITCH
    x, y = [v.ravel() for v in np.meshgrid(c, c, indexing="ij")]
    r = np.hypot(x, y)
    keep = (r - SIZE / np.sqrt(2) >= INNER) & (r + SIZE / np.sqrt(2) <= OUTER)
    x, y = x[keep], y[keep]
    m = len(x)
    pos = np.zeros((2 * m, 3))
    pos[:m, :2] = pos[m:, :2] = np.column_stack((x, y))
    pos[m:, 2] = DZ
    disk = np.repeat([0, 1], m)
    return pos, disk


def _geometry_keys(pos, disk):
    g = dict(geometry_crystal_crystal=np.arange(len(disk)), geometry_crystal_disk=disk,
             geometry_disk_disk=np.array([0, 1]), geometry_vd_id=np.array([73, 75]),
             geometry_vd_position_z=np.array([-1.0, DZ - 1.0]),
             geometry_disk_size_x=np.full(2, INNER), geometry_disk_size_y=np.full(2, OUTER))
    for k, axis in enumerate("xyz"):
        g["geometry_crystal_position_" + axis] = pos[:, k]
        g["geometry_crystal_size_" + axis] = np.full(len(disk), DEPTH if axis == "z" else SIZE)
        g["geometry_disk_origin_" + axis] = np.array([0.0, DZ]) if axis == "z" else np.zeros(2)
        for coord in "xyz":
            g["geometry_disk_axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    return g


def _containing(pos, disk, p, d):
    inside = ((np.abs(p[:, None, :2] - pos[None, :, :2]) <= SIZE / 2).all(axis=2) &
              (p[:, None, 2] - pos[None, :, 2] >= 0) & (p[:, None, 2] - pos[None, :, 2] <= DEPTH) &
              (disk[None, :] == d[:, None]))
    return np.where(inside.any(axis=1), inside.argmax(axis=1), -1)


def _entries(rng, n):
    d = rng.integers(0, 2, n)
    r = rng.uniform(INNER + 10, OUTER - 10, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    cos = rng.uniform(0.3, 1.0, n)
    sin = np.sqrt(1 - cos ** 2)
    a = rng.uniform(-np.pi, np.pi, n)
    return dict(energy=rng.uniform(30.0, 108.0, n), cosine=cos, radius=r, disk=d, time=rng.uniform(500, 1500, n),
                event_row=np.arange(n), position=np.column_stack((r * np.cos(phi), r * np.sin(phi), DZ * d)),
                direction=np.column_stack((sin * np.cos(a), sin * np.sin(a), cos)))


@pytest.fixture(scope="module")
def library():
    rng = np.random.default_rng(20261107)
    pos, disk = _grid()
    e = _entries(rng, 600)
    rows, p, sd = [], [], []
    for i in range(600):
        k = int(rng.integers(0, 14))
        off = np.column_stack((rng.normal(0, 30, k), rng.normal(0, 30, k), rng.uniform(1, 190, k)))
        other = rng.random(k) < 0.15
        q = e["position"][i] + off
        q[other, 2] += DZ * (1 - 2 * e["disk"][i])
        dd = np.where(other, 1 - e["disk"][i], e["disk"][i])
        c = _containing(pos, disk, q, dd)
        ok = c >= 0
        rows += [i] * int(ok.sum())
        p.append(q[ok])
        sd.append(dd[ok])
    p, sd = np.concatenate(p), np.concatenate(sd)
    n = len(rows)
    s = dict(entry=np.array(rows), energy_g4=rng.uniform(0.1, 20, n), energy_birks=rng.uniform(0.1, 20, n),
             time=e["time"][rows] + rng.uniform(0, 3, n), crystal=_containing(pos, disk, p, sd), disk=sd, position=p)
    lib = build(e, s, development=np.ones(600, bool))
    lib.update(_geometry_keys(pos, disk))
    return lib


def _uniforms(entries, lib, seed):
    """The uniforms library.sample consumes, in query order (one array per disk that has donors)."""
    rng = np.random.default_rng(seed)
    u = np.zeros(len(entries["energy"]))
    for disk in np.unique(entries["disk"]):
        where = np.flatnonzero(entries["disk"] == disk)
        if np.any(lib["entry_disk"] == disk):
            u[where] = rng.random(len(where))
    return u


def _queries(n=300, seed=5):
    q = _entries(np.random.default_rng(seed), n)
    q["energy"][:3] = [120.0, 5.0, 60.0]   # out-of-range and sparse cells use the fallback
    return q


@pytest.mark.parametrize("neighbors", [0, 3])
@pytest.mark.parametrize("placement", ["crystal", "geometric"])
def test_draw_and_placement_match_host(library, neighbors, placement):
    query = _queries()
    host = sample(library, query, np.random.default_rng(11), orientation="disk", radius_neighbors=neighbors)
    placed = place_crystals(library, query, host) if placement == "crystal" else place(library, host)
    lib = dc.pack_library(library, radius_neighbors=neighbors, placement=placement)
    q, _ = dc.entries_batch(query, _timing(1), np.arange(len(query["energy"])), 320)
    u = np.zeros(320)
    u[:300] = _uniforms(query, library, 11)

    @jax.jit
    def stages(q, u):
        donor, fallback, slot = dc.sample_donors(lib, q, u)
        st, over = dc.gather_steps(lib, q, donor, slot, 4096)
        crystal, depth, mapped = dc.place_steps(lib, q, st)
        return donor, fallback, st, crystal, depth, mapped, over

    donor, fallback, st, crystal, depth, mapped, over = jax.tree_util.tree_map(
        np.asarray, stages({k: jnp.asarray(v) for k, v in q.items()}, jnp.asarray(u)))
    np.testing.assert_array_equal(donor[:300], host["donor"])
    np.testing.assert_array_equal(fallback[:300], host["fallback"])
    n = len(host["entry"])
    assert int(over) == 0 and st["valid"].sum() == n
    np.testing.assert_allclose(st["position"][:n], host["position"], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(st["time"][:n], host["time"])
    np.testing.assert_array_equal(mapped[:n], placed["mapped"])
    np.testing.assert_array_equal(library["geometry_crystal_crystal"][crystal[:n][mapped[:n]]],
                                  placed["crystal"][placed["mapped"]])
    np.testing.assert_allclose(depth[:n][mapped[:n]], placed["local_position"][placed["mapped"], 2], atol=1e-9)
    assert 0 < mapped.sum() < n


def _pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _geom(pos, disk):
    near = ((np.abs(pos[:, None, 0] - pos[None, :, 0]) < 1.5 * PITCH) &
            (np.abs(pos[:, None, 1] - pos[None, :, 1]) < 1.5 * PITCH) & (disk[:, None] == disk[None, :]))
    return Geometry(pos, disk, near & ~np.eye(len(disk), dtype=bool), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, DZ]]))


def _sim(n):
    return {"lru": np.zeros(n, np.float32), "pe_per_mev": np.full((n, 2), 30.0, np.float32),
            "adc_per_mev": np.full((n, 2), 16.0, np.float32)}


def _timing(n):
    return {"pbtime": np.full(n, -200.0, np.float32), "mbtime": np.full(n, 1695.0, np.float32),
            "event_length": np.full(n, 1700.0), "onspill": np.ones(n, bool),
            "marker": np.full(n, 200.0, np.float32), "crystal_length": np.full(n, 200.0, np.float32)}


def test_response_to_clusters_matches_host_chain():
    """Photon statistics, travel time and the bunch-time smear off, constant noise (the stock fit objective divides by
    the baseline): digis exact, clusters to roundoff."""
    pos, disk = _grid()
    geom = _geom(pos, disk)
    rng = np.random.default_rng(3)
    n_ev, per = 6, 12
    ev = np.repeat(np.arange(n_ev), per)
    seedc = rng.integers(0, len(disk), n_ev)
    near = np.argsort(np.linalg.norm(pos[None, :, :] - pos[seedc][:, None, :], axis=2), axis=1)[:, :5]
    crystal = near[ev, rng.integers(0, 5, ev.size)]
    steps = {"event": ev, "crystal": crystal, "energy_g4": rng.uniform(0.5, 25, ev.size).astype(np.float32),
             "energy_birks": rng.uniform(0.5, 25, ev.size).astype(np.float32),
             "time": rng.uniform(600, 1400, n_ev)[ev] + rng.uniform(0, 1.5, ev.size),
             "z_local": rng.uniform(5, 195, ev.size).astype(np.float32),
             "lru": np.zeros(ev.size, np.float32), "yields": np.full((ev.size, 2), 30.0, np.float32)}
    timing = _timing(n_ev)
    pulse = _pulse()
    ro = ReadoutConfig(pe_stat=False, travel=False)
    noise = np.full(10000, 12.0)
    host = run_chain(steps, timing, lambda s: np.full(np.shape(s), np.float32(16) / np.float32(30)), pulse, noise,
                     geom, np.random.default_rng(1), ro_cfg=ro, pbt_reco_sigma=0.0)
    table = PropagationTable(np.ones((1, 1), np.float32), np.zeros(1, np.float32), np.float32(1.0))
    resp = dc.pack_response(_sim(len(disk)), pulse, noise, table, geom, ro_cfg=ro, pbt_reco_sigma=0.0)
    caps = dc.Caps(events=8, steps=ev.size, rows=128, digis=128, digi_length=64, hits=128, event_hits=32, rounds=6)

    @jax.jit
    def downstream(t, key):
        q = {"event": jnp.asarray(ev)}
        st = dict(row=jnp.arange(ev.size), time=jnp.asarray(steps["time"]), valid=jnp.ones(ev.size, bool),
                  energy_g4=jnp.asarray(steps["energy_g4"]), energy_birks=jnp.asarray(steps["energy_birks"]))
        c = jnp.asarray(crystal)
        e, s, pt, w, ok, _ = dc.photoelectrons(resp, q, t, st, c, jnp.asarray(steps["z_local"]),
                                               jnp.ones(ev.size, bool), key, 0)
        dg, over = dc.digitise(resp, t, e, s, pt, w, ok, key, caps)
        hits, _, _ = dc.fit_and_hits(resp, t, dg, t["pbtime"], caps)
        cl, over_cl = dc.clusters(resp, hits, caps.events, caps)
        return dg, cl, over, over_cl

    t = {k: jnp.asarray(v) for k, v in dc.entries_batch({k: np.zeros((0, 3) if k in ("position", "direction") else 0)
                                                         for k in ("energy", "cosine", "radius", "disk", "time",
                                                                   "position", "direction")},
                                                        timing, np.zeros(0, int), caps.events)[1].items()}
    dg, cl, over, over_cl = jax.tree_util.tree_map(np.asarray, downstream(t, jax.random.PRNGKey(0)))
    assert not any(int(v) for v in over.values()) and not any(int(v) for v in over_cl.values())
    ok = dg["ok"]
    d = host["digis"]
    np.testing.assert_array_equal(dg["t0"][ok], d["t0"])
    np.testing.assert_array_equal(dg["length"][ok], np.diff(d["offsets"]))
    flat = np.concatenate([dg["samples"][i, :n] for i, n in zip(np.flatnonzero(ok), dg["length"][ok])])
    np.testing.assert_array_equal(flat, d["samples"])
    got, ref = dc.flat_clusters(cl), host["clusters"]
    assert len(ref["event"]) >= n_ev
    for k in ("event", "size", "disk", "split"):
        np.testing.assert_array_equal(got[k], ref[k])
    for k in ("energy", "x", "y", "z"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got["time"], ref["time"], atol=1e-3)


def test_full_chain_runs_and_conserves_scale(library):
    """Everything on (draws, PE statistics, travel, noise, smear): clusters carry the sampled visible energy."""
    pos, disk = _grid()
    geom = _geom(pos, disk)
    pulse = _pulse()
    table = PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32), np.float32(400.0))
    noise = np.random.default_rng(2).normal(12.0, 1.0, 20000)
    resp = dc.pack_response(_sim(len(disk)), pulse, noise, table, geom)
    lib = dc.pack_library(library, radius_neighbors=0, placement="crystal")
    query = _queries(60, seed=9)
    caps = dc.Caps(events=64, steps=2048, photoelectrons=1 << 19, rows=1024, digis=1024, digi_length=96,
                   hits=1024, event_hits=48, rounds=6)
    q, t = dc.entries_batch(query, _timing(60), np.arange(60), caps.events)
    chain = dc.make_chain(lib, resp, caps)
    out = dc.run(chain, q, t, jax.random.PRNGKey(7))
    cl = dc.flat_clusters(out["clusters"])
    assert out["stats"]["draws"] == 60 and out["stats"]["mapped"] > 0
    assert len(cl["event"]) > 0 and set(cl["event"]) <= set(range(60))
    host = sample(library, query, np.random.default_rng(0), orientation="disk")
    visible = np.bincount(host["entry"], weights=host["energy_birks"], minlength=60)
    # leading cluster energy against the donor's visible energy: same scale, not paired (independent draws)
    lead = np.zeros(64)
    np.maximum.at(lead, cl["event"], cl["energy"])
    assert 0.5 < lead[:60].sum() / visible.sum() < 1.5
    again = dc.run(chain, q, t, jax.random.PRNGKey(7))
    np.testing.assert_array_equal(again["clusters"]["energy"], out["clusters"]["energy"])
