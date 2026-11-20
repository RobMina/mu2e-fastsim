"""Entrant device chain (calofastsim.entrant_chain) against the host entrant sampler (calofastsim.showers.entrants)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("scipy")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from calofastsim import device_chain as dc  # noqa: E402
from calofastsim import entrant_chain as ec  # noqa: E402
from calofastsim.cluster import Geometry  # noqa: E402
from calofastsim.pulse import PulseCache  # noqa: E402
from calofastsim.readout import PropagationTable  # noqa: E402
from calofastsim.showers import entrants as E  # noqa: E402

PITCH, SIZE, DEPTH, INNER, OUTER, DZ, Z0 = 34.3, 34.0, 200.0, 374.0, 660.0, 700.0, -100.0
ENVELOPE = (340.0, 700.0)        # disk inner / outer radius of the envelope (geometry size_x / size_y)


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
    pos[:m, 2] = Z0
    pos[m:, 2] = DZ + Z0
    return pos, np.repeat([0, 1], m)


def _containing(pos, disk, p, d):
    inside = ((np.abs(p[:, None, :2] - pos[None, :, :2]) <= SIZE / 2).all(axis=2) &
              (p[:, None, 2] - pos[None, :, 2] >= 0) & (p[:, None, 2] - pos[None, :, 2] <= DEPTH) &
              (disk[None, :] == d[:, None]))
    return np.where(inside.any(axis=1), inside.argmax(axis=1), -1)


def _geometry(pos, disk):
    g = dict(disk_disk=np.array([0, 1]), disk_run=np.ones(2, int), crystal_crystal=np.arange(len(disk)) + 7,
             crystal_disk=disk, crystal_run=np.ones(len(disk), int),
             disk_size_x=np.full(2, ENVELOPE[0]), disk_size_y=np.full(2, ENVELOPE[1]), disk_size_z=np.full(2, 400.0))
    for k, axis in enumerate("xyz"):
        g["crystal_position_" + axis] = pos[:, k]
        g["crystal_size_" + axis] = np.full(len(disk), DEPTH if axis == "z" else SIZE)
        g["disk_origin_" + axis] = np.array([0.0, DZ]) if axis == "z" else np.zeros(2)
        for coord in "xyz":
            g["disk_axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    return g


def _states(rng, n, face=None):
    """Entry states in the disk frame: front entries (radius 350-690) and inner-edge entries (radius 340, depth in the
    crystals), CE-like energies; a few photons."""
    face = rng.choice([E.FRONT, E.INNER], n, p=[0.7, 0.3]) if face is None else np.full(n, face)
    disk = rng.integers(0, 2, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    r = np.where(face == E.INNER, ENVELOPE[0], rng.uniform(350.0, 690.0, n))
    z = np.where(face == E.INNER, rng.uniform(Z0 + 5, Z0 + DEPTH - 5, n), Z0 - 1.0)
    pos = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    p = rng.uniform(40.0, 108.0, n)
    cos = rng.uniform(0.3, 1.0, n)
    sin = np.sqrt(1 - cos ** 2)
    a = rng.uniform(-0.5, 0.5, n)
    # front: cos along the disk axis; inner edge: cos along the outward radius
    along_axis = np.column_stack((sin * np.cos(phi + a), sin * np.sin(phi + a), cos))
    radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(n)))
    tangential = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(n)))
    outward = cos[:, None] * radial + sin[:, None] * (np.cos(a)[:, None] * tangential + np.sin(a)[:, None] * [0, 0, 1])
    mom = p[:, None] * np.where((face == E.INNER)[:, None], outward, along_axis)
    pdg = np.where(rng.random(n) < 0.1, 22, 11)
    return dict(face=face, disk=disk, local_position=pos, local_momentum=mom, pdg=pdg,
                species=E.species_code(pdg), time=rng.uniform(500, 1500, n))


@pytest.fixture(scope="module")
def library():
    rng = np.random.default_rng(20261119)
    pos, disk = _grid()
    n = 500
    ent = _states(rng, n)
    ent["event_row"] = np.arange(n)
    rows, p, sd = [], [], []
    for i in range(n):
        k = int(rng.integers(0, 16))
        start = ent["local_position"][i] + np.array([0.0, 0.0, DZ * ent["disk"][i]])
        direction = ent["local_momentum"][i] / np.linalg.norm(ent["local_momentum"][i])
        q = start + direction * rng.uniform(1, 150, k)[:, None] + rng.normal(0, 20, (k, 3))
        other = rng.random(k) < 0.15
        q[other, 2] += DZ * (1 - 2 * ent["disk"][i])
        dd = np.where(other, 1 - ent["disk"][i], ent["disk"][i])
        c = _containing(pos, disk, q, dd)
        ok = c >= 0
        rows += [i] * int(ok.sum())
        p.append(q[ok])
        sd.append(dd[ok])
    p, sd = np.concatenate(p), np.concatenate(sd)
    m = len(rows)
    geometry = _geometry(pos, disk)
    css = dict(entrant=np.array(rows), position=p, time=ent["time"][rows] + rng.uniform(0, 3, m),
               energy_g4=rng.uniform(0.1, 20, m), energy_birks=rng.uniform(0.1, 20, m),
               crystal=geometry["crystal_crystal"][_containing(pos, disk, p, sd)], disk=sd)
    return E.build(ent, css, geometry)


class _Fixed:
    """A NumPy-generator stand-in that hands entrants.sample fixed uniforms."""

    def __init__(self, u):
        self.u = u

    def random(self, n):
        return self.u[:n]


def _queries(n=300, seed=5):
    rng = np.random.default_rng(seed)
    q = _states(rng, n)
    q["pdg"][:2] = -11                                  # no e+ in the library: unsupported, donor -1
    q["local_momentum"][2:4] *= 3.0                     # above the energy bins: fallback
    q["local_position"][4, :2] *= 0.95                  # a front entry below the first radius bin edge
    return q


def _batch(query, n_rows, n_events=None):
    n = len(query["face"])
    ent = dict(query, event=np.arange(n) if n_events is None else np.arange(n) % n_events)
    return ec.entrants_batch(ent, _timing(1), n_rows, n_events or n_rows)


def _stages(lib, q, u):
    @jax.jit
    def go(arrays, q, u):
        import dataclasses
        lb = dataclasses.replace(lib, arrays=arrays)
        donor, fallback, edge, slot, position = ec.sample_entrants(lb, q, u)
        qq = dict(q, position=position)
        st, over = dc.gather_steps(lb, qq, donor, slot, 8192)
        crystal, depth, mapped = dc.place_steps(lb, qq, st)
        return donor, fallback, edge, st, crystal, depth, mapped, over
    return jax.tree_util.tree_map(np.asarray, go(lib.arrays, {k: jnp.asarray(v) for k, v in q.items()}, jnp.asarray(u)))


@pytest.mark.parametrize("k,dmax,far", [(1, 20.0, "g4"), (3, 20.0, "cell"), (1, 1e4, "g4"), (2, 60.0, "g4")])
def test_nearest_draw_matches_host(library, k, dmax, far):
    """selection="nearest" (the Mu2eG4 host's donorSelection: nearest): device donors, far flags and placed steps
    equal the host sampler's."""
    query = _queries()
    u = np.random.default_rng(13).random(320)
    host = E.sample(library, query, _Fixed(u), selection="nearest", nearest_k=k, max_entry_distance=dmax, far=far,
                    key_neighbors=3)
    lib = ec.pack_entrants(library, placement="geometric", key_neighbors=3, selection="nearest", nearest_k=k,
                           max_entry_distance=dmax, far=far)
    assert lib.meta["window"] >= 1 and lib.meta["selection"] == "nearest"
    q, _ = _batch(query, 320)

    @jax.jit
    def go(arrays, q, u):
        import dataclasses
        lb = dataclasses.replace(lib, arrays=arrays)
        return ec.sample_entrants(lb, q, u, return_far=True)
    donor, fallback, edge, slot, position, far_dev = jax.tree_util.tree_map(
        np.asarray, go(lib.arrays, {kk: jnp.asarray(v) for kk, v in q.items()}, jnp.asarray(u)))
    np.testing.assert_array_equal(donor[:300], host["donor"])
    np.testing.assert_array_equal(far_dev[:300], host["far"])
    np.testing.assert_array_equal(edge[:300], host["far"] if far == "g4" else np.zeros(300, bool))
    ok = host["donor"] >= 0
    np.testing.assert_array_equal(fallback[:300][ok], host["fallback"][ok])
    assert ok.sum() > (100 if far == "cell" or dmax > 1e3 else 0)
    if far == "g4":
        assert (host["donor"][host["far"]] == -1).all()
    _, _, _, st, _, _, _, over = _stages(lib, q, u)
    n = len(host["entry"])
    assert int(over) == 0 and st["valid"].sum() == n
    np.testing.assert_allclose(st["position"][:n], host["position"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nearest_k1_argmin_picks_the_lexsort_donors(library, dtype):
    """k = 1: the device's argmin path (default) gives the same donors and far flags as the sorted-window path, in
    both dtypes, for several distances and both far rules, and the host's donors."""
    import dataclasses
    query = _queries(300, seed=17)
    u = np.random.default_rng(19).random(320)
    q, _ = _batch(query, 320)
    qd = {kk: jnp.asarray(v) for kk, v in q.items()}
    for dmax, far in ((5.0, "g4"), (20.0, "cell"), (80.0, "g4"), (1e4, "cell")):
        lib = ec.pack_entrants(library, key_neighbors=3, selection="nearest", nearest_k=1, max_entry_distance=dmax,
                               far=far, dtype=dtype)
        sorted_lib = dataclasses.replace(lib, meta=dict(lib.meta, nearest_argmin=False))
        out = [jax.tree_util.tree_map(np.asarray, jax.jit(
            lambda arrays, q, u, lb=lb: ec.sample_entrants(dataclasses.replace(lb, arrays=arrays), q, u,
                                                           return_far=True))(lb.arrays, qd, jnp.asarray(u)))
               for lb in (lib, sorted_lib)]
        for got, want in zip(out[0], out[1]):
            np.testing.assert_array_equal(got, want)
        host = E.sample(library, query, _Fixed(u), selection="nearest", nearest_k=1, max_entry_distance=dmax,
                        far=far, key_neighbors=3, dtype=dtype)
        np.testing.assert_array_equal(out[0][0][:300], host["donor"])
        assert (out[0][0][:300] >= 0).sum() > 0


def test_nearest_distance_counts_the_half_turn_image():
    """entry_distance2: the 180-degree image of a donor is at distance 0; front distances are chords on the face,
    inner-edge distances sqrt(dz^2 + (rho dphi)^2) with dphi folded into [-pi/2, pi/2]."""
    d = E.entry_distance2(np.array([E.FRONT, E.FRONT, E.INNER, E.INNER]), np.array([400.0, 400.0, 30.0, 30.0]),
                          np.array([0.1, 0.1, 0.2, 0.2]), np.array([400.0, 400.0, 336.0, 336.0]),
                          np.array([400.0, 410.0, 30.0, 25.0]), np.array([0.1 + np.pi, 0.1, 0.2 - np.pi, 0.21]))
    np.testing.assert_allclose(np.sqrt(d), [0.0, 10.0, 0.0, np.hypot(5.0, 336.0 * 0.01)], atol=1e-9)
    assert E.fold_half_turn(np.array([3 * np.pi / 4]))[0] == pytest.approx(-np.pi / 4)


@pytest.mark.parametrize("neighbors,psi", [(0, None), (3, None), (3, E.PSI_BINS_15)])
def test_draw_and_steps_match_host(library, neighbors, psi):
    query = _queries()
    u = np.random.default_rng(11).random(320)
    host = E.sample(library, query, _Fixed(u), key_neighbors=neighbors, psi_bins=psi)
    lib = ec.pack_entrants(library, placement="geometric", key_neighbors=neighbors, psi_bins=psi)
    q, _ = _batch(query, 320)
    donor, fallback, edge, st, crystal, depth, mapped, over = _stages(lib, q, u)
    np.testing.assert_array_equal(donor[:300], host["donor"])
    ok = host["donor"] >= 0
    assert ok.sum() > 250 and (~ok).sum() >= 2
    np.testing.assert_array_equal(fallback[:300][ok], host["fallback"][ok])
    assert host["fallback"][2:4].all() and not edge.any()
    n = len(host["entry"])
    assert int(over) == 0 and st["valid"].sum() == n
    np.testing.assert_array_equal(st["row"][:n], host["entry"])
    np.testing.assert_allclose(st["position"][:n], host["position"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(st["time"][:n], host["time"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(st["energy_birks"][:n], host["energy_birks"])
    np.testing.assert_array_equal(st["disk"][:n], host["disk"])
    # geometric placement: the crystal that contains each placed step
    pos, disk = _grid()
    want = _containing(pos, disk, host["position"], host["disk"])
    np.testing.assert_array_equal(np.where(mapped[:n], crystal[:n], -1), want)
    np.testing.assert_allclose(depth[:n][mapped[:n]], host["position"][want >= 0, 2] - pos[want[want >= 0], 2], atol=1e-9)
    assert 0 < mapped[:n].sum() < n


def test_host_key_neighbors_rule(library):
    """K >= every cell size keeps the uniform draw; K = 1 takes the entrant nearest in the key (ties: lowest row)."""
    query = _queries(seed=31)
    u = np.random.default_rng(5).random(300)
    plain = E.sample(library, query, _Fixed(u))
    wide = E.sample(library, query, _Fixed(u), key_neighbors=10_000)
    np.testing.assert_array_equal(wide["donor"], plain["donor"])
    one = E.sample(library, query, _Fixed(u), key_neighbors=1)
    _, _, key, _ = E.entry_features(query["face"], query["local_position"], query["local_momentum"], query["pdg"])
    labels = np.column_stack((library["entry_species"], library["entry_face"], library["entry_disk"],
                              library["entry_cell"]))
    moved = 0
    for i in np.flatnonzero(one["donor"] >= 0):
        pool = np.flatnonzero(np.all(labels == labels[plain["donor"][i]], axis=1))
        d = np.abs(library["entry_key"][pool] - key[i])
        best = pool[np.flatnonzero(d == d.min())[0]]
        assert one["donor"][i] == best
        moved += one["donor"][i] != plain["donor"][i]
    assert moved > 0
    with pytest.raises(ValueError):
        E.sample(library, query, _Fixed(u), key_neighbors=-1)


def test_psi_cells_split_front_entries_by_transverse_direction(library):
    """psi bins keep a front query's donor in its transverse-direction bin (or the nearest occupied one)."""
    query = _queries(seed=41)
    query["face"][:] = E.FRONT
    query["local_position"][:, 2] = Z0 - 1.0
    query["pdg"][:] = 11
    u = np.random.default_rng(6).random(300)
    psi = E.transverse_angle(query["local_position"], query["local_momentum"])
    out = E.sample(library, query, _Fixed(u), psi_bins=E.PSI_BINS_15)
    ok = (out["donor"] >= 0) & ~out["fallback"]
    assert ok.sum() > 20
    np.testing.assert_array_equal(E.psi_cells(library["entry_face"][out["donor"][ok]],
                                              library["entry_psi"][out["donor"][ok]], E.PSI_BINS_15),
                                  E.psi_cells(query["face"][ok], psi[ok], E.PSI_BINS_15))
    old = {k: v for k, v in library.items() if k not in ("entry_psi", "entry_local_momentum")}
    with pytest.raises(ValueError):
        E.sample(old, query, _Fixed(u), psi_bins=E.PSI_BINS_15)
    with pytest.raises(ValueError):
        ec.pack_entrants(old, psi_bins=E.PSI_BINS_15)


def _own_uniforms(library, lib, rows):
    """Uniforms that draw each library entrant in ``rows`` at its own entry."""
    a = {k: np.asarray(v) for k, v in lib.arrays.items()}
    labels = np.column_stack((library["entry_species"], library["entry_face"], library["entry_disk"], library["entry_cell"]))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    u = np.zeros(len(rows))
    for i, j in enumerate(rows):
        c = inverse[j]
        members = a["order"][a["starts"][c]:a["starts"][c] + a["counts"][c]]
        u[i] = (np.flatnonzero(members == j)[0] + 0.5) / a["counts"][c]
    return u


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_crystal_placement_returns_the_donor_crystals_at_the_donor_entry(library, dtype):
    rows = np.arange(0, 500, 3)
    query = dict(face=library["entry_face"][rows], disk=library["entry_disk"][rows], pdg=library["entry_pdg"][rows],
                 time=library["entry_time"][rows], local_position=library["entry_local_position"][rows])
    # the library keeps the entry momentum only through its features: rebuild one with the same energy and cosine
    e, c, phi = library["entry_energy"][rows], library["entry_cosine"][rows], library["entry_azimuth"][rows]
    s = np.sqrt(1 - c ** 2)
    radial, tangential = np.column_stack((np.cos(phi), np.sin(phi), 0 * phi)), np.column_stack((-np.sin(phi), np.cos(phi), 0 * phi))
    inner = (query["face"] == E.INNER)[:, None]
    direction = np.where(inner, c[:, None] * radial + s[:, None] * tangential,
                         s[:, None] * radial + c[:, None] * np.array([0.0, 0.0, 1.0]))
    mass = np.where(np.abs(query["pdg"]) == 11, E.MASS[11], 0.0)
    query["local_momentum"] = np.sqrt(np.maximum(e ** 2 - mass ** 2, 0))[:, None] * direction
    lib = ec.pack_entrants(library, placement="crystal", dtype=dtype)
    q, _ = _batch(query, len(rows))
    u = _own_uniforms(library, lib, rows)
    donor, _, _, st, crystal, depth, mapped, _ = _stages(lib, q, u)
    np.testing.assert_array_equal(donor, rows)
    n = int(st["valid"].sum())
    src = np.concatenate([np.arange(library["offsets"][j], library["offsets"][j + 1]) for j in rows])
    assert n == len(src) > 0
    ids = library["geometry_crystal_crystal"]
    m = mapped[:n]
    assert m.all() if dtype == np.float64 else m.mean() > 0.999      # float32: steps within roundoff of a face
    np.testing.assert_array_equal(ids[crystal[:n][m]], library["step_crystal"][src][m])
    tol = 1e-9 if dtype == np.float64 else 0.05
    pos, disk = _grid()
    want_depth = st["position"][:n, 2].astype(np.float64) - pos[crystal[:n], 2]
    np.testing.assert_allclose(depth[:n][m], want_depth[m], atol=tol)


def test_float32_steps_close_to_host(library):
    query = _queries(seed=8)
    u = np.random.default_rng(3).random(300)
    host = E.sample(library, query, _Fixed(u))
    lib = ec.pack_entrants(library, placement="crystal", dtype=np.float32)
    q, _ = _batch(query, 300)
    donor, _, _, st, crystal, depth, mapped, _ = _stages(lib, q, u)
    np.testing.assert_array_equal(donor, host["donor"])
    n = len(host["entry"])
    assert st["position"].dtype == np.float32
    np.testing.assert_allclose(st["position"][:n], host["position"], rtol=0, atol=0.05)
    np.testing.assert_allclose(st["time"][:n], host["time"], rtol=0, atol=1e-9)
    lib64 = ec.pack_entrants(library, placement="crystal")
    _, _, _, _, c64, _, m64, _ = _stages(lib64, q, u)
    agree = (np.where(mapped[:n], crystal[:n], -1) == np.where(m64[:n], c64[:n], -1)).mean()
    assert agree > 0.99


def test_edge_rule_matches_host_and_leaves_the_entrant_undrawn(library):
    query = _queries(seed=12)
    lib = ec.pack_entrants(library, d_edge=40.0)
    q, _ = _batch(query, 300)
    donor, _, edge, *_ = _stages(lib, q, np.random.default_rng(2).random(300))
    want = ec.near_edge(query["face"], query["local_position"], ENVELOPE[0], ENVELOPE[1], 40.0)
    assert 0 < want.sum() < 300
    np.testing.assert_array_equal(edge, want)
    assert (donor[edge] == -1).all()
    with pytest.raises(ValueError):
        ec.pack_entrants(library, d_edge=-1.0)


def _pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _geom(pos, disk):
    near = ((np.abs(pos[:, None, 0] - pos[None, :, 0]) < 1.5 * PITCH) &
            (np.abs(pos[:, None, 1] - pos[None, :, 1]) < 1.5 * PITCH) & (disk[:, None] == disk[None, :]))
    return Geometry(pos, disk, near & ~np.eye(len(disk), dtype=bool), np.array([[0.0, 0.0, Z0], [0.0, 0.0, DZ + Z0]]))


def _sim(n):
    return {"lru": np.zeros(n, np.float32), "pe_per_mev": np.full((n, 2), 30.0, np.float32),
            "adc_per_mev": np.full((n, 2), 16.0, np.float32)}


def _timing(n):
    return {"pbtime": np.full(n, -200.0, np.float32), "mbtime": np.full(n, 1695.0, np.float32),
            "event_length": np.full(n, 1700.0), "onspill": np.ones(n, bool),
            "marker": np.full(n, 200.0, np.float32), "crystal_length": np.full(n, 200.0, np.float32)}


@pytest.mark.parametrize("digi_response", ["parametric", "waveform"])
def test_chain_several_entrants_per_event(library, digi_response):
    """Two entrants per event; clusters carry the drawn visible energy; same key, same clusters."""
    pos, disk = _grid()
    resp = dc.pack_response(_sim(len(disk)), _pulse(), np.random.default_rng(2).normal(12.0, 1.0, 20000),
                            PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32),
                                             np.float32(400.0)), _geom(pos, disk))
    lib = ec.pack_entrants(library)
    query = _queries(60, seed=9)
    query["pdg"][:] = 11
    query["local_momentum"][2:4] *= 0.5
    caps = dc.Caps(events=32, steps=2048, photoelectrons=1 << 19, rows=1024, digis=1024, digi_length=96,
                   hits=1024, event_hits=48, rounds=6)
    q, t = ec.entrants_batch(dict(query, event=np.arange(60) // 2), _timing(30), 64, caps.events)
    chain = ec.make_entrant_chain(lib, resp, caps, digi_response=digi_response, truth=True)
    out = ec.run(chain, q, t, jax.random.PRNGKey(7))
    cl = dc.flat_clusters(out["clusters"])
    assert out["stats"]["draws"] == 60 and out["stats"]["mapped"] > 0 and out["stats"]["g4_steps"] == 0
    assert len(cl["event"]) > 0 and set(cl["event"]) <= set(range(30))
    host = E.sample(library, query, np.random.default_rng(0))
    ev = np.arange(60)[host["entry"]] // 2
    visible = np.bincount(ev, weights=host["energy_birks"], minlength=30)
    lead = np.zeros(32)
    np.maximum.at(lead, cl["event"], cl["energy"])
    assert 0.3 < lead[:30].sum() / visible.sum() < 1.5
    again = ec.run(chain, q, t, jax.random.PRNGKey(7))
    np.testing.assert_array_equal(again["clusters"]["energy"], out["clusters"]["energy"])


def test_g4_steps_join_the_drawn_steps(library):
    """Edge entrants left to full simulation: their Geant4 steps, passed as g4, are clustered with the drawn ones."""
    pos, disk = _grid()
    resp = dc.pack_response(_sim(len(disk)), _pulse(), None,
                            PropagationTable(np.ones((1, 1), np.float32), np.zeros(1, np.float32), np.float32(1.0)),
                            _geom(pos, disk), add_noise=False)
    lib = ec.pack_entrants(library, d_edge=1e4)                  # every front entrant is an edge entrant
    query = _queries(40, seed=21)
    query["face"][:] = E.FRONT
    query["local_position"][:, 2] = Z0 - 1.0
    query["pdg"][:] = 11
    caps = dc.Caps(events=40, steps=1024, rows=1024, digis=1024, hits=1024, event_hits=48, rounds=6)
    q, t = ec.entrants_batch(dict(query, event=np.arange(40)), _timing(40), 40, caps.events)
    host = E.sample(library, query, np.random.default_rng(4))
    m = _containing(pos, disk, host["position"], host["disk"])
    ok = m >= 0
    g4 = dc.steps_batch(dict(event=host["entry"][ok], crystal=m[ok], energy_g4=host["energy_g4"][ok],
                             energy_birks=host["energy_birks"][ok], time=host["time"][ok],
                             z_local=host["position"][ok, 2] - pos[m[ok], 2]), 2048)
    chain = ec.make_entrant_chain(lib, resp, caps)
    out = ec.run(chain, q, t, jax.random.PRNGKey(3), g4=g4)
    assert out["stats"]["draws"] == 0 and out["stats"]["edge"] == 40 and out["stats"]["g4_steps"] == ok.sum()
    got = dc.flat_clusters(out["clusters"])
    assert len(got["event"]) > 0
    # visible energy is conserved into clusters on the same scale as the steps
    assert 0.3 < got["energy"].sum() / host["energy_birks"][ok].sum() < 1.5
    # steps tagged with their entrant row: only the steps of entrants left to Geant4 join
    tagged = dict(g4, entrant=np.full(2048, -1, np.int32))
    tagged["entrant"][:ok.sum()] = host["entry"][ok]
    out_t = ec.run(chain, q, t, jax.random.PRNGKey(3), g4=tagged)
    assert out_t["stats"]["g4_steps"] == ok.sum()
    drawn = ec.make_entrant_chain(ec.pack_entrants(library), resp, caps)        # no edge entrants
    out_d = ec.run(drawn, q, t, jax.random.PRNGKey(3), g4=tagged)
    assert out_d["stats"]["g4_steps"] == 0 and out_d["stats"]["edge"] == 0


def test_bench_entry_point_runs(tmp_path, monkeypatch):
    """calofastsim.bench entrant entry points: entrants of export tables -> clusters through the shared-timer path."""
    from calofastsim import bench
    from test_showers_entrants import tables, write
    write(tmp_path / "a.root", tables())
    ent, css, geo, _ = E.read(tmp_path / "a.root")
    E.save(tmp_path / "lib.npz", E.build(ent, css, geo), dict(name="bench-test"))
    monkeypatch.setenv("CALO_BENCH_TABLES", str(tmp_path / "a.root"))
    monkeypatch.setenv("CALO_BENCH_LIBRARY", str(tmp_path / "lib.npz"))
    events = bench.load_entrants([tmp_path / "a.root"])
    assert [len(e["face"]) for e in events] == [2, 1, 0]       # event 1: CE front + photon inner; event 2: CE inner
    pos = np.column_stack([geo["crystal_position_" + a] for a in "xyz"])
    geom = Geometry(pos, geo["crystal_disk"], np.zeros((3, 3), bool), np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]]))

    def conditions(dtype="float64", directory=None, fit_hessian="analytic"):
        return dc.pack_response(_sim(3), _pulse(), np.full(10000, 12.0),
                                PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32),
                                                 np.float32(400.0)), geom, dtype=np.dtype(dtype), fit_hessian=fit_hessian)
    monkeypatch.setattr(bench, "conditions", conditions)
    monkeypatch.setattr(bench, "_ENTRANT_CHAINS", {})
    prepared = bench.bench_prepare_entrants(6, seed=5, batch=4)
    assert len(prepared["batches"]) == 2
    for dtype in ("float64", "float32"):
        outs = bench.bench_entrants(prepared, dtype=dtype, key_neighbors=8, rows=64, hits=64)
        assert len(outs) == 2 and not any(bench.overflow(outs).values())
        near = bench.bench_entrants(prepared, dtype=dtype, key_neighbors=8, rows=64, hits=64, selection="nearest",
                                    far="cell")
        assert len(near) == 2 and not any(bench.overflow(near).values())
    # the chain cache is keyed by the library path: a library swapped within the process is not silently reused
    import shutil
    shutil.copy(tmp_path / "lib.npz", tmp_path / "lib2.npz")
    first = bench.entrant_chain("float64", batch=4, key_neighbors=8, rows=64, hits=64)
    monkeypatch.setenv("CALO_BENCH_LIBRARY", str(tmp_path / "lib2.npz"))
    assert bench.entrant_chain("float64", batch=4, key_neighbors=8, rows=64, hits=64) is not first


def test_read_skips_failed_events_on_request(tmp_path):
    """entrants.read(skip_failed=True): a failed-transport event and its rows are dropped; the rest is unchanged."""
    from test_showers_entrants import tables, write
    data = tables()
    write(tmp_path / "ok.root", data)
    data["events"]["g4_status"] = np.array([0, 1, 0], dtype=np.int32)       # event 2 (inner-edge CE) failed
    write(tmp_path / "bad.root", data)
    with pytest.raises(ValueError):
        E.read(tmp_path / "bad.root")
    ent, css, _, cen = E.read(tmp_path / "bad.root", skip_failed=True)
    ref, rcss, _, rcen = E.read(tmp_path / "ok.root")
    assert cen["failed_events_skipped"] == 1 and rcen["failed_events_skipped"] == 0
    assert cen["events"] == 2 and len(ent["face"]) == len(ref["face"]) - 1
    keep = ref["event_row"] != 1
    np.testing.assert_array_equal(ent["face"], ref["face"][keep])
    np.testing.assert_array_equal(ent["event_row"], np.where(ref["event_row"][keep] > 1, ref["event_row"][keep] - 1,
                                                             ref["event_row"][keep]))
    # read_many threads skip_failed through; event rows of the second file are offset by the completed events
    with pytest.raises(ValueError):
        E.read_many([tmp_path / "ok.root", tmp_path / "bad.root"])
    many, mcss, _, mcen = E.read_many([tmp_path / "bad.root", tmp_path / "ok.root"], skip_failed=True)
    assert [c["failed_events_skipped"] for c in mcen] == [1, 0]
    np.testing.assert_array_equal(many["event_row"], np.concatenate([ent["event_row"], ref["event_row"] + 2]))
    assert len(mcss["entrant"]) == len(css["entrant"]) + len(rcss["entrant"])


def test_w7_default_settings_match_the_bench():
    """W7's default shower source: the bench entry point's defaults are entrant_chain.W7_DEFAULT."""
    import inspect
    from calofastsim import bench
    sig = inspect.signature(bench.bench_entrants).parameters
    assert sig["key_neighbors"].default == ec.W7_DEFAULT["key_neighbors"] == 8
    assert sig["placement"].default == ec.W7_DEFAULT["placement"] == "crystal"
    assert ec.W7_DEFAULT["d_edge"] == 0.0 and ec.W7_DEFAULT["psi_bins"] is None
    assert len(ec.W7_LIBRARY_HASH) == 64 and ec.W7_LIBRARY_HASH.startswith("406bd7b4")


def test_w7_arm_writes_the_scoring_clusters(tmp_path, monkeypatch):
    """calofastsim.w7_arm: entrants of exports -> clusters.npz with the W7 populations; failed events dropped; the
    library hash is checked."""
    from calofastsim import w7_arm
    from calofastsim.showers import export
    from test_showers_entrants import tables, write
    data = tables()
    write(tmp_path / "a.root", data)
    ent, css, geo, _ = E.read(tmp_path / "a.root")
    E.save(tmp_path / "lib.npz", E.build(ent, css, geo), dict(name="w7-arm-test"))
    data["events"]["g4_status"] = np.array([0, 0, 1], dtype=np.int32)          # event 3 failed in the second export
    write(tmp_path / "b.root", data)
    pos = np.column_stack([geo["crystal_position_" + a] for a in "xyz"])
    geom = Geometry(pos, geo["crystal_disk"], np.zeros((3, 3), bool), np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]]))

    def response(ci, dtype):
        return dc.pack_response(_sim(3), _pulse(), np.full(10000, 12.0),
                                PropagationTable(np.array([[0.5, 1.0]], np.float32), np.array([0.5, 1.5], np.float32),
                                                 np.float32(400.0)), geom, dtype=np.dtype(dtype))
    monkeypatch.setattr(w7_arm, "response", response)
    monkeypatch.setattr(w7_arm, "stock_bunch_times",
                        lambda path: (np.array([[1, 0, 1], [1, 0, 2], [1, 0, 3]]), np.array([-210.0, -205.0, -201.0])))
    # primary mode needs the production primary identity, absent from these tables: event 1 is the primary front entry
    monkeypatch.setattr(export, "read", lambda path, input_mode: ({"event_row": np.array([0])}, None, None, None, None))
    args = ["--tables", str(tmp_path / "a.root"), str(tmp_path / "b.root"), "--stock", "sa", "sb",
            "--library", str(tmp_path / "lib.npz"), "--cal1-inputs", str(tmp_path), "--batch", "4"]
    with pytest.raises(SystemExit):
        w7_arm.main([str(tmp_path / "refused")] + args)                       # not W7's library
    for dtype in ("float64", "float32"):
        out = tmp_path / dtype
        s = w7_arm.main([str(out)] + args + ["--any-library", "--dtype", dtype])
        z = np.load(out / "clusters.npz")
        np.testing.assert_array_equal(z["all_ids"], [[1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 0, 1], [1, 0, 2]])
        np.testing.assert_array_equal(z["front_ids"], [[1, 0, 1], [1, 0, 1]])
        assert s["front_definition"] == ["primary disk-front entry", "front-face e- entrant"]
        assert [c["failed_events_skipped"] for c in s["census"]] == [0, 1]
        assert s["stats"]["draws"] == 6 and not s["w7_library"] and not any(s["overflow"].values())
        assert s["settings"] == {"placement": "crystal", "key_neighbors": 8, "d_edge": 0.0}
        ids = {tuple(r) for r in z["all_ids"]}
        assert len(z["ids"]) == len(z["energy"]) and all(tuple(r) in ids for r in z["ids"])
    # the nearest rule with Geant4 steps for far entrants: each entrant finds its own library entry (distance 0)
    s = w7_arm.main([str(tmp_path / "nearest")] + args + ["--any-library", "--selection", "nearest"])
    assert s["stats"]["draws"] == 6 and s["stats"]["far"] == 0 and s["stats"]["g4_steps"] == 0
    assert s["settings"]["selection"] == "nearest" and not any(s["overflow"].values())
