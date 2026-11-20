"""Entrant sampler's nearest-entry donor rule (entrants.sample entry_neighbors, the Mu2eG4 model's default)."""
import math

import numpy as np
import pytest

from calofastsim.showers import entrants as E

PITCH, SIZE, DEPTH, INNER, OUTER, DZ, Z0 = 34.3, 34.0, 200.0, 374.0, 660.0, 700.0, -100.0
ENVELOPE = (340.0, 700.0)


def _grid():
    n = int(np.ceil(OUTER / PITCH)) + 1
    c = np.arange(-n, n + 1) * PITCH
    x, y = [v.ravel() for v in np.meshgrid(c, c, indexing="ij")]
    r = np.hypot(x, y)
    keep = (r - SIZE / np.sqrt(2) >= INNER) & (r + SIZE / np.sqrt(2) <= OUTER)
    m = int(keep.sum())
    pos = np.zeros((2 * m, 3))
    pos[:m, :2] = pos[m:, :2] = np.column_stack((x[keep], y[keep]))
    pos[:m, 2], pos[m:, 2] = Z0, DZ + Z0
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


def _states(rng, n):
    """CE-like electrons entering the front face (radius 350-690) or the inner edge (radius 340)."""
    face = rng.choice([E.FRONT, E.INNER], n, p=[0.7, 0.3])
    disk = rng.integers(0, 2, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    r = np.where(face == E.INNER, ENVELOPE[0], rng.uniform(350.0, 690.0, n))
    z = np.where(face == E.INNER, rng.uniform(Z0 + 5, Z0 + DEPTH - 5, n), Z0 - 1.0)
    pos = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    p = rng.uniform(96.0, 108.0, n)
    cos = rng.uniform(0.75, 1.0, n)
    sin = np.sqrt(1 - cos ** 2)
    a = rng.uniform(-0.5, 0.5, n)
    along_axis = np.column_stack((sin * np.cos(phi + a), sin * np.sin(phi + a), cos))
    radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(n)))
    tangential = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(n)))
    outward = cos[:, None] * radial + sin[:, None] * (np.cos(a)[:, None] * tangential + np.sin(a)[:, None] * [0, 0, 1])
    mom = p[:, None] * np.where((face == E.INNER)[:, None], outward, along_axis)
    pdg = np.full(n, 11)
    return dict(face=face, disk=disk, local_position=pos, local_momentum=mom, pdg=pdg,
                species=E.species_code(pdg), time=rng.uniform(500, 1500, n))


@pytest.fixture(scope="module")
def lattice():
    """A library on a square crystal lattice (the Mu2e disk's symmetry: 180-degree turn and mirrors, not 90)."""
    rng = np.random.default_rng(20261120)
    pos, disk = _grid()
    n = 2000
    ent = _states(rng, n)
    ent["event_row"] = np.arange(n)
    rows, p, sd = [], [], []
    for i in range(n):
        k = int(rng.integers(4, 24))
        start = ent["local_position"][i] + np.array([0.0, 0.0, DZ * ent["disk"][i]])
        direction = ent["local_momentum"][i] / np.linalg.norm(ent["local_momentum"][i])
        q = start + direction * rng.uniform(1, 150, k)[:, None] + rng.normal(0, 15, (k, 3))
        dd = np.full(k, ent["disk"][i])
        ok = _containing(pos, disk, q, dd) >= 0
        rows += [i] * int(ok.sum())
        p.append(q[ok])
        sd.append(dd[ok])
    p, sd = np.concatenate(p), np.concatenate(sd)
    m = len(rows)
    geometry = _geometry(pos, disk)
    css = dict(entrant=np.array(rows), position=p, time=ent["time"][rows] + rng.uniform(0, 3, m),
               energy_g4=rng.uniform(0.1, 20, m), energy_birks=rng.uniform(0.1, 20, m),
               crystal=geometry["crystal_crystal"][_containing(pos, disk, p, sd)], disk=sd)
    return E.build(ent, css, geometry), ent, (pos, disk)


def _subset(ent, idx):
    return {k: np.asarray(v)[idx] for k, v in ent.items() if k in ("face", "disk", "local_position", "local_momentum",
                                                                    "pdg", "time")}


def _turn(q, angle):
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return dict(q, local_position=q["local_position"] @ r.T, local_momentum=q["local_momentum"] @ r.T)


def test_entry_distance2_against_points():
    rng = np.random.default_rng(1)
    kq, kd = rng.uniform(350, 690, 40), rng.uniform(350, 690, 60)
    pq, pd = rng.uniform(-np.pi, np.pi, 40), rng.uniform(-np.pi, np.pi, 60)
    front = E.entry_distance2(E.FRONT, kq, pq, kq, kd, pd)
    a = np.column_stack((kq * np.cos(pq), kq * np.sin(pq)))
    b = np.column_stack((kd * np.cos(pd), kd * np.sin(pd)))
    direct = np.minimum(((a[:, None] - b[None]) ** 2).sum(-1), ((a[:, None] + b[None]) ** 2).sum(-1))
    np.testing.assert_allclose(front, direct, rtol=1e-9, atol=1e-6)
    zq, zd, rho = rng.uniform(-190, 190, 40), rng.uniform(-190, 190, 60), np.full(40, 340.)
    inner = E.entry_distance2(E.INNER, zq, pq, rho, zd, pd)
    want = np.array([[(zq[i] - zd[j]) ** 2 + (340. * math.remainder(pq[i] - pd[j], math.pi)) ** 2
                      for j in range(60)] for i in range(40)])
    np.testing.assert_allclose(inner, want, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("k", [1, 3, 7, 1000])
def test_nearest_entries_against_brute_force(k):
    rng = np.random.default_rng(k)
    rows = np.sort(rng.choice(5000, 300, replace=False))
    dk, dp = rng.uniform(350, 690, 300), rng.uniform(-np.pi, np.pi, 300)
    dk[5], dp[5] = dk[4], dp[4] + np.pi                     # a tie: the lower row wins at equal distance
    qk, qp, u = rng.uniform(350, 690, 50), rng.uniform(-np.pi, np.pi, 50), rng.random(50)
    qk[0], qp[0] = dk[4], dp[4]
    donors, distance = E.nearest_entries(E.FRONT, qk, qp, qk, rows, dk, dp, k, u, chunk=7 * 300)
    d2 = E.entry_distance2(E.FRONT, qk, qp, qk, dk, dp)
    n = min(k, len(rows))
    for i in range(50):
        ranked = sorted(range(len(rows)), key=lambda j: (d2[i, j], rows[j]))[:n]
        pick = ranked[min(int(u[i] * n), n - 1)]
        assert donors[i] == rows[pick]
        assert distance[i] == pytest.approx(np.sqrt(d2[i, pick]), abs=1e-9)
    if k == 1:
        assert donors[0] == rows[4] and distance[0] == pytest.approx(0., abs=1e-6)


def test_sample_takes_the_nearest_entry_of_the_energy_cosine_group(lattice):
    library, ent, _ = lattice
    rng = np.random.default_rng(7)
    q = _states(rng, 400)
    out = E.sample(library, q, rng, entry_neighbors=1)
    energy, cosine, key, phi = E.entry_features(q["face"], q["local_position"], q["local_momentum"], q["pdg"])
    cell, outside = E.cells(q["face"], energy, cosine, key, E._binning(library))
    rho = np.hypot(q["local_position"][:, 0], q["local_position"][:, 1])
    labels = np.column_stack((library["entry_species"], library["entry_face"], library["entry_disk"],
                              library["entry_cell"][:, :2]))
    assert (out["donor"] >= 0).sum() > 350 and not out["fallback"].any()
    for i in np.flatnonzero(out["donor"] >= 0):
        pool = np.flatnonzero(np.all(labels == (0, q["face"][i], q["disk"][i], *cell[i, :2]), axis=1))
        d2 = E.entry_distance2(int(q["face"][i]), key[i:i + 1], phi[i:i + 1], rho[i:i + 1],
                               library["entry_key"][pool], library["entry_azimuth"][pool])[0]
        assert out["donor"][i] == pool[np.lexsort((pool, d2))[0]]
        assert out["distance"][i] == pytest.approx(np.sqrt(d2.min()), abs=1e-9)
    assert np.isnan(out["distance"][out["donor"] < 0]).all()


def test_turned_query_reuses_the_donor_turned_by_180_degrees(lattice):
    library, ent, _ = lattice
    idx = np.arange(0, 2000, 37)
    own = _subset(ent, idx)
    u = np.random.default_rng(3).random(len(idx))

    class Fixed:
        def random(self, n):
            return u[:n]

    self_out = E.sample(library, own, Fixed(), entry_neighbors=1, max_entry_distance=1e-6)
    np.testing.assert_array_equal(self_out["donor"], idx)
    turned = E.sample(library, _turn(own, np.pi), Fixed(), entry_neighbors=1, max_entry_distance=1e-6)
    np.testing.assert_array_equal(turned["donor"], idx)
    np.testing.assert_array_equal(turned["offsets"], self_out["offsets"])
    flip = self_out["position"].copy()
    flip[:, :2] *= -1                                                    # 180 degrees about the disk axis (x=y=0)
    np.testing.assert_allclose(turned["position"], flip, atol=1e-9)


def test_nearest_keeps_more_energy_on_the_lattice_than_the_cell_rule(lattice):
    library, _, (pos, disk) = lattice
    rng = np.random.default_rng(11)
    q = _states(rng, 1500)
    u = rng.random(1500)

    class Fixed:
        def random(self, n):
            return u[:n]

    def contained(out):
        inside = _containing(pos, disk, out["position"], out["disk"]) >= 0
        return float(out["energy_g4"][inside].sum() / out["energy_g4"].sum())

    cell = contained(E.sample(library, q, Fixed()))
    near = contained(E.sample(library, q, Fixed(), entry_neighbors=1))
    assert near > cell + 0.01, (near, cell)


def test_cuts_dtype_and_arguments(lattice):
    library, ent, _ = lattice
    idx = np.arange(1, 2000, 41)
    own = _subset(ent, idx)
    rng = np.random.default_rng(2)
    front = own["face"] == E.FRONT
    moved = dict(own, local_position=own["local_position"].copy())
    moved["local_position"][front, :2] *= 1 + 30. / np.hypot(*own["local_position"][front, :2].T)[:, None]
    near = E.sample(library, moved, rng, entry_neighbors=1)
    far = E.sample(library, moved, np.random.default_rng(2), entry_neighbors=1, max_entry_distance=1e-6)
    assert (near["donor"][front] >= 0).any() and (far["donor"][front] < 0).all()
    # energy: 3 % more momentum stays in the 95-110 MeV bin; 1 % deviation rejects, 5 % keeps the own donor
    hot = dict(own, local_momentum=own["local_momentum"] * 1.03)
    e = E.entry_features(own["face"], own["local_position"], own["local_momentum"], own["pdg"])[0]
    same = (e > 96) & (e * 1.03 < 109)
    tight = E.sample(library, hot, np.random.default_rng(2), entry_neighbors=1, max_energy_deviation=0.01)
    loose = E.sample(library, hot, np.random.default_rng(2), entry_neighbors=1, max_energy_deviation=0.05)
    assert same.sum() > 10
    assert (tight["donor"][same] < 0).all() and (loose["donor"][same] >= 0).all()
    # float32 placement equals float64 to float precision
    o64 = E.sample(library, own, np.random.default_rng(4), entry_neighbors=1)
    o32 = E.sample(library, own, np.random.default_rng(4), entry_neighbors=1, dtype=np.float32)
    assert o32["position"].dtype == np.float32
    np.testing.assert_array_equal(o32["donor"], o64["donor"])
    np.testing.assert_allclose(o32["position"], o64["position"], atol=2e-3)
    for bad in (dict(entry_neighbors=-1), dict(entry_neighbors=1, key_neighbors=2),
                dict(entry_neighbors=1, psi_bins=E.PSI_BINS_15), dict(max_entry_distance=-1.),
                dict(max_energy_deviation=float("nan"))):
        with pytest.raises(ValueError):
            E.sample(library, own, rng, **bad)
