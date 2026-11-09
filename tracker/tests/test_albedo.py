"""trkfastsim.sim.albedo: library building from dumper-table frames, the device sampler and the record merge."""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from trkfastsim.sim import albedo as AL  # noqa: E402

AX = AL.AXIS_XY[0]


def _frames():
    """Event 1: the CE (key 5) leaves at radius 400 (phi 0) and comes back; a shower electron (key 9, grandchild of
    the CE) and a photon (key 8) come back too; an unrelated electron (key 3) crosses upstream. Event 2: leaves, never
    returns. Event 3: leaves at radius 100 (another bin), never returns."""
    pa = pd.DataFrame([(1, 1, 1, 5, -1, 11, 0, 1), (1, 1, 1, 7, 5, 22, 0, 0), (1, 1, 1, 9, 7, 11, 0, 0),
                       (1, 1, 1, 8, 5, 22, 0, 0), (1, 1, 1, 3, -1, 11, 0, 0),
                       (1, 1, 2, 5, -1, 11, 1, 0), (1, 1, 3, 2, -1, 11, 0, 1)],
                      columns=["run", "subrun", "event", "key", "parent", "pdg", "isprim", "ismcprim"])
    z = AL.VD_Z
    v = pd.DataFrame([(1, 1, 1, 15, 5, AX + 400, 0, z, -10, 60, 70, 10.0),         # CE exit (radial -10, azimuthal 60)
                      (1, 1, 1, 15, 5, AX + 380, 40, z, -3, 4, -5, 14.0),           # CE back
                      (1, 1, 1, 15, 9, AX + 300, -60, z, 1, 1, -2, 13.0),           # shower electron back
                      (1, 1, 1, 15, 8, AX + 300, -60, z, 1, 1, -2, 13.5),           # photon back (not kept)
                      (1, 1, 1, 15, 3, AX + 300, -60, z, 1, 1, -2, 15.0),           # not a descendant
                      (1, 1, 1, 15, 5, AX + 380, 40, z, 1, 1, -2, 9.0),             # before the exit (not kept)
                      (1, 1, 2, 15, 5, AX + 410, 0, z, -10, 60, 70, 10.0),
                      (1, 1, 3, 15, 2, AX + 100, 0, z, 0, 60, 70, 10.0),
                      (1, 1, 3, 13, 2, AX + 100, 0, 8539.5, 0, 60, 70, 1.0)],      # another detector
                     columns=["run", "subrun", "event", "vdid", "sp", "x", "y", "z", "px", "py", "pz", "t"])
    return v, pa


def test_build_keeps_the_electron_returns_of_each_exit():
    lib = AL.build(*_frames())
    assert lib.bin_exits.sum() == 3 and len(lib.g_bin) == 1 and lib.g_count.tolist() == [2]
    assert lib.isce.tolist() == [False, True]                       # time order: shower electron at 13, CE at 14
    assert np.allclose(lib.dt, [3.0, 4.0])
    assert np.allclose(lib.r, [np.hypot(300, 60), np.hypot(380, 40)])
    assert np.allclose(lib.dphi, [np.arctan2(-60, 300), np.arctan2(40, 380)])
    i = 1                                                            # the CE: momentum (-3, 4) at phi = atan2(40, 380)
    ph = np.arctan2(40, 380)
    assert np.allclose([lib.pr[i], lib.pphi[i]], [-3 * np.cos(ph) + 4 * np.sin(ph), 3 * np.sin(ph) + 4 * np.cos(ph)])
    b = AL.bin_of(np.array([400.0]), np.array([70 / np.sqrt(10 ** 2 + 60 ** 2 + 70 ** 2)]), np.array([-10.0]))
    assert lib.g_bin.tolist() == b.tolist() and lib.bin_exits[b[0]] == 2
    ce = lib.select("ce")
    assert ce.g_count.tolist() == [1] and ce.isce.tolist() == [True] and ce.bin_exits.sum() == 3
    assert lib.select("all") is lib


def test_save_load_round_trip(tmp_path):
    lib = AL.build(*_frames())
    lib.save(tmp_path / "a.npz")
    back = AL.Library.load(tmp_path / "a.npz")
    for k in AL.Library.FIELDS:
        assert np.array_equal(getattr(lib, k), getattr(back, k)), k


def test_helix_back_to_the_plane_inverts():
    rng = np.random.default_rng(1)
    n = 50
    x0 = np.c_[AX + rng.uniform(-600, 600, n), rng.uniform(-600, 600, n), np.full(n, AL.VD_Z)]
    u0 = rng.normal(size=(n, 3)); u0[:, 2] = np.abs(u0[:, 2]) + 0.3; u0 /= np.linalg.norm(u0, axis=1)[:, None]
    p = rng.uniform(5, 105, n)
    x1, u1 = AL.back_to_plane(jnp.asarray(x0), jnp.asarray(u0), jnp.asarray(p), 1.0, z_plane=AL.VD_Z + 60.0)
    x2, u2 = AL.back_to_plane(x1, u1, jnp.asarray(p), 1.0)
    assert np.allclose(np.asarray(x2), x0, atol=1e-7) and np.allclose(np.asarray(u2), u0, atol=1e-10)
    # the helix radius is p_t / (0.3 B): the transverse displacement over a half turn is its diameter
    pt = p * np.hypot(u0[:, 0], u0[:, 1])
    rho = pt / (AL.KB * 1.0)
    s_half = np.pi * rho / np.hypot(u0[:, 0], u0[:, 1])
    x3, _ = AL.back_to_plane(jnp.asarray(x0), jnp.asarray(u0), jnp.asarray(p), 1.0,
                             z_plane=jnp.asarray(AL.VD_Z + s_half * u0[:, 2]))
    assert np.allclose(np.hypot(*(np.asarray(x3)[:, :2] - x0[:, :2]).T), 2 * rho, rtol=1e-9)


def test_sampler_rotates_the_returns_to_the_fast_exit():
    lib = AL.build(*_frames())
    lib.bin_exits = np.bincount(lib.g_bin, minlength=AL.NBIN).astype(np.int64)    # every exit returns
    D = AL.DeviceLibrary(lib)
    # two fast exits in the library's bin, rotated by +90 degrees and -30 degrees, and one in an empty bin
    phis = np.array([np.pi / 2, -np.pi / 6, 0.0])
    r = np.array([400.0, 400.0, 100.0])
    ut = np.array([-10.0, 60.0])     # radial, azimuthal (as the library exit)
    x = np.c_[AX + r * np.cos(phis), r * np.sin(phis), np.full(3, AL.VD_Z)]
    u = np.stack([[ut[0] * np.cos(f) - ut[1] * np.sin(f), ut[0] * np.sin(f) + ut[1] * np.cos(f), 70.0] for f in phis])
    p = np.linalg.norm(u, axis=1)
    u = u / p[:, None]
    t = np.array([100.0, 200.0, 300.0])
    g, phi, te = D.pick(jax.random.key(1), jnp.asarray(x), jnp.asarray(u), jnp.asarray(p), jnp.asarray(t),
                        jnp.ones(3, bool), 1.0)
    assert np.asarray(g).tolist() == [0, 0, -1] and np.allclose(np.asarray(te), t)
    rr = D.returns(g, phi, te, jnp.arange(3), 8)
    v = np.asarray(rr["valid"])
    assert v.sum() == 4 and np.asarray(rr["track"])[v].tolist() == [0, 0, 1, 1]
    xr, pr = np.asarray(rr["x"])[v], np.asarray(rr["p"])[v]
    # the CE return of the second fast exit: library position (380, 40) and momentum (-3, 4, -5) rotated by -30 deg
    c, s = np.cos(-np.pi / 6), np.sin(-np.pi / 6)
    assert np.allclose(xr[3, :2] - [AX, 0], [380 * c - 40 * s, 380 * s + 40 * c])
    assert np.allclose(pr[3], [-3 * c - 4 * s, -3 * s + 4 * c, -5])
    assert np.allclose(np.asarray(rr["t"])[v], [103.0, 104.0, 203.0, 204.0])
    assert np.asarray(rr["isce"])[v].tolist() == [False, True, False, True]


def test_merge_appends_after_the_last_record_and_drops_beyond_the_cap():
    R = 6
    valid = np.zeros((3, R), bool)
    valid[0, [0, 1, 3]] = True             # a gap: the extra records go after slot 3
    valid[1, :2] = True
    steps = {"valid": jnp.asarray(valid), "t": jnp.asarray(np.arange(3 * R, dtype=float).reshape(3, R))}
    ev = np.zeros((3, R), bool)
    ev[0, :3] = True; ev[1, :2] = True; ev[2, :1] = True
    extra = {"valid": jnp.asarray(ev), "t": jnp.asarray(100 + np.arange(3 * R, dtype=float).reshape(3, R))}
    track = jnp.asarray([0, 1, 0])          # rows 0 and 2 of extra both go to track 0
    out, dropped = AL.merge_records(steps, extra, track, jnp.asarray([True, True, True]), R)
    t, v = np.asarray(out["t"]), np.asarray(out["valid"])
    assert v[0].tolist() == [True, True, False, True, True, True] and int(dropped) == 2
    assert t[0, 4:].tolist() == [100.0, 101.0]
    assert v[1].tolist() == [True, True, True, True, False, False] and t[1, 2:4].tolist() == [106.0, 107.0]
    assert np.array_equal(t[2], np.asarray(steps["t"])[2])


def test_the_shipped_library_loads_with_its_counts():
    lib = AL.Library.load()
    assert AL.DEFAULT.exists() and lib.bin_exits.sum() == 36649 and len(lib.g_bin) == 1305 and len(lib.r) == 1550
    assert int(lib.isce.sum()) == 370 and len(lib.select("ce").g_bin) == 352
    assert lib.bin_exits.shape == (AL.NBIN,) and (lib.g_count > 0).all()
