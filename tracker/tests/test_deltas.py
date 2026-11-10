"""trkfastsim.sim.deltas: library building from dumper-table frames and the device placement of delta steps."""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from trkfastsim.sim import deltas as DL  # noqa: E402

NS = DL.NSTRAW


def _geom(n=2 * NS):
    """Straws along x, panel p = js // 96; straw k of a panel at y = 6 k, z = 5 * (k % 2) + 100 p."""
    js = np.arange(n)
    smid = np.c_[np.zeros(n), 6.0 * (js % NS), 5.0 * (js % 2) + 100.0 * (js // NS)]
    return smid, np.tile([1.0, 0.0, 0.0], (n, 1))


def _frames():
    smid, _ = _geom()
    row = lambda e, straw, sp, x, t, eion, ln, panel=0: dict(run=1, subrun=1, event=e, plane=0, panel=panel, straw=straw,
        sp=sp, eion=eion, len=ln, width=0.01, shape=0, ion=1, t=t, x0=x - 0.5, x1=x + 0.5,
        y0=smid[straw, 1] + 0.3, y1=smid[straw, 1] + 0.3, z0=smid[straw, 2] - 1.0 + 100.0 * panel,
        z1=smid[straw, 2] + 1.0 + 100.0 * panel, px=0.0, py=0.0, pz=1.0)
    st = pd.DataFrame([row(1, 4, 1, 10.0, 100.0, 2e-3, 5.0),         # CE crossing, straw 4
                       row(1, 4, 7, 12.0, 100.5, 1e-3, 1.0),         # delta in the same straw
                       row(1, 5, 7, 11.0, 100.8, 5e-4, 0.5),         # delta in the next straw of the panel
                       row(1, 4, 8, 11.0, 100.4, 9e-4, 0.4, 1),      # delta in another panel: not attributed
                       row(1, 4, 9, 11.0, 100.4, 9e-4, 0.4),         # not a descendant: not attributed
                       row(2, 20, 1, -30.0, 50.0, 1e-3, 3.0)])       # CE crossing without deltas, another bin
    pa = pd.DataFrame([(1, 1, 1, 1, -1, 11, 1, 0, 0.0), (1, 1, 1, 7, 1, 11, 0, 0, 0.0), (1, 1, 1, 8, 1, 11, 0, 0, 0.0),
                       (1, 1, 1, 9, -1, 11, 0, 0, 0.0), (1, 1, 2, 1, -1, 11, 0, 1, 0.0)],
                      columns=["run", "subrun", "event", "key", "parent", "pdg", "isprim", "ismcprim", "sz"])
    return st, pa


def test_build_attributes_same_straw_and_same_panel_steps():
    smid, sdir = _geom()
    lib = DL.build(*_frames(), smid, sdir)
    assert lib.bin_cross.sum() == 2 and lib.g_count.tolist() == [2]
    s = lib.steps
    assert sorted(s["dk"].tolist()) == [0, 1]
    i0 = int(np.flatnonzero(s["dk"] == 0)[0])
    assert np.isclose(s["du"][i0], 2.0) and np.isclose(s["dt"][i0], 0.5) and np.isclose(s["eion"][i0], 1e-3)
    assert np.isclose(s["dv"][i0], 0.3) or np.isclose(s["dv"][i0], -0.3)      # transverse offset from its own wire


def test_save_load_round_trip(tmp_path):
    smid, sdir = _geom()
    lib = DL.build(*_frames(), smid, sdir)
    lib.save(tmp_path / "d.npz")
    b = DL.Library.load(tmp_path / "d.npz")
    assert np.array_equal(lib.g_count, b.g_count) and all(np.array_equal(lib.steps[k], b.steps[k]) for k in DL.STEP)


def test_device_places_the_steps_relative_to_the_record():
    smid, sdir = _geom()
    lib = DL.build(*_frames(), smid, sdir)
    lib.bin_cross = np.bincount(lib.g_bin, minlength=len(DL.LEN_EDGES) - 1).astype(np.int64)   # every crossing has
    D = DL.DeviceLibrary(lib)
    T = dict(smid=jnp.asarray(smid), sdir=jnp.asarray(sdir), sid=jnp.arange(2 * NS), plane=jnp.zeros(2 * NS, int),
             panel=jnp.arange(2 * NS) // NS, straw=jnp.arange(2 * NS) % NS, upanel=jnp.arange(2 * NS) // NS)
    # one fast record in straw 30 of panel 1 at along-wire x = -50, path 5 mm (the library's bin), one in straw 95
    R = 4
    st = {k: jnp.zeros((2, R)) for k in ("eion", "width", "t", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz")}
    st.update(valid=jnp.asarray([[True, False, False, False], [True, False, False, False]]),
              straw_index=jnp.asarray([[NS + 30, 0, 0, 0], [NS + 95, 0, 0, 0]], jnp.int32),
              len=jnp.full((2, R), 5.0), shape=jnp.zeros((2, R), jnp.int32), ion=jnp.zeros((2, R), jnp.int32),
              sid=jnp.zeros((2, R), int), plane=jnp.zeros((2, R), int), panel=jnp.zeros((2, R), int),
              straw=jnp.zeros((2, R), int), upanel=jnp.zeros((2, R), int))
    for r, j in ((0, NS + 30), (1, NS + 95)):
        for c, v in zip(("x0", "x1", "y0", "y1", "z0", "z1", "t"), (-50.5, -49.5, smid[j, 1], smid[j, 1], smid[j, 2], smid[j, 2], 7.0)):
            st[c] = st[c].at[r, 0].set(v)
    g = D.pick(jax.random.key(1), st)
    assert np.asarray(g)[:, 0].tolist() == [0, 0] and (np.asarray(g)[:, 1:] == -1).all()
    out, row, ok = D.make(g, st, T, 8)
    ok = np.asarray(ok)
    assert ok.sum() == 3                                   # straw 95 + 1 is outside the panel: dropped
    j = np.asarray(out["straw_index"])[ok, 0]
    assert sorted(j.tolist()) == sorted([NS + 30, NS + 31, NS + 95])
    i = int(np.flatnonzero(ok & (np.asarray(out["straw_index"])[:, 0] == NS + 30))[0])
    mid = 0.5 * (np.asarray(out["x0"])[i, 0] + np.asarray(out["x1"])[i, 0])
    assert np.isclose(mid, -50.0 + 2.0)                    # the record's along-wire point + du
    assert np.isclose(np.asarray(out["t"])[i, 0], 7.5) and np.isclose(np.asarray(out["eion"])[i, 0], 1e-3)
    assert np.asarray(row)[ok].tolist().count(0) == 2
