"""Synthetic whole-shower tests, independent of any private physics input."""
import numpy as np
import pytest

from calofastsim.showers import Binning, build, frame, load, sample, save
from calofastsim.showers.export import join, keys


def fixture():
    entries = dict(energy=np.array([100., 100., 70., 100.]), cosine=np.ones(4), radius=np.full(4, 500.),
                   time=np.arange(4, dtype=float), disk=np.array([0, 0, 1, 0]), event_row=np.arange(4),
                   position=np.zeros((4, 3)), direction=np.tile([0., 0., 1.], (4, 1)))
    steps = dict(entry=np.array([0, 0, 2, 3]), energy_g4=np.array([20., 30., 60., 999.]),
                 energy_birks=np.array([19., 29., 59., 998.]), time=np.array([1., 2., 4., 5.]),
                 crystal=np.array([0, 1, 2, 0]), disk=np.array([0, 1, 1, 0]),
                 position=np.array([[1., 0., 0.], [2., 0., 0.], [0., 1., 0.], [1., 0., 0.]]))
    return entries, steps


def test_frame_roundtrip():
    u = np.array([[1, 0, 0], [0, 0, 1], [.2, .7, -.3]])
    r = frame(u)
    np.testing.assert_allclose(r.transpose(0, 2, 1) @ r, np.broadcast_to(np.eye(3), r.shape), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(r), 1)
    np.testing.assert_allclose(r[:, :, 2], u / np.linalg.norm(u, axis=1)[:, None])


def test_development_only_and_empty_showers():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, True, True, False]))
    np.testing.assert_array_equal(lib["offsets"], [0, 2, 2, 3])
    np.testing.assert_array_equal(lib["entry_event_row"], [0, 1, 2])
    assert 999 not in lib["step_energy_g4"]
    np.testing.assert_array_equal(lib["step_disk"], [0, 1, 1])


def test_roundtrip_same_entry():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, False, False, False]))
    query = {k: v[:1] for k, v in e.items()}
    draw = sample(lib, query, np.random.default_rng(3))
    np.testing.assert_allclose(draw["position"], s["position"][:2])
    np.testing.assert_allclose(draw["time"], s["time"][:2])
    np.testing.assert_array_equal(draw["energy_birks"], s["energy_birks"][:2])
    assert draw["supported"].all() and not draw["fallback"].any()


def test_draw_preserves_joint_pattern_and_zeros():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, True, False, False]))
    query = {k: np.repeat(v[:1], 10000, axis=0) for k, v in e.items()}
    draw = sample(lib, query, np.random.default_rng(10))
    sizes = np.diff(draw["offsets"])
    assert set(sizes) == {0, 2}
    assert abs(np.mean(sizes == 0) - .5) < .025
    np.testing.assert_array_equal(np.bincount(draw["entry"], weights=draw["energy_g4"], minlength=10000),
                                  np.where(draw["donor"] == 0, 50, 0))


def test_shift_rotation_and_times():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, False, False, False]))
    q = {k: v[:1].copy() for k, v in e.items()}
    q["position"][:] = [10, 20, 30]
    q["time"][:] = 100
    q["direction"][:] = [1, 0, 0]
    draw = sample(lib, q, np.random.default_rng(0))
    delta = draw["position"] - q["position"]
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), [1, 2])
    np.testing.assert_allclose(delta @ q["direction"][0], 0)
    np.testing.assert_allclose(draw["time"], [101, 102])


def test_fallback_never_silently_uses_other_disk():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, False, False, False]))
    q = {k: v.copy() for k, v in e.items()}
    q["energy"][0] = 10
    q["energy"][1] = 200
    draw = sample(lib, q, np.random.default_rng(0))
    np.testing.assert_array_equal(draw["supported"], [True, True, False, True])
    np.testing.assert_array_equal(draw["fallback"], [True, True, True, False])
    assert draw["donor"][2] == -1


def test_empty_query():
    e, s = fixture()
    lib = build(e, s, development=np.array([True, False, False, False]))
    draw = sample(lib, {k: v[:0] for k, v in e.items()}, np.random.default_rng(0))
    assert draw["position"].shape == (0, 3)
    np.testing.assert_array_equal(draw["offsets"], [0])


def test_empty_development_responses():
    e, s = fixture()
    lib = build(e, s, development=np.array([False, True, False, False]))
    draw = sample(lib, e, np.random.default_rng(0))
    assert draw["position"].shape == (0, 3)
    assert draw["supported"][0]
    np.testing.assert_array_equal(draw["offsets"], [0, 0, 0, 0, 0])


def test_save_load(tmp_path):
    e, s = fixture()
    lib = build(e, s, development=np.array([True, True, True, False]))
    path = tmp_path / "library.npz"
    save(path, lib, {"events": 4})
    result, metadata = load(path)
    assert metadata == {"events": 4}
    assert set(result) == set(lib)
    for k in lib:
        np.testing.assert_array_equal(result[k], lib[k])


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "badstep", "negative", "nomask", "zero_direction"])
def test_bad_inputs(mutation):
    e, s = fixture()
    mask = np.array([True, True, True, False])
    if mutation == "duplicate":
        e["event_row"][1] = 0
    elif mutation == "nan":
        s["position"][0, 0] = np.nan
    elif mutation == "badstep":
        s["entry"][0] = 100
    elif mutation == "negative":
        s["energy_birks"][0] = -1
    elif mutation == "nomask":
        mask = mask.astype(int)
    else:
        e["direction"][0] = 0
    with pytest.raises(ValueError):
        build(e, s, development=mask)


def test_bad_edges_and_no_entries():
    e, s = fixture()
    with pytest.raises(ValueError):
        Binning(energy=(1, 1)).arrays()
    with pytest.raises(ValueError):
        build(e, s, development=np.zeros(4, bool))


def test_identity_join_includes_event_and_product():
    t = dict(run=np.array([1, 1, 1]), subrun=np.array([0, 0, 1]), event=np.array([2, 1, 1]),
             product=np.array([7, 8, 7]))
    k = keys(t, tuple(t))
    np.testing.assert_array_equal(join(k, k[[2, 0]]), [2, 0])
    with pytest.raises(ValueError):
        join(k[:1], k[1:])
