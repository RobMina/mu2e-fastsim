"""Geometry placement and paired-statistics regression tests."""
import numpy as np
import pytest

from calofastsim.showers.library import build, sample
from calofastsim.showers.response import cluster_sum, place
from calofastsim.showers.compare import paired_hist, paired_mean, paired_ratio


def geometry():
    lib = dict(geometry_crystal_crystal=np.array([10, 11]), geometry_crystal_disk=np.array([0, 1]),
               geometry_disk_disk=np.array([0, 1]), geometry_vd_id=np.array([73, 75]),
               geometry_vd_position_z=np.array([100., 200.]),
               geometry_disk_size_x=np.array([0., 0.]), geometry_disk_size_y=np.array([20., 20.]))
    for axis in "xyz":
        lib["geometry_crystal_position_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        lib["geometry_crystal_size_" + axis] = np.full(2, 20.)
        lib["geometry_disk_origin_" + axis] = np.array([100., 200.]) if axis == "z" else np.zeros(2)
        for coord in "xyz":
            lib["geometry_disk_axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    return lib


def source():
    e = dict(energy=np.array([100.]), cosine=np.array([1.]), radius=np.array([5.]), time=np.array([1000.]),
             disk=np.array([0]), event_row=np.array([0]), position=np.array([[5., 0., 100.]]),
             direction=np.array([[0., 0., 1.]]))
    s = dict(entry=np.array([0, 0]), energy_g4=np.array([40., 30.]), energy_birks=np.array([39., 29.]),
             time=np.array([1001., 1002.]), crystal=np.array([10, 11]), disk=np.array([0, 1]),
             position=np.array([[5., 0., 110.], [5., 0., 210.]]))
    lib = build(e, s, development=np.array([True]))
    lib.update(geometry())
    return lib, e, s


def test_disk_rotation_keeps_longitudinal_coordinates():
    lib, e, _ = source()
    e = {k: v.copy() for k, v in e.items()}
    e["position"][:] = [0, 5, 100]
    e["direction"][:] = [.6, 0, .8]
    draw = sample(lib, e, np.random.default_rng(0), orientation="disk")
    np.testing.assert_allclose(draw["position"], [[0, 5, 110], [0, 5, 210]], atol=1e-12)
    placed = place(lib, draw)
    assert placed["mapped"].all()
    np.testing.assert_array_equal(placed["crystal"], [10, 11])


def test_placement_and_no_snap_into_detector():
    lib, e, s = source()
    s["position"][1, 0] = 100
    placed = place(lib, s)
    np.testing.assert_array_equal(placed["mapped"], [True, False])
    np.testing.assert_array_equal(placed["crystal"], [10, -1])
    np.testing.assert_allclose(placed["local_position"][0], [5, 0, 10])


def test_cluster_sum_zero_and_explicit_response():
    _, _, s = source()
    out = cluster_sum(s, 3)
    np.testing.assert_allclose(out["energy"], [68, 0, 0])
    assert np.isnan(out["time"][1:]).all()
    np.testing.assert_allclose(out["position"][0], np.average(s["position"], axis=0, weights=s["energy_birks"]))
    out = cluster_sum(s, 1, energies=np.array([10, 20]), accepted=np.array([False, True]))
    np.testing.assert_array_equal(out["energy"], [20])
    with pytest.raises(ValueError):
        cluster_sum(s, 1, energies=np.array([-1, 20]))


def test_paired_statistics_include_zeros():
    f, s, g = np.array([1, 0, 0, 1]), np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    assert paired_mean(f, s, g) == dict(fast=.5, stock=.5, difference=0., error=.5)
    assert paired_ratio(f, s, g)["ratio"] == 1
    assert paired_ratio(f, np.zeros(4), g)["ratio"] is None
    h = paired_hist(f, s, g, [-.5, .5, 1.5])
    np.testing.assert_array_equal(h["fast"], [2, 2])
    np.testing.assert_array_equal(h["stock"], [2, 2])
    assert h["chi2"] == 0 and h["ndf"] == 2
