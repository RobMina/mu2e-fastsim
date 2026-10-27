"""Synthetic single-foil configuration/census guards, without Geant4 or external files."""
import numpy as np
import pytest

from beamfastsim.foilclosure import render, validate_case, MMU
from beamfastsim.foilreport import census, highest, kinetic


def fixture():
    case = dict(energy_MeV=2., thickness_mm=.2, cosine=1., pdg=13, events=2)
    p0 = np.sqrt(2. * (2. + 2 * MMU))
    p1 = np.sqrt(1. * (1. + 2 * MMU))
    p = dict(run=[1, 1, 1], evt=[1, 2, 1], trk=[1, 1, 2], pdg=[13, 13, 11],
        parent_id=[-1, -1, 1], code=[56, 56, 31], pstop=[p1, 0, 0], zstop=[20, .01, .05],
        codestop=[182, 32, 1], p=[p0, p0, .5], time=[0., 0., .01], z=[-.101, -.101, 0.], tstop=[1., 5., .02])
    s = dict(run=[1]*7, evt=[1, 1, 1, 1, 2, 2, 2], trk=[1]*7, pdg=[13]*7,
        sid=[1, 2, 2, 1, 1, 2, 2], code=[182, 31, 182, 182, 182, 31, 32], x=[0.]*7, y=[0.]*7,
        z=[-.101, -.1, 0, .1, -.101, -.1, .01], px=[0.]*7, py=[0.]*7,
        pz=[p0, p0, p1, p1, p0, p0, 0], ke=[2, 2, 1, 1, 2, 2, 0], time=[0, 0, .01, .02, 0, 0, .01])
    return {k: np.asarray(v) for k, v in p.items()}, {k: np.asarray(v) for k, v in s.items()}, case


def test_census_identity_exit_and_physical_stop():
    p, s, case = fixture()
    report, store = census(p, s, case)
    assert (report["primaries"], report["exits"], report["physical_stops"], report["other"]) == (2, 1, 1, 0)
    np.testing.assert_allclose(store["exit_MeV"], [1., 0.])
    assert report["hard_electrons"] == 1
    np.testing.assert_allclose(store["chord_sum_mm"][0], .2)
    np.testing.assert_array_equal(store["foil_steps"], [2, 2])
    assert store["hard_count"].tolist() == [1, 0]


def test_subthreshold_exit_is_not_physical_stop():
    p, s, case = fixture()
    s["pz"][3] = np.sqrt(.05 * (.05 + 2 * MMU))
    report, store = census(p, s, case)
    assert report["physical_stops"] == 1
    assert report["threshold_stops"] == 2 and report["subthreshold_exits"] == 1
    assert store["exit"][0]


def test_missing_event_or_entrance_fails():
    p, s, case = fixture()
    with pytest.raises(ValueError, match="coverage"):
        census({**p, "evt": np.array([1, 1, 1])}, s, case)
    with pytest.raises(ValueError, match="never enters"):
        census(p, {**s, "sid": np.ones(7)}, case)
    s["evt"][0] = 3
    with pytest.raises(ValueError, match="outside"):
        census(p, s, case)


def test_wrong_parent_is_not_hard_primary_energy():
    p, s, case = fixture()
    p["parent_id"][-1] = 9
    report, _ = census(p, s, case)
    assert report["hard_electrons"] == 0


@pytest.mark.parametrize("args", [(0, .2, 1, 13, 1, 1), (5, -1, 1, 13, 1, 1), (5, .2, .4, 13, 1, 1),
                                   (5, .2, 1, 11, 1, 1), (5, .2, 1, 13, 0, 1), (5, .2, 1, 13, 2**24, 1)])
def test_case_domain(args):
    with pytest.raises(ValueError):
        validate_case(*args)


def test_render_muon_charge_thickness_and_angle():
    geom, gun = render(5., .1056, .8, -13)
    assert '"G4_Al"' in geom and '0.0528' in geom
    assert 'particleGun.id = -13' in gun
    assert 'particleGun.czmin = 0.8' in gun
    assert 'particleGun.mean = -1' in gun
    assert geom.isascii() and gun.isascii()


def test_highest_tree_cycle(tmp_path):
    import uproot
    path = tmp_path / "data.root"
    with uproot.recreate(path) as out:
        out.mktree("readback/tpart", {"evt": np.array([1, 2])})
        out.mktree("readback/tpart", {"evt": np.array([1, 2, 3])})
    with uproot.open(path) as root:
        arrays, meta = highest(root, "readback/tpart")
        assert meta["key"] == "readback/tpart;2" and meta["entries"] == 3
        assert len(arrays["evt"]) == 3
    np.testing.assert_allclose(kinetic(np.array([0., 20.])), [0., 20**2 / (np.sqrt(20**2 + MMU**2) + MMU)])


@pytest.mark.parametrize("message", ["***** Illegal parameter (1)", "illegal process (or type) name",
                                      "Command aborted (1)", "***** Batch is interrupted!!"])
def test_macro_warning_is_failure_even_if_art_exits_zero(message):
    from beamfastsim.foilclosure import macro_errors
    assert macro_errors(message + "\nArt has completed and will exit with status 0.")
    assert not macro_errors("/run/initialize\nArt has completed and will exit with status 0.")


def test_no_msc_rejects_scattering_step():
    p, s, case = fixture()
    s["code"][1] = 29
    with pytest.raises(ValueError, match="scattering"):
        census(p, s, {**case, "no_msc": True})


def test_comparison_bounds_and_stop_atom():
    from beamfastsim.foilcompare import contrast, verdict
    assert verdict(.01, .001, .02)["state"] == "PASS"
    assert verdict(.1, .001, .02)["state"] == "FAIL"
    assert verdict(.01, .01, .02)["state"] == "INDETERMINATE"
    c, r = np.tile([0., .2, .3], 100), np.tile([.05, .2, .3], 100)
    aligned = contrast(c, r, 1.)
    assert aligned["cdf"]["distance"] == 0.
    assert aligned["mean_exit"]["delta"] == 0.
    raw = contrast(c, r, 1., threshold_aligned=False)
    assert raw["cdf"]["distance"] == pytest.approx(1/3)


def test_undefined_endpoint_is_unresolved_not_a_stop():
    p, s, case = fixture()
    for name in ("pstop", "zstop", "tstop", "codestop"):
        p[name][1] = 0
    report, store = census(p, s, case)
    assert report["physical_stops"] == 0 and report["other"] == 1
    assert np.isnan(store["exit_MeV"][1])


def test_zero_count_binomial_interval_is_not_zero_width():
    from beamfastsim.foilcompare import stop_interval
    small = stop_interval(np.ones(2), np.ones(2))
    assert small["state"] == "INDETERMINATE"
    assert small["interval_99pct"][1] > .5
    large = stop_interval(np.ones(50000), np.ones(5000))
    assert large["state"] == "PASS" and large["interval_99pct"][1] > 0


def test_raw_range_endpoint_bridge_changes_bypass_only():
    from beamfastsim.foilendpoint import RawRangeLaw
    e = np.geomspace(.001, 10., 100)
    law = RawRangeLaw(dict(energy=e, values=e / 2), 1., maximum=5.)
    mean, forced = law.mean([1., 1., 1.], [.4, .475, .51], "range")
    np.testing.assert_allclose(mean, [.8, .95, 1.], atol=1e-12)
    assert forced.tolist() == [False, False, True]


def test_table_audit_rejects_tampered_export_before_loading(tmp_path):
    import json
    from beamfastsim.foilreport import table_audit
    from beamfastsim.reference_audit import _identity
    ref, exp = tmp_path / "ref", tmp_path / "export"
    ref.mkdir()
    exp.mkdir()
    (ref / "couple.dat").write_text("reference")
    (ref / "manifest.json").write_text(json.dumps(dict(artifacts=[_identity(ref / "couple.dat")])))
    (exp / "job.log").write_text("Art has completed and will exit with status 0.")
    (exp / "table.asc").write_text("original")
    (exp / "receipt.json").write_text(json.dumps(dict(complete=True, artifacts=[_identity(exp / "job.log")],
                                                     tables=[_identity(exp / "table.asc")])))
    (exp / "table.asc").write_text("changed")
    with pytest.raises(ValueError, match="slab table export"):
        table_audit(exp, ref)
