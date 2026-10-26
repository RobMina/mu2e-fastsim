"""Saved-entry study/report contracts and replica statistics with small synthetic tables."""
import json

import numpy as np
import pytest

pytest.importorskip("jax")
from beamfastsim import fluctstudy as fs, fluctreport as fr


def test_replica_not_pair_count_is_mc_denominator():
    values = np.array([[0., 0., 1.], [1., 1., 1.]])
    result = fs.mc_summary(values.sum(axis=1))
    assert result == dict(mean=2., conditional_mc_se=1.)
    paired = fs.paired(dict(stop=values.astype(bool), exit_MeV=values),
                       dict(stop=np.zeros((2, 3), bool), exit_MeV=np.zeros((2, 3))))
    assert paired["stop_count_delta"] == result


def test_distribution_stop_atom_and_fixed_cohorts():
    values = np.array([[0., .1, 2.], [0., .2, 3.]])
    report = fr.distribution(values, np.array([1., 1., 4.]), np.array([True, True, False]))
    assert report["entries"] == 2
    assert report["cdf_le"]["0.0"] == dict(mean=.5, conditional_mc_se=0.)
    assert report["cdf_le"]["0.1"] == dict(mean=.75, conditional_mc_se=.25)
    assert report["mean_within_entry_exit_variance_MeV2"] == pytest.approx(.0025)
    assert fr.distribution(values, np.array([1., 1., 4.]), np.zeros(3, bool)) == dict(entries=0)
    contrast = fr.contrast(values, values, np.ones(3, bool))
    assert contrast["mean_exit_delta_MeV"] == dict(mean=0., conditional_mc_se=0.)


def test_same_power_ranges_and_no_cut_law_conflation():
    e = np.geomspace(.01, 100., 257)
    tables = {k: dict(energy=e, values=np.full_like(e, p)) for k, p in (("DEDX", 2.), ("DEDXnr", 3.))}
    laws = fs.laws(tables, .03410575, 20., points=1001)
    np.testing.assert_allclose(laws["g4_cut"].ranges, (laws["g4_cut"].energy - .1) / 2.)
    np.testing.assert_allclose(laws["g4_full"].ranges, (laws["g4_full"].energy - .1) / 3.)
    assert laws["g4_cut"].cut == .03410575 and laws["g4_full"].cut == 1e9


def test_incomplete_study_fails_closed(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "summary.json").write_text(json.dumps(dict(complete=False)))
    with pytest.raises(ValueError, match="complete study"):
        fr.report_saved(tmp_path)


def test_invalid_cli_before_any_input_reads(tmp_path):
    args = ["--layer-report", "absent", "--table-report", "absent", "--out", str(tmp_path / "out")]
    for extra in (["--replicas", "1"], ["--seed", "-1"], ["--slices", "4", "4"],
                  ["--budget-seconds", "nan"], ["--slices", "0"]):
        with pytest.raises(ValueError):
            fs.main(args + extra)
    assert not (tmp_path / "out").exists()
