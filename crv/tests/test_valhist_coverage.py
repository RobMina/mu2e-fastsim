"""Synthetic ROOT regression tests; all fixtures use uproot's writable API."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import uproot
from uproot.writing import identify

from valhist.coverage import audit


KEY = "Validation/CrvDigis_stock/hits"


def histogram(values, variances=None, edges=None, title="fills"):
    data = np.asarray(values, dtype=np.float64)
    ndim = data.ndim
    if edges is None:
        edges = [np.arange(size - 1, dtype=float) for size in data.shape]
    axes = [identify.to_TAxis("xyz"[i] + "axis", "", len(e) - 1, e[0], e[-1], np.asarray(e, dtype=float))
            for i, e in enumerate(edges)]
    args = dict(fName="h", fTitle=title, data=data.ravel(order="F"), fEntries=10,
                fTsumw=float(data.sum()), fTsumw2=10, fTsumwx=0, fTsumwx2=0,
                fSumw2=None if variances is None else np.asarray(variances, dtype=float).ravel(order="F"),
                fXaxis=axes[0])
    if ndim >= 2:
        args.update(fYaxis=axes[1], fTsumwy=0, fTsumwy2=0, fTsumwxy=0)
    if ndim >= 3:
        args.update(fZaxis=axes[2], fTsumwz=0, fTsumwz2=0, fTsumwxz=0, fTsumwyz=0)
    return getattr(identify, f"to_TH{ndim}x")(**args)


def profile():
    return identify.to_TProfile(
        fName="p", fTitle="profile", data=np.array([0., 3., 0.]),
        fEntries=2, fTsumw=2, fTsumw2=2, fTsumwx=1, fTsumwx2=1,
        fTsumwy=3, fTsumwy2=5, fSumw2=np.array([0., 5., 0.]),
        fBinEntries=np.array([0., 2., 0.]), fBinSumw2=np.array([0., 2., 0.]),
        fXaxis=identify.to_TAxis("xaxis", "", 1, 0, 1))


def files(tmp_path, ref, cand):
    paths = tmp_path / "ref.root", tmp_path / "cand.root"
    for path, objects in zip(paths, (ref, cand)):
        with uproot.recreate(path) as root:
            for key, obj in objects.items():
                root[key] = obj
    return paths


def run(tmp_path, ref, cand, **kwargs):
    paths = files(tmp_path, ref, cand)
    summary = audit(*paths, tmp_path / "audit", **kwargs)
    report = json.loads((tmp_path / "audit" / "histograms.json").read_text(),
                        parse_constant=lambda token: pytest.fail("invalid JSON number " + token))
    return summary, report


def test_flow_is_compared_separately_and_sumw2_preserved(tmp_path):
    summary, report = run(tmp_path,
        {KEY: histogram([2, 3, 5, 7], [4, 9, 25, 49])},
        {KEY: histogram([4, 3, 5, 9], [16, 9, 25, 81])})
    r = report[KEY]
    assert r["status"] == "compared"
    assert r["ref_data"]["regions"]["visible"]["value_sum"] == 8
    assert r["ref_data"]["regions"]["flow"]["value_sum"] == 9
    assert r["ref_data"]["regions"]["inclusive"]["variance_sum"] == 87
    assert r["normalization"]["visible"]["difference"] == 0
    assert r["normalization"]["flow"]["difference"] == 4
    assert r["bin_differences"]["visible"]["equal"]
    assert r["closure"]["values"]["differing_bins"] == 2
    assert not r["closure"]["exact_values_and_variances"]
    assert r["weighted_diagnostics"]["flow"]["sum_squared_difference_over_variance_sum"] == pytest.approx(4 / 20 + 4 / 130)
    assert summary["value_differences"] == [KEY]


def test_exact_closure_requires_variances_and_distinguishes_variance_only(tmp_path):
    keys = [KEY + s for s in ("_equal", "_variance", "_missing")]
    summary, report = run(tmp_path,
        {keys[0]: histogram([1, 4, 2], [2, 16, 4]), keys[1]: histogram([1, 4, 2], [2, 16, 4]), keys[2]: histogram([0, 4, 0])},
        {keys[0]: histogram([1, 4, 2], [2, 16, 4]), keys[1]: histogram([1, 4, 2], [2, 17, 4]), keys[2]: histogram([0, 4, 0])})
    assert report[keys[0]]["closure"]["exact_values_and_variances"] is True
    assert report[keys[1]]["closure"]["values"]["equal"]
    assert report[keys[1]]["closure"]["exact_values_and_variances"] is False
    assert summary["variance_differences"] == [keys[1]]
    assert report[keys[2]]["closure"]["exact_values_and_variances"] is None
    assert summary["variance_coverage_gaps"] == [keys[2]]
    assert report[keys[2]]["weighted_diagnostics"]["visible"]["status"] == "unavailable_source_sumw2"


@pytest.mark.parametrize("ndim", [2, 3])
def test_multidimensional_corner_flows_and_sumw2_layout(tmp_path, ndim):
    a = np.arange(4 ** ndim, dtype=float).reshape((4,) * ndim)
    b = a.copy()
    b[(0,) * ndim] += 11
    b[(1,) * ndim] += 7
    va = a * a + 1
    summary, report = run(tmp_path, {KEY: histogram(a, va)}, {KEY: histogram(b, va)})
    r = report[KEY]
    assert r["ref_data"]["dimensions"] == ndim
    assert r["normalization"]["flow"]["difference"] == 11
    assert r["normalization"]["visible"]["difference"] == 7
    assert r["closure"]["source_variances"]["equal"]
    assert r["weighted_diagnostics"]["visible"]["sum_squared_difference_over_variance_sum"] == pytest.approx(49 / (2 * va[(1,) * ndim]))
    assert not summary["coverage_gaps"]


def test_normalization_and_shape_are_independent_descriptions(tmp_path):
    summary, report = run(tmp_path,
        {KEY: histogram([0, 2, 4, 0], [0, 2, 4, 0])},
        {KEY: histogram([0, 6, 12, 0], [0, 6, 12, 0])})
    r = report[KEY]
    assert r["normalization"]["visible"]["relative_difference"] == 2
    assert r["shape"]["visible"]["total_variation"] == 0
    assert r["shape"]["flow"]["status"] == "undefined_zero_total"
    assert "flag" not in r and "pvalue" not in r
    assert "pass" not in summary


def test_missing_type_edges_profiles_and_nonhistograms_are_explicit(tmp_path):
    names = {s: KEY + "_" + s for s in ("left", "right", "type", "edges", "profile", "string")}
    ref = {names["left"]: histogram([0, 1, 0]), names["type"]: histogram([0, 1, 0]),
           names["edges"]: histogram([0, 1, 0], edges=[np.array([0., 1.])]),
           names["profile"]: profile(), names["string"]: "not a histogram"}
    cand = {names["right"]: histogram([0, 1, 0]), names["type"]: histogram(np.zeros((3, 3))),
            names["edges"]: histogram([0, 1, 0], edges=[np.array([0., 1. + 1e-12])]),
            names["profile"]: profile(), names["string"]: "not a histogram"}
    summary, report = run(tmp_path, ref, cand)
    for suffix, issue in (("left", "missing_cand"), ("right", "missing_ref"), ("type", "type_mismatch"),
                          ("edges", "binning_mismatch"), ("profile", "unsupported_profile"), ("string", "unsupported_object")):
        assert issue in summary["coverage_gaps"][names[suffix]]
        assert "closure" not in report[names[suffix]]
    assert "binning_mismatch" in report[names["type"]]["issues"]


def test_derived_dqm_maps_never_receive_count_statistics(tmp_path):
    keys = ["crvRecoDQM/rateMap", "dqmCrv/MPV", "crvDigiDQM/ambiguous", "Validation/CrvReco/efficiency"]
    summary, report = run(tmp_path,
        {k: histogram(np.ones((3, 3)), np.ones((3, 3)), title="map") for k in keys},
        {k: histogram(np.full((3, 3), 2), np.ones((3, 3)), title="map") for k in keys})
    assert summary["statuses"] == {"compared": 4}
    for r in report.values():
        assert r["classification"]["ref"] == "non_count"
        assert r["normalization"]["status"] == "not_applicable_non_count"
        assert r["weighted_diagnostics"]["status"] == "not_applicable_non_count"
        assert not r["closure"]["values"]["equal"]


def test_highest_cycle_not_first_and_type_changes_not_hidden(tmp_path):
    ref, cand = files(tmp_path, {KEY: histogram([0, 1, 0], [0, 1, 0])}, {KEY: histogram([0, 9, 0], [0, 9, 0])})
    with uproot.update(ref) as root:
        root[KEY] = histogram([0, 9, 0], [0, 9, 0])
    audit(ref, cand, tmp_path / "audit")
    report = json.loads((tmp_path / "audit" / "histograms.json").read_text())
    assert report[KEY]["ref"][0]["cycle"] == 2
    assert report[KEY]["closure"]["exact_values_and_variances"]
    with uproot.update(ref) as root:
        root[KEY] = "replaced with unsupported object"
    summary = audit(ref, cand, tmp_path / "audit2")
    assert "type_mismatch" in summary["coverage_gaps"][KEY]
    assert "unsupported_object" in summary["coverage_gaps"][KEY]


def test_validation_alias_prefix_matching_and_collisions(tmp_path):
    candidate = KEY.replace("CrvDigis_stock", "CrvDigis_fast_extra")
    canonical = KEY.replace("CrvDigis_stock", "CrvDigis_stock_extra")
    ref, cand = files(tmp_path, {canonical: histogram([0, 1, 0])}, {candidate: histogram([0, 1, 0])})
    summary = audit(ref, cand, tmp_path / "audit", aliases=["CrvDigis_fast=CrvDigis_stock"], select=r"stock_extra")
    assert summary["statuses"] == {"compared": 1}
    with uproot.update(cand) as root:
        root[canonical] = histogram([0, 1, 0])
    summary = audit(ref, cand, tmp_path / "audit2", aliases=["CrvDigis_fast=CrvDigis_stock"])
    assert summary["coverage_gaps"][canonical] == ["alias_collision"]
    for bad in ("no_equals", "a=", "a/b=c", "a=b=c"):
        with pytest.raises(ValueError):
            audit(ref, cand, tmp_path / "unused", aliases=[bad])


def test_zero_variance_signed_weights_and_nonfinite_do_not_hide_problems(tmp_path):
    keys = [KEY + s for s in ("_zero", "_signed", "_nan", "_badvar")]
    summary, report = run(tmp_path,
        {keys[0]: histogram([0, 1, 0], [0, 0, 0]), keys[1]: histogram([0, -1, 0], [0, 1, 0]),
         keys[2]: histogram([0, np.nan, 0], [0, 1, 0]), keys[3]: histogram([0, 1, 0], [0, -1, 0])},
        {k: histogram([0, 2, 0], [0, 0, 0]) for k in keys})
    weighted = report[keys[0]]["weighted_diagnostics"]["visible"]
    assert weighted["zero_variance_different_bins"] == 1
    assert weighted["positive_variance_bins"] == 0
    assert weighted["status"] == "undefined_zero_variance_difference"
    assert weighted["sum_squared_difference_over_variance_sum"] is None
    assert report[keys[1]]["shape"]["visible"]["status"] == "undefined_signed_weights"
    assert "ref_nonfinite_values" in summary["coverage_gaps"][keys[2]]
    assert "ref_invalid_variances" in summary["coverage_gaps"][keys[3]]
    assert report[keys[2]]["ref_data"]["regions"]["visible"]["value_sum"] is None


def test_cli_new_output_and_default_selection_preserve_sources(tmp_path):
    ref, cand = files(tmp_path, {KEY: histogram([0, 0, 0]), "other/h": histogram([0, 1, 0])}, {KEY: histogram([0, 1, 0])})
    before = [p.read_bytes() for p in (ref, cand)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "-m", "valhist.coverage", str(ref), str(cand), "--outdir", str(tmp_path / "cli")]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "cli" / "summary.json").read_text())
    assert summary["excluded_ref_objects"] == 1
    assert summary["union_keys"] == 1
    report = json.loads((tmp_path / "cli" / "histograms.json").read_text())
    assert report[KEY]["normalization"]["visible"]["relative_difference"] is None
    assert report[KEY]["shape"]["visible"]["status"] == "undefined_zero_total"
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "must be new" in result.stderr
    assert before == [p.read_bytes() for p in (ref, cand)]
    summary = audit(ref, cand, tmp_path / "empty", select="does_not_match")
    assert summary["empty_selection"]
    summary = audit(ref, cand, tmp_path / "all", select="")
    assert summary["only_ref"] == ["other/h"]


def test_equal_total_shape_difference_and_sumw2_presence_gap(tmp_path):
    summary, report = run(tmp_path,
        {KEY: histogram([0, 2, 4, 0], [0, 4, 16, 0])},
        {KEY: histogram([0, 4, 2, 0])})
    r = report[KEY]
    assert r["normalization"]["visible"]["difference"] == 0
    assert r["shape"]["visible"]["total_variation"] == pytest.approx(1 / 3)
    assert not r["closure"]["sumw2_presence_equal"]
    assert r["closure"]["exact_values_and_variances"] is None
    assert summary["variance_coverage_gaps"] == [KEY]


def test_malformed_sumw2_and_numeric_overflow_are_reported(tmp_path):
    bad = KEY + "_malformed"
    huge = KEY + "_overflow"
    with np.errstate(over="ignore"):
        summary, report = run(tmp_path,
            {bad: histogram([0, 1, 0], [1, 2]), huge: histogram([0, 1e308, 1e308, 0])},
            {bad: histogram([0, 1, 0], [0, 1, 0]), huge: histogram([0, 1, 1, 0])})
    assert "ref_malformed_sumw2" in summary["coverage_gaps"][bad]
    assert "numeric_overflow" in summary["coverage_gaps"][huge]
    assert report[huge]["ref_data"]["regions"]["visible"]["value_sum"] is None


def test_alias_does_not_touch_other_directories_or_unrelated_prefixes(tmp_path):
    keys = ["other/CrvDigis_fast/h", "Validation/CrvDigis_faster/h"]
    ref, cand = files(tmp_path, {k: histogram([0, 1, 0]) for k in keys}, {k: histogram([0, 1, 0]) for k in keys})
    summary = audit(ref, cand, tmp_path / "audit", select="", aliases=["CrvDigis_fast=CrvDigis_stock"])
    report = json.loads((tmp_path / "audit" / "histograms.json").read_text())
    assert sorted(report) == sorted(keys)
    assert summary["statuses"] == {"compared": 2}


def test_explicit_reviewed_count_key_and_metadata(tmp_path):
    from uproot.models.TNamed import Model_TNamed
    from uproot.models.TObject import Model_TObject
    def named(title):
        obj = Model_TNamed.empty()
        obj._deeply_writable = True
        obj._bases = [Model_TObject.empty()]
        obj._members = dict(fName='dqmBinningVersion', fTitle=title)
        return obj
    key = 'crvRecoDQM/NPc'
    tag = 'crvRecoDQM/dqmBinningVersion'
    ref, cand = files(tmp_path, {key: histogram([0, 1, 2, 0]), tag: named('3')},
                       {key: histogram([0, 2, 1, 0]), tag: named('3')})
    result = audit(ref, cand, tmp_path / 'audit', count_keys=[key])
    reports = json.loads((tmp_path / 'audit/histograms.json').read_text())
    assert reports[key]['shape']['visible']['total_variation'] == pytest.approx(1 / 3)
    assert reports[tag]['metadata_equal'] is True
    assert result['classes_ref'] == {'TH1D': 1, 'TNamed': 1}
    assert result['dimensions'] == {1: 1}
    assert not result['coverage_gaps']
    with uproot.update(cand) as f:
        f[tag] = named('4')
    result = audit(ref, cand, tmp_path / 'audit2')
    assert result['coverage_gaps'][tag] == ['metadata_mismatch']
