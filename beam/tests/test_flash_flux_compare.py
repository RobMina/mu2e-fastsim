"""Synthetic paired statistics and local plotting; no physics-validation claim."""
import json

import numpy as np
import pytest

from beamfastsim.flash_flux_compare import FIELDS, compare, event_histograms, main, paired, read_arm, spectrum


def test_paired_ratio_and_standard_error_with_known_zeros():
    lo, hi = np.array([1., 3., 0.]), np.array([3., 2., 1.])
    n, ratio = 7, 1.5
    expected = np.sqrt(np.sum((hi-ratio*lo)**2)/(n*(n-1)))/(lo.sum()/n)
    got = paired(lo, hi, n)
    assert got["ratio"] == ratio
    assert got["ratio_se"] == pytest.approx(expected)
    assert got["low_per_event"] == pytest.approx(4/7)
    assert got["low_se"] == pytest.approx(np.std(np.r_[lo, np.zeros(4)], ddof=1)/np.sqrt(n))


def test_perfect_covariance_cancels_ratio_uncertainty():
    lo = np.array([0., 3., 20., 1.])
    got = paired(lo, 2.5*lo, 9)
    assert got["ratio"] == 2.5 and got["ratio_se"] == 0
    assert got["low_se"] > 0 and got["high_se"] > 0


@pytest.mark.parametrize("lo,hi,ratio,se", [([0, 0], [1, 2], None, None),
                                                   ([0, 0], [0, 0], None, None),
                                                   ([1, 2], [0, 0], 0, 0),
                                                   ([1], [2], 2, None), ([], [], None, None)])
def test_zero_arms_and_one_event(lo, hi, ratio, se):
    got = paired(lo, hi, max(1, len(lo)))
    assert got["ratio"] == ratio and got["ratio_se"] == se
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("n", [0, -1, 1, 2.5, True, np.nan, np.inf])
def test_invalid_denominators(n):
    with pytest.raises(ValueError, match="denominator"):
        paired([1, 2], [2, 3], n)


@pytest.mark.parametrize("lo,hi", [([1], [1, 2]), ([[1]], [[2]]), ([np.nan], [1]), ([1], [-1])])
def test_invalid_totals(lo, hi):
    with pytest.raises(ValueError):
        paired(lo, hi, 3)


def arm(path, factor=1, n=6, mapping=True, event=None):
    path.mkdir()
    if event is None:
        event = np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 4]], dtype=int)
    parent = np.tile([0, 0, 1, 3], factor)
    energy = np.tile([.1, 1., 10., 100.], factor)
    weight = np.full(len(parent), .5)
    edges = np.array([.1, 1., 10., 100.])
    values = [np.bincount(parent, weights=w, minlength=4) for w in (np.ones(len(parent)), weight, energy, weight*energy)]
    totals = dict(zip(FIELDS, values))
    np.savez(path / "event_totals.npz", event=event, **totals)
    report = dict(source_events=n, totals={k: float(v.sum()) for k, v in totals.items()},
                  energy_edges_mev=edges.tolist(), histograms=dict(transparent=np.histogram(energy, edges)[0].tolist(),
                                                                 uncollided=np.histogram(energy, edges, weights=weight)[0].tolist()))
    (path / "flux.json").write_text(json.dumps(report))
    if mapping:
        np.savez(path / "births.npz", source_run=event[:, 0], source_subrun=event[:, 1], source_event=event[:, 2])
        np.savez(path / "foil_exit.npz", parent=parent, energy=energy, weight=weight, foil_exit=np.ones(len(parent), bool))
    return event, totals


def inputs(tmp_path, **kwargs):
    low, high, out = (tmp_path / k for k in ("low", "high", "out"))
    arm(low, **kwargs)
    arm(high, factor=2, **kwargs)
    return low, high, out


def test_cli_writes_strict_json_and_pngs_with_paired_shape(tmp_path):
    low, high, out = inputs(tmp_path)
    assert main(["--low", str(low), "--high", str(high), "--out", str(out), "--source-events", "6"]) == 0
    result = json.loads((out / "numbers.json").read_text())
    assert result["known_zero_input_events"] == 2
    assert result["shard_count"] == 2
    assert result["metrics"]["transparent_photons"]["ratio"] == 2
    assert result["metrics"]["transparent_photons"]["ratio_se"] == 0
    assert result["metrics"]["transparent_photons"]["shard_ratio_se"] == 0
    assert [s["represented_events"] for s in result["per_shard"]] == [3, 1]
    for kind in ("transparent", "uncollided"):
        bins = result["spectra"][kind]["bins"]
        assert all(b["ratio"] == 2 and b["ratio_se"] == 0 for b in bins)
        assert all(b["shape_difference_per_event"] == 0 and b["shape_se"] == 0 and b["shape_pull"] is None for b in bins)
    for name in ("ratio_summary.png", "energy_spectrum.png"):
        assert (out / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = (out / "numbers.json").read_text()
    assert "NOT flux" in text and "no cascades/annihilation" in text
    assert str(tmp_path) not in text


def test_missing_mapping_is_explicit_no_shape_uncertainty(tmp_path, monkeypatch):
    low, high, out = inputs(tmp_path, mapping=False)
    monkeypatch.setattr("beamfastsim.flash_flux_compare.plot", lambda *_: None)
    got = compare(low, high, out, 6)
    for s in got["spectra"].values():
        assert "shape uncertainty not evaluated" in s["uncertainty"]
        assert all(b["ratio"] == 2 and b["ratio_se"] is None for b in s["bins"])


def test_unequal_shard_sizes_use_sum_ratio_and_cluster_residuals(tmp_path, monkeypatch):
    low, high, out = inputs(tmp_path, mapping=False)
    with np.load(high / "event_totals.npz") as z:
        data = {k: z[k] for k in z.files}
    data["transparent_photons"] = np.array([1., 0., 0., 5.])
    np.savez(high / "event_totals.npz", **data)
    report = json.loads((high / "flux.json").read_text())
    report["totals"]["transparent_photons"] = 6.
    (high / "flux.json").write_text(json.dumps(report))
    monkeypatch.setattr("beamfastsim.flash_flux_compare.plot", lambda *_: None)
    got = compare(low, high, out, 6)["metrics"]["transparent_photons"]
    assert got["ratio"] == 6/4
    residual = np.array([1., 5.]) - 1.5*np.array([3., 1.])
    assert got["shard_ratio_se"] == pytest.approx(np.sqrt(2*np.sum(residual**2))/4)
    assert got["ratio"] != pytest.approx(np.mean([1/3, 5]))


def test_shape_pull_includes_fitted_ratio_covariance():
    lo = np.array([[2., 0.], [0., 1.], [3., 0.]])
    hi = np.array([[1., 1.], [1., 3.], [1., 1.]])
    lt, ht, n = lo.sum(axis=1), hi.sum(axis=1), 5
    fit = ht.sum()/lt.sum()
    got = spectrum(lo, hi, lt, ht, n)
    for i in range(2):
        residual = hi[:, i] - fit*lo[:, i]
        influence = residual-lo[:, i].sum()/lt.sum()*(ht-fit*lt)
        se = np.std(np.r_[influence, 0., 0.], ddof=1)/np.sqrt(n)
        assert got[i]["shape_se"] == pytest.approx(se)
        assert got[i]["shape_pull"] == pytest.approx(residual.sum()/n/se)


def test_event_histograms_keep_last_edge_and_zero_response_event(tmp_path):
    path = tmp_path / "arm"
    event, _ = arm(path)
    h = event_histograms(path, event, np.array([.1, 1., 10., 100.]))
    np.testing.assert_array_equal(h["transparent"], [[1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]])


@pytest.mark.parametrize("change", ["identity", "order", "duplicate", "denominator", "total", "bins", "histogram"])
def test_reject_incompatible_comparisons(tmp_path, change):
    low, high, out = inputs(tmp_path)
    with np.load(high / "event_totals.npz") as z:
        data = {k: z[k] for k in z.files}
    report = json.loads((high / "flux.json").read_text())
    if change == "identity":
        data["event"][0, 2] = 99
    elif change == "order":
        data = {k: v[::-1] for k, v in data.items()}
    elif change == "duplicate":
        data["event"][0] = data["event"][1]
    elif change == "denominator":
        report["source_events"] = 7
    elif change == "total":
        report["totals"]["transparent_photons"] += 1
    elif change == "bins":
        report["energy_edges_mev"][1] = 2
    else:
        report["histograms"]["transparent"][0] += 1
    np.savez(high / "event_totals.npz", **data)
    (high / "flux.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare(low, high, out, 6)
    assert not out.exists()


@pytest.mark.parametrize("key", ["dropped_births", "capped_parents"])
def test_truncation_is_rejected(tmp_path, key):
    low, high, out = inputs(tmp_path)
    report = json.loads((high / "flux.json").read_text())
    report[key] = 1
    (high / "flux.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Truncated"):
        compare(low, high, out, 6)


def test_wrong_parent_mapping_rejected_even_if_histogram_matches(tmp_path):
    low, high, out = inputs(tmp_path)
    with np.load(high / "foil_exit.npz") as z:
        photons = {k: z[k] for k in z.files}
    photons["parent"][:] = 2
    np.savez(high / "foil_exit.npz", **photons)
    with pytest.raises(ValueError, match="parent mapping"):
        compare(low, high, out, 6)


def test_zero_response_comparison_plots_without_infinity(tmp_path):
    low, high, out = (tmp_path / k for k in ("low", "high", "out"))
    arm(low, factor=0)
    arm(high, factor=0)
    result = compare(low, high, out, 6)
    assert all(m["ratio"] is None and m["ratio_se"] is None for m in result["metrics"].values())
    text = (out / "numbers.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_shape_chi2_uses_covariance_and_fitted_normalization():
    from beamfastsim.flash_flux_compare import shape_chi2
    lo = np.array([[2., 0.], [0., 1.], [3., 0.], [0., 1.]])
    hi = np.array([[1., 1.], [1., 3.], [1., 1.], [0., 3.]])
    lt, ht = lo.sum(axis=1), hi.sum(axis=1)
    result = shape_chi2(lo, hi, lt, ht, 4)
    assert result["ndf"] == 1 and result["chi2"] > 0
    exact = shape_chi2(lo, 2 * lo, lt, 2 * lt, 4)
    assert exact["ndf"] == 0 and exact["chi2"] is None


@pytest.mark.parametrize("large", [2**53 + 1, 2**63, 2**64 - 1])
def test_event_histograms_preserve_mixed_integer_source_ids(tmp_path, large):
    event = np.array([[1, 1, large - i] for i in range(4)], dtype=np.uint64)
    path = tmp_path / "arm"
    _, totals = arm(path, event=event)
    source = dict(source_run=event[:, 0].astype(np.int64),
                  source_subrun=event[:, 1], source_event=event[:, 2])
    np.savez(path / "births.npz", **source)
    hist = event_histograms(path, event, np.array([.1, 1., 10., 100.]), totals)
    np.testing.assert_array_equal(hist["transparent"], [[1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]])
    # Adjacent values above float's exact range must not match an absent ID.
    source["source_event"] = source["source_event"].copy()
    source["source_event"][0] = large - 4
    np.savez(path / "births.npz", **source)
    with pytest.raises(ValueError, match="absent from event universe"):
        event_histograms(path, event, np.array([.1, 1., 10., 100.]))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_read_arm_canonicalizes_exact_event_ids(tmp_path, dtype):
    event = np.array([[1, 1, 2**53 + i] for i in range(4)], dtype=dtype)
    path = tmp_path / "arm"
    arm(path, event=event)
    _, data = read_arm(path, 6)
    assert data["event"].dtype == np.dtype("uint64")
    assert data["event"].tolist() == event.tolist()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("offset", [0, 1])
def test_compare_exact_mixed_integer_event_ids(tmp_path, monkeypatch, dtype, offset):
    low, high, out = (tmp_path / k for k in ("low", "high", "out"))
    event = np.array([[1, 1, 2**53 + 2*i] for i in range(4)], dtype=dtype)
    other = event.astype(np.uint64 if dtype == np.int64 else np.int64)
    other[0, 2] += offset
    arm(low, event=event)
    arm(high, factor=2, event=other)
    monkeypatch.setattr("beamfastsim.flash_flux_compare.plot", lambda *_: None)
    if offset:
        with pytest.raises(ValueError, match="Exact same event arrays required"):
            compare(low, high, out, 6)
        assert not out.exists()
    else:
        result = compare(low, high, out, 6)
        assert result["metrics"]["transparent_photons"]["ratio"] == 2
        assert result["spectra"]["transparent"]["uncertainty"] == "paired event delta method"


@pytest.mark.parametrize("column", [0, 1, 2])
def test_negative_event_and_birth_ids_rejected(tmp_path, column):
    path = tmp_path / "arm"
    event, totals = arm(path)
    event[0, column] = -1
    np.savez(path / "event_totals.npz", event=event, **totals)
    with pytest.raises(ValueError, match="nonnegative integer"):
        read_arm(path, 6)
    with pytest.raises(ValueError, match="nonnegative integer"):
        event_histograms(path, event, np.array([.1, 1., 10., 100.]))
    np.savez(path / "births.npz", **{"source_" + k: event[:, i]
                                   for i, k in enumerate(("run", "subrun", "event"))})
    # A signed negative must not wrap into a valid unsigned universe identity.
    unsigned_event = event.astype(np.uint64)
    with pytest.raises(ValueError, match="nonnegative integer"):
        event_histograms(path, unsigned_event, np.array([.1, 1., 10., 100.]))
