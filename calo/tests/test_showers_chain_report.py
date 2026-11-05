"""Saved-report CLI regressions preserve unavailable errors, ratios and warm rates."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest

from calofastsim.showers import chain_report
from calofastsim.showers.measure_chain import compare
from calofastsim.showers.reconstruct import report as reconstruction_report


PAIRS = ("fast_visible_vs_stock_visible", "fast_response_vs_stock_response", "proxy_wall_vs_stock_visible")


def sample(n, *, zero_stock=False, zero_fast=False):
    arrays = {k: np.zeros(n) for k in ("fast_visible", "fast_response", "stock_visible", "stock_response", "proxy_wall", "proxy_vacuum")}
    selected = dict(fast_visible=[0, 100, 102], fast_response=[0, 1, 100], stock_visible=[0, 1, 100, 101],
                    stock_response=[0, 100], proxy_wall=[0, 1, 100], proxy_vacuum=[0, 1, 100, 101])
    for key, rows in selected.items():
        arrays[key][[i for i in rows if i < n]] = 1 if key.startswith("proxy") else 20
        if (zero_stock and key.startswith("stock")) or (zero_fast and key.startswith("fast")):
            arrays[key][:] = 0
    return arrays


def saved_chain(n=200, *, rate=100., zero_stock=False, zero_fast=False):
    """Use measured statistics, adding only the metadata and timing required by the CLI."""
    result = compare(sample(n, zero_stock=zero_stock, zero_fast=zero_fast), np.arange(n))
    warm = n / rate if rate is not None else None
    measurements = [dict(chain_seconds=4.)]
    if warm is not None:
        measurements.append(dict(chain_seconds=warm))
    result.update(error_method="paired 100-source-event block jackknife; library training uncertainty excluded",
                  model="saved CE-chain regression", versions={"numpy": np.__version__}, backend="numpy",
                  timing=dict(measurements=measurements, warm_chain_seconds=warm, warm_generated_per_second=rate,
                              stock_generated_per_second=23.9, stock_grid_slot_equivalent=rate / 23.9 if rate is not None else None,
                              boundary_note="primary-only response sum, not matched reconstruction"))
    return result


def saved_reco(n=200, *, rate=25., zero_stock=False):
    arrays = sample(n, zero_stock=zero_stock)
    fast = dict(energy=arrays["fast_response"], **{k: np.zeros(n) for k in ("time", "x", "y", "z")})
    stock = dict(fast, energy=arrays["stock_response"])
    result = reconstruction_report(fast, stock, np.arange(n))
    result.update(timing=dict(generated_per_second=rate), reco_commit="saved-regression", device=["cpu"])
    return result


@pytest.fixture
def rendering(monkeypatch):
    """Capture actual plotting calls while still rendering the CLI's figures."""
    calls, figures = {}, {}
    for name in ("errorbar", "plot", "bar"):
        original = getattr(Axes, name)
        calls[name] = []

        def capture(self, *args, _name=name, _original=original, **kwargs):
            calls[_name].append((self, args, kwargs))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Axes, name, capture)
    original_savefig = Figure.savefig

    def savefig(self, path, *args, **kwargs):
        figures[path.name] = self
        return original_savefig(self, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    return calls, figures


def run_cli(tmp_path, monkeypatch, capsys, reports, reco=None):
    argv = ["chain_report", "--out", str(tmp_path / "rendered")]
    for prefix, group in (("", reports), ("reco-", reco or {})):
        for device, result in group.items():
            option = prefix + ("cpu" if device == "CPU" else "gpu")
            path = tmp_path / (option + ".json")
            path.write_text(json.dumps(result, allow_nan=False))
            argv.extend(["--" + option, str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    chain_report.main()
    out = tmp_path / "rendered"
    summary = json.loads((out / "chain_numbers.json").read_text())
    assert json.loads(capsys.readouterr().out) == summary
    assert json.loads((out / "numbers.json").read_text())["generated_event_chain"] == summary
    for device, result in reports.items():
        assert summary[device] == chain_report.public_numbers(result)
    for device, result in (reco or {}).items():
        assert summary["reconstruction"][device] == chain_report.reco_numbers(result)
    assert (out / "calo_chain.png").stat().st_size > 0
    if reco:
        assert (out / "calo_cluster.png").stat().st_size > 0
    return summary


def assert_rendering(rendering, reports, reco=None):
    """Require exact measured values; no null statistic may become a zero-valued glyph."""
    calls, figures = rendering
    ratio_axis, rate_axis = figures["calo_chain.png"].axes
    pairs = [(report[key]["ratio"], report[key]["ratio_error"]) for report in reports.values() for key in PAIRS]
    pairs += [(report["acceptance_ratio"]["ratio"], report["acceptance_ratio"]["error"]) for report in (reco or {}).values()]
    measured = [(i, value, error) for i, (value, error) in enumerate(pairs) if value is not None and error is not None]
    unknown = [(i, value) for i, (value, error) in enumerate(pairs) if value is not None and error is None]
    undefined = [i for i, (value, _) in enumerate(pairs) if value is None]
    errorbars = [(args, kwargs) for axis, args, kwargs in calls["errorbar"] if axis is ratio_axis]
    markers = [(args, kwargs) for axis, args, kwargs in calls["plot"] if axis is ratio_axis]
    assert len(errorbars) == bool(measured)
    assert len(markers) == bool(unknown)
    if measured:
        args, kwargs = errorbars[0]
        np.testing.assert_array_equal(args[0], [value for _, value, _ in measured])
        np.testing.assert_array_equal(args[1], [i for i, _, _ in measured])
        np.testing.assert_array_equal(kwargs["xerr"], [error for _, _, error in measured])
        assert np.isfinite(kwargs["xerr"]).all()
    if unknown:
        args, kwargs = markers[0]
        np.testing.assert_array_equal(args[0], [value for _, value in unknown])
        np.testing.assert_array_equal(args[1], [i for i, _ in unknown])
        assert args[2] == "o" and kwargs["linestyle"] == "none"
        assert "xerr" not in kwargs
    labels = [tick.get_text() for tick in ratio_axis.get_yticklabels()]
    assert len(labels) == len(pairs)
    assert [i for i, label in enumerate(labels) if "error unavailable" in label] == [i for i, _ in unknown]
    assert [text.get_position()[1] for text in ratio_axis.texts if text.get_text() == "undefined"] == undefined
    speed = reco if reco else reports
    key = "generated_per_second" if reco else "warm_generated_per_second"
    rates = [report["timing"][key] for report in speed.values()]
    available = [(i, rate) for i, rate in enumerate(rates) if rate is not None]
    bars = [(args, kwargs) for axis, args, kwargs in calls["bar"] if axis is rate_axis]
    assert len(bars) == bool(available)
    if available:
        args, _ = bars[0]
        np.testing.assert_array_equal(args[0], [i for i, _ in available])
        np.testing.assert_array_equal(args[1], [rate for _, rate in available])
        assert np.isfinite(args[1]).all()
    assert [patch.get_height() for patch in rate_axis.patches] == [rate for _, rate in available]
    assert [tick.get_text() for tick in rate_axis.get_xticklabels()] == list(speed)
    assert rate_axis.get_xlim() == (-.5, len(speed) - .5)
    assert [text.get_position()[0] for text in rate_axis.texts if text.get_text() == "unavailable"] == [i for i, rate in enumerate(rates) if rate is None]
    assert sorted(text.get_text() for text in rate_axis.texts if text.get_text() != "unavailable") == sorted(f"{rate:.1f}" for _, rate in available)
    assert np.isfinite(rate_axis.get_ylim()).all() and rate_axis.get_ylim()[1] > 0


@pytest.mark.parametrize("case", ["two-block", "one-block", "zero-stock", "zero-fast", "repeat-one", "mixed", "both-unavailable"])
def test_saved_chain_cli(tmp_path, monkeypatch, capsys, rendering, case):
    options = {"one-block": dict(n=8), "zero-stock": dict(zero_stock=True), "zero-fast": dict(zero_fast=True),
               "repeat-one": dict(rate=None), "mixed": dict(n=8, rate=None), "both-unavailable": dict(rate=None)}
    reports = {"CPU": saved_chain(**options.get(case, {}))}
    if case in ("mixed", "both-unavailable"):
        reports["MIG + host"] = saved_chain(rate=200. if case == "mixed" else None)
        reports["MIG + host"]["backend"] = "device"
    if case == "two-block":
        assert all(reports["CPU"][key]["ratio_error"] > 0 for key in PAIRS)
    if case == "one-block":
        assert all(reports["CPU"][key]["ratio_error"] is None for key in PAIRS)
    if case == "zero-stock":
        assert all(reports["CPU"][key]["ratio"] is None for key in PAIRS)
    if case == "repeat-one":
        assert len(reports["CPU"]["timing"]["measurements"]) == 1
    summary = run_cli(tmp_path, monkeypatch, capsys, reports)
    if reports["CPU"]["timing"]["warm_generated_per_second"] is None:
        assert summary["CPU"]["warm_generated_per_second"] is None
        assert summary["CPU"]["stock_grid_slot_equivalent"] is None
    assert_rendering(rendering, reports)


@pytest.mark.parametrize("case", ["two-block", "one-block", "zero-stock", "mixed", "both-unavailable"])
def test_saved_reconstruction_cli(tmp_path, monkeypatch, capsys, rendering, case):
    reports = {"CPU": saved_chain(rate=None), "MIG + host": saved_chain(rate=200.)}
    options = {"one-block": dict(n=8, rate=None), "zero-stock": dict(zero_stock=True),
               "mixed": dict(n=8, rate=None), "both-unavailable": dict(zero_stock=True, rate=None)}
    reco = {"CPU": saved_reco(**options.get(case, {}))}
    if case in ("mixed", "both-unavailable"):
        reco["MIG + host"] = saved_reco(rate=50. if case == "mixed" else None)
        reco["MIG + host"]["device"] = ["gpu"]
    if case == "one-block":
        assert reco["CPU"]["acceptance_ratio"]["error"] is None
    if case == "zero-stock":
        assert reco["CPU"]["acceptance_ratio"]["ratio"] is None
    summary = run_cli(tmp_path, monkeypatch, capsys, reports, reco)
    if reco["CPU"]["timing"]["generated_per_second"] is None:
        assert summary["reconstruction"]["CPU"]["timing"]["generated_per_second"] is None
    assert_rendering(rendering, reports, reco)
