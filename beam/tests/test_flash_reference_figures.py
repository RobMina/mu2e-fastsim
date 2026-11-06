"""Reference figure inputs preserve full denominators and energy components."""
import copy
import json

import numpy as np
import pytest

from beamfastsim.flash_reference_figures import MISSING_SE_NOTE, components, main, plot_components, plot_spectra
from beamfastsim.flash_response_compare import compare as compare_response
from beamfastsim.flash_s1_spectrum import compare as compare_spectrum
from test_flash_response_compare import sample
from test_flash_s1_spectrum import case as spectrum_case


@pytest.fixture
def case():
    origin = dict(input_events=10, foil_ancestry=dict(status=[0, 0, 1, 2],
        foil_born_observed=[False, True, True, True], ancestry_complete=[True]*4,
        ionizing_mev=[1., 2., 3., .5]))
    comparison = dict(events=10, totals={k: dict(low=dict(sum_mev=v, event_mean_se=.2))
        for k, v in (("raw", 6.5), ("owned", 3.), ("residual", 3.), ("unknown", .5))})
    return origin, comparison


def test_units_and_component_order(case):
    values, error = components(*case, "low")
    np.testing.assert_array_equal(values, [300000., 200000., 100000., 50000.])
    assert error == 200000.


def test_mismatched_events_fail(case):
    case[0]["input_events"] = 9
    with pytest.raises(ValueError, match="denominators"):
        components(*case, "low")


def test_energy_mismatch_fails(case):
    case[0]["foil_ancestry"]["ionizing_mev"][0] += 1
    with pytest.raises(ValueError, match="components"):
        components(*case, "low")


def test_unknown_metadata_fails(case):
    case[0]["foil_ancestry"]["ancestry_complete"][0] = False
    with pytest.raises(ValueError, match="complete"):
        components(*case, "low")


@pytest.fixture
def figures(monkeypatch):
    pytest.importorskip("matplotlib.pyplot")
    from matplotlib.figure import Figure

    saved = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


def single_event_spectrum(spectrum_case):
    fast, report, source, photons, refs = copy.deepcopy(spectrum_case)
    fast = {k: v[1:2].copy() for k, v in fast.items()}
    source = {k: v[:1].copy() for k, v in source.items()}
    photons = {k: v[:2].copy() for k, v in photons.items()}
    reference = {}
    for k, v in refs[0].items():
        if k.endswith("report_json"):
            reference[k] = v.copy()
        else:
            reference[k] = v[1:2].copy() if k.startswith(("event_", "evt_")) else v[:1].copy()
    diagnostics = json.loads(str(reference["report_json"]))
    diagnostics.update(expected_events=1, input_events=1, primary_birth_count=1,
                       primary_uncollided_exit_count=1, unresolved_count=0,
                       unknown_history_count=0, primary_no_recorded_exit_count=0)
    reference["report_json"] = np.asarray(json.dumps(diagnostics))
    report.update(source_events=1, loopers=0)
    return compare_spectrum(fast, report, source, photons, [reference])


def single_event_components():
    origin = dict(input_events=1, foil_ancestry=dict(status=[1], foil_born_observed=[True],
        ancestry_complete=[True], ionizing_mev=[1.]))
    data = sample(n=1, parents=(0,), response=(1.,))
    return origin, compare_response(data, data)


@pytest.mark.parametrize("mixed_samples", [False, True])
def test_single_event_spectra_keep_central_values_without_error_bars(
        tmp_path, spectrum_case, figures, mixed_samples):
    low = single_event_spectrum(spectrum_case)
    high = compare_spectrum(*spectrum_case) if mixed_samples else low
    assert low["input_events"] == 1
    assert low["spectra"]["photons"]["fast_total_event_se"] == [None] * 9
    json.dumps(low, allow_nan=False)
    out = tmp_path / "single-spectrum.png"
    plot_spectra(low, high, out)
    assert out.stat().st_size > 0
    figure = figures[-1]
    assert any(MISSING_SE_NOTE in text.get_text() for text in figure.texts)
    for row, metric in enumerate(("photons", "energy_mev")):
        for col, report in enumerate((low, high)):
            ax = figure.axes[row * 2 + col]
            for container, model in zip(ax.containers, ("fast", "g4")):
                assert container.has_yerr == (report["input_events"] > 1)
                expected = np.asarray(report["spectra"][metric][model + "_total"]) / report["input_events"]
                np.testing.assert_array_equal(container.lines[0].get_ydata(), expected)
    assert low["spectra"]["photons"]["fast_total_event_se"] == [None] * 9


def test_single_event_components_preserve_unavailable_error(tmp_path, figures):
    origin, comparison = single_event_components()
    values, error = components(origin, comparison, "low")
    np.testing.assert_array_equal(values, [1e6, 0., 0., 0.])
    assert error is None
    out = tmp_path / "single-components.png"
    plot_components(origin, origin, comparison, out)
    assert out.stat().st_size > 0
    figure = figures[-1]
    assert any(MISSING_SE_NOTE in text.get_text() for text in figure.texts)
    assert not any(hasattr(container, "has_yerr") for container in figure.axes[0].containers)
    assert comparison["totals"]["raw"]["low"]["event_mean_se"] is None


@pytest.mark.parametrize("zero", [False, True])
def test_multi_event_spectra_render_with_available_errors(tmp_path, spectrum_case, figures, zero):
    report = compare_spectrum(*spectrum_case)
    if zero:
        for metric in report["spectra"].values():
            for name in ("fast_total", "g4_total", "fast_total_event_se", "g4_total_event_se"):
                metric[name] = [0.] * 9
    out = tmp_path / "spectra.png"
    plot_spectra(report, report, out)
    assert out.stat().st_size > 0
    figure = figures[-1]
    assert not any(MISSING_SE_NOTE in text.get_text() for text in figure.texts)
    assert all(container.has_yerr for ax in figure.axes for container in ax.containers)


@pytest.mark.parametrize("zero", [False, True])
def test_multi_event_components_render_with_available_errors(tmp_path, case, figures, zero):
    origin, comparison = case
    if zero:
        origin["foil_ancestry"] = {k: [] for k in origin["foil_ancestry"]}
        for total in comparison["totals"].values():
            total["low"].update(sum_mev=0., event_mean_se=0.)
    for total in comparison["totals"].values():
        total["high"] = dict(total["low"])
    out = tmp_path / "components.png"
    plot_components(origin, origin, comparison, out)
    assert out.stat().st_size > 0
    figure = figures[-1]
    assert not any(MISSING_SE_NOTE in text.get_text() for text in figure.texts)
    assert figure.axes[0].containers[-1].has_yerr
    assert figure.axes[0].get_ylim()[1] > 0


def test_cli_single_event_reports_write_figures_and_provenance(tmp_path, spectrum_case, figures):
    spectrum = single_event_spectrum(spectrum_case)
    origin, comparison = single_event_components()
    reports = {"spectrum-low": spectrum, "spectrum-high": spectrum,
               "origins-low": origin, "origins-high": origin, "comparison": comparison}
    argv = []
    for name, report in reports.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(report, allow_nan=False))
        argv.extend(["--" + name, str(path)])
    out = tmp_path / "figures"
    assert main([*argv, "--out", str(out)]) == 0
    assert (out / "flash_reference_spectra.png").stat().st_size > 0
    assert (out / "flash_reference_components.png").stat().st_size > 0
    provenance = json.loads((out / "inputs.json").read_text())
    assert provenance["comparison"] == str(tmp_path / "comparison.json")
    assert all(any(MISSING_SE_NOTE in text.get_text() for text in fig.texts) for fig in figures)
