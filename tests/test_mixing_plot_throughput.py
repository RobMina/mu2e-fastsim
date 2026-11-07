"""The slide figure consumes measured process rates, not response-only stage timers."""
import json

import pytest

from mixing.plot_throughput import ENDPOINT_SCOPE, hardware_label, main, make_figure, public_numbers


def receipt():
    return {"events": 3, "hardware": {"label": "fast endpoint"},
            "stock": {"external_process": {"events_per_s": 1.5}, "hardware": {"label": "stock endpoint"}},
            "runs": [{"python_process": {"events_per_s": rate}, "validation": {"outputs": {"equal": True}},
                      "cache_state": "fresh_private_cache" if index == 0 else "populated_cache_fresh_process"}
                     for index, rate in enumerate((2., 3., 3.1))]}


def test_measured_figure_and_no_overwrite(tmp_path):
    path, out = tmp_path / "receipt.json", tmp_path / "plot.png"
    path.write_text(json.dumps(receipt()))
    main([str(path), "--out", str(out)])
    assert out.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(SystemExit):
        main([str(path), "--out", str(out)])


@pytest.mark.parametrize("mutation", ["failed", "missing", "nonfinite"])
def test_bad_measurement_never_becomes_figure(tmp_path, mutation):
    data = receipt()
    if mutation == "failed":
        data["runs"][0]["validation"]["outputs"]["equal"] = False
    elif mutation == "missing":
        data["runs"].pop()
    else:
        data["runs"][0]["python_process"]["events_per_s"] = float("nan")
    with pytest.raises(ValueError):
        make_figure(data, tmp_path / "plot.png")
    assert not (tmp_path / "plot.png").exists()


@pytest.mark.parametrize("metadata", [None, {}, {"text": "raw provenance only"}])
def test_missing_label_renders_hardware_not_recorded(tmp_path, metadata):
    assert hardware_label(metadata) == "hardware not recorded"
    data = receipt()
    data["hardware"] = metadata
    data["stock"]["hardware"] = metadata
    make_figure(data, tmp_path / "plot.png")
    assert (tmp_path / "plot.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("label", ["", "   ", "device\nhost", "device\rhost", 2])
@pytest.mark.parametrize("side", ["fast", "stock"])
def test_invalid_label_raises(tmp_path, label, side):
    data = receipt()
    (data if side == "fast" else data["stock"])["hardware"] = {"label": label}
    with pytest.raises(ValueError, match="single-line public label"):
        make_figure(data, tmp_path / "plot.png")
    assert not (tmp_path / "plot.png").exists()


def test_wrong_cache_order_raises(tmp_path):
    data = receipt()
    data["runs"][0]["cache_state"], data["runs"][1]["cache_state"] = (
        data["runs"][1]["cache_state"], data["runs"][0]["cache_state"])
    with pytest.raises(ValueError, match="expected a fresh cache"):
        make_figure(data, tmp_path / "plot.png")
    assert not (tmp_path / "plot.png").exists()


def test_public_number_export_drops_internal_provenance():
    data = receipt()
    data.update(denominator="all events", boundary="steps to output", excluded=["transport"],
                cache_scope="private", comparison_scope="historical only", rootdir="/private/path")
    data["stock"]["external_process"]["wall_s"] = 2.
    for run in data["runs"]:
        run["python_process"]["wall_s"] = 1.
        run["outer_container_process"] = {"wall_s": 1.1, "events_per_s": 2.7}
        run["stages"] = {"setup_s": .1, "write_s": .1, "passes": [{"combined_response_s": .7}]}
        run["directory"] = "/private/job123"
    public = public_numbers(data)
    assert len(public["fast"]) == 3
    assert public["fast"][0]["events_per_s"] == 2.
    assert "/private" not in json.dumps(public)
    assert "hardware" not in public and "rootdir" not in public
    assert public["endpoint_scope"] == ENDPOINT_SCOPE
    assert public["stock"]["endpoint"] == "stock endpoint"
    assert [run["endpoint"] for run in public["fast"]] == ["fast endpoint"] * 3
