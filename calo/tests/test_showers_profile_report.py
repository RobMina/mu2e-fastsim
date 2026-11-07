"""Saved full-route timing summaries reject unmatched workload or event universes."""
import json

import numpy as np
import pytest

from calofastsim.showers.profile_report import summarize


def inputs(tmp_path):
    paths = {}
    for name, warm in (("CPU", 40.), ("MIG", 10.)):
        path = tmp_path / name
        paths[name] = path
        for placement in ("geometric", "crystal"):
            out = path / placement
            out.mkdir(parents=True)
            source = dict(events=1000, timing=dict(method="single-call synchronized"), provenance={k: "fixed" for k in (
                "seed", "reco_commit", "reco_inputs", "stock_ntuple", "inputs", "library")},
                repeats=[dict(total_seconds=t, transport_seconds=t * .8, handoff_seconds=t * .1,
                              reco_seconds=t * .1, stage_seconds=dict(template_fit=t * .1), n_nonfinite_reco=0,
                              n_refit_skipped=1) for t in (warm * 2, warm, warm)],
                fast_selected=1, device=[name])
            (out / "numbers.json").write_text(json.dumps(source))
            arrays = {"fast_" + k: np.full(1000, 0. if k == "energy" else np.nan) for k in ("energy", "time", "x", "y", "z")}
            arrays["fast_energy"][0] = 20.
            np.savez(out / "clusters.npz", event_row=np.arange(1000), **arrays)
    return paths


def test_complete_rate_and_exact_fields(tmp_path):
    result = summarize(inputs(tmp_path))
    assert result["devices"]["CPU"]["geometric"]["generated_per_second"] == 25
    assert result["devices"]["MIG"]["geometric"]["generated_per_second"] == 100
    row = result["devices"]["MIG"]["crystal"]
    assert row["transport_fraction"] == .8
    assert row["vs_first_device"]["selection_equal"]
    assert all(field["exact"] for field in row["vs_first_device"]["fields"].values())


@pytest.mark.parametrize("kind", ["timing", "inputs", "rows"])
def test_unmatched_profiles_fail(tmp_path, kind):
    paths = inputs(tmp_path)
    path = paths["MIG"] / "crystal"
    if kind == "rows":
        with np.load(path / "clusters.npz") as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays["event_row"] = arrays["event_row"][::-1]
        np.savez(path / "clusters.npz", **arrays)
    else:
        source = json.loads((path / "numbers.json").read_text())
        if kind == "timing":
            source["timing"]["method"] = "sum of stages"
        else:
            source["provenance"]["seed"] = "different"
        (path / "numbers.json").write_text(json.dumps(source))
    with pytest.raises(ValueError):
        summarize(paths)
