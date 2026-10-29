"""Saved optical rows are checked by identity and exact masks, not regenerated draws."""
import json

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import optical_timing_probe as probe, optical_timing_report as report
from crvfastsim.veto.replay import sha256
from test_veto_optical_timing_probe import sample, call, fake_optics


def tables(monkeypatch):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    st["endpoint_roundoff"] = [False] * 5 + [True]
    q, rng = call(st, ev, geom)
    q["endpoint_roundoff_step"] = st.endpoint_roundoff.to_numpy()[q.step_index]
    frames, channels = probe.summaries(q, ev, geom)
    return st, frames, q, channels, geom


def test_saved_recalculation_and_exact_counts(monkeypatch):
    st, frames, q, channels, geom = tables(monkeypatch)
    report.validate_saved(st, frames, q, channels, geom)
    stats = report.block_summary(st, q, frames)
    assert stats["frames"] == 3 and stats["photons"] == 10
    assert stats["extreme_step_photons"] == 2
    assert stats["extreme_photon_rows"][0]["source_step_row"] == 15


@pytest.mark.parametrize("column", ["source_step_row", "event_row", "chan", "ordinal", "photon_id", "run",
                                   "time_raw_wrapped_relative", "raw_folded_residual_ns", "historical_kept",
                                   "stock_step_kept", "rescued_survives_step", "endpoint_roundoff_step"])
def test_corrupt_photon_field_rejected(monkeypatch, column):
    st, frames, q, channels, geom = tables(monkeypatch)
    value = q.loc[0, column]
    q.loc[0, column] = not value if q[column].dtype == bool else value + 1
    with pytest.raises((ValueError, AssertionError)):
        report.validate_saved(st, frames, q, channels, geom)


@pytest.mark.parametrize("which", ["frames", "channels", "step"])
def test_corrupt_summary_or_parent_identity_rejected(monkeypatch, which):
    st, frames, q, channels, geom = tables(monkeypatch)
    if which == "frames":
        frames.loc[0, "photons"] += 1
    elif which == "channels":
        channels.loc[0, "historical_kept"] += 1
    else:
        st.loc[0, "source_row"] += 1
    with pytest.raises((ValueError, AssertionError)):
        report.validate_saved(st, frames, q, channels, geom)


def test_report_end_to_end_hashes_and_no_new_draw(tmp_path, monkeypatch):
    st, frames, q, channels, geom = tables(monkeypatch)
    counter = tmp_path / "counters.txt"
    counter.write_text("synthetic geometry")
    monkeypatch.setattr(probe, "COUNTERS_SHA", sha256(counter))
    monkeypatch.setattr(report.geometry, "from_offline_config", lambda path: geom)
    artifacts = {}
    for name, frame in (("steps", st), ("events", frames), ("photons", q), ("channels", channels)):
        path = tmp_path / f"input0_{name}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[name] = dict(path=path.name, sha256=sha256(path), rows=len(frame))
    stats = report.block_summary(st, q, frames)
    block = {k: stats[k] for k in ("photons", "counts", "extreme_step_photons", "extreme_step_counts", "max_raw_folded_circular_residual_ns")}
    block.update(seed=21, input_sha256="1" * 64, input_bindings={}, artifacts=artifacts,
                 selected_events=frames.to_dict(orient="records"))
    meta = dict(schema="crv-optical-timing-probe-v1", rng_contract=probe.RNG_CONTRACT, F=None, timing_closure=False,
                assets={str(counter): sha256(counter)}, inputs={}, code_sha256={}, blocks=[block])
    (tmp_path / "summary.json").write_text(json.dumps(meta))
    monkeypatch.setattr(probe.photons, "make_photons", lambda *args: pytest.fail("must not regenerate photons"))
    out = tmp_path / "verified.json"
    result = report.run(tmp_path, out)
    assert result["all_saved_fields_verified"] is True and result["new_optical_draw"] is False
    with pytest.raises(ValueError, match="new report"):
        report.run(tmp_path, out)
    (tmp_path / "input0_photons.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="bound input"):
        report.run(tmp_path, tmp_path / "other.json")
