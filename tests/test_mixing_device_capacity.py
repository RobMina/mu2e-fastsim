"""Capacity derivation from measured demand, and overflow messages that name demand and capacity."""
import json

import pytest

from mixing import device_capacity as dcap
from mixing.device_backend import _overflow_detail


TEMPLATE = {"max_donors": 8, "rows": {"tracker": 100, "crv": 50},
            "response": {"tracker": {"clusters_per_step": 64, "long_steps": 16, "clusters_per_long_step": 1024,
                                     "drift_clusters": 1000, "straws": 1024, "clusters_per_end": 32,
                                     "end_tiers": [[256, 96], [8, 6144]], "pairs_per_straw": 8, "electronics_batch": 64},
                         "crv": {"substeps": 100, "photons": 100, "thermal": 10, "ct1": 7, "charge_clusters": 20,
                                 "charges_per_cluster": 256, "waveform_samples": 256, "digis": 9, "reco_waveforms": 9,
                                 "reco_samples": 192, "coinc_hits": 64, "waveform": "scatter"}}}


def test_derive_scales_measured_demand_and_keeps_layout():
    usage = {"rows/tracker": 400, "rows/crv": 10, "max_donors": 30, "response/tracker/drift_clusters": 8000,
             "response/tracker/straws": 900, "response/tracker/long_steps": 3, "response/tracker/clusters_per_long_step": 300,
             "response/tracker/end_tier_1_straws": 70, "response/tracker/end_tier_2_straws": 1,
             "response/tracker/clusters_per_end": 2000, "response/tracker/clusters_per_step": 20,
             "response/crv/photons": 1000, "response/crv/digis": 40, "response/crv/charges_per_cluster": 10,
             "response/crv/reco_samples": 500}
    caps = dcap.derive(TEMPLATE, usage, 1.25)
    assert caps["rows"] == {"tracker": 500, "crv": 13} and caps["max_donors"] == 38
    t = caps["response"]["tracker"]
    assert t["drift_clusters"] == 10000 and t["straws"] == 1152 and t["straws"] % 64 == 0
    assert t["long_steps"] == 4 and t["clusters_per_long_step"] == 375
    assert t["end_tiers"] == [[128, 96], [2, 6144]]           # widths kept, counts multiples of their batch
    assert t["clusters_per_step"] == 64 and t["clusters_per_end"] == 32   # padding choices kept
    c = caps["response"]["crv"]
    assert c["photons"] == 1250 and c["digis"] == 50 and c["ct1"] == 7 and c["waveform"] == "scatter"
    assert c["charges_per_cluster"] == 256 and c["reco_samples"] == 625  # floors never shrink
    assert TEMPLATE["rows"]["tracker"] == 100                  # template untouched
    with pytest.raises(ValueError, match="margin"):
        dcap.derive(TEMPLATE, usage, 0.9)


def test_demands_reads_ledgers_and_bench_passes(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"backend": {"capacity_usage": {"rows/tracker": 5, "response/crv/digis": 7}}},
            {"backend": {"capacity_usage": {"rows/tracker": 9}}}]
    (run / "batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    bench = tmp_path / "bench" / "pass_01"
    bench.mkdir(parents=True)
    (bench / "pass_metrics.json").write_text(json.dumps({"capacity_usage_max": {"response/crv/digis": 11}}))
    usage, batches = dcap.demands([run, tmp_path / "bench"])
    assert usage == {"rows/tracker": 9, "response/crv/digis": 11} and batches == 3
    with pytest.raises(ValueError, match="no capacity usage"):
        dcap.demands([tmp_path])
    out = tmp_path / "caps.json"
    (tmp_path / "template.json").write_text(json.dumps(TEMPLATE))
    dcap.main([str(run), "--template", str(tmp_path / "template.json"), "--out", str(out)])
    assert json.loads(out.read_text())["rows"]["tracker"] == 12
    with pytest.raises(FileExistsError):
        dcap.main([str(run), "--template", str(tmp_path / "template.json"), "--out", str(out)])


def test_overflow_detail_names_demand_and_capacity():
    usage = {"response/tracker/straws": 1024, "response/crv/photons": 150, "rows/tracker": 120, "max_donors": 9}
    assert _overflow_detail("response/tracker/straws", usage, TEMPLATE) == \
        "response/tracker/straws (demand >= 1024, capacity 1024)"
    assert _overflow_detail("response/crv/crv_photons", usage, TEMPLATE) == \
        "response/crv/crv_photons (demand >= 150, capacity 100)"
    assert _overflow_detail("row_capacity", usage, TEMPLATE) == "row_capacity (tracker 120/100)"
    assert _overflow_detail("donor_capacity", usage, TEMPLATE) == "donor_capacity (demand 9, capacity 8)"
    assert _overflow_detail("response/tracker/timing", usage, TEMPLATE) == "response/tracker/timing"
