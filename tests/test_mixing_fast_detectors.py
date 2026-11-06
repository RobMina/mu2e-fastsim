"""Mixed-detector driver preserves complete IDs and distinguishes diagnostic output from art."""
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from mixing.crv_compare import histogram, rows_to_events, compare
from mixing.fast_detectors import write_output, flatten_output


def test_histograms_keep_empty_events_and_flows():
    np.testing.assert_array_equal(histogram([0, 2, 2], [-1, 1, 9], [0, 2], 4),
                                  [[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        histogram([0], [np.nan], [0, 2], 3)


def test_full_ids_do_not_pack_run_or_truncate_event():
    keys = np.array([[1, 2, 3], [2, 2, 3], [1, 2, 2**30]])
    rows = ak.Array(dict(zip(("run", "subrun", "event"), keys[[2, 0]].T)))
    np.testing.assert_array_equal(rows_to_events(rows, keys), [2, 0])
    with pytest.raises(ValueError):
        rows_to_events(ak.Array({"run": [9], "subrun": [2], "event": [3]}), keys)


def test_output_keeps_zero_event_census_and_timing(tmp_path):
    keys = np.array([[1, 2, 3], [2, 2, 3], [1, 2, 2**30]], np.uint64)
    timing = SimpleNamespace(mc_pbtime=np.array([-210., -215., -220.]),
                             reco_pbtime=np.array([-209.5, -215.1, -219.9]), event_length=np.array([1675., 1700., 1675.]))
    D = {"ev": np.array([1]), "bar": np.array([0]), "sipm": np.array([2]),
         "starttdc": np.array([30]), "adc": np.ones((1, 12), np.int16)}
    P = {"ev": np.array([1]), "idx": np.array([0]), "bar": np.array([0]), "sipm": np.array([2]),
         "pe": np.array([10.]), "time": np.array([600.]), "seq": np.array([0])}
    C = {"ev": np.array([], np.int64), "pe": np.array([]), "time": np.array([])}
    output = tmp_path / "fast.root"
    write_output(output, keys, timing, (D, P, C))
    with uproot.open(output) as root:
        events = root["crvfast/events"].arrays(library="np")
        np.testing.assert_array_equal(events["ndigis"], [0, 1, 0])
        np.testing.assert_array_equal(events["event"], keys[:, 2])
        np.testing.assert_array_equal(events["reco_pbtime"], timing.reco_pbtime)
        assert root["crvfast/clusters"].num_entries == 0
        assert "trkdigi/events" not in root
    from mixing.inspect import summarize_fast
    receipt = summarize_fast(output)
    assert receipt["events"] == 3 and receipt["crv"]["empty_pulses_events"] == 2
    stock = tmp_path / "stock.root"
    with uproot.recreate(stock) as root:
        root.mktree("crvdump/events", dict(zip(("run", "subrun", "event"), keys.T)))
        for name, data in (("digis", D), ("pulses", P), ("clusters", C)):
            ids = keys[data["ev"]]
            root.mktree("crvdump/" + name, {**dict(zip(("run", "subrun", "event"), ids.T)),
                         **{("avgtime" if name == "clusters" and k == "time" else k): v for k, v in data.items() if k != "ev"}})
    report = compare(stock, keys, D, P, C, SimpleNamespace(sector=np.array([0]), sector_names=["T1"]))
    np.testing.assert_array_equal(report["digis_total"]["difference"], [0])
    np.testing.assert_array_equal(report["pulses_total"]["difference"], [0])
    assert report["clusters_total"]["events"] == 3


def test_detector_step_fcl_contains_no_detector_response():
    root = Path(__file__).resolve().parents[1] / "mixing"
    text = (root / "fcl/four_stream_detector_steps.fcl").read_text()
    schedule = text.split("physics.p1:", 1)[1]
    assert "CrvDAQSequence" not in schedule and "CrvRecoSequence" not in schedule
    assert "EWMProducer" not in schedule and "crvdump" not in schedule
    assert "PBISim, MuStopPileupMixer, EleBeamFlashMixer, MuBeamFlashMixer, NeutralsFlashMixer" in schedule
    for prefix in ("primary", "stop", "ele", "mu", "neutral"):
        assert prefix + "Steps" in schedule and prefix + "Crv" in schedule
    assert "MIXING_EXPORT_CRV_CONDITIONS" in (root / "run_response.sh").read_text()


def test_sampling_receipt_saves_exact_donor_arrays_without_pickle(tmp_path):
    import json
    from mixing.fast_detectors import save_sampling
    keys = np.array([[2**45, 1, 2**60], [1, 1, 2]], np.uint64)
    provenance = {"seed": 123, "selections": {"MuStopPileup": {"source_event_ids": keys}},
                  "intensity": np.array([123, 456]), "stream_order": ("MuStopPileup",)}
    save_sampling(tmp_path, provenance)
    metadata = json.loads((tmp_path / "sampling.json").read_text())
    with np.load(tmp_path / "sampling.npz", allow_pickle=False) as saved:
        record = metadata["selections"]["MuStopPileup"]["source_event_ids"]
        np.testing.assert_array_equal(saved[record["npz_key"]], keys)
        assert record["dtype"] == "uint64"
        np.testing.assert_array_equal(saved[metadata["intensity"]["npz_key"]], [123, 456])
