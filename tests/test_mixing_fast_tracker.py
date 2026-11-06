"""Detector-only bridge retains empty events and exact development baseline fields."""
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from mixing.fast_tracker import compare_reference, comparison, digi_columns, field_at_origin


def test_bridge_and_exact_reference(tmp_path):
    keys = np.array([[1, 2, 3], [1, 2, 4]])
    digi = {"ev": np.array([0]), "sid": np.array([1024]), "tdc": np.array([[30000, 30001]]),
            "tot": np.array([[2, 3]]), "pmp": np.array([24]), "adc": np.array([[10, 20, 30]])}
    cols = digi_columns(keys, digi, 3)
    assert ak.num(cols["d_sid"]).tolist() == [1, 0]
    assert ak.sum(cols["d_nadc"]) == 3
    path = tmp_path / "reference.root"
    with uproot.recreate(path) as root:
        root.mktree("trkdigi/events", cols)
    rows = ak.Array(cols)
    assert len(compare_reference(keys, rows, path)) == 2
    report = comparison(rows, cols)
    np.testing.assert_array_equal(report["planes"]["difference"], np.zeros(36))
    cols["d_pmp"] = ak.Array([[25], []])
    with pytest.raises(ValueError, match="changed saved tracker output"):
        compare_reference(keys, ak.Array(cols), path)


def test_profile_passes_keep_two_clean_warm_repeats():
    from mixing.fast_tracker import profile_kind
    assert [profile_kind(i, True) for i in range(5)] == [
        "uninstrumented", "host_profile", "synchronized_stages", "uninstrumented", "uninstrumented"]
    assert all(profile_kind(i, False) == "uninstrumented" for i in range(5))


def test_host_profile_writes_receipts_even_after_error(tmp_path):
    import pstats
    from mixing.fast_tracker import diagnostic_profile
    with pytest.raises(RuntimeError, match="diagnostic"):
        with diagnostic_profile(tmp_path, host=True):
            raise RuntimeError("diagnostic")
    assert pstats.Stats(str(tmp_path / "host_warm.pstats")).total_calls > 0
    assert (tmp_path / "host_warm_cumulative.txt").exists()
    assert (tmp_path / "host_warm_tottime.txt").exists()
    assert not (tmp_path / "stages_warm.json").exists()


def test_disabled_profile_writes_nothing(tmp_path):
    from mixing.fast_tracker import diagnostic_profile
    with diagnostic_profile(tmp_path) as receipt:
        assert receipt == {}
    assert not tuple(tmp_path.iterdir())


def test_profile_requires_separate_warm_repeats():
    from mixing.fast_tracker import main
    with pytest.raises(SystemExit, match="2"):
        main(["input.root", "--gdml", "geometry.gdml", "--detector-origin", "0", "0", "0",
              "--out", "unused", "--stock", "stock.root", "--backend", "jax", "--profile", "--repeats", "3"])


def test_origin_field_is_not_a_nominal_guess(tmp_path):
    path = tmp_path / "field.csv"
    path.write_text("# x,y,z,bx,by,bz\n-3904,0,10171,0,0,1.0123\n-3903,0,10171,0,0,1.1\n")
    assert field_at_origin(path, [-3904, 0, 10171]) == 1.0123
    with pytest.raises(ValueError, match="exactly one"):
        field_at_origin(path, [-3904, 0, 24171])


def test_seed_replay_preserves_crv_except_exposed_truth_time(tmp_path):
    from mixing.fast_tracker import compare_crv
    original, candidate = tmp_path / "old.root", tmp_path / "new.root"
    def save(path, pbtime, pulses=2):
        with uproot.recreate(path) as root:
            root.mktree("crvdump/events", {"event": np.array([1]), "pbtime": np.array([pbtime])})
            for tree in ("digis", "pulses", "clusters"):
                root.mktree(f"crvdump/{tree}", {"event": np.array([1]), "count": np.array([pulses])})
    save(original, -212.5)
    save(candidate, -220.)
    assert compare_crv(candidate, original)
    save(candidate, -220., 3)
    with pytest.raises(ValueError, match="changed saved CRV"):
        compare_crv(candidate, original)


ROOT = Path(__file__).resolve().parents[1] / "mixing"


def test_fast_inputs_preserve_crv_seed_and_separate_truth_timing():
    text = (ROOT / "fcl/four_stream_fast_inputs.fcl").read_text()
    assert "services.SeedService.policy: preDefinedSeed" in text
    assert "services.SeedService.EWMProducer: 20261108" in text
    assert "services.SeedService.timingTruth: 20261108" in text
    assert "physics.producers.EWMProducer.RecoFromMCTruth: true" in text
    assert "physics.producers.EWMProducer.RecoFromMCTruthErr: 0.5" in text
    assert "physics.producers.timingTruth.RecoFromMCTruth: true" in text
    assert "physics.analyzers.crvdump.protonBunchTime: timingTruth" in text
    scheduled = text.split("physics.p1:", 1)[1]
    assert "makeSD" not in scheduled and "trkdigi" not in scheduled
    assert "CrvDAQPackage.CrvDAQSequence" in scheduled


def test_import_uses_existing_detector_bridge_only():
    text = (ROOT / "fcl/import_fast_tracker.fcl").read_text()
    assert 'fileName: "response.root"' in text
    assert 'tableFile: "fast.root"' in text
    assert "module_type: StrawDigisFromTable" in text
    assert "p1: [tableSD]" in text
    assert "MixBackgroundFrames" not in text
    assert "TrackerMC.DigiSim" not in text
