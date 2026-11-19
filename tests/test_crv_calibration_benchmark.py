"""The synthetic EX pilot uses the unmodified fastsim scorer."""

import pytest

from benchmarks.fastsim_score import score


@pytest.mark.parametrize("phase", ["dev", "final"])
def test_ex_pilot_recovery_and_scorer(tmp_path, phase):
    # Also run this test in an umbrella checkout without installed detector packages.
    pytest.importorskip("crvfastsim")
    from benchmarks.fastsim_score import crv_calibration as card

    results = card.run(tmp_path, phase)
    censored = results["censored"]
    for baseline in (results["nominal"], results["positive_only"]):
        for name in ("nll_per_crossing", "detection_brier", "delta_attenuation_per_mm_abs_error",
                     "delta_slowness_ns_per_mm_abs_error"):
            assert censored[name] < baseline[name]
    assert censored["delta_attenuation_per_mm_abs_error"] < 2e-5
    assert censored["delta_slowness_ns_per_mm_abs_error"] < 1e-4

    ref = tmp_path / "reference"
    for arm in results:
        report = score.score(tmp_path / "submissions" / arm, ref)
        assert report["n_arms_scored"] == 2 and report["throughput_geomean"] is None
        if arm == "censored":
            assert report["validity_score"] == 1.
        else:
            assert report["validity_score"] < 1.
