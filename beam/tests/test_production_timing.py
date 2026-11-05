"""Synthetic production log parsing, including filtered-out events and duplicate reports."""
import pytest

from beamfastsim import production_timing as T


def _log(tmp_path, name, g4_events=100, suffix=""):
    p = tmp_path / name
    p.write_text(f"PrimaryPath:g4run:Mu2eG4 0.01 0.02 0.1 0.019 0.003 {g4_events}\n"
                 "TrigReport Events total = 100 passed = 40 failed = 60\n"
                 "TimeReport CPU = 3.0 Real = 4.0\n" + suffix)
    return p


def test_module_and_whole_job_denominators_are_separate(tmp_path):
    a = _log(tmp_path, "a.log")
    b = _log(tmp_path, "b.log", g4_events=50)
    r = T.summarize([a, b])
    assert r["generated_events"] == 200
    assert r["g4_ms_per_generated"] == pytest.approx(15)
    assert r["whole_art_wall_ms_per_generated"] == pytest.approx(40)
    assert r["stock_g4_gen_per_s"] == pytest.approx(1 / 0.015)
    assert r["stock_whole_art_gen_per_s"] == 25


def test_multiple_job_reports_fail_instead_of_double_counting(tmp_path):
    p = _log(tmp_path, "double.log", suffix="TimeReport CPU = 3.0 Real = 4.0\n")
    with pytest.raises(ValueError, match="exactly one"):
        T.read(p)
