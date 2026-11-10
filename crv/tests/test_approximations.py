"""W5 approximation levels reach runchain (crvfastsim.approximations) and are recorded."""
import json

import numpy as np
import pytest

from mu2efastsim import approximations as A
from crvfastsim import approximations as CA
from crvfastsim import cli
from crvfastsim.analyses import runchain


def _runchain_kw(monkeypatch, extra):
    seen = {}
    monkeypatch.setitem(cli.ANALYSES, "runchain", (lambda inputs, outdir, **kw: seen.update(kw), "stub"))
    cli.main(["runchain", "gen.root", "--outdir", "o", *extra])
    return seen


def test_declaration_loads():
    assert A.load("crvfastsim") is CA.WORKFLOW
    assert A.load("crvfastsim", "W5") is CA.WORKFLOW


def test_levels_are_runchain_options():
    dests = {a.dest for a in cli.build_parser().subparser["runchain"]._actions}
    for a in CA.WORKFLOW.approximations:
        for level in a.levels:
            for key in a.sets.get(level, {}):
                assert key in dests, (a.name, key)


def test_default_passes_todays_options(monkeypatch):
    kw = _runchain_kw(monkeypatch, ["--jax"])
    assert kw["legacy_transport"] is False and kw["below_msc"] is False and kw["jax"] is True
    assert kw["approximations"].non_default() == []


def test_each_level_reaches_its_runchain_option(monkeypatch):
    kw = _runchain_kw(monkeypatch, ["--approx", "muon_eloss=entry_momentum"])
    assert kw["legacy_transport"] is True
    kw = _runchain_kw(monkeypatch, ["--approx", "below_msc=scattered"])
    assert kw["below_msc"] is True
    kw = _runchain_kw(monkeypatch, ["--approx-preset", "fidelity"])
    assert kw["below_msc"] is True and kw["jax"] is False and kw["legacy_transport"] is False
    assert kw["approximations"].non_default() == ["below_msc", "pulse_fit"]


def test_option_against_the_level_is_refused(monkeypatch):
    with pytest.raises(SystemExit):
        _runchain_kw(monkeypatch, ["--jax", "--approx", "pulse_fit=full_length"])
    with pytest.raises(SystemExit):
        _runchain_kw(monkeypatch, ["--approx", "secondaries_library=off"])


def test_record_of_runchain_marks_the_numpy_full_length_fit():
    r = runchain.approximations_record(None, jax=False)
    assert r["levels"]["pulse_fit"] == "full_length" and r["selected"]["pulse_fit"] == "short_window"
    assert "pulse_fit" in r["effective"]
    r = runchain.approximations_record(None, jax=True)
    assert r["non_default"] == [] and r["effective"] == {}
    r = runchain.approximations_record({"preset": "fidelity"}, jax=False)
    assert r["preset"] == "fidelity" and r["levels"]["below_msc"] == "scattered"
    json.dumps(r)


def test_save_reco_carries_the_record(tmp_path):
    path = tmp_path / "r.npz"
    rec = runchain.approximations_record(None, jax=True)
    runchain._save_reco(path, [], 3, "run1a_v01", rec)
    with np.load(path) as z:
        assert json.loads(str(z["approximations"])) == json.loads(json.dumps(rec, sort_keys=True))


def test_merge_refuses_shards_with_different_levels(tmp_path):
    base = {"generated_events": 1, "timers": {}, "wall_s": 1.0, "cl_type_counts": {}}
    a = dict(base, approximations=runchain.approximations_record(None, True))
    b = dict(base, approximations=runchain.approximations_record({"below_msc": "scattered"}, True))
    for name, j in (("a.json", a), ("b.json", b)):
        (tmp_path / name).write_text(json.dumps(j))
    with pytest.raises(ValueError, match="approximation levels"):
        runchain.merge([str(tmp_path / "a.json"), str(tmp_path / "b.json")], str(tmp_path))
    (tmp_path / "c.json").write_text(json.dumps(a))
    res = runchain.merge([str(tmp_path / "a.json"), str(tmp_path / "c.json")], str(tmp_path))
    assert res["approximations"] == a["approximations"]
