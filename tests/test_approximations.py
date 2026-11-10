"""Every workflow's approximation declarations: they load, the default levels set nothing, the documentation table
is the generated one, and the W6 and W8 commands record them."""
import json
import sys
from pathlib import Path

import pytest

from mu2efastsim import approximations as A

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("beamfastsim", "crvfastsim", "calofastsim", "mixing", "trigger")
DOC = ROOT / "docs" / "approximations.md"
BEGIN, END = "<!-- table: python3 -m mu2efastsim.approximations -->", "<!-- end table -->"


@pytest.fixture(autouse=True)
def calo_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "calo"))


def _all():
    return {name: w for package in PACKAGES for name, w in A.declared(package).items()}


def test_every_workflow_is_declared():
    assert sorted(_all()) == ["W1", "W2-W4", "W5", "W6", "W7", "W8"]


@pytest.mark.parametrize("preset", A.PRESETS)
def test_presets_resolve_and_default_sets_nothing(preset):
    for w in _all().values():
        sel = w.select(preset)
        json.dumps(sel.record())
        if preset == "default":
            assert sel.arguments() == {} and sel.arguments("cli") == {}


def test_every_approximation_names_its_stock_alternative():
    for w in _all().values():
        for a in w.approximations:
            assert a.stock and a.switch and a.description, (w.name, a.name)
        for name, text in w.fixed.items():
            assert text, (w.name, name)


def test_documentation_table_is_generated(capsys):
    assert A.main(list(PACKAGES)) == 0
    table = capsys.readouterr().out
    text = DOC.read_text()
    assert BEGIN in text and END in text
    assert text.split(BEGIN)[1].split(END)[0].strip() == table.strip(), (
        "docs/approximations.md is out of date: regenerate the table with python3 -m mu2efastsim.approximations "
        + " ".join(PACKAGES))


def test_mixing_stream_benchmark_records_the_approximations(monkeypatch, tmp_path):
    from mixing import stream_benchmark as S
    seen = {}

    def run(args):
        seen["args"] = args
        return {"events": 0, "measured_s": 0, "events_per_s": 0}
    monkeypatch.setattr(S, "run", run)
    argv = [x for name in ("frames-manifest", "truth", "stock", "photon-csv", "expanded", "tracker-conditions", "out")
            for x in ("--" + name, str(tmp_path / name))]
    argv += ["--events", "1", "--batch-events", "1", "--max-selected-rows", "1", "--approx-preset", "fidelity"]
    S.main(argv)
    rec = seen["args"].approximations
    assert rec["workflow"] == "W6" and rec["preset"] == "fidelity" and rec["levels"] == {}
    assert rec["fixed"] == sorted(A.load("mixing").fixed)
    with pytest.raises(SystemExit):
        S.main(argv + ["--approx", "finite_donor_pools=off"])


def test_trigger_decide_records_the_approximations(monkeypatch):
    pytest.importorskip("jax")
    from trigger.device import run as R
    seen = {}
    monkeypatch.setattr(R, "cmd_decide", lambda args: seen.update(record=args.approximations_record))
    argv = ["decide", "--dump", "d.root", "--menu", "m", "--tc-mva", "t", "--output", "o.json"]
    R.main(argv)
    assert seen["record"]["workflow"] == "W8" and seen["record"]["preset"] == "default"
    with pytest.raises(SystemExit):
        R.main(argv + ["--approx", "compact_helix_fit=off"])


def test_bench_finds_each_workflow_package():
    for package in PACKAGES:
        assert A.declared(package), package
    assert A.load("beamfastsim", "W3").name == "W2-W4"
    assert A.load("trigger", "W8", required=False) is not None
    assert "beamfastsim" in sys.modules
