"""The approximation scheme (mu2efastsim.approximations) and its use by the shared timer (mu2efastsim.bench)."""
import argparse
import json
import sys
import textwrap

import pytest

from mu2efastsim import approximations as A
from mu2efastsim import bench


def _workflow():
    return A.Workflow("WX", "test workflow", (
        A.Approximation("model", "a model", levels=("fast", "port"), default="fast", sets={"port": {"exact": True}},
                        fidelity="port", alternative=A.PORT, stock="the port", switch="--exact"),
        A.Approximation("extra", "an extra", levels=("on", "off"), default="on", sets={"off": {"extra": None}},
                        cli={"off": {"extra": "none"}}, fast="off", alternative=A.NONE, stock="none in code"),
    ), fixed={"no_switch": "stock does it"}, aliases=("WY",))


def test_default_levels_set_nothing():
    w = _workflow()
    for preset in (None, "default"):
        sel = w.select() if preset is None else w.select(preset)
        assert sel.arguments() == {} and sel.arguments("cli") == {}
        assert sel.apply_kwargs({"a": 1}) == {"a": 1}
        assert sel.non_default() == []


def test_presets_pick_fast_and_fidelity_levels():
    w = _workflow()
    assert w.select("fast").levels == {"model": "fast", "extra": "off"}
    assert w.select("fast").arguments() == {"extra": None}
    assert w.select("fast").arguments("cli") == {"extra": "none"}
    assert w.select("fidelity").levels == {"model": "port", "extra": "on"}
    assert w.select("fidelity").arguments() == {"exact": True}


def test_overrides_after_preset_and_refusals():
    w = _workflow()
    assert w.select("fidelity", {"model": "fast"}).arguments() == {}
    with pytest.raises(ValueError, match="no level 'off'"):
        w.select(overrides={"model": "off"})
    with pytest.raises(ValueError, match="no switch yet"):
        w.select(overrides={"no_switch": "off"})
    with pytest.raises(ValueError, match="unknown approximation"):
        w.select(overrides={"nope": "on"})
    with pytest.raises(ValueError, match="unknown preset"):
        w.select("fastest")


def test_yaml_booleans_map_to_on_off():
    assert _workflow().select(overrides={"extra": False}).levels["extra"] == "off"
    assert _workflow().select(overrides={"extra": True}).levels["extra"] == "on"


def test_declaration_checks():
    with pytest.raises(ValueError, match="default"):
        A.Approximation("x", "", levels=("a",), default="b")
    with pytest.raises(ValueError, match="sets nothing"):
        A.Approximation("x", "", levels=("a", "b"), default="a")
    with pytest.raises(ValueError, match="must set nothing"):
        A.Approximation("x", "", levels=("a", "b"), default="a", sets={"a": {"k": 1}, "b": {"k": 2}})
    with pytest.raises(ValueError, match="duplicate"):
        a = A.Approximation("x", "", levels=("a",), default="a")
        A.Workflow("W", "", (a, a))


def test_apply_kwargs_refuses_conflicts_and_unreachable_arguments():
    sel = _workflow().select("fidelity")
    assert sel.apply_kwargs({"exact": True}) == {"exact": True}
    with pytest.raises(ValueError, match="given"):
        sel.apply_kwargs({"exact": False})
    with pytest.raises(ValueError, match="takes no 'exact'"):
        sel.apply_kwargs({}, accepted={"seed"})


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--exact", action="store_true")
    p.add_argument("--extra", default="default")
    return A.add_arguments(p)


def test_command_line_options_reach_the_namespace():
    p = _parser()
    args = p.parse_args(["--approx-preset", "fast", "--approx", "model=port"])
    sel = A.from_args(args, _workflow())
    sel.apply_namespace(args, p)
    assert args.exact is True and args.extra == "none"
    assert A.requested(args) and not A.requested(p.parse_args([]))


def test_explicit_option_against_the_level_is_refused():
    p = _parser()
    args = p.parse_args(["--extra", "mine", "--approx", "extra=off"])
    with pytest.raises(ValueError, match="given"):
        A.from_args(args, _workflow()).apply_namespace(args, p)


def test_config_section_json_file_and_workflow_keys(tmp_path):
    w = _workflow()
    f = tmp_path / "approx.json"
    f.write_text(json.dumps({"approximations": {"WY": {"preset": "fast", "model": "port"}, "W5": {"x": "y"}}}))
    args = _parser().parse_args(["--approximations", str(f)])
    sel = A.from_args(args, w)
    assert sel.preset == "fast" and sel.levels == {"model": "port", "extra": "off"}
    args = _parser().parse_args(["--approximations", '{"preset": "fidelity"}', "--approx-preset", "default"])
    assert A.from_args(args, w).preset == "default"
    with pytest.raises(ValueError, match="keyed by workflow"):
        A.read_section({"W5": {"x": "y"}}, "WX")


def test_yaml_config(tmp_path):
    pytest.importorskip("yaml")
    f = tmp_path / "approx.yaml"
    f.write_text("approximations:\n  preset: fidelity\n  extra: off\n")
    assert A.read_section(str(f), "WX") == ("fidelity", {"extra": False})
    assert _workflow().select(*A.read_section(str(f), "WX")).levels == {"model": "port", "extra": "off"}


def test_record_effective_levels_and_table():
    sel = _workflow().select()
    sel.set_effective("extra", "off", "the other options turn it off")
    r = sel.record()
    assert r["levels"] == {"model": "fast", "extra": "off"} and r["selected"]["extra"] == "on"
    assert r["non_default"] == ["extra"] and r["fixed"] == ["no_switch"] and r["no_fidelity_path"] == ["extra"]
    assert r["no_stock_path"] == ["extra"]
    json.dumps(r)
    table = A.table_markdown([_workflow()])
    assert "`model`" in table and "`no_switch` (no switch)" in table and table.count("\n") == 5


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    pkg = tmp_path / "approxfake"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "flow.py").write_text(textwrap.dedent("""
        CALLS = []
        def run(n, exact=False):
            CALLS.append(exact)
            return n
        def run_any(n, **kw):
            CALLS.append(kw.get("exact", False))
            return n
        def run_fixed(n):
            return n
    """))
    (pkg / "approximations.py").write_text(textwrap.dedent("""
        from mu2efastsim.approximations import PORT, Approximation, Workflow
        WORKFLOW = Workflow("WX", "test", (Approximation("model", "m", levels=("fast", "port"), default="fast",
                                                         sets={"port": {"exact": True}}, fidelity="port",
                                                         alternative=PORT),))
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield sys.modules
    for name in [m for m in sys.modules if m.startswith("approxfake")]:
        del sys.modules[name]


def test_bench_records_the_default_levels_and_passes_nothing(fake_package, tmp_path):
    out = tmp_path / "r.json"
    assert bench.main(["approxfake.flow:run", "--events", "3", "--passes", "2", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["kwargs"] == {} and r["approximations"]["levels"] == {"model": "fast"}
    assert fake_package["approxfake.flow"].CALLS == [False, False]


def test_bench_level_reaches_the_entry_point(fake_package, tmp_path):
    out = tmp_path / "r.json"
    assert bench.main(["approxfake.flow:run", "--events", "3", "--passes", "2", "--approx-preset", "fidelity",
                       "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["kwargs"] == {"exact": True} and r["approximations"]["non_default"] == ["model"]
    assert fake_package["approxfake.flow"].CALLS == [True, True]
    assert bench.main(["approxfake.flow:run_any", "--events", "3", "--passes", "2", "--approx", "model=port"]) == 0
    assert fake_package["approxfake.flow"].CALLS[-2:] == [True, True]


def test_bench_refuses_a_level_the_entry_point_cannot_take(fake_package, capsys):
    assert bench.main(["approxfake.flow:run_fixed", "--events", "3", "--passes", "2",
                       "--approx", "model=port"]) == 2
    assert "takes no 'exact'" in capsys.readouterr().err


def test_bench_without_declarations_records_none(tmp_path):
    out = tmp_path / "r.json"
    assert bench.main(["math:fabs", "--events", "1", "--passes", "2", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["approximations"] is None


def test_load_reports_missing_and_ambiguous_declarations():
    assert A.load("json", required=False) is None
    with pytest.raises(ValueError, match="declares no approximations"):
        A.load("json")
    assert A.declared("mu2efastsim") == {}
