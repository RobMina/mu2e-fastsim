import pytest

from fastsim_tools import approximations_doc as ad
from fastsim_tools.tools import repo_root

TEXT = ad.read(repo_root())


def test_table_parses():
    rows = ad.parse_table(TEXT)
    assert rows
    assert {r["workflow"] for r in rows} >= {"W1", "W2-W4", "W5", "W6", "W7", "W8", "W9"}
    for r in rows:
        if r["switchable"]:
            assert r["levels"] and r["default"] == r["levels"][0]
            assert r["fast"] in r["levels"]
            assert r["fidelity"] is None or r["fidelity"] in r["levels"]
            assert r["stock_alternative"] in ("exact port", "stock art", "none")
        else:
            assert r["levels"] == [] and r["stock"]


def test_bare_pipe_inside_a_cell():
    rows = {r["name"]: r for r in ad.parse_table(TEXT)}
    assert "(|p|, tan dip, r_max)" in rows["tracker_rows"]["description"]


def test_against_the_declarations():
    """The parsed table carries the packages' declarations (where the packages import here)."""
    approximations = pytest.importorskip("mu2efastsim.approximations")
    workflows = []
    for package in ("beamfastsim", "crvfastsim", "crvfastsim.design", "calofastsim", "mixing", "trigger"):
        try:
            workflows += list(approximations.declared(package).values())
        except ImportError:   # calofastsim needs its numerical stack
            pytest.skip(f"{package} not importable here")
    rows = ad.parse_table(TEXT)
    by_key = {(r["workflow"], r["name"]): r for r in rows}
    n = 0
    for w in workflows:
        for a in w.approximations:
            r = by_key[(w.name, a.name)]
            assert r["switchable"] and r["default"] == a.default and set(r["levels"]) == set(a.levels)
            assert r["fast"] == (a.fast or a.default) and r["fidelity"] == a.fidelity
            n += 1
        for name in w.fixed:
            assert not by_key[(w.name, name)]["switchable"]
            n += 1
    assert n == len(rows)


def test_describe_one_workflow_with_preset():
    d = ad.describe(TEXT, "W1", "fast")
    w1 = d["workflows"]["W1"]
    assert {a["name"] for a in w1["switchable"]} == {"ts_transport", "ts_vacuum_step"}
    assert w1["selected"] == {"ts_transport": "condensed", "ts_vacuum_step": "long"}
    assert any("stock_ds_entrance" in n for n in w1["notes"])
    assert set(d["presets"]) == {"default", "fast", "fidelity"}


def test_fidelity_without_a_path_stays_default():
    d = ad.describe(TEXT, "W2-W4", "fidelity")
    sel = d["workflows"]["W2-W4"]["selected"]
    rows = {a["name"]: a for a in d["workflows"]["W2-W4"]["switchable"]}
    for name, level in sel.items():
        assert level == (rows[name]["fidelity"] or rows[name]["default"])


def test_notes_shared_label():
    notes = ad.parse_notes(TEXT)
    assert ad.notes_for(notes, "W8")   # "W6, W8" bullet
    assert ad.notes_for(notes, "W6")


def test_errors():
    with pytest.raises(ValueError, match="no approximations"):
        ad.describe(TEXT, "W99")
    with pytest.raises(ValueError, match="preset"):
        ad.describe(TEXT, "W1", "fastest")
    with pytest.raises(ValueError, match="markers"):
        ad.parse_table("no table here")
