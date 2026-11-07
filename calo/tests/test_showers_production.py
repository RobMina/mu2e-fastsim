"""Synthetic production TTrees audit exact primary identities and saved links."""
import numpy as np
import pytest
import uproot

from calofastsim.showers.export import read
from test_showers_export import tables, write


EVENTS = [(1, 0, 1), (1, 1, 1), (2, 0, 1)]


def production_tables():
    data = tables()
    for table in data.values():
        if "event" in table:
            old = table["event"] - 1
            for j, name in enumerate(("run", "subrun", "event")):
                table[name] = np.asarray(EVENTS)[old, j].copy()
    for name in ("disks", "crystals", "virtualdetectors"):
        table = data[name]
        data[name] = {k: np.concatenate([v, np.full_like(v, 2) if k == "run" else v])
                      for k, v in table.items()}
    event = {k: data["events"][k].copy() for k in ("run", "subrun", "event")}
    data["primaries"] = dict(event, product_id=np.full(3, 200), index=np.zeros(3, int),
                             sim_product_id=np.full(3, 100), sim_key=np.full(3, 20), sim_valid=np.ones(3, bool))
    data["genparticles"] = dict(event, product_id=np.full(3, 300), index=np.zeros(3, int))
    particles = []
    for i, identity in enumerate(EVENTS):
        # Only self_parent edges extend the physical incident CE.
        specs = [(100, 1, 2212, 0, False), (100, 10, 13, 1, False),
                 (100, 20, 11, 10, False), (100, 99, 11, 20, False),
                 (999, 20, 11, 1, False)]
        if i == 0:
            specs.append((100, 21, 11, 20, True))
        for product, key, pdg, parent, continuation in specs:
            particles.append(dict(zip(("run", "subrun", "event"), identity),
                                  sim_product_id=product, sim_key=key, sim_valid=True, pdg=pdg,
                                  parent_valid=parent != 0, parent_product_id=100, parent_key=parent,
                                  generator_valid=key == 10, generator_product_id=300, generator_key=0,
                                  self_parent=continuation, is_ce_endpoint=product == 100 and key == 20))
    data["particles"] = {k: np.asarray([p[k] for p in particles]) for k in particles[0]}
    data["steps"]["sim_valid"] = np.ones(5, bool)
    data["steps"]["sim_key"][0] = 21
    append(data, "steps", 0)
    data["steps"]["sim_product_id"][-1] = 999
    data["steps"]["sim_key"][-1] = 20
    data["steps"]["time"][-1] = 1.
    data["showers"].update(sim_valid=np.ones(3, bool), sim_product_id=np.array([100, 999, 100]),
                            sim_key=np.array([99, 20, 99]))
    data["events"].update(input_mode=np.ones(3, int), transport_product_id=np.full(3, 100),
                           transport_particles_present=np.ones(3, bool), primary_present=np.ones(3, bool),
                           primary_product_id=np.full(3, 200), n_primaries=np.ones(3, int),
                           n_particles=np.array([6, 5, 5]), n_steps=np.array([4, 1, 1]),
                           showers_present=np.ones(3, bool), unresolved_links=np.zeros(3, int))
    del data["generated"]
    return data


def append(data, table, row):
    data[table] = {k: np.concatenate([v, v[row:row + 1]]) for k, v in data[table].items()}


def drop(data, table, row):
    data[table] = {k: np.delete(v, row) for k, v in data[table].items()}


def put(data, table, field, value, row=0):
    data[table][field][row] = value


@pytest.mark.parametrize("shuffle", [False, True])
def test_primary_whole_event_response(tmp_path, shuffle):
    data = production_tables()
    if shuffle:
        rng = np.random.default_rng(81)
        for name, table in data.items():
            order = rng.permutation(len(next(iter(table.values()))))
            data[name] = {k: v[order] for k, v in table.items()}
    path = tmp_path / "production.root"
    write(path, data)
    with uproot.open(path) as source:
        assert all(source["dump/" + name].classname == "TTree" for name in data)
    entries, css, geometry, census, total = read(path, input_mode="primary")
    identities = list(zip(*(data["events"][k] for k in ("run", "subrun", "event"))))
    selected = {identities[row]: (disk, time) for row, disk, time in
                zip(entries["event_row"], entries["disk"], entries["time"])}
    assert selected == {EVENTS[0]: (0, 10.), EVENTS[1]: (0, 20.)}
    np.testing.assert_allclose(entries["direction"], [[0, 0, 1], [0, 0, 1]])
    np.testing.assert_allclose(entries["energy"], np.sqrt(100. ** 2 + .51099895 ** 2))
    order = np.argsort(css["time"])
    np.testing.assert_array_equal(css["disk"][order], [0, 1])
    np.testing.assert_array_equal(css["energy_birks"][order], [39, 29])
    np.testing.assert_array_equal(css["energy_g4"][order], [40, 30])
    np.testing.assert_allclose(css["position"][order], [[5, 0, 110], [5, 0, 210]])
    assert all(identities[entries["event_row"][i]] == EVENTS[0] for i in css["entry"])
    assert dict(zip(identities, total)) == dict(zip(EVENTS, [68, 0, 19]))
    assert set(geometry["disk_run"]) == {1, 2}
    assert census == dict(events=3, front_entry_events=2, empty_css_events=1, non_front_entry_events=1,
                          non_front_entry_positive_events=1, css_rows=3, retained_css_rows=2,
                          direct_ce_particles=4, ce_descendant_css_rows=2, other_ancestry_css_rows=1)


def test_reverse_continuation_selects_other_disk(tmp_path):
    data = production_tables()
    data["steps"]["momentum_z"][0] = -100.
    path = tmp_path / "reverse.root"
    write(path, data)
    entries, css, _, _, _ = read(path, input_mode="primary")
    np.testing.assert_array_equal(entries["disk"], [1, 0])
    np.testing.assert_array_equal(entries["time"], [12, 20])
    np.testing.assert_array_equal(css["disk"], [0, 1])


BAD_CELLS = [
    ("header-mode", "events", "input_mode", 0, 0),
    ("header-transport", "events", "transport_product_id", 999, 0),
    ("header-primary", "events", "primary_product_id", 999, 0),
    ("missing-product", "events", "transport_particles_present", False, 0),
    ("missing-primary-product", "events", "primary_present", False, 0),
    ("missing-showers", "events", "showers_present", False, 0),
    ("unresolved-links", "events", "unresolved_links", 1, 0),
    ("primary-product", "primaries", "product_id", 999, 0),
    ("primary-index", "primaries", "index", 1, 0),
    ("primary-transport", "primaries", "sim_product_id", 999, 0),
    ("missing-primary-target", "primaries", "sim_key", 404, 0),
    ("null-primary", "primaries", "sim_valid", False, 0),
    ("null-particle", "particles", "sim_valid", False, 2),
    ("wrong-pdg", "particles", "pdg", -11, 2),
    ("wrong-process", "particles", "is_ce_endpoint", False, 2),
    ("missing-parent", "particles", "parent_key", 404, 2),
    ("wrong-parent-product", "particles", "parent_product_id", 999, 2),
    ("missing-generator", "particles", "generator_key", 404, 1),
    ("wrong-generator-product", "particles", "generator_product_id", 999, 1),
    ("cycle", "particles", "parent_key", 20, 1),
    ("self-cycle", "particles", "parent_key", 20, 2),
    ("continuation-species", "particles", "pdg", 22, 5),
    ("root-continuation", "particles", "self_parent", True, 0),
    ("primary-continuation", "particles", "self_parent", True, 2),
    ("null-step", "steps", "sim_valid", False, 0),
    ("null-shower", "showers", "sim_valid", False, 0),
    ("missing-step-target", "steps", "sim_key", 404, 0),
    ("missing-shower-target", "showers", "sim_key", 404, 0),
    ("particle-count", "events", "n_particles", 5, 0),
    ("step-count", "events", "n_steps", 3, 0),
    ("shower-count", "events", "n_showers", 1, 0),
    ("primary-count", "events", "n_primaries", 2, 0),
    ("failed-status", "events", "g4_status", 1, 1),
]


@pytest.mark.parametrize("case,table,field,value,row", BAD_CELLS, ids=[c[0] for c in BAD_CELLS])
def test_reject_malformed_cells(tmp_path, case, table, field, value, row):
    data = production_tables()
    put(data, table, field, value, row)
    path = tmp_path / (case + ".root")
    write(path, data)
    with pytest.raises(ValueError):
        read(path, input_mode="primary")


@pytest.mark.parametrize("case", ["duplicate-event", "duplicate-particle", "duplicate-generator",
                                  "duplicate-primary", "duplicate-pointer", "missing-primary", "multiple-primary"])
def test_reject_malformed_rows(tmp_path, case):
    data = production_tables()
    if case == "missing-primary":
        drop(data, "primaries", 0)
        put(data, "events", "n_primaries", 0)
    else:
        table = {"duplicate-event": "events", "duplicate-particle": "particles",
                 "duplicate-generator": "genparticles"}.get(case, "primaries")
        append(data, table, 0)
        if table == "primaries":
            put(data, "events", "n_primaries", 2)
            if case != "duplicate-primary":
                put(data, table, "index", 1, -1)
            if case == "multiple-primary":
                put(data, table, "sim_key", 21, -1)
    path = tmp_path / (case + ".root")
    write(path, data)
    with pytest.raises(ValueError):
        read(path, input_mode="primary")


@pytest.mark.parametrize("mode", [None, "stage", "unknown"])
def test_primary_requires_explicit_mode(tmp_path, mode):
    path = tmp_path / "explicit.root"
    write(path, production_tables())
    with pytest.raises(ValueError):
        read(path, **({} if mode is None else {"input_mode": mode}))


def test_legacy_stage_default_unchanged(tmp_path):
    path = tmp_path / "stage.root"
    write(path, tables())
    default = read(path)
    explicit = read(path, input_mode="stage")
    for left, right in zip(default[:4], explicit[:4]):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    np.testing.assert_array_equal(default[4], [68, 0, 19])
    np.testing.assert_array_equal(default[4], explicit[4])
    assert "direct_ce_particles" not in default[3]


@pytest.mark.parametrize("cycle", [False, True])
def test_long_continuation_chain(tmp_path, cycle):
    data = production_tables()
    particles = data["particles"]
    tail = {k: np.repeat(v[5:6], 65) for k, v in particles.items()}
    tail["sim_key"] = np.arange(1000, 1065)
    tail["parent_key"] = np.r_[21, np.arange(1000, 1064)]
    if cycle:
        tail["parent_key"][0] = 1064
    data["particles"] = {k: np.r_[v, tail[k]] for k, v in particles.items()}
    data["events"]["n_particles"][0] += 65
    data["steps"]["sim_key"][0] = 1064
    path = tmp_path / "long.root"
    write(path, data)
    if cycle:
        with pytest.raises(ValueError, match="cyclic"):
            read(path, input_mode="primary")
    else:
        entries, _, _, census, _ = read(path, input_mode="primary")
        np.testing.assert_array_equal(entries["time"], [10, 20])
        assert census["direct_ce_particles"] == 69


def test_present_empty_steps_and_showers_are_valid(tmp_path):
    data = production_tables()
    for name in ("steps", "showers"):
        data[name] = {k: v[:0] for k, v in data[name].items()}
        data["events"]["n_" + name][:] = 0
    path = tmp_path / "empty.root"
    write(path, data)
    entries, css, _, census, total = read(path, input_mode="primary")
    assert len(entries["event_row"]) == len(css["entry"]) == 0
    assert census["events"] == census["empty_css_events"] == 3
    np.testing.assert_array_equal(total, [0, 0, 0])


def test_saved_table_check_cli(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from calofastsim.showers.export import main

    path, out = tmp_path / "production.root", tmp_path / "check.json"
    write(path, production_tables())
    monkeypatch.setattr(sys, "argv", ["export", str(path), "--input-mode", "primary", "--out", str(out)])
    main()
    result = json.loads(out.read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result["census"]["front_entry_events"] == 2
    assert result["input_mode"] == "primary"
    assert "not unconditional" in result["denominator"]
    assert "separately" in result["geometry"]
