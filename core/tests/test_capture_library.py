"""Captures to a component_response bank: synthetic records through the strict audit, save and load."""
import json

import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_library, component_response as cr
from mu2efastsim.library import store
from test_capture import BIG, tables  # noqa: F401 -- reuse the shared synthetic capture fixture
from test_capture_links import add_birth_exit, make_links

ENERGY = 10.


def bankable(tables):
    """The shared fixture with total energies above |p|, a box envelope inside a world, and an exit at birth."""
    for name, prefixes in capture.STATES.items():
        for prefix in prefixes:
            tables[name][prefix + "energy"][:] = ENERGY
    v = tables["volumes"]
    v["logical"][:] = ["box", "world", "box", "world"]
    v["motherLogical"][:] = ["world", "", "world", ""]
    v["material"][:] = ["G4_AIR", "G4_Galactic", "G4_AIR", "G4_Galactic"]
    links = make_links(tables)
    add_birth_exit(tables, links)
    return tables, links


def write(path, tables, links):
    with uproot.recreate(path) as root:
        for name, table in {**tables, **links}.items():
            root.mktree("dump/" + name, table)
    return path


def test_capture_converts_to_a_bank_and_loads_back(tmp_path, tables):
    tables, links = bankable(tables)
    path = write(tmp_path / "capture.root", tables, links)
    manifest = capture_library.convert([path], tmp_path / "bank.npz", 2, relative_to=tmp_path)
    assert manifest["kind"] == "replay" and manifest["name"] == "secondaries.component.synthetic-box"
    assert manifest["validation"]["bank_admissible"] is False
    assert {"generator", "seeds", "release", "split"} <= set(manifest["unrecorded"])
    sample, = manifest["source"]["samples"]
    assert sample["dataset"] == "capture.root"
    assert sample["events"] == 2 and sample["exits_at_birth"] == 1 and len(sample["sha256"]) == 64
    assert manifest["geometry"]["volumes"] == 1 and manifest["geometry"]["volume_names"] == ["box"]
    bank = cr.load_library(tmp_path / "bank.npz")
    assert bank.component == cr.ComponentType("synthetic-box", "synthetic-not-physics", (1., 1., 1.), 1)
    np.testing.assert_array_equal(bank.entrances["pdg"], [13, 13])
    np.testing.assert_allclose(bank.entrances["position"], [[0, 0, -1]] * 2)
    np.testing.assert_allclose(bank.entrances["momentum"], [[0, 0, 1]] * 2)
    np.testing.assert_array_equal(bank.particle_offsets, [0, 1, 4])
    np.testing.assert_array_equal(bank.particles["parent"], [-1, -1, 0, 0])
    np.testing.assert_array_equal(bank.particles["exited"], [True, False, False, True])
    np.testing.assert_array_equal(bank.step_offsets, [0, 1, 3])
    np.testing.assert_array_equal(bank.steps["particle"], [0, 0, 1])
    np.testing.assert_array_equal(bank.steps["volume"], [0, 0, 0])
    np.testing.assert_allclose(bank.steps["t0"], [0, 0, 1])
    np.testing.assert_allclose(bank.steps["t1"], [2, 1, 2])
    np.testing.assert_allclose(bank.steps["edep"], [0, 2, 1])
    np.testing.assert_array_equal(bank.exit_offsets, [0, 1, 2])
    np.testing.assert_array_equal(bank.exits["particle"], [0, 2])
    np.testing.assert_allclose(bank.exits["position"], [[0, 0, 1]] * 2)
    np.testing.assert_allclose(bank.exits["energy"], [ENERGY, 1.])
    np.testing.assert_allclose(bank.exits["time"], [2, 1])


def test_replay_in_the_capture_placement_reproduces_its_global_steps(tmp_path, tables):
    tables, links = bankable(tables)
    library, _ = capture_library.library_from_captures([write(tmp_path / "c.root", tables, links)], 2, relative_to=tmp_path)
    e, t, s = tables["events"], tables["tracks"], tables["steps"]
    primary = np.flatnonzero(t["parent"] == 0)
    for row in range(2):
        placement = cr.Placement(library.component, 0, e["rotation"][row].reshape(3, 3), e["translation"][row],
                                 np.array([BIG + 1], dtype=np.int64))
        p = primary[row]
        incoming = dict(event=np.array([[1, 0, row]]), track=np.array([1]), parent=np.array([-1]),
                        pdg=t["pdg"][[p]].astype(np.int64), position=capture.vectors(t, "entrance_", "xyz")[[p]],
                        momentum=capture.vectors(t, "entrance_p", "xyz")[[p]], energy=t["entrance_energy"][[p]],
                        time=t["entrance_time"][[p]])
        for seed in range(8):
            out = cr.replay(library, placement, incoming, np.array([0]), np.random.default_rng(seed), next_track=2)
            trial = int(out.trials[0])
            mine = (s["event"] == e["event"][trial]) & s["preInside"]
            # Steps of the chosen trial, placed where its own capture event sat, must agree when the placements match.
            if trial == row:
                np.testing.assert_allclose(out.steps["x0"], capture.vectors(s, "pre_", "xyz")[mine], atol=1e-12)
                np.testing.assert_allclose(out.steps["x1"], capture.vectors(s, "post_", "xyz")[mine], atol=1e-12)
                np.testing.assert_allclose(out.steps["p1"], capture.vectors(s, "post_p", "xyz")[mine], atol=1e-12)
                np.testing.assert_allclose(out.steps["t1"], s["post_time"][mine], atol=1e-12)
            np.testing.assert_allclose(out.steps["edep"], s["deposit"][mine])


def test_cli_records_given_provenance(tmp_path, tables, capsys):
    tables, links = bankable(tables)
    path = write(tmp_path / "capture.root", tables, links)
    provenance, info = tmp_path / "provenance.json", tmp_path / "info.json"
    provenance.write_text(json.dumps({"seeds": {"k001.baseSeed": 401000}, "source": {"campaign": "u6"}}))
    info.write_text(json.dumps([{"placement": "nominal"}]))
    capture_library.main([str(path), "--expected-events", "2", "--output", str(tmp_path / "bank.npz"),
                          "--relative-to", str(tmp_path), "--provenance", str(provenance),
                          "--sample-info", str(info)])
    printed = json.loads(capsys.readouterr().out)
    assert "seeds" not in printed["unrecorded"] and "generator" in printed["unrecorded"]
    manifest = store.load(tmp_path / "bank.npz").manifest
    assert manifest["seeds"] == {"k001.baseSeed": 401000} and manifest["source"]["campaign"] == "u6"
    assert manifest["source"]["samples"][0]["placement"] == "nominal"
    assert manifest["source"]["supplied"] == ["campaign"] and "samples" in manifest["source"]["recorded"]
    assert manifest["source"]["samples"][0]["supplied"] == ["placement"]
    assert "supplied" not in manifest["geometry"] and "hash" in manifest["geometry"]["recorded"]
    assert cr.load_library(tmp_path / "bank.npz").entrances["pdg"].tolist() == [13, 13]


def test_loaded_bank_replays_bit_for_bit(tmp_path, tables):
    tables, links = bankable(tables)
    path = write(tmp_path / "capture.root", tables, links)
    library, provenance = capture_library.library_from_captures([path], 2, relative_to=tmp_path)
    cr.save_library(tmp_path / "bank.npz", library, **provenance)
    loaded = cr.load_library(tmp_path / "bank.npz")
    e, t = tables["events"], tables["tracks"]
    p = np.flatnonzero(t["parent"] == 0)[1]
    placement = cr.Placement(library.component, 3, e["rotation"][1].reshape(3, 3), e["translation"][1],
                             np.array([7], dtype=np.int64))
    n = 16
    incoming = dict(event=np.column_stack([np.zeros(n, int), np.zeros(n, int), np.arange(n)]), track=np.ones(n, int),
                    parent=np.full(n, -1), pdg=np.full(n, 13, dtype=np.int64),
                    position=np.repeat(capture.vectors(t, "entrance_", "xyz")[[p]], n, 0),
                    momentum=np.repeat(capture.vectors(t, "entrance_p", "xyz")[[p]], n, 0),
                    energy=np.full(n, ENERGY), time=np.full(n, 5.))
    a, b = (cr.replay(lib, placement, incoming, np.zeros(n, int), np.random.default_rng(11), next_track=2)
            for lib in (library, loaded))
    assert np.array_equal(a.trials, b.trials) and a.next_track == b.next_track and len(set(a.trials)) == 2
    for name in ("particles", "steps", "exits"):
        x, y = getattr(a, name), getattr(b, name)
        assert set(x) == set(y) and all(np.array_equal(x[k], y[k]) and x[k].dtype == y[k].dtype for k in x)


def test_dataset_outside_the_named_tree_is_refused(tmp_path, tables):
    tables, links = bankable(tables)
    (tmp_path / "tree").mkdir()
    path = write(tmp_path / "capture.root", tables, links)
    with pytest.raises(ValueError, match="outside tree"):
        capture_library.library_from_captures([path], 2, relative_to=tmp_path / "tree")


def test_given_provenance_cannot_replace_recorded_keys(tmp_path, tables):
    tables, links = bankable(tables)
    path = write(tmp_path / "capture.root", tables, links)
    with pytest.raises(ValueError, match="sample_info cannot replace"):
        capture_library.library_from_captures([path], 2, relative_to=tmp_path, sample_info=[{"sha256": "0" * 64}])
    with pytest.raises(ValueError, match="geometry cannot replace"):
        capture_library.convert([path], tmp_path / "bank.npz", 2, relative_to=tmp_path, geometry={"hash": "x"})
    with pytest.raises(ValueError, match="one entry per capture"):
        capture_library.library_from_captures([path], 2, relative_to=tmp_path, sample_info=[{}, {}])


def test_capture_without_links_is_refused(tmp_path, tables):
    tables, _ = bankable(tables)
    path = tmp_path / "capture.root"
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree("dump/" + name, table)
    with pytest.raises(ValueError, match="links and manifest"):
        capture_library.library_from_captures([path], 2, relative_to=tmp_path)


def test_capture_failing_the_audit_is_refused(tmp_path, tables):
    tables, links = bankable(tables)
    tables["events"]["bankAdmissible"][0] = True
    with pytest.raises(ValueError, match="bankAdmissible"):
        capture_library.library_from_captures([write(tmp_path / "c.root", tables, links)], 2, relative_to=tmp_path)


@pytest.mark.parametrize("field, value, message", [("physicsTag", "other-physics", "different component type"),
                                                    ("material", "G4_WATER", "different volume inventory")])
def test_captures_of_different_components_are_refused(tmp_path, tables, field, value, message):
    tables, links = bankable(tables)
    first = write(tmp_path / "a.root", tables, links)
    if field == "material":
        tables["volumes"]["material"][tables["volumes"]["name"] == "box"] = value
    else:
        tables["events"][field][:] = value
    second = write(tmp_path / "b.root", tables, links)
    with pytest.raises(ValueError, match=message):
        capture_library.library_from_captures([first, second], 2, relative_to=tmp_path)


def test_two_captures_concatenate_in_file_then_event_order(tmp_path, tables):
    tables, links = bankable(tables)
    a, b = write(tmp_path / "a.root", tables, links), write(tmp_path / "b.root", tables, links)
    library, provenance = capture_library.library_from_captures([a, b], 2, relative_to=tmp_path)
    np.testing.assert_array_equal(library.particle_offsets, [0, 1, 4, 5, 8])
    np.testing.assert_array_equal(library.exits["particle"], [0, 2, 0, 2])
    assert [s["dataset"] for s in provenance["source"]["samples"]] == ["a.root", "b.root"]


def test_inventory_hash_ignores_the_envelope_placement(tables):
    tables, _ = bankable(tables)
    rows, envelope = capture_library._volumes(tables)
    inventory = capture_library._inventory(tables, rows, envelope)
    moved = {k: v.copy() for k, v in tables["volumes"].items()}
    moved["translation"] = moved["translation"] + 5.
    tables["volumes"] = moved
    assert capture_library.inventory_hash(capture_library._inventory(tables, rows, envelope)) == \
        capture_library.inventory_hash(inventory)
