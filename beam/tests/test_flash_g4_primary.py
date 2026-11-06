"""Synthetic full-key continuation ancestry and recorded-process diagnostics."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim.flash_g4_photons import extract
from beamfastsim.flash_g4_primary import GEOMETRY_PROCESSES, main, select
from test_flash_g4_photons import table


@pytest.fixture
def fixture():
    pid, key = np.uint64(2**63 + 17), np.uint64(2**63 + 101)
    a = table("particles", 8, event=np.ones(8, np.uint64), product=np.full(8, pid),
              key=key + np.arange(8, dtype=np.uint64), pdg=[11, 11, 22, 11, 22, 22, 22, 22],
              stage=[1, 4, 4, 4, 4, 4, 4, 4], creation=[17, 56, 16, 17, 16, 2, 16, 16],
              start_volume=np.full(8, 9), energy=np.full(8, 5.), x=np.full(8, 2.),
              parent_state=[0, 1, 1, 1, 1, 1, 1, 1], parent_product=np.full(8, pid),
              parent_key=key + np.array([0, 0, 1, 1, 3, 1, 1, 1], np.uint64))
    s = table("steps", 5, event=np.ones(5, np.uint64), product=np.full(5, pid + np.uint64(1)),
              index=np.array([50, 40, 30, 20, 10], np.uint64), particle_state=np.ones(5, int),
              particle_product=np.full(5, pid), particle_key=key + np.array([2, 4, 5, 6, 7], np.uint64),
              x=np.full(5, 2.), post_x=np.full(5, 2.), post_z=[2., 2., 2., .5, 1.],
              pz=np.full(5, 5.), post_pz=np.full(5, 5.), end_process=[49, 49, 49, 40, 12])
    p = table("products", 10, event=np.r_[np.repeat([1, 2, 3], 3), 0],
              product=np.r_[np.tile(pid + np.arange(3, dtype=np.uint64), 3), pid + np.uint64(3)],
              scope=[0]*9 + [1], kind=[1, 3, 3]*3 + [5], size=[8, 5, 0, 0, 0, 0, 0, 0, 0, 5],
              label=["g4run", "g4run", "BeamFilter"]*3 + ["g4run"],
              instance=["", "photonSteps", "Beam"]*3 + [""])
    t = dict(particles=a, steps=s, products=p,
             events=table("events", 3, event=[1, 2, 3], ordinal=[0, 1, 2]),
             volumes=table("volumes", 1, product=pid + np.uint64(3), stage=4, index=9, name="TargetFoil_00"))
    g = SimpleNamespace(beam=SimpleNamespace(axis_x=0., foil_z=np.array([0.]),
                        foil_ht=np.array([1.]), foil_rin=np.array([1.]), foil_rout=np.array([3.])))
    return t, g


def run(fixture, **kwargs):
    tables, geom = fixture
    return select(tables, extract(tables, geom, **kwargs))


def add_prefix(tables, process=12, index=49):
    s = tables["steps"]
    row = {k: v[:1].copy() for k, v in s.items()}
    row["index"][:] = index
    row["post_z"][:] = .5
    row["end_process"][:] = process
    s["z"][0] = .5
    tables["steps"] = {k: np.concatenate((v, row[k])) for k, v in s.items()}
    tables["products"]["size"][1] += 1


def test_primary_secondary_annihilation_exact_uint64_and_zeros(fixture):
    out = run(fixture)
    np.testing.assert_array_equal(out["primary_brems"], [True, False, False, True, True])
    np.testing.assert_array_equal(out["uncollided_exit"], [True, False, False, False, False])
    np.testing.assert_array_equal(out["evt_primary_births"], [3, 0, 0])
    np.testing.assert_array_equal(out["evt_primary_exits"], [1, 0, 0])
    np.testing.assert_array_equal(out["evt_primary_energy"], [5., 0., 0.])
    assert out["product"].dtype == np.uint64 and out["key"][0] == np.uint64(2**63 + 103)
    assert out["no_recorded_exit"][-2:].all() and out["unresolved"][-1]
    report = json.loads(str(out["report_json"]))
    assert report["primary_birth_count"] == 3 and sum(report["primary_exit_bins"]) == 1
    assert GEOMETRY_PROCESSES == (48, 49, 182)


@pytest.mark.parametrize("change", ["parent_product", "missing_previous", "bad_state", "same_stage", "wrong_species"])
def test_unknown_ancestry_never_guessed(fixture, change):
    a = fixture[0]["particles"]
    if change == "parent_product":
        a["parent_product"][2] += np.uint64(1)
    elif change == "missing_previous":
        a["parent_key"][1] += np.uint64(999)
    elif change == "bad_state":
        a["parent_state"][2] = 2
    elif change == "same_stage":
        a["stage"][0] = 4
    else:
        a["pdg"][0] = -11
    out = run(fixture)
    assert out["unknown_ancestry"][0] and not out["primary_brems"][0]


def test_repeated_stages_and_positrons(fixture):
    a = fixture[0]["particles"]
    a["creation"][0] = 56  # the earlier stage was itself a continuation; no root walk needed
    a["pdg"][:2] = -11
    assert run(fixture)["primary_brems"][0]
    a["stage"][1] = 3  # copied previous-stage source cannot be this new-stage parent
    assert run(fixture)["unknown_ancestry"][0]


@pytest.mark.parametrize("process,accepted", [(48, True), (49, True), (182, True),
                                              (12, False), (65, False), (0, False), (186, False)])
def test_completed_prefix_process_not_float32_energy(fixture, process, accepted):
    add_prefix(fixture[0], process)
    # Identical momenta/energies deliberately do not rescue a scattered photon.
    out = run(fixture)
    assert out["uncollided_exit"][0] == accepted
    assert out["prior_non_geometry_steps"][0] == int(not accepted)


def test_crossing_endpoint_interaction_strictly_outside_allowed(fixture):
    fixture[0]["steps"]["end_process"][0] = 12
    fixture[0]["steps"]["post_pz"][0] = -1.
    assert run(fixture)["uncollided_exit"][0]
    fixture[0]["steps"]["post_z"][0] = 1.
    out = run(fixture)
    assert out["ambiguous"][0] and not out["uncollided_exit"][0]


@pytest.mark.parametrize("failure", [None, "scatter", "gap", "intervening"])
def test_two_segment_prefix_includes_transport_anchor(fixture, failure):
    t = fixture[0]
    add_prefix(t, 49)
    s = t["steps"]
    s["index"][-1] = 41
    s["post_z"][-1] = s["z"][0] = 1 - 1e-8
    s["end_process"][0] = 12  # strictly outside endpoint is not pre-crossing scattering
    if failure == "scatter":
        s["end_process"][-1] = 12
    elif failure == "gap":
        s["z"][0] += 1e-3
    elif failure == "intervening":
        add_prefix(t, 12, 45)
        t["steps"]["z"][0] = 1 - 1e-8
    out = run(fixture, boundary_tolerance_mm=1e-7)
    assert out["uncollided_exit"][0] == (failure is None)
    if failure is None:
        assert out["two_segment_reconstruction"][0] and out["first_stepindex"][0] == 41


def test_unresolved_record_pointer_and_empty(fixture):
    t, g = fixture
    t["steps"]["particle_key"][1] += np.uint64(1000)
    out = run(fixture)
    assert out["unknown_history"][0] and not out["uncollided_exit"].any()
    for name in ("particles", "steps"):
        t[name] = {k: v[:0] for k, v in t[name].items()}
    out = run(fixture)
    assert len(out["primary_brems"]) == 0 and out["evt_primary_births"].tolist() == [0, 0, 0]
    empty = {name: {k: v[:0] for k, v in table.items()} for name, table in t.items()}
    assert select(empty, extract(empty, g))["evt_primary_energy"].size == 0


def test_cli_root_npz_roundtrip(fixture, tmp_path, monkeypatch):
    import uproot
    import beamfastsim.flash_g4_primary as module
    t, g = fixture
    path, output = tmp_path / "input.root", tmp_path / "out.npz"
    with uproot.recreate(path) as root:
        for name, arrays in t.items():
            root.mktree("export/" + name, arrays)
    monkeypatch.setattr(module, "from_offline", lambda *args: g)
    args = ["--input", str(path), "--out", str(output), "--geom", "frozen.txt",
            "--offline-root", "/local", "--events", "3"]
    assert main(args) == 0
    with np.load(output, allow_pickle=False) as out:
        assert len(out["key"]) == 5 and out["evt_primary_births"].tolist() == [3, 0, 0]
        assert "spatial_report_json" in out.files
    assert json.loads(output.with_suffix(".json").read_text())["primary_uncollided_exit_count"] == 1
    assert main(args[:-1] + ["4"]) == 1


def test_full_key_collisions_and_uint64_step_sentinel(fixture):
    t, _ = fixture
    a = t["particles"]
    extra = {k: np.repeat(v[1:2], 2) for k, v in a.items()}
    extra["product"][0] += np.uint64(1)
    extra["event"][1] = 2
    extra["creation"][:] = 17  # same key, wrong product/event must never win
    t["particles"] = {k: np.concatenate((v, extra[k])) for k, v in a.items()}
    top = np.iinfo(np.uint64).max
    t["steps"]["index"][0] = top
    add_prefix(t, 48, top - 1)
    out = run(fixture)
    assert out["primary_brems"][0] and out["uncollided_exit"][0]
    assert out["first_stepindex"][0] == top and out["exit_candidate"][0]
    # Exact duplicate identity is ambiguous, never resolved by input row order.
    t["particles"]["product"][-2] = a["product"][1]
    with pytest.raises(ValueError, match="ambiguous"):
        run(fixture)


def test_later_scattering_does_not_veto_first_exit(fixture):
    add_prefix(fixture[0], 12, 51)
    out = run(fixture)
    assert out["first_stepindex"][0] == 50
    assert out["uncollided_exit"][0] and out["prior_non_geometry_steps"][0] == 0


def test_invalid_record_before_crossing_is_unresolved(fixture):
    add_prefix(fixture[0], 49)
    fixture[0]["steps"]["post_z"][-1] = np.nan
    out = run(fixture)
    assert out["unknown_history"][0] and not out["uncollided_exit"][0]
