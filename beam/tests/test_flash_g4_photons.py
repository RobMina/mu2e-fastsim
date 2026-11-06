"""Analytic annular segments and product-qualified synthetic exporter fixtures."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim.cechain import C_MM_NS
from beamfastsim.flash_exit import Z_PLUS, OUTER_RIM, INNER_HOLE
from beamfastsim.flash_g4_photons import FIELDS, extract, main, read_tables, segment_exit


def segments(pre, post, process=49, post_momentum=None, **kwargs):
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    momentum = post - pre
    if post_momentum is None:
        post_momentum = momentum
    return segment_exit(pre, post, momentum, post_momentum, process, 0., 1., 1., 3., 0., **kwargs)


def test_axial_rim_hole_and_never_exit():
    out = segments([[2, 0, 0]] * 4, [[2, 0, 2], [4, 0, 0], [0, 0, 0], [2, 0, .5]])
    np.testing.assert_array_equal(out["exit"], [True, True, True, False])
    np.testing.assert_array_equal(out["face"], [Z_PLUS, OUTER_RIM, INNER_HOLE, 0])
    np.testing.assert_allclose(out["distance_mm"][:3], 1.)
    np.testing.assert_allclose(out["segmentflight_ns"][:3], 1. / C_MM_NS)
    np.testing.assert_allclose(out["position_mm"][:3], [[2, 0, 1], [3, 0, 0], [1, 0, 0]])


def test_boundary_birth_outward_inward_and_scattering_endpoint():
    out = segments([[2, 0, 1], [2, 0, 1], [2, 0, 0], [2, 0, 0]],
                   [[2, 0, 2], [2, 0, 0], [2, 0, 1], [2, 0, 1]],
                   process=np.array([49, 49, 12, 49]))
    np.testing.assert_array_equal(out["exit"], [True, False, False, True])
    np.testing.assert_array_equal(out["ambiguous"], [False, False, True, False])
    assert out["distance_mm"][0] == 0
    # Even transport-only with inward/zero post momentum is not established outward.
    out = segments([[2, 0, 0]], [[2, 0, 1]], post_momentum=[[0, 0, -1]])
    assert out["endpoint_ambiguous"][0]


def test_tolerance_only_reports_near_boundary_never_nudges():
    pre = [[2, 0, 0], [2, 0, 1 + 1e-8]]
    post = [[2, 0, 1 - 1e-8], [2, 0, 2]]
    assert not segments(pre, post)["candidate"].any()
    out = segments(pre, post, boundary_tolerance_mm=1e-7)
    assert out["candidate"].all()
    assert out["boundary_ambiguous"].all()
    assert not out["exit"].any()
    assert np.isnan(out["position_mm"]).all()
    for tol in (-1, np.nan, [1]):
        with pytest.raises(ValueError, match="tolerance"):
            segments(pre, post, boundary_tolerance_mm=tol)


def test_hole_crossing_with_both_endpoints_in_material_and_invalid_segments():
    # A segment can leave/reenter the annulus with both endpoints inside.
    out = segments([[2, 0, 0]], [[-2, 0, 0]])
    assert out["exit"][0] and out["face"][0] == INNER_HOLE
    out = segments([[2, 0, 0], [2, 0, 0]], [[2, 0, 0], [np.nan, 0, 0]])
    assert out["invalid"].all() and not out["exit"].any()


def table(tree, n, **columns):
    out = {field: np.zeros(n, np.uint64) for field in FIELDS[tree]}
    for field in ("label", "instance", "name"):
        if field in out:
            out[field] = np.full(n, "", dtype="U30")
    for field in ("energy", "x", "y", "z", "px", "py", "pz", "post_x", "post_y", "post_z",
                  "post_px", "post_py", "post_pz", "time", "length"):
        if field in out:
            out[field] = np.zeros(n, float)
    if "run" in out:
        out["run"][:] = 1
        out["subrun"][:] = 7
    out.update({name: np.broadcast_to(value, (n,)).copy() for name, value in columns.items()})
    return out


@pytest.fixture
def sample():
    pid = np.uint64(2**63 + 1)
    products = table("products", 14, event=np.r_[np.repeat([1, 2, 3, 4], 3), 0, 1],
                     product=np.r_[np.tile(pid + np.arange(3, dtype=np.uint64), 4), pid + np.uint64(3), pid + np.uint64(4)],
                     kind=np.r_[np.tile([1, 3, 3], 4), 5, 1], scope=np.r_[np.zeros(12, int), 1, 0],
                     label=np.concatenate((np.tile(["g4run", "g4run", "BeamFilter"], 4), ["g4run", "upstream"])),
                     instance=np.concatenate((np.tile(["", "photonSteps", "Beam"], 4), ["", ""])),
                     size=[7, 10, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1])
    particles = table("particles", 9, event=[1]*7 + [2, 1],
                      product=np.r_[np.full(8, pid, np.uint64), pid + np.uint64(4)],
                      key=[10, 11, 12, 13, 14, 15, 16, 10, 10], pdg=np.full(9, 22),
                      stage=[2, 2, 2, 2, 2, 1, 2, 2, 2], creation=np.full(9, 16),
                      start_volume=np.full(9, 9), energy=np.full(9, 5.),
                      parent_state=np.ones(9, np.uint32), parent_product=np.full(9, pid, np.uint64),
                      parent_key=np.full(9, 999, np.uint64), x=np.full(9, 2.), z=[0, 0, 0, 0, 1, 0, 0, 0, 0])
    # Deliberately unsorted collection rows; copied and other-product photons share keys.
    steps = table("steps", 10, event=np.ones(10, int), product=np.full(10, pid + np.uint64(1), np.uint64),
                  index=[43, 0, 1, 2, 3, 4, 5, 40, 42, 50],
                  particle_state=np.ones(10, np.uint32),
                  particle_product=np.r_[np.full(9, pid, np.uint64), pid + np.uint64(4)],
                  particle_key=[16, 10, 11, 12, 13, 14, 15, 16, 16, 10],
                  x=np.full(10, 2.), z=np.array([0, 0, 0, 0, 0, 1, 0, 0, 2, 0], float),
                  post_x=np.array([2, 2, 4, 0, 2, 2, 2, 2, 2, 2], float),
                  post_z=[-2, 2, 0, 0, .5, 2, 2, 2, 0, 2],
                  px=[0, 0, 5, -5, 0, 0, 0, 0, 0, 0],
                  pz=[-5, 5, 0, 0, 5, 5, 5, 5, -5, 5],
                  post_px=[0, 0, 5, -5, 0, 0, 0, 0, 0, 0],
                  post_pz=[-5, 5, 0, 0, 5, 5, 5, 5, -5, 5],
                  time=np.arange(10, dtype=float) + 100, end_process=np.full(10, 49), length=np.full(10, 2.))
    volumes = table("volumes", 3, product=np.full(3, pid + np.uint64(3), np.uint64),
                    stage=[1, 2, 2], index=[9, 9, 10], name=["TargetFoil_01", "TargetFoil_00", "TargetFoil_01"])
    tables = dict(events=table("events", 4, event=[1, 2, 3, 4], ordinal=[0, 1, 2, 3]),
                  products=products, particles=particles, steps=steps, volumes=volumes)
    geom = SimpleNamespace(beam=SimpleNamespace(axis_x=0., foil_z=np.array([0., 10.]),
                           foil_ht=np.array([1., .5]), foil_rin=np.array([1., 0.]), foil_rout=np.array([3., 2.])))
    return tables, geom


def test_all_births_reentry_copied_ancestor_and_event_zeros(sample):
    tables, geom = sample
    out = extract(tables, geom)
    np.testing.assert_array_equal(out["key"], [10, 11, 12, 13, 14, 16, 10])
    np.testing.assert_array_equal(out["exit"], [True, True, True, False, True, True, False])
    np.testing.assert_array_equal(out["event_births"], [6, 1, 0, 0])
    np.testing.assert_array_equal(out["event_exits"], [5, 0, 0, 0])
    assert out["input_event_count"] == 4
    assert out["first_stepindex"][5] == 40  # not the later exit or file row order
    assert out["first_pre_time_ns"][5] == 107
    assert out["distance_mm"][4] == 0  # boundary-born outward gamma
    assert out["recorded_steps"][-1] == 0  # no steps, still a birth
    assert np.isnan(out["pre_energy_mev"][-1])
    assert out["product"].dtype == np.uint64 and out["product"][0] > 2**63
    np.testing.assert_allclose(out["pre_energy_mev"][out["exit"]], 5.)
    assert json.loads(out["report_json"].item())["copied_photons"] == 1
    assert not out["event_photon_steps_size_mismatch"].any()


def test_missing_join_is_reported_and_never_removes_birth(sample):
    tables, geom = sample
    tables["steps"]["particle_key"][1] = 888
    out = extract(tables, geom)
    assert len(out["key"]) == 7 and not out["exit"][0]
    assert json.loads(out["report_json"].item())["unresolved_photon_step_pointers"] == 1
    tables["particles"]["start_volume"][0] = 12345
    out = extract(tables, geom)
    assert json.loads(out["report_json"].item())["new_photons_without_volume"] == 1
    tables["products"]["label"][-2] = "unavailable"
    out = extract(tables, geom)
    assert len(out["key"]) == 0
    assert json.loads(out["report_json"].item())["photons_without_stage"] == 8


def test_first_ambiguous_candidate_not_replaced_by_later_exit(sample):
    tables, geom = sample
    tables["steps"]["post_z"][7] = 1.
    tables["steps"]["end_process"][7] = 12  # Compton at the boundary
    out = extract(tables, geom)
    assert out["first_stepindex"][5] == 40
    assert out["ambiguous"][5] and not out["exit"][5]
    assert out["first_end_process"][5] == 12


def test_missing_post_fields_fail_and_ambiguous_join_fails(sample):
    tables, geom = sample
    post_z = tables["steps"].pop("post_z")
    with pytest.raises(ValueError, match="post_z"):
        extract(tables, geom)
    tables["steps"]["post_z"] = post_z
    tables["volumes"]["index"][-1] = 9
    with pytest.raises(ValueError, match="ambiguous"):
        extract(tables, geom)


def test_geometry_annuli_not_nominal_or_nearest_z(sample):
    tables, geom = sample
    tables["particles"]["start_volume"][0] = 10
    # Metadata assigns foil 1 even though these positions still sit in foil 0.
    out = extract(tables, geom)
    assert out["foil_index"][0] == 1
    assert not out["birth_inside_geometry"][0] and not out["exit"][0]
    assert out["recorded_outside_start_segments"][0] == 1


def test_empty_collections_and_empty_ledger(sample):
    tables, geom = sample
    for name in ("particles", "steps"):
        tables[name] = {key: value[:0] for key, value in tables[name].items()}
    out = extract(tables, geom)
    assert len(out["key"]) == 0
    np.testing.assert_array_equal(out["event_births"], [0, 0, 0, 0])
    empty = {name: {key: value[:0] for key, value in table.items()} for name, table in tables.items()}
    out = extract(empty, geom)
    assert out["input_event_count"] == 0


def test_root_npz_roundtrip_and_no_exact_time_claim(sample, tmp_path, monkeypatch, capsys):
    import uproot
    import beamfastsim.flash_g4_photons as module

    tables, geom = sample
    path, output = tmp_path / "inclusive.root", tmp_path / "photons.npz"
    with uproot.recreate(path) as root:
        for name, arrays in tables.items():
            root.mktree("export/" + name, arrays)
    loaded = read_tables(path)
    assert loaded["steps"]["post_z"].shape == (10,)
    monkeypatch.setattr(module, "from_offline", lambda *args: geom)
    args = ["--input", str(path), "--out", str(output), "--geom", "frozen.txt",
            "--offline-root", "/local", "--events", "4"]
    assert main(args) == 0
    with np.load(output, allow_pickle=False) as out:
        assert out["event_births"].tolist() == [6, 1, 0, 0]
        assert out["product"].dtype == np.uint64
        assert "exit_time_ns" not in out.files
        assert "no exact crossing time" in json.loads(out["report_json"].item())["timing"]
    assert "s/file" in capsys.readouterr().out
    assert main(args[:-1] + ["5"]) == 1


def test_tolerance_checks_near_endpoint_process_on_both_sides():
    out = segments([[2, 0, 0]] * 2, [[2, 0, 1 - 1e-8], [2, 0, 1 + 1e-8]],
                   process=12, boundary_tolerance_mm=1e-7)
    assert out["candidate"].all() and out["ambiguous"].all()
    assert not out["exit"].any()


def test_all_event_input_state_denominator_and_missing_products(sample):
    tables, geom = sample
    # Declared source counts are retained even where source step rows are missing;
    # an explicit mismatch prevents interpreting this as a complete source export.
    tables["products"]["size"][[2, 5, 8, 11]] = [2, 3, 0, 5]
    tables["products"]["label"][10] = "missingRecorder"
    out = extract(tables, geom)
    np.testing.assert_array_equal(out["event_input_states_declared"], [2, 3, 0, 5])
    np.testing.assert_array_equal(out["event_input_states_size_mismatch"], [True, True, False, True])
    np.testing.assert_array_equal(out["event_photon_steps_present"], [True, True, True, False])
    assert out["input_state_count_declared"] == 10 and out["input_event_count"] == 4


def test_subrun_volume_join_never_reuses_other_subrun_metadata(sample):
    tables, geom = sample
    for name in ("events", "products", "particles"):
        table = tables[name]
        table["subrun"][table["event"] == 2] = 8
    out = extract(tables, geom)
    assert len(out["key"]) == 6
    assert json.loads(out["report_json"].item())["photons_without_stage"] == 1
    np.testing.assert_array_equal(out["event_births"], [6, 0, 0, 0])


def continuation_sample(sample, *, endpoint=1 - 1e-8, next_start=None, next_end=2.):
    """Photon 16 has collection indices 40, 42, 43, stored out of order."""
    tables, geom = sample
    s = tables["steps"]
    s["post_z"][7] = endpoint  # index 40, earliest candidate
    s["z"][8] = endpoint if next_start is None else next_start
    s["post_z"][8] = next_end  # index 42, continuation
    s["pz"][8] = s["post_pz"][8] = 5.
    return tables, geom


@pytest.mark.parametrize("endpoint,next_start,reconstructed", [
    (1 - 1e-8, None, True), (1 - 1e-8, 1 + 1e-8, True),
    (1 + 1e-8, None, False),  # already a strict single-segment transport exit
])
def test_two_segment_endpoint_short_and_just_outside(sample, endpoint, next_start, reconstructed):
    tables, geom = continuation_sample(sample, endpoint=endpoint, next_start=next_start)
    saved = {k: v.copy() for k, v in tables["steps"].items()}
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    i = 5
    assert out["exit"][i] and not out["ambiguous"][i]
    assert out["two_segment_reconstruction"][i] == reconstructed
    assert out["first_stepindex"][i] == 40 and out["first_pre_time_ns"][i] == 107
    assert out["first_end_process"][i] == 49
    np.testing.assert_allclose(out["position_mm"][i], [2, 0, 1], atol=0)
    assert out["distance_mm"][i] == 1.
    assert out["segmentflight_ns"][i] == 1. / C_MM_NS
    if reconstructed:
        assert out["continuation_stepindex"][i] == 42
        assert out["chord_length_mm"][i] == 2.
        assert out["first_segment_boundary_ambiguous"][i]
        assert out["recorded_ambiguous_segments"][i] >= 1
    for k, v in saved.items():
        np.testing.assert_array_equal(tables["steps"][k], v)
    assert "exit_time_ns" not in out
    assert json.loads(out["report_json"].item())["two_segment_reconstructions"] == int(reconstructed)


@pytest.mark.parametrize("failure", ["gap", "energy", "previous_post_energy", "direction",
                                     "previous_post_direction", "chord", "process", "no_next",
                                     "different_photon", "intervening_step", "not_far_outside"])
def test_two_segment_requires_explicit_continuity(sample, failure):
    tables, geom = continuation_sample(sample)
    s = tables["steps"]
    if failure == "gap":
        s["z"][8] += 1e-5
    elif failure == "energy":
        s["pz"][8] = s["post_pz"][8] = 4.
    elif failure == "previous_post_energy":
        s["post_pz"][7] = 4.
    elif failure == "direction":
        s["pz"][8] = -5.
    elif failure == "previous_post_direction":
        s["post_pz"][7] = -5.
    elif failure == "chord":
        s["post_x"][8] += .01
    elif failure == "process":
        s["end_process"][7] = 12
    elif failure == "no_next":
        s["particle_key"][[8, 0]] = 13
    elif failure == "different_photon":
        s["particle_product"][8] = tables["particles"]["product"][-1]
        s["particle_key"][8] = 10
    elif failure == "intervening_step":
        s["index"][0] = 41  # broken segment prevents skipping ahead to index 42
    elif failure == "not_far_outside":
        s["post_z"][8] = 1 + 5e-8
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["first_stepindex"][5] == 40
    assert out["ambiguous"][5] and not out["exit"][5]
    assert not out["two_segment_reconstruction"][5]


def test_two_segment_same_energy_next_process_outside_is_not_boundary_process(sample):
    tables, geom = continuation_sample(sample)
    tables["steps"]["end_process"][8] = 12
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["two_segment_reconstruction"][5]
    # A deflection at that endpoint is conservatively not straight evidence.
    tables["steps"]["post_pz"][8] = -5.
    assert extract(tables, geom, boundary_tolerance_mm=1e-7)["ambiguous"][5]


@pytest.mark.parametrize("next_x,resolved", [(0., True), (-2., False), (-4., False)])
def test_two_segment_hole_reentry_not_fake_crossing(sample, next_x, resolved):
    tables, geom = sample
    s = tables["steps"]
    s["post_x"][7], s["post_z"][7] = 1 + 1e-8, 0.
    s["x"][8], s["z"][8] = 1 + 1e-8, 0.
    s["post_x"][8], s["post_z"][8] = next_x, 0.
    for prefix in ("", "post_"):
        s[prefix + "px"][[7, 8]] = -5.
        s[prefix + "pz"][[7, 8]] = 0.
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["two_segment_reconstruction"][5] == resolved
    assert out["exit"][5] == resolved
    assert out["ambiguous"][5] != resolved
    if resolved:
        assert out["face"][5] == INNER_HOLE
        np.testing.assert_allclose(out["position_mm"][5], [1, 0, 0])


def test_two_segment_boundary_birth_and_outside_start_not_repaired(sample):
    tables, geom = continuation_sample(sample)
    s = tables["steps"]
    s["z"][7] = 1 + 1e-8
    s["post_z"][7] = s["z"][8] = 1 + 2e-8
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["ambiguous"][5] and not out["two_segment_reconstruction"][5]
    assert out["exit"][4] and out["distance_mm"][4] == 0  # exact boundary birth
    assert not out["two_segment_reconstruction"][4]


def test_two_segment_outer_rim_and_no_undeclared_tolerance(sample):
    tables, geom = sample
    s = tables["steps"]
    s["post_x"][7], s["post_z"][7] = 3 - 1e-8, 0.
    s["x"][8], s["z"][8] = 3 - 1e-8, 0.
    s["post_x"][8], s["post_z"][8] = 4., 0.
    for prefix in ("", "post_"):
        s[prefix + "px"][[7, 8]] = 5.
        s[prefix + "pz"][[7, 8]] = 0.
    strict = extract(tables, geom)
    assert strict["first_stepindex"][5] == 42
    assert not strict["two_segment_reconstruction"].any()
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["first_stepindex"][5] == 40
    assert out["two_segment_reconstruction"][5]
    assert out["face"][5] == OUTER_RIM
    np.testing.assert_allclose(out["position_mm"][5], [3, 0, 0])


def test_two_segment_does_not_skip_earlier_nontransport_candidate(sample):
    tables, geom = continuation_sample(sample)
    s = tables["steps"]
    # Earliest near-short segment is nontransport; later otherwise-good evidence
    # cannot replace it. File row order remains deliberately scrambled.
    s["end_process"][7] = 12
    s["z"][8], s["post_z"][8] = 0., 1 - 1e-8
    s["z"][0], s["post_z"][0] = 1 - 1e-8, 2.
    s["pz"][0] = s["post_pz"][0] = 5.
    out = extract(tables, geom, boundary_tolerance_mm=1e-7)
    assert out["first_stepindex"][5] == 40
    assert out["ambiguous"][5] and not out["exit"][5]
    assert not out["two_segment_reconstruction"].any()
