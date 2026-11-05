"""Source compression rules tested independently of private pilot results."""
import numpy as np
import pytest

from calofastsim.compression import ancestor_rows, compress, containment


def fixture(times=(0., .1, .3), depths=(1., 1., 1.)):
    particles = dict(run=np.ones(3, dtype=int), subrun=np.zeros(3, dtype=int), event=np.ones(3, dtype=int),
                     sim_product_id=np.full(3, 7), sim_key=np.arange(3), parent_valid=np.array([False, True, True]),
                     parent_product_id=np.array([0, 7, 7]), parent_key=np.array([0, 0, 1]))
    for name, values in (("start_x", [-20., 0., 0.]), ("start_y", [0., 0., 0.]), ("start_z", [1., 1., 1.]),
                          ("end_x", [0., 0., 0.]), ("end_y", [0., 0., 0.]), ("end_z", [2., 2., 2.])):
        particles[name] = np.array(values)
    n = len(times)
    steps = dict(run=np.ones(n, dtype=int), subrun=np.zeros(n, dtype=int), event=np.ones(n, dtype=int),
                 sim_product_id=np.full(n, 7), sim_key=np.full(n, 2), volume=np.zeros(n, dtype=int), time=np.asarray(times),
                 energy_g4=np.ones(n, dtype=np.float32), energy_visible=np.full(n, .5, dtype=np.float32))
    for name in ("position_x", "position_y", "momentum_x", "momentum_y"):
        steps[name] = np.zeros(n)
    steps["position_z"] = np.asarray(depths)
    steps["momentum_z"] = np.arange(n, dtype=float) + 1
    crystals = dict(run=np.array([1, 1]), crystal=np.array([0, 1]), disk=np.array([0, 0]))
    disks = dict(run=np.array([1]), disk=np.array([0]))
    for axis in "xyz":
        crystals[f"local_position_{axis}"] = np.array([0., 10.]) if axis == "x" else np.zeros(2)
        crystals[f"size_{axis}"] = np.full(2, 20. if axis == "z" else 4.)
        disks[f"origin_{axis}"] = np.zeros(1)
        disks[f"front_face_origin_{axis}"] = np.zeros(1)
        disks[f"size_{axis}"] = np.array([dict(x=0., y=100., z=40.)[axis]])
        for component in "xyz":
            disks[f"axis_{axis}_{component}"] = np.array([float(axis == component)])
    return particles, steps, crystals, disks


def run(args, **kwargs):
    return compress(*args, num_z_slices=2, delta_time=.2, energy_threshold=kwargs.get("threshold", 0.), crystal_length=20.)


def test_exact_ancestry_and_first_time_not_previous_time_grouping():
    args = fixture()
    np.testing.assert_array_equal(ancestor_rows(*args), [0, 0, 0])
    result = run(args)
    np.testing.assert_array_equal(result["n_compress"], [2, 1])
    np.testing.assert_array_equal(result["energy_g4"], [2., 1.])
    np.testing.assert_array_equal(result["energy_birks"], [1., .5])
    np.testing.assert_array_equal(result["time"], np.array([.05, .3], dtype=np.float32).astype(np.float64))
    np.testing.assert_array_equal(result["momentum_in"], [2., 3.])
    np.testing.assert_array_equal(result["member_index"], [0, 1, 2])
    np.testing.assert_array_equal(result["member_offsets"], [0, 2, 3])


def test_inside_secondary_leaving_disk_is_not_folded_into_parent():
    args = fixture()
    args[0]["end_z"][1] = 100
    np.testing.assert_array_equal(ancestor_rows(*args), [1, 1, 1])
    args[0]["start_x"][2] = 10
    inside, _ = containment(args[0], args[2], args[3])
    np.testing.assert_array_equal(inside, [False, True, True])


def test_flush_trigger_order_then_final_bucket_order_not_time_order():
    args = fixture(times=(0., .1, .3, .4), depths=(15., 1., 15., 1.))
    result = run(args)
    np.testing.assert_array_equal(result["position_z"], [15., 1., 1., 15.])
    np.testing.assert_array_equal(result["member_index"], [0, 1, 3, 2])


def test_threshold_is_on_whole_ancestor_crystal_before_compression():
    result = run(fixture(), threshold=3.)
    assert result["n_compress"].sum() == 3
    result = run(fixture(), threshold=3.001)
    assert len(result["n_compress"]) == 0
    np.testing.assert_array_equal(result["excluded_raw_index"], [0, 1, 2])


def test_slice_epsilon_clamping_and_float_delta_boundary():
    args = fixture(times=(0., float(np.float32(.2)), 1.), depths=(-.01, 10., 25.))
    result = run(args)
    np.testing.assert_array_equal(result["n_compress"], [2, 1])
    np.testing.assert_array_equal(result["position_z"], [4.995, 25.])


def test_equal_time_domain_flagged_even_if_outputs_look_equal():
    result = run(fixture(times=(0., 0., 1.)))
    assert np.all(result["equal_time_order_unspecified"])
    np.testing.assert_array_equal(result["member_index"], [0, 1, 2])


def test_empty_raw_collection_and_unsorted_particle_storage():
    args = list(fixture(times=(), depths=()))
    result = run(args)
    assert len(result["n_compress"]) == 0
    np.testing.assert_array_equal(result["member_offsets"], [0])
    args = list(fixture())
    args[0] = {name: value[[2, 0, 1]] for name, value in args[0].items()}
    np.testing.assert_array_equal(run(args)["sim_key"], [0, 0])


@pytest.mark.parametrize("change", ["nan", "negative", "bad_parent", "unresolved_step", "missing_crystal", "bad_frame"])
def test_invalid_input_fails_closed(change):
    args = fixture()
    if change == "nan":
        args[1]["time"][0] = np.nan
    elif change == "negative":
        args[1]["energy_g4"][0] = -1
    elif change == "bad_parent":
        args[0]["parent_key"][1] = 99
    elif change == "unresolved_step":
        args[1]["sim_product_id"][0] = 8
    elif change == "missing_crystal":
        args[1]["volume"][0] = 2
    else:
        args[3]["axis_x_x"][0] = 2
    with pytest.raises(ValueError):
        run(args)


def test_replay_checker_flags_ties_and_never_certifies_unobserved_membership():
    from test_controls import load_script
    prediction = run(fixture())
    saved = {name: value.copy() for name, value in prediction.items()}
    report = load_script("replay_compression").compare(prediction, saved)
    assert report["ordered_layout_counts_exact"]
    assert report["all_numeric_fields_exact"]
    assert report["bitwise_row_replay_certified"]
    assert not report["constituent_identity_certified"]
    prediction["equal_time_order_unspecified"][:] = True
    report = load_script("replay_compression").compare(prediction, saved)
    assert not report["bitwise_row_replay_certified"]
    saved["sim_key"][:] = 99
    assert not load_script("replay_compression").compare(prediction, saved)["ordered_layout_counts_exact"]


def test_configuration_requires_explicit_supported_source_and_parameters(tmp_path):
    from test_controls import load_script
    path = tmp_path / "expanded.fcl"
    block = '''CaloShowerStepMaker: {
      caloStepPointCollection: [ "g4run:calorimeter" ]
      compressData: true
      numZSlices: 20
      deltaTime: 2e-1
      eDepThreshold: 0
    }'''
    path.write_text(block)
    assert load_script("replay_compression").configuration(path) == dict(num_z_slices=20, delta_time=.2, energy_threshold=0.)
    for text in (block + block, block.replace("true", "false"), block.replace("g4run:calorimeter", "other"), block.replace("deltaTime: 2e-1", "")):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_script("replay_compression").configuration(path)


def test_cross_bucket_equal_trigger_times_have_unspecified_flush_order():
    args = fixture(times=(0., 0., 1., 1.), depths=(1., 15., 1., 15.))
    prediction = run(args)
    np.testing.assert_array_equal(prediction["flush_is_final"], [False, False, True, True])
    np.testing.assert_array_equal(prediction["flush_trigger_time"][:2], [1., 1.])
    saved = {name: value.copy() for name, value in prediction.items()}
    row_fields = ("run", "subrun", "event", "sim_product_id", "sim_key", "crystal", "n_compress",
                  "energy_g4", "energy_birks", "time", "momentum_in", "position_x", "position_y", "position_z")
    for name in row_fields:
        saved[name] = saved[name][[1, 0, 2, 3]]
    from test_controls import load_script
    report = load_script("replay_compression").compare(prediction, saved)
    assert report["row_multiset_replay_within_tolerance"]
    assert not report["all_numeric_fields_exact"]
    assert not report["bitwise_row_replay_certified"]


def test_inside_boundary_does_not_reuse_unrelated_product_key():
    args = fixture()
    p = args[0]
    p["sim_product_id"][0] = 8
    p["parent_product_id"][1] = 8
    assert np.all(run(args)["sim_product_id"] == 8)


def test_double_weighted_position_float_constructor_time_and_maximum_momentum():
    args = fixture(times=(1., 1.1, 2.), depths=(1., 2., 3.))
    args[1]["energy_g4"] = np.array([1., 3., 2.], dtype=np.float32)
    result = run(args)
    assert result["position_z"][0] == 1.75
    assert result["time"][0] == float(np.float32((1. + 3 * 1.1) / 4))
    assert result["momentum_in"][0] == 2


def test_two_disk_leakage_keeps_crystal_identity_and_does_not_fold_cross_disk_particle():
    p, steps, crystals, disks = fixture()
    disks = {name: np.r_[value, value] for name, value in disks.items()}
    disks["disk"][1] = 1
    disks["origin_z"][1] = disks["front_face_origin_z"][1] = 100
    crystals["disk"][1] = 1
    crystals["local_position_x"][1] = 0
    p["end_z"][1] = 101
    p["start_z"][2] = 101
    p["end_z"][2] = 102
    steps["volume"][:] = 1
    steps["position_z"][:] += 100
    result = run((p, steps, crystals, disks))
    np.testing.assert_array_equal(result["sim_key"], [1, 1])
    np.testing.assert_array_equal(result["crystal"], [1, 1])
    np.testing.assert_array_equal(result["position_z"], [1., 1.])


@pytest.mark.parametrize("change", ["run", "duplicate_disk", "nan_position", "huge_depth", "bad_size"])
def test_unsupported_geometry_and_overflow_domains_refused(change):
    args = fixture()
    if change == "run":
        args[0]["run"][:] = 99
        args[1]["run"][:] = 99
    elif change == "duplicate_disk":
        args = list(args)
        args[3] = {name: np.r_[value, value] for name, value in args[3].items()}
    elif change == "nan_position":
        args[2]["local_position_x"][0] = np.nan
    elif change == "huge_depth":
        args[1]["position_z"][0] = 1e20
    else:
        args[2]["size_y"][0] = -1
    with pytest.raises(ValueError):
        run(args)


def test_reverse_equal_time_variant_is_a_diagnostic_not_stock_sort_emulation():
    args = fixture(times=(0., 0., 1., 1.), depths=(1., 15., 1., 15.))
    stable = run(args)
    reverse = compress(*args, num_z_slices=2, delta_time=.2, energy_threshold=0., crystal_length=20., equal_time_order="reverse")
    np.testing.assert_array_equal(stable["bucket"], [0, 1, 0, 1])
    np.testing.assert_array_equal(reverse["bucket"], [1, 0, 0, 1])
    assert reverse["equal_time_order_unspecified"].all()
    with pytest.raises(ValueError, match="ordering"):
        compress(*args, num_z_slices=2, delta_time=.2, energy_threshold=0., crystal_length=20., equal_time_order="guessed_cpp")


def membership_arrays(prediction, args, mode):
    import json
    from calofastsim.ancestry import SIM_FIELDS
    from calofastsim.identities import EVENT_FIELDS
    member = prediction["member_index"]
    arrays = {f"predicted_{name}": prediction[name] for name in SIM_FIELDS + ("crystal", "n_compress")}
    arrays.update({name: prediction[name] for name in ("member_offsets", "bucket", "generation")})
    for name in EVENT_FIELDS:
        arrays[f"raw_{name}"] = args[1][name][member]
    arrays["raw_product_id"] = np.full(len(member), 9)
    arrays["raw_index"] = member
    arrays["provenance_json"] = np.asarray(json.dumps(dict(input_sha256="a" * 64, expanded_fcl_sha256="b" * 64,
                                                          configuration={}, crystal_length_mm=20., equal_time_order=mode)))
    return arrays


def test_alternative_order_partitions_join_only_by_raw_and_algorithmic_identities():
    from test_controls import load_script
    args = fixture(times=(0., 0., 1., 1.), depths=(1., 15., 1., 15.))
    left = membership_arrays(run(args), args, "stable")
    other = compress(*args, num_z_slices=2, delta_time=.2, energy_threshold=0., crystal_length=20., equal_time_order="reverse")
    right = membership_arrays(other, args, "reverse")
    result = load_script("compare_compression_membership").compare(left, right)
    assert result["raw_constituent_group_assignment_equal"]
    assert not result["group_output_order_equal"]
    assert not result["stock_constituent_identity_certified"]
    right["raw_index"][1] = right["raw_index"][0]
    with pytest.raises(ValueError, match="raw identity"):
        load_script("compare_compression_membership").compare(left, right)
