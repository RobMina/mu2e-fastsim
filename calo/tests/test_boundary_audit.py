"""Recorded momentum and ancestral time coverage are necessary, not sufficient gates."""
import numpy as np
import pytest

from test_controls import load_script


def fixture():
    events = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([2]), g4_status=np.array([0]), n_showers=np.array([1]))
    particles = dict(run=np.array([1, 1]), subrun=np.array([0, 0]), event=np.array([2, 2]),
                     sim_product_id=np.array([7, 7]), sim_key=np.array([10, 11]),
                     parent_valid=np.array([False, True]), parent_product_id=np.array([0, 7]), parent_key=np.array([0, 10]),
                     start_time=np.array([0., 1.5], dtype=np.float32))
    steps = dict(run=np.array([1, 1]), subrun=np.array([0, 0]), event=np.array([2, 2]),
                 sim_product_id=np.array([7, 7]), sim_key=np.array([10, 11]), kind=np.array([0, 1]),
                 volume=np.array([73, 0]), time=np.array([1., 2.]), energy_g4=np.array([0., 3.]))
    for prefix in ("position", "momentum"):
        for axis in "xyz":
            steps[f"{prefix}_{axis}"] = np.zeros(2)
    steps["momentum_z"][0] = 10.
    showers = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([2]), sim_product_id=np.array([7]),
                   sim_key=np.array([10]), crystal=np.array([0]), time=np.array([2.]), energy_g4=np.array([3.]), n_compress=np.array([1]))
    disks = dict(run=np.array([1]), disk=np.array([0]))
    for prefix in ("origin", "axis_x", "axis_y", "axis_z"):
        for axis in "xyz":
            disks[f"{prefix}_{axis}"] = np.array([float(prefix == f"axis_{axis}")])
    return events, particles, steps, showers, disks


def test_ancestral_incoming_coverage_is_not_a_certificate():
    report = load_script("audit_boundary").describe(*fixture())
    assert report["raw_steps_without_earlier_incoming_ancestor_row"] == 0
    assert report["shower_rows_without_earlier_incoming_ancestor_row"] == 0
    assert report["raw_constituent_count"] == report["stored_compressed_constituent_count"] == 1
    assert report["compressed_constituent_causality_certified"] is False
    assert report["material_exterior_coverage_certified"] is False
    assert report["zero_response_certified"] is False


@pytest.mark.parametrize("change", ["outgoing", "late_crossing", "wrong_identity", "different_event"])
def test_outgoing_late_and_nonancestor_rows_cannot_supply_coverage(change):
    args = list(fixture())
    steps = args[2]
    if change == "outgoing":
        steps["momentum_z"][0] *= -1
    elif change == "late_crossing":
        steps["time"][0] = 3
    elif change == "wrong_identity":
        args[1]["parent_valid"][1] = False
        args[3]["sim_key"][0] = 11
    else:
        steps["event"][0] = 99
    if change == "different_event":
        with pytest.raises(ValueError, match="identity"):
            load_script("audit_boundary").describe(*args)
    else:
        report = load_script("audit_boundary").describe(*args)
        assert report["raw_steps_without_earlier_incoming_ancestor_row"] == 1
        assert report["raw_step_gap_event_columns"]["event"] == [2]


def test_front_and_back_and_radial_momentum_signs():
    _, _, steps, _, disks = fixture()
    shell = {name: np.repeat(value[:1], 4) for name, value in steps.items()}
    shell["volume"] = np.array([73, 74, 77, 78])
    shell["position_x"] = np.array([0., 0., 335., 719.])
    shell["momentum_x"] = np.array([0., 0., 1., -1.])
    shell["momentum_z"] = np.array([1., -1., 0., 0.])
    dot, disk = load_script("audit_boundary").directions(shell, disks)
    np.testing.assert_array_equal(dot, [-1., -1., -1., -1.])
    np.testing.assert_array_equal(disk, [0, 0, 0, 0])


def test_raw_only_response_is_not_promoted_to_zero():
    args = list(fixture())
    args[0]["n_showers"][0] = 0
    args[3] = {name: value[:0] for name, value in args[3].items()}
    report = load_script("audit_boundary").describe(*args)
    assert report["empty_shower_events_with_raw_crystal_rows"] == 1
    assert report["raw_only_event_crystal_groups"] == 1
    assert report["raw_only_total_energy_mev"] == 3
    assert report["zero_response_certified"] is False


def test_empty_collections_are_valid_diagnostic_input():
    args = list(fixture())
    args[0]["n_showers"][0] = 0
    args[2] = {name: value[:0] for name, value in args[2].items()}
    args[3] = {name: value[:0] for name, value in args[3].items()}
    report = load_script("audit_boundary").describe(*args)
    assert report["shell_rows"] == report["raw_crystal_rows"] == 0
    assert report["events_with_incoming_momentum_rows"] == 0


def test_failed_transport_refused():
    args = fixture()
    args[0]["g4_status"][0] = 1
    with pytest.raises(ValueError, match="status-zero"):
        load_script("audit_boundary").describe(*args)


def test_secondary_born_before_parent_entrance_cannot_inherit_later_crossing():
    args = fixture()
    args[1]["start_time"][1] = .5
    report = load_script("audit_boundary").describe(*args)
    assert report["raw_steps_without_earlier_incoming_ancestor_row"] == 0
    assert report["raw_steps_without_birth_order_compatible_incoming_ancestor"] == 1


def test_causal_pointer_doubling_respects_intervening_birth_bounds():
    script = load_script("audit_boundary")
    parent = np.array([0, 0, 1, 2])
    earliest = np.array([10., np.inf, 30., np.inf])
    np.testing.assert_array_equal(script.causal_times(parent, earliest, np.array([0., 5., 20., 40.])),
                                  [10., np.inf, 30., 30.])


def test_independent_secondary_entrances_remain_correlated_in_one_event():
    args = list(fixture())
    args[1]["start_time"][1] = .5
    args[2] = {name: np.r_[value[:1], value[:1], value[1:]] for name, value in args[2].items()}
    args[2]["sim_key"][1] = 11
    report = load_script("audit_boundary").describe(*args)
    assert report["events_with_multiple_birth_order_distinct_entrance_lineages"] == 1
    assert report["entrance_lineage_count_histogram"] == {"2": 1}
    assert report["raw_steps_without_birth_order_compatible_incoming_ancestor"] == 0


def test_float_birth_uncertainty_is_counted_not_hidden():
    args = fixture()
    args[1]["start_time"][1] = 1.
    report = load_script("audit_boundary").describe(*args)
    assert report["raw_steps_without_birth_order_compatible_incoming_ancestor"] == 0
    assert report["raw_steps_depending_on_float_birth_ambiguity"] == 1
