"""Reference figure inputs preserve full denominators and energy components."""
import numpy as np
import pytest

from beamfastsim.flash_reference_figures import components


@pytest.fixture
def case():
    origin = dict(input_events=10, foil_ancestry=dict(status=[0, 0, 1, 2],
        foil_born_observed=[False, True, True, True], ancestry_complete=[True]*4,
        ionizing_mev=[1., 2., 3., .5]))
    comparison = dict(events=10, totals={k: dict(low=dict(sum_mev=v, event_mean_se=.2))
        for k, v in (("raw", 6.5), ("owned", 3.), ("residual", 3.), ("unknown", .5))})
    return origin, comparison


def test_units_and_component_order(case):
    values, error = components(*case, "low")
    np.testing.assert_array_equal(values, [300000., 200000., 100000., 50000.])
    assert error == 200000.


def test_mismatched_events_fail(case):
    case[0]["input_events"] = 9
    with pytest.raises(ValueError, match="denominators"):
        components(*case, "low")


def test_energy_mismatch_fails(case):
    case[0]["foil_ancestry"]["ionizing_mev"][0] += 1
    with pytest.raises(ValueError, match="components"):
        components(*case, "low")


def test_unknown_metadata_fails(case):
    case[0]["foil_ancestry"]["ancestry_complete"][0] = False
    with pytest.raises(ValueError, match="complete"):
        components(*case, "low")
