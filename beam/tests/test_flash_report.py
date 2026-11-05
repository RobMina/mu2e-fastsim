"""No comparison may infer a generated denominator from nonempty detector events."""
import pytest
from beamfastsim.flash_report import compare, compare_regions


def report(energy, denominator):
    return dict(normalization=dict(certified_generated_events=denominator), files=[dict(summary=dict(
        eion_sum_mev=energy, photon_birth=dict(z_edges_mm=[0, 1], r_edges_mm=[0, 1],
        eion_sum_mev=[[energy * .75]], no_photon=dict(eion_sum_mev=energy * .25),
        nonfinite_birth=dict(eion_sum_mev=0), outside_bins=dict(eion_sum_mev=0))))])


def test_difference_ledger_and_unresolved_energy():
    result = compare(report(2., 10), report(4., 10))
    assert result["high_over_low"] == 2.
    assert result["delta_mev_per_generated"] == .2
    assert result["fraction_of_design_delta_by_birth_bin"][0][0] == pytest.approx(.75)
    assert result["unresolved_delta_mev_per_generated"] == .05


def test_missing_denominator_fails_closed():
    with pytest.raises(ValueError, match="denominators"):
        compare(report(1., None), report(2., 10))


def test_mismatched_bins_or_energy_do_not_silently_compare():
    b = report(4., 10)
    b["files"][0]["summary"]["photon_birth"]["z_edges_mm"] = [1, 2]
    with pytest.raises(ValueError, match="bins differ"):
        compare(report(2., 10), b)
    b = report(4., 10)
    b["files"][0]["summary"]["eion_sum_mev"] = 6.
    with pytest.raises(ValueError, match="accounting"):
        compare(report(2., 10), b)


def test_region_ledger_rejects_uncertified_denominator():
    def region_report(energy, denominator):
        return dict(normalization=dict(certified_generated_events=denominator),
                    by_region=dict(region=["foil", "other"], eion_sum_mev=energy))
    result = compare_regions(region_report([2., 1.], 10), region_report([6., 2.], 10))
    assert result["fraction_of_design_delta"] == pytest.approx([.8, .2])
    with pytest.raises(ValueError, match="denominators"):
        compare_regions(region_report([2., 1.], None), region_report([6., 2.], 10))
