"""Exact all-input event matching for source-vs-Geant4 component diagnostics."""
import json

import numpy as np
import pytest

from beamfastsim.flash_s1_reference import compare


@pytest.fixture
def case():
    fast = dict(event=np.array([[1, 2, 10], [1, 2, 11], [1, 3, 12], [1, 3, 13]], np.uint64),
                uncollided_photons=np.array([0., 1., 2., 1.]), uncollided_energy_mev=np.array([0., 3., 6., 3.]))
    report = dict(source_events=4, dropped_births=0, capped_parents=0, loopers=1, scope="limited")
    reference = {"event_" + k: fast["event"][:, i].copy() for i, k in enumerate(("run", "subrun", "event"))}
    reference.update(evt_primary_exits=np.array([0, 1, 1, 0]), evt_primary_energy=np.array([0., 3., 3., 0.]))
    diagnostics = {k: 0 for k in ("primary_birth_count", "primary_uncollided_exit_count", "ambiguous_count",
        "unresolved_count", "unknown_ancestry_count", "unknown_history_count", "primary_no_recorded_exit_count",
        "primary_prior_non_geometry_count", "source_states_declared", "source_missing_events", "source_size_mismatch_events")}
    reference["report_json"] = np.asarray(json.dumps(diagnostics))
    return fast, report, reference


def test_paired_uncertainty_and_event_zeros(case):
    fast, report, g4 = case
    out = compare(fast, report, [g4])
    assert out["metrics"]["photons"]["fast_over_g4"] == 2
    assert out["metrics"]["photons"]["g4_total"] == 2
    assert out["metrics"]["energy_mev"]["fast_total"] == 12
    assert out["input_events"] == 4 and out["fast_loopers"] == 1


def test_disjoint_shards_order_and_full_uint64(case):
    fast, report, g4 = case
    fast["event"] += np.uint64(2**63 + 1)
    for name in ("event_run", "event_subrun", "event_event"):
        g4[name] += np.uint64(2**63 + 1)
    a = {k: v if k == "report_json" else v[:2] for k, v in g4.items()}
    b = {k: v if k == "report_json" else v[2:] for k, v in g4.items()}
    assert compare(fast, report, [b, a])["metrics"]["photons"]["fast_over_g4"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        compare(fast, report, [a, a])


def test_missing_event_and_failed_source_not_imputed(case):
    fast, report, g4 = case
    g4["event_event"][0] = 99
    with pytest.raises(ValueError, match="universes"):
        compare(fast, report, [g4])
    g4["event_event"][0] = 10
    report["capped_parents"] = 1
    with pytest.raises(ValueError, match="truncated"):
        compare(fast, report, [g4])


def test_generated_fast_totals_compare_with_verified_zero_source_event(case):
    from types import SimpleNamespace
    from beamfastsim.flash_flux import exit_records, summarize
    _, _, reference = case
    payload = dict(source_run=np.array([1], np.int64), source_subrun=np.array([2], np.int64),
                   source_event=np.array([10], np.uint64), source_pdg=np.array([11]),
                   source_time_ns=np.array([0.]), photon_valid=np.array([[True]]),
                   photon_position=np.array([[[2., 0., 0.]]]),
                   photon_direction=np.array([[[0., 0., 1.]]]), photon_energy=np.array([[5.]]),
                   photon_time=np.array([[0.]]), overflow=np.array([0]), capped=np.array([False]),
                   loopers=np.array([False]))
    beam = SimpleNamespace(axis_x=0., foil_z=np.array([0.]), foil_ht=np.array([1.]),
                           foil_rin=np.array([1.]), foil_rout=np.array([3.]))
    ledger = np.array([[1, 2, 10], [1, 2, 11]], np.uint64)
    # G4 explicitly verifies both inputs, with zero states in the second event.
    g4 = {"event_" + k: ledger[:, i] for i, k in enumerate(("run", "subrun", "event"))}
    g4.update(evt_primary_exits=np.array([1, 0]), evt_primary_energy=np.array([5., 0.]),
              event_input_states_declared=np.array([1, 0]), event_input_states_present=np.array([True, True]),
              event_input_states_size_mismatch=np.array([False, False]), report_json=reference["report_json"])
    photons = exit_records(payload, beam)
    report, fast = summarize(payload, photons, source_events=2, event_ledger=ledger)
    result = compare(fast, report, [g4])
    assert result["input_events"] == 2
    np.testing.assert_array_equal(fast["source_parents"], [1, 0])
    np.testing.assert_array_equal(fast["uncollided_photons"], [photons["weight"][0], 0.])
    assert result["metrics"]["photons"]["fast_mean_per_event"] == pytest.approx(photons["weight"][0] / 2)
    assert result["metrics"]["energy_mev"]["fast_total"] == pytest.approx(photons["weight"][0] * 5)
    # Neither a scalar denominator nor an unrelated ledger proves the missing ID.
    report, incomplete = summarize(payload, photons, source_events=2)
    with pytest.raises(ValueError, match="universes"):
        compare(incomplete, report, [g4])
    wrong = ledger.copy()
    wrong[1, 2] = 12
    report, incomplete = summarize(payload, photons, source_events=2, event_ledger=wrong)
    with pytest.raises(ValueError, match="universes"):
        compare(incomplete, report, [g4])
