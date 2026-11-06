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
