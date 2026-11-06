"""Saved-only, exact-key absolute S1 spectra with full-event paired uncertainty."""
import copy
import json

import numpy as np
import pytest

from beamfastsim.flash_s1_spectrum import (
    ENERGY_LOWER_EDGES_MEV, compare, event_histograms, main, paired_columns,
)


@pytest.fixture
def case():
    event = np.array([[1, 2, 10], [1, 2, 11], [1, 3, 12], [1, 3, 13]], np.uint64)
    source = {"source_" + k: event[1:, i].copy() for i, k in enumerate(("run", "subrun", "event"))}
    source.update(source_particle_key=np.array([5, 5, 5], np.uint64),
                  loopers=np.array([False, True, False]), overflow=np.zeros(3, int), capped=np.zeros(3, bool))
    photons = dict(parent=np.array([0, 0, 1, 1, 1, 1]), energy=np.array([.001, 1., 10., 100., 1000., .0005]),
                   weight=np.array([.5, .5, .25, .25, .25, 0.]), foil_exit=np.ones(6, bool),
                   attenuation_supported=np.array([True, True, True, True, True, False]),
                   path_in_foil_mm=np.full(6, .1))
    fi = np.array([1, 1, 2, 2, 2, 2])
    fast = dict(event=event)
    for k, weight in (("uncollided_photons", photons["weight"]),
                      ("uncollided_energy_mev", photons["weight"] * photons["energy"]),
                      ("transparent_photons", photons["foil_exit"].astype(float)),
                      ("transparent_energy_mev", photons["energy"])):
        fast[k] = np.bincount(fi, weights=weight, minlength=4)
    report = dict(source_events=4, dropped_births=0, capped_parents=0, loopers=1, scope="limited")
    reference = {"event_" + k: event[:, i].copy() for i, k in enumerate(("run", "subrun", "event"))}
    reference.update({k: event[[1, 2, 2, 3], i].copy() for i, k in enumerate(("run", "subrun", "event"))})
    reference.update(product=np.array([7, 7, 8, 7], np.uint64), key=np.array([1, 1, 1, 1], np.uint64),
                     pre_energy_mev=np.array([1., 10., 100., np.nan]), birth_energy_mev=np.array([1., 10., 100., .0008]),
                     primary_brems=np.ones(4, bool), uncollided_exit=np.array([True, True, True, False]),
                     exit=np.array([True, True, True, False]), unresolved=np.array([False, False, False, True]),
                     unknown_ancestry=np.zeros(4, bool), unknown_history=np.array([False, False, False, True]),
                     ambiguous=np.zeros(4, bool), birth_inside_geometry=np.ones(4, bool),
                     no_recorded_exit=np.array([False, False, False, True]), prior_non_geometry_steps=np.zeros(4, int),
                     evt_primary_exits=np.array([0, 1, 2, 0]), evt_primary_energy=np.array([0., 1., 110., 0.]))
    diagnostics = {k: 0 for k in ("primary_birth_count", "primary_uncollided_exit_count", "ambiguous_count",
        "unresolved_count", "unknown_ancestry_count", "unknown_history_count", "primary_no_recorded_exit_count",
        "primary_prior_non_geometry_count", "source_states_declared", "source_missing_events", "source_size_mismatch_events")}
    diagnostics.update(primary_birth_count=4, primary_uncollided_exit_count=3, unresolved_count=1,
                       unknown_history_count=1, primary_no_recorded_exit_count=1, input_events=4,
                       expected_events=4, geometry_process_ids=[48, 49, 182])
    reference["report_json"] = np.asarray(json.dumps(diagnostics))
    reference["spatial_report_json"] = np.asarray(json.dumps(dict(boundary_tolerance_mm=1e-7)))
    return fast, report, source, photons, [reference]


def test_absolute_spectra_zeros_unsupported_unknown_and_tail_ids(case):
    out = compare(*case)
    fast, _, _, _, references = case
    assert out["input_events"] == 4
    assert out["reference_comparison"]["fast_loopers"] == 1
    for metric, saved in (("photons", "uncollided_photons"), ("energy_mev", "uncollided_energy_mev")):
        assert sum(out["spectra"][metric]["fast_total"]) == pytest.approx(fast[saved].sum())
    tail = out["tails"]["energy_mev"]
    assert tail["fast_total"][0] == pytest.approx(278.0005)
    assert tail["g4_total"][0] == 111.
    assert tail["both_zero_events"][0] == 2  # absent source event AND present zero-output event
    assert tail["fast_total"][-1] == 275.
    assert tail["g4_total"][-1] == 100.
    ranked = tail["ranked_events"]["absolute_difference"]
    assert ranked["event_ids"][0][0] == [1, 3, 12]
    assert ranked["fast_looper_event"][0][0] is True
    assert ranked["event_ids"][0][2] == [None, None, None]
    assert ranked["top_k_score_fraction"][0][-1] == 1.
    assert ranked["top_k_signed_difference"][0][0] == 167.5
    assert out["fast_unmodelled"]["below_1kev_foil_births"]["rows"] == 1
    assert out["fast_unmodelled"]["below_1kev_foil_births"]["known_energy_mev"] == .0005
    assert out["g4_unknown"]["primary_no_recorded_exit_rows"]["rows"] == 1
    assert out["g4_unknown"]["primary_no_recorded_exit_rows"]["known_energy_mev"] == .0008
    assert out["provenance"]["g4_shards"][0]["spatial"]["boundary_tolerance_mm"] == 1e-7
    json.dumps(out, allow_nan=False)


def test_bins_include_every_boundary_underflow_and_open_overflow():
    lower = ENERGY_LOWER_EDGES_MEV
    energy = np.r_[lower[1:], np.nextafter(lower[1:], 0), 1e8]
    h = event_histograms(np.zeros(len(energy), int), energy, np.ones(len(energy)), 3)
    expected = np.r_[1., np.full(len(lower) - 2, 2.), 2.]
    np.testing.assert_array_equal(h["photons"][0], expected)
    assert h["photons"].sum() == len(energy)
    assert h["energy_mev"].sum() == pytest.approx(energy.sum())
    assert not h["photons"][1:].any()


def test_paired_variance_is_event_not_photon_and_keeps_zero_rows(case):
    out = compare(*case)
    fast, _, _, _, refs = case
    difference = fast["uncollided_energy_mev"] - refs[0]["evt_primary_energy"]
    expected = np.sqrt(4 * np.var(difference, ddof=1))
    assert out["tails"]["energy_mev"]["difference_total_event_se"][0] == pytest.approx(expected)
    # A perfectly common paired contribution cancels, unlike independent-arm SE.
    h = np.array([[0., 1.], [1., 2.], [3., 5.], [0., 0.]])
    paired = paired_columns(h, h)
    assert paired["difference_total_event_se"] == [0., 0.]
    assert paired["fast_total_event_se"][0] > 0
    assert paired_columns(h[:1], h[:1])["difference_total_event_se"] == [None, None]


def split_reference(reference):
    pieces = []
    for event_slice, photon_slice in ((slice(0, 2), slice(0, 1)), (slice(2, 4), slice(1, 4))):
        part = {}
        for name, value in reference.items():
            if name.endswith("report_json"):
                part[name] = value.copy()
            else:
                part[name] = value[event_slice if name.startswith(("event_", "evt_")) else photon_slice].copy()
        report = json.loads(str(part["report_json"]))
        report.update(expected_events=2, input_events=2)
        part["report_json"] = np.asarray(json.dumps(report))
        pieces.append(part)
    return pieces


def test_disjoint_shards_reordering_and_full_uint64(case):
    fast, report, source, photons, refs = case
    bump = np.uint64(2**63 + 17)
    fast["event"] += bump
    for name in ("source_run", "source_subrun", "source_event", "source_particle_key"):
        source[name] += bump
    for name in ("event_run", "event_subrun", "event_event", "run", "subrun", "event", "product", "key"):
        refs[0][name] += bump
    pieces = split_reference(refs[0])
    out = compare(fast, report, source, photons, pieces[::-1])
    expected_id = fast["event"][2].tolist()
    assert out["tails"]["energy_mev"]["ranked_events"]["fast"]["event_ids"][0][0] == expected_id
    assert out["reference_comparison"]["metrics"]["photons"]["g4_total"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        compare(fast, report, source, photons, [pieces[0], pieces[0]])


@pytest.mark.parametrize("change,match", [
    ("float_event", "integer"), ("duplicate_event", "Duplicate"), ("mismatched_event", "universes"),
    ("duplicate_photon", "Duplicate"), ("orphan_photon", "absent"),
    ("duplicate_source", "Duplicate"), ("orphan_source", "absent"), ("float_source", "dtype"),
    ("bad_parent", "parent"), ("fast_totals", "reconstruction"), ("g4_totals", "reconstruction"),
    ("capped", "truncated"), ("saved_overflow", "truncated"), ("bad_selection", "gates"),
    ("support", "support"), ("unsupported_weight", "support"), ("incomplete_shard", "expected-event"),
])
def test_rejects_broken_keys_totals_or_gates(case, change, match):
    fast, report, source, photons, refs = case
    g = refs[0]
    if change == "float_event":
        fast["event"] = fast["event"].astype(float)
    elif change == "duplicate_event":
        fast["event"][0] = fast["event"][1]
    elif change == "mismatched_event":
        g["event_event"][0] = 99
    elif change == "duplicate_photon":
        g["product"][2] = g["product"][1]
    elif change == "orphan_photon":
        g["event"][0] = 99
    elif change == "duplicate_source":
        for name in ("source_run", "source_subrun", "source_event", "source_particle_key"):
            source[name][1] = source[name][0]
    elif change == "orphan_source":
        source["source_event"][0] = 99
    elif change == "float_source":
        source["source_particle_key"] = source["source_particle_key"].astype(float)
    elif change == "bad_parent":
        photons["parent"][0] = 3
    elif change == "fast_totals":
        fast["uncollided_energy_mev"][0] = 1.
    elif change == "g4_totals":
        g["evt_primary_energy"][0] = 1.
    elif change == "capped":
        report["capped_parents"] = 1
    elif change == "saved_overflow":
        source["overflow"][0] = 1
    elif change == "bad_selection":
        g["uncollided_exit"][-1] = True
    elif change == "support":
        photons["attenuation_supported"][-1] = True
    elif change == "unsupported_weight":
        photons["weight"][-1] = .1
    elif change == "incomplete_shard":
        r = json.loads(str(g["report_json"]))
        r["expected_events"] = 5
        g["report_json"] = np.asarray(json.dumps(r))
    with pytest.raises(ValueError, match=match):
        compare(*case)


def test_empty_photon_populations_are_explicit_zeros(case):
    fast, _, _, photons, refs = case
    for k in fast:
        if k != "event":
            fast[k][:] = 0
    for k in photons:
        photons[k] = photons[k][:0]
    g = refs[0]
    for k in g:
        if k.startswith("evt_"):
            g[k][:] = 0
        elif not k.startswith("event_") and not k.endswith("report_json"):
            g[k] = g[k][:0]
    out = compare(*case)
    assert out["tails"]["photons"]["both_zero_events"] == [4] * 7
    ranking = out["tails"]["photons"]["ranked_events"]["fast"]
    assert ranking["event_ids"][0][0] == [None, None, None]
    assert ranking["top_k_score_fraction"][0] == [None] * 4
    json.dumps(out, allow_nan=False)


def test_cli_round_trip_and_small_json(case, tmp_path):
    fast, report, source, photons, refs = case
    directory = tmp_path / "fast"
    directory.mkdir()
    np.savez(directory / "event_totals.npz", **fast)
    np.savez(directory / "births.npz", **source, ignored_large_field=np.ones((3, 100)))
    np.savez(directory / "foil_exit.npz", **photons)
    (directory / "flux.json").write_text(json.dumps(report))
    path = tmp_path / "g4.npz"
    np.savez(path, **refs[0])
    output = tmp_path / "result.json"
    assert main(["--fast", str(directory), "--g4", str(path), "--out", str(output), "--top-events", "2"]) == 0
    saved = json.loads(output.read_text())
    assert saved["provenance"]["inputs"]["g4"] == [str(path)]
    assert saved["tails"]["energy_mev"]["g4_total"][0] == 111.
    assert output.stat().st_size < 200_000


def test_saved_inputs_not_mutated_and_top_limit(case):
    before = copy.deepcopy(case)
    compare(*case)
    np.testing.assert_array_equal(case[0]["event"], before[0]["event"])
    np.testing.assert_array_equal(case[4][0]["unresolved"], before[4][0]["unresolved"])
    with pytest.raises(ValueError, match="top_events"):
        compare(*case, top_events=0)


def test_event_count_and_energy_tails_are_diagnostics_on_full_ledger(case):
    out = compare(*case)
    energy = out["event_tails"]["energy_mev"]
    assert energy["event_total_thresholds"] == [1., 10., 30., 100.]
    assert energy["events_above"]["fast_total"] == [1., 1., 1., 1.]
    assert energy["events_above"]["g4_total"] == [2., 1., 1., 1.]
    assert energy["events_above"]["both_zero_events"] == [2, 3, 3, 3]
    assert energy["ranked_events"]["fast"]["event_ids"][0][0] == [1, 3, 12]
    assert energy["contribution_above"]["difference_total"][-1] == 167.5
    assert out["event_tails"]["photons"]["events_above"]["fast_total"] == [1., 0., 0., 0.]
    # No tail selection was applied to the actual inclusive reference/spectrum.
    assert out["reference_comparison"]["input_events"] == 4
    assert sum(out["spectra"]["energy_mev"]["fast_total"]) == pytest.approx(278.0005)


def test_other_unsupported_attenuation_kept_distinct(case):
    case[3]["energy"][-1] = .00155955  # explicitly unsupported open K-edge gap
    case[0]["transparent_energy_mev"][2] += .00155955 - .0005
    out = compare(*case)
    assert out["fast_unmodelled"]["below_1kev_foil_births"]["rows"] == 0
    assert out["fast_unmodelled"]["other_unsupported_foil_births"]["rows"] == 1


def test_shard_photons_must_belong_to_own_ledger(case):
    fast, report, source, photons, refs = case
    shards = split_reference(refs[0])
    # Globally valid event ID but wrong shard: reject before totals can mask it.
    shards[0]["subrun"][0] = 3
    shards[0]["event"][0] = 12
    with pytest.raises(ValueError, match="own shard"):
        compare(fast, report, source, photons, shards)


def test_ranking_signed_contribution_excludes_zero_score_padding(case):
    out = compare(*case, top_events=20)
    # At event-energy >=1 fast contributes only event 12; event 11 contributes
    # G4 only and must not contaminate fast top-k's signed difference via padding.
    ranked = out["event_tails"]["energy_mev"]["ranked_events"]["fast"]
    assert ranked["top_k_signed_difference"][0] == [167.5] * 4
    assert ranked["event_ids"][0][1] == [None, None, None]
    assert ranked["g4"][0][1] is None
    assert ranked["difference"][0][1] is None
