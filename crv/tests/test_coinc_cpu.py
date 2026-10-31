"""Exact host attachment acceleration, including independent stock-order boundary cases."""
import copy

import numpy as np
import pytest

from crvfastsim.reco import coinc as C
from crvfastsim.reco.coinc_cpu import attach_opposite_side_pulses
import test_coinc_attachment_cpu as cases


@pytest.fixture(autouse=True)
def numpy_mode(monkeypatch):
    monkeypatch.setenv("CRVFS_CPU_ATTACHMENT", "numpy")


def test_all_independent_attachment_expectations(request):
    for item in (cases.test_attachment_snapshot_no_cascade_or_clustered_pulse_reuse,
                 cases.test_attachment_supports_both_missing_ends_and_empty_inputs):
        item()


@pytest.mark.parametrize("seed", range(12))
def test_random_native_snapshots_equal_reference(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    for count in (0, 1, 7, 50, 150):
        n = count + 10
        hits = [cases.hit(i, bar=int(rng.integers(0, 8)), sipm=int(rng.integers(0, 4)),
                          t=float(rng.integers(70, 141)), seq=int(rng.integers(0, 3))) for i in range(n)]
        order = rng.permutation(n)
        clusters = [[hits[i] for i in q] for q in np.array_split(order[:count], 5)]
        ref, candidate = copy.deepcopy(clusters), copy.deepcopy(clusters)
        monkeypatch.setenv("CRVFS_CPU_ATTACHMENT", "reference")
        C._attach_opposite_side_pulses(ref, hits, 20.)
        attach_opposite_side_pulses(candidate, hits, 20.)
        assert candidate == ref
        assert [[id(h) for h in cl[len(orig):]] for cl, orig in zip(candidate, clusters)] == [
            [id(h) for h in cl[len(orig):]] for cl, orig in zip(ref, clusters)]


@pytest.mark.parametrize("vectorized", [False, True])
def test_full_independent_finder_cases(vectorized):
    for fun in (cases.test_attachment_default_and_disabled_legacy,
                cases.test_side0_candidate_and_absent_coincidence_seeds,
                cases.test_filtered_candidates_only_and_inclusive_filter_boundary,
                cases.test_compensated_pe_reaches_attached_properties,
                cases.test_original_secondary_member_occupies_side,
                cases.test_attachment_rescues_min_cluster_pes,
                cases.test_candidate_tie_and_sorted_bar_sipm_append,
                cases.test_nearest_cluster_tie_and_loser_not_reassigned,
                cases.test_nearest_cluster_order_not_time_or_pulse_index,
                cases.test_same_physical_bar_not_counter_or_neighbor):
        fun(vectorized)
    for time, attached in ((np.nextafter(120., -np.inf), True), (120., False),
                           (np.nextafter(120., np.inf), False), (np.nextafter(80., np.inf), True),
                           (80., False), (np.nextafter(80., -np.inf), False)):
        cases.test_strict_attachment_window_both_boundaries(vectorized, time, attached)
    for candidate, anchor in ((True, False), (False, True), (True, True)):
        cases.test_secondary_candidate_and_anchor_excluded_with_ignore_false(vectorized, candidate, anchor)


def test_invalid_mode_refused(monkeypatch):
    monkeypatch.setenv("CRVFS_CPU_ATTACHMENT", "typo")
    with pytest.raises(ValueError, match="CRVFS_CPU_ATTACHMENT"):
        C._attach_opposite_side_pulses([], [], 20.)
