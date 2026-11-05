"""Synthetic flat ROOT fixtures exercise retained ancestry, not inferred primaries."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim import flash_ancestry as fa


def particles(keys=(), parents=(), pdgs=(), **columns):
    n = len(keys)
    table = {field: np.zeros(n, dtype=float) for field in fa.PARTICLE}
    table.update(run=np.ones(n, dtype=int), subrun=np.ones(n, dtype=int), event=np.ones(n, dtype=int),
                 key=np.asarray(keys, dtype=int), parent=np.asarray(parents, dtype=int),
                 pdg=np.asarray(pdgs, dtype=int), ccode=np.full(n, 7, dtype=int), genidx=np.full(n, -1, dtype=int))
    table.update({name: np.asarray(value) for name, value in columns.items()})
    return table


def steps(keys=(), **columns):
    n = len(keys)
    table = {field: np.zeros(n) for field in fa.STEP}
    table.update(run=np.ones(n, dtype=int), subrun=np.ones(n, dtype=int), event=np.ones(n, dtype=int),
                 sp=np.asarray(keys, dtype=int), eion=np.ones(n))
    table.update({name: np.asarray(value) for name, value in columns.items()})
    return table


def generators(indices=(), **columns):
    n = len(indices)
    table = {field: np.ones(n, dtype=int) for field in fa.GEN}
    table.update(idx=np.asarray(indices, dtype=int), pdg=np.full(n, 13, dtype=int))
    table.update({name: np.asarray(value) for name, value in columns.items()})
    return table


def write_root(path, s, p, g=None):
    if g is None:
        g = generators()
    with uproot.recreate(path) as root:
        for name, table in (("sgsteps", s), ("particles", p), ("gen", g)):
            root.mktree(f"dump/{name}", {key: value.dtype for key, value in table.items()}).extend(table)


def test_nearest_strict_photon_and_direct_mother():
    p = particles([1, 2, 3, 4, 5], [-1, 1, 2, 3, 4], [13, 22, 11, 22, 11],
                  sx=[0, 3, 0, 6, 0], sy=[0, 4, 0, 8, 0], sz=[0, 10, 0, 20, 0])
    s = steps([3, 5, 4])
    a = fa.trace_steps(s, p)
    assert a["photon_key"].tolist() == [2, 4, 2]
    assert a["root_key"].tolist() == [1, 1, 1]
    assert a["direct_electron_gamma_mother"].tolist() == [True, True, False]
    report = fa.summarize(s, a, [0, 15, 25], [0, 7, 11])
    assert report["photon_birth"]["step_count"] == [[2, 0], [0, 1]]
    assert report["eion_sum_mev"] == 3


@pytest.mark.parametrize("field", fa.EVENT)
def test_full_identity_prevents_cross_event_or_run_joins(field):
    p = particles([1, 2, 1, 2], [-1, 1, -1, 1], [22, 11, 13, 11], **{field: [1, 1, 2, 2]})
    a = fa.trace_steps(steps([2, 2, 2], **{field: [1, 2, 3]}), p)
    assert a["photon_status"].tolist() == ["found", "no_retained_photon", "missing_particle"]
    assert a["root_index"].tolist() == [0, 2, -1]


def test_missing_parent_and_step_id_do_not_select_last_particle():
    p = particles([2, 99], [99, -1], [11, 22], event=[1, 2])
    s = steps([2, -1, 42])
    a = fa.trace_steps(s, p)
    assert a["root_status"].tolist() == ["missing_parent", "missing_particle", "missing_particle"]
    assert a["photon_index"].tolist() == [-1, -1, -1]
    assert a["root_key"].tolist() == [-1, -1, -1]
    assert np.isnan(a["photon_sz"]).all()
    report = fa.summarize(s, a, [0, 1], [0, 1])
    assert report["photon_birth"]["no_photon"]["eion_sum_mev"] == 3


def test_compressed_photon_path_and_unresolved_root_are_separate():
    p = particles([1, 2, 3], [2, 3, 999], [11, 11, 22])
    a = fa.trace_steps(steps([1]), p)
    assert a["photon_status"].tolist() == ["found"]
    assert a["photon_key"].tolist() == [3]
    assert a["root_status"].tolist() == ["missing_parent"]
    assert a["root_key"].tolist() == [-1]
    assert not a["direct_electron_gamma_mother"][0]
    report = fa.summarize(steps([1]), a, [-1, 1], [0, 1])
    assert report["retained_roots"]["key"] == []
    assert report["retained_roots"]["step_count"] == []


def test_cycles_self_loops_and_depth_limit():
    p = particles([1, 2, 3, 4, 5], [2, 1, 3, 5, -1], [11, 13, 11, 11, 13])
    a = fa.trace_steps(steps([1, 2, 3, 4]), p, max_depth=8)
    assert a["root_status"].tolist() == ["cycle", "cycle", "cycle", "root_reached"]
    assert a["photon_status"].tolist() == ["cycle", "cycle", "cycle", "no_retained_photon"]
    assert (a["root_index"][:3] == -1).all()
    a = fa.trace_steps(steps([4, 5]), p, max_depth=0)
    assert a["root_status"].tolist() == ["depth_limit", "root_reached"]


def test_cycle_with_retained_photon_does_not_claim_root():
    a = fa.trace_steps(steps([1]), particles([1, 2], [2, 1], [11, 22]), max_depth=8)
    assert a["photon_key"].tolist() == [2]
    assert a["root_key"].tolist() == [-1]
    assert a["root_status"].tolist() == ["cycle"]


def test_retained_root_not_original_primary_and_generator_join_scoped(tmp_path):
    path = tmp_path / "sample.root"
    p = particles([9, 1], [-1, 9], [22, 11], genidx=[0, -1])
    s = steps([1])
    a = fa.trace_steps(s, p, generators([0], event=[2]))
    assert a["root_key"].tolist() == [9]
    assert a["root_gen_index"].tolist() == [-1]
    a = fa.trace_steps(s, p, generators([0]))
    assert a["root_gen_pdg"].tolist() == [13]
    write_root(path, s, p, generators([0]))
    report = fa.analyze([path], z_bins=[-1, 1], r_bins=[0, 1])
    assert "not the original primary" in report["interpretation"]
    assert report["normalization"]["certified_generated_events"] is None
    assert report["normalization"]["eion_mev_per_generated_event"] is None
    assert report["counts"]["represented_step_events"] == 1
    roots = report["files"][0]["summary"]["retained_roots"]
    assert roots["key"] == [9]
    assert roots["run"] == roots["subrun"] == roots["event"] == [1]
    assert roots["linked_gen_idx"] == [0]
    assert roots["step_count"] == roots["eion_sum_mev"] == [1]
    certified = fa.analyze([path], z_bins=[-1, 1], r_bins=[0, 1], certified_generated_events=10)
    assert certified["normalization"]["eion_mev_per_generated_event"] == .1


def test_duplicate_particle_keys_rejected_even_without_steps():
    p = particles([1, 1], [-1, -1], [11, 22])
    with pytest.raises(ValueError, match="Duplicate"):
        fa.trace_steps(steps(), p)


def test_empty_particles_and_missing_ids():
    a = fa.trace_steps(steps([1, -1]), particles())
    assert a["photon_status"].tolist() == ["missing_particle", "missing_particle"]
    assert (a["root_key"] == -1).all()
    with pytest.raises(ValueError, match="identity"):
        fa.trace_steps(steps([1], event=[np.nan]), particles())
    with pytest.raises(KeyError):
        fa.trace_steps({k: v for k, v in steps([1]).items() if k != "run"}, particles())


def test_zero_entries_cli_highest_cycle_and_hashes(tmp_path):
    path, out = tmp_path / "empty.root", tmp_path / "report.json"
    write_root(path, steps([1]), particles([1], [-1], [11]))
    with uproot.update(path) as root:
        for name, table in (("sgsteps", steps()), ("particles", particles()), ("gen", generators())):
            root.mktree(f"dump/{name}", {key: value.dtype for key, value in table.items()}).extend(table)
    artifact = tmp_path / "config.txt"
    artifact.write_text("fixture\n")
    assert fa.main([str(path), "--out", str(out), "--z-bins", "0", "10", "--r-bins", "0", "10",
                    "--artifact", str(artifact)]) == 0
    report = json.loads(out.read_text())
    assert report["counts"] == dict(step_rows=0, particle_rows=0, gen_rows=0, represented_step_events=0,
                                    represented_gen_events=0, represented_any_tree_events=0)
    assert report["eion_sum_mev"] == 0
    assert report["normalization"]["certified_generated_events"] is None
    assert report["files"][0]["trees"]["sgsteps"]["selected_key"] == "dump/sgsteps;2"
    assert report["files"][0]["trees"]["particles"]["cycles"] == {"1": 1, "2": 0}
    assert len(report["files"][0]["sha256"]) == len(report["artifacts"][0]["sha256"]) == 64
    assert report["files"][0]["summary"]["photon_birth"]["step_count"] == [[0]]


def test_bins_account_for_outliers_and_nonfinite_birth():
    p = particles([1, 2, 3, 4], [-1, 1, -1, 3], [22, 11, 22, 11], sz=[99, 0, np.nan, 0])
    s = steps([2, 4], eion=[2, 3])
    report = fa.summarize(s, fa.trace_steps(s, p), [0, 1], [0, 1])
    assert report["photon_birth"]["outside_bins"]["eion_sum_mev"] == 2
    assert report["photon_birth"]["nonfinite_birth"]["eion_sum_mev"] == 3
    with pytest.raises(ValueError, match="bins"):
        fa.summarize(s, fa.trace_steps(s, p), [0, 0], [0, 1])


def test_disjoint_files_and_reject_overlapping_event_ids(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    write_root(a, steps([1]), particles([1], [-1], [11]))
    write_root(b, steps([1], event=[2]), particles([1], [-1], [11], event=[2]))
    report = fa.analyze([a, b], z_bins=[0, 1], r_bins=[0, 1])
    assert report["counts"]["represented_step_events"] == 2
    assert report["eion_sum_mev"] == 2
    with pytest.raises(ValueError, match="Overlapping"):
        fa.analyze([a, a], z_bins=[0, 1], r_bins=[0, 1])
    with pytest.raises(ValueError, match="smaller"):
        fa.analyze([a, b], z_bins=[0, 1], r_bins=[0, 1], certified_generated_events=1)


def test_no_input_and_invalid_controls():
    with pytest.raises(ValueError, match="At least one"):
        fa.analyze([], z_bins=[0, 1], r_bins=[0, 1])
    with pytest.raises(ValueError, match="max_depth"):
        fa.trace_steps(steps(), particles(), max_depth=-1)
    with pytest.raises(ValueError, match="Certified"):
        fa.analyze([], z_bins=[0, 1], r_bins=[0, 1], certified_generated_events=0)


def test_generator_rows_not_generated_event_count_and_final_bin_edges(tmp_path):
    path = tmp_path / "gen_only_events.root"
    p = particles([1, 2], [-1, 1], [22, 11], sx=[3, 0], sy=[4, 0], sz=[10, 0])
    s = steps([2], eion=[2.5])
    write_root(path, s, p, generators([0, 1, 0], event=[1, 1, 2]))
    report = fa.analyze([path], z_bins=[0, 10], r_bins=[0, 5])
    assert report["counts"]["gen_rows"] == 3
    assert report["counts"]["represented_gen_events"] == 2
    assert report["counts"]["represented_step_events"] == 1
    assert report["normalization"]["eion_mev_per_generated_event"] is None
    assert report["files"][0]["summary"]["photon_birth"]["eion_sum_mev"] == [[2.5]]


def test_invalid_missing_or_overflowing_event_id_rejected():
    p = particles([1], [-1], [11])
    for value in (-1, 1.5, np.inf, 2**63):
        with pytest.raises(ValueError, match="identity"):
            fa.trace_steps(steps([1], event=[value]), p)


def test_vectorized_many_events_reusing_keys():
    count = 20000
    event = np.repeat(np.arange(1, count + 1), 2)
    p = particles(np.tile([1, 2], count), np.tile([-1, 1], count), np.tile([22, 11], count), event=event)
    s = steps(np.full(count, 2), event=np.arange(1, count + 1))
    a = fa.trace_steps(s, p)
    np.testing.assert_array_equal(a["photon_index"], np.arange(0, count * 2, 2))
    np.testing.assert_array_equal(a["root_index"], a["photon_index"])
    assert np.all(a["direct_electron_gamma_mother"])
