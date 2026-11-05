"""Synthetic identity contracts only; no native transport/rate validation."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim import tsreference as ref
from beamfastsim import tspair as pair


def _rows(tracks, pdgs=13, parents=0, parent_pdgs=0, codes=14, vd=0):
    n = len(tracks)
    a = {k: np.zeros(n, dtype=np.int64) for k in ref.INTEGER}
    a.update(run=np.full(n, 7, np.int64), subrun=np.full(n, 3, np.int64),
             evt=np.ones(n, np.int64), trk=np.asarray(tracks, np.int64))
    for k, values in (("pdg", pdgs), ("parent_id", parents),
                      ("parent_pdg", parent_pdgs), ("code", codes)):
        a[k] = np.broadcast_to(np.asarray(values, np.int64), (n,)).copy()
    a.update({k: np.zeros(n) for k in ref.BORN + ref.END})
    a.update({k: np.zeros((n, 8)) for k in ref.VD})
    if vd:
        a["isvd"][:, vd - 1] = 1
    return a


def _write(path, rows, update=False):
    types = {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype for k, v in rows.items()}
    with (uproot.update(path) if update else uproot.recreate(path)) as f:
        f.mktree("mutree/ntpart", types).extend(rows)
    return path


def _audit(tmp_path, incoming, outgoing):
    a, b = tmp_path / "stage1.root", tmp_path / "stage2.root"
    _write(a, incoming)
    _write(b, outgoing)
    return pair.audit([a], [b])


@pytest.mark.parametrize("pdg", ref.PDGS)
def test_new_id_primary_links_parent_not_same_track(tmp_path, pdg):
    report = _audit(tmp_path, _rows([10], pdg, vd=1),
                    _rows([10, 101], pdg, [0, 10], [0, pdg], [14, 56], vd=8))
    assert report["gate"]
    assert report["exit_mapping"]["root_index"] == [0, 0]
    assert report["exit_mapping"]["status"] == ["copied_stage1", "resolved_direct_same_particle"]
    assert report["counts"]["native_continued_tracks"][str(pdg)] == 1
    assert report["counts"]["resolved_exit_root_ids_total"] == 1
    assert report["root_exit_multiplicity"] == [1]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pion,muon", [(-211, 13), (211, -13)])
def test_true_pion_decay_then_new_id_muon_continuation(tmp_path, pion, muon):
    report = _audit(tmp_path, _rows([10], pion, vd=1),
                    _rows([10, 101, 102, 200], [pion, pion, muon, muon],
                          [0, 10, 101, 102], [0, pion, pion, muon], [14, 56, 14, 56], vd=8))
    assert report["gate"]
    assert report["exit_mapping"]["status"] == ["copied_stage1", "resolved_direct_same_particle",
                                                "resolved_decay_descendant", "resolved_decay_descendant"]
    assert report["root_exit_multiplicity"] == [3]
    assert report["counts"]["resolved_exit_root_ids"][str(pion)] == 1
    assert report["counts"]["resolved_exit_root_ids"][str(muon)] == 0
    assert report["counts"]["native_continued_tracks"][str(muon)] == 1


def test_primary_parent_root_does_not_require_copied_row(tmp_path):
    report = _audit(tmp_path, _rows([10], vd=1), _rows([101], parents=10, parent_pdgs=13, codes=56, vd=8))
    assert report["gate"] and report["exit_mapping"]["status"] == ["resolved_direct_same_particle"]


@pytest.mark.parametrize("code,parent", [(56, 99), (14, 10), (14, 0), (56, 0)])
def test_missing_links_and_exit_without_parent_are_not_matched(tmp_path, code, parent):
    report = _audit(tmp_path, _rows([10], vd=1),
                    _rows([101], parents=parent, parent_pdgs=13, codes=code, vd=8))
    assert not report["gate"]
    assert report["exit_mapping"]["root_index"] == [-1]
    assert report["exit_mapping"]["status"] == ["missing_link"]
    assert report["counts"]["pool_issues"]["missing_link"] == 1


@pytest.mark.parametrize("code", [0, -1, 99999, 183, 184, 185, 76])
def test_unknown_codes_are_unresolved_not_decay(tmp_path, code):
    report = _audit(tmp_path, _rows([10], vd=1),
                    _rows([10, 101], parents=[0, 10], parent_pdgs=[0, 13], codes=[14, code], vd=8))
    assert not report["gate"]
    assert report["exit_mapping"]["status"][1] == "unknown_code"


def test_missing_code_is_explicit(tmp_path):
    outgoing = _rows([101], parents=10, parent_pdgs=13, codes=56, vd=8)
    del outgoing["code"]
    report = _audit(tmp_path, _rows([10], vd=1), outgoing)
    assert not report["gate"] and report["exit_mapping"]["status"] == ["missing_code"]


@pytest.mark.parametrize("pdg,parent_pdg", [(-13, 13), (13, -13)])
def test_continuation_species_and_parent_pdg_crosschecks(tmp_path, pdg, parent_pdg):
    report = _audit(tmp_path, _rows([10], vd=1),
                    _rows([101], pdg, 10, parent_pdg, 56, vd=8))
    assert not report["gate"] and report["exit_mapping"]["status"] == ["pdg_conflict"]


def test_unphysical_decay_charge_is_conflict(tmp_path):
    report = _audit(tmp_path, _rows([10], -211, vd=1),
                    _rows([10, 101], [-211, -13], [0, 10], [0, -211], [14, 14], vd=8))
    assert not report["gate"] and report["exit_mapping"]["status"][1] == "pdg_conflict"


def test_known_nondecay_process_not_mislabeled_decay(tmp_path):
    report = _audit(tmp_path, _rows([10], vd=1),
                    _rows([10, 101], parents=[0, 10], parent_pdgs=[0, 13], codes=[14, 20], vd=8))
    assert report["gate"] and report["exit_mapping"]["status"][1] == "resolved_other_descendant"


@pytest.mark.parametrize("which", ["input", "output"])
def test_duplicate_native_roots_or_output_identity_rejected(tmp_path, which):
    a = _rows([10, 10], vd=1) if which == "input" else _rows([10], vd=1)
    b = _rows([101, 101], parents=10, parent_pdgs=13, codes=56, vd=8) if which == "output" else _rows([101])
    with pytest.raises(ValueError, match="duplicate particle identity"):
        _audit(tmp_path, a, b)


@pytest.mark.parametrize("which", ["input", "output"])
def test_crossfile_events_overlap_even_different_tracks_and_unselected_species(tmp_path, which):
    one = _write(tmp_path / "one.root", _rows([10], vd=1))
    two = _write(tmp_path / "two.root", _rows([99], pdgs=11, vd=1))
    with pytest.raises(ValueError, match="overlapping native event"):
        pair.audit([one, two] if which == "input" else [one],
                   [one, two] if which == "output" else [one])


def test_duplicate_files_and_hashes_rejected_within_cohort(tmp_path):
    path = _write(tmp_path / "a.root", _rows([10], vd=1))
    copy = tmp_path / "copy.root"
    copy.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate/byte-identical"):
        pair.audit([path, copy], [path])
    # Sharing a file between the two cohorts is permitted and copied, not new.
    report = pair.audit([path], [path], exit_vd=1)
    assert report["counts"]["copied_stage1_exit_hits"] == 1
    assert report["counts"]["resolved_exit_tracks"] == 0


def test_disjoint_event_shards_can_be_repartitioned_across_files(tmp_path):
    a, b = _rows([10], vd=1), _rows([10], vd=1)
    b["evt"][:] = 2
    first = _write(tmp_path / "one.root", a)
    second = _write(tmp_path / "two.root", b)
    output = _rows([101, 101], parents=10, parent_pdgs=13, codes=56, vd=8)
    output["evt"][:] = [2, 1]
    end = _write(tmp_path / "end.root", output)
    report = pair.audit([first, second], [end])
    assert report["gate"] and report["exit_mapping"]["root_index"] == [1, 0]
    assert report["counts"]["resolved_exit_root_ids_total"] == 2


def test_int64_exact_and_complete_event_key(tmp_path):
    incoming = _rows([2**60 + 1, 2**60 + 2], vd=1)
    incoming["run"][:] = [2**60 + 1, 2**60 + 2]
    incoming["evt"][:] = 2**60 + 3
    outgoing = _rows([2**60 + 5, 2**60 + 6], parents=[2**60 + 2, 2**60 + 1], parent_pdgs=13, codes=56, vd=8)
    outgoing["run"][:] = [2**60 + 2, 2**60 + 1]
    outgoing["evt"][:] = 2**60 + 3
    report = _audit(tmp_path, incoming, outgoing)
    assert report["gate"] and report["exit_mapping"]["root_index"] == [1, 0]
    outgoing["subrun"][:] = 9
    report = _audit(tmp_path, incoming, outgoing)
    assert not report["gate"] and report["exit_mapping"]["root_index"] == [-1, -1]


@pytest.mark.parametrize("parents", [[101], [102, 101]])
def test_cycles_fail_gate_without_per_particle_walk(tmp_path, parents):
    tracks = [101] if len(parents) == 1 else [101, 102]
    report = _audit(tmp_path, _rows([10], vd=1),
                    _rows(tracks, parents=parents, parent_pdgs=13, codes=56, vd=8))
    assert not report["gate"]
    assert set(report["exit_mapping"]["status"]) == {"cycle"}
    assert report["counts"]["pool_issues"]["cycle"] == len(parents)


def test_depth_limit_is_unresolved(tmp_path):
    tracks = np.arange(100, 231)
    outgoing = _rows(tracks, parents=np.r_[10, tracks[:-1]], parent_pdgs=13, codes=56)
    outgoing["isvd"][-1, 7] = 1
    report = _audit(tmp_path, _rows([10], vd=1), outgoing)
    assert not report["gate"] and report["exit_mapping"]["status"] == ["depth_limit"]


def test_current_stage_hits_not_proximity_or_same_track(tmp_path):
    incoming = _rows([10, 11], vd=1)
    # Every stored time/position is identical, but only an explicit parent links.
    outgoing = _rows([10, 101, 102], parents=[0, 11, 999], parent_pdgs=[0, 13, 13], codes=[14, 56, 56], vd=8)
    report = _audit(tmp_path, incoming, outgoing)
    assert report["exit_mapping"]["root_index"] == [0, 1, -1]
    assert report["root_exit_multiplicity"] == [0, 1]
    assert report["counts"]["resolved_exit_root_ids_total"] == 1


def test_code56_same_track_never_automatically_resolves(tmp_path):
    report = _audit(tmp_path, _rows([10], codes=56, vd=1),
                    _rows([10, 101], parents=[0, 10], parent_pdgs=[0, 13], codes=56, vd=8))
    assert not report["gate"]
    assert report["exit_mapping"]["status"] == ["identity_conflict", "resolved_direct_same_particle"]
    assert report["counts"]["native_continued_tracks"]["13"] == 1


def test_reusing_root_id_with_new_primary_parent_is_conflict(tmp_path):
    report = _audit(tmp_path, _rows([10], vd=1), _rows([10], parents=10, parent_pdgs=13, codes=56, vd=8))
    assert not report["gate"] and report["exit_mapping"]["status"] == ["identity_conflict"]


def test_highest_cycle_and_cli_receipts_and_alias_guards(tmp_path):
    a = _write(tmp_path / "a.root", _rows([10], vd=1))
    b = _write(tmp_path / "b.root", _rows([101], parents=999, parent_pdgs=13, codes=56, vd=8))
    _write(b, _rows([101], parents=10, parent_pdgs=13, codes=56, vd=8), update=True)
    dest = tmp_path / "report.JSON"
    assert pair.main(["--input", str(a), "--output", str(b), "--out", str(dest)]) == 0
    report = json.loads(dest.read_text())
    assert report["output"]["files"][0]["cycle"] == 2
    alias = tmp_path / "alias.json"
    alias.symlink_to(a)
    with pytest.raises(SystemExit):
        pair.main(["--input", str(a), "--output", str(b), "--out", str(alias)])
    with pytest.raises(SystemExit):
        pair.main(["--input", str(a), "--output", str(b), "--out", str(tmp_path / "bad.root")])


def test_cli_unresolved_writes_report_and_returns_two(tmp_path):
    a = _write(tmp_path / "a.root", _rows([10], vd=1))
    b = _write(tmp_path / "b.root", _rows([101], parents=999, parent_pdgs=13, codes=56, vd=8))
    dest = tmp_path / "out.json"
    assert pair.main(["--input", str(a), "--output", str(b), "--out", str(dest)]) == 2
    assert not json.loads(dest.read_text())["gate"]
