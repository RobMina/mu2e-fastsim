"""Same-stage references retain recrossings, branches, unrelated hits and empty events."""
import numpy as np
import pytest

from beamfastsim import tsrawreference as ts
from test_tssmoke import particles, hits


def test_same_stage_decay_and_loss_no_restart():
    a, report = ts.build(particles(), hits([2, 4], [-211, -13], [10, 9]), hits([3], [13], [50]))
    assert report["aligned_reference_available"]
    assert a["stock_reached"].tolist() == [True, False]
    assert a["stock_pdg"].tolist() == [13, 0]
    assert a["stock_id"][0].tolist() == [1, 1, 1, 3]
    assert np.isnan(a["stock_t"][1])
    assert a["parent_shard_event_counts"].tolist() == [2]
    assert len(a["event_id"]) == 2


def test_recrossing_first_time_and_exact_key_mapping():
    a, report = ts.build(particles(), hits([4, 2, 2], [-13, -211, -211], [9, 20, 10]), hits([3], [13], [50]))
    assert a["t"].tolist() == [10, 9]
    assert a["crossing_source"].tolist() == [1, 0, 0]
    assert a["crossing_t"].tolist() == [9, 20, 10]
    assert report["tracks_with_repeated_forward_records"] == 1


def test_multiple_ds_rows_not_silently_collapsed():
    a, report = ts.build(particles(), hits([2], [-211], [10]), hits([3, 3], [13, 13], [50, 60]))
    assert not report["aligned_reference_available"]
    assert "stock_reached" not in a
    assert a["endpoint_source"].tolist() == [0, 0]
    assert a["stock_multiplicity"].tolist() == [2]


def test_nested_sources_are_explicit():
    a, report = ts.build(particles(), hits([2, 3], [-211, 13], [10, 30]), hits([3], [13], [50]))
    assert report["nested_source_tracks"] == 1
    assert not report["aligned_reference_available"]
    assert a["source_parent_source"].tolist() == [-1, 0]
    assert a["endpoint_source"].tolist() == [1]


def test_unresolved_endpoint_does_not_create_false_loss():
    p = particles()
    p["parent_id"][2] = 99
    p["parent_pdg"][2] = 2112
    a, report = ts.build(p, hits([2], [-211], [10]), hits([3], [13], [50]))
    assert a["endpoint_ancestry_incomplete"].tolist() == [True]
    assert not report["aligned_reference_available"]
    assert "stock_reached" not in a
    assert report["source_tracks_without_ds_mu_pi_descendant"] is None


def test_missing_upstream_parent_before_source_and_unrelated_endpoint():
    p = particles()
    p["parent_id"][1] = 99
    p["parent_pdg"][1] = 2112
    a, report = ts.build(p, hits([2], [-211], [10]), hits([3, 4], [13, -13], [50, 60]))
    assert report["aligned_reference_available"]
    assert a["endpoint_source"].tolist() == [0, -1]
    assert a["source_parent_ancestry_incomplete"].tolist() == [True]
    assert a["stock_reached"].tolist() == [True]


def test_empty_cohort_keeps_events():
    a, report = ts.build(particles(), hits([], [], []), hits([], [], []))
    assert report["aligned_reference_available"]
    assert a["x"].shape == (0, 3)
    assert a["stock_id"].shape == (0, 4)
    assert a["parent_shard_event_counts"].tolist() == [2]


def test_nonfinite_state_rejected():
    vd = hits([2], [-211], [10])
    vd["px"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ts.build(particles(), vd, hits([], [], []))


def test_combine_offsets_all_foreign_keys_and_rejects_duplicate_events(tmp_path):
    import json
    paths = []
    for index in (0, 1):
        p, vd, beam = particles(), hits([2], [-211], [10]), hits([3], [13], [50])
        for table in (p, vd, beam):
            table["subrun"][:] = 920000 + index
        a, report = ts.build(p, vd, beam)
        path = tmp_path / f"reference-{index}.npz"
        ts.save(a, path)
        path.with_suffix(".json").write_text(json.dumps(report))
        paths.append(path)
    a, report = ts.combine(paths)
    assert report["events"] == 4
    assert a["endpoint_source"].tolist() == [0, 1]
    assert a["crossing_source"].tolist() == [0, 1]
    assert a["source_parent_source"].tolist() == [-1, -1]
    with pytest.raises(ValueError, match="duplicate"):
        ts.combine([paths[0], paths[0]])


def test_heldout_merge_requires_explicit_cohort_and_rejects_mixing(tmp_path):
    import json
    paths = []
    for index in (99, 100, 199):
        p, vd, beam = particles(), hits([2], [-211], [10]), hits([3], [13], [50])
        for table in (p, vd, beam):
            table['subrun'][:] = 920000 + index
        a, report = ts.build(p, vd, beam)
        path = tmp_path / f'reference-{index}.npz'
        ts.save(a, path)
        path.with_suffix('.json').write_text(json.dumps(report))
        paths.append(path)
    with pytest.raises(ValueError, match='development'):
        ts.combine(paths[1:], root_cohort=True)
    a, report = ts.combine(paths[1:], root_cohort=True, cohort='heldout')
    assert report['cohort'] == 'heldout'
    assert report['events'] == 4
    assert a['shard'].tolist() == [920100, 920199]
    with pytest.raises(ValueError, match='heldout'):
        ts.combine(paths, root_cohort=True, cohort='heldout')
    with pytest.raises(ValueError, match='cohort'):
        ts.combine(paths[1:], cohort='all')


def test_explicit_event_inventory_retains_particle_free_event():
    a, _ = ts.build(particles(), hits([], [], []), hits([], [], []),
                    event_id=np.array([[1, 1, 1], [1, 1, 2], [1, 1, 3]], dtype="i8"))
    assert a["parent_shard_event_counts"].tolist() == [3]
    with pytest.raises(ValueError, match="cover observed"):
        ts.build(particles(), hits([], [], []), hits([], [], []), event_id=np.array([[1, 1, 1]]))


def test_root_lineage_remaps_descendant_endpoint_and_preserves_empty_events():
    a, _ = ts.build(particles(), hits([2, 3, 4], [-211, 13, -13], [10, 30, 12]), hits([3], [13], [50]))
    b = ts.root_lineages(a)
    assert b['id'][:, 3].tolist() == [2, 4]
    assert b['pdg'].tolist() == [-211, -13]
    assert b['stock_reached'].tolist() == [True, False]
    assert b['stock_id'][0, 3] == 3
    assert b['stock_t'][0] == 50 and b['t'][0] == 10
    assert b['crossing_source'].tolist() == [0, 0, 1]
    assert b['nested_id'][:, 3].tolist() == [3]
    assert b['nested_root_id'][:, 3].tolist() == [2]
    assert b['parent_shard_event_counts'].tolist() == [2]
    assert a['id'][:, 3].tolist() == [2, 3, 4]


def test_root_lineage_keeps_root_even_when_every_nested_track_is_lost():
    a, _ = ts.build(particles(), hits([2, 3], [-211, 13], [10, 30]), hits([], [], []))
    b = ts.root_lineages(a)
    assert b['id'][:, 3].tolist() == [2]
    assert b['stock_reached'].tolist() == [False]
    assert len(b['event_id']) == 2


def test_root_lineage_does_not_merge_unrelated_species_or_events():
    a, _ = ts.build(particles(), hits([2], [-211], [10]), hits([4], [-13], [50]))
    b = ts.root_lineages(a)
    assert b['endpoint_source'].tolist() == [-1]
    assert b['stock_reached'].tolist() == [False]


@pytest.mark.parametrize('bad', ['cycle', 'time', 'event', 'range'])
def test_root_lineage_invalid_parent_rejected(bad):
    a, _ = ts.build(particles(), hits([2, 3], [-211, 13], [10, 30]), hits([], [], []))
    if bad == 'cycle':
        a['source_parent_source'][:] = [1, 0]
        a['t'][:] = 10
    elif bad == 'time':
        a['t'][1] = 9
    elif bad == 'event':
        a['id'][1, 2] += 1
    else:
        a['source_parent_source'][1] = 99
    with pytest.raises(ValueError):
        ts.root_lineages(a)


def test_root_lineage_multiple_endpoints_and_incomplete_ancestry_rejected():
    a, _ = ts.build(particles(), hits([2, 3], [-211, 13], [10, 30]), hits([2, 3], [-211, 13], [50, 60]))
    with pytest.raises(ValueError, match='multiple DS'):
        ts.root_lineages(a)
    a['endpoint_ancestry_incomplete'][0] = True
    with pytest.raises(ValueError, match='unresolved DS'):
        ts.root_lineages(a)


def test_empty_root_lineage_projection():
    a, _ = ts.build(particles(), hits([], [], []), hits([], [], []))
    b = ts.root_lineages(a)
    assert b['id'].shape == (0, 4)
    assert b['nested_root_id'].shape == (0, 4)
    assert b['parent_shard_event_counts'].tolist() == [2]


def test_combine_root_lineages_keeps_exact_nested_ids(tmp_path):
    import json
    paths = []
    for index in (0, 1):
        p, vd, beam = particles(), hits([2, 3], [-211, 13], [10, 30]), hits([3], [13], [50])
        for table in (p, vd, beam):
            table['subrun'][:] = 920000 + index
        a, report = ts.build(p, vd, beam)
        path = tmp_path / f'reference-{index}.npz'
        ts.save(a, path)
        path.with_suffix('.json').write_text(json.dumps(report))
        paths.append(path)
    a, report = ts.combine(paths, root_cohort=True)
    assert report['recorded_source_tracks'] == 4
    assert report['source_tracks'] == report['nested_tracks_represented_by_ancestor'] == 2
    assert a['endpoint_source'].tolist() == [0, 1]
    assert a['crossing_source'].tolist() == [0, 0, 1, 1]
    assert a['nested_root_id'][:, 1].tolist() == [920000, 920001]
    assert len(a['event_id']) == 4
