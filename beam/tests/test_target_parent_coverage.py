"""Exact development-source support contracts, not source-independence evidence."""
import json

import numpy as np
import pytest

from beamfastsim.target_parent_coverage import audit, integer_rows, main


def test_exact_source_subrun_coverage_and_catalog(tmp_path):
    parent = tmp_path / 'parent.npz'
    stops = tmp_path / 'window.npz'
    source = tmp_path / 'draw.npz'
    cat = tmp_path / 'catalog.json'
    rows = dict(run=np.ones(6, int), subrun=np.repeat([2, 3, 4], 2),
                evt=np.tile([1, 2], 3), trk=np.ones(6, int))
    np.savez(parent, **rows)
    selected = np.array([5, 0, 4, 2])
    np.savez(stops, **{k: v[selected] for k, v in rows.items()})
    np.savez(source, stop_idx=np.array([0, 2, 0, 1]))
    cat.write_text(json.dumps([dict(metadata={'rs.first_run': 1, 'rs.last_run': 1,
        'rs.first_subrun': s, 'rs.last_subrun': s}) for s in (2, 3, 4)]))
    r = audit(parent, [(source, stops)], cat)
    assert r['unique_used_stop_rows'] == 3
    assert r['used_run_subrun_clusters'] == 2
    assert r['unused_run_subrun_keys'] == [[1, 3]]
    assert r['parent_labels_present_in_catalog']
    assert not r['parent_independence_established']
    assert not r['enough_unused_labels_for_24_cluster_design']
    assert r['development_draws'][0]['source_rows'] == 4
    out = tmp_path / 'coverage.json'
    args = ['--parent', str(parent), '--draw', str(source), str(stops), '--out', str(out)]
    main(args)
    assert json.loads(out.read_text())['unused_run_subrun_clusters'] == 1
    with pytest.raises(FileExistsError):
        main(args)


def test_missing_source_identity_and_invalid_indices(tmp_path):
    parent, stops, source = (tmp_path / name for name in ('parent.npz', 'stops.npz', 'source.npz'))
    rows = dict(run=np.array([1]), subrun=np.array([2]), evt=np.array([3]), trk=np.array([4]))
    np.savez(parent, **rows)
    np.savez(stops, **dict(rows, trk=np.array([5])))
    np.savez(source, stop_idx=np.array([0]))
    with pytest.raises(ValueError, match='exactly join'):
        audit(parent, [(source, stops)])
    np.savez(source, stop_idx=np.array([1]))
    with pytest.raises(ValueError, match='outside'):
        audit(parent, [(source, stops)])
    with pytest.raises(ValueError, match='integer'):
        integer_rows([.5])
