"""Pooled pilot limits remain event-level and do not conceal duplicate references."""
import json

import pytest

from beamfastsim.target_inclusive_summary import summarize
from beamfastsim.target_parent_audit import checksum


def test_pooled_summary_counts_events_and_electrons_separately(tmp_path):
    paths, fast_paths = [], []
    for i in range(2):
        path, fast_path = tmp_path / f'census{i}.json', tmp_path / f'fast{i}.json'
        path.write_text(json.dumps(dict(mode='inclusive-matching-background-filter-diagnostic', process='dio',
            generation_census_reconciled=True, reference_sha256=str(i) * 64, all_status_zero=True,
            generated_events=10, generated_electrons=11, generated_positrons=0, primary_vd13_electrons=8,
            passed_events=7, cuts={'450.0': dict(selected_electrons=6, path_rejected_electrons=i,
                g4_selected=6, status_zero_event_miss_diagnostic=dict(proxy_events=5, missed_by_filter=i))})))
        fast_path.write_text(json.dumps(dict(mode='inclusive-same-start-electron-marginal-diagnostic',
            generated_electrons=11, census_sha256=checksum(path), cuts={'450.0': dict(migration=dict(both=4, base_only=2, cand_only=i))})))
        paths.append(path)
        fast_paths.append(fast_path)
    report = summarize(paths, fast_paths)
    assert report['generated_electrons'] == 22 and report['generated_events'] == 20
    cut = report['cuts']['450.0']
    assert cut['event_miss_diagnostic']['proxy_events'] == 10
    assert cut['event_miss_diagnostic']['miss_fraction'] == .1
    assert cut['fast_over_g4'] == .75
    assert cut['paired_file_cluster_ratio_err'] == pytest.approx(1 / 12)
    assert not report['generated_yield_validated']
    with pytest.raises(ValueError, match='distinct'):
        summarize([paths[0], paths[0]])
    with pytest.raises(ValueError, match='match every'):
        summarize(paths, fast_paths[::-1])


def test_rpc_pooled_event_uncertainty_not_replaced_by_zero_file_spread(tmp_path):
    paths, fast_paths = [], []
    for i in range(4):
        path, fast_path = tmp_path / f'c{i}.json', tmp_path / f'f{i}.json'
        path.write_text(json.dumps(dict(mode='inclusive-matching-background-filter-diagnostic', process='rpc-internal',
            generation_census_reconciled=True, reference_sha256=str(i) * 64, all_status_zero=True,
            generated_events=10, generated_electrons=10, generated_positrons=10, primary_vd13_electrons=8,
            passed_events=7, cuts={'450.0': dict(selected_electrons=5, path_rejected_electrons=0,
                status_zero_event_miss_diagnostic=dict(proxy_events=5, missed_by_filter=0))})))
        fast_path.write_text(json.dumps(dict(mode='inclusive-same-start-electron-marginal-diagnostic',
            generated_electrons=10, census_sha256=checksum(path),
            cuts={'450.0': dict(migration=dict(both=4, base_only=1, cand_only=2))})))
        paths.append(path)
        fast_paths.append(fast_path)
    report = summarize(paths, fast_paths)
    assert report['cuts']['450.0']['paired_file_cluster_ratio_err'] == 0
    assert report['cuts']['450.0']['paired_event_ratio_err'] > 0
