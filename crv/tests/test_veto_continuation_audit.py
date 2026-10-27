"""Synthetic checks for provenance binding and selected-exposure diagnostics."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto.source_provenance import extraction_index, identity_summary, source_name
from crvfastsim.veto.selection_exposure import correlate, joined_diagnostic


def test_source_binding_and_ambiguous_list(tmp_path):
    assert source_name('steps.dts.mu2e.X.a.001430_00000744.parquet') == 'dts.mu2e.X.a.001430_00000744.art'
    with pytest.raises(ValueError):
        source_name('dump.root')
    listing = tmp_path / 'list.txt'
    listing.write_text('X /known/a.art\nX /known/b.art\n')
    assert extraction_index(listing)['b.art']['line'] == 2
    listing.write_text('X /known/a.art\nX /other/a.art\n')
    with pytest.raises(ValueError, match='duplicate'):
        extraction_index(listing)


def test_full_tuple_identity_summary():
    frame = pd.DataFrame({'run': [1, 1, 1], 'subrun': [1, 0, 1], 'event': [1, 2**20 + 1, 1]})
    result = identity_summary(frame)
    assert result['unique_events'] == 2
    assert result['event_min_max'] == [1, 2**20 + 1]
    assert result == identity_summary(frame.iloc[::-1])


def test_exposure_join_keeps_empty_selection_denominator():
    selected = pd.DataFrame({'run': [1, 1], 'subrun': [0, 0], 'event': [1, 3],
                             't_mid': [np.array([10., 20.]), np.array([30.])]})
    exposure = pd.DataFrame({'run': [1, 1, 1], 'subrun': [0, 0, 0], 'event': [1, 2, 3],
                             'nprotons': [10., 20., 30.], 'pbtime': [-2., -1., 0.]})
    result = joined_diagnostic(selected, exposure)
    assert result['ntuple_events'] == 3
    assert result['selected_events_matched'] == 2
    assert result['selected_tracks_matched'] == 3
    assert result['pbi_mean_selected_tracks'] == pytest.approx(50 / 3)
    assert sum(r['ntuple_events'] for r in result['pbi_quantile_strata']) == 3
    assert sum(r['selected_tracks'] for r in result['pbi_quantile_strata']) == 3
    with pytest.raises(ValueError, match='ambiguous'):
        joined_diagnostic(pd.concat([selected, selected]), exposure)
    exposure.loc[0, 'nprotons'] = np.nan
    with pytest.raises(ValueError, match='invalid exposure'):
        joined_diagnostic(selected, exposure)


def test_legacy_response_refuses_unbound_ce_schema():
    from crvfastsim.veto.exposure import require_legacy_response
    require_legacy_response({'schema': 'crv-mixed-replay-v1'})
    for manifest in ({'schema': 'crv-mixed-replay-v2'}, {},
                     {'schema': 'crv-mixed-replay-v1', 'exposure_contract': {}}):
        with pytest.raises(ValueError, match='exposure binding'):
            require_legacy_response(manifest)


def test_legacy_pool_refuses_v2_metadata(monkeypatch):
    from crvfastsim.veto import catsteps
    from crvfastsim.veto.mixframes import Pool
    monkeypatch.setattr(catsteps, 'read_meta', lambda path: {'schema': 'crv-cat-v2'})
    with pytest.raises(ValueError, match='legacy Pool refuses'):
        Pool(['synthetic.parquet'])


def test_saved_sam_parentage(tmp_path):
    from crvfastsim.veto.source_provenance import sam_summary
    path = tmp_path / 'sam.txt'
    path.write_text('File Name: x.art\nFile Id: 7\nEvent Count: 4\ndh.sha256: abc\n'
                    'Parents: dts.mu2e.NeutralsFlash.MDC2025ac.001430_00000001.art\n')
    result = sam_summary(path)
    assert result['event_count'] == 4
    assert result['parent_count'] == 1
    assert result['file_id'] == '7'


def test_selection_cli_momentum_window_and_ledger(tmp_path):
    import awkward as ak
    import uproot
    import json
    from crvfastsim.veto.selection_exposure import run
    selection = tmp_path / 'selection.parquet'
    ak.to_parquet(ak.Array({'run': [1, 1, 1], 'subrun': [0, 0, 0], 'event': [1, 2, 3],
                           't_mid': [[10., 20.], [30.], [40.]], 'p_mid': [[85., 100.], [200.], [105.]]}), selection)
    source = tmp_path / 'ntuple.root'
    values = {'evtinfo/run': np.array([1, 1, 1], np.int32), 'evtinfo/subrun': np.zeros(3, np.int32),
              'evtinfo/event': np.array([1, 2, 3], np.int32), 'evtinfomc/nprotons': np.array([10, 20, 30]),
              'evtinfomc/pbtime': np.array([-2., -1., 0.]), 'evtinfo/nprotons': np.zeros(3),
              'evtinfo/pbtime': np.array([-2.1, -1.1, 0.1])}
    with uproot.recreate(source) as root:
        root.mktree('EventNtuple/ntuple', {k: v.dtype for k, v in values.items()})
        root['EventNtuple/ntuple'].extend(values)
    out = tmp_path / 'result.json'
    run(str(selection), [str(source)], out, 'EventNtuple/ntuple')
    report = json.loads(out.read_text())
    assert report['momentum_window']['selected_tracks'] == 2
    assert report['diagnostic']['selected_tracks_matched'] == 2
    assert report['diagnostic']['ntuple_events'] == 3
    assert report['measured_field_checks']['nprotons']['unique_values'] == 1
    assert out.with_suffix('.parquet').exists()
    with pytest.raises(ValueError, match='new report'):
        run(str(selection), [str(source)], out, 'EventNtuple/ntuple')


def test_degenerate_correlation():
    assert correlate([1., 1.], [2., 3.]) is None
    assert correlate([1., 2.], [2., 4.]) == pytest.approx(1)
